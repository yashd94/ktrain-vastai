"""Choose which backbones to extract. Owned code.

Copied out of ``legacy/jobs/build_model_list.py`` plus ktrain's
``filter_timm_models``, and adopted the same way ``models.py`` was: kprelogits
owns this now and does not track ktrain.

Two ways to produce a selection:

  ``scan_timm``   walks the whole pretrained registry, filtering to
                  224x224/3-channel/<=max_params. Expensive -- cache misses get
                  instantiated (with ``pretrained=False``) purely to count
                  parameters -- so it needs no GPU but should be run once, not
                  once per shard. Results accumulate in a JSON param cache.
  ``from_cache``  replays the same filter against an existing cache. No timm,
                  no network. Blind to models added to timm since.

The selection is then subsampled ``sorted(names)[offset::stride]``. Sorting
before striding is not cosmetic: ``scan_timm`` returns models in an order that
depends on which entries were already cached, so an unsorted stride is not
reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SELECTION_NAME = "selected_models.json"
CACHE_NAME = "timm_model_param_cache.json"
SELECTION_SCHEMA = 1


def guess_size_bucket(name: str) -> int:
    """Rough param estimate (millions) from the model name.

    Only used to order cache misses largest-first, so that a scan interrupted
    partway has already resolved the expensive entries.
    """
    n = name.lower()
    if any(k in n for k in ("huge", "_h_", "_h.", "xlarge", "_xl")):
        return 600
    if any(k in n for k in ("large", "_l_", "_l.")):
        return 300
    if any(k in n for k in ("base", "_b_", "_b.")):
        return 80
    if any(k in n for k in ("small", "_s_", "_s.")):
        return 25
    if any(k in n for k in ("tiny", "_t_", "_t.", "nano", "micro", "pico")):
        return 10

    m = re.search(r"efficientnet.*b(\d)", n)
    if m:
        return [5, 8, 9, 12, 19, 30, 43, 66][min(int(m.group(1)), 7)]
    m = re.search(r"resne[tx]t?(\d+)", n)
    if m:
        return {18: 12, 26: 16, 34: 22, 50: 25, 101: 45, 152: 60}.get(int(m.group(1)), 30)
    m = re.search(r"densenet(\d+)", n)
    if m:
        return {121: 8, 161: 29, 169: 14, 201: 20}.get(int(m.group(1)), 15)
    if "vgg" in n:
        return 140
    return 50


def from_cache(cache_path: Path, max_params: int) -> List[Dict[str, Any]]:
    """Reproduce the scan's selection using only a prior param cache.

    Cache semantics: ``bad_input`` means the model failed the 224x224/3-channel
    check and has ``params=None``; ``too_large`` means it exceeded the
    max_params in force *when the cache was built*, so its recorded count is
    still valid and is re-tested against ours here.
    """
    with open(cache_path) as f:
        cache = json.load(f)

    compatible, too_large, unusable = [], 0, 0
    for name, rec in cache.items():
        params, status = rec.get("params"), rec.get("status")
        if params is None or status not in ("ok", "too_large"):
            unusable += 1
        elif params > max_params:
            too_large += 1
        else:
            compatible.append({"name": name, "params": params})

    print(f"Cache: {len(cache)} entries from {cache_path}")
    print(f"  compatible (<={max_params/1e6:.0f}M, 224x224): {len(compatible)}")
    print(f"  skipped (too large):                  {too_large}")
    print(f"  skipped (bad input / unresolved):     {unusable}")
    return compatible


def scan_timm(max_params: int = 200_000_000,
              cache_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Walk the timm registry, resolving and caching param counts.

    Stage 1 drops anything whose pretrained cfg is not (3, 224, 224). Stage 2
    resolves a param count from Hub metadata when possible and instantiates the
    model (no weights downloaded) when not. Stage 3 applies the ceiling. The
    cache is rewritten after every miss so an interrupted scan keeps its work.
    """
    import timm
    from huggingface_hub import model_info as hf_model_info

    cache_path = Path(cache_path) if cache_path else None
    cache: Dict[str, Any] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        print(f"Cache: {len(cache)} entries from {cache_path}")

    all_models = timm.list_models(pretrained=True)
    print(f"Total pretrained timm models: {len(all_models)}")

    compatible: List[Dict[str, Any]] = []
    skipped_size = skipped_other = 0
    for name in all_models:
        rec = cache.get(name)
        if rec is None:
            continue
        if rec.get("status") == "ok" and rec["params"] <= max_params:
            compatible.append({"name": name, "params": rec["params"]})
        elif rec.get("status") == "too_large" or (rec.get("params") and
                                                  rec["params"] > max_params):
            skipped_size += 1
        else:
            skipped_other += 1

    uncached = sorted((n for n in all_models if n not in cache),
                      key=guess_size_bucket, reverse=True)
    print(f"Uncached models remaining: {len(uncached)}")

    for name in uncached:
        try:
            cfg = timm.get_pretrained_cfg(name)
            size = tuple(cfg.input_size) if cfg and getattr(cfg, "input_size", None) else None
            if size != (3, 224, 224):
                cache[name] = {"params": None, "status": "bad_input"}
                skipped_other += 1
                continue

            n_params = None
            try:                                    # from Hub metadata
                hf_id = getattr(cfg, "hf_hub_id", None) or f"timm/{name}"
                info = hf_model_info(hf_id)
                st = getattr(info, "safetensors", None)
                if st is not None and hasattr(st, "total"):
                    n_params = st.total
                elif getattr(info, "siblings", None):
                    for sib in info.siblings:
                        if sib.rfilename.endswith((".safetensors", ".bin")) and sib.size:
                            n_params = sib.size // 4
                            break
            except Exception:
                pass
            if n_params is None:                    # or by instantiating
                model = None
                try:
                    model = timm.create_model(name, pretrained=False, num_classes=0)
                    n_params = sum(p.numel() for p in model.parameters())
                except Exception:
                    pass
                finally:
                    del model

            if n_params is None:
                cache[name] = {"params": None, "status": "unknown_params"}
                skipped_other += 1
            elif n_params > max_params:
                cache[name] = {"params": n_params, "status": "too_large"}
                skipped_size += 1
            else:
                cache[name] = {"params": n_params, "status": "ok"}
                compatible.append({"name": name, "params": n_params})
        except Exception as e:
            cache[name] = {"params": None, "status": "error", "error": str(e)}
            skipped_other += 1

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))

    print(f"Compatible (<={max_params/1e6:.0f}M, 224x224): {len(compatible)}")
    print(f"Skipped (too large): {skipped_size}   (error/unknown): {skipped_other}")
    return compatible


def build_selection(compatible: List[Dict[str, Any]], *, max_params: int,
                    stride: int = 50, offset: int = 0, limit: int = 0,
                    source: str = "cache") -> Dict[str, Any]:
    """Sort, stride, cap -- and record exactly how, so it can be reproduced."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if offset < 0 or (stride > 1 and offset >= stride):
        raise ValueError(f"offset must be in [0, {stride}), got {offset}")

    ordered = sorted(compatible, key=lambda m: m["name"])
    selected = ordered[offset::stride]
    if limit:
        selected = selected[:limit]
    return {
        "schema": SELECTION_SCHEMA,
        "max_params": max_params,
        "stride": stride,
        "offset": offset,
        "limit": limit,
        "selection": "sorted by name, then [offset::stride]",
        "source": source,
        "total_compatible": len(ordered),
        "n_selected": len(selected),
        "models": selected,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    results = Path(os.environ.get("RESULTS_DIR", "/workspace/results"))
    shared = results / "_shared"
    bucket = os.environ.get("S3_BUCKET", "")
    prefix = os.environ.get("S3_PREFIX", "ktrain/results").strip("/")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-params", type=int,
                   default=int(os.environ.get("MAX_NUM_PARAMS") or 200_000_000))
    p.add_argument("--stride", type=int, default=int(os.environ.get("MODEL_STRIDE") or 50),
                   help="Keep every Nth model after sorting by name")
    p.add_argument("--offset", type=int, default=int(os.environ.get("MODEL_OFFSET") or 0))
    p.add_argument("--limit", type=int, default=int(os.environ.get("MAX_MODELS") or 0),
                   help="Cap applied after striding; 0 = no cap")
    p.add_argument("--cache", type=Path, default=shared / CACHE_NAME)
    p.add_argument("--output", type=Path, default=shared / SELECTION_NAME)
    p.add_argument("--from-cache", action="store_true",
                   help="Derive the list from --cache alone: no timm, no network")
    p.add_argument("--s3-uri", default=(f"s3://{bucket}/{prefix}" if bucket else ""),
                   help="Optional destination; selection + cache go to {uri}/_shared/")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MODEL SELECTION")
    print(f"  max_params = {args.max_params/1e6:.0f}M")
    print(f"  stride     = every {args.stride} (offset {args.offset})")
    print(f"  cache      = {args.cache}")
    print(f"  source     = {'cache only (no scan)' if args.from_cache else 'live timm scan'}")
    print("=" * 70, flush=True)

    if args.from_cache:
        if not args.cache.exists():
            print(f"ERROR: --from-cache needs an existing cache at {args.cache}",
                  file=sys.stderr)
            return 2
        compatible = from_cache(args.cache, args.max_params)
    else:
        compatible = scan_timm(args.max_params, args.cache)

    selection = build_selection(compatible, max_params=args.max_params,
                                stride=args.stride, offset=args.offset,
                                limit=args.limit,
                                source="cache" if args.from_cache else "scan")
    args.output.write_text(json.dumps(selection, indent=2))

    sel = selection["models"]
    print(f"\nCompatible: {selection['total_compatible']}   "
          f"selected: {len(sel)}")
    if sel:
        print(f"  first 5: {', '.join(m['name'] for m in sel[:5])}")
        print(f"  params:  {min(m['params'] for m in sel)/1e6:.1f}M - "
              f"{max(m['params'] for m in sel)/1e6:.1f}M")
    print(f"Selection: {args.output}")

    if args.s3_uri:
        from .ops import s3
        for local, key in ((args.output, SELECTION_NAME), (args.cache, CACHE_NAME)):
            if local.exists():
                try:
                    s3.upload(local, f"{args.s3_uri.rstrip('/')}/_shared/{key}")
                    print(f"  uploaded -> {args.s3_uri}/_shared/{key}")
                except Exception as e:
                    print(f"  WARNING: upload of {key} failed: {e}")

    if not sel:
        print("ERROR: selection is empty -- extraction would have nothing to do",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
