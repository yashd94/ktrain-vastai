"""The GPU pass: run every selected backbone over the dataset once.

This is the only stage that needs a GPU and the only one whose output is worth
more than the machine that produced it, so the driver is organized around not
losing bundles:

  *Atomic writes.* ``save_atomic`` writes to a tmp file and renames, so a crash
  or a concurrent ``aws s3 sync`` can never publish a truncated .npz -- which
  would be worse than a missing one, since a truncated file looks done.

  *S3 is the resume oracle.* With ``cleanup`` on, local disk is emptied as we
  go; asking the filesystem what is done would re-extract everything after an
  interruption.

  *Cleanup only after a CONFIRMED upload.* An upload failure is loud, keeps the
  local copy regardless of the flag, and does not stop the run -- there is no
  version of "throw away the remaining GPU time" that helps.

  *A final sync of everything on disk*, not just what this run produced. A
  resumed run's pending list excludes earlier models, so per-model uploads
  alone would strand their bundles on a box about to be destroyed.

Every bundle is stamped with the full schema from ``contracts.py`` -- dataset,
weights revision, preprocessing spec, producer version -- so a consumer can
validate provenance without trusting a filename. ``bundle_payload`` is a pure function precisely so
that the contract can be tested without a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .contracts import PRELOGIT_BUNDLE_SCHEMA_VERSION

from . import producer_version
from .config import ExtractConfig, preflight, read_hf_token
from .models import PREPROCESSING_SPEC

DTYPES = {"fp32": np.float32, "fp16": np.float16}


# ---------------------------------------------------------------------------
# Bundle assembly (pure -- no GPU, no I/O)
# ---------------------------------------------------------------------------

def bundle_key(*, data_flag: str, model_name: str, max_train: int,
               precision: str, weights_revision: str, preprocessing: str) -> str:
    """Content identity of a bundle.

    Bundle *filenames* encode only (dataset, model), which is what makes them a
    shared cache -- but it also means two runs with different ``max_train`` or
    different weights would collide. This hash is stamped inside so a consumer
    can tell two same-named bundles apart after the fact, and preflight's
    published-config check stops the collision happening in the first place.
    """
    blob = json.dumps({"data_flag": data_flag, "model_name": model_name,
                       "max_train": max_train, "precision": precision,
                       "weights_revision": weights_revision,
                       "preprocessing": preprocessing},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def bundle_payload(features: Dict[str, Any], *, data_flag: str, model_name: str,
                   feat_dim: int, num_classes: int, max_train: int,
                   precision: str, weights_revision: str,
                   preprocessing: str = PREPROCESSING_SPEC,
                   producer: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Assemble the .npz members for one bundle, contract included.

    ``features`` holds ``x_{train,val,test}`` / ``y_{train,val,test}`` as torch
    tensors or arrays; anything ``np.asarray`` accepts works, which is what
    lets the contract be tested with synthetic data.
    """
    if precision not in DTYPES:
        raise ValueError(f"precision must be one of {sorted(DTYPES)}, got {precision!r}")
    dtype = DTYPES[precision]

    payload: Dict[str, np.ndarray] = {}
    for split in ("train", "val", "test"):
        payload[f"x_{split}"] = np.asarray(features[f"x_{split}"]).astype(dtype)
        payload[f"y_{split}"] = np.asarray(features[f"y_{split}"]).astype(np.int64).reshape(-1)

    payload.update(
        schema_version=np.int64(PRELOGIT_BUNDLE_SCHEMA_VERSION),
        dataset=np.array(data_flag),
        model_name=np.array(model_name),
        feat_dim=np.int64(feat_dim),
        num_classes=np.int64(num_classes),
        max_train=np.int64(max_train),
        precision=np.array(precision),
        weights_revision=np.array(weights_revision),
        preprocessing=np.array(preprocessing),
        producer_version=np.array(producer or producer_version()),
        bundle_key=np.array(bundle_key(
            data_flag=data_flag, model_name=model_name, max_train=max_train,
            precision=precision, weights_revision=weights_revision,
            preprocessing=preprocessing)),
    )
    return payload


def save_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
    """Write via tmp + rename. A partial .npz must never become visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Local bundle inspection
# ---------------------------------------------------------------------------

def bundle_status(path: Path) -> str:
    """``missing`` | ``ok`` | ``legacy`` | ``corrupt``.

    ``corrupt`` files are deleted so the next pass re-extracts them; a
    truncated bundle that merely got skipped is how a run finishes "clean" with
    a hole in it. ``legacy`` means readable but predating the contract -- see
    ``kprelogits/restamp.py``, which upgrades those without a GPU.
    """
    if not path.exists():
        return "missing"
    try:
        with np.load(path, allow_pickle=False) as z:
            from .contracts import ContractError, validate_prelogit_bundle
            try:
                validate_prelogit_bundle(z)
                return "ok"
            except ContractError:
                return "legacy" if {"x_train", "x_test"} <= set(z.files) else "corrupt"
    except Exception:
        print(f"  corrupt, will re-extract: {path.name}")
        path.unlink(missing_ok=True)
        return "corrupt"


def partition_scheduled(cfg, scheduled: List[str], have: set):
    """Split scheduled models into ``(pending, legacy, skipped_count)``.

    Three outcomes, and the distinction between the last two is the point.
    A bundle already in S3, or a contract-valid one on disk, is genuine
    progress and counts as *skipped*. A pre-contract bundle on disk is
    neither: re-extracting it would waste a GPU on features that are already
    correct, but counting it as done would let a shard whose every bundle is
    legacy report complete having uploaded nothing usable. It is its own
    category so the caller can refuse to proceed.
    """
    pending: List[str] = []
    legacy: List[str] = []
    skipped = 0
    for name in scheduled:
        if cfg.bundle_name(name) in have:
            skipped += 1
            continue
        status = bundle_status(cfg.bundle_path(name))
        if status == "ok":
            skipped += 1
        elif status == "legacy":
            legacy.append(name)
        else:
            pending.append(name)
    return pending, legacy, skipped


def drop_hf_weights(model_name: str) -> int:
    """Delete one model's cached weights, returning bytes freed.

    Every backbone in the sweep is a distinct checkpoint, so the cache has no
    reuse value once its features exist -- but only this model's entry goes, in
    case a restored cache holds weights for models still to come.
    """
    hub = os.environ.get("HF_HUB_CACHE") or (
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None)
    if not hub or not os.path.isdir(hub):
        return 0
    freed = 0
    for d in Path(hub).glob(f"models--*--{model_name}"):
        try:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass
    return freed


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-flag")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--results-dir", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--max-train", type=int, help="0 = full train split")
    p.add_argument("--feature-batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--precision", choices=sorted(DTYPES))
    p.add_argument("--rank", type=int)
    p.add_argument("--shards", type=int)
    p.add_argument("--s3-uri")
    p.add_argument("--hf-token-file", type=Path,
                   help="Path to a mode-600 file holding an HF token. Never "
                        "pass the token itself: it would be visible in ps.")
    p.add_argument("--cleanup", action="store_true", default=None,
                   help="After a CONFIRMED upload, delete the local .npz and "
                        "that model's HF weights")
    p.add_argument("--dry-run", action="store_true",
                   help="Preflight and print the resolved config; extract nothing")
    p.add_argument("--no-probe", action="store_true",
                   help="Skip the S3 and HuggingFace preflight probes (offline)")
    return p.parse_args(argv)


def config_from_args(args) -> ExtractConfig:
    return ExtractConfig.from_env(
        data_flag=args.data_flag, data_dir=args.data_dir,
        results_dir=args.results_dir, selection=args.selection,
        max_train=args.max_train, feature_batch_size=args.feature_batch_size,
        num_workers=args.num_workers, precision=args.precision,
        rank=args.rank, shards=args.shards, s3_uri=args.s3_uri,
        cleanup=args.cleanup, hf_token_file=args.hf_token_file,
    )


def _banner(cfg: ExtractConfig, device, n_models: int) -> None:
    from .models import device_name
    print("=" * 70)
    print("PRELOGIT EXTRACTION")
    print(f"  dataset    = {cfg.data_flag}")
    print(f"  device     = {device} ({device_name(device)})")
    print(f"  models     = {n_models} in selection")
    print(f"  shard      = {cfg.rank}/{cfg.shards}")
    print(f"  max_train  = {cfg.max_train or 'full split'}")
    print(f"  precision  = {cfg.precision}")
    print(f"  out        = {cfg.features_dir}")
    print(f"  s3         = {cfg.s3_prefix or '(none -- bundles stay on this instance)'}")
    print(f"  cleanup    = {'on' if cfg.cleanup else 'off'}")
    print(f"  producer   = {producer_version()}")
    print("=" * 70, flush=True)


def run(cfg: ExtractConfig) -> int:
    """Extract this shard's models. Returns a process exit code.

    Any escaping exception is recorded in the shard state as ``failed`` before
    it propagates -- a shard that dies silently looks identical to one still
    working, and the driver would wait on it forever.
    """
    from .ops import state

    st = state.StateFile.create(
        data_flag=cfg.data_flag, rank=cfg.rank, shards=cfg.shards,
        results_dir=cfg.results_dir, s3_prefix=cfg.s3_prefix)
    st.write()
    try:
        return _run(cfg, st)
    except BaseException as e:
        st.fail(f"{type(e).__name__}: {e}")
        raise


def _run(cfg: ExtractConfig, st) -> int:
    import torch

    from . import models as M
    from .ops import s3

    with open(cfg.selection_path) as f:
        model_names = [m["name"] for m in json.load(f)["models"]]
    scheduled = M.select_model_shard(model_names, cfg.shards, cfg.rank)

    device = M.pick_device()
    _banner(cfg, device, len(model_names))
    if device.type != "cuda":
        print("WARNING: no CUDA device -- extraction will be extremely slow.\n")
    if cfg.shards > 1:
        print(f"Shard {cfg.rank}/{cfg.shards}: {len(scheduled)} of "
              f"{len(model_names)} models")

    st.state.scheduled = len(scheduled)
    st.write()

    if cfg.s3_prefix:
        try:
            s3.put_json(cfg.resolved(), f"{cfg.s3_prefix}/{cfg.config_key}")
        except Exception as e:
            print(f"  WARNING: could not publish resolved config: {e}")

    # S3 first (the cleanup case), then local disk.
    have = s3.list_names(cfg.s3_prefix, suffix="_features.npz") if cfg.s3_prefix else set()
    if have:
        print(f"{len(have)} bundle(s) already in S3")
    pending, legacy, skipped = partition_scheduled(cfg, scheduled, have)

    if legacy:
        print(f"WARNING: {len(legacy)} local bundle(s) predate the contract and "
              f"are NOT re-extracted (e.g. {legacy[0]}). Upgrade them without a "
              f"GPU: python -m kprelogits.restamp --dir {cfg.features_dir}")
    st.state.skipped = skipped
    print(f"{st.state.skipped} already done; {len(pending)} to do.\n", flush=True)
    if legacy and not pending:
        st.fail(f"{len(legacy)} scheduled bundle(s) predate the contract and "
                f"nothing else was pending; restamp them, then re-run")
        return 1
    if not pending:
        st.advance("done", current=None)
        print("Nothing to do -- every scheduled model already has a bundle.")
        return 0

    st.advance("running")

    # Loaders are built ONCE and reused across backbones.
    train_loader, val_loader, test_loader, info = M.get_loaders(
        cfg.data_flag, str(cfg.data_dir), max_train=cfg.max_train or None,
        batch_size=cfg.feature_batch_size, num_workers=cfg.num_workers,
        seed=cfg.seed)
    num_classes = len(info["label"])

    done: List[dict] = []
    upload_failures: List[str] = []
    for i, name in enumerate(pending, 1):
        out = cfg.bundle_path(name)
        print(f"[{i}/{len(pending)}] {name}", flush=True)
        st.begin_model(name)
        t0 = time.perf_counter()

        try:
            encoder, feat_dim = M.build_encoder(name, device)
        except Exception as e:
            print(f"    FAILED build_encoder: {type(e).__name__}: {e}", flush=True)
            st.model_failed(name, "build_encoder", f"{type(e).__name__}: {e}")
            continue
        try:
            feats = M.extract_all_splits(
                encoder, train_loader, val_loader, test_loader, device,
                model_name=name, feature_batch_size=cfg.feature_batch_size)
        except Exception as e:
            print(f"    FAILED extract: {type(e).__name__}: {e}", flush=True)
            st.model_failed(name, "extract", f"{type(e).__name__}: {e}")
            del encoder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        # extract_all_splits drops ITS reference; this drops ours. Without it
        # the name stays bound until the next iteration rebinds it -- which
        # happens inside build_encoder(next model), so two backbones briefly
        # coexist in VRAM and one OOM becomes a cascade.
        del encoder

        save_atomic(out, bundle_payload(
            feats, data_flag=cfg.data_flag, model_name=name, feat_dim=feat_dim,
            num_classes=num_classes, max_train=cfg.max_train,
            precision=cfg.precision, weights_revision=M.weights_revision(name)))
        del feats

        dt = time.perf_counter() - t0
        mb = out.stat().st_size / 1e6
        print(f"    feat_dim={feat_dim}  {mb:.0f} MB  {dt:.1f}s", flush=True)
        done.append({"model": name, "feat_dim": int(feat_dim),
                     "seconds": round(dt, 1), "megabytes": round(mb, 1)})
        st.model_extracted()

        if not cfg.s3_prefix:
            continue
        dest = f"{cfg.s3_prefix}/{out.name}"
        try:
            s3.upload(out, dest)
        except Exception as e:
            # Loud, not fatal: the local copy survives and the final sync will
            # retry it. Deliberately NOT deleted, whatever cleanup says.
            print(f"    UPLOAD FAILED: {e}", flush=True)
            upload_failures.append(out.name)
            continue
        st.model_uploaded()
        msg = f"    -> {dest}"
        if cfg.cleanup:
            local_mb = out.stat().st_size / 1e6
            out.unlink(missing_ok=True)
            msg += f"  [freed {local_mb + drop_hf_weights(name) / 1e6:.0f} MB]"
        print(msg, flush=True)

    # Safety net: push everything still on disk, including earlier runs' work.
    sync_ok = True
    if cfg.s3_prefix:
        st.advance("uploading")
        print(f"\nSyncing all local bundles -> {cfg.s3_prefix}/", flush=True)
        try:
            s3.sync_dir(cfg.features_dir, cfg.s3_prefix)
            print("  sync complete", flush=True)
        except Exception as e:
            sync_ok = False
            print(f"  SYNC FAILED: {e}", flush=True)
            print("  Bundles are still on the instance -- DO NOT DESTROY IT.",
                  flush=True)

        # The sync exists precisely to retry the uploads that failed, so ask S3
        # what actually landed rather than leaving them recorded as failures.
        # Without this a bundle that failed once and synced fine still marks the
        # shard failed and blocks teardown -- a false alarm that costs rented
        # hours to investigate, on a run where nothing was lost.
        if sync_ok and upload_failures:
            landed = s3.list_names(cfg.s3_prefix, suffix="_features.npz")
            recovered = [n for n in upload_failures if n in landed]
            for n in recovered:
                upload_failures.remove(n)
                st.model_uploaded()
            if recovered:
                print(f"  recovered by sync: {', '.join(recovered)}", flush=True)

    report_path = (Path(cfg.results_dir) / f"{cfg.data_flag}_lp"
                   / f"extract_report_shard{cfg.rank}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(
        {"config": cfg.resolved(), "done": done,
         "failed": st.state.errors, "upload_failures": upload_failures}, indent=2))
    if cfg.s3_prefix:
        try:
            s3.upload(report_path, f"{cfg.s3_prefix}/{report_path.name}")
        except Exception as e:
            print(f"  WARNING: report upload failed: {e}")

    failed = st.state.failed
    st.advance("done" if (sync_ok and not upload_failures) else "failed",
               current=None)

    print("\n" + "=" * 70)
    print(f"extracted {len(done)}  failed {failed}")
    for f in st.state.errors:
        print(f"  FAILED {f['model']} @ {f['stage']}: {f['error'][:120]}")
    if cfg.s3_prefix:
        for u in upload_failures:
            print(f"  UPLOAD FAILED (still on local disk): {u}")
        print(f"state: {cfg.s3_prefix}/{cfg.state_key}")
    else:
        print("WARNING: no S3 destination -- bundles exist only on this instance.")
    print("=" * 70)
    return 0 if (sync_ok and not upload_failures and not failed) else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)

    probe = not args.no_probe
    rep = preflight(cfg, check_s3=probe, check_hf=probe)
    print("preflight:")
    print(rep.render(), flush=True)
    if not rep.ok:
        rep.raise_if_failed()

    if args.dry_run:
        print("\nresolved config:")
        print(json.dumps(cfg.resolved(), indent=2, sort_keys=True))
        print(f"\nfacts: {json.dumps(rep.facts, sort_keys=True)}")
        print("\nDRY RUN -- nothing extracted.")
        return 0

    token = read_hf_token(cfg)
    if token:
        # timm/huggingface_hub read this; it never reaches a command line.
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    if cfg.hf_home:
        os.environ.setdefault("HF_HOME", str(cfg.hf_home))

    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
