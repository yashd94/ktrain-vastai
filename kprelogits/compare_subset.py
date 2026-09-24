"""Check a full-split bundle against the subsampled bundle of the same model.

A ``max_train=0`` bundle holds every train row in dataset order; a
``max_train=10000`` bundle holds the seed-42 subsample, in permutation order.
``models.train_subset_indices`` is the map between them, so the full bundle's
rows at those indices must BE the subsampled bundle: labels exactly, features
to GPU noise. Val and test are extracted whole either way and must match
directly.

That makes each existing 10k bundle a free reference for its full-split
successor -- the same tiered test as ``ops.smoke.compare_bundles``, which
proves the extraction path, plus an exact label check on 10,000 train rows,
which proves the row order. A full bundle whose rows were shuffled, or
subsampled after all, fails the labels; one with the wrong weights or
preprocessing fails the cosine.

Usage (bundles are local; fetch the pair from S3 first)::

    python -m kprelogits.compare_subset FULL.npz SUBSET.npz [FULL SUBSET ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Identity fields that must agree between the two bundles. max_train is
# excluded on purpose -- it is the one field that differs.
SAME = ("dataset", "model_name", "feat_dim", "num_classes", "precision",
        "preprocessing", "weights_revision")


def feature_agreement(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, float, bool]:
    """``(cos_min, cos_mean, rel_fro, max_abs, ok)`` for two same-shape arrays.

    ``ok`` is the gate ``ops.smoke`` established: per-row cosine above 0.999
    and relative Frobenius error under 1e-2. fp16 autocast and TF32 cudnn make
    ~1e-2 the noise between two correct extractions on different cards; a
    cosine below ~0.99 means preprocessing, weights or normalization diverged.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    cos = (a * b).sum(1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))
    ok = bool(cos.min() > 0.999 and rel < 1e-2)
    return float(cos.min()), float(cos.mean()), rel, float(np.abs(a - b).max()), ok


def compare(full: Path, subset: Path, *, seed: int = 42) -> bool:
    """True when ``full`` is the full-split extension of ``subset``."""
    from .contracts import validate_prelogit_bundle
    from .models import train_subset_indices

    ok = True
    with np.load(full, allow_pickle=False) as z:
        fm = validate_prelogit_bundle(z)
        F = {k: z[k] for k in z.files if k.startswith(("x_", "y_"))}
    with np.load(subset, allow_pickle=False) as z:
        sm = validate_prelogit_bundle(z)
        S = {k: z[k] for k in z.files if k.startswith(("x_", "y_"))}
    print(f"{fm.model_name}")

    if fm.max_train != 0:
        print(f"  NOT FULL: {full.name} has max_train={fm.max_train}")
        ok = False
    for field in SAME:
        if getattr(fm, field) != getattr(sm, field):
            print(f"  MISMATCH {field}: {getattr(fm, field)!r} vs "
                  f"{getattr(sm, field)!r}")
            ok = False

    for split in ("train", "val", "test"):
        if not np.isfinite(F[f"x_{split}"]).all():
            print(f"  NON-FINITE: x_{split} of the full bundle")
            ok = False

    n_full = F["x_train"].shape[0]
    idx = train_subset_indices(n_full, sm.max_train, seed)
    if idx is None or len(idx) != S["x_train"].shape[0]:
        print(f"  MISMATCH train rows: full has {n_full}, subset has "
              f"{S['x_train'].shape[0]} at max_train={sm.max_train} -- the "
              f"subset is not a proper subsample of this split")
        return False
    print(f"  train {n_full} rows; the {len(idx)} subsampled rows are compared")

    pairs = {"train": (F["x_train"][idx], F["y_train"][idx]),
             "val": (F["x_val"], F["y_val"]),
             "test": (F["x_test"], F["y_test"])}
    for split, (x, y) in pairs.items():
        xs, ys = S[f"x_{split}"], S[f"y_{split}"]
        if x.shape != xs.shape:
            print(f"  MISMATCH x_{split}: {x.shape} vs {xs.shape}")
            ok = False
            continue
        if not np.array_equal(y, ys):
            print(f"  MISMATCH y_{split}: labels differ -- rows are not aligned")
            ok = False
        cmin, cmean, rel, mx, good = feature_agreement(x, xs)
        ok &= good
        print(f"  x_{split}: cos min={cmin:.6f} mean={cmean:.6f}  "
              f"rel_fro={rel:.2e}  max|d|={mx:.2e}  {'OK' if good else 'FAIL'}")
    return ok


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pairs", nargs="+", type=Path, metavar="FULL SUBSET")
    p.add_argument("--seed", type=int, default=42,
                   help="the subsample seed (ExtractConfig.seed; default 42)")
    args = p.parse_args(argv)
    if len(args.pairs) % 2:
        p.error("give bundles in pairs: FULL SUBSET [FULL SUBSET ...]")
    results = [compare(f, s, seed=args.seed)
               for f, s in zip(args.pairs[::2], args.pairs[1::2])]
    print(f"\n{sum(results)}/{len(results)} pair(s) agree")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
