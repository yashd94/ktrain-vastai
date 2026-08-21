"""Upgrade pre-contract bundles in place, without a GPU.

Bundles written before ``contracts.py`` existed carry the arrays and the
basic metadata (``feat_dim``, ``num_classes``, ``model_name``, ``max_train``,
``precision``) but none of the provenance the contract requires. Re-extracting
them would mean renting GPUs again to recompute features that are already
correct; restamping adds the missing scalars in seconds.

What it will NOT do is invent provenance. A restamped bundle records
``weights_revision="unknown:pre-contract"`` and a ``producer_version`` that
names both the legacy producer and this tool, because the actual timm weights
those features came from are genuinely unrecoverable after the fact. That is
the honest answer, and it is visible to anything that reads the manifest --
unlike the alternative, which is a plausible-looking string nobody can check.

``preprocessing`` is the one field asserted rather than observed: the legacy
extractor called ktrain's ``_image_transform``, which is byte-for-byte the
pipeline ``models.PREPROCESSING_SPEC`` describes. If that ever stops being
true, this assertion is where it breaks.

Arrays are copied through untouched -- no dtype change, no re-standardization.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .contracts import (
    PRELOGIT_BUNDLE_SCHEMA_VERSION, ContractError, validate_prelogit_bundle,
)

from . import producer_version
from .extract import bundle_key
from .models import PREPROCESSING_SPEC

UNKNOWN_WEIGHTS = "unknown:pre-contract"
LEGACY_PRODUCER = "legacy/jobs/extract_features.py"

# What a legacy bundle must already have for a restamp to be possible.
_LEGACY_REQUIRED = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test",
                    "feat_dim", "num_classes", "model_name", "max_train", "precision")


def restamp_payload(z, *, data_flag: str) -> Dict[str, np.ndarray]:
    """Build the contract-complete payload for one opened legacy bundle."""
    missing = [k for k in _LEGACY_REQUIRED if k not in z.files]
    if missing:
        raise ContractError(f"not restampable, missing: {missing}")

    payload = {k: z[k] for k in z.files}
    model_name = str(z["model_name"])
    max_train = int(z["max_train"])
    precision = str(z["precision"])

    payload.update(
        schema_version=np.int64(PRELOGIT_BUNDLE_SCHEMA_VERSION),
        dataset=np.array(data_flag),
        weights_revision=np.array(UNKNOWN_WEIGHTS),
        preprocessing=np.array(PREPROCESSING_SPEC),
        producer_version=np.array(f"{LEGACY_PRODUCER} restamped-by:{producer_version()}"),
        bundle_key=np.array(bundle_key(
            data_flag=data_flag, model_name=model_name, max_train=max_train,
            precision=precision, weights_revision=UNKNOWN_WEIGHTS,
            preprocessing=PREPROCESSING_SPEC)),
    )
    return payload


def restamp_file(path: Path, *, data_flag: str,
                 out_dir: Optional[Path] = None) -> str:
    """Restamp one bundle. Returns ``ok`` | ``already`` | an error string.

    Writes through a tmp file and renames, so an interrupted restamp cannot
    leave a half-written bundle where a valid one used to be.
    """
    dest = (Path(out_dir) / path.name) if out_dir else path
    with np.load(path, allow_pickle=False) as z:
        try:
            validate_prelogit_bundle(z)
            return "already"
        except ContractError:
            pass
        payload = restamp_payload(z, data_flag=data_flag)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **payload)
    with np.load(tmp, allow_pickle=False) as z2:
        validate_prelogit_bundle(z2)      # never publish an unvalidated result
    os.replace(tmp, dest)
    return "ok"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True,
                   help="Directory of *_features.npz bundles")
    p.add_argument("--data-flag", required=True,
                   help="Dataset the bundles came from; there is no way to "
                        "recover this from a legacy bundle, so it must be stated")
    p.add_argument("--out-dir", type=Path,
                   help="Write upgraded copies here instead of in place")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    bundles: List[Path] = sorted(Path(args.dir).glob("*_features.npz"))
    if not bundles:
        print(f"no *_features.npz under {args.dir}", file=sys.stderr)
        return 1

    counts = {"ok": 0, "already": 0, "error": 0}
    for b in bundles:
        if args.dry_run:
            with np.load(b, allow_pickle=False) as z:
                try:
                    validate_prelogit_bundle(z)
                    status = "already"
                except ContractError as e:
                    status = "would restamp" if all(
                        k in z.files for k in _LEGACY_REQUIRED) else f"ERROR {e}"
            print(f"  {b.name}: {status}")
            continue
        try:
            status = restamp_file(b, data_flag=args.data_flag, out_dir=args.out_dir)
            counts[status] += 1
        except Exception as e:
            counts["error"] += 1
            status = f"ERROR {type(e).__name__}: {e}"
        print(f"  {b.name}: {status}")

    if not args.dry_run:
        print(f"\nrestamped {counts['ok']}, already current {counts['already']}, "
              f"failed {counts['error']}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
