"""The prelogit bundle format: what ``kprelogits`` writes, per (dataset,
backbone), as frozen pre-logit features for train/val/test.

This module is the *specification* of that format, and it lives here because
kprelogits is the producer. It used to live in a third package (``kmetrics``)
on the theory that a format shared by two packages should sit on neutral
ground, owned by neither. That held while kmetrics sat beside both. It stopped
holding when kmetrics moved into the consumer's repo: importing it from here
would make the producer of the format depend on the consumer's library, which
is the coupling the neutral-ground rule existed to prevent, merely inverted.

So the producer owns the spec and a consumer keeps a reader. If the two ever
drift, the consumer refuses a bundle at load time with a message naming the
member at fault -- loud and early, which is the failure mode worth having.

A "schema version" is a single integer, bumped on any breaking change to
required keys, shapes, or dtypes. Consumers must check it before trusting a
file; ``validate_prelogit_bundle`` does the check and raises ``ContractError``
naming exactly what is wrong, because a bundle written by an old or diverged
producer must fail loudly rather than silently mis-score.

Provenance: the prelogit half of ``kmetrics/contracts.py`` @ ``29fefa2``,
carried over unchanged. The ProbeArtifact half did not come with it -- it is
written and read by kprobe alone, and kprelogits never touched it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRELOGIT_BUNDLE_SCHEMA_VERSION = 1


class ContractError(ValueError):
    """A bundle does not satisfy its declared schema version."""


# Required .npz keys and the dtype family each must satisfy. Per-split arrays
# (x_train/x_val/x_test, y_train/y_val/y_test) are required; test is a full
# required member even though not every consumer uses it today, because a
# bundle is the frozen, reusable asset and "we forgot to freeze test" is
# exactly the kind of gap that forces a costly re-extraction later.
PRELOGIT_REQUIRED_ARRAYS = (
    "x_train", "y_train", "x_val", "y_val", "x_test", "y_test",
)
PRELOGIT_REQUIRED_SCALARS = (
    "schema_version",   # int: PRELOGIT_BUNDLE_SCHEMA_VERSION
    "dataset",           # str: "octmnist" | "dermamnist" | ...
    "model_name",         # str: the timm backbone name, unsanitized
    "feat_dim",            # int
    "num_classes",           # int
    "max_train",               # int, 0 = full split
    "precision",                 # str: "fp32" | "fp16"
    "weights_revision",            # str: pretrained-weight identity (e.g. a
                                     #      timm/HF tag or content hash) --
                                     #      REQUIRED so two extractions of the
                                     #      same model name are distinguishable
    "preprocessing",                 # str: short description (resize/crop/
                                       #      normalize spec) or a hash of it
    "producer_version",                # str: kprelogits package/commit identity
)


@dataclass(frozen=True)
class PrelogitManifest:
    """The scalar half of a prelogit bundle -- everything except the arrays.

    Embedded in every bundle (as the scalar .npz members above) so a consumer
    can validate provenance without loading the feature arrays.
    """
    dataset: str
    model_name: str
    feat_dim: int
    num_classes: int
    max_train: int
    precision: str
    weights_revision: str
    preprocessing: str
    producer_version: str
    schema_version: int = PRELOGIT_BUNDLE_SCHEMA_VERSION


def validate_prelogit_bundle(npz: Any, *, strict_version: bool = True) -> PrelogitManifest:
    """Validate an opened ``np.load(...)`` bundle; return its manifest.

    Raises ``ContractError`` on any missing key, wrong schema version (if
    ``strict_version``), or a shape mismatch among the per-split arrays.
    """
    files = set(npz.files)
    missing = set(PRELOGIT_REQUIRED_ARRAYS) - files
    missing |= set(PRELOGIT_REQUIRED_SCALARS) - files
    if missing:
        raise ContractError(f"prelogit bundle missing required members: {sorted(missing)}")

    version = int(npz["schema_version"])
    if strict_version and version != PRELOGIT_BUNDLE_SCHEMA_VERSION:
        raise ContractError(
            f"prelogit bundle schema_version={version}, expected "
            f"{PRELOGIT_BUNDLE_SCHEMA_VERSION}. Re-extract, or pass "
            f"strict_version=False if you have specifically handled the diff."
        )

    feat_dim = int(npz["feat_dim"])
    if feat_dim < 1:
        # Every other check here is a consistency check, and a bundle with no
        # feature columns is perfectly consistent: x is (n, 0), feat_dim is 0.
        # Four InceptionNeXt bundles shipped exactly like that.
        raise ContractError(f"feat_dim={feat_dim}: a bundle with no feature "
                            f"columns carries nothing to fit on")
    for split in ("train", "val", "test"):
        x, y = npz[f"x_{split}"], npz[f"y_{split}"]
        if x.ndim != 2 or x.shape[1] != feat_dim:
            raise ContractError(
                f"x_{split} shape {x.shape} inconsistent with feat_dim={feat_dim}"
            )
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ContractError(
                f"y_{split} shape {y.shape} inconsistent with x_{split} shape {x.shape}"
            )

    return PrelogitManifest(
        dataset=str(npz["dataset"]),
        model_name=str(npz["model_name"]),
        feat_dim=feat_dim,
        num_classes=int(npz["num_classes"]),
        max_train=int(npz["max_train"]),
        precision=str(npz["precision"]),
        weights_revision=str(npz["weights_revision"]),
        preprocessing=str(npz["preprocessing"]),
        producer_version=str(npz["producer_version"]),
        schema_version=version,
    )
