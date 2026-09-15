"""The producer must satisfy its own contract -- provable without a GPU.

kprelogits writes bundles and kprobe (a separate repo) reads them, blind to
each other, agreeing only on ``kprelogits.contracts``. That agreement is
exactly the kind of thing that holds until someone adds a field on one side,
which is why the whole round trip is tested here -- assemble a payload, write
it, read it back and validate -- on synthetic features that need neither timm
nor a rented card.

These tests deliberately do not import the consumer. When kprobe lived in this
repo they called its real loader, which was the stronger test; reaching across
a repo boundary to keep that would be a dependency on the consumer, which is
the thing the contract exists to avoid. The consumer proves it can read the
format on its own side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.contracts import (  # noqa: E402
    PRELOGIT_BUNDLE_SCHEMA_VERSION, ContractError, validate_prelogit_bundle,
)
from kprelogits.extract import (  # noqa: E402
    bundle_key, bundle_payload, bundle_status, partition_scheduled, save_atomic,
)
from kprelogits.models import PREPROCESSING_SPEC, preprocessing_spec  # noqa: E402
from kprelogits.restamp import UNKNOWN_WEIGHTS, restamp_file  # noqa: E402

D, K = 8, 4
SIZES = {"train": 40, "val": 12, "test": 16}


def _features(seed=0):
    rng = np.random.default_rng(seed)
    f = {}
    for split, n in SIZES.items():
        f[f"x_{split}"] = rng.normal(size=(n, D)).astype(np.float32)
        f[f"y_{split}"] = rng.integers(0, K, n)
    return f


def _payload(**kw):
    base = dict(data_flag="octmnist", model_name="synthetic_net.fake_in1k",
                feat_dim=D, num_classes=K, max_train=0, precision="fp32",
                weights_revision="hf:timm/synthetic_net.fake_in1k@abc123")
    return bundle_payload(_features(), **{**base, **kw})


def _write(tmp_path, payload, name="octmnist_synthetic_net_features.npz"):
    p = tmp_path / name
    save_atomic(p, payload)
    return p


def test_payload_satisfies_the_contract(tmp_path):
    p = _write(tmp_path, _payload())
    with np.load(p, allow_pickle=False) as z:
        man = validate_prelogit_bundle(z)
    assert man.schema_version == PRELOGIT_BUNDLE_SCHEMA_VERSION
    assert man.dataset == "octmnist"
    assert man.feat_dim == D and man.num_classes == K
    assert man.preprocessing == PREPROCESSING_SPEC
    assert man.producer_version.startswith("kprelogits/")


def test_a_consumer_gets_the_shapes_and_dtypes_promised(tmp_path):
    """What a reader can rely on beyond the contract's own checks.

    The validator proves the members are present and mutually consistent; it
    does not pin dtypes. A consumer that builds float32 tensors from x and
    indexes labels with y needs those to be float32 and int64 specifically,
    and silently getting float64 features would cost memory on every fit
    without failing anything.
    """
    with np.load(_write(tmp_path, _payload()), allow_pickle=False) as z:
        man = validate_prelogit_bundle(z)
        assert man.feat_dim == D and man.num_classes == K
        assert man.model_name == "synthetic_net.fake_in1k"
        for split, n in SIZES.items():
            assert z[f"x_{split}"].shape == (n, D)
            assert z[f"y_{split}"].shape == (n,)
            assert z[f"x_{split}"].dtype == np.float32
            assert z[f"y_{split}"].dtype == np.int64


def test_fp16_halves_storage_and_still_validates(tmp_path):
    with np.load(_write(tmp_path, _payload(precision="fp16")), allow_pickle=False) as z:
        assert validate_prelogit_bundle(z).precision == "fp16"
        assert z["x_train"].dtype == np.float16


def test_unknown_precision_rejected():
    with pytest.raises(ValueError, match="precision must be"):
        _payload(precision="bf16")


def test_labels_are_flattened():
    """MedMNIST hands back (n, 1) labels; a bundle carrying that shape fails
    the contract's y.ndim check downstream."""
    f = _features()
    f["y_train"] = f["y_train"].reshape(-1, 1)
    p = bundle_payload(f, data_flag="octmnist", model_name="m", feat_dim=D,
                       num_classes=K, max_train=0, precision="fp32",
                       weights_revision="x")
    assert p["y_train"].ndim == 1


def test_declared_feat_dim_must_match_the_arrays(tmp_path):
    p = _write(tmp_path, _payload(feat_dim=D + 1))
    with np.load(p, allow_pickle=False) as z:
        with pytest.raises(ContractError, match="inconsistent with feat_dim"):
            validate_prelogit_bundle(z)


# ---- the preprocessing claim ----------------------------------------------

def test_grayscale_spec_is_frozen():
    """Every bundle in S3 was written with this exact string. Changing it would
    make already-extracted octmnist features look like a different pipeline."""
    assert preprocessing_spec(1) == "resize224x224|gray2rgb|totensor|imagenet_norm"
    assert PREPROCESSING_SPEC == preprocessing_spec(1)


def test_rgb_spec_does_not_claim_a_step_that_never_ran():
    """image_transform expands grayscale to RGB only when n_channels == 1.
    dermamnist is 3-channel, so a bundle claiming gray2rgb would be asserting
    a transform the loader provably skipped -- and nothing downstream checks
    the string against the pipeline, so it would never be caught."""
    assert preprocessing_spec(3) == "resize224x224|totensor|imagenet_norm"
    assert "gray2rgb" not in preprocessing_spec(3)


def test_channel_counts_give_different_specs():
    """The two feature sets are not interchangeable; the manifest must say so."""
    assert preprocessing_spec(1) != preprocessing_spec(3)
    assert _key(preprocessing=preprocessing_spec(1)) != \
        _key(preprocessing=preprocessing_spec(3))


# ---- bundle_key -----------------------------------------------------------

def _key(**kw):
    base = dict(data_flag="octmnist", model_name="m", max_train=10_000,
                precision="fp32", weights_revision="hf:x@1",
                preprocessing=PREPROCESSING_SPEC)
    return bundle_key(**{**base, **kw})


@pytest.mark.parametrize("change", [
    {"max_train": 5_000}, {"precision": "fp16"}, {"weights_revision": "hf:x@2"},
    {"preprocessing": "resize256"}, {"data_flag": "dermamnist"}, {"model_name": "n"},
])
def test_bundle_key_separates_incompatible_content(change):
    """Filenames encode only (dataset, model); this is what distinguishes two
    bundles that would otherwise collide in the shared cache."""
    assert _key() != _key(**change)


def test_bundle_key_is_stable():
    assert _key() == _key()


# ---- atomicity and status -------------------------------------------------

def test_save_atomic_leaves_no_tmp_file(tmp_path):
    p = _write(tmp_path, _payload())
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_bundle_status_classifies(tmp_path):
    assert bundle_status(tmp_path / "absent.npz") == "missing"
    assert bundle_status(_write(tmp_path, _payload())) == "ok"

    truncated = tmp_path / "truncated_features.npz"
    truncated.write_bytes(b"PK\x03\x04 not really a zip")
    assert bundle_status(truncated) == "corrupt"
    # A corrupt bundle is DELETED, not skipped: skipping is how a run finishes
    # "clean" with a hole in it.
    assert not truncated.exists()


# ---- restamp --------------------------------------------------------------

def _legacy(tmp_path):
    """A bundle in the pre-contract schema, as legacy/jobs wrote them."""
    f = _features(1)
    p = tmp_path / "octmnist_legacy_net_features.npz"
    np.savez(p, **f, feat_dim=np.int64(D), num_classes=np.int64(K),
             model_name=np.array("legacy_net"), max_train=np.int64(10_000),
             precision=np.array("fp32"))
    return p


def test_legacy_bundle_is_detected_not_deleted(tmp_path):
    p = _legacy(tmp_path)
    assert bundle_status(p) == "legacy"
    assert p.exists()


def test_restamp_makes_a_legacy_bundle_contract_valid(tmp_path):
    p = _legacy(tmp_path)
    assert restamp_file(p, data_flag="octmnist") == "ok"
    with np.load(p, allow_pickle=False) as z:
        man = validate_prelogit_bundle(z)
    assert man.dataset == "octmnist"
    assert man.model_name == "legacy_net" and man.max_train == 10_000


def test_restamp_does_not_invent_provenance(tmp_path):
    """The timm weights those features came from are genuinely unrecoverable;
    saying so is the point."""
    p = _legacy(tmp_path)
    restamp_file(p, data_flag="octmnist")
    with np.load(p, allow_pickle=False) as z:
        assert str(z["weights_revision"]) == UNKNOWN_WEIGHTS
        assert "legacy/jobs/extract_features.py" in str(z["producer_version"])


def test_restamp_preserves_arrays_exactly(tmp_path):
    p = _legacy(tmp_path)
    with np.load(p, allow_pickle=False) as z:
        before = {k: z[k].copy() for k in
                  ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")}
    restamp_file(p, data_flag="octmnist")
    with np.load(p, allow_pickle=False) as z:
        for k, v in before.items():
            assert np.array_equal(z[k], v) and z[k].dtype == v.dtype


def test_restamp_is_idempotent(tmp_path):
    p = _legacy(tmp_path)
    assert restamp_file(p, data_flag="octmnist") == "ok"
    assert restamp_file(p, data_flag="octmnist") == "already"


def test_a_restamped_bundle_reads_back_as_a_valid_one(tmp_path):
    """Restamping is only worth anything if the result is indistinguishable
    from a freshly extracted bundle to a consumer that only knows the
    contract."""
    p = _legacy(tmp_path)
    restamp_file(p, data_flag="octmnist")
    with np.load(p, allow_pickle=False) as z:
        assert validate_prelogit_bundle(z).model_name == "legacy_net"


def test_restamp_refuses_a_bundle_missing_arrays(tmp_path):
    p = tmp_path / "half_features.npz"
    np.savez(p, x_train=np.zeros((3, D), np.float32), y_train=np.zeros(3, np.int64))
    with pytest.raises(ContractError, match="not restampable"):
        restamp_file(p, data_flag="octmnist")


# ---- what counts as progress ----------------------------------------------

class _Cfg:
    """Just the two methods partition_scheduled needs."""

    def __init__(self, tmp_path):
        self.dir = tmp_path

    def bundle_name(self, model):
        return f"octmnist_{model}_features.npz"

    def bundle_path(self, model):
        return self.dir / self.bundle_name(model)


def test_a_bundle_in_s3_counts_as_done(tmp_path):
    cfg = _Cfg(tmp_path)
    pending, legacy, skipped = partition_scheduled(
        cfg, ["a"], {"octmnist_a_features.npz"})
    assert (pending, legacy, skipped) == ([], [], 1)


def test_a_missing_bundle_is_pending(tmp_path):
    cfg = _Cfg(tmp_path)
    assert partition_scheduled(cfg, ["a"], set()) == (["a"], [], 0)


def test_a_valid_local_bundle_counts_as_done(tmp_path):
    cfg = _Cfg(tmp_path)
    _write(tmp_path, _payload(), name=cfg.bundle_name("a"))
    assert partition_scheduled(cfg, ["a"], set()) == ([], [], 1)


def test_a_legacy_bundle_is_neither_pending_nor_done(tmp_path):
    """The trap this function exists to close. A pre-contract bundle must not
    be re-extracted (the features are fine, and a GPU costs money) but must not
    count as progress either -- otherwise a shard whose every bundle is legacy
    reports done and complete, having uploaded nothing usable, and the early
    return skips the final sync on the way out."""
    cfg = _Cfg(tmp_path)
    p = tmp_path / cfg.bundle_name("a")
    f = _features(1)
    np.savez(p, **f, feat_dim=np.int64(D), num_classes=np.int64(K),
             model_name=np.array("a"), max_train=np.int64(10_000),
             precision=np.array("fp32"))
    pending, legacy, skipped = partition_scheduled(cfg, ["a"], set())
    assert (pending, legacy, skipped) == ([], ["a"], 0)


def test_skipped_never_counts_a_legacy_bundle(tmp_path):
    """is_complete asserts extracted + skipped == scheduled, so an inflated
    skipped count is exactly how a shard claims work it never did."""
    cfg = _Cfg(tmp_path)
    legacy_p = tmp_path / cfg.bundle_name("old")
    f = _features(2)
    np.savez(legacy_p, **f, feat_dim=np.int64(D), num_classes=np.int64(K),
             model_name=np.array("old"), max_train=np.int64(10_000),
             precision=np.array("fp32"))
    _write(tmp_path, _payload(), name=cfg.bundle_name("good"))

    pending, legacy, skipped = partition_scheduled(
        cfg, ["old", "good", "new"], set())
    assert pending == ["new"] and legacy == ["old"] and skipped == 1
    assert skipped + len(pending) + len(legacy) == 3
