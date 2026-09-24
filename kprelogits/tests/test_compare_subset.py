"""compare_subset: a full-split bundle checked against its 10k sibling.

The check is only worth anything if it fails on the ways a full-split run can
go wrong -- rows in the wrong order, a subsample after all, different weights
-- and passes on GPU noise. Synthetic bundles, built with the real payload
code and the real subsample indices.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

pytest.importorskip("torch")

from kprelogits import compare_subset  # noqa: E402
from kprelogits.extract import bundle_payload, save_atomic  # noqa: E402
from kprelogits.models import train_subset_indices  # noqa: E402

D, K = 5, 4
N_TRAIN, N_SUB, N_VAL, N_TEST = 60, 20, 7, 9


def _full(rng):
    return {"x_train": rng.normal(size=(N_TRAIN, D)).astype(np.float32),
            "y_train": rng.integers(0, K, N_TRAIN),
            "x_val": rng.normal(size=(N_VAL, D)).astype(np.float32),
            "y_val": rng.integers(0, K, N_VAL),
            "x_test": rng.normal(size=(N_TEST, D)).astype(np.float32),
            "y_test": rng.integers(0, K, N_TEST)}


def _save(path, f, *, max_train, weights="hf:x@1"):
    save_atomic(path, bundle_payload(
        f, data_flag="octmnist", model_name="m", feat_dim=D, num_classes=K,
        max_train=max_train, precision="fp32", weights_revision=weights))
    return path


def _pair(tmp_path, *, seed=42, weights="hf:x@1", full_max_train=0, mutate=None):
    f = _full(np.random.default_rng(0))
    idx = train_subset_indices(N_TRAIN, N_SUB, seed)
    sub = {**f, "x_train": f["x_train"][idx], "y_train": f["y_train"][idx]}
    if mutate:
        f = mutate(f)
    return (_save(tmp_path / "full.npz", f, max_train=full_max_train),
            _save(tmp_path / "sub.npz", sub, max_train=N_SUB, weights=weights))


# ---- the indices ----------------------------------------------------------

def test_indices_are_the_seeded_permutation_the_extractor_always_used():
    import torch
    g = torch.Generator().manual_seed(42)
    assert train_subset_indices(97_477, 10_000) == \
        torch.randperm(97_477, generator=g)[:10_000].tolist()


def test_no_subsample_means_no_indices():
    assert train_subset_indices(100, 0) is None
    assert train_subset_indices(100, None) is None
    assert train_subset_indices(100, 100) is None      # DermaMNIST's case


# ---- the comparison -------------------------------------------------------

def test_the_full_split_extension_of_a_subset_passes(tmp_path):
    assert compare_subset.compare(*_pair(tmp_path))


def test_gpu_noise_passes(tmp_path):
    rng = np.random.default_rng(1)

    def jitter(f):
        return {k: (v * (1 + rng.normal(scale=1e-4, size=v.shape))).astype(np.float32)
                if k.startswith("x_") else v for k, v in f.items()}
    assert compare_subset.compare(*_pair(tmp_path, mutate=jitter))


def test_shuffled_train_rows_fail_on_labels(tmp_path, capsys):
    """The failure only this check can see: every row present, in the wrong
    order, so features and labels are each plausible and the pairing is not."""
    def shuffle(f):
        p = np.random.default_rng(2).permutation(N_TRAIN)
        return {**f, "x_train": f["x_train"][p], "y_train": f["y_train"][p]}
    assert not compare_subset.compare(*_pair(tmp_path, mutate=shuffle))
    assert "labels differ" in capsys.readouterr().out


def test_a_subset_drawn_with_another_seed_fails(tmp_path):
    assert not compare_subset.compare(*_pair(tmp_path, seed=7))


def test_a_bundle_that_is_not_full_split_fails(tmp_path, capsys):
    assert not compare_subset.compare(*_pair(tmp_path, full_max_train=50))
    assert "NOT FULL" in capsys.readouterr().out


def test_different_weights_fail(tmp_path, capsys):
    assert not compare_subset.compare(*_pair(tmp_path, weights="hf:x@2"))
    assert "weights_revision" in capsys.readouterr().out


def test_different_features_fail(tmp_path):
    def replace(f):
        return {**f, "x_val": np.random.default_rng(3).normal(
            size=(N_VAL, D)).astype(np.float32)}
    assert not compare_subset.compare(*_pair(tmp_path, mutate=replace))


def test_non_finite_features_fail(tmp_path, capsys):
    def poison(f):
        x = f["x_train"].copy()
        x[-1, 0] = np.nan
        return {**f, "x_train": x}
    assert not compare_subset.compare(*_pair(tmp_path, mutate=poison))
    assert "NON-FINITE" in capsys.readouterr().out


def test_the_cli_exit_code_reflects_every_pair(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    good = _pair(tmp_path / "a")
    bad = _pair(tmp_path / "b", weights="hf:x@2")
    assert compare_subset.main([str(p) for p in good]) == 0
    assert compare_subset.main([str(p) for p in (*good, *bad)]) == 1
