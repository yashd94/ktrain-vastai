"""Zero-width features: produced by a timm bug, invisible to every other check.

timm's MlpClassifierHead (InceptionNeXt) builds ``Linear(hidden, 0)`` when
constructed with num_classes=0, so four backbones shipped bundles whose x
arrays are (n, 0) -- labels intact, feat_dim=0, shape-consistent, and useless.
They passed the contract, the manifest audit and the size audit, because every
one of those checks asks whether the parts agree, and they did.

build_encoder is exercised with a fake timm (the real one is not installed
here) and real torch where available.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.contracts import ContractError, validate_prelogit_bundle  # noqa: E402
from kprelogits.extract import bundle_payload, save_atomic  # noqa: E402


# ---- the contract ---------------------------------------------------------

def _zero_width_bundle(tmp_path):
    rng = np.random.default_rng(0)
    feats = {}
    for split, n in {"train": 20, "val": 6, "test": 8}.items():
        feats[f"x_{split}"] = np.zeros((n, 0), np.float32)
        feats[f"y_{split}"] = rng.integers(0, 4, n)
    p = tmp_path / "octmnist_inception_next_atto.sail_in1k_features.npz"
    save_atomic(p, bundle_payload(feats, data_flag="octmnist",
                                  model_name="inception_next_atto.sail_in1k",
                                  feat_dim=0, num_classes=4, max_train=0,
                                  precision="fp32", weights_revision="hf:x@1"))
    return p


def test_the_contract_rejects_a_bundle_with_no_feature_columns(tmp_path):
    with np.load(_zero_width_bundle(tmp_path), allow_pickle=False) as z:
        with pytest.raises(ContractError, match="feat_dim=0"):
            validate_prelogit_bundle(z)


# ---- build_encoder --------------------------------------------------------

torch = pytest.importorskip("torch")
from kprelogits import models as M  # noqa: E402


class _MlpHead(torch.nn.Module):
    """Mimics timm's MlpClassifierHead: fc2 is Linear(hidden, 0) when built
    headless, and reset() swaps it for Identity."""

    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 6)
        self.fc2 = torch.nn.Linear(6, num_classes)

    def forward(self, x):
        return self.fc2(self.fc1(x.mean(dim=(2, 3))))


class _Broken(torch.nn.Module):
    def __init__(self, num_classes, still_zero=False):
        super().__init__()
        self.head = _MlpHead(num_classes)
        self.resets, self.still_zero = 0, still_zero

    def forward(self, x):
        return self.head(x)

    def reset_classifier(self, num_classes, global_pool=None):
        self.resets += 1
        if not self.still_zero:
            self.head.fc2 = torch.nn.Identity()


class _Working(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.fc(x.mean(dim=(2, 3)))

    def reset_classifier(self, *a, **k):
        raise AssertionError("a model that already yields features must not be reset")


def _fake_timm(monkeypatch, model):
    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(
        create_model=lambda *a, **k: model))
    return model


def test_a_zero_width_head_is_reset_to_its_pre_logits(monkeypatch):
    m = _fake_timm(monkeypatch, _Broken(num_classes=0))
    enc, d = M.build_encoder("inception_next_atto.sail_in1k", torch.device("cpu"))
    assert d == 6 and m.resets == 1


def test_a_model_that_already_yields_features_is_left_exactly_as_built(monkeypatch):
    """The fix must not touch the 837 architectures that worked, or bundles
    extracted after it stop being comparable with those extracted before."""
    _fake_timm(monkeypatch, _Working())
    enc, d = M.build_encoder("coat_tiny.in1k", torch.device("cpu"))
    assert d == 5


def test_still_zero_after_reset_is_a_hard_failure(monkeypatch):
    """Width 0 is the one thing no later check can catch."""
    _fake_timm(monkeypatch, _Broken(num_classes=0, still_zero=True))
    with pytest.raises(RuntimeError, match="zero-width"):
        M.build_encoder("some_future_model", torch.device("cpu"))
