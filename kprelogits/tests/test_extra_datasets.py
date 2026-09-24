"""Non-MedMNIST datasets in MedMNIST npz layout, and empty splits.

dermamniste (DermaMNIST-E, val/test only) is the first: its train split is
empty, and a (0, ...) split must come out as a (0, feat_dim) bundle member
rather than a torch.cat([]) failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits import models as M  # noqa: E402
from kprelogits.contracts import validate_prelogit_bundle  # noqa: E402
from kprelogits.extract import bundle_payload  # noqa: E402

torch = pytest.importorskip("torch")


def _npz(tmp_path, flag="dermamniste", n_train=0, n_val=5, n_test=9):
    rng = np.random.default_rng(0)
    d = {}
    for split, n in {"train": n_train, "val": n_val, "test": n_test}.items():
        d[f"{split}_images"] = rng.integers(0, 255, (n, 224, 224, 3), np.uint8)
        d[f"{split}_labels"] = rng.integers(0, 7, (n, 1)).astype(np.uint8)
    np.savez(tmp_path / f"{flag}_224.npz", **d)
    return tmp_path


def _numpy_transform(_n_channels):
    # torchvision may be absent here; the real transform is not under test.
    return lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float()


def test_extra_dataset_loads_and_empty_train_widens(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "image_transform", _numpy_transform)
    root = _npz(tmp_path)
    tr, va, te, info = M.get_loaders("dermamniste", str(root), max_train=None,
                                     batch_size=4, num_workers=0)
    assert (len(tr.dataset), len(va.dataset), len(te.dataset)) == (0, 5, 9)
    assert len(info["label"]) == 7 and info["n_channels"] == 3

    enc = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    f = M.extract_all_splits(enc, tr, va, te, torch.device("cpu"),
                             model_name="fake", feature_batch_size=4)
    assert tuple(f["x_train"].shape) == (0, 3)
    assert tuple(f["y_train"].shape) == (0,)
    assert tuple(f["x_val"].shape) == (5, 3) and tuple(f["x_test"].shape) == (9, 3)
    assert f["y_test"].dtype == torch.int64

    p = bundle_payload(f, data_flag="dermamniste", model_name="fake", feat_dim=3,
                       num_classes=7, max_train=0, precision="fp16",
                       weights_revision="x", preprocessing=M.preprocessing_spec(3))
    out = tmp_path / "b.npz"
    np.savez(out, **p)
    man = validate_prelogit_bundle(np.load(out))
    assert man.dataset == "dermamniste" and man.max_train == 0


def test_unknown_flag_still_goes_to_medmnist(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "image_transform", _numpy_transform)
    monkeypatch.setitem(sys.modules, "medmnist", None)  # import must fail loudly
    with pytest.raises(ImportError):
        M.get_loaders("octmnist", str(tmp_path), num_workers=0)
