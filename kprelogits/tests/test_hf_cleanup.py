"""Per-model weight cleanup must free the weights, in both hub cache layouts.

Old layout: ``models--org--name/blobs/<sha>`` with ``snapshots/<rev>/file``
symlinking into that model's own blobs dir. Removing the model dir frees it.

New layout (huggingface_hub 1.x): a hub-wide ``blobs/<xx>/<sha>`` store, and
``models--*/snapshots/<rev>/file`` symlinks into it. Removing the model dir
frees only the symlinks -- which is how a 45 GB box filled up on 2026-09-25
with cleanup on. Orphaned blobs must go; blobs another model still links
must stay.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.extract import drop_hf_weights, drop_orphaned_blobs  # noqa: E402

MB = 1024 * 1024


def _old_layout(hub: Path, name: str, size: int) -> Path:
    d = hub / f"models--timm--{name}"
    blob = d / "blobs" / ("a" * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"w" * size)
    snap = d / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    return blob


def _new_layout(hub: Path, name: str, sha: str, size: int) -> Path:
    blob = hub / "blobs" / sha[:2] / sha
    if not blob.exists():
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"w" * size)
    snap = hub / f"models--timm--{name}" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    (hub / f"models--timm--{name}" / "refs").mkdir()
    (hub / f"models--timm--{name}" / "refs" / "main").write_text("rev1")
    return blob


def test_old_layout_is_freed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    hub = tmp_path / "hub"
    blob = _old_layout(hub, "resnet18.a1_in1k", 3 * MB)
    freed = drop_hf_weights("resnet18.a1_in1k")
    assert freed >= 3 * MB
    assert not blob.exists() and not (hub / "models--timm--resnet18.a1_in1k").exists()


def test_new_layout_frees_shared_blob(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    hub = tmp_path / "hub"
    blob = _new_layout(hub, "resnet18.a1_in1k", "b" * 64, 5 * MB)
    freed = drop_hf_weights("resnet18.a1_in1k")
    assert freed >= 5 * MB
    assert not blob.exists(), "the shared blob is the weights; it must go"


def test_blob_still_linked_by_another_model_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    hub = tmp_path / "hub"
    done = _new_layout(hub, "resnet18.a1_in1k", "c" * 64, 2 * MB)
    pending = _new_layout(hub, "resnet50.a1_in1k", "d" * 64, 4 * MB)
    shared = _new_layout(hub, "resnet34.a1_in1k", "e" * 64, 1 * MB)
    # a second model links the same shared blob (identical file content)
    snap = hub / "models--timm--resnet34b.a1_in1k" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    os.symlink(os.path.relpath(shared, snap), snap / "model.safetensors")

    drop_hf_weights("resnet18.a1_in1k")
    assert not done.exists()
    assert pending.exists(), "weights for a model still to come must survive"
    assert shared.exists()

    drop_hf_weights("resnet34.a1_in1k")
    assert shared.exists(), "resnet34b still links it"
    drop_hf_weights("resnet34b.a1_in1k")
    assert not shared.exists()


def test_orphan_sweep_ignores_missing_store_and_partial_downloads(tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    assert drop_orphaned_blobs(hub) == 0          # no blobs dir at all
    (hub / "blobs" / "ff").mkdir(parents=True)
    part = hub / "blobs" / "ff" / ("f" * 64 + ".incomplete")
    part.write_bytes(b"x" * 10)
    # An unreferenced partial file is an orphan too: nothing is mid-download
    # when cleanup runs, so a leftover .incomplete is dead weight.
    assert drop_orphaned_blobs(hub) == 10
    assert not part.exists()


def test_no_hub_env_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert drop_hf_weights("anything") == 0
