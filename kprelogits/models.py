"""Loaders, encoders, and feature extraction. Owned code.

Copied out of ``ktrain/medmnist/model_comparison/train.py`` and adopted: the
underscore-private helpers became public names, the Colab-era module globals
became explicit arguments, and the parts that only mattered for head training
were dropped. **This is now canonical.** kprelogits does not track ktrain, and
divergence between the two is expected rather than a bug to be reconciled --
which is what lets kprelogits spin out with no submodule.

Two behaviours are preserved deliberately, because they were learned the hard
way in the original:

  *Measure feat_dim, never trust it.* ``encoder.num_features`` is wrong for
  some backbones (mobilenetv3), so we push a zero batch through and read the
  shape. A wrong feat_dim silently produces a bundle kprobe would happily fit.

  *Halve the batch on OOM and retry.* A sweep across a hundred backbones will
  meet models that do not fit at batch 128 on the rented card. Failing the
  model costs the whole rental slot; retrying at 64, 32, ... costs seconds.

``timm``, ``medmnist`` and ``torchvision`` are imported inside the functions
that need them, so the module (and everything in ``config.py`` that touches
``select_model_shard`` / ``safe_name``) imports fine on a laptop with none of
them installed.
"""

from __future__ import annotations

import contextlib
import io
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The preprocessing string stamped into every bundle. Changing ANY of the
# transform below must change this string, or two incompatible feature sets
# become indistinguishable after the fact.
PREPROCESSING_SPEC = "resize224x224|gray2rgb|totensor|imagenet_norm"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = (3, 224, 224)


def safe_name(model_name: str) -> str:
    """Filesystem-safe form of a timm model name (they may contain ``/``)."""
    return model_name.replace("/", "__")


def select_model_shard(model_names: Sequence[str], shards: Optional[int] = 1,
                       rank: int = 0) -> List[str]:
    """Split a model list across workers by index modulo.

    Round-robin rather than contiguous blocks so that a systematic ordering
    (the selection is sorted by name, and names correlate with size) spreads
    evenly instead of putting every large model on one box.
    """
    if shards in (None, 0, 1):
        return list(model_names)
    if shards < 1:
        raise ValueError("shards must be >= 1")
    if rank < 0 or rank >= shards:
        raise ValueError(f"rank must be in [0, {shards}), got {rank}")
    return [n for i, n in enumerate(model_names) if i % shards == rank]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def image_transform(n_channels: int):
    """The one preprocessing pipeline. Mirrors ``PREPROCESSING_SPEC``."""
    from torchvision import transforms

    tx = [transforms.Resize((224, 224))]
    if n_channels == 1:
        # timm backbones expect 3-channel input; MedMNIST grayscale sets are
        # expanded rather than the first conv being rewritten.
        tx.append(transforms.Grayscale(num_output_channels=3))
    tx += [transforms.ToTensor(),
           transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    return transforms.Compose(tx)


def get_loaders(data_flag: str, root: str, *, max_train: Optional[int] = 10_000,
                batch_size: int = 128, num_workers: int = 4,
                prefetch_factor: int = 4, seed: int = 42):
    """Non-shuffled loaders over all three splits, plus the dataset info dict.

    Built ONCE per run and reused for every backbone. The loss_comparison code
    this replaces rebuilt them per model, re-reading a multi-GB npz each time.

    ``max_train`` subsamples the train split with a seeded permutation, so the
    same subsample is shared by every backbone and by every later re-extraction
    -- otherwise bundles would not be comparable across runs.
    """
    import os

    import medmnist
    import torch
    from medmnist import INFO
    from torch.utils.data import DataLoader

    os.makedirs(root, exist_ok=True)
    ds_info = INFO[data_flag]
    DataClass = getattr(medmnist, ds_info["python_class"])
    tx = image_transform(int(ds_info.get("n_channels", 3)))

    kw = dict(transform=tx, download=True, size=224, root=root)
    with contextlib.redirect_stderr(io.StringIO()):
        train_ds = DataClass(split="train", **kw)
        val_ds = DataClass(split="val", **kw)
        test_ds = DataClass(split="test", **kw)

    full_train = len(train_ds)
    if max_train and full_train > max_train:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(full_train, generator=g)[:max_train]
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
        print(f"  subsampled train: {len(train_ds)}/{full_train} (seed {seed})")

    ldr_kw: Dict[str, Any] = dict(batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        ldr_kw["prefetch_factor"] = prefetch_factor
    loaders = tuple(DataLoader(ds, **ldr_kw) for ds in (train_ds, val_ds, test_ds))
    print(f"  {data_flag}: train={len(loaders[0].dataset)} "
          f"val={len(loaders[1].dataset)} test={len(loaders[2].dataset)} "
          f"classes={len(ds_info['label'])}")
    return (*loaders, ds_info)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def build_encoder(model_name: str, device):
    """Frozen, eval-mode, globally-pooled encoder + its MEASURED feature width."""
    import timm
    import torch

    with contextlib.redirect_stderr(io.StringIO()):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unexpected keys.*pretrained weights")
            encoder = timm.create_model(model_name, pretrained=True,
                                        num_classes=0, global_pool="avg")
    encoder.eval().to(device)
    for p in encoder.parameters():
        p.requires_grad = False

    with torch.inference_mode():
        feat_dim = encoder(torch.zeros(1, *INPUT_SIZE, device=device)).shape[-1]
    return encoder, int(feat_dim)


def weights_revision(model_name: str) -> str:
    """Identity of the pretrained weights, for the bundle manifest.

    The model name alone is not enough: timm re-points a name at new weights,
    and two bundles extracted months apart under one name would otherwise be
    indistinguishable. Resolves to the Hub repo plus its commit sha when the
    network allows, degrading to the repo id, then the config's URL, then
    ``unknown`` -- each step still more specific than the name.
    """
    try:
        import timm
        cfg = timm.get_pretrained_cfg(model_name)
    except Exception:
        return "unknown"

    hf_id = getattr(cfg, "hf_hub_id", None)
    tag = getattr(cfg, "tag", None)
    if hf_id:
        try:
            from huggingface_hub import model_info
            sha = getattr(model_info(hf_id), "sha", None)
            if sha:
                return f"hf:{hf_id}@{sha[:12]}"
        except Exception:
            pass
        return f"hf:{hf_id}" + (f"#{tag}" if tag else "")
    url = getattr(cfg, "url", None)
    return f"url:{url}" if url else "unknown"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_split(encoder, loader, device, *, split_name: str = "",
                  model_name: str = "", initial_batch_size: Optional[int] = None,
                  timing: bool = False):
    """Run the encoder over one split, halving the batch on CUDA OOM.

    Returns ``(features, labels, timing_stats|None)`` as CPU float32 / int64
    tensors. AMP is used on CUDA only; features are cast back to float32 before
    leaving the GPU so precision is a bundle-level decision, not an artifact of
    where it ran.
    """
    import torch
    from torch.utils.data import DataLoader

    batch_size = initial_batch_size or loader.batch_size or 128
    dataset = loader.dataset
    use_amp = device.type == "cuda"
    tag = f" {split_name}" if split_name else ""

    while batch_size >= 1:
        feats, labels = [], []
        t_data = t_forward = 0.0
        kw: Dict[str, Any] = dict(batch_size=batch_size, shuffle=False,
                                  num_workers=loader.num_workers,
                                  pin_memory=loader.pin_memory)
        if loader.num_workers > 0 and getattr(loader, "prefetch_factor", None):
            kw["prefetch_factor"] = loader.prefetch_factor
        retry_loader = DataLoader(dataset, **kw)
        try:
            with torch.inference_mode():
                t_prev = time.perf_counter()
                for x, y in retry_loader:
                    t_ready = time.perf_counter()
                    t_data += t_ready - t_prev
                    x = x.to(device, non_blocking=True)
                    y = y.view(-1).long()   # .squeeze(1) is unsafe at batch 1
                    with torch.autocast(device_type=device.type,
                                        dtype=torch.float16, enabled=use_amp):
                        f = encoder(x)
                    feats.append(f.float().cpu())
                    labels.append(y.cpu())
                    if timing and device.type == "cuda":
                        torch.cuda.synchronize()
                    t_prev = time.perf_counter()
                    t_forward += t_prev - t_ready
            stats = ({"data": t_data, "forward": t_forward, "batch_size": batch_size}
                     if timing else None)
            return torch.cat(feats), torch.cat(labels), stats
        except torch.cuda.OutOfMemoryError:
            del feats, labels, retry_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size //= 2
            if batch_size >= 1:
                print(f"    {model_name or 'model'} extract{tag} OOM, "
                      f"retrying at batch={batch_size}", flush=True)

    raise RuntimeError(f"OOM during extraction{tag} even at batch_size=1")


def extract_all_splits(encoder, train_loader, val_loader, test_loader, device, *,
                       model_name: str = "", feature_batch_size: Optional[int] = None,
                       timing: bool = True) -> Dict[str, Any]:
    """Extract all three splits; TAKES OWNERSHIP of ``encoder``.

    The encoder is deleted and the CUDA cache emptied on both success and
    failure. Without that, a failed model's backbone stays resident and the
    next ``build_encoder`` briefly holds two in VRAM -- which turns one OOM
    into a cascade.
    """
    import torch

    kw = dict(model_name=model_name, initial_batch_size=feature_batch_size,
              timing=timing)
    try:
        t0 = time.perf_counter()
        x_train, y_train, t_tr = extract_split(encoder, train_loader, device,
                                               split_name="train", **kw)
        x_val, y_val, t_va = extract_split(encoder, val_loader, device,
                                           split_name="val", **kw)
        x_test, y_test, t_te = extract_split(encoder, test_loader, device,
                                             split_name="test", **kw)
        t1 = time.perf_counter()
    finally:
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dict(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                x_test=x_test, y_test=y_test,
                split_timings=(t_tr, t_va, t_te), seconds=t1 - t0)


def pick_device():
    """CUDA if present, else MPS, else CPU. Extraction on CPU is viable only
    for smoke tests -- the caller is expected to warn."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device) -> str:
    import torch
    if device.type == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "cuda"
    return device.type
