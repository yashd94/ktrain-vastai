import os, sys
os.environ.setdefault("DATA_ROOT", "/workspace/data")
os.environ.setdefault("RESULTS_ROOT", "/workspace/results")
sys.path.insert(0, "/workspace")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

PARALLEL_RANK   = int(os.environ.get("PARALLEL_RANK", "0"))
PARALLEL_SHARDS = int(os.environ.get("PARALLEL_SHARDS", "1"))
DATA_FLAG       = os.environ.get("DATA_FLAG", "breakhis")

# --- BreakHis adapter (only used when DATA_FLAG == "breakhis") ---
if DATA_FLAG == "breakhis":
    from ktrain.medmnist.breakhis_adapter import BreakHisDataset, BREAKHIS_INFO
    from torch.utils.data import DataLoader
    from torchvision import transforms

    def get_loaders(data_flag="breakhis", root="/workspace/data", max_train=None):
        n_channels = 3
        tx = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])
        kw = dict(transform=tx, root=root, size=224)
        train_ds = BreakHisDataset(split="train", **kw)
        val_ds   = BreakHisDataset(split="val",   **kw)
        test_ds  = BreakHisDataset(split="test",  **kw)

        ldr_kw = dict(batch_size=128, shuffle=False,
                      num_workers=4, pin_memory=True, prefetch_factor=4)
        return (
            DataLoader(train_ds, **ldr_kw),
            DataLoader(val_ds,   **ldr_kw),
            DataLoader(test_ds,  **ldr_kw),
            BREAKHIS_INFO,
        )
else:
    from ktrain.medmnist.model_comparison.train import get_loaders

# --- Rest of run_worker.py unchanged ---
from ktrain.medmnist.model_comparison.train import (
    filter_timm_models, verify_image_quality,
    run_sweep, print_cost_analysis, SHARED_MODEL_CACHE,
)
from ktrain.medmnist.config import dataset_ckpt_dir, dataset_pred_dir

info = BREAKHIS_INFO if DATA_FLAG == "breakhis" else None
CKPT_DIR = str(dataset_ckpt_dir(DATA_FLAG))
PRED_DIR = str(dataset_pred_dir(DATA_FLAG))

# Skip verify_image_quality for BreakHis (MedMNIST-specific check)
if DATA_FLAG != "breakhis":
    verify_image_quality(data_flag=DATA_FLAG)

train_loader, val_loader, test_loader, info = get_loaders(
    data_flag=DATA_FLAG, root="/workspace/data"
)

model_list  = filter_timm_models(max_params=200_000_000, cache_path=SHARED_MODEL_CACHE)
model_names = [m["name"] for m in model_list]
print(f"Total models: {len(model_names)}")

summaries = run_sweep(
    model_names,
    train_loader, val_loader, test_loader, info,
    ckpt_dir=CKPT_DIR,
    pred_dir=PRED_DIR,
    data_flag=DATA_FLAG,
    skip_existing=True,
    parallel_shards=PARALLEL_SHARDS,
    parallel_rank=PARALLEL_RANK,
    timing=True,
)

print_cost_analysis(summaries, total_models=len(model_names))

s3 = os.environ.get("S3_BUCKET")
if s3:
    os.system(f"aws s3 sync /workspace/results s3://{s3}/{DATA_FLAG}/results/")
