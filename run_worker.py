import os, sys
import ssl
import urllib.request

# Disable SSL verification — safe on Vast.ai
ssl._create_default_https_context = ssl._create_unverified_context

# Override Colab-hardcoded paths BEFORE importing ktrain
os.environ.setdefault("DATA_ROOT", "/workspace/data")
os.environ.setdefault("RESULTS_ROOT", "/workspace/results")
sys.path.insert(0, "/workspace")

# Parallelism config — set via Vast.ai env vars
PARALLEL_RANK   = int(os.environ.get("PARALLEL_RANK", "0"))
PARALLEL_SHARDS = int(os.environ.get("PARALLEL_SHARDS", "1"))
DATA_FLAG       = os.environ.get("DATA_FLAG", "dermamnist")

print(f"Worker rank={PARALLEL_RANK}/{PARALLEL_SHARDS}, dataset={DATA_FLAG}")

from ktrain.medmnist.model_comparison.train import (
    filter_timm_models, verify_image_quality, get_loaders,
    run_sweep, print_cost_analysis, SHARED_MODEL_CACHE,
)
from ktrain.medmnist.config import dataset_ckpt_dir, dataset_pred_dir

CKPT_DIR = str(dataset_ckpt_dir(DATA_FLAG))
PRED_DIR = str(dataset_pred_dir(DATA_FLAG))

verify_image_quality(data_flag=DATA_FLAG)
train_loader, val_loader, test_loader, info = get_loaders(data_flag=DATA_FLAG)

model_list = filter_timm_models(max_params=5_000_000, cache_path=SHARED_MODEL_CACHE)
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

# Sync results to S3 if configured
s3 = os.environ.get("S3_BUCKET")
if s3:
    os.system(f"aws s3 sync /workspace/results s3://{s3}/ktrain/results/")
    print(f"Synced to s3://{s3}/ktrain/results/")

