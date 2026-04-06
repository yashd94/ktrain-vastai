# ktrain — Vast.ai Docker Runner

Companion tooling for *"Pandora's Regret: Decision-Aligned Evaluation for
Sequential Search"* (Flores, Deshpande, Brea and Wilson, 2026).

Runs the `ktrain` model comparison sweep — linear-probing frozen `timm`
backbones on MedMNIST datasets — on Vast.ai GPU instances instead of
Google Colab.

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the worker image on top of `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` |
| `run_worker.py` | Entrypoint script — replaces the Colab notebook cells |

---

## How It Works

### Dockerfile

- Starts from `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime` (PyTorch,
  CUDA 12.1, and cuDNN pre-matched — no version conflicts)
- Installs system tools (`git`, `curl`, `awscli`) and Python deps
  (`medmnist`, `timm`, `huggingface_hub`, etc.)
- Copies the local `ktrain/` repo and `run_worker.py` into `/workspace`
- Sets `PYTHONPATH=/workspace` to mirror the Colab `sys.path` setup
- Creates `/workspace/data` and `/workspace/results` for runtime I/O

> **Note:** The `ktrain` repo is is included as a submodule.
> On creation, a PAT is used to clone from source.

### run_worker.py

Replaces the Colab notebook entrypoint. On startup it:

1. Overrides the Colab-hardcoded `/content/` paths to `/workspace/data`
   and `/workspace/results` via environment variables
2. Reads `PARALLEL_RANK` and `PARALLEL_SHARDS` from the environment to
   select its shard of the model list (via `_select_model_shard()`)
3. Calls `filter_timm_models()` to discover all compatible pretrained
   `timm` models (≤200M params, 224×224 input) — results are cached to
   avoid repeat Hub round-trips
4. Calls `verify_image_quality()` to confirm MedMNIST is serving genuine
   224×224 images (not nearest-neighbour upsamples from 28×28)
5. Loads train/val/test data via `get_loaders()`
6. Runs `run_sweep()` — for each model: extracts frozen backbone features,
   trains a linear head (20 epochs, AdamW), saves predictions (`.npz`)
   and checkpoint (`.pt`) to the results directory
7. Prints a GPU cost projection via `print_cost_analysis()`
8. Optionally syncs results to S3 if `S3_BUCKET` is set

`skip_existing=True` is always set — runs are **idempotent**. If an
instance is preempted or crashes, relaunch it and it resumes from where
it left off.

---

## Parallelization

Each model is fully independent (no inter-node communication). The sweep
is sharded across N workers by model-list index:

| Worker | `PARALLEL_RANK` | `PARALLEL_SHARDS` | Models assigned |
|---|---|---|---|
| 0 | `0` | `4` | indices 0, 4, 8, 12, … |
| 1 | `1` | `4` | indices 1, 5, 9, 13, … |
| 2 | `2` | `4` | indices 2, 6, 10, 14, … |
| 3 | `3` | `4` | indices 3, 7, 11, 15, … |

Each worker writes its own
`summary_shard{RANK}_of_{SHARDS}.json` to the results directory.

---

## Build and Push

> Requires Docker with `buildx` support. If building on Apple Silicon
> (M1/M2/M3), you **must** use `--platform linux/amd64` — Vast.ai runs
> x86_64 hosts.

```bash
# One-time buildx setup (Apple Silicon / cross-platform builds)
docker buildx create --use --name multibuilder
docker buildx inspect --bootstrap

# Build and push directly to Docker Hub
docker buildx build \
  --platform linux/amd64 \
  -t YOUR_DOCKERHUB_USERNAME/ktrain-worker:latest \
  --push \
  .

