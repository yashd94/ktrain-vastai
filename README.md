# ktrain-vastai

Companion tooling for *"Pandora's Regret: Decision-Aligned Evaluation for
Sequential Search"* (Flores, Deshpande, Brea and Wilson, 2026).

This repo is now **`kprelogits`** and nothing else — the rented-GPU feature
extraction stage. Everything it grew out of has moved to its own directory
beside it. See [docs/restructure.md](docs/restructure.md) for what moved where
and why.

---

## Layout

```
kprelogits/   Vast.ai GPU feature extraction. Writes frozen prelogit bundles
              to S3, stamped with the contract in kprelogits/contracts.py and
              validated on read. Typed config + preflight, per-shard state
              files, instance teardown, HF-token staging.
              See kprelogits/README.md.

docs/        restructure.md   — what moved where, and why
             restructure-plan.md — the original plan (historical)

feature_cache/  local prelogit bundles.
```

That is the whole repo. `kprelogits` is a leaf: it imports numpy, torch, timm,
medmnist, and the stdlib, and nothing else — no sibling package, no submodule.

### Moved out

| where | what it is |
|---|---|
| `../kprobe` | linear probing: CE / label smoothing / focal / MSE / Pandora arms, plus temperature / Platt / prevalence calibration. Carries its own copy of `kmetrics`. |
| `../ksim` | component-level diagnostic cost simulators — the multiplex simulator and the imperfect-test DP. Standalone. |
| `../legacy` | the original OCTMNIST study: `jobs/`, `scripts/`, its docs, `out/`, and the 82M of `artifacts/`. **Unversioned** — this repo's history at `bc0f669^` is the only version-controlled copy, so don't rewrite it. |
| `../ktrain` | the paper's archived Colab-first reference implementation, previously a submodule pinned here at `2de9717`. |

`ktrain` was a submodule for two reasons, and both left with `legacy/`:
`legacy/jobs/*` imported it, and `legacy/Dockerfile` did
`COPY ktrain/ /workspace/ktrain/` so the worker image could resolve
`import ktrain.medmnist...`. Nothing in `kprelogits` ever imported it — the
extraction helpers were copied out and adopted precisely to sever that. What
remains are provenance citations in prose, which need the commit id, not the
files. (One leftover: `select.py`'s `S3_PREFIX` still defaults to
`ktrain/results`. That prefix does not exist in the bucket — bundles live under
`medmnist_prelogits/` — so it is a dead default, not a live path. Pass
`--s3-uri` or set `S3_PREFIX` if you want the selection document uploaded.)

## Dependencies

`kprelogits` imports **nothing from the other repos** — numpy, torch, timm,
medmnist, and the stdlib, and that is all. Its entire outward interface is the
bundle format in [kprelogits/contracts.py](kprelogits/contracts.py).

That file is the specification of what kprelogits writes, and it lives with the
producer deliberately. It used to sit in a third package (`kmetrics`) on the
theory that a format shared by two packages belongs to neither — sound while
`kmetrics` sat beside both, and void once it moved into the consumer's repo,
since importing it from here would make the producer depend on the consumer.
So: the producer owns the spec, a consumer keeps a reader, and if the two ever
drift the consumer refuses a bundle at load time naming the member at fault.

Nothing in `kprelogits` imports from the study code or from `ktrain`, full stop
— not even at test time. That was true before those directories left, which is
why they could leave.

## Running it

Every command below was run as written. Config is **environment-driven** with
CLI overrides, and none of it is optional — the defaults point at `/workspace`,
because the extractor's home is a rented box, not a laptop.

| variable | what it does | default |
|---|---|---|
| `S3_BUCKET` | where bundles go; unset means local-only | *(none)* |
| `S3_FEATURES_PREFIX` | prefix under the bucket | `medmnist_prelogits` |
| `DATA_FLAG` | dataset | `octmnist` |
| `MAX_TRAIN` | train subsample, 0 = full split | `10000` |
| `DATA_DIR` / `RESULTS_DIR` | where MedMNIST lands / where bundles are written | `/workspace/data`, `/workspace/results` |
| `PARALLEL_RANK` / `PARALLEL_SHARDS` | round-robin shard of the model list | `0` / `1` |
| `FEATURE_PRECISION` | `fp32` or `fp16` | `fp32` |
| `CLEANUP` | `1` deletes each bundle after confirmed upload | `0` |
| `FEATURE_BATCH_SIZE`, `NUM_WORKERS` | loader tuning | `128`, `4` |
| `HF_TOKEN_FILE`, `HF_HOME` | a *path* to the token, never the token | *(none)* |

The live bucket is
`pandora-linear-probe-inputs-939723541836-us-east-1-an`, holding
`medmnist_prelogits/octmnist/` (841 bundles), the pre-staged datasets under
`medmnist/`, an HF weight cache under `hf_cache/`, and
`timm_model_param_cache.json` at the root.

**1. Choose backbones.** The param cache is not in this repo; fetch it:

```bash
aws s3 cp s3://pandora-linear-probe-inputs-939723541836-us-east-1-an/timm_model_param_cache.json .
python -m kprelogits.select --from-cache --cache timm_model_param_cache.json \
    --max-params 200000000 --stride 1 --output results/_shared/selected_models.json
```

`--stride 1` reproduces the 865-model corpus that produced the existing
bundles. A larger stride subsamples it — `--stride 50` gives 18 models, which
is useful for a cheap trial and is *not* what is in S3.

**2. Preflight, locally, before renting anything.** It reports every problem at
once and touches no GPU:

```bash
S3_BUCKET=pandora-linear-probe-inputs-939723541836-us-east-1-an \
DATA_DIR=./data RESULTS_DIR=./results \
python -m kprelogits.extract --selection results/_shared/selected_models.json --dry-run
```

Add `--no-probe` to skip the S3 and HuggingFace checks and work fully offline.

**3. Extract.** On the box, with credentials staged and the dataset present:

```bash
python -m kprelogits.extract --selection /workspace/selection.json \
    --shards 4 --rank 0 --cleanup
```

Pull the dataset from S3 first rather than letting MedMNIST fetch it from
source — it is already staged, and a slow origin download inside the worker is
indistinguishable from a wedged job:

```bash
aws s3 cp s3://.../medmnist/octmnist_224.npz /workspace/data/octmnist_224.npz
```

### Getting onto the box

There is **one automated path, and it runs a single backbone**:
`python -m kprelogits.ops.smoke --go` (below). It rents, stages code and
credentials, launches, polls S3 state, validates, and tears down.

For a multi-shard sweep there is **no driver yet** — `ops/vast.py` has every
piece (`search_offers`, `create`, `attach_ssh_key`, `ssh_endpoints`,
`stage_credentials`, `teardown`, `preteardown_check`) and `smoke.py` composes
them for one box, but nothing fans that out over N shards and waits on all of
them. Generalising `smoke.py` is the next piece of work; until then a sweep
means driving `smoke.py`'s sequence by hand per shard, with
`preteardown_check(prefix, expected_shards=N, expected_bundles=M)` as the gate
before destroying anything.

### Restamping what is already there

The 841 bundles in S3 predate `contracts.py` and record only `feat_dim`,
`max_train`, `model_name`, `num_classes`, `precision` — no `weights_revision`,
no `preprocessing`, no producer. `restamp` fixes that with no GPU, but works on
a **local directory**, so bundles have to come down and go back up:

```bash
aws s3 sync s3://.../medmnist_prelogits/octmnist/ ./bundles/
python -m kprelogits.restamp --dir ./bundles --data-flag octmnist
aws s3 sync ./bundles/ s3://.../medmnist_prelogits/octmnist/
```

That is ~5 GB down and up. Restamping does not invent provenance — it records
`weights_revision="unknown:pre-contract"`, which documents the absence rather
than papering over it. Re-extracting is the only way to *earn* real
provenance; see Status for what one such comparison found.

## The worker image

```bash
docker build -t <dockerhub-user>/kprelogits-worker:latest .
docker push  <dockerhub-user>/kprelogits-worker:latest
```

Pass that tag as `image=` to `kprelogits.ops.vast.create()`.

`Dockerfile` is the legacy worker rebuilt for this package: same base
(`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`), same apt set, same
`/workspace` layout — all proven on the GPUs this project rents — minus
everything that existed to mount the `ktrain` submodule and put it on
`PYTHONPATH`. Beyond the base image it installs only `timm`, `medmnist`, and
`huggingface_hub`, which is kprelogits' entire third-party surface.

`docker/entrypoint.sh` sources `/workspace/.env-secrets` before exec'ing the
extractor, then passes its arguments through untouched. That file is what
`ops.vast.stage_credentials()` writes over stdin, and sourcing it is the only
sanctioned way AWS keys and the HF token reach the job: `vastai create -e KEY=…`
would put them in the instance's command line, where anyone else on a shared
host can read them out of `ps`, and baking them into a layer would outlive the
run. A missing secrets file is not fatal — preflight reports what is actually
missing with far more context.

`timm` is deliberately unpinned, as it was in the legacy image. A timm upgrade
can change which weights a given `model_name` resolves to; every bundle records
that identity in `weights_revision`, so bundles from two timm versions stay
*distinguishable* — but they are not interchangeable, so pin before a run that
has to match an earlier one.

## Testing

```bash
python -m pytest kprelogits -q -m "not slow"
```

The `slow` marker covers long scientific gates; nothing in `kprelogits`
currently carries it.

## Status

`kprelogits` builds and its suite passes (164 tests).

- **The GPU path is verified.** On 2026-08-20 `kprelogits/ops/smoke.py` rented
  an RTX 3060 Ti, extracted `mobilenetv2_050.lamb_in1k` (the smallest of the
  pilot 50) in 94 s, uploaded the bundle, and tore the box down. The bundle
  satisfies the contract, resolves real provenance
  (`hf:timm/mobilenetv2_050.lamb_in1k@8990230b131c`), and matches the
  production bundle extracted months earlier on different hardware: labels
  bit-identical on all three splits, per-row cosine ≥ 0.999997, relative
  Frobenius error ~6e-5 — about 160x inside the tolerance gate.
- **The worker image has never been built.** `Dockerfile` exists and is
  described above, but there is no Docker on this machine. Only
  `docker/entrypoint.sh` is tested. The smoke driver sidesteps this entirely:
  it ships the package as source to a stock PyTorch image, which is how the
  legacy pipeline worked.

## The smoke test

```bash
python -m kprelogits.ops.smoke          # plan: prints offers and every command, rents nothing
python -m kprelogits.ops.smoke --go     # rent, extract, validate, destroy
python -m kprelogits.ops.smoke --abort  # destroy anything labelled smoke-*
```

`vastai` must be on PATH (`.venv/bin`). `--abort` is the first command of any
post-mortem: if the driver dies mid-run, that is what reclaims the box.

Three details are load-bearing rather than incidental:

**A per-attempt S3 prefix.** Bundle keys are `{dataset}_{model}_features.npz` —
no `max_train`, no precision. The production prefix already holds this model's
bundle, so writing there would overwrite real data; and because the worker's
resume oracle *is* S3, a merely-fresh-once prefix would make the second attempt
skip the model and report success having done nothing.

**SSH mode, not the image entrypoint.** The worker image starts extracting at
boot, which races credential staging — the job would reach S3 before its keys
did. Renting with `ssh=True` leaves sshd holding the box open so credentials
land first and the job starts deliberately.

**S3 state files are the only liveness signal.** Never `pgrep` over SSH: the
remote shell matches its own pattern, so the poll reports the job alive forever
or reports a kill that never happened. Three incidents, one of them four
concurrent overwriting fits.

Validation is tiered, because equality is the wrong test. Labels and shapes
must match the reference bundle *exactly* — that proves the seed-42 subsample
and the whole data path are byte-equivalent, and it costs nothing. Features are
checked by cosine similarity (min > 0.999) instead, because both extractions
run fp16 autocast and cudnn convolutions default to TF32 on Ampere and later:
~1e-2 relative difference is what correct code produces on two different
cards. A cosine below ~0.99 is not GPU noise — it means preprocessing, weights,
or normalization diverged, which is the failure worth catching.

In the one run so far the agreement came in at ~6e-5, some 160x inside that
gate. That is a much stronger result than the fp16/TF32 analysis predicted, and
it is worth not over-reading: `mobilenetv2_050` is the smallest and shallowest
of the 50, and depth and attention are exactly what accumulate this kind of
difference. Whether the agreement holds for `convnext_small` or a `swin` is
untested.

Packaging is deferred: no `pyproject.toml`, imports rely on the repo root being
on `sys.path`.
