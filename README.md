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

**1. Choose backbones.** The param cache is committed at
[artifacts/timm_model_param_cache.json](artifacts/timm_model_param_cache.json),
a byte-identical copy of the one at the bucket root. It maps each of 1,699 timm
model names to `{"params": int | null, "status": "ok" | "bad_input" |
"too_large"}`: `bad_input` means the pretrained config is not 3x224x224 (params
null), and `too_large` means the model was over the ceiling in force when the
cache was built (its count is still valid). It is public, for reuse elsewhere, at
<https://raw.githubusercontent.com/yashd94/ktrain-vastai/main/artifacts/timm_model_param_cache.json>.

```bash
python -m kprelogits.select --from-cache --cache artifacts/timm_model_param_cache.json \
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

Two drivers, both of which rent, stage code and credentials, launch, poll S3
state files, and tear down. Neither needs anything typed on the box.

`ops/smoke.py` runs **one backbone on one box** and validates the result
against its predecessor in S3. It is the acceptance test for the GPU path, not
a way to get work done.

`ops/run.py` runs **a model list across N boxes** — the sweep driver:

```bash
# plan: prints the shard split, the offers, the exact remote commands. Rents nothing.
python -m kprelogits.ops.run --selection artifacts/dermamnist_pilot50.json \
    --data-flag dermamnist --shards 4

# same command, with hardware
python -m kprelogits.ops.run --selection artifacts/dermamnist_pilot50.json \
    --data-flag dermamnist --shards 4 --go

# post-mortem: destroy anything labelled run-*
python -m kprelogits.ops.run --abort
```

Default is plan mode; `--go` is what spends money. Models are split
round-robin (`select_model_shard`) rather than in blocks, because the
selection is name-sorted and names correlate with size — blocks would pile the
heavy backbones onto one box.

Two properties worth knowing, because they are what make it safe to leave
running. **Every instance id is in the teardown list before it can fail** — the
context is entered before the first rental and holds a mutable list, so a
Ctrl-C during provisioning cannot strand a box. And **each box is destroyed as
its own shard lands**, gated not on the shard's `done` claim but on that
shard's bundles being present in S3 by name, which the driver checks by
recomputing the worker's own partition locally.

Re-running the same command after a partial run extracts only what is missing:
the worker's resume oracle is S3, so bundles that landed are not recomputed.

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

That is **~88 GB each way** for the 841 OCTMNIST bundles (measured
2026-09-13) -- about $8 of S3 egress to a laptop, and more free disk than
most laptops have, so run it in batches or from a machine in us-east-1, where
the transfer is free. Restamping does not invent provenance — it records
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

`timm` is **pinned to 1.0.29**, identically in the Dockerfile and in
`PIP_PACKAGES` (`kprelogits/ops/smoke.py`), which is what rented boxes install;
a test fails if the two disagree. `weights_revision` records the HF weights
commit, not the timm version, and timm decides what a model's pre-logits *are*:
InceptionNeXt's headless constructor returned zero-width features in timm
1.0.x, and `build_encoder`'s workaround is verified against 1.0.29. Two sweeps
under different timm versions could differ with nothing in either manifest to
say so. Change the pin deliberately, in both places, and re-verify when you do.

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
