# kprelogits

Frozen pre-logit extraction on rented GPUs.

Runs each timm backbone over a dataset exactly once and writes the pre-logit
features to a bundle in S3. This is the only stage that needs a GPU, and the
only stage whose output costs more than the machine that produced it — so most
of the package is about not losing bundles and not paying for idle boxes.

Bundles are the hand-off to `kprobe`, which lives in its own repo. The two
never import each other; they agree only on the format specified in
[contracts.py](contracts.py), which kprelogits stamps into every bundle and a
consumer validates on read. The producer owns the spec because the producer is
what would have to change to break it.

## Pipeline

```bash
# 1. choose backbones (no GPU; no network once the cache is local).
#    The cache is NOT in this repo -- it lives at the bucket root.
aws s3 cp s3://$S3_BUCKET/timm_model_param_cache.json .
python -m kprelogits.select --from-cache --cache timm_model_param_cache.json \
    --max-params 200000000 --stride 1 --output results/_shared/selected_models.json

# 2. check everything before renting anything (env matters -- see below)
S3_BUCKET=... DATA_DIR=./data RESULTS_DIR=./results \
python -m kprelogits.extract --selection results/_shared/selected_models.json --dry-run

# 3. extract (on the box)
python -m kprelogits.extract --selection /workspace/selection.json \
    --shards 4 --rank 0 --cleanup
```

`--stride 1` reproduces the 865-model corpus behind the bundles already in S3;
a larger stride subsamples it for a cheap trial.

Configuration is environment-driven with CLI overrides, and the defaults assume
a rented box (`/workspace/data`, `/workspace/results`, no bucket). The full
variable list is in the [root README](../README.md#running-it) — a bare
`python -m kprelogits.extract --dry-run` on a laptop fails, correctly, because
it can find no selection file.

`--dry-run` runs the full preflight and prints the resolved config without
touching a GPU. `--no-probe` additionally skips the S3 and HuggingFace checks
so it works offline.

## Preflight

`config.preflight()` reports **every** problem at once, so one fix cycle
suffices. Structural checks are pure and offline; the probing checks
(`check_s3`, `check_hf`) are opt-out.

| check | why it exists |
|---|---|
| `0 <= rank < shards` | a shard that schedules nothing still bills |
| selection exists, parses, has named models | the file that never got uploaded |
| `cleanup` implies `s3_uri` | deleting bundles with nowhere to put them |
| S3 prefix listable | credentials that were never staged |
| published config matches `bundle_identity()` | see *bundle keys* below |
| gated HF repos have a token | a 401 twenty minutes into a rental |
| free disk | bundles run 0.5–2 GB each |
| oversized models | **warns**, does not skip — unlike ktrain's silent gate |

## Bundle keys and the shared cache

Bundle filenames are `{dataset}_{model}_features.npz` — keyed by dataset and
model only, which is what makes them a *shared* cache rather than a per-run
artifact. It also means two runs differing in `max_train`, `precision`, or the
underlying weights would write different content under the same name.

Two defences: every bundle carries a `bundle_key` hash of exactly those fields,
and preflight compares this run's `bundle_identity()` against the
`extract_config_shard*.json` documents already published beside the bundles,
refusing to proceed on a mismatch.

## Ops

`ops/state.py` — each shard publishes a small JSON document to S3 after every
transition (`pending → running → uploading → done | failed`) with counts of
scheduled / skipped / extracted / uploaded / failed.

**Never poll process liveness over SSH.** `ssh host 'pgrep -f extract'` matches
the remote shell running that very command, so the count is never zero, and
`pkill` kills its own shell first and returns empty output that reads like
success. This cost the project three incidents. The bracket trick (`[e]xtract`)
does not save you either — a shell variable *named* after the pattern
self-matches too. Read the state files instead.

`ops/vast.py` — instance lifecycle. The registry file is **not** the truth; the
API is. `reconcile()` reports instances that are billing but untracked (one such
orphan ran 4.3 hours unnoticed), `teardown()` destroys in `finally` and re-lists
to confirm, and `preteardown_check()` replaces the manual checklist: every shard
terminal and complete, *and* the expected bundles actually in S3. A shard saying
`done` while S3 is short is the failure that nearly lost a set of fits existing
only on a box about to be destroyed.

`ops/s3.py` — one retry policy. Reads degrade to a warning (a failed listing
costs re-extraction at worst); writes raise (a silent upload failure throws away
a GPU pass).

`ops/run.py` — the sweep driver: rents N boxes, launches one shard on each,
releases each box as its own shard's bundles appear in S3, and tears down the
rest in `finally`. The teardown list is entered before the first rental and
mutated as boxes come and go, so there is no window in which a live instance
is unaccounted for. `ops/smoke.py` is the same shape for one backbone on one
box, and exists to test the GPU path rather than to do work.

## Credentials

**Never put a secret in a command line.** Vast hosts are shared and `ps` is
world-readable; a key was exposed this way earlier in the project. `create()`
refuses credential-shaped keys in `vastai create -e`, and `stage_credentials()`
pipes them over stdin into a `umask 077` file:

```python
vast.stage_credentials("root@host", {
    "AWS_ACCESS_KEY_ID": ..., "AWS_SECRET_ACCESS_KEY": ..., "HF_TOKEN": ...,
})
```

Remote scripts then do `set -a; . /workspace/.env-secrets; set +a`. The HF token
is read from `--hf-token-file` (a path, never the token) or the environment, and
never appears in the resolved config that gets published to S3.

## Migrating pre-contract bundles

Bundles written before `contracts.py` existed carry the arrays but none of
the provenance. Re-extracting would mean renting GPUs to recompute correct
features; restamping adds the missing scalars in seconds:

```bash
python -m kprelogits.restamp --dir feature_cache --data-flag octmnist
```

It does not invent provenance: restamped bundles record
`weights_revision="unknown:pre-contract"` and a `producer_version` naming both
the legacy producer and this tool, because the timm weights those features came
from are genuinely unrecoverable after the fact. Arrays pass through untouched.

## Provenance

`models.py` and `select.py` were copied out of
`ktrain/medmnist/model_comparison/train.py` and `../legacy/jobs/build_model_list.py`
and are **now canonical**. kprelogits does not track ktrain; divergence is
expected rather than a bug to reconcile. That is what lets this package spin out
with no submodule.

## Testing

```bash
python -m pytest kprelogits/tests -q
```

Fully offline — no GPU, no timm, no S3, no vastai CLI. `timm`, `medmnist`, and
`torchvision` are imported inside the functions that need them, so the package
imports on a laptop with none of them installed. The round trip is tested end
to end (`bundle_payload` → `np.savez` → `validate_prelogit_bundle`) on
synthetic features. It deliberately does not import a consumer: reaching across
a repo boundary for a stronger test would be a dependency on the very thing the
contract exists to decouple.

What the tests cannot cover is the timm forward pass itself. That is now
verified on real hardware — see the root README's Status — by
`python -m kprelogits.ops.smoke --go`, which rents a box, extracts one
backbone, checks the bundle against its predecessor in S3, and tears down.
