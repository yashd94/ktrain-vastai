> **Historical.** This is the plan as approved in August 2026, kept for the
> reasoning behind the decisions — not a description of the repo today. It was
> carried out in full, and then went further: `kmetrics`, `kprobe` and `ksim`
> have since left for their own repos, and `ktrain` and the study code with
> them. See [restructure.md](restructure.md) for what actually happened and
> the [root README](../README.md) for what is here now.

# Restructure into kprelogits / kprobe / kmetrics

## Context

The repo currently mixes three lifecycles: GPU feature extraction on Vast, linear-probe
fitting + recalibration, and metric/selection evaluation — all tangled through `jobs/`
and a pinned Colab-first `ktrain` submodule whose metrics layer has already forked three
ways. The user wants three core folders, each a future spin-out repo:

- **`kprelogits/`** — the Vast feature-extraction pipeline, with the ops hardening
  discussed earlier (typed config + preflight, task-state files, instance teardown,
  HF-token support). Results to S3.
- **`kprobe/`** — linear probing, run locally: CE, label smoothing, focal, MSE,
  Pandora α∈{0,1,2}; recalibration as a first-class step (head on 90% of train,
  calibrator on held-out 10%, ratio configurable); target-prevalence-aware calibration
  because OCTMNIST val prior ≠ balanced test prior. Downloads prelogits from S3;
  writes logits locally only.
- **`kmetrics/`** (in-repo for now) — built fresh from this repo's implementations:
  scoring metrics, per-example decompositions, the top-1-of-K selection eval
  (p_top1 + Δregret), and the simulators (`multiplex/`, `octmnist_sim/` core).

User decisions (from AskUserQuestion):
1. kprobe carries **exactly the listed arm set** — historical arms stay in
   `jobs/fit_heads.py` as the frozen OCTMNIST record.
2. Recal split = **split of the train set** (90/10 default, configurable). Val stays
   untouched for selection.
3. Target-prevalence Platt = **importance-weighted fit** (weight examples by
   `target_prior[y]/empirical_prior[y]`).
4. kmetrics starts **fresh from this repo's code** (per_example/shift_lib, which carry
   verification tests) — the sibling `../kmetrics` repo is reconciled later, not vendored.

Dependency direction (one-way, enforced): `kmetrics ← kprobe ← experiments (jobs/)`;
`kprelogits` standalone. Nothing in the three new packages imports from `jobs/` or `ktrain/`.
`jobs/` remains the frozen record of the OCTMNIST study and keeps working throughout —
existing modules become thin re-export shims where their content moves, never duplicates.

---

## Phase 1 — `kmetrics/` (foundation; others import it)

New package at repo root:

```
kmetrics/
  __init__.py
  pandora.py         # numpy Pandora scoring
  search_cost.py     # simulated search cost (p/cost greedy, stable-sort ties)
  metrics.py         # per-example decompositions + exact scalar metrics
  costs.py           # load_costs_from_csv + the canonical cost.csv
  selection.py       # top-1-of-K engine: p_top1, mean_regret, tie expectation, arm_seed
  simulators/
    multiplex/       # moved from /multiplex
    octmnist/        # moved from /octmnist_sim (core only)
  tests/
```

**Sources (copy with provenance headers, then shim the originals):**
- `pandora.py`, `search_cost.py`: from `ktrain/medmnist/metrics/` — these two files are
  byte-identical to the sibling repo, so "fresh from this repo" and "reconcile with
  sibling" coincide. Keep the `priors=` kwarg (`_adjust_for_priors`).
- `metrics.py`: from `jobs/finite_sample_selection/per_example.py` (vectorized 2-D forms)
  and `shift_lib.py`'s batched 3-D forms (`decomposable_per_example`, `ece_samples`,
  `macro_auc_samples`) — collapse the three coexisting implementations (ktrain scalar,
  per_example 2-D, shift_lib 3-D) into axis-general functions. Copy the four scalar
  metrics per_example imports from ktrain (`balanced_accuracy`, `cohens_kappa`, `ece`,
  `macro_auc_ovr` in `ktrain/medmnist/metrics/baselines.py`).
  `verify_decomposition` becomes a kmetrics unit test.
- `costs.py`: `load_costs_from_csv` from `ktrain/medmnist/config.py` + one canonical
  `cost.csv` (copy of `ktrain/medmnist/cost.csv`, which both datasets' rows live in).
- `selection.py`: from `jobs/finite_sample_selection/summary_stats.py` — `_score`,
  `_tie_mask`, `_gather`, `_summarise`, `arm_seed`, `assert_distinct_arm_seeds`,
  TIE_RTOL/ATOL — plus `shift_lib.score_selection` (the per-replicate-truth variant).
  One scorer, two entry points (fixed truth / per-replicate truth). **`arm_seed` must be
  moved verbatim** — it determines every published by-condition number.
- `simulators/multiplex/`: `multiplex/` moves wholesale (self-contained: sim, tests,
  README, `run_meta_eval.py`).
- `simulators/octmnist/`: `octmnist_sim/` core moves — `dp.py`, `nodes.py`,
  `obs_models.py`, `policies.py`, `config.py`, `constants.py`, `metrics.py`,
  `experiments.py`, `figures.py`, `build_zoo.py`, `cli.py`, `tests/`,
  `SPEC_DEVIATIONS.md`. **`selection_figures.py` and `compare_simulators.py` do NOT
  move** — they import experiment code (`shift_lib`, `summary_stats`, `model_pool`) and
  are exhibits of the OCTMNIST study; they stay in a slimmed top-level `octmnist_sim/`
  alongside `run_all.sh`, importing the simulator core from `kmetrics.simulators.octmnist`.

**Shims:** `jobs/finite_sample_selection/per_example.py` keeps its public API
(re-exporting from kmetrics) plus the experiment-owned metric-set declarations
(`DECOMPOSABLE_METRICS`, `EXACT_ONLY_METRICS`, `HIGHER_IS_BETTER`, `TRUTH_METRIC` stay —
they are the study's declared set, not library code). Same pattern for the moved
functions in `summary_stats.py` and `shift_lib.py`.

**What stays out of kmetrics:** the differentiable Pandora training loss (kprobe, per
user), the metric-set declarations, `model_pool.py`, `dataset_spec.py`, the shift
experiment, all report/figure scripts.

**Phase-1 verification (blocking):**
1. `bash jobs/finite_sample_selection/run_all.sh` → `finite_sample_selection.csv` and
   `finite_sample_selection_by_condition.csv` **byte-identical** to current committed copies.
2. `.venv-fit/bin/python3 jobs/finite_sample_selection/shift_tests.py out/finite_sample_selection/octmnist` → ALL PASS.
3. `python -m pytest kmetrics/simulators/multiplex/test_multiplex_sim.py kmetrics/simulators/octmnist/tests -q` → pass in new location.
4. New kmetrics tests: per-example vs scalar decomposition check; selection scorer vs a
   pinned row of the published by-condition CSV (locks `arm_seed` + tie handling).

---

## Phase 2 — `kprobe/` (imports kmetrics)

New package at repo root:

```
kprobe/
  __init__.py
  config.py          # ProbeConfig dataclass + validation
  data.py            # fetch/load/standardize prelogit bundles
  splits.py          # stratified train/calib split
  losses.py          # ce / label-smoothing / focal / mse loss factories
  pandora_loss.py    # moved from jobs/pandora_loss.py
  fit.py             # L-BFGS full-batch protocol; Adam+polish path for pandora arms
  calibrate.py       # TS / weighted Platt / prevalence adjustment
  run.py             # CLI: probe a dataset end to end, default output set
  tests/             # includes moved jobs/test_pandora_loss.py
```

**Data layer** (`data.py`): adapt from `jobs/fit_heads.py` —
`fetch_features` (:179-201, S3 `aws s3 cp` with `.part`+rename),
`load_splits` (:204-212), `standardize` (:215-227, `CONST_DIM_STD=1e-8`, `STD_EPS=1e-6`).
Reads the same bundle schema `extract_features.py` writes
(`x_{train,val,test}`, `y_*`, `feat_dim`, `num_classes`, `model_name`, `max_train`,
`precision`). Logits land locally under `out/kprobe/logits/<flag>/<backbone>/<arm>/`
(fit_heads layout, no `@lam` suffix); **no S3 upload of logits**.

**Splits** (`splits.py`): generalize `fit_heads.stratified_half` (:477-494) to
`stratified_split(y, calib_frac=0.1, seed)` — per-class shuffle, first `1-frac` → head-fit
indices, rest → calibration indices, both sorted. Head trains on the 90%; calibrators fit
on the 10%; **val is never consumed by calibration**.

**Losses** (`losses.py`): factories for `ce`, `ls_eps{ε}` (`F.cross_entropy(...,
label_smoothing=ε)`), `focal_g{γ}`, `mse` (Brier) — port the loss closures from
`fit_heads.make_loss` (the `mse` branch was deleted from fit_heads in the cuts; recreate
from `docs/arms.md` arm D: `((softmax(z)-onehot)**2).sum(1).mean()`).
Pandora arms: `pandora_a{α}_{w|u}` with α free-form (grammar from
`fit_pandora.parse_arm` :77-91); α∈{0,1,2} is the declared surface. `pandora_loss.py`
and `test_pandora_loss.py` move from `jobs/` (git mv; leave one-line deprecation stubs).

**Fit** (`fit.py`): two protocols, both from existing proven code —
- CE-family: full-batch L-BFGS (`fit_heads.fit_head` protocol: history_size=10,
  max_iter=500, grad-norm gate 1e-5 as diagnostic).
- Pandora: Adam (batch 1024, cosine + 5% warmup, early stop on val sim cost, patience
  300) + float64 L-BFGS polish, restarts {zeros, ce_warm, rand×N} — from
  `fit_pandora.fit_adam`/`fit_one` (:132-358). Val sim cost via `kmetrics.search_cost`.
- Compute: **torch on CPU by default, `device="mps"` opt-in flag** (recommendation:
  keep the proven torch code; L-BFGS full-batch is already fast on CPU; do not rewrite
  in numpy). Respect `FIT_THREADS`.

**Calibration** (`calibrate.py`) — all post-CE, fit on the calibration split, all
returned as `(M, v)` affine maps so `W' = W@M, b' = b@M+v` keeps everything a linear
probe (the `fit_heads.run_one_recal` :1261 idiom):
- `temperature(z_cal, y_cal)` — NLL-minimizing scalar T, fit on **val-prevalence data
  as-is** (per user: "TS on validation prevalence").
- `platt_weighted(z_cal, y_cal, target_prior)` — Platt (`exp(a)*z + c`) with each
  example weighted `target_prior[y] / empirical_prior[y]` in the loss. With
  `target_prior=None` or equal to empirical, reduces exactly to plain Platt (test this).
- `prevalence(source_prior, target_prior)` — closed-form **logit offset**
  `v = log(target/source)`, `M = I`. This is the ONE canonical convention; document that
  it equals `refit_selection.prevalence_shift` (:102) and
  `shift_lib.adjust_probs(p, w)` with `w = target/source`, and that
  `kmetrics.pandora`'s `priors=` kwarg divides (inverse convention) — the shift_tests
  equivalence assert (:178-193) already pins this.
- Fitting engine: reuse `fit_heads.fit_recalibration`'s L-BFGS setup (:576-578) with
  `objective="ce"` only (Pandora-objective recal is out of kprobe's declared scope).
- IDR/penalized isotonic: **not in this phase** — leave a documented seam
  (`calibrate.py` docstring) since the user plans it against the prevalence-adjusted
  versions later.

**Config** (`config.py`): `ProbeConfig` dataclass — dataset flag, backbone list or
selection file, arms, λ (fixed value or grid + selection metric), `calib_frac` (default
0.10), `calib_seed`, **`target_prior`** (array | `"balanced"` | `"train"` | None;
OCTMNIST default `"balanced"` because test is balanced and val is not — document the
val/test shift: val (0.344, 0.105, 0.080, 0.472) vs balanced test), device, S3 source
URI, output dir. `validate()` checks cross-field constraints before any download.

**Default output set per (backbone, arm):** raw logits + three calibrated variants
(`+ts`, `+platt_t`, `+preval`) written as sibling arm dirs (e.g. `ce`, `ce+ts`,
`ce+platt_t`, `ce+preval`), one `fits.jsonl` record each with a **fixed schema**
(fix the union-schema problem the explorer found in fit_heads records: every record has
every field, null where N/A; includes resolved config + calib split spec).

**Phase-2 verification (blocking):**
1. Reproduction gate, `check_ce_reproduces.py` pattern (jobs/check_ce_reproduces.py):
   fit `ce` on 2-3 backbones from the cached S3 bundles at the recorded λ and compare
   val/test simulated cost to the recorded `fits.jsonl` values. NOTE: kprobe's ce will
   differ slightly (head sees 90% of train, not 100%) — run the gate with
   `calib_frac=0` to reproduce exactly, then once at 0.10 to record the (expected,
   small) delta. Both numbers go in the PR description.
2. `pytest kprobe/tests` — pandora loss §1.3 tests (moved), platt-weighted-reduces-to-
   platt, prevalence offset equals shift_lib.adjust_probs route, TS matches
   `fit_heads.fit_temperature` on identical inputs.
3. `kprobe/run.py --data-flag octmnist --backbones <2 names> --arms ce,ls_eps0.1 --dry-run`
   prints resolved config; without `--dry-run` produces the full default output set.

---

## Phase 3 — `kprelogits/` (standalone; can run parallel to Phase 2)

New package at repo root:

```
kprelogits/
  __init__.py
  config.py          # ExtractConfig + preflight()
  extract.py         # from jobs/extract_features.py
  models.py          # extraction helpers copied OUT of ktrain (severs the submodule dep)
  select.py          # from jobs/build_model_list.py
  ops/
    state.py         # per-shard task-state JSON (replaces sentinel-grep + pgrep)
    vast.py          # instance registry + finally-teardown (thin wrapper over vastai CLI)
    s3.py            # upload/resume/cleanup helpers factored from extract.py
  README.md
```

**Moves:**
- `extract.py` from `jobs/extract_features.py` — keep the good parts verbatim: atomic
  `save_atomic` (:127-132), S3-listing resume (`s3_already_have` :69-84), cleanup gated
  on confirmed upload (:258-270), per-model upload + final sync. **Bundle schema
  unchanged** so kprobe and all legacy consumers keep working.
- `models.py`: copy from `ktrain/medmnist/model_comparison/train.py` —
  `get_loaders` (:362), `_build_encoder` (:421), `_extract_all_splits` (:529),
  `_select_model_shard` (:686), `_safe` (:670) — with a provenance header (source file
  + submodule SHA). This severs kprelogits' ktrain dependency for spin-out.
- `select.py` from `jobs/build_model_list.py` (same treatment: it imports
  `filter_timm_models` from ktrain — copy that too).

**Config + preflight** (`config.py`): `ExtractConfig` dataclass covering the current env
surface (dataset, data/results dirs, selection path, `max_train`, batch size,
rank/shards, precision, S3 bucket+prefix, cleanup, HF cache dirs, optional HF token).
`preflight()` validates before any rental: rank < shards; `MODEL_OFFSET < MODEL_STRIDE`;
`max_params ≤ 200e6` (or warn about ktrain's silent gate); S3 credentials present and
bucket listable when S3 is configured; selection file exists and parses; **HF token
present when the selection contains gated models** (probe via
`huggingface_hub.model_info` on a sample). Resolved config (defaults included) is
serialized to JSON and uploaded next to the results
(`{s3}/{flag}/extract_config_shard{rank}.json`).

**HF token** (new capability — none exists today): accept `HF_TOKEN` env or
`--hf-token-file`; on the box, stage via a `umask 077` file sourced by the job, per the
`docs/vastai.md` §6 credential pattern. **Never in `vastai create -e` args and never in
`ps`-visible command lines.**

**Ops** (`ops/`):
- `state.py`: each shard writes `{s3}/{flag}/state_shard{rank}.json` transitions
  (`pending → running → uploading → done | failed`, with timestamp, host, counts of
  done/failed models). The driver polls S3 state files — **never `pgrep`** (the
  self-match trap in `docs/vastai.md` §4 / memory note).
- `vast.py`: `rent(config) -> instance_id` recording to a registry file AND verified
  against `vastai show instances --raw`; `with teardown(ids):` context manager that
  destroys instances in `finally` and re-lists to confirm burn rate — the 4.3-hour
  orphan and the α=10 near-loss from `docs/vastai.md` §8 are the motivating incidents.
  Pre-teardown check = state files show `done` + expected S3 key count, replacing the
  §9 manual checklist.
- `s3.py`: the upload/retry/resume code factored out of `extract.py` so retry policy
  lives in one place.

`scripts/run_job.sh JOB=extract` gains a sibling entry pointing at
`kprelogits/extract.py`; the old `jobs/extract_features.py` becomes a deprecation stub.
`vast_rent.sh` / `vast_deploy.sh` are NOT rewritten (historical records + working
tooling) — `ops/vast.py` wraps the same CLI, and the shell scripts can be retired when
kprelogits spins out.

**Phase-3 verification:**
1. `preflight()` unit tests: each violated constraint produces its named error; a
   valid config passes.
2. Local extraction smoke test: run `extract.py` on 1-2 tiny timm backbones
   (e.g. the smallest in the selection) on CPU/MPS with `--max-train 256`,
   no S3; assert the bundle schema (keys/dtypes/shapes) matches an existing S3 bundle's
   exactly, and that a `kprobe` ce fit on the new bundle runs end to end.
3. Dry-run of the vast wrapper (no `--go`): prints offers + resolved config, rents nothing.

---

## Phase 4 — documentation + repo hygiene

- Top-level `README.md`: new layout section (kprelogits / kprobe / kmetrics /
  jobs=frozen-experiment / ktrain=pinned-archive) + the one-way dependency rule.
- `docs/`: short `docs/restructure.md` recording what moved where and the shim policy;
  update `ktrain/CLAUDE.md`-adjacent pointers if any break.
- `deprecated/` untouched; `ktrain` submodule untouched (still pinned; still imported
  by the frozen `jobs/` record — bands 2–3 of the dependency map: `train_heads.py`,
  `run_worker.py`, `fit_heads.py`'s historical `PandoraLoss` import for arms G/H).
- Commit in phase order: one commit per phase, each ending at a green verification gate.

## Out of scope (explicitly deferred)

- IDR / penalized isotonic calibrators (seam documented in `kprobe/calibrate.py`).
- The §2–§5 experiment work from the revised spec (recal cross, sharpening sweep, shift
  calibrators, Dirichlet severity) — these become consumers of kprobe/kmetrics.
- DermaMNIST arm declaration and shift_lib K=4 generalization (guards already in place
  in `dataset_spec.py`).
- The pending regeneration debt (shift grid, refit selection, octmnist_sim exhibits) —
  unchanged by this restructuring; Phase-1 verification only requires the two no-shift
  CSVs, which are current. Regenerate the rest after Phase 1 so the reruns exercise the
  new import paths.
- Reconciliation with the sibling `../kmetrics` repo (its `baselines.py` diverged; the
  in-repo kmetrics becomes canonical and the sibling is updated from it at spin-out).
- Actual spin-out into separate repos.

## Key risks

- **Byte-identical regression is the contract.** Phase 1 moves the code that produced
  published numbers; the two run_all.sh CSVs must not change by one byte. If they do,
  stop and diff before proceeding.
- `octmnist_sim` split (core → kmetrics, exhibits stay) touches the deferred-rerun
  exhibits; their imports change but their outputs are already known-stale, so verify
  imports only, not outputs.
- kprobe's 90/10 head is a *different estimator* than the historical 100% head — the
  reproduction gate must compare like with like (`calib_frac=0`).

---

# Revision (2026-08-13) — after critique review

Superseding decisions, per user + agent-critique review:

1. **No byte-identical reproduction.** Old results are disposable (changed since
   run, unreported). kmetrics numerics are pinned by fixture tests (verified to
   1e-13 against the old code before freezing), not by regenerating legacy CSVs.
2. **Simulators → `ksim/`**, a separate package importing kmetrics. kmetrics
   stays a small pure metrics+selection library.
3. **Artifact contracts live in kmetrics** (`kmetrics/contracts.py`): versioned
   prelogit-bundle schema + ProbeArtifact schema. kprelogits and kprobe both
   depend on kmetrics for the contract; they still never import each other.
4. **Legacy: aggressive.** Tag pre-restructure commit as the legacy snapshot,
   physically move `jobs/`, `scripts/`, `deprecated/`, and the octmnist_sim
   exhibits under `legacy/`. Not maintained, not required to run. The §2–§5
   experiments are rebuilt on kprobe/kmetrics, not extended in jobs/.
5. **Run identity split.** Probe fits: `runs/<run-id>/` keyed by resolved-config
   hash, config saved inside, resume validates the hash. Prelogit bundles: a
   shared content-keyed cache with the config manifest embedded per bundle and
   validated on read — deliberately NOT run-scoped.
6. **ProbeArtifact is the primary kprobe output** (standardizer, head,
   calibration map + kind, class order, split spec, target prior, resolved
   config, schema version); logits are derived products regenerable for any
   split (train/val, one day test).
7. **Library defaults stay experiment-free**: kprobe `target_prior=None`; the
   OCTMNIST experiment config sets balanced. Copied ktrain internals become
   owned code (renamed, tested, canonical). Packaging (pyproject) deferred to
   spin-out. One orchestration path: kprelogits ops; shell scripts archived in
   legacy/.
