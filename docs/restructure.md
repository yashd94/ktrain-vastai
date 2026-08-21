# The 2026-08 restructure: what moved where

Before this, the repo mixed three lifecycles in one tree: GPU feature
extraction on rented hardware, linear-probe fitting and recalibration, and
metric/selection evaluation — all routed through `jobs/` and a pinned,
Colab-first `ktrain` submodule whose metrics layer had already forked three
ways. Any change to how a head was fit could reach the code that produced
published numbers, and the same formula existed in three places with no test
saying they agreed.

> **Note.** This records the August 2026 split into four in-repo packages.
> That was the right first move and it did not stay put: `kmetrics`, `kprobe`
> and `ksim` have since become separate repos, and `ktrain` and the study code
> left too, so this repo is `kprelogits` alone. The provenance table below
> still names the in-repo destinations, because that is where the code went at
> the time and the trail only makes sense read in order. See *The spin-out*
> below for where things ended up.

The split is by lifecycle, not by topic. Extraction happens rarely, costs money
per hour, and must survive a machine vanishing. Probing happens constantly,
locally, and is cheap to redo. Scoring is pure arithmetic that both of the above
and every future experiment depend on. Those are three different jobs with three
different failure modes, and they are now three packages.

See [restructure-plan.md](restructure-plan.md) for the plan and the decisions behind it.

## Provenance

Everything below was **copied and adopted**, not vendored. The new copy is
canonical; divergence from the source is expected rather than a bug to
reconcile. That is what lets each package spin out with no submodule.

| destination | source | notes |
|---|---|---|
| `kmetrics/pandora.py` | `ktrain/medmnist/metrics/pandora.py` @ `2de9717` | verbatim; was byte-identical to the sibling kmetrics repo |
| `kmetrics/search_cost.py` | `ktrain/medmnist/metrics/search_cost.py` @ `2de9717` | verbatim |
| `kmetrics/metrics.py` | `jobs/finite_sample_selection/per_example.py` (2-D) + `shift_lib.py` (3-D) + ktrain's scalars | three implementations collapsed into one axis-general path |
| `kmetrics/selection.py` | `jobs/finite_sample_selection/summary_stats.py` + `shift_lib.score_selection` | one scorer, two entry points; `arm_seed` moved verbatim |
| `kmetrics/costs.py`, `cost.csv` | `ktrain/medmnist/config.py`, `ktrain/medmnist/cost.csv` | one canonical table, both datasets |
| `kmetrics/contracts.py` | new | the prelogit-bundle and ProbeArtifact schemas |
| `ksim/multiplex/` | `multiplex/` | moved wholesale, self-contained |
| `ksim/octmnist/` | `octmnist_sim/` core | exhibits stayed with the study code they import |
| `kprobe/data.py` | `jobs/fit_heads.py` fetch/load/standardize | standardize now returns kept-width mu/sigma |
| `kprobe/splits.py` | `jobs/fit_heads.py::stratified_half` | 50/50 of VAL generalized to an arbitrary fraction of TRAIN |
| `kprobe/losses.py` | `jobs/fit_heads.py::make_loss` | MSE recreated from `legacy/docs/arms.md` arm D |
| `kprobe/fit.py` | `jobs/fit_heads.py::fit_head`, `jobs/fit_pandora.py` | protocols kept, drivers left behind |
| `kprobe/calibrate.py` | `jobs/fit_heads.py::run_one_recal` idiom | every map affine, returned as `(M, v)` |
| `kprobe/pandora_loss.py` | `jobs/pandora_loss.py` | moved |
| `kprelogits/extract.py` | `jobs/extract_features.py` | now stamps the full contract |
| `kprelogits/models.py` | `ktrain/.../model_comparison/train.py` helpers | underscore-privates became owned public API |
| `kprelogits/select.py` | `jobs/build_model_list.py` + ktrain's `filter_timm_models` | |
| `kprelogits/ops/`, `config.py`, `restamp.py` | new | preflight, state files, teardown, migration |

Everything not listed — the OCTMNIST study's drivers, shell scripts, reports,
figures, and outputs — moved under `legacy/` wholesale. It is not maintained
and nothing above imports from it.

## The one-way rule

As designed, in one tree:

```
kmetrics  <-  ksim
kmetrics  <-  kprobe        <-  (future experiment code)
kmetrics  <-  kprelogits
```

`kprelogits` and `kprobe` never import each other. They communicate only
through the artifact contracts, so **the file format is the interface**. This is
what lets the extraction pipeline live on a rented GPU box with no probing code
installed, and the probe run locally with no timm, no medmnist, and no CUDA.

Nothing in the four packages imports from `legacy/` or `ktrain/` — not even at
test time. The one remaining coupling runs the other way:
`legacy/jobs/finite_sample_selection/per_example.py::verify_decomposition`
cross-checks kmetrics' decomposition against ktrain's archived reference, for
that study's own callers.

## The spin-out (2026-08-20) and what it did to that rule

`kprobe` and `ksim` left for `../kprobe` and `../ksim`; `kmetrics` went with
`kprobe`, which was the only package that used more than a sliver of it.

`ksim` cost nothing: despite the arrow above, it never actually imported
`kmetrics` — it carries its own `octmnist/metrics.py`. Its only tie was
inbound, from the `legacy/octmnist_sim/` exhibits.

`kprelogits` was the interesting one. It used **only the prelogit half** of
`kmetrics/contracts.py` — the schema version, `ContractError`, and
`validate_prelogit_bundle` — and nothing else in the package at all: no
metrics, costs, pandora, selection, or search_cost. It never touched
`ProbeArtifact`, which is written and read by `kprobe` alone. So the file was
always two unrelated contracts sharing a module, and it split cleanly.

That half now lives in `kprelogits/contracts.py`, and **kprelogits imports
nothing from any of the other repos.**

The reasoning matters more than the move. Putting the format spec in a third
package was a bid for *neutral ground*: neither producer nor consumer could
unilaterally redefine what they hand each other. That was sound while
`kmetrics` sat beside both. It expired the moment `kmetrics` moved into the
consumer's repo — continuing to import it would have made the producer of the
format depend on the consumer's library, which is the coupling neutrality was
protecting against, merely inverted and harder to see.

So the producer owns the spec and a consumer keeps a reader. The two copies can
drift; if they do, the consumer refuses a bundle at load time with a message
naming the member at fault. Loud and early beats a shared import that no longer
means what it meant.

## The two contracts

**Prelogit bundle** (`kprelogits` → `kprobe`). An `.npz` per (dataset,
backbone) holding frozen features for all three splits, plus scalars naming the
dataset, the weights revision, the preprocessing spec, and the producer
version. Bundles are a *shared content-keyed cache*, deliberately **not**
run-scoped: the same features serve every probing run, and the embedded
manifest — not a filename or a directory — is what a consumer validates.

**ProbeArtifact** (`kprobe`'s primary output). Standardizer, head, calibration
spec, split spec, class order, resolved config, schema version. Logits are a
*derived* product: an artifact regenerates them for any split, so what gets
stored is the thing that makes regeneration possible. `ProbeArtifact.apply()`
is the single code path from raw features to calibrated logits.

Run identity splits accordingly: probe fits live in `runs/<flag>-<run-id>/`
keyed by a hash of the resolved config, so two differently-configured runs
cannot overwrite or silently resume from one another; bundles are keyed by
content and shared across runs.

## Decisions worth remembering

**No byte-identical reproduction.** Old results are disposable — most changed
after they were run and none are reported anywhere. kmetrics' numerics were
verified against the legacy code before freezing; the legacy CSVs are not a
regression contract.

**Legacy is archived, not shimmed.** No re-export stubs. A file is either
maintained code in one of the four packages or a frozen record under `legacy/`.

**Library defaults stay experiment-free.** `kprobe` defaults
`target_prior=None`; the OCTMNIST experiment config sets `"balanced"`.
"Calibrated for the training population", "for a balanced benchmark", and "for
a deployment population" are three different claims, and a probing library
should not silently pick one.

**Recalibration is first-class.** The head fits on `1 - calib_frac` of TRAIN
(default 0.90), calibrators fit on the rest, and VAL is never consumed by
calibration — it stays free for selection.

**One orchestration path.** `kprelogits.ops`. The shell scripts are archived.

**Packaging is deferred** to spin-out; no `pyproject.toml` yet.

## What the rebuild changed on purpose

**float64 heads.** The float32 L-BFGS fit stalls at grad-norm ~3e-4 — the
float32 noise floor — which made the 1e-5 convergence gate unreachable and
reported `converged=False` for every arm. The same fit in float64 runs to
9e-7. `legacy/docs/fit.md` §3 recommended this. It costs ~3s per fit and moves
costs by under $0.30; it is also why one backbone in the migration gate comes
out *better* than the legacy record rather than identical.

**Prevalence correction is in the default output set.** OCTMNIST's train and
val priors are near-identical (`.342/.107/.078/.473` vs `.344/.105/.080/.472`)
while test is exactly balanced. On `convnext_small`, correcting for that moves
test accuracy 0.777 → 0.916 and cost $266.87 → $236.74. Temperature scaling
moves accuracy by exactly zero — it is monotone, so it cannot move the argmax —
and slightly *worsens* test cost, being fit at train prevalence and scored on a
balanced split.

**Class names travel with the costs.** `kmetrics.costs` serves `load_class_names`
from the same table, in the same order, as the cost vector. Artifacts recorded
`class_order` as `["0","1","2","3"]`, which is only interpretable next to the
code that wrote it; more to the point, a cost vector mis-ordered against the
labels is unfalsifiable after the fact, and one table in one order is what
rules that out.

**Bundles carry provenance.** Legacy bundles recorded feat_dim, num_classes,
model name, max_train, and precision — enough to fit, not enough to know what
you fit on. Bundles now name their dataset, weights revision, preprocessing
spec, and producer. `python -m kprelogits.restamp` upgrades pre-contract
bundles in place without a GPU, recording
`weights_revision="unknown:pre-contract"` rather than inventing one, since the
timm weights those features came from are genuinely unrecoverable.

## Known gaps

- **Packaging.** No installable distributions; imports rely on the repo root
  being on `sys.path`.
- **λ sweeps.** `kprobe` carries an already-selected λ per arm (a mapping, or
  one scalar for every arm); it does not *select* λ. Sweeping and selecting on
  pooled val cost is still the legacy pipeline's job.
- **IDR / penalized isotonic calibrators** are out of scope for now; the seam
  is documented in `kprobe/calibrate.py` and would need `CalibrationSpec`
  widened beyond affine maps.
- **The timm forward pass is untested** in CI (no timm locally). Verify a real
  extraction of one small backbone at `--max-train 256` before committing a
  sweep to it.
- **No CLI driver for the vast.ai wrapper.** `kprelogits/ops/vast.py` exposes
  `search_offers`, `create(dry_run=...)`, `teardown()` and
  `preteardown_check()` as tested library functions, but nothing ties them
  together — so "one orchestration path" is currently a library with no entry
  point, while the shell scripts that had one are archived.
