"""``ExtractConfig`` and ``preflight`` -- fail before you rent.

A feature-extraction run costs real money the moment an instance boots, and
every failure mode this package has actually hit was knowable in advance: a
rank outside the shard range, a selection file that never got uploaded, S3
credentials that were never staged, a gated HuggingFace repo with no token, a
``--cleanup`` flag with nowhere to clean up *to*. ``preflight()`` checks all of
it against a resolved config and reports every problem at once, so a broken run
is caught on the laptop rather than twenty minutes into a rental.

Checks come in two kinds. Structural checks are pure -- no network, no
credentials, safe to run in a unit test. Probing checks (``check_s3``,
``check_hf``) talk to the outside world and can be turned off. Both feed the
same report, and ``raise_if_failed()`` is the single gate the driver calls.

The resolved config is JSON-serialized next to the bundles
(``{s3}/{flag}/extract_config_shard{rank}.json``) so a bundle's provenance is
recoverable from S3 alone, and so a *later* run can detect that it is about to
write cache-incompatible bundles under names an earlier run already used
(see ``bundle_identity``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import producer_version

# Selection entries above this are not skipped by kprelogits -- unlike ktrain's
# run_model(), which had a silent module-level gate. They are merely slow and
# OOM-prone, so preflight warns rather than errors.
LARGE_MODEL_PARAMS = 200_000_000

PRECISIONS = ("fp32", "fp16")

# Env vars that must never reach a remote command line (see ops/vast.py).
SECRET_ENV_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PASSWD")


@dataclass
class ExtractConfig:
    """One extraction run, fully resolved."""

    data_flag: str = "octmnist"
    data_dir: Path = Path("/workspace/data")
    results_dir: Path = Path("/workspace/results")

    # None -> {results_dir}/_shared/selected_models.json
    selection: Optional[Path] = None

    max_train: int = 10_000          # 0 = the full train split
    feature_batch_size: int = 128
    num_workers: int = 4
    precision: str = "fp32"

    rank: int = 0
    shards: int = 1

    s3_uri: str = ""                 # "" = keep bundles local (and say so loudly)
    cleanup: bool = False            # delete local .npz + weights AFTER upload

    hf_token_file: Optional[Path] = None   # never the token itself, only a path
    hf_home: Optional[Path] = None

    seed: int = 42

    # ---- derived locations -------------------------------------------------

    @property
    def selection_path(self) -> Path:
        if self.selection is not None:
            return Path(self.selection)
        return Path(self.results_dir) / "_shared" / "selected_models.json"

    @property
    def features_dir(self) -> Path:
        return Path(self.results_dir) / f"{self.data_flag}_lp" / "features"

    @property
    def s3_prefix(self) -> str:
        """Where this dataset's bundles live. Empty when S3 is not configured."""
        return f"{self.s3_uri.rstrip('/')}/{self.data_flag}" if self.s3_uri else ""

    def bundle_name(self, model_name: str) -> str:
        from .models import safe_name
        return f"{self.data_flag}_{safe_name(model_name)}_features.npz"

    def bundle_path(self, model_name: str) -> Path:
        return self.features_dir / self.bundle_name(model_name)

    @property
    def config_key(self) -> str:
        return f"extract_config_shard{self.rank}.json"

    @property
    def state_key(self) -> str:
        return f"state_shard{self.rank}.json"

    # ---- serialization -----------------------------------------------------

    def resolved(self) -> Dict[str, Any]:
        """JSON-safe dict of every field. Carries no secrets -- ``hf_token_file``
        is a path, and the token itself is never stored on the config."""
        d = asdict(self)
        for k in ("data_dir", "results_dir", "selection", "hf_token_file", "hf_home"):
            d[k] = str(d[k]) if d[k] is not None else None
        d["selection_resolved"] = str(self.selection_path)
        d["producer_version"] = producer_version()
        return d

    def bundle_identity(self) -> Dict[str, Any]:
        """The fields that determine what is *inside* a bundle of a given name.

        Bundle filenames are keyed by (dataset, model) only, so two runs
        differing in ``max_train`` or ``precision`` would write different
        content under the same key -- a silent cache poisoning. preflight
        compares this dict against any config already published alongside the
        bundles and refuses to proceed on a mismatch.
        """
        return {
            "data_flag": self.data_flag,
            "max_train": self.max_train,
            "precision": self.precision,
            "seed": self.seed,
        }

    # ---- construction ------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> "ExtractConfig":
        """Build from the container's env surface, then apply explicit overrides.

        The env names are the ones the vast.ai job scripts already export; the
        overrides are what a CLI flag sets.
        """
        def _s(name: str, default: str) -> str:
            return os.environ.get(name, "") or default

        def _i(name: str, default: int) -> int:
            raw = os.environ.get(name, "")
            return int(raw) if raw else default

        results = Path(_s("RESULTS_DIR", "/workspace/results"))
        bucket = _s("S3_BUCKET", "")
        prefix = _s("S3_FEATURES_PREFIX", "medmnist_prelogits").strip("/")

        cfg = cls(
            data_flag=_s("DATA_FLAG", "octmnist"),
            data_dir=Path(_s("DATA_DIR", "/workspace/data")),
            results_dir=results,
            max_train=_i("MAX_TRAIN", 10_000),
            feature_batch_size=_i("FEATURE_BATCH_SIZE", 128),
            num_workers=_i("NUM_WORKERS", 4),
            precision=_s("FEATURE_PRECISION", "fp32"),
            rank=_i("PARALLEL_RANK", 0),
            shards=_i("PARALLEL_SHARDS", 1),
            s3_uri=(f"s3://{bucket}/{prefix}" if bucket else ""),
            cleanup=_s("CLEANUP", "") == "1",
            hf_token_file=(Path(_s("HF_TOKEN_FILE", "")) if _s("HF_TOKEN_FILE", "") else None),
            hf_home=(Path(_s("HF_HOME", "")) if _s("HF_HOME", "") else None),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

@dataclass
class PreflightReport:
    """Every problem found, not just the first one."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> "PreflightReport":
        if self.errors:
            raise ValueError(
                "preflight failed:\n"
                + "\n".join(f"  ERROR   {e}" for e in self.errors)
                + ("\n" + "\n".join(f"  warning {w}" for w in self.warnings)
                   if self.warnings else "")
            )
        return self

    def render(self) -> str:
        lines = [f"  warning {w}" for w in self.warnings]
        lines += [f"  ERROR   {e}" for e in self.errors]
        return "\n".join(lines) if lines else "  all checks passed"


def read_hf_token(cfg: ExtractConfig) -> Optional[str]:
    """Resolve the HuggingFace token from a file, then the environment.

    The file form is what the box uses: staged once under ``umask 077`` and
    read here, so the token never appears in a command line or in ``ps``.
    """
    if cfg.hf_token_file is not None:
        p = Path(cfg.hf_token_file)
        if p.exists():
            tok = p.read_text().strip()
            if tok:
                return tok
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        tok = os.environ.get(name, "").strip()
        if tok:
            return tok
    return None


def _check_structure(cfg: ExtractConfig, rep: PreflightReport) -> None:
    if not cfg.data_flag:
        rep.errors.append("data_flag is empty")
    if cfg.shards < 1:
        rep.errors.append(f"shards must be >= 1, got {cfg.shards}")
    elif not (0 <= cfg.rank < cfg.shards):
        rep.errors.append(
            f"rank must be in [0, shards) = [0, {cfg.shards}), got {cfg.rank}")
    if cfg.precision not in PRECISIONS:
        rep.errors.append(f"precision must be one of {PRECISIONS}, got {cfg.precision!r}")
    if cfg.max_train < 0:
        rep.errors.append(f"max_train must be >= 0 (0 = full split), got {cfg.max_train}")
    if cfg.feature_batch_size < 1:
        rep.errors.append(f"feature_batch_size must be >= 1, got {cfg.feature_batch_size}")
    if cfg.num_workers < 0:
        rep.errors.append(f"num_workers must be >= 0, got {cfg.num_workers}")
    if cfg.cleanup and not cfg.s3_uri:
        rep.errors.append(
            "cleanup=True with no s3_uri would delete bundles with nowhere to "
            "put them -- set S3_BUCKET or turn cleanup off")
    if cfg.s3_uri and not cfg.s3_uri.startswith("s3://"):
        rep.errors.append(f"s3_uri must start with s3://, got {cfg.s3_uri!r}")
    if not cfg.s3_uri:
        rep.warnings.append(
            "no s3_uri -- bundles will exist only on this instance and are lost "
            "when it is destroyed")
    if cfg.hf_token_file is not None and not Path(cfg.hf_token_file).exists():
        rep.errors.append(f"hf_token_file does not exist: {cfg.hf_token_file}")


def _check_selection(cfg: ExtractConfig, rep: PreflightReport) -> Optional[List[dict]]:
    path = cfg.selection_path
    if not path.exists():
        rep.errors.append(
            f"no selection file at {path} -- run kprelogits/select.py first")
        return None
    try:
        with open(path) as f:
            sel = json.load(f)
    except Exception as e:
        rep.errors.append(f"selection {path} does not parse: {type(e).__name__}: {e}")
        return None
    models = sel.get("models")
    if not isinstance(models, list) or not models:
        rep.errors.append(f"selection {path} has no non-empty 'models' list")
        return None
    bad = [m for m in models if not isinstance(m, dict) or not m.get("name")]
    if bad:
        rep.errors.append(
            f"selection {path}: {len(bad)} entries lack a 'name' field")
        return None

    rep.facts["n_selected"] = len(models)
    # Only meaningful once the shard geometry itself is valid; _check_structure
    # has already reported it if not, and preflight's contract is to report
    # every problem rather than stop at the first.
    if cfg.shards >= 1 and 0 <= cfg.rank < cfg.shards:
        from .models import select_model_shard
        scheduled = select_model_shard([m["name"] for m in models], cfg.shards, cfg.rank)
        rep.facts["n_scheduled"] = len(scheduled)
        if not scheduled:
            rep.errors.append(
                f"shard {cfg.rank}/{cfg.shards} schedules 0 of {len(models)} models "
                f"-- this instance would do nothing")

    oversized = [m["name"] for m in models
                 if isinstance(m.get("params"), int) and m["params"] > LARGE_MODEL_PARAMS]
    if oversized:
        rep.warnings.append(
            f"{len(oversized)} selected model(s) exceed "
            f"{LARGE_MODEL_PARAMS/1e6:.0f}M params (e.g. {oversized[0]}); "
            f"kprelogits will extract them anyway -- expect OOM backoff and "
            f"long runtimes")
    return models


def _check_disk(cfg: ExtractConfig, rep: PreflightReport) -> None:
    probe = Path(cfg.results_dir)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / 1e9
    except Exception as e:
        rep.warnings.append(f"could not stat free disk at {probe}: {e}")
        return
    rep.facts["free_gb"] = round(free_gb, 1)
    if free_gb < 10:
        rep.errors.append(f"only {free_gb:.1f} GB free at {probe} -- one bundle "
                          f"can exceed that")
    elif free_gb < 50 and not cfg.cleanup:
        rep.warnings.append(
            f"{free_gb:.1f} GB free at {probe} and cleanup is off; bundles "
            f"accumulate at ~0.5-2 GB each -- consider cleanup=True")


def _check_s3(cfg: ExtractConfig, rep: PreflightReport) -> None:
    if not cfg.s3_uri:
        return
    from .ops import s3
    if not s3.have_aws():
        rep.errors.append("aws CLI not found on PATH but s3_uri is set")
        return
    ok, detail = s3.bucket_listable(cfg.s3_prefix)
    if not ok:
        rep.errors.append(f"cannot list {cfg.s3_prefix}: {detail}")
        return
    rep.facts["s3_reachable"] = True

    # Cache-poisoning guard: a bundle name encodes only (dataset, model), so a
    # run with a different max_train/precision would overwrite content under a
    # name an earlier run already published.
    for key, other in s3.read_published_configs(cfg.s3_prefix).items():
        theirs = {k: other.get(k) for k in cfg.bundle_identity()}
        if any(v is not None for v in theirs.values()) and theirs != cfg.bundle_identity():
            rep.errors.append(
                f"{key} in S3 was written with {theirs}, this run has "
                f"{cfg.bundle_identity()}. Bundle names do not encode these, so "
                f"proceeding would mix incompatible features under one cache "
                f"key. Use a different S3_FEATURES_PREFIX or match the settings.")


def _is_network_error(e: BaseException) -> bool:
    """A failure to reach the Hub at all, as opposed to the Hub answering 'no'.

    requests' exceptions (ConnectionError, SSLError, Timeout, and the Hub's
    HTTP errors) all derive from OSError; GatedRepoError and
    RepositoryNotFoundError are caught before this is consulted.
    """
    return isinstance(e, (OSError, TimeoutError)) or type(e).__name__ in {
        "LocalEntryNotFoundError", "OfflineModeIsEnabled"}


def _check_hf(cfg: ExtractConfig, models: Optional[List[dict]],
              rep: PreflightReport, sample: int = 5) -> None:
    token = read_hf_token(cfg)
    rep.facts["hf_token"] = "present" if token else "absent"
    if not models:
        return
    try:
        from huggingface_hub import model_info
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except Exception:
        rep.warnings.append(
            "huggingface_hub not importable -- skipped the gated-repo probe; a "
            "gated backbone will fail at download time instead of here")
        return

    names = [m["name"] for m in models][:max(1, sample)]
    gated, unknown, unreachable = [], [], []
    for name in names:
        try:
            model_info(f"timm/{name}", token=token)
        except GatedRepoError:
            gated.append(name)
        except RepositoryNotFoundError:
            # Also what the Hub returns for a gated repo to an unauthorized
            # caller, so it is only conclusive when we did send a token.
            (unknown if token else gated).append(name)
        except Exception as e:
            unknown.append(f"{name} ({type(e).__name__})")
            if _is_network_error(e):
                unreachable.append(f"{type(e).__name__}: "
                                   f"{' '.join(str(e).split())[:160]}")

    # Every probe failing to CONNECT is not a warning: it is a machine that
    # cannot download a single weight file. On 2026-09-14 a vast host whose
    # DNS answered huggingface.co with a Meta IPv6 address passed preflight
    # on exactly this -- "could not resolve 5 repo(s)" was only a warning --
    # then failed every model of its shard one TLS error at a time.
    if names and len(unreachable) == len(names):
        rep.errors.append(
            f"cannot reach the HuggingFace Hub from this machine: all "
            f"{len(names)} probes failed to connect (e.g. {unreachable[0]}). "
            f"Every weight download would fail; use a machine with working "
            f"HTTPS to huggingface.co.")
    if gated and not token:
        rep.errors.append(
            f"no HF token, and {len(gated)} of {len(names)} probed repos are "
            f"gated or private (e.g. timm/{gated[0]}). Stage a token with "
            f"hf_token_file, or drop the gated models from the selection.")
    elif gated:
        rep.errors.append(
            f"HF token present but still denied for: {', '.join(gated)} -- the "
            f"token lacks access to these repos")
    if unknown:
        rep.warnings.append(
            f"could not resolve {len(unknown)} repo(s) during the HF probe: "
            f"{', '.join(unknown[:3])}")


def cuda_arch_supported(capability, arch_list) -> bool:
    """Whether a torch build can run kernels on a GPU of this compute capability.

    A cubin built for sm_Mm runs on any sm_Mn with the same major and n >= m --
    sm_86 code runs on an sm_89 RTX 4060 Ti, which is why those cards work
    under a torch that lists no sm_89 -- and PTX for compute_Mm can be JIT-
    compiled for anything at or above it. Nothing else runs. The image's torch
    2.5.1/cu124 stops at sm_90, so an RTX 5060 Ti (12.0) fails its first CUDA
    op with "no kernel image is available for execution on the device".
    """
    major, minor = capability
    for arch in arch_list:
        kind, _, num = str(arch).partition("_")
        if len(num) < 2 or not num.isdigit():
            continue
        a_major, a_minor = int(num[:-1]), int(num[-1])
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def _check_cuda_arch(rep: PreflightReport) -> None:
    """On a GPU box, refuse a card this torch build has no kernels for.

    Failing here costs one box, which the driver releases and replaces.
    Failing later cost a whole shard: on 2026-09-14 a Blackwell box passed
    every other check, launched, and failed each model in turn. On a laptop
    there is nothing to judge, so the absence is recorded and nothing more.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            rep.facts["cuda"] = "absent"
            return
        cap = tuple(torch.cuda.get_device_capability(0))
        arches = list(torch.cuda.get_arch_list())
    except Exception as e:
        rep.facts["cuda"] = f"unprobed ({type(e).__name__})"
        return
    rep.facts["cuda_capability"] = f"{cap[0]}.{cap[1]}"
    if not cuda_arch_supported(cap, arches):
        rep.errors.append(
            f"GPU compute capability {cap[0]}.{cap[1]} has no kernels in this "
            f"torch build (arch list {arches}); every model would fail with 'no "
            f"kernel image is available'. Rent a card this build supports.")


def preflight(cfg: ExtractConfig, *, check_s3: bool = True,
              check_hf: bool = True, hf_sample: int = 5) -> PreflightReport:
    """Validate a config. Structural checks always run; the probing checks
    (S3 reachability, HF gating) are opt-out so tests stay offline."""
    rep = PreflightReport()
    _check_structure(cfg, rep)
    models = _check_selection(cfg, rep)
    _check_disk(cfg, rep)
    _check_cuda_arch(rep)
    if check_s3:
        _check_s3(cfg, rep)
    if check_hf:
        _check_hf(cfg, models, rep, sample=hf_sample)
    return rep


def check_no_secrets(env: Dict[str, str]) -> None:
    """Refuse an env mapping that would put a credential on a command line.

    Vast hosts are shared and ``ps`` is world-readable; a key was exposed this
    way earlier in the project. Anything credential-shaped must be staged in a
    ``umask 077`` file instead (``ops.vast.stage_credentials``).
    """
    leaked = [k for k in env if any(h in k.upper() for h in SECRET_ENV_HINTS)]
    if leaked:
        raise ValueError(
            f"refusing to place {sorted(leaked)} on a command line -- these are "
            f"visible in `ps` on a shared host. Stage them with "
            f"ops.vast.stage_credentials() instead.")


def which_aws() -> Optional[str]:
    """Small helper used by preflight output and the ops layer."""
    return shutil.which("aws")


def git_available() -> bool:
    try:
        return subprocess.run(["git", "--version"], capture_output=True).returncode == 0
    except Exception:
        return False
