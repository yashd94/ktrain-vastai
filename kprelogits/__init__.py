"""kprelogits -- frozen pre-logit extraction on rented GPUs.

Runs each timm backbone over a dataset exactly once and writes the pre-logit
features to a bundle in S3. This is the only stage that needs a GPU, and the
only stage whose output is expensive enough that losing it means re-renting
hardware -- which is why so much of this package is about *not losing it*:
atomic writes, upload-confirmed cleanup, per-shard state files, and instance
teardown that runs in ``finally``.

Bundles are the hand-off to kprobe, which now lives in its own repo. The two
never import each other; they agree only on the format specified in
``contracts.py``, which this package stamps into every bundle and a consumer
validates on read.

Layout:
    contracts.py  the prelogit bundle format -- the whole public interface
    config.py   ExtractConfig + preflight() -- fail before you rent
    models.py   loaders, encoder, extraction (owned code, formerly ktrain's)
    select.py   which backbones to run
    extract.py  the driver
    ops/        state files, S3, vast.ai instance lifecycle
"""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path

__version__ = "0.1.0"

_PKG_DIR = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def producer_version() -> str:
    """Identity stamped into every bundle: package version + git description.

    A bundle whose numbers came from a dirty tree must say so -- otherwise the
    only record of which code produced a multi-GB artifact is a version string
    that nobody bumped. Falls back to the bare version outside a git checkout.
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(_PKG_DIR), "describe", "--always", "--dirty", "--tags"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return f"kprelogits/{__version__}+{r.stdout.strip()}"
    except Exception:
        pass
    return f"kprelogits/{__version__}"
