"""S3 access via the ``aws`` CLI, with one retry policy.

The CLI rather than boto3 because it is what the container already has, and
because ``aws s3 sync`` is the safety net that rescues a resumed run's earlier
work. Everything funnels through ``_run`` so that "how many times do we retry,
and how loudly do we fail" is answered in exactly one place -- the legacy
extract script had three different answers inline.

Failure policy: reads (``ls``) degrade to a warning and an empty result,
because a failed listing should cost re-extraction at worst. Writes raise,
because a silent upload failure is how a GPU pass gets thrown away.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RETRIES = 3
BACKOFF_SECONDS = (2, 8)   # after attempt 1, after attempt 2


class S3Error(RuntimeError):
    """A write to S3 did not succeed after retries."""


def have_aws() -> bool:
    return shutil.which("aws") is not None


def _run(args: List[str], *, retries: int = 1) -> subprocess.CompletedProcess:
    """Run an aws command, retrying transient failures with backoff."""
    last: Optional[subprocess.CompletedProcess] = None
    for attempt in range(retries):
        last = subprocess.run(args, capture_output=True, text=True)
        if last.returncode == 0:
            return last
        if attempt + 1 < retries:
            time.sleep(BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)])
    assert last is not None
    return last


def _listing(uri: str) -> Tuple[bool, set, str]:
    """``(ok, names, error)`` for one prefix.

    ``aws s3 ls`` exits 1 both when it cannot reach the bucket AND when the
    prefix simply holds nothing -- and S3 has no directories, so a prefix with
    no objects under it does not exist in any meaningful sense. The two are
    distinguishable only by stderr: a real failure explains itself, an empty
    listing says nothing at all.

    Conflating them means every first run into a fresh prefix looks like a
    credentials or networking failure, which is exactly the case a resume
    oracle and a preflight both have to get right.
    """
    r = _run(["aws", "s3", "ls", uri.rstrip("/") + "/"])
    err = (r.stderr or "").strip()
    if r.returncode != 0 and err:
        return False, set(), err[:300]
    names = {ln.split()[-1] for ln in r.stdout.splitlines() if ln.strip()}
    return True, names, ""


def bucket_listable(uri: str) -> Tuple[bool, str]:
    """Can we see this prefix at all? Used by preflight, before any rental."""
    if not have_aws():
        return False, "aws CLI not on PATH"
    ok, _, err = _listing(uri)
    return ok, err


def list_names(uri: str, *, suffix: str = "") -> set:
    """Filenames directly under ``uri``. Empty (with a warning) on failure.

    This is the resume oracle: with cleanup on, local disk is emptied as we go,
    so S3 -- not the filesystem -- is the truth about what is already done.
    """
    if not uri:
        return set()
    ok, names, err = _listing(uri)
    if not ok:
        print(f"  WARNING: could not list {uri} ({err[:120]}); "
              f"falling back to local disk only")
        return set()
    return {n for n in names if n.endswith(suffix)} if suffix else names


def upload(local: Path, dest: str, *, retries: int = RETRIES) -> None:
    """Copy one file up. Raises ``S3Error`` -- callers decide whether that is
    fatal, but nobody gets to mistake a failure for a success."""
    r = _run(["aws", "s3", "cp", str(local), dest, "--only-show-errors"],
             retries=retries)
    if r.returncode != 0:
        raise S3Error(f"upload {local} -> {dest} failed rc={r.returncode}: "
                      f"{r.stderr.strip()[:300]}")


def download(src: str, local: Path, *, retries: int = RETRIES) -> None:
    r = _run(["aws", "s3", "cp", src, str(local), "--only-show-errors"],
             retries=retries)
    if r.returncode != 0:
        raise S3Error(f"download {src} failed rc={r.returncode}: "
                      f"{r.stderr.strip()[:300]}")


def put_json(obj, dest: str, *, retries: int = RETRIES) -> None:
    """Publish a small JSON document (config, state) to S3."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "payload.json"
        p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
        upload(p, dest, retries=retries)


def get_json(src: str) -> Optional[dict]:
    """Fetch and parse a small JSON document; None if absent or unparseable."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "payload.json"
        try:
            download(src, p, retries=1)
            return json.loads(p.read_text())
        except Exception:
            return None


def read_published_configs(prefix: str) -> Dict[str, dict]:
    """Every ``extract_config_shard*.json`` already published under ``prefix``.

    preflight uses these to detect that a new run would write bundles whose
    content disagrees with bundles already cached under the same names.
    """
    out: Dict[str, dict] = {}
    for name in list_names(prefix):
        if name.startswith("extract_config_shard") and name.endswith(".json"):
            doc = get_json(f"{prefix.rstrip('/')}/{name}")
            if doc is not None:
                out[name] = doc
    return out


def sync_dir(local_dir: Path, dest: str, *,
             exclude: Tuple[str, ...] = ("*.tmp", "*.part")) -> None:
    """Push every file in a directory. The safety net for a resumed run.

    Per-file uploads only cover what THIS run produced; a resumed run skips
    already-extracted models, so without this their bundles stay stranded on a
    box that is about to be destroyed.
    """
    args = ["aws", "s3", "sync", str(local_dir), dest.rstrip("/") + "/"]
    for pat in exclude:
        args += ["--exclude", pat]
    args.append("--only-show-errors")
    r = _run(args, retries=RETRIES)
    if r.returncode != 0:
        raise S3Error(f"sync {local_dir} -> {dest} failed rc={r.returncode}: "
                      f"{r.stderr.strip()[:300]}")
