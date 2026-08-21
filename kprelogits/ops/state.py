"""Per-shard task state -- the supported way to ask "is that box done?".

**Never poll process liveness over SSH.** ``ssh host 'pgrep -f extract'``
matches the remote shell running that very command, so the count is never zero
and ``pkill`` kills its own shell first, returning empty output that reads like
success. This cost the project three incidents (see the ``vastai`` ops doc, and
the standing ``pgrep-self-match-in-ssh-polls`` note). The bracket trick
(``[e]xtract``) does not save you either: a shell variable *named* after the
pattern self-matches too.

So a shard publishes what it is doing. ``ShardState`` is a small JSON document
written atomically to local disk and mirrored to S3 after every transition; the
driver reads S3 and never touches the box. Phases are linear:

    pending -> running -> uploading -> done
                  |            |
                  +------------+-------> failed

``done`` means the shard finished its model list AND its final sync returned 0.
That distinction is the whole point: "the process exited" and "the features are
in S3" are different facts, and only the second one makes a box safe to
destroy.
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import producer_version

PHASES = ("pending", "running", "uploading", "done", "failed")
TERMINAL = ("done", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ShardState:
    """What one shard is doing, as published to S3."""

    data_flag: str
    rank: int
    shards: int
    phase: str = "pending"
    host: str = field(default_factory=socket.gethostname)
    pid: int = field(default_factory=os.getpid)
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    scheduled: int = 0          # models this shard is responsible for
    skipped: int = 0            # already present in S3 / on disk at start
    extracted: int = 0
    uploaded: int = 0
    failed: int = 0
    current: Optional[str] = None
    errors: List[Dict[str, str]] = field(default_factory=list)
    producer: str = field(default_factory=producer_version)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_complete(self) -> bool:
        """Every scheduled model accounted for and everything uploaded.

        Deliberately strict: a shard with a single failed model is not
        complete, so the pre-teardown check makes a human look at it.
        """
        return (self.phase == "done"
                and self.failed == 0
                and self.extracted + self.skipped == self.scheduled
                and self.uploaded == self.extracted)


class StateFile:
    """A ``ShardState`` bound to a local path and (optionally) an S3 key.

    Every mutation writes through: local atomically (tmp + ``os.replace``, so a
    reader never sees a half-written document), then S3 best-effort. An S3 push
    that fails is reported and does not abort extraction -- losing telemetry is
    not a reason to throw away GPU time -- but it does leave the last-known
    state stale, which the driver's final sync corrects.
    """

    def __init__(self, state: ShardState, path: Path, s3_prefix: str = "") -> None:
        self.state = state
        self.path = Path(path)
        self.s3_prefix = s3_prefix
        self.push_failures = 0

    @classmethod
    def create(cls, *, data_flag: str, rank: int, shards: int,
               results_dir: Path, s3_prefix: str = "", **kw) -> "StateFile":
        st = ShardState(data_flag=data_flag, rank=rank, shards=shards, **kw)
        path = Path(results_dir) / f"{data_flag}_lp" / f"state_shard{rank}.json"
        return cls(st, path, s3_prefix)

    @property
    def s3_uri(self) -> str:
        return f"{self.s3_prefix.rstrip('/')}/{self.path.name}" if self.s3_prefix else ""

    def write(self) -> None:
        self.state.updated_at = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state.to_json(), indent=2, default=str))
        os.replace(tmp, self.path)
        if self.s3_uri:
            from . import s3
            try:
                s3.put_json(self.state.to_json(), self.s3_uri, retries=1)
            except Exception as e:
                self.push_failures += 1
                print(f"  WARNING: state push failed ({type(e).__name__}: "
                      f"{str(e)[:120]}); local state at {self.path}")

    # ---- transitions -------------------------------------------------------

    def advance(self, phase: str, **fields) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected {PHASES}")
        self.state.phase = phase
        for k, v in fields.items():
            setattr(self.state, k, v)
        self.write()

    def begin_model(self, name: str) -> None:
        self.state.current = name
        self.write()

    def model_extracted(self) -> None:
        self.state.extracted += 1
        self.write()

    def model_uploaded(self) -> None:
        self.state.uploaded += 1
        self.write()

    def model_failed(self, name: str, stage: str, error: str) -> None:
        self.state.failed += 1
        self.state.errors.append({"model": name, "stage": stage,
                                  "error": str(error)[:300]})
        self.write()

    def fail(self, error: str) -> None:
        """Terminal failure of the shard itself (not one model)."""
        self.state.errors.append({"model": "*", "stage": "shard", "error": str(error)[:300]})
        self.advance("failed", current=None)


# ---------------------------------------------------------------------------
# Driver side
# ---------------------------------------------------------------------------

def read_states(s3_prefix: str) -> Dict[int, ShardState]:
    """Fetch every published shard state under a prefix, keyed by rank."""
    from . import s3
    out: Dict[int, ShardState] = {}
    for name in s3.list_names(s3_prefix):
        if not (name.startswith("state_shard") and name.endswith(".json")):
            continue
        doc = s3.get_json(f"{s3_prefix.rstrip('/')}/{name}")
        if doc is None:
            continue
        known = {f for f in ShardState.__dataclass_fields__}
        out[int(doc.get("rank", -1))] = ShardState(
            **{k: v for k, v in doc.items() if k in known})
    return out


def summarize(states: Dict[int, ShardState]) -> str:
    if not states:
        return "no shard states published"
    lines = []
    for rank in sorted(states):
        s = states[rank]
        lines.append(
            f"  shard {rank}/{s.shards} {s.phase:<9} "
            f"{s.extracted + s.skipped}/{s.scheduled} done, "
            f"{s.uploaded} uploaded, {s.failed} failed  "
            f"[{s.host} @ {s.updated_at}]"
            + (f"  <- {s.current}" if s.current else ""))
    return "\n".join(lines)
