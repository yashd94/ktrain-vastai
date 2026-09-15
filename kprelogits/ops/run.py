"""Rent N boxes, extract one model list across them, destroy every box.

This is the multi-shard driver `ops` was missing. Every part of it already
existed and was tested -- offers, create, per-instance key attach, credential
staging, shard state files, teardown, the pre-teardown check -- and nothing
composed them across more than one box, so a real sweep meant driving four
rentals by hand. Hand-driven rentals are how this project paid for a 4.3-hour
orphan.

It borrows the connection details from ``smoke``, which is backwards on paper
-- production importing the smoke test -- and deliberate: those functions
encode eight rentals' worth of findings (both ssh routes tried every round,
``PubkeyAcceptedAlgorithms`` for RSA keys, a per-run known_hosts, secrets
sourced in every remote shell, a launch that may legitimately hang the
channel). Re-typing them here to respect a layering diagram would mean
re-earning them. When they move to a shared module, both callers move.

Three things differ from ``smoke``, beyond the box count:

**Every instance id enters the teardown list the moment it exists.** ``smoke``
acquires outside the teardown context and covers the gap with a BaseException
handler. With four boxes that gap is four times as wide and spans the whole
acquisition loop, so here the context is entered *first*, holding a mutable
list, and an id is appended as soon as ``create`` returns one -- including when
the box turns out to be unusable and is destroyed a moment later.

**A shard's box dies when its shard does**, not when the run ends. The model
list is not uniform and shards finish minutes to an hour apart; a box that has
uploaded its last bundle is pure burn. Early release is gated on the same
evidence as the final check: that shard's *own* bundles, listed in S3 by name.

**Each box is staged and launched before the next is rented, and finished
shards are released during acquisition rather than after it.** Renting eight
boxes serially takes the better part of an hour once a couple fail to boot,
which is long enough for the first shard to finish inside that window --
so watching only after the last box launches means paying for every early
finisher to idle until the slowest rental completes.

Usage::

    python -m kprelogits.ops.run --selection artifacts/dermamnist_pilot50.json \\
        --data-flag dermamnist --shards 4              # plan; rents nothing
    python -m kprelogits.ops.run ... --shards 4 --go   # rent and run
    python -m kprelogits.ops.run --abort               # destroy anything run-*
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import s3, state, vast
from .smoke import (  # see the module docstring on why these come from here
    DATA_TIMEOUT, IMAGE, OFFER_QUERY, PIP_PACKAGES, REMOTE_PATH, SETUP_TIMEOUT,
    SOURCE_SECRETS, _sh, _ssh, aws_credentials, code_tarball, tail_remote_log,
    wait_for_running, wait_for_ssh,
)

BUCKET = "pandora-linear-probe-inputs-939723541836-us-east-1-an"
PREFIX = "medmnist_prelogits"

# A little over smoke's 40 GB: cleanup deletes each bundle after upload, so
# what accumulates is the HuggingFace cache -- a dozen backbones of up to 50M
# params, a few GB -- on top of the 1.1 GB dataset and the image.
#
# It must stay UNDER the ``disk_space`` floor in OFFER_QUERY. Asking for more
# than the offer filter guarantees means a create that fails, or succeeds onto
# a host with no room, and the disk-full failure lands hours later with the
# work already paid for.
DISK_GB = 45
assert DISK_GB < 50, "OFFER_QUERY filters disk_space>=50; keep DISK_GB below it"

MAX_TRAIN = 10_000

# Where the selection lands on the box. A constant, and used by BOTH the scp
# destination and the --selection argument, because the local file is named
# after whatever study it belongs to and the remote name must not depend on
# that. Copying to a directory instead ("scp file host:/workspace/") keeps the
# local basename, which is how the first attempt shipped
# dermamnist_pilot50.json and then asked the worker for selection.json.
REMOTE_SELECTION = "/workspace/selection.json"
REMOTE_TARBALL = "/workspace/kprelogits.tgz"

# smoke's offer query, with the VRAM floor raised. The full corpus runs to
# 196M params; at batch 128 over 224x224 those do not fit an 8 GB card, and an
# OOM would fail the same large models on every shard at once. 16 GB is
# plentiful on the marketplace (64 offers at the time of writing) and costs
# about $0.02/hr more.
RUN_OFFER_QUERY = OFFER_QUERY.replace("gpu_ram>=8", "gpu_ram>=16")

# Deadlines, sized for a shard of ~100 backbones rather than smoke's one.
FIRST_STATE_DEADLINE = 5 * 60
# The worker publishes state on every model transition, so silence longer than
# one model's extraction means something is wedged. The largest backbones are
# ~196M params over ~10k images, plus a cold HuggingFace download of ~800 MB.
STALE_DEADLINE = 30 * 60
# Generous on purpose: the whole corpus is ~36B params of forward passes, and
# the cost of a deadline that fires early is a destroyed box mid-shard, while
# the cost of one that fires late is bounded by the stale deadline anyway.
JOB_DEADLINE = 12 * 60 * 60
POLL_SECONDS = 45


def run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def s3_prefix(data_flag: str, prefix: str = PREFIX) -> str:
    """Where bundles land -- matching ``ExtractConfig.s3_prefix`` exactly.

    The worker appends the data flag itself, so a mismatch here would poll a
    prefix nothing writes to and report a job that never started.
    """
    return f"s3://{BUCKET}/{prefix}/{data_flag}"


def worker_env(data_flag: str, rank: int, shards: int,
               prefix: str = PREFIX) -> Dict[str, str]:
    """Non-secret config for one shard. Credentials are staged, never here."""
    return {
        "S3_BUCKET": BUCKET,
        "S3_FEATURES_PREFIX": prefix,
        "DATA_FLAG": data_flag,
        "MAX_TRAIN": str(MAX_TRAIN),
        "DATA_DIR": "/workspace/data",
        "RESULTS_DIR": "/workspace/results",
        "PARALLEL_RANK": str(rank),
        "PARALLEL_SHARDS": str(shards),
        # Upload then delete. Not for disk (bundles are small here) but because
        # S3 is the resume oracle: a bundle that exists only on the box is work
        # that vanishes with the box.
        "CLEANUP": "1",
        "HF_HOME": "/workspace/hf",
    }


def shard_models(models: Sequence[str], shards: int, rank: int) -> List[str]:
    """The names one shard owns -- the worker's own split, computed locally.

    Duplicating the worker's partition is what makes early release safe: it
    turns "shard 2 says done" into "these eleven named bundles are in S3".
    """
    from ..models import select_model_shard
    return select_model_shard(list(models), shards, rank)


def missing_bundles(prefix: str, data_flag: str,
                    names: Sequence[str]) -> List[str]:
    """Which of ``names`` are not yet in S3. The only completion evidence."""
    have = s3.list_names(prefix, suffix="_features.npz")
    return [n for n in names if f"{data_flag}_{n}_features.npz" not in have]


def remote_script(data_flag: str, rank: int, shards: int, prefix: str,
                  selection_remote: str = REMOTE_SELECTION) -> List[str]:
    """The four remote commands for one shard, in order.

    Every command that reaches S3 sources the staged secrets first: each ssh
    call is its own shell, nothing carries over, and an unsourced ``aws`` fails
    with "Unable to locate credentials" -- trivial to fix, expensive to
    discover on a box that is already billing.
    """
    env = " ".join(f"{k}={v}" for k, v in
                   sorted(worker_env(data_flag, rank, shards, prefix).items()))
    base = f"{REMOTE_PATH}; {SOURCE_SECRETS} cd /workspace && {env} python -m kprelogits.extract"
    return [
        f"{REMOTE_PATH}; pip install -q {PIP_PACKAGES}",
        f"{REMOTE_PATH}; {SOURCE_SECRETS} aws s3 cp "
        f"s3://{BUCKET}/medmnist/{data_flag}_224.npz "
        f"/workspace/data/{data_flag}_224.npz",
        f"{base} --selection {selection_remote} --dry-run",
        # setsid + nohup + all three descriptors redirected: the job must
        # outlive this ssh session, and ssh must not hold the channel open on
        # an inherited descriptor. The trailing echo says the shell got that
        # far; it is not a claim that the job succeeded.
        f"{base.replace('python -m', 'setsid nohup python -m')} "
        f"--selection {selection_remote} "
        f"> /workspace/extract.log 2>&1 < /dev/null & sleep 2; echo launched",
    ]


# ---------------------------------------------------------------------------
# One box
# ---------------------------------------------------------------------------

class Box:
    """One rented instance and the shard it is running."""

    def __init__(self, rank: int, instance_id: int, target: List[str]) -> None:
        self.rank = rank
        self.instance_id = instance_id
        self.target = target
        self.host = target[-1].split("@")[-1]
        self.user = target[-1].split("@")[0]
        self.port = int(target[target.index("-p") + 1])
        self.released = False

    def __repr__(self) -> str:
        return f"<box rank={self.rank} id={self.instance_id} {self.host}:{self.port}>"


class NoUsableBox(RuntimeError):
    """Every offer tried for one rank failed to give a box that boots and
    answers ssh. A RuntimeError subclass so a caller can skip that rank instead
    of ending the run -- see acquire_with_refresh."""


def acquire(rank: int, offers: List[dict], pubkey: str, label_base: str,
            live: List[int], known_hosts: Optional[Path],
            env: Dict[str, str], tried: Optional[set] = None
            ) -> Tuple[Box, List[dict]]:
    """Rent until one box boots and answers ssh. Returns it and the unused offers.

    ``live`` is the teardown list, mutated in place: an id goes in the instant
    ``create`` hands one back, before anything can go wrong with it. A box that
    then proves unusable is destroyed here rather than left to the outer
    teardown, because the next rental is about to start and two live instances
    is how a burn rate doubles unnoticed.

    Consumed offers are dropped from the returned list so the next rank cannot
    try to rent a machine this one is already holding.
    """
    label = f"{label_base}-r{rank}"
    remaining = list(offers)
    last_error = None

    while remaining:
        offer = remaining.pop(0)
        if tried is not None:
            tried.add(int(offer["id"]))
        print(f"  [r{rank}] offer {offer.get('id')} "
              f"${float(offer.get('dph_total') or 0):.3f}/hr {offer.get('gpu_name')} "
              f"cc={offer.get('compute_cap')} "
              f"cpu_eff={float(offer.get('cpu_cores_effective') or 0):.1f} "
              f"down={offer.get('inet_down')}Mbps {offer.get('geolocation')}",
              flush=True)
        instance_id = None
        try:
            try:
                instance_id = vast.create(int(offer["id"]), image=IMAGE,
                                          disk=DISK_GB, label=label, ssh=True,
                                          env=env)
            except Exception as ce:
                # A create that failed locally does not prove nothing was
                # rented. Look for the label before moving on, or the lost box
                # bills unattended.
                stray = vast.find_by_label(label)
                if stray:
                    instance_id = int(stray[0]["id"])
                    print(f"  [r{rank}] create reported {type(ce).__name__} but "
                          f"{instance_id} exists under {label} -- adopting",
                          flush=True)
                else:
                    raise
            live.append(instance_id)
            print(f"  [r{rank}] instance {instance_id} (teardown list: {live})",
                  flush=True)

            inst = wait_for_running(instance_id)
            print(f"  [r{rank}] {inst.get('gpu_name')} "
                  f"cpu_eff={inst.get('cpu_cores_effective')}/{inst.get('cpu_cores')} "
                  f"${float(inst.get('dph_total') or 0):.3f}/hr", flush=True)

            # Per-instance, not account-wide: team accounts cannot hold
            # account-wide keys, and this grant dies with the box anyway.
            vast.attach_ssh_key(instance_id, pubkey)
            routes = vast.ssh_endpoints(instance_id)
            target = wait_for_ssh(
                routes, known_hosts=known_hosts,
                on_retry=lambda: vast.attach_ssh_key(instance_id, pubkey))
            return Box(rank, instance_id, target), remaining

        # BaseException: this spans ten minutes of provisioning plus ssh
        # retries, the likeliest moment for a Ctrl-C, and an interrupt that
        # walks past a live instance leaves it billing.
        except BaseException as e:
            if instance_id is not None:
                print(f"  [r{rank}] box {instance_id} unusable: "
                      f"{type(e).__name__}: {str(e)[:160]}", flush=True)
                try:
                    if vast.destroy(instance_id) and instance_id in live:
                        live.remove(instance_id)
                except Exception as de:
                    print(f"  [r{rank}] DESTROY FAILED {instance_id}: {de} -- "
                          f"left in the teardown list", flush=True)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            last_error = e
            print(f"  [r{rank}] trying the next offer", flush=True)

    raise NoUsableBox(f"rank {rank}: no offer produced a usable box; "
                      f"last: {last_error}")


def acquire_with_refresh(rank: int, offers: List[dict], tried: set, *, search,
                         pubkey: str, label_base: str, live: List[int],
                         known_hosts: Optional[Path], env: Dict[str, str]):
    """acquire(), except that running out of offers is neither final nor fatal.

    Offers are a snapshot taken before the first rental, and on a bad day one
    rank can burn through all of them. 2026-09-13's OCTMNIST sweep lost seven
    boxes in a row on rank 6, left rank 7 nothing, and the exception that
    followed ended the run -- whose teardown then destroyed ranks 4 and 6 in
    the middle of their shards. So: re-search the live marketplace once,
    skipping every offer already tried; if that fails too, skip the rank and
    return ``(None, [])``. Its models stay unextracted and a re-run picks them
    up (S3 is the resume oracle), which costs far less than the working boxes
    a fatal error would take down with it.
    """
    last = None
    for attempt in (1, 2):
        try:
            return acquire(rank, offers, pubkey, label_base, live, known_hosts,
                           env, tried=tried)
        except NoUsableBox as e:
            last = e
        if attempt == 1:
            offers = [o for o in search() if int(o["id"]) not in tried]
            print(f"  [r{rank}] offers exhausted; re-searched the marketplace: "
                  f"{len(offers)} untried offer(s)", flush=True)
            if not offers:
                break
    print(f"  [r{rank}] SKIPPED -- no usable box ({last}). Its models stay "
          f"unextracted; re-running this command picks them up.", flush=True)
    return None, []


STAGE_ATTEMPTS = 3


def bring_up(rank: int, offers: List[dict], tried: set, *, stage, live: List[int],
             attempts: int = STAGE_ATTEMPTS, **acquire_kw):
    """Acquire a box for ``rank`` and stage + launch it. A box that fails
    staging is released and replaced; a rank that runs out is skipped.

    Staging used to run bare in the acquisition loop, so ONE box whose remote
    preflight refused -- a GPU this torch build cannot run, a pip or data-pull
    failure -- raised straight out of it, and the teardown destroyed every
    other box in the run. That is the same run-killing shape as an exhausted
    offer list, and it gets the same answer. Returns ``(box | None, offers)``.
    """
    for attempt in range(1, attempts + 1):
        box, offers = acquire_with_refresh(rank, offers, tried, live=live,
                                           **acquire_kw)
        if box is None:
            return None, offers
        try:
            stage(box)
            return box, offers
        except Exception as e:
            print(f"  [r{rank}] staging failed on {box} (attempt {attempt}/"
                  f"{attempts}): {type(e).__name__}: "
                  f"{' '.join(str(e).split())[:300]} -- releasing it", flush=True)
            release(box, live)
    print(f"  [r{rank}] SKIPPED -- {attempts} boxes failed staging. Its models "
          f"stay unextracted; re-running this command picks them up.", flush=True)
    return None, offers


def stage_and_launch(box: Box, *, tarball: Path, selection: Path,
                     creds: Dict[str, str], data_flag: str, shards: int,
                     prefix: str, known_hosts: Optional[Path]) -> None:
    """Ship code, stage credentials, pull data, preflight, launch."""
    r = box.rank
    scp = vast.scp_args(box.host, box.port, box.user, known_hosts=known_hosts)
    # One scp per file, each to an explicit destination PATH rather than a
    # directory. Copying into a directory keeps the local basename, and the
    # remote commands refer to fixed names -- so a selection file named after
    # its study arrives as itself and the worker is asked for something else.
    for local, remote in ((tarball, REMOTE_TARBALL),
                          (selection, REMOTE_SELECTION)):
        _sh(["scp", *scp, str(local), f"{box.user}@{box.host}:{remote}"],
            timeout=300, label=f"r{r} scp {Path(local).name}")
    _ssh(box.target, f"cd /workspace && tar xzf {REMOTE_TARBALL} && "
                     f"rm -f {REMOTE_TARBALL} && mkdir -p data results hf && "
                     f"test -s {REMOTE_SELECTION}",
         timeout=120, label=f"r{r} untar")
    vast.stage_credentials(box.target, creds)

    cmds = remote_script(data_flag, r, shards, prefix)
    _ssh(box.target, cmds[0], timeout=SETUP_TIMEOUT, label=f"r{r} pip")
    _ssh(box.target, cmds[1], timeout=DATA_TIMEOUT, label=f"r{r} data pull")
    out = _ssh(box.target, cmds[2], timeout=300, label=f"r{r} dry-run")
    print(f"  [r{r}] preflight tail: {out.strip().splitlines()[-1][:120]}"
          if out.strip() else f"  [r{r}] preflight produced no output")

    # A hung launch is not a failed launch. Detaching through ssh is fiddly and
    # the channel stays open while anything holds it; what matters is whether
    # the worker started, and the state file answers that directly.
    try:
        _ssh(box.target, cmds[3], timeout=60, label=f"r{r} launch",
             stdin_null=True)
        print(f"  [r{r}] launched", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [r{r}] launch did not return (ssh held the channel); "
              f"the state file decides", flush=True)
    except Exception as e:
        print(f"  [r{r}] launch errored: {type(e).__name__}: {e}", flush=True)


# ---------------------------------------------------------------------------
# Watching all of them
# ---------------------------------------------------------------------------

def release(box: Box, live: List[int]) -> None:
    """Destroy one box and drop it from the teardown list, if it really died."""
    print(f"  [r{box.rank}] releasing instance {box.instance_id}", flush=True)
    try:
        if vast.destroy(box.instance_id):
            if box.instance_id in live:
                live.remove(box.instance_id)
            box.released = True
        else:
            print(f"  [r{box.rank}] destroy unconfirmed -- left in the "
                  f"teardown list", flush=True)
    except Exception as e:
        print(f"  [r{box.rank}] destroy error {type(e).__name__}: {e} -- "
              f"left in the teardown list", flush=True)


class Watch:
    """Watches shards and releases each box as its own shard lands.

    A class rather than one ``poll`` function because the *acquisition loop*
    needs to release boxes too. Renting eight boxes serially takes the better
    part of an hour once a couple of them fail to boot, and the first shard
    can finish inside that window -- so a driver that only starts watching
    after the last box launches pays for every finished shard to sit idle
    until the slowest rental completes. Acquisition sweeps without deadlines
    (a shard that has not launched yet must not be judged for having published
    no state); the polling loop sweeps with them.
    """

    def __init__(self, *, prefix: str, data_flag: str,
                 assignments: Dict[int, List[str]], live: List[int],
                 scratch: Path) -> None:
        self.prefix = prefix
        self.data_flag = data_flag
        self.assignments = assignments
        self.live = live
        self.scratch = scratch
        self.boxes: Dict[int, Box] = {}
        self.pending: set = set()
        self.verdicts: Dict[int, str] = {}
        self.launched_at: Dict[int, float] = {}
        self._last_seen: Dict[int, tuple] = {}
        self._last_change: Dict[int, float] = {}

    def add(self, box: Box) -> None:
        """Register a launched box. Its deadlines start now, not at run start."""
        self.boxes[box.rank] = box
        self.pending.add(box.rank)
        self.launched_at[box.rank] = time.time()
        self._last_change[box.rank] = time.time()

    def _land(self, rank: int, verdict: str, *, rescue_it: bool) -> None:
        box = self.boxes[rank]
        self.verdicts[rank] = verdict
        if rescue_it:
            tail_remote_log(box.target)
            rescue(box, self.scratch / f"rescue-r{rank}", self.data_flag)
        release(box, self.live)
        self.pending.discard(rank)

    # Allowance for the box's clock running behind this one.
    CLOCK_SKEW = 120

    def _foreign(self, st, rank: int) -> bool:
        """Whether a state file was written by some other run's worker.

        State files are named by rank alone, so a prefix that has seen an
        earlier run already holds a ``state_shard{rank}.json`` -- usually a
        ``done`` one -- before this box's worker gets around to overwriting it.
        Read as this box's, it condemns a box that launched seconds ago: "done
        but 109 bundles missing", rescue, destroy. Re-running into a prefix is
        the documented way to resume, and the way to reshard, so this is not a
        corner case. It went unnoticed only because the launch command tends to
        hold the ssh channel for its full 60 s timeout, by which time the new
        worker has written.

        A different shard count, or a start before this box was launched, means
        the file is not this box's. Unparseable start times count as foreign:
        acting on a file that cannot be attributed is how boxes get destroyed.
        """
        shards = getattr(st, "shards", None)
        if shards is not None and shards != len(self.assignments):
            return True
        raw = getattr(st, "started_at", None)
        if not raw:
            return False
        try:
            from datetime import datetime
            started = datetime.fromisoformat(raw).timestamp()
        except (TypeError, ValueError):
            return True
        return started < self.launched_at[rank] - self.CLOCK_SKEW

    def sweep(self, *, deadlines: bool, quiet: bool = False) -> None:
        """One pass over every pending shard. Releases those that have landed."""
        if not self.pending:
            return
        states = state.read_states(self.prefix)
        if not quiet:
            print(f"  {state.summarize(states)}", flush=True)

        for rank in sorted(self.pending):
            st = states.get(rank)
            if st is not None and self._foreign(st, rank):
                st = None       # an earlier run's file under this rank's name
            since_launch = time.time() - self.launched_at[rank]

            if st is None:
                if deadlines and since_launch > FIRST_STATE_DEADLINE:
                    self._land(rank, "no state file; the worker never started",
                               rescue_it=True)
                continue

            fp = (st.phase, st.extracted, st.uploaded, st.failed, st.updated_at,
                  st.current)
            if fp != self._last_seen.get(rank):
                self._last_seen[rank] = fp
                self._last_change[rank] = time.time()

            if st.phase in ("done", "failed"):
                # "done" is a claim. The check of it is whether this shard's
                # own bundles are actually in S3, by name.
                missing = missing_bundles(self.prefix, self.data_flag,
                                          self.assignments[rank])
                if st.phase == "done" and not missing and st.is_complete:
                    self._land(rank, "done", rescue_it=False)
                else:
                    why = (f"failed: {st.errors[-1] if st.errors else '?'}"
                           if st.phase == "failed"
                           else f"done but {len(missing)} bundle(s) missing "
                                f"from S3: {missing[:3]}")
                    self._land(rank, why, rescue_it=True)
                continue

            if deadlines and time.time() - self._last_change[rank] > STALE_DEADLINE:
                self._land(rank, f"no state change for "
                                 f"{STALE_DEADLINE // 60} min while {st.phase}",
                           rescue_it=True)

    def poll(self) -> Dict[int, str]:
        """Sweep to a decision for every shard. Returns rank -> verdict."""
        started = time.time()
        while self.pending:
            elapsed = time.time() - started
            if elapsed > JOB_DEADLINE:
                for r in sorted(self.pending):
                    self.verdicts[r] = (f"job deadline "
                                        f"({JOB_DEADLINE // 3600}h) exceeded")
                break
            print(f"  [{elapsed / 60:5.1f}m]", end=" ", flush=True)
            self.sweep(deadlines=True)
            if self.pending:
                time.sleep(POLL_SECONDS)
        return self.verdicts


def rescue(box: Box, out_dir: Path, data_flag: str) -> None:
    """Pull what is worth keeping off a box about to be destroyed.

    Bounded on purpose. The worker prints DO NOT DESTROY when its final sync
    fails, which is right for a shard holding hours of work -- but the answer
    is to rescue it and then destroy, not to leave an instance billing while
    someone decides.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for remote in ("/workspace/extract.log",
                   f"/workspace/results/{data_flag}_lp/"
                   f"extract_report_shard{box.rank}.json",
                   f"/workspace/results/{data_flag}_lp/features/"):
        try:
            _sh(["scp", "-r", *vast.scp_args(box.host, box.port, box.user),
                 f"{box.user}@{box.host}:{remote}", str(out_dir)],
                timeout=600, label=f"r{box.rank} rescue {remote}")
        except Exception as e:
            print(f"  [r{box.rank}] rescue of {remote} failed: "
                  f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def load_models(selection: Path) -> List[str]:
    doc = json.loads(Path(selection).read_text())
    names = [m["name"] for m in doc["models"]]
    if not names:
        raise ValueError(f"{selection} has no models")
    return names


def plan(args, models: List[str]) -> int:
    prefix = s3_prefix(args.data_flag, args.prefix)
    print(f"RUN PLAN  {args.data_flag}  {len(models)} models  "
          f"{args.shards} shard(s)")
    print(f"  selection  {args.selection}")
    print(f"  s3 prefix  {prefix}")
    print(f"  image      {IMAGE}   disk {DISK_GB} GB")
    print(f"  deadlines  first-state {FIRST_STATE_DEADLINE // 60}m  "
          f"stale {STALE_DEADLINE // 60}m  job {JOB_DEADLINE // 3600}h")

    print("\n  shard assignment (round-robin over the name-sorted list):")
    for r in range(args.shards):
        mine = shard_models(models, args.shards, r)
        print(f"    r{r}: {len(mine):3d} models  {mine[0]} ... {mine[-1]}")

    try:
        have = s3.list_names(prefix, suffix="_features.npz")
        done = [n for n in models if f"{args.data_flag}_{n}_features.npz" in have]
        print(f"\n  already in S3: {len(done)}/{len(models)} "
              f"({len(have)} bundle(s) under the prefix in total)")
        if done:
            print(f"    the worker skips these: {done[:4]}"
                  f"{' ...' if len(done) > 4 else ''}")
    except Exception as e:
        print(f"\n  could not list {prefix}: {type(e).__name__}: {e}")

    print(f"\n  worker env (rank 0 of {args.shards}):")
    for k, v in sorted(worker_env(args.data_flag, 0, args.shards,
                                  args.prefix).items()):
        print(f"    {k}={v}")

    print(f"\n  offer query: {RUN_OFFER_QUERY}")
    try:
        offers = vast.search_offers(RUN_OFFER_QUERY, limit=args.shards + 3)
        print(f"  {len(offers)} offer(s) available, cheapest first:")
        for o in offers[:args.shards + 3]:
            print(f"    id={o.get('id')} ${float(o.get('dph_total') or 0):.3f}/hr "
                  f"{o.get('gpu_name')} cc={o.get('compute_cap')} "
                  f"down={o.get('inet_down')}Mbps {o.get('geolocation')}")
        if len(offers) >= args.shards:
            rate = sum(float(o.get("dph_total") or 0) for o in offers[:args.shards])
            print(f"  {args.shards} boxes would burn ${rate:.3f}/hr together")
    except Exception as e:
        print(f"  could not search offers: {type(e).__name__}: {e}")

    print("\n  then, per box:")
    for line in remote_script(args.data_flag, 0, args.shards, args.prefix):
        print(f"    {line[:200]}")
    print("\n  rents nothing. Re-run with --go to execute.")
    return 0


def abort(label_base: str = "run-") -> int:
    """Destroy anything this driver rented. The first post-mortem command."""
    rec = vast.reconcile()
    print(rec.render())
    mine = [i for i in vast.show_instances()
            if str(i.get("label") or "").startswith(label_base)]
    if not mine:
        print(f"nothing labelled {label_base}* is running")
        return 0
    print(f"destroying {len(mine)}: {[i.get('id') for i in mine]}")
    for i in mine:
        vast.destroy(int(i["id"]))
    remaining = vast.show_instances()
    print(f"after: ${vast.burn_rate(remaining):.3f}/hr")
    print(vast.describe(remaining))
    return 0 if not remaining else 1


def go(args, models: List[str], rid: str, scratch: Path) -> int:
    prefix = s3_prefix(args.data_flag, args.prefix)
    label_base = f"run-{rid}"
    assignments = {r: shard_models(models, args.shards, r)
                   for r in range(args.shards)}
    print(f"=== RUN {rid}  {args.data_flag}  {len(models)} models "
          f"across {args.shards} shard(s) -> {prefix}\n")

    # -- Phase 1: everything that can fail for free ------------------------
    print("[1/4] pre-rental checks")
    rec = vast.reconcile()
    print(rec.render())
    if rec.orphans:
        print("ORPHANS are billing. Resolve them before renting more.")
        return 1

    ok, msg = s3.bucket_listable(f"s3://{BUCKET}")
    print(f"  bucket listable: {ok} {msg}")
    if not ok:
        return 1

    creds = aws_credentials()
    # Staged over stdin into a umask-077 file with the AWS keys, never in argv
    # and never in worker_env -- ``create`` refuses token-shaped keys in -e,
    # and ps is world-readable on a shared vast host. SOURCE_SECRETS exports it
    # in every remote shell, which is where ``read_hf_token`` finds it.
    if args.hf_token_file:
        tok = Path(args.hf_token_file).read_text().strip()
        if not tok:
            print(f"  {args.hf_token_file} is empty")
            return 1
        creds["HF_TOKEN"] = tok
    print(f"  staging: {sorted(creds)}")
    if not args.hf_token_file:
        print("  no HF token: weights revisions come from the local HF cache "
              "(no API call), but gated repos will fail per-model at download")

    if not vast.DEFAULT_SSH_KEY.exists():
        print(f"  no local private key at {vast.DEFAULT_SSH_KEY}")
        return 1
    pubkey = vast.public_key()
    print(f"  ssh key {vast.DEFAULT_SSH_KEY}.pub ({pubkey.split()[0]})")

    tarball = code_tarball(scratch / "kprelogits.tgz")
    selection = Path(args.selection)
    print(f"  code {tarball.stat().st_size // 1024} KB, "
          f"selection {selection} ({len(models)} models)")

    already = len(models) - len(missing_bundles(prefix, args.data_flag, models))
    print(f"  already in S3: {already}/{len(models)}")
    if already == len(models):
        print("  everything is already extracted; nothing to rent")
        return 0

    # The real preflight, locally, against the exact prefix the boxes will use.
    # It is the same code either way, so anything it rejects it would reject on
    # rented hardware -- after the boot, the pip install and the data pull have
    # all been paid for.
    local = scratch / f"preflight-{rid}"
    (local / "data").mkdir(parents=True, exist_ok=True)
    (local / "results").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **worker_env(args.data_flag, 0, args.shards, args.prefix),
           "DATA_DIR": str(local / "data"), "RESULTS_DIR": str(local / "results")}
    r = subprocess.run(
        [sys.executable, "-m", "kprelogits.extract",
         "--selection", str(selection), "--dry-run"],
        env=env, capture_output=True, text=True, timeout=300,
        cwd=str(Path(__file__).resolve().parents[2]))
    if r.returncode != 0:
        print("  local preflight FAILED -- not renting:")
        print((r.stdout or "")[-1200:])
        print((r.stderr or "")[-1200:])
        return 1
    print("  local preflight passed")

    # One extra offer per shard, plus a couple: an offer can vanish between the
    # search and the rental, and a box can boot and never answer ssh.
    offers = vast.search_offers(RUN_OFFER_QUERY, limit=args.shards * 2 + 4)
    if len(offers) < args.shards:
        print(f"  only {len(offers)} offer(s) for {args.shards} shards; "
              f"nothing rented")
        return 1
    print(f"  {len(offers)} offer(s) held for {args.shards} shard(s)")

    # -- Phase 2/3: rent, stage and launch, one box at a time --------------
    known_hosts = scratch / f"known_hosts-{rid}"
    known_hosts.write_text("")

    live: List[int] = []
    verdicts: Dict[int, str] = {}
    watch = Watch(prefix=prefix, data_flag=args.data_flag,
                  assignments=assignments, live=live, scratch=scratch)

    # Entered BEFORE anything is rented, holding a list that is mutated as
    # boxes come and go. Nothing below can leave an instance outside it.
    with vast.teardown(live):
        print(f"\n[2/4] renting and launching {args.shards} box(es)")
        skipped: Dict[int, str] = {}
        tried: set = set()
        search = lambda: vast.search_offers(RUN_OFFER_QUERY,  # noqa: E731
                                            limit=args.shards * 2 + 4)
        stage = lambda b: stage_and_launch(  # noqa: E731
            b, tarball=tarball, selection=selection, creds=creds,
            data_flag=args.data_flag, shards=args.shards, prefix=args.prefix,
            known_hosts=known_hosts)
        for rank in range(args.shards):
            box, offers = bring_up(
                rank, offers, tried, stage=stage, live=live, search=search,
                pubkey=pubkey, label_base=label_base, known_hosts=known_hosts,
                env=worker_env(args.data_flag, rank, args.shards, args.prefix))
            if box is None:
                skipped[rank] = "no box that booted, staged and launched"
                continue
            # Only now: a box is watched from the moment its worker was asked
            # to start, so a box that failed staging is never watched at all.
            watch.add(box)
            print(f"  [r{rank}] {box} -- {len(assignments[rank])} models",
                  flush=True)
            # Release anything that finished while this box was provisioning,
            # and condemn anything that never started. Deadlines are safe here:
            # ranks not yet rented are not in ``pending`` at all, and every
            # deadline counts from that box's own launch. Leaving them off
            # meant a worker that never started on rank 0 went unnoticed until
            # rank 7 launched -- an hour of billing a dead box, at eight boxes.
            watch.sweep(deadlines=True, quiet=True)

        print(f"\n[3/4] watching {len(watch.pending)} shard(s); "
              f"each box is destroyed as its shard lands")
        verdicts = watch.poll()
        verdicts.update({r: f"skipped: {why}" for r, why in skipped.items()})

    # -- Phase 4: report, boxes already gone -------------------------------
    print("\n[4/4] result")
    for rank in sorted(verdicts):
        mark = "OK  " if verdicts[rank] == "done" else "FAIL"
        print(f"  {mark} r{rank} ({len(assignments[rank])} models): "
              f"{verdicts[rank]}")

    check = vast.preteardown_check(prefix, expected_shards=args.shards,
                                   expected_bundles=len(models))
    print(check.render())

    missing = missing_bundles(prefix, args.data_flag, models)
    good = not missing and all(v == "done" for v in verdicts.values())
    print(f"\n  {len(models) - len(missing)}/{len(models)} bundles in {prefix}")
    if missing:
        print(f"  missing ({len(missing)}): {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
        print(f"  re-running this command extracts exactly those: the worker's "
              f"resume oracle is S3, so what landed is not recomputed.")
    print(f"\n{'RUN COMPLETE' if good else 'RUN INCOMPLETE'}")
    return 0 if good else 1


def install_teardown_signals() -> None:
    """Make SIGTERM and SIGHUP run the teardown instead of skipping it.

    Python's default for both is to exit on the spot, and exiting on the spot
    skips every ``finally`` -- including the one that destroys the boxes. Both
    arrive in ordinary circumstances: a closed terminal sends SIGHUP, and a
    quitting parent app sends SIGTERM to its children. Raising SystemExit runs
    the finally blocks instead: ``acquire`` destroys a half-provisioned box and
    re-raises, and the teardown context destroys everything else. (SIGKILL
    cannot be caught; ``--abort`` is the answer to that one.)
    """
    import signal

    def _exit(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--selection", type=Path,
                   help="selection JSON ({'models': [{'name': ...}]})")
    p.add_argument("--data-flag", default="dermamnist")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--prefix", default=PREFIX,
                   help=f"S3_FEATURES_PREFIX (default {PREFIX}); the worker "
                        f"appends the data flag")
    p.add_argument("--go", action="store_true",
                   help="actually rent hardware (default prints a plan)")
    p.add_argument("--abort", action="store_true",
                   help="destroy anything labelled run-*, then exit")
    p.add_argument("--hf-token-file", type=Path,
                   help="path to a file holding an HF token (a PATH, never the "
                        "token itself -- it would land in ps on a shared host). "
                        "Staged with the AWS keys; raises the anonymous rate "
                        "limit and unlocks gated repos.")
    p.add_argument("--scratch", type=Path, default=Path("out/run"))
    args = p.parse_args(argv)

    if args.abort:
        return abort()
    if not args.selection:
        p.error("--selection is required")
    if args.shards < 1:
        p.error("--shards must be >= 1")

    models = load_models(args.selection)
    if not args.go:
        return plan(args, models)

    rid = run_id()
    args.scratch.mkdir(parents=True, exist_ok=True)
    install_teardown_signals()
    try:
        return go(args, models, rid, args.scratch)
    except KeyboardInterrupt:
        print("\ninterrupted -- teardown ran for every box that was rented; "
              "confirm with: python -m kprelogits.ops.run --abort")
        return 130


if __name__ == "__main__":
    sys.exit(main())
