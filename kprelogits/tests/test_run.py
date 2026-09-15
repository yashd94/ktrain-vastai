"""The multi-shard driver, tested without renting anything.

What is worth testing here is not the happy path -- that needs four GPUs --
but the things that cost money when they are wrong: a shard partition that
drops or duplicates a model, a polled prefix that nothing writes to, a remote
command that reaches S3 without credentials, and above all an instance id that
exists before it is in the teardown list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.ops import run  # noqa: E402

MODELS = [f"m{i:02d}" for i in range(50)]


# ---- the shard partition --------------------------------------------------

@pytest.mark.parametrize("shards", [1, 2, 3, 4, 7])
def test_shards_partition_exactly(shards):
    """Every model on exactly one box. A dropped model is a hole nobody sees
    until a fit fails; a duplicated one is two GPUs doing the same work."""
    parts = [run.shard_models(MODELS, shards, r) for r in range(shards)]
    flat = [n for p in parts for n in p]
    assert sorted(flat) == sorted(MODELS)
    assert len(flat) == len(set(flat))


def test_shards_are_balanced():
    """Round-robin, not contiguous blocks: the list is name-sorted and names
    correlate with architecture family and size, so blocks would put the heavy
    models on one box and leave the others idle."""
    sizes = [len(run.shard_models(MODELS, 4, r)) for r in range(4)]
    assert max(sizes) - min(sizes) <= 1


def test_shard_assignment_matches_the_worker():
    """The driver duplicates the worker's split to decide early release. If
    the two ever disagree, a box is destroyed while it still owns work."""
    from kprelogits.models import select_model_shard
    for r in range(4):
        assert run.shard_models(MODELS, 4, r) == select_model_shard(MODELS, 4, r)


# ---- the prefix everything agrees on --------------------------------------

def test_driver_prefix_matches_the_worker_config(monkeypatch):
    """The driver polls a prefix; the worker writes one. They are computed by
    different code, and if they disagree the driver watches empty ground and
    reports a job that never started -- while the job runs fine, unwatched."""
    from kprelogits.config import ExtractConfig

    env = run.worker_env("dermamnist", 0, 4)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = ExtractConfig.from_env()
    assert cfg.s3_prefix == run.s3_prefix("dermamnist")
    assert cfg.rank == 0 and cfg.shards == 4
    assert cfg.data_flag == "dermamnist"


def test_prefix_separates_datasets():
    assert run.s3_prefix("octmnist") != run.s3_prefix("dermamnist")
    assert run.s3_prefix("dermamnist").endswith("/dermamnist")


def test_worker_env_carries_rank_and_no_secrets():
    for rank in range(4):
        env = run.worker_env("dermamnist", rank, 4)
        assert env["PARALLEL_RANK"] == str(rank)
        assert env["PARALLEL_SHARDS"] == "4"
        assert env["CLEANUP"] == "1"      # S3 is the resume oracle
    joined = " ".join(f"{k}={v}" for k, v in env.items()).upper()
    for shape in ("KEY", "SECRET", "TOKEN", "PASSWORD"):
        assert shape not in joined


# ---- the remote commands --------------------------------------------------

def test_every_s3_command_sources_the_staged_secrets():
    """Each ssh call is a fresh shell. An unsourced `aws` fails with "Unable to
    locate credentials" -- trivial to fix, and discovered on a billing box."""
    cmds = run.remote_script("dermamnist", 2, 4, run.PREFIX)
    for cmd in cmds:
        if "aws s3" in cmd or "kprelogits.extract" in cmd:
            assert ". /workspace/.env-secrets" in cmd, cmd


def test_every_remote_command_fixes_path():
    """Non-interactive ssh shells do not source .bashrc and never see conda."""
    for cmd in run.remote_script("dermamnist", 0, 4, run.PREFIX):
        assert "export PATH=/opt/conda/bin:$PATH" in cmd


def test_launch_is_detached_and_carries_the_rank():
    cmds = run.remote_script("dermamnist", 3, 4, run.PREFIX)
    launch = cmds[-1]
    assert "setsid nohup" in launch
    assert "> /workspace/extract.log 2>&1 < /dev/null &" in launch
    assert "PARALLEL_RANK=3" in launch and "PARALLEL_SHARDS=4" in launch
    assert "--dry-run" not in launch


def test_dry_run_precedes_the_launch():
    cmds = run.remote_script("dermamnist", 0, 4, run.PREFIX)
    assert "--dry-run" in cmds[2] and "--dry-run" not in cmds[3]
    assert cmds.index(cmds[2]) < cmds.index(cmds[3])


def test_data_pull_names_the_right_dataset():
    cmd = run.remote_script("dermamnist", 0, 4, run.PREFIX)[1]
    assert "medmnist/dermamnist_224.npz" in cmd
    assert "octmnist" not in cmd


# ---- staging matches what the remote commands ask for ---------------------

class _CP:
    returncode = 0
    stdout = ""
    stderr = ""


def _fake_stage(monkeypatch, tmp_path, sel_name="dermamnist_pilot50.json"):
    """Run stage_and_launch against mocks; return (scp cmds, ssh cmds)."""
    sent, ran = [], []
    monkeypatch.setattr(run, "_sh", lambda cmd, **k: (sent.append(cmd), _CP())[1])
    monkeypatch.setattr(run, "_ssh", lambda t, c, **k: (ran.append(c), "")[1])
    monkeypatch.setattr(run.vast, "scp_args", lambda *a, **k: [])
    monkeypatch.setattr(run.vast, "stage_credentials", lambda *a, **k: None)

    tar = tmp_path / "kprelogits.tgz"
    tar.write_text("x")
    sel = tmp_path / sel_name
    sel.write_text('{"models": []}')
    run.stage_and_launch(run.Box(0, 1, ["-p", "22", "root@h"]), tarball=tar,
                         selection=sel, creds={}, data_flag="dermamnist",
                         shards=4, prefix=run.PREFIX, known_hosts=None)
    return sent, ran


def test_the_selection_is_staged_where_the_worker_is_told_to_look(monkeypatch,
                                                                  tmp_path):
    """The bug that killed the first attempt. The local selection is named
    after its study; the remote command asks for a fixed path. Copying into a
    directory keeps the local basename, so the worker preflighted against a
    file that was never there -- and it cost a rental to find out."""
    sent, ran = _fake_stage(monkeypatch, tmp_path)
    destinations = [c[-1].split(":", 1)[1] for c in sent if c[0] == "scp"]
    assert run.REMOTE_SELECTION in destinations
    assert not any(d.endswith("/") for d in destinations), \
        "scp to a directory keeps the local basename; name the destination file"


def test_every_remote_path_referenced_was_actually_staged(monkeypatch, tmp_path):
    """Generalises the above: no remote command may name a /workspace file
    that nothing put there."""
    sent, ran = _fake_stage(monkeypatch, tmp_path)
    staged = {c[-1].split(":", 1)[1] for c in sent if c[0] == "scp"}
    for cmd in ran:
        for token in cmd.split():
            if token.startswith("/workspace/") and token.endswith(".json"):
                assert token in staged, f"{token} is referenced but never staged"


def test_staging_survives_a_selection_named_anything(monkeypatch, tmp_path):
    for name in ("selection.json", "pilot50.json", "a.b.c.json"):
        sent, ran = _fake_stage(monkeypatch, tmp_path, sel_name=name)
        dests = [c[-1].split(":", 1)[1] for c in sent if c[0] == "scp"]
        assert run.REMOTE_SELECTION in dests
        launch = [c for c in ran if "setsid" in c][0]
        assert f"--selection {run.REMOTE_SELECTION}" in launch


# ---- completion evidence --------------------------------------------------

def test_missing_bundles_reads_s3_names(monkeypatch):
    monkeypatch.setattr(run.s3, "list_names",
                        lambda *a, **k: {"dermamnist_m01_features.npz"})
    assert run.missing_bundles("s3://b/p", "dermamnist", ["m01", "m02"]) == ["m02"]
    assert run.missing_bundles("s3://b/p", "dermamnist", ["m01"]) == []


def test_missing_bundles_is_dataset_scoped(monkeypatch):
    """An octmnist bundle of the same backbone must not count as dermamnist
    progress -- the filenames differ only by the dataset prefix."""
    monkeypatch.setattr(run.s3, "list_names",
                        lambda *a, **k: {"octmnist_m01_features.npz"})
    assert run.missing_bundles("s3://b/p", "dermamnist", ["m01"]) == ["m01"]


# ---- the orphan window ----------------------------------------------------

class _Box:
    def __init__(self, rank=0, instance_id=1):
        self.rank, self.instance_id = rank, instance_id
        self.released = False


def test_release_keeps_an_unconfirmed_destroy_in_the_teardown_list(monkeypatch):
    """destroy() returns False when the API still lists the instance. Dropping
    it from the list on that answer is how a box goes on billing unwatched."""
    monkeypatch.setattr(run.vast, "destroy", lambda *a, **k: False)
    live = [1]
    run.release(_Box(instance_id=1), live)
    assert live == [1]


def test_release_drops_a_confirmed_destroy(monkeypatch):
    monkeypatch.setattr(run.vast, "destroy", lambda *a, **k: True)
    live = [1, 2]
    run.release(_Box(instance_id=1), live)
    assert live == [2]


def test_a_destroy_that_raises_leaves_the_id_for_teardown(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("api down")
    monkeypatch.setattr(run.vast, "destroy", boom)
    live = [1]
    run.release(_Box(instance_id=1), live)
    assert live == [1]


def test_acquire_registers_the_instance_before_it_can_fail(monkeypatch):
    """The whole point of the teardown list. If wait_for_running throws --
    ten minutes of provisioning, the likeliest place to lose patience -- the
    id must already be in the list, and the box destroyed on the way out."""
    seen = {}
    monkeypatch.setattr(run.vast, "create", lambda *a, **k: 4242)
    monkeypatch.setattr(run, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(RuntimeError("never booted")))
    monkeypatch.setattr(run.vast, "destroy",
                        lambda i, *a, **k: seen.setdefault("destroyed", i) or True)

    live: list = []
    with pytest.raises(RuntimeError, match="no offer produced a usable box"):
        run.acquire(0, [{"id": 1}], "ssh-ed25519 AAAA", "run-x", live, None, {})
    assert seen["destroyed"] == 4242
    assert live == []          # destroyed AND confirmed, so correctly dropped


def test_acquire_keeps_the_id_when_the_destroy_fails(monkeypatch):
    monkeypatch.setattr(run.vast, "create", lambda *a, **k: 99)
    monkeypatch.setattr(run, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(RuntimeError("no boot")))
    monkeypatch.setattr(run.vast, "destroy", lambda *a, **k: False)

    live: list = []
    with pytest.raises(RuntimeError):
        run.acquire(0, [{"id": 1}], "k", "run-x", live, None, {})
    assert live == [99], "an unconfirmed destroy must stay in the teardown list"


def test_acquire_adopts_an_instance_a_failed_create_left_behind(monkeypatch):
    """A create that errors locally does not prove nothing was rented."""
    def failing_create(*a, **k):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(run.vast, "create", failing_create)
    monkeypatch.setattr(run.vast, "find_by_label", lambda label: [{"id": 777}])
    monkeypatch.setattr(run, "wait_for_running", lambda i: {"gpu_name": "x"})
    monkeypatch.setattr(run.vast, "attach_ssh_key", lambda *a, **k: None)
    monkeypatch.setattr(run.vast, "ssh_endpoints", lambda i: [("proxy", "h", 22)])
    monkeypatch.setattr(run, "wait_for_ssh",
                        lambda *a, **k: ["-p", "22", "root@h"])

    live: list = []
    box, rest = run.acquire(0, [{"id": 1}], "k", "run-x", live, None, {})
    assert box.instance_id == 777 and live == [777]


def test_acquire_does_not_reuse_a_consumed_offer(monkeypatch):
    """Two ranks renting the same offer id is a create failure at best."""
    monkeypatch.setattr(run.vast, "create", lambda oid, **k: 100 + oid)
    monkeypatch.setattr(run, "wait_for_running", lambda i: {"gpu_name": "x"})
    monkeypatch.setattr(run.vast, "attach_ssh_key", lambda *a, **k: None)
    monkeypatch.setattr(run.vast, "ssh_endpoints", lambda i: [("proxy", "h", 22)])
    monkeypatch.setattr(run, "wait_for_ssh",
                        lambda *a, **k: ["-p", "22", "root@h"])

    live: list = []
    offers = [{"id": 1}, {"id": 2}]
    box0, rest = run.acquire(0, offers, "k", "run-x", live, None, {})
    box1, rest = run.acquire(1, rest, "k", "run-x", live, None, {})
    assert box0.instance_id != box1.instance_id
    assert rest == [] and sorted(live) == [101, 102]


def test_ctrl_c_during_acquisition_destroys_and_reraises(monkeypatch):
    """A KeyboardInterrupt must not walk past a live instance."""
    seen = {}
    monkeypatch.setattr(run.vast, "create", lambda *a, **k: 5)
    monkeypatch.setattr(run, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(run.vast, "destroy",
                        lambda i, *a, **k: seen.setdefault("d", i) or True)
    with pytest.raises(KeyboardInterrupt):
        run.acquire(0, [{"id": 1}], "k", "run-x", [], None, {})
    assert seen["d"] == 5


# ---- releasing during acquisition -----------------------------------------

def _iso(offset_s=0):
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


class _St:
    """Just the ShardState fields sweep() reads."""

    def __init__(self, phase, extracted=0, uploaded=0, failed=0, complete=True,
                 started_at=None, shards=None):
        self.phase, self.extracted = phase, extracted
        self.uploaded, self.failed = uploaded, failed
        self.updated_at, self.current, self.errors = "t", None, []
        self.is_complete = complete
        self.started_at = started_at or _iso()
        self.shards = shards


def _watch(monkeypatch, tmp_path, states, *, have=(), assignments=None):
    monkeypatch.setattr(run.state, "read_states", lambda p: states)
    monkeypatch.setattr(run.state, "summarize", lambda s: "")
    monkeypatch.setattr(run.s3, "list_names", lambda *a, **k: set(have))
    monkeypatch.setattr(run.vast, "destroy", lambda *a, **k: True)
    monkeypatch.setattr(run, "tail_remote_log", lambda *a, **k: None)
    monkeypatch.setattr(run, "rescue", lambda *a, **k: None)
    w = run.Watch(prefix="s3://b/p", data_flag="dermamnist",
                  assignments=assignments or {0: ["m00"]}, live=[],
                  scratch=tmp_path)
    return w


def test_sweep_releases_a_shard_whose_bundles_are_in_s3(monkeypatch, tmp_path):
    w = _watch(monkeypatch, tmp_path, {0: _St("done", 1, 1)},
               have={"dermamnist_m00_features.npz"})
    w.live.append(1)
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=False, quiet=True)
    assert w.verdicts == {0: "done"} and w.pending == set() and w.live == []


def test_sweep_refuses_to_release_on_a_done_claim_with_no_bundle(monkeypatch,
                                                                tmp_path):
    """The check that makes early release safe: S3, not the shard's word."""
    w = _watch(monkeypatch, tmp_path, {0: _St("done", 1, 1)}, have=set())
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=False, quiet=True)
    assert "missing" in w.verdicts[0]


def test_the_deadline_flag_is_what_condemns_a_silent_shard(monkeypatch, tmp_path):
    """Without deadlines, a silent shard is simply not ready; with them, past
    FIRST_STATE_DEADLINE, it is a worker that never started. Acquisition sweeps
    WITH deadlines -- safe because they count from each box's own launch (see
    the next test) -- so a dead rank 0 is caught while rank 7 is still being
    rented, not after."""
    w = _watch(monkeypatch, tmp_path, {})
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.launched_at[0] = 0.0                      # long past FIRST_STATE_DEADLINE
    w.sweep(deadlines=False, quiet=True)
    assert w.pending == {0} and w.verdicts == {}

    w.sweep(deadlines=True, quiet=True)
    assert w.verdicts[0].startswith("no state file")


def test_deadlines_are_measured_from_launch_not_from_run_start(monkeypatch,
                                                               tmp_path):
    """Rank 7 launches an hour after rank 0. A deadline counted from the run's
    start would condemn it on its first sweep."""
    w = _watch(monkeypatch, tmp_path, {})
    w.add(run.Box(7, 8, ["-p", "22", "root@h"]))
    w.sweep(deadlines=True, quiet=True)
    assert w.pending == {7} and w.verdicts == {}


def test_an_earlier_runs_done_state_does_not_condemn_a_new_box(monkeypatch,
                                                              tmp_path):
    """State files are named by rank alone. A prefix that has seen a run holds
    a stale 'done' state_shard0.json until the new worker overwrites it, and
    the sweep right after launch can read it first: 'done but 109 missing',
    rescue, destroy -- on a box that is working fine."""
    stale = _St("done", 13, 13, started_at=_iso(-3600))
    w = _watch(monkeypatch, tmp_path, {0: stale}, have=set())
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=True, quiet=True)
    assert w.pending == {0} and w.verdicts == {}


def test_a_state_started_after_launch_is_this_boxs(monkeypatch, tmp_path):
    fresh = _St("done", 1, 1, started_at=_iso(+30))
    w = _watch(monkeypatch, tmp_path, {0: fresh},
               have={"dermamnist_m00_features.npz"})
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=True, quiet=True)
    assert w.verdicts == {0: "done"}


def test_a_box_clock_a_minute_behind_is_still_trusted(monkeypatch, tmp_path):
    skewed = _St("done", 1, 1, started_at=_iso(-60))
    w = _watch(monkeypatch, tmp_path, {0: skewed},
               have={"dermamnist_m00_features.npz"})
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=True, quiet=True)
    assert w.verdicts == {0: "done"}


def test_a_state_from_another_shard_count_is_foreign(monkeypatch, tmp_path):
    """A 4-shard run's rank 0 and an 8-shard run's rank 0 share a filename but
    own different models."""
    other = _St("done", 13, 13, shards=4)
    w = _watch(monkeypatch, tmp_path, {0: other},
               have={"dermamnist_m00_features.npz"},
               assignments={r: [f"m{r:02d}"] for r in range(8)})
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=True, quiet=True)
    assert w.pending == {0} and w.verdicts == {}


def test_an_unattributable_start_time_is_not_acted_on(monkeypatch, tmp_path):
    odd = _St("done", 13, 13, started_at="yesterday-ish")
    w = _watch(monkeypatch, tmp_path, {0: odd}, have=set())
    w.add(run.Box(0, 1, ["-p", "22", "root@h"]))
    w.sweep(deadlines=False, quiet=True)
    assert w.pending == {0}


def test_sweep_is_a_noop_with_nothing_pending(monkeypatch, tmp_path):
    calls = []
    w = _watch(monkeypatch, tmp_path, {})
    monkeypatch.setattr(run.state, "read_states", lambda p: calls.append(p) or {})
    w.sweep(deadlines=True, quiet=True)
    assert calls == []


# ---- the box target parsing ----------------------------------------------

def test_box_parses_the_ssh_target():
    box = run.Box(2, 55, ["-i", "/k", "-p", "40123", "root@1.2.3.4"])
    assert (box.host, box.port, box.user) == ("1.2.3.4", 40123, "root")
    assert box.rank == 2 and box.instance_id == 55


# ---- dying without skipping the teardown ----------------------------------

def test_hangup_and_term_raise_systemexit_so_finally_blocks_run():
    """Default SIGTERM/SIGHUP handling exits without running `finally`, which
    is where the boxes get destroyed. An app quit sends exactly these."""
    import signal
    sigs = (signal.SIGTERM, signal.SIGHUP)
    old = {s: signal.getsignal(s) for s in sigs}
    try:
        run.install_teardown_signals()
        for s in sigs:
            with pytest.raises(SystemExit) as e:
                signal.getsignal(s)(s, None)
            assert e.value.code == 128 + s
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def test_a_signal_mid_block_still_reaches_the_finally():
    """What the handler buys, end to end: the teardown line runs."""
    import os
    import signal
    old = signal.getsignal(signal.SIGTERM)
    ran = []
    try:
        run.install_teardown_signals()
        with pytest.raises(SystemExit):
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                signal.pause() if hasattr(signal, "pause") else None
            finally:
                ran.append("teardown")
    finally:
        signal.signal(signal.SIGTERM, old)
    assert ran == ["teardown"]


# ---- the timm pin ---------------------------------------------------------

def test_timm_is_pinned_identically_on_every_install_path():
    """No bundle records the timm version, and timm decides what a model's
    pre-logits are. The rented-box path (PIP_PACKAGES) and the image path
    (Dockerfile) must pin the same release, or two sweeps can differ with
    nothing in either manifest to say so."""
    from kprelogits.ops.smoke import PIP_PACKAGES
    docker = (REPO / "Dockerfile").read_text()
    docker_pip = " ".join(l for l in docker.splitlines() if "pip install" in l)
    for where, text in (("PIP_PACKAGES", PIP_PACKAGES), ("Dockerfile", docker_pip)):
        toks = text.replace("\\", " ").split()
        assert "timm==1.0.29" in toks, f"{where} does not pin timm==1.0.29"
        assert "timm" not in toks, f"{where} also installs an unpinned timm"


def test_the_pin_reaches_the_remote_install():
    pip = run.remote_script("dermamnist", 0, 8, run.PREFIX)[0]
    assert "timm==1.0.29" in pip


# ---- a rank that gets no box is skipped, not fatal ------------------------

def _refresh_kw(search, live=None):
    return dict(search=search, pubkey="k", label_base="run-x",
                live=live if live is not None else [], known_hosts=None, env={})


def test_no_usable_box_is_still_a_runtime_error():
    """Callers that catch RuntimeError keep working; the subclass is what lets
    the driver tell 'this rank got no box' from everything else."""
    assert issubclass(run.NoUsableBox, RuntimeError)


def test_acquire_records_every_offer_it_tries(monkeypatch):
    monkeypatch.setattr(run.vast, "create", lambda *a, **k: 7)
    monkeypatch.setattr(run, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(RuntimeError("no boot")))
    monkeypatch.setattr(run.vast, "destroy", lambda *a, **k: True)
    tried: set = set()
    with pytest.raises(run.NoUsableBox):
        run.acquire(0, [{"id": 11}, {"id": 12}], "k", "run-x", [], None, {}, tried=tried)
    assert tried == {11, 12}


def test_exhausted_offers_are_refreshed_from_the_live_marketplace(monkeypatch):
    calls = []

    def fake_acquire(rank, offers, *a, tried=None, **k):
        calls.append([o["id"] for o in offers])
        if len(calls) == 1:
            tried.update(o["id"] for o in offers)
            raise run.NoUsableBox("all dead")
        return "BOX", []

    monkeypatch.setattr(run, "acquire", fake_acquire)
    tried: set = set()
    box, _ = run.acquire_with_refresh(
        6, [{"id": 1}, {"id": 2}], tried,
        **_refresh_kw(lambda: [{"id": 2}, {"id": 3}, {"id": 4}]))
    assert box == "BOX"
    assert calls == [[1, 2], [3, 4]], "the re-search must skip offers already tried"


def test_a_rank_that_never_gets_a_box_is_skipped_not_fatal(monkeypatch, capsys):
    """The 2026-09-13 failure: rank 7 exhausted every offer, the exception
    ended the run, and the teardown destroyed ranks 4 and 6 mid-shard."""
    def always_dead(*a, **k):
        raise run.NoUsableBox("all dead")

    monkeypatch.setattr(run, "acquire", always_dead)
    box, offers = run.acquire_with_refresh(7, [{"id": 1}], set(),
                                           **_refresh_kw(lambda: [{"id": 9}]))
    assert box is None and offers == []
    assert "SKIPPED" in capsys.readouterr().out


def test_an_empty_marketplace_skips_without_retrying(monkeypatch):
    calls = []

    def dead(*a, **k):
        calls.append(1)
        raise run.NoUsableBox("all dead")

    monkeypatch.setattr(run, "acquire", dead)
    box, _ = run.acquire_with_refresh(7, [{"id": 1}], set(), **_refresh_kw(lambda: []))
    assert box is None and len(calls) == 1


def test_other_errors_still_propagate(monkeypatch):
    """Only 'no usable box' is skippable. Anything else -- a KeyboardInterrupt
    above all -- must still reach the teardown."""
    def interrupted(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run, "acquire", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run.acquire_with_refresh(0, [{"id": 1}], set(), **_refresh_kw(lambda: []))


# ---- a box that fails staging is replaced, not fatal ---------------------

class _FakeBox:
    def __init__(self, n):
        self.instance_id, self.rank = n, 0

    def __repr__(self):
        return f"<box {self.instance_id}>"


def _bring_up(monkeypatch, boxes, stage):
    released = []
    it = iter(boxes)
    monkeypatch.setattr(run, "acquire_with_refresh",
                        lambda rank, offers, tried, **k: (next(it, None), offers))
    monkeypatch.setattr(run, "release", lambda box, live: released.append(box.instance_id))
    box, _ = run.bring_up(3, [], set(), stage=stage, live=[], search=lambda: [],
                          pubkey="k", label_base="run-x", known_hosts=None, env={})
    return box, released


def test_a_box_that_fails_staging_is_released_and_replaced(monkeypatch):
    """The shape of 2026-09-14: one box's remote preflight refuses (a GPU the
    image cannot run). It must cost that box, not the run."""
    staged = []

    def stage(b):
        staged.append(b.instance_id)
        if b.instance_id == 1:
            raise RuntimeError("r3 dry-run failed rc=1: GPU compute capability "
                               "12.0 has no kernels in this torch build")

    box, released = _bring_up(monkeypatch, [_FakeBox(1), _FakeBox(2)], stage)
    assert box.instance_id == 2 and released == [1] and staged == [1, 2]


def test_a_rank_whose_boxes_all_fail_staging_is_skipped(monkeypatch):
    def stage(b):
        raise RuntimeError("pip install failed")

    box, released = _bring_up(monkeypatch, [_FakeBox(i) for i in range(10)], stage)
    assert box is None and released == [0, 1, 2]      # STAGE_ATTEMPTS boxes, then skip


def test_no_box_at_all_is_skipped_without_staging(monkeypatch):
    box, released = _bring_up(monkeypatch, [], lambda b: pytest.fail("nothing to stage"))
    assert box is None and released == []


def test_an_interrupt_during_staging_still_reaches_the_teardown(monkeypatch):
    def stage(b):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _bring_up(monkeypatch, [_FakeBox(1)], stage)
