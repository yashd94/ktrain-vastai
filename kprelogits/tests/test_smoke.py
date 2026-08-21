"""The smoke driver's decisions, tested without renting anything.

The driver spends money, so the parts that decide *when to stop spending* are
the parts worth pinning: the per-attempt prefix, the abort rules, and the
bundle comparison that says whether the run proved anything. Nothing here
touches vast.ai, S3, or a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.ops import smoke, state  # noqa: E402


# ---- the per-attempt prefix ----------------------------------------------

def test_each_attempt_gets_its_own_prefix():
    """Reusing a smoke prefix makes attempt 2 a silent no-op: the worker's
    resume oracle is S3, so it would skip the model and report success."""
    assert smoke.smoke_prefix("a") != smoke.smoke_prefix("b")


def test_the_smoke_prefix_is_not_the_production_one():
    """Bundle keys carry only (dataset, model), so writing to the production
    prefix would overwrite a real bundle rather than sit beside it."""
    p = smoke.smoke_prefix("x")
    assert smoke.SMOKE_PREFIX in p
    assert f"/{smoke.PROD_PREFIX}/" not in p
    assert smoke.prod_bundle_uri() != p


def test_worker_env_carries_the_run_specific_prefix():
    env = smoke.worker_env("run7")
    assert "run-run7" in env["S3_FEATURES_PREFIX"]
    assert env["MAX_TRAIN"] == str(smoke.MAX_TRAIN)


def test_worker_env_carries_no_credentials():
    """`vastai create -e` values land in the instance's command line. Anything
    secret has to go through stage_credentials instead."""
    from kprelogits.config import check_no_secrets
    check_no_secrets(smoke.worker_env("r"))


def test_selection_names_exactly_one_model():
    doc = smoke.selection_doc()
    assert [m["name"] for m in doc["models"]] == [smoke.MODEL]


def test_remote_commands_fix_the_path_before_anything_else():
    """A non-interactive ssh shell never sources .bashrc, so conda's python
    and pip are not on PATH unless every command puts them there."""
    for line in smoke.remote_script("r").splitlines():
        assert line.startswith(smoke.REMOTE_PATH), line


def test_the_launch_command_survives_the_ssh_session_ending():
    launch = smoke.remote_script("r").splitlines()[-1]
    assert "setsid nohup" in launch and "< /dev/null" in launch


def test_credentials_are_sourced_not_passed():
    launch = smoke.remote_script("r").splitlines()[-1]
    assert ". /workspace/.env-secrets" in launch
    assert "AWS_SECRET" not in launch


# ---- abort rules ----------------------------------------------------------

class _Clock:
    """A fake clock so deadline logic is tested in microseconds, not minutes."""

    def __init__(self):
        self.t = 1_000.0

    def time(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


def _poll(monkeypatch, states_sequence):
    """Drive poll_to_completion over a scripted sequence of shard states."""
    clock = _Clock()
    monkeypatch.setattr(smoke.time, "time", clock.time)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)
    seq = list(states_sequence)

    def fake_read(prefix):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(smoke.state, "read_states", fake_read)
    return smoke.poll_to_completion("s3://bucket/prefix")


def _st(phase, **kw):
    base = dict(data_flag="octmnist", rank=0, shards=1, phase=phase,
                scheduled=1, extracted=0, uploaded=0, failed=0)
    base.update(kw)
    return {0: state.ShardState(**base)}


def test_done_is_success(monkeypatch):
    ok, why = _poll(monkeypatch, [_st("running"), _st("done")])
    assert ok and why == "done"


def test_failed_is_reported_with_the_error(monkeypatch):
    st = _st("failed", failed=1)
    st[0].errors = [{"model": "m", "stage": "extract", "error": "CUDA oom"}]
    ok, why = _poll(monkeypatch, [st])
    assert not ok and "CUDA oom" in why


def test_a_worker_that_never_starts_is_not_waited_on_forever(monkeypatch):
    """No state file means the launch itself failed -- there is nothing to
    wait for, and the box bills the whole time."""
    ok, why = _poll(monkeypatch, [{}])
    assert not ok and "never started" in why


def test_a_wedged_worker_is_cut_off(monkeypatch):
    """Phase stays running and nothing changes: the job is stuck, and the
    only signal that distinguishes stuck from slow is the update timestamp."""
    ok, why = _poll(monkeypatch, [_st("running", updated_at="fixed")])
    assert not ok and "no state change" in why


def test_progress_resets_the_staleness_clock(monkeypatch):
    """A slow job that is still reporting must not be killed for being slow."""
    moving = [_st("running", updated_at=f"t{i}") for i in range(40)]
    ok, why = _poll(monkeypatch, moving + [_st("done")])
    assert ok


# ---- bundle comparison ----------------------------------------------------

D, K = 6, 4
SIZES = {"train": 12, "val": 5, "test": 7}


def _bundle(tmp_path, name, *, features=None, labels=None, model=smoke.MODEL):
    from kprelogits.extract import bundle_payload, save_atomic
    rng = np.random.default_rng(0)
    f = {}
    for split, n in SIZES.items():
        f[f"x_{split}"] = (features[split] if features else
                           rng.normal(size=(n, D)).astype(np.float32))
        f[f"y_{split}"] = (labels[split] if labels else
                           rng.integers(0, K, n))
    p = tmp_path / name
    save_atomic(p, bundle_payload(
        f, data_flag="octmnist", model_name=model, feat_dim=D, num_classes=K,
        max_train=0, precision="fp32", weights_revision="hf:x@1"))
    return p


def test_a_bundle_validates_against_the_contract(tmp_path, capsys):
    assert smoke.compare_bundles(_bundle(tmp_path, "a.npz"), None)
    assert "contract OK" in capsys.readouterr().out


def test_identical_features_pass(tmp_path):
    rng = np.random.default_rng(1)
    feats = {s: rng.normal(size=(n, D)).astype(np.float32)
             for s, n in SIZES.items()}
    labs = {s: rng.integers(0, K, n) for s, n in SIZES.items()}
    a = _bundle(tmp_path, "a.npz", features=feats, labels=labs)
    b = _bundle(tmp_path, "b.npz", features=feats, labels=labs)
    assert smoke.compare_bundles(a, b)


def test_gpu_noise_passes(tmp_path):
    """Both extractions run fp16 autocast, and cudnn convolutions default to
    TF32 on Ampere+. Small relative differences are what CORRECT code produces
    on two different cards; failing them would fail every real run."""
    rng = np.random.default_rng(2)
    feats = {s: rng.normal(size=(n, D)).astype(np.float32)
             for s, n in SIZES.items()}
    labs = {s: rng.integers(0, K, n) for s, n in SIZES.items()}
    jittered = {s: (v * (1 + rng.normal(scale=1e-4, size=v.shape))).astype(np.float32)
                for s, v in feats.items()}
    a = _bundle(tmp_path, "a.npz", features=feats, labels=labs)
    b = _bundle(tmp_path, "b.npz", features=jittered, labels=labs)
    assert smoke.compare_bundles(a, b)


def test_different_labels_fail(tmp_path, capsys):
    """Labels are the sharpest check available: they are integers, so any
    difference means the seed-42 subsample or the data path changed."""
    rng = np.random.default_rng(3)
    feats = {s: rng.normal(size=(n, D)).astype(np.float32)
             for s, n in SIZES.items()}
    a = _bundle(tmp_path, "a.npz", features=feats,
                labels={s: np.zeros(n, dtype=np.int64) for s, n in SIZES.items()})
    b = _bundle(tmp_path, "b.npz", features=feats,
                labels={s: np.ones(n, dtype=np.int64) for s, n in SIZES.items()})
    assert not smoke.compare_bundles(a, b)
    assert "labels differ" in capsys.readouterr().out


def test_structurally_different_features_fail(tmp_path):
    """The failure this exists to catch: wrong preprocessing or wrong weights
    produce features that are not a small perturbation of the reference."""
    rng = np.random.default_rng(4)
    labs = {s: rng.integers(0, K, n) for s, n in SIZES.items()}
    a = _bundle(tmp_path, "a.npz",
                features={s: rng.normal(size=(n, D)).astype(np.float32)
                          for s, n in SIZES.items()}, labels=labs)
    b = _bundle(tmp_path, "b.npz",
                features={s: rng.normal(size=(n, D)).astype(np.float32)
                          for s, n in SIZES.items()}, labels=labs)
    assert not smoke.compare_bundles(a, b)


def test_a_shape_mismatch_is_reported_not_crashed(tmp_path, capsys):
    a = _bundle(tmp_path, "a.npz")
    rng = np.random.default_rng(5)
    wide = {s: rng.normal(size=(n, D + 2)).astype(np.float32)
            for s, n in SIZES.items()}
    b = _bundle(tmp_path, "b.npz", features=wide)
    assert not smoke.compare_bundles(a, b)
    assert "MISMATCH" in capsys.readouterr().out


def test_non_finite_features_fail(tmp_path, capsys):
    """The contract checks names, versions and shapes -- not values. A NaN
    survives upload and validation and only surfaces as a fit that will not
    converge, by which time the GPU is long gone."""
    rng = np.random.default_rng(6)
    feats = {s: rng.normal(size=(n, D)).astype(np.float32)
             for s, n in SIZES.items()}
    feats["train"][0, 0] = np.nan
    assert not smoke.compare_bundles(_bundle(tmp_path, "a.npz", features=feats), None)
    assert "NON-FINITE" in capsys.readouterr().out


def test_labels_outside_num_classes_fail(tmp_path, capsys):
    labs = {s: np.full(n, 99, dtype=np.int64) for s, n in SIZES.items()}
    assert not smoke.compare_bundles(_bundle(tmp_path, "a.npz", labels=labs), None)
    assert "OUT OF RANGE" in capsys.readouterr().out


def test_ssh_wait_reattaches_the_key_before_giving_up(monkeypatch):
    """A key attached seconds ago may not have propagated. One lost attach
    must not look the same as a box that will never answer."""
    clock = _Clock()
    monkeypatch.setattr(smoke.time, "time", clock.time)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)

    class R:
        returncode, stdout, stderr = 255, "", "auth failed"

    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: R())
    calls = []
    with pytest.raises(RuntimeError, match="never came up"):
        smoke.wait_for_ssh([("proxy", "h", 22)], on_retry=lambda: calls.append(1))
    assert calls, "the key was never re-attached"


def test_ssh_wait_returns_as_soon_as_the_box_answers(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(smoke.time, "time", clock.time)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: R())
    calls = []
    got = smoke.wait_for_ssh([("proxy", "h", 22)], on_retry=lambda: calls.append(1))
    assert got[-1] == "root@h"
    assert not calls, "re-attached despite a working connection"


def test_a_dead_route_falls_through_to_a_working_one(monkeypatch):
    """The failure that cost a rental: the direct route was firewalled and
    timed out, and nothing ever tried the proxy that was working all along."""
    clock = _Clock()
    monkeypatch.setattr(smoke.time, "time", clock.time)
    monkeypatch.setattr(smoke.time, "sleep", clock.sleep)

    def fake_run(args, **kw):
        class R:
            stdout = stderr = ""
            returncode = 0 if "ssh4.vast.ai" in args[-2] else 255
        if "ssh4.vast.ai" not in args[-2]:
            R.stderr = "Operation timed out"
        return R()

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    got = smoke.wait_for_ssh([("direct", "70.70.1.1", 41400),
                              ("proxy", "ssh4.vast.ai", 28014)])
    assert got[-1] == "root@ssh4.vast.ai"


def test_every_s3_command_sources_the_credentials():
    """Each ssh call is a fresh shell, so nothing carries over. An unsourced
    `aws` dies with 'Unable to locate credentials' -- on a rented box, after
    the slow parts have already been paid for."""
    for line in smoke.remote_script("r").splitlines():
        if "aws " in line or "kprelogits.extract" in line:
            assert smoke.SOURCE_SECRETS in line, line


def test_ssh_can_detach_its_own_stdin(monkeypatch):
    """Without -n, ssh holds the channel open for a backgrounded remote job,
    which is indistinguishable from the job hanging."""
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(smoke.subprocess, "run",
                        lambda cmd, **k: (seen.update(cmd=cmd), R())[1])
    smoke._ssh(["root@h"], "true", timeout=5, label="x", stdin_null=True)
    assert seen["cmd"][:2] == ["ssh", "-n"]

    smoke._ssh(["root@h"], "true", timeout=5, label="x")
    assert "-n" not in seen["cmd"]


# ---- acquiring a usable box ----------------------------------------------

def test_a_bad_box_is_destroyed_and_the_next_offer_tried(monkeypatch):
    """Boxes fail for reasons that are nobody's bug -- the image never pulls,
    sshd never takes the key. That should cost one short rental, not the run.
    And the dud must die BEFORE the next is rented, or the burn rate doubles."""
    events = []
    monkeypatch.setattr(smoke.vast, "create",
                        lambda oid, **k: (events.append(("create", oid)), oid)[1])
    monkeypatch.setattr(smoke.vast, "destroy",
                        lambda i, **k: events.append(("destroy", i)))
    monkeypatch.setattr(smoke.vast, "attach_ssh_key", lambda *a: None)
    monkeypatch.setattr(smoke.vast, "ssh_endpoints", lambda i, **k: [("proxy", "h", 1)])
    monkeypatch.setattr(smoke, "wait_for_running", lambda i: {"gpu_name": "x"})

    def flaky(routes, **kw):
        if events[-1][1] == 11:
            raise RuntimeError("ssh never came up")
        return ["root@h", "-p", "1", "root@h"]

    monkeypatch.setattr(smoke, "wait_for_ssh", flaky)
    got, _ = smoke.acquire_box([{"id": 11}, {"id": 22}], "ssh-rsa k", "rid")

    assert got == 22
    assert events == [("create", 11), ("destroy", 11), ("create", 22)]


def test_running_out_of_offers_is_an_error_not_a_silent_pass(monkeypatch):
    monkeypatch.setattr(smoke.vast, "create", lambda oid, **k: oid)
    monkeypatch.setattr(smoke.vast, "destroy", lambda i, **k: None)
    monkeypatch.setattr(smoke, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(RuntimeError("stuck loading")))
    with pytest.raises(RuntimeError, match="no offer produced"):
        smoke.acquire_box([{"id": 1}, {"id": 2}], "k", "rid")


def test_ctrl_c_during_provisioning_still_destroys_the_box(monkeypatch):
    """The worst orphan window: acquire_box spans up to ten minutes of
    provisioning and ssh retries, OUTSIDE the teardown context, which is the
    likeliest moment for someone to lose patience. Catching only Exception
    would let KeyboardInterrupt walk past a live, billing instance."""
    killed = []
    monkeypatch.setattr(smoke.vast, "create", lambda oid, **k: 99)
    monkeypatch.setattr(smoke.vast, "destroy", lambda i, **k: killed.append(i))
    monkeypatch.setattr(smoke, "wait_for_running",
                        lambda i: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(KeyboardInterrupt):
        smoke.acquire_box([{"id": 1}], "k", "rid")
    assert killed == [99], "interrupt left the instance billing"


def test_an_offer_that_vanished_moves_on_to_the_next(monkeypatch):
    """Offers are live marketplace inventory and one can be taken between the
    search and the rental. That should cost the next offer, not the run."""
    made = []

    def create(oid, **k):
        if oid == 1:
            raise smoke.vast.VastError("create failed rc=1: no such offer")
        made.append(oid)
        return oid

    monkeypatch.setattr(smoke.vast, "create", create)
    monkeypatch.setattr(smoke.vast, "find_by_label", lambda label: [])
    monkeypatch.setattr(smoke.vast, "destroy", lambda i, **k: None)
    monkeypatch.setattr(smoke.vast, "attach_ssh_key", lambda *a: None)
    monkeypatch.setattr(smoke.vast, "ssh_endpoints", lambda i, **k: [("proxy", "h", 1)])
    monkeypatch.setattr(smoke, "wait_for_running", lambda i: {"gpu_name": "x"})
    monkeypatch.setattr(smoke, "wait_for_ssh",
                        lambda routes, **k: ["root@h", "-p", "1", "root@h"])

    got, _ = smoke.acquire_box([{"id": 1}, {"id": 2}], "k", "rid")
    assert got == 2 and made == [2]


def test_an_ambiguous_create_adopts_the_box_it_may_have_made(monkeypatch, capsys):
    """'The request failed locally' does not prove 'nothing was rented'. The
    label is the only handle left on a box whose id we never saw."""
    monkeypatch.setattr(smoke.vast, "create", lambda oid, **k: (_ for _ in ()).throw(
        smoke.vast.VastError("timed out")))
    monkeypatch.setattr(smoke.vast, "find_by_label",
                        lambda label: [{"id": 4242, "label": label}])
    monkeypatch.setattr(smoke.vast, "attach_ssh_key", lambda *a: None)
    monkeypatch.setattr(smoke.vast, "ssh_endpoints", lambda i, **k: [("proxy", "h", 1)])
    monkeypatch.setattr(smoke, "wait_for_running", lambda i: {"gpu_name": "x"})
    monkeypatch.setattr(smoke, "wait_for_ssh",
                        lambda routes, **k: ["root@h", "-p", "1", "root@h"])

    got, _ = smoke.acquire_box([{"id": 1}], "k", "rid")
    assert got == 4242
    assert "adopting it" in capsys.readouterr().out
