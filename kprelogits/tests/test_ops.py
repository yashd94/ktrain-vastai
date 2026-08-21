"""Ops invariants: state files, instance reconciliation, credential staging.

These are the mechanisms that replaced things which had already gone wrong --
polling `pgrep` over SSH, trusting the registry file, putting AWS keys in a
remote command line. Each test pins the property the replacement exists for.
No network, no vastai CLI, no S3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.ops import state, vast  # noqa: E402
from kprelogits.select import build_selection, from_cache  # noqa: E402


# ---------------------------------------------------------------------------
# Shard state
# ---------------------------------------------------------------------------

@pytest.fixture
def sf(tmp_path):
    return state.StateFile.create(data_flag="octmnist", rank=0, shards=1,
                                  results_dir=tmp_path, scheduled=3)


def _read(sf):
    return json.loads(sf.path.read_text())


def test_state_starts_pending_and_is_readable(sf):
    sf.write()
    assert _read(sf)["phase"] == "pending"
    assert _read(sf)["scheduled"] == 3


def test_every_transition_writes_through(sf):
    sf.advance("running")
    assert _read(sf)["phase"] == "running"
    sf.begin_model("convnext_small")
    assert _read(sf)["current"] == "convnext_small"
    sf.model_extracted()
    sf.model_uploaded()
    assert (_read(sf)["extracted"], _read(sf)["uploaded"]) == (1, 1)


def test_unknown_phase_rejected(sf):
    with pytest.raises(ValueError, match="unknown phase"):
        sf.advance("finished")


def test_no_tmp_file_survives_a_write(sf):
    """A reader must never catch a half-written document; the tmp+rename is
    what guarantees that."""
    sf.advance("running")
    assert not list(sf.path.parent.glob("*.tmp"))


def test_failures_are_recorded_with_their_stage(sf):
    sf.model_failed("bad_net", "build_encoder", "RuntimeError: no such model")
    doc = _read(sf)
    assert doc["failed"] == 1
    assert doc["errors"][0]["stage"] == "build_encoder"


def test_complete_requires_everything_accounted_for(sf):
    """`done` is not enough. A shard whose process exited and a shard whose
    features reached S3 are different facts, and only the second is safe."""
    sf.state.extracted, sf.state.uploaded = 3, 3
    sf.advance("done")
    assert sf.state.is_complete

    sf.state.failed = 1
    assert not sf.state.is_complete


def test_uploaded_short_of_extracted_is_incomplete(sf):
    sf.state.extracted, sf.state.uploaded = 3, 2
    sf.advance("done")
    assert not sf.state.is_complete


def test_skipped_models_count_toward_completion(sf):
    """A resumed run extracts nothing and is still complete."""
    sf.state.skipped = 3
    sf.advance("done")
    assert sf.state.is_complete


def test_shard_failure_is_terminal_and_explains_itself(sf):
    sf.fail("RuntimeError: CUDA driver crashed")
    doc = _read(sf)
    assert doc["phase"] == "failed"
    assert "CUDA" in doc["errors"][-1]["error"]


def test_read_states_round_trips(monkeypatch, tmp_path):
    """The driver reads S3, never the box -- so the published document must
    reconstitute into the same dataclass."""
    sf = state.StateFile.create(data_flag="octmnist", rank=1, shards=2,
                                results_dir=tmp_path, scheduled=5)
    sf.state.extracted = 5
    sf.advance("done")
    published = json.loads(sf.path.read_text())

    from kprelogits.ops import s3
    monkeypatch.setattr(s3, "list_names", lambda *a, **k: {"state_shard1.json"})
    monkeypatch.setattr(s3, "get_json", lambda uri: published)

    got = state.read_states("s3://bucket/prefix/octmnist")
    assert got[1].phase == "done" and got[1].extracted == 5
    assert "shard 1/2" in state.summarize(got)


def test_read_states_ignores_unknown_fields(monkeypatch):
    """A newer producer adding a field must not crash an older driver."""
    from kprelogits.ops import s3
    doc = {"data_flag": "octmnist", "rank": 0, "shards": 1, "phase": "done",
           "future_field": 42}
    monkeypatch.setattr(s3, "list_names", lambda *a, **k: {"state_shard0.json"})
    monkeypatch.setattr(s3, "get_json", lambda uri: doc)
    assert state.read_states("s3://b/p")[0].phase == "done"


# ---------------------------------------------------------------------------
# Instance registry and reconciliation
# ---------------------------------------------------------------------------

def _inst(i, dph=0.5, label="kprelogits"):
    return {"id": i, "dph_total": dph, "label": label, "actual_status": "running"}


def test_registry_round_trip(tmp_path):
    r = vast.Registry(tmp_path / ".vast_instances")
    r.add(101, label="shard0")
    r.add(102, label="shard1")
    assert r.ids() == [101, 102]
    r.remove(101)
    assert r.ids() == [102]


def test_registry_reads_the_historic_line_format(tmp_path):
    p = tmp_path / ".vast_instances"
    p.write_text("101\n102\n")
    assert vast.Registry(p).ids() == [101, 102]


def test_orphans_are_what_the_api_knows_and_the_file_does_not(tmp_path):
    """The 4.3-hour orphan: rented ad hoc, never in the file, billing all
    along. Reconciliation is against the API precisely because of this."""
    r = vast.Registry(tmp_path / ".vast_instances")
    r.add(101)
    rec = vast.reconcile(r, instances=[_inst(101, 0.4), _inst(999, 1.1, "ad-hoc")])
    assert [i["id"] for i in rec.orphans] == [999]
    assert rec.burn == pytest.approx(1.5)
    assert "ORPHAN" in rec.render()


def test_stale_entries_are_reported_separately(tmp_path):
    r = vast.Registry(tmp_path / ".vast_instances")
    r.add(101)
    r.add(102)
    rec = vast.reconcile(r, instances=[_inst(101)])
    assert rec.stale == [102] and not rec.orphans


def test_burn_rate_survives_missing_prices():
    assert vast.burn_rate([{"id": 1}, _inst(2, 0.25)]) == pytest.approx(0.25)


def test_create_dry_run_rents_nothing(tmp_path, capsys):
    got = vast.create(12345, image="img", registry=vast.Registry(tmp_path / "reg"),
                      dry_run=True)
    assert got is None
    assert "DRY RUN" in capsys.readouterr().out
    assert not (tmp_path / "reg").exists()


def test_create_refuses_credentials_in_env():
    """`vastai create -e` puts values on a command line. Staging exists for
    exactly this."""
    with pytest.raises(ValueError, match="ps"):
        vast.create(1, image="img", env={"AWS_SECRET_ACCESS_KEY": "sekrit"},
                    dry_run=True)


def test_create_allows_ordinary_env(tmp_path, capsys):
    vast.create(1, image="img", env={"DATA_FLAG": "octmnist"},
                registry=vast.Registry(tmp_path / "reg"), dry_run=True)
    assert "DATA_FLAG=octmnist" in capsys.readouterr().out


def test_env_is_one_argument_not_repeated_flags(capsys):
    """The CLI's parser rejects repeated -e flags outright; the whole set has
    to arrive as a single --env string. Getting this wrong fails at create,
    which is the good case -- it fails before anything is rented."""
    vast.create(1, image="img", env={"A": "1", "B": "2"}, dry_run=True)
    out = capsys.readouterr().out
    assert out.count("--env") == 1
    assert "--env -e A=1 -e B=2" in out


def test_env_values_with_whitespace_are_refused():
    """Inside that single string a space would split into a bogus extra flag,
    and the box would come up misconfigured rather than not at all."""
    with pytest.raises(ValueError, match="whitespace"):
        vast.create(1, image="img", env={"X": "a b"}, dry_run=True)


def test_ssh_mode_is_off_by_default(capsys):
    vast.create(1, image="img", dry_run=True)
    assert "--ssh" not in capsys.readouterr().out


def test_ssh_mode_asks_for_a_direct_port(capsys):
    """Without --ssh, Vast runs the image's entrypoint -- and a stock image
    whose CMD is a shell exits at once, taking the instance with it."""
    vast.create(1, image="img", ssh=True, dry_run=True)
    out = capsys.readouterr().out
    assert "--ssh" in out and "--direct" in out


# ---- reaching the box -----------------------------------------------------

def _inst_ssh(**kw):
    base = {"id": 1, "ssh_host": "ssh4.vast.ai", "ssh_port": 28014,
            "public_ipaddr": "70.70.1.1", "direct_port_start": 41400}
    base.update(kw)
    return base


def test_proxy_route_comes_first():
    """The direct route hangs until timeout when the host is firewalled; the
    proxy always accepts a connection. SSH here carries commands and a small
    tarball, so the proxy's lower bandwidth costs nothing."""
    routes = vast.ssh_routes(_inst_ssh())
    assert routes[0] == ("proxy", "ssh4.vast.ai", 28014)
    assert routes[1] == ("direct", "70.70.1.1", 41400)


def test_a_host_offering_no_direct_port_yields_only_the_proxy():
    """direct_port_start is -1 on hosts that do not offer one at all."""
    routes = vast.ssh_routes(_inst_ssh(direct_port_start=-1))
    assert [r[0] for r in routes] == ["proxy"]


def test_an_instance_with_no_ssh_fields_yields_nothing():
    assert vast.ssh_routes({"id": 1}) == []


def test_ssh_endpoints_waits_for_the_fields_to_appear(monkeypatch):
    """A box that just reached running may not have published ssh fields yet,
    so one empty answer says nothing about reachability."""
    answers = [[], [_inst_ssh()]]
    monkeypatch.setattr(vast, "show_instances", lambda: answers.pop(0))
    slept = []
    routes = vast.ssh_endpoints(1, retries=3, delay=5, sleep=slept.append)
    assert routes[0][0] == "proxy" and slept == [5]


def test_ssh_endpoints_gives_up_with_the_fix_in_the_message(monkeypatch):
    monkeypatch.setattr(vast, "show_instances", lambda: [])
    with pytest.raises(vast.VastError, match="attach ssh"):
        vast.ssh_endpoints(77, retries=2, sleep=lambda _: None)


def test_ssh_args_carries_the_port_and_refuses_to_hang():
    a = vast.ssh_args("h", 2222)
    assert "-p" in a and "2222" in a and a[-1] == "root@h"
    # BatchMode: a missing key must fail now, not block on a password prompt
    # that no one is there to answer.
    assert "BatchMode=yes" in a


def test_teardown_runs_on_exception(monkeypatch):
    """The whole point of the context manager: a crash and a clean exit end at
    the same place."""
    destroyed = []
    monkeypatch.setattr(vast, "destroy", lambda i, **k: destroyed.append(i) or True)
    monkeypatch.setattr(vast, "show_instances", lambda: [])

    with pytest.raises(RuntimeError):
        with vast.teardown([101, 102]):
            raise RuntimeError("extraction blew up")
    assert destroyed == [101, 102]


def test_teardown_continues_past_a_failing_destroy(monkeypatch, capsys):
    """One unreachable box must not leave the others billing."""
    seen = []

    def flaky(i, **k):
        seen.append(i)
        if i == 101:
            raise vast.VastError("api timeout")
        return True

    monkeypatch.setattr(vast, "destroy", flaky)
    monkeypatch.setattr(vast, "show_instances", lambda: [])
    with vast.teardown([101, 102]):
        pass
    assert seen == [101, 102]
    assert "teardown error on 101" in capsys.readouterr().out


def test_teardown_disabled_says_so_loudly(monkeypatch, capsys):
    monkeypatch.setattr(vast, "destroy", lambda i, **k: pytest.fail("must not destroy"))
    with vast.teardown([101], enabled=False):
        pass
    assert "still billing" in capsys.readouterr().out


# ---- credential staging ---------------------------------------------------

def test_credentials_go_over_stdin_never_argv(monkeypatch):
    """If the payload appears in argv it appears in `ps` on a shared host."""
    calls = {}

    class R:
        returncode = 0
        stderr = ""

    def fake_run(args, **kw):
        calls["args"] = args
        calls["input"] = kw.get("input")
        return R()

    monkeypatch.setattr(vast.subprocess, "run", fake_run)
    vast.stage_credentials("root@host", {"AWS_SECRET_ACCESS_KEY": "sekrit"})

    assert "sekrit" not in " ".join(calls["args"])
    assert "AWS_SECRET_ACCESS_KEY=sekrit" in calls["input"]
    assert "umask 077" in " ".join(calls["args"])


def test_credentials_can_reach_a_nonstandard_port(monkeypatch):
    """Vast hands out a non-22 port, which a bare `user@host` cannot carry."""
    calls = {}

    class R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(vast.subprocess, "run",
                        lambda args, **kw: (calls.update(args=args, input=kw.get("input")), R())[1])
    vast.stage_credentials(vast.ssh_args("ssh4.vast.ai", 28014),
                           {"AWS_SECRET_ACCESS_KEY": "sekrit"})

    assert "28014" in calls["args"] and "root@ssh4.vast.ai" in calls["args"]
    assert "sekrit" not in " ".join(calls["args"])


def test_credential_values_are_quoted_for_the_shell(monkeypatch):
    """The staged file is SOURCED, so its contents are shell code. A value
    carrying `$(...)` would execute on the box at source time -- in the one
    helper whose entire purpose is handling secrets safely."""
    calls = {}

    class R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(vast.subprocess, "run",
                        lambda args, **kw: (calls.update(input=kw.get("input")), R())[1])
    vast.stage_credentials("root@h", {"AWS_SECRET_ACCESS_KEY": "a b$(id)'c"})

    line = calls["input"].strip()
    assert line.startswith("AWS_SECRET_ACCESS_KEY=")
    # Round-trip through the shell's own parser: what a POSIX shell would
    # assign must be the value we handed in, character for character.
    import shlex as _shlex
    assert _shlex.split(line)[0] == "AWS_SECRET_ACCESS_KEY=a b$(id)'c"


@pytest.mark.parametrize("path", ["/workspace/../etc/x; rm -rf /", "/tmp/$(id)"])
def test_unsafe_remote_paths_refused(path):
    with pytest.raises(ValueError, match="unsafe remote path"):
        vast.stage_credentials("host", {"A": "b"}, remote_path=path)


def test_invalid_env_name_refused():
    with pytest.raises(ValueError, match="invalid env var name"):
        vast.stage_credentials("host", {"BAD NAME": "b"})


# ---- pre-teardown ---------------------------------------------------------

def _preteardown(monkeypatch, states, bundles):
    from kprelogits.ops import s3
    monkeypatch.setattr(state, "read_states", lambda prefix: states)
    monkeypatch.setattr(s3, "list_names", lambda *a, **k: bundles)
    return vast.preteardown_check("s3://b/p/octmnist", expected_shards=len(states) or 1,
                                  expected_bundles=len(bundles) or None)


def _done_state(rank, n=2):
    return state.ShardState(data_flag="octmnist", rank=rank, shards=1,
                            phase="done", scheduled=n, extracted=n, uploaded=n)


def test_teardown_cleared_when_states_and_bundles_agree(monkeypatch):
    chk = _preteardown(monkeypatch, {0: _done_state(0)}, {"a_features.npz", "b_features.npz"})
    assert chk.ok, chk.render()


def test_running_shard_blocks_teardown(monkeypatch):
    st = _done_state(0)
    st.phase = "running"
    chk = _preteardown(monkeypatch, {0: st}, {"a_features.npz", "b_features.npz"})
    assert not chk.ok and any("is running" in b for b in chk.blockers)


def test_missing_state_blocks_teardown(monkeypatch):
    """No published state is indistinguishable from a box still working, so it
    must block -- this is the case `pgrep` used to answer wrongly."""
    from kprelogits.ops import s3
    monkeypatch.setattr(state, "read_states", lambda prefix: {})
    monkeypatch.setattr(s3, "list_names", lambda *a, **k: set())
    chk = vast.preteardown_check("s3://b/p", expected_shards=2)
    assert not chk.ok and any("no published state" in b for b in chk.blockers)


def test_bundles_missing_from_s3_block_teardown(monkeypatch):
    """A shard saying `done` while S3 is short is the α=10 near-loss: fits
    that existed only on a box about to be destroyed."""
    from kprelogits.ops import s3
    monkeypatch.setattr(state, "read_states", lambda prefix: {0: _done_state(0, 5)})
    monkeypatch.setattr(s3, "list_names", lambda *a, **k: {"a_features.npz"})
    chk = vast.preteardown_check("s3://b/p", expected_shards=1, expected_bundles=5)
    assert not chk.ok and any("only 1 of 5" in b for b in chk.blockers)
    assert "DO NOT DESTROY" in chk.render()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@pytest.fixture
def cache(tmp_path):
    p = tmp_path / "cache.json"
    p.write_text(json.dumps({
        "big_net": {"params": 900_000_000, "status": "too_large"},
        "ok_a": {"params": 10_000_000, "status": "ok"},
        "ok_b": {"params": 20_000_000, "status": "ok"},
        "ok_c": {"params": 30_000_000, "status": "ok"},
        "ok_d": {"params": 40_000_000, "status": "ok"},
        "grey_net": {"params": None, "status": "bad_input"},
        "broken": {"params": None, "status": "error"},
    }))
    return p


def test_from_cache_applies_our_ceiling_not_the_recorded_one(cache):
    """`too_large` records keep a valid param count, so raising the ceiling
    must let them back in without a rescan."""
    assert {m["name"] for m in from_cache(cache, 200_000_000)} == \
        {"ok_a", "ok_b", "ok_c", "ok_d"}
    assert "big_net" in {m["name"] for m in from_cache(cache, 1_000_000_000)}


def test_from_cache_drops_unresolved_entries(cache):
    names = {m["name"] for m in from_cache(cache, 10**9)}
    assert "grey_net" not in names and "broken" not in names


def test_selection_sorts_before_striding():
    """Scan order depends on what was already cached, so an unsorted stride is
    not reproducible."""
    unordered = [{"name": n, "params": 1} for n in ["d", "a", "c", "b"]]
    sel = build_selection(unordered, max_params=1, stride=2)
    assert [m["name"] for m in sel["models"]] == ["a", "c"]


def test_offset_partitions_the_stride():
    models = [{"name": f"m{i:02d}", "params": 1} for i in range(10)]
    got = set()
    for off in range(3):
        got |= {m["name"] for m in
                build_selection(models, max_params=1, stride=3, offset=off)["models"]}
    assert len(got) == 10


def test_limit_caps_after_striding():
    models = [{"name": f"m{i:02d}", "params": 1} for i in range(10)]
    sel = build_selection(models, max_params=1, stride=2, limit=3)
    assert sel["n_selected"] == 3


@pytest.mark.parametrize("kw,match", [
    (dict(stride=0), "stride must be"),
    (dict(stride=3, offset=3), r"offset must be in \[0, 3\)"),
    (dict(stride=3, offset=-1), "offset must be in"),
])
def test_invalid_stride_or_offset_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        build_selection([{"name": "a", "params": 1}], max_params=1, **kw)


def test_selection_records_how_it_was_made():
    """The selection is the input to a run that costs money; it has to say
    what it is."""
    sel = build_selection([{"name": "a", "params": 1}], max_params=5, stride=1,
                          source="cache")
    assert sel["schema"] == 1 and sel["max_params"] == 5
    assert sel["source"] == "cache" and "offset::stride" in sel["selection"]


# ---- ssh identity ---------------------------------------------------------

def test_ssh_args_passes_an_identity_when_one_exists(tmp_path):
    """With BatchMode and an empty agent, ssh offers nothing and the box
    rejects the login -- while billing by the minute."""
    key = tmp_path / "id_rsa"
    key.write_text("x")
    a = vast.ssh_args("h", 22, key=key)
    assert a[0] == "-i" and a[1] == str(key)


def test_ssh_args_omits_a_missing_identity(tmp_path):
    """A -i pointing at nothing makes ssh fail outright; better to let it try
    its defaults than to hand it a path that cannot work."""
    a = vast.ssh_args("h", 22, key=tmp_path / "absent")
    assert "-i" not in a


def test_account_ssh_keys_reads_the_api(monkeypatch):
    class R:
        returncode, stdout, stderr = 0, '[{"id": 1, "public_key": "ssh-rsa AAA"}]', ""
    monkeypatch.setattr(vast, "_vastai", lambda *a, **k: R())
    assert len(vast.account_ssh_keys()) == 1


def test_no_account_keys_is_an_empty_list_not_a_crash(monkeypatch):
    """The state that caused a wasted rental: the API answers fine, with
    nothing in it."""
    class R:
        returncode, stdout, stderr = 0, "[]", ""
    monkeypatch.setattr(vast, "_vastai", lambda *a, **k: R())
    assert vast.account_ssh_keys() == []


# ---- per-instance keys ----------------------------------------------------

def test_public_key_reads_only_the_public_half(tmp_path):
    (tmp_path / "k.pub").write_text("ssh-rsa AAAAB3 abe@host\n")
    assert vast.public_key(tmp_path / "k").startswith("ssh-rsa ")


def test_a_missing_public_key_is_caught_before_renting(tmp_path):
    with pytest.raises(vast.VastError, match="no public key"):
        vast.public_key(tmp_path / "absent")


def test_a_private_key_is_refused_as_a_public_one(tmp_path):
    """Pointing at the wrong half would send private material to a third
    party. It must fail, and fail before anything is rented."""
    (tmp_path / "k.pub").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    with pytest.raises(vast.VastError, match="does not look like"):
        vast.public_key(tmp_path / "k")


def test_attach_ssh_key_targets_one_instance(monkeypatch):
    """Team accounts cannot hold account-wide keys, and per-instance is the
    tighter grant regardless: it dies with the box."""
    seen = {}

    class R:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(vast, "_vastai", lambda a, **k: (seen.update(a=a), R())[1])
    vast.attach_ssh_key(4242, "ssh-rsa AAAA")
    assert seen["a"] == ["attach", "ssh", "4242", "ssh-rsa AAAA"]


def test_attach_failure_is_loud(monkeypatch):
    class R:
        returncode, stdout, stderr = 1, "", "nope"

    monkeypatch.setattr(vast, "_vastai", lambda *a, **k: R())
    with pytest.raises(vast.VastError, match="attach ssh key"):
        vast.attach_ssh_key(1, "ssh-rsa AAAA")


# ---- empty prefixes are not errors ----------------------------------------

def _aws(rc, out="", err=""):
    class R:
        returncode, stdout, stderr = rc, out, err
    return lambda args, **kw: R()


def test_an_empty_prefix_is_listable(monkeypatch):
    """S3 has no directories, so a fresh prefix holds nothing and `aws s3 ls`
    exits 1 with empty stderr. Calling that unreachable makes every first run
    look like a credentials failure."""
    from kprelogits.ops import s3
    monkeypatch.setattr(s3, "_run", _aws(1, "", ""))
    ok, err = s3.bucket_listable("s3://bucket/brand-new-prefix")
    assert ok and err == ""


def test_a_real_s3_error_is_still_an_error(monkeypatch):
    from kprelogits.ops import s3
    monkeypatch.setattr(s3, "_run", _aws(1, "", "An error occurred (NoSuchBucket)"))
    ok, err = s3.bucket_listable("s3://nope")
    assert not ok and "NoSuchBucket" in err


def test_listing_an_empty_prefix_returns_nothing_quietly(monkeypatch, capsys):
    """The resume oracle must read 'nothing done yet', not warn that S3 is
    unreachable and silently fall back to local disk."""
    from kprelogits.ops import s3
    monkeypatch.setattr(s3, "_run", _aws(1, "", ""))
    assert s3.list_names("s3://bucket/fresh") == set()
    assert "WARNING" not in capsys.readouterr().out


def test_listing_parses_names(monkeypatch):
    from kprelogits.ops import s3
    listing = ("2026-08-05 11:58:56   67245484 octmnist_a_features.npz\n"
               "2026-08-05 11:58:57   67245484 octmnist_b_features.npz\n")
    monkeypatch.setattr(s3, "_run", _aws(0, listing, ""))
    assert s3.list_names("s3://b/p", suffix="_features.npz") == {
        "octmnist_a_features.npz", "octmnist_b_features.npz"}


# ---- connection settings shared by ssh and scp ----------------------------

def test_scp_and_ssh_agree_on_everything_but_the_port_flag(tmp_path):
    """Authenticating over ssh and then failing to scp -- because the flags
    were retyped and the identity dropped -- is a confusing way to lose a box."""
    key = tmp_path / "id_rsa"
    key.write_text("x")
    (tmp_path / "id_rsa.pub").write_text("ssh-rsa AAAA")
    a = vast.ssh_args("h", 2222, key=key)
    b = vast.scp_args("h", 2222, key=key)
    assert "-p" in a and "-P" in b          # the one difference
    for opt in ("-i", "BatchMode=yes", "StrictHostKeyChecking=accept-new"):
        assert opt in a and opt in b


def test_rsa_keys_get_the_legacy_algorithm_opt_in(tmp_path):
    """OpenSSH 8.8+ refuses SHA-1 RSA by default and Vast hosts run a range of
    sshd versions, so the same key works on some boxes and not others."""
    key = tmp_path / "id_rsa"
    key.write_text("x")
    assert "PubkeyAcceptedAlgorithms=+ssh-rsa" in vast.ssh_args("h", 22, key=key)


def test_modern_keys_do_not_re_enable_legacy_algorithms(tmp_path):
    """An ed25519 key needs no compatibility concession; making one anyway
    weakens every connection for nothing."""
    key = tmp_path / "id_ed25519"
    key.write_text("x")
    assert "PubkeyAcceptedAlgorithms=+ssh-rsa" not in vast.ssh_args("h", 22, key=key)


def test_a_per_run_known_hosts_file_is_honoured(tmp_path):
    """Vast recycles host:port, so a new box at a seen address trips OpenSSH's
    man-in-the-middle check -- a guaranteed false alarm that costs a rental."""
    kh = tmp_path / "known_hosts"
    assert f"UserKnownHostsFile={kh}" in vast.ssh_args("h", 22, known_hosts=kh)
    assert f"UserKnownHostsFile={kh}" in vast.scp_args("h", 22, known_hosts=kh)


def test_ed25519_is_preferred_over_rsa(tmp_path, monkeypatch):
    """Vast's own `create ssh-key` generates ed25519, so a user following its
    setup instructions must not be told they have no key."""
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    for n in ("id_rsa", "id_ed25519"):
        (ssh / n).write_text("x")
        (ssh / f"{n}.pub").write_text("k")
    monkeypatch.delenv("KPRELOGITS_SSH_KEY", raising=False)
    monkeypatch.setattr(vast.Path, "home", staticmethod(lambda: tmp_path))
    assert vast.default_ssh_key().name == "id_ed25519"


def test_a_key_without_its_public_half_is_not_usable(tmp_path, monkeypatch):
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_rsa").write_text("x")      # no .pub: cannot attach it anywhere
    monkeypatch.delenv("KPRELOGITS_SSH_KEY", raising=False)
    monkeypatch.setattr(vast.Path, "home", staticmethod(lambda: tmp_path))
    assert vast.default_ssh_key() is None


# ---- destroy confirmation and ambiguous creates ---------------------------

def test_destroy_waits_out_an_asynchronous_deletion(monkeypatch, tmp_path):
    """Deletion is async; one immediate check can still see the box and cry
    STILL BILLING about one that is already on its way out."""
    seen = [[_inst(5)], [_inst(5)], []]
    monkeypatch.setattr(vast, "_vastai", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(vast, "show_instances", lambda: seen.pop(0))
    r = vast.Registry(tmp_path / "reg")
    r.add(5)
    assert vast.destroy(5, registry=r, sleep=lambda _: None) is True
    assert r.ids() == []


def test_a_box_that_never_disappears_is_reported_as_billing(monkeypatch, capsys):
    monkeypatch.setattr(vast, "_vastai", lambda *a, **k: type(
        "R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(vast, "show_instances", lambda: [_inst(5)])
    assert vast.destroy(5, confirm_tries=2, sleep=lambda _: None) is False
    assert "STILL BILLING" in capsys.readouterr().out


def test_find_by_label_recovers_an_ambiguously_created_box(monkeypatch):
    """A create that fails locally does not prove nothing was rented: Vast may
    have accepted the contract and lost the answer on the way back."""
    monkeypatch.setattr(vast, "show_instances",
                        lambda: [_inst(1, label="other"), _inst(2, label="smoke-x")])
    assert [i["id"] for i in vast.find_by_label("smoke-x")] == [2]


def test_a_stopped_instance_is_flagged_as_still_costing(capsys):
    """dph_total is compute. A stopped box reports ~none of it and keeps
    billing for the disk it holds, so a reassuring $0.000/hr is not the same
    as costing nothing -- only destroying releases the storage."""
    out = vast.describe([{"id": 7, "dph_total": 0.0, "actual_status": "stopped"}])
    assert "still" in out and "storage" in out and "7" in out


def test_running_instances_get_no_storage_note():
    out = vast.describe([_inst(1)])
    assert "storage" not in out
