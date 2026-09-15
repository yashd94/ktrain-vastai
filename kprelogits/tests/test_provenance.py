"""Where a bundle's weights revision comes from.

The revision used to come from one Hub API call per model, anonymously, inside
a bare ``except: pass``. Under a rate limit that call fails and the manifest
records ``hf:timm/x`` with no sha -- valid-looking, unpinned, and invisible.
It now comes from the local HF cache the download just populated, which needs
no request and names the commit that was actually loaded. timm is not
installed here, so these tests stand in fakes for it and for the Hub client.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits import models as M  # noqa: E402

SHA = "e5c9e1af0cd6" + "0" * 28          # 40 hex, prefix matches a real bundle
OTHER = "a" * 40


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return tmp_path / "hub"


def _cache(hub: Path, repo: str, sha: str, ref: str = "main") -> None:
    d = hub / f"models--{repo.replace('/', '--')}" / "refs"
    d.mkdir(parents=True, exist_ok=True)
    (d / ref).write_text(sha + "\n")


def _fake_timm(monkeypatch, hf_hub_id):
    cfg = types.SimpleNamespace(hf_hub_id=hf_hub_id, tag=None, url=None)
    monkeypatch.setitem(sys.modules, "timm", types.SimpleNamespace(
        get_pretrained_cfg=lambda name: cfg))


def _fake_hub_api(monkeypatch, *, sha=None, raises=None):
    calls = []

    def model_info(repo, **kw):
        calls.append(repo)
        if raises:
            raise raises
        return types.SimpleNamespace(sha=sha)

    mod = types.ModuleType("huggingface_hub")
    mod.model_info = model_info
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return calls


# ---- reading the cache ----------------------------------------------------

def test_reads_the_commit_the_download_recorded(hub):
    _cache(hub, "timm/convnext_atto_ols.a2_in1k", SHA)
    assert M.cached_hf_sha("timm/convnext_atto_ols.a2_in1k") == SHA


def test_no_cache_entry_is_none_not_an_error(hub):
    assert M.cached_hf_sha("timm/never_downloaded") is None


def test_a_malformed_ref_is_not_trusted(hub):
    _cache(hub, "timm/x", "not-a-sha")
    assert M.cached_hf_sha("timm/x") is None


def test_a_pinned_sha_in_the_id_is_returned_as_is(hub):
    assert M.cached_hf_sha(f"timm/x@{OTHER}") == OTHER


def test_a_named_revision_reads_its_own_ref(hub):
    _cache(hub, "timm/x", SHA, ref="main")
    _cache(hub, "timm/x", OTHER, ref="v2")
    assert M.cached_hf_sha("timm/x@v2") == OTHER


def test_hf_hub_cache_overrides_hf_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "unused"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    _cache(tmp_path / "elsewhere", "timm/x", SHA)
    assert M.cached_hf_sha("timm/x") == SHA


# ---- what weights_revision stamps ----------------------------------------

def test_the_cache_is_preferred_and_the_api_is_never_called(hub, monkeypatch):
    """The point of the change: no request per model, so no rate limit."""
    _fake_timm(monkeypatch, "timm/m")
    _cache(hub, "timm/m", SHA)
    calls = _fake_hub_api(monkeypatch, sha=OTHER)
    assert M.weights_revision("m") == f"hf:timm/m@{SHA[:12]}"
    assert calls == []


def test_the_cache_wins_over_a_newer_hub_head(hub, monkeypatch):
    """If the repo is pushed after the download, the manifest must name what
    was loaded, not what the Hub now calls main."""
    _fake_timm(monkeypatch, "timm/m")
    _cache(hub, "timm/m", SHA)
    _fake_hub_api(monkeypatch, sha=OTHER)
    assert SHA[:12] in M.weights_revision("m")


def test_falls_back_to_the_api_when_the_cache_has_nothing(hub, monkeypatch):
    _fake_timm(monkeypatch, "timm/m")
    calls = _fake_hub_api(monkeypatch, sha=OTHER)
    assert M.weights_revision("m") == f"hf:timm/m@{OTHER[:12]}"
    assert calls == ["timm/m"]


def test_an_unpinned_result_is_announced_not_swallowed(hub, monkeypatch, capsys):
    """Both sources failing still yields a usable manifest, but the log must
    say so -- an unpinned revision looks exactly as valid as a pinned one."""
    _fake_timm(monkeypatch, "timm/m")
    _fake_hub_api(monkeypatch, raises=RuntimeError("429 Too Many Requests"))
    assert M.weights_revision("m") == "hf:timm/m"
    out = capsys.readouterr().out
    assert "unpinned" in out and "429" in out


def test_the_stamp_format_matches_bundles_already_in_s3(hub, monkeypatch):
    """50 dermamnist bundles already carry hf:<repo>@<12 hex>. A format change
    here would make one sweep's manifests unlike the next's for no reason."""
    _fake_timm(monkeypatch, "timm/convnext_atto_ols.a2_in1k")
    _cache(hub, "timm/convnext_atto_ols.a2_in1k", SHA)
    _fake_hub_api(monkeypatch, sha=OTHER)
    assert (M.weights_revision("convnext_atto_ols.a2_in1k")
            == "hf:timm/convnext_atto_ols.a2_in1k@e5c9e1af0cd6")


def test_a_revision_suffix_does_not_leak_into_the_repo_name(hub, monkeypatch):
    _fake_timm(monkeypatch, f"timm/m@{OTHER}")
    _fake_hub_api(monkeypatch, raises=AssertionError("must not be called"))
    assert M.weights_revision("m") == f"hf:timm/m@{OTHER[:12]}"
