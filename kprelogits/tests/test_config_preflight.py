"""Preflight must catch, on the laptop, every failure that would otherwise
surface twenty minutes into a paid rental.

Each test names one such failure. They all run offline: the probing checks
(S3, HuggingFace) are switched off, which is exactly the split the report was
designed around.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits.config import (  # noqa: E402
    ExtractConfig, check_no_secrets, preflight, read_hf_token,
)


@pytest.fixture
def selection(tmp_path):
    p = tmp_path / "_shared" / "selected_models.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "schema": 1,
        "models": [{"name": f"model_{i}", "params": 10_000_000} for i in range(6)],
    }))
    return p


def _cfg(tmp_path, selection, **kw):
    base = dict(data_flag="octmnist", data_dir=tmp_path / "data",
                results_dir=tmp_path / "results", selection=selection)
    return ExtractConfig(**{**base, **kw})


def _run(cfg):
    return preflight(cfg, check_s3=False, check_hf=False)


def test_valid_config_passes(tmp_path, selection):
    rep = _run(_cfg(tmp_path, selection))
    assert rep.ok, rep.render()
    assert rep.facts["n_selected"] == 6


@pytest.mark.parametrize("kw,match", [
    (dict(rank=2, shards=2), "rank must be in"),
    (dict(rank=-1), "rank must be in"),
    (dict(shards=0), "shards must be >= 1"),
    (dict(precision="bf16"), "precision must be"),
    (dict(max_train=-1), "max_train must be"),
    (dict(feature_batch_size=0), "feature_batch_size"),
    (dict(cleanup=True), "nowhere to put them"),
    (dict(s3_uri="/not/a/uri"), "must start with s3://"),
])
def test_structural_violations_are_named(tmp_path, selection, kw, match):
    rep = _run(_cfg(tmp_path, selection, **kw))
    assert not rep.ok
    assert any(match in e for e in rep.errors), rep.errors


def test_all_errors_reported_at_once(tmp_path, selection):
    """One preflight, one fix cycle -- reporting only the first error would
    mean re-running the whole check for each mistake."""
    rep = _run(_cfg(tmp_path, selection, rank=9, shards=2, precision="bf16",
                    max_train=-5))
    assert len(rep.errors) >= 3


def test_missing_selection_is_an_error(tmp_path):
    rep = _run(_cfg(tmp_path, tmp_path / "nope.json"))
    assert any("no selection file" in e for e in rep.errors)


def test_unparseable_selection_is_an_error(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text("{not json")
    rep = _run(_cfg(tmp_path, p))
    assert any("does not parse" in e for e in rep.errors)


def test_selection_without_models_is_an_error(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"schema": 1, "models": []}))
    rep = _run(_cfg(tmp_path, p))
    assert any("no non-empty 'models'" in e for e in rep.errors)


def test_selection_entries_need_names(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"models": [{"params": 1}, {"name": "ok"}]}))
    rep = _run(_cfg(tmp_path, p))
    assert any("lack a 'name'" in e for e in rep.errors)


def test_shard_that_schedules_nothing_is_an_error(tmp_path, selection):
    """A box that would boot, find no work, and bill anyway."""
    rep = _run(_cfg(tmp_path, selection, rank=7, shards=8))
    assert rep.facts["n_scheduled"] == 0
    assert any("would do nothing" in e for e in rep.errors)


def test_shards_partition_the_selection(tmp_path, selection):
    counts = [_run(_cfg(tmp_path, selection, rank=r, shards=3)).facts["n_scheduled"]
              for r in range(3)]
    assert sum(counts) == 6


def test_oversized_models_warn_but_do_not_block(tmp_path):
    """kprelogits has no silent size gate (ktrain's run_model did) -- big
    models are slow, not skipped, so this is a warning."""
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"models": [{"name": "giant", "params": 900_000_000}]}))
    rep = _run(_cfg(tmp_path, p))
    assert rep.ok
    assert any("exceed" in w for w in rep.warnings)


def test_no_s3_warns_about_losing_everything(tmp_path, selection):
    rep = _run(_cfg(tmp_path, selection))
    assert any("lost" in w for w in rep.warnings)


def test_missing_hf_token_file_is_an_error(tmp_path, selection):
    rep = _run(_cfg(tmp_path, selection, hf_token_file=tmp_path / "absent"))
    assert any("hf_token_file does not exist" in e for e in rep.errors)


# ---- token handling -------------------------------------------------------

def test_token_read_from_file_not_environment(tmp_path, selection, monkeypatch):
    tok = tmp_path / "hf.token"
    tok.write_text("hf_secret\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert read_hf_token(_cfg(tmp_path, selection, hf_token_file=tok)) == "hf_secret"


def test_token_falls_back_to_environment(tmp_path, selection, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert read_hf_token(_cfg(tmp_path, selection)) == "from_env"


def test_resolved_config_carries_no_token(tmp_path, selection):
    """The resolved config is uploaded to S3 next to the bundles; a token in it
    would be a published secret."""
    tok = tmp_path / "hf.token"
    tok.write_text("hf_secret")
    blob = json.dumps(_cfg(tmp_path, selection, hf_token_file=tok).resolved())
    assert "hf_secret" not in blob
    assert str(tok) in blob


# ---- the ps-visibility guard ----------------------------------------------

@pytest.mark.parametrize("key", [
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "HF_TOKEN", "MY_PASSWORD",
])
def test_credential_shaped_env_is_refused(key):
    """Vast hosts are shared and ps is world-readable; a key was exposed this
    way earlier in the project."""
    with pytest.raises(ValueError, match="ps"):
        check_no_secrets({key: "value"})


def test_ordinary_env_is_allowed():
    check_no_secrets({"DATA_FLAG": "octmnist", "PARALLEL_RANK": "0"})


# ---- identity and derived paths -------------------------------------------

def test_bundle_identity_tracks_content_determining_fields(tmp_path, selection):
    a = _cfg(tmp_path, selection)
    assert a.bundle_identity() != _cfg(tmp_path, selection, max_train=500).bundle_identity()
    assert a.bundle_identity() != _cfg(tmp_path, selection, precision="fp16").bundle_identity()
    # rank/shards do not change what is inside a bundle
    assert a.bundle_identity() == _cfg(tmp_path, selection, rank=1, shards=4).bundle_identity()


def test_derived_paths(tmp_path, selection):
    cfg = _cfg(tmp_path, selection, s3_uri="s3://bucket/prelogits")
    assert cfg.s3_prefix == "s3://bucket/prelogits/octmnist"
    assert cfg.bundle_name("convnext_small.fb_in1k") == \
        "octmnist_convnext_small.fb_in1k_features.npz"
    assert cfg.bundle_name("org/model") == "octmnist_org__model_features.npz"
    assert cfg.features_dir.name == "features"


def test_from_env_reads_the_container_surface(monkeypatch):
    monkeypatch.setenv("DATA_FLAG", "dermamnist")
    monkeypatch.setenv("PARALLEL_RANK", "2")
    monkeypatch.setenv("PARALLEL_SHARDS", "4")
    monkeypatch.setenv("S3_BUCKET", "mybucket")
    monkeypatch.setenv("CLEANUP", "1")
    cfg = ExtractConfig.from_env()
    assert (cfg.data_flag, cfg.rank, cfg.shards, cfg.cleanup) == ("dermamnist", 2, 4, True)
    assert cfg.s3_uri == "s3://mybucket/medmnist_prelogits"


def test_from_env_overrides_win(monkeypatch):
    monkeypatch.setenv("DATA_FLAG", "dermamnist")
    assert ExtractConfig.from_env(data_flag="octmnist").data_flag == "octmnist"
    assert ExtractConfig.from_env(data_flag=None).data_flag == "dermamnist"
