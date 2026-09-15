"""A machine that cannot reach the HuggingFace Hub must fail preflight.

On 2026-09-14 a vast host's DNS answered huggingface.co with a Meta IPv6
address. TLS verification correctly refused every connection -- and preflight
passed anyway, because its Hub probe reported "could not resolve 5 repo(s)" as
a warning. The shard then failed its models one TLS error at a time. Every
probe failing to connect means no weight file will download, which is an
error; a box that fails preflight is released and replaced by the driver.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits import config  # noqa: E402


class GatedRepoError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


class SSLError(ConnectionError):
    """requests.exceptions.SSLError is, ultimately, an OSError."""


def _probe(monkeypatch, outcome):
    """Run _check_hf over five repos; ``outcome(name)`` returns or raises."""
    hub = types.ModuleType("huggingface_hub")
    hub.model_info = lambda repo, token=None: outcome(repo)
    utils = types.ModuleType("huggingface_hub.utils")
    utils.GatedRepoError, utils.RepositoryNotFoundError = GatedRepoError, RepositoryNotFoundError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    monkeypatch.setattr(config, "read_hf_token", lambda cfg: None)
    rep = config.PreflightReport()
    config._check_hf(None, [{"name": f"m{i}"} for i in range(5)], rep, sample=5)
    return rep


def _raise(e):
    def f(repo):
        raise e
    return f


def test_a_hub_that_cannot_be_reached_fails_preflight(monkeypatch):
    rep = _probe(monkeypatch, _raise(SSLError(
        "certificate verify failed: Hostname mismatch, certificate is not "
        "valid for 'huggingface.co'")))
    assert any("cannot reach the HuggingFace Hub" in e for e in rep.errors)


def test_connection_refused_and_timeouts_count_too(monkeypatch):
    rep = _probe(monkeypatch, _raise(TimeoutError("timed out")))
    assert any("cannot reach" in e for e in rep.errors)


def test_one_reachable_repo_keeps_it_a_warning(monkeypatch):
    """Partial failure is a flaky repo, not a dead network."""
    def outcome(repo):
        if repo.endswith("m0"):
            return object()
        raise ConnectionError("reset")
    rep = _probe(monkeypatch, outcome)
    assert not any("cannot reach" in e for e in rep.errors)
    assert any("could not resolve" in w for w in rep.warnings)


def test_a_non_network_error_is_not_mistaken_for_an_outage(monkeypatch):
    rep = _probe(monkeypatch, _raise(ValueError("odd metadata")))
    assert not any("cannot reach" in e for e in rep.errors)


def test_a_healthy_hub_passes(monkeypatch):
    rep = _probe(monkeypatch, lambda repo: object())
    assert rep.errors == []
