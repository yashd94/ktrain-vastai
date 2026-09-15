"""Which GPUs the image's torch can actually run.

On 2026-09-14 an RTX 5060 Ti (Blackwell, compute capability 12.0) passed every
check, launched, and then failed every model of its shard with "no kernel
image is available for execution on the device": the image's torch 2.5.1/cu124
carries kernels up to sm_90. The rule has to follow CUDA's real compatibility,
though, not an exact match -- sm_86 code runs on the sm_89 RTX 4060 Tis that
extracted hundreds of bundles without complaint.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from kprelogits import config  # noqa: E402
from kprelogits.config import cuda_arch_supported  # noqa: E402

TORCH_251_CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


@pytest.mark.parametrize("cap, ok", [
    ((7, 5), True),     # RTX 2080 Ti, Quadro RTX 5000
    ((8, 0), True),
    ((8, 6), True),     # RTX A4000, 3080
    ((8, 9), True),     # RTX 4060 Ti -- via the sm_86 cubins
    ((9, 0), True),
    ((10, 0), False),
    ((12, 0), False),   # RTX 5060 Ti -- the 2026-09-14 failure
    ((3, 7), False),
])
def test_which_cards_the_image_can_run(cap, ok):
    assert cuda_arch_supported(cap, TORCH_251_CU124) is ok


def test_an_ada_card_runs_on_ampere_cubins():
    """An exact-match rule would reject working cards: this torch lists no
    sm_89, yet the 4060 Tis extracted fine."""
    assert "sm_89" not in TORCH_251_CU124
    assert cuda_arch_supported((8, 9), TORCH_251_CU124)


def test_ptx_is_forward_compatible():
    assert cuda_arch_supported((12, 0), ["sm_90", "compute_90"])


def test_malformed_arch_entries_are_ignored():
    assert not cuda_arch_supported((8, 6), ["gfx90a", "sm_", "compute"])


def _fake_torch(monkeypatch, cap, arches, available=True):
    cuda = types.SimpleNamespace(is_available=lambda: available,
                                 get_device_capability=lambda i=0: cap,
                                 get_arch_list=lambda: arches)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def test_preflight_refuses_a_card_with_no_kernels(monkeypatch):
    _fake_torch(monkeypatch, (12, 0), TORCH_251_CU124)
    rep = config.PreflightReport()
    config._check_cuda_arch(rep)
    assert any("12.0" in e and "no kernels" in e for e in rep.errors)


def test_preflight_passes_a_supported_card(monkeypatch):
    _fake_torch(monkeypatch, (8, 9), TORCH_251_CU124)
    rep = config.PreflightReport()
    config._check_cuda_arch(rep)
    assert rep.errors == [] and rep.facts["cuda_capability"] == "8.9"


def test_no_gpu_is_recorded_not_refused(monkeypatch):
    """A laptop preflight has nothing to judge."""
    _fake_torch(monkeypatch, (0, 0), [], available=False)
    rep = config.PreflightReport()
    config._check_cuda_arch(rep)
    assert rep.errors == [] and rep.facts["cuda"] == "absent"


def test_the_offer_query_excludes_what_the_image_cannot_run():
    """Keep the query and the image in step: a card the preflight would refuse
    should not be rented in the first place."""
    from kprelogits.ops.smoke import OFFER_QUERY
    assert "compute_cap<=900" in OFFER_QUERY and "compute_cap>=750" in OFFER_QUERY
