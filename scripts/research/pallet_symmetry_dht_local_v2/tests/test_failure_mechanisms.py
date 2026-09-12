from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.hough import Lattice
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.geometry import (
    candidate_utility, continuous_line_targets, decode_single_mode, single_mode_wls,
)
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.model import LocalLineFusionV2
from scripts.research.pallet_symmetry_dht_local_v2.symdht_local_v2.runner import state_sha, utility_sha


def corrected_loss(error, correction, diagonal=800.):
    after = abs(error - correction)
    point = F.smooth_l1_loss(torch.tensor([after / diagonal]), torch.zeros(1)).item()
    no_harm = .5 * max(after - abs(error), 0.) / diagonal
    return point + no_harm


def test_A_useful_corrections_have_strict_loss_ordering_without_anchor():
    for error in (2., 5., 10.):
        baseline = corrected_loss(error, 0.)
        partial = corrected_loss(error, error / 2)
        exact = corrected_loss(error, error)
        assert exact < partial < baseline
    assert corrected_loss(5., 5.) < corrected_loss(5., 0.)


def _fixture():
    base = torch.zeros(1, 9, 2); valid = torch.ones(1, 9, dtype=torch.bool)
    sigma = torch.ones(1, 9); image_hw = torch.tensor([[800., 800.]])
    lines = torch.zeros(1, 12, 3); lines[..., 0] = 1
    return base, valid, sigma, image_hw, lines


def test_B_support_and_correction_utility_are_separate():
    base, valid, sigma, hw, lines = _fixture()
    target = base.clone(); target[0, 1, 1] = 10
    lattice = Lattice(); box = torch.tensor([[-20., -20., 20., 20.]])
    _, _, support, _ = continuous_line_targets(target, valid, box, lattice)
    assert support[0, 0]  # The GT structural edge exists even when this candidate is harmful.
    lines[:, 0, 2] = -10
    utility, gains, usable = candidate_utility(base, valid, sigma, lines, target, valid, torch.ones(1), hw)
    assert usable[0, 0] and utility[0, 0] == 0 and gains[0, 0] < 0
    corrected, _ = single_mode_wls(base, valid, sigma, lines, torch.zeros(1, 12), torch.ones(1), hw)
    assert torch.equal(corrected, base)
    base[:, :2, 0] = 5; target[:, :2, 0] = 0; lines[:, 0, 2] = 0
    utility, gains, _ = candidate_utility(base, valid, sigma, lines, target, valid, torch.ones(1), hw)
    assert utility[0, 0] == 1 and gains[0, 0] > 0
    corrected, _ = single_mode_wls(base, valid, sigma, lines, utility, torch.ones(1), hw)
    assert torch.linalg.vector_norm(corrected[0, :2] - target[0, :2], dim=-1).mean() < torch.linalg.vector_norm(base[0, :2] - target[0, :2], dim=-1).mean()


def test_C_soft_target_refines_between_rho_bins_toward_continuous_GT():
    lattice = Lattice(); points = torch.zeros(1, 9, 2); valid = torch.ones(1, 9, dtype=torch.bool)
    # Edge 0 is the vertical line x=203.9 in a 400px ROI.
    points[0, 0] = torch.tensor([203.9, 50.]); points[0, 1] = torch.tensor([203.9, 350.])
    box = torch.tensor([[0., 0., 400., 400.]])
    soft, hard, support, _ = continuous_line_targets(points, valid, box, lattice)
    decoded = decode_single_mode(soft.clamp_min(1e-30).log(), torch.ones_like(soft, dtype=torch.bool), lattice, box)
    x_soft = float(-decoded["raw_line"][0, 0, 2] / decoded["raw_line"][0, 0, 0])
    hard_line = lattice.lines[hard[0, 0]]
    x_hard = 200 + 200 * float(-hard_line[2] / hard_line[0])
    assert support[0, 0] and abs(x_soft - 203.9) < abs(x_hard - 203.9)


def test_D_alternative_modes_are_not_summed_into_a_compromise():
    base, valid, sigma, hw, _ = _fixture(); lattice = Lattice(); box = torch.tensor([[0., 0., 100., 100.]])
    logits = torch.full((1, 12, lattice.bins), -20.)
    # Both x=0 and x=10 alternatives exist. The correct x=0 mode is more reliable.
    offset = lattice.lines[:, 2]; theta0 = lattice.lines[:, 1].abs() < 1e-8
    correct = torch.where(theta0, (offset - 1.).abs(), torch.full_like(offset, 100.)).argmin()
    wrong = torch.where(theta0, (offset - .8).abs(), torch.full_like(offset, 100.)).argmin()
    logits[:, :, correct] = 20.; logits[:, :, wrong] = 10.
    decoded = decode_single_mode(logits, torch.ones_like(logits, dtype=torch.bool), lattice, box)
    selected = decoded["raw_line"]
    utility = torch.zeros(1, 12); utility[:, 0] = 1
    corrected, _ = single_mode_wls(base, valid, sigma, selected, utility, torch.ones(1), hw)
    # Existing lattice quantization leaves a sub-pixel offset, but the wrong
    # alternative is not added to the normal equations (v1 compromise: +2.5px).
    assert abs(float(corrected[0, 0, 0])) < 1.0
    assert abs(float(corrected[0, 0, 0]) - 2.5) > 1.0


def test_same_seed_common_and_utility_initialization_are_byte_identical():
    for seed in (1, 2, 3):
        direct = LocalLineFusionV2(arm="direct", common_seed=seed)
        hough = LocalLineFusionV2(arm="hough", common_seed=seed)
        assert state_sha(direct) == state_sha(hough)
        assert utility_sha(direct) == utility_sha(hough)


def test_center_is_bit_exact_and_model_is_finite():
    obs = {"features": torch.randn(1, 192, 24, 24), "content": torch.ones(1, 1, 24, 24),
           "box": torch.tensor([[0., 0., 100., 100.]]), "image_hw": torch.tensor([[100., 100.]]),
           "base_points": torch.rand(1, 9, 2) * 100, "point_valid": torch.ones(1, 9, dtype=torch.bool),
           "point_sigma": torch.full((1, 9), math.sqrt(2) / 2)}
    for arm in ("direct", "hough"):
        out = LocalLineFusionV2(arm=arm, common_seed=1)(obs)
        assert torch.equal(out["points"][:, 8], obs["base_points"][:, 8])
        assert torch.isfinite(out["points"]).all()
