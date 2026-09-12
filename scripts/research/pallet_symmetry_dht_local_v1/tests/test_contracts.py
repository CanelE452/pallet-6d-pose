from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from symdht_local.constants import C2_YAW, IDENTITY
from symdht_local.data import OBSERVATION_KEYS
from symdht_local.export import affine_fixtures
from symdht_local.geometry import edge_graph_preserved, local_wls_fusion
from symdht_local.hough import Lattice, decode_modes, reliability
from symdht_local.model import LocalLineFusion
from symdht_local.objective import objective
from symdht_local.runner import score, score_point


def observation(batch=1):
    base = torch.tensor([
        [20., 20.], [80., 20.], [80., 80.], [20., 80.],
        [30., 30.], [70., 30.], [70., 70.], [30., 70.], [50., 50.],
    ]).expand(batch, -1, -1).clone()
    return {
        "features": torch.randn(batch, 192, 24, 24),
        "content": torch.ones(batch, 1, 24, 24),
        "box": torch.tensor([[0., 0., 100., 100.]]).expand(batch, -1).clone(),
        "image_hw": torch.tensor([[100., 100.]]).expand(batch, -1).clone(),
        "dims": torch.tensor([[1.2, .8, .15]]).expand(batch, -1).clone(),
        "base_points": base,
        "point_valid": torch.ones(batch, 9, dtype=torch.bool),
        "point_sigma": torch.full((batch, 9), .5),
    }


def test_explicit_c2_is_bijection_fixes_center_and_preserves_graph():
    assert edge_graph_preserved(list(IDENTITY))
    assert edge_graph_preserved(list(C2_YAW))
    assert C2_YAW[8] == 8


def test_affine_ramp_and_impulse_fixtures():
    result = affine_fixtures()
    assert result["PASS"], result


def test_uniform_logits_have_exact_zero_reliability_and_correction():
    lattice = Lattice()
    logits = torch.zeros(1, 12, lattice.bins)
    valid = torch.ones_like(logits, dtype=torch.bool)
    null = torch.zeros(1, 12)
    decoded = decode_modes(logits, valid, null, lattice, torch.tensor([[0., 0., 100., 100.]]))
    assert torch.equal(decoded["concentration"], torch.zeros_like(decoded["concentration"]))
    assert torch.equal(decoded["absolute_weight"], torch.zeros_like(decoded["absolute_weight"]))
    obs = observation()
    corrected, delta = local_wls_fusion(obs["base_points"], obs["point_valid"], obs["point_sigma"],
                                        decoded["raw_lines"], decoded["absolute_weight"],
                                        torch.tensor([.5]), obs["image_hw"])
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(corrected, obs["base_points"])


def test_null_dominant_is_near_zero_and_modes_keep_absolute_mass():
    lattice = Lattice()
    torch.manual_seed(4)
    logits = torch.randn(1, 12, lattice.bins)
    valid = torch.ones_like(logits, dtype=torch.bool)
    decoded = decode_modes(logits, valid, torch.full((1, 12), 100.), lattice,
                           torch.tensor([[0., 0., 100., 100.]]), modes=3)
    assert decoded["absolute_weight"].abs().max() < 1e-30
    assert torch.all(decoded["mode_mass"].sum(-1) <= 1 + 1e-6)
    assert torch.allclose(decoded["absolute_weight"],
                          decoded["reliability"][..., None] * decoded["mode_mass"])


def test_one_line_moves_only_normal_and_center_is_exact():
    obs = observation()
    lines = torch.zeros(1, 12, 1, 3)
    weights = torch.zeros(1, 12, 1)
    lines[:, 0, 0] = torch.tensor([1., 0., -10.])
    weights[:, 0, 0] = 1
    corrected, delta = local_wls_fusion(obs["base_points"], obs["point_valid"],
                                        torch.ones(1, 9), lines, weights,
                                        torch.ones(1), torch.tensor([[1000., 1000.]]))
    assert torch.equal(delta[0, :2, 1], torch.zeros(2))
    assert torch.equal(delta[0, 2:8], torch.zeros_like(delta[0, 2:8]))
    assert torch.equal(corrected[:, 8], obs["base_points"][:, 8])


def test_max_shift_is_one_percent_raw_diagonal():
    obs = observation()
    lines = torch.zeros(1, 12, 1, 3); weights = torch.zeros(1, 12, 1)
    lines[:, 0, 0] = torch.tensor([1., 0., -10000.]); weights[:, 0, 0] = 1
    _, delta = local_wls_fusion(obs["base_points"], obs["point_valid"],
                                torch.full((1, 9), 100.), lines, weights,
                                torch.ones(1), obs["image_hw"])
    limit = .01 * 100 * 2 ** .5
    assert float(torch.linalg.vector_norm(delta[0, 0])) <= limit + 1e-5


def test_corrected_point_loss_reaches_each_line_head_and_update_is_finite():
    torch.manual_seed(8)
    for arm in ("direct", "hough"):
        model = LocalLineFusion(arm=arm)
        obs = observation()
        before = [parameter.detach().clone() for parameter in model.line_head.parameters()]
        output = model(obs)
        target = obs["base_points"].clone(); target[:, :8, 0] += 2
        loss = torch.nn.functional.smooth_l1_loss(output["points"][:, :8], target[:, :8])
        loss.backward()
        line_grad = sum(float(parameter.grad.abs().sum()) for parameter in model.line_head.parameters()
                        if parameter.grad is not None)
        assert torch.isfinite(loss) and line_grad > 0, (arm, line_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.line_head.parameters()))
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_center_is_bit_exact_through_full_model():
    for arm in ("direct", "hough"):
        obs = observation()
        output = LocalLineFusion(arm=arm)(obs)
        assert torch.equal(output["points"][:, 8], obs["base_points"][:, 8])


def test_unmatched_nan_target_has_finite_training_objective():
    obs = observation()
    model = LocalLineFusion(arm="direct")
    output = model(obs)
    batch = {**obs, "target_points": torch.full((1, 9, 2), float("nan")),
             "target_valid": torch.zeros(1, 9, dtype=torch.bool),
             "symmetry_permutations": [torch.tensor([list(IDENTITY)])]}
    result = objective(output, batch, model,
                       {"point": 1., "line": .1, "null": .05, "no_harm": .5, "anchor": .05})
    assert torch.isfinite(result["loss"])


def test_gt_free_point_and_head_scoring_with_missing_supervision(tmp_path: Path):
    obs = observation()
    artifact = {key: value for key, value in obs.items()}
    obs_path = tmp_path / "observation.pt"; torch.save(artifact, obs_path)
    import hashlib
    obs_sha = hashlib.sha256(obs_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "symdht_local_export_v1", "role": "fixture",
        "records": [{"frame_id": "fixture", "observation": "observation.pt",
                     "observation_sha256": obs_sha, "supervision": "DOES_NOT_EXIST.pt",
                     "supervision_sha256": "0" * 64,
                     "symmetry_permutations": [list(IDENTITY)]}],
    }
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
    point_output = tmp_path / "point.json"
    score_point(argparse.Namespace(manifest=str(manifest_path), output=str(point_output)))
    assert json.loads(point_output.read_text())["GT_opened"] is False
    model = LocalLineFusion(arm="direct")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"schema": "symdht_local_checkpoint_v1", "model_config": model.config(),
                "state_dict": model.state_dict(), "seed": 1, "steps": 1}, checkpoint_path)
    head_output = tmp_path / "head.json"
    score(argparse.Namespace(manifest=str(manifest_path), checkpoint=str(checkpoint_path),
                             output=str(head_output), device="cpu"))
    assert json.loads(head_output.read_text())["GT_opened"] is False


def test_dht_is_line_aligned_voting_not_global_pooling():
    lattice = Lattice()
    feature = torch.zeros(1, 1, 24, 24); feature[0, 0, 7, 11] = 1
    vote, mass = lattice(feature, torch.ones(1, 1, 24, 24))
    assert vote.std() > 0
    assert mass.max() > 0 and (mass == 0).any()
