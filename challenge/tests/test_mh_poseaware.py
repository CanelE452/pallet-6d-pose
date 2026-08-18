"""Wiring tests for the pose-aware corner arm (PHASE 7).

The claim P1 rests on is that the pose term reaches the corner branch and nothing
else.  These check it on the real model rather than by reading the code, because
`SplitLate` shares a parent class with the arms that do share their late block.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stage0" / "multihead"))
sys.path.insert(0, str(ROOT / "Deep_Object_Pose" / "train"))

torch = pytest.importorskip("torch")
PA = pytest.importorskip("mh_poseaware")
MD = pytest.importorskip("mh_data")
CG = pytest.importorskip("mh_cigm")


def _needs_gpu():
    if not torch.cuda.is_available():
        pytest.skip("model construction needs the training device")


def test_pose_triple_is_self_consistent():
    """Projecting the object points with the GT pose must land on the label."""
    stems = [r["stem"] for r in MD.load_split() if r["split"] == "MH_DEV"][:4]
    for stem in stems:
        label = MD.read_label(stem)
        points = CG.object_points(label)
        K = CG.intrinsics(label)
        rotation, translation = CG.gt_pose(label)
        camera = (rotation @ points.T).T + translation
        uv = (K @ (camera / camera[:, 2:3]).T).T[:, :2]
        truth = np.asarray(label["objects"][0]["projected_cuboid"], float)[:8]
        assert np.abs(uv - truth).max() < 1e-3


def test_corner_pixels_matches_grid_to_pixels():
    """The differentiable path must use the same scale as the numpy path."""
    _needs_gpu()
    sampler, _ = PA.build_pose_loss()
    beliefs = torch.zeros(2, 9, MD.GRID, MD.GRID, device=PA.MH.DEV)
    beliefs[0, :, 10, 20] = 1.0
    beliefs[1, :, 30, 40] = 1.0
    resolution = [(640.0, 480.0), (1280.0, 720.0)]
    pixels = PA.corner_pixels(beliefs, resolution, sampler).cpu().numpy()
    for index, (width, height) in enumerate(resolution):
        grid = np.stack([pixels[index][:, 0] * MD.GRID / width,
                         pixels[index][:, 1] * MD.GRID / height], 1)
        expected = CG.grid_to_pixels(grid, width, height)
        assert np.allclose(pixels[index], expected, atol=1e-4)


def test_pose_gradient_does_not_reach_line():
    """L_pose must move the corner late block and leave the line one at zero."""
    _needs_gpu()
    import mh_screen as MS
    MS.deterministic()
    _, _, _, features = MS.lattice()
    sampler, pose_loss = PA.build_pose_loss()
    model, _ = PA.build_model(1)
    stems = [r["stem"] for r in MD.load_split() if r["split"] == "MH_TRAIN"][:2]
    pack = MD.load_pack(stems)
    out = model(pack["images"], features)
    targets = PA.pose_targets(stems, pack)
    if not bool(targets["mask"].any()):
        pytest.skip("no usable frame in this pair")
    pixels = PA.corner_pixels(out["beliefs"][-1], pack["resolution"], sampler)
    value, _ = pose_loss(pixels, targets["X"], targets["K"], targets["R_gt"],
                         targets["t_gt"], targets["diag"], targets["mask"])
    value.backward()

    line_grad = sum(float(p.grad.abs().sum()) for p in
                    model.line_late.parameters() if p.grad is not None)
    corner_grad = sum(float(p.grad.abs().sum()) for p in
                      model.corner_late.parameters() if p.grad is not None)
    assert line_grad == 0.0
    assert corner_grad > 0.0


def test_locked_constants():
    assert PA.LAMBDA_POSE_CANDIDATES == (1e-5, 3e-5, 1e-4, 3e-4)
    assert PA.LAMBDA_POSE_CANDIDATES_REJECTED_V1 == (0.01, 0.1, 1.0)
    assert PA.SOURCE_STEP == 18000
    assert PA.STEPS == 3000
    assert PA.MARKS == (250, 500, 1000, 2000, 3000)
    assert PA.GATE == {"t_gain_pct": 10.0, "R_degrade_pct": 3.0,
                       "geometry_gain_pct": 10.0, "line_degrade_pct": 0.5}
