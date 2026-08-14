"""Phase D/J tests for the predicted-seed Gauss-Newton refinement."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "Deep_Object_Pose/train/diffpnp3d_loss.py"
RUNNER = ROOT / "scripts/stage0/paper_s2/paper_s2_predseed_diffpnp_screen.py"
OUT = ROOT / "data/pallet/results/paper_s2_predseed_diffpnp_screen"
EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def _dpl():
    for extra in (ROOT / "Deep_Object_Pose/train", ROOT / "Deep_Object_Pose/common"):
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    spec = importlib.util.spec_from_file_location("DPL", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene():
    """A cuboid, a pose, and its exact projections."""
    rng = np.random.default_rng(0)
    points = np.array([[x, y, z] for x in (-0.55, 0.55)
                       for y in (-0.6, 0.6) for z in (-0.07, 0.07)], float)
    points = np.vstack([points, points.mean(axis=0)])
    K = np.array([[614.18, 0, 329.28], [0, 614.31, 234.53], [0, 0, 1.0]])
    angle = 0.35
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]])
    t = np.array([0.05, -0.02, 3.2])
    camera = (R @ points.T).T + t
    uv = (K @ camera.T).T
    return points, K, R, t, uv[:, :2] / uv[:, 2:3]


# -- D1 identity ------------------------------------------------------------
def test_exact_pose_and_exact_points_barely_move() -> None:
    dpl = _dpl()
    points, K, R, t, observed = _scene()
    pose, health = dpl.refine_pose_from_predicted_seed(observed, points, K, R, t)
    assert np.isfinite(pose["R"]).all() and np.isfinite(pose["t"]).all()
    assert np.linalg.norm(pose["t"] - t) < 1e-6
    assert np.abs(pose["R"] - R).max() < 1e-6
    assert health["observed_after"] < 1e-6


# -- D2 perturbation --------------------------------------------------------
def test_a_perturbed_seed_converges_back() -> None:
    dpl = _dpl()
    points, K, R, t, observed = _scene()
    axis = np.array([0.02, -0.015, 0.01])
    angle = np.linalg.norm(axis)
    unit = axis / angle
    cross = np.array([[0, -unit[2], unit[1]], [unit[2], 0, -unit[0]],
                      [-unit[1], unit[0], 0]])
    perturb = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
    pose, health = dpl.refine_pose_from_predicted_seed(
        observed, points, K, perturb @ R, t + np.array([0.01, -0.008, 0.03]))
    assert health["observed_after"] < health["observed_before"]
    assert np.linalg.norm(pose["t"] - t) < np.linalg.norm(
        t + np.array([0.01, -0.008, 0.03]) - t)
    camera = (pose["R"] @ points.T).T + pose["t"]
    assert camera[:, 2].min() > 0


# -- D3 gradient ------------------------------------------------------------
def test_gradient_reaches_the_observed_coordinates() -> None:
    dpl = _dpl()
    points, K, R, t, observed = _scene()
    obs = torch.tensor(observed, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(points, dtype=torch.float64)[None]
    Km = torch.tensor(K, dtype=torch.float64)[None]
    rvec = dpl._rotation_to_rodrigues(torch.tensor(R, dtype=torch.float64))[None]
    tvec = torch.tensor(t, dtype=torch.float64)[None]
    uv, _ = dpl._project_batch(rvec, tvec, X, Km)
    ((uv - obs[None]) ** 2).sum().backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all()


# -- D4 GT isolation --------------------------------------------------------
def test_the_refiner_never_sees_ground_truth() -> None:
    import inspect

    dpl = _dpl()
    signature = inspect.signature(dpl.refine_pose_from_predicted_seed)
    for name in signature.parameters:
        assert "gt" not in name.lower(), name
    source = inspect.getsource(dpl.refine_pose_from_predicted_seed).lower()
    for banned in ("r_gt", "t_gt", "projected_gt", "gt_points", "gt_pose",
                   "json.load", "read_json"):
        assert banned not in source, banned


def test_the_seed_is_detached() -> None:
    import inspect

    source = inspect.getsource(_dpl().refine_pose_from_predicted_seed)
    assert ".detach()" in source


# -- solver configuration ---------------------------------------------------
def test_solver_constants_are_the_fixed_ones() -> None:
    dpl = _dpl()
    assert dpl.GN_STEPS == 4
    assert dpl.GN_DAMPING == 1e-3
    assert dpl.GN_DELTA_CLIP == 0.5
    assert dpl.GN_COND_MAX == 1e8


def test_a_step_that_raises_the_residual_is_rejected() -> None:
    import inspect

    source = inspect.getsource(_dpl().refine_pose_from_predicted_seed)
    assert "trial_res >= best_res" in source
    assert 'health["rejected"] += 1' in source


def test_guards_return_the_seed_unchanged() -> None:
    dpl = _dpl()
    points, K, R, t, observed = _scene()
    broken = observed.copy()
    broken[:] = np.nan
    pose, health = dpl.refine_pose_from_predicted_seed(broken, points, K, R, t)
    assert np.allclose(pose["R"], R) and np.allclose(pose["t"], t)


def test_no_optimizer_is_created_in_the_runner() -> None:
    source = RUNNER.read_text("utf-8")
    for banned in ("optim.", "Optimizer", ".backward()", ".step()", "requires_grad_(True)"):
        assert banned not in source, banned


def test_checkpoint_untouched_and_no_weights_staged() -> None:
    if EP57.is_file():
        assert hashlib.sha256(EP57.read_bytes()).hexdigest() == EP57_SHA
    out = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    assert not [line for line in out.splitlines() if line.endswith((".pth", ".pt"))]


# -- results ----------------------------------------------------------------
def _gate():
    path = OUT / "predseed_diffpnp_gate.json"
    if not path.is_file():
        pytest.skip("screen not run")
    return json.loads(path.read_text("utf-8"))


def test_same_seventy_frames_on_both_arms() -> None:
    gate = _gate()
    assert gate["summary"]["n_frames"] == 70
    provenance = json.loads(
        (OUT / "predseed_diffpnp_provenance.json").read_text("utf-8"))
    assert provenance["n_valid_frames"] == 70
    assert provenance["training_steps"] == 0
    assert provenance["optimizer_steps"] == 0
    assert provenance["baseline_gate"]["passed"] is True


def test_centroid_is_part_of_the_correspondence_set() -> None:
    path = OUT / "predseed_diffpnp_frames.csv"
    if not path.is_file():
        pytest.skip("screen not run")
    table = pd.read_csv(path)
    assert bool(table.centroid_used.all())
    assert int(table.n_correspondence.max()) == 9


def test_gate_has_all_twelve_pre_fixed_conditions() -> None:
    gate = _gate()
    assert len(gate["conditions"]) == 12
    assert gate["conditions"][0]["name"].startswith("1 GT reproj -5%")
    assert gate["conditions"][11]["name"].startswith("12 >=5 frames")


import pandas as pd  # noqa: E402
