"""Unit tests for the theta-only Point+Line solver (PHASE 3).

These pin the two properties the whole diagnostic rests on -- that the line term
carries no `rho`, and that it vanishes at the truth -- plus the contract points
that a later edit could silently break.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "stage0" / "multihead"))

TH = pytest.importorskip("mh_theta")
DG = pytest.importorskip("mh_diagnose")
CG = pytest.importorskip("mh_cigm")


def _frame():
    """A synthetic frame: a cuboid, an exact pose, and the lines it induces."""
    import cv2
    model = np.array([[-0.5, -0.4, -0.1], [0.5, -0.4, -0.1],
                      [0.5, 0.4, -0.1], [-0.5, 0.4, -0.1],
                      [-0.5, -0.4, 0.1], [0.5, -0.4, 0.1],
                      [0.5, 0.4, 0.1], [-0.5, 0.4, 0.1]], float)
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    rotation, _ = cv2.Rodrigues(np.array([0.15, -0.25, 0.05]))
    translation = np.array([0.02, -0.03, 3.0])
    camera = (rotation @ model.T).T + translation
    projected = (K @ (camera / camera[:, 2:3]).T).T[:, :2]
    return model, K, rotation, translation, projected


def _lines_through(projected, edges):
    """Exact normalised pixel lines through each projected edge."""
    rows = []
    for a, b in edges:
        pa, pb = projected[a], projected[b]
        direction = pb - pa
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        rows.append([normal[0], normal[1], -float(normal @ pa)])
    return np.asarray(rows)


def test_line_normal_is_independent_of_rho():
    """`_line_in_pixels` splits into a rho-free normal and a rho-only offset."""
    first = DG._line_in_pixels(np.float64(0.7), np.float64(5.0), 640.0, 480.0)
    second = DG._line_in_pixels(np.float64(0.7), np.float64(-11.0), 640.0, 480.0)
    assert first[0] == pytest.approx(second[0], abs=1e-12)
    assert first[1] == pytest.approx(second[1], abs=1e-12)
    assert first[2] != pytest.approx(second[2])


def test_theta_residual_ignores_rho():
    """Shifting every line's offset must not move the theta residual at all."""
    import cv2
    model, K, rotation, translation, projected = _frame()
    edges = CG.EDGES
    lines = _lines_through(projected, edges)
    shifted = lines.copy()
    shifted[:, 2] += np.linspace(-40.0, 40.0, len(edges))

    rvec, _ = cv2.Rodrigues(rotation)
    params = np.concatenate([rvec.reshape(3), translation])
    corner_px = projected + 1.5            # a wrong point branch, on purpose
    use = np.ones(len(edges), bool)

    a = TH.theta_residual(params, model, K, corner_px, lines, edges, use, 1.0)
    b = TH.theta_residual(params, model, K, corner_px, shifted, edges, use, 1.0)
    assert np.allclose(a, b, atol=1e-9)


def test_theta_residual_vanishes_at_the_truth():
    """With the true pose and the true lines, only the point term survives."""
    import cv2
    model, K, rotation, translation, projected = _frame()
    edges = CG.EDGES
    lines = _lines_through(projected, edges)
    rvec, _ = cv2.Rodrigues(rotation)
    params = np.concatenate([rvec.reshape(3), translation])
    use = np.ones(len(edges), bool)

    residual = TH.theta_residual(params, model, K, projected, lines, edges,
                                 use, 1.0)
    assert np.abs(residual).max() < 1e-6


def test_theta_residual_is_undirected():
    """Swapping an edge's endpoints may only flip the sign."""
    import cv2
    model, K, rotation, translation, projected = _frame()
    edges = CG.EDGES
    flipped = [(b, a) for a, b in edges]
    lines = _lines_through(projected, edges)
    rvec, _ = cv2.Rodrigues(rotation)
    params = np.concatenate([rvec.reshape(3), translation + 0.05])
    use = np.ones(len(edges), bool)

    a = TH.theta_residual(params, model, K, projected, lines, edges, use, 1.0)
    b = TH.theta_residual(params, model, K, projected, lines, flipped, use, 1.0)
    point = len(model)
    assert np.allclose(a[:point], b[:point])
    assert np.allclose(a[point:], -b[point:], atol=1e-9)


def test_zero_weight_is_point_only():
    """lambda_theta = 0 must reproduce the point-only residual exactly."""
    import cv2
    model, K, rotation, translation, projected = _frame()
    edges = CG.EDGES
    lines = _lines_through(projected, edges)
    rvec, _ = cv2.Rodrigues(rotation)
    params = np.concatenate([rvec.reshape(3), translation])
    use = np.ones(len(edges), bool)
    corner_px = projected + 2.0

    joint = TH.theta_residual(params, model, K, corner_px, lines, edges, use,
                              0.0)
    point = TH.theta_residual(params, model, K, corner_px, lines, edges,
                              np.zeros(len(edges), bool), 1.0)
    assert np.allclose(joint[:len(model)], point)
    assert np.allclose(joint[len(model):], 0.0)


def test_grid_and_gate_are_locked_constants():
    """The screen is pre-registered; these must not drift between runs."""
    assert TH.LAMBDA_GRID == (0.03, 0.1, 0.3, 1.0, 3.0)
    assert TH.GATE == {"ALL_R_gain_pct": 5.0, "ALL_t_degrade_pct": 3.0,
                       "Vlt8_R_gain_pct": 10.0, "Vlt8_t_degrade_pct": 5.0,
                       "R_p90_degrade_pct": 5.0}
    assert TH.BOOTSTRAP == 10_000
    assert DG.HUBER_PX == 5.0


def test_confirmation_population_is_disjoint_from_d2():
    """D3 exists to be read once; overlapping D2 would defeat that."""
    import json
    out = ROOT / "data/pallet/results/paper_s2_multihead"
    d2 = set(json.loads((out / "d2_mh_dev512_manifest.json").read_text())["stems"])
    d3 = set(json.loads((out / "d3_mh_conf512_manifest.json").read_text())["stems"])
    assert len(d3) == 512
    assert not (d2 & d3)
