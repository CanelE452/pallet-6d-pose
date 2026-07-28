"""Core regression tests for the frozen PAPER_S2 ep57 diagnostic.

These tests protect conventions that can silently reverse the diagnosis:
camera yaw, the exact missing-point sentinel, and the operational clamped
7x7 local-softargmax/moment calculation.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0" / "paper_s2_frozen_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("paper_s2_frozen_diagnostic", SCRIPT)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DIAGNOSTIC)

TRAIN_DIR = ROOT / "Deep_Object_Pose" / "train"
sys.path.insert(0, str(TRAIN_DIR))
from diffpnp3d_loss import LocalSoftArgmax2D  # noqa: E402


def test_yaw_uses_local_forward_z_axis() -> None:
    angle_deg = 37.0
    angle = math.radians(angle_deg)
    rotation_y = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )
    assert DIAGNOSTIC.yaw_deg(rotation_y) == pytest_approx(angle_deg)


def test_only_exact_minus_one_pair_is_missing() -> None:
    assert not DIAGNOSTIC.point_valid((-1.0, -1.0))
    assert DIAGNOSTIC.point_valid((-1.0, 12.0))
    assert DIAGNOSTIC.point_valid((-2.0, -3.0))
    assert DIAGNOSTIC.point_valid((640.0, 480.0))
    assert not DIAGNOSTIC.point_valid((float("nan"), 1.0))


def test_boundary_local_softargmax_and_covariance_match_training_decoder() -> None:
    heatmap = np.full((50, 50), -8.0, dtype=np.float32)
    heatmap[0, 0] = 2.0
    heatmap[0, 1] = 1.25
    heatmap[1, 0] = 0.75
    heatmap[1, 1] = 0.25
    heatmap[2, 0] = -0.25

    scale_x, scale_y = 12.8, 9.6
    observed = DIAGNOSTIC.heatmap_stats(heatmap, scale_x, scale_y, None)

    decoder = LocalSoftArgmax2D(
        window=7,
        temperature=0.1,
        orig_size=(640, 480),
        belief_size=(50, 50),
    )
    coords, confidence = decoder(torch.from_numpy(heatmap)[None, None])

    assert observed["softargmax_x"] == pytest_approx(float(coords[0, 0, 0]), 1e-6)
    assert observed["softargmax_y"] == pytest_approx(float(coords[0, 0, 1]), 1e-6)
    assert observed["cov_grid_xx"] == pytest_approx(
        float(confidence["var_x"][0, 0]), 1e-6
    )
    assert observed["cov_grid_xy"] == pytest_approx(
        float(confidence["cov_xy"][0, 0]), 1e-6
    )
    assert observed["cov_grid_yy"] == pytest_approx(
        float(confidence["var_y"][0, 0]), 1e-6
    )
    assert observed["cov_px_xx"] == pytest_approx(
        observed["cov_grid_xx"] * scale_x**2, 1e-9
    )
    assert observed["cov_px_xy"] == pytest_approx(
        observed["cov_grid_xy"] * scale_x * scale_y, 1e-9
    )
    assert observed["cov_px_yy"] == pytest_approx(
        observed["cov_grid_yy"] * scale_y**2, 1e-9
    )


def pytest_approx(value: float, absolute_tolerance: float = 1e-12):
    """Keep this audit test independent of pytest import order/plugins."""
    import pytest

    return pytest.approx(value, abs=absolute_tolerance, rel=0.0)
