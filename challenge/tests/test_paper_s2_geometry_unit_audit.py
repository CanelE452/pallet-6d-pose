"""Regression tests for the numerical PAPER_S2 geometry audit.

The legacy BPnP mismatch is intentionally not asserted as a passing gate here;
the canonical PAPER_S2 checkpoint uses the separate unrolled-GN path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage0" / "paper_s2" / "paper_s2_geometry_unit_audit.py"
SPEC = importlib.util.spec_from_file_location("paper_s2_geometry_unit_audit", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def test_paper_s2_local_softargmax_and_covariance_gate() -> None:
    _global_result, local_result = AUDIT._softargmax_cases()
    assert all(local_result["strict_gate"].values())
    assert local_result["operational_gate"]["pass"]
    assert local_result["covariance_scale_check"]["pass"]


def test_paper_s2_diffpnp3d_oracle_and_gradient_gate() -> None:
    result = AUDIT._diffpnp3d_audit()
    assert all(result["gate"].values())
    assert result["invalid_mask"]["loss"] == 0.0
    assert result["invalid_mask"]["gated_out"] == 1
    assert result["nan_guard_backward"]["gradient_finite_by_frame"] == [True, True]


def test_legacy_bpnp_mismatch_remains_explicit() -> None:
    result = AUDIT._legacy_bpnp_audit()
    assert result["gate"]["oracle_reprojection_le_1px"]
    assert result["gate"]["oracle_yaw_le_1deg"]
    # This documents why legacy BPnP and PAPER_S2 DiffPnP3D must not be
    # collapsed into one method name.  Remove only after the forward/backward
    # pair is made numerically consistent and the audit gate is updated.
    assert not result["gate"]["finite_difference_rel_le_1e_2"]
