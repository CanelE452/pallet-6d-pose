"""Integrity tests for the PGBC feasibility gates (no training happened)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data/pallet/results/paper_s2/paper_s2_pgbc_screen"
SCRIPT = ROOT / "scripts/stage0/paper_s2/paper_s2_pgbc_screen.py"
EP57 = ROOT / "weights/paper_s2_stageB/net_epoch_0057.pth"
EP57_SHA = "c0055fe7c4210f636705668c7c56dd98fb75414c403d5a5a3aa03359b08bc896"


def _module():
    spec = importlib.util.spec_from_file_location("PGBC", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate():
    path = OUT / "pgbc_gate.json"
    if not path.is_file():
        pytest.skip("gates not run")
    return json.loads(path.read_text("utf-8"))


def test_canonical_checkpoint_is_untouched() -> None:
    if not EP57.is_file():
        pytest.skip("checkpoint not present")
    assert hashlib.sha256(EP57.read_bytes()).hexdigest() == EP57_SHA


def test_no_weights_are_tracked_or_staged() -> None:
    for args in (["git", "ls-files", "weights/paper_s2_pgbc_screen"],
                 ["git", "diff", "--cached", "--name-only"]):
        out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout
        assert not [line for line in out.splitlines() if line.endswith((".pth", ".pt"))]


def test_baseline_reproduction_is_recorded_and_passed() -> None:
    gate = _gate()["baseline_gate"]
    assert gate["passed"] is True
    assert gate["strict_n"] == 87
    assert gate["gt2d_pose_success"] == 87
    assert gate["pred_pose_success"] == 70
    assert abs(gate["yaw_median_deg"] - 6.025) <= 0.1
    assert abs(gate["fixed_gt_reproj_median_px"] - 23.162) <= 0.25


def test_residual_amplitude_is_the_fixed_value() -> None:
    module = _module()
    assert module.AMPLITUDE == 0.25
    assert _gate()["amplitude"] == 0.25


def test_oracle_residual_is_bounded_and_three_by_three() -> None:
    module = _module()
    delta = module.oracle_residual((50, 50), np.array([25.0, 30.0]))
    assert np.abs(delta).max() <= module.AMPLITUDE + 1e-9
    assert int((delta > 0).sum()) == 9
    assert delta[29:32, 24:27].min() > 0


def test_gate_thresholds_are_the_pre_specified_ones() -> None:
    gate = _gate()
    assert gate["G0"]["threshold"]["share_of_far_corners_with_50pct_reduction"] == 0.80
    assert gate["G1"]["threshold"]["every_fold_auc_or_accuracy"] == 0.75
    assert gate["G1"]["threshold"]["gt_beats_wrong"] == 0.70
    assert gate["G2"]["threshold"] == {"error_reduction": 0.20, "bias_reduction": 0.20}


def test_every_gate_failed_so_nothing_was_implemented() -> None:
    gate = _gate()
    assert not gate["G0"]["passed"]
    assert not gate["G1"]["passed"]
    assert not gate["G2"]["passed"]
    for name in ("PGBC_OVERFIT_GATE.md", "PGBC_SESSION_CV.md", "PGBC_DIFFPNP_GATE.md"):
        assert "NOT RUN" in (OUT / name).read_text("utf-8")
    assert not (ROOT / "Deep_Object_Pose/common/pallet_graph_belief_corrector.py").exists()


def test_probe_control_sits_at_chance_by_construction() -> None:
    """Corner ID and dims are identical within a pair, so the control must be 0.5."""
    control = _gate()["G1"]["control_no_feature_auc"]
    assert len(control) == 3
    assert all(abs(value - 0.5) < 0.05 for value in control)


def test_g1_folds_are_session_disjoint() -> None:
    module = _module()
    sessions = ["a"] * 20 + ["b"] * 16 + ["c"] * 15 + ["d"] * 12 + ["e"] * 8
    folds = module.session_folds(sessions, 3)
    flat = [name for fold in folds for name in fold]
    assert len(flat) == len(set(flat)), "a session appears in two folds"
    assert set(flat) == {"a", "b", "c", "d", "e"}


def test_g2_never_feeds_gt_or_centroid_to_the_solver() -> None:
    source = SCRIPT.read_text("utf-8")
    body = source[source.index("def run_g2"):source.index("def g2_gate")]
    assert "gt_points" in body  # only as the error reference
    assert "held[8] = None" in body, "centroid must not be a PnP correspondence"
    assert "geometry.solve(held)" in body
    assert "geometry.solve(geometry.gt_points)" not in body


def test_only_the_primary_population_is_used() -> None:
    source = SCRIPT.read_text("utf-8")
    assert 'f["population"] == "primary"' in source


def test_out_of_grid_gt_is_flagged_not_dropped() -> None:
    import pandas as pd

    path = OUT / "pgbc_g0_residual_capacity.csv"
    if not path.is_file():
        pytest.skip("G0 not run")
    table = pd.read_csv(path)
    assert "gt_in_grid" in table.columns
    assert _gate()["G0"]["gt_outside_grid"] >= 0


def test_no_forbidden_branch_was_introduced() -> None:
    source = SCRIPT.read_text("utf-8").lower()
    for banned in ("semanticline", "maskhead", "votingoffset", "edge_branch"):
        assert banned not in source
