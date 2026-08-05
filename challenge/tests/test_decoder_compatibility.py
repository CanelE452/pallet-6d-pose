"""Phase M for the decoder compatibility calibration.

The audit's claim is narrow: one field moved, it was chosen on N87 alone, and
it did not work. These tests pin the "one field" and the "N87 alone", and pin
the failure itself so a later edit cannot quietly turn it into a pass.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tokenize

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
CAL = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration")
RUNNER = STAGE0 / "paper_s2_eval56.py"
WRAPPER = STAGE0 / "decoder_paths.py"
SEALED = ("capturenight08", "capturenight09", "capturepallet07",
          "capturepallet09", "testset_full8_manifest", "handannot17")


@pytest.fixture(scope="module")
def runner():
    for path in (STAGE0, ROOT / "Deep_Object_Pose/common",
                 ROOT / "Deep_Object_Pose/train", ROOT / "challenge/scripts",
                 ROOT / "scripts/data_prep/eval"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("eval56_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cal_source():
    text = RUNNER.read_text("utf-8")
    return text[text.index("CAL_OUT = DEC_OUT"):]


@pytest.fixture(scope="module")
def cal_code(cal_source):
    """Comments and strings stripped, so prose about a thing is not a use."""
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(cal_source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING))


@pytest.fixture(scope="module")
def gate():
    return pd.read_csv(CAL / "sigma_calibration_metrics.csv")


# 1, 2
def test_head_and_checkpoint(runner):
    log = subprocess.run(["git", "log", "--format=%H"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert "9c329fcba5e5abddabb837dba7e8710de16f0e54" in log
    assert hashlib.sha256(runner.EP57.read_bytes()).hexdigest() == runner.EP57_SHA


# 3, 4, 29
def test_no_training_no_optimizer(cal_source, cal_code):
    for name in ("optim", "backward", "zero_grad", "requires_grad", "torch.save",
                 "load_state_dict("):
        assert name not in cal_code or name == "load_state_dict(", name
    called = {node.func.attr for node in ast.walk(ast.parse(cal_source))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"backward", "step", "zero_grad", "save"}


# 5
def test_d0_n87_baseline_parity():
    baseline = pd.read_csv(CAL / "n87_d0_baseline.csv")
    assert len(baseline) == 87
    assert int(baseline.pose_success.sum()) == 70
    median = float(np.nanmedian(pd.to_numeric(baseline.reproj_fixed_gt_px,
                                              errors="coerce")))
    assert abs(median - 23.161629) <= 1e-4, median


# 6
def test_sigma3_reproduces_the_blocked_state(gate):
    row = gate[gate.sigma_arm == "S30"].iloc[0]
    assert float(row.sigma) == 3.0
    assert int(row.objects_built) <= 12
    assert not bool(row.passed)


# 7, 26
def test_deployment_source_is_reused(cal_code):
    wrapper = WRAPPER.read_text("utf-8")
    assert "ObjectDetector.find_object_poses(" in wrapper
    assert "ObjectDetector" in cal_code and "find_objects" in cal_code
    for name in ("peaks_binary", "np.average(j_values", "def find_objects",
                 "def solve_pnp"):
        assert name not in wrapper and name not in cal_code, name


# 8
def test_direct_cache_parity_still_holds():
    parity = pd.read_csv(CAL.parent / "decoder_direct_cache_parity.csv")
    assert float(parity.max_point_delta_px.max()) <= 1e-6
    assert float(parity.max_pose_delta.max()) <= 1e-6


# 9-15: only sigma moves
def test_config_with_sigma_changes_nothing_else(runner):
    import decoder_paths as DP
    base, gates = runner.dec_config()
    assert (base.thresh_map, base.thresh_points, base.threshold,
            base.thresh_angle) == (0.30, 0.30, 0.30, 0.50)
    for sigma in runner.CAL_SIGMA_GRID.values():
        armed = DP.config_with_sigma(base, sigma)
        assert armed.sigma == sigma
        for field in ("thresh_map", "thresh_points", "threshold", "thresh_angle",
                      "mask_edges", "mask_faces", "vertex", "softmax"):
            assert getattr(armed, field) == getattr(base, field), field
    assert gates["min_detected_keypoints"] == 7
    assert gates["max_reproj_error_px"] == 8.0
    assert gates["z_min_m"] == 0.3 and gates["z_max_m"] == 5.0
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def cal_run"):]
    assert "BLOCKED: deployment thresholds are not the recorded ones" in body


# 16, 22
def test_sigma_grid_is_exactly_the_pre_registered_one(runner, gate):
    assert runner.CAL_SIGMA_GRID == {"S00": 0.0, "S05": 0.5, "S10": 1.0,
                                     "S15": 1.5, "S20": 2.0, "S25": 2.5,
                                     "S30": 3.0}
    assert set(gate.sigma_arm) == set(runner.CAL_SIGMA_GRID)
    stored = json.loads((CAL / "sigma_gate.json").read_text("utf-8"))
    assert stored["grid"] == runner.CAL_SIGMA_GRID


# 17, 18
def test_selection_used_n87_only_and_leakage_is_disclosed(runner):
    sweep = pd.read_csv(CAL / "sigma_calibration_frames.csv")
    assert set(sweep.set) == {"N87"}, set(sweep.set)
    assert not (CAL / "holdout_compatibility.csv").exists(), \
        "holdouts must stay unspent when no sigma is selected"
    decision = json.loads((CAL / "compatibility_final_decision.json").read_text("utf-8"))
    assert decision["holdouts_spent"] == {"eval56": False, "wood": False}
    # the overlap is real and must be stated, not asserted away
    assert decision["leakage_disclosure"]["n87_inter_eval56"] == 12
    assert decision["leakage_disclosure"]["n87_inter_wood"] == 0
    text = (CAL / "COMPATIBILITY_N87_GATE.md").read_text("utf-8")
    assert "12 frames" in text


def test_n87_membership_matches_the_disclosure(runner):
    frames = runner.cal_n87_frames()
    assert len(frames) == 87
    n87 = {f["fid"] for f in frames}
    eval56 = json.loads((runner.OUT / "eval56_manifest.json").read_text("utf-8"))
    wood = json.loads((runner.OUT / "wood_manifest.json").read_text("utf-8"))
    assert len(n87 & {f["frame_id"] for f in eval56["frames"]}) == 12
    assert len(n87 & {f["frame_id"] for f in wood["frames"]}) == 0


# 19, 20, 21
def test_a_single_sigma_would_have_been_used_everywhere(runner):
    selected = json.loads((CAL / "selected_sigma.json").read_text("utf-8"))
    assert selected.get("selected") is None
    assert selected["verdict"] == "CONFIG_ONLY_RESCUE = FAIL"
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def cal_run"):]
    # Phase E arms every holdout with the one selected value
    assert "single = {selected[\"sigma_arm\"]: selected[\"sigma\"]}" in body


# 23
def test_selection_rule_is_automatic(runner):
    source = __import__("inspect").getsource(runner.cal_select_sigma)
    assert "sort_values" in source
    assert "pnp_candidates" in source and "reproj_arm_median" in source
    for name in ("input(", "manual", "choose"):
        assert name not in source
    fake = pd.DataFrame([
        {"sigma_arm": "A", "sigma": 1.0, "passed": True, "pnp_candidates": 70,
         "reproj_arm_median": 20.0, "gate_pass_frames": 40},
        {"sigma_arm": "B", "sigma": 2.0, "passed": True, "pnp_candidates": 70,
         "reproj_arm_median": 19.0, "gate_pass_frames": 30},
    ])
    assert runner.cal_select_sigma(fake)["sigma_arm"] == "B"
    assert runner.cal_select_sigma(fake[fake.passed == False]) is None


# 24, 25
def test_production_selection_used_and_gt_free(cal_code):
    assert "production_selection" in cal_code
    body = WRAPPER.read_text("utf-8")
    body = body[body.index("def production_selection"):]
    code = " ".join(t.string for t in
                    tokenize.generate_tokens(io.StringIO(body).readline)
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))
    assert "oracle" not in code and "gt_points" not in code


# 27
def test_tensor_cache_is_float32(cal_source):
    assert 'astype(np.float32)' in cal_source
    assert 'assert array.dtype == np.float32' in RUNNER.read_text("utf-8")


# 28
def test_target_width_audit_uses_ideal_targets_only(runner):
    source = __import__("inspect").getsource(runner.cal_target_feasibility)
    assert "cal_ideal_belief" in source and "cal_ideal_affinity" in source
    assert "cv2.imread" not in source
    table = pd.read_csv(CAL / "target_width_feasibility.csv")
    assert set(table.target_sigma) == {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
    assert (table.deployment_sigma == 3.0).all()


# 30
def test_no_sealed_session(cal_source, runner):
    for token in SEALED:
        assert token not in cal_source or "SEALED" in cal_source
    for frame in runner.cal_n87_frames():
        assert not frame.get("is_final_test", False)
        for token in SEALED:
            assert token not in frame["image_path"]


# 31, 32
def test_no_source_write_and_no_new_root(cal_code):
    for name in ("imwrite", "shutil", "unlink"):
        assert name not in cal_code, name
    assert CAL.parent.name == "decoder_reconciliation"
    assert CAL.parent.parent.name == "paper_s2_eval56"


# 33
def test_no_weights_tracked_or_staged():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith(".pth")]
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in staged if p.endswith(".pth")]


# the findings, pinned
def test_no_sigma_passes_and_the_ceiling_is_not_the_smoothing(gate):
    assert not gate.passed.any()
    assert int(gate.centroid_survival.max()) == 74
    best = gate.loc[gate.centroid_survival.idxmax()]
    assert float(best.sigma) <= 0.5, "the best survival must be at no smoothing"


def test_ep57_is_about_half_the_width_of_the_challenge_models():
    widths = pd.read_csv(CAL / "blob_width_metrics.csv")
    ep57 = widths[widths.model == "M0_ep57"].sigma_from_half_max.median()
    for control in ("M1_challenge0123", "M2_challengenight"):
        other = widths[widths.model == control].sigma_from_half_max.median()
        assert other / ep57 > 1.7, (control, other, ep57)


def test_role_minima_differ_and_ep57_centroid_is_short():
    table = pd.read_csv(CAL / "target_width_feasibility.csv")
    centre = table[table.placement == "center"].set_index("target_sigma")
    assert not bool(centre.loc[2.0, "centroid_ok"])
    assert bool(centre.loc[2.0, "corner_peak_ok"])
    assert bool(centre.loc[2.5, "centroid_ok"])
    widths = pd.read_csv(CAL / "blob_width_metrics.csv")
    ep57 = widths[widths.model == "M0_ep57"]
    assert ep57[ep57.role == "centroid"].sigma_from_half_max.median() < 2.5
