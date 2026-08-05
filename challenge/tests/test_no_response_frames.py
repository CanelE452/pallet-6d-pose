"""Tests for the no-response frame analysis.

The finding is that the 13 blocking frames are a global collapse, not a
centroid-specific one, and that an oracle centroid still reaches no pose. Both
are pinned here so a later edit cannot quietly revive the width hypothesis.
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
NRF = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/no_response_frames")
RUNNER = STAGE0 / "paper_s2_eval56.py"
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
def nrf_source():
    text = RUNNER.read_text("utf-8")
    return text[text.index("NRF_OUT = CAL_OUT"):]


@pytest.fixture(scope="module")
def nrf_code(nrf_source):
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(nrf_source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING))


@pytest.fixture(scope="module")
def taxonomy():
    return pd.read_csv(NRF / "nrf_taxonomy.csv", dtype={"frame_id": str})


@pytest.fixture(scope="module")
def counterfactuals():
    return pd.read_csv(NRF / "nrf_counterfactuals.csv", dtype={"frame_id": str})


def test_head_and_checkpoint(runner):
    log = subprocess.run(["git", "log", "--format=%H"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert "e373402e2841b6e3b4c1d2af043b3f4022e50118" in log
    assert hashlib.sha256(runner.EP57.read_bytes()).hexdigest() == runner.EP57_SHA


def test_no_training_no_optimizer(nrf_source, nrf_code):
    for name in ("optim", "backward", "zero_grad", "requires_grad", "torch.save"):
        assert name not in nrf_code, name
    called = {node.func.attr for node in ast.walk(ast.parse(nrf_source))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"backward", "step", "zero_grad", "save"}


def test_membership_is_exact_and_hashed(runner):
    members = json.loads((NRF / "nrf_membership.json").read_text("utf-8"))
    assert len(members["R0"]) == 9 and len(members["R1"]) == 4
    assert len(members["C0"]) == 13
    assert len(set(members["R0"]) | set(members["R1"])) == 13
    assert not (set(members["R0"]) | set(members["R1"])) & set(members["C0"])
    payload = json.dumps({k: members[k] for k in ("R0", "R1", "C0")}, sort_keys=True)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == \
        members["membership_sha256"]


def test_matching_rule_is_recorded_and_deterministic(runner):
    members = json.loads((NRF / "nrf_membership.json").read_text("utf-8"))
    assert "same domain" in members["matching_rule"]
    assert "no replacement" in members["matching_rule"]
    again = runner.nrf_membership()
    assert again["membership_sha256"] == members["membership_sha256"]


def test_controls_share_the_domain_of_their_pair(runner):
    members = json.loads((NRF / "nrf_membership.json").read_text("utf-8"))
    meta = {f["frame_id"]: f for f in runner.cal_n87_frames()}
    for dead_id, control_id in members["pairs"].items():
        assert meta[dead_id]["domain"] == meta[control_id]["domain"]


def test_thresholds_untouched(runner):
    config, gates = runner.dec_config()
    assert (config.thresh_map, config.thresh_points, config.threshold,
            config.thresh_angle, config.sigma) == (0.30, 0.30, 0.30, 0.50, 3)
    assert gates["min_detected_keypoints"] == 7


def test_only_channel_eight_is_ever_substituted(runner):
    source = __import__("inspect").getsource(runner.nrf_counterfactuals)
    assert "armed[8] = replacement" in source
    for forbidden in ("armed[0", "armed[:8", "armed[NEAR", "armed[FAR",
                      "affinity ="):
        assert forbidden not in source, forbidden


# the findings
def test_every_no_response_frame_is_a_global_collapse(taxonomy):
    dead = taxonomy[taxonomy.group != "C0"]
    assert len(dead) == 13
    assert (dead["class"] == "T2_GLOBAL_NO_RESPONSE").all()
    assert int((dead["class"] == "T1_CENTROID_ONLY_NO_RESPONSE").sum()) == 0
    # 12 of 13 have no corner above the gate at all
    assert int((dead.corners_above_030 == 0).sum()) >= 12


def test_corners_are_dead_too(taxonomy):
    dead = taxonomy[taxonomy.group != "C0"]
    control = taxonomy[taxonomy.group == "C0"]
    assert dead.corner_peak_median.median() < 0.20
    assert control.corner_peak_median.median() > 0.50
    assert dead.corner_gt_mass_5x5.median() < 0.05
    assert control.corner_gt_mass_5x5.median() > 0.20


def test_oracle_centroid_builds_objects_but_solves_nothing(counterfactuals):
    oracle = counterfactuals[counterfactuals.arm == "U0_gt_ideal_s25"]
    assert int(oracle.object_built.sum()) == 13
    assert int(oracle.pnp_success.sum()) == 0
    width = counterfactuals[counterfactuals.arm == "U1_width_only_s25"]
    assert int(width.object_built.sum()) == 0


def test_gate_selects_width_not_primary():
    gate = json.loads((NRF / "nrf_gate.json").read_text("utf-8"))
    assert gate["width_not_primary"] is True
    assert gate["role_specific_target_width"] is False
    assert gate["dual_bandwidth_head"] is False
    assert gate["target_defect"] is False
    assert gate["T1"] == 0 and gate["T2"] == 13


def test_no_response_population_is_truncated_not_dark():
    table = pd.read_csv(NRF / "nrf_domain_association.csv",
                        dtype={"frame_id": str, "pair": str})
    dead = table[table.role == "dead"].set_index("pair")
    control = table[table.role == "control"].set_index("pair")
    assert int(dead.is_truncated.sum()) >= 9
    assert int(control.is_truncated.sum()) <= 3
    inframe_dead = pd.to_numeric(dead.n_gt_inframe, errors="coerce")
    inframe_ctrl = pd.to_numeric(control.loc[dead.index, "n_gt_inframe"],
                                 errors="coerce")
    # no dead frame has MORE in-frame corners than its control; 3 pairs tie
    assert int((inframe_dead > inframe_ctrl).sum()) == 0
    assert int((inframe_dead < inframe_ctrl).sum()) >= 9
    # brightness does not separate them
    luma_dead = pd.to_numeric(dead.luma_p10, errors="coerce").median()
    luma_ctrl = pd.to_numeric(control.luma_p10, errors="coerce").median()
    assert abs(luma_dead - luma_ctrl) < 10


def test_confirmatory_set_is_fixed_as_eval44_clean_plus_wood():
    text = (NRF / "NRF_FINAL_DECISION.md").read_text("utf-8")
    assert "eval44-clean" in text and "wood" in text
    assert "12 frames" in text


def test_no_sealed_session_and_no_final_test(runner, nrf_source):
    for token in SEALED:
        assert token not in nrf_source or "SEALED" in nrf_source
    members = json.loads((NRF / "nrf_membership.json").read_text("utf-8"))
    meta = {f["frame_id"]: f for f in runner.cal_n87_frames()}
    for uid in members["R0"] + members["R1"] + members["C0"]:
        assert not meta[uid].get("is_final_test", False)


def test_no_weights_tracked_or_staged():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith(".pth")]
