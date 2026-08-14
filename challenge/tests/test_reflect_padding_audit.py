"""Tests for the reflect-padding audit.

Two things are pinned: that only the input border changed, and the finding
itself -- response recovers, localisation does not, and grey beats reflect.
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

import cv2
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
PAD = (ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
       / "compatibility_calibration/reflect_padding_audit")
RUNNER = STAGE0 / "paper_s2" / "paper_s2_eval56.py"
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
def pad_source():
    text = RUNNER.read_text("utf-8")
    return text[text.index("PAD_OUT = CAL_OUT"):]


@pytest.fixture(scope="module")
def pad_code(pad_source):
    return " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(pad_source).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING))


@pytest.fixture(scope="module")
def frames():
    return pd.read_csv(PAD / "padding_frames.csv", dtype={"frame_id": str})


@pytest.fixture(scope="module")
def gate():
    return json.loads((PAD / "padding_gate.json").read_text("utf-8"))


# 1, 2
def test_head_and_checkpoint(runner):
    log = subprocess.run(["git", "log", "--format=%H"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert "82ec98b2dad63b0cf1c0507e17a67905fac7fc20" in log
    assert hashlib.sha256(runner.EP57.read_bytes()).hexdigest() == runner.EP57_SHA


# 3, 4
def test_no_training_no_optimizer(pad_source, pad_code):
    for name in ("optim", "backward", "zero_grad", "requires_grad", "torch.save"):
        assert name not in pad_code, name
    called = {node.func.attr for node in ast.walk(ast.parse(pad_source))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"backward", "step", "zero_grad", "save"}


# 5, 6, 7
def test_membership_is_the_prior_one(runner):
    members = json.loads((PAD / "padding_membership.json").read_text("utf-8"))
    prior = json.loads((runner.NRF_OUT / "nrf_membership.json").read_text("utf-8"))
    assert members["control_sha256"] == prior["membership_sha256"]
    assert members["control_sha256"].startswith("9230daa96f515e11")
    assert sorted(members["D13"]) == sorted(prior["R0"] + prior["R1"])
    assert sorted(members["C13"]) == sorted(prior["C0"])
    assert len(members["D13"]) == 13 and len(members["C13"]) == 13


# 8, 9
def test_e44_is_a_true_set_difference_and_wood_is_disjoint(runner):
    members = json.loads((PAD / "padding_membership.json").read_text("utf-8"))
    eval56 = json.loads((runner.OUT / "eval56_manifest.json").read_text("utf-8"))
    n87 = {f["fid"] for f in runner.cal_n87_frames()}
    ids = {f["frame_id"] for f in eval56["frames"]}
    assert set(members["E44"]) == ids - n87
    assert len(members["E44"]) == 44
    assert len(members["n87_eval56_overlap"]) == 12
    assert members["D13_inter_E44"] == [] and members["C13_inter_E44"] == []
    assert members["D13_inter_W45"] == [] and members["C13_inter_W45"] == []


# 10, 11
def test_pad_ratio_comes_from_the_inference_source(runner):
    source = (ROOT / "challenge/scripts/dope_predict_mp4_pad.py").read_text("utf-8")
    assert '"--pad", type=int, default=100' in source
    assert runner.PAD_PIXELS == 100
    stored = json.loads((PAD / "padding_gate.json").read_text("utf-8"))
    assert stored["pad_pixels"] == 100
    assert "dope_predict_mp4_pad.py" in stored["source"]


# 12, 13, 14, 15
def test_arms_are_exactly_the_pre_registered_four(runner, frames):
    assert runner.PAD_ARMS == ("A0_original", "A1_reflect", "A2_replicate",
                               "A3_constant127")
    assert set(frames.arm) == set(runner.PAD_ARMS)
    assert runner.PAD_CONSTANT_VALUE == (127, 127, 127)
    source = __import__("inspect").getsource(runner.pad_apply)
    assert 'pad_frame(image, PAD_PIXELS, "reflect")' in source
    assert 'pad_frame(image, PAD_PIXELS, "replicate")' in source
    assert "cv2.BORDER_CONSTANT" in source


# 16
def test_all_arms_share_one_geometry(runner):
    for arm in runner.PAD_ARMS[1:]:
        geometry = runner.pad_geometry(arm, 640, 480)
        assert geometry == {"left": 100, "top": 100, "canvas_w": 840.0,
                            "canvas_h": 680.0}
    image = np.zeros((480, 640, 3), np.uint8)
    for arm in runner.PAD_ARMS:
        assert runner.pad_apply(image, arm).shape == (480, 640, 3)


# 17, 18, 19
def test_intrinsics_and_roundtrip(runner):
    K = np.array([[607.5, 0, 321.3], [0, 606.8, 241.7], [0, 0, 1.0]])
    padded = runner.pad_intrinsics(K, "A1_reflect", 640, 480)
    assert padded[0, 0] == K[0, 0] and padded[1, 1] == K[1, 1]
    assert padded[0, 2] == K[0, 2] + 100 and padded[1, 2] == K[1, 2] + 100
    rng = np.random.default_rng(1)
    points = np.stack([rng.uniform(-0.7, 0.7, 64), rng.uniform(-0.2, 0.2, 64),
                       rng.uniform(1.5, 4.0, 64)], 1)
    uv = (K @ points.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    uvp = (padded @ points.T).T
    uvp = uvp[:, :2] / uvp[:, 2:3]
    assert np.abs(uvp - (uv + 100)).max() <= 1e-6
    ok_a, r_a, t_a = cv2.solvePnP(points[:9], uv[:9], K, np.zeros((4, 1)),
                                  flags=cv2.SOLVEPNP_EPNP)
    ok_b, r_b, t_b = cv2.solvePnP(points[:9], uv[:9] + 100, padded,
                                  np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    assert ok_a and ok_b
    assert np.abs(r_a - r_b).max() <= 1e-6 and np.abs(t_a - t_b).max() <= 1e-6


# 20, 21
def test_no_gt_in_preprocessing(runner, pad_code):
    for name in ("pad_apply", "pad_geometry", "pad_intrinsics"):
        source = __import__("inspect").getsource(getattr(runner, name))
        code = " ".join(t.string for t in
                        tokenize.generate_tokens(io.StringIO(source).readline)
                        if t.type not in (tokenize.COMMENT, tokenize.STRING))
        for forbidden in ("gt", "GT", "oracle", "required_pad"):
            assert forbidden not in code, (name, forbidden)
    decode = __import__("inspect").getsource(runner.pad_decode)
    assert "[None] * MD.N_KP" in decode      # GT is never handed to the decoder


# 22, 23, 24
def test_preprocessing_and_dtype(runner, pad_source):
    assert "FZ.preprocess_squash(pad_apply(image, arm))" in pad_source
    assert pad_source.count("astype(np.float32)") >= 4
    forward = __import__("inspect").getsource(runner.pad_forward)
    assert forward.count("model(tensor)") == 1


# 26, 27, 28, 29
def test_decoder_config_untouched(runner, pad_source):
    config, gates = runner.dec_config()
    assert (config.sigma, config.thresh_map, config.thresh_points,
            config.threshold, config.thresh_angle) == (3, 0.30, 0.30, 0.30, 0.50)
    assert gates["min_detected_keypoints"] == 7
    assert "BLOCKED: deployment config is not the recorded one" in pad_source


# 30, 31, 34
def test_selection_used_development_only_and_holdouts_unspent(frames):
    assert set(frames.group) == {"D13", "C13"}
    selected = json.loads((PAD / "selected_padding.json").read_text("utf-8"))
    assert selected["selected"] is None
    assert not (PAD / "padding_confirmatory.csv").exists()


# 32
def test_selection_rule_is_automatic(runner):
    source = __import__("inspect").getsource(runner.pad_select)
    assert "sorted(" in source and "input(" not in source
    fake = [{"arm": "A2_replicate", "passed": True, "R4": 10, "d0_pnp": 9,
             "rescued_reproj_median": 20.0, "c13_worsened": 2, "new_gt50_frac": 0.1},
            {"arm": "A1_reflect", "passed": True, "R4": 10, "d0_pnp": 10,
             "rescued_reproj_median": 25.0, "c13_worsened": 1, "new_gt50_frac": 0.1}]
    assert runner.pad_select(fake)["arm"] == "A1_reflect"
    assert runner.pad_select([{**fake[0], "passed": False}]) is None


# 35, 36, 37, 38
def test_no_source_writes_no_new_root_no_weights(pad_code):
    for name in ("imwrite", "shutil", "rmtree"):
        assert name not in pad_code, name
    assert PAD.parent.name == "compatibility_calibration"
    assert PAD.parent.parent.name == "decoder_reconciliation"
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith(".pth")]


def test_no_sealed_session(runner, pad_source):
    for token in SEALED:
        assert token not in pad_source or "SEALED" in pad_source
    members = json.loads((PAD / "padding_membership.json").read_text("utf-8"))
    meta = {f["frame_id"]: f for f in runner.cal_n87_frames()}
    for uid in members["D13"] + members["C13"]:
        assert not meta[uid].get("is_final_test", False)


# the findings
def test_response_recovers_but_baseline_had_none(frames):
    base = frames[(frames.arm == "A0_original") & (frames.group == "D13")]
    assert int(base.R_centroid.sum()) == 0
    assert int(base.R4.sum()) == 0
    assert int(base.D0_pose_success.sum()) == 0
    best = frames[(frames.arm == "A3_constant127") & (frames.group == "D13")]
    assert int(best.R_centroid.sum()) >= 10
    assert int(best.R4.sum()) >= 8
    assert int(best.D0_pose_success.sum()) >= 8


def test_constant_grey_is_not_worse_than_reflect(gate):
    rows = {g["arm"]: g for g in gate["rows"]}
    assert rows["A3_constant127"]["R4"] >= rows["A1_reflect"]["R4"]
    assert rows["A3_constant127"]["centroid_recovered"] >= \
        rows["A1_reflect"]["centroid_recovered"]
    assert rows["A3_constant127"]["d0_pnp"] >= rows["A1_reflect"]["d0_pnp"]


def test_localisation_fails_on_every_arm(gate):
    for row in gate["rows"]:
        assert not row["passed"]
        assert row["conditions"]["new corner <=20px >= 60%"] is False
        assert row["conditions"]["rescued reproj <= 30px"] is False


def test_deployment_never_gets_past_smoothing(frames):
    dead = frames[frames.group == "D13"]
    assert (dead.P2_failure_stage.isin(("1_no_raw_response",
                                        "2_centroid_lost_in_smoothing"))).all()
    padded = dead[dead.arm != "A0_original"]
    assert int((padded.P2_failure_stage == "2_centroid_lost_in_smoothing").sum()) >= 20
