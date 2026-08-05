"""Phase Q for the decoder reconciliation audit.

The audit claims three decoders saw one model output and that P2 is the
project's own deployment code rather than a convenient rewrite.  These tests
pin both claims, plus the parity that lets the P0 column be compared with the
existing results.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "scripts/stage0"
DEC = ROOT / "data/pallet/results/paper_s2_eval56/decoder_reconciliation"
RUNNER = STAGE0 / "paper_s2_eval56.py"
WRAPPER = STAGE0 / "decoder_paths.py"
DETECTOR = ROOT / "Deep_Object_Pose/common/detector.py"
RUN_LIVE = ROOT / "challenge/scripts/run_live.py"
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
def metrics():
    return pd.read_csv(DEC / "decoder_arm_metrics.csv")


@pytest.fixture(scope="module")
def wrapper_source():
    return WRAPPER.read_text("utf-8")


# 1
def test_head_is_at_or_after_the_expected_commit():
    log = subprocess.run(["git", "log", "--format=%H"], cwd=ROOT,
                         capture_output=True, text=True).stdout.split()
    assert "88d25c55be0a9ef9275781177b7eb248ba96f648" in log


# 2
def test_ep57_sha_unchanged(runner):
    assert hashlib.sha256(runner.EP57.read_bytes()).hexdigest() == runner.EP57_SHA


# 3, 4
def test_no_training_and_no_optimizer(wrapper_source, runner):
    audit = RUNNER.read_text("utf-8")
    audit = audit[audit.index("def dec_config"):]
    for name in ("optim", "backward", "zero_grad", "loss.", "torch.save"):
        assert name not in wrapper_source, name
        assert name not in audit, name
    called = {node.func.attr for node in ast.walk(ast.parse(audit))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not called & {"backward", "step", "zero_grad", "save"}


# 5, 6
@pytest.mark.parametrize("label,expected", [
    ("eval56", {"pnp_success": 50, "reproj_median_px": 11.5578,
                "corner_median_px": 7.2411, "near_median_px": 4.6755,
                "far_median_px": 11.4063, "t50": 45, "t100": 17,
                "nan_corner": 119, "frames": 56}),
    ("wood", {"pnp_success": 44, "reproj_median_px": 9.2839,
              "corner_median_px": 9.2255, "near_median_px": 6.7325,
              "far_median_px": 14.1798, "t50": 40, "t100": 36,
              "nan_corner": 51, "frames": 45}),
])
def test_p0_baseline_parity(metrics, label, expected):
    row = metrics[(metrics.set == label) & (metrics.arm == "B0")
                  & (metrics.path == "P0")].iloc[0]
    for key, want in expected.items():
        if isinstance(want, float):
            assert abs(float(row[key]) - want) <= 0.10, (label, key)
        else:
            assert int(row[key]) == want, (label, key)


# 7-11: each arm's P0 column reproduces the number the arm was judged on
@pytest.mark.parametrize("label,arm,column,want", [
    ("eval56", "E2", "far_median_px", 9.6422),
    ("eval56", "E2", "reproj_median_px", 11.7433),
    ("eval56", "S1", "reproj_median_px", 8.5191),
    ("eval56", "C1", "pnp_success", 55),
    ("eval56", "N2", "reproj_median_px", 11.6680),
    ("eval56", "N3", "pnp_success", 52),
    ("wood", "E2", "far_median_px", 11.8776),
    ("wood", "N2", "reproj_median_px", 8.8733),
])
def test_arm_p0_parity(metrics, label, arm, column, want):
    row = metrics[(metrics.set == label) & (metrics.arm == arm)
                  & (metrics.path == "P0")].iloc[0]
    if isinstance(want, float):
        assert abs(float(row[column]) - want) <= 0.10
    else:
        assert int(row[column]) == want


# 12
def test_p0_uses_the_frozen_mechanism_decoder():
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def dec_evaluate_frame"):body.index("def dec_p2_points")]
    assert 'MD.decode_all(belief, scale_x, scale_y, frame.gt_points)["D0"]' in body


# 13, 15, 26
def test_p1_and_p2_reuse_the_repository_functions(wrapper_source):
    assert "from filter_pr_camfacing import extract_keypoints_from_belief" in wrapper_source
    assert "from detector import ObjectDetector" in wrapper_source
    assert "from cuboid_pnp_solver import CuboidPNPSolver" in wrapper_source
    assert "ObjectDetector.find_object_poses(" in wrapper_source
    assert "extract_keypoints_from_belief(belief)" in wrapper_source
    # nothing that would amount to a private reimplementation
    for name in ("def find_objects", "def solve_pnp", "gaussian_filter(",
                 "peaks_binary", "np.average(j_values"):
        assert name not in wrapper_source, name


# 14
def test_deployment_entrypoints_all_reach_the_same_decoder():
    entrypoints = [ROOT / "scripts/dope/run_dope_live.py", RUN_LIVE,
                   ROOT / "challenge/25y_automatic_lifter-master"
                   "/25y_automatic_lifter-master/depth_cam/calib/dope_inference.py"]
    for path in entrypoints:
        text = path.read_text("utf-8")
        assert "from detector import ModelData, ObjectDetector" in text, path
        assert "find_object_poses" in text, path


# 16
def test_p2_direct_cache_parity():
    parity = pd.read_csv(DEC / "decoder_direct_cache_parity.csv")
    assert len(parity) >= 10
    assert float(parity.max_point_delta_px.max()) <= 1e-6
    assert float(parity.max_pose_delta.max()) <= 1e-6


# 17, 18, 19
def test_one_tensor_feeds_every_path(runner):
    frames = pd.read_parquet(DEC / "decoder_frames.parquet")
    assert frames.belief_sha.notna().all()
    assert frames.affinity_sha.notna().all()
    # a single row carries one belief and one affinity hash, and that row holds
    # the P0, P1 and P2 result computed from them
    for path in ("P0", "P1", "P2"):
        assert f"{path}_pose_success" in frames.columns
    assert frames.groupby(["set", "arm", "frame_id"]).belief_sha.nunique().max() == 1
    body = RUNNER.read_text("utf-8")
    assert "assert array.dtype == np.float32" in body


# 20, 21, 22
def test_resize_intrinsics_and_dimensions_are_shared(runner, wrapper_source):
    import decoder_paths as DP
    assert DP.INPUT_SIZE == 400 and DP.BELIEF_SIZE == 50
    assert DP.SCALE_FACTOR == DP.INPUT_SIZE // DP.BELIEF_SIZE
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def dec_forward_arm"):]
    assert body.count("FZ.preprocess_squash") >= 4
    K = np.array([[600.0, 0, 320.0], [0, 610.0, 240.0], [0, 0, 1.0]])
    scaled = DP.squash_intrinsics(K, 640, 480)
    assert np.isclose(scaled[0, 0], 600.0 * 400 / 640)
    assert np.isclose(scaled[1, 1], 610.0 * 400 / 480)


# 23
def test_centroid_convention_is_recorded_and_used(runner):
    trace = (DEC / "DECODER_PATH_TRACE.md").read_text("utf-8")
    assert "centroid" in trace.lower()
    detector = DETECTOR.read_text("utf-8")
    assert "points = obj[1] + [(obj[0][0] * scale_factor, obj[0][1] * scale_factor)]" in detector


# 24, 25
def test_p0_has_no_smoothing_and_no_affinity():
    frozen = (STAGE0 / "paper_s2_frozen_diagnostic.py").read_text("utf-8")
    stats = frozen[frozen.index("def heatmap_stats"):frozen.index("def build_cache")
                   if "def build_cache" in frozen else len(frozen)]
    for name in ("gaussian", "sigma", "affinit", "peaks_binary"):
        assert name not in stats.lower()[:6000], name


# 27
def test_p2_uses_affinity(wrapper_source):
    assert "def run_p2" in wrapper_source
    body = wrapper_source[wrapper_source.index("def run_p2"):]
    assert "aff = torch.from_numpy" in body
    assert "find_object_poses(vertex2, aff," in body


# 28, 29, 30
def test_production_selection_is_primary_and_gt_free(wrapper_source):
    """Comments and the docstring say "no GT is consulted"; strip them first so
    the statement about GT is not read as a use of GT."""
    import io
    import tokenize

    body = wrapper_source[wrapper_source.index("def production_selection"):]
    code = " ".join(
        token.string
        for token in tokenize.generate_tokens(io.StringIO(body).readline)
        if token.type not in (tokenize.COMMENT, tokenize.STRING))
    for name in ("gt", "GT", "oracle", "gt_points"):
        assert name not in code, name
    # tokenising splits the attribute access, so match the parts
    assert "GATES" in code and "evaluate_result" in code


# 31, 32
def test_gate_registry_points_at_the_recorded_definitions(runner):
    registry = json.loads((DEC / "decoder_gate_registry.json").read_text("utf-8"))
    for arm, entry in registry["registry"].items():
        assert entry["source"], arm
    assert (ROOT / "data/pallet/results/paper_s2_eval56"
            / "role_stage_static_gate.json").is_file()
    assert (ROOT / "data/pallet/results/paper_s2_eval56/pfdr"
            / "pfdr_gate.json").is_file()
    verdicts = pd.read_csv(DEC / "decoder_verdict_matrix.csv")
    p0 = verdicts[verdicts.path == "P0"]
    assert (p0.verdict == "REJECT").all(), p0[p0.verdict != "REJECT"]


# 33
def test_no_sealed_session_is_touched(wrapper_source):
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def dec_config"):]
    for token in SEALED:
        assert token not in body and token not in wrapper_source


# 34, 35
def test_no_source_data_written_and_no_new_result_root(wrapper_source):
    body = RUNNER.read_text("utf-8")
    body = body[body.index("def dec_config"):]
    for name in ("cv2.imwrite", "to_json", "shutil"):
        assert name not in body and name not in wrapper_source
    assert DEC.parent.name == "paper_s2_eval56"
    assert DEC.parent.parent == ROOT / "data/pallet/results"


# 36
def test_no_weights_tracked_or_staged():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in tracked if p.endswith(".pth")]
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    assert not [p for p in staged if p.endswith(".pth")]


# the two findings, pinned
def test_cuboid_model_is_the_yaw180_partner():
    sys.path.insert(0, str(ROOT / "Deep_Object_Pose/common"))
    sys.path.insert(0, str(ROOT / "challenge/scripts"))
    from cuboid import Cuboid3d
    import annotate_pnp as APNP
    import decoder_paths as DP

    w, d, h = 1.10, 1.30, 0.11
    camfacing = np.asarray(APNP.make_pallet_keypoints_3d(w, d, h), float)[:9]
    cuboid = np.asarray(Cuboid3d([w, h, d]).get_vertices(), float)
    assert np.abs(cuboid - camfacing @ DP.RY180.T).max() < 1e-12
    order = [int(np.argmin(np.linalg.norm(camfacing - cuboid[i], axis=1)))
             for i in range(9)]
    assert tuple(order) == DP.CUBOID_SWAP_MAP
    assert "swap_map = [5, 4, 7, 6, 1, 0, 3, 2, 8]" in RUN_LIVE.read_text("utf-8")


def test_deployment_decoder_builds_no_object_on_ep57(metrics):
    """The BLOCKED finding: pinned so it cannot be lost in a later edit."""
    for label in ("eval56", "wood"):
        row = metrics[(metrics.set == label) & (metrics.arm == "B0")
                      & (metrics.path == "P2")].iloc[0]
        assert int(row.P2_objects_total) == 0, label
        assert int(row.pnp_success) == 0, label
