"""§13 — V2 per-keypoint filter 와 ambiguity score 를 계약으로 못 박는다.

V1 에서 3D 규약을 잘못 골라 필터가 1000 장 중 1 장만 통과시킨 사고가 있었다.
그래서 "완벽한 투영이면 잔차가 0" 부터 검사한다.

GT 를 인자에 두지 않는다는 것도 AST 로 강제한다 — 문자열 검사는 주석에 속는다
(`forbidden-token-tests-must-use-ast`).
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "self_training_yolo" / "v2" / "keypoint_scores.py"


def load_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "annotate"))
    spec = importlib.util.spec_from_file_location("v2_keypoint_scores", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["v2_keypoint_scores"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scores_module():
    return load_module()


# registry 는 x/y/z 로 준다.  y 가 height 다 (pseudo_label_filters.registry_hypotheses).
DIMENSIONS = {"x": 1.10, "y": 0.11, "z": 1.30}
CAMERA = np.array([[614.0, 0.0, 320.0], [0.0, 614.0, 240.0], [0.0, 0.0, 1.0]])


def perfect_projection(scores_module):
    """registry 규약 그대로의 3D 점을 실제로 투영해 만든 완벽한 관측."""

    from annotate_pnp import make_pallet_keypoints_3d

    points_3d = make_pallet_keypoints_3d(
        DIMENSIONS["x"], DIMENSIONS["z"], DIMENSIONS["y"])
    rvec = np.array([[0.35], [0.20], [0.10]])
    tvec = np.array([[0.05], [0.02], [3.0]])
    projected, _ = cv2.projectPoints(
        np.ascontiguousarray(points_3d[:9], dtype=np.float64), rvec, tvec, CAMERA, None)
    return projected.reshape(-1, 2)


@pytest.fixture(scope="module")
def clean(scores_module):
    keypoints = perfect_projection(scores_module)
    confidence = np.full(9, 0.99)
    return keypoints, confidence


# ── 완벽한 투영 ────────────────────────────────────────────────────────

def test_perfect_projection_gives_near_zero_removal_residuals(scores_module, clean):
    keypoints, confidence = clean
    result = scores_module.per_keypoint_scores(
        keypoints, confidence, CAMERA, DIMENSIONS,
        flip_keypoints_2d=keypoints, flip_conf=confidence)
    assert max(result["r_remove"]) < 1e-3, result["r_remove"]
    assert all(result["keep_corner"]), result["keep_corner"]


def test_perfect_flip_gives_near_zero_flip_residuals(scores_module, clean):
    keypoints, confidence = clean
    result = scores_module.per_keypoint_scores(
        keypoints, confidence, CAMERA, DIMENSIONS,
        flip_keypoints_2d=keypoints, flip_conf=confidence)
    assert max(result["r_flip"]) < 1e-9


# ── 한 코너만 망가뜨리면 그 코너만 떨어진다 ───────────────────────────

def test_a_single_corrupted_corner_is_isolated(scores_module, clean):
    keypoints, confidence = clean
    corrupted = keypoints.copy()
    corrupted[3] += np.array([70.0, -55.0])
    result = scores_module.per_keypoint_scores(
        corrupted, confidence, CAMERA, DIMENSIONS,
        flip_keypoints_2d=corrupted, flip_conf=confidence)
    assert not result["keep_corner"][3], "망가진 코너가 살아남았다"
    survivors = [flag for index, flag in enumerate(result["keep_corner"]) if index != 3]
    assert sum(survivors) >= 5, (
        f"한 코너 오염이 나머지를 무너뜨렸다: {result['keep_corner']}")


def test_a_semantic_swap_raises_the_flip_residual_of_the_affected_points(
        scores_module, clean):
    keypoints, confidence = clean
    swapped = keypoints.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    result = scores_module.per_keypoint_scores(
        keypoints, confidence, CAMERA, DIMENSIONS,
        flip_keypoints_2d=swapped, flip_conf=confidence)
    assert result["r_flip"][0] > result["r_flip"][2]
    assert result["r_flip"][1] > result["r_flip"][2]


# ── 무차원성 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("factor", [0.5, 2.0])
def test_scores_are_scale_invariant(scores_module, clean, factor):
    keypoints, confidence = clean
    base = scores_module.per_keypoint_scores(
        keypoints, confidence, CAMERA, DIMENSIONS,
        flip_keypoints_2d=keypoints, flip_conf=confidence)
    scaled_camera = CAMERA.copy()
    scaled_camera[:2] *= factor
    scaled = scores_module.per_keypoint_scores(
        keypoints * factor, confidence, scaled_camera, DIMENSIONS,
        flip_keypoints_2d=keypoints * factor, flip_conf=confidence)
    assert np.allclose(base["r_remove"], scaled["r_remove"], atol=1e-6)
    assert base["q"] == pytest.approx(scaled["q"], abs=1e-9)


# ── ambiguity q ────────────────────────────────────────────────────────

def test_q_is_invariant_under_the_90_degree_permutation(scores_module, clean):
    """min/max 라 90 도 순열이 두 변을 맞바꿔도 값이 그대로여야 한다."""

    keypoints, _ = clean
    permuted = keypoints[list(scores_module.YAW90)]
    assert scores_module.ambiguity_q(keypoints) == pytest.approx(
        scores_module.ambiguity_q(permuted), abs=1e-9)


def test_q_reaches_one_for_a_square_projection(scores_module):
    square = np.zeros((9, 2))
    square[0] = [100.0, 100.0]     # 근좌
    square[1] = [200.0, 100.0]     # 근우  -> width 100
    square[4] = [100.0, 200.0]     # 원좌  -> depth 100
    square[5] = [200.0, 200.0]     # 원우
    assert scores_module.ambiguity_q(square) == pytest.approx(1.0)


def test_q_is_small_for_an_elongated_projection(scores_module):
    elongated = np.zeros((9, 2))
    elongated[0] = [100.0, 100.0]
    elongated[1] = [400.0, 100.0]
    elongated[4] = [100.0, 130.0]
    elongated[5] = [400.0, 130.0]
    assert scores_module.ambiguity_q(elongated) == pytest.approx(0.1)


def test_q_uses_both_parallel_edges_not_just_one(scores_module):
    """한쪽 변만 쓰면 원근에서 90 도 불변이 깨진다 — 정의를 고정한다."""

    trapezoid = np.zeros((9, 2))
    trapezoid[0] = [100.0, 100.0]
    trapezoid[1] = [300.0, 100.0]      # 근면 width 200
    trapezoid[4] = [140.0, 200.0]
    trapezoid[5] = [260.0, 200.0]      # 원면 width 120  (원근 축소)
    width = 0.5 * (200.0 + 120.0)
    depth = 0.5 * (np.hypot(40.0, 100.0) + np.hypot(40.0, 100.0))
    assert scores_module.ambiguity_q(trapezoid) == pytest.approx(
        min(width, depth) / max(width, depth))


def test_ambiguous_view_uses_the_locked_threshold(scores_module, clean):
    keypoints, confidence = clean
    square = keypoints.copy()
    square[0] = np.array([100.0, 100.0])
    square[1] = np.array([200.0, 100.0])
    square[4] = np.array([100.0, 200.0])
    square[5] = np.array([200.0, 200.0])
    result = scores_module.per_keypoint_scores(
        square, confidence, CAMERA, DIMENSIONS, ambiguity_threshold=0.75)
    assert result["ambiguous_view"] is True


# ── visibility 벡터 ────────────────────────────────────────────────────

def test_ambiguity_masking_keeps_the_frame_and_the_centroid(scores_module):
    scores = {
        "keep_corner": [True] * 8,
        "keep_centroid": True,
        "ambiguous_view": True,
    }
    visibility = scores_module.visibility_vector(scores, ambiguity_aware=True)
    assert visibility[:8] == [0] * 8, "모호한 시점의 semantic corner 가 안 꺼졌다"
    assert visibility[8] == 2, "centroid 까지 꺼버리면 §7 위반"
    assert len(visibility) == 9


def test_ambiguity_masking_is_off_when_not_requested(scores_module):
    scores = {
        "keep_corner": [True] * 8,
        "keep_centroid": True,
        "ambiguous_view": True,
    }
    assert scores_module.visibility_vector(scores, ambiguity_aware=False)[:8] == [2] * 8


# ── GT 를 받지 않는다 (AST 로 강제) ───────────────────────────────────

FORBIDDEN_ARGUMENTS = (
    "gt", "ground_truth", "gt_xy", "gt_pose", "pose_transform",
    "canonical_pose", "axis_assignment", "target", "annotation",
)


def test_the_public_functions_take_no_ground_truth_argument():
    tree = ast.parse(MODULE_PATH.read_text())
    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        for argument in node.args.args + node.args.kwonlyargs:
            if any(token in argument.arg.lower() for token in FORBIDDEN_ARGUMENTS):
                offending.append(f"{node.name}({argument.arg})")
    assert not offending, f"GT 성 인자가 있다: {offending}"


def test_the_module_never_reads_the_evaluation_workspace():
    """평가 GT 경로를 이 모듈이 열면 안 된다."""

    tree = ast.parse(MODULE_PATH.read_text())
    literals = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    for text in literals:
        assert "evaluation" not in text, f"평가 경로 문자열이 있다: {text}"
        assert "real_gt" not in text, f"GT 경로 문자열이 있다: {text}"
