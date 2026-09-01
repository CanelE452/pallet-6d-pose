"""YOLO self-training 용 무차원 geometry consistency score 계약.

score 가 "무차원" 이라는 주장은 테스트로 강제한다.  해상도를 바꾸거나 물체를
멀리 놓아도 값이 거의 같아야 threshold 0.05 를 한 벌로 쓸 수 있다.

GT 누수 방지도 여기서 막는다.  filter 함수가 GT pose 나 GT axis assignment 를
인자로 받는 순간, "label-free adaptation" 주장이 무너진다.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "self_training_yolo"))

import pseudo_label_filters as plf  # noqa: E402

PLASTIC_XYZ = {"x": 1.1, "y": 0.11, "z": 1.3}
N_CORNERS = 8
# dataset 의 flip_idx 와 같은 페어링.  (0,1)(2,3)(4,5)(6,7) 스왑, 8 고정.
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6, 8]


def camera_matrix(fx=614.18, fy=614.31, cx=329.28, cy=234.53) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def perfect_projection(distance=4.0, scale=1.0, K=None):
    """registry 치수 그대로의 cuboid 를 실제로 투영해 만든 무결점 예측."""

    K = camera_matrix() if K is None else K
    keypoints_3d = plf.cuboid_keypoints_3d(
        width=PLASTIC_XYZ["x"], depth=PLASTIC_XYZ["z"], height=PLASTIC_XYZ["y"]
    )
    rvec = np.array([[0.35], [0.25], [0.10]], dtype=np.float64)
    tvec = np.array([[0.15], [0.05], [distance * scale]], dtype=np.float64)
    projected, _ = cv2.projectPoints(keypoints_3d, rvec, tvec, K, None)
    return projected.reshape(-1, 2), K


def all_valid() -> np.ndarray:
    return np.ones(9, dtype=bool)


# ── 1. 무결점 투영이면 세 score 모두 0 에 붙는다 ─────────────────────────

def test_perfect_projection_scores_are_near_zero() -> None:
    points, K = perfect_projection()
    scores = plf.geometry_scores(
        points, all_valid(), K, PLASTIC_XYZ,
        flip_keypoints_2d=points, flip_valid=all_valid(),
    )
    assert scores["s_reproj"] < 1e-3
    assert scores["s_remove"] < 1e-3
    assert scores["s_flip"] < 1e-9


# ── 2. keypoint 하나를 망가뜨리면 removal score 가 오른다 ────────────────

def test_corrupted_keypoint_raises_removal_score() -> None:
    points, K = perfect_projection()
    clean = plf.geometry_scores(points, all_valid(), K, PLASTIC_XYZ)

    corrupted = points.copy()
    diagonal = plf.projected_diagonal(points)
    corrupted[2] += np.array([0.35 * diagonal, -0.25 * diagonal])
    damaged = plf.geometry_scores(corrupted, all_valid(), K, PLASTIC_XYZ)

    assert damaged["s_remove"] > clean["s_remove"]
    # 실질적으로 갈라야 한다.  0.05 게이트가 의미를 가지려면 통과/기각이 갈릴 만큼.
    assert damaged["s_remove"] > 0.05
    assert clean["s_remove"] < 0.05


def test_removal_score_is_more_sensitive_than_plain_reprojection() -> None:
    """한 점만 망가진 경우, 그 점이 해에 기여하지 않는 removal 쪽이 더 민감하다."""

    points, K = perfect_projection()
    diagonal = plf.projected_diagonal(points)
    corrupted = points.copy()
    corrupted[5] += np.array([0.30 * diagonal, 0.20 * diagonal])
    scores = plf.geometry_scores(corrupted, all_valid(), K, PLASTIC_XYZ)
    assert scores["s_remove"] > scores["s_reproj"]


# ── 3~4. flip 대응이 맞으면 0, semantic pair 를 틀리면 커진다 ────────────

def test_correct_semantic_flip_gives_zero_flip_score() -> None:
    points, K = perfect_projection()
    scores = plf.geometry_scores(
        points, all_valid(), K, PLASTIC_XYZ,
        flip_keypoints_2d=points, flip_valid=all_valid(),
    )
    assert scores["s_flip"] < 1e-9


def test_wrong_semantic_pair_raises_flip_score() -> None:
    points, K = perfect_projection()
    mispaired = points[FLIP_IDX]  # 좌우 짝을 되돌리지 않은 상태를 흉내낸다
    scores = plf.geometry_scores(
        points, all_valid(), K, PLASTIC_XYZ,
        flip_keypoints_2d=mispaired, flip_valid=all_valid(),
    )
    assert scores["s_flip"] > 0.05


def test_flip_index_is_an_involution_matching_canonical_pairs() -> None:
    """dump_teacher_predictions 가 쓰는 페어링이 canonical_filters 와 같은가."""

    sys.path.insert(0, str(REPO_ROOT / "scripts" / "data_prep"))
    from canonical_filters import FLIP_PAIRS

    assert sorted(FLIP_IDX) == list(range(9))
    assert [FLIP_IDX[FLIP_IDX[i]] for i in range(9)] == list(range(9))
    assert FLIP_IDX[8] == 8
    derived = {tuple(sorted((a, b))) for a, b in FLIP_PAIRS}
    from_idx = {
        tuple(sorted((i, FLIP_IDX[i]))) for i in range(N_CORNERS) if FLIP_IDX[i] != i
    }
    assert derived == from_idx


# ── 5~6. 무차원성: 해상도와 거리를 바꿔도 값이 유지된다 ──────────────────

def test_score_is_invariant_to_image_resolution() -> None:
    base_points, base_K = perfect_projection()
    factor = 2.0
    scaled_K = camera_matrix(
        fx=614.18 * factor, fy=614.31 * factor,
        cx=329.28 * factor, cy=234.53 * factor,
    )
    scaled_points, _ = perfect_projection(K=scaled_K)

    diagonal = plf.projected_diagonal(base_points)
    base_points = base_points.copy()
    base_points[3] += np.array([0.12 * diagonal, 0.08 * diagonal])
    scaled_points = scaled_points.copy()
    scaled_points[3] += np.array([0.12 * diagonal * factor, 0.08 * diagonal * factor])

    base = plf.geometry_scores(base_points, all_valid(), base_K, PLASTIC_XYZ)
    scaled = plf.geometry_scores(scaled_points, all_valid(), scaled_K, PLASTIC_XYZ)
    assert base["projected_diagonal_px"] * factor == pytest.approx(
        scaled["projected_diagonal_px"], rel=1e-6
    )
    assert base["s_reproj"] == pytest.approx(scaled["s_reproj"], rel=0.02)
    assert base["s_remove"] == pytest.approx(scaled["s_remove"], rel=0.02)


def test_score_is_invariant_to_object_distance() -> None:
    near_points, K = perfect_projection(distance=3.0)
    far_points, _ = perfect_projection(distance=6.0)

    near_diagonal = plf.projected_diagonal(near_points)
    far_diagonal = plf.projected_diagonal(far_points)
    near_points = near_points.copy()
    far_points = far_points.copy()
    near_points[6] += np.array([0.15 * near_diagonal, 0.0])
    far_points[6] += np.array([0.15 * far_diagonal, 0.0])

    near = plf.geometry_scores(near_points, all_valid(), K, PLASTIC_XYZ)
    far = plf.geometry_scores(far_points, all_valid(), K, PLASTIC_XYZ)
    # 픽셀 스케일은 거리에 따라 2 배 가까이 달라진다.
    assert near["projected_diagonal_px"] > 1.9 * far["projected_diagonal_px"]
    # 무차원 score 는 그 2 배 변동을 흡수해야 한다.  완전히 같을 수는 없다 —
    # 가까울수록 원근 왜곡이 커서 같은 비율의 오차가 조금 다르게 퍼진다.
    # 요구 조건은 "0.05 게이트의 판정이 거리 때문에 뒤집히지 않는다" 이다.
    assert near["s_reproj"] == pytest.approx(far["s_reproj"], rel=0.25)
    assert near["s_remove"] == pytest.approx(far["s_remove"], rel=0.25)


# ── 7. GT 를 인자로 받지 않는다 ─────────────────────────────────────────

FORBIDDEN_ARGUMENT_TOKENS = (
    "gt", "ground_truth", "pose_transform", "axis_assignment",
    "canonical_pose", "target", "label",
)


@pytest.mark.parametrize(
    "function",
    [plf.geometry_scores, plf.registry_hypotheses, plf.cuboid_keypoints_3d,
     plf.projected_diagonal],
)
def test_filters_never_accept_ground_truth(function) -> None:
    for name in inspect.signature(function).parameters:
        lowered = name.lower()
        for token in FORBIDDEN_ARGUMENT_TOKENS:
            assert token not in lowered, f"{function.__name__} takes {name!r}"


def test_two_registry_hypotheses_are_scored_without_choosing_by_truth() -> None:
    hypotheses = plf.registry_hypotheses(PLASTIC_XYZ)
    assert len(hypotheses) == 2, "plastic 은 W/D 가 달라 hypothesis 가 둘이어야 한다"
    points, K = perfect_projection()
    scores = plf.geometry_scores(points, all_valid(), K, PLASTIC_XYZ)
    assert len(scores["hypotheses"]) == 2
    # 보고되는 값은 두 hypothesis 의 최소값이다 — 정답으로 고른 것이 아니다.
    assert scores["s_reproj"] == pytest.approx(
        min(item["s_reproj"] for item in scores["hypotheses"])
    )


def test_square_footprint_collapses_to_one_hypothesis() -> None:
    assert len(plf.registry_hypotheses({"x": 1.1, "y": 0.15, "z": 1.1})) == 1


# ── 8. paper-facing 산출물에 내부 용어 LOO 가 나오지 않는다 ──────────────

PAPER_FACING = (
    REPO_ROOT / "_docs" / "paper" / "EXPERIMENTS.md",
    REPO_ROOT / "_docs" / "paper" / "SELF_TRAINING_METHOD_LOCK.md",
)


def test_paper_facing_documents_do_not_use_the_internal_loo_term() -> None:
    import re

    pattern = re.compile(r"\bLOO\b")
    for path in PAPER_FACING:
        if not path.exists():
            continue
        offending = [
            f"{path.name}:{number}"
            for number, line in enumerate(path.read_text().splitlines(), 1)
            if pattern.search(line)
        ]
        assert not offending, f"paper-facing 문서에 LOO 가 남아 있다: {offending}"


def test_module_docstring_declares_the_reader_facing_names() -> None:
    text = plf.__doc__ or ""
    assert "single-keypoint-removal reprojection consistency" in text
    assert "horizontal-flip keypoint consistency" in text


def test_no_absolute_pixel_threshold_is_hardcoded_in_the_filters() -> None:
    """10px / 20px 같은 DOPE 시절 절대 threshold 가 스며들지 않았는지 AST 로 본다."""

    source = Path(plf.__file__).read_text()
    tree = ast.parse(source)

    # 점 개수 상한/하한 같은 정당한 정수는 이름을 가진 모듈 상수로만 허용한다.
    named = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    named.add(id(node.value))

    suspicious = []
    for node in ast.walk(tree):
        if id(node) in named:
            continue
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        # px threshold 는 관례적으로 5~100 사이의 값으로 나타난다.
        # 인덱스(0..8)나 비교용 점 개수(4,5)는 그 범위 밖이거나 상수로 승격돼 있다.
        if 6 <= value <= 100:
            suspicious.append((value, getattr(node, "lineno", None)))
    assert not suspicious, f"절대 px 상수로 보이는 값: {suspicious}"
