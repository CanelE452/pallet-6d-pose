"""Contract tests for the 180-degree-equivalent pose metrics.

    python3 -m pytest scripts/paper/pose_metric_closure_v1/test_symmetry_aware_pose_metrics.py -q

These must pass **before** any model result is computed.  Tests 3 and 4 are the
reason the track exists: a wrong long/short axis choice has to stay visible as
roughly 90 degrees, on both pallets.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d  # noqa: E402
from symmetry_aware_pose_metrics import (  # noqa: E402
    AUC_INTEGRATION_POINTS,
    SYMMETRY_GROUP,
    cuboid_model_points,
    model_diameter_m,
    pose_auc,
    rotation_error_degrees,
    symmetry_aware_add_m,
    translation_components_m,
    translation_error_m,
    wrap180,
    yaw_error_degrees,
)

PLASTIC = (1.10, 0.11, 1.30)
WOOD = (0.80, 0.14, 0.59)
IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)


def ry(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


# ---------------------------------------------------------------- Test 1


def test_perfect_pose_is_zero_everywhere():
    points = cuboid_model_points(PLASTIC)
    assert rotation_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)
    assert yaw_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)
    assert translation_error_m(ORIGIN, ORIGIN) == pytest.approx(0.0, abs=1e-12)
    assert oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                           IDENTITY, ORIGIN, PLASTIC) == pytest.approx(1.0, abs=1e-9)
    assert symmetry_aware_add_m(points, IDENTITY, ORIGIN,
                                IDENTITY, ORIGIN) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- Test 2


@pytest.mark.parametrize("extents", [PLASTIC, WOOD])
def test_180_degree_yaw_is_free(extents):
    """180 도는 선언된 등가류다 — 모든 지표가 완벽으로 나와야 한다."""

    points = cuboid_model_points(extents)
    flipped = ry(180.0)
    assert rotation_error_degrees(flipped, IDENTITY) == pytest.approx(0.0, abs=1e-6)
    assert yaw_error_degrees(flipped, IDENTITY) == pytest.approx(0.0, abs=1e-6)
    assert oriented_iou_3d(flipped, ORIGIN, extents,
                           IDENTITY, ORIGIN, extents) == pytest.approx(1.0, abs=1e-6)
    assert symmetry_aware_add_m(points, flipped, ORIGIN,
                                IDENTITY, ORIGIN) == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------ Test 3 and 4


@pytest.mark.parametrize("extents,name", [(PLASTIC, "plastic"), (WOOD, "wood")])
def test_90_degree_yaw_stays_a_failure(extents, name):
    """가장 중요한 테스트 — 틀린 장단축 선택이 90 도 근처로 남아야 한다."""

    points = cuboid_model_points(extents)
    swapped = ry(90.0)

    assert yaw_error_degrees(swapped, IDENTITY) == pytest.approx(90.0, abs=1e-6), name
    assert rotation_error_degrees(swapped, IDENTITY) > 45.0, name

    iou = oriented_iou_3d(swapped, ORIGIN, extents, IDENTITY, ORIGIN, extents)
    assert iou < 1.0, f"{name}: a 90-degree swap must not score a perfect overlap"

    add = symmetry_aware_add_m(points, swapped, ORIGIN, IDENTITY, ORIGIN)
    assert add > 0.05, f"{name}: symmetry-aware ADD must penalise the swap, got {add}"


def test_yaw_error_is_bounded_to_ninety():
    for degrees in (0, 10, 45, 89, 90, 91, 135, 179, 180, 181, 269, 271, 359):
        value = yaw_error_degrees(ry(float(degrees)), IDENTITY)
        assert 0.0 <= value <= 90.0 + 1e-9, degrees


@pytest.mark.parametrize("degrees,expected", [
    (10.0, 10.0), (80.0, 80.0), (100.0, 80.0), (170.0, 10.0),
    (190.0, 10.0), (260.0, 80.0), (280.0, 80.0), (350.0, 10.0),
])
def test_yaw_folds_about_180(degrees, expected):
    assert yaw_error_degrees(ry(degrees), IDENTITY) == pytest.approx(expected, abs=1e-6)


def test_wrap180():
    assert wrap180(190.0) == pytest.approx(-170.0)
    assert wrap180(-190.0) == pytest.approx(170.0)
    assert wrap180(180.0) == pytest.approx(180.0)


# ---------------------------------------------------------------- Test 5


def test_known_translation_is_ten_centimetres():
    offset = np.array([0.1, 0.0, 0.0])
    assert translation_error_m(offset, ORIGIN) == pytest.approx(0.1, rel=1e-12)
    assert translation_error_m(offset, ORIGIN) * 100.0 == pytest.approx(10.0, rel=1e-12)


def test_translation_components_split_lateral_and_depth():
    offset = np.array([0.03, 0.04, 0.12])
    parts = translation_components_m(offset, ORIGIN)
    assert parts["lateral_m"] == pytest.approx(0.05, rel=1e-9)
    assert parts["depth_m"] == pytest.approx(0.12, rel=1e-9)
    assert parts["total_m"] == pytest.approx(np.linalg.norm(offset), rel=1e-9)


def test_translation_does_not_leak_into_rotation():
    points = cuboid_model_points(PLASTIC)
    offset = np.array([0.2, 0.0, 0.0])
    assert rotation_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)
    # 순수 평행이동이면 대칭 ADD 는 이동 거리와 같다.
    assert symmetry_aware_add_m(points, IDENTITY, offset, IDENTITY, ORIGIN) == \
        pytest.approx(0.2, rel=1e-9)


# ------------------------------------------------------- symmetry hygiene


def test_symmetry_group_is_exactly_identity_and_180():
    assert len(SYMMETRY_GROUP) == 2
    assert np.allclose(SYMMETRY_GROUP[0], np.eye(3))
    assert np.allclose(SYMMETRY_GROUP[1], np.diag([-1.0, 1.0, -1.0]))
    for member in SYMMETRY_GROUP:
        assert float(np.linalg.det(member)) == pytest.approx(1.0, abs=1e-9)


def test_group_aware_add_does_not_behave_like_unrestricted_nearest_neighbour():
    """무제한 최근접 ADD-S 는 정사각 90 도를 0 으로 만든다.  우리 정의는 아니어야 한다."""

    square = cuboid_model_points((1.10, 0.11, 1.10))
    swapped = ry(90.0)
    ours = symmetry_aware_add_m(square, swapped, ORIGIN, IDENTITY, ORIGIN)

    from challenge.evaluation_v2.pose_metrics import adds_error_m
    unrestricted = adds_error_m(square, swapped, ORIGIN, IDENTITY, ORIGIN)

    assert unrestricted == pytest.approx(0.0, abs=1e-9)
    assert ours > 0.5, "group-aware ADD must not forgive a 90-degree swap"


# ------------------------------------------------------------------ AUC


def test_pose_auc_analytic():
    diameter = model_diameter_m(cuboid_model_points(PLASTIC))
    assert pose_auc([0.0] * 10, diameter) == pytest.approx(1.0, rel=1e-9)
    assert pose_auc([diameter] * 10, diameter) == pytest.approx(0.0, abs=1e-12)
    assert pose_auc([0.0] * 5 + [diameter] * 5, diameter) == pytest.approx(0.5, rel=1e-9)


def test_pose_auc_is_monotone():
    diameter = model_diameter_m(cuboid_model_points(PLASTIC))
    assert pose_auc([0.01] * 10, diameter) > pose_auc([0.05] * 10, diameter)


def test_pose_auc_resolution_is_frozen():
    assert AUC_INTEGRATION_POINTS == 1001


def test_pose_auc_rejects_invalid_input():
    diameter = model_diameter_m(cuboid_model_points(PLASTIC))
    for bad in ([], [-0.1], [float("nan")]):
        with pytest.raises(ValueError):
            pose_auc(bad, diameter)
    with pytest.raises(ValueError):
        pose_auc([0.1], 0.0)


def test_model_diameter_is_the_space_diagonal():
    expected = float(np.linalg.norm(PLASTIC))
    assert model_diameter_m(cuboid_model_points(PLASTIC)) == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------- Test 6 and 7


def test_prediction_path_signature_takes_no_ground_truth():
    """Test 6 — GT 가 예측 selector 에 들어갈 수 없다는 것을 시그니처로 강제."""

    import inspect
    from challenge.evaluation_v2.pnp_selector import select_pnp_hypotheses

    parameters = set(inspect.signature(select_pnp_hypotheses).parameters)
    assert parameters == {"predicted_keypoints", "camera_intrinsics",
                          "physical_dimensions", "config"}
    forbidden = {"gt", "ground_truth", "target", "target_pose", "expected_hypothesis",
                 "axis_assignment", "reviewed_axis"}
    assert not (parameters & forbidden)


def test_oracle_mode_is_tagged_and_separate():
    """Test 7 — oracle 경로는 반드시 스스로 oracle 이라고 표시해야 한다."""

    from pose_evaluation_paths import evaluate_frame, ORACLE_TAG

    points = cuboid_model_points(PLASTIC)
    main = evaluate_frame(points, IDENTITY, ORIGIN, IDENTITY, ORIGIN,
                          PLASTIC, mode="main")
    oracle = evaluate_frame(points, IDENTITY, ORIGIN, IDENTITY, ORIGIN,
                            PLASTIC, mode="oracle")
    assert main["mode"] == "main"
    assert main["is_oracle"] is False
    assert oracle["mode"] == ORACLE_TAG
    assert oracle["is_oracle"] is True
