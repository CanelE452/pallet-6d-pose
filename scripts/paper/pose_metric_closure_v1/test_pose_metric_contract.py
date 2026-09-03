"""Analytical contract tests for the pose metrics, written before any pose number.

모델을 올리지 않고 추론하지 않는다.  합성 해석해만 쓴다.  목적은 하나 —
**pose metric 이 열렸을 때 그 숫자가 실제로 무엇을 재는지 미리 확정**하는 것.

    python3 -m pytest scripts/paper/pose_metric_closure_v1/test_pose_metric_contract.py -q

가장 중요한 테스트는 `test_ninety_degree_wd_swap_is_not_free` 다.  직사각 팔레트에서
틀린 W/D 가설이 0 오차로 통과하면 evaluator 가 selector 실패를 숨기는 것이고,
그러면 pose 표 전체가 무의미해진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.pose_metrics import (  # noqa: E402
    add_error_m,
    model_diameter_m,
    pose_auc,
    rotation_error_degrees,
    transformed_model_points,
    translation_error_m,
    yaw_error_degrees,
)

# registry: plastic_standard_110x130x11 — x = 1.10 m, y = 0.11 m (height), z = 1.30 m
WIDTH_X, HEIGHT_Y, DEPTH_Z = 1.10, 0.11, 1.30
IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)


def cuboid(width: float = WIDTH_X, height: float = HEIGHT_Y,
           depth: float = DEPTH_Z) -> np.ndarray:
    """8 corners of an axis-aligned box centred at the origin."""

    half = np.array([width, height, depth], dtype=np.float64) / 2.0
    return np.array([[sx * half[0], sy * half[1], sz * half[2]]
                     for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                     for sz in (-1.0, 1.0)], dtype=np.float64)


def rotation_about_y(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


# ---------------------------------------------------------------- identity


def test_identity_pose_is_zero_error():
    points = cuboid()
    assert rotation_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)
    assert yaw_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)
    assert translation_error_m(ORIGIN, ORIGIN) == pytest.approx(0.0, abs=1e-12)
    assert add_error_m(points, IDENTITY, ORIGIN, IDENTITY, ORIGIN) == pytest.approx(
        0.0, abs=1e-12)


# ------------------------------------------------------------- translation


@pytest.mark.parametrize("offset_cm", [1.0, 5.0, 23.7])
def test_known_translation_matches_expected_centimetres(offset_cm):
    offset = np.array([offset_cm / 100.0, 0.0, 0.0])
    assert translation_error_m(offset, ORIGIN) == pytest.approx(offset_cm / 100.0, rel=1e-12)
    # 순수 평행이동이면 ADD 는 이동 거리와 같다.
    assert add_error_m(cuboid(), IDENTITY, offset, IDENTITY, ORIGIN) == pytest.approx(
        offset_cm / 100.0, rel=1e-12)


def test_translation_does_not_leak_into_rotation():
    assert rotation_error_degrees(IDENTITY, IDENTITY) == pytest.approx(0.0, abs=1e-9)


# -------------------------------------------------------------------- yaw


@pytest.mark.parametrize("degrees", [1.0, 10.0, 45.0, 89.0])
def test_known_yaw_is_reported_as_that_yaw(degrees):
    predicted = rotation_about_y(degrees)
    assert yaw_error_degrees(predicted, IDENTITY) == pytest.approx(degrees, abs=1e-6)
    assert rotation_error_degrees(predicted, IDENTITY) == pytest.approx(degrees, abs=1e-6)


def test_yaw_is_measured_about_the_height_axis():
    """registry: y 가 높이축이다.  x 축 회전은 yaw 로 세지 않는다."""

    angle = np.radians(30.0)
    cos, sin = np.cos(angle), np.sin(angle)
    roll_about_x = np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    assert yaw_error_degrees(roll_about_x, IDENTITY) == pytest.approx(0.0, abs=1e-6)
    assert rotation_error_degrees(roll_about_x, IDENTITY) == pytest.approx(30.0, abs=1e-6)


# ------------------------------------------- the test this track exists for


def test_ninety_degree_wd_swap_is_not_free():
    """직사각 팔레트에서 틀린 W/D 가설은 반드시 큰 오차를 내야 한다.

    이게 0 이면 evaluator 가 selector 실패를 흡수해버리고, 그러면
    "우리 pose 가 정확하다" 는 문장이 selector 가 아니라 metric 의 관대함에서 나온다.
    """

    points = cuboid()
    swapped = rotation_about_y(90.0)

    assert rotation_error_degrees(swapped, IDENTITY) == pytest.approx(90.0, abs=1e-6)
    assert yaw_error_degrees(swapped, IDENTITY) == pytest.approx(90.0, abs=1e-6)

    error = add_error_m(points, swapped, ORIGIN, IDENTITY, ORIGIN)
    assert error > 0.0, "90-degree W/D swap must not be free"

    # 1.10 x 1.30 에서 코너별 변위는 아래로 해석적으로 결정된다.
    expected = float(np.linalg.norm(
        transformed_model_points(points, swapped, ORIGIN) - points, axis=1).mean())
    assert error == pytest.approx(expected, rel=1e-12)

    # 관측된 실패 규모와 같은 자릿수인지 — 진단에서 rotation 85.3도, ADD-S 정규화 0.709
    assert error > 0.05, f"swap error {error:.4f} m is implausibly small"


def test_add_is_index_wise_so_even_a_square_swap_is_not_free():
    """ADD 는 대응점 기준이라 정사각이어도 90도 swap 이 공짜가 아니다.

    처음엔 정사각이면 0 이 나올 거라 가정했는데 1.10 m 가 나왔다.  코드가 맞다 —
    `add_error_m` 은 코너 i 를 코너 i 와 비교하므로, 코너 집합이 자기 자신으로
    사상돼도 **인덱스가 돌면** 벌점이 붙는다.
    """

    square = cuboid(width=1.10, depth=1.10)
    swapped = rotation_about_y(90.0)
    assert add_error_m(square, swapped, ORIGIN, IDENTITY, ORIGIN) > 0.5


def test_unrestricted_adds_forgives_the_square_swap():
    """반대로 무제한 ADD-S 는 그 swap 을 용서한다 — symmetry 계약이 필요한 이유.

    `adds_error_m` 의 docstring 이 스스로 "not paper symmetry policy" 라고 적어둔 것도
    같은 이유다.  90도를 근거 없이 symmetry 집합에 넣으면 selector 실패가 지표에서
    사라지고, forklift 관점에서 fork-entry 면이 바뀐 pose 가 정답으로 통과한다.
    """

    from challenge.evaluation_v2.pose_metrics import adds_error_m

    square = cuboid(width=1.10, depth=1.10)
    swapped = rotation_about_y(90.0)
    assert adds_error_m(square, swapped, ORIGIN, IDENTITY, ORIGIN) == pytest.approx(
        0.0, abs=1e-9)

    # 직사각(실제 plastic)에서는 무제한 ADD-S 조차 공짜가 아니다.
    rectangular = cuboid()
    assert adds_error_m(rectangular, swapped, ORIGIN, IDENTITY, ORIGIN) > 0.05


def test_set_distance_shrinks_as_the_footprint_approaches_square():
    """가로세로가 비슷해질수록 잘못된 가설의 기하적 비용이 줄어든다.

    plastic 은 1.10 x 1.30 이라 비 1.182 — 공짜는 아니지만 작다.  selector 가
    어려운 이유가 metric 이 아니라 **물체 형상**에 있다는 것을 기록한다.
    """

    from challenge.evaluation_v2.pose_metrics import adds_error_m

    swapped = rotation_about_y(90.0)
    errors = [adds_error_m(cuboid(width=1.10, depth=depth), swapped, ORIGIN,
                           IDENTITY, ORIGIN)
              for depth in (1.10, 1.15, 1.30, 1.80)]
    assert errors == sorted(errors), "cost must grow with the aspect difference"
    assert errors[0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------- AUC


def test_pose_auc_on_analytic_sequences():
    # 모든 오차가 0 이면 곡선이 항상 1 이라 면적도 1.
    assert pose_auc([0.0] * 10, max_fraction=0.1) == pytest.approx(1.0, rel=1e-9)
    # 모든 오차가 임계 상한을 넘으면 0.
    assert pose_auc([1.0] * 10, max_fraction=0.1) == pytest.approx(0.0, abs=1e-12)
    # 절반만 0 이면 0.5.
    assert pose_auc([0.0] * 5 + [1.0] * 5, max_fraction=0.1) == pytest.approx(
        0.5, rel=1e-9)


def test_pose_auc_is_monotone_in_error():
    better = pose_auc([0.01] * 10, max_fraction=0.1)
    worse = pose_auc([0.05] * 10, max_fraction=0.1)
    assert better > worse


def test_pose_auc_rejects_invalid_input():
    for bad in ([], [-0.1], [float("nan")]):
        with pytest.raises(ValueError):
            pose_auc(bad, max_fraction=0.1)


def test_model_diameter_is_the_space_diagonal():
    expected = float(np.linalg.norm([WIDTH_X, HEIGHT_Y, DEPTH_Z]))
    assert model_diameter_m(cuboid()) == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------ gate cannot be bypassed


def test_blocked_gate_returns_nulls_not_numbers():
    from challenge.evaluation_v2.pose_metrics import PoseMetricGate, blocked_pose_metrics

    gate = PoseMetricGate(
        canonical_migration_status="NOT_PASS",
        selector_status="FAIL",
        symmetry_status="UNREVIEWED",
        final_manifest_status="NOT_FROZEN",
        passed=False,
        blocked_reasons=("POSE_METRICS_BLOCKED_NO_RELIABLE_AXIS_SELECTOR",),
    )
    blocked = blocked_pose_metrics(gate)
    numeric = [(key, value) for key, value in blocked.items()
               if isinstance(value, (int, float)) and not isinstance(value, bool)]
    assert not numeric, f"blocked pose metrics must be null, got {numeric}"
