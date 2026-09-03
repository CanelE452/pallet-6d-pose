"""Analytical tests for the exact oriented 3D IoU.

    python3 -m pytest scripts/paper/pose_metric_closure_v1/test_oriented_iou3d.py -q

The 90-degree test is the one that matters: on a rectangular pallet a wrong W/D
hypothesis must not score a perfect overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from challenge.evaluation_v2.oriented_iou3d import (  # noqa: E402
    box_volume,
    intersection_volume,
    oriented_iou_3d,
)

PLASTIC = (1.10, 0.11, 1.30)     # x, y(height), z  — registry
WOOD = (0.80, 0.14, 0.59)
IDENTITY = np.eye(3)
ORIGIN = np.zeros(3)


def rotation_about_y(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])


def test_same_box_is_one():
    assert oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                           IDENTITY, ORIGIN, PLASTIC) == pytest.approx(1.0, abs=1e-9)


def test_disjoint_boxes_are_zero():
    far = np.array([10.0, 0.0, 0.0])
    assert oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                           IDENTITY, far, PLASTIC) == pytest.approx(0.0, abs=1e-12)


def test_touching_boxes_are_zero():
    """면이 정확히 맞닿으면 부피가 0 이다."""

    touching = np.array([PLASTIC[0], 0.0, 0.0])
    assert oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                           IDENTITY, touching, PLASTIC) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
def test_known_axis_shift_matches_closed_form(fraction):
    """x 축으로 fraction 만큼 밀면 겹침 비율이 해석적으로 결정된다."""

    shift = np.array([PLASTIC[0] * fraction, 0.0, 0.0])
    overlap = 1.0 - fraction
    expected = overlap / (2.0 - overlap)
    assert oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                           IDENTITY, shift, PLASTIC) == pytest.approx(expected, rel=1e-6)


def test_half_shift_intersection_volume_is_half():
    shift = np.array([PLASTIC[0] * 0.5, 0.0, 0.0])
    inter = intersection_volume(IDENTITY, ORIGIN, PLASTIC, IDENTITY, shift, PLASTIC)
    assert inter == pytest.approx(box_volume(PLASTIC) * 0.5, rel=1e-6)


def test_180_degree_rotation_is_identical_box():
    """직육면체는 180도 회전에서 자기 자신이다 — 평가 등가류의 근거."""

    for extents in (PLASTIC, WOOD):
        assert oriented_iou_3d(rotation_about_y(180.0), ORIGIN, extents,
                               IDENTITY, ORIGIN, extents) == pytest.approx(1.0, abs=1e-6)


def test_90_degree_rotation_is_not_free_on_plastic():
    """가장 중요한 테스트 — 틀린 W/D 가설이 완전 겹침으로 통과하면 FAIL."""

    value = oriented_iou_3d(rotation_about_y(90.0), ORIGIN, PLASTIC,
                            IDENTITY, ORIGIN, PLASTIC)
    assert value < 1.0, "a 90-degree swap must not score a perfect overlap"
    # 1.10 x 1.30 footprint: 교집합 단면은 1.10 x 1.10, 합집합은 2*1.43 - 1.21
    expected = (1.10 * 1.10) / (2.0 * 1.10 * 1.30 - 1.10 * 1.10)
    assert value == pytest.approx(expected, rel=1e-6)
    assert 0.7 < value < 0.9


def test_90_degree_rotation_is_not_free_on_wood():
    value = oriented_iou_3d(rotation_about_y(90.0), ORIGIN, WOOD,
                            IDENTITY, ORIGIN, WOOD)
    assert value < 1.0
    expected = (0.59 * 0.59) / (2.0 * 0.80 * 0.59 - 0.59 * 0.59)
    assert value == pytest.approx(expected, rel=1e-6)


def test_wood_penalises_the_swap_more_than_plastic():
    """종횡비가 클수록 틀린 가설의 비용이 크다 — wood 1.356 대 plastic 1.182."""

    plastic = oriented_iou_3d(rotation_about_y(90.0), ORIGIN, PLASTIC,
                              IDENTITY, ORIGIN, PLASTIC)
    wood = oriented_iou_3d(rotation_about_y(90.0), ORIGIN, WOOD,
                           IDENTITY, ORIGIN, WOOD)
    assert wood < plastic


def test_square_footprint_makes_the_swap_free():
    """대조군: 정사각이면 90도가 실제로 공짜다.  plastic 이 그에 가깝다는 게 문제의 근원."""

    square = (1.10, 0.11, 1.10)
    assert oriented_iou_3d(rotation_about_y(90.0), ORIGIN, square,
                           IDENTITY, ORIGIN, square) == pytest.approx(1.0, abs=1e-6)


def test_is_not_the_axis_aligned_approximation():
    """축정렬 근사였다면 45도 회전에서 값이 크게 달라진다."""

    rotated = oriented_iou_3d(rotation_about_y(45.0), ORIGIN, PLASTIC,
                              IDENTITY, ORIGIN, PLASTIC)
    assert 0.0 < rotated < 1.0
    # 축정렬 bounding box 로 근사하면 두 상자가 같은 AABB 를 공유해 1.0 이 된다.
    assert rotated < 0.95, "an axis-aligned approximation would report ~1.0 here"


def test_symmetric_in_its_arguments():
    shift = np.array([0.3, 0.0, 0.2])
    forward = oriented_iou_3d(IDENTITY, ORIGIN, PLASTIC,
                              rotation_about_y(30.0), shift, PLASTIC)
    backward = oriented_iou_3d(rotation_about_y(30.0), shift, PLASTIC,
                               IDENTITY, ORIGIN, PLASTIC)
    assert forward == pytest.approx(backward, rel=1e-9)


def test_result_is_bounded():
    for degrees in (0.0, 15.0, 45.0, 90.0, 137.0, 180.0):
        value = oriented_iou_3d(rotation_about_y(degrees), ORIGIN, PLASTIC,
                                IDENTITY, ORIGIN, PLASTIC)
        assert 0.0 <= value <= 1.0


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        oriented_iou_3d(np.zeros((3, 3)), ORIGIN, PLASTIC, IDENTITY, ORIGIN, PLASTIC)
    with pytest.raises(ValueError):
        oriented_iou_3d(IDENTITY, ORIGIN, (0.0, 1.0, 1.0), IDENTITY, ORIGIN, PLASTIC)
