"""Executable contract for canonical and camera-facing pallet frames."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.annotate import pallet_geometry as geometry


EXPECTED_PERMUTATIONS = {
    geometry.AxisAssignment.YAW_0: (0, 1, 2, 3, 4, 5, 6, 7, 8),
    geometry.AxisAssignment.YAW_90: (1, 5, 6, 2, 0, 4, 7, 3, 8),
    geometry.AxisAssignment.YAW_180: (5, 4, 7, 6, 1, 0, 3, 2, 8),
    geometry.AxisAssignment.YAW_270: (4, 0, 3, 7, 5, 1, 2, 6, 8),
}


def test_named_dimensions_remove_legacy_tuple_ambiguity() -> None:
    physical = geometry.canonical_dimensions()
    assert isinstance(physical, geometry.PhysicalDimensionsXYZ)
    assert physical.as_dict() == {"x": 1.10, "y": 0.11, "z": 1.30}

    camera = geometry.camera_facing_dimensions(
        geometry.AxisAssignment.YAW_90)
    assert isinstance(camera, geometry.CameraFacingDimensionsWHD)
    assert camera.as_dict() == {
        "width": 1.30,
        "height": 0.11,
        "depth": 1.10,
    }
    # The only positional legacy adapter makes its W,D,H order explicit.
    assert camera.as_legacy_wdh_tuple() == (1.30, 1.10, 0.11)
    with pytest.raises(TypeError):
        geometry.PhysicalDimensionsXYZ(1.10, 0.11, 1.30)


def test_canonical_centroid_origin_y_down_right_handed_contract() -> None:
    points = geometry.canonical_keypoints_3d()
    assert points.shape == (9, 3)
    np.testing.assert_array_equal(points[8], np.zeros(3))

    # camera_dynamic_0123_v4: 0,1,4,5 are image/object top (-Y), while
    # 2,3,6,7 are bottom (+Y).  With +X right and +Z forward this is a
    # right-handed OpenCV-style basis: X cross Y = Z.
    np.testing.assert_array_equal(
        points[[0, 1, 4, 5], 1], np.full(4, -0.11 / 2.0))
    np.testing.assert_array_equal(
        points[[2, 3, 6, 7], 1], np.full(4, +0.11 / 2.0))
    np.testing.assert_array_equal(
        np.cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        [0.0, 0.0, 1.0],
    )


def test_all_four_signed_yaws_are_exact_proper_rotations() -> None:
    expected = {
        geometry.AxisAssignment.YAW_0: np.eye(3),
        geometry.AxisAssignment.YAW_90: np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]),
        geometry.AxisAssignment.YAW_180: np.array([
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]),
        geometry.AxisAssignment.YAW_270: np.array([
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]),
    }
    for assignment, expected_rotation in expected.items():
        rotation = geometry.canonical_to_camera_facing_transform(assignment)
        np.testing.assert_array_equal(rotation, expected_rotation)
        np.testing.assert_array_equal(rotation.T @ rotation, np.eye(3))
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=0.0)

    # The sign convention is fixed explicitly: at +90, +X maps to -Z and
    # +Z maps to +X.  It is not inferred from an LR swap.
    yaw_90 = expected[geometry.AxisAssignment.YAW_90]
    np.testing.assert_array_equal(yaw_90 @ [1.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    np.testing.assert_array_equal(yaw_90 @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])


@pytest.mark.parametrize("assignment", list(geometry.AxisAssignment))
def test_permutation_is_exact_coordinate_matching(
        assignment: geometry.AxisAssignment) -> None:
    rotation = geometry.canonical_to_camera_facing_transform(assignment)
    permutation = geometry.canonical_to_camera_facing_keypoint_permutation(
        assignment)
    assert permutation == EXPECTED_PERMUTATIONS[assignment]
    assert sorted(permutation) == list(range(9))
    assert permutation[8] == 8

    canonical_reordered = geometry.canonical_keypoints_3d()[list(permutation)]
    transformed = (rotation @ canonical_reordered.T).T
    np.testing.assert_array_equal(
        transformed, geometry.camera_facing_keypoints_3d(assignment))


def test_dimensions_determine_only_yaw_parity() -> None:
    short_width = geometry.CameraFacingDimensionsWHD(
        width_m=1.10, height_m=0.11, depth_m=1.30)
    long_width = geometry.CameraFacingDimensionsWHD(
        width_m=1.30, height_m=0.11, depth_m=1.10)
    assert geometry.axis_assignment_candidates_from_camera_facing_dimensions(
        short_width) == (
            geometry.AxisAssignment.YAW_0,
            geometry.AxisAssignment.YAW_180,
        )
    assert geometry.axis_assignment_candidates_from_camera_facing_dimensions(
        long_width) == (
            geometry.AxisAssignment.YAW_90,
            geometry.AxisAssignment.YAW_270,
        )


@pytest.mark.parametrize("assignment", list(geometry.AxisAssignment))
def test_pose_conversion_preserves_projection_and_centroid_translation(
        assignment: geometry.AxisAssignment) -> None:
    # A nontrivial proper camera-facing rotation, built as exact quarter turns.
    rotation_cf = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation_cf = np.array([0.2, -0.1, 3.5])
    rotation_can, translation_can = geometry.camera_facing_to_canonical_pose(
        rotation_cf, translation_cf, assignment)
    axis_rotation = geometry.canonical_to_camera_facing_transform(assignment)
    np.testing.assert_array_equal(rotation_can, rotation_cf @ axis_rotation)
    np.testing.assert_array_equal(translation_can, translation_cf)
    assert np.linalg.det(rotation_can) == pytest.approx(1.0)

    permutation = geometry.canonical_to_camera_facing_keypoint_permutation(
        assignment)
    points_cf = geometry.camera_facing_keypoints_3d(assignment)
    points_can = geometry.canonical_keypoints_3d()[list(permutation)]
    camera_from_cf = (rotation_cf @ points_cf.T).T + translation_cf
    camera_from_can = (rotation_can @ points_can.T).T + translation_can
    np.testing.assert_array_equal(camera_from_can, camera_from_cf)


def test_reflection_and_parity_only_pose_requests_are_rejected() -> None:
    reflection = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="reflection"):
        geometry.validate_proper_rotation(reflection)
    with pytest.raises(ValueError, match="parity-only"):
        geometry.camera_facing_to_canonical_pose(
            np.eye(3), np.zeros(3), "SHORT_WIDTH")
    with pytest.raises(TypeError, match="axis_assignment"):
        geometry.camera_facing_to_canonical_pose(
            np.eye(3), np.zeros(3),
            (geometry.AxisAssignment.YAW_0, geometry.AxisAssignment.YAW_180),
        )

