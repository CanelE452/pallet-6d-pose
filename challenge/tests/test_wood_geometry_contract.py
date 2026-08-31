"""Wood W/D parity remains distinct from its canonical physical axes."""

from __future__ import annotations

import numpy as np

from scripts.annotate import pallet_geometry as geometry
from scripts.annotate.object_geometry_registry import WOOD_OBJECT_TYPE, load_object_geometry_registry


def test_wood_parities_and_dynamic_face_names_are_exact() -> None:
    spec = load_object_geometry_registry().resolve(WOOD_OBJECT_TYPE)
    yaw0 = geometry.camera_facing_dimensions(
        geometry.AxisAssignment.YAW_0, spec.physical_dimensions
    )
    yaw90 = geometry.camera_facing_dimensions(
        geometry.AxisAssignment.YAW_90, spec.physical_dimensions
    )
    assert (yaw0.width_m, yaw0.height_m, yaw0.depth_m) == (0.80, 0.14, 0.59)
    assert (yaw90.width_m, yaw90.height_m, yaw90.depth_m) == (0.59, 0.14, 0.80)
    assert geometry.axis_assignment_candidates_from_camera_facing_dimensions(
        yaw0, physical_dimensions=spec.physical_dimensions
    ) == (geometry.AxisAssignment.YAW_0, geometry.AxisAssignment.YAW_180)
    assert geometry.axis_assignment_candidates_from_camera_facing_dimensions(
        yaw90, physical_dimensions=spec.physical_dimensions
    ) == (geometry.AxisAssignment.YAW_90, geometry.AxisAssignment.YAW_270)
    # Wood X>Z, so YAW_0 is the long visible width (the inverse of plastic).
    assert geometry.camera_facing_hypothesis_name(
        geometry.AxisAssignment.YAW_0, spec.physical_dimensions
    ) == "long-face-front"
    assert geometry.camera_facing_hypothesis_name(
        geometry.AxisAssignment.YAW_90, spec.physical_dimensions
    ) == "short-face-front"


def test_wood_keypoint_permutations_are_bijective_proper_rotations() -> None:
    spec = load_object_geometry_registry().resolve(WOOD_OBJECT_TYPE)
    for assignment in geometry.AxisAssignment:
        permutation = geometry.canonical_to_camera_facing_keypoint_permutation(
            assignment, spec.physical_dimensions
        )
        assert set(permutation) == set(range(9))
        rotation = geometry.canonical_to_camera_facing_transform(assignment)
        assert np.linalg.det(rotation) == 1.0
        canonical = geometry.canonical_keypoints_3d(spec.physical_dimensions)
        camera_facing = geometry.camera_facing_keypoints_3d(
            assignment, spec.physical_dimensions
        )
        assert np.array_equal((rotation @ canonical.T).T[list(permutation)], camera_facing)
