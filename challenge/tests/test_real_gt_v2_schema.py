"""Validation tests for the additive real-pallet GT v2 contract."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from scripts.annotate import pallet_geometry as geometry
from scripts.annotate import real_gt_v2_schema as schema


def _pose_record(
        assignment: geometry.AxisAssignment,
        transform_cf: np.ndarray,
) -> dict:
    rotation, translation = geometry.camera_facing_to_canonical_pose(
        transform_cf[:3, :3], transform_cf[:3, 3], assignment)
    return {
        "axis_assignment": assignment.value,
        "pose_transform": geometry.make_pose_transform(
            rotation, translation).tolist(),
        "canonical_to_camera_facing_rotation": (
            geometry.canonical_to_camera_facing_transform(
                assignment).tolist()),
        "canonical_to_camera_facing_keypoint_permutation": list(
            geometry.canonical_to_camera_facing_keypoint_permutation(
                assignment)),
        "projection_parity_max_px": 0.0,
    }


def valid_document(*, long_width: bool = False) -> dict:
    transform_cf = np.eye(4, dtype=float)
    transform_cf[:3, 3] = [0.1, -0.2, 3.0]
    if long_width:
        dimensions = {"width": 1.30, "height": 0.11, "depth": 1.10}
        assignments = (
            geometry.AxisAssignment.YAW_90,
            geometry.AxisAssignment.YAW_270,
        )
    else:
        dimensions = {"width": 1.10, "height": 0.11, "depth": 1.30}
        assignments = (
            geometry.AxisAssignment.YAW_0,
            geometry.AxisAssignment.YAW_180,
        )
    annotations = [{
        "xy": [float(index), float(index + 1)],
        "visibility": 0,
        "in_frame": True,
        "source": "unknown",
        "reason": "unknown",
    } for index in range(schema.KEYPOINT_COUNT)]
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "camera_data": {
            "width": 640,
            "height": 480,
            "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
        },
        "objects": [{
            # Historical fields remain available to historical consumers.
            "dimensions_m": copy.deepcopy(dimensions),
            "pose_transform": transform_cf.tolist(),
            "manual_kps": [entry["xy"] for entry in annotations],
            "keypoint_frame": schema.KEYPOINT_FRAME,
            "physical_dimensions_m": {"x": 1.10, "y": 0.11, "z": 1.30},
            "camera_facing_pnp": {
                "axis_assignment": None,
                "axis_assignment_candidates": [
                    assignment.value for assignment in assignments],
                "dimensions_m": copy.deepcopy(dimensions),
                "pose_transform": transform_cf.tolist(),
            },
            "canonical_pose": None,
            "canonical_pose_candidates": [
                _pose_record(assignment, transform_cf)
                for assignment in assignments
            ],
            "legacy": {
                "dimensions_m": copy.deepcopy(dimensions),
                "pose_transform": transform_cf.tolist(),
                "fix_swap": None,
            },
            "keypoint_annotations": annotations,
            "occlusion_level": "unknown",
            "truncation": {
                "is_truncated": False,
                "outside_keypoints": [],
                "bbox_outside_fraction": 0.0,
            },
            "migration_status": "MANUAL_REVIEW_REQUIRED",
        }],
    }


@pytest.mark.parametrize("long_width", [False, True])
def test_unresolved_signed_pose_is_valid_when_both_candidates_are_preserved(
        long_width: bool) -> None:
    document = valid_document(long_width=long_width)
    assert schema.validate_gt_v2(document) is None
    assert schema.validate_gt_document(document) == "v2"
    assert document["objects"][0]["canonical_pose"] is None


@pytest.mark.parametrize("object_count", [0, 2])
def test_v2_requires_exactly_one_pallet_object(object_count: int) -> None:
    document = valid_document()
    original = copy.deepcopy(document["objects"][0])
    document["objects"] = [copy.deepcopy(original) for _ in range(object_count)]
    with pytest.raises(schema.SchemaValidationError, match="exactly one"):
        schema.validate_gt_v2(document)


def test_confirmed_signed_pose_must_match_selected_axis() -> None:
    document = valid_document()
    obj = document["objects"][0]
    obj["camera_facing_pnp"]["axis_assignment"] = "YAW_180"
    obj["canonical_pose"] = copy.deepcopy(obj["canonical_pose_candidates"][1])
    obj["migration_status"] = "CANONICAL_POSE_CONFIRMED"
    schema.validate_gt_v2(document)

    broken = copy.deepcopy(document)
    broken["objects"][0]["canonical_pose"] = copy.deepcopy(
        broken["objects"][0]["canonical_pose_candidates"][0])
    with pytest.raises(schema.SchemaValidationError, match="must match"):
        schema.validate_gt_v2(broken)


def test_parity_only_state_cannot_claim_a_singular_canonical_pose() -> None:
    document = valid_document()
    obj = document["objects"][0]
    obj["canonical_pose"] = copy.deepcopy(obj["canonical_pose_candidates"][0])
    with pytest.raises(schema.SchemaValidationError, match="must be null"):
        schema.validate_gt_v2(document)

    document = valid_document()
    obj = document["objects"][0]
    obj["camera_facing_pnp"]["axis_assignment"] = "YAW_0"
    with pytest.raises(schema.SchemaValidationError, match="is required"):
        schema.validate_gt_v2(document)


@pytest.mark.parametrize("visibility", [0, 1, 2])
def test_visibility_accepts_only_the_three_integer_states(visibility: int) -> None:
    document = valid_document()
    document["objects"][0]["keypoint_annotations"][0]["visibility"] = visibility
    schema.validate_gt_v2(document)


@pytest.mark.parametrize("visibility", [True, False, 1.0, -1, 3, "1", None])
def test_visibility_rejects_non_schema_values(visibility: object) -> None:
    document = valid_document()
    document["objects"][0]["keypoint_annotations"][0]["visibility"] = visibility
    with pytest.raises(schema.SchemaValidationError, match="visibility"):
        schema.validate_gt_v2(document)


@pytest.mark.parametrize("visibility", [1, 2])
def test_labeled_visibility_requires_xy(visibility: int) -> None:
    document = valid_document()
    annotation = document["objects"][0]["keypoint_annotations"][3]
    annotation["visibility"] = visibility
    annotation["xy"] = None
    with pytest.raises(schema.SchemaValidationError, match="xy"):
        schema.validate_gt_v2(document)


def test_unknown_visibility_may_retain_legacy_xy_or_null() -> None:
    document = valid_document()
    document["objects"][0]["keypoint_annotations"][0]["xy"] = None
    document["objects"][0]["keypoint_annotations"][0]["in_frame"] = False
    schema.validate_gt_v2(document)


def test_local_ultralytics_mapping_masks_zero_and_preserves_one_two() -> None:
    annotations = valid_document()["objects"][0]["keypoint_annotations"]
    annotations[0].update({"xy": [12.0, 34.0], "visibility": 0})
    annotations[1].update({"xy": [56.0, 78.0], "visibility": 1})
    annotations[2].update({"xy": [90.0, 12.0], "visibility": 2})

    converted = schema.keypoint_annotations_to_ultralytics(annotations)

    assert converted[0] == [0.0, 0.0, 0.0]
    assert converted[1] == [56.0, 78.0, 1.0]
    assert converted[2] == [90.0, 12.0, 2.0]


def test_reflection_is_rejected_from_pose_records() -> None:
    document = valid_document()
    candidate = document["objects"][0]["canonical_pose_candidates"][0]
    candidate["canonical_to_camera_facing_rotation"] = np.diag(
        [-1.0, 1.0, 1.0]).tolist()
    with pytest.raises(schema.SchemaValidationError, match="reflection"):
        schema.validate_gt_v2(document)


def test_axis_rotation_permutation_and_pose_are_cross_checked() -> None:
    document = valid_document()
    obj = document["objects"][0]
    candidate = obj["canonical_pose_candidates"][0]
    candidate["canonical_to_camera_facing_keypoint_permutation"] = list(
        geometry.canonical_to_camera_facing_keypoint_permutation("YAW_180"))
    with pytest.raises(schema.SchemaValidationError, match="coordinate-set"):
        schema.validate_gt_v2(document)

    document = valid_document()
    obj = document["objects"][0]
    # YAW_180 cannot reuse the unrotated camera-facing transform as canonical.
    obj["canonical_pose_candidates"][1]["pose_transform"] = copy.deepcopy(
        obj["camera_facing_pnp"]["pose_transform"])
    with pytest.raises(schema.SchemaValidationError, match="proper camera-facing"):
        schema.validate_gt_v2(document)


def test_dimensions_and_candidate_pair_are_fixed_by_physical_contract() -> None:
    document = valid_document()
    document["objects"][0]["physical_dimensions_m"]["z"] = 1.31
    with pytest.raises(schema.SchemaValidationError, match="canonical value"):
        schema.validate_gt_v2(document)

    document = valid_document()
    obj = document["objects"][0]
    obj["camera_facing_pnp"]["axis_assignment_candidates"] = [
        "YAW_90", "YAW_270"]
    obj["canonical_pose_candidates"] = []
    with pytest.raises(schema.SchemaValidationError, match="W/D parity"):
        schema.validate_gt_v2(document)


def test_truncation_types_and_indices_are_strict() -> None:
    document = valid_document()
    document["objects"][0]["truncation"]["is_truncated"] = 1
    with pytest.raises(schema.SchemaValidationError, match="bool or null"):
        schema.validate_gt_v2(document)

    document = valid_document()
    document["objects"][0]["truncation"]["outside_keypoints"] = [2, 2]
    with pytest.raises(schema.SchemaValidationError, match="duplicate"):
        schema.validate_gt_v2(document)


def test_legacy_loader_acknowledges_but_does_not_promote_old_schema() -> None:
    legacy = {"objects": [{"dimensions_m": {"width": 1.1}}]}
    assert schema.validate_gt_document(legacy) == "legacy"
    with pytest.raises(schema.SchemaValidationError, match="schema_version"):
        schema.validate_gt_document(legacy, allow_legacy=False)
