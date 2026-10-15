"""A single immutable observation pass and separate physically rigid output."""
from copy import deepcopy

import numpy as np
import pytest

from . import common as C
from .anchored_completion_api import AnchoredCompletion
from .test_boundary_rule_api import Provider, inputs


def test_three_point_runtime_preserves_both_rotations_and_returns_one_rigid_model():
    image, raw, K, initial, observation = inputs(3)
    originals = image.copy(), raw.copy(), deepcopy(initial), deepcopy(observation)
    result = AnchoredCompletion(Provider(observation), {"supplement": False}).refine(
        image, raw, K, initial, [1.3, .15, 1.1])
    assert result["solution"]["accepted"]
    assert result["solution"]["estimated_dof"] == 3
    assert result["actual_pose"]["R_cf"] == initial["R_cf"]
    assert result["actual_pose"]["R_physical"] == initial["R_physical"]
    projected = C.project(C.cuboid(*initial["cf_extents"]), result["actual_pose"]["R_cf"],
                          result["actual_pose"]["centroid"], K)
    np.testing.assert_array_equal(result["completed_points"][:8], projected)
    np.testing.assert_array_equal(result["measured_points"], observation["fused_points"])
    assert result["completed_points"][8] == raw[8].tolist()
    assert set(map(int, result["observed_measurements_by_id"])) == set(result["used_ids"])
    assert not result["provenance"]["projected_coordinates_used_as_measurements"]
    assert not result["provenance"]["rigid_completion_is_observed"]
    np.testing.assert_array_equal(image, originals[0])
    np.testing.assert_array_equal(raw, originals[1])
    assert initial == originals[2] and observation == originals[3]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_insufficient_evidence_keeps_pose_exact_and_labels_initial_completion(count):
    image, raw, K, initial, observation = inputs(count)
    result = AnchoredCompletion(Provider(observation), {"supplement": False}).refine(
        image, raw, K, initial, [1.3, .15, 1.1])
    assert result["actual_pose"] == initial
    assert not result["solution"]["accepted"]
    assert result["used_ids"] == [] and result["observed_measurements_by_id"] == {}
    expected = C.project(C.cuboid(*initial["cf_extents"]), initial["R_cf"], initial["centroid"], K)
    np.testing.assert_array_equal(result["rigid_model_corners"], expected)
    np.testing.assert_array_equal(result["original_predictions"], raw)
    np.testing.assert_array_equal(result["measured_points"], raw)
    assert result["solution"]["uncertainty"]["local_covariance"] is None


@pytest.mark.parametrize("tamper", ["raw", "image", "hidden", "centre", "bound"])
def test_runtime_rejects_provider_mutation_or_unbound_moved_pixels(tamper):
    image, raw, K, initial, observation = inputs(3)
    originals = image.copy(), raw.copy(), deepcopy(initial)
    with pytest.raises(ValueError):
        AnchoredCompletion(Provider(observation, tamper), {"supplement": False}).refine(
            image, raw, K, initial, [1.3, .15, 1.1])
    np.testing.assert_array_equal(image, originals[0])
    np.testing.assert_array_equal(raw, originals[1])
    assert initial == originals[2]


def test_other_instance_region_excludes_measurement_without_certifying_depth():
    image, raw, K, initial, observation = inputs(3)
    k = observation["correction"]["observed_refined_ids"][0]
    cue = {k: dict(schema="independent_occluding_region_v1", native_corner=k,
        query_xy=raw[k].tolist(), independent_of_target_palette=True, different_instance_from_target=True,
        region_contains_query=True, region_observed_rgb=True, instance_id="other:fixture",
        source_reference="independent_region_fixture", source_type="instance_segmentation")}
    result = AnchoredCompletion(Provider(observation), {"supplement": False}).refine(
        image, raw, K, initial, [1.3, .15, 1.1], occluding_evidence=cue)
    assert k not in result["anchor_ids"] and k not in result["used_ids"]
    assert not result["keypoint_status"][k]["external_occlusion_confirmed"]
    assert result["actual_pose"] == initial
