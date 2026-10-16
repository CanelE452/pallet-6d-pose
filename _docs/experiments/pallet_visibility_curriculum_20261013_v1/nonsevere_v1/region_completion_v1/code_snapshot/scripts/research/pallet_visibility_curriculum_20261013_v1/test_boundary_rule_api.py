"""Live-rule contract: limited DOF, output completion and optional occluder cues."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from . import common as C
from . import boundary_rules as B
from .boundary_rule_api import BoundaryRuleCorrector
from .test_boundary_rules import scene, observation


class Provider:
    def __init__(self, evidence, tamper=None):
        self.evidence = evidence
        self.tamper = tamper

    def assess(self, image, raw, K, initial):
        result = deepcopy(self.evidence)
        if self.tamper == "raw":
            raw[0] += [1., 0.]
            result["raw_points"] = raw.tolist()
        elif self.tamper == "image":
            image[0, 0] = [255, 255, 255]
        elif self.tamper == "hidden":
            k = np.flatnonzero(result["self_hidden"])[0]
            result["fused_points"][k][0] += 1.
        elif self.tamper == "centre":
            result["fused_points"][8][0] += 1.
        elif self.tamper == "bound":
            k = result["correction"]["observed_refined_ids"][0]
            result["fused_points"][k][0] += 7.
        return result


def inputs(count):
    xyz, R, t, K = scene()
    q = C.project(xyz, R, t, K)
    t0 = t + [.03, -.02, .04]
    H = C.self_hidden(xyz, R, t0)
    evidence = observation(q, xyz, K, R, t0, count, H)
    Q = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))[0]
    initial = dict(available=True, R_cf=R.tolist(), R_physical=(R @ Q).tolist(),
                   centroid=t0.tolist(), cf_extents=[1.1, .15, 1.3])
    return np.zeros((480, 640, 3), np.uint8), np.asarray(evidence["raw_points"]), K, initial, evidence


def test_three_points_change_translation_only_and_complete_five_unmeasured_slots():
    image, raw, K, initial, evidence = inputs(3)
    before = raw.copy()
    with patch.object(B.M, "_fit", side_effect=AssertionError("Full pose fit forbidden for three points")):
        result = BoundaryRuleCorrector(Provider(evidence)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    assert result["decision"]["accepted"]
    assert result["decision"]["estimated_dof"] == 3
    assert result["actual_pose"]["R_cf"] == initial["R_cf"]
    assert result["actual_pose"]["R_physical"] == initial["R_physical"]
    assert len(result["projected_completion_ids"]) == 5
    ids = result["projected_completion_ids"]
    np.testing.assert_allclose(np.asarray(result["completed_points"])[ids], np.asarray(result["projected_points"])[ids])
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(result["observed_points"], before)
    assert result["completed_points"][8] == before[8].tolist()
    assert not result["provenance"]["projected_completion_used_for_fit"]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_two_or_fewer_preserve_initial_pose_and_original_including_hidden_queries(count):
    image, raw, K, initial, evidence = inputs(count)
    result = BoundaryRuleCorrector(Provider(evidence)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    assert not result["decision"]["accepted"]
    assert result["actual_pose"] == initial
    assert result["decision"]["used_ids"] == []
    assert result["projected_completion_ids"] == []
    np.testing.assert_array_equal(result["completed_points"], raw)


def test_independent_other_instance_can_disable_three_point_rule_despite_high_RGB_score():
    image, raw, K, initial, evidence = inputs(3)
    k = evidence["correction"]["observed_refined_ids"][0]
    external = {k: dict(schema="independent_occluding_region_v1", native_corner=k,
        query_xy=raw[k].tolist(), independent_of_target_palette=True, different_instance_from_target=True,
        region_contains_query=True, region_observed_rgb=True, instance_id="fixture:other",
        source_reference="analytic_instance_record", source_type="instance_segmentation")}
    result = BoundaryRuleCorrector(Provider(evidence)).refine(image, raw, K, initial,
        [1.3, .15, 1.1], occluding_evidence=external)
    assert result["decision"]["image_supported_boundary_count"] == 2
    assert not result["decision"]["accepted"]
    assert result["keypoint_status"][k]["state"] == "OTHER_OBJECT_SUSPECTED"
    assert not result["keypoint_status"][k]["external_occlusion_confirmed"]
    assert result["actual_pose"] == initial


@pytest.mark.parametrize("tamper", ["raw", "image", "hidden", "centre", "bound"])
def test_rule_provider_cannot_mutate_inputs_or_invent_excluded_measurements(tamper):
    image, raw, K, initial, evidence = inputs(3)
    snapshots = image.copy(), raw.copy(), deepcopy(initial)
    with pytest.raises(ValueError):
        BoundaryRuleCorrector(Provider(evidence, tamper)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    np.testing.assert_array_equal(image, snapshots[0])
    np.testing.assert_array_equal(raw, snapshots[1])
    assert initial == snapshots[2]

