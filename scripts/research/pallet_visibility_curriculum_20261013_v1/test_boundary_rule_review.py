"""Independent line/fusion identity, pose-probe and limited-DOF review checks."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from . import boundary_rules as B
from .boundary_evidence import summarize_boundary_evidence
from .boundary_rule_api import BoundaryRuleCorrector
from .test_boundary_rule_api import Provider, inputs


@pytest.mark.parametrize("bad", ["negative", "asymmetric", "nonfinite"])
def test_bad_covariance_removes_strong_support_without_reclassifying_RGB_visibility(bad):
    _, _, _, _, observation = inputs(5)
    k = observation["correction"]["observed_refined_ids"][0]
    record = observation["intersection_covariance"][str(k)]
    record["covariance_px2"] = ({"negative": [[-1., 0.], [0., 1.]],
        "asymmetric": [[1., .5], [0., 1.]], "nonfinite": [[np.nan, 0.], [0., 1.]]}[bad])
    evidence = summarize_boundary_evidence(observation, image_hw=(480, 640))
    assert k not in evidence["strong_ids"]
    assert evidence["records"][k]["p_visible"] == .9
    assert evidence["records"][k]["RGB_material_supported"]
    assert not evidence["records"][k]["external_occlusion_confirmed"]


def test_arbitrary_small_fused_shift_cannot_borrow_unrelated_true_line_support():
    image, raw, K, initial, observation = inputs(5)
    k = observation["correction"]["observed_refined_ids"][0]
    observation["fused_points"][k][0] += .5
    # The independently observed intersection and all actual samples stay raw.
    with pytest.raises(ValueError):
        BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])


def consistent_sigma2_observation():
    image, raw, K, initial, observation = inputs(5)
    old = raw.copy()
    raw[:8] -= [1., 0.]
    observation["raw_points"] = raw.tolist()
    observation["fused_points"] = raw.tolist()
    observation["fusion"] = []
    for k in observation["correction"]["observed_refined_ids"]:
        observation["fused_points"][k] = (raw[k] + .5 * (old[k] - raw[k])).tolist()
        covariance = observation["intersection_covariance"][str(k)]
        covariance.update(eigenvalues_px2=[1., 1.], largest_std_px=1., candidate=old[k].tolist())
        observation["fusion"].append(dict(corner=k, alpha=.5, detector_sigma_assumed_px=2.,
            largest_intersection_variance_px2=1., applied_shift_px=.5,
            covariance_is_semantic_correctness_probability=False))
    return image, raw, K, initial, observation


def test_consistent_sigma2_fusion_still_accepts_without_changing_corner_identity():
    image, raw, K, initial, observation = consistent_sigma2_observation()
    result = BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    assert result["decision"]["image_supported_boundary_count"] == 5
    assert result["decision"]["accepted"]
    np.testing.assert_array_equal(result["corrected_measurements"], observation["fused_points"])
    assert not result["provenance"]["projected_completion_used_for_fit"]


@pytest.mark.parametrize("field,value", [("alpha", .25), ("largest_intersection_variance_px2", 2.)])
def test_sigma2_alpha_and_covariance_records_cannot_be_changed_independently(field, value):
    image, raw, K, initial, observation = consistent_sigma2_observation()
    observation["fusion"][0][field] = value
    with pytest.raises(ValueError):
        BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])


def test_conditional_truncation_must_match_initial_pose_and_actual_canvas():
    image, raw, K, initial, observation = inputs(5)
    k = observation["correction"]["observed_refined_ids"][0]
    observation["boundary_visibility"]["conditional_truncation"][k] = True
    observation["boundary_visibility"]["geometric_boundary_candidates"][k] = False
    # Self-consistent provider flags must not override the supplied initial geometry.
    with pytest.raises(ValueError):
        BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])


def test_three_point_optimizer_cannot_smuggle_rotation_into_translation_branch():
    image, raw, K, initial, observation = inputs(3)
    R0 = np.asarray(initial["R_cf"])
    changed_R = cv2.Rodrigues(np.array([.01, 0., 0.]))[0] @ R0
    with patch.object(B, "_fit_translation", return_value=(changed_R, np.array([.02, .01, 3.5]), {})):
        result = BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    assert not result["decision"]["accepted"]
    assert result["actual_pose"] == initial
    assert result["decision"]["fallback_reason"] == "translation_rule_changed_rotation"
    assert result["projected_completion_ids"] == []


def test_four_point_near_equal_distinct_probe_solutions_trigger_full_API_fallback():
    image, raw, K, initial, observation = inputs(4)
    R0 = np.asarray(initial["R_cf"]); t0 = np.asarray(initial["centroid"])
    alternative = cv2.Rodrigues(np.array([0., np.deg2rad(6), 0.]))[0] @ R0
    t = np.array([.02, .01, 3.5])
    calls = [(R0.copy(), t.copy(), {})] + [(alternative.copy(), t.copy(), {})] + [(R0.copy(), t.copy(), {})] * 5
    def objective(*args):
        R, pose_t = args[3], args[4]
        value = 100. if np.array_equal(pose_t, t0) else 1. if np.allclose(R, R0) else 1.05
        return dict(objective=value, data_objective=value, prior_cost_sum=0., frozen_weight_sum=1.)
    with patch.object(B.M, "_fit", side_effect=calls), patch.object(B.M, "_aggregate_objective", side_effect=objective):
        result = BoundaryRuleCorrector(Provider(observation)).refine(image, raw, K, initial, [1.3, .15, 1.1])
    assert not result["decision"]["accepted"]
    assert result["decision"]["fallback_reason"] == "four_point_near_equal_distinct_pose_candidates"
    assert result["solution"]["ambiguity_probe_conflicts"]
    assert result["actual_pose"] == initial
    np.testing.assert_array_equal(result["completed_points"], raw)


def test_explicit_actual_canvas_cannot_be_fractional():
    _, _, _, _, observation = inputs(3)
    with pytest.raises(ValueError):
        summarize_boundary_evidence(observation, image_hw=(480.5, 640))
