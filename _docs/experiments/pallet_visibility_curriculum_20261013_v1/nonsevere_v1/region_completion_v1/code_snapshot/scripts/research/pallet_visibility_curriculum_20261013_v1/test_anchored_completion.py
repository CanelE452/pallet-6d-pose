"""Independent observations, restricted DOF and fixed-reference anchor budgets."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from . import anchored_completion as A
from . import boundary_rules as B
from . import curriculum as M
from . import common as C
from . import appearance as P
from . import interior_observation as I
from .test_boundary_rules import scene, observation


def inputs(count=3, rotation_error=.006):
    xyz, R, t, K = scene()
    q = M.project(xyz, R, t, K)
    R0 = cv2.Rodrigues(np.array([rotation_error, -.5*rotation_error, 0.]))[0] @ R
    t0 = t+np.array([.025, -.015, .03])
    return xyz, R, t, K, R0, t0, observation(q, xyz, K, R0, t0, count)


def test_three_boundary_translation_seed_can_gain_full_pose_with_true_interior():
    xyz, R, t, K, R0, t0, obs = inputs()
    before = deepcopy(obs)
    result = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert result["boundary_seed_applied"]
    assert result["boundary_seed"]["estimated_dof"] == 3
    np.testing.assert_array_equal(result["boundary_seed"]["R"], R0)
    assert result["accepted"] and result["estimated_dof"] == 6
    assert result["final_supplemental_ids"]
    assert all(not obs["boundary_visibility"]["expected_outline"][k] for k in result["eligible_supplemental_ids"])
    assert M._rotation_angle(np.asarray(result["R"]) @ R.T) < M._rotation_angle(R0 @ R.T)
    assert result["final_anchor_guard"]["usable"]
    assert obs == before
    np.testing.assert_array_equal(result["measured_points"], obs["fused_points"])
    np.testing.assert_array_equal(np.asarray(result["completed_points"])[:8], result["projected_points"])
    assert result["completed_points"][8] == obs["raw_points"][8]


def test_two_boundary_plus_one_interior_bootstraps_only_translation():
    xyz, R, t, K, R0, t0, obs = inputs(2, rotation_error=0.)
    inside = np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"]))
    for k in inside[1:]:
        obs["visibility_appearance"]["local_accuracy_score"][k] = .12
    result = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert result["accepted"] and result["estimated_dof"] == 3
    assert len(result["used_ids"]) == 3
    np.testing.assert_array_equal(result["R"], R0)
    assert np.linalg.norm(np.asarray(result["t"])-t) < np.linalg.norm(t0-t)/10
    assert result["anchor_reference"]["source"] == "first_accepted_measured_bootstrap"
    assert result["uncertainty"]["estimated_subspace"] == "translation_with_exact_fixed_rotation"


def test_colour_visible_weak_outline_does_not_become_true_interior_or_image_anchor():
    xyz, R, t, K, R0, t0, obs = inputs(1, rotation_error=0.)
    outline = np.asarray(obs["boundary_visibility"]["expected_outline"])
    for k in np.flatnonzero(~outline):
        obs["visibility_appearance"]["local_accuracy_score"][k] = .12
    result = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert result["anchor_ids"] == obs["correction"]["observed_refined_ids"]
    assert result["eligible_supplemental_ids"] == []
    assert not result["accepted"]
    all_visible = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"supplemental_pool": "all_visible"})
    assert all_visible["eligible_supplemental_ids"]
    assert all(all_visible["keypoint_status"][k]["measurement_role"] == "DETECTOR_VISIBLE_WEAK_BOUNDARY"
               for k in all_visible["eligible_supplemental_ids"])


def test_patch_boundary_within_model_does_not_change_its_geometric_interior_role():
    xyz, R, t, K, R0, t0, obs = inputs(2, rotation_error=0.)
    nonhull = np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"]))
    for k in nonhull:
        obs["visibility_appearance"]["states"][k] = "VISIBLE_BOUNDARY"
    result = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert result["eligible_supplemental_ids"] == nonhull.tolist()
    for k in nonhull:
        status = result["keypoint_status"][k]
        assert status["measurement_role"] == "DETECTOR_VISIBLE_MODEL_INTERIOR"
        assert status["RGB_state_at_raw_query"] == "VISIBLE_BOUNDARY"
        assert not status["expected_outline"] and not status["strong_measured_boundary"]


def test_corrupted_visible_interior_cannot_drag_corrected_boundary_anchors():
    xyz, R, t, K, R0, t0, obs = inputs(5, rotation_error=.008)
    interior = np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"]))
    for k in interior:
        obs["raw_points"][k] = (np.asarray(obs["raw_points"][k])+[16., -9.]).tolist()
        obs["fused_points"][k] = obs["raw_points"][k][:]
    protected = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    strict = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"final_guard_policy": "strict_eligible_pool"})
    unprotected = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"anchor_protection": False})
    assert protected["boundary_seed_applied"]
    assert protected["rejected_addition_ids"]
    assert any(p.get("reason") == "protected_anchor_budget_exceeded" for s in protected["stages"] for p in s.get("probes", []))
    # Anchor priority preserves the corrected seed and explicitly discloses
    # the weak-pool exception. The strict ablation restores exact initial.
    assert protected["accepted"] and protected["final_anchor_guard"]["usable"]
    assert protected["final_guard"]["independent_boundary_seed_exception"]
    assert not protected["final_guard"]["selected_checkpoint_passes_full_pool_guard"]
    np.testing.assert_array_equal(protected["R"], protected["boundary_seed"]["R"])
    np.testing.assert_array_equal(protected["t"], protected["boundary_seed"]["t"])
    assert not strict["accepted"]
    assert strict["fallback_reason"] == "final_immutable_pool_guard_rejected_all_checkpoints"
    np.testing.assert_array_equal(strict["R"], R0)
    np.testing.assert_array_equal(strict["t"], t0)
    ids = protected["anchor_ids"]
    baseline = M.project(xyz, protected["anchor_reference"]["R"], protected["anchor_reference"]["t"], K)
    protected_shift = np.linalg.norm(np.asarray(protected["projected_points"])[ids]-baseline[ids], axis=1)
    assert protected_shift.max() <= 2.+1e-8
    assert strict["final_guard"]["discarded_checkpoint_count"] == 1
    assert unprotected["final_supplemental_ids"]
    assert np.linalg.norm(np.asarray(unprotected["projected_points"])[ids]-baseline[ids], axis=1).max() > 2.


def test_four_point_bootstrap_cannot_bypass_finite_ambiguity_veto():
    xyz, R, t, K, R0, t0, obs = inputs(2, rotation_error=.004)
    conflict = [dict(best_probe=0, alternative_probe=1, objective_gap=.01)]
    with patch.object(B, "_ambiguity_records", return_value=conflict):
        result = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    four = [s for s in result["stages"] if len(s.get("attempted_ids", [])) == 4]
    assert four and len(four[0]["probes"]) == 7
    assert not four[0]["accepted"]
    assert four[0]["reason"] == "four_point_near_equal_distinct_pose_candidates"
    assert result["estimated_dof"] in (0, 3)


def test_missing_self_truncated_and_other_object_points_never_enter_optimizer():
    xyz, R, t, K, R0, t0, obs = inputs(2)
    nonhull = np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"]))
    k = int(nonhull[0])
    obs["self_hidden"][k] = obs["fixed_H"][k] = True
    if len(nonhull) > 1:
        j = int(nonhull[1])
        obs["raw_points"][j] = obs["fused_points"][j] = [-1., -1.]
    captured = []
    original = M._fit
    def fit(*args, **kwargs):
        captured.extend(args[7])
        np.testing.assert_array_equal(args[1], np.asarray(obs["fused_points"])[:8])
        return original(*args, **kwargs)
    with patch.object(M, "_fit", fit):
        result = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"supplemental_pool": "all_visible"})
    assert k not in captured and k not in result["used_ids"]
    assert j not in captured and j not in result["used_ids"]
    assert result["provenance"]["completed_points_used_for_fit"] is False
    assert result["completed_points"][8] == obs["raw_points"][8]


def test_observability_does_not_gain_data_rank_from_pose_prior_or_completion():
    xyz, R, t, K, R0, t0, obs = inputs(1)
    result = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"supplement": False})
    assert not result["accepted"] and result["used_ids"] == []
    np.testing.assert_array_equal(result["R"], R0)
    np.testing.assert_array_equal(result["t"], t0)
    uncertainty = result["uncertainty"]
    assert uncertainty["full_rigid_data_rank"] == 0
    assert len(uncertainty["full_rigid_nullspace"]) == 6
    assert uncertainty["estimated_parameter_count"] == 0
    assert uncertainty["local_covariance"] is None
    assert not uncertainty["calibrated_confidence"]


def test_completion_covariance_matches_independent_finite_difference_propagation():
    xyz, R, t, K = scene()
    K[0, 1] = 7.
    info = A._uncertainty(xyz, K, R, t, t, list(range(8)), np.ones(8), 6, M.CurriculumConfig())
    scale = info["depth_scale"]
    columns = []
    for j in range(6):
        delta = np.zeros(6)
        delta[j] = 1e-6
        plus = M.project(xyz, cv2.Rodrigues(delta[:3])[0] @ R, t+delta[3:]*scale, K)
        minus = M.project(xyz, cv2.Rodrigues(-delta[:3])[0] @ R, t-delta[3:]*scale, K)
        columns.append((plus-minus).reshape(-1)/(2e-6))
    derivative = np.stack(columns, axis=1).reshape(8, 2, 6)
    covariance = np.asarray(info["local_covariance"])
    propagated = np.einsum("pik,kl,pjl->pij", derivative, covariance, derivative)
    np.testing.assert_allclose(info["projected_covariance_px2"], propagated, atol=1e-7, rtol=1e-7)
    assert np.linalg.eigvalsh(np.asarray(info["projected_covariance_px2"])).min() >= -1e-10
    assert not info["calibrated_confidence"]
    assert "model_dimension_error" in info["uncertainty_sources_not_modelled"]


def test_translation_covariance_conditions_on_rotation_instead_of_certifying_it():
    xyz, R, t, K = scene()
    info = A._uncertainty(xyz, K, R, t, t, [1, 3, 4], np.ones(8), 3, M.CurriculumConfig())
    assert np.asarray(info["local_covariance"]).shape == (3, 3)
    assert np.asarray(info["projected_covariance_px2"]).shape == (8, 2, 2)
    assert info["conditions_on_exact_initial_rotation"]
    assert "rotation_local_std_degrees" not in info
    assert "initial_rotation_error_in_translation_only_branch" in info["uncertainty_sources_not_modelled"]


def RGB_junction_inputs():
    """Render three visible faces; bias only the detector's internal junction."""
    xyz, R, t, K = scene()
    q = C.project(xyz, R, t, K)
    t0 = t+np.array([.025, -.015, .03])
    H = C.self_hidden(xyz, R, t0)
    hull = cv2.convexHull(C.project(xyz, R, t0, K).astype(np.float32), returnPoints=False).ravel()
    inside = np.flatnonzero(~np.isin(np.arange(8), hull) & ~H)
    query = q.copy()
    query[inside] += [4., -3.]
    obs = observation(query, xyz, K, R, t0, 2, H)
    obs["visibility_appearance"]["local_accuracy_score"] = [.12]*8
    image = np.full((480, 640, 3), 230, np.uint8)
    camera = -R.T @ t
    normals = np.array([[0, 0, -1], [0, 0, 1], [0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0]])
    colors = [(35, 75, 130), (90, 140, 210), (60, 120, 200), (180, 210, 230), (30, 65, 80), (140, 70, 55)]
    for ids, normal, color in zip(P.FACES, normals, colors):
        if normal @ (camera-xyz[list(ids)].mean(0)) > 0:
            cv2.fillConvexPoly(image, np.rint(q[list(ids)]).astype(np.int32), color)
    initial = dict(available=True, R_cf=R.tolist(), centroid=t0.tolist(), cf_extents=np.ptp(xyz, axis=0).tolist())
    typed = I.infer_interior_observations(image, obs["raw_points"], K, initial, obs)
    obs["interior_observations"] = typed
    obs["image_pixel_sha256"] = I.image_pixel_sha256(image)
    return xyz, R, t, K, t0, obs, inside.tolist()


def test_actual_RGB_junction_can_supplement_accuracy_floor_without_changing_anchors():
    xyz, R, t, K, t0, obs, inside = RGB_junction_inputs()
    before = deepcopy(obs)
    off = A.solve_anchored_completion(obs, xyz, K, R, t0,
        {"allow_detector_supplements": False, "use_supplemental_observations": False})
    on = A.solve_anchored_completion(obs, xyz, K, R, t0, {"allow_detector_supplements": False})
    assert not off["accepted"] and off["used_ids"] == []
    assert on["accepted"] and on["estimated_dof"] == 3
    assert on["RGB_junction_measurement_ids"] == inside
    np.testing.assert_array_equal(on["R"], R)
    assert np.linalg.norm(np.asarray(on["t"])-t) < np.linalg.norm(t0-t)
    for k in inside:
        record = obs["interior_observations"]["supplemental_observations"][str(k)]
        np.testing.assert_array_equal(on["measured_points"][k], record["measured_xy"])
        assert on["keypoint_status"][k]["measurement_role"] == "RGB_FACE_JUNCTION_INTERIOR"
        assert not on["keypoint_status"][k]["requested_3D_corner_certified"]
    for k in on["anchor_ids"]:
        assert on["measured_points"][k] == obs["fused_points"][k]
    assert obs == before
    np.testing.assert_array_equal(on["source_measured_points"], obs["fused_points"])
    np.testing.assert_array_equal(on["raw_points"], obs["raw_points"])


def test_RGB_junction_and_other_real_visible_points_can_transition_to_full_pose():
    xyz, R, t, K, t0, obs, inside = RGB_junction_inputs()
    result = A.solve_anchored_completion(obs, xyz, K, R, t0,
        {"supplemental_pool": "all_visible", "supplemental_accuracy": .12})
    assert result["accepted"] and result["estimated_dof"] == 6
    assert any(stage["accepted"] and stage["estimated_dof"] == 3 for stage in result["stages"])
    assert any(stage["accepted"] and stage["estimated_dof"] == 6 for stage in result["stages"])
    assert all(not obs["self_hidden"][k] for k in result["used_ids"])
    assert result["final_anchor_guard"]["usable"]
    assert set(inside) <= set(result["RGB_junction_measurement_ids"])


@pytest.mark.parametrize("tamper", ["query", "blend", "pixel", "projection"])
def test_typed_RGB_measurement_must_bind_actual_query_blend_image_and_source(tamper):
    xyz, R, t, K, t0, obs, inside = RGB_junction_inputs()
    bad = deepcopy(obs)
    typed = bad["interior_observations"]
    record = typed["supplemental_observations"][str(inside[0])]
    if tamper == "query":
        record["raw_query_xy"][0] += .5
    elif tamper == "blend":
        record["measured_xy"][0] += .25
    elif tamper == "pixel":
        bad["image_pixel_sha256"] = "0"*64
    else:
        typed["provenance"]["projected_coordinates_used_as_measurements"] = True
    with pytest.raises(ValueError):
        A.solve_anchored_completion(bad, xyz, K, R, t0)
