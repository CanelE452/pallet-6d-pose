"""Independent observation protection and completion contract canaries."""
from copy import deepcopy
from pathlib import Path
import runpy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from . import anchored_completion as A
from . import boundary_rules as B
from . import common as C
from . import curriculum as M
from . import interior_observation as I
from .anchored_completion_api import AnchoredCompletion
from .test_boundary_rule_api import inputs
from .test_boundary_rule_api import Provider
from .test_boundary_rules import scene, observation as make_observation
from .test_interior_observation import toy_scene


def target_with_max_horizontal_shift(xyz, R, t, K, pixels):
    base = C.project(xyz, R, t, K)
    small = np.array([.01, 0., 0.])
    displacement = np.linalg.norm(C.project(xyz, R, t + small, K) - base, axis=1)
    delta = small * pixels / displacement.max()
    return base, t + delta


def test_anchor_guard_uses_fixed_seed_to_prevent_cumulative_small_step_drift():
    xyz, R, t, K = scene()
    base, target_t = target_with_max_horizontal_shift(xyz, R, t, K, 2.7)
    measured = C.project(xyz, R, target_t, K)
    reference = dict(R=R, t=t, source="fixed_independent_seed")
    cfg, pose = A.AnchoredCompletionConfig(), M.CurriculumConfig()
    for fraction in (1 / 3, 2 / 3):
        result = A._anchor_guard(xyz, measured, K, R, t + fraction * (target_t - t),
                                 list(range(8)), reference, R, t, cfg, pose)
        assert result["usable"]
    final = A._anchor_guard(xyz, measured, K, R, target_t, list(range(8)),
                            reference, R, t, cfg, pose)
    assert not final["usable"]
    assert all(r["per_point_loss_pass"] for r in final["records"])
    assert any(not r["projection_shift_pass"] for r in final["records"])
    # Each individual update is below2px; only the frozen seed comparison catches it.
    assert np.linalg.norm(measured - C.project(xyz, R, t + 2 / 3 * (target_t - t), K), axis=1).max() < 2.
    np.testing.assert_array_equal(base, C.project(xyz, reference["R"], reference["t"], K))


def test_one_anchor_cannot_be_sacrificed_to_improve_aggregate_other_anchor_error():
    xyz, R, t, K = scene()
    base, candidate_t = target_with_max_horizontal_shift(xyz, R, t, K, 1.2)
    candidate = C.project(xyz, R, candidate_t, K)
    sacrificial = int(np.argmax(np.linalg.norm(candidate - base, axis=1)))
    measured = candidate.copy()
    measured[sacrificial] = base[sacrificial]
    before = np.linalg.norm(base - measured, axis=1)
    after = np.linalg.norm(candidate - measured, axis=1)
    assert np.sum(A._huber(after, 8.)) < np.sum(A._huber(before, 8.))
    result = A._anchor_guard(xyz, measured, K, R, candidate_t, list(range(8)),
        dict(R=R, t=t, source="fixed_seed"), R, t,
        A.AnchoredCompletionConfig(), M.CurriculumConfig())
    assert not result["usable"]
    assert all(r["projection_shift_pass"] for r in result["records"])
    assert not result["records"][sacrificial]["per_point_loss_pass"]


def test_bootstrap_has_explicit_initial_residual_guard_without_fake_corrected_seed():
    xyz, R, t, K = scene()
    _, target_t = target_with_max_horizontal_shift(xyz, R, t, K, 2.7)
    measured = C.project(xyz, R, target_t, K)
    result = A._anchor_guard(xyz, measured, K, R, target_t, list(range(8)), None, R, t,
        A.AnchoredCompletionConfig(), M.CurriculumConfig())
    assert result["usable"]
    assert result["reference_source"] == "initial_residual_only_before_bootstrap"
    assert not result["shift_budget_applied"]
    assert all(r["huber_increase_px2"] < 0 for r in result["records"])


def test_accepted_boundary_seed_cannot_bypass_final_immutable_pool_guard():
    _, raw, K, initial, observation = inputs(3)
    snapshot = deepcopy(observation)
    xyz = C.cuboid(*initial["cf_extents"])
    R0, t0 = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
    seed = B.solve_boundary_rules(observation, xyz, K, R0, t0,
        dict(coordinate_source="RAW", image_hw=(480, 640)))
    assert seed["accepted"]
    def objective(*args):
        value = 1. if np.array_equal(np.asarray(args[4]), t0) else 2.
        return dict(objective=value, data_objective=value, prior_cost_sum=0., frozen_weight_sum=1.)
    with patch.object(B, "solve_boundary_rules", return_value=seed), patch.object(M, "_aggregate_objective", side_effect=objective):
        result = A.solve_anchored_completion(observation, xyz, K, R0, t0,
            dict(coordinate_source="RAW", supplement=False, image_hw=(480, 640),
                 final_guard_policy="strict_eligible_pool"))
    assert result["boundary_seed_applied"]
    assert not result["accepted"]
    assert result["fallback_reason"] == "final_immutable_pool_guard_rejected_all_checkpoints"
    assert result["used_ids"] == []
    assert result["estimated_dof"] == 0
    assert not result["final_guard"]["checkpoints"][0]["passes_immutable_pool_initial_guard"]
    np.testing.assert_array_equal(result["R"], R0)
    np.testing.assert_array_equal(result["t"], t0)
    np.testing.assert_array_equal(result["raw_points"], raw)
    np.testing.assert_array_equal(result["measured_points"], raw)
    np.testing.assert_allclose(np.asarray(result["completed_points"])[:8], C.project(xyz, R0, t0, K))
    assert result["completed_points"][8:] == raw[8:].tolist()
    assert observation == snapshot
    assert not result["provenance"]["completed_points_used_for_fit"]


def test_anchor_priority_explicitly_preserves_seed_without_claiming_full_pool_gain():
    _, raw, K, initial, observation = inputs(3)
    snapshot = deepcopy(observation)
    xyz = C.cuboid(*initial["cf_extents"])
    R0, t0 = np.asarray(initial["R_cf"]), np.asarray(initial["centroid"])
    seed = B.solve_boundary_rules(observation, xyz, K, R0, t0,
        dict(coordinate_source="RAW", image_hw=(480, 640)))
    assert seed["accepted"]
    def objective(*args):
        value = 1. if np.array_equal(np.asarray(args[4]), t0) else 2.
        return dict(objective=value, data_objective=value, prior_cost_sum=0., frozen_weight_sum=1.)
    with patch.object(B, "solve_boundary_rules", return_value=seed), patch.object(M, "_aggregate_objective", side_effect=objective):
        result = A.solve_anchored_completion(observation, xyz, K, R0, t0,
            dict(coordinate_source="RAW", supplement=False, image_hw=(480, 640),
                 final_guard_policy="anchor_priority"))
    assert result["accepted"]
    assert result["final_guard"]["independent_boundary_seed_exception"]
    assert not result["final_guard"]["selected_checkpoint_passes_full_pool_guard"]
    assert result["final_guard"]["independent_boundary_seed_objective_gain"] > 0
    assert result["final_anchor_guard"]["usable"]
    assert all(r["projection_shift_px"] == 0 and r["huber_increase_px2"] == 0
               for r in result["final_anchor_guard"]["records"])
    assert result["used_ids"] == seed["used_ids"]
    assert result["estimated_dof"] == seed["estimated_dof"]
    np.testing.assert_array_equal(result["R"], seed["R"])
    np.testing.assert_array_equal(result["t"], seed["t"])
    np.testing.assert_array_equal(result["measured_points"], raw)
    assert result["completed_points"][8:] == raw[8:].tolist()
    assert observation == snapshot


def test_no_support_has_no_pose_covariance_or_prior_manufactured_observability():
    xyz, R, t, K = scene()
    result = A._uncertainty(xyz, K, R, t, t, [], np.ones(8), 0, M.CurriculumConfig())
    assert result["full_rigid_data_rank"] == 0
    assert np.asarray(result["full_rigid_nullspace"]).shape == (6, 6)
    assert result["local_covariance"] is None
    assert result["estimated_parameter_count"] == 0
    assert not result["calibrated_confidence"]
    assert not result["global_pose_uniqueness_proved"]


def full_junction_fixture():
    image, q, K, initial, base, k = toy_scene()
    xyz = C.cuboid(*initial["cf_extents"])
    observation = make_observation(q[:8], xyz, K, np.asarray(initial["R_cf"]),
                                   np.asarray(initial["centroid"]), 0, base["self_hidden"])
    observation["raw_points"] = q.tolist()
    observation["fused_points"] = q.tolist()
    observation["visibility_appearance"] = deepcopy(base["visibility_appearance"])
    initial["R_physical"] = deepcopy(initial["R_cf"])
    return image, q, K, initial, observation, k


@pytest.mark.parametrize("cached", [False, True])
def test_actual_API_hash_overrides_user_rules_for_foreign_same_shape_image(cached):
    image, q, K, initial, observation, _ = full_junction_fixture()
    foreign = image.copy()
    foreign[0, 0, 0] ^= 1
    record = I.infer_interior_observations(foreign, q, K, initial, observation)
    assert record["supplemental_observations"]
    forged_rule = dict(image_pixel_sha256=I.image_pixel_sha256(foreign))
    if cached:
        observation["interior_observations"] = record
        api = AnchoredCompletion(Provider(observation), forged_rule)
    else:
        api = AnchoredCompletion(Provider(observation), forged_rule,
                                interior_provider=lambda *_: deepcopy(record))
    with pytest.raises(ValueError, match="actual-image binding"):
        api.refine(image, q, K, initial, initial["cf_extents"])


@pytest.mark.parametrize("cached", [False, True])
def test_bound_junction_is_separate_measurement_without_modifying_RGB_semantics(cached):
    image, q, K, initial, observation, k = full_junction_fixture()
    originals = deepcopy(observation), q.copy(), image.copy(), deepcopy(initial)
    record = I.infer_interior_observations(image, q, K, initial, observation)
    expected = record["supplemental_observations"][str(k)]["measured_xy"]
    if cached:
        observation["interior_observations"] = record
        api = AnchoredCompletion(Provider(observation), {"allow_detector_supplements": False})
    else:
        api = AnchoredCompletion(Provider(observation), {"allow_detector_supplements": False},
                                interior_provider=lambda *_: deepcopy(record))
    result = api.refine(image, q, K, initial, initial["cf_extents"])
    assert result["solution"]["RGB_junction_measurement_ids"] == [k]
    np.testing.assert_array_equal(np.asarray(result["measured_points"])[k], expected)
    np.testing.assert_array_equal(result["solution"]["source_measured_points"], q)
    np.testing.assert_array_equal(result["measured_points"][8:], q[8:])
    assert result["observation_result"]["visibility_appearance"] == originals[0]["visibility_appearance"]
    assert result["solution"]["configuration"]["image_pixel_sha256"] == I.image_pixel_sha256(image)
    assert not result["solution"]["accepted"]  # One actual junction cannot create three-dimensional support.
    assert result["used_ids"] == []
    np.testing.assert_array_equal(q, originals[1])
    np.testing.assert_array_equal(image, originals[2])
    assert initial == originals[3]


def test_null_hash_cached_validation_explicitly_reports_unverified_image_binding():
    image, q, K, initial, observation, k = full_junction_fixture()
    record = I.infer_interior_observations(image, q, K, initial, observation)
    verified = I.validate_interior_observations(record, q, K, initial, observation,
        image_hw=image.shape[:2], image_pixel_sha256=None)
    assert not verified[str(k)]["actual_image_hash_binding_verified"]
    assert not verified[str(k)]["source_RGB_gradient_values_verified_by_validator"]
    assert not verified[str(k)]["semantic_corner_identity_verified"]


def test_optional_junction_rejects_usable_probability_substitution():
    image, q, K, initial, observation, _ = full_junction_fixture()
    observation["visibility_appearance"]["diagnostics"]["joint_observation"] = {"target": "visible_and_accurate"}
    with pytest.raises(ValueError, match="not P_usable"):
        I.infer_interior_observations(image, q, K, initial, observation)


def test_paper_coordinate_compatibility_uses_correct_general_physical_row_basis():
    root = Path(__file__).resolve().parents[3]
    runner = root / "_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/anchored_completion_v1/experiment.py"
    compatible = runpy.run_path(str(runner))["compatible"]
    phase = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))[0]
    physical_basis = cv2.Rodrigues(np.array([.3, -.2, .1]))[0]
    native = C.cuboid(1.2, .3, 1.)
    reference = C.cuboid(1., .3, 1.2)
    mapping = np.argmin(np.linalg.norm((native @ phase)[:, None] - reference[None], axis=2), axis=1)
    assert sorted(mapping.tolist()) == list(range(8))
    # Both descriptions project exactly the same camera corners and physical pose.
    np.testing.assert_allclose(native, reference[mapping] @ phase.T, atol=1e-15, rtol=0)
    initial = dict(R_cf=np.eye(3).tolist(), R_physical=physical_basis.tolist(), cf_extents=[1.2, .3, 1.])
    truth = dict(body_R=phase.tolist(), R=physical_basis.tolist(), body_xyz=[1., .3, 1.2], order=2)
    assert compatible(initial, truth, mapping)
    # An actual width mismatch must still be rejected without a GT-selected permutation.
    wrong = dict(truth, body_xyz=[1., .3, 1.4])
    assert not compatible(initial, wrong, mapping)
