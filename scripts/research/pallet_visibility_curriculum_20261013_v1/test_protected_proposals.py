"""Finite pose selection, immutable observations and nonrigid 2D readout."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from . import anchored_completion as A
from . import curriculum as M
from . import protected_proposals as P
from .test_anchored_completion import inputs


def biased_interior_seed():
    xyz, R, t, K, R0, t0, obs = inputs(5, rotation_error=.008)
    for k in np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"])):
        obs["raw_points"][k] = (np.asarray(obs["raw_points"][k])+[16., -9.]).tolist()
        obs["fused_points"][k] = obs["raw_points"][k][:]
    base = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert base["accepted"] and base["final_guard"]["independent_boundary_seed_exception"]
    return xyz, K, R0, t0, base


def proposal(base, xyz, delta=.002, name="LEGACY_PROGRESSIVE"):
    return dict(name=name, R=deepcopy(base["R"]),
        t=(np.asarray(base["t"])+[delta, -.5*delta, 0.]).tolist(),
        cf_extents=np.ptp(xyz, axis=0).tolist(), coordinate_frame="ORIGINAL_NATIVE_CF",
        native_order_preserved=True)


def test_best_protected_proposal_requires_dual_gain_and_does_not_refit_or_change_q():
    xyz, K, R0, t0, base = biased_interior_seed()
    original = deepcopy(base)
    candidates = [proposal(base, xyz, .001), proposal(base, xyz, .002, "LEARNED_JOINT")]
    before = deepcopy(candidates)
    with patch.object(M, "_fit", side_effect=AssertionError("No new pose fitting allowed")):
        selected = P.select_protected_pose(base, xyz, K, R0, t0, candidates)
    assert selected["proposal_selected"] and selected["selected_pose_source"] == "LEARNED_JOINT"
    np.testing.assert_array_equal(selected["t"], candidates[1]["t"])
    assert selected["used_ids"] == base["used_ids"]
    assert selected["estimated_dof"] == base["estimated_dof"]
    assert selected["anchor_reference"] == base["anchor_reference"]
    assert selected["measured_points"] == base["measured_points"]
    assert selected["frozen_pool_weights"] == base["frozen_pool_weights"]
    assert selected["final_anchor_guard"]["usable"]
    assert not selected["final_guard"]["independent_boundary_seed_exception"]
    assert selected["base_final_guard"] == base["final_guard"]
    assert selected["uncertainty"]["estimated_parameter_count"] == base["estimated_dof"]
    assert base == original and candidates == before


def test_current_gain_without_initial_gain_does_not_inherit_seed_exception():
    xyz, K, R0, t0, base = biased_interior_seed()
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [proposal(base, xyz, .0001)])
    audit = selected["proposal_audit"][0]
    assert audit["current_objective_gain"] > 1e-7
    assert audit["initial_objective_gain"] < 0
    assert audit["reason"] == "no_initial_pool_objective_gain"
    assert not selected["proposal_selected"]
    assert selected["R"] == base["R"] and selected["t"] == base["t"]
    assert selected["final_guard"] == base["final_guard"]
    assert selected["final_guard"]["independent_boundary_seed_exception"]


def test_pool_gain_cannot_compensate_for_a_degraded_individual_anchor():
    xyz, K, R0, t0, base = biased_interior_seed()
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [proposal(base, xyz, .02)])
    audit = selected["proposal_audit"][0]
    assert audit["current_objective_gain"] > 0 and audit["initial_objective_gain"] > 0
    assert audit["reason"] == "protected_anchor_budget_exceeded"
    assert any(not r["per_point_loss_pass"] for r in audit["anchor_guard"]["records"])
    assert not selected["proposal_selected"]


def test_three_DOF_keeps_exact_initial_R_and_original_support_despite_larger_pool():
    xyz, R, t, K, R0, t0, obs = inputs(3, rotation_error=.006)
    # The corrupted supplement is eligible but cannot pass the fixed anchor
    # guard. It must not silently promote retained three-point support to 6DOF.
    for k in np.flatnonzero(~np.asarray(obs["boundary_visibility"]["expected_outline"])):
        obs["raw_points"][k] = (np.asarray(obs["raw_points"][k])+[12., -8.]).tolist()
        obs["fused_points"][k] = obs["raw_points"][k][:]
    base = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    assert base["estimated_dof"] == 3 and len(base["immutable_pool_ids"]) > 3
    candidate = proposal(base, xyz, .001)
    candidate["R"] = (cv2.Rodrigues(np.array([.08, -.03, .01]))[0] @ R0).tolist()
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [candidate])
    audit = selected["proposal_audit"][0]
    assert audit["proposal_rotation_discarded"]
    np.testing.assert_array_equal(audit["evaluated_R"], R0)
    np.testing.assert_array_equal(selected["R"], R0)
    assert selected["used_ids"] == base["used_ids"] and selected["estimated_dof"] == 3


@pytest.mark.parametrize("bad", ["rotation", "frame", "dimensions", "order", "NaN"])
def test_invalid_pose_or_model_binding_is_audited_and_preserves_base(bad):
    xyz, K, R0, t0, base = biased_interior_seed()
    candidate = proposal(base, xyz)
    if bad == "rotation":
        candidate["R"] = (np.asarray(candidate["R"])*1.01).tolist()
    elif bad == "frame":
        candidate["coordinate_frame"] = "PHYSICAL_BODY"
    elif bad == "dimensions":
        candidate["cf_extents"][0] += .01
    elif bad == "order":
        candidate["native_order_preserved"] = False
    else:
        candidate["t"][2] = np.nan
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [candidate])
    assert selected["proposal_audit"][0]["reason"] == "invalid_pose_proposal"
    assert selected["R"] == base["R"] and selected["t"] == base["t"]
    assert selected["final_guard"] == base["final_guard"]


def test_malformed_candidate_cannot_crash_or_lose_a_later_valid_proposal():
    xyz, K, R0, t0, base = biased_interior_seed()
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [None, [], proposal(base, xyz)])
    assert [a["reason"] for a in selected["proposal_audit"][:2]] == ["invalid_pose_proposal"]*2
    assert selected["proposal_selected"] and selected["proposal_audit"][2]["selected"]


def test_fixed_initial_reference_and_anchor_reference_cannot_be_rebound():
    xyz, K, R0, t0, base = biased_interior_seed()
    with pytest.raises(ValueError, match="frozen V1 initial"):
        P.select_protected_pose(base, xyz, K, R0, t0+[.001, 0, 0], [])
    bad = deepcopy(base)
    bad["anchor_reference"] = None
    with pytest.raises(ValueError, match="frozen anchor reference"):
        P.select_protected_pose(bad, xyz, K, R0, t0, [])


def test_extra_proposal_coordinates_are_never_measurements_and_equal_cost_is_rejected():
    xyz, K, R0, t0, base = biased_interior_seed()
    candidate = proposal(base, xyz, 0.)
    candidate["points"] = np.full((9, 2), np.nan).tolist()
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [candidate])
    assert not selected["proposal_selected"]
    assert selected["proposal_audit"][0]["reason"] == "no_current_pool_objective_gain"
    assert selected["measured_points"] == base["measured_points"]
    assert selected["proposal_audit"][0]["extra_coordinates_used_as_measurements"] is False


def test_two_or_fewer_pose_support_stays_initial_while_readout_keeps_real_anchors():
    xyz, R, t, K, R0, t0, obs = inputs(2)
    base = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"supplement": False})
    selected = P.select_protected_pose(base, xyz, K, R0, t0, [proposal(base, xyz)])
    assert selected["proposal_audit"][0]["reason"] == "insufficient_retained_independent_support"
    np.testing.assert_array_equal(selected["R"], R0)
    np.testing.assert_array_equal(selected["t"], t0)
    readout = P.state_keypoint_readout(selected)
    assert readout["preserved_measurement_ids"] == base["anchor_ids"]
    for k in base["anchor_ids"]:
        assert readout["reported_points"][k] == base["measured_points"][k]
    for k in set(range(8))-set(base["anchor_ids"]):
        assert readout["reported_points"][k] == selected["projected_points"][k]
    assert readout["reported_points"][8] == base["raw_points"][8]
    assert not readout["mixed_points_are_one_rigid_pose"]


def test_readout_uses_only_authoritative_anchor_or_actual_used_visible_roles():
    xyz, R, t, K, R0, t0, obs = inputs(5)
    no_anchors = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"anchors": False, "supplement": False})
    assert no_anchors["boundary_seed"]["evidence"]["strong_ids"]
    assert P.state_keypoint_readout(no_anchors)["preserved_measurement_ids"] == []
    base = A.solve_anchored_completion(obs, xyz, K, R0, t0)
    readout = P.state_keypoint_readout(base)
    assert set(readout["preserved_measurement_ids"]) == set(base["anchor_ids"]) | set(base["used_ids"])
    for k in readout["preserved_measurement_ids"]:
        assert readout["reported_points"][k] == base["measured_points"][k]
    bad = deepcopy(base)
    bad["keypoint_status"][base["anchor_ids"][0]]["self_occluded_conditional"] = True
    with pytest.raises(ValueError, match="non-excluded authoritative"):
        P.state_keypoint_readout(bad)


def test_readout_rejects_slot_or_center_mismatch():
    xyz, R, t, K, R0, t0, obs = inputs(2)
    base = A.solve_anchored_completion(obs, xyz, K, R0, t0, {"supplement": False})
    for mode in ("slots", "center"):
        bad = deepcopy(base)
        if mode == "slots":
            bad["raw_points"] = bad["raw_points"][:8]
        else:
            bad["measured_points"][8][0] += .01
        with pytest.raises(ValueError):
            P.state_keypoint_readout(bad)
