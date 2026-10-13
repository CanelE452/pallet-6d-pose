"""Focused runtime contract checks using generated pixels and no labels/files."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import common as C
from .ot_domains_v2 import apply_real_crop, fit_external_o_error_model
from .state_visibility_v2 import StateVisibilityModel, geometry_state_gates
from .statewise_joint_api_v2 import StatewiseJointV2, registered_geometry


FAST = {"solver": {"global_rotation_starts": 0, "max_p3p_triples": 2,
                   "max_joint_iterations": 100}, "bi": {"line_resampling": 16}}


def inputs():
    image = np.full((480, 640, 3), 75, dtype=np.uint8)
    K = np.array([[600., 15., 320.], [0., 620., 240.], [0., 0., 1.]])
    xyz = C.cuboid(1.3, .11, 1.1)
    R = Rotation.from_euler("xyz", [.25, -.3, .05]).as_matrix()
    Q = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    t = np.array([0., 0., 3.2])
    coarse = dict(R_cf=R.tolist(), R_physical=(R @ Q).tolist(), centroid=t.tolist(),
                  cf_extents=[1.3, .11, 1.1], selected_hypothesis="registered_native_test")
    raw = np.vstack([C.project(xyz, R, t, K), [312.3, 247.8]])
    return image, raw, K, xyz, coarse


class StateProducer:
    def __init__(self, state_overrides=None):
        self.overrides = state_overrides or {}
        self.calls = []

    def __call__(self, image, raw, hidden, roles, projected, model, *, pallet_mask,
                 independent_truncation=None):
        self.calls.append(dict(raw=raw.copy(), hidden=hidden.copy(), mask=pallet_mask.copy()))
        records, ids = geometry_state_gates(image, raw, hidden, roles, projected,
                                            independent_truncation=independent_truncation)
        for native_id in ids:
            state = self.overrides.get(int(native_id), roles[native_id])
            records[native_id].update(state=state, rgb_evaluated=True,
                                     p_visible=0. if state == "O" else 1.,
                                     reason="injected_test_actual_visibility")
        return dict(states=[r["state"] for r in records], records=records,
                    rgb_queried_ids=ids.tolist(), diagnostics={"test_injection": True})


class Masks:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return dict(state_pallet_mask=np.full(image.shape[:2], .2),
                    pallet_mask=np.full(image.shape[:2], .8),
                    provenance={"Tiny_for_state": True, "SAM_for_boundary": True})


def test_runtime_chains_current_state_BI_and_all_OT_with_registered_physical_basis():
    image, raw, K, xyz, coarse = inputs()
    producer, masks = StateProducer({0: "O", 7: "O"}), Masks()
    before_raw, before_pose = raw.copy(), deepcopy(coarse)
    runtime = StatewiseJointV2(mask_provider=masks, config=FAST, state_producer=producer)
    result = runtime.refine(image, raw, K, xyz, coarse)
    assert result["accepted"]
    assert result["states"][6] == "S"
    assert {d["native_corner"] for d in result["ot_domains"]["domains"]} == {0, 7}
    assert {d["native_corner"] for d in result["joint_pose"]["joint_constraint_audit"]} == {0, 7}
    assert all(d["feasible"] for d in result["joint_pose"]["joint_constraint_audit"])
    assert result["visible_correction"]["queried_native_ids"] == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(producer.calls[0]["mask"], np.full(image.shape[:2], .2))
    assert result["mask_audit"]["shared_mask"] is False
    for measurement in result["frozen_visible_measurements"]:
        native_id = measurement["native_corner"]
        np.testing.assert_array_equal(result["keypoint_readout"]["reported_points"][native_id], measurement["xy"])
    readout = result["keypoint_readout"]
    assert readout["reported_points"][8] == raw[8].tolist()
    assert readout["point_sources"][6] == "FINAL_JOINT_RIGID_S_COMPLETION"
    pose = result["actual_pose"]
    Q = np.array(coarse["R_cf"]).T @ np.array(coarse["R_physical"])
    np.testing.assert_allclose(pose["R_physical"], np.array(pose["R_cf"]) @ Q, atol=1e-12)
    np.testing.assert_array_equal(raw, before_raw)
    assert coarse == before_pose
    assert result["provenance"]["old_boundary_observation_used"] is False
    assert result["provenance"]["S_image_queries"] == 0
    json.dumps(result, allow_nan=False)


def test_S_raw_perturbation_cannot_change_mask_prompts_scale_features_or_pose():
    image, raw, K, xyz, coarse = inputs()
    producer, masks = StateProducer({0: "O", 7: "O"}), Masks()
    runtime = StatewiseJointV2(mask_provider=masks, config=FAST, state_producer=producer)
    first = runtime.refine(image, raw, K, xyz, coarse)
    raw[6] = [1e8, -1e8]
    second = runtime.refine(image, raw, K, xyz, coarse)
    for call in masks.calls:
        assert 6 not in call["prompt_native_ids"]
        assert call["prompt_box_xyxy"] == np.r_[call["prompt_points"].min(0), call["prompt_points"].max(0)].tolist()
    np.testing.assert_array_equal(masks.calls[0]["prompt_points"], masks.calls[1]["prompt_points"])
    np.testing.assert_array_equal(producer.calls[0]["raw"][6], [-1, -1])
    np.testing.assert_array_equal(producer.calls[0]["raw"], producer.calls[1]["raw"])
    assert first["states"] == second["states"]
    assert first["ot_domains"]["nonself_prediction_scale_px"] == second["ot_domains"]["nonself_prediction_scale_px"]
    assert first["frozen_visible_measurements"] == second["frozen_visible_measurements"]
    assert first["joint_pose"] == second["joint_pose"]
    assert first["actual_pose"] == second["actual_pose"]
    assert first["keypoint_readout"] == second["keypoint_readout"]


def test_rejected_joint_keeps_BI_and_S_fallback_without_solving_OT_or_U():
    image, raw, K, xyz, coarse = inputs()
    # Only one B/I measurement survives; O/U must not become fake fit points.
    producer = StateProducer({0: "O", 2: "U", 3: "U", 4: "U", 5: "U", 7: "U"})
    result = StatewiseJointV2(config=FAST, state_producer=producer).refine(
        image, raw, K, xyz, coarse, pallet_mask=np.zeros(image.shape[:2]))
    assert not result["accepted"]
    assert result["joint_pose"]["fallback_reason"] == "fewer_than_two_corrected_BI_for_joint_pose"
    assert result["actual_pose"]["source"] == "explicit_registered_coarse_pose_fallback_S_output_only"
    assert result["actual_pose"]["joint_feasible"] is False
    readout = result["keypoint_readout"]
    for native_id in [0, 2, 3, 4, 5, 7]:
        assert readout["reported_points"][native_id] is None
    assert readout["point_sources"][6] == "EXPLICIT_FALLBACK_RIGID_S_PROJECTION"
    np.testing.assert_allclose(readout["reported_points"][6], C.project(xyz, coarse["R_cf"], coarse["centroid"], K)[6])
    assert readout["reported_points"][8] == raw[8].tolist()
    assert result["fallback_audit"]["O_T_reported_as_solved"] is False


def test_geometry_gated_T_uses_outside_union_without_old_pose_side_veto():
    image, raw, K, xyz, coarse = inputs()
    # An actual cropped canvas changes K and translates existing image queries.
    crop = image[:, 250:]
    crop_K = K.copy(); crop_K[0, 2] -= 250
    crop_raw = raw.copy(); crop_raw[:, 0] -= 250
    result = StatewiseJointV2(config={**FAST, "solver": {**FAST["solver"], "max_joint_iterations": 40}},
        state_producer=StateProducer()).refine(crop, crop_raw, crop_K, xyz, coarse,
                                               pallet_mask=np.zeros(crop.shape[:2]))
    T_domains = [d for d in result["ot_domains"]["domains"] if d["state"] == "T"]
    assert T_domains
    assert all(d["outside_canvas_side"] == "unknown" and d["halfplanes"] == [] for d in T_domains)
    for record in result["state_observation"]["records"]:
        if record["state"] == "T":
            assert record["physical_truncation_certified"] is False
    assert result["joint_pose"]["search"]["T_union_branches_required"] == 4 ** len(T_domains)
    assert result["provenance"]["initial_pose_is_initialization_only"] is True


def test_registered_order_and_state_S_gate_are_checked_before_image_measurements():
    image, raw, K, xyz, coarse = inputs()
    masks = Masks()
    runtime = StatewiseJointV2(mask_provider=masks, config=FAST, state_producer=StateProducer())
    with pytest.raises(ValueError, match="registered original cuboid"):
        runtime.refine(image, raw, K, xyz[[1, 0, 2, 3, 4, 5, 6, 7]], coarse)
    assert not masks.calls

    class InvalidState(StateProducer):
        def __call__(self, *args, **kwargs):
            observation = super().__call__(*args, **kwargs)
            observation["states"][6] = "O"
            observation["records"][6]["state"] = "O"
            return observation

    with pytest.raises(ValueError, match="geometric S must remain fixed"):
        StatewiseJointV2(config=FAST, state_producer=InvalidState()).refine(
            image, raw, K, xyz, coarse, pallet_mask=np.zeros(image.shape[:2]))


def test_registered_geometry_does_not_depend_on_raw_or_metadata():
    image, _, K, xyz, coarse = inputs()
    geometry = registered_geometry(image.shape[:2], K, xyz, coarse)
    assert geometry["initial_self_hidden"].tolist() == [False, False, False, False, False, False, True, False]
    assert geometry["raw_points_used_for_geometry"] is False
    assert geometry["B_I_role_is_geometry_not_actual_visibility"] is True


def test_RGB26_without_bound_segmenter_requires_an_explicit_current_mask():
    image, raw, K, xyz, coarse = inputs()
    model = StateVisibilityModel({"feature_family": "NON_S_RGB26"}, {}, None)
    with pytest.raises(ValueError, match="no bound segmenter; supply a current-RGB"):
        StatewiseJointV2(state_model=model, config=FAST).refine(image, raw, K, xyz, coarse)


def test_crop_binding_is_checked_before_RGB_even_when_no_T_domain_exists():
    image, raw, K, xyz, coarse = inputs()
    crop = apply_real_crop(image, K, raw, [0, 0, 640, 480])
    crop["image_bgr"][0, 0] = 255
    masks = Masks()
    with pytest.raises(ValueError, match="Real crop must bind the actual current RGB"):
        StatewiseJointV2(mask_provider=masks, config=FAST, state_producer=StateProducer()).refine(
            crop["image_bgr"], crop["raw_points"], crop["K"], xyz, coarse,
            crop_transform=crop["transform"])
    assert not masks.calls


def test_inconsistent_BI_seed_does_not_become_an_unchecked_S_fallback_pose():
    image, raw, K, xyz, coarse = inputs()
    raw[0] += [100., -80.]
    result = StatewiseJointV2(config=FAST, state_producer=StateProducer()).refine(
        image, raw, K, xyz, coarse, pallet_mask=np.zeros(image.shape[:2]))
    assert not result["accepted"]
    assert result["fallback_audit"]["corrected_BI_seed_rejected_for_measurement_budget"]
    assert result["actual_pose"]["source"] == "explicit_registered_coarse_pose_fallback_S_output_only"
    np.testing.assert_array_equal(result["actual_pose"]["R_cf"], coarse["R_cf"])
    assert result["keypoint_readout"]["reported_points"][0] == raw[0].tolist()


def test_two_BI_OT_runtime_readout_is_explicitly_a_feasible_family_representative():
    image,raw,K,xyz,coarse=inputs()
    producer=StateProducer({0:"O",3:"O",4:"U",5:"U",7:"U"})
    result=StatewiseJointV2(config={**FAST,"solver":{**FAST["solver"],"analytic_jacobian":True}},
        state_producer=producer).refine(image,raw,K,xyz,coarse,pallet_mask=np.zeros(image.shape[:2]))
    assert result["accepted"] and result["joint_pose"]["status"]=="joint_feasible_pose_family"
    assert result["ambiguity_audit"]["material_ambiguity"]
    readout=result["keypoint_readout"]
    assert readout["OT_constraints_satisfied"] and readout["OT_feasible_hypothesis_available"]
    assert readout["OT_solved"] is False and readout["OT_unique_solution_claimed"] is False
    assert readout["point_sources"][0]=="AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION"
    assert readout["point_sources"][3]=="AMBIGUOUS_JOINT_POSE_FAMILY_PROJECTION"
    assert readout["pose_family_candidates"]
    for measurement in result["frozen_visible_measurements"]:
        np.testing.assert_array_equal(readout["reported_points"][measurement["native_corner"]],measurement["xy"])
    assert result["provenance"]["S_image_queries"]==0


def test_failed_pair_joint_does_not_project_S_from_arbitrary_pair_only_seed():
    image,raw,K,xyz,coarse=inputs()
    # Generated, deliberately incompatible tiny O domains exercise genuine
    # joint failure. These are contract-test arrays, not real CAL data/labels.
    raw[[0,3]]=[50.,50.]
    model=fit_external_o_error_model([dict(split="CAL",state="O",external_label_confirmed=True,
        label_source="HUMAN",raw_xy=[50.,50.],target_xy=[50.,50.],scale_px=100.,
        record_id="synthetic_incompatible_ranges",session="synthetic",image_hw=[480,640],native_corner=0)],
        minimum_radius_px=.01)
    producer=StateProducer({0:"O",3:"O",4:"U",5:"U",7:"U"})
    result=StatewiseJointV2(ot_model=model,
        config={**FAST,"solver":{**FAST["solver"],"analytic_jacobian":True,"global_rotation_starts":4}},
        state_producer=producer).refine(image,raw,K,xyz,coarse,pallet_mask=np.zeros(image.shape[:2]))
    seed=result["joint_pose"]["corrected_BI_seed"]
    assert not result["accepted"] and seed["visible_support_guards_pass"]
    assert seed["contains_underconstrained_pose_family"]
    assert np.linalg.norm(np.array(seed["R"])-np.array(coarse["R_cf"])) > .5
    assert result["fallback_audit"]["corrected_BI_seed_rejected_as_underconstrained_family"]
    assert result["actual_pose"]["source"]=="explicit_registered_coarse_pose_fallback_S_output_only"
    np.testing.assert_array_equal(result["actual_pose"]["R_cf"],coarse["R_cf"])
    np.testing.assert_array_equal(result["actual_pose"]["centroid"],coarse["centroid"])
    np.testing.assert_allclose(result["keypoint_readout"]["reported_points"][6],
                               C.project(xyz,coarse["R_cf"],coarse["centroid"],K)[6],atol=1e-12,rtol=0)
    assert result["keypoint_readout"]["reported_points"][0] is None
    assert result["keypoint_readout"]["reported_points"][3] is None
