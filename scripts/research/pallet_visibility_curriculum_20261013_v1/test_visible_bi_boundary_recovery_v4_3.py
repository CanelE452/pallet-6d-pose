"""Boundary recovery evidence, finite uncertainty, and excluded-state invariants."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import common as C
from .test_visible_bi_correction_v2 import scene
from .visible_bi_correction_v2 import correct_visible_bi, DEFAULT_CONFIG as LEGACY
from .visible_bi_boundary_recovery_v4_3 import (
    recover_visible_boundary, _external_mask_and_contours, _shared_edge_consistency)


def displaced_scene():
    image, mask, q, K, X, pose, states, P = scene()
    for k in (1, 2):
        outward = P[k]-P.mean(0)
        q[k] = P[k]+20.*outward/np.linalg.norm(outward)
    base = correct_visible_bi(image, q, K, pose, states, pallet_mask=mask, native_xyz=X)
    assert not base["records"][1]["corrected"] and not base["records"][2]["corrected"]
    return image, mask, q, K, X, pose, states, P, base


def short_edge_scene():
    X = C.cuboid(1.1, .11, 1.3)
    R = Rotation.from_euler("xyz", [0, -35, 0], degrees=True).as_matrix()
    t = np.array([.1, .15, 3.5])
    K = np.array([[614., 0., 329.], [0., 614., 234.], [0., 0., 1.]])
    P = C.project(X, R, t, K)
    image = np.full((480, 640, 3), 230, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    faces = [([0, 1, 2, 3], [0, 0, -1]), ([4, 5, 6, 7], [0, 0, 1]),
             ([0, 1, 5, 4], [0, -1, 0]), ([3, 2, 6, 7], [0, 1, 0]),
             ([0, 3, 7, 4], [-1, 0, 0]), ([1, 2, 6, 5], [1, 0, 0])]
    for j, (ids, normal) in enumerate(faces):
        if np.asarray(normal)@R.T@(-t) > 0:
            poly = np.rint(P[ids]).astype(np.int32)
            cv2.fillConvexPoly(image, poly, (40+10*j, 90+10*j, 60))
            cv2.fillConvexPoly(mask, poly, 1)
    hull = set(cv2.convexHull(P.astype(np.float32), returnPoints=False).ravel())
    states = ["S" if C.self_hidden(X, R, t)[k] else ("B" if k in hull else "I") for k in range(8)]
    pose = dict(R_cf=R.tolist(), centroid=t.tolist(), cf_extents=[1.1, .11, 1.3])
    q = np.vstack([P, [320., 240.]])
    # A synthetic displacement leaves six of fifteen along-edge samples;
    # production reach is contour-derived and has no fixed 24px parameter.
    q[1] += [24., -4.]; q[2] += [24., 8.]
    base = correct_visible_bi(image, q, K, pose, states, pallet_mask=mask, native_xyz=X)
    return image, mask.astype(float), q, K, X, pose, states, P, base


def run(args, **cfg):
    image, mask, q, K, X, pose, states, _, base = args
    return recover_visible_boundary(image, q, K, pose, states, pallet_mask=mask,
                                    base_correction=base, native_xyz=X, config=cfg)


def test_beyond_local_radius_is_actual_image_intersection_with_fresh_covariance():
    args = displaced_scene(); output = run(args)
    for k in (1, 2):
        row = output["boundary_recovery_records"][k]
        assert row["recovered"] and row["image_derived_reach_px"] > LEGACY["normal_search_radius_px"]
        assert row["blend_alpha"] == 1. and not row["original_12px_trust_would_pass"]
        assert np.linalg.norm(np.asarray(output["measured_points"][k])-args[7][k]) < 2.
        np.testing.assert_array_equal(output["measured_points"][k], row["intersection_xy"])
        np.testing.assert_array_equal(row["covariance_px2"], row["conditional_covariance"]["covariance_px2"])
        assert np.linalg.eigvalsh(row["covariance_px2"]).min() > 0
        assert row["conditional_covariance"]["successful_pair_resamples"] >= 16
        for ray in row["selected_rays"]:
            assert ray["supported"] and ray["inlier_count"] >= LEGACY["minimum_inlier_count"]
            assert ray["inlier_fraction"] >= LEGACY["minimum_inlier_fraction"]
            assert ray["inlier_span_fraction"] >= LEGACY["minimum_inlier_span_fraction"]
            assert ray["inlier_rms_px"] <= LEGACY["maximum_line_rms_px"]
            assert ray["angle_change_deg"] <= LEGACY["maximum_direction_change_deg"]
    assert output["observed_shared_edge_audit"]
    assert all(r["supported"] for r in output["observed_shared_edge_audit"])


def test_recentering_restores_coverage_without_lowering_line_thresholds():
    args = short_edge_scene()
    wide_only = run(args, recenter_on_observed_line=False)
    output = run(args)
    assert not wide_only["boundary_recovery_records"][2]["recovered"]
    row = output["boundary_recovery_records"][2]
    assert row["recovered"]
    assert any(r["reason"] == "insufficient_along_edge_coverage" for r in row["expanded_rays"])
    assert any(a["remeasured_ray"].get("inlier_fraction", 0) >= LEGACY["minimum_inlier_fraction"]
               for a in row["observed_line_recentering"])
    assert np.linalg.norm(np.asarray(output["measured_points"][2])-args[7][2]) < 2.


def test_successful_original_B_and_all_I_are_exactly_preserved_and_inputs_immutable():
    args = displaced_scene(); original = deepcopy(args[8]); q_before = args[2].copy()
    result = run(args)
    for old, new in zip(original["visible_measurements"], result["visible_measurements"]):
        k = old["native_corner"]
        if args[6][k] == "I" or original["records"][k]["corrected"]:
            assert old == new
            assert original["records"][k] == result["records"][k]
    assert args[8] == original
    np.testing.assert_array_equal(args[2], q_before)
    np.testing.assert_array_equal(result["measured_points"][8], q_before[8])


def test_S_O_T_U_coordinates_do_not_affect_recovery_or_covariance():
    args = list(displaced_scene())
    states = args[6].copy(); states[3] = "O"; states[4] = "T"; states[5] = "U"
    args[6] = states
    args[8] = correct_visible_bi(args[0], args[2], args[3], args[5], states,
                                 pallet_mask=args[1], native_xyz=args[4])
    first = run(args)
    args[2] = args[2].copy()
    for k, state in enumerate(states):
        if state in ("S", "O", "T", "U"):
            args[2][k] = [1e9, -1e9]
    args[8] = correct_visible_bi(args[0], args[2], args[3], args[5], states,
                                 pallet_mask=args[1], native_xyz=args[4])
    second = run(args)
    assert first["visible_measurements"] == second["visible_measurements"]
    assert first["boundary_recovery_records"] == second["boundary_recovery_records"]
    assert first["observed_shared_edge_audit"] == second["observed_shared_edge_audit"]
    assert set(second["recovery_queried_native_ids"]) <= {k for k, s in enumerate(states) if s == "B"}


def test_mask_without_actual_RGB_transition_cannot_recover():
    args = list(displaced_scene()); args[0] = np.full_like(args[0], 127)
    args[8] = correct_visible_bi(args[0], args[2], args[3], args[5], args[6], pallet_mask=args[1], native_xyz=args[4])
    output = run(args)
    assert output["recovered_native_ids"] == []
    np.testing.assert_array_equal(output["measured_points"], args[8]["measured_points"])
    assert any(r.get("reason") == "insufficient_actual_RGB_boundary_samples"
               for row in output["boundary_recovery_records"] for r in row.get("expanded_rays", []))


def test_empty_mask_does_not_substitute_model_projected_corner():
    args = list(displaced_scene()); args[1] = np.zeros_like(args[1])
    args[8] = correct_visible_bi(args[0], args[2], args[3], args[5], args[6], pallet_mask=args[1], native_xyz=args[4])
    output = run(args)
    assert output["recovered_native_ids"] == [] and output["recovery_queried_native_ids"] == []
    np.testing.assert_array_equal(output["measured_points"], args[8]["measured_points"])


def test_interior_mask_holes_are_not_exterior_line_evidence():
    mask = np.zeros((80, 80), float); mask[10:70, 10:70] = 1.; mask[30:50, 30:50] = 0.
    exterior, contours = _external_mask_and_contours(mask)
    assert len(contours) == 1 and exterior[40, 40] == 1
    assert exterior[0, 0] == 0


def test_explicit_cap_and_alpha_controls_preserve_candidate_not_hide_the_old_veto():
    args = displaced_scene(); full = run(args)
    capped = run(args, maximum_corner_shift_px=12.)
    half = run(args, maximum_alpha=.5)
    for k in (1, 2):
        assert capped["boundary_recovery_records"][k]["reason"] == "explicit_recovery_corner_shift_cap"
        assert not capped["boundary_recovery_records"][k]["recovered"]
        assert full["boundary_recovery_records"][k]["intersection_xy"] == half["boundary_recovery_records"][k]["intersection_xy"]
        assert full["boundary_recovery_records"][k]["conditional_covariance"] == half["boundary_recovery_records"][k]["conditional_covariance"]
        np.testing.assert_allclose(half["measured_points"][k], .5*(args[2][k]+full["measured_points"][k]))


def test_shared_endpoint_line_disagreement_is_rejected():
    a = dict(native_neighbor=2, normal=[1., 0.], offset=-10.)
    b = dict(native_neighbor=3, normal=[0., 1.], offset=-20.)
    c = dict(native_neighbor=1, normal=[1., 0.], offset=-15.)
    d = dict(native_neighbor=0, normal=[0., 1.], offset=-40.)
    proposals = {1: dict(first=a, second=b, point=np.array([10., 20.])),
                 2: dict(first=c, second=d, point=np.array([15., 40.]))}
    audit, invalid = _shared_edge_consistency(proposals, 12.)
    assert invalid == {1, 2} and not audit[0]["supported"]


@pytest.mark.parametrize("mutation,match", [("image", "image binding"), ("raw", "raw native"),
                                              ("states", "state/native"), ("native", "Registered native")])
def test_bound_inputs_and_original_native_order_are_required(mutation, match):
    image, mask, q, K, X, pose, states, _, base = displaced_scene()
    if mutation == "image": image = image.copy(); image[0, 0] = 0
    if mutation == "raw": q = q.copy(); q[1, 0] += 1
    if mutation == "states": states = states.copy(); states[1] = "O"
    if mutation == "native": X = X[::-1]
    with pytest.raises(ValueError, match=match):
        recover_visible_boundary(image, q, K, pose, states, pallet_mask=mask, base_correction=base, native_xyz=X)


def test_unknown_config_does_not_silently_change_guards():
    args = displaced_scene()
    with pytest.raises(ValueError, match="Unknown boundary"):
        run(args, minimum_inlier_fraction=.1)


def test_post_correction_unqueried_T_to_U_keeps_authoritative_state_and_all_BI_exact():
    args = list(displaced_scene())
    states = args[6].copy(); states[4] = "T"
    base = correct_visible_bi(args[0], args[2], args[3], args[5], states,
                              pallet_mask=args[1], native_xyz=args[4])
    before = deepcopy(base)
    states[4] = "U"; args[6] = states; args[8] = base
    output = run(args)
    assert output["records"][4]["state"] == "U" and not output["records"][4]["pixel_queried"]
    assert output["records"][4]["original_correction_state"] == "T"
    assert output["provenance"]["recovery_post_correction_excluded_T_to_U_native_ids"] == [4]
    assert base == before
    for old, new in zip(before["visible_measurements"], output["visible_measurements"]):
        if before["records"][old["native_corner"]]["corrected"] or old["state"] == "I":
            assert old == new


def test_other_excluded_transitions_and_BI_role_metadata_mismatch_reject():
    args = list(displaced_scene())
    states = args[6].copy(); states[4] = "O"
    args[8] = correct_visible_bi(args[0], args[2], args[3], args[5], states,
                                 pallet_mask=args[1], native_xyz=args[4])
    states[4] = "U"; args[6] = states
    with pytest.raises(ValueError, match="state/native"):
        run(args)
    args = list(displaced_scene())
    args[8]["visible_measurements"][0]["state"] = "B"  # Native0 is I.
    with pytest.raises(ValueError, match="state/native"):
        run(args)
