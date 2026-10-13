"""Synthetic, prediction-free checks for the new simultaneous-domain solver."""
from copy import deepcopy
from itertools import product
import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .joint_ot_pnp import project_native, solve_joint_ot_pnp


# Historical four-root fixture proves the algebraic domain intersection. Its
# reference BI0 is model-self-hidden, so explicitly retain legacy semantics
# here; the physically visible fixture below tests the corrected defaults.
FAST = {"global_rotation_starts": 0, "max_joint_iterations": 150,
        "protect_seed_projection": True, "enforce_visible_geometry": False,
        "enforce_visible_canvas": False}


def scene():
    # Known native ordering and original dimensions: no generated image or GT.
    xyz = np.array(list(product([-.5, .5], [-.2, .2], [-.6, .6])))
    K = np.array([[600., 25., 320.], [0., 620., 240.], [0., 0., 1.]])
    R = Rotation.from_rotvec([-2.796081998477284, -1.1554161099022129,
                             .4701428657425449]).as_matrix()
    t = np.array([.261626142029378, -.2012353094507891, 3.1347318581769867])
    q, _ = project_native(xyz, R, t, K)
    return xyz, K, R, t, q


def visible(q, ids, protection=.05):
    return [{"native_corner": i, "xy": q[i].tolist(), "covariance_px2": [[1., 0.], [0., 1.]],
             "protection_px": protection, "provenance": {"coordinate_source": "corrected_image_BI"}}
            for i in ids]


def states_for(ids, domains=()):
    states = ["S"] * 8
    for i in ids:
        states[i] = "B" if i % 2 == 0 else "I"
    for domain in domains:
        states[domain["native_corner"]] = domain["state"]
    return states


def p3p_roots(xyz, K, q, ids):
    homogeneous = np.c_[q[ids], np.ones(len(ids))] @ np.linalg.inv(K).T
    normalized = homogeneous[:, :2] / homogeneous[:, 2:]
    count, rotations, translations = cv2.solveP3P(
        xyz[ids], normalized, np.eye(3), None, flags=cv2.SOLVEPNP_P3P)
    roots = [(cv2.Rodrigues(r)[0], t.reshape(3)) for r, t in zip(rotations, translations)]
    assert count == 4  # This scene independently exhibits the P3P ambiguity.
    return roots


def segment_ellipse(native_id, first, second):
    delta = second - first
    return {"native_corner": native_id, "state": "O", "shape": "ellipse",
            "center_xy": ((first + second) / 2).tolist(),
            "half_axes_px": [np.linalg.norm(delta) / 2 + 3., 2.],
            "angle_deg": float(np.rad2deg(np.arctan2(delta[1], delta[0])))}


def ellipse_contains(domain, xy):
    angle = np.deg2rad(domain["angle_deg"])
    basis = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    normalized = basis.T @ (xy - domain["center_xy"]) / domain["half_axes_px"]
    return bool(normalized @ normalized <= 1.)


def test_two_O_regions_reject_complementary_P3P_branches_jointly():
    xyz, K, R, t, q = scene()
    ids = [0, 2, 6]
    roots = p3p_roots(xyz, K, q, ids)
    projections = [project_native(xyz, root_R, root_t, K)[0] for root_R, root_t in roots]
    true = int(np.argmin([np.linalg.norm(root_R - R) for root_R, _ in roots]))
    rivals = [i for i in range(4) if i != true]
    # O1 admits true + rival A. O5 admits true + rival B. Each is deliberately
    # insufficient alone; their common feasible finite branch is the true one.
    d1 = segment_ellipse(1, q[1], projections[rivals[0]][1])
    d2 = segment_ellipse(5, q[5], projections[rivals[1]][5])
    admission = [[ellipse_contains(d, p[d["native_corner"]]) for p in projections]
                 for d in [d1, d2]]
    assert sum(admission[0]) == 2 and sum(admission[1]) == 2
    assert [i for i in range(4) if admission[0][i] and admission[1][i]] == [true]
    first = solve_joint_ot_pnp(xyz, K, states_for(ids, [d1]), visible(q, ids), [d1], [480, 640], config=FAST)
    second = solve_joint_ot_pnp(xyz, K, states_for(ids, [d2]), visible(q, ids), [d2], [480, 640], config=FAST)
    joint = solve_joint_ot_pnp(xyz, K, states_for(ids, [d1, d2]), visible(q, ids), [d1, d2],
                               [480, 640], config=FAST)
    assert first["accepted"] and first["ambiguity"]["detected"]
    assert second["accepted"] and second["ambiguity"]["detected"]
    assert joint["accepted"] and not joint["ambiguity"]["detected"]
    assert {a["native_corner"] for a in joint["joint_constraint_audit"]} == {1, 5}
    assert all(a["feasible"] for a in joint["joint_constraint_audit"])
    np.testing.assert_allclose(joint["R"], R, atol=2e-5, rtol=0)
    np.testing.assert_allclose(joint["t"], t, atol=2e-5, rtol=0)


def test_three_BI_fits_rotation_without_initial_pose_trust_veto():
    xyz, K, R, t, q = scene()
    ids = [0, 2, 6]
    domain = {"native_corner": 5, "state": "O", "shape": "box",
              "bounds_xyxy": [q[5, 0] - 1, q[5, 1] - 1, q[5, 0] + 1, q[5, 1] + 1]}
    bad_R = Rotation.from_euler("x", 50, degrees=True).as_matrix() @ R
    output = solve_joint_ot_pnp(xyz, K, states_for(ids, [domain]), visible(q, ids), [domain], [480, 640],
                               initializations=[{"R": bad_R, "t": t}], config=FAST)
    assert output["accepted"]
    fitted = np.asarray(output["R"])
    rotation_from_start = np.rad2deg(Rotation.from_matrix(fitted @ bad_R.T).magnitude())
    assert rotation_from_start > 45
    assert output["rank"]["visible"]["rank"] == 6
    assert output["provenance"]["rotation_fixed_for_three_points"] is False
    np.testing.assert_allclose(fitted, R, atol=2e-5)


def test_BI_noisy_observations_are_preserved_but_not_hard_clamped():
    xyz, K, _, _, q = scene()
    noisy = q.copy()
    noisy[:6] += np.array([[.8, -.2], [-.4, .6], [.2, .9], [-.7, -.5], [.3, -.6], [.1, .4]])
    ids = list(range(6))
    measurements = visible(noisy, ids, protection=3.)
    before = deepcopy(measurements)
    output = solve_joint_ot_pnp(xyz, K, states_for(ids), measurements, [], [480, 640],
                               config={**FAST, "max_p3p_triples": 2})
    assert output["accepted"] and measurements == before
    np.testing.assert_array_equal(np.asarray(output["reported_native"])[ids], noisy[ids])
    assert np.max(np.linalg.norm(np.asarray(output["rigid_projection8"])[ids] - noisy[ids], axis=1)) > .1
    assert output["mixed_readout_is_rigid_pose"] is False
    assert all(a["protection_pass"] for a in output["visible_protection_audit"])
    json.dumps(output, allow_nan=False)


def test_T_unknown_side_enumerates_union_and_intersects_spatial_halfplanes():
    xyz, K, _, _, _ = scene()
    R = Rotation.from_euler("y", .7).as_matrix(); t = np.array([1., 0., 2.8])
    q, _ = project_native(xyz, R, t, K)
    ids = [0, 1, 2, 3, 4, 6]
    domain = {"native_corner": 5, "state": "T", "shape": "halfplane",
              "halfplanes": [[0, 1, 210], [0, -1, -180]], "outside_margin_px": 0}
    output = solve_joint_ot_pnp(xyz, K, states_for(ids, [domain]), visible(q, ids), [domain], [480, 640],
                               config={**FAST, "max_p3p_triples": 1})
    assert output["accepted"]
    assert output["search"]["T_union_branches_required"] == 4
    audit = output["joint_constraint_audit"][0]
    assert audit["outside_canvas_side"] == "right" and audit["feasible"]
    assert np.asarray(output["projected_native"])[5, 0] >= 640
    assert 180 <= np.asarray(output["projected_native"])[5, 1] <= 210
    assert len(audit["inequalities"]) == 3


def test_infeasible_joint_domain_is_not_silently_replaced_by_BI_seed():
    xyz, K, _, _, q = scene()
    ids = [0, 1, 2, 3, 4, 6]
    domain = {"native_corner": 5, "state": "O", "shape": "ellipse",
              "center_xy": [q[5, 0] + 80, q[5, 1]], "half_axes_px": [1., 1.]}
    output = solve_joint_ot_pnp(xyz, K, states_for(ids, [domain]), visible(q, ids), [domain], [480, 640],
                               config={**FAST, "max_p3p_triples": 1})
    assert not output["accepted"]
    assert output["R"] is None and output["projected_native"] is None
    assert output["corrected_BI_seed"]["admissible_as_joint_fallback"] is False
    assert output["search"]["global_feasibility_proved"] is False
    assert output["status"] == "no_joint_feasible_pose_found"
    assert all(not a["joint_feasible"] for a in output["candidates"])
    for native_id in ids:
        np.testing.assert_array_equal(output["reported_native"][native_id], q[native_id])


def test_S_and_U_cannot_supply_region_or_fit_coordinates():
    xyz, K, _, _, q = scene()
    ids = [0, 2, 6]
    states = states_for(ids)
    states[7] = "U"
    for native_id in [1, 7]:
        for perturbation in [0, 1e9]:
            domain = {"native_corner": native_id, "state": states[native_id], "shape": "ellipse",
                      "center_xy": [perturbation, -perturbation], "half_axes_px": [1., 1.]}
            with pytest.raises(ValueError, match="Only final O/T"):
                solve_joint_ot_pnp(xyz, K, states, visible(q, ids), [domain], [480, 640], config=FAST)
            with pytest.raises(ValueError, match="Only unique final B/I"):
                solve_joint_ot_pnp(xyz, K, states, visible(q, ids + [native_id]), [], [480, 640], config=FAST)
    output = solve_joint_ot_pnp(xyz, K, states, visible(q, ids), [], [480, 640], config=FAST)
    assert output["provenance"]["S_image_terms"] == 0
    assert output["provenance"]["S_region_terms"] == 0
    for native_id in [1, 3, 4, 5, 7]:
        assert output["output_corner_sources"][native_id] == "RIGID_MODEL_PROJECTION_ONLY"


def test_every_OT_domain_is_required_and_native_projection_uses_original_basis():
    xyz, K, _, _, q = scene()
    ids = [0, 2, 6]
    states = states_for(ids); states[5] = "O"
    with pytest.raises(ValueError, match="Every final O/T"):
        solve_joint_ot_pnp(xyz, K, states, visible(q, ids), [], [480, 640], config=FAST)
    output = solve_joint_ot_pnp(xyz, K, states_for(ids), visible(q, ids), [], [480, 640], config=FAST)
    reprojection, depth = project_native(xyz, np.array(output["R"]), np.array(output["t"]), K)
    np.testing.assert_allclose(output["rigid_projection8"], reprojection, atol=1e-10)
    assert np.all(depth > 0)
    assert output["ambiguity"]["detected"]  # Full local rank does not prove uniqueness.
    assert output["ambiguity"]["global_uniqueness_proved"] is False


def test_all_S_has_no_image_term_and_no_invented_pose():
    xyz, K, R, t, _ = scene()
    first = solve_joint_ot_pnp(xyz, K, ["S"] * 8, [], [], [480, 640], config=FAST)
    # Arbitrarily perturbed prior starts have no admitted image coordinates.
    second = solve_joint_ot_pnp(xyz, K, ["S"] * 8, [], [], [480, 640],
                               initializations=[{"R": R, "t": t + [10, 20, 30]}], config=FAST)
    assert first == second
    assert not first["accepted"] and first["reported_native"] == [None] * 8


def test_seed_shift_protection_cannot_hide_inconsistent_BI_measurements():
    xyz, K, _, _, q = scene()
    ids = list(range(6))
    inconsistent = q.copy()
    inconsistent[4] += [24., -20.]
    records = visible(inconsistent, ids, protection=2.)
    relaxed = solve_joint_ot_pnp(xyz, K, states_for(ids), records, [], [480, 640],
        config={**FAST, "max_p3p_triples": 2, "maximum_visible_mahalanobis": 100.})
    bounded = solve_joint_ot_pnp(xyz, K, states_for(ids), records, [], [480, 640],
        config={**FAST, "max_p3p_triples": 2, "maximum_visible_mahalanobis": 3.})
    assert relaxed["accepted"]
    assert all(a["protection_pass"] for a in relaxed["visible_protection_audit"])
    assert max(a["measurement_mahalanobis"] for a in relaxed["visible_protection_audit"]) > 10
    assert not bounded["accepted"] and bounded["R"] is None
    assert not bounded["corrected_BI_seed"]["visible_measurement_budgets_pass"]
    assert bounded["fallback_diagnostics"]["corrected_BI_seed_failed_measurement_ids"]
    for native_id in ids:
        np.testing.assert_array_equal(bounded["reported_native"][native_id], inconsistent[native_id])
