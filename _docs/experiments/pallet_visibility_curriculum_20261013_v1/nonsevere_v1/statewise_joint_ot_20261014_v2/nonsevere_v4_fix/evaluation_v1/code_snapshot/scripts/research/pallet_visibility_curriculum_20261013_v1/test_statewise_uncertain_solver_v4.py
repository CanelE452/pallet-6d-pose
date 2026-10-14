"""Mathematical fixtures for the new pure solver; no dataset/GT/image reads."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import joint_ot_pnp as V3
from . import statewise_uncertain_solver_v4 as V4


def fixture(count=4):
    xyz = np.array(list(product([-.4, .4], [-.22, .22], [-.09, .09])))
    R = Rotation.from_euler("xyz", [13, -21, 7], degrees=True).as_matrix()
    t = np.array([.06, -.025, 2.4])
    K = np.array([[720., 17., 320.], [0., 705., 240.], [0., 0., 1.]])
    q, _ = V3.project_native(xyz, R, t, K)
    camera = -R.T @ t
    normal = V3.registered_incident_normals(xyz)
    direct = [i for i, n in enumerate(normal) if np.max(n @ (camera - xyz[i])) > 0]
    # Deterministic native ordering; all observations are physical nonself.
    ids = direct[:count]
    states = ["U"] * 8
    visible = []
    for k, i in enumerate(ids):
        states[i] = "B" if k % 2 == 0 else "I"
        visible.append({"native_corner": i, "xy": q[i].tolist(),
                        "covariance_px2": [[1., 0.], [0., 1.]], "protection_px": 2.})
    available = [i for i in range(8) if i not in ids]
    oid, sid = available[:2]
    states[oid] = "O"
    states[sid] = "S"
    domains = [{"native_corner": oid, "state": "O", "shape": "ellipse",
                "center_xy": q[oid].tolist(), "half_axes_px": [18., 18.]}]
    coarse = {"R": R.tolist(), "t": t.tolist(), "physical_registration_Q": np.eye(3).tolist()}
    cfg = {"max_p3p_triples": 0, "global_rotation_starts": 0,
           "max_seed_nfev": 80, "max_joint_iterations": 100,
           "optimizer_ftol": 1e-9}
    return xyz, K, states, visible, domains, (480, 640), coarse, cfg


def solve(args, **config):
    xyz, K, states, visible, domains, hw, coarse, cfg = args
    return V4.solve_uncertain_joint_ot_pnp(xyz, K, states, visible, domains, hw, coarse,
                                          config={**cfg, **config})


def test_frozen_observations_effective_covariance_and_physical_registration():
    args = fixture()
    original = deepcopy(args)
    visible = args[3]
    visible[0]["fit_covariance_px2"] = [[25., 2.], [2., 16.]]
    visible[0]["fit_reliability_weight"] = .4
    before = deepcopy(visible)
    Q = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    args[6]["physical_registration_Q"] = Q.tolist()
    result = solve(args, state_BI_total_error_floor_px2={"B": [[9., 0.], [0., 9.]],
                                                       "I": [[16., 0.], [0., 16.]]})
    assert result["accepted"]
    assert visible == before
    assert args[0].tolist() == original[0].tolist()
    for source, output in zip(visible, result["visible_observations"]):
        assert output["xy"] == source["xy"]
        assert output["covariance_px2"] == source["covariance_px2"]
        assert result["reported_native"][source["native_corner"]] == source["xy"]
    assert result["visible_observations"][0]["fit_covariance_px2"] == [[25., 2.], [2., 16.]]
    assert result["visible_observations"][1]["fit_covariance_px2"] == [[17., 0.], [0., 17.]]
    assert np.allclose(result["R_physical"], np.asarray(result["R"]) @ Q)
    assert all(r["feasible"] for r in result["joint_constraint_audit"])


def test_soft_loss_keeps_large_inconsistent_BI_residual_feasible():
    args = fixture(count=6)
    for k, m in enumerate(args[3]):
        m["xy"] = (np.asarray(m["xy"]) + np.array([(-1)**k * 15., (-1)**(k // 2) * 11.])).tolist()
    result = solve(args, prior_weight=8., state_BI_total_error_floor_px2={
        "B": [[36., 0.], [0., 36.]], "I": [[36., 0.], [0., 36.]]})
    assert result["accepted"]
    assert max(r["measurement_mahalanobis"] for r in result["visible_protection_audit"]) > 3.
    assert max(r["observed_conditional_mahalanobis"] for r in result["visible_protection_audit"]) > 3.
    assert all(not r["hard_measurement_budget_active"] for r in result["visible_protection_audit"])
    assert all(r["visible_support_guards_pass"] for r in result["visible_protection_audit"])
    assert result["provenance"]["B_I_absolute_covariance_residual_constraints"] is False
    assert all(r["feasible"] for r in result["joint_constraint_audit"])


def test_joint_complementary_domains_cannot_compensate_and_fallback_is_explicit():
    args = fixture()
    domain = args[4][0]
    domain["half_axes_px"] = [2., 2.]
    other = deepcopy(domain)
    other["center_xy"][0] += 30.
    # Each region alone is attainable; their same-corner intersection is empty.
    assert solve(args)["accepted"]
    args_other = deepcopy(args)
    args_other[4][:] = [other]
    assert solve(args_other)["accepted"]
    args[4].append(other)
    result = solve(args)
    assert not result["accepted"]
    assert result["fallback_reason"] == "all_searched_joint_branches_infeasible_or_not_converged"
    assert not result["OT_constraints_satisfied"]
    assert result["reported_native"][domain["native_corner"]] is None
    assert result["fallback_pose"]["source"] == "SOFT_BI_AND_REGISTERED_COARSE_PRIOR_SEED"
    assert result["BI_prior_seed"]["fitted_BI_and_coarse_prior"]
    assert not result["BI_prior_seed"]["fitted_only_corrected_BI"]


def test_OT_can_move_pose_despite_finite_coarse_prior():
    args = fixture()
    domain = args[4][0]
    domain["center_xy"][0] += 13.
    domain["half_axes_px"] = [1., 1.]
    with_ot = solve(args, prior_weight=1.)
    no_ot = deepcopy(args)
    no_ot[2][domain["native_corner"]] = "U"
    no_ot[4].clear()
    without_ot = solve(no_ot, prior_weight=1.)
    assert with_ot["accepted"] and without_ot["accepted"]
    assert np.linalg.norm(np.asarray(with_ot["projected_native"])[domain["native_corner"]]
                          - np.asarray(without_ot["projected_native"])[domain["native_corner"]]) > 10.
    assert with_ot["joint_constraint_audit"][0]["feasible"]
    assert with_ot["objective_components"]["coarse_pose_prior_loss"] > 0.
    assert with_ot["objective_components"]["soft_BI_robust_loss"] > 0.


def test_two_BI_always_family_and_failed_joint_keeps_coarse_S_fallback():
    args = fixture(count=2)
    result = solve(args)
    assert result["accepted"]
    assert result["status"] == "joint_feasible_pose_family"
    assert result["ambiguity"]["detected"]
    assert not result["OT_unique_solution_claimed"] and not result["OT_solved"]
    assert result["pose_family_candidates"]
    assert not result["search"]["P3P_or_SQPNP_called"]
    no_ot = deepcopy(args)
    no_ot[2][args[4][0]["native_corner"]] = "U"
    no_ot[4].clear()
    assert not solve(no_ot)["accepted"]
    assert solve(no_ot)["fallback_reason"] == "two_BI_requires_OT_ranges"
    other = deepcopy(args[4][0])
    args[4][0]["half_axes_px"] = other["half_axes_px"] = [1., 1.]
    other["center_xy"][1] += 40.
    args[4].append(other)
    failed = solve(args)
    assert not failed["accepted"]
    assert failed["BI_prior_seed"]["contains_underconstrained_pose_family"]
    assert failed["fallback_pose"]["source"] == "INITIAL_REGISTERED_COARSE_PNP"
    assert failed["fallback_pose"]["R"] == args[6]["R"]
    assert failed["fallback_pose"]["t"] == args[6]["t"]


def test_S_raw_metadata_is_irrelevant_and_S_cannot_be_an_observation():
    args = fixture()
    a = solve(args)
    perturbed = deepcopy(args)
    perturbed[6]["raw_keypoints"] = [[1e12, -1e12]] * 8
    b = solve(perturbed)
    assert a["R"] == b["R"] and a["t"] == b["t"] and a["objective"] == b["objective"]
    assert a["provenance"]["S_image_terms"] == a["provenance"]["S_region_terms"] == 0
    sid = args[2].index("S")
    q, _ = V3.project_native(args[0], np.asarray(a["R"]), np.asarray(a["t"]), args[1])
    assert np.allclose(a["reported_native"][sid], q[sid])
    args[3].append({"native_corner": sid, "xy": [320., 240.],
                    "covariance_px2": [[1., 0.], [0., 1.]], "protection_px": 2.})
    with pytest.raises(ValueError, match="Only unique final B/I"):
        solve(args)


@pytest.mark.parametrize("phi", [[1e-9, -2e-9, 3e-9], [.2, -.7, .4], [3.13, .01, -.02]])
def test_camera_tangent_ray_logdepth_prior_exact_jacobian(phi):
    scale = .9
    center_xyz = np.array([.03, -.05, .04])
    R0 = Rotation.from_rotvec([-.05, .08, -.02]).as_matrix()
    center0 = np.array([.11, -.07, 2.3])
    p = np.r_[phi, [.12, -.08, 2.7]]
    # Anisotropic covariance with cross terms exercises all six components.
    matrix = np.eye(6)
    matrix[0, 3] = .15
    whitening = np.linalg.inv(np.linalg.cholesky(matrix @ matrix.T))
    fun = lambda x: V4._prior_residual_and_jacobian(x, scale, center_xyz, R0, center0, whitening)
    residual, analytic = fun(p)
    numerical = V3._jacobian(lambda x: fun(x)[0], p)
    assert np.isfinite(residual).all()
    assert np.allclose(analytic, numerical, atol=2e-6, rtol=2e-6)


def test_soft_constraint_jacobian_has_no_residual_budget_rows():
    args = fixture(count=2)
    xyz, K, states, visible, domains, hw, coarse, _ = args
    cfg = V4._configuration({})
    xyz, K, _, measurements, domains, hw = V3._inputs(xyz, K, states, visible, domains, hw, V4._helper_config(cfg))
    scale = np.linalg.norm(np.ptp(xyz, axis=0))
    p = V3._pose_vector(np.asarray(coarse["R"]), np.asarray(coarse["t"]), scale)
    values, jacobian = V4._constraints(p, xyz, K, measurements, domains, (None,), hw, scale, cfg)
    assert len(values) == 8 + 2 * 5 + 1
    scalar = lambda x: V4._constraints(x, xyz, K, measurements, domains, (None,), hw, scale, cfg, analytic=False)
    assert np.allclose(values, scalar(p), atol=1e-12)
    assert np.allclose(jacobian, V3._jacobian(scalar, p), atol=1e-6, rtol=2e-6)


def test_simultaneous_O_and_unknown_side_T_use_physical_outside_union():
    args = fixture()
    xyz, K, states, visible, domains, _, coarse, cfg = args
    q, _ = V3.project_native(xyz, np.asarray(coarse["R"]), np.asarray(coarse["t"]), K)
    # The four B/I selected above are left-side vertices. A true crop makes
    # the right-side native vertex T without supplying its precise coordinate.
    width = 360
    assert all(m["xy"][0] < width for m in visible)
    tid = next(i for i, state in enumerate(states) if state == "U" and q[i, 0] > width)
    states[tid] = "T"
    domains.append({"native_corner": tid, "state": "T", "shape": "halfplane",
                    "outside_canvas_side": "unknown", "canvas_hw": [480, width]})
    result = V4.solve_uncertain_joint_ot_pnp(xyz, K, states, visible, domains, (480, width),
                                            coarse, config=cfg)
    assert result["accepted"]
    assert result["search"]["T_union_branches_searched"] == 4
    assert all(d["feasible"] for d in result["joint_constraint_audit"])
    assert len(result["joint_constraint_audit"]) == 2
    taudit = next(d for d in result["joint_constraint_audit"] if d["state"] == "T")
    assert taudit["outside_canvas_side"] == "right"
    assert taudit["projected_xy"][0] >= width - 1e-7
    assert "center_xy" not in domains[-1]


def test_weighted_total_error_objective_gradient_and_native_basis_invariance():
    args = fixture()
    xyz, K, states, visible, domains, hw, coarse, _ = args
    cfg = V4._configuration({"prior_weight": .25})
    xyz, K, _, measurements, _, _ = V3._inputs(xyz, K, states, visible, domains, hw, V4._helper_config(cfg))
    for k, m in enumerate(measurements):
        m["fit_reliability_weight"] = .4 + .3 * k
        m["xy"] = m["xy"] + [1.7, -2.1]
        m["whitening"] = np.array([[.3, 0.], [-.1, .5]])
    scale = np.linalg.norm(np.ptp(xyz, axis=0))
    R0, t0 = np.asarray(coarse["R"]), np.asarray(coarse["t"])
    prior = {"R": R0, "center": R0 @ xyz.mean(axis=0) + t0,
             "whitening": np.diag([4., 5., 3., 20., 25., 8.])}
    R = Rotation.from_rotvec([.015, -.01, .012]).as_matrix() @ R0
    p = V3._pose_vector(R, t0 + [.015, -.006, .03], scale)
    value, gradient, _ = V4._objective_pair(p, xyz, K, measurements, scale, prior, cfg)
    numerical = V3._jacobian(lambda x: np.array([V4._objective_pair(x, xyz, K, measurements, scale, prior, cfg)[0]]), p)[0]
    assert np.allclose(gradient, numerical, atol=2e-5, rtol=2e-6)
    H = Rotation.from_euler("xyz", [27., 9., -33.], degrees=True).as_matrix()
    alternate_xyz = xyz @ H.T
    alternate_prior = {**prior, "R": R0 @ H.T}
    alternate_p = V3._pose_vector(R @ H.T, t0 + [.015, -.006, .03], scale)
    alternate_value = V4._objective_pair(alternate_p, alternate_xyz, K, measurements, scale,
                                        alternate_prior, cfg)[0]
    assert np.isclose(value, alternate_value, atol=1e-9)


def test_zero_prior_control_does_not_claim_a_prior_was_applied():
    result = solve(fixture(), prior_weight=0.)
    assert result["accepted"]
    assert not result["prior_audit"]["enabled"]
    assert not result["provenance"]["initial_pose_prior"]
    assert result["provenance"]["coarse_pose_used_for_initialization"]
    assert not result["search"]["registered_coarse_pose_is_prior"]
    assert result["objective_components"]["coarse_pose_prior_loss"] == 0.
    assert result["BI_prior_seed"]["fitted_only_corrected_BI"]
    assert not result["BI_prior_seed"]["fitted_BI_and_coarse_prior"]


def test_forbidden_legacy_hard_guard_config_and_invalid_cal_covariance_rejected():
    args = fixture()
    with pytest.raises(ValueError, match="Unknown v4"):
        solve(args, maximum_visible_mahalanobis=3.)
    with pytest.raises(ValueError, match="Unknown v4"):
        solve(args, protect_seed_projection=True)
    args[3][0]["fit_covariance_px2"] = [[-1., 0.], [0., 1.]]
    with pytest.raises(ValueError, match="positive definite"):
        solve(args)
