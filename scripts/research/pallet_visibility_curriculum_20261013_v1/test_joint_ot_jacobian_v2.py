"""Central-difference verification of analytic SE3/OT derivatives, no data."""
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from .joint_ot_jacobian_v2 import (constraint_values_and_jacobian,
                                 projection_and_jacobian,
                                 visible_objective_and_gradient,
                                 visible_residual_and_jacobian)
from .joint_ot_pnp import (_DEFAULTS, _constraint_values, _inputs, _objective,
                          _visible_residual, project_native, solve_joint_ot_pnp)


def central_difference(function, p, step=1e-6):
    columns = []
    for index in range(6):
        delta = np.zeros(6); delta[index] = step
        columns.append((function(p + delta) - function(p - delta)) / (2 * step))
    return np.stack(columns, axis=-1)


def scene(rotvec):
    # Deliberately retain this known ordering and nonzero skew in K.
    xyz = np.array(list(product([-.65, .65], [-.055, .055], [-.55, .55])))
    K = np.array([[605., 47., 317.], [0., 620., 256.], [0., 0., 1.]])
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = np.r_[rotvec, np.array([.15, -.1, 3.6]) / scale]
    return p, xyz, K, scale


ROTVECS = [np.zeros(3), np.array([1e-10, -2e-10, 3e-10]), np.array([.2, -.6, .3]),
           np.array([1., -2., 3.]) / np.sqrt(14) * (np.pi - 1e-8),
           np.array([1., -2., 3.]) / np.sqrt(14) * (np.pi + 1e-8)]


@pytest.mark.parametrize("rotvec", ROTVECS)
def test_projection_and_depth_exact_jacobians_near_zero_and_pi(rotvec):
    p, xyz, K, scale = scene(rotvec)
    q, depth, jq, jdepth = projection_and_jacobian(p, xyz, K, scale)
    R = Rotation.from_rotvec(rotvec).as_matrix()
    expected, expected_depth = project_native(xyz, R, p[3:] * scale, K)
    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(depth, expected_depth, atol=1e-12, rtol=0)
    numeric_q = central_difference(lambda x: project_native(
        xyz, Rotation.from_rotvec(x[:3]).as_matrix(), x[3:] * scale, K)[0], p)
    numeric_depth = central_difference(lambda x: project_native(
        xyz, Rotation.from_rotvec(x[:3]).as_matrix(), x[3:] * scale, K)[1], p)
    np.testing.assert_allclose(jq, numeric_q, atol=8e-7, rtol=5e-7)
    np.testing.assert_allclose(jdepth, numeric_depth, atol=5e-9, rtol=5e-7)


def problem(rotvec):
    p, xyz, K, scale = scene(rotvec)
    reference = projection_and_jacobian(p, xyz, K, scale)[0]
    measurements = [dict(native_corner=i, xy=(reference[i] + [.8 * (i + 1), -.6]).tolist(),
                         covariance_px2=[[2.3, .4], [.4, 1.2]], protection_px=2.)
                    for i in [0, 2, 6]]
    domains = [dict(native_corner=1,state="O",shape="ellipse",center_xy=(reference[1]+[2,-1]).tolist(),
                    half_axes_px=[7., 13.],angle_deg=37.,halfplanes=[[1.,2.,900.]]),
               dict(native_corner=3,state="O",shape="box",
                    bounds_xyxy=[10.,20.,630.,470.],halfplanes=[[3.,-4.,800.]]),
               dict(native_corner=5,state="T",shape="halfplane",outside_canvas_side="right",
                    outside_margin_px=.3,halfplanes=[[-1.,0.,-610.],[0.,1.,430.]])]
    states = ["B" if i in [0,2,6] else "S" for i in range(8)]
    for domain in domains: states[domain["native_corner"]] = domain["state"]
    cfg = dict(_DEFAULTS)
    xyz, K, _, measurements, domains, hw = _inputs(xyz,K,states,measurements,domains,[480,640],cfg)
    return p, xyz, K, scale, reference, measurements, domains, hw, cfg


@pytest.mark.parametrize("rotvec", ROTVECS)
def test_whitened_BI_and_radial_huber_match_current_problem_and_derivatives(rotvec):
    p, xyz, K, scale, _, measurements, _, _, cfg = problem(rotvec)
    residual, jacobian = visible_residual_and_jacobian(p,xyz,K,measurements,scale)
    old_residual = lambda x: _visible_residual(x,xyz,K,measurements,scale)
    np.testing.assert_allclose(residual,old_residual(p),atol=1e-12,rtol=0)
    np.testing.assert_allclose(jacobian,central_difference(old_residual,p),atol=8e-7,rtol=5e-7)
    value, gradient = visible_objective_and_gradient(p,xyz,K,measurements,scale,cfg["huber_scale"])
    old_objective = lambda x: np.asarray(_objective(x,xyz,K,measurements,scale,cfg["huber_scale"]))
    np.testing.assert_allclose(value,old_objective(p),atol=1e-12,rtol=0)
    np.testing.assert_allclose(gradient,central_difference(old_objective,p),atol=2e-5,rtol=5e-7)


@pytest.mark.parametrize("rotvec", ROTVECS)
@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_all_joint_constraint_values_and_jacobians_match_numeric_problem(rotvec,side):
    p, xyz, K, scale, reference, measurements, domains, hw, cfg = problem(rotvec)
    sides = [None,None,side]
    p = p + np.array([.001,-.002,.0015,.001,-.002,.003])
    values,jacobian = constraint_values_and_jacobian(p,xyz,K,measurements,domains,sides,reference,hw,scale,cfg)
    old = lambda x: _constraint_values(x,xyz,K,measurements,domains,sides,reference,hw,scale,cfg)
    np.testing.assert_allclose(values,old(p),atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(jacobian,central_difference(old,p),atol=3e-5,rtol=2e-6)
    per_BI = (1 + int(cfg["protect_seed_projection"]) + int(cfg["enforce_visible_geometry"])
              + 4 * int(cfg["enforce_visible_canvas"]))
    # Every BI and OT inequality stays individual; none is averaged.
    count = 8 + len(measurements) * per_BI + 10
    assert values.shape == (count,) and jacobian.shape == (count,6)


def test_analytic_functions_do_not_modify_fixed_measurements_or_native_arrays():
    p, xyz, K, scale, _, measurements, _, _, cfg = problem([.2,-.6,.3])
    before_xyz,before_K,before_p = xyz.copy(),K.copy(),p.copy()
    before_q = [m["xy"].copy() for m in measurements]
    for array in [xyz,K,p]: array.setflags(write=False)
    visible_objective_and_gradient(p,xyz,K,measurements,scale,cfg["huber_scale"])
    np.testing.assert_array_equal(xyz,before_xyz)
    np.testing.assert_array_equal(K,before_K)
    np.testing.assert_array_equal(p,before_p)
    for measurement,q in zip(measurements,before_q): np.testing.assert_array_equal(measurement["xy"],q)


def test_analytic_solver_preserves_continuous_joint_pose_and_individual_constraints():
    from .test_joint_ot_pnp import FAST, scene as ambiguous_scene, states_for, visible
    xyz,K,R,t,q = ambiguous_scene()
    ids=[0,2,6]
    domain=dict(native_corner=5,state="O",shape="box",
                bounds_xyxy=[q[5,0]-1,q[5,1]-1,q[5,0]+1,q[5,1]+1])
    numerical = solve_joint_ot_pnp(xyz,K,states_for(ids,[domain]),visible(q,ids),[domain],[480,640],
                                    config={**FAST,"analytic_jacobian":False})
    exact = solve_joint_ot_pnp(xyz,K,states_for(ids,[domain]),visible(q,ids),[domain],[480,640],
                               config={**FAST,"analytic_jacobian":True})
    assert numerical["accepted"] and exact["accepted"]
    assert exact["search"]["analytic_jacobian_used"]
    assert numerical["search"]["initialization_count"] == exact["search"]["initialization_count"]
    np.testing.assert_allclose(exact["R"],numerical["R"],atol=2e-5,rtol=0)
    np.testing.assert_allclose(exact["t"],numerical["t"],atol=2e-5,rtol=0)
    np.testing.assert_allclose(exact["R"],R,atol=2e-5,rtol=0)
    assert all(a["feasible"] for a in exact["joint_constraint_audit"])
    assert all(a["measurement_budget_pass"] and a["protection_pass"] for a in exact["visible_protection_audit"])


def test_analytic_solver_does_not_accept_inconsistent_absolute_BI_budgets():
    from .test_joint_ot_pnp import FAST, scene as ambiguous_scene, states_for, visible
    xyz,K,_,_,q = ambiguous_scene();ids=list(range(6));q[4]+=[24,-20]
    exact=solve_joint_ot_pnp(xyz,K,states_for(ids),visible(q,ids,protection=2.),[],[480,640],
                             config={**FAST,"max_p3p_triples":2,"analytic_jacobian":True})
    assert not exact["accepted"] and exact["R"] is None
    assert exact["fallback_diagnostics"]["corrected_BI_seed_failed_measurement_ids"]
