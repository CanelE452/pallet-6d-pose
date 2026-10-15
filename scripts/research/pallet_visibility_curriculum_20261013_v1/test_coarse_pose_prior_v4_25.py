"""Frozen-P1 equality, derivative/gauge and covariance tests; synthetic only."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import coarse_pose_prior_v4_25 as V
from . import bounded_geometric_priors_problem_v4_22 as OS
from .common import cuboid
from .test_complete_bi_image_chain_evidence_v4_20 import fixture as native_fixture


def ingredients(mode='FULL', weight=1., strength=1.):
    xyz = cuboid(.8, .4, .6) + [.13, -.2, .07]
    R0 = Rotation.from_rotvec([.1, -.25, .07]).as_matrix()
    Q = Rotation.from_euler('y', 90., degrees=True).as_matrix()
    coarse = dict(R=R0.tolist(), t=[.2, .1, 3.], physical_registration_Q=Q.tolist())
    lower = np.diag([.2, .25, .3, .04, .05, .1])
    lower[3, 0] = .01
    lower[4, 1] = -.02
    lower[5, 2] = .015
    covariance = lower @ lower.T
    bias = np.array([.01, -.02, .03, .004, -.003, .02])
    prepared = V.prepare_coarse_pose_prior(xyz, coarse, covariance, bias,
        calibration_model_digest_sha256='a' * 64, mode=mode,
        factor_weight=weight, bias_strength=strength)
    return xyz, coarse, covariance, bias, prepared, V.rebind_coarse_pose_prior(prepared, xyz, Q)


def test_exact_original_P1_residual_jacobian_and_value_same_basis():
    xyz, coarse, covariance, bias, _, bound = ingredients()
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = V.BIAS.V3._pose_vector(Rotation.from_rotvec([.2, -.1, .14]).as_matrix(), [.3, -.2, 2.8], scale)
    R0, t0 = np.asarray(coarse['R']), np.asarray(coarse['t'])
    expected, expected_j = V.BIAS._prior_residual_and_jacobian(p, scale, xyz.mean(axis=0), R0,
        R0 @ xyz.mean(axis=0) + t0, np.linalg.inv(np.linalg.cholesky(covariance)), bias)
    residual, jacobian = V.coarse_pose_prior_residual_and_jacobian(p, xyz, scale, bound)
    value, gradient, audit = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    assert np.array_equal(residual, expected) and np.array_equal(jacobian, expected_j)
    assert value == float(expected @ expected)
    assert np.array_equal(gradient, 2. * expected @ expected_j)
    assert audit['mean_bias_subtracted_once'] and audit['S_U_raw_image_2D_or_regions'] == 0


@pytest.mark.parametrize('mode', V.MODES)
@pytest.mark.parametrize('rotvec', [[.2, -.1, .14], [4.1, -.5, .2]])
def test_analytic_gradient_in_small_and_large_rotation_charts(mode, rotvec):
    xyz, _, _, _, _, bound = ingredients(mode)
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = np.r_[rotvec, np.array([.3, -.2, 2.8]) / scale]
    value, gradient, _ = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    numeric = []
    for j in range(6):
        step = 1e-6 * max(1., abs(p[j]))
        d = np.zeros(6)
        d[j] = step
        numeric.append((V.coarse_pose_prior_value_and_gradient(p + d, xyz, scale, bound)[0]
            - V.coarse_pose_prior_value_and_gradient(p - d, xyz, scale, bound)[0]) / (2 * step))
    assert np.isfinite(value)
    assert np.allclose(gradient, numeric, atol=3e-5, rtol=3e-6)


@pytest.mark.parametrize('mode', V.MODES)
def test_nonzero_model_centers_and_two_Q_bases_preserve_common_physical_residual(mode):
    xyz, coarse, _, _, prepared, first = ingredients(mode)
    Q0 = np.asarray(coarse['physical_registration_Q'])
    Q1 = np.eye(3)
    centered_physical = (xyz - xyz.mean(axis=0)) @ Q0
    other_xyz = centered_physical @ Q1.T + [-.15, .08, -.04]
    second = V.rebind_coarse_pose_prior(first, other_xyz, Q1)
    assert second['source_binding_sha256'] == prepared['source_binding_sha256']
    assert second['original_first_center_camera_xyz'] == first['original_first_center_camera_xyz']
    Rp = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    camera_center = np.array([.3, -.1, 2.7])
    residuals, values = [], []
    for points, Q, bound in ((xyz, Q0, first), (other_xyz, Q1, second)):
        R = Rp @ Q.T
        t = camera_center - R @ points.mean(axis=0)
        scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-3)
        p = V.BIAS.V3._pose_vector(R, t, scale)
        residuals.append(V.coarse_pose_prior_residual_and_jacobian(p, points, scale, bound)[0])
        values.append(V.coarse_pose_prior_value_and_gradient(p, points, scale, bound)[0])
    assert np.allclose(residuals[0], residuals[1], atol=2e-13, rtol=1e-12)
    assert values[0] == pytest.approx(values[1], abs=2e-12, rel=1e-12)


def test_translation_uses_marginal_covariance_not_full_precision_or_conditional():
    xyz = cuboid(.8, .4, .6)
    coarse = dict(R=np.eye(3).tolist(), t=[0., 0., 3.], physical_registration_Q=np.eye(3).tolist())
    covariance = np.block([[np.eye(3), .8 * np.eye(3)], [.8 * np.eye(3), 2. * np.eye(3)]])
    prepared = V.prepare_coarse_pose_prior(xyz, coarse, covariance, np.zeros(6),
        calibration_model_digest_sha256='b' * 64, mode='TRANSLATION')
    bound = V.rebind_coarse_pose_prior(prepared, xyz, np.eye(3))
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = V.BIAS.V3._pose_vector(np.eye(3), [.3, -.2, 2.7], scale)
    raw, _ = V.BIAS._prior_residual_and_jacobian(p, scale, xyz.mean(axis=0), np.eye(3),
        np.array([0., 0., 3.]), np.eye(6), None)
    expected = float(raw[3:] @ np.linalg.inv(covariance[3:, 3:]) @ raw[3:])
    conditional = float(raw[3:] @ np.linalg.inv(covariance)[3:, 3:] @ raw[3:])
    actual = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)[0]
    assert actual == pytest.approx(expected, abs=1e-14)
    assert not np.isclose(actual, conditional)


@pytest.mark.parametrize('mode', V.MODES)
def test_bias_once_no_original_pose_shift_and_no_input_mutation(mode):
    xyz, coarse, covariance, bias, prepared, bound = ingredients(mode)
    original = deepcopy((xyz, coarse, covariance, bias, prepared, bound))
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = V.BIAS.V3._pose_vector(np.asarray(coarse['R']), np.asarray(coarse['t']), scale)
    covariance_use, bias_use = (covariance, bias) if mode == 'FULL' else (covariance[3:, 3:], bias[3:])
    expected = float(bias_use @ np.linalg.inv(covariance_use) @ bias_use)
    first = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    second = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    assert first[0] == pytest.approx(expected, rel=1e-12, abs=1e-14)
    assert first[0] != pytest.approx(4. * expected)
    assert first[0] == second[0] and np.array_equal(first[1], second[1])
    assert np.array_equal(xyz, original[0]) and coarse == original[1]
    assert np.array_equal(covariance, original[2]) and np.array_equal(bias, original[3])
    assert prepared == original[4] and bound == original[5]


@pytest.mark.parametrize('case', ['not_SPD', 'nonsymmetric', 'bad_R', 'bad_Q', 'physical_R_mismatch',
    'negative_coarse_depth', 'bad_digest', 'bad_bias', 'bad_mode', 'bad_strength'])
def test_covariance_rotation_and_parameter_validation(case):
    xyz, coarse, covariance, bias, _, _ = ingredients()
    kwargs = dict(calibration_model_digest_sha256='a' * 64)
    if case == 'not_SPD':
        covariance[0, 0] = -1.
    elif case == 'nonsymmetric':
        covariance[0, 1] += .1
    elif case == 'bad_R':
        coarse['R'] = np.diag([-1., 1., 1.]).tolist()
    elif case == 'bad_Q':
        coarse['physical_registration_Q'] = np.zeros((3, 3)).tolist()
    elif case == 'physical_R_mismatch':
        coarse['R_physical'] = np.eye(3).tolist()
    elif case == 'negative_coarse_depth':
        coarse['t'] = [0., 0., -3.]
    elif case == 'bad_digest':
        kwargs['calibration_model_digest_sha256'] = 'not a digest'
    elif case == 'bad_bias':
        bias[0] = np.nan
    elif case == 'bad_mode':
        kwargs['mode'] = 'CONDITIONAL'
    else:
        kwargs['bias_strength'] = 2.
    with pytest.raises(ValueError):
        V.prepare_coarse_pose_prior(xyz, coarse, covariance, bias, **kwargs)


def test_wrapper_keeps_native_ledger_and_constraints_and_adds_one_named_prior():
    native, _, _, _, R, t = native_fixture()
    base = OS.BoundedGeometricPriorsProblem(native, [], dict(epsilon=.2, huber_scale=3.), None)
    original = dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=np.eye(3).tolist())
    prepared = V.prepare_coarse_pose_prior(native.xyz, original, np.diag([.1, .2, .3, .01, .02, .03]),
        [.01, -.01, .02, .001, -.002, .003], calibration_model_digest_sha256='c' * 64)
    bound = V.rebind_coarse_pose_prior(prepared, native.xyz, np.eye(3))
    wrapper = V.CoarsePosePriorProblem(base, bound)
    p = V.BIAS.V3._pose_vector(R, t, native.scale) + [.01, -.02, .005, .008, -.011, .025]
    bv, bg, bp = base.objective(p)
    pv, pg, _ = V.coarse_pose_prior_value_and_gradient(p, native.xyz, native.scale, bound)
    value, gradient, parts = wrapper.objective(p)
    assert value == bv + pv and np.array_equal(gradient, bg + pg)
    assert parts['native_costs'] == bp['native_costs']
    assert parts['native_observation_objective'] == bp['native_observation_objective']
    assert [r['kind'] for r in parts['regularizer_costs']] == list(OS.REGULARIZER_KINDS) + [V.REGULARIZER_KIND]
    assert parts['regularizer_objective'] == bp['regularizer_objective'] + pv
    assert parts['coarse_pose_prior_count'] == 1
    a, j = wrapper.constraints(p, wrapper.sides[0])
    b, jj = base.constraints(p, base.sides[0])
    assert np.array_equal(a, b) and np.array_equal(j, jj)
    pose_R, pose_t = V.BIAS.V3._pose(p, native.scale)
    old, new = base.evaluate(pose_R, pose_t), wrapper.evaluate(pose_R, pose_t)
    assert new['accepted'] == old['accepted']
    assert new['objective'] == pytest.approx(old['objective'] + pv)
    with pytest.raises(ValueError, match='exactly O/S once'):
        V.CoarsePosePriorProblem(wrapper, bound).objective(p)
    different_Q = Rotation.from_euler('y', 90., degrees=True).as_matrix()
    with pytest.raises(ValueError, match='identical physical registration Q'):
        V.CoarsePosePriorProblem(base, V.rebind_coarse_pose_prior(prepared, native.xyz, different_Q))


def test_zero_weight_and_zero_bias_strength_do_not_shift_inputs_or_objective():
    xyz, coarse, _, _, _, bound = ingredients(weight=0., strength=0.)
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = V.BIAS.V3._pose_vector(np.asarray(coarse['R']), np.asarray(coarse['t']), scale)
    value, gradient, audit = V.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    assert value == 0. and np.array_equal(gradient, np.zeros(6))
    assert audit['factor_weight'] == audit['bias_strength'] == 0.
