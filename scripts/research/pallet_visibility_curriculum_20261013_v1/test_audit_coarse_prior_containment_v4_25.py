"""Independent v25 audit arithmetic; synthetic only, no archive/model reads."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import audit_coarse_prior_containment_v4_25 as A
from . import coarse_pose_prior_v4_25 as PRIOR
from .test_coarse_pose_prior_v4_25 import ingredients


@pytest.mark.parametrize('mode', ['FULL', 'TRANSLATION'])
@pytest.mark.parametrize('rotvec', [[.2, -.1, .14], [4.1, -.5, .2]])
def test_direct_old_API_auditor_matches_helper_value_gradient_and_source_binding(mode, rotvec):
    xyz, coarse, covariance, bias, prepared, bound = ingredients(mode)
    original = dict(native_xyz=xyz.tolist(), coarse_pose=coarse,
        physical_registration_Q=coarse['physical_registration_Q'])
    task = dict(families={'ORIGINAL_WD': original})
    model = dict(prior_covariance6=covariance.tolist(), prior_fit=dict(median_bias6=bias.tolist()))
    fields = A._original_prior_fields(task, model, 'a' * 64, mode)
    assert all(fields[key] == value for key, value in prepared.items() if key != 'source_binding_sha256')
    assert A._source_binding(fields) == prepared['source_binding_sha256']
    scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
    p = np.r_[rotvec, np.array([.3, -.2, 2.8]) / scale]
    value, gradient, audit = A.direct_old_P1_prior(p, xyz, scale, fields, original['physical_registration_Q'])
    expected, expected_gradient, expected_audit = PRIOR.coarse_pose_prior_value_and_gradient(p, xyz, scale, bound)
    assert value == expected and np.array_equal(gradient, expected_gradient)
    assert audit['residual'] == expected_audit['whitened_residual']
    numeric = []
    for j in range(6):
        step = 1e-6 * max(1., abs(p[j]))
        delta = np.zeros(6)
        delta[j] = step
        numeric.append((A.direct_old_P1_prior(p + delta, xyz, scale, fields, original['physical_registration_Q'])[0]
            - A.direct_old_P1_prior(p - delta, xyz, scale, fields, original['physical_registration_Q'])[0]) / (2 * step))
    assert np.allclose(gradient, numeric, atol=3e-5, rtol=3e-6)


@pytest.mark.parametrize('mode', ['FULL', 'TRANSLATION'])
def test_manual_auditor_rebinds_original_target_to_second_nonzero_mean_basis(mode):
    xyz, coarse, covariance, bias, _, _ = ingredients(mode)
    task = dict(families={'ORIGINAL_WD': dict(native_xyz=xyz.tolist(), coarse_pose=coarse,
        physical_registration_Q=coarse['physical_registration_Q'])})
    fields = A._original_prior_fields(task, dict(prior_covariance6=covariance.tolist(),
        prior_fit=dict(median_bias6=bias.tolist())), 'a' * 64, mode)
    Q0 = np.asarray(coarse['physical_registration_Q'])
    Q1 = np.eye(3)
    points1 = (xyz - xyz.mean(axis=0)) @ Q0 @ Q1.T + [-.15, .08, -.04]
    physical_R = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    center = np.array([.3, -.1, 2.7])
    costs, residuals = [], []
    for points, Q in ((xyz, Q0), (points1, Q1)):
        R = physical_R @ Q.T
        t = center - R @ points.mean(axis=0)
        scale = float(np.linalg.norm(np.ptp(points, axis=0)))
        p = A.BIAS.V3._pose_vector(R, t, scale)
        cost, _, audit = A.direct_old_P1_prior(p, points, scale, fields, Q)
        costs.append(cost)
        residuals.append(audit['residual'])
    assert costs[0] == pytest.approx(costs[1], abs=2e-12, rel=1e-12)
    assert np.allclose(residuals[0], residuals[1], atol=2e-13, rtol=1e-12)


def test_three_prior_and_optional_containment_ledger_no_old_mask_or_duplicate_shape():
    shape = A._energies(1., 2., 3., 4., 5., .7, True)
    assert shape == dict(regularizer=9., regularized_native=10., weighted_containment=3.5, total=13.5)
    native = A._energies(1., 2., 3., 4., 5., .7, False)
    assert native == dict(regularizer=9., regularized_native=10., weighted_containment=0., total=10.)


def test_same_mode_native_endpoint_names_use_frozen_global_family_order():
    task = dict(families={'ORIGINAL_WD': {}, 'SWAPPED_WD': {}})
    profile = next(arms for _, mode, enabled, arms in A.V.PROFILES if mode == 'FULL' and not enabled)
    endpoint = dict(R=np.eye(3).tolist(), t=[0., 0., 3.])
    row = dict(bounded_prior_refinement_audit={profile[0]: dict(family_refinements={
        'ORIGINAL_WD': dict(feasible_endpoints=[endpoint, endpoint]),
        'SWAPPED_WD': dict(feasible_endpoints=[endpoint])})})
    records = A._native_endpoint_bank(task, row, 'FULL')
    assert [r['name'] for r in records] == ['ORIGINAL_WD/NATIVE_ENDPOINT_0',
        'ORIGINAL_WD/NATIVE_ENDPOINT_1', 'SWAPPED_WD/NATIVE_ENDPOINT_2']
