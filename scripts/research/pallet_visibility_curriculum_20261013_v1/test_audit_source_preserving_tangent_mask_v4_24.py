"""Independent-audit accounting fixtures; synthetic, no files/references."""
from copy import deepcopy

import numpy as np
import pytest

from . import audit_source_preserving_tangent_mask_v4_24 as A
from .test_source_preserving_complete_bi_evidence_v4_24 import fixture
from .test_calibrate_complete_bi_edge_mixture_v4_20d import component, row
from .test_calibrate_archived_complete_tangent_v4_24 import point as tpoint, row as trow


def test_continuous_mask_count_one_and_fixedmask_solution_separate():
    c = A._selection_energies(7., 5., 3., 2., continuous_mask=True, mask_selector=True)
    assert c == dict(solution_objective=18., selector_objective=18.,
        regularized_native_objective=12., weighted_mask_objective=6.)
    b = A._selection_energies(7., 5., 3., 2., continuous_mask=False, mask_selector=True)
    assert b['solution_objective'] == 12. and b['selector_objective'] == 18.
    n = A._selection_energies(7., 5., 3., 2., continuous_mask=False, mask_selector=False)
    assert n['solution_objective'] == n['selector_objective'] == 12.


def test_factor_buckets_preserve_kind_vertex_and_sum_each_exact_factor():
    records = [dict(kind='COMPLETE_BI', native_corner=2, weighted_robust_cost=3.),
        dict(kind='PROJECTED_NATIVE_EDGE', native_corner=2, native_neighbor=3, weighted_robust_cost=5.),
        dict(kind='PROJECTED_NATIVE_EDGE', native_corner=2, native_neighbor=3, weighted_robust_cost=7.)]
    buckets = A._native_cost_buckets(records)
    assert buckets == {('COMPLETE_BI', 2, None): 3., ('PROJECTED_NATIVE_EDGE', 2, 3): 12.}


def test_fallback_readout_requires_S_projection_and_U_partial_OT_unresolved():
    xyz = np.array([[-1., -1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., -1., -1.],
        [-1., 1., -1.], [-1., 1., 1.], [1., 1., 1.], [1., 1., -1.]])
    K = np.array([[100., 0., 100.], [0., 100., 100.], [0., 0., 1.]])
    R, t = np.eye(3), np.array([0., 0., 5.])
    projection = A.V.C.project(xyz, R, t, K)
    task = dict(families={'ORIGINAL_WD': dict(native_xyz=xyz.tolist())}, spec=dict(K=K.tolist()),
        states=['S', 'U', 'B', 'O', 'T', 'B', 'I', 'I'], boundary_merged=[dict(native_corner=2)])
    result = dict(WD_family='ORIGINAL_WD', accepted=False,
        actual_pose=dict(R_cf=R.tolist(), centroid=t.tolist()),
        keypoint_readout=dict(reported_points=[projection[0].tolist(), None, None, None, None,
            projection[5].tolist(), projection[6].tolist(), projection[7].tolist()]))
    assert all(condition for condition, _ in A._model_readout_checks(task, result))
    result['keypoint_readout']['reported_points'][1] = [1., 2.]
    assert not all(condition for condition, _ in A._model_readout_checks(task, result))


@pytest.mark.parametrize('source_preserving', [False, True])
def test_independent_reconstruction_TCAL_original_axis_and_hardguards(source_preserving):
    base, obs, kw, (k, _, _), R, t = fixture()
    calibration = A.SOURCE_CAL if source_preserving else A.ALL_CAL
    model = calibration.fit_native_edge_mixture_model([
        row('synthetic_a', 'A', [component(3., .05)]),
        row('synthetic_b', 'B', [component(-4., -.04)])])
    tm = A.TCAL.fit_archived_complete_tangent_model([
        trow('synthetic_a', 'A', [tpoint(3.)]), trow('synthetic_b', 'B', [tpoint(-4.)])])
    family = dict(native_xyz=base.xyz.tolist(), physical_registration_Q=np.eye(3).tolist(),
        coarse_pose=dict(R=R.tolist(), t=t.tolist(), physical_registration_Q=np.eye(3).tolist()))
    task = dict(spec=dict(K=base.K.tolist(), states=list(base.states), visible=kw['visible_measurements'],
        domains=[], image_hw=list(base.image_hw), config=dict(prior_weight=0., analytic_jacobian=True)),
        states=list(base.states), boundary_merged=[], u_regions=[], image_sha256=kw['image_binding']['sha256'],
        initial_self_hidden=kw['initial_self_hidden'], new_chain_observation=obs,
        complete_BI_evidence=dict(junction_proposals=kw['junction_proposals'],
            initial_projection_xy=kw['initial_projection_xy']))
    before = deepcopy(task)
    problem, raw = A._reconstruct_native(task, family, model, tm, source_preserving=source_preserving)
    record = next(r for r in problem.plan['corner_plans'] if r['native_corner'] == k)
    tangent = np.asarray(record['tangent_unit_xy'])
    covariance = np.asarray(record['original_measurement_archive']['original_effective_fit_covariance_px2'])
    assert record['tangent_variance_px2'] == pytest.approx(1. + tm['gamma'] * (tangent @ covariance @ tangent))
    signature = A._plan_signature(problem.plan)
    extra = deepcopy(problem.plan)
    extra['corner_plans'][0]['irrelevant_prose'] = 'does not change a likelihood'
    assert signature == A._plan_signature(extra)
    changed = deepcopy(problem.plan)
    next(r for r in changed['corner_plans'] if r['native_corner'] == k)['tangent_variance_px2'] += 1.
    assert signature != A._plan_signature(changed)
    p = A.NATIVE.V._pose_vector(R, t, problem.scale) + [.01, -.02, .005, .008, -.011, .025]
    value, gradient, parts = problem.objective(p)
    numeric = A.NATIVE.V._jacobian(lambda x: np.array([problem.objective(x)[0]]), p)[0]
    assert np.allclose(gradient, numeric, atol=4e-5, rtol=4e-6)
    assert value == pytest.approx(sum(c['weighted_robust_cost'] for c in parts['native_costs']))
    v, j = problem.constraints(p, problem.sides[0])
    vv, jj = raw.constraints(p, raw.sides[0])
    assert np.array_equal(v, vv) and np.array_equal(j, jj)
    assert task == before
