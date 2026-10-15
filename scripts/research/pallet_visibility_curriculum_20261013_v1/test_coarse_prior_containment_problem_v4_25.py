"""Synthetic N/O/S/coarse/C composition and family-before-auth contracts."""
from copy import deepcopy

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from . import coarse_prior_containment_problem_v4_25 as NEW
from . import coarse_pose_prior_v4_25 as COARSE
from .test_continuous_outline_problem_v4_24 import DummyBase, numeric, PRIORS
from .test_global_outline_analytic_factor_v4_11 import scene
from .test_native_edge_problem_v4_17 import problem as native_fixture


class ThreePriorBase(DummyBase):
    def objective(self, p):
        value, gradient, parts = super().objective(p)
        extra = float(.6 * (p[2] - p[5]) ** 2)
        derivative = np.array([0., 0., 1.2 * (p[2] - p[5]), 0., 0., -1.2 * (p[2] - p[5])])
        self.last_parts = dict(parts,
            regularizer_costs=parts['regularizer_costs'] + [dict(kind='CAL_COARSE_POSE_PRIOR', weighted_cost=extra)],
            regularizer_objective=parts['regularizer_objective'] + extra,
            total_regularized_objective=value + extra, CAL_coarse_pose_prior_objective=extra,
            coarse_pose_prior_count=1)
        return value + extra, gradient + derivative, self.last_parts


@pytest.mark.parametrize('translation', [[0., .3, 2.2], [-.9, .3, 2.2]])
def test_N_plus_three_priors_plus_one_weighted_C_gradient_eval_and_ledger(translation):
    xyz, K, scale, p, observation = scene(translation, [.12, .65, .07])
    p = p + [.013, -.017, .011, .009, -.007, .019]
    base = ThreePriorBase(xyz, K, scale)
    wrapped = NEW.ContinuousContainmentProblem(base, observation, {'outline_factor_weight': 2.75})
    nv, ng, parts = base.objective(p)
    saved = deepcopy(parts)
    cv, cg, _ = NEW.FACTOR.outline_value_and_gradient(p, xyz, K, scale, observation)
    assert cv > 0.
    value, gradient, audit = wrapped.objective(p)
    assert value == pytest.approx(nv + 2.75 * cv, abs=1e-11)
    np.testing.assert_allclose(gradient, ng + 2.75 * cg, atol=1e-11, rtol=1e-13)
    np.testing.assert_allclose(gradient, numeric(lambda pp: wrapped.objective(pp)[0], p),
                               atol=3e-5, rtol=2e-5)
    assert parts == saved
    assert audit['native_costs'] == saved['native_costs']
    assert audit['regularizer_costs'] == saved['regularizer_costs']
    assert [r['kind'] for r in audit['regularizer_costs']] == list(NEW.REGULARIZER_KINDS)
    for key in ('native_observation_objective', 'regularizer_objective', 'total_regularized_objective',
                'CAL_coarse_pose_prior_objective', 'coarse_pose_prior_count'):
        assert audit[key] == saved[key]
    assert audit['containment_count_in_optimizer'] == 1
    assert audit['native_plus_coarse_regularized_objective'] == nv
    assert audit['weighted_containment_objective'] == pytest.approx(2.75 * cv)
    assert audit['total_optimization_objective'] == pytest.approx(value)
    assert audit['containment_not_in_native_costs_or_three_model_regularizers']
    assert audit['old_bidirectional_full_outline_factor_used'] is False
    R, t = Rotation.from_rotvec(p[:3]).as_matrix(), p[3:] * scale
    checked = wrapped.evaluate(R, t)
    assert checked['objective'] == pytest.approx(value, abs=1e-9)
    assert checked['native_costs'] == audit['native_costs']
    assert checked['regularizer_costs'] == audit['regularizer_costs']
    assert checked['scalar_and_vector_guard_token'] == 'EXACT_BASE_GUARDS'
    assert checked['containment_count_in_optimizer'] == 1


def test_real_native_OS_CAL_coarse_and_containment_preserve_all_scalar_and_Jacobian_guards():
    args, _, _, native = native_fixture(typed=True)
    R, t = np.asarray(args[6]['R']), np.asarray(args[6]['t'])
    p = NEW.G._pose_vector(R, t, native.scale)
    q = NEW.G.project_native(native.xyz, R, t, native.K)[0]
    oid = next(k for k, state in enumerate(native.states) if state == 'O')
    region = dict(native_corner=oid, state='O', shape='ellipse', localization_available=True,
        independent_precise_support=False, center_xy=(q[oid] - [24., 10.]).tolist(),
        half_axes_px=[4., 7.], angle_deg=17.)
    regions = PRIORS.O.prepare_o_localization_regions([region], native.states, [s == 'S' for s in native.states])
    visibility_model = dict(schema=PRIORS.S.MODEL_SCHEMA, fit_partition='CAL',
        fixed_pseudocounts=PRIORS.S.PSEUDOCOUNTS, geometry_tolerance_deg=2., epsilon=.3)
    visibility_model['model_digest_sha256'] = PRIORS.S.model_digest(visibility_model)
    visibility = PRIORS.S.prepare_bounded_coarse_visibility_prior(native.xyz, R, t,
        [s == 'S' for s in native.states], visibility_model)
    bounded = PRIORS.BoundedGeometricPriorsProblem(native, regions, dict(epsilon=.2, huber_scale=3.), visibility)
    prepared = COARSE.prepare_coarse_pose_prior(native.xyz,
        dict(R=R, t=t, physical_registration_Q=native.Q),
        np.diag([.03, .04, .05, .02, .025, .06]) ** 2,
        [.01, -.005, .002, .003, -.001, .012], calibration_model_digest_sha256='a' * 64)
    base = COARSE.CoarsePosePriorProblem(bounded, COARSE.rebind_coarse_pose_prior(prepared, native.xyz, native.Q))
    mask = np.zeros(native.image_hw, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(cv2.convexHull(q.astype(np.float32))).astype(np.int32), 1)
    observation = NEW.CONFIG.observe_global_outline(mask, native.image_hw)
    wrapped = NEW.ContinuousContainmentProblem(base, observation, {'outline_factor_weight': 1.7})
    assert wrapped.Q is native.Q and wrapped.measurements is native.measurements
    assert wrapped.domains is native.domains and wrapped.regions is native.regions
    for sides in native.sides:
        before = base.constraints(p, sides)
        after = wrapped.constraints(p, sides)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)
    be, we = base.evaluate(R, t), wrapped.evaluate(R, t)
    for key in ('accepted', 'minimum_joint_slack', 'visible_protection_audit', 'joint_constraint_audit',
                'U_region_constraint_audit', 'boundary_observation_audit', 'R_physical',
                'all_original_hard_guards_recomputed', 'native_costs', 'regularizer_costs'):
        assert we[key] == be[key]
    assert [r['kind'] for r in we['regularizer_costs']] == list(NEW.REGULARIZER_KINDS)
    assert we['objective'] == pytest.approx(wrapped.objective(p)[0], abs=1e-9)
    assert we['containment_count_in_optimizer'] == we['coarse_pose_prior_count'] == 1


@pytest.mark.parametrize('supported,weight', [(False, 2.), (True, 0.)])
def test_unsupported_or_zero_weight_exact_numeric_identity_without_projection(supported, weight, monkeypatch):
    xyz, K, scale, p, _ = scene([0., .3, 2.2], [.12, .65, .07])
    base = ThreePriorBase(xyz, K, scale)
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Inactive containment queried model geometry')
    base.xyz = base.K = Poison()
    def forbidden(*args, **kwargs):
        raise AssertionError('Inactive containment factor was called')
    monkeypatch.setattr(NEW.FACTOR, 'outline_value_and_gradient', forbidden)
    wrapped = NEW.ContinuousContainmentProblem(base, dict(supported=supported), {'outline_factor_weight': weight})
    bv, bg, bp = base.objective(p)
    value, gradient, audit = wrapped.objective(p)
    assert value == bv
    np.testing.assert_array_equal(gradient, bg)
    assert audit['native_costs'] == bp['native_costs'] and audit['regularizer_costs'] == bp['regularizer_costs']
    assert audit['containment_count_in_optimizer'] == 0
    assert audit['global_containment_cost'] == audit['weighted_containment_objective'] == 0.
    R, t = Rotation.from_rotvec(p[:3]).as_matrix(), p[3:] * scale
    assert wrapped.evaluate(R, t)['objective'] == base.evaluate(R, t)['objective']
    for a, b in zip(base.constraints(p, base.sides[0]), wrapped.constraints(p, base.sides[0])):
        np.testing.assert_array_equal(a, b)


def test_base_rejection_returns_exact_object_and_invalid_containment_cannot_forge_acceptance(monkeypatch):
    xyz, K, scale, p, observation = scene([0., .3, 2.2], [.12, .65, .07])
    base = ThreePriorBase(xyz, K, scale)
    wrapped = NEW.ContinuousContainmentProblem(base, observation, {})
    rejected = dict(accepted=False, objective=None, reason='original hard guard rejected')
    monkeypatch.setattr(base, 'evaluate', lambda *args: rejected)
    def forbidden(*args, **kwargs):
        raise AssertionError('A rejected native candidate queried containment')
    with monkeypatch.context() as patch:
        patch.setattr(NEW.FACTOR, 'outline_value_and_gradient', forbidden)
        assert wrapped.evaluate(object(), object()) is rejected
    monkeypatch.undo()
    monkeypatch.setattr(NEW.FACTOR, 'outline_value_and_gradient',
                        lambda *a, **kw: (1e12, np.zeros(6), dict(valid_projection=False)))
    checked = wrapped.evaluate(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:] * scale)
    assert checked['accepted'] is False and checked['objective'] is None
    assert checked['containment_numeric_validity_failed_not_new_physical_guard']
    with pytest.raises(ValueError, match='valid conditional foreground-containment'):
        wrapped.objective(p)


STATES = ['B', 'I', 'U', 'O', 'T', 'B', 'S', 'S']


def candidate(name, family='ORIGINAL_WD', native=(1., 1.), priors=(0., 0., 0.), shape=0.):
    return dict(name=name, family=family, accepted=True, all_joint_constraints_satisfied=True,
        native_costs=[dict(kind='BI_COMPLETE', native_corner=k, weighted_robust_cost=c) for k, c in zip((0, 1), native)],
        regularizer_costs=[dict(kind=kind, weighted_cost=c) for kind, c in zip(NEW.REGULARIZER_KINDS, priors)],
        global_containment_cost=shape, global_mask_cost=shape,
        historical_whole_mask_cost=object(), actual_pose=object(), frame_id=object(), metrics=object())


def test_three_priors_plus_containment_sum_once_and_no_native_Pareto_prefilter():
    rows = [candidate('native_frontier', native=(1., 1.), priors=(5., 3., 2.), shape=2.),
            candidate('dominated_but_regularized_best', native=(2., 2.), priors=(.2, .3, .4), shape=.5)]
    native = NEW.OLD.AUTH.select_hierarchical_native_candidates(STATES, rows)
    assert native['pareto_frontier_indices'] == [0]
    selected = NEW.select_candidates(STATES, rows, containment_weight=2.)
    assert selected['selected_name'] == 'dominated_but_regularized_best'
    assert len(selected['candidates']) == 2
    chosen = selected['candidates'][1]
    assert chosen['total_energy'] == pytest.approx(4. + .2 + .3 + .4 + 2. * .5)
    assert chosen['weighted_containment_objective'] == 1.
    assert selected['native_Pareto_is_audit_not_regularizer_prefilter']
    assert not selected['old_full_outline_factor_or_tie_break_used']


def test_native_numeric_ties_are_stable_without_old_wholemask_or_containment_tiebreak():
    rows = [candidate('first', shape=100.), candidate('later_shape_better', native=(1. + 1e-15, 1.), shape=0.)]
    # AUTH's legacy numeric tie selects shape-better; the new selector must not reuse that selection.
    assert NEW.OLD.AUTH.select_hierarchical_native_candidates(STATES, rows)['selected_name'] == 'later_shape_better'
    selected = NEW.select_candidates(STATES, rows, containment_weight=0.)
    assert selected['selected_name'] == 'first'
    assert [r['global_containment_cost'] for r in selected['candidates']] == [100., 0.]


def test_locked_family_filters_raw_before_auth_and_keeps_globally_dominated_candidate():
    class Poison:
        def __iter__(self):
            raise AssertionError('Excluded family fields were authenticated')
    rows = [candidate('better_wrong_family', native=(1., 1.)),
            candidate('allowed_dominated', family='SWAPPED_WD', native=(2., 2.))]
    assert NEW.OLD.AUTH.select_hierarchical_native_candidates(STATES, rows)['pareto_frontier_indices'] == [0]
    rows[0]['native_costs'] = Poison()
    rows.append(dict(family='SWAPPED_WD', accepted=False, native_costs=Poison()))
    selected = NEW.select_candidates(STATES, rows, locked_family='SWAPPED_WD')
    assert selected['selected_name'] == 'allowed_dominated'
    assert selected['selected_family'] == 'SWAPPED_WD'
    assert selected['family_filter_before_authentication']
    assert selected['predicted_e_family_constraint'] == 'SWAPPED_WD'
    assert not selected['e_numeric_pose_used'] and not selected['old_best_pose_fallback']


def test_absent_locked_family_has_no_selection_and_no_old_numeric_pose_fallback():
    selected = NEW.select_candidates(STATES, [candidate('original_only')], locked_family='SWAPPED_WD')
    assert selected['selected_name'] is selected['selected_family'] is None
    assert selected['candidates'] == []
    assert not selected['e_numeric_pose_used'] and not selected['old_best_pose_fallback']
    assert not any(k in selected for k in ('actual_pose', 'R', 't'))


@pytest.mark.parametrize('bad', ['missing', 'wrong_order', 'negative', 'old_mask_scalar'])
def test_incomplete_or_polluted_cost_ledgers_are_rejected(bad):
    row = candidate('one')
    if bad == 'missing':
        row['regularizer_costs'].pop()
    elif bad == 'wrong_order':
        row['regularizer_costs'].reverse()
    elif bad == 'negative':
        row['regularizer_costs'][2]['weighted_cost'] = -1.
    else:
        row['global_mask_cost'] = 5.
    with pytest.raises(ValueError):
        NEW.select_candidates(STATES, [row], containment_weight=0.)


@pytest.mark.parametrize('family', ['ORIGINAL', 'SWAPPED', True])
def test_invalid_family_labels_rejected(family):
    with pytest.raises(ValueError, match='GT-free e family label'):
        NEW.select_candidates(STATES, [candidate('one')], locked_family=family)
