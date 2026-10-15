"""Integration contracts for replacement factors and fresh joint feasibility."""
from copy import deepcopy

import numpy as np
import pytest

from . import native_edge_problem_v4_17 as NEW
from . import hierarchical_native_candidate_score_v4_17 as AUTH
from . import projected_native_edge_factor_v4_17 as EDGE
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from .test_hierarchical_native_candidate_score_v4_17 import fixture, auth
from .test_statewise_partial_boundary_u_solver_v4_6 import region


def problem(typed=True, replace=True, with_region=True):
    a, b, kw = fixture(typed)
    bundle = auth(a, [b], kw)
    if not replace:
        bundle = dict(edges=[], replaces_normal_factor_refs=[])
    regions = [region(a)] if with_region else []
    result = NEW.prepare_native_edge_problem(*a[:7], config=a[7],
        boundary_observations=[b], u_regions=regions, edge_bundle=bundle)
    return a, b, bundle, result


@pytest.mark.parametrize('typed', [False, True])
def test_replacement_exactly_once_and_full_gradient(typed):
    a, b, bundle, p = problem(typed)
    at = NEW.V._pose_vector(np.asarray(a[6]['R']), np.asarray(a[6]['t']), p.scale)
    at += np.array([.01, -.02, .005, .008, -.011, .025])
    value, derivative, audit = p.objective(at)
    numerical = NEW.V._jacobian(lambda pp: np.array([p.objective(pp)[0]]), at)[0]
    assert np.allclose(derivative, numerical, atol=3e-5, rtol=3e-6)
    assert value == pytest.approx(sum(v['weighted_robust_cost'] for v in audit['native_costs']))
    assert len(p.edges) == 1 and len(p.replaced) == 1 and p.remaining_normals == []
    assert b['native_corner'] not in [m['native_corner'] for m in p.complete]
    edge_value = EDGE.projected_edge_value_and_gradient(at, p.xyz, p.K, p.scale,
        p.edges[0], weight=p.edges[0]['fit_reliability_weight'], huber_scale=p.cfg['huber_scale'])[0]
    edge_costs = [v for v in audit['native_costs'] if v['kind'] == 'PROJECTED_NATIVE_EDGE']
    assert len(edge_costs) == 1 and edge_costs[0]['weighted_robust_cost'] == edge_value
    assert not any(v['kind'] == 'UNREPLACED_PARTIAL_NORMAL' for v in audit['native_costs'])
    assert audit['coarse_pose_prior_loss'] == audit['global_outline_objective_weight'] == 0.
    assert audit['S_image_terms'] == audit['U_image_terms'] == 0
    assert all(p.states[v['native_corner']] in ('B', 'I') for v in audit['native_costs'])


def test_unreplaced_line_keeps_old_normal_and_uncertain_archived_tangent_once():
    a, b, bundle, p = problem(False, replace=False)
    at = NEW.V._pose_vector(np.asarray(a[6]['R']), np.asarray(a[6]['t']), p.scale)
    at += np.array([.01, -.02, .005, .008, -.011, .025])
    _, _, audit = p.objective(at)
    assert len(p.remaining_normals) == 1 and not p.edges and not p.replaced
    assert sum(v['kind'] == 'UNREPLACED_PARTIAL_NORMAL' for v in audit['native_costs']) == 1
    archived = T._tangent_records(p.boundary, 4.)
    assert sum(v['kind'] == 'UNCERTAIN_ARCHIVED_TANGENT' for v in audit['native_costs']) == len(archived)


def test_fresh_constraints_and_scalar_crosscheck_reject_changed_U_without_point_term():
    a, b, bundle, p = problem()
    R, t = np.asarray(a[6]['R']), np.asarray(a[6]['t'])
    accepted = p.evaluate(R, t)
    assert accepted['all_original_hard_guards_recomputed']
    # Other synthetic canvas/visibility constraints can reject this fixture;
    # the U guard itself must independently detect a translated region.
    assert all(v['feasible'] for v in accepted['U_region_constraint_audit'])
    at = NEW.V._pose_vector(R, t, p.scale)
    value, derivative, _ = p.objective(at)
    p.regions[0]['center_xy'] += 2000.
    rejected = p.evaluate(R, t)
    assert not rejected['accepted']
    assert not all(v['feasible'] for v in rejected['U_region_constraint_audit'])
    shifted_value, shifted_derivative, _ = p.objective(at)
    assert shifted_value == value and np.array_equal(derivative, shifted_derivative)
    assert not p.evaluate(R, np.array([0., 0., -4.]))['accepted']
    assert not p.evaluate(R * 2., t)['accepted']


def test_all_hard_guard_jacobians_and_input_immutability():
    a, b, kw = fixture(True)
    saved_visible, saved_boundary = deepcopy(a[3]), deepcopy(b)
    bundle = auth(a, [b], kw); saved_bundle = deepcopy(bundle)
    p = NEW.prepare_native_edge_problem(*a[:7], config=a[7],
        boundary_observations=[b], u_regions=[region(a)], edge_bundle=bundle)
    at = NEW.V._pose_vector(np.asarray(a[6]['R']), np.asarray(a[6]['t']), p.scale)
    values, jacobian = p.constraints(at, p.sides[0])
    numeric = NEW.V._jacobian(lambda pp: p.constraints(pp, p.sides[0])[0], at)
    assert np.allclose(jacobian, numeric, atol=3e-5, rtol=3e-6)
    p.objective(at); p.evaluate(np.asarray(a[6]['R']), np.asarray(a[6]['t']))
    assert a[3] == saved_visible and b == saved_boundary and bundle == saved_bundle


def test_duplicate_replacement_and_strengthened_noise_are_rejected():
    a, b, kw = fixture(True); bundle = auth(a, [b], kw)
    def construct(other):
        return NEW.prepare_native_edge_problem(*a[:7], config=a[7],
            boundary_observations=[b], u_regions=[], edge_bundle=other)
    bad = deepcopy(bundle); bad['replaces_normal_factor_refs'] *= 2
    with pytest.raises(ValueError, match='only once'):
        construct(bad)
    bad = deepcopy(bundle); bad['edges'][0]['normal_variance_px2'] *= .5
    with pytest.raises(ValueError, match='strengthened'):
        construct(bad)


def test_S_gate_precedes_copy_of_poison_coordinates_and_noise():
    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError('S fields read before gate')
    a, b, kw = fixture(True)
    bundle = auth(a, [b], kw)
    k = next(k for k, s in enumerate(a[2]) if s == 'S')
    bad = dict(edges=[dict(native_corner=k, native_neighbor=0, support_anchor_xy=Poison())],
        replaces_normal_factor_refs=[])
    with pytest.raises(ValueError, match='Only BI'):
        NEW.prepare_native_edge_problem(*a[:7], config=a[7], boundary_observations=[b],
            u_regions=[], edge_bundle=bad)
