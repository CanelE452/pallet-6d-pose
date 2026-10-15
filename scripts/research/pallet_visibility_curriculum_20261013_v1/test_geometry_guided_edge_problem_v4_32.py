"""Frozen numerical objective and guard equivalence on synthetic V32 evidence."""
from copy import copy
import numpy as np
import pytest
from . import geometry_guided_edge_problem_v4_32 as G
from . import geometry_guided_edge_authentication_v4_32 as A
from .test_geometry_guided_edge_authentication_v4_32 import fixture


@pytest.mark.parametrize('two,partial,policy', [
    (False, False, 'ALL_BI_REPLACE'), (True, False, 'ALL_BI_REPLACE'),
    (False, True, 'RGB_NONFALLBACK_PRESERVE'), (True, True, 'RGB_NONFALLBACK_PRESERVE')])
def test_values_gradient_all_constraints_J_and_native_ledger_bitexact(two, partial, policy):
    base, observation, kwargs, _ = fixture(two=two, partial=partial)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    numeric = {**bundle, 'observation_family': G.PARENT.OBSERVATION_FAMILY, 'population': G.PARENT.POPULATION}
    old = G.PARENT.GeometryGuidedEdgeProblem(base, numeric, source_policy=policy, tangent_gamma=.3)
    new = G.GeometryGuidedEdgeProblem(base, bundle, source_policy=policy, tangent_gamma=.3)
    p = G.V._pose_vector(np.asarray(kwargs['initial_geometry']['R_cf']),
        np.asarray(kwargs['initial_geometry']['centroid']), base.scale) + [.01, -.02, .005, .008, -.011, .025]
    a, ja, ap = old.objective(p); b, jb, bp = new.objective(p)
    assert a == b and np.array_equal(ja, jb) and ap['native_costs'] == bp['native_costs']
    for side in old.sides:
        c, jc = old.constraints(p, side); d, jd = new.constraints(p, side)
        assert np.array_equal(c, d) and np.array_equal(jc, jd)
    assert old.guard_boundary == new.guard_boundary
    assert bp['v32_cost_plan']['selection_profile'] == G.SELECTION_PROFILE
    assert bp['v32_cost_plan']['numerical_factor_and_guard_math'] == 'UNCHANGED_V29'


@pytest.mark.parametrize('reason', ['NO_AUTHENTICATED_INITIAL_ONLY_LINES', 'NEW_V32_MODEL_UNAVAILABLE'])
def test_empty_uniform_abstention_equals_original_no_owner_baseline(reason):
    base, observation, kwargs, _ = fixture(partial=True)
    bundle = G.empty_geometry_guided_evidence(base, initial_geometry=kwargs['initial_geometry'],
        image_binding=kwargs['image_binding'], association_configuration=observation['association_configuration'], reason=reason)
    new = G.GeometryGuidedEdgeProblem(base, bundle, source_policy='ALL_BI_REPLACE', tangent_gamma=0.)
    p = np.array([.1, -.2, .01, .03, -.04, 2.])
    a, ja, _ = base.objective(p); b, jb, _ = new.objective(p)
    assert a == b and np.array_equal(ja, jb) and new.plan['new_owner_ids'] == []
    for side in base.sides:
        c, jc = base.constraints(p, side); d, jd = new.constraints(p, side)
        assert np.array_equal(c, d) and np.array_equal(jc, jd)
    assert not bundle['old_V31_noise_substitution'] and not bundle['old_noise_substitution']


def test_trial_WD_changes_do_not_rewrite_original_ray_or_channel_association():
    base, observation, kwargs, _ = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    H = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    trial = copy(base); trial.xyz = np.asarray(base.xyz) @ H; trial.Q = H.T @ base.Q
    actual = G.plan_geometry_guided_replacement(trial, bundle, source_policy='ALL_BI_REPLACE')
    expected = G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')
    assert actual['initial_model_ray_binding'] == expected['initial_model_ray_binding']
    assert actual['original_BI_query_and_native_ray_association_recomputed'] == expected['original_BI_query_and_native_ray_association_recomputed']
