"""The new V31 admission changes no frozen objective, derivatives or guards."""
from copy import copy, deepcopy
import numpy as np
import pytest

from . import geometry_guided_edge_problem_v4_31 as G
from . import geometry_guided_edge_authentication_v4_31 as A
from .test_geometry_guided_edge_authentication_v4_31 import fixture


@pytest.mark.parametrize('two,partial,policy', [
    (False, False, 'ALL_BI_REPLACE'), (True, False, 'ALL_BI_REPLACE'),
    (False, True, 'RGB_NONFALLBACK_PRESERVE'), (True, True, 'RGB_NONFALLBACK_PRESERVE')])
def test_same_admitted_bundle_preserves_parent_value_gradient_hardguard_J_exact(two, partial, policy):
    base, observation, kwargs, _ = fixture(two=two, partial=partial)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    parent_bundle = {**bundle, 'observation_family': G.PARENT.OBSERVATION_FAMILY, 'population': G.PARENT.POPULATION}
    old = G.PARENT.GeometryGuidedEdgeProblem(base, parent_bundle, source_policy=policy, tangent_gamma=.3)
    new = G.GeometryGuidedEdgeProblem(base, bundle, source_policy=policy, tangent_gamma=.3)
    geometry = kwargs['initial_geometry']
    p = G.V._pose_vector(np.asarray(geometry['R_cf']), np.asarray(geometry['centroid']), base.scale)
    p = p + [.01, -.02, .005, .008, -.011, .025]
    av, aj, ap = old.objective(p); bv, bj, bp = new.objective(p)
    assert av == bv and np.array_equal(aj, bj)
    assert ap['native_costs'] == bp['native_costs']
    for side in old.sides:
        a, ja = old.constraints(p, side); b, jb = new.constraints(p, side)
        assert np.array_equal(a, b) and np.array_equal(ja, jb)
    assert new.guard_boundary == old.guard_boundary
    assert bp['v31_cost_plan']['selection_profile'] == G.SELECTION_PROFILE
    assert not bp['v31_cost_plan']['old_V29_or_V30_noise_relabelled']


@pytest.mark.parametrize('reason', ['NO_AUTHENTICATED_NEW_LINES', 'MODEL_UNAVAILABLE_UNIFORM_POLICY'])
def test_empty_evidence_or_new_noise_unavailable_exact_no_owner_baseline(reason):
    base, observation, kwargs, _ = fixture(partial=True)
    bundle = G.empty_geometry_guided_evidence(base, initial_geometry=kwargs['initial_geometry'],
        image_binding=kwargs['image_binding'], association_configuration=observation['association_configuration'], reason=reason)
    new = G.GeometryGuidedEdgeProblem(base, bundle, source_policy='ALL_BI_REPLACE', tangent_gamma=0.)
    p = np.array([.1, -.2, .01, .03, -.04, 2.])
    av, aj, _ = base.objective(p); bv, bj, _ = new.objective(p)
    assert av == bv and np.array_equal(aj, bj)
    for side in base.sides:
        a, ja = base.constraints(p, side); b, jb = new.constraints(p, side)
        assert np.array_equal(a, b) and np.array_equal(ja, jb)
    assert bundle['selection_profile'] == G.SELECTION_PROFILE and not bundle['old_noise_substitution']
    assert new.plan['new_owner_ids'] == [] and new.guard_boundary == base.boundary


def test_original_ray_binding_is_not_replaced_by_trial_WD_basis_or_trial_pose():
    base, observation, kwargs, _ = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    angle = np.pi / 2
    H = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    trial = copy(base)
    trial.xyz = np.asarray(base.xyz) @ H
    trial.Q = H.T @ base.Q
    # Geometry binding is original, while native problem uses a reindexed WD basis.
    plan = G.plan_geometry_guided_replacement(trial, bundle, source_policy='ALL_BI_REPLACE')
    assert plan['initial_model_ray_binding'] == bundle['initial_model_ray_binding']
    assert plan['original_BI_query_and_native_ray_association_recomputed'] == G.plan_geometry_guided_replacement(
        base, bundle, source_policy='ALL_BI_REPLACE')['original_BI_query_and_native_ray_association_recomputed']


def test_nonzero_native_origin_uses_model_center_not_native_t_as_center():
    base, observation, kwargs, _ = fixture()
    shifted = copy(base); offset = np.array([.3, -.2, .1])
    shifted.xyz = np.asarray(base.xyz) + offset
    geometry = deepcopy(kwargs['initial_geometry'])
    geometry['native_xyz'] = shifted.xyz.tolist()
    # Physical center of the native box changes under this synthetic origin;
    # rebinding source and trial alike keeps native t=center-R*mean(X).
    center = np.asarray(geometry['centroid']) + np.asarray(geometry['R_cf']) @ offset
    geometry['centroid'] = center.tolist()
    native_t = center - np.asarray(geometry['R_cf']) @ shifted.xyz.mean(axis=0)
    projection, _ = G.V.project_native(shifted.xyz, np.asarray(geometry['R_cf']), native_t, base.K)
    geometry['initial_projection_xy'] = projection.tolist()
    binding = G.bind_initial_geometry(shifted, geometry)
    assert np.allclose(binding['initial_projection_xy'][0], projection[0], atol=1e-12)
    wrong = deepcopy(geometry)
    wrong['initial_projection_xy'], _ = G.V.project_native(shifted.xyz, np.asarray(geometry['R_cf']), center, base.K)
    with pytest.raises(ValueError, match='original native model geometry'):
        G.bind_initial_geometry(shifted, wrong)


def test_core_cannot_accept_old_population_or_forged_new_query_even_with_pass_ledger():
    base, observation, kwargs, _ = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    bad = deepcopy(bundle); bad['population'] = G.V30.POPULATION
    with pytest.raises(ValueError, match='Separate V31'):
        G.GeometryGuidedEdgeProblem(base, bad, source_policy='ALL_BI_REPLACE', tangent_gamma=.3)
    bad = deepcopy(bundle); guide = bad['groups'][0]['selected_chain_ref']['guide_reference']
    guide['search_anchor_kind'] = 'MODEL_PROJECTION_SEARCH_ONLY_BAND'
    identity = G._digest({k: v for k, v in guide.items() if k != 'guide_reference_id'})
    guide['guide_reference_id'] = identity
    bad['groups'][0]['selected_chain_ref']['guide_reference_id'] = identity
    bad['original_BI_query_and_native_ray_association_recomputed'] = [{'passes': True}]
    with pytest.raises(ValueError, match='model_projection_query'):
        G.GeometryGuidedEdgeProblem(base, bad, source_policy='ALL_BI_REPLACE', tangent_gamma=.3)
