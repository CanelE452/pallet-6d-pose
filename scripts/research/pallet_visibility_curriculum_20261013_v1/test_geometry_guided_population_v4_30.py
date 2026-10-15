"""Version boundary and unchanged physical energy/guards; synthetic data only."""
from copy import deepcopy
import numpy as np
import pytest

from . import geometry_guided_edge_problem_v4_30 as G
from . import geometry_guided_edge_authentication_v4_30 as A
from .test_geometry_guided_edge_problem_v4_29 import fixture as problem_fixture
from .test_geometry_guided_edge_authentication_v4_29 import fixture as auth_fixture


def new_bundle(bundle):
    return {**bundle, 'observation_family': G.OBSERVATION_FAMILY, 'population': G.POPULATION,
        'selection_profile': G.SELECTION_PROFILE,
        'association_configuration': dict(normal_search_radius_px=43.)}


def new_auth_fixture():
    base, obs, kw, ids = auth_fixture()
    obs['selection_profile'] = G.SELECTION_PROFILE
    obs['association_configuration'] = dict(normal_search_radius_px=43.)
    obs['chain_observation']['configuration'] = deepcopy(obs['association_configuration'])
    kw['noise_provenance'].update(observation_family=G.OBSERVATION_FAMILY,
        selection_profile=G.SELECTION_PROFILE)
    return base, obs, kw, ids


@pytest.mark.parametrize('partial,two,policy', [(False, False, 'ALL_BI_REPLACE'),
    (True, False, 'ALL_BI_REPLACE'), (False, True, 'ALL_BI_REPLACE'),
    (True, False, 'RGB_NONFALLBACK_PRESERVE')])
def test_same_factors_have_bit_exact_parent_energy_gradient_and_guards(partial, two, policy):
    base, bundle, _, R, t = problem_fixture(partial=partial, two=two)
    before = G._digest(bundle)
    old = G.PARENT.GeometryGuidedEdgeProblem(base, bundle, source_policy=policy, tangent_gamma=.2)
    new = G.GeometryGuidedEdgeProblem(base, new_bundle(bundle), source_policy=policy, tangent_gamma=.2)
    p = G.V._pose_vector(R, t, base.scale) + [.01, -.02, .005, .008, -.011, .025]
    ov, oj, op = old.objective(p)
    nv, nj, nparts = new.objective(p)
    assert ov == nv and np.array_equal(oj, nj)
    assert op['native_costs'] == nparts['native_costs']
    for sides in old.sides:
        os, od = old.constraints(p, sides)
        ns, nd = new.constraints(p, sides)
        assert np.array_equal(os, ns) and np.array_equal(od, nd)
    assert old.evaluate(R, t)['accepted'] == new.evaluate(R, t)['accepted']
    assert nparts['v30_cost_plan'] is new.plan
    assert new.plan['selection_profile'] == G.SELECTION_PROFILE
    assert G._digest(bundle) == before
    assert G.PARENT.OBSERVATION_FAMILY.endswith('_V29')


@pytest.mark.parametrize('field,value', [('observation_family', G.PARENT.OBSERVATION_FAMILY),
    ('population', G.PARENT.POPULATION), ('selection_profile', 'OLD_WINNER')])
def test_old_population_or_unbound_profile_rejects_before_line_payload(field, value):
    base, old, _, _, _ = problem_fixture()
    bundle = new_bundle(old)
    bundle[field] = value
    bundle['groups'] = [None]
    with pytest.raises(ValueError, match='Separate V30'):
        G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')


def test_actual_provenance_and_new_family_profile_are_bound_together():
    base, obs, kw, _ = new_auth_fixture()
    before = G._digest(obs)
    result = A.authenticate_geometry_guided_edges(base, obs, **kw)
    assert result['observation_family'] == G.OBSERVATION_FAMILY
    assert result['producer_generation'] == G.RAW_OBSERVATION_GENERATION
    assert result['selection_profile'] == G.SELECTION_PROFILE
    assert all(c['passes'] for c in result['original_BI_association_recomputed'])
    assert G._digest(obs) == before


def test_old_noise_or_old_winner_cannot_be_relabelled_by_only_one_tag():
    base, obs, kw, _ = new_auth_fixture()
    bad = deepcopy(kw)
    bad['noise_provenance']['observation_family'] = G.PARENT.OBSERVATION_FAMILY
    with pytest.raises(ValueError, match='Separately fitted V30'):
        A.authenticate_geometry_guided_edges(base, obs, **bad)
    obs.pop('selection_profile')
    with pytest.raises(ValueError, match='New V30 selected'):
        A.authenticate_geometry_guided_edges(base, obs, **kw)


def test_far_line_fails_original_q_association_before_raw_TLS_payload():
    base, obs, kw, _ = new_auth_fixture()
    obs['joint_graph']['line_candidates'][0]['offset'] += 500.
    obs['chain_observation']['edge_records'][0]['chains'] = None
    with pytest.raises(ValueError, match='original B/I association'):
        A.authenticate_geometry_guided_edges(base, obs, **kw)


def test_S_gate_precedes_node_line_and_archive_reads():
    base, obs, kw, _ = new_auth_fixture()
    hidden = next(k for k, state in enumerate(base.states) if state == 'S')
    obs['chain_observation']['edge_records'][0]['native_neighbor'] = hidden
    obs['joint_graph']['line_candidates'] = []
    with pytest.raises(ValueError, match='State/S gates'):
        A.authenticate_geometry_guided_edges(base, obs, **kw)


def test_association_reach_cannot_differ_from_measured_chain_configuration():
    base, obs, kw, _ = new_auth_fixture()
    obs['association_configuration']['normal_search_radius_px'] = 5000.
    with pytest.raises(ValueError, match='original raw-chain configuration'):
        A.authenticate_geometry_guided_edges(base, obs, **kw)
