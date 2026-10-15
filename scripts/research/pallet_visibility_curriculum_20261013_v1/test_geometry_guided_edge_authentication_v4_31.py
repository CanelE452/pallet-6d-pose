"""V31 raw-query/core rebinding contracts with synthetic provenance only."""
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
import pytest

from . import geometry_guided_edge_authentication_v4_31 as A
from . import geometry_guided_edge_problem_v4_31 as G
from .test_geometry_guided_edge_authentication_v4_29 import fixture as old_fixture


def _ref_digest(reference, chain):
    guide = reference['guide_reference']
    identity = G._digest({key: value for key, value in guide.items() if key != 'guide_reference_id'})
    guide['guide_reference_id'] = identity
    reference['guide_reference_id'] = identity
    chain.update(guide_reference=deepcopy(guide), guide_reference_id=identity,
                 generating_guide_reference_id=identity)


def fixture(two=False, partial=False):
    base, observation, kwargs, ids = old_fixture(two=two, partial=partial)
    geometry = kwargs['initial_geometry']
    digest = observation['initial_geometry_digest']
    for reference in observation['selected_chain_refs']:
        k, mate = reference['native_corner'], reference['native_neighbor']
        chain = observation['chain_observation']['edge_records'][reference['ray_index']]['chains'][reference['chain_index']]
        guide = reference['guide_reference']
        guide.update(guide_kind='FROZEN17C_ORIGINAL_MODEL_SEED',
            actual_search_anchor_xy=deepcopy(G._archive(base, k)['original_observed_xy']),
            original_geometry_digest=digest,
            original_reference_projection_xy=deepcopy([geometry['initial_projection_xy'][k], geometry['initial_projection_xy'][mate]]))
        _ref_digest(reference, chain)
    observation.update(selection_profile=G.SELECTION_PROFILE,
        association_configuration=dict(normal_search_radius_px=43.))
    observation['chain_observation']['configuration'] = deepcopy(observation['association_configuration'])
    kwargs['noise_provenance'].update(observation_family=G.OBSERVATION_FAMILY, noise_population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, raw_observation_generation=G.RAW_OBSERVATION_GENERATION)
    return base, observation, kwargs, ids


def test_actual_raw_reference_original_geometry_and_all_BI_archives_bound():
    base, observation, kwargs, (k, mate, _) = fixture()
    before = deepcopy((observation, kwargs, base.prepartial_bound_source_archives))
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    assert (bundle['observation_family'], bundle['population'], bundle['selection_profile']) == (
        G.OBSERVATION_FAMILY, G.POPULATION, G.SELECTION_PROFILE)
    assert bundle['groups'][0]['selected_chain_ref'] == observation['selected_chain_refs'][0]
    binding = bundle['initial_model_ray_binding']
    assert binding['initial_geometry_digest'] == observation['initial_geometry_digest']
    assert binding['initial_projection_xy'][6] is None
    assert binding['S_projection_slots_read'] == 0 and binding['unknown_observed_coordinate_slots_read'] == 0
    assert {o['native_corner'] for o in bundle['groups'][0]['activation_owners']} == {k, mate}
    plan = G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')
    assert plan['original_BI_query_and_native_ray_association_recomputed'][0]['passes']
    assert plan['numerical_factor_and_guard_math'] == 'UNCHANGED_V29'
    assert (observation, kwargs, base.prepartial_bound_source_archives) == before


@pytest.mark.parametrize('field,old', [
    ('observation_family', G.PARENT.OBSERVATION_FAMILY), ('noise_population', G.V30.POPULATION),
    ('selection_profile', G.V30.SELECTION_PROFILE), ('raw_observation_generation', 'relabelled')])
def test_old_noise_or_mixed_population_rejects_before_line_payload(field, old):
    base, observation, kwargs, _ = fixture()
    kwargs['noise_provenance'][field] = old
    observation['selected_chain_refs'] = object()
    with pytest.raises(ValueError, match='Separate V31'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


@pytest.mark.parametrize('change', ['model_anchor', 'wrong_query_anchor', 'reverse_ray', 'raw_ref_alias', 'old_profile'])
def test_new_admission_rejects_independent_of_old_pass_ledger(change):
    base, observation, kwargs, _ = fixture()
    reference = observation['selected_chain_refs'][0]
    chain = observation['chain_observation']['edge_records'][0]['chains'][0]
    guide = reference['guide_reference']
    if change == 'model_anchor':
        guide['search_anchor_kind'] = 'MODEL_PROJECTION_SEARCH_ONLY_BAND'
    elif change == 'wrong_query_anchor':
        guide['actual_search_anchor_xy'][0] += 2.
    elif change == 'reverse_ray':
        direction = (-np.asarray(guide['actual_search_direction_xy'])).tolist()
        guide['actual_search_direction_xy'] = direction
        chain['generating_guide_direction_xy'] = direction
    elif change == 'raw_ref_alias':
        chain['guide_reference'] = {'forged': True}
    else:
        observation['selection_profile'] = G.V30.SELECTION_PROFILE
    if change not in ('raw_ref_alias', 'old_profile'):
        _ref_digest(reference, chain)
    observation['original_BI_query_and_native_ray_association_recomputed'] = [{'passes': True}]
    with pytest.raises(ValueError):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_core_rechecks_q_source_geometry_and_actual_query_after_authentication():
    base, observation, kwargs, (k, _, _) = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    bad = deepcopy(bundle)
    bad['groups'][0]['selected_chain_ref']['guide_reference']['search_anchor_kind'] = 'MODEL_PROJECTION_SEARCH_ONLY_BAND'
    with pytest.raises(ValueError, match='dictionary/digest'):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')
    mutated = copy(base)
    mutated.prepartial_bound_source_archives = deepcopy(base.prepartial_bound_source_archives)
    mutated.prepartial_bound_source_archives[str(k)]['original_measurement_source'] = 'new source'
    with pytest.raises(ValueError, match='immutable BI archive'):
        G.plan_geometry_guided_replacement(mutated, bundle, source_policy='ALL_BI_REPLACE')
    bad = deepcopy(bundle)
    del bad['initial_model_ray_binding']
    with pytest.raises(KeyError):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')
    bad = deepcopy(bundle)
    bad['initial_model_ray_binding']['initial_projection_xy'][0][1] += 5.
    with pytest.raises(ValueError, match='original native model geometry'):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')
    bad = deepcopy(bundle)
    bad['groups'][0]['noise_provenance']['observation_family'] = G.V30.OBSERVATION_FAMILY
    with pytest.raises(ValueError, match='Separate V31'):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')


def test_both_BI_mate_measurement_is_mandatory_instead_of_model_ray_rescue():
    base, observation, kwargs, (_, mate, _) = fixture()
    base = copy(base)
    base.measurements = [m for m in base.measurements if m['native_corner'] != mate]
    with pytest.raises(ValueError, match='both_BI_mate_missing'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_old_actual_RGB_or_lattice_gate_remains_mandatory_after_V31_admission():
    base, observation, kwargs, _ = fixture()
    chain = observation['chain_observation']['edge_records'][0]['chains'][0]
    chain['actual_inlier_image_lattice_xy'][0][0] += 1
    with pytest.raises(ValueError):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_unknown_mate_uses_initial_model_only_and_skips_S_and_unknown_observed_slots():
    base, observation, kwargs, (k, mate, _) = fixture()
    base = copy(base)
    base.states = tuple('U' if i == mate else s for i, s in enumerate(base.states))
    base.measurements = [m for m in base.measurements if m['native_corner'] != mate]
    class Poison:
        def __array__(self, *args, **kw): raise AssertionError('Hidden/unknown observed coordinate read')
        def __deepcopy__(self, memo): raise AssertionError('Hidden/unknown coordinate copied')
    class Forbidden(dict):
        def __getitem__(self, key): raise AssertionError('Unknown observed archive read')
    base.prepartial_bound_source_archives = dict(base.prepartial_bound_source_archives)
    base.prepartial_bound_source_archives[str(mate)] = Forbidden()
    kwargs['initial_geometry']['initial_projection_xy'][6] = Poison()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    check = bundle['original_BI_query_and_native_ray_association_recomputed'][0]
    assert check['reference_source'] == 'ORIGINAL_INITIAL_MODEL_NATIVE_RAY_FOR_UNKNOWN_MATE'
    assert check['unknown_S_U_observed_coordinates_read'] == 0
    assert bundle['initial_model_ray_binding']['initial_projection_xy'][6] is None
    # The runtime trial pose does not rescue a missing original unknown ray.
    bad = deepcopy(bundle)
    bad['initial_model_ray_binding']['initial_projection_xy'][mate] = None
    with pytest.raises(ValueError):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')


def test_S_gate_precedes_raw_chain_fields_and_core_query_fields():
    base, observation, kwargs, _ = fixture()
    hidden = next(i for i, s in enumerate(base.states) if s == 'S')
    class Forbidden(dict):
        def __getitem__(self, key):
            if key in ('native_corner', 'native_neighbor'): return super().__getitem__(key)
            raise AssertionError('S query payload read')
    observation['chain_observation']['edge_records'] = [Forbidden(native_corner=hidden, native_neighbor=0)]
    with pytest.raises(ValueError, match='State/S'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    base, observation, kwargs, _ = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    bundle['groups'] = [Forbidden(native_corner=hidden, native_neighbor=0)]
    with pytest.raises(ValueError, match='State/S'):
        G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')


def test_initial_projection_must_match_original_geometry_even_if_old_digest_excludes_it():
    base, observation, kwargs, _ = fixture()
    kwargs['initial_geometry']['initial_projection_xy'][0][0] += 1.
    with pytest.raises(ValueError, match='original native model geometry'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_new_empty_authentication_is_valid_and_original_globals_source_are_unchanged():
    modules = (G.PARENT, G.V30, A.PARENT, G.ASSOCIATION)
    before = [(Path(m.__file__).read_bytes(), dict(vars(m))) for m in modules]
    base, observation, kwargs, _ = fixture()
    observation['selected_chain_refs'] = []
    observation['joint_graph']['selected_line_names'] = []
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    assert bundle['groups'] == []
    assert G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')['new_owner_ids'] == []
    for module, (content, namespace) in zip(modules, before):
        assert Path(module.__file__).read_bytes() == content
        assert all(vars(module)[key] is value for key, value in namespace.items())
