"""Synthetic original producer references; no cached rows or image/label IO."""
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
import pytest

from . import geometry_guided_edge_authentication_v4_32 as A
from . import geometry_guided_edge_problem_v4_32 as G
from . import image_edge_observations_v4_29 as PRODUCER
from .test_geometry_guided_edge_authentication_v4_29 import fixture as original_fixture


def fixture(two=False, partial=False, measured=False):
    base, observation, kwargs, ids = original_fixture(two=two, partial=partial)
    geometry = kwargs['initial_geometry']
    points = G.bound_BI_points(base)
    for reference in observation['selected_chain_refs']:
        ri, ci = reference['ray_index'], reference['chain_index']
        ray = observation['chain_observation']['edge_records'][ri]
        chain = ray['chains'][ci]
        if measured:
            k, mate = ray['native_corner'], ray['native_neighbor']
            anchor = np.asarray(points[k]['record']['xy'])
            direction = np.asarray(points[mate]['record']['xy']) - anchor
            direction /= np.linalg.norm(direction)
            ray['guide_proposal'] = dict(guide_id='MEASURED_BI_ENDPOINTS',
                guide_kind='MEASURED_BI_ENDPOINT_DIRECTION', direction_xy=direction.tolist(),
                search_anchor_kind='OBSERVED_BI_REFERENCE_BAND', anchor_xy=anchor.tolist(),
                observed_BI_reference_xy=anchor.tolist(), physical_axis=0)
            ray['actual_query_strip_audit'] = dict(synthetic=True, actual_query_center='immutable BI q')
        produced = PRODUCER._chain_reference(ray, chain, ri, ci, geometry, base.states, points)
        reference.update(deepcopy(produced))
        chain.update(guide_reference=deepcopy(produced['guide_reference']),
            guide_reference_id=produced['guide_reference_id'],
            generating_guide_reference_id=produced['guide_reference_id'],
            generating_guide_direction_xy=deepcopy(produced['guide_reference']['actual_search_direction_xy']))
    observation.update(selection_profile=G.SELECTION_PROFILE,
        association_configuration=dict(normal_search_radius_px=43.))
    observation['chain_observation']['configuration'] = deepcopy(observation['association_configuration'])
    kwargs['noise_provenance'].update(observation_family=G.OBSERVATION_FAMILY,
        noise_population=G.POPULATION, selection_profile=G.SELECTION_PROFILE,
        raw_observation_generation=G.RAW_OBSERVATION_GENERATION)
    return base, observation, kwargs, ids


def reseal(reference, chain):
    guide = reference['guide_reference']
    identity = G._digest({k: v for k, v in guide.items() if k != 'guide_reference_id'})
    guide['guide_reference_id'] = reference['guide_reference_id'] = identity
    chain.update(guide_reference=deepcopy(guide), guide_reference_id=identity,
        generating_guide_reference_id=identity,
        generating_guide_direction_xy=deepcopy(guide['actual_search_direction_xy']))


def test_original_raw_query_aliases_are_copied_then_independently_rechecked():
    base, observation, kwargs, _ = fixture()
    before = deepcopy((observation, kwargs, base.prepartial_bound_source_archives))
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    raw = observation['chain_observation']['edge_records'][0]['chains'][0]
    group = bundle['groups'][0]
    assert group['authenticated_raw_query_aliases'] == {k: raw[k] for k in G.RAW_QUERY_ALIAS_KEYS}
    assert group['authenticated_raw_query_aliases'] is not raw
    plan = G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')
    assert plan['initial_only_query_population'] and not plan['external_poseguide_consumed']
    assert plan['original_BI_query_and_native_ray_association_recomputed'][0]['passes']
    assert plan['initial_source_closure_sha256'] == G.ASSOCIATION.INITIAL_SOURCE_CLOSURE_SHA256
    assert (observation, kwargs, base.prepartial_bound_source_archives) == before


def test_measured_BI_endpoint_source_is_admitted_with_exact_original_query_dictionary():
    base, observation, kwargs, _ = fixture(measured=True)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    check = bundle['original_BI_query_and_native_ray_association_recomputed'][0]
    assert check['initial_only_query_source']['guide_kind'] == 'MEASURED_BI_ENDPOINT_DIRECTION'
    assert check['exact_original_query_source_and_direction_bound']
    assert bundle['groups'][0]['selected_chain_ref'] == observation['selected_chain_refs'][0]
    G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')


@pytest.mark.parametrize('field,value', [('observation_family', G.PREVIOUS.OBSERVATION_FAMILY),
    ('noise_population', G.PREVIOUS.POPULATION), ('selection_profile', G.PREVIOUS.SELECTION_PROFILE),
    ('raw_observation_generation', 'renamed'), ('fit_partition', 'TEST')])
def test_old_or_mixed_noise_rejects_before_raw_payload(field, value):
    base, observation, kwargs, _ = fixture()
    kwargs['noise_provenance'][field] = value
    observation['selected_chain_refs'] = object()
    with pytest.raises(ValueError, match='Separate V32'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


@pytest.mark.parametrize('change', ['cached', 'unknown_kind', 'model_band', 'missing_alias'])
def test_source_preflight_rejects_before_selected_TLS_lookup(change):
    base, observation, kwargs, _ = fixture()
    chain = observation['chain_observation']['edge_records'][0]['chains'][0]
    if change == 'missing_alias':
        del chain['generating_guide_reference_id']
    else:
        ref = chain['guide_reference']
        if change == 'model_band': ref['search_anchor_kind'] = 'MODEL_PROJECTION_SEARCH_ONLY_BAND'
        else: ref['guide_kind'] = 'PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR' if change == 'cached' else 'UNKNOWN'
    class Poison(dict):
        def __getitem__(self, key): raise AssertionError('Forbidden selected TLS looked up')
    observation['joint_graph']['line_candidates'] = [Poison()]
    with pytest.raises(ValueError, match='initial-only query source'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


@pytest.mark.parametrize('key', G.RAW_QUERY_ALIAS_KEYS)
def test_core_does_not_invent_missing_authenticated_aliases(key):
    base, observation, kwargs, _ = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    del bundle['groups'][0]['authenticated_raw_query_aliases'][key]
    with pytest.raises(ValueError, match='original raw query aliases'):
        G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')


def test_core_rejects_ref_or_q_archive_forgery_and_alias_mismatch():
    base, observation, kwargs, (k, _, _) = fixture()
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    bad = deepcopy(bundle)
    bad['groups'][0]['authenticated_raw_query_aliases']['generating_guide_direction_xy'][0] += .1
    with pytest.raises(ValueError, match='original raw query aliases'):
        G.plan_geometry_guided_replacement(base, bad, source_policy='ALL_BI_REPLACE')
    changed = copy(base)
    changed.prepartial_bound_source_archives = deepcopy(base.prepartial_bound_source_archives)
    changed.prepartial_bound_source_archives[str(k)]['original_measurement_source'] = 'forged'
    with pytest.raises(ValueError, match='immutable BI archive'):
        G.plan_geometry_guided_replacement(changed, bundle, source_policy='ALL_BI_REPLACE')


def test_S_gate_and_unknown_mate_gate_do_not_read_forbidden_observed_slots():
    base, observation, kwargs, (_, mate, _) = fixture()
    class Poison:
        def __array__(self, *a, **kw): raise AssertionError('S projection read')
        def __deepcopy__(self, memo): raise AssertionError('S projection copied')
    class Forbidden(dict):
        def __getitem__(self, key): raise AssertionError('Unknown observed archive read')
    hidden = next(i for i, state in enumerate(base.states) if state == 'S')
    kwargs['initial_geometry']['initial_projection_xy'][hidden] = Poison()
    base = copy(base)
    base.states = tuple('U' if i == mate else state for i, state in enumerate(base.states))
    base.measurements = [m for m in base.measurements if m['native_corner'] != mate]
    base.prepartial_bound_source_archives = dict(base.prepartial_bound_source_archives)
    base.prepartial_bound_source_archives[str(mate)] = Forbidden()
    # Original17c's actual query uses its original model ray for an unknown mate.
    ref = observation['selected_chain_refs'][0]
    chain = observation['chain_observation']['edge_records'][0]['chains'][0]
    guide = ref['guide_reference']
    guide['actual_search_direction_xy'] = deepcopy(guide['seed_direction_xy'])
    guide['actual_search_direction_source'] = 'original model direction'
    reseal(ref, chain)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    assert bundle['initial_model_ray_binding']['initial_projection_xy'][hidden] is None
    assert bundle['original_BI_query_and_native_ray_association_recomputed'][0]['unknown_S_U_observed_coordinates_read'] == 0
    G.plan_geometry_guided_replacement(base, bundle, source_policy='ALL_BI_REPLACE')
    class StateOnly(dict):
        def __getitem__(self, key):
            if key in ('native_corner', 'native_neighbor'): return super().__getitem__(key)
            raise AssertionError('S query payload read')
    observation['chain_observation']['edge_records'] = [StateOnly(native_corner=hidden, native_neighbor=0)]
    with pytest.raises(ValueError, match='State/S'):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_actual_gradient_lattice_provenance_and_partial_grade_remain_required():
    base, observation, kwargs, _ = fixture()
    chain = observation['chain_observation']['edge_records'][0]['chains'][0]
    chain.update(proposal_grade='PARTIAL_LINE_PROPOSAL', inlier_span_fraction=.53,
        corrected_endpoint_normal_guard_pass=False)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    assert bundle['groups'][0]['proposal_grade'] == 'PARTIAL_LINE_PROPOSAL'
    assert not bundle['groups'][0]['full_2D_corner_measurement']
    chain['actual_inlier_image_lattice_xy'][0][0] += 1
    with pytest.raises(ValueError):
        A.authenticate_geometry_guided_edges(base, observation, **kwargs)


def test_private_namespaces_and_dependency_source_are_not_mutated():
    modules = (G.PREVIOUS, A.PREVIOUS, G.PARENT, A.PARENT, G.ASSOCIATION, PRODUCER)
    before = [(Path(m.__file__).read_bytes(), dict(vars(m))) for m in modules]
    base, observation, kwargs, _ = fixture(two=True)
    bundle = A.authenticate_geometry_guided_edges(base, observation, **kwargs)
    G.GeometryGuidedEdgeProblem(base, bundle, source_policy='ALL_BI_REPLACE', tangent_gamma=.3)
    assert A._raw_authenticate.__globals__ is not A.PARENT.authenticate_geometry_guided_edges.__globals__
    for module, (content, namespace) in zip(modules, before):
        assert Path(module.__file__).read_bytes() == content
        assert all(vars(module)[key] is value for key, value in namespace.items())
