"""Fake producer metadata and graph fixtures; actual rows/images/models=0."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from . import initial_only_observed_ray_association_v4_32 as V
from .test_geometry_graph_exact_memo_v4_29 import graph_fixture


class Poison:
    def __array__(self, *args, **kwargs):
        raise AssertionError('Forbidden numeric payload converted')

    def __getitem__(self, key):
        raise AssertionError('Forbidden payload indexed')

    def get(self, *args, **kwargs):
        raise AssertionError('Forbidden payload queried')


def seal(chain):
    ref = chain['guide_reference']
    ref.pop('guide_reference_id', None)
    identity = V.G._digest(ref)
    ref['guide_reference_id'] = identity
    chain.update(guide_reference_id=identity, generating_guide_reference_id=identity,
                 generating_guide_direction_xy=deepcopy(ref['actual_search_direction_xy']))


def augmented_graph(*, measured=False):
    args, kw, obs = graph_fixture()
    geometry, states, visible = args[2:5]
    points = {m['native_corner']: dict(record=m) for m in visible}
    for ri, ray in enumerate(obs['edge_records']):
        k, mate = ray['native_corner'], ray['native_neighbor']
        q = np.asarray(points[k]['record']['xy'])
        d = np.asarray(points[mate]['record']['xy']) - q
        direction = (d / np.linalg.norm(d)).tolist()
        if measured:
            ray['guide_proposal'] = dict(guide_id='MEASURED_BI_ENDPOINTS',
                guide_kind='MEASURED_BI_ENDPOINT_DIRECTION', direction_xy=direction,
                search_anchor_kind='OBSERVED_BI_REFERENCE_BAND', anchor_xy=q.tolist(),
                observed_BI_reference_xy=q.tolist(), vanishing_point_h=None, physical_axis=0)
            ray['actual_query_strip_audit'] = dict(synthetic=True)
        for ci, chain in enumerate(ray['chains']):
            chain['angle_change_deg'] = 0.
            ref = V.P._chain_reference(ray, chain, ri, ci, geometry, states, points)
            chain['guide_reference'] = ref['guide_reference']
            seal(chain)
    return args, kw, obs


def fixture(*, measured=False, mate_state='I'):
    args, kw, obs = augmented_graph(measured=measured)
    states = ['B', mate_state] + ['S'] * 6
    hidden = [False, False] + [True] * 6
    geometry = args[2]
    geometry['initial_self_hidden'] = hidden
    q = [10., 19.]; other = [40., 19.]
    geometry['initial_projection_xy'] = [[110., 35.], [150., 35.]] + [Poison()] * 6
    points = {0: dict(record=dict(xy=q, source='RGB_CORRECTED')),
              1: dict(record=dict(xy=other, source='VISIBLE_BACKBONE_FALLBACK'))}
    # Construct this fake record with the actual frozen producer reference builder.
    chain = deepcopy(obs['edge_records'][0]['chains'][0])
    ray = dict(native_corner=0, native_neighbor=1, along_interval_px=[4., 26.])
    if measured:
        ray.update(guide_proposal=dict(guide_id='MEASURED_BI_ENDPOINTS',
            guide_kind='MEASURED_BI_ENDPOINT_DIRECTION', direction_xy=[1., 0.],
            search_anchor_kind='OBSERVED_BI_REFERENCE_BAND', anchor_xy=q,
            observed_BI_reference_xy=q, vanishing_point_h=None, physical_axis=0),
            actual_query_strip_audit=dict(synthetic=True))
    eligible = points if mate_state in ('B', 'I') else {0: points[0]}
    ref = V.P._chain_reference(ray, chain, 7, 3, geometry, states, eligible)
    chain['guide_reference'] = ref['guide_reference']; seal(chain)
    node = dict(name='0-1/7:3', native_edge=[0, 1], native_corner=0, native_neighbor=1,
        normal=[0., 1.], offset=0., provenance=['7:3'], line_record=chain)
    return node, points, states, hidden, dict(normal_search_radius_px=43.), geometry


def assess(values):
    node, points, states, hidden, cfg, geometry = values
    return V.assess_initial_only_observed_ray_association(
        node, points, states, hidden, cfg, geometry=geometry)


@pytest.mark.parametrize('measured', [False, True])
def test_real_producer_reference_builder_two_allowed_query_kinds(measured):
    values = fixture(measured=measured)
    result = assess(values)
    assert result['passes'] and result['exact_original_query_source_and_direction_bound']
    assert result['normal_association']['endpoints'][0]['absolute_distance_px'] == 19.
    assert not result['extra_cached_poseguide_consumed']


def test_state_gates_before_poisoned_q_line_reference_or_unknown_projection():
    values = fixture(); node, points, states, hidden, _, geometry = values
    node['line_record'] = node['normal'] = Poison(); points[0] = Poison()
    geometry['initial_projection_xy'] = Poison(); hidden[1] = True
    assert assess(values)['reason'] == 'initial_S_gate_before_query_or_numeric_payload'
    hidden[1] = False; states[0] = 'U'
    assert assess(values)['reason'] == 'non_BI_anchor_before_query_or_numeric_payload'


@pytest.mark.parametrize('kind', ['PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR', 'UNKNOWN'])
def test_forbidden_cached_guide_rejected_before_numeric_fields_or_ref_hash(kind):
    values = fixture(); node, points, _, _, _, geometry = values
    node['line_record'] = dict(guide_reference=dict(guide_kind=kind,
        guide_id='ORIGINAL_WD/CACHED_0', actual_search_anchor_xy=Poison(),
        actual_search_direction_xy=Poison(), guide_reference_id=Poison()))
    node['normal'] = Poison(); points[0] = Poison(); geometry['initial_projection_xy'] = Poison()
    result = assess(values)
    assert result['reason'] == 'query_guide_kind_not_initial_only'
    assert result['q_TLS_numeric_fields_read'] is False


@pytest.mark.parametrize('measured', [False, True])
def test_allowed_kind_cannot_disguise_cached_source_id(measured):
    values = fixture(measured=measured)
    ref = values[0]['line_record']['guide_reference']; ref['guide_id'] = 'ORIGINAL_WD/CACHED_0'
    ref['actual_search_anchor_xy'] = Poison(); values[0]['normal'] = Poison(); values[1][0] = Poison()
    assert 'source_id' in assess(values)['reason']


def test_model_anchor_tag_rejected_even_when_anchor_numerically_equals_q():
    values = fixture(); ref = values[0]['line_record']['guide_reference']
    ref['search_anchor_kind'] = 'MODEL_PROJECTION_SEARCH_ONLY_BAND'
    assert assess(values)['reason'] == 'model_projection_search_band_not_an_original_q_query'


def test_digest_alias_native_ids_and_original_geometry_are_exact_bindings():
    values = fixture(); chain = values[0]['line_record']
    chain['guide_reference']['actual_search_anchor_xy'][0] += .1
    assert assess(values)['reason'] == 'original_query_dictionary_digest_mismatch'
    values = fixture(); values[0]['line_record']['generating_guide_reference_id'] = 'f' * 64
    assert assess(values)['reason'] == 'missing_or_disagreeing_exact_query_source_ids'
    values = fixture(); chain = values[0]['line_record']; chain['guide_reference']['native_neighbor'] = 2; seal(chain)
    assert assess(values)['reason'] == 'query_directed_native_ids_disagree'
    values = fixture(); chain = values[0]['line_record']; chain['guide_reference']['original_geometry_digest'] = 'a' * 64; seal(chain)
    assert assess(values)['reason'] == 'query_original_geometry_digest_mismatch'


def test_anchor_metadata_tolerance_retains19px_correction_and_parent_normal43():
    values = fixture(); chain = values[0]['line_record']; ref = chain['guide_reference']
    ref['actual_search_anchor_xy'][0] += 5e-7; seal(chain)
    assert assess(values)['passes']
    ref['actual_search_anchor_xy'][0] += 2e-6; seal(chain)
    assert assess(values)['reason'] == 'actual_query_anchor_does_not_equal_original_source_BI_q'
    values = fixture(); values[1][1]['record']['xy'][1] = 44.
    assert assess(values)['reason'] == 'parent_V30_original_BI_normal_gate_failed'


def test_parent_signed_ray_and_exact_producer_direction_cannot_be_pose_rescued():
    values = fixture(); chain = values[0]['line_record']
    chain['guide_reference']['actual_search_direction_xy'] = [-1., 0.]; seal(chain)
    assert assess(values)['reason'] == 'actual_query_direction_opposes_fixed_native_ray'
    values = fixture(); chain = values[0]['line_record']
    chain['guide_reference']['actual_search_direction_xy'] = [.8, .6]; seal(chain)
    assert assess(values)['reason'] == 'actual_query_direction_or_seed_differs_from_exact_producer_construction'
    values = fixture(); del values[1][1]
    assert assess(values)['reason'] == 'both_BI_mate_missing_immutable_measurement'


@pytest.mark.parametrize('mate_state', ['O', 'T', 'U'])
def test_unknown_mate_uses_only_original_model_ray_and_never_unknown_q(mate_state):
    values = fixture(mate_state=mate_state); values[1][1] = Poison()
    assert assess(values)['passes']
    assert assess(values)['reference_source'] == 'ORIGINAL_INITIAL_MODEL_NATIVE_RAY_FOR_UNKNOWN_MATE'
    values[5]['initial_projection_xy'][1] = [70., 35.]
    assert assess(values)['reason'] == 'actual_query_direction_opposes_fixed_native_ray'


def test_tls_sign_invariance_and_all_original_payloads_unchanged():
    values = fixture(); before = deepcopy(values[:5]); result = assess(values)
    values[0]['normal'] = [0., -1.]; values[0]['offset'] = -values[0]['offset']
    flipped = assess(values)
    assert result['passes'] and flipped['passes'] and result['signed_direction_cosine'] == flipped['signed_direction_cosine']
    assert values[1:5] == before[1:5]


@pytest.mark.parametrize('guides', [[], [dict(guide_id='e', geometry=Poison())], Poison()])
def test_rank_forbids_every_external_guide_before_any_geometry_or_rows(guides):
    with pytest.raises(ValueError, match='external pose_guides'):
        V.rank_initial_only_observed_ray_graph(Poison(), Poison(), Poison(), Poison(),
            image_binding=Poison(), pose_guides=guides)


@pytest.mark.parametrize('cache', [1, 4096])
def test_all_admitted_fullgraph_bitexact_to_parent_same_math_no_global_or_input_mutation(cache):
    args, kw, obs = augmented_graph(); before = deepcopy((args, kw, obs))
    paths = [Path(m.__file__) for m in (V.PARENT, V.F, V.G, V.B, V.M)]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    globals_before = {k: id(v) for k, v in V.G.rank_joint_geometry_line_graph.__globals__.items()}
    cfg = dict(graph_configuration=dict(maximum_beam_states=5, reported_joint_alternatives=5))
    expected = V.PARENT.rank_observed_anchor_native_ray_graph(obs, args[2], args[3], args[4],
        **kw, configuration=cfg, pallet_mask=args[1], maximum_cache_entries=cache, intersection_cache_limit=cache)
    result = V.rank_initial_only_observed_ray_graph(obs, args[2], args[3], args[4],
        **kw, configuration=cfg, pallet_mask=args[1], maximum_cache_entries=cache, intersection_cache_limit=cache)
    assert result['rejected_node_count'] == 0 and result['candidate_nodes_checked'] == 12
    assert json.dumps(expected['graph'], sort_keys=True) == json.dumps(result['graph'], sort_keys=True)
    assert V.G.NMS.G.C.finite((args, kw, obs)) == V.G.NMS.G.C.finite(before)
    assert globals_before == {k: id(v) for k, v in V.G.rank_joint_geometry_line_graph.__globals__.items()}
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert result['external_pose_guides_consumed'] == 0


def test_rank_rejects_cached_numeric_poison_before_node_and_preserves_good_raw_index():
    args, kw, obs = augmented_graph(); good = deepcopy(obs['edge_records'][0]['chains'][0])
    bad = dict(guide_reference=dict(guide_kind='PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR',
        guide_id='ORIGINAL_WD/CACHED_0', actual_search_anchor_xy=Poison()),
        normal=Poison(), observed_points=Poison(), offset=Poison())
    obs['edge_records'][0]['chains'] = [bad, good]
    result = V.rank_initial_only_observed_ray_graph(obs, args[2], args[3], args[4], **kw, pallet_mask=args[1])
    assert result['source_rejected_before_numeric_node_count'] == 1
    assert result['node_association_ledger'][0]['node_provenance'] == ['0:0']
    assert result['node_association_ledger'][0]['reason'] == 'query_guide_kind_not_initial_only'
    assert any(n['provenance'] == ['0:1'] for n in result['graph']['line_candidates'])
    assert obs['edge_records'][0]['chains'][0] is bad
    assert obs['edge_records'][0]['chains'][1] == good
