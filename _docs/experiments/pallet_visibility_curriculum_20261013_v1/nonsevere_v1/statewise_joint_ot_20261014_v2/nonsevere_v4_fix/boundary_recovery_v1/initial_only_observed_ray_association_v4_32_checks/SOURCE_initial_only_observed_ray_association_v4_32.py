"""Opt-in initial-only admission of existing observed B/I ray measurements.

Only original17c and measured-BI-endpoint actual queries are admitted. All
physical-VP queries, including ORIGINAL_INITIAL, are excluded. External pose
guides are forbidden, including in the unchanged graph scoring/ordered terms.
No cached anchor/pixel/TLS/ref is rewritten and no measurement is created.
The independent initial geometry/q ancestry must be bound by the caller.
This separately selected population needs a new CAL model and evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType

import numpy as np

from . import observed_anchor_native_ray_association_v4_31 as PARENT

P, G, B, M, F = PARENT.P, PARENT.G, PARENT.B, PARENT.M, PARENT.F
PARENT_ADMISSION_SHA256 = 'ccdb311a4468825919a6731620e6836a0b52bffc775cc594d90c6bea8d2babad'
INITIAL_SOURCE_CLOSURE_SHA256 = 'c10f6dd26a35b2da960efe3a6533a43af5c8db636f9161b1c3e1652b55ef75ad'
SELECTION_PROFILE = 'INITIAL_ONLY_ORIGINAL17C_OR_MEASURED_BI_QUERY_FIXED_NATIVE_RAY_V32'
OBSERVATION_GENERATION = 'CACHED_FULL_PROVENANCE_V29_INITIAL_ONLY_OBSERVED_NATIVE_RAY_SELECTION_V32'
ALLOWED_GUIDE_KINDS = ('FROZEN17C_ORIGINAL_MODEL_SEED', 'MEASURED_BI_ENDPOINT_DIRECTION')
GEOMETRY_DIGEST_FIELDS = ('native_xyz', 'R_cf', 'centroid', 'physical_registration_Q',
                          'K', 'image_hw', 'initial_self_hidden')
PRODUCER_NUMERIC_TOLERANCE = PARENT.ORTHOGONAL_NUMERIC_TOLERANCE
RECIPE = dict(selection_profile=SELECTION_PROFILE, observation_generation=OBSERVATION_GENERATION,
    parent_raw_observation_generation=P.OBSERVATION_GENERATION,
    parent_selection_profile=PARENT.SELECTION_PROFILE,
    allowed_actual_query_guide_kinds=list(ALLOWED_GUIDE_KINDS),
    allowed_source_ids=['ORIGINAL_INITIAL', 'MEASURED_BI_ENDPOINTS'],
    physical_VP_queries_allowed=False, model_projection_query_allowed=False,
    external_pose_guides_allowed=False, graph_pose_guides=None,
    graph_context='unchanged G28 builder with internal ORIGINAL_INITIAL only; measured BI endpoint order remains',
    guide_provenance='exact producer kind/id/geometry_ref_id, query-dict digest and raw chain aliases before TLS/q payload',
    direction_provenance='original model seed normal; actual base query uses BI endpoint delta or original unknown model ray; measured endpoint query uses BI delta',
    normal_anchor_signed_gate='unchanged F31 normal43, original q-anchor metadata1e-6 and fixed signed native ray existing10px',
    source_closure_precondition=dict(sha256=INITIAL_SOURCE_CLOSURE_SHA256,
        caller_must_bind_independent_initial_geometry_and_archived_q=True),
    before_exact_fit_dedup=True, raw_refs_indices_coefficients_q_covariance_immutable=True,
    raw_extra_cached_queries_preserved_in_archive_but_not_consumed=True,
    new_actual_measurements=0, new_RGB_NMS_TLS_NN_posefit_labels=0,
    old_V29_V30_V31_noise_transferred_or_validated=False,
    separate_CAL46_inventory_fixed44_new_noise_and_uniform_evaluation_required=True,
    finite_native_edge_identity_or_pose_uniqueness_certified=False)


def _source_gate(node, states, hidden, *, chain=None):
    """Inspect state and exact query source before any q/TLS/support numbers."""
    k, mate, edge = PARENT._ids(node)
    base = dict(native_edge=edge, source_native_corner=k, source_native_neighbor=mate,
        original_indices_preserved=True, q_TLS_numeric_fields_read=False)

    def reject(reason):
        return dict(base, passes=False, reason=reason)

    if any(hidden[i] or states[i] == 'S' for i in edge):
        return reject('initial_S_gate_before_query_or_numeric_payload')
    if states[k] not in ('B', 'I'):
        return reject('non_BI_anchor_before_query_or_numeric_payload')
    if states[mate] not in ('B', 'I', 'O', 'T', 'U'):
        return reject('invalid_authoritative_mate_state')
    if chain is None:
        chain = node.get('line_record')
    if not isinstance(chain, dict):
        return reject('missing_original_query_record')
    reference = chain.get('guide_reference')
    if not isinstance(reference, dict):
        return reject('missing_original_query_reference')
    kind = reference.get('guide_kind')
    # Forbidden guide numerical payload is never hashed, copied or converted.
    if kind not in ALLOWED_GUIDE_KINDS:
        return dict(reject('query_guide_kind_not_initial_only'), guide_kind=kind)
    source_id = reference.get('guide_id')
    if reference.get('search_anchor_kind') == 'MODEL_PROJECTION_SEARCH_ONLY_BAND':
        return reject('model_projection_search_band_not_an_original_q_query')
    if kind == ALLOWED_GUIDE_KINDS[0]:
        if (source_id != 'ORIGINAL_INITIAL'
                or reference.get('search_anchor_kind') is not None
                or reference.get('original_query_strip_metadata_is_frozen17c') is not True
                or reference.get('strip_direction_source') !=
                   'BI endpoint vector if both BI, otherwise original model direction'):
            return reject('original17c_producer_source_id_or_query_kind_mismatch')
    else:
        if (source_id != 'MEASURED_BI_ENDPOINTS'
                or reference.get('search_anchor_kind') != 'OBSERVED_BI_REFERENCE_BAND'
                or reference.get('actual_search_direction_source') != kind):
            return reject('measured_BI_endpoint_producer_source_id_or_query_kind_mismatch')
        if states[mate] not in ('B', 'I'):
            return reject('measured_BI_endpoint_query_requires_BI_mate')
    if reference.get('geometry_ref_id') != 'ORIGINAL_INITIAL':
        return reject('query_geometry_source_is_not_original_initial')
    if (reference.get('native_corner') != k or reference.get('native_neighbor') != mate
            or reference.get('physical_native_ids') != [k, mate]):
        return reject('query_directed_native_ids_disagree')
    identity = reference.get('guide_reference_id')
    if (not isinstance(identity, str) or len(identity) != 64
            or chain.get('guide_reference_id') != identity
            or chain.get('generating_guide_reference_id') != identity):
        return reject('missing_or_disagreeing_exact_query_source_ids')
    try:
        actual = G._digest({key: value for key, value in reference.items()
                            if key != 'guide_reference_id'})
    except (TypeError, ValueError):
        return reject('invalid_original_query_dictionary')
    if identity != actual:
        return reject('original_query_dictionary_digest_mismatch')
    return dict(base, passes=True, reason='initial_only_query_source_pass',
        guide_kind=kind, guide_id=source_id, guide_reference_id=identity)


def assess_initial_only_observed_ray_association(node, points, states, initial_self_hidden,
                                                cfg, *, geometry):
    """Strict source binding followed by the unchanged F31 numerical gates."""
    source = _source_gate(node, states, initial_self_hidden)
    if not source['passes']:
        return source
    checked = PARENT.assess_observed_anchor_native_ray_association(
        node, points, states, initial_self_hidden, cfg, geometry=geometry)
    if not checked['passes']:
        return dict(checked, initial_only_query_source=source)
    k, mate, _ = PARENT._ids(node)
    line = node['line_record']; reference = line['guide_reference']

    def reject(reason):
        return dict(checked, passes=False, reason=reason, initial_only_query_source=source)

    if reference['original_geometry_digest'] != G._digest(
            {key: geometry[key] for key in GEOMETRY_DIGEST_FIELDS}):
        return reject('query_original_geometry_digest_mismatch')
    # The caller must bind this ORIGINAL geometry, not a trial/selected pose.
    start = np.asarray(geometry['initial_projection_xy'][k], float)
    other = np.asarray(geometry['initial_projection_xy'][mate], float)
    archived_projection = np.asarray(reference['original_reference_projection_xy'], float)
    if (start.shape != (2,) or other.shape != (2,)
            or archived_projection.shape != (2, 2)
            or not np.isfinite([*start, *other, *archived_projection.ravel()]).all()
            or not np.allclose(archived_projection, [start, other], atol=1e-6, rtol=0)):
        return reject('query_original_model_projection_binding_mismatch')
    model_delta = other - start
    model_length = float(np.linalg.norm(model_delta))
    if (source['guide_kind'] == ALLOWED_GUIDE_KINDS[0]
            and model_length < PARENT.MINIMUM_REFERENCE_LENGTH_PX):
        return reject('original_producer_model_seed_shorter_than_existing10px')
    actual = np.asarray(reference['actual_search_direction_xy'], float)
    seed = np.asarray(reference['seed_direction_xy'], float)
    seed_normal = np.asarray(reference['seed_normal_xy'], float)
    if source['guide_kind'] == ALLOWED_GUIDE_KINDS[0]:
        expected_seed = model_delta / model_length
        expected_source = ('gated immutable BI endpoint hemisphere'
                           if states[mate] in ('B', 'I') else 'original model direction')
        if reference.get('actual_search_direction_source') != expected_source:
            return reject('original17c_actual_direction_source_mismatch')
    else:
        # F31 above has already required both immutable BI records/length>=10.
        expected_seed = np.asarray(checked['fixed_reference_direction_unit_xy'])
        q = np.asarray(points[k]['record']['xy'], float)
        archived_q = np.asarray(reference['original_observed_BI_reference_xy'], float)
        if archived_q.shape != (2,) or not np.allclose(archived_q, q, atol=1e-6, rtol=0):
            return reject('measured_BI_endpoint_reference_q_mismatch')
    expected_actual = np.asarray(checked['fixed_reference_direction_unit_xy'])
    if (seed.shape != (2,) or seed_normal.shape != (2,)
            or not np.isfinite([*seed, *seed_normal]).all()
            or not np.allclose(seed, expected_seed, atol=PRODUCER_NUMERIC_TOLERANCE, rtol=0)
            or not np.allclose(seed_normal, [-seed[1], seed[0]],
                               atol=PRODUCER_NUMERIC_TOLERANCE, rtol=0)
            or not np.allclose(actual, expected_actual, atol=PRODUCER_NUMERIC_TOLERANCE, rtol=0)):
        return reject('actual_query_direction_or_seed_differs_from_exact_producer_construction')
    return dict(checked, initial_only_query_source=source,
        exact_original_query_source_and_direction_bound=True,
        extra_cached_poseguide_consumed=False)


def rank_initial_only_observed_ray_graph(chain_observation, geometry, states, visible_measurements, *,
        image_binding, configuration=None, pose_guides=None, pallet_mask=None, visibility_context=None,
        maximum_cache_entries=4096, intersection_cache_limit=4096):
    """Same graph arithmetic; raw source rejection before node numbers/dedup."""
    if pose_guides is not None:
        raise ValueError('V32 forbids external pose_guides; use internal ORIGINAL_INITIAL only')
    if hashlib.sha256(Path(PARENT.__file__).read_bytes()).hexdigest() != PARENT_ADMISSION_SHA256:
        raise ValueError('Exact frozen F31 source required')
    if hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest() != PARENT.PARENT_ASSOCIATION_SHA256:
        raise ValueError('Exact frozen R30 private rank hook required')
    ledger = []

    def node_factory(record, chain, reference_index, cfg):
        k, mate = record['native_corner'], record['native_neighbor']
        stub = dict(name=f'{min(k, mate)}-{max(k, mate)}/{reference_index}',
            native_edge=sorted([k, mate]), native_corner=k, native_neighbor=mate,
            provenance=[reference_index])
        verdict = _source_gate(stub, states, geometry['initial_self_hidden'], chain=chain)
        if not verdict['passes']:
            stub['_initial_only_rejection'] = verdict
            return stub
        return P._fresh_node(record, chain, reference_index, cfg)

    def check(node, points, current_states, hidden, cfg):
        verdict = node.get('_initial_only_rejection')
        if verdict is None:
            verdict = assess_initial_only_observed_ray_association(
                node, points, current_states, hidden, cfg, geometry=geometry)
        ledger.append(dict(line_name=node['name'], node_provenance=list(node['provenance']), **verdict))
        return verdict['passes']

    namespace = dict(G.rank_joint_geometry_line_graph.__globals__)
    namespace.update(_association_pass=check, _node=node_factory,
        _score_graph=M.make_ordered_scorer(maximum_cache_entries=maximum_cache_entries),
        _bounded_beam_step=B.bounded_beam_step,
        _new_intersection_cache=lambda: B.BoundedIntersectionCache(intersection_cache_limit))
    rank = FunctionType(F._filtered_rank_code(), namespace)
    rank.__kwdefaults__ = deepcopy(G.rank_joint_geometry_line_graph.__kwdefaults__)
    graph = rank(chain_observation, geometry, states, visible_measurements,
        image_binding=image_binding, configuration=configuration, pose_guides=None,
        pallet_mask=pallet_mask, visibility_context=visibility_context)
    graph['joint_alternatives'] = [a for a in graph['joint_alternatives']
                                   if not a['ordered_support_unavailable']]
    return dict(selection_profile=SELECTION_PROFILE, observation_generation=OBSERVATION_GENERATION,
        parent_raw_observation_generation=P.OBSERVATION_GENERATION,
        parent_selection_profile=PARENT.SELECTION_PROFILE, graph=graph,
        node_association_ledger=ledger, rejected_node_count=sum(not r['passes'] for r in ledger),
        candidate_nodes_checked=len(ledger), external_pose_guides_consumed=0,
        graph_geometry_source='ORIGINAL_INITIAL_ONLY',
        source_rejected_before_numeric_node_count=sum(not r.get('initial_only_query_source', r)['passes']
            for r in ledger),
        raw_archive_unchanged=True, original_chain_indices_and_node_provenance_preserved=True,
        old_V29_V30_V31_noise_validated_for_this_population=False,
        new_CAL46_noise_and_uniform_validation_required=True,
        new_RGB_NMS_TLS_posefit_labels_or_precise_KPs=0, operational_opt_in_only=True)
