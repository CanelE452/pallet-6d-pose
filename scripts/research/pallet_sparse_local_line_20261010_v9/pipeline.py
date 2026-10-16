"""Pure packet adapter for the one missing sparse local IMAGE_ROLE C2 arm.

``solve_local_packet(packet)`` accepts exactly the V8 normalized parent packet
plus ``role_observation``: the same-frame full V7 sealed IMAGE_ROLE observation
row. Original N3 coordinates supply display/fallback only. Its stored initial
pose and the parent's available point-only pose supply local starts. No image,
model, truth, decoding, four-point solve, initial pose computation or file read
occurs here. Whole-file SHA, full-population and cleanup provenance remain the
driver's responsibility, before it makes one of these normalized packets.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

from ..pallet_same_observation_controls_20261010_v8 import controls as V8
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from .solver import LocalPointLineBank, POLICY


METHOD = 'ROLE_BOUNDARY_LOCAL_POINT_LINE'
COMPARATOR = V8.PRIMARY


def validate_local_packet(packet):
    V8.require(isinstance(packet, Mapping) and set(packet) == {
        'parent_row', 'native_N3_points', 'parent_completion', 'role_observation'},
        'use the explicit GT-free sparse local packet')
    normalized = {k: packet[k] for k in ('parent_row', 'native_N3_points', 'parent_completion')}
    p = V8.validate_parent(normalized)
    obs = packet['role_observation']
    V8.require(isinstance(obs, Mapping), 'full same-frame sealed ROLE observation required')
    V8.require(not set(obs) & {'corner', 'pose', 'GT', 'ground_truth', 'human_states_native',
                              'reference_pose', 'baseline_corner', 'baseline_pose', 'mask_audit'},
               'scored/truth fields cannot enter local observations')
    V8.require(obs.get('id') == p['row']['id'] and obs.get('session') == p['row']['session'] and
               obs.get('head_arm') == 'IMAGE_ROLE', 'ROLE observation frame/session/head differs')
    V8.require(obs.get('GT_input') is False, 'ROLE observation is not sealed GT-free input')
    V8.require(V8.semantic_sha256(obs.get('initial_N3_pose')) == V8.semantic_sha256(p['row']['initial_pose']) and
               obs.get('predicted_N3_hidden') == p['H'], 'stored initial N3 pose/H identity differs')
    V8.require(np.array_equal(np.asarray(obs.get('native_N3_points'), float), p['native'], equal_nan=True),
               'original native N3 observation identity differs')
    # V7 run.py serializes decoder fields directly into each observation row.
    observation = obs
    V8.require(all(isinstance(observation.get(k), list) for k in ('corners', 'lines', 'queries')),
               'full original decoded ROLE observation packet absent')
    expected_queries = 0 if p['row']['selected_index'] is None and obs.get('no_detection') is True else 84
    V8.require(len(observation['queries']) == expected_queries and
               [q.get('query') for q in observation['queries']] == list(range(expected_queries)),
               'original query identities/order differ')
    V8.require(observation.get('raw_logits_sha256') == p['row'].get('observation_raw_logits_sha256'),
               'parent and original ROLE logits semantic identity differs')
    for line in observation['lines']:
        edge = line.get('edge')
        V8.require(isinstance(edge, int) and not isinstance(edge, bool) and 0 <= edge < 12,
                   'invalid original semantic line ID')
        queries = line.get('queries')
        V8.require(isinstance(queries, list) and all(isinstance(q, int) and not isinstance(q, bool) and
                   0 <= q < 84 and q // 7 == edge for q in queries), 'original line query identities differ')
        V8.require(all(observation['queries'][q].get('calibrated_model_coverage') is True and
                       observation['queries'][q].get('selected_xy') is not None for q in queries),
                   'line lacks selected source-CAL-supported query evidence')
        V8.require(np.array_equal(np.asarray(line.get('support_points'), float),
                                  np.asarray([observation['queries'][q]['selected_xy'] for q in queries], float)),
                   'line support points are not the sealed query observations')
    lookup = {int(c['id']): c for c in observation['corners']}
    V8.require(len(lookup) == len(observation['corners']), 'duplicate original corner ID')
    for k in p['contract']['validated_boundary_corner_ids']:
        V8.require(k in lookup and np.array_equal(np.asarray(lookup[k]['xy'], float), p['q'][k]),
                   'parent sparse corner differs from its original ROLE observation')
        admitted = [c for c in p['contract']['corner_admission'] if c['id'] == k]
        V8.require(len(admitted) == 1 and admitted[0]['accepted'] is True and
                   list(admitted[0]['edges']) == list(lookup[k]['edges']),
                   'sparse corner source-edge admission identity differs')
    p.update(observation=copy.deepcopy(observation), observation_row=obs)
    return p


def solve_local_packet(packet):
    """Return one full normalized geometry row and its actual-call ledger."""
    p = validate_local_packet(packet)
    before = V8.semantic_sha256(packet)
    parent = p['row']
    bank = LocalPointLineBank(p['q'], p['K'], p['xyz'], image_size=p['size'])
    solved = bank.solve(p['observation'], adopted_boundary_corner_ids=p['contract']['validated_boundary_corner_ids'],
        hidden=p['H'], initial_pose=copy.deepcopy(parent['initial_pose']),
        point_result=copy.deepcopy(parent['solver']))
    V8.require(solved['prior_used'] is False and solved['initial_dimension_prior_used'] is False and
               solved['known_dimension_constraint_used'] is False,
               'an initial residual/dimension prior entered local fit')
    result = assemble(p['native'], p['q'], copy.deepcopy(parent['initial_pose']), p['H'],
                      solved, 'VALIDATED_ROLE_ONLY', p['contract'])
    row = {key: copy.deepcopy(parent[key]) for key in (
        'id', 'session', 'K', 'xyz', 'raw_hw', 'selected_index', 'fixed_metadata',
        'head_arm', 'head_mode', 'head_checkpoint_sha256', 'calibration_binding')}
    row.update(result)
    for key in ('mask_diagnostic', 'observation_raw_logits_sha256', 'selected_queries', 'source_lines',
                'model_query_anchor', 'initial_pose_source', 'Base_proposal_feature_dependence_retained'):
        if key in parent:
            row[key] = copy.deepcopy(parent[key])
    row.update(method=METHOD, comparator=COMPARATOR, observation_supply='BOUNDARY_ONLY',
        selected_corner_ids=list(p['contract']['validated_boundary_corner_ids']),
        predicted_initial_N3_hidden=list(p['H']), original_self_hidden_initial=list(p['H']),
        initial_H_applied=True, diagnostic_no_mask=False,
        local_point_line_policy=copy.deepcopy(POLICY), local_point_line_refinement=True,
        pose_estimation_kind='LOCAL_POINT_LINE_REFINEMENT', independent_PnP_or_PnL=False,
        partial_lines_not_pose_inputs=False, final_numeric_pose_has_initial_prior=False,
        initial_pose_used_only_for_H_and_fallback=False,
        initial_pose_may_supply_local_optimizer_start=True,
        conditional_H_from_initial_N3=True, missing_sparse_filled_for_numeric_fit=False,
        native_N3_is_numeric_final_pose_observation=False, display_missing_uses_native_N3=True,
        selection_validation_fit_excludes_candidate_corner=False,
        selection_validation_is_fully_statistically_independent=False,
        selection_validation_has_shared_initial_prior=False,
        parent_method=V8.PARENT_METHOD, parent_geometry_row_semantic_sha256=V8.semantic_sha256(parent),
        parent_observation_row_semantic_sha256=V8.semantic_sha256(p['observation_row']),
        parent_native_N3_semantic_sha256=V8.semantic_sha256(p['native']),
        original_native_N3_points=p['native'].tolist(),
        native_reference_id=dict(id=parent['id'], head_arm='IMAGE_ROLE'),
        parent_completion=copy.deepcopy(p['completion']), same_sparse_observation=True,
        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,
        head_forward_calls=0, calibration_or_decode_calls=0, oracle=False,
        stored_jacobian_fit_ids=list(solved.get('fit_input_ids', [])) if solved.get('available') else [],
        stored_jacobian_fit_line_edges=list(solved.get('fit_line_edges', [])) if solved.get('available') else [],
        stored_jacobian_scope='whole actual point+unused line pool, not8px diagnostic inliers',
        full_candidate_and_operation_witnesses_preserved=True, solver_replay_latency_claim=False)
    V8.require(np.array_equal(np.asarray(row['input_points']), p['q'], equal_nan=True), 'sparse input was altered')
    V8.require(np.array_equal(np.asarray(row['native_points'])[8], p['native'][8], equal_nan=True), 'center changed')
    V8.require(set(p['H']).isdisjoint(solved.get('fit_input_ids', [])), 'self-hidden initial point entered fit')
    V8.require(row['reprojections_reused_as_observations'] is False, 'projection was re-fit')
    V8.require(before == V8.semantic_sha256(packet), 'immutable local parent packet was mutated')
    ledger = dict(schema='same_sparse_ROLE_local_point_line_ledger_v9', method=METHOD,
        parent_packet_semantic_sha256=before, logical_pose_paths=1,
        coordinate_bank_count=1, coordinate_banks=[dict(input_hash=bank.bank.digest, **dict(bank.bank.ledger))],
        local_operation_counts=copy.deepcopy(solved['operation_counts']),
        local_bank_counts=copy.deepcopy(bank.counts),
        factor_pool=copy.deepcopy(solved.get('factor_pool')), factor_pool_binding=solved.get('factor_pool_binding'),
        original_observation_binding=solved['observation_binding'],
        original_line_edges=[int(line['edge']) for line in p['observation']['lines']],
        used_line_edges=list(solved['line_edges']), consumed_edges=list(solved['consumed_edges']),
        stored_initial_pose_used_as_start=solved['initial_pose_start_used'],
        residual_prior_used=False, native_N3_coordinates_used_for_numeric_fit=False,
        detector_calls=0, N3_calls=0, head_calls=0, initial_pose_calls=0, PnP_calls=0,
        calibration_calls=0, query_decode_calls=0, new_training_updates=0, new_RGB=0,
        parent_unchanged=True, parent_full_file_provenance_is_caller_responsibility=True,
        latency_benchmark=False, local_only=True, global_uniqueness_proven=False,
        policy=copy.deepcopy(POLICY))
    return row, ledger
