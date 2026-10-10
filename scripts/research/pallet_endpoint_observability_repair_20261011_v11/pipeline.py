"""Filter independently contradicted unused lines, then fit one local proposal.

The two output views share one actual unchanged V9 local solve. The safe view
can decline its selected proposal using the existing V1 normalized-column
weak-condition predicate. It never selects another candidate or fits again.
The sealed packet API consumes completed V10 evidence; deployment uses the
same kernel with fresh native endpoint validation, without sealed receipts.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math

import numpy as np

from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_partial_line_heldout_20261010_v6 import solver as V6
from ..pallet_same_observation_controls_20261010_v8 import controls as V8
from ..pallet_sparse_local_line_20261010_v9 import pipeline as V9
from ..pallet_sparse_local_line_20261010_v9.solver import LocalPointLineBank
from ..pallet_sparse_local_line_20261010_v9 import solver as S9
from ..pallet_unused_line_endpoint_audit_20261011_v10 import audit as A10

METHODS = ('FILTERED_LOCAL', 'FILTERED_LOCAL_WEAK_SAFE')
PRIMARY = METHODS[1]
COMPARATOR = V9.METHOD
PARENT_PRIMARY = V8.PARENT_METHOD
POINT_COMPARATOR = V8.PRIMARY
DECISIONS = ('SUPPORTED_RETAIN', 'CONTRADICTED_REJECT', 'UNVERIFIED_RETAIN')
WEAK_RELATIVE_THRESHOLD = 1e-4
POLICY = dict(schema='same_sparse_ROLE_endpoint_observability_repair_policy_v11',
    methods=list(METHODS), primary=PRIMARY,
    input='unchanged sparse V7 IMAGE_ROLE points and actual unconsumed physical/CAL line pool',
    endpoint_policy=copy.deepcopy(A10.POLICY),
    filter='remove only actual unused CONTRADICTED_REJECT edges; retain UNKNOWN',
    actual_corners_and_consumed_support_lines_unchanged=True,
    local_policy=copy.deepcopy(S9.POLICY), actual_local_solves_per_frame=1,
    reported_views_per_actual_solver=2, no_alternative_reselection=True,
    weak_origin='unchanged V1 HypothesisBank._jacobian strict s_min < s_max * 1e-4',
    weak_relative_threshold=WEAK_RELATIVE_THRESHOLD,
    weak_normalization='same V1/V5 unit L2 Jacobian columns, scale floor 1e-300',
    weak_matrices=['observed_normal_joint', 'modeled_normal_joint'],
    weak_scope='selected candidate whole actual filtered pool; no new SVD or inlier gate',
    weak_safe='rank-only NEW proposal with either weak matrix -> WEAK_OBSERVABILITY then unchanged N3 fallback',
    rank_only_proposal_candidates_preserved_before_gate=True,
    numerical_NEW_is_not_pose_accuracy_success=True,
    unavailable_validation_is_UNKNOWN_not_frame_failure=True,
    native_N3_only_validator_and_display_fallback_not_final_fit=True,
    missing_sparse_filled_for_numeric_fit=False, initial_pose_is_local_start_only=True,
    H_initial_coordinates_excluded_from_fit=True,
    accepted_NEW_H_projection_once_without_refit=True,
    no_new_seed_threshold_training_RGB_or_GT_selection=True,
    known_DEV_evidence=True, unseen_evaluation=False, global_uniqueness_proven=False)


def require(value, message):
    V8.require(value, message)


def _copy(value):
    return copy.deepcopy(value)


def _delta(after, before):
    require(set(after) == set(before), 'native bank ledger fields differ')
    return {key: int(after[key] - before[key]) for key in after}


def _source_line(line):
    return dict(edge=line['edge'], endpoints=list(V5.EDGES[line['edge']]),
        normal=line['normal'].tolist(), offset=line['offset'],
        support_query_ids=list(line['support_query_ids']), raw=_copy(line['source_line']))


def _endpoint_records(native, lines, H, observer):
    """The unchanged V6/V10 endpoint test, on one shared fresh native bank."""
    records = []
    for line in lines:
        edge = line['edge']; endpoints = tuple(V5.EDGES[edge])
        blocked = tuple(sorted(set(H) | set(endpoints)))
        before = native.ledger.copy()
        if observer is not None:
            observer(dict(phase='endpoint_request_started', edge=edge,
                validation_excluded=list(blocked), bank_before=before))
        try:
            solved = native.solve(excluded=endpoints, hidden=H, robust=True)
        except BaseException as error:
            if observer is not None:
                observer(dict(phase='endpoint_request_exception', edge=edge,
                    bank_after=native.ledger.copy(), error=dict(type=type(error).__name__, message=str(error))))
            raise
        after = native.ledger.copy()
        problem = V6._witness_problem(solved, blocked)
        distances, rms = None, None
        if problem is None:
            projected = np.asarray(solved['projected'], float)
            distances = projected[list(endpoints)] @ line['normal'] - line['offset']
            rms = float(np.sqrt(np.mean(distances ** 2)))
            if not np.isfinite(rms): problem = 'VALIDATION_ENDPOINT_RMS_NONFINITE'
        if problem is not None:
            decision, reason = 'UNVERIFIED_RETAIN', problem
        elif rms <= V6.POLICY['endpoint_RMS_threshold_px']:
            decision, reason = 'SUPPORTED_RETAIN', 'AVAILABLE_NATIVE_N3_ENDPOINT_RMS_WITHIN_FIXED_8PX'
        else:
            decision, reason = 'CONTRADICTED_REJECT', 'AVAILABLE_NATIVE_N3_ENDPOINT_RMS_EXCEEDS_FIXED_8PX'
        record = dict(edge=edge, endpoints=list(endpoints), hidden=list(H),
            validator_temporary_excluded=list(endpoints), validation_excluded=list(blocked),
            actual_self_hidden_separate_from_temporary_exclusion=True,
            native_input_hash=native.digest, attempted_pose_calls=1,
            decision=decision, reason=reason, endpoint_RMS_px=rms,
            signed_endpoint_distances_px=None if distances is None else distances.tolist(),
            source_line=_source_line(line), solved=_copy(solved),
            bank_before=before, bank_after=after, bank_delta=_delta(after, before),
            observed_line_supplied_to_validation_solver=False,
            global_cached_generation_may_include_blocked_IDs=True,
            active_scoring_generators_refits_exclude_blocked_IDs=True,
            physical_ownership_certified=False, validation_pose_may_be_wrong_consensus=True)
        records.append(record)
        if observer is not None:
            observer(dict(phase='endpoint_request_returned', edge=edge,
                bank_after=after, record=_copy(record)))
    return records


def _validate_saved_records(audit, native, lines, H, q, adopted, K, xyz, image_size,
                            consumed, checks, contract_error):
    """Only replay completed, same-source V10 decisions; no new point solve."""
    require(isinstance(audit, Mapping) and not A10.TRUTH_FIELDS.intersection(audit),
        'GT-free V10 endpoint evidence required')
    require(audit.get('method') == A10.METHOD and audit.get('policy') == A10.POLICY and
        audit.get('GT_free') is True and audit.get('new_final_pose_computed') is False and
        audit.get('original_V9_pose_and_status_preserved') is True,
        'frozen V10 audit policy/status differs')
    for key, value in (('original_sparse_points', q), ('original_native_N3_points', native.points),
                       ('K', K), ('xyz', xyz)):
        require(np.array_equal(np.asarray(audit.get(key), float), value, equal_nan=True),
            'V10 numeric source identity differs: ' + key)
    require(audit['raw_hw'] == [image_size[1], image_size[0]] and audit['initial_H'] == list(H) and
        audit['actual_adopted_boundary_ids'] == list(adopted) and
        audit['actual_sparse_used_ids'] == [k for k in range(8) if k in set(audit['actual_sparse_used_ids'])] and
        audit['consumed_edges'] == consumed and audit['original_line_contract_checks'] == checks and
        audit['contract_error'] == contract_error, 'V10 H/adopted/factor contract differs')
    records = audit['endpoint_records']
    require(isinstance(records, list) and [r['edge'] for r in records] == [l['edge'] for l in lines] ==
        audit['original_actual_unused_edges'], 'V10 actual unused unique edge pool differs')
    for line, record in zip(lines, records):
        endpoints = list(V5.EDGES[line['edge']]); blocked = sorted(set(H) | set(endpoints))
        require(record['hidden'] == list(H) and record['validation_excluded'] == blocked and
            record['validator_temporary_excluded'] == endpoints and record['endpoints'] == endpoints and
            record['native_input_hash'] == native.digest and record['decision'] in DECISIONS and
            record['observed_line_supplied_to_validation_solver'] is False and
            record['physical_ownership_certified'] is False and
            V8.semantic_sha256(record['source_line']) == V8.semantic_sha256(_source_line(line)),
            'V10 endpoint source/mask/line identity differs')
        problem = V6._witness_problem(record['solved'], blocked)
        rms = record['endpoint_RMS_px']
        if problem is None and (not isinstance(rms, (int, float)) or not math.isfinite(rms)):
            problem = 'VALIDATION_ENDPOINT_RMS_NONFINITE'
        expected = ('UNVERIFIED_RETAIN' if problem is not None else
            'SUPPORTED_RETAIN' if rms <= V6.POLICY['endpoint_RMS_threshold_px'] else 'CONTRADICTED_REJECT')
        require(record['decision'] == expected, 'V10 stored fixed endpoint decision differs')
    return _copy(records)


def weak_witness(solved):
    """Apply inherited strict threshold to the already computed spectra."""
    best = solved.get('best_candidate')
    result = dict(schema='selected_filtered_local_observability_gate_v11',
        source='fresh unchanged V9 selected candidate stored normalized spectra',
        relative_threshold=WEAK_RELATIVE_THRESHOLD, new_Jacobian_SVD_calls=0,
        selected_candidate_index=None, observed_weak=False, modeled_weak=False,
        observed_singular_values=None, modeled_singular_values=None,
        observed_witness=None, modeled_witness=None, rank_only_proposal_available=bool(solved['available']),
        applied=False, no_alternative_reselection=True, numeric_NEW_is_not_pose_accuracy_success=True)
    if best is None: return result
    result['selected_candidate_index'] = best['candidate_index']
    geometry = best['point_line_geometry']
    for label, key in (('observed', 'observed_normal_joint'), ('modeled', 'modeled_normal_joint')):
        witness = geometry[key]; singular = witness['singular_values']
        require(witness.get('normalization') == 'unit L2 Jacobian columns' and
            isinstance(singular, list) and len(singular) == 6 and
            all(type(s) in (int, float) and math.isfinite(s) and s >= 0 for s in singular) and
            all(a >= b for a, b in zip(singular, singular[1:])),
            'fresh selected candidate normalized singular witness differs')
        result[label + '_witness'] = _copy(witness)
        result[label + '_singular_values'] = list(singular)
        result[label + '_weak'] = bool(singular[-1] < singular[0] * WEAK_RELATIVE_THRESHOLD)
    result['applied'] = bool(solved['available'] and (result['observed_weak'] or result['modeled_weak']))
    return result


def weak_safe_view(solved, gate, input_points):
    """Decline the selected proposal; preserve all candidates and attempts."""
    result = _copy(solved)
    if not gate['applied']: return result
    result['observability_gate'] = _copy(gate)
    result['fresh_filtered_solver_semantic_sha256'] = V8.semantic_sha256(solved)
    result['rejected_local_pose'] = {key: _copy(solved[key]) for key in
        ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected', 'points_final',
         'fit_input_ids', 'fit_line_edges', 'hidden_reprojected', 'state', 'reason')}
    result.update(available=False, pose_available=False, new_pose_estimated=False,
        no_pose=True, fallback_used=False, state='WEAK_OBSERVABILITY',
        reason='selected_filtered_local_observed_or_modeled_normal_joint_is_weak_under_existing_V1_predicate',
        output_status='NO_NEW_POSE', local_refinement_estimated=False,
        R_cf=None, R_physical=None, centroid=None, cf_extents=None, projected=None,
        fit_input_ids=[], fit_line_edges=[], input_inliers=[], final_inliers=[], inliers=[],
        final_line_inliers=[], hidden_reprojected=False,
        point_inlier_count=0, unique_line_inlier_count=0, total_inlier_factor_count=0,
        points_final=np.asarray(input_points, float).tolist(),
        rank_only_proposal_declined=True, weak_condition_is_explicit_availability_gate=True)
    return result


def repair_observations(points, native_points, K, xyz, observation, hidden,
                        adopted_boundary_corner_ids, *, initial_pose, point_result,
                        image_size=(640, 480), endpoint_audit=None, observer=None):
    """Pure numeric kernel: fresh endpoint audit OR sealed replay, then one fit.

    No models, images, truth, files, reference poses or full-population receipts
    enter this function. A caller separately proves frozen file provenance.
    """
    require(observer is None or callable(observer), 'observer must be callable')
    require(isinstance(observation, Mapping) and not A10.TRUTH_FIELDS.intersection(observation),
        'actual GT-free decoded observation required')
    original_inputs = dict(points=points, native=native_points, K=K, xyz=xyz,
        observation=observation, hidden=hidden, adopted=adopted_boundary_corner_ids,
        initial_pose=initial_pose, point_result=point_result)
    before_input = V8.semantic_sha256(original_inputs)
    q, native_q, K, xyz = (np.asarray(v, float) for v in (points, native_points, K, xyz))
    H = tuple(V8._ids(list(hidden), 'hidden'))
    adopted = tuple(V8._ids(list(adopted_boundary_corner_ids), 'adopted_boundary_corner_ids'))
    bank = LocalPointLineBank(q, K, xyz, image_size=image_size)
    U = tuple(k for k in bank.bank.eligible if k not in H)
    lines, consumed, checks, contract_error = [], [], [], None
    try:
        lines, consumed, checks = bank.factor_validator._lines(observation, adopted, set(U))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        contract_error = dict(type=type(error).__name__, message=str(error))
    native = PoseBank(native_q, K, xyz, image_size=image_size)
    before_native = native.ledger.copy()
    if endpoint_audit is None:
        records = _endpoint_records(native, lines, H, observer)
        source = 'FRESH_NATIVE_ENDPOINT_LOO'
    else:
        records = _validate_saved_records(endpoint_audit, native, lines, H, q, adopted,
            K, xyz, image_size, consumed, checks, contract_error)
        require(endpoint_audit['actual_sparse_used_ids'] == list(U), 'V10 active sparse IDs differ')
        source = 'SEALED_V10_ENDPOINT_DECISION_REPLAY'
    rejected = [r['edge'] for r in records if r['decision'] == 'CONTRADICTED_REJECT']
    unknown = [r['edge'] for r in records if r['decision'] == 'UNVERIFIED_RETAIN']
    retained = [l['edge'] for l in lines if l['edge'] not in rejected]
    require(not set(consumed) & set(rejected), 'a point-consumed support line was rejected')
    filtered = _copy(observation)
    if contract_error is None:
        filtered['lines'] = [_copy(line) for line in observation['lines'] if line['edge'] not in rejected]
    require(V8.semantic_sha256(filtered['corners']) == V8.semantic_sha256(observation['corners']) and
        V8.semantic_sha256(filtered['queries']) == V8.semantic_sha256(observation['queries']),
        'point/query observations changed during endpoint filter')
    if observer is not None:
        observer(dict(phase='filtered_local_request_started', rejected_edge_ids=rejected,
            retained_unused_edge_ids=retained, native_bank_after=native.ledger.copy()))
    try:
        solved = bank.solve(filtered, adopted_boundary_corner_ids=adopted, hidden=H,
            initial_pose=_copy(initial_pose), point_result=_copy(point_result))
    except BaseException as error:
        if observer is not None:
            observer(dict(phase='filtered_local_request_exception',
                local_bank_counts=_copy(bank.counts), local_coordinate_bank_counts=bank.bank.ledger.copy(),
                error=dict(type=type(error).__name__, message=str(error))))
        raise
    require(solved['operation_counts']['local_solve_calls'] == 1 and
        solved['prior_used'] is False and solved['initial_pose_residual_prior'] is False,
        'one unchanged prior-free local solve required')
    require(solved['used'] == list(U) and solved['hidden'] == list(H) and
        solved['line_edges'] == retained and solved['consumed_edges'] == consumed,
        'fresh filtered factor pool or H differs')
    gate = weak_witness(solved)
    proposal_sha = V8.semantic_sha256(solved)
    views = {METHODS[0]: _copy(solved), METHODS[1]: weak_safe_view(solved, gate, q)}
    if observer is not None:
        observer(dict(phase='filtered_local_request_returned',
            full_rank_only_solver=_copy(solved), observability_gate=_copy(gate)))
    filter_witness = dict(schema='actual_unused_line_endpoint_filter_v11',
        endpoint_decision_source=source, endpoint_audit_semantic_sha256=
            None if endpoint_audit is None else V8.semantic_sha256(endpoint_audit),
        original_observation_semantic_sha256=V8.semantic_sha256(observation),
        filtered_observation_semantic_sha256=V8.semantic_sha256(filtered),
        original_observation_binding=S9._binding(observation), filtered_observation_binding=S9._binding(filtered),
        original_factor_pool_binding=None if contract_error else S9._binding(dict(points=q[list(U)], point_ids=U,
            lines=[dict(edge=l['edge'], normal=l['normal'], offset=l['offset'],
                        support_query_ids=l['support_query_ids']) for l in lines])),
        filtered_factor_pool_binding=_copy(solved.get('factor_pool_binding')),
        original_actual_unused_edge_ids=[l['edge'] for l in lines],
        rejected_edge_ids=rejected, retained_unused_edge_ids=retained, unknown_retained_edge_ids=unknown,
        consumed_edges=consumed, original_line_contract_checks=checks, contract_error=contract_error,
        original_actual_source_lines=[_source_line(l) for l in lines], endpoint_records=_copy(records),
        all_rejected_raw_duplicate_records_removed=True, corners_and_queries_unchanged=True,
        native_N3_filled_for_final_fit=False, no_endpoint_projection_as_observation=True,
        physical_ownership_truth='UNKNOWN')
    ledger = dict(schema='endpoint_observability_repair_ledger_v11',
        actual_filtered_local_solve_calls=1, reported_output_views=2,
        endpoint_decision_source=source, endpoint_validation_requests=
            len(records) if endpoint_audit is None else 0,
        endpoint_decision_records=len(records), endpoint_coordinate_bank_count=1,
        endpoint_native_input_hash=native.digest, endpoint_native_bank_before=before_native,
        endpoint_native_bank_after=native.ledger.copy(),
        endpoint_native_bank_delta=_delta(native.ledger, before_native),
        local_operation_counts=_copy(solved['operation_counts']), local_bank_counts=_copy(bank.counts),
        local_coordinate_bank=dict(input_hash=bank.bank.digest, **dict(bank.bank.ledger)),
        observation_filter=filter_witness, observability_gate=_copy(gate),
        full_rank_only_solver=_copy(solved), fresh_filtered_solver_semantic_sha256=proposal_sha,
        raw_input_semantic_sha256=before_input, no_second_local_optimizer_or_alternative_reselection=True,
        original_H_preserved=True, model_head_decode_initial_pose_calls=0,
        new_GT_scores_training_RGB_calls=0, latency_claim=False,
        whole_file_population_cleanup_provenance_is_caller_responsibility=True)
    require(before_input == V8.semantic_sha256(original_inputs), 'original kernel inputs mutated')
    return views, ledger


def assemble_view(parent, native, q, initial, H, contract, solved, method, ledger, *, fresh=False):
    """Original V2 display/fallback contract, applied to one recorded view."""
    require(method in METHODS, 'unknown V11 output view')
    assembled = assemble(native, q, initial, H, solved, 'VALIDATED_ROLE_ONLY', contract)
    row = {key: _copy(parent[key]) for key in ('id', 'session', 'K', 'xyz', 'raw_hw',
        'selected_index', 'fixed_metadata', 'head_arm', 'head_mode', 'head_checkpoint_sha256',
        'calibration_binding') if key in parent}
    for key in ('mask_diagnostic', 'observation_raw_logits_sha256', 'selected_queries', 'source_lines',
                'model_query_anchor', 'Base_proposal_feature_dependence_retained'):
        if key in parent: row[key] = _copy(parent[key])
    row.update(assembled)
    row.update(method=method, comparator=COMPARATOR, parent_method=PARENT_PRIMARY,
        observation_supply='BOUNDARY_ONLY', selected_corner_ids=list(contract['validated_boundary_corner_ids']),
        original_native_N3_points=np.asarray(native, float).tolist(),
        predicted_initial_N3_hidden=list(H), original_self_hidden_initial=list(H),
        initial_H_applied=True, diagnostic_no_mask=False,
        endpoint_observability_repair_policy=_copy(POLICY),
        local_point_line_policy=_copy(S9.POLICY), local_point_line_refinement=True,
        observation_filter=_copy(ledger['observation_filter']),
        observability_gate={**_copy(ledger['observability_gate']),
            'output_view_applies_gate': method == PRIMARY},
        fresh_filtered_solver_semantic_sha256=ledger['fresh_filtered_solver_semantic_sha256'],
        rank_only_control_shares_actual_solver_with_weak_safe=True,
        full_candidate_and_operation_witnesses_preserved=True,
        pose_estimation_kind='LOCAL_POINT_LINE_REFINEMENT', independent_PnP_or_PnL=False,
        partial_lines_not_pose_inputs=False, final_numeric_pose_has_initial_prior=False,
        initial_pose_used_only_for_H_and_fallback=False, initial_pose_may_supply_local_optimizer_start=True,
        conditional_H_from_initial_N3=True, missing_sparse_filled_for_numeric_fit=False,
        native_N3_is_numeric_final_pose_observation=False, display_missing_uses_native_N3=True,
        selection_validation_fit_excludes_candidate_corner=False,
        selection_validation_is_fully_statistically_independent=False,
        selection_validation_has_shared_initial_prior=False,
        original_lines_used_only_if_not_consumed_by_actual_adopted_points=True,
        stored_jacobian_fit_ids=list(solved['fit_input_ids']) if solved['available'] else [],
        stored_jacobian_fit_line_edges=list(solved['fit_line_edges']) if solved['available'] else [],
        stored_jacobian_scope='whole actual filtered pool, not8px diagnostic support',
        numerical_NEW_is_not_pose_accuracy_success=True, global_uniqueness_proven=False,
        initial_pose_source='fixed N3 and available sparse point pose; LOCAL starts only',
        head_forward_calls=0 if not fresh else int(parent['selected_index'] is not None),
        calibration_or_decode_calls=0 if not fresh else int(parent['selected_index'] is not None),
        fresh_complete_capture=fresh, cached_accuracy_row_used=False,
        sealed_observations_input=not fresh,
        solver_replay_latency_claim=False, oracle=False)
    require(np.array_equal(np.asarray(row['input_points']), np.asarray(q), equal_nan=True) and
        np.array_equal(np.asarray(row['native_points'])[8], np.asarray(native)[8], equal_nan=True),
        'sparse fit or detector center changed')
    require(set(H).isdisjoint(solved['fit_input_ids']) and row['reprojections_reused_as_observations'] is False,
        'self-hidden coordinate entered fit or projection was reused')
    return row


def solve_repair_packet(local_packet, audit_row, *, observer=None):
    """Completed V7/V10 packet replay -> two rows, one actual filtered fit."""
    before = V8.semantic_sha256(dict(local_packet=local_packet, audit_row=audit_row))
    p = V9.validate_local_packet(local_packet); parent = p['row']
    require(audit_row.get('id') == parent['id'] and audit_row.get('session') == parent['session'] and
        audit_row.get('parent_geometry_row_semantic_sha256') == V8.semantic_sha256(parent) and
        audit_row.get('original_ROLE_observation_semantic_sha256') == V8.semantic_sha256(p['observation_row']) and
        audit_row.get('original_native_N3_semantic_sha256') == V8.semantic_sha256(p['native']),
        'completed V10 same-source frame/ROLE/native lineage differs')
    views, ledger = repair_observations(p['q'], p['native'], p['K'], p['xyz'], p['observation'], p['H'],
        p['contract']['validated_boundary_corner_ids'], initial_pose=parent['initial_pose'],
        point_result=parent['solver'], image_size=p['size'], endpoint_audit=audit_row, observer=observer)
    require(ledger['observation_filter']['original_factor_pool_binding'] == audit_row['factor_pool_binding'],
        'immutable V10 original V9 factor-pool binding differs')
    rows = [assemble_view(parent, p['native'], p['q'], parent['initial_pose'], p['H'], p['contract'],
        views[method], method, ledger) for method in METHODS]
    for row in rows:
        row.update(parent_geometry_row_semantic_sha256=V8.semantic_sha256(parent),
            parent_observation_row_semantic_sha256=V8.semantic_sha256(p['observation_row']),
            parent_native_N3_semantic_sha256=V8.semantic_sha256(p['native']),
            parent_completion=_copy(p['completion']), same_sparse_observation=True,
            endpoint_audit_semantic_sha256=V8.semantic_sha256(audit_row),
            original_V9_geometry_row_semantic_sha256=audit_row['local_geometry_row_semantic_sha256'])
    ledger.update(id=parent['id'], session=parent['session'], input_packet_semantic_sha256=before,
        parent_geometry_row_semantic_sha256=V8.semantic_sha256(parent),
        parent_observation_row_semantic_sha256=V8.semantic_sha256(p['observation_row']),
        endpoint_audit_semantic_sha256=V8.semantic_sha256(audit_row),
        original_V9_geometry_row_semantic_sha256=audit_row['local_geometry_row_semantic_sha256'])
    require(before == V8.semantic_sha256(dict(local_packet=local_packet, audit_row=audit_row)),
        'sealed repair packet mutated')
    return rows, ledger
