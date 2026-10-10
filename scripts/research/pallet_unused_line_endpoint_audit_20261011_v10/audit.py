"""Audit unused V9 lines using the unchanged V6 native endpoint heldout test.

The explicit packet contains GT-free frozen V7/V9 observations only. This
function requests native point-only validation poses; it never runs a new
LOCAL fit, modifies observations, filters a final fit, scores GT, or produces
a replacement final pose. Whole-file/population/cleanup/quiet/byte protection
and actual primitive entry accounting belong to the separately frozen driver.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy


METHOD = 'FROZEN_V9_UNUSED_LINE_ENDPOINT_AUDIT'
POLICY = dict(
    schema='frozen_sparse_ROLE_unused_line_endpoint_diagnosis_policy_v10',
    population='all existing easy153 and medium92 frozen V9 rows, in cohort order',
    actual_factor_pool='unchanged V9 sparse ROLE points and unconsumed physical/CAL-supported lines',
    validation_coordinates='original V7 OBS.native_N3_points, never corrected display coordinates',
    validation_solver='unchanged V4 finite four-ID point-only robust PoseBank; both registry dimensions',
    validation_exclusions='initial H union both endpoints of each actual unused semantic edge',
    observed_line_used_for_validation_pose_selection=False,
    initial_rotation_translation_dimension_or_projection_prior=False,
    endpoint_RMS_threshold_px=8.,
    endpoint_RMS='sqrt((d_a**2+d_b**2)/2), signed distances to normalized actual observed line',
    decision='same V6 SUPPORTED_RETAIN / CONTRADICTED_REJECT / UNVERIFIED_RETAIN',
    unavailable_or_ambiguous_is_UNKNOWN=True,
    one_validation_request_per_unique_original_unused_edge=True,
    same_native_coordinate_bank_shared_across_endpoint_masks=True,
    consumed_edges_or_same_line_points_not_double_counted=True,
    native_OR_sparse_coordinates_not_filled=True,
    stored_local_conditioning_and_objective_gaps='copy original witnesses; do not infer correctness from rank6',
    new_local_fits=0, new_final_pose_paths=0, new_GT_scores=0,
    new_detector_N3_head_calibration_decode_training_RGB_calls=0,
    actual_observation_packet_or_V9_geometry_modified=False,
    endpoint_projection_used_as_new_image_observation=False,
    independent_real_physical_boundary_ownership_certified=False,
    no_accuracy_or_unseen_generalization_claim=True,
    no_threshold_or_performance_setting_search=True,
)
TRUTH_FIELDS = frozenset(('corner', 'pose', 'GT', 'ground_truth', 'reference_pose',
    'human_states_native', 'baseline_corner', 'baseline_pose', 'mask_audit'))


def require(value, message):
    if not value:
        raise ValueError(message)


def delta(after, before):
    require(set(after) == set(before), 'native bank ledger schema changed')
    return {key: int(after[key] - before[key]) for key in after}


def stored_local_diagnosis(solved):
    """Copy existing conditioning/start/cost evidence without a new J or SVD."""
    fields = ('state', 'reason', 'available', 'used', 'eligible', 'hidden',
        'actual_adopted_boundary_ids', 'line_edges', 'consumed_edges', 'factor_pool',
        'factor_pool_binding', 'registered_dimensions', 'candidate_count',
        'viable_candidate_count', 'best_candidate', 'per_dimension_best',
        'alternatives', 'numerical_tied_solution_count', 'numerical_duplicate_solution_count',
        'multiple_solutions', 'unresolved_ambiguity', 'point_line_geometry',
        'diagnostic_inlier_geometry', 'diagnostic_inlier_geometry_unavailable',
        'diagnostic_inlier_support_rank6', 'diagnostic_inlier_scalar_residuals',
        'local_start_cache', 'start_source_witnesses', 'optimizer_attempts',
        'all_candidate_solutions', 'operation_counts', 'hypothesis_bank_counts')
    return dict(source='immutable V9 solver fields; no new LOCAL/Jacobian/SVD/objective evaluation',
        copied_fields={key: copy.deepcopy(solved[key]) for key in fields if key in solved},
        full_rank_is_not_good_conditioning_or_correct_correspondence=True,
        low_in_pool_loss_is_not_independent_validation=True,
        local_start_pool_is_not_exhaustive_global_branch_enumeration=True,
        real_physical_corner_or_edge_ownership='UNKNOWN; no independent real-image ownership annotation',
        initial_pose_is_start_only_not_residual_prior=True)


def audit_packet(packet, *, observer=None):
    """Return one independent line-diagnosis row and its actual-call ledger.

    ``packet`` has exactly ``local_packet`` (the unchanged V9 normalized
    parent packet) and ``local_row`` (same-frame unscored sealed V9 geometry).
    A driver must verify current whole-file SHA, complete245 populations and
    parent successful cleanup before assembling it. Only this function's
    point-only validation requests are new numerical calls.
    """
    import numpy as np
    from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
    from ..pallet_partial_line_independent_20261010_v5 import solver as V5
    from ..pallet_partial_line_heldout_20261010_v6 import solver as V6
    from ..pallet_sparse_local_line_20261010_v9 import pipeline as V9
    from ..pallet_sparse_local_line_20261010_v9 import solver as S9
    from ..pallet_same_observation_controls_20261010_v8 import controls as V8

    require(isinstance(packet, Mapping) and set(packet) == {'local_packet', 'local_row'},
            'explicit two-part frozen diagnosis packet required')
    require(observer is None or callable(observer), 'optional execution observer must be callable')
    p = V9.validate_local_packet(packet['local_packet'])
    local = packet['local_row']
    require(isinstance(local, Mapping) and not TRUTH_FIELDS.intersection(local),
            'use sealed GT-free V9 geometry, never scored rows')
    require(local['method'] == V9.METHOD and local['id'] == p['row']['id'] and
        local['session'] == p['row']['session'] and local['head_arm'] == 'IMAGE_ROLE' and
        local['observation_supply'] == 'BOUNDARY_ONLY' and local['oracle'] is False and
        local['initial_H_applied'] is True and local['same_sparse_observation'] is True,
        'same-frame frozen sparse ROLE LOCAL row required')
    require(local['parent_geometry_row_semantic_sha256'] == V8.semantic_sha256(p['row']) and
        local['parent_observation_row_semantic_sha256'] == V8.semantic_sha256(p['observation_row']) and
        local['parent_native_N3_semantic_sha256'] == V8.semantic_sha256(p['native']),
        'local source row provenance differs')
    require(np.array_equal(np.asarray(local['input_points'], float), p['q'], equal_nan=True) and
        np.array_equal(np.asarray(local['original_native_N3_points'], float), p['native'], equal_nan=True) and
        np.array_equal(np.asarray(local['K'], float), p['K']) and
        np.array_equal(np.asarray(local['xyz'], float), p['xyz']) and
        local['raw_hw'] == p['row']['raw_hw'] and local['hidden_initial'] == p['H'],
        'saved local sparse/native/K/registry/H contract differs')
    solved = local['solver']
    require(isinstance(solved, Mapping) and solved['solver'] == 'LOCAL_POINT_LINE' and
        solved['initial_pose_residual_prior'] is False and solved['global_uniqueness_proven'] is False and
        solved['reprojected_points_reused_as_observations'] is False,
        'frozen numerical LOCAL policy differs')
    require(V6.POLICY['endpoint_RMS_threshold_px'] == POLICY['endpoint_RMS_threshold_px'] and
        V6.POLICY['validation_coordinates'] == 'unchanged native N3 RGB observations only' and
        V6.POLICY['unavailable_is_no_match'] is False and
        V6.POLICY['observed_line_used_to_choose_validation_pose'] is False,
        'unchanged original V6 endpoint audit policy required')

    before_packet = V8.semantic_sha256(packet)
    H = tuple(p['H'])
    adopted = tuple(p['contract']['validated_boundary_corner_ids'])
    # This factor validator reconstructs exactly the actual V9 line pool.
    # Its solve method and local optimizer are never called.
    factors = V5.PointLineBank(p['q'], p['K'], p['xyz'], image_size=p['size'])
    U = tuple(k for k in factors.bank.eligible if k not in H)
    require(list(U) == solved['used'] and set(H).isdisjoint(solved['fit_input_ids']),
        'local actual pool or self-hidden fit exclusion differs')
    lines, consumed, checks, contract_error = [], [], [], None
    try:
        lines, consumed, checks = factors._lines(p['observation'], adopted, set(U))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        require(solved['state'] == 'INVALID_OBSERVATION_CONTRACT' and solved['reason'] == str(error),
            'frozen/local factor contract validation disagrees')
        contract_error = dict(type=type(error).__name__, message=str(error))
    if contract_error is None:
        require([line['edge'] for line in lines] == solved['line_edges'] and
            consumed == solved['consumed_edges'] and checks == solved['line_contract_checks'],
            'not the immutable V9 unused-line pool')
        expected_binding = S9._binding(dict(points=p['q'][list(U)], point_ids=U,
            lines=[dict(edge=l['edge'], normal=l['normal'], offset=l['offset'],
                        support_query_ids=l['support_query_ids']) for l in lines]))
        require(expected_binding == solved['factor_pool_binding'], 'V9 factor-pool semantic binding differs')

    # One immutable native-N3 bank per frame. No initial pose, line, selected
    # branch or saved LOCAL pose is supplied to this numerical initializer.
    native = PoseBank(p['native'], p['K'], p['xyz'], image_size=p['size'])
    require(native.known_dimension_index is None, 'native validation dimension prior supplied')
    bank_before = native.ledger.copy()
    records = []
    if observer is not None:
        observer(dict(phase='native_bank_ready', native_input_hash=native.digest,
            native_bank_before=bank_before, actual_unused_edges=[line['edge'] for line in lines]))
    for line in lines:
        edge = line['edge']
        endpoints = tuple(V5.EDGES[edge])
        blocked = tuple(sorted(set(H) | set(endpoints)))
        prior = native.ledger.copy()
        # Exactly the original V6 request, including typed under-four failure.
        if observer is not None:
            observer(dict(phase='endpoint_request_started', edge=edge,
                validation_excluded=list(blocked), bank_before=prior))
        try:
            validated = native.solve(excluded=endpoints, hidden=H, robust=True)
        except BaseException as error:
            if observer is not None:
                observer(dict(phase='endpoint_request_exception', edge=edge,
                    bank_after=native.ledger.copy(),
                    error=dict(type=type(error).__name__, message=str(error))))
            raise
        after = native.ledger.copy()
        if observer is not None:
            observer(dict(phase='endpoint_request_returned', edge=edge,
                bank_after=after, solved=copy.deepcopy(validated)))
        problem = V6._witness_problem(validated, blocked)
        distances, rms = None, None
        if problem is None:
            projected = np.asarray(validated['projected'], float)
            distances = projected[list(endpoints)] @ line['normal'] - line['offset']
            rms = float(np.sqrt(np.mean(distances ** 2)))
            if not np.isfinite(rms):
                problem = 'VALIDATION_ENDPOINT_RMS_NONFINITE'
        if problem is not None:
            decision, reason = 'UNVERIFIED_RETAIN', problem
        elif rms <= V6.POLICY['endpoint_RMS_threshold_px']:
            decision, reason = 'SUPPORTED_RETAIN', 'AVAILABLE_NATIVE_N3_ENDPOINT_RMS_WITHIN_FIXED_8PX'
        else:
            decision, reason = 'CONTRADICTED_REJECT', 'AVAILABLE_NATIVE_N3_ENDPOINT_RMS_EXCEEDS_FIXED_8PX'
        records.append(dict(edge=edge, endpoints=list(endpoints), hidden=list(H),
            validator_temporary_excluded=list(endpoints), validation_excluded=list(blocked),
            actual_self_hidden_separate_from_temporary_exclusion=True,
            native_input_hash=native.digest, attempted_pose_calls=1,
            decision=decision, reason=reason, endpoint_RMS_px=rms,
            signed_endpoint_distances_px=None if distances is None else distances.tolist(),
            source_line=dict(edge=edge, endpoints=list(endpoints), normal=line['normal'].tolist(),
                offset=line['offset'], support_query_ids=list(line['support_query_ids']),
                raw=copy.deepcopy(line['source_line'])),
            solved=copy.deepcopy(validated), bank_before=prior, bank_after=after,
            bank_delta=delta(after, prior), observed_line_supplied_to_validation_solver=False,
            global_cached_generation_may_include_blocked_IDs=True,
            active_scoring_generators_refits_exclude_blocked_IDs=True,
            physical_ownership_certified=False, validation_pose_may_be_wrong_consensus=True))

    bank_after = native.ledger.copy()
    require(all(factors.counts[key] == 0 for key in factors.counts), 'factor helper performed a pose/J fit')
    require(factors.bank.ledger == {key: 0 for key in factors.bank.ledger},
        'sparse factor helper generated numerical hypotheses')
    require(before_packet == V8.semantic_sha256(packet), 'input packet mutated during diagnosis')
    rejected = [r['edge'] for r in records if r['decision'] == 'CONTRADICTED_REJECT']
    unknown = [r['edge'] for r in records if r['decision'] == 'UNVERIFIED_RETAIN']
    supported = [r['edge'] for r in records if r['decision'] == 'SUPPORTED_RETAIN']
    require(not set(consumed).intersection(rejected) and len(records) == len(lines),
        'consumed/duplicate factor entered endpoint audit')
    row = dict(schema='frozen_sparse_ROLE_unused_line_endpoint_diagnosis_row_v10',
        id=local['id'], session=local['session'], method=METHOD, policy=copy.deepcopy(POLICY),
        local_geometry_row_semantic_sha256=V8.semantic_sha256(local),
        parent_geometry_row_semantic_sha256=local['parent_geometry_row_semantic_sha256'],
        original_ROLE_observation_semantic_sha256=local['parent_observation_row_semantic_sha256'],
        original_native_N3_semantic_sha256=local['parent_native_N3_semantic_sha256'],
        K=p['K'].tolist(), xyz=p['xyz'].tolist(), raw_hw=copy.deepcopy(local['raw_hw']),
        original_native_N3_points=p['native'].tolist(), original_sparse_points=p['q'].tolist(),
        initial_H=list(H), actual_sparse_used_ids=list(U), actual_adopted_boundary_ids=list(adopted),
        factor_pool_binding=copy.deepcopy(solved.get('factor_pool_binding')),
        original_actual_unused_edges=[l['edge'] for l in lines], consumed_edges=consumed,
        original_line_contract_checks=checks, contract_error=contract_error,
        endpoint_records=records, diagnostic_contradicted_edges=rejected,
        diagnostic_unknown_edges=unknown, diagnostic_supported_edges=supported,
        stored_local_diagnosis=stored_local_diagnosis(solved),
        native_bank_before=bank_before, native_bank_after=bank_after,
        native_bank_delta=delta(bank_after, bank_before), original_input_packet_unchanged=True,
        no_final_observation_filter_applied=True, new_final_pose_computed=False,
        new_LOCAL_optimizer_calls=0, new_GT_scores=0, new_model_image_calls=0,
        original_V9_pose_and_status_preserved=True, GT_free=True,
        physical_ownership_truth='UNKNOWN', classification_mismatch_is_not_pose_failure=True)
    ledger = dict(schema='frozen_sparse_ROLE_unused_line_endpoint_diagnosis_ledger_v10',
        id=local['id'], session=local['session'], input_packet_semantic_sha256=before_packet,
        input_packet_unchanged=True, endpoint_validation_requests=len(records),
        available_validation_poses=sum(r['solved']['available'] is True for r in records),
        checkable_validation_poses=len(supported)+len(rejected), unverified_validation_poses=len(unknown),
        native_coordinate_bank_count=1, native_coordinate_bank_hash=native.digest,
        native_bank_before=bank_before, native_bank_after=bank_after,
        native_bank_delta=delta(bank_after, bank_before),
        factor_validator_pose_or_J_operations=dict(factors.counts),
        factor_validator_native_solver_operations=factors.bank.ledger.copy(),
        new_final_pose_paths=0, new_LOCAL_optimizer_calls=0, new_GT_scores=0,
        new_detector_N3_head_decode_calls=0, new_local_conditioning_Jacobian_or_SVD_calls=0,
        native_point_validation_may_compute_Jacobian_and_rank=True,
        native_point_validation_SVD_entries_not_instrumented_here=True,
        candidate_conditioning_fields_copied_without_recompute=True,
        endpoint_validation_hypotheses_are_actual_new_point_PnP_calls=True,
        latency_or_accuracy_improvement_claim=False)
    return row, ledger
