"""Three fixed controls on one immutable V7 sparse ROLE observation.

``solve_controls(packet)`` is an in-memory functional interface. The caller
supplies ``parent_row`` from GT-free V7 GEOMETRY_SEALED (method
IMAGE_ROLE_BOUNDARY_ONLY), ``native_N3_points`` from its same-frame sealed
OBSERVATIONS, and ``parent_completion`` with the following normalized fields:
complete=True, frames=245, geometry_rows=1960, observation_rows=735,
fixed_rows=490, cleanup_error=None, protocol_sha256, geometry_sha256, and
inference_receipt_sha256. The caller must independently verify those hashes,
the full population and successful model cleanup before constructing packets.
This pure module checks their structural declarations; it performs no file
reads and cannot certify whole-file provenance from one row.

The V7 row's ``native_points`` is its final display output, NOT the original
N3 observation. Original N3 points therefore must be supplied explicitly.
The sparse input remains sparse; native N3 supplies display/fallback only.
One matching prior-free V4 bank is shared among three actual solve calls.
H+robust is an honest control replay, not a copied result or new head inference.
Ordinary solves/refits all active U and may return NEW with fewer than four
8px diagnostic inliers; the four-inlier gate belongs to robust only.

No head, image, GT, calibration, query decoding, initial pose estimation,
learning, renderer, timing benchmark or performance-based policy is here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
import operator

import numpy as np

from ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank
from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble

PARENT_METHOD = 'IMAGE_ROLE_BOUNDARY_ONLY'
PRIMARY = 'ROLE_BOUNDARY_H_ROBUST'
METHODS = (PRIMARY, 'ROLE_BOUNDARY_H_STANDARD', 'ROLE_BOUNDARY_NO_MASK_ROBUST')
ROLE_CHECKPOINT_SHA256 = '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d'
PARITY_ATOL = 1e-7
POLICY = dict(
    schema='same_sparse_ROLE_solver_mask_controls_v8',
    parent_method=PARENT_METHOD, methods=list(METHODS), primary=PRIMARY,
    primary_is_unchanged_parent_diagnostic_control=True,
    controls_change_observation_admission=False,
    shared_input='exact sealed V7 sparse q/K/physical xyz/image size',
    final_solver='unchanged prior-free V4 PoseBank',
    initial_pose='sealed N3 initial pose for unchanged fallback only',
    hidden='sealed original N3 H; no mask update or human repair',
    routes=dict(ROLE_BOUNDARY_H_ROBUST=dict(robust=True, apply_H=True),
                ROLE_BOUNDARY_H_STANDARD=dict(robust=False, apply_H=True),
                ROLE_BOUNDARY_NO_MASK_ROBUST=dict(robust=True, apply_H=False)),
    ordinary='whole active U SSE/LM/rank; 8px inliers are diagnostic only',
    robust='same finite 4-ID consensus; at least4 final inliers',
    same_coordinate_bank_shared_across_masks_and_solver_modes=True,
    missing_sparse_coordinates_filled_for_numeric_fit=False,
    native_N3_is_numeric_final_pose_observation=False,
    H_projection='NEW final R,t only; never fit projections again',
    nomask='explicit diagnostic; original predicted H retained separately',
    same_U_does_not_imply_same_H_display=True,
    parent_numeric_parity_atol=PARITY_ATOL,
    latency_claim=False, model_forward_calls=0, new_training_updates=0, new_RGB=0,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ids(values, label):
    require(isinstance(values, (list, tuple)), label + ' must be a list/tuple')
    try:
        result = [operator.index(v) for v in values]
    except TypeError as error:
        raise ValueError(label + ' contains a noninteger ID') from error
    require(all(not isinstance(v, bool) for v in values) and
            result == sorted(set(result)) and all(0 <= v < 8 for v in result),
            label + ' must be sorted unique native corner IDs0..7')
    return result


def _canonical(value):
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def semantic_sha256(value):
    """A packet/row semantic digest; never substitute it for file-byte SHA."""
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _sha(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), label + ' is not SHA256')


def validate_parent(packet):
    """Structural, GT-free validation before constructing any numeric bank."""
    require(isinstance(packet, Mapping) and
            set(packet) == {'parent_row', 'native_N3_points', 'parent_completion'},
            'use only the explicit normalized GT-free parent packet')
    row, completion = packet['parent_row'], packet['parent_completion']
    require(isinstance(row, Mapping) and isinstance(completion, Mapping), 'parent mappings required')
    require(completion.get('complete') is True and completion.get('frames') == 245 and
            completion.get('geometry_rows') == 1960 and completion.get('observation_rows') == 735 and
            completion.get('fixed_rows') == 490 and 'cleanup_error' in completion and
            completion['cleanup_error'] is None, 'incomplete or unclean parent population')
    for name in ('protocol_sha256', 'geometry_sha256', 'inference_receipt_sha256'):
        _sha(completion.get(name), 'parent ' + name)
    require(not set(row) & {'corner', 'pose', 'mask_audit', 'baseline_corner', 'baseline_pose',
                           'GT', 'ground_truth', 'reference_pose', 'human_states_native'},
            'scored/truth fields cannot enter the control packet')
    required = {'id', 'session', 'method', 'K', 'xyz', 'raw_hw', 'selected_index', 'fixed_metadata',
                'input_points', 'native_points', 'initial_pose', 'hidden_initial',
                'predicted_initial_N3_hidden', 'observation_contract', 'solver', 'actual_pose',
                'output_status', 'new_pose_estimated', 'fallback_used', 'pose_available', 'no_pose',
                'hidden_reprojected', 'hidden_after', 'hidden_set_changed', 'reprojected_ids',
                'output_coordinate_sources', 'head_arm', 'head_mode', 'observation_supply',
                'head_checkpoint_sha256', 'calibration_binding', 'selected_corner_ids',
                'oracle', 'partial_lines_not_pose_inputs', 'final_numeric_pose_has_initial_prior'}
    require(required <= set(row), 'required sealed parent fields absent')
    require(row['method'] == PARENT_METHOD and row['head_arm'] == row['head_mode'] == 'IMAGE_ROLE' and
            row['observation_supply'] == 'BOUNDARY_ONLY' and
            row['head_checkpoint_sha256'] == ROLE_CHECKPOINT_SHA256,
            'use the unchanged corrected ROLE boundary-only parent')
    require(row['oracle'] is False and row['partial_lines_not_pose_inputs'] is True and
            row['final_numeric_pose_has_initial_prior'] is False,
            'parent is not the frozen observation-only nonoracle point route')
    require(isinstance(row['id'], str) and isinstance(row['session'], str), 'parent identity absent')
    require(isinstance(row['fixed_metadata'], Mapping) and row['fixed_metadata'].get('preserved') is True,
            'original detector metadata preservation absent')
    require(isinstance(row['calibration_binding'], Mapping), 'parent CAL binding absent')
    _sha(row['calibration_binding'].get('sha256'), 'parent CAL binding')
    q, native = np.asarray(row['input_points'], dtype=np.float64), np.asarray(packet['native_N3_points'], dtype=np.float64)
    K, xyz = np.asarray(row['K'], dtype=np.float64), np.asarray(row['xyz'], dtype=np.float64)
    require(q.shape == native.shape == (9, 2) and not np.isinf(q).any() and not np.isinf(native).any(),
            'sparse/native points must be [9,2], finite or missing NaN/null')
    require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0,
            'invalid camera matrix')
    require(xyz.shape == (3,) and np.isfinite(xyz).all() and (xyz > 0).all(), 'invalid physical W,H,D')
    hw = row['raw_hw']
    require(isinstance(hw, (list, tuple)) and len(hw) == 2 and
            all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in hw), 'invalid raw image size')
    H = _ids(row['hidden_initial'], 'parent H')
    require(H == _ids(row['predicted_initial_N3_hidden'], 'predicted initial N3 H'), 'parent H declarations differ')
    contract = row['observation_contract']
    require(isinstance(contract, Mapping), 'parent observation contract absent')
    admitted = _ids(contract.get('validated_boundary_corner_ids'), 'admitted corner IDs')
    require(admitted == _ids(row['selected_corner_ids'], 'selected proposal IDs'), 'parent proposal identities differ')
    finite = np.isfinite(q[:8]).all(axis=1)
    require(list(np.flatnonzero(finite)) == admitted and
            all(np.isnan(q[k]).all() for k in range(8) if k not in admitted),
            'missing sparse coordinates were filled or partially nonfinite')
    require(np.array_equal(q[8], native[8], equal_nan=True), 'detector center differs')
    require(contract.get('missing_ROLE_ONLY_corner_is_NaN') is True and
            contract.get('boundary_only_native_distance_used_for_admission') is False and
            contract.get('boundary_only_LOO_used_for_admission') is False and
            contract.get('native_N3_used_for_numeric_final_fit') is False and
            contract.get('heldout_pose_calls') == 0, 'boundary-only admission contract differs')
    require(isinstance(row['initial_pose'], Mapping) and isinstance(row['solver'], Mapping),
            'sealed initial/solver witnesses absent')
    require(row['solver'].get('prior_used') is False and
            row['solver'].get('known_dimension_constraint_used') is False and
            row['solver'].get('hidden') == H, 'parent numeric pose prior or H differs')
    return dict(row=row, completion=completion, q=q.copy(), native=native.copy(),
                K=K.copy(), xyz=xyz.copy(), H=H, size=(hw[1], hw[0]), contract=copy.deepcopy(contract))


def _numeric_parity(left, right, label, numeric, exact):
    if left is None or right is None:
        require(left is None and right is None, label + ' missing values differ')
        exact[label] = True
        numeric[label] = 0.
        return
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    require(x.shape == y.shape and np.array_equal(np.isnan(x), np.isnan(y)) and
            np.array_equal(np.isposinf(x), np.isposinf(y)) and np.array_equal(np.isneginf(x), np.isneginf(y)),
            label + ' shape/nonfinite pattern differs')
    valid = np.isfinite(x) & np.isfinite(y)
    difference = float(np.max(np.abs(x[valid] - y[valid]))) if valid.any() else 0.
    numeric[label] = difference
    exact[label] = bool(np.array_equal(x, y, equal_nan=True))
    require(difference <= PARITY_ATOL, label + ' numeric parity exceeds fixed tolerance')


def solver_parity(current, reference):
    """Compare numerical/categorical output; exclude operation/cache receipts."""
    numeric, exact = {}, {}
    for key in ('available', 'state', 'reason', 'used', 'fit_input_ids', 'final_inliers', 'prior_used',
                'known_dimension_constraint_used', 'unresolved_ambiguity', 'weak_four_point_consensus'):
        require(key in current and key in reference, 'solver parity field missing: ' + key)
        require(current[key] == reference[key], 'solver parity differs: ' + key)
    for key in ('generator_ids', 'selected_hypothesis', 'dimension_index'):
        require((key in current) == (key in reference), 'solver parity field presence differs: ' + key)
        require(current.get(key) == reference.get(key), 'solver parity differs: ' + key)
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected',
                'residuals_used_px', 'sse_px2', 'truncated_sse_px2'):
        require((key in current) == (key in reference), 'solver parity field presence differs: ' + key)
        _numeric_parity(current.get(key), reference.get(key), 'solver.' + key, numeric, exact)
    return dict(passed=True, atol=PARITY_ATOL, rtol=0, max_abs_differences=numeric,
                numeric_exact_equality=exact, operation_and_cache_counts_compared=False)


def parent_parity(current, parent):
    result = solver_parity(current['solver'], parent['solver'])
    for key in ('output_status', 'new_pose_estimated', 'fallback_used', 'pose_available', 'no_pose',
                'hidden_initial', 'hidden_after', 'hidden_set_changed', 'hidden_reprojected',
                'reprojected_ids', 'output_coordinate_sources'):
        require(current[key] == parent[key], 'parent replay output differs: ' + key)
    for key in ('input_points', 'native_points'):
        _numeric_parity(current[key], parent[key], key, result['max_abs_differences'], result['numeric_exact_equality'])
    for key in ('available', 'selected_hypothesis'):
        require(current['actual_pose'].get(key) == parent['actual_pose'].get(key), 'parent actual pose differs: ' + key)
    for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected'):
        _numeric_parity(current['actual_pose'].get(key), parent['actual_pose'].get(key),
                        'actual_pose.' + key, result['max_abs_differences'], result['numeric_exact_equality'])
    return result


def solve_controls(packet):
    """Return ``(three_normalized_rows, ledger)`` without mutating inputs.

    Each result is ready for the caller's GT-free row writer/scoring adapter.
    The unchanged parent H+robust control is first replayed and verified before
    either new diagnostic route is solved. All three calls count as work.
    Equal H/no-H U is explicitly recorded and checked for identical numerical
    outcomes; their H replacement/display fields are intentionally distinct.
    """
    p = validate_parent(packet)
    before = semantic_sha256(packet)
    row = p['row']
    bank = PoseBank(p['q'], p['K'], p['xyz'], image_size=p['size'])
    results, control_parity = [], None
    for method in METHODS:
        spec = POLICY['routes'][method]
        H = p['H'] if spec['apply_H'] else []
        solved = bank.solve(hidden=H, robust=spec['robust'])
        require(solved['prior_used'] is False and solved['known_dimension_constraint_used'] is False,
                'initial/external dimension prior entered the same-observation control')
        result = assemble(p['native'], p['q'], copy.deepcopy(row['initial_pose']), H,
                          solved, 'VALIDATED_ROLE_ONLY', p['contract'])
        normalized = {key: copy.deepcopy(row[key]) for key in
                      ('id', 'session', 'K', 'xyz', 'raw_hw', 'selected_index', 'fixed_metadata',
                       'head_arm', 'head_mode', 'head_checkpoint_sha256', 'calibration_binding')}
        normalized.update(result)
        for key in ('mask_diagnostic', 'observation_raw_logits_sha256', 'selected_queries', 'source_lines',
                    'model_query_anchor', 'initial_pose_source', 'Base_proposal_feature_dependence_retained'):
            if key in row:
                normalized[key] = copy.deepcopy(row[key])
        normalized.update(method=method, observation_supply='BOUNDARY_ONLY',
            selected_corner_ids=list(p['contract']['validated_boundary_corner_ids']),
            predicted_initial_N3_hidden=list(p['H']), original_self_hidden_initial=list(p['H']),
            initial_H_applied=bool(spec['apply_H']), diagnostic_no_mask=not spec['apply_H'],
            same_observation_policy=copy.deepcopy(POLICY), partial_lines_not_pose_inputs=True,
            final_numeric_pose_has_initial_prior=False, initial_pose_used_only_for_H_and_fallback=True,
            conditional_H_from_initial_N3=True, no_match_filled_as_boundary_observation=False,
            missing_sparse_filled_for_numeric_fit=False, display_missing_uses_native_N3=True,
            selection_validation_fit_excludes_candidate_corner=False,
            selection_validation_is_fully_statistically_independent=False,
            selection_validation_has_shared_initial_prior=False,
            parent_method=PARENT_METHOD, parent_geometry_row_semantic_sha256=semantic_sha256(row),
            parent_native_N3_semantic_sha256=semantic_sha256(p['native']),
            parent_completion=copy.deepcopy(p['completion']),
            existing_control_replay=method == PRIMARY, same_sparse_observation=True,
            head_forward_calls=0, calibration_or_decode_calls=0, oracle=False)
        require(np.array_equal(np.asarray(normalized['input_points']), p['q'], equal_nan=True), 'sparse numeric input changed')
        require(np.array_equal(np.asarray(normalized['native_points'])[8], p['native'][8], equal_nan=True), 'center changed')
        require(set(H).isdisjoint(solved.get('fit_input_ids', [])), 'masked initial coordinates entered final fit')
        require(normalized['reprojections_reused_as_observations'] is False, 'reprojection was refit')
        if method == PRIMARY:
            control_parity = parent_parity(normalized, row)
        results.append(normalized)
    first, last = results[0]['solver'], results[2]['solver']
    same_U = first['used'] == last['used']
    equal_U_parity = solver_parity(last, first) if same_U else None
    require(before == semantic_sha256(packet), 'immutable parent packet was mutated')
    ledger = dict(schema='same_sparse_observation_control_ledger_v8',
        methods=list(METHODS), parent_packet_semantic_sha256=before, coordinate_bank_count=1,
        coordinate_banks=[dict(input_hash=bank.digest, prior_backend=False, **dict(bank.ledger))],
        logical_pose_paths=3, existing_control_replay_pose_paths=1, new_diagnostic_pose_paths=2,
        detector_calls=0, N3_calls=0, head_calls=0, initial_pose_calls=0,
        calibration_calls=0, query_decode_calls=0, new_training_updates=0, new_RGB=0,
        parent_replay_parity=control_parity, H_and_no_H_used_U_equal=same_U,
        H_and_no_H_used_U=dict(masked=first['used'], unmasked=last['used']),
        H_and_no_H_equal_U_numeric_parity=equal_U_parity,
        H_and_no_H_display_compared_as_same=False,
        actual_H_removed_sparse_ids=sorted(set(last['used']) - set(first['used'])),
        missing_sparse_coordinates_filled_for_numeric_fit=False,
        parent_unchanged=True, parent_full_file_provenance_is_caller_responsibility=True,
        this_module_independently_verified_full_file_provenance=False,
        latency_benchmark=False, policy=copy.deepcopy(POLICY))
    return results, ledger
