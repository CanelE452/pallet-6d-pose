"""Public-only scalar audit of explicit two-endpoint native-N3 validation.

Reads saved observations, sealed geometry and scored rows. No research module,
GT cache, image, model, PnP, optimizer, NumPy, Jacobian or SVD is imported.
Run only after the complete v6 geometry and scoring artifacts are available.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_partial_line_heldout_20261010_v6'
PRIVATE = Path('/tmp/pallet-partial-line-heldout-private-20261010-v6')
PRIMARY = 'N3_INDEPENDENT_CORNERWISE_ROLE_ENDPOINT_VALIDATED_LINES'
CONTROL = 'N3_INDEPENDENT_CORNERWISE_ROLE_PARTIAL_LINES'
METHODS = (PRIMARY, 'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK', CONTROL)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
SOURCE_REGISTRY_EDGES = frozenset((0, 2, 4, 6, 8, 9, 10, 11))
OBS_WRAPPER_KEYS = frozenset(('id', 'session', 'GT_input', 'original_base_points',
                            'native_N3_points', 'initial_N3_pose', 'predicted_N3_hidden'))
NO_PRIOR_FLAGS = ('prior_used', 'initial_pose_used', 'initial_projection_used',
                  'initial_dimension_prior_used', 'known_dimension_constraint_used',
                  'excluded_image_coordinates_used_for_scoring',
                  'excluded_image_coordinates_used_for_equivalence',
                  'reprojected_points_reused_as_observations')
ATOL, RTOL, RADIUS = 1e-7, 1e-12, 8.0


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def binding(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    absolute = path.resolve()
    name = str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name
    return dict(path=name, sha256=digest.hexdigest(), bytes=path.stat().st_size)


def bound(path, expected):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'binding differs: ' + path.name)


def guard(folder, protocol, output):
    for path in (folder.absolute(), protocol.absolute(), output.absolute()):
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink input/output ancestry')
    target = output.resolve()
    require(target.name == 'VALIDATION_CHECKS.json', 'only a new VALIDATION_CHECKS.json receipt may be written')
    require(target == DOC.resolve() / 'VALIDATION_CHECKS.json' or target.is_relative_to(PRIVATE.resolve()),
            'output must be new VALIDATION_CHECKS.json or the new v6 private subtree')
    require(not target.exists() and not target.is_symlink(), 'preserve existing check receipt')
    require(target.parent.is_dir(), 'output parent must already exist')
    require(target not in (folder.resolve(), protocol.resolve()), 'output overlaps input')
    return folder.resolve(), protocol.resolve(), target


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


class Audit:
    def __init__(self):
        self.checks = self.failure_count = 0
        self.failures = []
        self.max_difference = 0.0

    def ok(self, value, label, actual=None, expected=None):
        self.checks += 1
        if not value:
            self.failure_count += 1
            if len(self.failures) < 100:
                self.failures.append(dict(check=label, actual=safe(actual), expected=safe(expected)))

    def eq(self, actual, expected, label):
        self.ok(actual == expected, label, actual, expected)

    def close(self, actual, expected, label):
        if isinstance(expected, (list, tuple)):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
                self.ok(False, label + '.shape', actual, expected)
                return
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.close(a, e, label + '.' + str(i))
            return
        if expected is None:
            self.eq(actual, None, label + '.stored_nonfinite_or_absent')
            return
        valid = type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
        difference = abs(actual - expected) if valid else None
        if valid:
            self.max_difference = max(self.max_difference, difference)
        self.ok(valid and difference <= ATOL + RTOL * abs(expected), label, actual, expected)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def finite(value, shape):
    if not isinstance(value, list) or len(value) != shape[0]:
        return False
    if len(shape) == 1:
        return all(type(v) in (int, float) and math.isfinite(v) for v in value)
    return all(finite(v, shape[1:]) for v in value)


def valid_ids(value):
    return (isinstance(value, list) and len(set(value)) == len(value) and
            all(type(k) is int and 0 <= k < 8 for k in value))


def eligible(points, hw):
    height, width = hw
    return [k for k, p in enumerate(points[:8]) if finite(p, (2,)) and p != [-1, -1] and
            0 <= p[0] < width and 0 <= p[1] < height]


def project(dimensions, R, t, K):
    require(finite(dimensions, (3,)) and min(dimensions) > 0 and finite(R, (3, 3)) and
            finite(t, (3,)) and finite(K, (3, 3)), 'invalid stored numeric pose/camera')
    a, b, c = (v / 2 for v in dimensions)
    model = ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c),
             (-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c))
    output = []
    for p in model:
        camera = [dot(axis, p) + shift for axis, shift in zip(R, t)]
        uvw = [dot(axis, camera) for axis in K]
        require(all(math.isfinite(v) for v in uvw) and uvw[2] > 0, 'invalid saved projected depth')
        output.append([uvw[0] / uvw[2], uvw[1] / uvw[2]])
    return output


def semantic_binding(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=True).encode('utf-8')
    return dict(sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded),
                encoding='sorted compact JSON UTF-8; numpy values converted; nonfinite tokens preserved')


def native_hash(points, K, xyz, hw):
    # Finite public values can reconstruct the original float64 bytes exactly.
    # A JSON null cannot certify a particular original NaN payload: do not guess.
    if not (finite(points, (9, 2)) and finite(K, (3, 3)) and finite(xyz, (3,))):
        return None
    values = [v for p in points for v in p] + [v for row in K for v in row] + xyz
    encoded = b''.join(struct.pack('=d', float(v)) for v in values)
    encoded += str((hw[1], hw[0])).encode()
    return hashlib.sha256(encoded).hexdigest()


def ledger(before, after, delta, audit, label):
    audit.eq(set(before), set(after), label + '.keys')
    audit.eq(delta, {k: after[k] - before[k] for k in after}, label + '.delta')
    audit.ok(all(type(v) is int and v >= 0 for v in delta.values()), label + '.nonnegative')


def source_pool(row, observation, audit, label):
    U = row['solver']['used']
    adopted = [k for k in row['observation_contract']['hybrid_boundary_corner_ids'] if k in U]
    corners = {c['id']: c for c in observation['corners']}
    audit.eq(len(corners), len(observation['corners']), label + '.unique_corner_ids')
    consumed = set()
    for k in adopted:
        corner = corners[k]
        audit.eq(row['input_points'][k], corner['xy'], label + '.actual_boundary.' + str(k))
        edges = corner['edges']
        audit.ok(len(edges) == 2 and len(set(edges)) == 2 and
                 all(type(e) is int and 0 <= e < 12 and k in EDGES[e] for e in edges),
                 label + '.incident_edges.' + str(k))
        consumed.update(edges)
    seen, allowed = {}, {}
    for raw in observation['lines']:
        edge = raw['edge']
        require(type(edge) is int and 0 <= edge < 12, 'invalid semantic source edge')
        if edge in seen:
            audit.eq(raw, seen[edge], label + '.consistent_duplicate.' + str(edge))
            continue
        seen[edge] = raw
        audit.eq(raw['endpoints'], list(EDGES[edge]), label + '.edge_identity.' + str(edge))
        norm = math.hypot(*raw['normal'])
        require(math.isfinite(norm) and norm > 0 and math.isfinite(raw['offset']), 'invalid source line')
        normal, offset = [v / norm for v in raw['normal']], raw['offset'] / norm
        support, queries, radii = raw['support_points'], raw['queries'], raw['query_radii_px']
        audit.ok(len(support) == len(queries) == len(radii) and len(set(queries)) >= 3 and
                 all(type(q) is int and 0 <= q < 84 and q // 7 == edge for q in queries),
                 label + '.support_query_identity.' + str(edge))
        errors = [abs(dot(p, normal) - offset) for p in support]
        good = finite(support, (len(support), 2)) and all(type(r) in (int, float) and
                    math.isfinite(r) and r >= 0 and error <= r + 1e-9 for error, r in zip(errors, radii))
        if edge in consumed:
            audit.ok(good and edge in SOURCE_REGISTRY_EDGES, label + '.consumed_support.' + str(edge))
        if edge in SOURCE_REGISTRY_EDGES and good and edge not in consumed:
            allowed[edge] = dict(normal=normal, offset=offset, queries=queries, raw=raw)
    audit.ok(consumed <= set(seen), label + '.consumed_raw_support_exists')
    audit.eq(row['solver']['consumed_edges'], sorted(consumed), label + '.consumed')
    return allowed, sorted(consumed), adopted


def point_candidate(candidate, points, K, U, blocked, audit, counts, label, lines=None):
    q = project(candidate['dimensions'], candidate['R_cf'], candidate['centroid'], K)
    audit.close(candidate['projected'], q, label + '.projection')
    rp = [math.hypot(q[k][0] - points[k][0], q[k][1] - points[k][1]) for k in U]
    ip = [k for k, r in zip(U, rp) if r <= RADIUS]
    for key in ('generator_ids', 'actual_fit_input_ids', 'inlier_ids'):
        audit.ok(valid_ids(candidate.get(key)) and set(candidate[key]) <= set(U) and
                 not (set(candidate[key]) & blocked), label + '.active_' + key)
    audit.eq(candidate['inlier_ids'], ip, label + '.point_inliers')
    if lines is None:
        audit.close(candidate['residuals_used_px'], rp, label + '.scoring_errors')
        audit.eq(candidate['inlier_count'], len(ip), label + '.point_count')
        audit.close(candidate['truncated_sse_px2'], math.fsum(min(r * r, 64) for r in rp), label + '.truncated_sse')
        audit.close(candidate['sse_px2'], math.fsum(r * r for r in rp), label + '.sse')
    else:
        audit.eq(candidate['same_scoring_point_ids'], U, label + '.same_point_pool')
        audit.eq(candidate['same_scoring_line_edges'], sorted(lines), label + '.same_line_pool')
        signed = [[dot(q[k], lines[e]['normal']) - lines[e]['offset'] for k in EDGES[e]] for e in sorted(lines)]
        rl = [math.sqrt(math.fsum(d * d for d in ds) / 2) for ds in signed]
        il = [e for e, r in zip(sorted(lines), rl) if r <= RADIUS]
        audit.close(candidate['point_residual_norms_px'], rp, label + '.point_errors')
        audit.close(candidate['line_residual_RMS_px'], rl, label + '.line_RMS')
        audit.close(candidate['line_endpoint_signed_residuals_px'], signed, label + '.endpoint_errors')
        audit.eq(candidate['line_inlier_edges'], il, label + '.line_inliers')
        audit.eq(candidate['point_inlier_count'], len(ip), label + '.point_count')
        audit.eq(candidate['line_inlier_count'], len(il), label + '.unique_edge_count')
        audit.eq(candidate['total_inlier_factor_count'], len(ip) + len(il), label + '.factor_count')
        audit.ok(set(candidate['actual_fit_line_edges']) <= set(lines), label + '.actual_fit_edges')
        audit.close(candidate['truncated_sse_px2'], math.fsum(min(r * r, 64) for r in rp + rl), label + '.truncated_sse')
        audit.close(candidate['sse_px2'], math.fsum(r * r for r in rp + rl), label + '.sse')
    counts['stored_scalar_checked_candidates'] += 1
    return q, rp, ip


def witness_problem(solved, blocked):
    """Availability/active-ID witness, independently expressed from saved fields."""
    if solved.get('available') is not True:
        return 'UNAVAILABLE'
    if any(solved.get(k) is not v for k, v in dict(new_pose_estimated=True, pose_available=True,
            no_pose=False, fallback_used=False, unresolved_ambiguity=False).items()) or solved.get('state') != 'NEW_POSE':
        return 'INCONSISTENT_AVAILABLE_STATUS'
    if any(solved.get(k) is not False for k in NO_PRIOR_FLAGS):
        return 'MISSING_PRIOR_FREE_WITNESS'
    if not valid_ids(solved.get('excluded')) or set(solved['excluded']) != blocked:
        return 'MISSING_FULL_EXCLUSION_WITNESS'
    U = solved.get('used')
    if not valid_ids(U) or len(U) < 4 or set(U) & blocked:
        return 'MISSING_ENDPOINT_FREE_SCORING_POOL'
    for key in ('generator_ids', 'fit_input_ids', 'final_inliers'):
        value = solved.get(key)
        if not valid_ids(value) or len(value) < 4 or not set(value) <= set(U):
            return 'MISSING_SELECTED_ACTIVE_ID_WITNESS'
    candidates = solved.get('all_candidate_solutions')
    if not isinstance(candidates, list) or not candidates:
        return 'MISSING_ALL_SCORED_CANDIDATE_WITNESS'
    for c in candidates:
        if any(not valid_ids(c.get(k)) or not set(c[k]) <= set(U)
               for k in ('generator_ids', 'actual_fit_input_ids', 'inlier_ids')):
            return 'MISSING_CANDIDATE_ENDPOINT_FREE_ACTIVE_ID_WITNESS'
        if not isinstance(c.get('residuals_used_px'), list) or len(c['residuals_used_px']) != len(U):
            return 'MISSING_CANDIDATE_SCORING_POOL_WITNESS'
    if not (finite(solved.get('projected'), (8, 2)) and finite(solved.get('R_cf'), (3, 3)) and
            finite(solved.get('R_physical'), (3, 3)) and finite(solved.get('centroid'), (3,)) and
            finite(solved.get('cf_extents'), (3,))):
        return 'MISSING_NUMERIC_POSE_WITNESS'
    return None


def endpoint_checks(row, observation, original, consumed, adopted, audit, counts, reasons, label):
    solver = row['solver']
    validation = solver['endpoint_validation']
    summary = row['observation_contract']['endpoint_validation']
    audit.eq(summary.get('full_witness_reference'), 'solver.endpoint_validation', label + '.contract_packet_reference')
    for key, value in summary.items():
        if key != 'full_witness_reference':
            audit.eq(value, validation.get(key), label + '.contract_summary_join.' + key)
    audit.eq(validation['schema'], 'endpoint_validation_packet_v6', label + '.schema')
    H, T = row['hidden_initial'], solver['temporary_excluded']
    native = observation['native_N3_points']
    audit.eq(validation['native_points'], native, label + '.native_RGB_coordinates')
    audit.eq(validation['hidden'], H, label + '.actual_H')
    audit.eq(validation['temporary_excluded'], T, label + '.caller_T')
    audit.eq(validation['original_unused_edges'], sorted(original), label + '.original_unused')
    audit.eq(validation['consumed_edges'], consumed, label + '.consumed')
    audit.eq(validation['actual_adopted_boundary_ids'], adopted, label + '.actual_adopted')
    audit.eq(validation['contract_state'], 'VALID', label + '.valid_support_contract')
    for key, value in dict(observed_line_used_for_validation_pose_choice=False,
            validation_projections_added_to_observations=False, initial_pose_or_dimension_prior_used=False,
            actual_self_hidden_separate_from_endpoint_exclusion=True,
            original_corner_selection_and_consumed_support_preserved=True,
            original_observation_preserved=True).items():
        audit.eq(validation[key], value, label + '.' + key)
    raw_observation = {k: v for k, v in observation.items() if k not in OBS_WRAPPER_KEYS}
    audit.eq(validation['original_observation_binding'], semantic_binding(raw_observation), label + '.original_semantic_binding')
    audit.eq(validation['original_line_edges'], sorted({l['edge'] for l in observation['lines']}), label + '.raw_edges')
    reconstructed_hash = native_hash(native, row['K'], row['xyz'], row['raw_hw'])
    if reconstructed_hash is not None:
        audit.eq(validation['native_input_hash'], reconstructed_hash, label + '.native_float64_input_hash')
        counts['finite_native_input_hash_recomputed'] += 1
    else:
        counts['native_input_hash_not_recomputed_nonfinite_JSON'] += 1
    audit.eq(validation['final_input_hash'], solver['input_hash'], label + '.final_input_hash')
    records = validation['records']
    audit.eq([r['edge'] for r in records], sorted(original), label + '.exact_one_record_per_eligible_edge')
    rejected, supported, unverified = set(), set(), set()
    previous = validation['validation_bank_before']
    for record in records:
        edge, line = record['edge'], original[record['edge']]
        rlabel = label + '.edge.' + str(edge)
        endpoints = list(EDGES[edge])
        temporary = sorted(set(T) | set(endpoints))
        blocked = set(H) | set(temporary)
        audit.eq(record['endpoints'], endpoints, rlabel + '.endpoints')
        audit.eq(record['hidden'], H, rlabel + '.H')
        audit.eq(record['temporary_excluded'], T, rlabel + '.caller_T')
        audit.eq(record['validator_temporary_excluded'], temporary, rlabel + '.explicit_both_endpoints')
        audit.eq(record['validation_excluded'], sorted(blocked), rlabel + '.explicit_H_T_endpoints')
        audit.eq(record['attempted_pose_calls'], 1, rlabel + '.one_solve')
        audit.eq(record['validation_robust'], True, rlabel + '.robust')
        audit.eq(record['native_input_hash'], validation['native_input_hash'], rlabel + '.native_hash')
        source = record['source_line']
        audit.eq(source['raw'], line['raw'], rlabel + '.original_line_retained')
        audit.eq(source['edge'], edge, rlabel + '.source_edge')
        audit.eq(source['endpoints'], endpoints, rlabel + '.source_endpoints')
        audit.close(source['normal'], line['normal'], rlabel + '.normalized_normal')
        audit.close(source['offset'], line['offset'], rlabel + '.normalized_offset')
        audit.eq(source['support_query_ids'], line['queries'], rlabel + '.original_query_ids')
        audit.eq(record['bank_before'], previous, rlabel + '.sequential_native_bank')
        ledger(record['bank_before'], record['bank_after'], record['bank_delta'], audit, rlabel + '.ledger')
        previous = record['bank_after']
        solved = record['solved']
        audit.eq(solved['operation_counts'], record['bank_delta'], rlabel + '.actual_solve_delta')
        audit.eq(solved['hypothesis_bank_counts'], record['bank_after'], rlabel + '.actual_bank_after')
        audit.eq(solved['input_hash'], validation['native_input_hash'], rlabel + '.actual_solver_native_hash')
        audit.eq(solved['hidden'], H, rlabel + '.solver_actual_H')
        audit.eq(solved['temporary_excluded'], temporary, rlabel + '.solver_explicit_endpoint_T')
        audit.eq(solved['excluded'], sorted(blocked), rlabel + '.solver_excluded_union')
        U = [k for k in eligible(native, row['raw_hw']) if k not in blocked]
        problem = witness_problem(solved, blocked)
        if 'used' in solved:
            audit.eq(solved['used'], U, rlabel + '.native_scoring_pool')
        else:
            audit.ok(problem is not None, rlabel + '.missing_scoring_witness_is_unverified')
            counts['missing_used_witness_records'] += 1
        for flag in NO_PRIOR_FLAGS:
            if flag in solved:
                audit.eq(solved[flag], False, rlabel + '.' + flag)
            else:
                audit.ok(problem is not None, rlabel + '.missing_' + flag + '_is_unverified')
                counts['missing_prior_free_flag_witnesses'] += 1
        if 'equivalence_uses_all_eight_model_predictions' in solved:
            audit.eq(solved['equivalence_uses_all_eight_model_predictions'], True, rlabel + '.modeled_equivalence_scope')
        audit.eq(record['global_cached_generation_may_include_blocked_IDs'], True, rlabel + '.global_cache_scope')
        audit.eq(record['active_scoring_generators_refits_exclude_blocked_IDs'], True, rlabel + '.active_validation_scope')
        for key in ('used', 'generator_ids', 'fit_input_ids', 'final_inliers'):
            if key in solved:
                value = solved[key]
                audit.ok(valid_ids(value) and not set(value) & blocked and set(value) <= set(U), rlabel + '.active_' + key)
            else:
                audit.ok(problem is not None, rlabel + '.missing_active_' + key + '_is_unverified')
                counts['missing_selected_ID_witnesses'] += 1
        candidates = solved.get('all_candidate_solutions')
        if isinstance(candidates, list):
            for i, candidate in enumerate(candidates):
                keys = ('generator_ids', 'actual_fit_input_ids', 'inlier_ids', 'residuals_used_px')
                if all(k in candidate for k in keys):
                    point_candidate(candidate, native, row['K'], U, blocked, audit, counts, rlabel + '.candidate.' + str(i))
                else:
                    audit.ok(problem is not None, rlabel + '.missing_active_candidate_witness_is_unverified')
                    counts['missing_active_candidate_witnesses'] += 1
        else:
            audit.ok(problem is not None, rlabel + '.missing_candidate_pool_is_unverified')
            counts['missing_all_candidate_witness_records'] += 1
        if problem is None:
            q = project(solved['cf_extents'], solved['R_cf'], solved['centroid'], row['K'])
            audit.close(solved['projected'], q, rlabel + '.validation_Rt_projection')
            signed = [dot(q[k], line['normal']) - line['offset'] for k in endpoints]
            rms = math.sqrt(math.fsum(d * d for d in signed) / 2)
            audit.close(record['signed_endpoint_distances_px'], signed, rlabel + '.endpoint_signed_distance')
            audit.close(record['endpoint_RMS_px'], rms, rlabel + '.endpoint_RMS')
            decision = 'SUPPORTED_RETAIN' if rms <= RADIUS else 'CONTRADICTED_REJECT'
            if decision == 'SUPPORTED_RETAIN':
                supported.add(edge)
            else:
                rejected.add(edge)
            audit.ok(solved['pose_available'] and solved['available'], rlabel + '.validated_new_pose')
            audit.eq(solved['reprojected_points_reused_as_observations'], False, rlabel + '.validation_no_refit')
            for k in H:
                audit.close(solved['points_final'][k], q[k], rlabel + '.validator_H_output.' + str(k))
            for k in set(temporary) - set(H):
                audit.eq(solved['points_final'][k], native[k], rlabel + '.heldout_coordinate_not_projection.' + str(k))
            audit.eq(solved['points_final'][8], native[8], rlabel + '.validator_center')
        else:
            decision = 'UNVERIFIED_RETAIN'
            unverified.add(edge)
            audit.eq(record['endpoint_RMS_px'], None, rlabel + '.unverified_no_RMS')
            audit.eq(record['signed_endpoint_distances_px'], None, rlabel + '.unverified_no_endpoint_errors')
            reasons[record['reason']] += 1
        audit.eq(record['decision'], decision, rlabel + '.fixed8_decision')
        counts['endpoint_decision_' + decision] += 1
        counts['explicit_endpoint_LOO_records'] += 1
    audit.eq(previous, validation['validation_bank_after'], label + '.validation_interval_end')
    ledger(validation['validation_bank_before'], validation['validation_bank_after'],
           validation['validation_bank_delta'], audit, label + '.validation_interval')
    audit.eq(validation['attempted_pose_calls'], len(original), label + '.attempted_count')
    available = sum(r['solved'].get('available') is True for r in records)
    audit.eq(validation['available_pose_calls'], available, label + '.available_pose_count')
    audit.eq(validation['unavailable_pose_calls'], len(records) - available, label + '.unavailable_pose_count')
    audit.eq(validation['checkable_pose_calls'], len(supported) + len(rejected), label + '.checkable_pose_count')
    audit.eq(validation['unverified_line_decisions'], len(unverified), label + '.unverified_line_count')
    audit.eq(validation['rejected_edges'], sorted(rejected), label + '.rejected')
    audit.eq(validation['retained_edges'], sorted(set(original) - rejected), label + '.retained_including_unverified')
    audit.ok(not (set(consumed) & rejected), label + '.consumed_not_removed')
    filtered = dict(raw_observation)
    filtered['lines'] = [l for l in raw_observation['lines'] if l['edge'] not in rejected]
    audit.eq(validation['filtered_observation_binding'], semantic_binding(filtered), label + '.only_rejected_edges_removed')
    audit.eq(validation['filtered_raw_line_edges'], sorted({l['edge'] for l in filtered['lines']}), label + '.filtered_raw_edges')
    shared = validation['native_final_bank_shared']
    aliases = dict(native='native', final='native' if shared else 'final')
    names = {'native'} if shared else {'native', 'final'}
    audit.eq(validation['bank_aliases'], aliases, label + '.bank_aliases')
    audit.eq(validation['unique_numeric_bank_count'], len(names), label + '.unique_bank_count')
    audit.eq(shared, validation['native_input_hash'] == validation['final_input_hash'], label + '.identical_coordinates_bank_reuse')
    for key in ('bank_before', 'bank_after', 'bank_delta'):
        audit.eq(set(validation[key]), names, label + '.' + key + '.unique_banks')
    for name in names:
        ledger(validation['bank_before'][name], validation['bank_after'][name], validation['bank_delta'][name], audit, label + '.bank.' + name)
        points = native if name == 'native' else row['input_points']
        dimension_count = 1 if abs(row['xyz'][0] - row['xyz'][2]) < 1e-9 else 2
        cap = dimension_count * math.comb(len(eligible(points, row['raw_hw'])), 4)
        for key in ('subsets_considered', 'subset_generic_calls'):
            audit.ok(validation['bank_after'][name][key] <= cap, label + '.cached_initial_four_subset_cap.' + name + '.' + key)
    audit.eq(validation['validation_bank_before'], validation['bank_before']['native'], label + '.validation_initial_bank')
    final_alias = aliases['final']
    audit.eq(validation['final_bank_before'], validation['validation_bank_after'] if shared else validation['bank_before']['final'], label + '.disjoint_final_interval')
    ledger(validation['final_bank_before'], validation['final_bank_after'], validation['final_bank_delta'], audit, label + '.final_interval')
    audit.eq(validation['final_bank_after'], validation['bank_after'][final_alias], label + '.final_bank_end')
    audit.eq(validation['validation_bank_after'], validation['bank_after']['native'] if not shared else validation['final_bank_before'], label + '.validation_native_end')
    final_counts = solver['final_only_operation_counts']
    aggregate = {k: sum(validation['bank_delta'][n][k] for n in names) for k in validation['validation_bank_delta']}
    aggregate.update({k: v for k, v in final_counts.items() if k not in aggregate})
    audit.eq(solver['operation_counts'], aggregate, label + '.distinct_banks_counted_once')
    for k, value in validation['final_bank_delta'].items():
        audit.eq(final_counts[k], value, label + '.final_only_count.' + k)
    counts['unique_numeric_bank_intervals'] += len(names)
    return {e: line for e, line in original.items() if e not in rejected}


def final_checks(row, observation, lines, audit, counts, label):
    solved, H = row['solver'], row['hidden_initial']
    blocked = set(H) | set(solved['temporary_excluded'])
    U = [k for k in eligible(row['input_points'], row['raw_hw']) if k not in blocked]
    audit.eq(solved['used'], U, label + '.actual_scoring_pool')
    audit.eq(solved['excluded'], sorted(blocked), label + '.final_excluded_union')
    audit.eq(solved['hidden'], H, label + '.final_H')
    for flag in NO_PRIOR_FLAGS:
        audit.eq(solved[flag], False, label + '.' + flag)
    for key in ('used', 'generator_ids', 'fit_input_ids', 'final_inliers'):
        value = solved.get(key, [])
        audit.ok(valid_ids(value) and set(value) <= set(U) and not set(value) & blocked, label + '.active_' + key)
    for i, candidate in enumerate(solved['all_candidate_solutions']):
        point_candidate(candidate, row['input_points'], row['K'], U, blocked, audit, counts,
                        label + '.candidate.' + str(i), lines if lines else None)
    if lines is not None:
        audit.eq(solved['line_edges'], sorted(lines), label + '.final_C2_line_pool')
        audit.eq(solved['factor_pool'], dict(point_ids=U, line_edges=sorted(lines), actual_point_factors=len(U),
                 unique_line_factors=len(lines), line_support_queries=sum(len(l['queries']) for l in lines.values()),
                 scalar_residuals=2 * (len(U) + len(lines))), label + '.one_edge_one_factor')
        audit.ok(not (set(solved['consumed_edges']) & set(lines)), label + '.no_point_line_duplicate')
        for attempt in solved.get('optimizer_attempts', []):
            audit.ok(set(attempt['fit_point_ids']) <= set(U) and len(attempt['fit_point_ids']) >= 4,
                     label + '.optimizer_actual_point_pool')
            audit.ok(set(attempt['fit_line_edges']) <= set(lines), label + '.optimizer_edge_pool')
            audit.eq(attempt['initial_pose_used'], False, label + '.optimizer_no_initial_prior')
        if not lines:
            audit.eq(solved['empty_line_pool_exact_point_delegate'], True, label + '.empty_point_delegate')
            audit.eq(solved.get('selected_pose_refined_with_lines', False), False, label + '.empty_no_line_refinement')
            audit.eq(solved['optimizer_attempts'], [], label + '.empty_no_optimizer')
            audit.eq(solved.get('final_line_inliers', []), [], label + '.empty_no_line_inlier')
        counts['C2_selected_unique_edge_factors'] += len(lines)
        counts['C2_support_queries'] += sum(len(l['queries']) for l in lines.values())
    new = solved['available']
    audit.eq(row['new_pose_estimated'], new, label + '.new_pose_status')
    audit.eq(row['actual_pose']['available'], row['pose_available'], label + '.operational_pose_object')
    audit.ok(not new or row['pose_available'], label + '.NEW_requires_operational_pose')
    audit.eq(row['reprojections_reused_as_observations'], False, label + '.output_not_refit')
    audit.eq(row['native_points'][8], observation['native_N3_points'][8], label + '.center_preserved')
    audit.eq(solved['global_uniqueness_proven'], False, label + '.no_global_uniqueness_claim')
    if new:
        q = project(solved['cf_extents'], solved['R_cf'], solved['centroid'], row['K'])
        audit.close(solved['projected'], q, label + '.final_Rt_projection')
        audit.ok(len(solved['final_inliers']) >= 4 and solved['unresolved_ambiguity'] is False,
                 label + '.accepted_four_point_consensus_no_unresolved_tie')
        for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
            audit.eq(row['actual_pose'][key], solved[key], label + '.actual_pose_' + key)
        expected = [list(p) for p in observation['native_N3_points']]
        for k in range(8):
            p = row['input_points'][k]
            if k not in H and finite(p, (2,)) and p != [-1, -1]:
                expected[k] = p
        for k in H:
            expected[k] = q[k]
        audit.close(row['native_points'], expected, label + '.actual_H_only_projection_and_observed_output')
        audit.eq(row['reprojected_ids'], H, label + '.H_reprojected_ids')
        counts['accepted_NEW_rows'] += 1
        counts['accepted_H_reprojected_corners'] += len(H)
        if lines:
            best = solved['best_candidate']
            _, _, ip = point_candidate(best, row['input_points'], row['K'], U, blocked,
                                       audit, Counter(), label + '.accepted_best', lines)
            audit.eq(solved['final_inliers'], ip, label + '.accepted_point_inlier_join')
            audit.eq(solved['final_line_inliers'], best['line_inlier_edges'], label + '.accepted_edge_inlier_join')
            audit.close(solved['truncated_sse_px2'], best['truncated_sse_px2'], label + '.accepted_group_SSE')
            audit.close(solved['line_residual_RMS_px'], best['line_residual_RMS_px'], label + '.accepted_line_errors')
            audit.eq(solved['point_inlier_count'], best['point_inlier_count'], label + '.accepted_actual_point_count')
            audit.eq(solved['unique_line_inlier_count'], best['line_inlier_count'], label + '.accepted_unique_edge_count')
            audit.eq(solved['total_inlier_factor_count'], best['total_inlier_factor_count'], label + '.accepted_total_factor_count')
    else:
        audit.eq(row['native_points'], observation['native_N3_points'], label + '.full_N3_fallback')
        audit.eq(row['reprojected_ids'], [], label + '.fallback_no_new_projection')
        counts['fallback_or_failed_rows'] += 1
    geometry = solved.get('point_line_geometry')
    if geometry:
        for key in ('observed_normal_joint', 'modeled_normal_joint', 'point_only',
                    'observed_normal_lines_only', 'modeled_normal_lines_only'):
            counts['stored_raw_' + key + '_rank' + str(geometry[key]['numerical_rank'])] += 1
        audit.eq(geometry['modeled_normals_held_fixed_for_derivative'], True, label + '.modeled_normal_rank_scope')
        audit.eq(geometry['global_unique_pose_proven'], False, label + '.local_rank_only')
        if new:
            audit.eq(geometry['observed_normal_joint']['numerical_rank'], 6, label + '.accepted_observed_rank_label')
            audit.eq(geometry['modeled_normal_joint']['numerical_rank'], 6, label + '.accepted_modeled_rank_label')
            counts['accepted_stored_local_rank6_rows'] += 1


def run(folder, protocol_path, output):
    folder, protocol_path, output = guard(folder, protocol_path, output)
    names = ('OBSERVATIONS.jsonl.gz', 'GEOMETRY_SEALED.jsonl.gz', 'PREDICTIONS.jsonl.gz',
             'FIXED_GEOMETRY_SEALED.jsonl.gz', 'GEOMETRY_SEAL.json', 'SCORING_RECEIPT.json')
    paths = {n: folder / n for n in names}
    paths['PROTOCOL.json'] = protocol_path
    before = {n: binding(p) for n, p in paths.items()}
    before['checker_code'] = binding(Path(__file__))
    audit, counts, reasons = Audit(), Counter(), Counter()
    result = dict(schema='public_only_v6_endpoint_validation_scalar_checks_v1', complete=False, passed=False,
        inputs=before, arithmetic_absolute_tolerance=ATOL, arithmetic_relative_tolerance=RTOL,
        fixed_endpoint_RMS_threshold_px=RADIUS, support_consistency_tolerance_px=1e-9,
        model_GT_image_PnP_optimizer_ray_training_RGB_calls=0,
        independent_SVD_Jacobian_recomputation=False, real_physical_boundary_ownership_certified=False,
        fully_statistically_independent=False,
        limits=['Checks saved scalar geometry and explicit active-ID exclusions, not a new perturbation experiment.',
                'Global lazy cached generators can include excluded IDs; only active scored/generator/refit IDs are exclusions witnesses.',
                'Native N3 and IMAGE_ROLE proposals/H remain conditionally dependent on the frozen image pipeline.',
                'No GT authenticity, real edge ownership, global pose uniqueness or runtime causality is independently certified.',
                'Local rank labels are counted from stored records; no Jacobian/SVD or model execution is replayed.',
                'Nonfinite JSON nulls do not certify original float64 NaN payloads; finite native input hashes are reconstructed.'])
    start = time.perf_counter()
    try:
        protocol, seal, scoring = read(protocol_path), read(paths['GEOMETRY_SEAL.json']), read(paths['SCORING_RECEIPT.json'])
        audit.eq(protocol['methods'], list(METHODS), 'fixed_methods')
        audit.eq(protocol['frames'], 245, 'fixed_scope')
        audit.ok(seal['complete'] and not seal['GT_read_allowed'] and seal['frames'] == 245 and
                 seal['rows'] == 980 and seal['fixed_rows'] == 490, 'complete_observation_only_seal')
        audit.ok(scoring['complete'] and scoring['scored_method_rows'] == 980 and scoring['new_pose_fits'] == 0,
                 'scoring_after_seal_without_new_fits')
        bound(protocol_path, seal['protocol'])
        bound(paths['GEOMETRY_SEAL.json'], scoring['geometry_seal'])
        for name, field in (('GEOMETRY_SEALED.jsonl.gz', 'geometry'), ('OBSERVATIONS.jsonl.gz', 'observations'),
                            ('FIXED_GEOMETRY_SEALED.jsonl.gz', 'fixed_geometry')):
            bound(paths[name], seal[field])
        bound(paths['PREDICTIONS.jsonl.gz'], scoring['predictions'])
        observations, geometry, predictions, controls = [rows(paths[n]) for n in names[:4]]
        obs = {r['id']: r for r in observations}
        sealed = {(r['method'], r['id']): r for r in geometry}
        scored = {(r['method'], r['id']): r for r in predictions}
        fixed = {(r['method'], r['id']): r for r in controls}
        audit.eq(len(observations), 245, 'observation_count')
        audit.eq(len(obs), 245, 'observation_unique')
        audit.eq(len(geometry), 980, 'geometry_count')
        audit.eq(len(sealed), 980, 'geometry_unique')
        audit.eq(len(predictions), 980, 'prediction_count')
        audit.eq(set(scored), set(sealed), 'scored_sealed_population')
        audit.eq(len(controls), 490, 'fixed_count')
        audit.eq(len(fixed), 490, 'fixed_unique')
        for method in METHODS:
            audit.eq({r['id'] for r in geometry if r['method'] == method}, set(obs), 'all_cohort_ids.' + method)
        for method in ('BASE', 'N3_SUBPIX'):
            audit.eq({r['id'] for r in controls if r['method'] == method}, set(obs), 'fixed_all_cohort.' + method)
        for row in geometry:
            label = row['method'] + '/' + row['id']
            for key, value in row.items():
                audit.eq(scored[(row['method'], row['id'])].get(key), value, label + '.sealed_scored_join.' + key)
            observation = obs[row['id']]
            audit.eq(observation['GT_input'], False, label + '.observation_GT_free')
            audit.eq(observation['native_N3_points'], fixed[('N3_SUBPIX', row['id'])]['native_points'], label + '.fixed_N3_join')
            expected_H = [] if row['method'] == 'N3_INDEPENDENT_ROBUST_NO_MASK' else observation['predicted_N3_hidden']
            audit.eq(row['hidden_initial'], expected_H, label + '.same_frozen_H_or_no_mask')
            lines = None
            if row['method'] in (PRIMARY, CONTROL):
                original, consumed, adopted = source_pool(row, observation, audit, label)
                lines = endpoint_checks(row, observation, original, consumed, adopted, audit, counts, reasons, label) if row['method'] == PRIMARY else original
                if row['method'] == PRIMARY:
                    control = sealed[(CONTROL, row['id'])]
                    audit.eq(row['input_points'], control['input_points'], label + '.unchanged_cornerwise_selection')
                    audit.eq(row['selected_corner_ids'], control['selected_corner_ids'], label + '.unchanged_adopted_corners')
                    audit.eq(row['cornerwise_selection'], control['cornerwise_selection'], label + '.same_point_only_LOO')
                else:
                    audit.ok('endpoint_validation' not in row['solver'], label + '.control_has_no_endpoint_filter')
            final_checks(row, observation, lines, audit, counts, label)
            counts['method_rows_' + row['method']] += 1
        for name, path in paths.items():
            audit.eq(binding(path), before[name], 'input_preserved.' + name)
        audit.eq(binding(Path(__file__)), before['checker_code'], 'checker_preserved')
        result['complete'] = True
    except Exception as error:
        audit.ok(False, 'exception', dict(type=type(error).__name__, message=str(error)))
    result.update(passed=result['complete'] and audit.failure_count == 0, checks=audit.checks,
        failure_count=audit.failure_count, failures=audit.failures,
        actual_saved_arithmetic_counts=dict(counts), unverified_reason_counts=dict(reasons),
        maximum_numeric_absolute_difference=audit.max_difference,
        saved_arithmetic_wall_seconds=time.perf_counter() - start,
        Python_version=sys.version, actual_checker_runs_this_invocation=1)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(safe(result), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=result['passed'], complete=result['complete'], checks=audit.checks,
                          failures=audit.failure_count, output=str(output)), ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DOC, help='complete saved v6 result folder')
    parser.add_argument('--protocol', type=Path, default=DOC / 'PROTOCOL.json', help='frozen v6 protocol')
    parser.add_argument('--output', type=Path, default=DOC / 'VALIDATION_CHECKS.json', help='new exclusive receipt file')
    args = parser.parse_args()
    run(args.input, args.protocol, args.output)


if __name__ == '__main__':
    main()
