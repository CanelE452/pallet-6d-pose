"""Standalone scalar checks for the three fixed V8 sparse ROLE controls.

Only the standard library is imported. Freeze completed GT-free geometry and
this code before its own arithmetic, then run once. No model, GT, image, PnP,
optimizer, ray, production module, Jacobian or SVD is called. Ordinary PnP is
checked against whole-U SSE/rank, robust against consensus/inlier rank. Every
stored final candidate witness is checked; chosen rechecks are counted apart.
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
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_same_observation_controls_20261010_v8'
PARENT = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE = Path('/tmp/pallet-same-observation-controls-private-20261010-v8')
COHORT = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
METHODS = ('ROLE_BOUNDARY_H_ROBUST', 'ROLE_BOUNDARY_H_STANDARD', 'ROLE_BOUNDARY_NO_MASK_ROBUST')
PARENT_METHODS = ('IMAGE_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_BOUNDARY_ONLY',
    'IMAGE_NO_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_CORNERWISE_HYBRID',
    'IMAGE_NO_ROLE_CORNERWISE_HYBRID', 'IMAGE_ROLE_CORNERWISE_HYBRID',
    'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK')
HEADS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
CHECKPOINT = '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d'
FALSE_FLAGS = ('prior_used', 'initial_pose_used', 'initial_projection_used',
    'initial_dimension_prior_used', 'known_dimension_constraint_used',
    'excluded_image_coordinates_used_for_scoring', 'excluded_image_coordinates_used_for_equivalence',
    'reprojected_points_reused_as_observations')
TRUTH_FIELDS = {'corner', 'pose', 'mask_audit', 'baseline_corner', 'baseline_pose',
    'GT', 'ground_truth', 'reference_pose', 'human_states_native'}
OUTPUTS = ('VALIDATION_PROTOCOL.json', 'VALIDATION_STARTED.json', 'VALIDATION_CHECKS.json')
ATOL, RTOL = 1e-7, 1e-12
RADIUS, TIE = 8.0, 1e-8
LEDGER_KEYS = ('generic_calls', 'subset_generic_calls', 'standard_generic_calls', 'refit_generic_calls',
    'lm_calls', 'subsets_considered', 'returned_solutions', 'invalid_solutions', 'generic_errors',
    'geometry_rejected', 'generic_cache_hits', 'generic_cache_misses')
PRIMITIVES = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix')
LIMITS = dict(new_model_GT_image_PnP_optimizer_ray_training_Jacobian_SVD_calls=0,
    supplemental_before_own_arithmetic=True, existing_parent_accuracy_already_known=True,
    fresh_unseen_accuracy_policy=False, stored_singular_witness_recount_only=True,
    local_rank6_is_not_global_uniqueness=True,
    numeric_hash='Reconstruct finite float64 inputs only. JSON null cannot certify original NaN bit payload.',
    candidate_count='Stored records per solve packet, with chosen rechecks separately; not unique physical poses.',
    consensus_border='Stored residuals are checked independently within tolerance, then their exact <=8px membership is recounted.',
    cache_check='Same q/K/xyz and byte-bound solver, sequential nonnegative bank deltas, once-only full four-ID generation, final ledger join; no new numerical generator invocation.',
    ordinary='Whole-U SSE and J support U; <=8px inliers diagnostic, fewer than4 can be NEW.',
    robust='Descending inlier count then truncated SSE; at least4 final inliers and J support inliers.',
    ambiguous='Ordinary tied SSE ignores inlier-count differences; robust ties also require equal inlier count. Distinct model projection/physical R/t means unavailable.',
    self_hidden='Applied H excluded from generator/fit/scoring; NEW replaces only H with final R,t projection, no refit. No-mask is explicitly H=[] and retains original H separately.',
    hardware_snapshot_scope='Recorded quiet/temperature and protection snapshots only; no independent historical hardware authenticity or guarantee against interference between probes.',
    no_performance_based_threshold_changes=True, no_automatic_retry=True)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'empty stored row: ' + Path(path).name)
            yield json.loads(line)


def binding(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink input')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    absolute = path.resolve()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
                sha256=digest.hexdigest(), bytes=path.stat().st_size)


def same_binding(left, right):
    return all(left.get(key) == right.get(key) for key in ('sha256', 'bytes'))


def safe(value):
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted([safe(item) for item in value], key=lambda item: json.dumps(item, sort_keys=True, allow_nan=False))
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()


def guard(args, stage):
    for path in (args.input.absolute(), args.output.absolute(), args.parent.absolute(), args.cohort.absolute(), args.protocol.absolute()):
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink input/output ancestry')
    require(args.input.is_dir() and args.output.is_dir() and args.parent.is_dir(), 'existing directories required')
    output = args.output.resolve()
    require(output == DOC.resolve() or output.is_relative_to(PRIVATE.resolve()), 'only new V8 DOC/private output')
    for name in OUTPUTS if stage == 'freeze' else OUTPUTS[1:]:
        require(not (output / name).exists() and not (output / name).is_symlink(), 'preserve first/existing ' + name)
    if stage == 'run':
        require((output / OUTPUTS[0]).is_file(), 'own freeze required')
    return args.input.resolve(), output


def input_paths(args):
    folder, parent = args.input.resolve(), args.parent.resolve()
    paths = {name: folder / name for name in ('GEOMETRY_SEAL.json', 'GEOMETRY_SEALED.jsonl.gz',
        'CONTROL_LEDGERS.jsonl.gz', 'CONTROL_RECEIPT.json', 'PARENT_POPULATION_CHECKS.json', 'CONTROL_STARTED.json', 'PREFLIGHT.json')}
    paths['PROTOCOL.json'] = args.protocol.resolve()
    for name in ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'GEOMETRY_SEALED.jsonl.gz', 'OBSERVATIONS.jsonl.gz',
                 'INFERENCE_RECEIPT.json', 'FIXED_GEOMETRY_SEALED.jsonl.gz', 'BASE_N3_PARITY.json'):
        paths['parent:' + name] = parent / name
    paths.update(checker_code=Path(__file__).resolve(), controls_code=CODE / 'controls.py',
        runner_code=CODE / 'runner.py', common_code=CODE / 'common.py', cohort=args.cohort.resolve(),
        V4_pose=REPO / 'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py',
        V1_solver=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
        V2_assemble=REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2/pipeline.py')
    for index, _ in enumerate(read(folder / 'CONTROL_RECEIPT.json')['resource_snapshots']):
        paths['resource:%03d' % index] = folder / ('RESOURCE_%03d.json' % index)
    return paths


def core_bindings(inputs, core):
    names = dict(checker_code='validation_checks.py', controls_code='controls.py', runner_code='runner.py',
                 common_code='common.py', V4_pose='v4_pose', V1_solver='v1_solver', V2_assemble='v2_assemble', cohort='cohort')
    names.update({key: key for key in inputs if key.startswith('parent:')})
    for key, core_key in names.items():
        require(same_binding(inputs[key], core['inputs'][core_key]), 'frozen V8 code/parent authority differs: ' + key)


def freeze(args):
    folder, output = guard(args, 'freeze')
    paths = input_paths(args)
    inputs = {key: binding(path) for key, path in paths.items()}
    seal, receipt, core, parent_seal, parent_receipt = [read(paths[key]) for key in
        ('GEOMETRY_SEAL.json', 'CONTROL_RECEIPT.json', 'PROTOCOL.json', 'parent:GEOMETRY_SEAL.json', 'parent:INFERENCE_RECEIPT.json')]
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
            seal['rows'] == 735 and seal['methods'] == list(METHODS), 'complete fixed GT-free V8 geometry required')
    require(receipt['complete'] is True and receipt['cleanup_error'] is None and receipt['error'] is None,
            'successful control cleanup required before supplemental freeze')
    require(core['methods'] == list(METHODS) and core['frames'] == 245, 'fixed V8 policy differs')
    core_bindings(inputs, core)
    require(parent_seal['complete'] is True and parent_seal['GT_read_allowed'] is False and parent_seal['frames'] == 245 and
            parent_seal['rows'] == 1960 and parent_seal['fixed_rows'] == 490 and parent_seal['methods'] == list(PARENT_METHODS),
            'complete fixed GT-free parent required')
    require(parent_receipt['complete'] is True and parent_receipt['cleanup_error'] is None and
            parent_receipt['inference_error'] is None, 'parent cleanup required')
    for key, expected in (('PROTOCOL.json', seal['protocol']), ('GEOMETRY_SEALED.jsonl.gz', seal['geometry']),
            ('CONTROL_LEDGERS.jsonl.gz', seal['ledgers']), ('CONTROL_RECEIPT.json', seal['control_receipt']),
            ('PARENT_POPULATION_CHECKS.json', seal['parent_population_checks']),
            ('parent:PROTOCOL.json', parent_seal['protocol']), ('parent:GEOMETRY_SEALED.jsonl.gz', parent_seal['geometry']),
            ('parent:OBSERVATIONS.jsonl.gz', parent_seal['observations']),
            ('parent:FIXED_GEOMETRY_SEALED.jsonl.gz', parent_seal['fixed_geometry']),
            ('parent:BASE_N3_PARITY.json', parent_seal['parity']), ('PROTOCOL.json', receipt['protocol']),
            ('PARENT_POPULATION_CHECKS.json', core['parent_population_checks'])):
        require(same_binding(inputs[key], expected), 'seal input binding differs: ' + key)
    write_new(output / OUTPUTS[0], dict(schema='supplemental_same_sparse_ROLE_scalar_protocol_v8',
        checker=inputs['checker_code'], inputs=inputs, limits=LIMITS, methods=list(METHODS), frames=245, rows=735,
        scalar_absolute_tolerance=ATOL, scalar_relative_tolerance=RTOL,
        frozen_after_complete_GT_free_control_geometry=True,
        thresholds=dict(consensus_px=RADIUS, objective_tie_px2=TIE, equivalent_projection_px=1e-5,
            equivalent_physical_rotation=1e-5, equivalent_translation_relative_diagonal=1e-5, stored_rank_relative=1e-10)))
    print('SAME_ROLE_SCALAR_FROZEN', binding(output / OUTPUTS[0])['sha256'], flush=True)


class Audit:
    def __init__(self):
        self.checks = 0
        self.failure_count = 0
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

    def near(self, actual, expected, label):
        if isinstance(expected, dict):
            valid = isinstance(actual, dict) and set(actual) == set(expected)
            self.ok(valid, label + '.fields', actual, expected)
            if valid:
                for key in expected:
                    self.near(actual[key], expected[key], label + '.' + key)
        elif isinstance(expected, (list, tuple)):
            valid = isinstance(actual, (list, tuple)) and len(actual) == len(expected)
            self.ok(valid, label + '.shape', actual, expected)
            if valid:
                for index, (a, e) in enumerate(zip(actual, expected)):
                    self.near(a, e, label + '.' + str(index))
        elif type(expected) in (int, float):
            valid = type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected)
            difference = abs(actual - expected) if valid else None
            if difference is not None:
                self.max_difference = max(self.max_difference, difference)
            self.ok(valid and difference <= ATOL + RTOL * abs(expected), label, actual, expected)
        else:
            self.eq(actual, expected, label)


def finite(value, shape):
    if not shape:
        return type(value) in (int, float) and math.isfinite(value)
    return isinstance(value, list) and len(value) == shape[0] and all(finite(v, shape[1:]) for v in value)


def ids(value):
    return isinstance(value, list) and all(type(k) is int and 0 <= k < 8 for k in value) and value == sorted(set(value))


def eligible(points, hw):
    h, w = hw
    return [k for k, p in enumerate(points[:8]) if finite(p, (2,)) and p != [-1, -1] and 0 <= p[0] < w and 0 <= p[1] < h]


def dimensions(xyz):
    return [xyz] if abs(xyz[0] - xyz[2]) < 1e-9 else [xyz, [xyz[2], xyz[1], xyz[0]]]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def model(dim):
    a, b, c = [v / 2 for v in dim]
    return ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c),
            (-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c))


def project(dim, R, t, K):
    require(finite(dim, (3,)) and min(dim) > 0 and finite(R, (3, 3)) and finite(t, (3,)) and finite(K, (3, 3)),
            'invalid candidate pose/camera')
    result = []
    for point in model(dim):
        camera = [dot(axis, point) + offset for axis, offset in zip(R, t)]
        q = [dot(axis, camera) for axis in K]
        require(camera[2] > 1e-9 and q[2] > 0 and all(math.isfinite(v) for v in q), 'invalid candidate model depth')
        result.append([q[0] / q[2], q[1] / q[2]])
    return result


def determinant(R):
    a, b, c = R
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])


def physical_rotation(R, dim):
    return R if dim == 0 else [[-row[2], row[1], row[0]] for row in R]


def equivalent(a, b, xyz):
    return (max(abs(x - y) for row, other in zip(a['projected'], b['projected']) for x, y in zip(row, other)) < 1e-5 and
            max(abs(x - y) for row, other in zip(a['R_physical'], b['R_physical']) for x, y in zip(row, other)) < 1e-5 and
            math.dist(a['centroid'], b['centroid']) / math.sqrt(dot(xyz, xyz)) < 1e-5)


def native_hash(points, K, xyz, hw):
    if not (finite(points, (9, 2)) and finite(K, (3, 3)) and finite(xyz, (3,))):
        return None
    flat = [v for point in points for v in point] + [v for axis in K for v in axis] + xyz
    return hashlib.sha256(b''.join(struct.pack('=d', float(v)) for v in flat) + str((hw[1], hw[0])).encode()).hexdigest()


def semantic_sha(value):
    # JSON-streamed data already has the source's finite/NaN->null mapping.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stored_rank(witness, length, audit, label):
    require(isinstance(witness, dict), 'missing stored rank witness: ' + label)
    singular = witness['singular_values']
    audit.ok(finite(singular, (length,)) and all(v >= 0 for v in singular), label + '.required_finite_singular_length')
    require(finite(singular, (length,)) and length > 0, 'malformed stored singular witness')
    audit.ok(all(a >= b for a, b in zip(singular, singular[1:])), label + '.descending')
    rank = sum(value > singular[0] * 1e-10 for value in singular)
    audit.eq(witness['numerical_rank'], rank, label + '.rank_recount')
    if 'relative_singular_values' in witness:
        audit.near(witness['relative_singular_values'], [v / max(singular[0], 1e-300) for v in singular], label + '.relative_values')
    if 'normalization' in witness:
        audit.eq(witness['normalization'], 'unit L2 Jacobian columns, rotation and translation', label + '.J_normalization')
        audit.near(witness['condition_number'], singular[0] / singular[-1] if singular[-1] > 0 else None, label + '.condition_number')
        audit.eq(witness['weak_condition'], singular[-1] < singular[0] * 1e-4, label + '.weak_condition')
    return rank


def ledger(snapshot, audit, label, eligible_count, dimension_count):
    audit.eq(set(snapshot), set(LEDGER_KEYS), label + '.exact_counter_fields')
    require(all(type(value) is int and value >= 0 for value in snapshot.values()), 'invalid bank operation counters')
    audit.eq(snapshot['generic_calls'], sum(snapshot[key] for key in ('subset_generic_calls', 'standard_generic_calls', 'refit_generic_calls')),
             label + '.generic_phase_sum')
    audit.eq(snapshot['generic_cache_misses'], snapshot['generic_calls'] + snapshot['geometry_rejected'], label + '.miss_accounting')
    cap = dimension_count * math.comb(eligible_count, 4) if eligible_count >= 4 else 0
    audit.ok(snapshot['subsets_considered'] in (0, cap), label + '.lazy_full_bank_once', snapshot['subsets_considered'], (0, cap))
    audit.ok(snapshot['subset_generic_calls'] <= snapshot['subsets_considered'] <= 70 * dimension_count,
             label + '.maximum70_four_ID_subsets_per_dimension')
    audit.ok(snapshot['invalid_solutions'] <= snapshot['returned_solutions'], label + '.invalid_not_more_than_returned')
    audit.ok(snapshot['generic_errors'] <= snapshot['generic_calls'], label + '.errors_not_more_than_calls')


def candidate(c, points, K, xyz, U, H, robust, audit, counts, label, scope):
    counts['candidate_scalar_arithmetic_invocations'] += 1
    counts[scope + '_arithmetic_invocations'] += 1
    d, dims = c['dimension_index'], dimensions(xyz)
    audit.ok(type(d) is int and 0 <= d < len(dims), label + '.registered_dimension')
    require(type(d) is int and 0 <= d < len(dims), 'invalid candidate dimension')
    audit.eq(c['dimensions'], dims[d], label + '.dimension_values')
    audit.eq(c['selected_hypothesis'], ('REGISTRY_WD', 'REGISTRY_DW')[d], label + '.dimension_name')
    audit.eq(c['dimension_constraint_allowed'], True, label + '.no_dimension_filter')
    for key in ('generator_ids', 'actual_fit_input_ids', 'inlier_ids'):
        audit.ok(ids(c[key]) and set(c[key]).issubset(U) and not set(c[key]).intersection(H), label + '.' + key + '.only_U')
    audit.ok(len(c['generator_ids']) >= 4 and len(c['actual_fit_input_ids']) >= 4, label + '.minimum4_fit')
    if not robust:
        audit.eq(c['generator_ids'], U, label + '.ordinary_generator_whole_U')
        audit.eq(c['actual_fit_input_ids'], U, label + '.ordinary_fit_whole_U')
    audit.ok(type(c['solution_index']) is int and c['solution_index'] >= 0, label + '.returned_branch_index')
    audit.ok(c['generator'] in ('SQPNP', 'IPPE_PLANE', 'SQPNP+LM', 'IPPE_PLANE+LM'), label + '.standard_generators')
    projected = project(c['dimensions'], c['R_cf'], c['centroid'], K)
    audit.ok(abs(determinant(c['R_cf']) - 1) <= 1e-6, label + '.proper_rotation_determinant')
    for i, row in enumerate(c['R_cf']):
        for j, other in enumerate(c['R_cf']):
            audit.ok(abs(dot(row, other) - int(i == j)) <= 1e-6, label + '.rotation_orthogonality.' + str(i) + '.' + str(j))
    audit.near(c['projected'], projected, label + '.independent_projection')
    audit.near(c['R_physical'], physical_rotation(c['R_cf'], d), label + '.registry_to_physical_rotation')
    residual = [math.dist(projected[k], points[k]) for k in U]
    audit.near(c['residuals_used_px'], residual, label + '.independent_only_U_residual')
    inliers = [k for k, value in zip(U, c['residuals_used_px']) if value <= RADIUS]
    audit.eq(c['inlier_ids'], inliers, label + '.fixed8px_membership')
    audit.eq(c['inlier_count'], len(inliers), label + '.inlier_count')
    audit.eq(c['score_count'], len(inliers) if robust else len(U), label + '.mode_score_count')
    sse = math.fsum(value * value for value in residual)
    truncated = math.fsum(min(value * value, 64.0) for value in residual)
    audit.near(c['sse_px2'], sse, label + '.whole_U_SSE')
    audit.near(c['truncated_sse_px2'], truncated, label + '.whole_U_truncated_SSE')
    audit.near(c['objective_px2'], truncated if robust else sse, label + '.mode_objective')
    counts['candidate_scalar_arithmetic_completed'] += 1
    counts[scope + '_arithmetic_completed'] += 1
    counts[scope + '_generator:' + c['generator']] += 1
    prefix = (-c['inlier_count'], c['objective_px2']) if robust else (c['objective_px2'],)
    return prefix + (d, tuple(c['generator_ids']), c['solution_index'], c['generator'])


def tied(candidate_record, best, robust):
    return (not robust or candidate_record['inlier_count'] == best['inlier_count']) and abs(candidate_record['objective_px2'] - best['objective_px2']) <= TIE


def pose_packet(row, audit, counts, previous, label):
    s, points, K, xyz, hw = row['solver'], row['input_points'], row['K'], row['xyz'], row['raw_hw']
    H = row['hidden_initial']
    robust = row['method'] != 'ROLE_BOUNDARY_H_STANDARD'
    E = eligible(points, hw)
    U = [k for k in E if k not in H]
    for flag in FALSE_FLAGS:
        audit.eq(s[flag], False, label + '.' + flag)
    audit.eq(s['solver'], 'OBSERVATION_ONLY_FINITE_SUBSET_ROBUST' if robust else 'OBSERVATION_ONLY_STANDARD', label + '.backend')
    audit.eq(s['equivalence_uses_all_eight_model_predictions'], True, label + '.equivalence_model_only')
    audit.eq(s['global_uniqueness_proven'], False, label + '.no_global_unique_claim')
    audit.eq(s['known_dimension_index'], None, label + '.no_known_dimension_index')
    audit.eq(s['known_dimension_provenance'], None, label + '.no_known_dimension_provenance')
    audit.eq(s['known_dimension_provenance_independently_verified'], False, label + '.no_external_fact_claim')
    audit.eq(s['eligible'], E, label + '.eligible_native_pixels')
    audit.eq(s['used'], U, label + '.exact_score_pool')
    audit.eq(s['hidden'], H, label + '.applied_H')
    audit.eq(s['actual_self_hidden_ids'], H, label + '.self_hidden_alias')
    audit.eq(s['temporary_excluded'], [], label + '.no_LOO_in_C3_controls')
    audit.eq(s['temporary_exclusion_is_self_occlusion'], False, label + '.temporary_not_H')
    audit.eq(s['excluded'], H, label + '.exclude_only_H')
    audit.eq(s['hidden_initial_excluded'], True, label + '.H_excluded_flag')
    audit.eq(s['residual_threshold_px'], RADIUS, label + '.fixed8px')
    for key in ('fit_input_ids', 'input_inliers', 'final_inliers', 'inliers'):
        audit.ok(ids(s[key]) and set(s[key]).issubset(U) and not set(s[key]).intersection(H), label + '.' + key + '.fit_excludes_H')
    reconstructed = native_hash(points, K, xyz, hw)
    if reconstructed is None:
        counts['nonfinite_full_bank_hash_not_independently_reconstructible'] += 1
    else:
        audit.eq(s['input_hash'], reconstructed, label + '.finite_float64_input_hash')
    op, after = s['operation_counts'], s['hypothesis_bank_counts']
    audit.eq(set(op), set(LEDGER_KEYS), label + '.interval_counter_fields')
    audit.eq(set(after), set(LEDGER_KEYS), label + '.cumulative_counter_fields')
    require(all(type(v) is int and v >= 0 for v in op.values()), 'invalid operation delta')
    before = {key: after[key] - op[key] for key in LEDGER_KEYS}
    audit.ok(all(value >= 0 for value in before.values()), label + '.nonnegative_previous')
    audit.eq(before, previous if previous is not None else {key: 0 for key in LEDGER_KEYS}, label + '.shared_bank_sequential_delta')
    ledger(after, audit, label + '.bank', len(E), len(dimensions(xyz)))
    audit.eq(op['lm_calls'], s['refit_count'], label + '.actual_LM_entries_equal_refit_attempts')
    if robust:
        audit.ok(type(s['refit_count']) is int and 0 <= s['refit_count'] <= 3, label + '.robust_max3_refit_starts')
        audit.eq(op['standard_generic_calls'], 0, label + '.robust_no_whole_U_standard_generator')
        audit.ok(op['refit_generic_calls'] <= s['refit_count'], label + '.actual_refit_generator_not_more_than_refits')
    else:
        audit.ok(s['refit_count'] in (0, 1), label + '.ordinary_max_one_LM')
        audit.eq(op['subset_generic_calls'], 0, label + '.ordinary_no_subset_generation')
        audit.eq(op['subsets_considered'], 0, label + '.ordinary_no_bank_rebuild')
        audit.eq(op['refit_generic_calls'], 0, label + '.ordinary_no_consensus_regeneration')
    audit.ok(0 <= s['LM_solution_count'] <= s['refit_count'], label + '.LM_returned_not_more_than_attempted')
    cs = s['all_candidate_solutions']
    for key in ('candidate_count', 'eligible_candidate_count', 'refit_solution_count', 'LM_solution_count'):
        audit.ok(type(s[key]) is int and s[key] >= 0, label + '.nonnegative_' + key)
    LM_records = [record for record in cs if record['generator'].endswith('+LM')]
    audit.eq(len(LM_records), s['LM_solution_count'], label + '.every_returned_LM_record_preserved')
    if robust:
        # Four-ID refit calls hit the already built exact same (dimension, IDs)
        # cache, so their object references are deduplicated with the original
        # hypotheses. Non-LM four-ID records are therefore precisely the
        # allowed initial subset hypotheses, not additional refit solutions.
        subset_records = [record for record in cs if not record['generator'].endswith('+LM') and len(record['generator_ids']) == 4]
        audit.eq(len(subset_records), s['eligible_candidate_count'], label + '.all_allowed_four_ID_hypotheses_preserved')
        audit.ok(s['eligible_candidate_count'] <= s['candidate_count'] <= after['returned_solutions'] - after['invalid_solutions'],
                 label + '.global_bank_count_includes_allowed_hypotheses')
        audit.ok(s['eligible_candidate_count'] + s['LM_solution_count'] <= len(cs) <=
                 s['eligible_candidate_count'] + s['refit_solution_count'] + s['LM_solution_count'],
                 label + '.deduplicated_pool_cardinality_bounds')
    else:
        audit.eq(s['candidate_count'], s['eligible_candidate_count'], label + '.ordinary_all_U_initial_count')
        audit.eq(s['refit_solution_count'], 0, label + '.ordinary_no_consensus_refit_generator_records')
        audit.eq(len(cs), s['eligible_candidate_count'] + s['LM_solution_count'], label + '.ordinary_all_returned_whole_U_and_LM_records')
    counts['stored_candidate_records_in_packets'] += len(cs)
    keys = []
    for index, record in enumerate(cs):
        keys.append(candidate(record, points, K, xyz, U, H, robust, audit, counts,
                              label + '.candidate.' + str(index), 'stored_candidate'))
        counts['stored_candidate_records_checked'] += 1
    if len(U) < 4:
        audit.eq(s['state'], 'INSUFFICIENT_OBSERVATIONS', label + '.less4_is_insufficient')
        audit.eq(cs, [], label + '.less4_no_candidate')
    if s['geometry'].get('image'):
        image_rank = stored_rank(s['geometry']['image'], min(2, len(U)), audit, label + '.image_layout')
        if s['state'] == 'NEW_POSE':
            audit.eq(image_rank, 2, label + '.accepted_image_rank2')
    if cs:
        best = s['best_candidate']
        candidate(best, points, K, xyz, U, H, robust, audit, counts, label + '.chosen_recheck', 'chosen_recheck')
        chosen_index = min(range(len(cs)), key=lambda index: keys[index])
        audit.eq(best, cs[chosen_index], label + '.best_mode_score_and_fixed_tie_order')
        audit.eq(s['fit_input_ids'], best['actual_fit_input_ids'], label + '.chosen_fit_IDs')
        audit.eq(s['input_inliers'], s['fit_input_ids'], label + '.input_inlier_alias')
        audit.eq(s['final_inliers'], best['inlier_ids'], label + '.chosen_final_inliers')
        audit.eq(s['inliers'], s['final_inliers'], label + '.final_inlier_alias')
        audit.near(s['residuals_used_px'], best['residuals_used_px'], label + '.chosen_residuals')
        if not robust:
            audit.eq(s['fit_input_ids'], U, label + '.ordinary_whole_U_fit')
            audit.eq(s['refit_count'], 1, label + '.ordinary_whole_U_LM_attempt')
        audit.eq(s['weak_four_point_consensus'], len(s['final_inliers']) == 4 if robust else len(U) == 4,
                 label + '.four_point_flag_mode_scope')
        per_dimension = []
        for dimension in range(len(dimensions(xyz))):
            indices = [index for index, record in enumerate(cs) if record['dimension_index'] == dimension]
            if indices:
                per_dimension.append(cs[min(indices, key=lambda index: keys[index])])
        audit.eq(s['per_dimension_best'], per_dimension, label + '.all_returned_dimensions_best')
        ranked_indices = sorted(range(len(cs)), key=lambda index: keys[index])
        alternatives = []
        numerical_ties = 0
        for index in ranked_indices[1:]:
            other = cs[index]
            is_tied = tied(other, best, robust)
            numerical_ties += int(is_tied)
            if equivalent(other, best, xyz):
                continue
            info = dict(other)
            info.update(numerical_objective_tie=is_tied,
                projection_max_abs_from_selected_px=max(abs(x - y) for a, b in zip(other['projected'], best['projected']) for x, y in zip(a, b)),
                physical_rotation_max_abs_from_selected=max(abs(x - y) for a, b in zip(other['R_physical'], best['R_physical']) for x, y in zip(a, b)))
            alternatives.append(info)
        ambiguity = any(record['numerical_objective_tie'] for record in alternatives)
        if s['state'] in ('NEW_POSE', 'AMBIGUOUS_PNP'):
            audit.near(s['alternatives'], alternatives, label + '.every_distinct_returned_alternative')
            audit.eq(s['numerical_tied_solution_count'], numerical_ties, label + '.all_mode_tied_branches')
            audit.eq(s['multiple_solutions'], bool(alternatives), label + '.multiple_physical_solutions')
            audit.eq(s['unresolved_ambiguity'], ambiguity, label + '.mode_ambiguity')
            audit.eq(s['state'] == 'AMBIGUOUS_PNP', ambiguity, label + '.ambiguous_unavailable')
        elif s['state'] in ('INSUFFICIENT_CONSENSUS', 'NUMERICAL_RANK_DEFICIENT'):
            audit.eq(s['alternatives'], [], label + '.earlier_gate_no_ambiguity_arbitration')
        rank_ids = s['final_inliers'] if robust else U
        J = s['geometry'].get('jacobian')
        audit.eq(row['stored_jacobian_fit_ids'], rank_ids if J is not None else [], label + '.declared_J_support_mode')
        if J is not None:
            counts['stored_J_packets'] += 1
            rank = stored_rank(J, 6, audit, label + '.stored_local_J')
            counts['stored_local_J_rank:' + str(rank)] += 1
            if s['state'] in ('NEW_POSE', 'AMBIGUOUS_PNP'):
                audit.eq(rank, 6, label + '.rank6_before_acceptance_or_ambiguity')
            if s['state'] == 'NUMERICAL_RANK_DEFICIENT':
                audit.ok(rank < 6, label + '.rank_failure_distinct')
        object_witness = s['geometry'].get('object')
        if object_witness is not None:
            object_rank = stored_rank(object_witness, 3, audit, label + '.stored_object_layout')
            if s['state'] == 'NEW_POSE':
                audit.ok(object_rank >= 2, label + '.accepted_object_rank_at_least2')
        if robust and len(s['final_inliers']) < 4:
            audit.eq(s['state'], 'INSUFFICIENT_CONSENSUS', label + '.robust_minimum4_gate')
    else:
        audit.eq(row['stored_jacobian_fit_ids'], [], label + '.no_candidates_no_J_support')
    new = s['available']
    audit.ok(type(new) is bool, label + '.available_boolean')
    audit.eq(s['pose_available'], new, label + '.numeric_available')
    audit.eq(s['new_pose_estimated'], new, label + '.numeric_new')
    audit.eq(s['no_pose'], not new, label + '.numeric_failure')
    audit.eq(s['fallback_used'], False, label + '.no_solver_baseline_fallback')
    audit.eq(s['state'] == 'NEW_POSE', new, label + '.NEW_state')
    expected = [list(point) for point in points]
    if new:
        audit.ok(bool(cs) and len(U) >= 4 and not s['unresolved_ambiguity'], label + '.accepted_actual_pool')
        for key, length in (('image', 2), ('object', 3), ('jacobian', 6)):
            audit.ok(isinstance(s['geometry'].get(key), dict), label + '.NEW_required_' + key + '_witness')
            require(isinstance(s['geometry'].get(key), dict), 'NEW rank witness missing')
            audit.eq(len(s['geometry'][key]['singular_values']), length, label + '.NEW_' + key + '_length')
        if robust:
            audit.ok(len(s['final_inliers']) >= 4, label + '.robust_accepted_consensus4')
        elif len(s['final_inliers']) < 4:
            counts['ordinary_NEW_with_fewer_than4_diagnostic_inliers'] += 1
        for key in ('R_cf', 'R_physical', 'centroid'):
            audit.near(s[key], s['best_candidate'][key], label + '.chosen_pose_' + key)
        audit.eq(s['cf_extents'], s['best_candidate']['dimensions'], label + '.chosen_dimensions')
        projected = project(s['cf_extents'], s['R_cf'], s['centroid'], K)
        audit.near(s['projected'], projected, label + '.final_Rt_projection')
        audit.near(s['sse_px2'], s['best_candidate']['sse_px2'], label + '.final_SSE')
        audit.near(s['truncated_sse_px2'], s['best_candidate']['truncated_sse_px2'], label + '.final_truncated_SSE')
        audit.near(s['reprojection_px'], math.fsum(s['residuals_used_px']) / len(U), label + '.mean_used_reprojection')
        for corner in H:
            expected[corner] = projected[corner]
        audit.eq(s['hidden_reprojected'], bool(H), label + '.H_replacement_flag')
    audit.near(s['points_final'], expected, label + '.solver_H_only_output')
    counts['pose_state:' + row['method'] + ':' + s['state']] += 1
    counts['actual_U_count:' + row['method'] + ':' + str(len(U))] += 1
    return after


def hidden_after(pose):
    R, t, dimensions_ = pose['R_cf'], pose['centroid'], pose['cf_extents']
    camera = [-math.fsum(R[j][i] * t[j] for j in range(3)) for i in range(3)]
    margin = -math.sin(math.radians(2.0))
    result = []
    for index, corner in enumerate(model(dimensions_)):
        ray = [a - b for a, b in zip(camera, corner)]
        norm = math.sqrt(dot(ray, ray))
        require(norm > 0, 'undefined stored camera-to-corner ray')
        if max((1 if coordinate > 0 else -1) * component / norm for coordinate, component in zip(corner, ray)) < margin:
            result.append(index)
    return result


def output_checks(row, parent, obs, audit, counts, label):
    native, H, s = row['original_native_N3_points'], row['hidden_initial'], row['solver']
    original_H = parent['hidden_initial']
    audit.eq(native, obs['native_N3_points'], label + '.original_native_N3_join')
    audit.eq(row['native_reference_id'], dict(id=row['id'], head_arm='IMAGE_ROLE'), label + '.native_reference_ID')
    audit.eq(row['parent_observation_row_semantic_sha256'], semantic_sha(obs), label + '.full_parent_observation_semantic_hash')
    audit.eq(row['parent_geometry_row_semantic_sha256'], semantic_sha(parent), label + '.full_parent_geometry_semantic_hash')
    audit.eq(row['parent_native_N3_semantic_sha256'], semantic_sha(native), label + '.native_semantic_hash')
    applied = row['method'] != 'ROLE_BOUNDARY_NO_MASK_ROBUST'
    audit.eq(H, original_H if applied else [], label + '.frozen_H_or_explicit_no_mask')
    audit.eq(row['predicted_initial_N3_hidden'], original_H, label + '.original_H_diagnostic')
    audit.eq(row['original_self_hidden_initial'], original_H, label + '.original_H_preserved')
    audit.eq(row['initial_H_applied'], applied, label + '.mask_route')
    audit.eq(row['diagnostic_no_mask'], not applied, label + '.no_mask_route')
    for key in ('K', 'xyz', 'raw_hw', 'selected_index', 'fixed_metadata', 'initial_pose', 'input_points',
                'head_arm', 'head_mode', 'head_checkpoint_sha256', 'calibration_binding', 'selected_corner_ids',
                'observation_contract', 'observation_raw_logits_sha256'):
        audit.eq(row[key], parent[key], label + '.same_parent_' + key)
    audit.eq(row['input_points'][8], native[8], label + '.input_center_preserved')
    audit.eq(row['head_arm'], 'IMAGE_ROLE', label + '.same_head')
    audit.eq(row['head_checkpoint_sha256'], CHECKPOINT, label + '.fixed_corrected_checkpoint')
    audit.eq(row['observation_raw_logits_sha256'], obs['raw_logits_sha256'], label + '.unchanged_observation_logits')
    admitted = row['observation_contract']['validated_boundary_corner_ids']
    audit.ok(ids(admitted), label + '.admitted_IDs')
    audit.eq([k for k in range(8) if finite(row['input_points'][k], (2,))], admitted, label + '.sparse_exact_admission')
    for corner in range(8):
        if corner not in admitted:
            audit.eq(row['input_points'][corner], [None, None], label + '.missing_sparse_never_filled.' + str(corner))
    for flag in ('native_N3_used_as_independent_RGB_observations', 'missing_sparse_filled_for_numeric_fit',
                 'no_match_filled_as_boundary_observation', 'final_numeric_pose_has_initial_prior',
                 'selection_validation_fit_excludes_candidate_corner', 'selection_validation_has_shared_initial_prior',
                 'reprojections_reused_as_observations', 'local_point_line_refinement', 'oracle'):
        audit.eq(row[flag], False, label + '.' + flag)
    for flag in ('partial_lines_not_pose_inputs', 'same_sparse_observation'):
        audit.eq(row[flag], True, label + '.' + flag)
    audit.eq(row['head_forward_calls'], 0, label + '.no_new_head')
    audit.eq(row['calibration_or_decode_calls'], 0, label + '.no_new_decode_CAL')
    new = s['available']
    fallback = not new and row['initial_pose']['available']
    audit.eq(row['new_pose_estimated'], new, label + '.driver_new')
    audit.eq(row['fallback_used'], fallback, label + '.declared_fallback')
    audit.eq(row['pose_available'], new or fallback, label + '.operational_available')
    audit.eq(row['no_pose'], not (new or fallback), label + '.complete_failure')
    audit.eq(row['actual_pose']['available'], new or fallback, label + '.actual_available')
    audit.eq(row['output_status'], 'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE', label + '.distinct_status')
    if new:
        for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected'):
            audit.eq(row['actual_pose'][key], s[key], label + '.actual_equals_solver_' + key)
        counts['NEW_driver_pose_joins'] += 1
    else:
        audit.eq(row['actual_pose'], row['initial_pose'], label + '.unchanged_initial_fallback_or_failure')
        counts['fallback_or_failure_driver_pose_joins'] += 1
    expected = [list(point) for point in native]
    if new:
        for corner in range(8):
            if corner not in H and finite(row['input_points'][corner], (2,)) and row['input_points'][corner] != [-1, -1]:
                expected[corner] = row['input_points'][corner]
        for corner in H:
            expected[corner] = s['projected'][corner]
    audit.near(row['native_points'], expected, label + '.observed_output_H_projection_or_full_native_fallback')
    audit.eq(row['native_points'][8], native[8], label + '.output_center_exact')
    audit.eq(row['reprojected_ids'], H if new else [], label + '.only_applied_H_replaced_once')
    audit.eq(row['hidden_reprojected'], bool(new and H), label + '.H_output_flag')
    after = hidden_after(s) if new else []
    audit.eq(row['hidden_after'], after, label + '.postfit_mask_diagnostic')
    audit.eq(row['hidden_set_changed'], bool(new and set(after) != set(H)), label + '.mask_change_recorded_not_frame_veto')
    sources = ['FINAL_POSE_REPROJECTION' if new and corner in H else
               'VALIDATED_BOUNDARY_INTERSECTION' if new and corner in admitted else 'NATIVE_N3_RGB'
               for corner in range(8)] + ['UNCHANGED_DETECTOR_CENTER']
    audit.eq(row['output_coordinate_sources'], sources, label + '.output_roles')
    counts['output_status:' + row['method'] + ':' + row['output_status']] += 1
    counts['postfit_H_set_changed_recorded'] += int(row['hidden_set_changed'])


def parent_primary_rows(path, audit, cohort):
    counts, seen = Counter(), set()
    for row in rows(path):
        key = (row['method'], row['id'])
        audit.ok(row['method'] in PARENT_METHODS and row['id'] in cohort and key not in seen, 'parent.unique_geometry_ID')
        audit.ok(not TRUTH_FIELDS.intersection(row), 'parent.geometry.GT_free')
        seen.add(key)
        counts[row['method']] += 1
        if row['method'] == 'IMAGE_ROLE_BOUNDARY_ONLY':
            yield row
    audit.eq(len(seen), 1960, 'parent.complete1960_geometry_rows')
    for method in PARENT_METHODS:
        audit.eq(counts[method], 245, 'parent.every_method245.' + method)
        audit.eq({fid for name, fid in seen if name == method}, cohort, 'parent.every_method_same_cohort.' + method)


def parent_ROLE_rows(path, audit, cohort):
    counts, seen = Counter(), set()
    for row in rows(path):
        key = (row['head_arm'], row['id'])
        audit.ok(row['head_arm'] in HEADS and row['id'] in cohort and key not in seen, 'parent.unique_observation_ID')
        audit.eq(row['GT_input'], False, 'parent.observation.GT_free')
        seen.add(key)
        counts[row['head_arm']] += 1
        if row['head_arm'] == 'IMAGE_ROLE':
            yield row
    audit.eq(len(seen), 735, 'parent.complete735_observation_rows')
    for head in HEADS:
        audit.eq(counts[head], 245, 'parent.every_head245.' + head)
        audit.eq({fid for name, fid in seen if name == head}, cohort, 'parent.every_head_same_cohort.' + head)


def parent_fixed_frames(path, audit, cohort):
    stream = iter(rows(path))
    seen = set()
    while True:
        first = next(stream, None)
        if first is None:
            break
        second = next(stream, None)
        require(second is not None, 'missing second fixed parent control')
        group = {}
        for row in (first, second):
            audit.eq(row['id'], first['id'], 'parent.fixed.same_frame_pair')
            key = (row['method'], row['id'])
            audit.ok(row['method'] in ('BASE', 'N3_SUBPIX') and row['id'] in cohort and key not in seen,
                     'parent.fixed.unique_known_control')
            audit.ok(not TRUTH_FIELDS.intersection(row), 'parent.fixed.GT_free')
            seen.add(key)
            group[row['method']] = row
        audit.eq(set(group), {'BASE', 'N3_SUBPIX'}, 'parent.fixed.exact_pair')
        yield group
    audit.eq(len(seen), 490, 'parent.complete490_fixed_controls')
    for method in ('BASE', 'N3_SUBPIX'):
        audit.eq({fid for name, fid in seen if name == method}, cohort, 'parent.every_fixed_same_cohort.' + method)


def numerical_same_U(first, last, audit, label):
    for key in ('available', 'state', 'reason', 'used', 'fit_input_ids', 'final_inliers', 'prior_used',
                'known_dimension_constraint_used', 'unresolved_ambiguity', 'weak_four_point_consensus'):
        audit.eq(first[key], last[key], label + '.' + key)
    for key in ('generator_ids', 'selected_hypothesis', 'dimension_index', 'R_cf', 'R_physical', 'centroid',
                'cf_extents', 'projected', 'residuals_used_px', 'sse_px2', 'truncated_sse_px2'):
        audit.eq(key in first, key in last, label + '.same_presence_' + key)
        audit.near(first.get(key), last.get(key), label + '.' + key)


def receipt_checks(paths, inputs, core, seal, receipt, population, audit):
    audit.eq(core['schema'], 'fixed_same_observation_C3_protocol_v8', 'V8.protocol_schema')
    audit.eq(core['new_accuracy_primary'], False, 'V8.diagnostic_not_new_primary')
    audit.eq(core['evaluation_is_unseen'], False, 'V8.existing_parent_accuracy_known')
    audit.eq(core['GT_tuning'], False, 'V8.no_GT_tuning')
    audit.eq(core['accuracy_policy_changes'], 0, 'V8.no_accuracy_policy_changes')
    audit.eq(core['requested_original_C3_diagnostic'], True, 'V8.fixed_C3_diagnostic')
    core_bindings(inputs, core)
    audit.eq(seal['cleanup_completed_before_seal'], True, 'V8.cleanup_before_seal')
    audit.eq(seal['no_scored_or_human_inputs'], True, 'V8.seal_no_scored_inputs')
    audit.eq(seal['latency_benchmark'], False, 'V8.seal_not_latency')
    audit.eq(seal['ledger_rows'], 245, 'V8.seal245_ledgers')
    audit.eq(receipt['schema'], 'fixed_same_observation_control_receipt_v8', 'V8.receipt_schema')
    audit.eq(receipt['geometry_ready_for_seal'], True, 'V8.receipt_ready_for_seal')
    audit.eq(receipt['formal_GT_permission_requires_complete_seal'], True, 'V8.GT_requires_seal')
    audit.eq(receipt['actual_complete_frames'], 245, 'V8.recorded_completed245')
    audit.ok(same_binding(receipt['protocol'], inputs['PROTOCOL.json']), 'V8.receipt_protocol_binding')
    for key in ('actual_detector_calls', 'actual_N3_calls', 'actual_head_calls', 'actual_initial_pose_calls',
                'actual_CAL_calls', 'actual_query_decode_calls', 'actual_model_constructors', 'new_training_updates', 'new_RGB'):
        audit.eq(receipt[key], 0, 'V8.receipt.' + key)
    for key in ('GT_access_during_controls', 'runtime_benchmark', 'deployment_latency_claim'):
        audit.eq(receipt[key], False, 'V8.receipt.' + key)
    audit.eq(receipt['source_asset_or_GT_attempted_reads'], [], 'V8.recorded_no_forbidden_reads')
    audit.eq(receipt['environment_and_monkeypatch_cleanup_completed'], True, 'V8.all_contexts_cleaned')
    audit.ok(same_binding(receipt['actual_OpenCV_entry_counter_code'], inputs['runner_code']),
             'V8.recorded_actual_entry_counter_code_binding')
    audit.eq(receipt['actual_OpenCV_primitive_names'], list(PRIMITIVES), 'V8.recorded_actual_entry_primitive_names')
    audit.eq(receipt['actual_OpenCV_entry_counter_scope'],
             'actual entries during the three solve_controls routes, including their projection/Jacobian/rotation work',
             'V8.recorded_actual_entry_counter_scope')
    environment = receipt['library_environment']
    audit.eq(environment['solve_OpenCV_threads'], 1, 'V8.recorded_solve_OpenCV_threads')
    audit.eq(environment['Torch_or_model_library_initialized'], False, 'V8.recorded_no_model_library_initialization')
    for key in ('python', 'numpy', 'opencv'):
        audit.ok(isinstance(environment[key], str) and bool(environment[key]), 'V8.recorded_library_version.' + key)
    audit.eq(set(environment['thread_environment']), {'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'},
             'V8.recorded_thread_environment_fields')
    audit.eq(receipt['no_automatic_retry'], True, 'V8.receipt_no_retry')
    serialization = receipt['serialization']
    audit.eq(serialization['mode'], 'per_frame_full_witness_stream', 'V8.complete_witness_stream')
    audit.eq(serialization['diagnostic_fields_removed'], 0, 'V8.no_witness_fields_removed')
    audit.eq(serialization['full_parent_population_retained'], False, 'V8.no_parent_full_population_retention')
    for name, count in (('geometry', 735), ('ledgers', 245)):
        value = serialization['streams'][name]
        audit.eq(value['rows'], count, 'V8.serialized.' + name)
        audit.eq(value['published'], True, 'V8.published.' + name)
        audit.eq(value['close_errors'], [], 'V8.stream_close.' + name)
        audit.eq(value['interruption_preservation_errors'], [], 'V8.stream_preservation.' + name)
    audit.eq(receipt['protection_before'], receipt['protection_after'], 'V8.recorded_protection_unchanged')
    protection = receipt['protection_after']
    for key in ('passed', 'source_checkout_unchanged'):
        audit.eq(protection[key], True, 'V8.protection.' + key)
    audit.eq(protection['changed_protected_paths'], [], 'V8.no_recorded_protected_change')
    audit.eq(population['passed'], True, 'V8.full_parent_population_passed')
    audit.eq(population['full_population_checked_before_first_control_pose_path'], True, 'V8.full_parent_before_fit')
    audit.eq(population['scored_or_human_fields_passed_to_solver'], False, 'V8.parent_no_scored_fields')
    audit.eq(population['source_model_image_GT_PnP_calls'], 0, 'V8.parent_proof_no_fits_models')
    audit.eq(population['full_parent_population_rows_retained'], 0, 'V8.parent_proof_no_population_retention')
    audit.eq(population['one_frame_primary_and_ROLE_observation_kept'], True, 'V8.parent_proof_one_frame_witness_scope')
    audit.eq(population['completion'], core['parent_completion'], 'V8.parent_completion_protocol_join')
    audit.eq(seal['parent_completion'], core['parent_completion'], 'V8.parent_completion_seal_join')
    started = read(paths['CONTROL_STARTED.json'])
    audit.ok(same_binding(started['protocol'], inputs['PROTOCOL.json']), 'V8.started_protocol_binding')
    for key, expected in (('configured_frames', 245), ('configured_rows', 735), ('configured_pose_paths', 735),
                          ('parent_checked_before_first_pose', True), ('no_automatic_retry', True)):
        audit.eq(started[key], expected, 'V8.started.' + key)
    preflight = read(paths['PREFLIGHT.json'])
    audit.eq(preflight['passed'], True, 'V8.preflight_passed')
    audit.eq(preflight['new_pose_paths'], 0, 'V8.preflight_no_fit')
    audit.ok(same_binding(preflight['protocol'], inputs['PROTOCOL.json']), 'V8.preflight_protocol_binding')
    snapshots = receipt['resource_snapshots']
    audit.ok(len(snapshots) >= 2, 'V8.recorded_resource_endpoints')
    audit.eq(snapshots[0]['phase'], 'before_control_imports', 'V8.resource_start')
    audit.eq(snapshots[-1]['phase'], 'after_control_cleanup', 'V8.resource_end')
    for index, snapshot in enumerate(snapshots):
        audit.eq(read(paths['resource:%03d' % index]), snapshot, 'V8.resource_durable.' + str(index))
        audit.eq(snapshot['quiet'], True, 'V8.resource_quiet.' + str(index))
        audit.eq(snapshot['temperature_under_80'], True, 'V8.resource_temperature.' + str(index))
    # These are byte-bound historical snapshots, never an independent claim
    # that transient hardware interference could not occur between probes.


def run(args):
    folder, output = guard(args, 'run')
    own_path = output / OUTPUTS[0]
    own = read(own_path)
    paths = input_paths(args)
    inputs = {key: binding(path) for key, path in paths.items()}
    require(inputs == own['inputs'] and own['checker'] == inputs['checker_code'] and own['limits'] == LIMITS and
            own['schema'] == 'supplemental_same_sparse_ROLE_scalar_protocol_v8' and
            own['scalar_absolute_tolerance'] == ATOL and own['scalar_relative_tolerance'] == RTOL,
            'own frozen code/input/protocol differs')
    protocol_binding = binding(own_path)
    write_new(output / OUTPUTS[1], dict(status='STARTED', protocol=protocol_binding, checker=inputs['checker_code'], no_automatic_retry=True))
    start = time.monotonic()
    audit, counts = Audit(), Counter()
    bank_operation_totals = Counter({key: 0 for key in LEDGER_KEYS})
    finished = False
    error = None
    try:
        core, seal, receipt, population = [read(paths[key]) for key in
            ('PROTOCOL.json', 'GEOMETRY_SEAL.json', 'CONTROL_RECEIPT.json', 'PARENT_POPULATION_CHECKS.json')]
        audit.eq(core['methods'], list(METHODS), 'V8.fixed_methods')
        audit.eq(core['frames'], 245, 'V8.fixed245')
        audit.eq(seal['complete'], True, 'V8.seal_complete')
        audit.eq(seal['GT_read_allowed'], False, 'V8.seal_before_GT')
        audit.eq(seal['rows'], 735, 'V8.seal735')
        audit.eq(receipt['complete'], True, 'V8.control_complete')
        audit.eq(receipt['cleanup_error'], None, 'V8.control_cleanup')
        audit.eq(receipt['error'], None, 'V8.control_error')
        audit.eq(population['complete'], True, 'V8.parent_population_completed_before_control')
        receipt_checks(paths, inputs, core, seal, receipt, population, audit)
        for key, expected in (('PROTOCOL.json', seal['protocol']), ('GEOMETRY_SEALED.jsonl.gz', seal['geometry']),
                ('CONTROL_LEDGERS.jsonl.gz', seal['ledgers']), ('CONTROL_RECEIPT.json', seal['control_receipt']),
                ('PARENT_POPULATION_CHECKS.json', seal['parent_population_checks'])):
            audit.ok(same_binding(inputs[key], expected), 'V8.seal_binding.' + key)
        cohort_document = read(paths['cohort'])
        cohort_order = cohort_document['ids']
        cohort = set(cohort_order)
        audit.eq(len(cohort_order), 245, 'V8.cohort_order245')
        audit.eq(len(cohort), 245, 'V8.cohort_unique245')
        audit.eq(cohort_document['counts'], dict(clean=153, moderate=92, severe_excluded=74, original=319), 'V8.fixed_difficulty_scope')
        audit.eq([frame['id'] for frame in cohort_document['frames']], cohort_order, 'V8.cohort_frame_order')
        audit.eq(Counter(frame['label'] for frame in cohort_document['frames']), Counter(clean=153, moderate=92), 'V8.only_clean_and_moderate')
        audit.eq(population['ordered_ids'], cohort_order, 'V8.proof_same_ordered245')
        audit.eq(read(paths['parent:BASE_N3_PARITY.json'])['passed'], True, 'V8.parent_fixed_parity_receipt')
        parent_rows = iter(parent_primary_rows(paths['parent:GEOMETRY_SEALED.jsonl.gz'], audit, cohort))
        observations = iter(parent_ROLE_rows(paths['parent:OBSERVATIONS.jsonl.gz'], audit, cohort))
        fixed_controls = iter(parent_fixed_frames(paths['parent:FIXED_GEOMETRY_SEALED.jsonl.gz'], audit, cohort))
        controls = iter(rows(paths['GEOMETRY_SEALED.jsonl.gz']))
        ledgers = iter(rows(paths['CONTROL_LEDGERS.jsonl.gz']))
        completed_IDs = set()
        removed_H_counts = Counter()
        for frame_index, fid in enumerate(cohort_order):
            parent, obs, bank = next(parent_rows, None), next(observations, None), next(ledgers, None)
            fixed = next(fixed_controls, None)
            require(parent is not None and obs is not None and bank is not None, 'missing full frame source/ledger')
            require(fixed is not None, 'missing full fixed-control frame')
            audit.eq(parent['id'], fid, 'frame.' + str(frame_index) + '.parent_order')
            audit.eq(obs['id'], fid, fid + '.observation_order')
            audit.eq(bank['id'], fid, fid + '.ledger_order')
            audit.eq(bank['session'], parent['session'], fid + '.ledger_session')
            audit.eq(fixed['N3_SUBPIX']['id'], fid, fid + '.fixed_order')
            audit.eq(fixed['N3_SUBPIX']['native_points'], obs['native_N3_points'], fid + '.native_N3_is_fixed_control')
            audit.eq(fixed['BASE']['native_points'], obs['original_base_points'], fid + '.native_BASE_is_fixed_control')
            audit.eq(parent['initial_pose'], obs['initial_N3_pose'], fid + '.same_original_initial_pose')
            audit.eq(parent['hidden_initial'], obs['predicted_N3_hidden'], fid + '.same_original_predicted_H')
            frame_rows = []
            previous = None
            for method in METHODS:
                row = next(controls, None)
                require(row is not None, 'missing control row')
                label = method + '/' + fid
                audit.eq(row['method'], method, label + '.fixed_route_order')
                audit.eq(row['id'], fid, label + '.same_frame')
                audit.eq(row['session'], parent['session'], label + '.same_session')
                audit.ok(not TRUTH_FIELDS.intersection(row), label + '.GT_free')
                audit.eq(row['full_candidate_and_operation_witnesses_preserved'], True, label + '.full_witness_preserved')
                audit.eq(row['solver_replay_latency_claim'], False, label + '.replay_not_latency')
                audit.eq(row['stored_jacobian_fit_ids_source'], 'unchanged V4 solve rank_ids: final_inliers if robust else used', label + '.J_support_source')
                output_checks(row, parent, obs, audit, counts, label)
                previous = pose_packet(row, audit, counts, previous, label + '.solver')
                counts['geometry_rows_checked'] += 1
                frame_rows.append(row)
            audit.eq(bank['methods'], list(METHODS), fid + '.ledger_methods')
            audit.eq(bank['coordinate_bank_count'], 1, fid + '.one_shared_bank')
            audit.eq(len(bank['coordinate_banks']), 1, fid + '.one_bank_record')
            final_bank = bank['coordinate_banks'][0]
            audit.eq(final_bank['prior_backend'], False, fid + '.no_prior_backend')
            hashes = [row['solver']['input_hash'] for row in frame_rows]
            audit.eq(hashes, [final_bank['input_hash']] * 3, fid + '.same_input_hash_all_masks_and_modes')
            audit.eq({key: value for key, value in final_bank.items() if key not in ('input_hash', 'prior_backend')}, previous,
                     fid + '.final_once_per_bank_ledger')
            bank_operation_totals.update(previous)
            audit.eq(bank['logical_pose_paths'], 3, fid + '.actual_logical3')
            audit.eq(bank['existing_control_replay_pose_paths'], 1, fid + '.actual_parent_replay1')
            audit.eq(bank['new_diagnostic_pose_paths'], 2, fid + '.actual_new_diagnostics2')
            for key in ('detector_calls', 'N3_calls', 'head_calls', 'initial_pose_calls', 'calibration_calls',
                        'query_decode_calls', 'new_training_updates', 'new_RGB'):
                audit.eq(bank[key], 0, fid + '.no_new_' + key)
            audit.eq(bank['parent_unchanged'], True, fid + '.parent_unchanged')
            audit.eq(bank['missing_sparse_coordinates_filled_for_numeric_fit'], False, fid + '.no_sparse_fill')
            audit.eq(bank['parent_full_file_provenance_is_caller_responsibility'], True, fid + '.caller_provenance_scope')
            audit.eq(bank['this_module_independently_verified_full_file_provenance'], False, fid + '.pure_controls_scope')
            audit.eq(bank['latency_benchmark'], False, fid + '.replay_not_latency')
            audit.eq(bank['parent_replay_parity']['passed'], True, fid + '.recorded_parent_replay_parity')
            first, ordinary, last = [row['solver'] for row in frame_rows]
            same_U = first['used'] == last['used']
            audit.eq(bank['H_and_no_H_used_U_equal'], same_U, fid + '.actual_same_U')
            audit.eq(bank['H_and_no_H_used_U'], dict(masked=first['used'], unmasked=last['used']), fid + '.used_pools_join')
            audit.eq(bank['actual_H_removed_sparse_ids'], sorted(set(last['used']) - set(first['used'])), fid + '.actual_H_excluded_admitted_points')
            removed_H_counts.update(bank['actual_H_removed_sparse_ids'])
            audit.eq(bank['H_and_no_H_display_compared_as_same'], False, fid + '.display_H_is_separate')
            if same_U:
                audit.eq(bank['H_and_no_H_equal_U_numeric_parity']['passed'], True, fid + '.recorded_equal_U_parity')
                numerical_same_U(first, last, audit, fid + '.independent_equal_U_parity')
                counts['same_U_H_and_no_H_frames'] += 1
            else:
                audit.eq(bank['H_and_no_H_equal_U_numeric_parity'], None, fid + '.different_U_not_equal_pose_claim')
            numerical_same_U(frame_rows[0]['solver'], parent['solver'], audit, fid + '.parent_control_solver_parity')
            for key in ('output_status', 'new_pose_estimated', 'fallback_used', 'pose_available', 'no_pose',
                        'hidden_initial', 'hidden_after', 'hidden_set_changed', 'hidden_reprojected',
                        'reprojected_ids', 'output_coordinate_sources', 'native_points'):
                audit.near(frame_rows[0][key], parent[key], fid + '.parent_control_output_parity.' + key)
            for key in ('available', 'selected_hypothesis'):
                audit.eq(frame_rows[0]['actual_pose'].get(key), parent['actual_pose'].get(key), fid + '.parent_actual_pose.' + key)
            for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected'):
                audit.near(frame_rows[0]['actual_pose'].get(key), parent['actual_pose'].get(key), fid + '.parent_actual_pose.' + key)
            completed_IDs.add(fid)
            counts['frames_checked'] += 1
            counts['unique_coordinate_banks_checked'] += 1
            counts['logical_pose_paths_recorded'] += bank['logical_pose_paths']
            counts['parent_replay_paths_recorded'] += bank['existing_control_replay_pose_paths']
            counts['new_diagnostic_paths_recorded'] += bank['new_diagnostic_pose_paths']
            if counts['frames_checked'] % 25 == 0:
                print('SAME_ROLE_SCALAR_FRAMES', counts['frames_checked'], 245, flush=True)
            # Full witnesses retained only for the current frame, then released.
            del parent, obs, bank, fixed, frame_rows
        for stream, name in ((parent_rows, 'parent_geometry'), (observations, 'parent_observations'), (fixed_controls, 'parent_fixed'),
                             (controls, 'control_geometry'), (ledgers, 'control_ledgers')):
            audit.eq(next(stream, None), None, 'V8.no_extra_' + name)
        audit.eq(completed_IDs, cohort, 'V8.same_all245_IDs')
        audit.eq(counts['frames_checked'], 245, 'V8.actual245_frames')
        audit.eq(counts['geometry_rows_checked'], 735, 'V8.actual735_rows')
        audit.eq(counts['logical_pose_paths_recorded'], 735, 'V8.actual735_logical_solves')
        audit.eq(counts['parent_replay_paths_recorded'], 245, 'V8.actual245_parent_replays')
        audit.eq(counts['new_diagnostic_paths_recorded'], 490, 'V8.actual490_new_diagnostic_solves')
        expected_calls = dict(frames_started=245, frames_complete=245, geometry_rows=735, ledger_rows=245,
            coordinate_banks_started=245, coordinate_banks_complete=245,
            logical_pose_paths_started=735, logical_pose_paths_complete=735)
        for method in METHODS:
            expected_calls[method + ':started'] = 245
            expected_calls[method + ':complete'] = 245
        audit.eq(receipt['actual_counts'], expected_calls, 'V8.independent_full_row_counts_equal_actual_entries_receipt')
        audit.eq(receipt['method_rows'], {method: 245 for method in METHODS}, 'V8.receipt_every_method245')
        status_counts = {key.removeprefix('output_status:'): value for key, value in counts.items() if key.startswith('output_status:')}
        audit.eq(receipt['output_status_counts'], status_counts, 'V8.receipt_status_counts_from_every_row')
        audit.eq(receipt['existing_control_replay_pose_paths_started'], 245, 'V8.receipt_replay_started245')
        audit.eq(receipt['existing_control_replay_pose_paths_complete'], 245, 'V8.receipt_replay_complete245')
        audit.eq(receipt['new_diagnostic_pose_paths_started'], 490, 'V8.receipt_newdiag_started490')
        audit.eq(receipt['new_diagnostic_pose_paths_complete'], 490, 'V8.receipt_newdiag_complete490')
        audit.eq(receipt['H_and_no_H_equal_used_U_frames'], counts['same_U_H_and_no_H_frames'], 'V8.receipt_same_U_frames')
        audit.eq(receipt['actual_H_removed_sparse_ID_counts'], {str(key): value for key, value in removed_H_counts.items()}, 'V8.receipt_H_exclusion_counts')
        entries = receipt['actual_OpenCV_entry_calls']
        audit.eq(set(entries), set(PRIMITIVES), 'V8.all6_recorded_OpenCV_counters')
        audit.ok(all(type(value) is int and value >= 0 for value in entries.values()), 'V8.valid_OpenCV_entry_values')
        audit.eq(entries['solvePnP'], 0, 'V8.no_initial_or_baseline_PnP')
        audit.eq(entries['cornerSubPix'], 0, 'V8.no_new_N3_correction')
        audit.eq(entries['solvePnPGeneric'], bank_operation_totals['generic_calls'], 'V8.actual_Generic_entries_equal_all_bank_totals')
        audit.eq(entries['solvePnPRefineLM'], bank_operation_totals['lm_calls'], 'V8.actual_LM_entries_equal_all_bank_totals')
        audit.eq(entries['projectPoints'], counts['stored_J_packets'], 'V8.actual_projectPoints_entries_equal_stored_J_packets')
        audit.eq(counts['stored_candidate_records_checked'], counts['stored_candidate_records_in_packets'], 'V8.every_stored_candidate_reached')
        audit.eq(counts['stored_candidate_arithmetic_invocations'], counts['stored_candidate_records_checked'], 'V8.every_stored_record_checked_once')
        audit.eq(counts['candidate_scalar_arithmetic_completed'], counts['candidate_scalar_arithmetic_invocations'], 'V8.every_candidate_arithmetic_completed')
        audit.eq(counts['candidate_scalar_arithmetic_invocations'], counts['stored_candidate_records_checked'] + counts['chosen_recheck_arithmetic_invocations'],
                 'V8.stored_records_and_chosen_rechecks_separate')
        audit.eq({key: binding(path) for key, path in paths.items()}, own['inputs'], 'V8.all_frozen_inputs_unchanged_after_math')
        audit.eq(binding(own_path), protocol_binding, 'V8.own_protocol_unchanged')
        finished = True
    except Exception as exception:
        error = dict(type=type(exception).__name__, message=str(exception))
        audit.ok(False, 'V8.scalar_exception', error, 'all checks complete')
    result = dict(schema='supplemental_same_sparse_ROLE_scalar_checks_v8', complete=finished,
        passed=finished and audit.failure_count == 0, protocol=protocol_binding, checker=inputs['checker_code'], inputs=inputs,
        limits=LIMITS, checks=audit.checks, failure_count=audit.failure_count,
        first_failure=audit.failures[0] if audit.failures else None, failures_first100=audit.failures,
        exception=error, actual_saved_arithmetic_counts=dict(counts), maximum_absolute_scalar_difference=audit.max_difference,
        independently_joined_unique_bank_operation_totals=dict(bank_operation_totals),
        full_geometry_population_retained=False, full_witness_diagnostics_removed_from_originals=0,
        Python=sys.version, elapsed_seconds=time.monotonic() - start)
    write_new(output / OUTPUTS[2], result)
    print('SAME_ROLE_SCALAR_CHECKS', 'PASS' if result['passed'] else 'FAIL', audit.checks,
          counts['geometry_rows_checked'], counts['stored_candidate_records_checked'], flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--cohort', type=Path, default=COHORT)
    parser.add_argument('--protocol', type=Path, default=DOC / 'PROTOCOL.json', help='frozen V8 formal protocol, including for private geometry input')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args)
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
