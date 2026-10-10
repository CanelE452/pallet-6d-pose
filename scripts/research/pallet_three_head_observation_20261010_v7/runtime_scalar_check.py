"""Supplemental scalar audit of a completed V7 fresh-path runtime receipt.

Freeze this checker and its byte-bound inputs AFTER accuracy/runtime completion,
then run once.  Only the standard library is imported.  No model, image, GT,
production module, PnP, optimizer, ray, Jacobian or SVD is executed.  The full
candidate witnesses stay in the original stream; this checker retains timing
scalars and small counters, never the full row population.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE = Path('/tmp/pallet-three-head-observation-private-20261010-v7')
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
HEADS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
METHODS = ('IMAGE_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_BOUNDARY_ONLY',
           'IMAGE_NO_ROLE_BOUNDARY_ONLY', 'GEOMETRY_ONLY_CORNERWISE_HYBRID',
           'IMAGE_NO_ROLE_CORNERWISE_HYBRID', 'IMAGE_ROLE_CORNERWISE_HYBRID',
           'N3_INDEPENDENT_ROBUST_H', 'N3_INDEPENDENT_ROBUST_NO_MASK')
ARMS = ('BASE', 'N3_SUBPIX', *METHODS)
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
          'final_pose_reprojection_and_metadata')
SUMMARY_STAGES = ('full', *STAGES, 'return')
MOMENTS = ('n', 'mean', 'sample_variance', 'sample_std', 'median', 'P90', 'maximum')
PRIMITIVES = ('solvePnP', 'solvePnPRefineLM', 'cornerSubPix', 'solvePnPGeneric')
FALSE_FLAGS = ('prior_used', 'initial_pose_used', 'initial_projection_used',
               'initial_dimension_prior_used', 'known_dimension_constraint_used',
               'excluded_image_coordinates_used_for_scoring',
               'excluded_image_coordinates_used_for_equivalence',
               'reprojected_points_reused_as_observations')
OUTPUTS = ('RUNTIME_VALIDATION_PROTOCOL.json', 'RUNTIME_VALIDATION_STARTED.json',
           'RUNTIME_VALIDATION_CHECKS.json')
ATOL, RTOL = 1e-7, 1e-12
LIMITS = dict(
    supplemental_after_completed_accuracy_and_runtime=True,
    accuracy_policy_or_model_changes=0,
    new_model_image_GT_PnP_training_ray_Jacobian_SVD_calls=0,
    independent_hardware_authenticity_claim=False,
    snapshot_can_exclude_all_transient_interference=False,
    raw_absolute_stage_timestamps_available=False,
    full_timing_is_original_full_ms_not_sum_derived=True,
    partition_check='Stored stage/return durations versus independently stored full_ms; absolute t0/t1/ticks were not serialized.',
    center_check='Only the recorded center/score/box preservation flag is counted. Original center/score/box values are absent from runtime rows; no independent coordinate recalculation is claimed.',
    fallback_pose_check='Recorded availability/status and parity flags only; the pre-fit initial pose is absent from runtime rows.',
    parity_check='Recorded parity tolerances/maxima and flags are validated; no new model/reference replay.',
    candidate_check='Counts stored final/LOO candidate witnesses without redoing their scalar geometry; the separate geometry checker covers candidate arithmetic.',
    H_camera_authority='Same frame ID joined to byte-bound public INPUTS K, cross-checked against the frozen runtime panel K.',
    protection_check='Recorded cleanup/protection snapshots are cross-checked, not an independent historical filesystem or process monitor.',
    no_automatic_retry=True)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def binding(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'missing/symlink input: ' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    absolute = path.resolve()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
                sha256=digest.hexdigest(), bytes=path.stat().st_size)


def bound(path, expected):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')),
            'binding differs: ' + Path(path).name)


def same_binding(actual, expected):
    return all(actual[k] == expected[k] for k in ('sha256', 'bytes'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()


def guard(args, stage):
    paths = (args.input.absolute(), args.output.absolute(), args.protocol.absolute(),
             args.panel.absolute(), args.cohort.absolute(), args.authority.absolute())
    for path in paths:
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink ancestry')
    require(args.input.is_dir() and args.output.is_dir(), 'input/output directories must already exist')
    destination = args.output.resolve()
    require(destination == DOC.resolve() or destination.is_relative_to(PRIVATE.resolve()),
            'only new V7 DOC/private outputs allowed')
    for name in OUTPUTS if stage == 'freeze' else OUTPUTS[1:]:
        path = destination / name
        require(not path.exists() and not path.is_symlink(), 'preserve first/existing ' + name)
    if stage == 'run':
        require((destination / OUTPUTS[0]).is_file(), 'freeze required before run')
    return args.input.resolve(), destination


def input_paths(args):
    folder = args.input.resolve()
    paths = {name: folder / name for name in ('RUNTIME.json', 'RUNTIME_ROWS.jsonl.gz',
        'RUNTIME_STARTED.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json', 'SCORING_RECEIPT.json')}
    paths.update(checker_code=Path(__file__).resolve(), accuracy_protocol=args.protocol.resolve(),
                 runtime_code=Path(__file__).with_name('runtime.py').resolve(),
                 panel=args.panel.resolve(), cohort=args.cohort.resolve(), authority=args.authority.resolve())
    runtime = read(folder / 'RUNTIME.json')
    for index, _ in enumerate(runtime['environment']['interference_snapshots']):
        paths['resource:%03d' % index] = folder / ('RUNTIME_RESOURCE_%03d.json' % index)
    return paths


def freeze(args):
    folder, output = guard(args, 'freeze')
    paths = input_paths(args)
    # Hash every byte-bound input before any row/statistic/projection arithmetic.
    bindings = {key: binding(path) for key, path in paths.items()}
    runtime, seal, inference, scoring, core = [read(paths[name]) for name in
        ('RUNTIME.json', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json', 'SCORING_RECEIPT.json', 'accuracy_protocol')]
    require(runtime['complete'] is True and runtime['status'] == 'DONE' and runtime['statistics_official'] is True,
            'completed official runtime required')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and seal['frames'] == 245 and
            seal['rows'] == 1960 and seal['fixed_rows'] == 490 and seal['methods'] == list(METHODS),
            'complete frozen GT-free geometry required')
    require(inference['complete'] is True and inference['cleanup_error'] is None and
            inference['inference_error'] is None and inference['actual_complete_frames'] == 245,
            'complete inference/cleanup required')
    require(scoring['complete'] is True and scoring['frames'] == 245 and core['methods'] == list(METHODS) and
            core['frames'] == 245, 'completed scoped scoring required')
    for key, expected in (('RUNTIME_ROWS.jsonl.gz', runtime['raw_rows']),
                          ('accuracy_protocol', runtime['protocol']),
                          ('accuracy_protocol', seal['protocol']),
                          ('GEOMETRY_SEAL.json', runtime['accuracy_seal']),
                          ('GEOMETRY_SEAL.json', scoring['geometry_seal']),
                          ('INFERENCE_RECEIPT.json', scoring['inference_receipt']),
                          ('runtime_code', runtime['runtime_code'])):
        require(same_binding(bindings[key], expected), 'finished receipt binding differs: ' + key)
    for key, frozen_name in (('runtime_code', 'runtime.py'), ('panel', 'runtime_panel'),
                             ('cohort', 'cohort'), ('authority', 'original_inputs')):
        require(same_binding(bindings[key], core['inputs'][frozen_name]), 'core authority binding differs: ' + key)
    write_new(output / OUTPUTS[0], dict(schema='supplemental_v7_runtime_scalar_protocol_v1',
        checker=bindings['checker_code'], inputs=bindings, limits=LIMITS,
        expected=dict(rows=1500, warmup_rows=200, measured_rows=1300, arms=list(ARMS),
            panel_frames=26, panel_sessions=13, warmup_per_arm=20, measured_per_arm=130,
            head_forward_entries=900, per_head_attempted_and_completed=300,
            summary_blocks=70, moment_slots=490),
        scalar_absolute_tolerance=ATOL, scalar_relative_tolerance=RTOL,
        schedule='20 warm blocks rotating all10 arms with odd reversal; 5 measured repeats, each26 images with the same repeat-dependent rotation/reversal',
        frozen_after_complete_accuracy_geometry_scoring_and_runtime=True,
        no_runtime_rows_or_moment_arithmetic_before_this_freeze=True))
    print('RUNTIME_SCALAR_FROZEN', binding(output / OUTPUTS[0])['sha256'], flush=True)


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [safe(v) for v in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False))
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


class Audit:
    def __init__(self):
        self.checks = 0
        self.failure_count = 0
        self.failures = []
        self.max_abs_difference = Counter()

    def ok(self, value, label, actual=None, expected=None):
        self.checks += 1
        if not value:
            self.failure_count += 1
            if len(self.failures) < 100:
                self.failures.append(dict(check=label, actual=safe(actual), expected=safe(expected)))

    def eq(self, actual, expected, label):
        self.ok(actual == expected, label, actual, expected)

    def near(self, actual, expected, label, scope='scalar'):
        if isinstance(expected, (tuple, list)):
            valid = isinstance(actual, (tuple, list)) and len(actual) == len(expected)
            self.ok(valid, label + '.shape', actual, expected)
            if valid:
                for index, (a, e) in enumerate(zip(actual, expected)):
                    self.near(a, e, label + '.' + str(index), scope)
        elif expected is None:
            self.eq(actual, None, label)
        else:
            valid = number(actual) and number(expected)
            difference = abs(actual - expected) if valid else None
            if difference is not None:
                self.max_abs_difference[scope] = max(self.max_abs_difference[scope], difference)
            self.ok(valid and difference <= ATOL + RTOL * abs(expected), label, actual, expected)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def primitive_counts(value, audit, label):
    require(isinstance(value, dict), 'primitive entry counters must be a mapping')
    # Production Counter serialization is sparse: a primitive which has not
    # entered the counter yet is absent, rather than an explicit zero.
    audit.ok(set(value).issubset(PRIMITIVES), label + '.no_unknown_primitive_keys', set(value), set(PRIMITIVES))
    result = {}
    for key in PRIMITIVES:
        count = value.get(key, 0)
        require(type(count) is int and count >= 0, 'invalid primitive entry counter')
        result[key] = count
    return result


def shaped(value, shape):
    if not shape:
        return number(value)
    return isinstance(value, (tuple, list)) and len(value) == shape[0] and all(
        shaped(v, shape[1:]) for v in value)


def ids(value):
    return isinstance(value, list) and all(type(k) is int and 0 <= k < 8 for k in value) and value == sorted(set(value))


def schedule():
    # Standalone implementation: never import or call the production scheduler.
    for phase in ('warmup', 'measured'):
        for block in range(20 if phase == 'warmup' else 5):
            rotation = block % 10
            order = [ARMS[(rotation + index) % 10] for index in range(10)]
            if block % 2:
                order = order[::-1]
            images = [block % 26] if phase == 'warmup' else range(26)
            for image in images:
                for position, arm in enumerate(order):
                    yield dict(phase=phase, arm=arm, arm_position=position, image_index=image,
                               **({'warmup_index': block} if phase == 'warmup' else {'repeat': block}))


def requested_head(arm):
    return next((head for head in HEADS if arm.startswith(head + '_')), None)


def compact_solver(solver):
    if solver is None:
        return None
    keys = (*FALSE_FLAGS, 'solver', 'available', 'pose_available', 'new_pose_estimated', 'no_pose',
            'fallback_used', 'state', 'hidden', 'excluded', 'temporary_excluded', 'eligible', 'used',
            'fit_input_ids', 'input_inliers', 'final_inliers', 'inliers', 'hidden_reprojected',
            'unresolved_ambiguity', 'global_uniqueness_proven', 'equivalence_uses_all_eight_model_predictions',
            'R_cf', 'R_physical', 'centroid', 'cf_extents', 'projected', 'residual_threshold_px')
    return {key: solver[key] for key in keys}


def compact_contract(contract):
    if contract is None:
        return None
    keys = ('heldout_pose_calls', 'boundary_only_native_distance_used_for_admission',
            'boundary_only_LOO_used_for_admission', 'native_N3_used_for_numeric_final_fit',
            'LOO_initial_prior_includes_heldout_influence', 'heldout_projection_is_observation',
            'actual_self_hidden_ids', 'final_self_hidden_reprojection_deferred',
            'final_reprojection_must_not_be_refit')
    result = {key: contract[key] for key in keys if key in contract}
    records = []
    for entry in contract.get('cornerwise_records', []):
        record = {key: entry[key] for key in ('id', 'heldout_pose_calls', 'fit_input_ids', 'final_inlier_ids',
            'actual_self_hidden_ids', 'projected_coordinate_used_as_observation', 'numeric_validation_pose_prior_used')}
        record['solver'] = compact_solver(entry['loo_solver']) if entry['heldout_pose_calls'] else None
        record['candidate_witness_records'] = len(entry['loo_solver']['all_candidate_solutions']) if entry['heldout_pose_calls'] else 0
        records.append(record)
    result['records'] = records
    if 'cornerwise_selection' in contract:
        summary = contract['cornerwise_selection']
        result['LOO_solve_calls'] = summary['LOO_solve_calls']
        result['summary_H'] = summary['actual_self_hidden_ids']
    return result


def compact_row(raw):
    # A single decoded row may contain large packets. Discard those references
    # immediately; retain no raw row population or candidate arrays.
    keys = ('phase', 'arm', 'warmup_index', 'repeat', 'arm_position', 'image_index', 'id', 'session',
            'final_points', 'hidden_initial', 'output_status', 'new_pose_estimated', 'fallback_used',
            'selected_index', 'GT_canary_active', 'primitive_entry_calls', 'parity_status',
            'actual_head_forward_entries', 'points_parity', 'pose_parity', 'raw_BASE_points_parity',
            'raw_N3_points_parity', 'candidate_center_score_box_preserved', 'observation_raw_logits_sha256')
    result = {key: raw[key] for key in keys if key in raw}
    result.update({key: value for key, value in raw.items() if key.endswith('_ms')})
    actual = raw['actual_pose']
    result['actual_pose'] = {key: actual[key] for key in ('available', 'R_cf', 'R_physical', 'centroid', 'cf_extents') if key in actual}
    result['solver'] = compact_solver(raw['solver'])
    result['candidate_witness_records'] = len(raw['solver']['all_candidate_solutions']) if raw['solver'] is not None else 0
    result['contract'] = compact_contract(raw['observation_contract'])
    selection = raw['cornerwise_selection']
    result['cornerwise_selection'] = ({key: selection[key] for key in ('LOO_solve_calls', 'actual_self_hidden_ids')}
                                    if selection is not None else None)
    # Absolute timing ticks are absent in the current writer. A future schema
    # must not silently inherit the current absence assertion.
    result['raw_timestamp_fields_present'] = any(key in raw for key in ('t0', 't1', 'ticks', 't0_ns', 't1_ns'))
    return result


def streamed_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'unexpected empty runtime row')
            raw = json.loads(line)
            row = compact_row(raw)
            del raw
            yield row


def projection(dimensions, rotation, translation, K):
    require(shaped(dimensions, (3,)) and min(dimensions) > 0 and shaped(rotation, (3, 3)) and
            shaped(translation, (3,)) and shaped(K, (3, 3)), 'invalid saved NEW pose/camera shape')
    a, b, c = [value / 2 for value in dimensions]
    corners = ((-a, -b, -c), (a, -b, -c), (a, b, -c), (-a, b, -c),
               (-a, -b, c), (a, -b, c), (a, b, c), (-a, b, c))
    result = []
    for point in corners:
        camera = [math.fsum(x * y for x, y in zip(axis, point)) + offset
                  for axis, offset in zip(rotation, translation)]
        homogeneous = [math.fsum(x * y for x, y in zip(axis, camera)) for axis in K]
        require(camera[2] > 1e-9 and homogeneous[2] > 0 and all(number(v) for v in homogeneous),
                'nonphysical saved NEW camera depth')
        result.append([homogeneous[0] / homogeneous[2], homogeneous[1] / homogeneous[2]])
    return result


def point_parity(record, audit, label):
    audit.eq(record['finite_support_equal'], True, label + '.finite_support_equal')
    audit.eq(record['atol_px'], 1e-7, label + '.fixed_atol')
    audit.eq(record['rtol'], 0, label + '.fixed_rtol')
    audit.ok(number(record['max_abs_px']) and 0 <= record['max_abs_px'] <= 1e-7,
             label + '.recorded_maximum_in_tolerance', record['max_abs_px'], '<=1e-7')


def pose_parity(record, available, audit, label):
    audit.eq(record['atol'], 1e-7, label + '.fixed_atol')
    audit.eq(record['rtol'], 0, label + '.fixed_rtol')
    fields = ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'reprojection_px') if available else ()
    audit.eq(set(record['max_abs_by_field']), set(fields), label + '.recorded_fields')
    for key, value in record['max_abs_by_field'].items():
        audit.ok(number(value) and 0 <= value <= 1e-7, label + '.' + key + '.recorded_maximum_in_tolerance', value, '<=1e-7')


def packet_contract(solver, H, temporary, audit, counts, label):
    for flag in FALSE_FLAGS:
        audit.eq(solver[flag], False, label + '.' + flag)
    audit.eq(solver['solver'], 'OBSERVATION_ONLY_FINITE_SUBSET_ROBUST', label + '.fixed_backend')
    audit.eq(solver['global_uniqueness_proven'], False, label + '.no_global_uniqueness_claim')
    audit.eq(solver['equivalence_uses_all_eight_model_predictions'], True, label + '.model_equivalence_only')
    audit.eq(solver['residual_threshold_px'], 8.0, label + '.fixed_consensus_radius')
    audit.eq(solver['hidden'], H, label + '.H_join')
    audit.eq(solver['temporary_excluded'], temporary, label + '.temporary_join')
    blocked = sorted(set(H) | set(temporary))
    audit.eq(solver['excluded'], blocked, label + '.exclusion_union')
    audit.ok(ids(solver['eligible']) and ids(solver['used']), label + '.eligible_and_used_ID_sets')
    audit.eq(solver['used'], [k for k in solver['eligible'] if k not in blocked], label + '.used_excludes_H_and_holdout')
    for key in ('fit_input_ids', 'input_inliers', 'final_inliers', 'inliers'):
        values = solver[key]
        audit.ok(ids(values) and set(values).issubset(solver['used']) and not set(values).intersection(blocked),
                 label + '.' + key + '.no_excluded_fit_or_consensus_ID', values, solver['used'])
    new = solver['available']
    audit.ok(type(new) is bool, label + '.available_is_bool')
    audit.eq(solver['pose_available'], new, label + '.pose_available')
    audit.eq(solver['new_pose_estimated'], new, label + '.new_flag')
    audit.eq(solver['no_pose'], not new, label + '.numeric_unavailable')
    audit.eq(solver['fallback_used'], False, label + '.solver_does_not_return_baseline')
    audit.eq(solver['state'] == 'NEW_POSE', new, label + '.NEW_state')
    if new:
        audit.ok(len(solver['used']) >= 4 and len(solver['final_inliers']) >= 4 and
                 solver['unresolved_ambiguity'] is False, label + '.accepted_consensus')
        audit.eq(solver['hidden_reprojected'], bool(H), label + '.H_reprojected_flag')
    counts['solver_state:' + solver['state']] += 1


def outcome(row, camera, audit, counts, label):
    H, solver, actual = row['hidden_initial'], row['solver'], row['actual_pose']
    new, fallback = row['new_pose_estimated'], row['fallback_used']
    audit.ok(ids(H), label + '.initial_H_ID_set')
    audit.ok(type(new) is bool and type(fallback) is bool and not (new and fallback), label + '.distinct_new_and_fallback')
    audit.ok(type(actual['available']) is bool, label + '.actual_available_bool')
    if row['arm'] in ('BASE', 'N3_SUBPIX'):
        audit.eq(solver, None, label + '.fixed_control_no_final_solver')
        audit.eq(H, [], label + '.fixed_control_no_H_replacement')
        audit.eq(new, False, label + '.fixed_control_not_new')
        audit.eq(fallback, False, label + '.fixed_control_not_fallback')
        audit.eq(row['output_status'], 'FRESH_FIXED_CONTROL', label + '.fixed_status')
    else:
        require(isinstance(solver, dict), 'research runtime row lacks final solver')
        packet_contract(solver, H, [], audit, counts, label + '.final')
        audit.eq(new, solver['available'], label + '.driver_new_is_solver_available')
        audit.eq(actual['available'], new or fallback, label + '.driver_available')
        status = 'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'
        audit.eq(row['output_status'], status, label + '.output_status')
        if row['arm'] == 'N3_INDEPENDENT_ROBUST_NO_MASK':
            audit.eq(H, [], label + '.explicit_no_mask')
        if new:
            for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents'):
                audit.eq(actual[key], solver[key], label + '.actual_driver_equals_solver_' + key)
            projected = projection(actual['cf_extents'], actual['R_cf'], actual['centroid'], camera)
            audit.near(solver['projected'], projected, label + '.Rt_projection', 'projection_px')
            require(isinstance(row['final_points'], list) and len(row['final_points']) == 9, 'final points shape differs')
            for corner in H:
                audit.near(row['final_points'][corner], projected[corner], label + '.H_replaced.' + str(corner), 'projection_px')
                counts['independent_hidden_point_projection_checks'] += 1
            counts['NEW_driver_Rt_joins'] += 1
    counts['output_status:' + row['arm'] + ':' + row['output_status']] += 1
    counts['stored_final_candidate_witness_records_not_recomputed'] += row['candidate_witness_records']
    contract = row['contract']
    selection = row['cornerwise_selection']
    if row['arm'].endswith('_BOUNDARY_ONLY'):
        require(isinstance(contract, dict), 'boundary route lacks recorded contract')
        for key in ('boundary_only_native_distance_used_for_admission', 'boundary_only_LOO_used_for_admission',
                    'native_N3_used_for_numeric_final_fit'):
            audit.eq(contract[key], False, label + '.' + key)
        audit.eq(contract['heldout_pose_calls'], 0, label + '.boundary_no_LOO')
        audit.eq(selection, None, label + '.boundary_no_selection_summary')
    elif row['arm'].endswith('_CORNERWISE_HYBRID'):
        require(isinstance(contract, dict) and isinstance(selection, dict), 'hybrid route lacks recorded LOO contract')
        for key in ('LOO_initial_prior_includes_heldout_influence', 'heldout_projection_is_observation'):
            audit.eq(contract[key], False, label + '.' + key)
        for key in ('final_self_hidden_reprojection_deferred', 'final_reprojection_must_not_be_refit'):
            audit.eq(contract[key], True, label + '.' + key)
        audit.eq(contract['actual_self_hidden_ids'], H, label + '.hybrid_H')
        calls = 0
        for index, record in enumerate(contract['records']):
            prefix = label + '.LOO.' + str(index)
            audit.ok(type(record['id']) is int and 0 <= record['id'] < 8, prefix + '.corner_ID')
            audit.eq(record['actual_self_hidden_ids'], H, prefix + '.H')
            audit.eq(record['projected_coordinate_used_as_observation'], False, prefix + '.projection_not_observation')
            audit.eq(record['numeric_validation_pose_prior_used'], False, prefix + '.no_initial_pose_prior')
            audit.ok(record['heldout_pose_calls'] in (0, 1) and type(record['heldout_pose_calls']) is int,
                     prefix + '.at_most_one_solve')
            calls += record['heldout_pose_calls']
            if record['heldout_pose_calls']:
                packet_contract(record['solver'], H, [record['id']], audit, counts, prefix + '.solver')
                audit.eq(record['fit_input_ids'], record['solver']['fit_input_ids'], prefix + '.fit_ID_join')
                audit.eq(record['final_inlier_ids'], record['solver']['final_inliers'], prefix + '.inlier_ID_join')
            counts['stored_LOO_candidate_witness_records_not_recomputed'] += record['candidate_witness_records']
        audit.ok(calls <= 8, label + '.LOO_max8')
        audit.eq(contract['heldout_pose_calls'], calls, label + '.contract_actual_LOO_calls')
        audit.eq(contract['LOO_solve_calls'], calls, label + '.contract_summary_LOO_calls')
        audit.eq(selection['LOO_solve_calls'], calls, label + '.driver_summary_LOO_calls')
        audit.eq(contract['summary_H'], H, label + '.contract_summary_H')
        audit.eq(selection['actual_self_hidden_ids'], H, label + '.driver_summary_H')
        counts['LOO_pose_paths'] += calls
        counts[requested_head(row['arm']) + '_LOO_pose_paths'] += calls


def panel_authority(paths, runtime, audit):
    panel_document, cohort = read(paths['panel']), read(paths['cohort'])
    frames = panel_document['frames']
    audit.eq(panel_document['count'], 26, 'panel.recorded_count')
    audit.eq(panel_document['sessions'], 13, 'panel.recorded_sessions')
    audit.eq(panel_document['selection_uses_accuracy'], False, 'panel.no_accuracy_selection')
    audit.eq(len(frames), 26, 'panel.frames')
    audit.eq(len({frame['frame_id'] for frame in frames}), 26, 'panel.unique_IDs')
    sessions = Counter(frame['session_id'] for frame in frames)
    audit.eq(len(sessions), 13, 'panel.unique_sessions')
    audit.ok(all(count == 2 for count in sessions.values()), 'panel.two_frames_each_session')
    audit.eq(panel_document['session_counts'], dict(sessions), 'panel.session_counts')
    audit.eq(runtime['panel'], [dict(id=frame['frame_id'], session=frame['session_id']) for frame in frames],
             'runtime.same_ordered_panel')
    eligible = set(cohort['ids'])
    audit.eq(len(eligible), 245, 'cohort.eligible245')
    audit.ok(all(frame['frame_id'] in eligible for frame in frames), 'panel.all_IDs_in_frozen_cohort')
    authority_document = read(paths['authority'])
    wanted = {frame['frame_id'] for frame in frames}
    # Do not retain historical cached point arrays, candidate metadata, or GT.
    authoritative = {}
    seen = set()
    for frame in authority_document['frames']:
        audit.ok(frame['id'] not in seen, 'authority.unique_ID.' + frame['id'])
        seen.add(frame['id'])
        if frame['id'] in wanted:
            authoritative[frame['id']] = {key: frame[key] for key in ('id', 'session', 'K', 'xyz', 'raw_hw', 'selected_index', 'image_sha256')}
    del authority_document
    audit.eq(len(seen), 319, 'authority.original_population319')
    audit.eq(set(authoritative), wanted, 'authority.all_panel_IDs')
    for frame in frames:
        original = authoritative[frame['frame_id']]
        label = 'panel.authority.' + frame['frame_id']
        audit.eq(frame['session_id'], original['session'], label + '.session')
        audit.eq(frame['camera_intrinsics'], original['K'], label + '.K_exact')
        audit.ok(shaped(original['K'], (3, 3)), label + '.finite_K')
        audit.eq(frame['dimensions_wdh_m'], [original['xyz'][0], original['xyz'][2], original['xyz'][1]], label + '.physical_WDH')
        audit.eq(frame['original_hw'], original['raw_hw'], label + '.image_shape')
        audit.eq(frame['input_selected_index'], original['selected_index'], label + '.selected_candidate')
        audit.eq(frame['image']['sha256'], original['image_sha256'], label + '.image_binding')
    return frames, authoritative


def receipts(paths, runtime, audit):
    core, seal, inference, scoring, journal = [read(paths[name]) for name in
        ('accuracy_protocol', 'GEOMETRY_SEAL.json', 'INFERENCE_RECEIPT.json', 'SCORING_RECEIPT.json', 'RUNTIME_STARTED.json')]
    audit.eq(runtime['schema'], 'fixed_three_head_point_observation_complete_runtime_v7', 'runtime.schema')
    for key in ('complete', 'statistics_official', 'model_initialization_complete', 'pipeline_close_complete',
                'image_decode_outside_intervals', 'runtime_rows_streamed_after_timed_interval',
                'only_timing_summary_fields_retained_in_memory', 'full_original_candidate_witnesses_preserved'):
        audit.eq(runtime[key], True, 'runtime.' + key)
    audit.eq(runtime['status'], 'DONE', 'runtime.status')
    audit.eq(runtime['arms'], list(ARMS), 'runtime.fixed10_arms')
    for key, expected in (('frames', 26), ('panel_sessions', 13), ('warmup_per_arm', 20), ('repeats', 5),
                          ('measured_per_arm', 130), ('configured_pipeline_calls', 1500), ('serialized_raw_rows', 1500),
                          ('actual_image_decode_calls', 26), ('new_training_updates', 0), ('new_GT_evaluations', 0)):
        audit.eq(runtime[key], expected, 'runtime.' + key)
    for key in ('cached_coordinate_replay_used_for_timing', 'GT_inference_access', 'other_benchmarks_parallel'):
        audit.eq(runtime[key], False, 'runtime.' + key)
    for key in ('head_hook_cleanup_errors', 'row_writer_close_errors', 'row_writer_preservation_errors'):
        audit.eq(runtime[key], [], 'runtime.' + key)
    boundaries = runtime['boundaries']
    for key in ('all_initial_and_final_PnP_included', 'all_observation_and_admission_work_included',
                'all_cornerwise_LOO_PnP_and_selection_included'):
        audit.eq(boundaries[key], True, 'runtime.boundaries.' + key)
    audit.eq(boundaries['excluded'], ['model/checkpoint load', 'RGB file/decode', 'GT scoring', 'parity checks',
                                     'durable journal', 'resource snapshots'], 'runtime.interval_exclusions')
    audit.eq(runtime['execution'], dict(pipeline_calls_started=1500, pipeline_calls_complete=1500), 'runtime.execution1500')
    audit.eq(journal['status'], 'DONE', 'runtime.journal.done')
    audit.eq(journal['counts'], runtime['execution'], 'runtime.journal.execution_join')
    audit.eq(journal['elapsed_seconds'], runtime['elapsed_seconds'], 'runtime.journal.elapsed_join')
    audit.ok(number(runtime['elapsed_seconds']) and runtime['elapsed_seconds'] > 0, 'runtime.positive_wall_time')
    audit.eq(core['methods'], list(METHODS), 'accuracy.fixed_methods')
    audit.eq(core['frames'], 245, 'accuracy.fixed245')
    audit.eq(seal['complete'], True, 'seal.complete')
    audit.eq(seal['GT_read_allowed'], False, 'seal.GT_free')
    audit.eq(inference['complete'], True, 'inference.complete')
    audit.eq(inference['cleanup_error'], None, 'inference.cleanup')
    audit.eq(inference['inference_error'], None, 'inference.error')
    audit.eq(inference['GT_access_during_inference'], False, 'inference.GT_free')
    audit.eq(inference['new_training_updates'], 0, 'inference.no_training')
    audit.eq(inference['new_RGB'], 0, 'inference.no_RGB_generation')
    audit.eq(inference['no_automatic_retry'], True, 'inference.no_retry')
    for key, rows in (('geometry', 1960), ('fixed', 490), ('observations', 735)):
        stream = inference['serialization']['streams'][key]
        audit.eq(stream['rows'], rows, 'inference.serialized.' + key)
        audit.eq(stream['published'], True, 'inference.published.' + key)
        audit.eq(stream['close_errors'], [], 'inference.close.' + key)
        audit.eq(stream['interruption_preservation_errors'], [], 'inference.preservation.' + key)
    audit.eq(scoring['complete'], True, 'scoring.complete')
    audit.eq(scoring['GT_access_only_after_complete_seal'], True, 'scoring.after_seal')
    for key in ('new_detector_forwards', 'new_head_forwards', 'new_pose_fits', 'new_rays'):
        audit.eq(scoring[key], 0, 'scoring.' + key)
    audit.eq(scoring['scoring_fitting_entries_forbidden'], True, 'scoring.fitting_forbidden')
    protection = runtime['prior_protection_after']
    for key in ('passed', 'source_checkout_unchanged'):
        audit.eq(protection[key], True, 'runtime.protection.' + key)
    audit.eq(protection['changed_protected_paths'], [], 'runtime.protection.changed_paths')
    audit.eq(protection, scoring['prior_tracked_and_user_checkout_protection'], 'runtime.same_recorded_protection_as_scoring')
    audit.ok(type(protection['protected_repository_files']) is int and protection['protected_repository_files'] > 0,
             'runtime.protection.positive_recorded_file_count')
    snapshots = runtime['environment']['interference_snapshots']
    audit.ok(len(snapshots) >= 7, 'runtime.resource_snapshots_include_endpoints_and_repeat_starts')
    audit.eq(snapshots[0]['phase'], 'before_models', 'runtime.resource_before_models')
    audit.eq(snapshots[-1]['phase'], 'after_complete', 'runtime.resource_after_complete')
    expected_jobs = list(schedule())
    job_set = {json.dumps(job, sort_keys=True) for job in expected_jobs}
    repeat_starts = {json.dumps(job, sort_keys=True) for job in expected_jobs
                     if job['phase'] == 'measured' and job['image_index'] == 0 and job['arm_position'] == 0}
    recorded_jobs = set()
    for index, snapshot in enumerate(snapshots):
        prefix = 'runtime.resource.' + str(index)
        audit.eq(read(paths['resource:%03d' % index]), snapshot, prefix + '.durable_snapshot_binding')
        for key in ('quiet', 'gpu_temperature_under_80'):
            audit.eq(snapshot[key], True, prefix + '.' + key)
        audit.eq(snapshot['system_changes'], False, prefix + '.system_changes')
        for key in ('foreign_gpu_pids', 'foreign_cpu_workloads'):
            audit.eq(snapshot[key], [], prefix + '.' + key)
        fields = snapshot['fields'].split(',')
        values = next(csv.reader([snapshot['gpu']], skipinitialspace=True))
        audit.eq(len(values), len(fields), prefix + '.recorded_GPU_columns')
        require(len(values) == len(fields), 'malformed recorded GPU snapshot')
        temperature = float(values[fields.index('temperature.gpu')].strip())
        audit.ok(number(temperature) and 0 <= temperature < 80, prefix + '.recorded_GPU_temperature_under80')
        if 'job' in snapshot:
            key = json.dumps(snapshot['job'], sort_keys=True)
            audit.ok(key in job_set, prefix + '.job_in_fixed_schedule')
            recorded_jobs.add(key)
    audit.ok(repeat_starts.issubset(recorded_jobs), 'runtime.resources.all5_repeat_start_snapshots')
    return core


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values):
    require(len(values) >= 2 and all(number(value) for value in values), 'insufficient/nonfinite stored timings')
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return dict(n=len(values), mean=mean, sample_variance=variance, sample_std=math.sqrt(variance),
                median=quantile(values, 0.5), P90=quantile(values, 0.9), maximum=max(values), unit='ms', ddof=1)


def run(args):
    folder, output = guard(args, 'run')
    protocol_path = output / OUTPUTS[0]
    protocol = read(protocol_path)
    paths = input_paths(args)
    actual_bindings = {key: binding(path) for key, path in paths.items()}
    require(actual_bindings == protocol['inputs'], 'supplemental frozen code/input bytes differ; no retry')
    require(protocol['schema'] == 'supplemental_v7_runtime_scalar_protocol_v1' and
            protocol['limits'] == LIMITS and protocol['scalar_absolute_tolerance'] == ATOL and
            protocol['scalar_relative_tolerance'] == RTOL and protocol['checker'] == actual_bindings['checker_code'],
            'supplemental checker protocol differs')
    own_protocol = binding(protocol_path)
    write_new(output / OUTPUTS[1], dict(status='STARTED', protocol=own_protocol,
        checker=actual_bindings['checker_code'], no_automatic_retry=True, limits=LIMITS))
    start = time.monotonic()
    audit, counts = Audit(), Counter()
    timings = {arm: {stage: [] for stage in SUMMARY_STAGES} for arm in ARMS}
    primitive_totals = Counter({key: 0 for key in PRIMITIVES})
    head_totals = {head: Counter(attempted=0, completed=0) for head in HEADS}
    phases = Counter()
    arm_phases = Counter()
    recomputed = {}
    finished = False
    failure = None
    try:
        runtime = read(paths['RUNTIME.json'])
        core = receipts(paths, runtime, audit)
        for name, expected in (('accuracy_protocol', runtime['protocol']), ('accuracy_protocol', read(paths['GEOMETRY_SEAL.json'])['protocol']),
                               ('GEOMETRY_SEAL.json', runtime['accuracy_seal']), ('runtime_code', runtime['runtime_code']),
                               ('RUNTIME_ROWS.jsonl.gz', runtime['raw_rows'])):
            audit.ok(same_binding(actual_bindings[name], expected), 'runtime.bound.' + name)
        for name, key in (('runtime_code', 'runtime.py'), ('panel', 'runtime_panel'), ('cohort', 'cohort'), ('authority', 'original_inputs')):
            audit.ok(same_binding(actual_bindings[name], core['inputs'][key]), 'runtime.core_input.' + name)
        frames, authority = panel_authority(paths, runtime, audit)
        jobs = iter(schedule())
        for index, row in enumerate(streamed_rows(paths['RUNTIME_ROWS.jsonl.gz'])):
            label = 'row.' + str(index)
            job = next(jobs, None)
            require(job is not None, 'more than1500 runtime rows')
            identity = {key: row[key] for key in ('phase', 'arm', 'warmup_index', 'repeat', 'arm_position', 'image_index') if key in row}
            audit.eq(identity, job, label + '.exact_rotating_schedule')
            require(row['arm'] in ARMS and row['phase'] in ('warmup', 'measured'), 'unknown runtime arm/phase')
            frame = frames[job['image_index']]
            audit.eq(row['id'], frame['frame_id'], label + '.ordered_panel_ID')
            audit.eq(row['session'], frame['session_id'], label + '.ordered_panel_session')
            canonical = authority[frame['frame_id']]
            audit.eq(row['selected_index'], canonical['selected_index'], label + '.preserved_candidate_index')
            audit.eq(row['raw_timestamp_fields_present'], False, label + '.absolute_timestamps_not_serialized')
            audit.eq(row['parity_status'], 'PASS', label + '.recorded_parity_PASS')
            audit.eq(row['GT_canary_active'], True, label + '.recorded_GT_canary')
            audit.eq(row['candidate_center_score_box_preserved'], True, label + '.recorded_center_score_box_preservation')
            counts['recorded_center_score_box_preservation_flags'] += 1
            for field in ('points_parity', 'raw_BASE_points_parity'):
                point_parity(row[field], audit, label + '.' + field)
            if row['arm'] == 'BASE':
                audit.ok('raw_N3_points_parity' not in row, label + '.BASE_no_N3_parity_record')
            else:
                point_parity(row['raw_N3_points_parity'], audit, label + '.raw_N3_points_parity')
            pose_parity(row['pose_parity'], row['actual_pose']['available'], audit, label + '.pose_parity')
            expected_head = requested_head(row['arm'])
            expected_entries = {head: dict(attempted=int(head == expected_head), completed=int(head == expected_head)) for head in HEADS}
            audit.eq(row['actual_head_forward_entries'], expected_entries, label + '.requested_head_pre_post')
            for head in HEADS:
                for entry in ('attempted', 'completed'):
                    value = row['actual_head_forward_entries'][head][entry]
                    require(type(value) is int and value >= 0, 'invalid head entry counter')
                    head_totals[head][entry] += value
            logits = row['observation_raw_logits_sha256']
            if expected_head is None:
                audit.eq(logits, None, label + '.native_control_no_logits')
            else:
                audit.ok(isinstance(logits, str) and len(logits) == 64 and all(c in '0123456789abcdef' for c in logits),
                         label + '.requested_head_logits_digest')
            entries = primitive_counts(row['primitive_entry_calls'], audit, label + '.primitive_counters')
            for key in PRIMITIVES:
                primitive_totals[key] += entries[key]
            audit.eq({key for key in row if key.endswith('_ms')}, {stage + '_ms' for stage in SUMMARY_STAGES},
                     label + '.exact7_timing_fields')
            for stage in SUMMARY_STAGES:
                value = row[stage + '_ms']
                require(number(value) and value >= 0, 'invalid stored timing')
                audit.ok(value > 0, label + '.positive_' + stage + '_ms', value, '>0')
                if row['phase'] == 'measured':
                    timings[row['arm']][stage].append(value)
            partition = math.fsum(row[stage + '_ms'] for stage in (*STAGES, 'return'))
            audit.near(row['full_ms'], partition, label + '.stored_partition_consistency_only', 'partition_ms')
            outcome(row, canonical['K'], audit, counts, label)
            phases[row['phase']] += 1
            arm_phases[(row['arm'], row['phase'])] += 1
            counts['rows_checked'] += 1
            if counts['rows_checked'] % 250 == 0:
                print('RUNTIME_SCALAR_ROWS', counts['rows_checked'], 1500, flush=True)
        audit.eq(next(jobs, None), None, 'runtime.no_missing_schedule_rows')
        audit.eq(counts['rows_checked'], 1500, 'runtime.raw_exact1500')
        audit.eq(dict(phases), dict(warmup=200, measured=1300), 'runtime.raw_phases')
        for arm in ARMS:
            audit.eq(arm_phases[(arm, 'warmup')], 20, arm + '.warm20')
            audit.eq(arm_phases[(arm, 'measured')], 130, arm + '.measured130')
            audit.eq(runtime['warmup_accounting'][arm], dict(configured=20, completed=20), arm + '.warm_receipt')
        audit.eq(set(runtime['warmup_accounting']), set(ARMS), 'runtime.warm_receipt_exact10arms')
        receipt_entries = primitive_counts(runtime['actual_OpenCV_entry_calls'], audit, 'runtime.primitive_receipt')
        audit.eq(dict(primitive_totals), receipt_entries, 'runtime.all4_primitive_row_sums_equal_receipt')
        actual_heads = {head: dict(values) for head, values in head_totals.items()}
        expected_heads = {head: dict(attempted=300, completed=300) for head in HEADS}
        audit.eq(actual_heads, expected_heads, 'runtime.raw_each_head300_pre_post')
        audit.eq(actual_heads, runtime['actual_head_forward_entries'], 'runtime.raw_head_entries_equal_receipt')
        audit.eq(math.fsum(value['attempted'] for value in actual_heads.values()), 900, 'runtime.total900_attempted_head_entries')
        audit.eq(math.fsum(value['completed'] for value in actual_heads.values()), 900, 'runtime.total900_completed_head_entries')
        expected_calls = dict(detector_calls=1500, initial_pose_calls=1500, N3_route_calls=1350,
            feature_initial_pose_calls=900, shared_feature_tensors=900, head_calls=900, final_pose_paths=1200,
            LOO_pose_paths=counts['LOO_pose_paths'])
        for head in HEADS:
            expected_calls[head + '_head_calls'] = 300
            expected_calls[head + '_LOO_pose_paths'] = counts[head + '_LOO_pose_paths']
            for supply in ('BOUNDARY_ONLY', 'CORNERWISE_HYBRID'):
                expected_calls[head + '_' + supply + '_final_pose_paths'] = 150
        audit.eq(runtime['actual_pipeline_calls'], expected_calls, 'runtime.complete_recorded_pipeline_counts_join_schedule_and_LOO_rows')
        forwards = runtime['model_forwards']
        audit.eq(forwards['N3'], 1350, 'runtime.recorded_N3_forwards')
        audit.eq(forwards['heads'], {head: 300 for head in HEADS}, 'runtime.recorded_head_forwards')
        audit.eq(forwards['detector_internal_initialization_calls'], 1, 'runtime.recorded_single_detector_initialization_forward')
        audit.eq(forwards['detector'], 1500 + forwards['detector_internal_initialization_calls'], 'runtime.recorded_detector_full_plus_initialization')
        audit.eq(set(runtime['summaries']), set(ARMS), 'runtime.summary_exact10arms')
        for arm in ARMS:
            audit.eq(set(runtime['summaries'][arm]), set(SUMMARY_STAGES), arm + '.summary_exact7stages')
            recomputed[arm] = {}
            for stage in SUMMARY_STAGES:
                values = timings[arm][stage]
                audit.eq(len(values), 130, arm + '.' + stage + '.raw_measured_count')
                independent = describe(values)
                recomputed[arm][stage] = independent
                stored = runtime['summaries'][arm][stage]
                audit.eq(set(stored), set(MOMENTS) | {'unit', 'ddof'}, arm + '.' + stage + '.summary_exact_fields')
                for field in MOMENTS:
                    label = 'summary.' + arm + '.' + stage + '.' + field
                    if field == 'n':
                        audit.eq(stored[field], independent[field], label)
                    else:
                        audit.near(stored[field], independent[field], label, 'moment_' + field)
                    counts['moment_slots_checked'] += 1
                audit.eq(stored['unit'], 'ms', arm + '.' + stage + '.unit')
                audit.eq(stored['ddof'], 1, arm + '.' + stage + '.ddof1')
                counts['summary_blocks_checked'] += 1
        audit.eq(counts['moment_slots_checked'], 490, 'runtime.all490_moment_slots')
        audit.eq(counts['summary_blocks_checked'], 70, 'runtime.all70_summary_blocks')
        audit.eq(counts['recorded_center_score_box_preservation_flags'], 1500, 'runtime.all1500_recorded_center_preservation_flags')
        audit.eq({key: binding(path) for key, path in paths.items()}, protocol['inputs'], 'runtime.all_frozen_inputs_unchanged_after_math')
        audit.eq(binding(protocol_path), own_protocol, 'runtime.own_protocol_unchanged_after_math')
        finished = True
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        audit.ok(False, 'runtime.scalar_exception', failure, 'all required checks complete')
    result = dict(schema='supplemental_v7_runtime_scalar_checks_v1', complete=finished,
        passed=finished and audit.failure_count == 0, protocol=own_protocol,
        checker=actual_bindings['checker_code'], inputs=actual_bindings, limits=LIMITS,
        checks=audit.checks, failure_count=audit.failure_count, first_failure=audit.failures[0] if audit.failures else None,
        failures_first100=audit.failures, exception=failure, counts=dict(counts), phase_counts=dict(phases),
        per_arm_phase_counts={arm: {phase: arm_phases[(arm, phase)] for phase in ('warmup', 'measured')} for arm in ARMS},
        actual_primitive_entry_sums=dict(primitive_totals), actual_head_entry_sums={key: dict(value) for key, value in head_totals.items()},
        maximum_absolute_differences=dict(audit.max_abs_difference), recomputed_original_full_and_stage_summaries=recomputed,
        stored_timing_scalars_retained=sum(len(values) for group in timings.values() for values in group.values()),
        full_runtime_rows_retained=0, full_candidate_witness_arrays_retained=0,
        elapsed_seconds=time.monotonic() - start,
        no_actual_model_image_GT_PnP_training_ray_Jacobian_SVD_calls=True)
    # Exclusive creation preserves the first failed or successful run. Never
    # overwrite a receipt, journal, protocol or raw stream.
    write_new(output / OUTPUTS[2], result)
    print('RUNTIME_SCALAR_CHECKS', 'PASS' if result['passed'] else 'FAIL', audit.checks,
          counts['rows_checked'], counts['moment_slots_checked'], flush=True)
    return 0 if result['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('freeze', 'run'))
    parser.add_argument('--input', type=Path, default=DOC)
    parser.add_argument('--output', type=Path, default=DOC)
    parser.add_argument('--protocol', type=Path, default=DOC / 'PROTOCOL.json')
    parser.add_argument('--panel', type=Path, default=CORRECTED / 'RUNTIME_PANEL.json')
    parser.add_argument('--cohort', type=Path, default=CORRECTED / 'COHORT.json')
    parser.add_argument('--authority', type=Path, default=REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1/INPUTS.json')
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args)
        return 0
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
