"""Own-frozen, standard-library audit of a completed fresh V9 runtime.

No production imports, model, image, GT, optimizer, PnP, ray, Jacobian or SVD
calls. Stream one full row, retain only timing scalars and counter totals.
Absolute clock ticks were not stored: check duration partition arithmetic, not
historical hardware authenticity. Full candidate math has its separate audit.
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
import time
import traceback

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / '_docs/experiments/pallet_sparse_local_line_20261010_v9'
PRIVATE = Path('/tmp/pallet-sparse-local-line-private-20261010-v9')
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
OLD = REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1'
PRIMARY = 'ROLE_BOUNDARY_LOCAL_POINT_LINE'
COMPARATOR = 'ROLE_BOUNDARY_H_ROBUST'
ARMS = ('BASE', 'N3_SUBPIX', COMPARATOR, PRIMARY)
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
          'final_pose_reprojection_and_metadata')
SUMMARY_STAGES = ('full', *STAGES, 'return')
MOMENTS = ('n', 'mean', 'sample_variance', 'sample_std', 'median', 'P90', 'maximum')
PRIMITIVES = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix')
OPTIMIZER = ('actual_SciPy_least_squares_entries', 'actual_SciPy_least_squares_returns',
             'actual_SciPy_least_squares_exceptions', 'actual_optimizer_residual_entries',
             'actual_optimizer_jacobian_entries')
OPTIMIZER_JOIN = dict(actual_SciPy_least_squares_entries='local_optimizer_calls',
    actual_SciPy_least_squares_returns='local_optimizer_completions',
    actual_optimizer_residual_entries='optimizer_residual_calls',
    actual_optimizer_jacobian_entries='optimizer_jacobian_calls')
POSE_FIELDS = ('R_cf', 'R_physical', 'centroid', 'cf_extents')
POSE_OPTIONAL_FIELDS = ('projected', 'reprojection_px', 'sse_px2', 'soft_l1_cost',
    'factor_sse_px2', 'truncated_sse_px2', 'residuals_used_px', 'line_residual_RMS_px')
RANK = 1e-10
CHECKPOINT = '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d'
OUTPUTS = ('RUNTIME_VALIDATION_PROTOCOL.json', 'RUNTIME_VALIDATION_STARTED.json',
           'RUNTIME_VALIDATION_CHECKS.json')
ATOL, RTOL = 1e-7, 1e-12
LIMITS = dict(supplemental_after_accuracy_and_runtime=True,
    original_failed_runtime_attempt_preserved=True, accuracy_runtime_policy_changes=0,
    new_model_image_GT_optimizer_PnP_ray_Jacobian_SVD_calls=0,
    full_ms_is_original_not_sum_derived=True, raw_absolute_clock_ticks_available=False,
    duration_check='Stored independent full_ms versus stored five stage durations and return duration.',
    hardware_authenticity_independently_certified=False,
    resource_snapshots_exclude_all_transient_interference=False,
    saved_accuracy_parity_is_record_validation_not_model_replay=True,
    candidate_math='Stored witness presence, pose/fit/operation joins only; candidate arithmetic belongs to completed geometry validation.',
    stored_SVD_singular_values_recount_only=True,
    local_rank6_is_not_global_unique_pose=True,
    selected_metadata='Fresh original detector candidates versus fresh returned prediction, plus public INPUTS selected metadata.',
    camera_authority='Byte-bound original INPUTS, joined to ordered frozen panel and scoped COHORT.',
    protection='Recorded before/after snapshots; no independent historical filesystem monitor.',
    automatic_retry=False)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def binding(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'missing/symlink input: ' + str(path))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    absolute = path.resolve()
    return dict(path=str(absolute.relative_to(REPO)) if absolute.is_relative_to(REPO) else path.name,
                sha256=h.hexdigest(), bytes=path.stat().st_size)


def same_binding(actual, expected):
    return isinstance(expected, dict) and all(actual[k] == expected.get(k) for k in ('sha256', 'bytes'))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((safe(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush()


def guard(args, stage):
    for path in (args.input, args.output, args.accuracy_protocol, args.runtime_protocol,
                 args.panel, args.cohort, args.authority, args.runtime_code, *args.history):
        require(not any(p.is_symlink() for p in (path, *path.parents)), 'symlink ancestry: ' + str(path))
    require(args.input.is_dir() and args.output.is_dir(), 'input/output directories must already exist')
    output = args.output.resolve()
    require(output == DOC.resolve() or output.is_relative_to(PRIVATE.resolve()), 'exclusive V9 DOC/private output only')
    for name in OUTPUTS if stage == 'freeze' else OUTPUTS[1:]:
        require(not (output / name).exists(), 'preserve first/existing ' + name)
    if stage == 'run':
        require((output / OUTPUTS[0]).is_file(), 'own supplemental freeze required')
    return args.input.resolve(), output


def input_paths(args):
    folder = args.input.resolve()
    paths = dict(runtime=folder/args.runtime_name, runtime_rows=folder/args.rows_name,
        runtime_started=folder/args.started_name, runtime_preflight=folder/args.preflight_name,
        runtime_protocol=args.runtime_protocol.resolve(), accuracy_protocol=args.accuracy_protocol.resolve(),
        geometry_seal=folder/'GEOMETRY_SEAL.json', control_receipt=folder/'CONTROL_RECEIPT.json',
        scoring_receipt=folder/'SCORING_RECEIPT.json', geometry_validation=folder/'VALIDATION_CHECKS.json',
        geometry_validation_protocol=folder/'VALIDATION_PROTOCOL.json',
        accuracy_geometry=folder/'GEOMETRY_SEALED.jsonl.gz', accuracy_ledgers=folder/'CONTROL_LEDGERS.jsonl.gz',
        parent_population_checks=folder/'PARENT_POPULATION_CHECKS.json',
        panel=args.panel.resolve(), cohort=args.cohort.resolve(), authority=args.authority.resolve(),
        checker_code=Path(__file__).resolve(), runtime_code=args.runtime_code.resolve(),
        original_runtime_code=Path(__file__).with_name('runtime.py'),
        original_runtime_protocol=folder/'RUNTIME_PROTOCOL.json',
        original_runtime_preflight=folder/'RUNTIME_PREFLIGHT.json',
        original_CLI_attempt=folder/'RUNTIME_CLI_ATTEMPT_A.json',
        deployment_code=Path(__file__).with_name('deployment.py'),
        runtime_contract=DOC/'RUNTIME_CONTRACT_KO.md', review_contract=DOC/'RUNTIME_REVIEW_CONTRACT_KO.md',
        frame_helper=REPO/'scripts/research/pallet_boundary_corner_refiner_20261010_v2/common.py')
    for name in ('solver.py','pipeline.py','validation_checks.py'):
        paths['accuracy_code:'+name]=Path(__file__).with_name(name)
    runtime = read(paths['runtime'])
    for index, _ in enumerate(runtime['resource_snapshots']):
        paths['resource:%03d' % index] = folder / ('%s%03d.json' % (args.resource_prefix, index))
    for index, path in enumerate(args.history):
        paths['original_failure_history:%03d' % index] = path.resolve()
    return paths


def freeze(args):
    _, output = guard(args, 'freeze')
    paths = input_paths(args)
    bindings = {key: binding(path) for key, path in paths.items()}
    r, seal, control, score, validation = (read(paths[key]) for key in
        ('runtime', 'geometry_seal', 'control_receipt', 'scoring_receipt', 'geometry_validation'))
    require(r['complete'] is True and r['status'] == 'DONE' and r['statistics_official'] is True,
            'completed official fresh runtime required')
    require(seal['complete'] is True and seal['GT_read_allowed'] is False and
        seal['frames'] == seal['rows'] == seal['ledger_rows'] == 245, 'complete GT-free V9 seal required')
    require(control['complete'] is True and control['environment_and_monkeypatch_cleanup_completed'] is True and
        not control['error'] and not control['cleanup_error'], 'completed V9 cleanup required')
    require(score['complete'] is True and score['actual_total_scored_rows'] == 980 and
        validation['complete'] is True and validation['passed'] is True, 'completed score and scalar geometry gate required')
    for key, expected in (('runtime_rows', r['raw_rows']), ('runtime_protocol', r['runtime_protocol']),
        ('accuracy_protocol', r['accuracy_protocol']), ('geometry_seal', r['accuracy_seal']),
        ('geometry_seal', score['geometry_seal']), ('runtime_code', r['runtime_code']),
        ('deployment_code', r['deployment_code'])):
        require(same_binding(bindings[key], expected), 'completed binding differs: ' + key)
    write_new(output/OUTPUTS[0], dict(schema='supplemental_sparse_local_line_runtime_scalar_protocol_v9',
        checker=bindings['checker_code'], inputs=bindings, limits=LIMITS,
        expected=dict(rows=600, warmup_rows=80, measured_rows=520, arms=list(ARMS),
            warmup_per_arm=20, measured_per_arm=130, panel_frames=26, panel_sessions=13,
            IMAGE_ROLE_attempted_and_completed=300, summary_blocks=28, moment_slots=196),
        scalar_absolute_tolerance=ATOL, scalar_relative_tolerance=RTOL,
        own_freeze_after_completed_runtime_before_own_row_arithmetic=True,
        no_automatic_retry=True))
    print('V9_RUNTIME_SCALAR_FROZEN', binding(output/OUTPUTS[0])['sha256'], flush=True)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def shaped(value, shape):
    if not shape:
        return number(value)
    return isinstance(value, list) and len(value) == shape[0] and all(shaped(v, shape[1:]) for v in value)


def ids(value):
    return isinstance(value, list) and all(type(k) is int and 0 <= k < 8 for k in value) and value == sorted(set(value))


class Audit:
    def __init__(self):
        self.checks = 0; self.failure_count = 0; self.failures = []; self.max_abs_difference = Counter()

    def ok(self, value, label, actual=None, expected=None):
        self.checks += 1
        if not value:
            self.failure_count += 1
            if len(self.failures) < 100:
                self.failures.append(dict(check=label, actual=brief(actual), expected=brief(expected)))

    def eq(self, actual, expected, label):
        self.ok(actual == expected, label, actual, expected)

    def near(self, actual, expected, label, scope='scalar'):
        if isinstance(expected, dict):
            valid = isinstance(actual, dict) and set(actual) == set(expected)
            self.ok(valid, label+'.keys', actual, expected)
            if valid:
                for key in expected:
                    self.near(actual[key], expected[key], label+'.'+key, scope)
        elif isinstance(expected, list):
            valid = isinstance(actual, list) and len(actual) == len(expected)
            self.ok(valid, label+'.shape', actual, expected)
            if valid:
                for index, (a, e) in enumerate(zip(actual, expected)):
                    self.near(a, e, label+'.'+str(index), scope)
        elif expected is None or isinstance(expected, (bool, str)):
            self.eq(actual, expected, label)
        else:
            valid = number(actual) and number(expected)
            difference = abs(actual-expected) if valid else None
            if difference is not None:
                self.max_abs_difference[scope] = max(self.max_abs_difference[scope], difference)
            self.ok(valid and difference <= ATOL+RTOL*abs(expected), label, actual, expected)


def brief(value):
    """Failure metadata never retains a complete candidate/row population."""
    if isinstance(value, dict) and len(value) > 20:
        return dict(type='dict', keys=sorted(str(k) for k in value), payload_not_retained=True)
    if isinstance(value, (list, tuple, set, frozenset)) and len(value) > 24:
        return dict(type=type(value).__name__, length=len(value), payload_not_retained=True)
    if isinstance(value, dict):
        return {str(k):brief(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [brief(v) for v in value]
    return safe(value)


def counts(value, known, audit, label):
    require(isinstance(value, dict), 'counter mapping required: '+label)
    audit.ok(set(value) <= set(known), label+'.no_unknown_keys', set(value), set(known))
    result = {}
    for key in known:
        n = value.get(key, 0)
        require(type(n) is int and n >= 0, 'nonnegative integer counter required: '+label+'.'+key)
        result[key] = n
    return result


def schedule():
    # Standalone schedule; never call the production scheduler.
    for phase in ('warmup', 'measured'):
        for block in range(20 if phase == 'warmup' else 5):
            rotation = block % 4
            order = [ARMS[(rotation+i) % 4] for i in range(4)]
            if block % 2:
                order.reverse()
            for image in ([block % 26] if phase == 'warmup' else range(26)):
                for position, arm in enumerate(order):
                    yield dict(phase=phase, arm=arm, arm_position=position, image_index=image,
                        **({'warmup_index':block} if phase == 'warmup' else {'repeat':block}))


def projection(extents, rotation, centroid, K):
    require(shaped(extents, (3,)) and min(extents) > 0 and shaped(rotation, (3,3)) and
        shaped(centroid, (3,)) and shaped(K, (3,3)), 'saved NEW physical pose/camera required')
    a,b,c = [x/2 for x in extents]
    corners = ((-a,-b,-c),(a,-b,-c),(a,b,-c),(-a,b,-c),
               (-a,-b,c),(a,-b,c),(a,b,c),(-a,b,c))
    result = []
    for point in corners:
        cam = [math.fsum(x*y for x,y in zip(axis, point))+offset for axis,offset in zip(rotation,centroid)]
        hom = [math.fsum(x*y for x,y in zip(axis,cam)) for axis in K]
        require(cam[2] > 1e-9 and hom[2] > 0 and all(number(v) for v in hom), 'nonphysical saved NEW depth')
        result.append([hom[0]/hom[2], hom[1]/hom[2]])
    return result


def point_parity(record, audit, label):
    audit.eq(record['finite_support_equal'], True, label+'.finite_support_equal')
    audit.eq(record['atol_px'], 1e-7, label+'.fixed_atol')
    audit.eq(record['rtol'], 0, label+'.fixed_rtol')
    audit.ok(number(record['max_abs_px']) and 0 <= record['max_abs_px'] <= 1e-7,
             label+'.stored_difference', record['max_abs_px'], '<=1e-7')


def pose_parity(record, actual, audit, label):
    audit.eq(record['atol'], 1e-7, label+'.fixed_atol'); audit.eq(record['rtol'], 0, label+'.fixed_rtol')
    require(isinstance(record['max_abs_by_field'], dict), 'stored pose parity mapping required')
    audit.eq(record['required_fields'], list(POSE_FIELDS), label+'.required_fields')
    audit.eq(record['missing_LOCAL_reprojection_px_not_fabricated'], True, label+'.no_fabricated_metric')
    audit.eq(record['post_interval_only'], True, label+'.after_interval')
    optional = [key for key in POSE_OPTIONAL_FIELDS if key in actual] if actual['available'] else []
    audit.eq(record['checked_optional_fields'], optional, label+'.natural_optional_fields')
    expected = set() if not actual['available'] else set(POSE_FIELDS) | set(optional)
    fields = set(record['max_abs_by_field'])
    audit.eq(fields, expected, label+'.actual_schema_fields')
    if not actual['available']:
        audit.eq(fields, set(), label+'.unavailable_no_numeric_parity')
    for key,value in record['max_abs_by_field'].items():
        audit.ok(key in actual, label+'.field_exists.'+key)
        audit.ok(number(value) and 0 <= value <= 1e-7, label+'.stored_difference.'+key, value, '<=1e-7')


def authority(paths, runtime, rp, audit):
    paneldoc, cohort, inputs = (read(paths[key]) for key in ('panel','cohort','authority'))
    panel = paneldoc['frames']; wanted = {f['frame_id'] for f in panel}
    audit.eq(len(panel),26,'panel.count'); audit.eq(len(wanted),26,'panel.unique_IDs')
    sessions = Counter(f['session_id'] for f in panel)
    audit.eq(len(sessions),13,'panel.sessions'); audit.eq(set(sessions.values()),{2},'panel.two_each_session')
    audit.eq(paneldoc['selection_uses_accuracy'],False,'panel.no_accuracy_selection')
    audit.eq(runtime['panel'],[dict(id=f['frame_id'],session=f['session_id']) for f in panel],'runtime.ordered_panel')
    eligible = {f['id']:f for f in cohort['frames']}
    audit.eq(len(eligible),245,'cohort.eligible245')
    audit.eq(Counter(f['label'] for f in eligible.values()),Counter(clean=153,moderate=92),'cohort.easy_medium')
    audit.ok(wanted <= set(eligible),'panel.no_severe_ID')
    # INPUTS contains cached RGB points/metadata, not reference GT. Keep only26.
    seen, selected, normalized = set(), {}, []
    object_types = {f['session_id']:f['object_type'] for f in panel}
    for f in inputs['frames']:
        require(f['id'] not in seen, 'duplicate original INPUTS frame')
        seen.add(f['id'])
        if f['id'] in eligible:
            cohort_frame = eligible[f['id']]
            audit.eq(f['session'],cohort_frame['session'],'authority.cohort_session.'+f['id'])
            audit.eq(f['image'],cohort_frame['image']['path'],'authority.cohort_path.'+f['id'])
            audit.eq(f['image_sha256'],cohort_frame['image']['sha256'],'authority.cohort_hash.'+f['id'])
            normalized.append(dict(id=f['id'],session=f['session'],label=cohort_frame['label'],
                image=cohort_frame['image'],raw_hw=f['raw_hw'],K=f['K'],xyz=f['xyz'],
                object_type=object_types[f['session']]))
        if f['id'] in wanted:
            selected[f['id']] = {key:f[key] for key in ('id','session','image','image_sha256',
                'K','xyz','raw_hw','points','selected_index','candidate_metadata','prediction_support')}
    del inputs
    audit.eq(len(seen),319,'authority.original319'); audit.eq(set(selected),wanted,'authority.all26')
    for f in panel:
        ref = selected[f['frame_id']]; label='panel.'+f['frame_id']
        audit.eq(f['session_id'],ref['session'],label+'.session')
        audit.eq(eligible[ref['id']]['session'],ref['session'],label+'.cohort_session')
        audit.eq(f['camera_intrinsics'],ref['K'],label+'.K')
        audit.eq(f['original_hw'],ref['raw_hw'],label+'.hw')
        audit.eq(f['input_selected_index'],ref['selected_index'],label+'.selection')
        audit.eq(f['image']['sha256'],ref['image_sha256'],label+'.image_hash')
        audit.eq(eligible[ref['id']]['image']['sha256'],ref['image_sha256'],label+'.cohort_image_hash')
        audit.eq(f['dimensions_wdh_m'],[ref['xyz'][0],ref['xyz'][2],ref['xyz'][1]],label+'.registered_dimensions')
    by_id={f['id']:f for f in normalized}
    expected_identities=[by_id[k] for k in cohort['ids']]
    proof=runtime['frame_input_normalization']
    audit.eq(proof,rp['frame_input_normalization'],'runtime.same_full245_normalization')
    audit.eq(proof['schema'],'authoritative_runtime_frame_join_v9_review','runtime.normalization_schema')
    for key,value in (('complete',True),('frames',245),('authority_frames',319),('clean',153),
                      ('moderate',92),('severe_excluded',74),('other_authority_fields_forwarded',False),
                      ('GT_pose_or_labels_forwarded',False),('saved_predictions_used_only_after_complete_interval',True)):
        audit.eq(proof[key],value,'runtime.normalization.'+key)
    audit.eq(proof['ids'],cohort['ids'],'runtime.full245_identity_order')
    audit.eq(proof['joined_frame_identities'],expected_identities,'runtime.full245_identity_values')
    for key,alias in (('cohort','cohort'),('original_inputs','authority'),('helper_code','frame_helper')):
        audit.ok(same_binding(proof[key],binding(paths[alias])),'runtime.normalization_bound.'+key)
    audit.eq(proof['inference_input_fields'],['RAM_RGB','K','xyz','metadata.id','metadata.session','metadata.object_type'],
        'runtime.normalization_only_fresh_inputs')
    del normalized,by_id,expected_identities
    return panel, selected


def metadata(row, output, ref, audit, tally, label):
    audit.eq(row['original_detector_prediction_recorded_after_complete_interval'], True, label+'.fresh_witness_saved_after_interval')
    original = row['original_fresh_detector_prediction']
    returned = output['prediction']
    audit.eq(original['selected_index'],returned['selected_index'],label+'.selection_preserved')
    audit.eq(row['selected_index'],original['selected_index'],label+'.driver_selection')
    audit.eq(row['selected_index'],ref['selected_index'],label+'.original_authority_selection')
    audit.eq(len(original['candidates']),len(returned['candidates']),label+'.all_candidate_count')
    require(len(original['candidates']) == len(returned['candidates']), 'fresh candidate population differs')
    for i,(before,after) in enumerate(zip(original['candidates'],returned['candidates'])):
        audit.eq(set(before),set(after),label+'.candidate_fields.'+str(i))
        audit.eq({k:v for k,v in before.items() if k != 'keypoints_xy'},
                 {k:v for k,v in after.items() if k != 'keypoints_xy'},label+'.all_conf_box_class.'+str(i))
        if i == original['selected_index']:
            audit.eq(before['keypoints_xy'][8],after['keypoints_xy'][8],label+'.selected_center')
            audit.eq(after['keypoints_xy'],output['native_points'],label+'.selected_final_points')
            audit.eq(before['keypoints_xy'],output['original_base_points'],label+'.selected_fresh_Base_points')
            saved = {k:v for k,v in before.items() if k != 'keypoints_xy'}
            audit.eq(saved,output['fixed_metadata']['candidate_metadata'],label+'.fresh_selected_metadata')
            audit.eq(saved,ref['candidate_metadata'],label+'.public_selected_metadata')
            tally['selected_original_and_returned_center_checks'] += 1
        else:
            audit.eq(before['keypoints_xy'],after['keypoints_xy'],label+'.nonselected_points.'+str(i))
            tally['nonselected_candidate_coordinate_checks'] += 1
        tally['full_candidate_conf_box_class_checks'] += 1
    audit.eq(output['fixed_metadata']['preserved'],True,label+'.metadata_preserved')
    audit.eq(row['fixed_metadata'],output['fixed_metadata'],label+'.driver_metadata')
    audit.eq(output['native_points'][8],output['native_N3_points'][8],label+'.returned_N3_center')
    audit.near(output['native_points'][8],ref['points']['BASE'][8],label+'.public_Base_center','center_px')
    tally['driver_center_checks'] += 1


def recorded_rank(witness, audit, tally, label):
    """Recount stored singular spectrum; neither a new J nor SVD is computed."""
    n=witness['scalar_rows']; singular=witness['singular_values']
    require(type(n) is int and n >= 0 and isinstance(singular,list),'stored rank witness required')
    audit.eq(witness['local_only'],True,label+'.local_only')
    audit.eq(len(singular),min(n,6),label+'.stored_singular_count')
    audit.ok(all(number(v) and v >= 0 for v in singular),label+'.finite_nonnegative_singulars')
    audit.ok(all(a >= b for a,b in zip(singular,singular[1:])),label+'.descending_singulars')
    rank=sum(v > singular[0]*RANK for v in singular) if singular else 0
    audit.eq(witness['numerical_rank'],rank,label+'.stored_spectrum_rank_recount')
    tally['stored_singular_rank_recounts_no_new_SVD']+=1
    return rank


def final_outcome(row, output, ref, runtime_protocol, audit, tally, label):
    H, s, actual = row['hidden_initial'],row['solver'],row['actual_pose']
    new, fallback = row['new_pose_estimated'],row['fallback_used']
    audit.ok(ids(H),label+'.sorted_H_IDs')
    audit.ok(type(new) is bool and type(fallback) is bool and not(new and fallback),label+'.distinct_new_fallback')
    for field in ('hidden_initial','actual_pose','solver','output_status','new_pose_estimated','fallback_used'):
        audit.eq(row[field],output[field],label+'.full_output_join.'+field)
    audit.eq(row['final_points'],output['native_points'],label+'.final_points_join')
    audit.eq(output['K'],ref['K'],label+'.K_authority'); audit.eq(output['xyz'],ref['xyz'],label+'.dimension_authority')
    audit.eq(output['raw_hw'],ref['raw_hw'],label+'.raw_hw_authority')
    for key in ('fresh_capture_input_only','saved_accuracy_coordinates_or_pose_as_input'):
        audit.eq(output[key],key == 'fresh_capture_input_only',label+'.'+key)
    point_parity(row['raw_BASE_points_parity'],audit,label+'.stored_Base_parity')
    audit.near(output['original_base_points'],ref['points']['BASE'],label+'.independent_Base_coordinates','Base_px')
    if row['arm'] != 'BASE':
        point_parity(row['raw_N3_points_parity'],audit,label+'.stored_N3_parity')
        audit.near(output['native_N3_points'],ref['points']['N3_SUBPIX'],label+'.independent_N3_coordinates','N3_px')
    else:
        audit.ok('raw_N3_points_parity' not in row,label+'.Base_has_no_N3_parity')
    point_parity(row['points_parity'],audit,label+'.stored_final_parity')
    pose_parity(row['pose_parity'],actual,audit,label+'.stored_pose_parity')
    if row['arm'] in ('BASE','N3_SUBPIX'):
        audit.eq(s,None,label+'.fixed_no_final_solver'); audit.eq(H,[],label+'.fixed_no_H')
        audit.eq(new,False,label+'.fixed_not_NEW'); audit.eq(fallback,False,label+'.fixed_not_fallback')
        audit.eq(row['output_status'],'FRESH_FIXED_CONTROL',label+'.fixed_status')
        audit.near(row['final_points'],ref['points'][row['arm']],label+'.fixed_original_points','fixed_px')
        audit.eq(row['fresh_point_bank_ledger'],None,label+'.fixed_no_point_bank')
        audit.eq(actual,output['initial_pose'],label+'.fixed_actual_initial')
    else:
        require(isinstance(s,dict), 'learned path requires full final solver')
        audit.eq(output['head_arm'],'IMAGE_ROLE',label+'.single_head')
        audit.eq(output['head_mode'],'IMAGE_ROLE',label+'.same_head_mode')
        audit.eq(output['head_checkpoint_sha256'],CHECKPOINT,label+'.fixed_ROLE_checkpoint')
        audit.eq(runtime_protocol['inputs']['fresh_v7:checkpoint:IMAGE_ROLE']['sha256'],CHECKPOINT,label+'.checkpoint_protocol')
        audit.ok(same_binding(output['calibration_binding'],runtime_protocol['inputs']['fresh_v7:calibration:IMAGE_ROLE']),
            label+'.CAL_protocol_join')
        audit.eq(output['observation_raw_logits_sha256'],row['full_fresh_ROLE_observation']['raw_logits_sha256'],label+'.fresh_logits_join')
        audit.eq(output['model_query_anchor'],'original Base predictions; unchanged training feature distribution',label+'.Base_query_anchor')
        audit.eq(new,s['available'],label+'.NEW_is_solver_available')
        audit.eq(actual['available'],new or fallback,label+'.actual_availability')
        audit.eq(row['output_status'],'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',label+'.status')
        for flag in ('prior_used','initial_projection_used','initial_dimension_prior_used','known_dimension_constraint_used',
                     'excluded_image_coordinates_used_for_scoring','excluded_image_coordinates_used_for_equivalence',
                     'reprojected_points_reused_as_observations'):
            audit.eq(s[flag],False,label+'.solver.'+flag)
        audit.eq(output['reprojections_reused_as_observations'],False,label+'.driver_no_refit')
        audit.eq(output['no_match_filled_as_boundary_observation'],False,label+'.no_missing_fill')
        audit.eq(s['hidden'],H,label+'.solver_H'); audit.eq(s['excluded'],H,label+'.excluded_H')
        audit.ok(ids(s['used']) and ids(s['fit_input_ids']),label+'.fit_ID_sets')
        audit.ok(set(H).isdisjoint(s['used']) and set(H).isdisjoint(s['fit_input_ids']),label+'.H_fit_exclusion')
        audit.ok(set(s['fit_input_ids']) <= set(s['used']),label+'.fit_subset_actual_pool')
        require(isinstance(s['all_candidate_solutions'],list), 'full candidate witnesses required')
        tally['stored_final_candidate_records_not_math_recomputed'] += len(s['all_candidate_solutions'])
        require(isinstance(row['fresh_point_bank_ledger'],dict), 'fresh point bank full ledger required')
        tally['fresh_point_bank_ledger_rows'] += 1
        if row['arm'] == PRIMARY:
            audit.eq(s['solver'],'LOCAL_POINT_LINE',label+'.LOCAL_backend')
            for flag in ('independent_PnP_or_PnL','global_uniqueness_proven','initial_pose_residual_prior',
                         'same_edge_point_line_double_count'):
                audit.eq(s[flag],False,label+'.LOCAL.'+flag)
            for flag in ('local_initialization_only','rank_is_local_only','inlier_counts_are_diagnostic_only',
                         'no_minimum_point_inlier_gate','final_factor_pool_is_same_for_every_start'):
                audit.eq(s[flag],True,label+'.LOCAL.'+flag)
            audit.eq(s['derived_line_points'],0,label+'.no_derived_observations')
            audit.eq(output['cached_accuracy_row_used'],False,label+'.no_accuracy_cache')
            audit.eq(output['parent_completion_receipt_used_as_deployment_input'],False,label+'.no_saved_parent_receipt')
            audit.eq(output['sealed_parent_packet_adapter_called'],False,label+'.no_saved_packet_adapter')
            audit.eq(output['fresh_point_start_recomputed_inside_call'],True,label+'.fresh_point_start')
            audit.eq(output['missing_sparse_filled_for_numeric_fit'],False,label+'.no_native_fill')
            audit.eq(output['native_N3_is_numeric_final_pose_observation'],False,label+'.no_native_final_observation')
            point = output['fresh_point_solver']
            tally['stored_point_start_candidate_records_not_math_recomputed'] += len(point['all_candidate_solutions'])
            audit.eq(point['fit_input_ids'],[k for k in point['fit_input_ids'] if k not in H],label+'.point_start_H_excluded')
            for entry,operation in OPTIMIZER_JOIN.items():
                audit.eq(row['local_optimizer_entry_calls'].get(entry,0),s['operation_counts'].get(operation,0),label+'.actual_local_callback_join.'+entry)
            if new:
                audit.eq(s['fit_input_ids'],s['used'],label+'.whole_actual_point_pool')
                audit.eq(s['fit_line_edges'],s['line_edges'],label+'.whole_actual_line_pool')
                audit.eq(s['inlier_geometry_is_acceptance_gate'],False,label+'.diagnostic_inliers_no_gate')
                for key in ('observed_normal_joint','modeled_normal_joint'):
                    audit.eq(recorded_rank(s['point_line_geometry'][key],audit,tally,label+'.stored_rank.'+key),
                        6,label+'.stored_whole_pool_rank6.'+key)
                if len(s['fit_input_ids']) < 4:
                    tally['LOCAL_NEW_with_fewer_than_four_actual_points'] += 1
                if not s['diagnostic_inlier_support_rank6']:
                    tally['LOCAL_NEW_whole_pool_rank6_weak_diagnostic_inlier_support'] += 1
            tally['LOCAL_optimizer_attempt_records'] += len(s['optimizer_attempts'])
        else:
            audit.eq(s['solver'],'OBSERVATION_ONLY_FINITE_SUBSET_ROBUST',label+'.point_backend')
            audit.eq(s['global_uniqueness_proven'],False,label+'.point_not_global_unique')
            audit.eq(s['prior_used'],False,label+'.point_no_prior')
            if new:
                audit.ok(len(s['final_inliers']) >= 4,label+'.robust_min4_consensus')
        if new:
            for key in ('R_cf','R_physical','centroid','cf_extents'):
                audit.eq(actual[key],s[key],label+'.actual_Rt_equals_solver.'+key)
            uv = projection(actual['cf_extents'],actual['R_cf'],actual['centroid'],ref['K'])
            audit.near(s['projected'],uv,label+'.independent_Rt_projection','projection_px')
            for k in H:
                audit.near(row['final_points'][k],uv[k],label+'.H_replaced.'+str(k),'projection_px')
                tally['independent_hidden_corner_projection_checks'] += 1
            for k in range(8):
                if k not in H:
                    p = output['input_points'][k]
                    is_observed = shaped(p,(2,)) and not all(x == -1 for x in p)
                    audit.eq(row['final_points'][k],p if is_observed else output['native_N3_points'][k],label+'.observed_or_display_native.'+str(k))
            audit.eq(output['reprojected_ids'],H,label+'.only_H_reprojected')
            tally['NEW_actual_Rt_solver_joins'] += 1
        else:
            audit.eq(actual,output['initial_pose'],label+'.fallback_actual_initial')
            audit.eq(row['final_points'],output['native_N3_points'],label+'.fallback_full_native_N3')
            audit.eq(output['reprojected_ids'],[],label+'.unavailable_no_H_reprojection')
        tally['hidden_set_changed_without_frame_veto'] += int(output['hidden_set_changed'])
    tally['output_status:'+row['arm']+':'+row['output_status']] += 1


def original_history(paths, bindings, runtime, rp, audit):
    history=runtime['original_runtime_history']
    audit.eq(history,rp['original_runtime_history'],'runtime.original_history_join')
    for name,alias in (('code','original_runtime_code'),('protocol','original_runtime_protocol'),
                       ('preflight','original_runtime_preflight'),('CLI_attempt_A','original_CLI_attempt'),
                       ('contract','runtime_contract')):
        audit.ok(same_binding(history[name],bindings[alias]),'runtime.original_history_bound.'+name)
        audit.ok(same_binding(history[name],rp['inputs']['runtime:original_history:'+name]),
            'runtime.frozen_original_history_bound.'+name)
    failure=read(paths['original_CLI_attempt'])
    audit.eq(failure['schema'],'fresh_runtime_cli_attempt_before_started_v9','original_failure.schema')
    audit.eq(failure['attempt'],'A','original_failure.attempt')
    audit.eq(failure['exit_code'],1,'original_failure.exit1')
    audit.eq(failure['error']['type'],'TypeError','original_failure.schema_error')
    for key in ('actual_image_decode_calls','actual_model_initializations',
                'actual_detector_N3_head_PnP_optimizer_calls','actual_timed_pipeline_calls'):
        audit.eq(failure[key],0,'original_failure.actual_zero.'+key)
    audit.eq(failure['original_runtime_and_protocol_preserved'],True,'original_failure.preserved')
    audit.eq(failure['automatic_retry'],False,'original_failure.no_auto_retry')
    for key in ('code','protocol','preflight'):
        audit.ok(same_binding(failure[key],history[key]),'original_failure.original_bytes.'+key)
    for name in ('RUNTIME_STARTED.json','RUNTIME.json','RUNTIME_ROWS.jsonl.gz'):
        audit.eq(failure['receipt_absence'][name],True,'original_failure.no_original_output_recorded.'+name)
        audit.eq((paths['runtime'].parent/name).exists(),False,'original_failure.outputs_still_absent.'+name)
    for packet_name,packet in (('receipt',runtime),('protocol',rp)):
        for key in ('input_schema_repair_only','original_schedule_timing_deployment_solver_unchanged','post_interval_pose_schema_parity'):
            audit.eq(packet[key],True,'runtime.'+packet_name+'.'+key)
        audit.eq(packet['original_attempt_timed_calls'],0,'runtime.'+packet_name+'.original_attempt_zero')


def provenance(paths, bindings, runtime, rp, audit):
    aliases=dict(runtime_code='runtime:runtime_code',deployment_code='runtime:deployment_code',
        review_contract='runtime:runtime_contract',panel='runtime:runtime_panel',
        accuracy_protocol='runtime:accuracy_protocol',accuracy_geometry='runtime:accuracy_geometry',
        accuracy_ledgers='runtime:accuracy_ledgers',geometry_seal='runtime:accuracy_seal',
        control_receipt='runtime:accuracy_receipt',geometry_validation_protocol='runtime:accuracy_validation_protocol',
        geometry_validation='runtime:accuracy_validation_checks',scoring_receipt='runtime:accuracy_scoring',
        cohort='fresh_v7:cohort',authority='fresh_v7:original_inputs')
    for alias,key in aliases.items():
        audit.ok(same_binding(bindings[alias],rp['inputs'][key]),'runtime.protocol_closure.'+alias)
    core=read(paths['accuracy_protocol']); seal=read(paths['geometry_seal']); score=read(paths['scoring_receipt'])
    for name in ('solver.py','pipeline.py','validation_checks.py'):
        audit.ok(same_binding(bindings['accuracy_code:'+name],core['inputs'][name]),'runtime.immutable_accuracy_code.'+name)
    for alias,key in (('accuracy_protocol','protocol'),('accuracy_geometry','geometry'),('accuracy_ledgers','ledgers'),
                      ('control_receipt','control_receipt'),('parent_population_checks','parent_population_checks')):
        audit.ok(same_binding(bindings[alias],seal[key]),'runtime.accuracy_seal_bound.'+alias)
    audit.ok(same_binding(score['control_receipt'],bindings['control_receipt']),'runtime.scoring_control_join')
    validation=read(paths['geometry_validation']); own=read(paths['geometry_validation_protocol'])
    audit.ok(same_binding(validation['own_protocol'],bindings['geometry_validation_protocol']),'runtime.geometry_gate_own_protocol')
    audit.ok(same_binding(validation['checker'],bindings['accuracy_code:validation_checks.py']),'runtime.geometry_gate_checker')
    audit.eq(validation['inputs'],own['inputs'],'runtime.geometry_gate_own_inputs')
    possible={(b['sha256'],b['bytes']) for b in rp['inputs'].values() if isinstance(b,dict) and 'sha256' in b and 'bytes' in b}
    require(isinstance(runtime['model_bindings'],list) and runtime['model_bindings'],'recorded model closure required')
    for index,b in enumerate(runtime['model_bindings']):
        audit.ok((b['sha256'],b['bytes']) in possible,'runtime.recorded_model_closure.'+str(index))
    for alias in ('Base','N3','checkpoint:IMAGE_ROLE','calibration:IMAGE_ROLE'):
        expected=rp['inputs']['fresh_v7:'+alias]
        audit.ok(any(same_binding(b,expected) for b in runtime['model_bindings']),'runtime.recorded_required_model_binding.'+alias)
    audit.eq(rp['deployed_single_head'],'IMAGE_ROLE','runtime.only_ROLE_loaded')
    audit.eq(rp['other_heads_loaded'],False,'runtime.no_other_heads')


def receipts(paths, bindings, runtime, audit):
    seal,control,score,core,rp,journal,preflight,validation = (read(paths[key]) for key in
        ('geometry_seal','control_receipt','scoring_receipt','accuracy_protocol','runtime_protocol',
         'runtime_started','runtime_preflight','geometry_validation'))
    for key in ('complete','statistics_official','model_initialization_complete','pipeline_close_complete',
        'image_decode_outside_intervals','runtime_rows_streamed_after_timed_interval',
        'only_timing_summaries_retained_in_memory','full_original_candidate_witnesses_preserved',
        'actual_optimizer_dispatch_count_join','all_recorded_resource_snapshots_quiet'):
        audit.eq(runtime[key],True,'runtime.'+key)
    audit.eq(runtime['status'],'DONE','runtime.done'); audit.eq(runtime['cleanup_errors'],[],'runtime.cleanup')
    audit.eq(runtime['arms'],list(ARMS),'runtime.arms')
    for key,n in (('frames',26),('panel_sessions',13),('warmup_per_arm',20),('measured_per_arm',130),('repeats',5),
        ('configured_pipeline_calls',600),('serialized_raw_rows',600),('actual_image_decode_calls',26),
        ('new_training_updates',0),('new_GT_evaluations',0),('new_RGB',0)):
        audit.eq(runtime[key],n,'runtime.'+key)
    for key in ('cached_coordinate_replay_used_for_timing','cached_accuracy_rows_used_as_inference_inputs',
                'sealed_parent_packet_adapter_called','GT_inference_access','other_benchmarks_parallel',
                'global_uniqueness_proven','automatic_retry'):
        audit.eq(runtime[key],False,'runtime.'+key)
    audit.eq(runtime['local_only'],True,'runtime.local_only')
    audit.eq(runtime['schema'],'fixed_fresh_sparse_local_line_complete_runtime_review_v9','runtime.review_schema')
    audit.eq(rp['schema'],'fixed_fresh_sparse_local_line_runtime_review_protocol_v9','runtime.protocol_review_schema')
    audit.eq(runtime['boundaries']['all_initial_feature_Base_point_robust_local_H_work_included'],True,'runtime.whole_path')
    audit.eq(runtime['execution'],dict(pipeline_calls_started=600,pipeline_calls_complete=600),'runtime.execution600')
    audit.eq(journal['status'],'DONE','runtime.journal_done')
    audit.eq(journal['counts'],runtime['execution'],'runtime.journal_counts')
    audit.eq(journal['elapsed_seconds'],runtime['elapsed_seconds'],'runtime.journal_elapsed')
    audit.ok(number(runtime['elapsed_seconds']) and runtime['elapsed_seconds'] > 0,'runtime.positive_elapsed')
    audit.eq(preflight['passed'],True,'preflight.pass')
    audit.ok(same_binding(preflight['runtime_protocol'],bindings['runtime_protocol']),'preflight.protocol')
    audit.eq(preflight['schema'],'independent_fresh_sparse_local_line_runtime_review_preflight_v9','preflight.schema')
    audit.eq(preflight['actual_pipeline_calls'],0,'preflight.calls0')
    audit.eq(preflight['automatic_retry'],False,'preflight.no_retry')
    audit.eq(preflight['resource_snapshot']['quiet'],True,'preflight.recorded_quiet')
    audit.eq(preflight['resource_snapshot']['temperature_under_80'],True,'preflight.recorded_thermal')
    audit.eq(core['frames'],245,'accuracy245'); audit.eq(core['methods'],[PRIMARY],'accuracy.method')
    audit.eq(seal['complete'],True,'accuracy.seal_complete'); audit.eq(seal['GT_read_allowed'],False,'accuracy.GT_free')
    for key in ('frames','rows','ledger_rows'):
        audit.eq(seal[key],245,'accuracy.seal_'+key)
    audit.eq(control['complete'],True,'accuracy.control_complete')
    audit.eq(control['environment_and_monkeypatch_cleanup_completed'],True,'accuracy.control_cleanup')
    audit.eq(control['error'],None,'accuracy.control_error'); audit.eq(control['cleanup_error'],None,'accuracy.control_cleanup_error')
    audit.eq(score['complete'],True,'accuracy.score_complete'); audit.eq(score['actual_total_scored_rows'],980,'accuracy.score980')
    audit.eq(validation['complete'],True,'accuracy.geometry_validation_complete')
    audit.eq(validation['passed'],True,'accuracy.geometry_validation_pass')
    gate=score['independent_geometry_validation']
    audit.eq(set(gate),{'protocol','receipt','checker'},'accuracy.score_gate_field_schema')
    for key,alias in (('protocol','geometry_validation_protocol'),('receipt','geometry_validation'),
                      ('checker','accuracy_code:validation_checks.py')):
        audit.ok(same_binding(gate[key],bindings[alias]),'accuracy.score_gate_bound.'+key)
    for key,expected in (('runtime_rows',runtime['raw_rows']),('runtime_protocol',runtime['runtime_protocol']),
        ('accuracy_protocol',runtime['accuracy_protocol']),('geometry_seal',runtime['accuracy_seal']),
        ('geometry_seal',score['geometry_seal']),('runtime_code',runtime['runtime_code']),
        ('deployment_code',runtime['deployment_code'])):
        audit.ok(same_binding(bindings[key],expected),'runtime.byte_binding.'+key)
    audit.eq(rp['arms'],list(ARMS),'runtime_protocol.arms')
    audit.eq(rp['stages'],list(STAGES),'runtime_protocol.stages')
    audit.eq(rp['configured_pipeline_calls'],600,'runtime_protocol600')
    all_jobs=list(schedule())
    audit.eq(rp['schedule'],dict(warmup=all_jobs[:80],measured=all_jobs[80:]),'runtime_protocol.fixed_schedule')
    for key in ('accuracy_policy_changed','cached_observation_replay_used_for_timing','GT_for_runtime_decisions','automatic_retry'):
        audit.eq(rp[key],False,'runtime_protocol.'+key)
    audit.eq(runtime['actual_OpenCV_primitive_names'],list(PRIMITIVES),'runtime.actual6_CV_names')
    audit.eq(runtime['actual_OpenCV_counter_entered'],True,'runtime.CV_counter_entered')
    audit.eq(runtime['actual_OpenCV_counter_scope'],'fresh timed calls and post-interval numerical parity; model loading excluded',
        'runtime.CV_counter_scope')
    audit.eq(runtime['prior_protection_before'],runtime['prior_protection_after'],'runtime.protection_unchanged')
    protection=runtime['prior_protection_after']
    audit.eq(protection['passed'],True,'runtime.protection_pass')
    audit.eq(protection['source_checkout_unchanged'],True,'runtime.source_unchanged')
    audit.eq(protection['changed_protected_paths'],[],'runtime.changed_paths')
    snaps=runtime['resource_snapshots']
    audit.ok(len(snaps) >= 7,'runtime.minimum_resource_snapshots')
    audit.eq(snaps[0]['phase'],'before_models','runtime.resource_start')
    audit.eq(snaps[-1]['phase'],'after_fresh_complete','runtime.resource_end')
    valid_jobs={json.dumps(j,sort_keys=True) for j in all_jobs}
    starts={json.dumps(j,sort_keys=True) for j in all_jobs if j['phase']=='measured' and j['image_index']==0 and j['arm_position']==0}
    witnessed=set()
    for index,snap in enumerate(snaps):
        label='runtime.resource.'+str(index)
        audit.eq(read(paths['resource:%03d'%index]),snap,label+'.durable_bytes')
        audit.eq(snap['quiet'],True,label+'.quiet'); audit.eq(snap['temperature_under_80'],True,label+'.thermal')
        audit.eq(snap['foreign_cpu_workloads'],[],label+'.recorded_no_competing_CPU')
        audit.eq(snap['foreign_gpu_pids'],[],label+'.recorded_no_competing_GPU')
        audit.eq(snap['system_changes'],False,label+'.no_system_changes')
        fields=snap['gpu_fields'].split(',')
        audit.ok('temperature.gpu' in fields,label+'.recorded_temperature_field')
        for gpu in csv.reader(snap['gpu'].splitlines(),skipinitialspace=True):
            audit.eq(len(gpu),len(fields),label+'.recorded_GPU_columns')
            require(len(gpu)==len(fields),'malformed recorded GPU resource snapshot')
            temperature=float(gpu[fields.index('temperature.gpu')])
            audit.ok(number(temperature) and 0 <= temperature < 80,label+'.recorded_temperature_under80')
        if snap['job'] is not None:
            key=json.dumps(snap['job'],sort_keys=True)
            audit.ok(key in valid_jobs,label+'.scheduled_job'); witnessed.add(key)
    audit.ok(starts <= witnessed,'runtime.all5_repeat_start_snapshots')
    original_history(paths,bindings,runtime,rp,audit)
    provenance(paths,bindings,runtime,rp,audit)
    return rp


def quantile(values, fraction):
    ordered=sorted(values); p=(len(ordered)-1)*fraction
    lo,hi=math.floor(p),math.ceil(p)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)


def describe(values):
    require(len(values) >= 2 and all(number(v) for v in values),'finite timing population required')
    mean=math.fsum(values)/len(values)
    var=math.fsum((v-mean)**2 for v in values)/(len(values)-1)
    return dict(n=len(values),mean=mean,sample_variance=var,sample_std=math.sqrt(var),
        median=quantile(values,.5),P90=quantile(values,.9),maximum=max(values),unit='ms',ddof=1)


def run(args):
    _,output=guard(args,'run'); protocol_path=output/OUTPUTS[0]
    own=read(protocol_path); paths=input_paths(args)
    bindings={key:binding(path) for key,path in paths.items()}
    require(own['inputs']==bindings and own['checker']==bindings['checker_code'],'own frozen bytes differ; preserve first attempt')
    require(own['schema']=='supplemental_sparse_local_line_runtime_scalar_protocol_v9' and own['limits']==LIMITS and
        own['scalar_absolute_tolerance']==ATOL and own['scalar_relative_tolerance']==RTOL,'own checker protocol differs')
    own_binding=binding(protocol_path)
    write_new(output/OUTPUTS[1],dict(status='STARTED',own_protocol=own_binding,checker=bindings['checker_code'],limits=LIMITS))
    began=time.monotonic(); audit=Audit(); tally=Counter(); phases=Counter(); arm_phases=Counter()
    primitive_total=Counter({k:0 for k in PRIMITIVES}); optimizer_total=Counter({k:0 for k in OPTIMIZER})
    head_total=Counter(attempted=0,completed=0)
    local_operation_total=Counter()
    timings={arm:{stage:[] for stage in SUMMARY_STAGES} for arm in ARMS}
    recomputed={}; completed=False; exception=None
    try:
        runtime=read(paths['runtime']); rp=receipts(paths,bindings,runtime,audit)
        panel,refs=authority(paths,runtime,rp,audit)
        audit.eq(rp['panel'],runtime['panel'],'runtime_protocol.ordered_panel')
        jobs=iter(schedule())
        with gzip.open(paths['runtime_rows'],'rt',encoding='utf-8') as stream:
            for index,line in enumerate(stream):
                require(bool(line.strip()),'blank runtime row')
                row=json.loads(line); label='row.'+str(index); expected=next(jobs,None)
                require(expected is not None,'unexpected additional runtime row')
                audit.eq({key:row.get(key) for key in expected},expected,label+'.schedule')
                item=panel[expected['image_index']]; ref=refs[item['frame_id']]
                audit.eq(row['id'],ref['id'],label+'.frame'); audit.eq(row['session'],ref['session'],label+'.session')
                for key,value in (('GT_canary_active',True),('fresh_capture',True),('accuracy_replay_input',False),
                    ('parity_status','PASS'),('candidate_center_score_box_preserved',True),
                    ('parity_checked_after_complete_timed_interval',True)):
                    audit.eq(row[key],value,label+'.'+key)
                audit.ok(not any(key in row for key in ('t0','t1','ticks','t0_ns','t1_ns')),label+'.absence_of_saved_absolute_ticks')
                for stage in SUMMARY_STAGES:
                    value=row[stage+'_ms']
                    audit.ok(number(value) and value > 0,label+'.positive_actual_duration.'+stage,value,'>0')
                    if row['phase']=='measured':
                        timings[row['arm']][stage].append(value)
                partition=math.fsum(row[stage+'_ms'] for stage in (*STAGES,'return'))
                audit.near(row['full_ms'],partition,label+'.duration_partition','full_partition_ms')
                requested=int(row['arm'] in (COMPARATOR,PRIMARY))
                audit.eq(row['actual_head_forward_entries'],{'IMAGE_ROLE':dict(attempted=requested,completed=requested)},label+'.requested_head')
                head_total.update(row['actual_head_forward_entries']['IMAGE_ROLE'])
                cv=counts(row['primitive_entry_calls'],PRIMITIVES,audit,label+'.primitive')
                opt=counts(row['local_optimizer_entry_calls'],OPTIMIZER,audit,label+'.optimizer')
                primitive_total.update(cv); optimizer_total.update(opt)
                audit.eq(opt['actual_SciPy_least_squares_entries'],
                    opt['actual_SciPy_least_squares_returns']+opt['actual_SciPy_least_squares_exceptions'],
                    label+'.optimizer_returns_or_recorded_exceptions')
                if row['arm'] != PRIMARY:
                    audit.eq(opt,{k:0 for k in OPTIMIZER},label+'.no_nonLOCAL_optimizer')
                out=row['full_fresh_output']
                audit.eq(out['method'],row['arm'],label+'.fresh_method')
                audit.eq(row['full_fresh_ROLE_observation'],out['observation'],label+'.fresh_observation_join')
                metadata(row,out,ref,audit,tally,label+'.metadata')
                final_outcome(row,out,ref,rp,audit,tally,label+'.outcome')
                if row['arm']==PRIMARY:
                    for key,value in row['solver']['operation_counts'].items():
                        require(type(value) is int and value >= 0,'invalid LOCAL operation counter')
                        local_operation_total[key]+=value
                phases[row['phase']]+=1; arm_phases[(row['arm'],row['phase'])]+=1
                tally['runtime_rows']+=1
                del out,row
        audit.eq(next(jobs,None),None,'runtime.complete_schedule')
        audit.eq(tally['runtime_rows'],600,'runtime600rows'); audit.eq(phases,Counter(warmup=80,measured=520),'runtime.phases')
        for arm in ARMS:
            audit.eq(arm_phases[(arm,'warmup')],20,'runtime.warm20.'+arm)
            audit.eq(arm_phases[(arm,'measured')],130,'runtime.measured130.'+arm)
            audit.eq(runtime['warmup_accounting'][arm],dict(configured=20,completed=20),'runtime.warm_accounting.'+arm)
        audit.eq(dict(primitive_total),counts(runtime['actual_OpenCV_entry_calls'],PRIMITIVES,audit,'receipt.primitive'),'runtime.actual_CV_row_sum')
        audit.eq(dict(optimizer_total),counts(runtime['actual_optimizer_entry_calls'],OPTIMIZER,audit,'receipt.optimizer'),'runtime.actual_optimizer_row_sum')
        audit.eq(runtime['actual_head_forward_entries'],{'IMAGE_ROLE':dict(head_total)},'runtime.head_row_sum')
        audit.eq(dict(head_total),dict(attempted=300,completed=300),'runtime.actual_head300')
        for entry,operation in OPTIMIZER_JOIN.items():
            audit.eq(optimizer_total[entry],runtime['local_operation_counts'].get(operation,0),'runtime.local_dispatch_join.'+entry)
        for key in set(local_operation_total)|set(runtime['local_operation_counts']):
            audit.eq(local_operation_total.get(key,0),runtime['local_operation_counts'].get(key,0),
                'runtime.all_local_operation_row_sum.'+key)
        for key,value in dict(detector_calls=600,initial_pose_calls=600,N3_route_calls=450,
            feature_initial_pose_calls=300,head_calls=300,IMAGE_ROLE_head_calls=300,final_pose_paths=300).items():
            audit.eq(runtime['actual_pipeline_calls'].get(key,0),value,'runtime.actual_pipeline.'+key)
        fresh=runtime['actual_fresh_path_calls']
        for key,value in dict(fresh_predict_calls_started=600,fresh_predict_calls_complete=600,
            fresh_sparse_point_paths_started=300,fresh_sparse_point_paths_complete=300,
            fresh_local_banks_started=150,fresh_local_banks_complete=150,
            fresh_local_paths_started=150,fresh_local_paths_complete=150).items():
            audit.eq(fresh.get(key,0),value,'runtime.actual_fresh.'+key)
        for arm in ARMS:
            for phase in ('started','complete'):
                audit.eq(fresh.get(arm+':'+phase,0),150,'runtime.actual_route150.'+arm+'.'+phase)
        forwards=runtime['model_forwards']
        audit.eq(forwards['IMAGE_ROLE'],300,'runtime.head_model_forwards')
        audit.eq(forwards['N3'],450,'runtime.N3_model_forwards')
        audit.eq(forwards['detector']-forwards['detector_internal_initialization_calls'],600,'runtime.detector_model_forwards_without_loading')
        for arm in ARMS:
            recomputed[arm]={}
            for stage in SUMMARY_STAGES:
                summary=describe(timings[arm][stage]); recomputed[arm][stage]=summary
                stored=runtime['summaries'][arm][stage]
                for moment in MOMENTS:
                    audit.near(stored[moment],summary[moment],'runtime.moment.'+arm+'.'+stage+'.'+moment,'moment_'+moment)
                    tally['independently_recomputed_moment_slots']+=1
                audit.eq(stored['unit'],'ms','runtime.summary_unit.'+arm+'.'+stage)
                audit.eq(stored['ddof'],1,'runtime.sample_variance_ddof.'+arm+'.'+stage)
                tally['summary_blocks']+=1
        audit.eq(tally['summary_blocks'],28,'runtime28summaryblocks')
        audit.eq(tally['independently_recomputed_moment_slots'],196,'runtime196moment_slots')
        audit.eq({k:binding(path) for k,path in paths.items()},bindings,'own_inputs_unchanged_after_scalar_math')
        completed=True
    except BaseException as error:
        exception=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    result=dict(schema='supplemental_sparse_local_line_runtime_scalar_checks_v9',
        complete=completed,passed=bool(completed and not audit.failure_count and exception is None),
        own_protocol=own_binding,checker=bindings['checker_code'],inputs=bindings,limits=LIMITS,
        checks=audit.checks,failure_count=audit.failure_count,failures=audit.failures,exception=exception,
        counts=dict(tally),phase_counts=dict(phases),actual_primitive_row_sums=dict(primitive_total),
        actual_optimizer_row_sums=dict(optimizer_total),actual_head_row_sums=dict(head_total),
        actual_local_operation_row_sums=dict(local_operation_total),
        independently_recomputed_summaries=recomputed,max_abs_difference=dict(audit.max_abs_difference),
        elapsed_seconds=time.monotonic()-began,first_attempt_preserved=True)
    write_new(output/OUTPUTS[2],result)
    print('V9_RUNTIME_SCALAR_CHECKED',result['passed'],result['counts'].get('runtime_rows',0),flush=True)
    require(result['passed'],'runtime scalar audit failed; receipt preserved, no automatic retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','run'))
    parser.add_argument('--input',type=Path,default=DOC)
    parser.add_argument('--output',type=Path,default=DOC)
    parser.add_argument('--accuracy-protocol',type=Path,default=DOC/'PROTOCOL.json')
    parser.add_argument('--runtime-protocol',type=Path,default=DOC/'RUNTIME_REVIEW_PROTOCOL.json')
    parser.add_argument('--runtime-code',type=Path,default=Path(__file__).with_name('runtime_review.py'))
    parser.add_argument('--runtime-name',default='RUNTIME_REVIEW.json')
    parser.add_argument('--rows-name',default='RUNTIME_REVIEW_ROWS.jsonl.gz')
    parser.add_argument('--started-name',default='RUNTIME_REVIEW_STARTED.json')
    parser.add_argument('--preflight-name',default='RUNTIME_REVIEW_PREFLIGHT.json')
    parser.add_argument('--resource-prefix',default='RUNTIME_REVIEW_RESOURCE_')
    parser.add_argument('--panel',type=Path,default=CORRECTED/'RUNTIME_PANEL.json')
    parser.add_argument('--cohort',type=Path,default=CORRECTED/'COHORT.json')
    parser.add_argument('--authority',type=Path,default=OLD/'INPUTS.json')
    parser.add_argument('--history',type=Path,action='append',default=[])
    args=parser.parse_args()
    for name in (args.runtime_name,args.rows_name,args.started_name,args.preflight_name):
        require(name == Path(name).name and name not in ('.','..'),'basename required')
    require(args.resource_prefix == Path(args.resource_prefix).name,'resource prefix basename required')
    {'freeze':freeze,'run':run}[args.stage](args)


if __name__ == '__main__':
    main()
