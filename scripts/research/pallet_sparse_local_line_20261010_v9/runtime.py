"""Own-frozen fresh whole-path benchmark of four fixed V9 routes.

Accuracy bytes and their accepted policy stay unchanged. Each of the 600 calls
captures the detector anew; local calls also recompute the sparse point solve
inside the same synchronized interval. Saved data are post-interval parity only.
"""
from __future__ import annotations

from collections import Counter
import copy
import importlib.metadata
from pathlib import Path
import sys
import time
import traceback

from . import common as C
from ..pallet_three_head_observation_20261010_v7 import common as V7
from ..pallet_three_head_observation_20261010_v7.runtime import describe, durable_owned

ARMS = ('BASE', 'N3_SUBPIX', C.COMPARATOR, C.PRIMARY)
STAGES = ('detector', 'N3_correction', 'initial_pose', 'boundary_observation',
          'final_pose_reprojection_and_metadata')
FRAMES, WARMUP, REPEATS = 26, 20, 5
RUNTIME_NAMES = ('RUNTIME_STARTED.json', 'RUNTIME.json', 'RUNTIME_ROWS.jsonl.gz',
                 'PENDING_RUNTIME_ROWS.jsonl.gz', 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz')


def schedules():
    warm, measured = [], []
    for index in range(WARMUP):
        order = list(ARMS[index % len(ARMS):] + ARMS[:index % len(ARMS)])
        if index % 2:
            order.reverse()
        for position, arm in enumerate(order):
            warm.append(dict(phase='warmup', arm=arm, warmup_index=index,
                arm_position=position, image_index=index % FRAMES))
    for repeat in range(REPEATS):
        order = list(ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)])
        if repeat % 2:
            order.reverse()
        for image in range(FRAMES):
            for position, arm in enumerate(order):
                measured.append(dict(phase='measured', arm=arm, repeat=repeat,
                    arm_position=position, image_index=image))
    C.require(len(warm) == 80 and len(measured) == 520 and
        Counter(job['arm'] for job in warm) == {arm:20 for arm in ARMS} and
        Counter(job['arm'] for job in measured) == {arm:130 for arm in ARMS}, 'fixed fresh600 schedule differs')
    return warm, measured


def accuracy_args(args):
    result = copy.copy(args)
    result.output = args.accuracy_output
    return result


def prerequisites(args):
    """Validate completed accuracy and retain only parity fields on fixed26."""
    a = accuracy_args(args)
    from .evaluator import sealed
    seal, _, validation, _, _ = sealed(a)
    folder = Path(args.accuracy_output)
    scoring = C.read(folder / 'SCORING_RECEIPT.json')
    C.require(scoring['complete'] is True and scoring['actual_total_scored_rows'] == 980 and
        C.same_binding(scoring['geometry_seal'], C.binding(folder / 'GEOMETRY_SEAL.json')) and
        scoring['independent_geometry_validation'] == validation, 'runtime requires completed frozen accuracy scoring')
    for key, name in (('predictions', 'PREDICTIONS.jsonl.gz'), ('comparator_predictions', 'COMPARATOR_PREDICTIONS.jsonl.gz'),
                      ('fixed_predictions', 'FIXED_PREDICTIONS.jsonl.gz')):
        C.bound(folder / name, scoring[key], 'runtime completed accuracy scores:' + key)
    cohort = C.read(args.cohort)
    eligible = {row['id']: row for row in cohort['frames']}
    panel_path = C.CORRECTED / 'RUNTIME_PANEL.json'
    panel = C.read(panel_path)['frames']
    panel_ids = {row['frame_id'] for row in panel}
    C.require(len(panel) == len(panel_ids) == 26 and len({row['session_id'] for row in panel}) == 13 and
        Counter(row['session_id'] for row in panel) == {s:2 for s in {row['session_id'] for row in panel}} and
        panel_ids <= set(eligible), 'existing fixed runtime panel changed')
    fields = ('native_points', 'actual_pose', 'hidden_initial', 'output_status',
              'new_pose_estimated', 'fallback_used', 'fixed_metadata')
    references = {arm:{} for arm in ARMS}
    streams = ((folder/'GEOMETRY_SEALED.jsonl.gz', (C.PRIMARY,)),
        (Path(args.point_parent)/'GEOMETRY_SEALED.jsonl.gz', tuple(C.V8.METHODS)),
        (Path(args.parent)/'FIXED_GEOMETRY_SEALED.jsonl.gz', ('BASE','N3_SUBPIX')))
    for path, methods in streams:
        keys, counts = set(), Counter()
        for row in C.rows(path):
            key = (row['method'], row['id'])
            C.require(row['method'] in methods and row['id'] in eligible and key not in keys,
                      'unknown/duplicate full accuracy runtime reference')
            keys.add(key); counts[row['method']] += 1
            if row['method'] in references and row['id'] in panel_ids:
                references[row['method']][row['id']] = {key:row[key] for key in fields}
        C.require(counts == {method:245 for method in methods} and len(keys) == 245*len(methods),
                  'runtime reference full population incomplete')
    C.require(all(set(values) == panel_ids for values in references.values()), 'runtime26 references incomplete')
    return panel, eligible, references, seal


def runtime_inputs(args):
    a = accuracy_args(args)
    protocol = C.verify_protocol(a)
    inputs = {key:value for key,value in protocol['inputs'].items()}
    # V7 constructor needs its unchanged source/checkpoint/calibration closure.
    # These hashes are frozen here, outside every measured inference interval.
    inputs.update({'fresh_v7:' + key: value for key,value in V7.inputs(args).items()})
    for key, path in dict(runtime_code=Path(__file__), deployment_code=C.CODE/'deployment.py',
        runtime_contract=C.DOC/'RUNTIME_CONTRACT_KO.md',
        runtime_panel=C.CORRECTED/'RUNTIME_PANEL.json', accuracy_protocol=Path(args.protocol),
        accuracy_geometry=Path(args.accuracy_output)/'GEOMETRY_SEALED.jsonl.gz',
        accuracy_ledgers=Path(args.accuracy_output)/'CONTROL_LEDGERS.jsonl.gz',
        accuracy_seal=Path(args.accuracy_output)/'GEOMETRY_SEAL.json',
        accuracy_receipt=Path(args.accuracy_output)/'CONTROL_RECEIPT.json',
        accuracy_validation_protocol=Path(args.accuracy_output)/'VALIDATION_PROTOCOL.json',
        accuracy_validation_checks=Path(args.accuracy_output)/'VALIDATION_CHECKS.json',
        accuracy_scoring=Path(args.accuracy_output)/'SCORING_RECEIPT.json').items():
        inputs['runtime:' + key] = C.binding(path)
    return inputs


def frozen(args):
    protocol = C.read(args.runtime_protocol)
    C.require(protocol['schema'] == 'fixed_fresh_sparse_local_line_runtime_protocol_v9' and
        protocol['arms'] == list(ARMS) and protocol['frames'] == 26 and
        protocol['warmup_per_arm'] == 20 and protocol['measured_per_arm'] == 130 and
        protocol['configured_pipeline_calls'] == 600 and protocol['stages'] == list(STAGES) and
        protocol['inputs'] == runtime_inputs(args), 'own frozen fresh runtime code/input policy differs')
    C.require(protocol['source_root'] == str(Path(args.source_root).resolve()) and
        protocol['baseline_root'] == str(Path(args.baseline_root).resolve()), 'runtime split roots differ')
    return protocol


def freeze(args):
    path = Path(args.runtime_protocol)
    C.require(path.resolve() == (C.DOC/'RUNTIME_PROTOCOL.json').resolve(), 'formal own runtime protocol must be new V9 DOC')
    destination = copy.copy(args); destination.output = str(C.DOC)
    C.output_path(destination, path.name)
    panel, _, _, _ = prerequisites(args)
    inputs = runtime_inputs(args)
    protection = C.protect(args)
    warm, measured = schedules()
    C.write_new(path, dict(schema='fixed_fresh_sparse_local_line_runtime_protocol_v9', inputs=inputs,
        arms=list(ARMS), stages=list(STAGES), frames=26, sessions=13, warmup_per_arm=20,
        measured_per_arm=130, repeats=5, configured_pipeline_calls=600,
        schedule=dict(warmup=warm, measured=measured), panel=[dict(id=f['frame_id'], session=f['session_id']) for f in panel],
        source_root=str(Path(args.source_root).resolve()), baseline_root=str(Path(args.baseline_root).resolve()),
        accuracy_policy_changed=False, own_freeze_after_completed_accuracy_before_own_runtime=True,
        fresh_detector_each_call=True, fresh_point_only_start_inside_LOCAL_interval=True,
        initial_N3_and_feature_Base_and_point_robust_and_local_and_H_reprojection_all_timed=True,
        cached_observation_replay_used_for_timing=False, saved_accuracy_inputs_used_only_for_post_interval_parity=True,
        protected_before=protection, automatic_retry=False, GT_for_runtime_decisions=False,
        timing_boundary='RAM BGR/K/registered dimensions -> fresh detector/N3/initial pose/ROLE feature+decode/sparse point robust/optional local+unused physical line factors/H replacement/prediction preservation/metadata return',
        excluded=['model/checkpoint loading','RGB file/decode','accuracy/GT scoring','post-interval parity','durable logging','resource snapshots'],
        deployed_single_head='IMAGE_ROLE', other_heads_loaded=False, local_global_uniqueness_proven=False,
        local_start_and_existing_factor_loss_rank_tie_policy_unchanged=True,
        frozen_after_known_DEV_accuracy=True, no_unseen_generalization_claim=True,
        actual_runtime_calls_at_freeze=0, new_training_updates=0, new_RGB=0))
    frozen(args)
    print('SPARSE_LOCAL_FRESH_RUNTIME_FROZEN', C.binding(path)['sha256'], flush=True)


def preflight(args):
    frozen(args); prerequisites(args)
    for name in RUNTIME_NAMES:
        C.output_path(args, name)
    snapshot = C.quiet_snapshot()
    passed = bool(snapshot['quiet'] and snapshot['temperature_under_80'])
    C.write_new(C.output_path(args, 'RUNTIME_PREFLIGHT.json'), dict(schema='independent_fresh_sparse_local_line_runtime_preflight_v9',
        passed=passed, runtime_protocol=C.binding(args.runtime_protocol), resource_snapshot=snapshot,
        protection=C.protect(args), actual_pipeline_calls=0, automatic_retry=False))
    C.require(passed, 'RUNTIME_PENDING: competing workload/thermal guard; preflight preserved')
    print('SPARSE_LOCAL_FRESH_RUNTIME_PREFLIGHT_PASS', flush=True)


def _measure_with_context(args):
    import cv2
    import numpy as np
    import scipy
    from .deployment import Pipeline, STAGES as DEPLOYMENT_STAGES
    from . import solver as solver_module
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_same_observation_controls_20261010_v8.runner import primitive_entries, PRIMITIVES
    protocol = frozen(args)
    panel, eligible, references, seal = prerequisites(args)
    C.require(tuple(DEPLOYMENT_STAGES) == STAGES, 'fresh adapter stage markers differ')
    preflight_receipt = C.read(Path(args.output)/'RUNTIME_PREFLIGHT.json')
    C.require(preflight_receipt['passed'] is True and C.same_binding(preflight_receipt['runtime_protocol'],
        C.binding(args.runtime_protocol)), 'independent fresh runtime preflight required')
    for name in RUNTIME_NAMES:
        C.output_path(args, name)
    before_protection = C.protect(args)
    destination = C.output_path(args, 'RUNTIME.json')
    row_path = C.output_path(args, 'RUNTIME_ROWS.jsonl.gz')
    journal = C.output_path(args, 'RUNTIME_STARTED.json')
    images = []
    for item in panel:
        frame = eligible[item['frame_id']]
        path = Path(args.source_root)/frame['image']
        C.require(V7.sha(path) == frame['image_sha256'], 'fresh runtime RGB hash differs')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        C.require(image is not None and list(image.shape[:2]) == frame['raw_hw'], 'fresh runtime RGB decode differs')
        images.append(image)
    result = dict(schema='fixed_fresh_sparse_local_line_complete_runtime_v9', complete=False, status='FAILED_RUNTIME',
        arms=list(ARMS), frames=26, panel_sessions=13, panel=protocol['panel'], runtime_protocol=C.binding(args.runtime_protocol),
        accuracy_protocol=C.binding(args.protocol), accuracy_seal=C.binding(Path(args.accuracy_output)/'GEOMETRY_SEAL.json'),
        runtime_code=C.binding(__file__), deployment_code=C.binding(C.CODE/'deployment.py'),
        warmup_per_arm=20, measured_per_arm=130, repeats=5, configured_pipeline_calls=600,
        schedule='cyclic arm rotation and odd-block reversal;20 warmups+5x26 measured per arm',
        boundaries=dict(full_pipeline=protocol['timing_boundary'], excluded=protocol['excluded'],
            all_initial_feature_Base_point_robust_local_H_work_included=True),
        cached_coordinate_replay_used_for_timing=False, cached_accuracy_rows_used_as_inference_inputs=False,
        sealed_parent_packet_adapter_called=False, GT_inference_access=False, actual_image_decode_calls=26,
        image_decode_outside_intervals=True, local_only=True, global_uniqueness_proven=False,
        other_benchmarks_parallel=False, new_training_updates=0, new_GT_evaluations=0, new_RGB=0,
        automatic_retry=False, statistic_source='actual synchronized fresh complete-path intervals')
    state, optimizer = Counter(), Counter()
    primitive = Counter({name:0 for name in PRIMITIVES})
    head_counts, hook_handles, cleanup_errors = Counter(), [], []
    rows, probes = [], []
    initialized, pipeline, writer, primitive_entered = False, None, None, False
    active_raw, raw_write_attempted = None, False
    original_optimizer = solver_module.least_squares
    began = time.monotonic()
    C.write_new(journal, dict(status='STARTED', runtime_protocol=C.binding(args.runtime_protocol), configured_calls=600, counts=dict(state)))

    def probe(phase, job=None):
        snapshot = dict(phase=phase, job=job, complete_calls=state['pipeline_calls_complete'], **C.quiet_snapshot())
        C.write_new(C.output_path(args, 'RUNTIME_RESOURCE_%03d.json' % len(probes)), snapshot)
        probes.append(snapshot)
        C.require(snapshot['quiet'] and snapshot['temperature_under_80'], 'RUNTIME_PENDING: competing workload/thermal guard')

    def summary(raw):
        return {key:value for key,value in raw.items() if key in ('arm','phase','parity_status') or key.endswith('_ms')}

    def counted_optimizer(fun, x0, *a, **kw):
        optimizer['actual_SciPy_least_squares_entries'] += 1
        def residual(*values, **options):
            optimizer['actual_optimizer_residual_entries'] += 1
            return fun(*values, **options)
        jac = kw.get('jac')
        if callable(jac):
            def jacobian(*values, **options):
                optimizer['actual_optimizer_jacobian_entries'] += 1
                return jac(*values, **options)
            kw['jac'] = jacobian
        try:
            fit = original_optimizer(residual, x0, *a, **kw)
        except BaseException:
            optimizer['actual_SciPy_least_squares_exceptions'] += 1
            raise
        optimizer['actual_SciPy_least_squares_returns'] += 1
        return fit

    try:
        probe('before_models')
        writer = RowWriter(row_path, interrupted_path=C.output_path(args, 'INTERRUPTED_RUNTIME_ROWS.jsonl.gz'))
        with V7.inference_canary():
            pipeline = Pipeline(args)
        initialized = True
        C.require(set(pipeline.heads) == {'IMAGE_ROLE'}, 'fresh runtime must load only the requested head')
        for arm, head in pipeline.heads.items():
            def entry(module, inputs, arm=arm):
                head_counts[arm+':attempted'] += 1
            def complete(module, inputs, output, arm=arm):
                head_counts[arm+':completed'] += 1
            hook_handles.extend([head.register_forward_pre_hook(entry), head.register_forward_hook(complete)])
        torch = pipeline.torch
        C.require(torch.cuda.is_available(), 'CUDA unavailable')
        result['environment'] = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
            scipy=scipy.__version__, opencv=cv2.__version__, ultralytics=importlib.metadata.version('ultralytics'),
            batch=1, actual_torch_threads=torch.get_num_threads(), actual_OpenCV_threads=cv2.getNumThreads(),
            numeric=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                CUDA_version=torch.version.cuda, cuDNN_version=torch.backends.cudnn.version()),
            device=dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability())),
            threadpools=[{k:v for k,v in pool.items() if k != 'filepath'} for pool in __import__('threadpoolctl').threadpool_info()])
        torch.cuda.reset_peak_memory_stats()
        solver_module.least_squares = counted_optimizer
        last_probe = time.monotonic()
        warm, measured = schedules()
        with V7.inference_canary(), pipeline.old.no_truth_reads(), torch.no_grad(), primitive_entries(primitive):
            primitive_entered = True
            for job in warm+measured:
                active_raw, raw_write_attempted = None, False
                C.require(time.monotonic()-began <= args.remaining_seconds, 'RUNTIME_PENDING: operating limit')
                if time.monotonic()-last_probe >= 5 or (job['phase'] == 'measured' and job['image_index'] == 0 and job['arm_position'] == 0):
                    probe('during_fresh_paths', job); last_probe = time.monotonic()
                frame = eligible[panel[job['image_index']]['frame_id']]
                state['pipeline_calls_started'] += 1
                durable_owned(journal, dict(status='RUNNING', active_job=job, counts=dict(state),
                    model_counts=dict(pipeline.base.counts), local_counts=dict(pipeline.counts), optimizer_entries=dict(optimizer)))
                primitive_before, head_before, optimizer_before = primitive.copy(), head_counts.copy(), optimizer.copy()
                ticks = []
                def mark(stage):
                    torch.cuda.synchronize()
                    ticks.append((stage, time.perf_counter_ns()))
                metadata = {key:frame[key] for key in ('id','session','object_type')}
                torch.cuda.synchronize(); t0 = time.perf_counter_ns()
                output = pipeline.predict(images[job['image_index']], frame['K'], frame['xyz'], metadata,
                    method=job['arm'], stage_callback=mark)
                torch.cuda.synchronize(); t1 = time.perf_counter_ns()
                times, previous = {}, t0
                for stage, tick in ticks:
                    times[stage+'_ms'] = (tick-previous)/1e6; previous = tick
                times['return_ms'] = (t1-previous)/1e6
                raw = dict(**job, id=frame['id'], session=frame['session'], full_ms=(t1-t0)/1e6, **times,
                    final_points=output['native_points'], actual_pose=output['actual_pose'], solver=output.get('solver'),
                    hidden_initial=output['hidden_initial'], output_status=output['output_status'],
                    new_pose_estimated=output['new_pose_estimated'], fallback_used=output['fallback_used'],
                    selected_index=output['selected_index'], fixed_metadata=output['fixed_metadata'],
                    full_fresh_output=output, full_fresh_ROLE_observation=output['observation'],
                    fresh_point_bank_ledger=pipeline.last_point_bank_ledger,
                    primitive_entry_calls={name:primitive[name]-primitive_before[name] for name in primitive},
                    local_optimizer_entry_calls={name:optimizer[name]-optimizer_before[name] for name in set(optimizer)|set(optimizer_before)},
                    actual_head_forward_entries={'IMAGE_ROLE':{key:head_counts['IMAGE_ROLE:'+key]-head_before['IMAGE_ROLE:'+key]
                        for key in ('attempted','completed')}},
                    GT_canary_active=True, fresh_capture=True, accuracy_replay_input=False, parity_status='PENDING')
                active_raw = raw
                C.require([stage for stage,_ in ticks] == list(STAGES), 'fresh runtime stage schedule differs')
                requested = int(job['arm'] in (C.COMPARATOR,C.PRIMARY))
                C.require(raw['actual_head_forward_entries'] == {'IMAGE_ROLE':dict(attempted=requested,completed=requested)},
                          'fresh requested-head forward hooks differ')
                reference = references[job['arm']][frame['id']]
                raw['points_parity'] = pipeline.runtime._point_parity(output['native_points'], reference['native_points'], 1e-7, (job['arm'],frame['id']))
                raw['pose_parity'] = pipeline.runtime._pose_parity(output['actual_pose'], reference['actual_pose'])
                C.require(all(C.finite(output[key]) == C.finite(reference[key]) for key in
                    ('hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')), 'fresh saved outcome metadata differs')
                raw['raw_BASE_points_parity'] = pipeline.runtime._point_parity(output['original_base_points'], frame['points']['BASE'], 1e-7, ('BASE',frame['id']))
                if job['arm'] != 'BASE':
                    raw['raw_N3_points_parity'] = pipeline.runtime._point_parity(output['native_N3_points'], frame['points']['N3_SUBPIX'], 1e-7, ('N3_SUBPIX',frame['id']))
                if output['new_pose_estimated']:
                    C.require(set(output['hidden_initial']).isdisjoint(output['solver']['fit_input_ids']), 'fresh runtime H fit leakage')
                    if output['hidden_initial']:
                        np.testing.assert_allclose(np.asarray(output['native_points'])[output['hidden_initial']],
                            np.asarray(output['solver']['projected'])[output['hidden_initial']], rtol=0, atol=1e-12)
                raw.update(parity_status='PASS', candidate_center_score_box_preserved=True,
                    parity_checked_after_complete_timed_interval=True)
                rows.append(summary(raw)); raw_write_attempted = True; writer.write(raw); active_raw = None
                state['pipeline_calls_complete'] += 1
                if state['pipeline_calls_complete'] % 50 == 0:
                    print('SPARSE_LOCAL_FRESH_RUNTIME',state['pipeline_calls_complete'],600,flush=True)
        base_counts, local_counts = pipeline.base.counts, pipeline.counts
        C.require(len(rows) == state['pipeline_calls_complete'] == 600 and
            base_counts['detector_calls'] == base_counts['initial_pose_calls'] == 600 and
            base_counts['N3_route_calls'] == 450 and base_counts['head_calls'] ==
            base_counts['feature_initial_pose_calls'] == base_counts['IMAGE_ROLE_head_calls'] == 300 and
            base_counts['final_pose_paths'] == 300 and
            head_counts['IMAGE_ROLE:attempted'] == head_counts['IMAGE_ROLE:completed'] == 300 and
            local_counts['fresh_predict_calls_started'] == local_counts['fresh_predict_calls_complete'] == 600 and
            local_counts['fresh_sparse_point_paths_started'] == local_counts['fresh_sparse_point_paths_complete'] == 300 and
            local_counts['fresh_local_banks_started'] == local_counts['fresh_local_banks_complete'] ==
            local_counts['fresh_local_paths_started'] == local_counts['fresh_local_paths_complete'] == 150 and
            all(local_counts[arm+':started'] == local_counts[arm+':complete'] == 150 for arm in ARMS), 'fresh600 path count join differs')
        for entry, operation in (('actual_SciPy_least_squares_entries','local_optimizer_calls'),
            ('actual_SciPy_least_squares_returns','local_optimizer_completions'),
            ('actual_optimizer_residual_entries','optimizer_residual_calls'), ('actual_optimizer_jacobian_entries','optimizer_jacobian_calls')):
            C.require(optimizer[entry] == pipeline.local_operation_counts[operation], 'fresh actual local dispatch differs:' + entry)
        probe('after_fresh_complete')
        result.update(complete=True, status='DONE', actual_optimizer_dispatch_count_join=True,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result.update(status='RUNTIME_PENDING' if 'RUNTIME_PENDING' in str(error) else 'FAILED_RUNTIME',
            reason=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
    finally:
        solver_module.least_squares = original_optimizer
        if writer is not None and active_raw is not None and not raw_write_attempted:
            rows.append(summary(active_raw))
            try:
                raw_write_attempted = True; writer.write(active_raw)
            except BaseException as error:
                cleanup_errors.append(dict(phase='failed_active_row', type=type(error).__name__, message=str(error)))
        for handle in hook_handles:
            try:
                handle.remove()
            except BaseException as error:
                cleanup_errors.append(dict(phase='head_hook', type=type(error).__name__, message=str(error)))
        result.update(actual_OpenCV_entry_calls=dict(primitive), actual_OpenCV_primitive_names=list(PRIMITIVES),
            actual_OpenCV_counter_entered=primitive_entered,
            actual_OpenCV_counter_scope='fresh timed calls and post-interval numerical parity; model loading excluded',
            actual_optimizer_entry_calls=dict(optimizer), actual_head_forward_entries={'IMAGE_ROLE':{
                key:head_counts['IMAGE_ROLE:'+key] for key in ('attempted','completed')}},
            model_initialization_complete=initialized, execution=dict(state), resource_snapshots=probes,
            elapsed_seconds=time.monotonic()-began, raw_row_write_attempted=raw_write_attempted)
        result['all_recorded_resource_snapshots_quiet'] = bool(probes and all(
            snapshot['quiet'] and snapshot['temperature_under_80'] for snapshot in probes))
        result['other_benchmarks_parallel'] = False if result['all_recorded_resource_snapshots_quiet'] else 'UNKNOWN_OR_GUARD_DETECTED_COMPETITION'
        if pipeline is not None:
            result.update(actual_pipeline_calls=dict(pipeline.base.counts), actual_fresh_path_calls=dict(pipeline.counts),
                local_operation_counts=dict(pipeline.local_operation_counts), model_bindings=pipeline.bindings,
                model_forwards=dict(detector=pipeline.models.detector_forwards, N3=pipeline.models.n3_forwards,
                    IMAGE_ROLE=head_counts['IMAGE_ROLE:attempted'], detector_internal_initialization_calls=max(0,
                        pipeline.models.detector_forwards-pipeline.base.counts['detector_calls'])))
            if not result['complete'] and pipeline.last_capture is not None:
                interrupted = dict(full_fresh_capture=pipeline.last_capture,
                    fresh_point_result=pipeline.last_point_result, fresh_point_bank_ledger=pipeline.last_point_bank_ledger,
                    partial_local_bank_counts=None if pipeline.last_local_bank is None else dict(pipeline.last_local_bank.counts))
                try:
                    C.write_new(C.output_path(args,'INTERRUPTED_RUNTIME_PACKET.json'), interrupted)
                except BaseException as error:
                    cleanup_errors.append(dict(phase='interrupted_packet',type=type(error).__name__,message=str(error)))
            try:
                pipeline.close(); result['pipeline_close_complete'] = True
            except BaseException as error:
                result['pipeline_close_complete'] = False
                cleanup_errors.append(dict(phase='pipeline_close',type=type(error).__name__,message=str(error)))
        try:
            frozen(args)
            after = C.protect(args); result['prior_protection_after'] = after
            C.require(after == before_protection,'runtime changed protected published/source bytes')
        except BaseException as error:
            cleanup_errors.append(dict(phase='protection',type=type(error).__name__,message=str(error)))
        result['prior_protection_before'] = before_protection
        if writer is not None:
            try:
                writer.close(); C.require(writer.count == len(rows), 'runtime serialized summary count differs')
                C.require(not writer.close_errors, 'runtime close/fsync failed')
                if result['complete'] and not cleanup_errors:
                    writer.promote()
                else:
                    writer.preserve_interrupted()
            except BaseException as error:
                cleanup_errors.append(dict(phase='runtime_stream',type=type(error).__name__,message=str(error)))
                writer.preserve_interrupted()
            cleanup_errors.extend(writer.close_errors + writer.preservation_errors)
        if cleanup_errors:
            result.update(complete=False,status='FAILED_RUNTIME_CLEANUP')
        result.update(cleanup_errors=cleanup_errors, statistics_official=bool(result['complete']),
            runtime_rows_streamed_after_timed_interval=True, only_timing_summaries_retained_in_memory=True,
            full_original_candidate_witnesses_preserved=True, serialized_raw_rows=0 if writer is None else writer.count,
            raw_rows=C.binding(row_path) if row_path.is_file() else C.binding(writer.interrupted_path)
                if writer is not None and writer.interrupted_path.is_file() else None,
            summaries={arm:{stage:describe([row[stage+'_ms'] for row in rows if row['arm'] == arm and
                row['phase'] == 'measured' and row['parity_status'] == 'PASS']) for stage in ('full',)+STAGES+('return',)} for arm in ARMS},
            warmup_accounting={arm:dict(configured=20, completed=sum(row['arm'] == arm and row['phase'] == 'warmup' and
                row['parity_status'] == 'PASS' for row in rows)) for arm in ARMS})
        C.write_new(destination,result)
        durable_owned(journal,dict(status=result['status'],counts=dict(state),elapsed_seconds=result['elapsed_seconds']))
    print('SPARSE_LOCAL_FRESH_RUNTIME',result['status'],len(rows),flush=True)
    C.require(result['complete'],'fresh runtime incomplete; original receipt/prefix preserved')
    return result


def measure(args):
    with C.legacy_context(args):
        return _measure_with_context(args)


def main():
    parser = C.parser(__doc__,('freeze','preflight','measure','schedule'))
    parser.set_defaults(output=str(C.DOC))
    parser.add_argument('--accuracy-output',default=str(C.DOC))
    parser.add_argument('--runtime-protocol',default=str(C.DOC/'RUNTIME_PROTOCOL.json'))
    parser.add_argument('--base-weights')
    parser.add_argument('--N3-weights')
    parser.add_argument('--calibration',default=str(V7.V.DOC/'CALIBRATION.json'))
    parser.add_argument('--calibration-root',default=str(V7.CAL_ROOT))
    args = parser.parse_args()
    if args.stage == 'schedule':
        warm, measured = schedules(); print(dict(arms=ARMS,warmup=len(warm),measured=len(measured),total=600))
    else:
        {'freeze':freeze,'preflight':preflight,'measure':measure}[args.stage](args)


if __name__ == '__main__':
    main()
