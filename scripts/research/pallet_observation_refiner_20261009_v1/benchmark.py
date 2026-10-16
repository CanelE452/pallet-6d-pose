"""Actual synchronized RAM-RGB deployment paths; never coordinate replay.

The optional learned adapter is instantiated before timing. Its ``predict``
method receives only image, captured detector inputs, camera/registry metadata,
and the frozen model wrapper. It returns selected correspondences and the
initial geometry pose; this module still performs the same fresh robust solver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib
import importlib.metadata
import os
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from .inference import hidden_mask, finish
from .solver import HypothesisBank

ARMS = ('BASE', 'N3_SUBPIX', 'N3_SUBPIX_GEOM_NOSELF_ROBUST')
WARMUP, REPEATS, FRAMES = 20, 5, 26


def schedules(arms=ARMS):
    arms = tuple(arms)
    warm, measured = [], []
    for index in range(WARMUP):
        order = list(arms[index % len(arms):] + arms[:index % len(arms)])
        if index % 2:
            order.reverse()
        for position, arm in enumerate(order):
            warm.append(dict(phase='warmup', arm=arm, warmup_index=index,
                             arm_position=position, image_index=index % FRAMES))
    for repeat in range(REPEATS):
        order = list(arms[repeat % len(arms):] + arms[:repeat % len(arms)])
        if repeat % 2:
            order.reverse()
        for image_index in range(FRAMES):
            for position, arm in enumerate(order):
                measured.append(dict(phase='measured', arm=arm, repeat=repeat,
                                     arm_position=position, image_index=image_index))
    assert Counter(r['arm'] for r in warm) == {a: WARMUP for a in arms}
    assert Counter(r['arm'] for r in measured) == {a: REPEATS * FRAMES for a in arms}
    assert len(warm + measured) == len(arms) * 150 <= 600
    return warm, measured


def _references(panel, arms, learned_reference=None):
    """Reduce sealed outputs to coordinate/pose parity fields before inference."""
    ids = {r['frame_id'] for r in panel}
    references = {arm: {} for arm in arms}
    path = C.DOC / 'POSE_DIAGNOSTICS.jsonl.gz'
    for row in C.iter_rows(path):
        fid, method = row['id'], row['method']
        if fid not in ids:
            continue
        for arm in ('BASE', 'N3_SUBPIX'):
            if method == arm + '_NO_MASK_STANDARD':
                references[arm][fid] = dict(points=row['input_points'], pose=row['initial_pose'])
        if method == 'N3_SUBPIX_GEOM_NOSELF_ROBUST':
            references[method][fid] = dict(points=row['native_points'], pose=row['actual_pose'],
                                          output_status=row['output_status'], hidden=row['hidden_initial'])
    bindings = [C.binding(path)]
    if 'IMAGE_ROLE' in arms:
        if learned_reference is None:
            raise ValueError('IMAGE_ROLE needs sealed prediction references before runtime')
        learned_reference = Path(learned_reference)
        for row in C.iter_rows(learned_reference):
            if row['id'] in ids and row['method'] in ('IMAGE_ROLE', 'IMAGE_ROLE_GEOM_ROBUST'):
                references['IMAGE_ROLE'][row['id']] = dict(points=row['native_points'], pose=row['actual_pose'],
                    output_status=row['output_status'], hidden=row['hidden_initial'])
        bindings.append(C.binding(learned_reference))
    assert all(set(v) == ids for v in references.values()), 'Incomplete fixed panel parity references'
    return references, bindings


def _preserve_metadata(before, after):
    assert before['selected_index'] == after['selected_index']
    assert len(before['candidates']) == len(after['candidates'])
    for i, (a, b) in enumerate(zip(before['candidates'], after['candidates'])):
        assert set(a) == set(b)
        for key in a:
            if key != 'keypoints_xy':
                assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]), equal_nan=True), key
        p, q = np.asarray(a['keypoints_xy']), np.asarray(b['keypoints_xy'])
        if i != before['selected_index']:
            assert np.array_equal(p, q, equal_nan=True)
        else:
            assert np.array_equal(p[8], q[8], equal_nan=True)


def _pose_parity(actual, expected, atol=1e-7):
    assert actual['available'] == expected['available']
    assert actual.get('selected_hypothesis') == expected.get('selected_hypothesis')
    differences = {}
    if actual['available']:
        for key in ('R_cf', 'R_physical', 'centroid', 'cf_extents', 'reprojection_px'):
            a, b = np.asarray(actual[key], float), np.asarray(expected[key], float)
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            differences[key] = float(np.max(np.abs(a - b)))
            assert differences[key] <= atol, (key, differences[key], atol)
    return dict(atol=atol, rtol=0, max_abs_by_field=differences)


def _adapter(spec, models):
    if spec is None:
        return None
    module, factory = spec.split(':', 1)
    return getattr(importlib.import_module(module), factory)(models)


def benchmark_adapter(models):
    """Shared fixed-detector neck -> frozen IMAGE_ROLE boundary observations."""
    from . import learned_infer as L
    class Adapter:
        def __init__(self):
            self.head = L.load_head('IMAGE_ROLE')
            self.forwards = 0
            self.bindings = [C.binding(L.checkpoint_path('IMAGE_ROLE')), C.binding(Path(L.__file__)),
                             C.binding(Path(L.M.__file__))]
            self.contract = dict(arm='IMAGE_ROLE', checkpoint='last at update3000',
                parameters=sum(p.numel() for p in self.head.parameters()),
                detector='same captured BASE forward and FP16-rounded shared neck',
                feature_role_pose='fresh standard pose from predicted Base correspondences',
                final_mask_pose='separate unchanged historical Base pose',
                unobserved_corner_inputs='NaN, never Base-fill to meet minimum',
                fallback='full original Base coordinates and old Base pose')

        def predict(self, image, captured, panel_frame, pipeline):
            raw = dict(candidates=pipeline.inf.serial(captured['candidates']),
                       selected_index=captured['selected_index'])
            index = raw['selected_index']
            original = np.full((9, 2), np.nan) if index is None else np.asarray(raw['candidates'][index]['keypoints_xy'], float)
            frame = dict(K=panel_frame['camera_intrinsics'],
                xyz=np.asarray(panel_frame['dimensions_wdh_m'])[[0, 2, 1]].tolist(),
                points={'BASE': original}, raw_hw=list(image.shape[:2]))
            initial = pipeline.pose.infer(original, np.asarray(frame['K']), np.asarray(frame['xyz']), source=False)
            hidden, _ = hidden_mask(initial)
            observed = np.full((9, 2), np.nan)
            observed[8] = original[8]
            if index is None:
                diagnostic = dict(no_detection=True, corners=[], lines=[], queries=[], feature_pose_computed=False)
            else:
                diagnostic = L.observe(image, captured, frame, self.head, 'IMAGE_ROLE')
                self.forwards += 1
                seen = set()
                for corner in diagnostic['corners']:
                    k = int(corner['id'])
                    assert 0 <= k < 8 and k not in seen
                    seen.add(k)
                    observed[k] = corner['xy']
                diagnostic['feature_pose_computed'] = True
            return dict(prediction=raw, initial_pose=initial, original_points=original,
                        observation_points=observed, hidden=hidden, excluded=hidden,
                        diagnostic=diagnostic)
    return Adapter()


def run(adapter_spec=None, learned_reference=None, remaining_seconds=3600.):
    path, journal = C.DOC / 'RUNTIME.json', C.SCRATCH / 'RUNTIME_STARTED.json'
    assert not path.exists() and not journal.exists(), 'Preserve completed or interrupted runtime attempts'
    assert C.read(C.DOC / 'SOLVER_CHECKS.json')['passed']
    assert C.read(C.DOC / 'CONTRACT_AUDIT.json')['status'] == 'PASS'
    C.source_modules()
    from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R
    import torch
    arms = ARMS + (('IMAGE_ROLE',) if adapter_spec else ())
    warm, measured = schedules(arms)
    panel, images, expected_raw, bindings = R._panel()
    references, ref_bindings = _references(panel, arms, learned_reference)
    bindings += ref_bindings
    started = time.monotonic()
    rows, models, adapter, originals = [], None, None, {}
    state = Counter()
    environment = dict(python=__import__('sys').version.split()[0], torch=torch.__version__,
        numpy=np.__version__, opencv=cv2.__version__, ultralytics=importlib.metadata.version('ultralytics'),
        batch=1, thread_contract=dict(torch=4, opencv=1), interference_snapshots=[])
    result = dict(schema='observation_refiner_actual_runtime_v1', complete=False, status='FAILED_RUNTIME',
        runtime_code=C.binding(Path(__file__)), arms=list(arms), frames=26, panel_sessions=13,
        panel=[dict(id=r['frame_id'], session=r['session_id'], image_key=r['image_key']) for r in panel],
        warmup_per_arm=20, repeats=5, measured_per_arm=130, max_pipeline_calls=len(warm + measured),
        schedule='cyclic route rotation; odd blocks reversed; every fixed panel image in five repeats',
        boundaries=dict(full_pipeline='RAM native BGR -> detector once with shared neck capture -> correction/observation -> initial old prediction-only pose -> fresh robust bank/solve -> hidden replacement',
            all_initial_and_final_PnP_included=True, accuracy_cache_replay_used=False,
            excluded=['model/checkpoint load', 'RGB file/decode', 'GT evaluation', 'parity checks',
                      'durable journal writes', 'interference snapshots']),
        GT_inference_access=False, cached_coordinate_replay_used_for_timing=False,
        other_benchmarks_parallel=False, input_bindings=bindings)
    C.write(journal, dict(status='STARTED', configured_pipeline_calls=len(warm + measured), counts=dict(state)))
    try:
        probe = R._interference()
        environment['interference_snapshots'].append(dict(phase='before_models', **probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']:
            raise RuntimeError('RUNTIME_PENDING: competing workload/thermal guard; preserve other jobs')
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        torch.manual_seed(1)
        np.random.seed(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        environment['numeric'] = dict(matmul_tf32=False, cudnn_tf32=True, cudnn_benchmark=False,
            cudnn_deterministic=False, CUDA_version=torch.version.cuda, cuDNN_version=torch.backends.cudnn.version())
        models = R.Pipeline()
        adapter = _adapter(adapter_spec, models)
        bindings += models.bindings
        if adapter:
            bindings += getattr(adapter, 'bindings', [])
            result['learned_adapter_contract'] = getattr(adapter, 'contract', {})
        result['fixed_method_contract'] = models.contract
        environment['threadpools'] = [{k: v for k, v in pool.items() if k != 'filepath'}
                                     for pool in __import__('threadpoolctl').threadpool_info()]
        environment['actual_torch_threads'] = torch.get_num_threads()
        environment['actual_opencv_threads'] = cv2.getNumThreads()
        environment['device'] = dict(name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
        torch.cuda.reset_peak_memory_stats()
        for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'cornerSubPix'):
            originals[name] = getattr(cv2, name)
            def wrapped(*args, _name=name, **kwargs):
                state[_name] += 1
                return originals[_name](*args, **kwargs)
            setattr(cv2, name, wrapped)
        last_probe = time.monotonic()
        with torch.no_grad():
            for job in warm + measured:
                if time.monotonic() - started > remaining_seconds:
                    raise RuntimeError('RUNTIME_PENDING: fixed runtime operating budget exhausted')
                if time.monotonic() - last_probe >= 5 or (job['phase'] == 'measured' and job['image_index'] == 0 and job['arm_position'] == 0):
                    probe = R._interference()
                    environment['interference_snapshots'].append(dict(job=job, **probe))
                    last_probe = time.monotonic()
                    if not probe['quiet'] or not probe['gpu_temperature_under_80']:
                        raise RuntimeError('RUNTIME_PENDING: resource interference; contaminated samples retained')
                state['pipeline_calls_started'] += 1
                C.write(journal, dict(status='RUNNING', active_job=job, counts=dict(state)))
                frame, image, arm = panel[job['image_index']], images[job['image_index']], job['arm']
                K, xyz = np.asarray(frame['camera_intrinsics']), np.asarray(frame['dimensions_wdh_m'])[[0, 2, 1]]
                before_counts = {key: state[key] for key in originals}
                with C.no_truth_reads():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter_ns()
                    state['detector_calls'] += 1
                    captured = models.extractor.predict(image)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter_ns()
                    if arm == 'IMAGE_ROLE':
                        # Adapter owns only boundary observation + its necessary initial pose.
                        selected = adapter.predict(image, captured, frame, models)
                        prediction = selected['prediction']
                        initial = selected['initial_pose']
                        observation_points = np.asarray(selected['observation_points'], float)
                        original_points = np.asarray(selected['original_points'], float)
                        hidden = list(selected['hidden'])
                        excluded = list(selected.get('excluded', hidden))
                        diagnostic = selected.get('diagnostic')
                        state['learned_observation_calls'] += 1
                        state['initial_pose_calls'] += 1
                        torch.cuda.synchronize()
                        t2 = time.perf_counter_ns()
                    else:
                        correction_arm = 'BASE' if arm == 'BASE' else 'N3_SUBPIX'
                        prediction, diagnostic, _ = models.correct(correction_arm, image, captured, frame)
                        original_points, _ = R._raw(prediction)
                        observation_points = original_points
                        torch.cuda.synchronize()
                        t2 = time.perf_counter_ns()
                        initial = models.pose.infer(original_points, K, xyz, source=False)
                        state['initial_pose_calls'] += 1
                        hidden, _ = hidden_mask(initial) if arm.endswith('GEOM_NOSELF_ROBUST') else ([], {})
                        excluded = hidden
                    torch.cuda.synchronize()
                    t3 = time.perf_counter_ns()
                    final = None
                    if arm in ('IMAGE_ROLE', 'N3_SUBPIX_GEOM_NOSELF_ROBUST'):
                        if observation_points is None:
                            observation_points = np.full((9, 2), np.nan)
                            original_points = observation_points.copy()
                        bank = HypothesisBank(observation_points, K, xyz, image_size=(image.shape[1], image.shape[0]))
                        final = finish(bank, original_points, initial, excluded=excluded, hidden=hidden, robust=True)
                        state['robust_solver_calls'] += 1
                        final_points, pose = np.asarray(final['native_points']), final['actual_pose']
                        if arm == 'IMAGE_ROLE' and final['new_pose_estimated']:
                            # Selected nonhidden observations become the displayed corners;
                            # a fallback retains the entire preset original Base output.
                            observed = np.isfinite(observation_points[:8]).all(1)
                            observed[np.asarray(hidden, int)] = False
                            final_points[np.flatnonzero(observed)] = observation_points[:8][observed]
                            final['native_points'] = final_points
                        if arm == 'IMAGE_ROLE':
                            state['feature_role_pose_calls'] += int(diagnostic.get('feature_pose_computed', False))
                        prediction = copy.deepcopy(prediction)
                        if prediction['selected_index'] is not None:
                            prediction['candidates'][prediction['selected_index']]['keypoints_xy'] = final_points
                    else:
                        final_points, pose = original_points, initial
                    torch.cuda.synchronize()
                    t4 = time.perf_counter_ns()
                raw = dict(candidates=models.inf.serial(captured['candidates']), selected_index=captured['selected_index'])
                row = dict(**job, id=frame['frame_id'], session=frame['session_id'], full_ms=(t4 - t0) / 1e6,
                    detector_ms=(t1 - t0) / 1e6, correction_ms=(t2 - t1) / 1e6,
                    initial_pose_ms=(t3 - t2) / 1e6, robust_and_reprojection_ms=(t4 - t3) / 1e6,
                    learned_initial_pose_in_correction_interval=arm == 'IMAGE_ROLE',
                    raw_points=R._raw(raw)[0], final_points=final_points, actual_pose=pose,
                    selected_index=prediction['selected_index'], hidden_initial=hidden,
                    call_counts={k: state[k] - before_counts[k] for k in originals},
                    correction_diagnostic=diagnostic, output_status=final['output_status'] if final else 'HISTORICAL_FIXED_PATH',
                    solver=final['solver'] if final else None, GT_canary_active=True, parity_status='PENDING')
                rows.append(row)
                R._check_raw(raw, expected_raw[job['image_index']])
                _preserve_metadata(raw, prediction)
                ref = references[arm][frame['frame_id']]
                row['final_points_parity'] = R._point_parity(final_points, ref['points'], 1e-7, (arm, frame['frame_id']))
                row['final_pose_parity'] = _pose_parity(pose, ref['pose'])
                if final:
                    assert final['output_status'] == ref['output_status'] and hidden == ref['hidden']
                    assert not set(hidden) & set(final['solver']['fit_input_ids'])
                    if final['hidden_reprojected']:
                        np.testing.assert_allclose(final_points[hidden], np.asarray(final['solver']['projected'])[hidden], atol=1e-12, rtol=0)
                row.update(RAW_cached_bitexact=True, candidate_center_score_box_preserved=True, parity_status='PASS')
                state['pipeline_calls_complete'] += 1
                if state['pipeline_calls_complete'] % 50 == 0:
                    print('RUNTIME', state['pipeline_calls_complete'], len(warm + measured), 'seconds', round(time.monotonic() - started, 2), flush=True)
        assert state['pipeline_calls_complete'] == len(rows) == len(warm + measured)
        assert state['detector_calls'] == len(rows) and state['initial_pose_calls'] == len(rows)
        probe = R._interference()
        environment['interference_snapshots'].append(dict(phase='after_complete', **probe))
        if not probe['quiet'] or not probe['gpu_temperature_under_80']:
            raise RuntimeError('RUNTIME_PENDING: endpoint interference; samples retained but not official')
        result.update(status='DONE', complete=True)
        environment['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except Exception as error:
        result.update(status='RUNTIME_PENDING' if 'RUNTIME_PENDING' in str(error) else 'FAILED_RUNTIME',
                      reason=dict(type=type(error).__name__, message=str(error)))
    finally:
        for name, original in originals.items():
            setattr(cv2, name, original)
        if models is not None:
            result['model_forwards'] = dict(detector=models.detector_forwards, fixed_N3=models.n3_forwards,
                detector_internal_initialization_calls=max(0, models.detector_forwards - state['detector_calls']),
                duplicate_N3_backbone=0, learned=int(getattr(adapter, 'forwards', state['learned_observation_calls'])) if adapter else 0)
            models.close()
        result['environment'] = environment
        result['execution'] = dict(state, elapsed_seconds=time.monotonic() - started, updates=0,
            image_decode_calls=len(images), image_decode_outside_intervals=True, extra_GT_evaluation_calls=0)
        result['summaries'] = {arm: {stage: R.describe([r[stage + '_ms'] for r in rows if r['arm'] == arm
            and r['phase'] == 'measured' and r['parity_status'] == 'PASS'])
            for stage in ('full', 'detector', 'correction', 'initial_pose', 'robust_and_reprojection')}
            for arm in arms}
        result['warmup_accounting'] = {arm: dict(configured=20, completed=sum(r['arm'] == arm
            and r['phase'] == 'warmup' and r['parity_status'] == 'PASS' for r in rows)) for arm in arms}
        result['statistics_official'] = bool(result['complete'])
        rows_path = C.DOC / 'RUNTIME_ROWS.jsonl.gz'
        C.save_rows(rows_path, rows)
        result['raw_rows'] = C.binding(rows_path)
        result['input_bindings'] = bindings
        C.write(path, result)
        C.write(journal, dict(status=result['status'], counts=dict(state), elapsed_seconds=result['execution']['elapsed_seconds']))
    print('RUNTIME', result['status'], result['execution'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('schedule', 'measure'))
    parser.add_argument('--image-role-adapter')
    parser.add_argument('--learned-reference', type=Path)
    parser.add_argument('--remaining-seconds', type=float, default=3600.)
    args = parser.parse_args()
    if args.stage == 'schedule':
        arms = ARMS + (('IMAGE_ROLE',) if args.image_role_adapter else ())
        warm, measured = schedules(arms)
        print(dict(arms=arms, warmup=len(warm), measured=len(measured), total=len(warm + measured)))
    else:
        run(args.image_role_adapter, args.learned_reference, args.remaining_seconds)
