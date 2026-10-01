"""Measured RESNET18/D1/P1 runtime, with all attempts and cache replay preserved.

Run only after training, source selection, DEV inference and CPU evaluation.
The timer starts from a decoded native BGR array. It includes preprocessing,
forward, CPU decoder/head-rule math and the same prediction-only canonical pose
path as evaluation. File loading, module construction and accuracy checks are
outside the timer. No reference pose or keypoint GT is loaded by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from common import ROOT, HERE, DOC, RAW, DEV, read, write, sha, bound, now, gpu, verify_lock

ARMS = ('RESNET18', 'D1', 'P1')
REPEATS = 5
WARMUP = 20
POINT_ATOL = 1e-10
POSE_CODE = ROOT / 'scripts/paper/pose_metric_closure_v1'
POSE_CONTRACT = ROOT / 'data/pallet/results/paper_pose_metric_closure_v1/POSE_EVAL_OBJECT_CONTRACT.json'


def verify(binding):
    path = ROOT / binding['path']
    assert sha(path) == binding['sha256'], path
    if 'bytes' in binding:
        assert path.stat().st_size == binding['bytes'], path
    return path


def select_items(items):
    """First two in original manifest order per session, no sorting by result."""
    counts = Counter()
    selected = []
    for item in items:
        session = item['session_id']
        if counts[session] < 2:
            selected.append(item)
            counts[session] += 1
    assert len(counts) == 13 and set(counts.values()) == {2}
    assert len(selected) == 26
    return selected


def schedule(frames=26, repeats=REPEATS):
    """All arms at every frame/repeat, rotating order across adjacent blocks."""
    rows = []
    for repeat in range(repeats):
        for image_index in range(frames):
            offset = (repeat + image_index) % len(ARMS)
            for position in range(len(ARMS)):
                arm = ARMS[(offset + position) % len(ARMS)]
                rows.append(dict(repeat=repeat, image_index=image_index,
                                 arm=arm, arm_position=position))
    return rows


def compare_array(actual, expected, name, atol=POINT_ATOL):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    assert a.shape == e.shape, (name, a.shape, e.shape)
    assert not np.isinf(a).any() and not np.isinf(e).any(), name
    assert np.array_equal(np.isnan(a), np.isnan(e)), (name, 'missing mask')
    mask = np.isfinite(e)
    error = float(np.max(np.abs(a[mask] - e[mask]))) if mask.any() else 0.
    assert error <= atol, (name, error, atol)
    return error


def check_prediction(base, refined, cached, arm):
    """Strict fixed-output equivalence; no fastest/most-accurate trial choice."""
    assert np.array_equal(base['valid'], np.asarray(cached['point_valid'], bool))
    errors = [compare_array(base['points_original'], cached['base_points'], 'base points')]
    if cached['box_original'] is None:
        assert base['bbox_original'] is None
    else:
        assert base['bbox_original'] is not None
        errors.append(compare_array(base['bbox_original'], cached['box_original'], 'fixed box'))
    errors.append(compare_array(base['score'], cached['score'], 'fixed score'))
    errors.append(compare_array(base['confidence'], cached['confidence'], 'belief confidence'))
    errors.append(compare_array(base['affine_input_to_net'], cached['affine_input_to_net'], 'affine', 0.))
    points = base['points_original'] if arm == 'RESNET18' else refined[arm]
    expected = cached['base_points'] if arm == 'RESNET18' else cached['refined'][arm]
    errors.append(compare_array(points, expected, arm + ' points'))
    assert np.array_equal(np.isfinite(points).all(-1), cached['point_valid'])
    assert np.array_equal(points[8], base['points_original'][8], equal_nan=True)
    return max(errors)


def modules():
    # Import definitions only; never call the accuracy loader/main functions.
    for path in (ROOT, POSE_CODE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import evaluation as E
    M = importlib.import_module('run_pose_evaluation')
    G = importlib.import_module('pose_evaluation_paths')
    return E, M, G


def inputs():
    protocol = verify_lock()
    runtime = protocol['runtime']
    assert runtime['batch'] == 1 and runtime['warmup'] == WARMUP and runtime['repeats'] == REPEATS
    assert runtime['arms'] == ['RESNET18', 'RESNET18+D1', 'RESNET18+P1']
    assert runtime['images'] == 'first two by manifest order per session, 26 total'
    assert read(DOC / 'TRAINING_COMPLETE.json')['complete']
    assert read(DOC / 'SYNTHETIC_HELDOUT.json')['complete']
    inference_complete = read(DOC / 'DEV_INFERENCE_COMPLETE.json')
    assert inference_complete['complete'] and inference_complete['frames'] == 319
    prediction_path = verify(inference_complete['predictions'])
    assert prediction_path == RAW / 'DEV_PREDICTIONS.json'
    predictions = read(prediction_path)
    assert predictions['schema'] == 'resnet18_refiner_dev_predictions_v1'
    assert predictions['complete'] and predictions['coordinate_system'] == 'original_unpadded_pixels'
    assert predictions['protocol'] == bound(DOC / 'PROTOCOL.json')
    assert predictions['selection'] == bound(DOC / 'SELECTION.json')
    assert protocol['schema'] == 'resnet18_local_refiner_protocol_v1'
    for key in ('baseline', 'baseline_protocol', 'baseline_training_complete'):
        assert predictions[key] == protocol[key]
        verify(predictions[key])
    for value in predictions['checkpoints'].values():
        verify(value)
    for value in predictions['code']:
        verify(value)
    lock_path = DOC / 'DEV_EVALUATION_LOCK.json'
    lock = read(lock_path)
    assert lock['complete'] and lock['positive_frames'] == 319 and lock['sessions'] == 13
    assert verify(lock['predictions']) == prediction_path
    verify(lock['code'])
    # K/physical dimensions are already extracted and checked by evaluation.
    # The lock contains bindings to GT containers; those values are not opened.
    mapping = {row['frame_id']: row for row in lock['frame_mapping']}
    cached = {row['frame_id']: row for row in predictions['frames']}
    manifest = read(DEV)['items']
    assert len(mapping) == len(cached) == len(manifest) == 319
    assert set(mapping) == set(cached) == {row['frame_id'] for row in manifest}
    E, M, G = modules()
    contract = G.load_pose_object_contract(str(POSE_CONTRACT))
    selected, images, cameras, specs, cache_rows = [], [], [], [], []
    for item in select_items(manifest):
        fid = item['frame_id']; meta = mapping[fid]; cache = cached[fid]
        path = ROOT / item['image_path']
        key = str(path.resolve().relative_to(ROOT.resolve()))
        assert key == cache['image_key'] == meta['image_key']
        assert item['session_id'] == cache['session_id'] == meta['session_id']
        raw = path.read_bytes()
        import hashlib
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == cache['image_sha256'], path
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        assert image is not None and list(image.shape[:2]) == cache['original_hw'] == meta['original_hw']
        camera = np.asarray(meta['camera_intrinsics'], np.float64)
        assert camera.shape == (3, 3) and np.isfinite(camera).all()
        assert camera[0, 0] > 0 and camera[1, 1] > 0
        spec = G.object_spec(contract, meta['object_type'])
        dims = meta['dimensions_m']
        assert [spec['long_m'], spec['short_m'], spec['height_m']] == [dims['long'], dims['short'], dims['height']]
        spec = {key: float(spec[key]) for key in ('long_m', 'short_m', 'height_m')}
        selected.append(dict(frame_id=fid, session_id=item['session_id'], image=bound(path),
            original_hw=list(image.shape[:2]), object_type=meta['object_type'],
            camera_intrinsics=camera.tolist(), dimensions_m=dims))
        images.append(image); cameras.append(camera); specs.append(spec); cache_rows.append(cache)
    code_paths = [HERE / name for name in ('runtime.py', 'inference.py', 'evaluation.py',
        'resnet_adapter.py', 'model.py', 'input_data.py', 'base_common.py',
        'refiner.py', 'common.py', 'selection.py', 'train.py')]
    code_paths += [Path(M.__file__), Path(G.__file__), ROOT / 'challenge/evaluation_v2/pnp_selector.py']
    plan = dict(schema='resnet18_refiner_runtime_plan_v1', complete=True,
        protocol=bound(DOC / 'PROTOCOL.json'), predictions=bound(prediction_path),
        baseline=predictions['baseline'], baseline_protocol=predictions['baseline_protocol'],
        baseline_training_complete=predictions['baseline_training_complete'],
        evaluation_mapping=bound(lock_path), selection=bound(DOC / 'SELECTION.json'),
        training=bound(DOC / 'TRAINING_COMPLETE.json'), object_contract=bound(POSE_CONTRACT),
        manifest=bound(DEV), codes=[bound(path) for path in code_paths], selected=selected,
        arms=list(ARMS), frames=26, sessions=13, warmup_per_arm=WARMUP, repeats=REPEATS,
        measured_calls=26 * REPEATS * len(ARMS), schedule=schedule(), batch=1,
        point_replay_atol_px=POINT_ATOL, point_replay_rtol=0.,
        threads=dict(torch=1, opencv=1),
        timing_scope='Decoded BGR through preprocessing, RESNET18, actual CPU decoder/head selection rule, then canonical prediction-only pose. Model initialization, file loading and replay checks excluded.',
        pose_failure_policy='All calls retained, including no-box and fewer-than-six-corner cases; latency status accompanies every raw row.',
        CPU_decode_included=True, GPU_synchronized=True, reference_values_read=False,
        image_annotation_loaded=False, reused_input_only_K_dimensions_from_evaluation_lock=True,
        retries_automatic=0, fastest_trial_selection=False,
        historical_YOLO_latency='Separate historical panel only; no direct timing rank against these newly measured RESNET18 arms.')
    return plan, images, cameras, specs, cache_rows, (E, M, G)


def measure(predictor, image, camera, spec, arm, pose_modules):
    E, M, G = pose_modules
    torch.cuda.synchronize(predictor.device)
    start = time.perf_counter_ns()
    base, refined, _ = predictor.predict(image, arms=[] if arm == 'RESNET18' else [arm])
    torch.cuda.synchronize(predictor.device)
    keypoints_end = time.perf_counter_ns()
    points = base['points_original'] if arm == 'RESNET18' else refined[arm]
    pose = (E.solve_without_gt(points, camera, spec, M, G)
            if base['bbox_original'] is not None else dict(status='NO_DETECTION'))
    end = time.perf_counter_ns()
    result = dict(keypoints_ms=(keypoints_end-start)/1e6,
        pose_ms=(end-keypoints_end)/1e6, full_ms=(end-start)/1e6,
        pose_status=pose['status'], finite_corners=int(base['valid'][:8].sum()),
        has_box=base['bbox_original'] is not None)
    if 'reason' in pose:
        result['pose_failure_reason'] = pose['reason']
    assert np.isfinite([result[key] for key in ('keypoints_ms', 'pose_ms', 'full_ms')]).all()
    assert result['keypoints_ms'] > 0 and result['full_ms'] >= result['keypoints_ms']
    return result, base, refined


def append(file, value):
    file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    file.flush(); os.fsync(file.fileno())


def describe(values):
    a = np.asarray(values, np.float64)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), minimum=float(a.min()), maximum=float(a.max()))


def summarize(rows):
    result = {}
    for arm in ARMS:
        chosen = [row for row in rows if row['arm'] == arm]
        assert len(chosen) == 26 * REPEATS
        result[arm] = dict(**{key: describe([row[key] for row in chosen])
            for key in ('keypoints_ms', 'pose_ms', 'full_ms')},
            pose_status_counts=dict(Counter(row['pose_status'] for row in chosen)),
            max_cache_replay_abs_px=max(row['cache_replay_max_abs_px'] for row in chosen),
            failures_included=True)
    return result


def run():
    assert not (DOC / 'RUNTIME.json').exists(), 'Completed timing is frozen; no repeat/fastest selection'
    torch.set_num_threads(1); cv2.setNumThreads(1)
    plan, images, cameras, specs, cache_rows, pose_modules = inputs()
    write(DOC / 'RUNTIME_PLAN.json', plan)
    directory = RAW / 'runtime'; directory.mkdir(parents=True, exist_ok=True)
    previous = sorted(directory.glob('attempt_*'))
    attempt = directory / f'attempt_{len(previous)+1:04d}'
    prior = []
    for old in previous:
        assert (old / 'FAILED.json').exists(), ('Unclosed prior attempt requires review', old)
        prior.append(dict(failure=bound(old / 'FAILED.json'),
                          raw_rows=bound(old / 'ROWS.jsonl')))
    attempt.mkdir()
    started = now(); rows = []; warmup_rows = []
    status = None
    with (attempt / 'ROWS.jsonl').open('x') as file:
        try:
            status = gpu()
            from inference import Predictor
            predictor = Predictor(device='cuda', heads=['D1', 'P1'])
            assert predictor.device.type == 'cuda'
            write(attempt / 'START.json', dict(started=started, plan=bound(DOC / 'RUNTIME_PLAN.json'),
                prior_attempts=prior, gpu=status))
            for j in range(WARMUP):
                for position in range(len(ARMS)):
                    arm = ARMS[(j + position) % len(ARMS)]; i = j % len(images)
                    timing, base, refined = measure(predictor, images[i], cameras[i], specs[i], arm, pose_modules)
                    row = dict(phase='warmup', warmup_index=j, arm=arm, image_index=i,
                        frame_id=plan['selected'][i]['frame_id'], **timing)
                    # Preserve raw latency even if the accuracy replay check fails.
                    append(file, dict(row, replay_status='PENDING'))
                    error = check_prediction(base, refined, cache_rows[i], arm)
                    append(file, dict(phase='warmup_replay', warmup_index=j, arm=arm,
                                     cache_replay_max_abs_px=error, PASS=True))
                    row['cache_replay_max_abs_px'] = error; warmup_rows.append(row)
            for index, job in enumerate(plan['schedule']):
                i = job['image_index']; arm = job['arm']
                timing, base, refined = measure(predictor, images[i], cameras[i], specs[i], arm, pose_modules)
                row = dict(phase='measured', call=index+1, **job,
                    frame_id=plan['selected'][i]['frame_id'], session_id=plan['selected'][i]['session_id'], **timing)
                append(file, dict(row, replay_status='PENDING'))
                error = check_prediction(base, refined, cache_rows[i], arm)
                append(file, dict(phase='measured_replay', call=index+1,
                    cache_replay_max_abs_px=error, PASS=True))
                row['cache_replay_max_abs_px'] = error; rows.append(row)
                if (index+1) % 78 == 0:
                    print('RESNET18_RUNTIME_CALLS', index+1, len(plan['schedule']), flush=True)
            assert len(rows) == plan['measured_calls'] and len(warmup_rows) == WARMUP * len(ARMS)
            for value in plan['codes']:
                verify(value)
            assert bound(DOC / 'PROTOCOL.json') == plan['protocol']
            assert bound(RAW / 'DEV_PREDICTIONS.json') == plan['predictions']
            result = dict(schema='resnet18_refiner_runtime_v1', complete=True, PASS=True,
                started=started, finished=now(), plan=bound(DOC / 'RUNTIME_PLAN.json'),
                attempt=attempt.name, raw_rows=bound(attempt / 'ROWS.jsonl'),
                previous_attempts=prior, code=bound(__file__), gpu=status,
                warmup_calls=len(warmup_rows), measured_calls=len(rows),
                frames=26, sessions=13, repeats=REPEATS, batch=1,
                all_cache_replays_PASS=True, point_atol_px=POINT_ATOL,
                results=summarize(rows), measurements=rows,
                no_fastest_selection=True, partial_or_failed_rows_discarded=False,
                prior_failed_attempts_retained_separately_not_pooled_into_completed_summary=True,
                reference_values_read=False, new_training=0,
                historical_YOLO_latency_is_separate=True)
            write(attempt / 'COMPLETE.json', result)
            write(DOC / 'RUNTIME.json', result)
        except BaseException as exc:
            file.flush(); os.fsync(file.fileno())
            write(attempt / 'FAILED.json', dict(complete=False, PASS=False, started=started,
                failed_at=now(), error_type=type(exc).__name__, error=str(exc),
                measured_replays_completed=len(rows), warmup_replays_completed=len(warmup_rows),
                raw_rows=bound(attempt / 'ROWS.jsonl'), plan=bound(DOC / 'RUNTIME_PLAN.json'),
                code=bound(__file__), prior_attempts=prior, no_automatic_retry=True))
            raise
    print('RESNET18_RUNTIME_COMPLETE', result['measured_calls'], sha(DOC / 'RUNTIME.json'), flush=True)
    return result


def selfcheck():
    items = [dict(session_id=f's{i%13}', frame_id=str(i)) for i in range(52)]
    chosen = select_items(items)
    assert [row['frame_id'] for row in chosen] == [str(i) for i in range(26)]
    jobs = schedule(); assert len(jobs) == 390
    assert Counter(row['arm'] for row in jobs) == {arm:130 for arm in ARMS}
    for arm in ARMS:
        positions = Counter(row['arm_position'] for row in jobs if row['arm'] == arm)
        assert max(positions.values()) - min(positions.values()) <= 1
    points = np.arange(18, dtype=float).reshape(9, 2); points[2] = np.nan
    valid = np.isfinite(points).all(-1); box = np.array([0., 1., 16., 17.])
    base = dict(points_original=points.copy(), valid=valid, bbox_original=box,
                score=.5, confidence=np.ones(9), affine_input_to_net=np.eye(3))
    cached = dict(base_points=points.tolist(), point_valid=valid.tolist(), box_original=box.tolist(),
                  score=.5, confidence=[1.]*9, affine_input_to_net=np.eye(3).tolist(),
                  refined={arm:points.tolist() for arm in ('D1','P1')})
    for arm in ARMS:
        assert check_prediction(base, {arm:points.copy()}, cached, arm) == 0.
    changed = points.copy(); changed[0,0] += 1e-9
    try:
        check_prediction(base, {'D1':changed}, cached, 'D1')
    except AssertionError:
        pass
    else:
        raise AssertionError('Replay tolerance failed to reject changed point')
    assert describe([1., 2., 3.])['mean'] == 2.
    print(json.dumps(dict(PASS=True, invented_only=True, actual_forward_calls=0,
        actual_reference_reads=0, schedule_calls=390, warmup_calls=60)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['selfcheck', 'measure'])
    args = parser.parse_args()
    if args.phase == 'selfcheck':
        selfcheck()
    else:
        run()
