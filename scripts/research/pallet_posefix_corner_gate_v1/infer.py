"""GT-free frozen predictions for the corner-selection screen.

The original image uses the already-frozen R0 and Replay predictions. A real
horizontal image flip is independently detected by frozen R0, then refined by
the same frozen Replay model. No annotation, metric, GT branch, or label file is
opened by this module. Geometry/selection and subsequent scoring are separate.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch

from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures

C = N.C
ROOT = N.ROOT
DOC = ROOT / '_docs/experiments/pallet_posefix_corner_gate_v1'
RAW = ROOT / 'data/pallet/results/pallet_posefix_corner_gate_v1'
CACHE = RAW / 'inference_cache'
PROTOCOL = DOC / 'PROTOCOL.json'
SAVED = N.RAW / 'PREDICTIONS.json'
PERM = np.array([1, 0, 3, 2, 5, 4, 7, 6, 8], dtype=int)
SCHEMA = 'pallet_posefix_corner_gate_v1.inference.2'
BENCH = DOC / 'CPU_BENCH_CPU4.json'
GPU_OVERRIDE = DOC / 'RUNTIME_OVERRIDE_GPU.json'


def freeze(path, value):
    """Never silently replace a completed result or a resumed cache row."""
    path = Path(path).resolve()
    assert path.is_relative_to(RAW) or path.is_relative_to(DOC), path
    if path.exists():
        assert N.E.read(path) == value, ('Immutable diagnostic differs', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(content)
    pending.replace(path)


def setup():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def serial_prediction(captured):
    """Copy only predictions; do not cache detector feature tensors."""
    candidates = []
    for candidate in captured['candidates']:
        output = {}
        for key, value in candidate.items():
            value = value.tolist() if hasattr(value, 'tolist') else value
            if isinstance(value, (list, tuple, float)):
                assert np.isfinite(np.asarray(value, dtype=float)).all(), key
            output[key] = value
        candidates.append(output)
    return dict(candidates=candidates, selected_index=captured['selected_index'])


def unflip_xy(points, width):
    points = np.asarray(points, dtype=np.float64).copy()
    assert points.shape == (9, 2)
    invalid = ~np.isfinite(points).all(-1) | (points == -1).all(-1)
    points[:, 0] = width - 1 - points[:, 0]
    points[invalid] = -1
    return points[PERM]


def unflip_prediction(prediction, width):
    result = copy.deepcopy(prediction)
    for candidate in result['candidates']:
        candidate['keypoints_xy'] = unflip_xy(candidate['keypoints_xy'], width).tolist()
        if 'keypoints_conf' in candidate:
            candidate['keypoints_conf'] = np.asarray(candidate['keypoints_conf'])[PERM].tolist()
        x0, y0, x1, y1 = candidate['box_xyxy']
        candidate['box_xyxy'] = [width - 1 - x1, y0, width - 1 - x0, y1]
    return result


def unflip_heatmap(summary, width):
    if summary is None:
        return None
    result = copy.deepcopy(summary)
    for key in ('expected_original_xy9', 'mode_original_xy9'):
        result[key] = unflip_xy(summary[key], width).tolist()
    for key in ('valid9', 'entropy_nats9', 'entropy_normalized9', 'peak_probability9', 'peak_mass7x7_9'):
        result[key] = np.asarray(summary[key])[PERM].tolist()
    result['coordinates'] = 'original unflipped image, fixed native flip permutation only'
    result['crop_matrix_note'] = 'crop_matrix is from the independently detected flipped image'
    return result


@torch.inference_mode()
def refine_summary(model, bgr, prediction):
    """One forward supplies the original expectation and mode/ambiguity probes."""
    result = copy.deepcopy(prediction)
    prepared = C.prepare_input(bgr, prediction)
    if prepared is None:
        return result, None
    device = next(model.parameters()).device
    args = [torch.as_tensor(prepared[key], device=device)[None]
            for key in ('rgb', 'points', 'valid')]
    logits = model(*args)
    assert logits.shape == (1, 9, 96, 72), tuple(logits.shape)
    assert torch.isfinite(logits).all()
    expected_crop = C.expectation(logits)[0].cpu().numpy()
    inverse = np.linalg.inv(prepared['matrix'])
    expected = C.transform_points(expected_crop, inverse)
    probabilities = logits[0].flatten(1).softmax(-1).reshape(9, 96, 72).cpu().numpy()
    modes = np.array([np.unravel_index(int(np.argmax(p)), p.shape)[::-1]
                      for p in probabilities], dtype=float)
    mode_xy = C.transform_points(modes * 4, inverse)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(axis=(1, 2))
    peak_probability, peak_mass = [], []
    for probability, (x, y) in zip(probabilities, modes.astype(int)):
        peak_probability.append(float(probability[y, x]))
        peak_mass.append(float(probability[max(y-3, 0):min(y+4, 96),
                                           max(x-3, 0):min(x+4, 72)].sum()))
    assert np.isfinite(expected).all() and np.isfinite(mode_xy).all()
    output = expected.copy()
    output[~prepared['valid']] = prepared['original_points'][~prepared['valid']]
    output[8] = prepared['original_points'][8]
    C.selected(result)['keypoints_xy'] = output.tolist()
    summary = dict(valid9=prepared['valid'].tolist(),
                   expected_original_xy9=expected.tolist(), mode_original_xy9=mode_xy.tolist(),
                   entropy_nats9=entropy.tolist(),
                   entropy_normalized9=(entropy / np.log(96 * 72)).tolist(),
                   peak_probability9=peak_probability, peak_mass7x7_9=peak_mass,
                   crop_matrix=prepared['matrix'].tolist(), heatmap_hw=[96, 72],
                   input_hw=[384, 288], heatmap_coordinate_multiplier=4,
                   coordinates='original image, native model keypoint indices 0..8',
                   centroid_preserved_in_prediction=True,
                   note='Diagnostic network expectation includes channel8; actual output keeps R0 centroid.')
    return result, summary


def bindings(device='cpu'):
    assert PROTOCOL.exists(), 'Root must freeze the GT-free selection protocol before inference'
    N.verify()
    fit = N.E.read(N.DOC / 'FIT.json')
    assert fit['complete'] and fit['step'] == 300
    N.F.verify(fit['checkpoint'])
    result = dict(protocol=N.E.bound(PROTOCOL), code=N.E.bound(Path(__file__)),
                saved_predictions=N.E.bound(SAVED), replay_checkpoint=fit['checkpoint'],
                replay_fit=N.E.bound(N.DOC / 'FIT.json'),
                replay_protocol=N.E.bound(N.DOC / 'PROTOCOL.json'),
                frozen_R0=N.E.bound(N.E.R0),
                features_code=N.E.bound(Path(__file__).parents[1] / 'pallet_line_pose_v1/features.py'),
                posefix_code=N.E.bound(Path(C.__file__)),
                architecture_code=N.E.bound(Path(__file__).parents[1] / 'pallet_sensors_submission_v1/prior_model.py'))
    if device == 'cuda':
        assert GPU_OVERRIDE.exists(), 'Explicit user-requested GPU runtime override must be recorded first'
        result['runtime_override'] = N.E.bound(GPU_OVERRIDE)
    return result


def load_models(source, device='cpu'):
    for value in source.values():
        N.F.verify(value)
    checkpoint = torch.load(ROOT / source['replay_checkpoint']['path'], map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 300
    assert checkpoint['protocol_sha256'] == source['replay_protocol']['sha256']
    model = C.PoseFixPallet9()
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device).eval().requires_grad_(False)
    # Passing a string makes Ultralytics select_device reset torch's thread
    # budget to its global NUM_THREADS. The torch.device early-return preserves
    # the explicitly bounded four-thread host execution on either backend.
    detector = FrozenYoloFeatures(N.E.R0, device=torch.device(device))
    assert all(parameter.device.type == device for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    if device == 'cpu':
        assert not torch.cuda.is_initialized(), 'The CPU mode must not initialize CUDA'
    assert torch.get_num_threads() == 4
    return model, detector


def records():
    saved = N.E.read(SAVED)
    assert set(saved) == {'DEV72', 'GREEN150'}
    assert len(saved['DEV72']) == 72 and len(saved['GREEN150']) == 150
    # Explicit allowlist: annotation paths/labels/GT metrics never leave this step.
    return [(dataset, {key: row[key] for key in ('id', 'image', 'raw_hw', 'predictions')})
            for dataset in ('DEV72', 'GREEN150') for row in saved[dataset]]


def cache_path(dataset, ident):
    safe = re.sub(r'[^A-Za-z0-9_.-]', '_', ident)
    return CACHE / f'{dataset}__{safe}.json'


@torch.inference_mode()
def infer_record(dataset, row, model, detector, source):
    start = time.monotonic()
    device = next(model.parameters()).device.type
    N.F.verify(row['image'])
    bgr = cv2.imread(str(ROOT / row['image']['path']))
    assert bgr is not None and list(bgr.shape[:2]) == row['raw_hw']
    r0 = copy.deepcopy(row['predictions']['R0'])
    saved = copy.deepcopy(row['predictions']['POSEFIX_RAW'])
    normal, heatmap = refine_summary(model, bgr, r0)
    selected_normal, selected_saved = C.selected(normal), C.selected(saved)
    assert (selected_normal is None) == (selected_saved is None)
    parity = 0.0
    if selected_normal is not None:
        current = np.asarray(selected_normal['keypoints_xy'])
        previous = np.asarray(selected_saved['keypoints_xy'])
        np.testing.assert_allclose(current[:8], previous[:8], atol=.01, rtol=0)
        np.testing.assert_array_equal(current[8], previous[8])
        parity = float(np.max(np.abs(current[:8] - previous[:8])))
    normal_seconds = time.monotonic() - start
    flip_start = time.monotonic()
    flipped_image = np.ascontiguousarray(bgr[:, ::-1])
    captured = detector.predict(flipped_image, cpu_features=True)
    assert torch.get_num_threads() == 4, 'Detector changed the four-thread CPU budget'
    flipped_raw = serial_prediction(captured)
    del captured
    flipped_refined, flipped_heatmap = refine_summary(model, flipped_image, flipped_raw)
    width = bgr.shape[1]
    if device == 'cpu':
        assert not torch.cuda.is_initialized()
    assert torch.get_num_threads() == 4
    return dict(schema=SCHEMA, dataset=dataset, id=row['id'], image=row['image'],
                raw_hw=row['raw_hw'], bindings=source,
                normal=dict(prediction=saved, heatmap=heatmap, parity_max_abs_px=parity,
                            prediction_source='Exact saved Replay; recomputed output used for parity and diagnostics only'),
                flip=dict(raw_prediction=flipped_raw, refined_prediction=flipped_refined,
                          raw_unflipped_prediction=unflip_prediction(flipped_raw, width),
                          refined_unflipped_prediction=unflip_prediction(flipped_refined, width),
                          heatmap_unflipped=unflip_heatmap(flipped_heatmap, width),
                          detected=C.selected(flipped_raw) is not None,
                          refiner_available=flipped_heatmap is not None,
                          missing_policy='Reject correction if missing/invalid; never fabricate flip output',
                          fixed_native_flip_permutation=PERM.tolist()),
                seconds=dict(normal=normal_seconds, flip=time.monotonic()-flip_start,
                             total=time.monotonic()-start),
                GT_used=False, training=False, device=device, cpu_threads=torch.get_num_threads())


def run(limit=None, device='cpu'):
    assert device in ('cpu', 'cuda')
    setup()
    source = bindings(device)
    bench_path = BENCH if device == 'cpu' else DOC / 'GPU_BENCH.json'
    if limit is None and (RAW / 'INFERENCE.json').exists():
        completed = N.E.read(RAW / 'INFERENCE.json')
        assert completed['complete'] and completed['count'] == 222
        assert completed['schema'] == SCHEMA and completed['bindings'] == source
        assert completed['device'] == device
        for binding in completed['cache_bindings']:
            N.F.verify(binding)
        for row in completed['records']:
            N.F.verify(row['image'])
        print(json.dumps(dict(stage='ALREADY_COMPLETE', count=222,
                              manifest=str(RAW / 'INFERENCE.json'))), flush=True)
        return completed
    rows = records()
    paths = [cache_path(dataset, row['id']) for dataset, row in rows]
    assert len(set(paths)) == len(paths), 'Sanitized cache-name collision'
    if limit is not None:
        assert 0 < limit <= len(rows)
        assert not bench_path.exists(), 'Benchmark already completed; read its result'
        rows = rows[:limit]
    started = time.monotonic()
    runtime_checks = []
    if device == 'cuda':
        runtime_checks.append(dict(stage='before_load', **N.E.gpu()))
    model, detector = load_models(source, device)
    all_records, cached = [], []
    try:
        for number, (dataset, row) in enumerate(rows, 1):
            if device == 'cuda' and number % 20 == 0:
                runtime_checks.append(dict(stage='during_inference', frame=number, **N.E.gpu()))
            path = cache_path(dataset, row['id'])
            if limit is None and path.exists():
                result = N.E.read(path)
                assert result['schema'] == SCHEMA and result['bindings'] == source
                assert result['device'] == device
                assert result['id'] == row['id'] and result['dataset'] == dataset
                assert result['image'] == row['image']
                N.F.verify(row['image'])
            else:
                result = infer_record(dataset, row, model, detector, source)
                if limit is None:
                    freeze(path, result)
            all_records.append(result)
            if limit is None:
                cached.append(N.E.bound(path))
            print(json.dumps(dict(stage='BENCH' if limit is not None else device.upper() + '_INFERENCE',
                                  number=number, total=len(rows), dataset=dataset, id=row['id'],
                                  seconds=result['seconds'],
                                  parity=result['normal']['parity_max_abs_px'],
                                  flip_detected=result['flip']['detected'])), flush=True)
    finally:
        detector.close()
        del model, detector
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    if device == 'cpu':
        assert not torch.cuda.is_initialized()
    else:
        runtime_checks.append(dict(stage='after_release', **N.E.gpu()))
    result = dict(schema=SCHEMA, complete=True, count=len(all_records), bindings=source,
                  elapsed_seconds=time.monotonic()-started, cpu_threads=torch.get_num_threads(), device=device,
                  GT_used=False, training=False, cuda_initialized=torch.cuda.is_initialized(),
                  runtime_checks=runtime_checks,
                  peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
                  fixed_native_flip_permutation=PERM.tolist(), records=all_records,
                  cache_bindings=cached,
                  parity_max_abs_px=max(r['normal']['parity_max_abs_px'] for r in all_records),
                  flip_missing=sum(not r['flip']['detected'] for r in all_records))
    if limit is not None:
        result['benchmark_only'] = True
        result['estimated_222_inference_seconds'] = float(np.mean([r['seconds']['total'] for r in all_records]) * 222)
        freeze(bench_path, result)
    else:
        assert len(all_records) == 222
        freeze(RAW / 'INFERENCE.json', result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('records', 'cache_bindings', 'bindings')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, help='Benchmark first N frozen rows; writes no main cache')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                        help='CUDA requires the explicit recorded runtime override; no automatic fallback')
    args = parser.parse_args()
    run(args.limit, args.device)
