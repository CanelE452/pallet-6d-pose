"""Resumable synthetic-only frozen YOLO feature and loss-target cache.

Main cache: <run-dir>/cache, requires an existing TRAIN_PROTOCOL.json.
--limit N creates a separate evenly spaced manifest-index smoke sample under
smoke/cache_limit_N. --stop-after-chunks is allowed only for smoke resume tests.
Array files alone never establish completion: committed chunk hashes, the
completion bitmap, source bindings and CACHE_COMPLETE.json must agree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.modules.head import Pose26
from ultralytics.utils import ops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as FX
import source_data as SD


SPECS = {
    'p3': ('float16', (64, 80, 80)), 'p4': ('float16', (128, 40, 40)),
    'points': ('float32', (9, 2)), 'boxes': ('float32', (4,)),
    'point_valid': ('bool', (9,)), 'gt_points': ('float32', (9, 2)),
    'gt_valid': ('bool', (9,)), 'gt_support': ('bool', (8,)),
    'input_shape': ('int16', (2,)), 'matched': ('bool', ()), 'detected': ('bool', ()),
    'score': ('float32', ()), 'gain': ('float64', ()), 'record_index': ('int32', ()),
    'matched_gt_index': ('int32', ()), 'matched_iou': ('float32', ()),
    'selected_candidate_index': ('int32', ()),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def dependencies():
    paths = [HERE/'features.py', HERE/'source_data.py', Path(__file__).resolve()]
    paths += [Path(inspect.getfile(obj)) for obj in (YOLO, BasePredictor, LetterBox, Pose26, ops.scale_coords)]
    return {str(path.resolve()): FX.sha(path) for path in dict.fromkeys(paths)}


def bindings(run_dir, source, indices, chunk_size, smoke):
    train_protocol = run_dir/'TRAIN_PROTOCOL.json'
    if not smoke and not train_protocol.is_file():
        raise ValueError('Full cache requires the root-frozen TRAIN_PROTOCOL.json')
    return dict(schema='pallet_line_pose_feature_cache_manifest_v1', stage='smoke' if smoke else 'main',
        source_manifest=str(run_dir/'SOURCE_MANIFEST.json'), source_manifest_sha256=FX.sha(run_dir/'SOURCE_MANIFEST.json'),
        source_audit_sha256=FX.sha(run_dir/'SOURCE_DATA_AUDIT.json'),
        train_protocol_sha256=None if smoke else FX.sha(train_protocol),
        dependency_sha256=dependencies(), backbone=str(FX.BASELINE), backbone_sha256=FX.BASELINE_SHA,
        n_records=len(indices), record_indices=indices, record_ids=[source['records'][i]['id'] for i in indices],
        index_order='Original SOURCE_MANIFEST index order; smoke uses evenly spaced global indices, independent of GT and predictions.',
        chunk_size=chunk_size, arrays={k: dict(file=k+'.npy', dtype=dtype, shape=[len(indices), *shape]) for k, (dtype, shape) in SPECS.items()},
        input_recipe=source['input_recipe'], feature_recipe='Actual frozen neck P3/P4, FP16 rounded; zero-pad only right/bottom to80x80/40x40. Actual input_shape must mask spatial padding.',
        point_recipe='Highest predicted box score first, no GT; branch_inputs maps returned canvas xy to label-affine input pixels. All finite9points observed; no kp_conf threshold.',
        target_recipe='SD.load_loss_targets with actual hooked input shape, then SD.match_loss_target on already-selected box at IoU>=0.5. Unmatched/missing rows retained with gt_valid/support false and gt_points NaN. No GT influences detection/feature extraction.',
        support_recipe='Matched SD.edge_valid; model loss separately requires finite endpoints and length>=2 input pixels.',
        minimum_match_iou=.5, random_seed_for_sampling=None)


def open_arrays(directory, manifest):
    arrays = {}
    for name, spec in manifest['arrays'].items():
        path = directory/spec['file']
        if path.exists():
            value = np.load(path, mmap_mode='r+')
            if list(value.shape) != spec['shape'] or str(value.dtype) != spec['dtype']:
                raise ValueError(f'Existing array contract differs: {name}')
        else:
            if (directory/'shards').exists() and any((directory/'shards').glob('*.json')):
                raise ValueError(f'Committed chunks exist but array is missing: {name}')
            value = np.lib.format.open_memmap(path, mode='w+', dtype=spec['dtype'], shape=tuple(spec['shape']))
        arrays[name] = value
    path = directory/'done.npy'
    if path.exists():
        done = np.load(path, mmap_mode='r+')
        if done.shape != (manifest['n_records'],) or done.dtype != np.dtype('bool'):
            raise ValueError('Completion bitmap contract differs')
    else:
        done = np.lib.format.open_memmap(path, mode='w+', dtype='bool', shape=(manifest['n_records'],))
        done[:] = False; done.flush()
    return arrays, done


def flush_arrays(directory, arrays):
    for name, value in arrays.items():
        value.flush()
        with (directory/(name+'.npy')).open('rb') as handle:
            os.fsync(handle.fileno())


def slice_hashes(arrays, start, end):
    return {name: hashlib.sha256(np.asarray(value[start:end]).tobytes(order='C')).hexdigest()
            for name, value in arrays.items()}


def chunk_path(directory, start, end):
    return directory/'shards'/f'{start:06d}_{end:06d}.json'


def restore_committed(directory, manifest, arrays, done, *, verify_data=True):
    count = 0
    for start in range(0, manifest['n_records'], manifest['chunk_size']):
        end = min(start+manifest['chunk_size'], manifest['n_records'])
        marker = chunk_path(directory, start, end)
        if not marker.exists():
            done[start:end] = False
            continue
        value = read(marker)
        if value['manifest_sha256'] != FX.sha(directory/'CACHE_MANIFEST.json') or value['start'] != start or value['end'] != end:
            raise ValueError(f'Chunk identity changed: {marker}')
        if verify_data and slice_hashes(arrays, start, end) != value['array_slice_sha256']:
            raise ValueError(f'Committed chunk data hash mismatch: {marker}')
        if not np.array_equal(arrays['record_index'][start:end], manifest['record_indices'][start:end]):
            raise ValueError('Cached global record indices changed')
        done[start:end] = True
        count += end-start
    done.flush()
    return count


def plain_parity(model, image, captured):
    result = model.predict(image, conf=.001, imgsz=640, rect=True, augment=False,
                           half=False, device='cuda', verbose=False, save=False, stream=False)[0]
    count = 0 if result.boxes is None else len(result.boxes)
    if count != len(captured['candidates']):
        raise ValueError('Feature hooks changed detector candidate count')
    for index, candidate in enumerate(captured['candidates']):
        for observed, expected in ((candidate['box_xyxy'], result.boxes.xyxy[index].cpu().numpy()),
            (candidate['keypoints_xy'], result.keypoints.xy[index].cpu().numpy()),
            (candidate['keypoints_conf'], result.keypoints.conf[index].cpu().numpy()),
            (candidate['score'], result.boxes.conf[index].item())):
            if not np.array_equal(np.asarray(observed), np.asarray(expected)):
                raise ValueError('Feature hook versus plain YOLO detector numerical parity failed')


def cache_row(record, detector, *, parity_model=None, input_capture=None):
    # This call is deliberately before reading targets or performing GT matching.
    sample = SD.load_inference_sample(record, verify_image_hash=True)
    captured = detector.predict(sample['image_bgr'], already_padded=True, cpu_features=True)
    selected = FX.branch_inputs(captured)
    tf = SD.transform(sample['image_bgr'].shape[:2], captured['input_shape'])
    if tf.to_dict() != sample['transform'].to_dict():
        raise ValueError('Source label affine does not match actual model input shape')
    if parity_model is not None:
        plain_parity(parity_model, sample['image_bgr'], captured)
        # CUDA division uses different float32 arithmetic from NumPy CPU /255.
        # Compare underlying integer pixels and reproduce the actual CUDA op,
        # without changing the canonical detector's input or the source helper.
        expected_pixels = np.rint(sample['image_chw']*255).astype(np.uint8)
        actual_pixels = np.rint(input_capture['input']*255).astype(np.uint8)
        expected_cuda = torch.from_numpy(expected_pixels).to('cuda').float().div_(255).cpu().numpy()
        input_capture['cpu_normalization_max_abs'] = max(input_capture.get('cpu_normalization_max_abs', 0.),
            float(np.max(np.abs(input_capture['input']-sample['image_chw']))))
        if not np.array_equal(actual_pixels, expected_pixels) or not np.array_equal(input_capture['input'], expected_cuda):
            raise ValueError('Actual YOLO input pixels differ from source preprocessing')
    if FX.sha(record['label']) != record['label_sha256']:
        raise ValueError('Source label changed after manifest freeze')
    targets = SD.load_loss_targets(record, tf)
    row = {name: np.zeros(shape, dtype=dtype) for name, (dtype, shape) in SPECS.items()}
    row.update(points=np.full((9, 2), np.nan, np.float32), boxes=np.full(4, np.nan, np.float32),
        gt_points=np.full((9, 2), np.nan, np.float32), input_shape=np.array(captured['input_shape'], np.int16),
        gain=np.float64(tf.gain), record_index=np.int32(record['index']),
        matched_gt_index=np.int32(-1), selected_candidate_index=np.int32(-1))
    for name, stride in (('p3', 8), ('p4', 16)):
        value = captured[name].numpy()
        if value.dtype != np.float16 or value.shape != SPECS[name][1] or not np.isfinite(value).all():
            raise ValueError('Invalid feature tensor shape/dtype/value')
        h, w = captured['input_shape'][0]//stride, captured['input_shape'][1]//stride
        if np.any(value[:, h:, :]) or np.any(value[:, :, w:]):
            raise ValueError('Feature padding outside the actual input is not zero')
        row[name] = value
    if selected is not None:
        # Stable highest-score selection is checked without GT.
        index = SD.select_top1([candidate['score'] for candidate in captured['candidates']])
        if index != captured['selected_index'] or index != selected['candidate_index'] or selected['gain'] != tf.gain:
            raise ValueError('Detector selection/gain contract differs')
        row.update(points=selected['points'], boxes=selected['boxes'], point_valid=selected['point_valid'],
                   detected=np.bool_(True), score=np.float32(captured['candidates'][index]['score']),
                   selected_candidate_index=np.int32(index))
        match = SD.match_loss_target(selected['boxes'], targets['boxes_xyxy'], minimum_iou=.5)
        row['matched_iou'] = np.float32(match['iou'])
        if match['matched']:
            gt_index = match['target_index']
            gt_points, gt_valid = targets['kps'][gt_index], targets['kp_valid'][gt_index]
            row.update(matched=np.bool_(True), matched_gt_index=np.int32(gt_index),
                gt_points=gt_points.astype(np.float32), gt_valid=gt_valid,
                gt_support=targets['edge_valid'][gt_index])
            raw = np.asarray(record['targets'][gt_index]['keypoints_normalized'])[:, :2]
            raw *= np.asarray(record['prepared_shape_hw'][::-1])
            if not np.allclose(tf.input_to_prepared(gt_points[gt_valid]), raw[gt_valid], atol=1e-9, rtol=1e-12):
                raise ValueError('Matched target coordinate roundtrip failed')
    if not row['matched'] and (row['gt_valid'].any() or row['gt_support'].any() or np.isfinite(row['gt_points']).any()):
        raise ValueError('An unmatched row acquired a loss target')
    return row


def run(run_dir, *, limit=0, chunk_size=128, stop_after_chunks=None):
    if not (run_dir/'PURPOSE.md').is_file():
        raise ValueError('Missing experiment PURPOSE.md')
    source = read(run_dir/'SOURCE_MANIFEST.json')
    audit = read(run_dir/'SOURCE_DATA_AUDIT.json')
    if (source['schema'] != 'pallet_line_pose_source_v1' or not audit['PASS']
            or audit['manifest_sha256'] != FX.sha(run_dir/'SOURCE_MANIFEST.json')
            or audit['source_data_py_sha256'] != FX.sha(HERE/'source_data.py')):
        raise ValueError('Source manifest/audit binding failed')
    records = source['records']
    if len(records) != 60000 or [r['index'] for r in records] != list(range(60000)):
        raise ValueError('Expected exactly the fixed ordered60000 source records')
    if chunk_size < 1 or limit < 0 or limit > len(records) or (stop_after_chunks is not None and (not limit or stop_after_chunks < 1)):
        raise ValueError('Invalid chunk/smoke/interruption options')
    indices = list(range(len(records))) if not limit else np.linspace(0, len(records)-1, limit, dtype=np.int64).tolist()
    directory = run_dir/'cache' if not limit else run_dir/'smoke'/f'cache_limit_{limit}'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory/'EXTRACTION.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = bindings(run_dir, source, indices, chunk_size, bool(limit))
    path = directory/'CACHE_MANIFEST.json'
    if path.exists():
        if read(path) != manifest:
            raise ValueError('Cache manifest/config/code/ordered index binding differs')
    else:
        needed = sum(np.prod(v['shape'])*np.dtype(v['dtype']).itemsize for v in manifest['arrays'].values())
        if shutil.disk_usage(directory).free < needed+2*1024**3:
            raise ValueError('Insufficient free disk for complete cache plus2GiB margin')
        write(path, manifest)
    (directory/'shards').mkdir(exist_ok=True)
    arrays, done = open_arrays(directory, manifest)
    restored = restore_committed(directory, manifest, arrays, done)
    if (directory/'CACHE_COMPLETE.json').exists():
        completion = read(directory/'CACHE_COMPLETE.json')
        shard_hashes = {p.name: FX.sha(p) for p in sorted((directory/'shards').glob('*.json'))}
        if (not done.all() or not completion['complete'] or not completion['PASS']
                or completion['manifest_sha256'] != FX.sha(path)
                or completion['n_records'] != len(indices) or completion['completed_rows'] != len(indices)
                or completion['shard_marker_sha256'] != shard_hashes
                or completion['bitmap_sha256'] != FX.sha(directory/'done.npy')):
            raise ValueError('Completion marker does not match verified committed chunks')
        print(f'Complete cache verified, no model loaded: {directory}', flush=True)
        return directory
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    detector = FX.FrozenYoloFeatures(device='cuda')
    parity_model = YOLO(str(FX.BASELINE), task='pose') if limit else None
    input_capture = {}
    handle = detector.yolo.model.model[0].register_forward_pre_hook(
        lambda module, inputs: input_capture.update(input=inputs[0][0].detach().cpu().numpy().copy())) if limit else None
    new_chunks, extracted = 0, 0
    started = time.perf_counter()
    try:
        for start in range(0, len(indices), chunk_size):
            end = min(start+chunk_size, len(indices))
            if done[start:end].all():
                continue
            for local in range(start, end):
                row = cache_row(records[indices[local]], detector, parity_model=parity_model, input_capture=input_capture)
                for name, value in row.items():
                    arrays[name][local] = value
                extracted += 1
            flush_arrays(directory, arrays)
            hashes = slice_hashes(arrays, start, end)
            marker = dict(complete=True, manifest_sha256=FX.sha(path), start=start, end=end,
                          array_slice_sha256=hashes, source_record_indices=indices[start:end])
            write(chunk_path(directory, start, end), marker)
            done[start:end] = True; done.flush()
            new_chunks += 1
            progress = dict(complete=False, total=len(indices), completed=int(done.sum()),
                resumed_rows=restored, rows_extracted_this_process=extracted, elapsed_seconds=time.perf_counter()-started,
                manifest_sha256=FX.sha(path))
            write(directory/'PROGRESS.json', progress)
            print(f'Feature cache {int(done.sum())}/{len(indices)} rows; resumed{restored}; chunk{new_chunks}', flush=True)
            if stop_after_chunks is not None and new_chunks >= stop_after_chunks:
                write(directory/'SMOKE_INTERRUPTION.json', dict(intentional=True, complete=False,
                    committed_rows=int(done.sum()), final_marker_exists=False, manifest_sha256=FX.sha(path)))
                return directory
    finally:
        if handle is not None:
            handle.remove()
        detector.close()
    if not done.all() or dependencies() != manifest['dependency_sha256'] or FX.sha(run_dir/'SOURCE_MANIFEST.json') != manifest['source_manifest_sha256']:
        raise ValueError('Incomplete cache or source changed during extraction')
    chunk_hashes = {p.name: FX.sha(p) for p in sorted((directory/'shards').glob('*.json'))}
    completion = dict(schema='pallet_line_pose_feature_cache_complete_v1', complete=True, PASS=True,
        manifest_sha256=FX.sha(path), n_records=len(indices), completed_rows=int(done.sum()),
        shard_marker_sha256=chunk_hashes, bitmap_sha256=FX.sha(directory/'done.npy'),
        source_sha256=manifest['dependency_sha256'], source_manifest_sha256=manifest['source_manifest_sha256'],
        detected=int(arrays['detected'].sum()), matched=int(arrays['matched'].sum()),
        n_valid_gt_corners=int(arrays['gt_valid'].sum()), n_supported_gt_lines=int(arrays['gt_support'].sum()),
        source_counts=dict(Counter(records[i]['source'] for i in indices)),
        shape_counts=dict(Counter(str(records[i]['prepared_shape_hw']) for i in indices)),
        partition_counts=dict(Counter(records[i]['partition'] for i in indices)),
        resumed_rows=restored, rows_extracted_this_process=extracted,
        smoke_plain_detector_candidate_parity=bool(limit), smoke_actual_input_pixels_exact=bool(limit),
        smoke_actual_cuda_normalization_exact=bool(limit),
        smoke_numpy_vs_cuda_normalization_max_abs=input_capture.get('cpu_normalization_max_abs'),
        normalization_note='Source helper NumPy CPU /255 can differ from canonical Torch CUDA /255 by float32 roundoff. Smoke checks identical integer input pixels and bit-exact canonical CUDA normalization; feature extraction always uses the unchanged canonical detector.',
        image_and_label_sha256_checked_per_extracted_row=True, missing_and_unmatched_rows_retained=True,
        elapsed_seconds_this_process=time.perf_counter()-started,
        runtime=dict(python=sys.executable, torch=torch.__version__, ultralytics=ultralytics.__version__,
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), inference_batch=1,
                     cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=False),
        scope='Cache extraction and label transform completion. Smoke rows are isolated; no neural training or real-model selection performed.')
    write(directory/'CACHE_COMPLETE.json', completion)
    write(directory/'PROGRESS.json', dict(complete=True, total=len(indices), completed=len(indices), manifest_sha256=FX.sha(path)))
    print(f'Feature cache complete: {directory}', flush=True)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Nonzero: isolated smoke sample, never the main cache')
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--stop-after-chunks', type=int, help='Intentional smoke-only interruption for resume verification')
    args = parser.parse_args()
    run(args.run_dir.resolve(), limit=args.limit, chunk_size=args.chunk_size, stop_after_chunks=args.stop_after_chunks)
