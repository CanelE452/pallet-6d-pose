"""Matched RAM-image-to-final-PnP wall time for fixed Base/N3/PoseFix.

The new receipt schema records every warmup/measurement.  Large original
checkpoints and images are read through --source-root, never copied. PoseFix
is the uncapped first raw pass used by the published accuracy comparator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def run(source_root, output):
    source = Path(source_root).resolve()
    if output.exists():
        receipt = json.loads(output.read_text())
        if receipt.get('schema') != 'joint_action_matched_runtime_v1' or not receipt.get('complete'):
            raise ValueError('Incomplete runtime receipt')
        for entry in receipt['bindings']:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('Runtime binding drift: ' + entry['path'])
        raw_path = output.parent / receipt['raw']['path']
        if sha(raw_path) != receipt['raw']['sha256']:
            raise ValueError('Runtime raw output drift')
        return receipt

    import cv2
    import numpy as np
    import torch
    from scripts.research.pallet_n3_completion_v3 import square_yolo as Y

    started = time.perf_counter()
    # Historical absolute imports are deliberately loaded from the explicitly
    # selected source checkout; hashes below make that boundary reviewable.
    sys.path.insert(0, str(source / 'scripts/research/pallet_dim_conditioned_p_v1'))
    E = importlib.import_module('dcp_env')
    inf = importlib.import_module('inference')
    pose = importlib.import_module('pose')
    sys.path.insert(0, str(source / 'scripts/research/pallet_sensors_submission_v1'))
    PM = importlib.import_module('prior_model')
    CM = importlib.import_module('posefix_contract_math')
    if Path(E.ROOT).resolve() != source:
        raise ValueError('Historical imports resolved to wrong source checkout')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for locked matched runtime')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.manual_seed(1)
    np.random.seed(1)
    paths = ('Base', 'N3_seed1', 'PoseFix_seed1')
    protocol_path = source / '_docs/experiments/pallet_dim_conditioned_p_v1/RUNTIME_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    old = json.loads((source / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json').read_text())
    selected = old['selected']
    if [r['image_key'] for r in selected] != protocol['keys'] or len(selected) != 26:
        raise ValueError('Fixed 26-frame membership drift')
    if len({r['session_id'] for r in selected}) != 13:
        raise ValueError('Expected exactly 13 sessions')
    n3spec = Y.selection_contract(source, verify_checkpoints=True)['methods']['N3_DIM_SYM_seed1']
    head, n3path = inf.load_head('N3_DIM_SYM', 1)
    ppath = source / 'data/pallet/results/pallet_sensors_submission_v1/runs/PRIOR1/last.pt'
    prior_receipt_path = source / '_docs/experiments/pallet_sensors_submission_v1/PRIOR1_COMPLETE.json'
    prior_receipt = json.loads(prior_receipt_path.read_text())
    if sha(ppath) != prior_receipt['checkpoint_sha256']:
        raise ValueError('PoseFix checkpoint binding drift')
    checkpoint = torch.load(ppath, map_location='cpu', weights_only=False)
    if not checkpoint['complete'] or checkpoint['step'] != 6000:
        raise ValueError('PoseFix is not the final fixed 6000-update checkpoint')
    prior = PM.PoseFixPallet9().cuda().eval().requires_grad_(False)
    prior.load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    normalization_path = source / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json'
    norm = json.loads(normalization_path.read_text())
    feature_module = E.old('features')
    extractor = feature_module.FrozenYoloFeatures(E.R0)
    images = []
    image_bindings = []
    for row in selected:
        path = source / row['image_key']
        if sha(path) != row['image']['sha256']:
            raise ValueError('Image bytes changed: ' + row['image_key'])
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot decode: ' + row['image_key'])
        images.append(image)
        image_bindings.append(binding(path))

    @torch.no_grad()
    def operation(name, index):
        image, row = images[index], selected[index]
        captured = extractor.predict(image)
        selected_index = captured['selected_index']
        points = None if selected_index is None else np.asarray(captured['candidates'][selected_index]['keypoints_xy']).copy()
        if name == 'N3_seed1':
            dimensions, order = inf.registry_input(row['object_type'])
            if dimensions.tolist() != row['dimensions_wdh_m']:
                raise ValueError('Deployment geometry registry drift')
            pred, _ = inf.predict_captured(head, 'N3_DIM_SYM', captured, dimensions, order,
                                           n3spec['temperature'], n3spec['rule'], image.shape[:2], norm)
            points = None if selected_index is None else np.asarray(pred['candidates'][selected_index]['keypoints_xy'])
        elif name == 'PoseFix_seed1' and points is not None:
            box = np.asarray(captured['candidates'][selected_index]['box_xyxy'], float)
            valid = np.isfinite(points).all(-1) & ~(points == -1).all(-1)
            if np.isfinite(box).all() and (box[2:] > box[:2]).all():
                matrix = CM.axis_aligned_crop_matrix(box)
                rgb = cv2.warpAffine(image, matrix[:2], (288, 384), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT)[:, :, ::-1].astype(np.float32)
                rgb -= np.array([123.68, 116.78, 103.94], np.float32)
                cp = CM.transform_points(np.where(valid[:, None], points, 0), matrix).astype(np.float32)
                tensors = [torch.as_tensor(v, device='cuda')[None] for v in (rgb.transpose(2, 0, 1), cp, valid)]
                with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
                    q = PM.expectation(prior(*tensors))[0].cpu().numpy()
                raw = CM.transform_points(q, np.linalg.inv(matrix))
                raw[~valid] = points[~valid]
                raw[8] = points[8]
                points = raw
        result = pose.infer(points, np.asarray(row['camera_intrinsics']),
                            np.asarray(row['dimensions_wdh_m'])[[0, 2, 1]], source=False)
        return {'selected_index': selected_index,
                'points': None if points is None else np.where(np.isfinite(points), points, np.nan).tolist(),
                'pose_available': bool(result['available']),
                'selected_hypothesis': result.get('selected_hypothesis')}

    def one(name, index, phase, repeat, position):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter_ns()
        out = operation(name, index)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        # JSON missing values are null, never a finite pose/error replacement.
        if out['points'] is not None:
            out['points'] = [[float(v) if np.isfinite(v) else None for v in p] for p in out['points']]
        return {'path': name, 'phase': phase, 'repeat': repeat, 'position': position,
                'frame_id': selected[index]['frame_id'], 'session': selected[index]['session_id'],
                'wall_ms': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'output': out}

    try:
        warmup = []
        for rep in range(20):
            for position in range(3):
                warmup.append(one(paths[(rep + position) % 3], rep % 26, 'warmup', rep, position))
        measurements = []
        for rep in range(5):
            for index in range(26):
                order = list(paths)
                offset = (rep + index) % 3
                order = order[offset:] + order[:offset]
                if rep % 2:
                    order.reverse()
                for position, name in enumerate(order):
                    measurements.append(one(name, index, 'measured', rep, position))
            print(f'Matched runtime repeat {rep + 1}/5', flush=True)
    finally:
        extractor.close()
    summary = {}
    for name in paths:
        rows = [r for r in measurements if r['path'] == name]
        values = [r['wall_ms'] for r in rows]
        summary[name] = {'count': len(rows), 'median_ms': float(np.median(values)),
                         'p90_ms': float(np.quantile(values, .9)),
                         'pose_success': sum(r['output']['pose_available'] for r in rows),
                         'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows)}
    code_paths = [Path(__file__), Path(inf.__file__), Path(E.__file__), Path(pose.__file__),
                  Path(feature_module.__file__), Path(PM.__file__), Path(CM.__file__)]
    bindings = [binding(p) for p in code_paths + [E.R0, n3path, ppath, protocol_path, normalization_path, prior_receipt_path]]
    bindings.extend(image_bindings)
    import ultralytics
    raw = {'schema': 'joint_action_matched_runtime_rows_v1', 'warmup': warmup, 'measurements': measurements}
    raw_path = output.parent / 'results/runtime_rows.json'
    write(raw_path, raw)
    receipt = {'schema': 'joint_action_matched_runtime_v1', 'complete': True,
               'source_root': str(source), 'summary': summary, 'bindings': bindings,
               'raw': {'path': str(raw_path.relative_to(output.parent)), 'sha256': sha(raw_path), 'bytes': raw_path.stat().st_size},
               'frames': 26, 'sessions': 13, 'warmup_per_arm': 20, 'repeats': 5,
               'measured_calls': 390, 'warmup_calls': 60, 'new_fit_count': 0, 'optimizer_updates': 0,
               'precision': 'FP32 models, frozen feature handoff FP16; matmul TF32 off; original YOLO/N3 cuDNN TF32 on and original PoseFix head cuDNN TF32 off; AMP off',
               'boundary': 'native uint8 BGR in RAM -> reflect/letterbox -> shared frozen detector and feature capture -> optional head crop/sampling/decoder -> prediction-only W/D wrapper and final PnP; image decode and model loading excluded',
               'posefix': 'purpose-adapted PRIOR1 last6000; uncapped first raw pass, same published accuracy method',
               'environment': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                               'opencv': cv2.__version__, 'ultralytics': ultralytics.__version__,
                               'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(),
                               'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()},
               'parameters': {'N3': sum(p.numel() for p in head.parameters()),
                              'PoseFix': sum(p.numel() for p in prior.parameters())},
               'memory_scope': 'all fixed models resident; process-scoped peaks, not isolated per-method memory',
               'elapsed_seconds': time.perf_counter() - started}
    write(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DOC / 'RUNTIME_MATCHED.json')
    args = parser.parse_args()
    result = run(args.source_root, args.output)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
