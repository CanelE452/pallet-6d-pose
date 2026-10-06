"""Matched deployable A cost: generate bank, sample RGB, select J, final F.

Base and the unchanged N3 seed1 are remeasured in this same panel. The
offline all-candidate oracle search is deliberately absent from inference.
"""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from .runtime import DOC, binding, sha, write


def run(source_root, cache_dir, output):
    source, cache = Path(source_root).resolve(), Path(cache_dir).resolve()
    if output.exists():
        old = json.loads(output.read_text())
        if not old.get('complete'):
            raise ValueError('Incomplete A runtime receipt')
        for item in old['bindings']:
            if sha(item['path']) != item['sha256']:
                raise ValueError('A runtime input changed: ' + item['path'])
        if sha(output.parent / old['raw']['path']) != old['raw']['sha256']:
            raise ValueError('A runtime rows changed')
        return old
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(source / 'scripts/research/pallet_dim_conditioned_p_v1'))
    E = importlib.import_module('dcp_env')
    inf = importlib.import_module('inference')
    if Path(E.ROOT).resolve() != source:
        raise ValueError('Historical imports resolved to wrong source checkout')
    from .a_common import POSE, sample, finite
    from .scorer import JointActionScorer
    from .a_inference import predict_captured_joint
    from scripts.research.pallet_n3_completion_v3 import square_yolo as Y
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    import os
    foreign = [r for r in processes.splitlines() if r.split(',')[0].strip() != str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    if foreign:
        raise RuntimeError('A runtime requires an idle measurement GPU: ' + repr(foreign))
    started = time.perf_counter()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    spec = Y.selection_contract(source, verify_checkpoints=True)['methods']['N3_DIM_SYM_seed1']
    n3, n3path = inf.load_head('N3_DIM_SYM', 1)
    paths = {'Frozen_J_seed1': n3path, 'GEO_seed1': cache / 'fits/GEO_seed1/last.pt',
             'PERM_seed1': cache / 'fits/PERM_seed1/last.pt'}
    config = json.loads((DOC / 'A_protocol.json').read_text())['config']
    heads = {}
    for name, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if not checkpoint['complete'] or checkpoint['step'] != 6000:
            raise ValueError('Wrong final A/N3 checkpoint: ' + name)
        head = JointActionScorer(5, **config).cuda().eval().requires_grad_(False)
        head.load_state_dict(checkpoint['model_state_dict'], strict=True)
        heads[name] = head
    norm_path = source / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json'
    norm = json.loads(norm_path.read_text())
    selected_path = source / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
    selected = json.loads(selected_path.read_text())['selected']
    protocol_path = source / '_docs/experiments/pallet_dim_conditioned_p_v1/RUNTIME_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    assert [r['image_key'] for r in selected] == protocol['keys'] and len(selected) == 26
    assert len({r['session_id'] for r in selected}) == 13
    images = []
    import inspect
    input_paths = [norm_path, selected_path, protocol_path, DOC / 'A_protocol.json', E.R0, Path(inf.__file__), Path(E.__file__),
                   Path(POSE.__file__), Path(inspect.getfile(POSE.solve)), Path(inspect.getfile(POSE.select_pnp_hypotheses)),
                   Path(inspect.getfile(JointActionScorer.__bases__[0])), Path(__file__),
                   source / '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json'] + list(paths.values())
    selector_module = inspect.getmodule(POSE.select_pnp_hypotheses)
    input_paths.append(Path(selector_module.geometry.__file__))
    input_paths += [Path(inspect.getfile(sample)), Path(inspect.getfile(finite))]
    for row in selected:
        path = source / row['image_key']
        assert sha(path) == row['image']['sha256']
        image = cv2.imread(str(path))
        assert image is not None
        images.append(image)
        input_paths.append(path)
    for name in ('a_inference.py', 'a_common.py', 'a_data.py', 'scorer.py', 'geometry.py'):
        input_paths.append(Path(__file__).parent / name)
    feature_module = E.old('features')
    input_paths.append(Path(feature_module.__file__))
    extractor = feature_module.FrozenYoloFeatures(E.R0)
    names = ('Base', 'N3_seed1', *heads)

    @torch.no_grad()
    def operation(name, i):
        image, row = images[i], selected[i]
        captured = extractor.predict(image)
        dimensions, order = inf.registry_input(row['object_type'])
        assert dimensions.tolist() == row['dimensions_wdh_m']
        K = np.asarray(row['camera_intrinsics'])
        if name in heads:
            pred, final_pose, diagnostic = predict_captured_joint(
                heads[name], captured, dimensions, order, K, image.shape[:2], norm,
                readout='J', arm='PERM' if name == 'PERM_seed1' else 'GEO', frame_id=row['frame_id'])
        else:
            pred = captured
            if name == 'N3_seed1':
                pred, _ = inf.predict_captured(n3, 'N3_DIM_SYM', captured, dimensions, order,
                                               spec['temperature'], spec['rule'], image.shape[:2], norm)
            ix = pred['selected_index']
            q = None if ix is None else np.asarray(pred['candidates'][ix]['keypoints_xy'])
            final_pose = POSE.infer(q, K, dimensions[[0, 2, 1]], False)
            diagnostic = {'actions': 0}
        ix = pred['selected_index']
        q = None if ix is None else np.asarray(pred['candidates'][ix]['keypoints_xy'])
        return final_pose, diagnostic, q

    def one(name, i, phase, rep, position):
        torch.cuda.synchronize()
        tick = time.perf_counter_ns()
        final, diagnostic, q = operation(name, i)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - tick) / 1e6
        points = None if q is None else [[float(v) if np.isfinite(v) else None for v in xy] for xy in q]
        return {'path': name, 'phase': phase, 'repeat': rep, 'position': position,
                'frame_id': selected[i]['frame_id'], 'session': selected[i]['session_id'], 'wall_ms': elapsed,
                'pose_available': bool(final['available']), 'final_W_D': final.get('selected_hypothesis'),
                'diagnostic': diagnostic, 'points': points}

    try:
        warmup = []
        for rep in range(20):
            for position in range(len(names)):
                warmup.append(one(names[(rep + position) % len(names)], rep % 26, 'warmup', rep, position))
        measurements = []
        for rep in range(5):
            for i in range(26):
                offset = (rep + i) % len(names)
                order = list(names[offset:] + names[:offset])
                if rep % 2:
                    order.reverse()
                for position, name in enumerate(order):
                    measurements.append(one(name, i, 'measured', rep, position))
            print(f'A matched runtime repeat {rep + 1}/5', flush=True)
    finally:
        extractor.close()
    summary = {}
    for name in names:
        rows = [r for r in measurements if r['path'] == name]
        values = [r['wall_ms'] for r in rows]
        summary[name] = {'count': len(rows), 'median_ms': float(np.median(values)),
                         'p90_ms': float(np.quantile(values, .9)), 'pose_success': sum(r['pose_available'] for r in rows),
                         'min_actions': min(r['diagnostic']['actions'] for r in rows),
                         'max_actions': max(r['diagnostic']['actions'] for r in rows)}
    raw_path = output.parent / 'results/runtime_A_rows.json'
    write(raw_path, {'schema': 'joint_action_A_runtime_rows_v1', 'warmup': warmup, 'measurements': measurements})
    result = {'schema': 'joint_action_A_runtime_v1', 'complete': True, 'summary': summary,
              'checkpoint_bindings': {name: binding(path) for name, path in paths.items()},
              'bindings': [binding(p) for p in dict.fromkeys(input_paths)],
              'raw': {'path': str(raw_path.relative_to(output.parent)), 'sha256': sha(raw_path), 'bytes': raw_path.stat().st_size},
              'frames': 26, 'sessions': 13, 'warmup_per_arm': 20, 'repeat': 5,
              'warmup_calls': len(warmup), 'measured_calls': len(measurements), 'optimizer_updates': 0,
              'environment': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__,
                              'opencv': cv2.__version__, 'gpu': torch.cuda.get_device_name()},
              'precision': 'same original YOLO/N3 cuDNN TF32 on; matmul TF32 off; FP32 models/FP16 features; no AMP',
              'boundary': 'native BGR RAM image -> preprocessing/detector/features -> prediction-only initial hypotheses and normalized projected motion bank -> fresh feature patch scores and hard J -> actual final W/D plus PnP F(q); no offline oracle candidate-F search',
              'elapsed_seconds': time.perf_counter() - started}
    write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DOC / 'RUNTIME_A.json')
    args = parser.parse_args()
    print(json.dumps(run(args.source_root, args.cache_dir, args.output)['summary'], indent=2))


if __name__ == '__main__':
    main()
