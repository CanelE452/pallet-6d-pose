"""Frozen synthetic YOLO26 pose baseline for paired DHT line fusion.

Standard Ultralytics predict owns letterboxing and its inverse. This adapter
only adds the established reflected image border, then subtracts that border
from returned original-canvas coordinates. Instance selection never sees GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nullable_vector(value):
    return [float(v) if np.isfinite(v) else None for v in value]


class Baseline:
    def __init__(self, config, device='cuda'):
        settings = config['models']['yolo']
        weights = Path(settings['weights'])
        if not weights.is_file() or file_sha(weights) != settings['weights_sha256']:
            raise ValueError('YOLO weights differ from the frozen synthetic R0 checkpoint')
        self.recipe = dict(pad_px=settings['pad_px'], imgsz=settings['input_size'],
                           box_confidence=settings['box_conf'], keypoint_confidence=settings['kp_conf'],
                           iou=.7, max_det=300, rect=True, half=False,
                           border='BORDER_REFLECT_101', batch=1,
                           normalization='Ultralytics BGR to RGB, float32 /255',
                           coordinates='Predictor inverse letterbox into padded original canvas, then subtract pad',
                           instance='highest predicted box confidence, no GT input')
        self.device = device
        self.model = YOLO(str(weights), task='pose')
        self.model.model.requires_grad_(False).eval()

    def predict(self, image):
        """Return raw corners and confidence validity in original-image pixels."""
        pad = self.recipe['pad_px']
        synchronize()
        begin = time.perf_counter()
        canvas = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        result = self.model.predict(
            canvas, imgsz=self.recipe['imgsz'], conf=self.recipe['box_confidence'],
            iou=self.recipe['iou'], max_det=self.recipe['max_det'],
            rect=self.recipe['rect'], half=False, device=self.device,
            verbose=False, save=False, save_txt=False, save_crop=False,
            augment=False, stream=False)[0]
        if tuple(result.orig_shape) != tuple(canvas.shape[:2]):
            raise RuntimeError('Ultralytics output does not use the padded original canvas')
        count = 0 if result.boxes is None else len(result.boxes)
        output = dict(kps=[[None, None] for _ in range(9)], kp_conf=[None]*9,
                      kp_valid=[False]*9, detected=False, n_instances=count,
                      box_xyxy=None, box_conf=None, selected_instance=None)
        if count:
            scores = result.boxes.conf.detach().cpu().numpy()
            selected = int(np.argmax(scores))
            if result.keypoints is None:
                raise RuntimeError('Detected pose instance has no keypoints object')
            keypoints = result.keypoints.data[selected].detach().cpu().numpy().copy()
            if keypoints.shape != (9, 3):
                raise RuntimeError(f'Expected9 camera-facing pose keypoints, got {keypoints.shape}')
            keypoints[:, :2] -= pad
            xy, confidence = keypoints[:, :2], keypoints[:, 2]
            valid = np.isfinite(keypoints).all(1) & (confidence >= self.recipe['keypoint_confidence'])
            box = result.boxes.xyxy[selected].detach().cpu().numpy() - pad
            output.update(kps=[nullable_vector(point) for point in xy],
                          kp_conf=nullable_vector(confidence), kp_valid=valid.tolist(),
                          detected=True, box_xyxy=nullable_vector(box),
                          box_conf=float(scores[selected]), selected_instance=selected)
        synchronize()
        output['inference_ms'] = (time.perf_counter()-begin)*1000
        output['ultralytics_speed_ms'] = {k: float(v) for k, v in result.speed.items()}
        return output


def distribution(values):
    return {'n': len(values), 'mean_ms': float(np.mean(values)),
            'median_ms': float(np.median(values)), 'p90_ms': float(np.percentile(values, 90))}


def source_hashes():
    import inspect
    from ultralytics.engine.predictor import BasePredictor
    from ultralytics.engine.results import Keypoints
    from ultralytics.models.yolo.pose.predict import PosePredictor
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ops
    paths = [Path(__file__).resolve(),
             *[Path(inspect.getfile(obj)).resolve() for obj in
               (BasePredictor, Keypoints, PosePredictor, LetterBox, ops.scale_coords)]]
    return {str(path): file_sha(path) for path in paths}


def load_image(record):
    if file_sha(record['image']) != record['image_sha256']:
        raise ValueError(f"Image bytes changed since source freeze: {record['id']}")
    image = cv2.imread(record['image'])
    if image is None or image.shape[:2] != (record['height'], record['width']):
        raise ValueError(f"Image geometry mismatch: {record['id']}")
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not (root / 'PURPOSE.md').is_file():
        raise ValueError('Prepared experiment PURPOSE.md is required')
    config_path, manifest_path = root / 'CONFIG.json', root / 'manifest.json'
    config, manifest = read_json(config_path), read_json(manifest_path)
    if config.get('schema') != 'dht_pose_integration_v1':
        raise ValueError('Unexpected experiment config schema')
    records = manifest['records']
    expected_counts = {'synth_val': 256, 'synth_test': 256, 'cross_v4': 128, 'real_dev': 52}
    if config['evaluation']['populations'] != list(expected_counts):
        raise ValueError('Only the prespecified synthetic evaluation and canonical DEV52 populations are allowed')
    if len(records) != 692 or len({r['id'] for r in records}) != 692:
        raise ValueError('Expected692 unique frames')
    seen = set()
    for population, count in expected_counts.items():
        indices = manifest['populations'][population]
        if len(indices) != count or len(set(indices)) != count:
            raise ValueError('Population count or duplicate index mismatch')
        for index in indices:
            if index in seen or records[index]['index'] != index or records[index]['population'] != population:
                raise ValueError('Manifest index/population overlap mismatch')
            seen.add(index)
    if seen != set(range(len(records))):
        raise ValueError('Evaluation manifest membership is incomplete')
    if file_sha(manifest['source_manifest']) != manifest['source_manifest_sha256']:
        raise ValueError('Original DHT source manifest changed')
    weights = Path(config['models']['yolo']['weights'])
    identity = dict(config_sha256=file_sha(config_path), manifest_sha256=file_sha(manifest_path),
                    weights_sha256=file_sha(weights), source_sha256=source_hashes())
    if identity['weights_sha256'] != config['models']['yolo']['weights_sha256']:
        raise ValueError('Unexpected YOLO model SHA')
    output_path = root / 'BASELINE_YOLO.json'
    if output_path.exists():
        previous = read_json(output_path)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError('Existing output provenance differs; use a new experiment directory')
        if previous.get('complete') and len(previous.get('records', [])) == len(records):
            print(f'Verified existing completed baseline: {output_path}', flush=True)
            return
        raise ValueError('Incomplete existing output must be investigated before rerunning')
    if config['runtime']['device'] != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This frozen runtime experiment requires CUDA')
    if config['runtime']['batch'] != 1 or config['runtime']['warmup'] != 5:
        raise ValueError('Expected frozen batch1/warmup5 timing protocol')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Baseline(config, device='cuda')
    first = records[0]
    if first['population'] != 'synth_val' or first['source_kind'] != 'synthetic':
        raise ValueError('Warmup must use the first synthetic validation image')
    warm_image = load_image(first)
    for _ in range(config['runtime']['warmup']):
        model.predict(warm_image)
    output_records = []
    for index, record in enumerate(records):
        image = load_image(record)
        prediction = model.predict(image)
        output_records.append({key: record[key] for key in ('id', 'population', 'group', 'width', 'height')} | prediction)
        if (index+1) % 100 == 0 or index+1 == len(records):
            print(f'YOLO baseline {index+1}/{len(records)}', flush=True)
    if (file_sha(config_path) != identity['config_sha256'] or file_sha(manifest_path) != identity['manifest_sha256']
            or source_hashes() != identity['source_sha256'] or file_sha(weights) != identity['weights_sha256']):
        raise RuntimeError('Frozen experiment inputs or source code changed during inference')
    output = dict(schema='dht_pose_baseline_v1', model='YOLO', complete=True,
                  **identity, weights=str(weights), recipe=model.recipe,
                  runtime=dict(device='cuda', python=sys.executable, python_version=sys.version.split()[0],
                               torch=torch.__version__, ultralytics=ultralytics.__version__,
                               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                               num_threads=torch.get_num_threads(), cudnn_benchmark=False,
                               matmul_tf32=False, batch=1, warmup=5, warmup_id=first['id']),
                  timing_definition='Synchronized GPU wall time per batch1 predict, including reflected padding, Ultralytics preprocessing/inference/postprocessing, coordinate inversion and CPU copy; excludes disk I/O, model loading, warmup and JSON writing.',
                  timing=distribution([r['inference_ms'] for r in output_records]),
                  timing_by_population={p: distribution([r['inference_ms'] for r in output_records if r['population'] == p]) for p in expected_counts},
                  n_frames=len(records), n_detected=sum(r['detected'] for r in output_records),
                  records=output_records)
    write_json(output_path, output)
    print(json.dumps({'complete': True, 'frames': len(records), 'detected': output['n_detected'],
                      'timing': output['timing'], 'output': str(output_path)}), flush=True)


if __name__ == '__main__':
    main()
