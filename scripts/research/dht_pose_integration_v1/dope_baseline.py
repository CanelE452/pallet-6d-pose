"""Frozen six-stage DOPE baseline with the existing local-peak decoder.

Inference only reads image pixels. It preserves semantic channel identities and
missing points; no GT matching, PnP, point imputation, or model fallback occurs.
The decoder selects a local maximum independently in each belief channel and
does not group multiple objects using affinity maps.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DECODER = ROOT / 'scripts/data_prep/filters/filter_pr_camfacing.py'
SOURCE_MODEL = ROOT / 'Deep_Object_Pose/common/models.py'
SOURCE_PREPROCESS = ROOT / 'scripts/stage0/eval_harness/eval_pvnet_heads.py'
SOURCE_CANVAS = (ROOT / 'data/pallet/eval_results/stage16_truncation_addon/'
                 'capturecad_b2_eval/eval_capturecad_b2.py')


def _canonical_functions():
    for sub in ('Deep_Object_Pose/common', 'scripts/data_prep/filters',
                'scripts/stage0/eval_harness', 'scripts/data_prep',
                'challenge/scripts/annotate', 'challenge/scripts/infer',
                'challenge/scripts/live'):
        sys.path.insert(0, str(ROOT / sub))
    from models import DopeNetwork
    from filter_pr_camfacing import extract_keypoints_from_belief
    from eval_pvnet_heads import preprocess
    spec = importlib.util.spec_from_file_location('integration_canonical_canvas',
                                                  SOURCE_CANVAS)
    canvas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canvas)
    return DopeNetwork, extract_keypoints_from_belief, preprocess, canvas


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _nullable(values):
    return [float(x) if np.isfinite(x) else None for x in values]


class FrozenDope:
    def __init__(self, weights, recipe, device):
        self.recipe = recipe
        self.device = torch.device(device)
        network, self.decoder, self.preprocess, self.canvas = _canonical_functions()
        # Load on CPU first; strict loading prevents silently dropped heads or
        # partially initialized parameters. This exact checkpoint has9 belief
        # and16 affinity channels at each of six stages.
        state = torch.load(str(weights), map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model = network()
        self.model.load_state_dict(state, strict=True)
        self.model.requires_grad_(False).to(self.device).eval()

    def predict(self, image):
        """Return semantic9 corners/centroid in the unpadded image coordinates."""
        pad = int(self.recipe['pad_px'])
        _synchronize(self.device)
        begin = time.perf_counter()
        height, width = image.shape[:2]
        canvas = self.canvas.pad_frame(image, pad)
        tensor, nw, nh, scale = self.preprocess(canvas)
        tensor = tensor.to(self.device)
        _synchronize(self.device)
        forward_begin = time.perf_counter()
        with torch.inference_mode():
            belief_stages, _ = self.model(tensor)
        _synchronize(self.device)
        forward_ms = (time.perf_counter() - forward_begin) * 1000
        belief = belief_stages[-1][0].detach().cpu().numpy()
        if belief.shape[0] != 9 or not np.isfinite(belief).all():
            raise RuntimeError(f'Invalid DOPE belief output: {belief.shape}')
        bh, bw = belief.shape[1:]
        peaks = self.decoder(belief, float(self.recipe['belief_threshold']))
        points = np.full((9, 2), np.nan, dtype=np.float64)
        valid = np.zeros(9, dtype=bool)
        confidence = np.asarray([p[2] for p in peaks], dtype=np.float64)
        for i, (x, y, _) in enumerate(peaks):
            if x >= 0 and np.isfinite([x, y]).all():
                points[i] = self.canvas.belief_to_orig_pad(
                    x, y, bw, bh, nw, nh, scale, pad, width, height)
                valid[i] = np.isfinite(points[i]).all()
        corners = points[:8][valid[:8]]
        box = None
        if len(corners) >= 3:
            box = [*corners.min(0), *corners.max(0)]
        # Keep partial predictions even when a box cannot be derived. The
        # evaluation denominator, including missing corners, belongs downstream.
        output = dict(
            kps=[_nullable(point) for point in points],
            kp_conf=_nullable(confidence), kp_valid=valid.tolist(),
            detected=bool(valid[:8].any()), n_instances=None,
            selected_instance=None,
            box_xyxy=_nullable(box) if box is not None else None,
            box_conf=None, box_source='bounding_box_of_detected_corners',
            candidate_rule='independent_highest_local_peak_per_semantic_channel',
            affinity_instance_grouping=False,
            n_detected_corners=int(valid[:8].sum()),
            input_shape_chw=list(tensor.shape[1:]), belief_shape=list(belief.shape),
            forward_ms=forward_ms,
        )
        _synchronize(self.device)
        output['inference_ms'] = (time.perf_counter() - begin) * 1000
        return output


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Baseline(FrozenDope):
    def __init__(self, config, device='cuda'):
        recipe = config['models']['dope']
        if recipe.get('preprocess') != 'stage0_reflect_resizeback_shortside400':
            raise ValueError('CONFIG must freeze the established stage0 DOPE recipe')
        if recipe.get('shortest_side') != 400 or recipe['pad_px'] != 100:
            raise ValueError('The canonical stage0 recipe uses short-side400 / pad100')
        if float(recipe['belief_threshold']) != 0.3:
            raise ValueError('The predeclared canonical belief threshold is0.3')
        if file_sha(recipe['weights']) != recipe['weights_sha256']:
            raise ValueError('DOPE checkpoint differs from frozen CONFIG')
        super().__init__(recipe['weights'], recipe, device)


def _write_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config_path, manifest_path = run_dir / 'CONFIG.json', run_dir / 'manifest.json'
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    config_hash, manifest_hash = file_sha(config_path), file_sha(manifest_path)
    records = manifest['records']
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Manifest is empty or has duplicate IDs')
    if config['runtime']['batch'] != 1 or config['runtime']['warmup'] != 5:
        raise ValueError('Expected the predeclared batch1 / warmup5 timing recipe')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    baseline = Baseline(config, args.device)
    warmup_record = next(r for r in records if r['population'] == 'synth_val')
    warmup_image = cv2.imread(warmup_record['image'])
    if warmup_image is None:
        raise ValueError('Warmup image could not be read')
    for _ in range(config['runtime']['warmup']):
        baseline.predict(warmup_image)
    outputs = []
    started = time.perf_counter()
    for index, record in enumerate(records):
        if file_sha(record['image']) != record['image_sha256']:
            raise ValueError(f"Image hash mismatch: {record['id']}")
        image = cv2.imread(record['image'])
        if image is None or image.shape[:2] != (record['height'], record['width']):
            raise ValueError(f"Image shape mismatch: {record['id']}")
        prediction = baseline.predict(image)
        prediction.update({k: record[k] for k in
                           ('id', 'population', 'group', 'width', 'height')})
        outputs.append(prediction)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"DOPE {index + 1}/{len(records)} "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    if file_sha(config_path) != config_hash or file_sha(manifest_path) != manifest_hash:
        raise ValueError('Experiment inputs changed during inference')
    source_paths = [Path(__file__), SOURCE_MODEL, SOURCE_DECODER,
                    SOURCE_PREPROCESS, SOURCE_CANVAS]
    times = [record['inference_ms'] for record in outputs]
    device = torch.device(args.device)
    payload = dict(
        schema='dht_pose_baseline_v1', model='DOPE', complete=True,
        config_sha256=config_hash, manifest_sha256=manifest_hash,
        weights_sha256=config['models']['dope']['weights_sha256'],
        source_sha256={str(p.relative_to(ROOT)): file_sha(p) for p in source_paths},
        recipe=config['models']['dope'], n_records=len(outputs),
        runtime=dict(device=str(device), torch=torch.__version__,
                     cuda=torch.version.cuda, opencv=cv2.__version__,
                     device_name=torch.cuda.get_device_name(device)
                     if device.type == 'cuda' else 'CPU', batch=1, warmup=5,
                     warmup_id=warmup_record['id'], cpu_threads=4,
                     opencv_threads=1, cudnn_benchmark=False),
        timing_definition=('Synchronized wall time per predict: reflected padding, '
                           'resize/normalization, host-to-device transfer, all six '
                           'DOPE belief/affinity stages, GPU-to-CPU transfer, '
                           'canonical peak decoding and coordinate inversion. '
                           'Excludes image decoding, hashing, model loading and '
                           'five warmups. Batch1. This is a component measurement.'),
        timing_ms=dict(n=len(times), mean=float(np.mean(times)),
                       median=float(np.median(times)), p90=float(np.percentile(times, 90))),
        limitation=('One channelwise candidate set; affinity fields are computed '
                    'but not used for instance grouping. No GT-based selection. '
                    'Declared training imagesize448 does not establish training '
                    'parity: the repository loader uses a400pixel crop.'),
        records=outputs,
    )
    _write_json(run_dir / 'BASELINE_DOPE.json', payload)
    print(f"Saved {run_dir / 'BASELINE_DOPE.json'}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
