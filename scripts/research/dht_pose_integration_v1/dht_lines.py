"""Decode frozen DHT checkpoints and expose an actual image-to-line pipeline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'deep_hough_side_v1'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(HERE.parent / 'hough_attention_transfer_v1'))
import core as C
from dht import SparseDHT
from network import DeepHoughSide, decode
import targets as T


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    C.write_json(path, value)


def source_hashes():
    return {str(p): C.sha(p) for p in [Path(__file__), OLD/'dht.py', OLD/'network.py', OLD/'targets.py',
                                      HERE.parent/'hough_attention_transfer_v1/core.py']}


def load_head(config, seed, device):
    folder = Path(config['dht']['run_dir'])
    path = folder / 'checkpoints' / f'DHT_seed{seed}_final.pth'
    if C.sha(path) != config['dht']['checkpoint_sha256'][str(seed)]:
        raise ValueError('DHT checkpoint SHA changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['steps'] != 6000 or saved['seed'] != seed or saved['arm'] != 'DHT':
        raise ValueError('Expected final6000step DHT checkpoint')
    if saved['config_sha256'] != config['dht']['config_sha256'] or C.sha(folder/'CONFIG.json') != saved['config_sha256']:
        raise ValueError('DHT checkpoint configuration mismatch')
    for name in ['dht.py', 'network.py', 'targets.py']:
        if saved['script_sha256'][name] != C.sha(OLD/name):
            raise ValueError(f'DHT source changed: {name}')
    lattice = SparseDHT(theta_bins=180, rho_step=.5, normalize=True).to(device)
    head = DeepHoughSide(lattice, seed).to(device)
    head.load_state_dict(saved['model'], strict=True)
    return head.requires_grad_(False).eval(), lattice


class ImageDHT:
    """Existing VGG+DHT image pipeline; no labels/GT mask accepted."""
    def __init__(self, config, seed=1, device='cuda'):
        self.config, self.device = config, torch.device(device)
        path = Path(config['dht']['backbone'])
        if C.sha(path) != config['dht']['backbone_sha256']:
            raise ValueError('DHT backbone SHA changed')
        self.backbone = C.load_backbone(str(path), self.device)
        self.head, self.lattice = load_head(config, seed, self.device)

    @torch.no_grad()
    def features(self, bgr):
        pad = self.config['dht']['pad_px']
        canvas = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(cv2.resize(canvas, (400, 400)), cv2.COLOR_BGR2RGB)
        values = ((rgb.astype(np.float32)/255-C.MEAN)/C.STD).transpose(2, 0, 1)
        result = self.backbone(torch.from_numpy(values[None]).to(self.device))
        # Match the stored-FP16 / read-FP32 path used by the trained head.
        return result.to(torch.float16).to(torch.float32)

    @torch.no_grad()
    def predict(self, bgr):
        theta, rho = decode(self.head(self.features(bgr)), self.lattice)
        height, width = bgr.shape[:2]
        return T.line_pixels(theta[0].cpu().numpy(), rho[0].cpu().numpy(), width, height)


def generate(run_dir):
    cfg, manifest = read(run_dir/'CONFIG.json'), read(run_dir/'manifest.json')
    identity = dict(config_sha256=C.sha(run_dir/'CONFIG.json'), manifest_sha256=C.sha(run_dir/'manifest.json'),
                    source_sha256=source_hashes(), checkpoint_sha256=cfg['dht']['checkpoint_sha256'])
    dest = run_dir/'DHT_LINES.json'
    if dest.exists():
        previous = read(dest)
        if all(previous.get(k) == v for k, v in identity.items()) and previous.get('complete'):
            print('Verified existing DHT lines', flush=True)
            return
        raise ValueError('Existing DHT line provenance mismatch')
    source = Path(cfg['dht']['source_cache'])
    if C.sha(source/'manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source feature manifest changed')
    cache = read(source/'CACHE.json')
    if cache['signature'] != dict(config_sha256=C.sha(source/'CONFIG.json'), manifest_sha256=C.sha(source/'manifest.json')):
        raise ValueError('Inherited cache provenance mismatch')
    original = read(source/'manifest.json')
    flat = [dict(r, population=p) for p, rows in original['populations'].items() for r in rows]
    records = manifest['records']
    for r in records:
        if flat[r['source_index']]['id'] != r['id'] or flat[r['source_index']]['population'] != r['population']:
            raise ValueError('Source index/identity mismatch')
    features = np.load(source/'features.npy', mmap_mode='r')
    if features.shape != (2740, 128, 50, 50) or features.dtype != np.float16:
        raise ValueError('Unexpected inherited features')
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = [{key: r[key] for key in ['id', 'population', 'group', 'width', 'height']} | {'seeds': {}} for r in records]
    predictions = {}
    for seed in cfg['dht']['seeds']:
        model, lattice = load_head(cfg, seed, device)
        with torch.no_grad():
            for start in range(0, len(records), 12):
                batch = records[start:start+12]
                values = torch.from_numpy(np.asarray(features[[r['source_index'] for r in batch]])).to(device, dtype=torch.float32)
                scores = model(values)
                theta, rho = [v.cpu().numpy() for v in decode(scores, lattice)]
                peaks = scores.masked_fill(~lattice.valid.flatten()[None, None], -1e9).softmax(-1).max(-1).values.cpu().numpy()
                for local, r in enumerate(batch):
                    lines = T.line_pixels(theta[local], rho[local], r['width'], r['height'])
                    if not np.isfinite(lines).all():
                        raise ValueError('Nonfinite DHT prediction')
                    output[start+local]['seeds'][str(seed)] = dict(lines=lines.tolist(), theta_deg=theta[local].tolist(),
                                                                  rho=rho[local].tolist(), peak_probability=peaks[local].tolist())
                    predictions[(r['id'], seed)] = T.pixel_errors(lines, r['gt_points'])
        del model
        print(f'DHT seed{seed}: {len(records)} frames', flush=True)
    # Reproduce independently saved prior evaluation; GT is used only here.
    max_angle = max_distance = 0.
    checked = 0
    with (Path(cfg['dht']['run_dir'])/'PER_ROLE.csv').open() as stream:
        for row in csv.DictReader(stream):
            key = (row['id'], int(row['seed']))
            if row['arm'] != 'DHT' or key not in predictions:
                continue
            angle, distance = predictions[key]
            role = int(row['role'])
            max_angle = max(max_angle, abs(angle[role]-float(row['angle_deg'])))
            max_distance = max(max_distance, abs(distance[role]-float(row['distance_px'])))
            checked += 1
    if checked == 0 or max_angle > 1e-3 or max_distance > 1e-3:
        raise ValueError(f'Prior DHT metric parity failed: {checked}, {max_angle}, {max_distance}')
    if identity['source_sha256'] != source_hashes() or identity['config_sha256'] != C.sha(run_dir/'CONFIG.json'):
        raise ValueError('DHT source/config changed while decoding')
    write(dest, dict(schema='dht_pose_lines_v1', complete=True, **identity, n_frames=len(records),
                     seeds=cfg['dht']['seeds'], records=output, source_cache=str(source),
                     prior_metric_parity=dict(PASS=True, checked_rows=checked, max_angle_delta_deg=max_angle, max_distance_delta_px=max_distance),
                     scope='Frozen DHT8 final6000 checkpoints; no fitting or GT used to predict or mask fusion inputs.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    generate(args.run_dir.resolve())


if __name__ == '__main__':
    main()
