"""Freeze the factorial design and cache identical-backbone padding inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent / 'deep_hough_side_v1'
sys.path.insert(0, str(HERE.parent / 'hough_attention_transfer_v1'))
import core as C


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def config_for(run_dir, stage='main'):
    inherited = read(ROOT / 'data/pallet/results/deep_hough_side_v1/CONFIG.json')
    return dict(
        schema='dht_padding_view_v1', stage=stage,
        conditions={
            'reflect100': dict(mode='reflect101', pad_px=100, constant_bgr=None),
            'black100': dict(mode='constant', pad_px=100, constant_bgr=[0, 0, 0]),
            'mean100': dict(mode='constant', pad_px=100, constant_bgr=[103.53, 116.28, 123.675]),
        },
        preprocessing=dict(input_size=400, feature_shape=[128, 50, 50], resize='INTER_LINEAR',
                           input_dtype='float32', mean_rgb=[.485, .456, .406], std_rgb=[.229, .224, .225]),
        model=dict(type='DHT8', backbone=inherited['backbone'], backbone_sha256=inherited['backbone_sha256'],
                   backbone_frozen=True, theta_bins=180, rho_step=.5, channels=16,
                   feature_coordinate='(original+pad+.5)*50/padded_size-.5',
                   target_angle_sigma_deg=1., target_rho_sigma_cell=.5,
                   target_semantics='Eight amodal structural side supporting lines; not all twelve cuboid edges or physical edge visibility',
                   source_sha256={name: C.sha(OLD / name) for name in ['dht.py', 'network.py', 'targets.py']}),
        training=dict(steps=6000 if stage == 'main' else 20, batch=12 if stage == 'main' else 4,
                      lr=.001, weight_decay=.0001, seeds=[1, 2, 3] if stage == 'main' else [1],
                      regimes=['base', 'low_balanced'], checkpoint_selection='fixed final step',
                      train_test_padding='same mode; no inference-only substitution in primary comparison'),
        evaluation=dict(populations=['synth_val', 'synth_test', 'cross_v4', 'low_val', 'low_test', 'real_dev']
                        if stage == 'main' else ['synth_val', 'low_val'],
                        success_angle_deg=5, success_distance_px=8,
                        distance='Mean perpendicular distance from the two GT endpoints to the predicted infinite line, original pixels',
                        primary='Per-seed median and P90 error plus fraction satisfying both 5deg and 8px; summarize across fixed seeds',
                        cohort_fields=['elevation_bin', 'side_bin', 'top_bin', 'size_bin'],
                        sparse_cell_frames=20,
                        missing_cell='NO_DATA: no accuracy claim; fewer than20 frames is descriptive only',
                        real_scope='Existing canonical manual DEV52 only; no final-test data or real training'),
        design=dict(primary='3 padding modes x 2 data mixtures x 3 seeds; 18 fresh DHT8 heads',
                    paired_padding='Same train indices, initialization seed, batch sequence, features backbone, geometry and budget',
                    data_mixture='base2048 vs fixed1024base+1024eligiblelow; count-matched, renderer/background composition can differ',
                    comparison='Padding effects within each mixture, mixture effects within each padding, then descriptive interactions',
                    interpretation='The mixture intervention is not a pure causal camera-view ablation',
                    preserved='8roles,100px pad width,400input,50feature grid; width/ROI/mask/12role changes reserved for separate experiments',
                    future_controls=['Padding width / no-padding with geometry-consistent targets',
                                     'Valid-region DHT aggregation', 'Twelve roles / front width edges',
                                     'Detected ROI / higher-resolution features'],
                    pretraining_limit='Inherited synthetic backbone source manifest unavailable; end-to-end pretraining/holdout overlap unverified'),
    )


def metadata(run_dir, source_manifest):
    run_dir.mkdir(parents=True, exist_ok=True)
    purpose = run_dir / 'PURPOSE.md'
    if not purpose.exists():
        purpose.write_text('# DHT padding and viewpoint factorial experiment\n\n'
                           'Configure and compare matched train/test padding with a fixed DHT8 backbone/readout, '
                           'and count-matched base versus low-view data mixtures. Main runs are not started by metadata preparation. '
                           'Only canonical real DEV52 is eligible for evaluation. Each configured run uses its fixed final step.\n')
    from cohorts import build_manifest
    build_manifest(run_dir, source_manifest, seed=17)
    cfg = config_for(run_dir)
    path = run_dir / 'CONFIG.json'
    if path.exists():
        if read(path) != cfg:
            raise RuntimeError('Existing CONFIG differs: use a new output directory for a changed experiment')
    else:
        write(path, cfg)
    log(f'Metadata fixed: {len(read(run_dir / "manifest.json")["records"])} frames, 18 planned main runs')


def signature(run_dir, cfg, condition):
    return dict(manifest_sha256=C.sha(run_dir / 'manifest.json'),
                backbone_sha256=cfg['model']['backbone_sha256'],
                preprocessing_sha256=canonical_sha({'condition': cfg['conditions'][condition], 'preprocessing': cfg['preprocessing']}))


def prepare_image(record, condition, pp):
    bgr = cv2.imread(record['image'])
    if bgr is None or tuple(bgr.shape[:2]) != (record['height'], record['width']):
        raise ValueError(f'Image/annotation dimensions disagree: {record["id"]}')
    bgr = bgr.astype(np.float32)
    pad = condition['pad_px']
    mode = cv2.BORDER_REFLECT_101 if condition['mode'] == 'reflect101' else cv2.BORDER_CONSTANT
    if condition['mode'] not in ('reflect101', 'constant'):
        raise ValueError(condition['mode'])
    canvas = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, mode, value=condition['constant_bgr'] or 0)
    resized = cv2.resize(canvas, (pp['input_size'], pp['input_size']), interpolation=cv2.INTER_LINEAR)
    rgb = resized[..., ::-1].copy() / 255
    return ((rgb - np.asarray(pp['mean_rgb'], np.float32)) / np.asarray(pp['std_rgb'], np.float32)).transpose(2, 0, 1)


def cache(run_dir, conditions=None):
    cfg, manifest = read(run_dir / 'CONFIG.json'), read(run_dir / 'manifest.json')
    records = manifest['records']
    conditions = conditions or list(cfg['conditions'])
    for name, expected in cfg['model']['source_sha256'].items():
        if C.sha(OLD / name) != expected:
            raise RuntimeError(f'Frozen model/target code changed: {name}')
    if C.sha(Path(cfg['model']['backbone'])) != cfg['model']['backbone_sha256']:
        raise RuntimeError('Backbone file changed')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = None
    for condition in conditions:
        dest = run_dir / 'cache' / condition
        dest.mkdir(parents=True, exist_ok=True)
        expected = signature(run_dir, cfg, condition)
        if (dest / 'CACHE.json').exists():
            old = read(dest / 'CACHE.json')
            if old['signature'] != expected or old['producer_sha256'] != C.sha(Path(__file__)):
                raise RuntimeError(f'Cache provenance changed: {condition}')
            features = np.load(dest / 'features.npy', mmap_mode='r')
            if list(features.shape) != [len(records), *cfg['preprocessing']['feature_shape']]:
                raise RuntimeError(f'Cache shape changed: {condition}')
            log(f'Reusing complete {condition} cache')
            continue
        if backbone is None:
            backbone = C.load_backbone(cfg['model']['backbone'], device)
        temp = dest / 'features.pending.npy'
        features = np.lib.format.open_memmap(temp, mode='w+', dtype=np.float16,
                                            shape=(len(records), *cfg['preprocessing']['feature_shape']))
        started = time.monotonic()
        for start in range(0, len(records), 12):
            batch = records[start:start + 12]
            pixels = np.stack([prepare_image(r, cfg['conditions'][condition], cfg['preprocessing']) for r in batch])
            with torch.no_grad():
                values = backbone(torch.from_numpy(pixels).to(device)).float()
            if tuple(values.shape[1:]) != tuple(cfg['preprocessing']['feature_shape']) or not torch.isfinite(values).all():
                raise RuntimeError('Nonfinite or unexpected frozen feature output')
            values = values.cpu().numpy().astype(np.float16)
            if not np.isfinite(values).all():
                raise RuntimeError('FP16 cache overflow')
            features[start:start + len(batch)] = values
            if start % 480 == 0:
                log(f'{condition} features {start}/{len(records)}')
        features.flush()
        del features
        temp.replace(dest / 'features.npy')
        write(dest / 'CACHE.json', dict(signature=expected, frames=len(records), shape=[len(records), *cfg['preprocessing']['feature_shape']],
              dtype='float16', producer_sha256=C.sha(Path(__file__)), features_sha256=C.sha(dest / 'features.npy'),
              elapsed_sec=time.monotonic() - started, device=str(device), torch=torch.__version__))
        log(f'Cache completed: {condition}, {time.monotonic() - started:.1f}s')


def smoke(main_dir):
    """Separate tiny synthetic-only configuration; no smoke metric is main evidence."""
    source = read(main_dir / 'manifest.json')
    ids = source['train_regimes']['base'][:8] + source['populations']['low_train'][:8]
    ids += source['populations']['synth_val'][:4] + source['populations']['low_val'][:4]
    assert len(ids) == len(set(ids)) == 24
    reverse = {old: new for new, old in enumerate(ids)}
    manifest = dict(schema='dht_padding_view_smoke_v1', records=[], populations={}, train_regimes={})
    for old in ids:
        r = copy.deepcopy(source['records'][old])
        r['index'] = reverse[old]
        manifest['records'].append(r)
        manifest['populations'].setdefault(r['population'], []).append(r['index'])
    manifest['train_regimes'] = dict(base=list(range(8)), low_balanced=list(range(4)) + list(range(8, 12)))
    manifest['source_main_manifest_sha256'] = C.sha(main_dir / 'manifest.json')
    out = main_dir / 'smoke'
    out.mkdir(exist_ok=True)
    cfg = config_for(out, stage='smoke')
    for name, value in [('CONFIG.json', cfg), ('manifest.json', manifest)]:
        if (out / name).exists() and read(out / name) != value:
            raise RuntimeError(f'Smoke provenance differs: {name}')
        write(out / name, value)
    (out / 'PURPOSE.md').write_text('# Pipeline smoke test only\n\n24 synthetic frames, 20 steps; independent weights and outputs. Not main experiment performance.\n')
    write(out / 'DATA_AUDIT.json', dict(stage='smoke', source_main_manifest_sha256=C.sha(main_dir / 'manifest.json'),
                                      frames=24, real_frames=0, training_frames_per_regime=8))
    cache(out)
    log(f'Smoke configuration/caches ready: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'data/pallet/results/dht_padding_view_v1')
    parser.add_argument('--source-manifest', type=Path, default=ROOT / 'data/pallet/results/hough_attention_transfer_v1/manifest.json')
    parser.add_argument('--phase', choices=['metadata', 'cache', 'smoke', 'all'], default='metadata')
    parser.add_argument('--conditions', nargs='+', choices=['reflect100', 'black100', 'mean100'])
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if args.phase in ('metadata', 'all'):
        metadata(out, args.source_manifest.resolve())
    if args.phase in ('cache', 'all'):
        cache(out, args.conditions)
    if args.phase in ('smoke', 'all'):
        smoke(out)


if __name__ == '__main__':
    main()
