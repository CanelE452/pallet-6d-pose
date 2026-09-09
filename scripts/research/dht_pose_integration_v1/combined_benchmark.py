"""Measure actual frozen baseline + image DHT + corner fusion latency.

The timing cohort is selected solely by frame ID hashes. Ground-truth points,
support, visibility and errors are never read by the timed pipelines or used
to choose records. Component timing sums are not combined-pipeline timings.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import dht_lines
import dope_baseline
import yolo_baseline
from fusion import fuse_corners
from resize_control import correct_legacy_resize_coordinates


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def synchronize():
    torch.cuda.synchronize()


def statistics(values):
    return dict(n=len(values), mean_ms=float(np.mean(values)),
                median_ms=float(np.median(values)), p90_ms=float(np.percentile(values, 90)))


def prediction_arrays(prediction):
    return np.asarray(prediction['kps'], dtype=np.float64), np.asarray(prediction['kp_valid'], dtype=bool)


def timed_pipeline(baseline, dht, image, lam, postprocess=None):
    """One actual end-to-end invocation, with zero-lambda DHT short circuit."""
    synchronize()
    begin = time.perf_counter()
    raw = baseline.predict(image)
    points, valid = prediction_arrays(raw)
    if postprocess is not None:
        points, valid = postprocess(points, valid, image, raw)
    lines = None
    if lam != 0:
        lines = dht.predict(image)
        points, valid = fuse_corners(points, valid, lines, lam, image.shape[1], image.shape[0])
    synchronize()
    elapsed = (time.perf_counter()-begin)*1000
    return elapsed, raw, points, valid, lines


def compare_arrays(actual, reference, tolerance=1e-3):
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if actual.shape != reference.shape:
        raise ValueError('Parity arrays have different shapes')
    finite_a, finite_b = np.isfinite(actual), np.isfinite(reference)
    same_finite = bool(np.array_equal(finite_a, finite_b))
    shared = finite_a & finite_b
    maximum = float(np.max(np.abs(actual[shared]-reference[shared]))) if shared.any() else 0.
    exact = same_finite and bool(np.array_equal(actual[shared].view(np.uint64), reference[shared].view(np.uint64)))
    return dict(PASS=same_finite and maximum <= tolerance, bitwise_equal=exact,
                finite_mask_equal=same_finite, max_abs_delta_px=maximum, tolerance_px=tolerance)


def summarize_parity(rows):
    return dict(PASS=all(r['PASS'] for r in rows), checked_frames=len(rows),
                bitwise_equal_frames=sum(r['bitwise_equal'] for r in rows),
                max_abs_delta_px=max(r['max_abs_delta_px'] for r in rows),
                tolerance_px=1e-3, mismatched_ids=[r['id'] for r in rows if not r['PASS']], rows=rows)


def choose_records(manifest):
    result = []
    for population in ('synth_test', 'real_dev'):
        candidates = [r for r in manifest['records'] if r['population'] == population]
        candidates.sort(key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        result.extend(candidates[:12])
    if len(result) != 24:
        raise ValueError('Expected12 fixed synthetic test and12 fixed real DEV benchmark records')
    return result


def verify_sources(paths):
    for path, expected in paths.items():
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != expected:
            raise ValueError(f'Source bytes differ from frozen upstream artifact: {path}')


def sources():
    result = {str(HERE / name): sha(HERE / name) for name in
              ('combined_benchmark.py', 'fusion.py', 'resize_control.py', 'dope_baseline.py', 'yolo_baseline.py', 'dht_lines.py')}
    result.update(dht_lines.source_hashes())
    result.update(yolo_baseline.source_hashes())
    return result


def corrected_dope(points, valid, image, raw):
    return correct_legacy_resize_coordinates(points, valid, image.shape[1], image.shape[0],
                                              raw['input_shape_chw'], pad=100, shortest_side=400)


def summarize_timing(rows):
    by_scope = {}
    for scope in ('all', 'synth_test', 'real_dev'):
        subset = rows if scope == 'all' else [r for r in rows if r['population'] == scope]
        arms = {arm: statistics([r['wall_ms'] for r in subset if r['arm'] == arm])
                for arm in ('baseline', 'selected', 'fixed_lambda1')}
        base = {(r['id'], r['repeat']): r['wall_ms'] for r in subset if r['arm'] == 'baseline'}
        paired = {arm: statistics([r['wall_ms']-base[(r['id'], r['repeat'])]
                                   for r in subset if r['arm'] == arm])
                  for arm in ('selected', 'fixed_lambda1')}
        by_scope[scope] = {'arms': arms, 'paired_added_latency_ms': paired}
    return by_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir/'PURPOSE.md').is_file():
        raise ValueError('Prepared results directory with PURPOSE.md is required')
    paths = {name: run_dir/name for name in
             ('CONFIG.json', 'manifest.json', 'SELECTION.json', 'BASELINE_DOPE.json', 'BASELINE_YOLO.json', 'DHT_LINES.json')}
    inputs = {name: read(path) for name, path in paths.items()}
    config, manifest, selection = (inputs[name] for name in ('CONFIG.json', 'manifest.json', 'SELECTION.json'))
    if config['schema'] != 'dht_pose_integration_v1':
        raise ValueError('Unexpected integration experiment schema')
    frozen = {name: sha(path) for name, path in paths.items()}
    for name, value in inputs.items():
        if name not in ('CONFIG.json', 'manifest.json'):
            if value['config_sha256'] != frozen['CONFIG.json'] or value['manifest_sha256'] != frozen['manifest.json']:
                raise ValueError(f'Upstream {name} uses different configuration or records')
            if 'source_sha256' in value:
                verify_sources(value['source_sha256'])
    if config['runtime']['combined_benchmark_seed'] != 1 or config['runtime']['combined_benchmark_records'] != 24:
        raise ValueError('Expected prespecified seed1/24-frame benchmark')
    if config['runtime']['batch'] != 1 or config['runtime']['warmup'] != 5:
        raise ValueError('Expected batch1/warmup5 timing protocol')
    variant_names = ('DOPE', 'DOPE_exact_resize', 'YOLO')
    for variant in variant_names:
        if selection['models'][variant]['lambda'] not in config['fusion']['lambda_grid']:
            raise ValueError('Selection contains an unregistered lambda')
    source_hashes = sources()
    output_path = run_dir/'RUNTIME.json'
    if output_path.exists():
        old = read(output_path)
        if old.get('input_sha256') == frozen and old.get('source_sha256') == source_hashes and old.get('complete'):
            print('Verified existing completed RUNTIME.json', flush=True)
            return
        raise ValueError('Existing runtime artifact provenance differs')
    selected = choose_records(manifest)
    images = {}
    for record in selected:
        if sha(record['image']) != record['image_sha256']:
            raise ValueError(f"Benchmark image changed: {record['id']}")
        image = cv2.imread(record['image'])
        if image is None or image.shape[:2] != (record['height'], record['width']):
            raise ValueError('Benchmark image geometry mismatch')
        images[record['id']] = image
    if not torch.cuda.is_available():
        raise RuntimeError('Actual combined CUDA latency requires CUDA')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dht_reference = {r['id']: r['seeds']['1']['lines'] for r in inputs['DHT_LINES.json']['records']}
    dht = dht_lines.ImageDHT(config, seed=1, device='cuda')
    warm_record = next(r for r in manifest['records'] if r['population'] == 'synth_val')
    if sha(warm_record['image']) != warm_record['image_sha256']:
        raise ValueError('Warmup source image changed')
    warm_image = cv2.imread(warm_record['image'])
    for _ in range(5):
        dht.predict(warm_image)
    by_model, parity, measurements = {}, {}, []
    for variant in variant_names:
        model_name = 'DOPE' if variant.startswith('DOPE') else 'YOLO'
        module = dope_baseline if model_name == 'DOPE' else yolo_baseline
        baseline = module.Baseline(config, device='cuda')
        postprocess = corrected_dope if variant == 'DOPE_exact_resize' else None
        lam = selection['models'][variant]['lambda']
        arms = {'baseline': 0., 'selected': float(lam), 'fixed_lambda1': 1.}
        baseline_reference = {r['id']: r for r in inputs[f'BASELINE_{model_name}.json']['records']}
        for _ in range(5):
            timed_pipeline(baseline, dht, warm_image, 1., postprocess)
        baseline_parity, line_parity, variant_rows = [], [], []
        for frame_index, record in enumerate(selected):
            image = images[record['id']]
            for repeat in range(3):
                labels = list(arms)
                shift = (frame_index+repeat) % len(labels)
                order = labels[shift:]+labels[:shift]
                for order_index, arm in enumerate(order):
                    elapsed, raw, points, valid, lines = timed_pipeline(baseline, dht, image, arms[arm], postprocess)
                    variant_rows.append(dict(id=record['id'], population=record['population'], model=variant,
                                             arm=arm, repeat=repeat, order_position=order_index,
                                             lam=arms[arm], dht_used=lines is not None, wall_ms=elapsed))
                    if repeat == 0 and arm == 'baseline':
                        reference = baseline_reference[record['id']]
                        agreement = compare_arrays(raw['kps'], reference['kps'])
                        agreement['valid_mask_equal'] = raw['kp_valid'] == reference['kp_valid']
                        agreement['detected_equal'] = raw['detected'] == reference['detected']
                        agreement['PASS'] &= agreement['valid_mask_equal'] and agreement['detected_equal']
                        if postprocess is not None:
                            reference_points, reference_valid = prediction_arrays(reference)
                            corrected, corrected_valid = corrected_dope(reference_points, reference_valid, image, reference)
                            agreement['corrected_points'] = compare_arrays(points, corrected)
                            agreement['PASS'] &= agreement['corrected_points']['PASS'] and np.array_equal(valid, corrected_valid)
                        baseline_parity.append(dict(id=record['id'], **agreement))
                    if repeat == 0 and arm == 'fixed_lambda1':
                        agreement = compare_arrays(lines, dht_reference[record['id']])
                        role_delta = np.max(np.abs(np.asarray(lines)-np.asarray(dht_reference[record['id']])), axis=(1, 2))
                        agreement['changed_role_indices'] = np.flatnonzero(role_delta > 1e-3).tolist()
                        line_parity.append(dict(id=record['id'], **agreement))
            print(f'{variant} benchmark {frame_index+1}/24', flush=True)
        parity[variant] = {'baseline_predictions': summarize_parity(baseline_parity),
                           'fresh_image_dht_seed1': summarize_parity(line_parity)}
        by_model[variant] = dict(selected_lambda=lam, selected_uses_dht=bool(lam != 0),
                                baseline_model=model_name, exact_resize_control=postprocess is not None,
                                by_population=summarize_timing(variant_rows))
        measurements.extend(variant_rows)
        del baseline
        gc.collect()
        torch.cuda.empty_cache()
    if any(sha(path) != frozen[name] for name, path in paths.items()) or sources() != source_hashes:
        raise ValueError('Inputs or source changed during benchmark')
    all_parity = all(p['PASS'] for item in parity.values() for p in item.values())
    output = dict(schema='dht_pose_runtime_v1', complete=True, parity_PASS=all_parity,
                  config_sha256=frozen['CONFIG.json'], manifest_sha256=frozen['manifest.json'],
                  input_sha256=frozen, source_sha256=source_hashes,
                  weights_sha256={name: config['models'][name]['weights_sha256'] for name in ('dope', 'yolo')} |
                                 {'dht_seed1': config['dht']['checkpoint_sha256']['1'], 'dht_backbone': config['dht']['backbone_sha256']},
                  runtime=dict(device='cuda', gpu=torch.cuda.get_device_name(), python=sys.executable,
                               torch=torch.__version__, cuda=torch.version.cuda, batch=1, warmup=5, repeats=3,
                               torch_threads=4, opencv_threads=1, cudnn_benchmark=False, matmul_tf32=False),
                  timing_definition='Actual synchronized wall timer around baseline.predict, optional exact DOPE resize correction, fresh-image VGG+DHT.predict, and corner fusion, returning CPU arrays. Disk I/O, JSON serialization, model loading and warmup are excluded. Lambda0 skips DHT and fusion. No feature sharing or feature cache is used in timed DHT calls.',
                  cohort_definition='First12 SHA256(frame ID)-ordered synth_test plus first12 SHA256(frame ID)-ordered real_dev. Independent of errors and GT. Three repeats; arm execution order rotates within each frame/repeat.',
                  warmup_id=warm_record['id'], n_unique_frames=24, invocations_per_model_arm=72,
                  selected_records=[{k:r[k] for k in ('id', 'population', 'image_sha256', 'width', 'height')} for r in selected],
                  by_model=by_model, parity=parity, measurements=measurements,
                  limitation='Latency applies to this single-GPU serialized two-network implementation and24-frame descriptive cohort. Repeats are paired invocations, not independent accuracy samples. No accuracy metric or GT-based decision is computed here. A parity failure must be reviewed before equating these timed predictions with corpus results.')
    write(output_path, output)
    print(json.dumps({'complete': True, 'parity_PASS': all_parity,
                      'output': str(output_path), 'models': {k:v['by_population']['all'] for k,v in by_model.items()}}), flush=True)


if __name__ == '__main__':
    main()
