"""Pack frozen P4/point/Hough observations; synthetic targets stay separate.

No detector forward, new proposal selection, real annotation read, or source
mutation occurs here. P4 retains its original stride16 grid and is padded only
on the bottom/right to40x40. Model inputs and loss targets are separate objects.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_dht_decoder_probe_v1 import geometry as G


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'pallet_dht_structured_cache_v2'
CALIBRATION_PREFIX = 'pallet_dht_structured_v2:calibration:20260909:'
INPUT_SPECS = {
    'p4': ((128, 40, 40), 'float32'),
    'baseline_points': ((9, 2), 'float32'),
    'point_conf': ((9,), 'float32'), 'point_valid': ((9,), 'bool'),
    'diagonal': ((), 'float32'), 'detected': ((), 'bool'),
    'raw_to_input_affine': ((2, 3), 'float32'),
    'input_shape_hw': ((2,), 'int64'),
    'line_h': ((12, 4, 3), 'float32'),
    'peak_logits': ((12, 4), 'float32'), 'peak_valid': ((12, 4), 'bool'),
    'intersections': ((8, 49, 2), 'float32'),
    'intersection_valid': ((8, 49), 'bool'),
}
TARGET_SPECS = {
    'points': ((9, 2), 'float32'), 'valid': ((9,), 'bool'),
    'loss_valid': ((9,), 'bool'), 'matched': ((), 'bool'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_sha256():
    return {str(Path(p).resolve()): sha(p) for p in (__file__, G.__file__)}


class BoundInputs:
    def __init__(self, snapshot):
        self.expected = snapshot['sha256']
        self.actual = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        expected = self.expected.get(str(path)) if expected is None else expected
        if expected is None:
            raise ValueError(f'Unbound packing input: {path}')
        digest = sha(path)
        if digest != expected:
            raise ValueError(f'Packing source changed: {path}')
        self.actual[str(path)] = digest
        return path

    def verify(self):
        for path, digest in self.actual.items():
            if sha(path) != digest:
                raise ValueError(f'Packing input changed during preparation: {path}')


def split_indices(records, config):
    """Preserve source index/order; only assign the registered synthetic split."""
    if config['calibration_prefix'] != CALIBRATION_PREFIX:
        raise ValueError('Unexpected calibration SHA prefix')
    if [r['index'] for r in records] != list(range(len(records))):
        raise ValueError('Source cache indices must be contiguous and ordered')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate source ID')
    pool = [r for r in records if r['population'] == 'synth_train']
    ranked = sorted(pool, key=lambda r: (hashlib.sha256(
        (config['calibration_prefix'] + r['id']).encode()).hexdigest(), r['id']))
    calibration = [r['index'] for r in ranked[:config['calibration_count']]]
    selected = set(calibration)
    populations = {
        'train': [r['index'] for r in pool if r['index'] not in selected],
        'calibration': calibration,
        'synth_val': [r['index'] for r in records if r['population'] == 'synth_val'],
        'real_dev': [r['index'] for r in records if r['population'] == 'real_dev'],
    }
    expected = dict(train=config['train_count'], calibration=config['calibration_count'],
        synth_val=config['synthetic_validation_count'], real_dev=config['real_dev_count'])
    if {key: len(value) for key, value in populations.items()} != expected:
        raise ValueError('Source population counts differ from the fixed protocol')
    all_indices = sum(populations.values(), [])
    if len(all_indices) != len(records) or len(set(all_indices)) != len(records):
        raise ValueError('Splits must partition every source record exactly once')
    return populations


def synthetic_target(source, cached):
    """Full source GT remains available even for unmatched synthetic detections.

    The independent loss_valid mask additionally requires the frozen IoU match
    and finite predicted-point availability. Real records return before any
    source_record or annotation lookup.
    """
    result = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in TARGET_SPECS.items()}
    if source['population'] == 'real_dev':
        return result
    record = source['source_record']
    if record['source_kind'] != 'synthetic' or len(record['targets']) != 1:
        raise ValueError('Expected the frozen single-instance synthetic source')
    target = np.asarray(record['targets'][0]['keypoints_normalized'], dtype=np.float64)
    if target.shape != (9, 3) or not np.isfinite(target).all():
        raise ValueError('Malformed synthetic label')
    h, w = record['prepared_shape_hw']
    if [h - 2 * record['reflect_pad_px'], w - 2 * record['reflect_pad_px']] != [source['height'], source['width']]:
        raise ValueError('Synthetic prepared/raw shape mismatch')
    result['points'] = (target[:, :2] * [w, h] - record['reflect_pad_px']).astype(np.float32)
    result['valid'] = target[:, 2] > 0
    matched = bool(cached.get('loss_matched', False))
    if matched and cached.get('matched_gt_index') != 0:
        raise ValueError('Frozen matched target index changed')
    result['matched'] = np.asarray(matched, dtype=bool)
    result['loss_valid'] = result['valid'] & np.asarray(cached['baseline']['point_valid'], bool) & matched
    return result


def pack_observation(record, frame, evidence):
    """Prediction-only transformation; no target argument is accepted."""
    p4 = np.asarray(frame['p4'])
    if p4.dtype != np.float32 or p4.ndim != 3 or p4.shape[0] != 128:
        raise ValueError('Expected original FP32 128-channel P4')
    _, h, w = p4.shape
    if max(h, w) > 40 or min(h, w) < 1 or not np.isfinite(p4).all():
        raise ValueError('Invalid P4 footprint')
    input_hw = np.asarray(frame['input_shape_hw'], dtype=np.int64)
    affine = np.asarray(frame['raw_to_input_affine'], dtype=np.float64)
    if not np.array_equal(input_hw, [16 * h, 16 * w]):
        raise ValueError('Original stride16 P4/input geometry mismatch')
    if not np.array_equal(input_hw, record['input_shape_hw']) or not np.array_equal(affine, record['raw_to_input_affine']):
        raise ValueError('Cached metadata affine/input shape changed')
    if not np.isfinite(affine).all() or affine.shape != (2, 3):
        raise ValueError('Malformed original-to-input affine')
    logits = np.asarray(frame['logits'])
    theta, rho = np.asarray(frame['theta']), np.asarray(frame['rho'])
    valid_lattice = np.asarray(frame['lattice_valid'], dtype=bool)
    if logits.shape != (12, len(theta), len(rho)):
        raise ValueError('Hough logits shape mismatch')
    line_h = np.zeros((12, 4, 3), np.float64)
    peak_logits = np.zeros((12, 4), np.float32)
    peak_valid = np.zeros((12, 4), bool)
    for role in range(12):
        peaks = G.topk_lines(logits[role], theta, rho, valid_lattice,
            dict(top_k=4, nms_angle_degrees=4., nms_rho_feature_cells=1.))
        for k, peak in enumerate(peaks):
            line_h[role, k] = G.feature_line_to_raw(peak['theta'], peak['rho'],
                (h, w), input_hw, affine)
            peak_logits[role, k] = peak['logit']
            peak_valid[role, k] = True
    old_h = np.asarray(evidence['line_fusion__line_peaks_h_raw'])
    old_valid = np.asarray(evidence['line_fusion__line_peak_valid'], dtype=bool)
    if old_h.shape != (12, 4, 3) or not np.array_equal(old_valid, peak_valid):
        raise ValueError('Saved line-mode availability differs')
    if not np.allclose(old_h[old_valid], line_h[old_valid], rtol=0, atol=1e-8):
        raise ValueError('Saved NMS line/affine provenance mismatch')
    baseline = record['baseline']
    padded = np.zeros((128, 40, 40), np.float32)
    padded[:, :h, :w] = p4
    result = dict(p4=padded, baseline_points=baseline['points'], point_conf=baseline['point_conf'],
        point_valid=baseline['point_valid'], diagonal=math.hypot(record['width'], record['height']),
        detected=baseline['detected'], raw_to_input_affine=affine, input_shape_hw=input_hw,
        line_h=line_h, peak_logits=peak_logits, peak_valid=peak_valid,
        intersections=evidence['line_fusion__candidates_xy'],
        intersection_valid=evidence['line_fusion__candidate_valid'])
    for key, (shape, dtype) in INPUT_SPECS.items():
        value = np.asarray(result[key], dtype=dtype)
        if value.shape != shape or (value.dtype.kind == 'f' and not np.isfinite(value).all()):
            raise ValueError(f'Malformed packed input: {key}')
        result[key] = value
    if not np.array_equal(result['p4'][:, :h, :w], p4):
        raise ValueError('P4 values changed')
    if result['p4'][:, h:, :].any() or result['p4'][:, :, w:].any():
        raise ValueError('Bottom/right P4 padding must be exactly zero')
    if not result['intersection_valid'][:, 0].all():
        raise ValueError('Existing identity candidate must remain available')
    return result


def prepare(run_dir):
    run = Path(run_dir).resolve()
    if not (run / 'PURPOSE.md').is_file():
        raise ValueError('New dataset output PURPOSE.md is required')
    protocol = read(run / 'PROTOCOL.json')
    protocol_digest = sha(run / 'PROTOCOL.json')
    code_digests = source_sha256()
    if protocol['schema'] != 'pallet_dht_structured_v2_protocol':
        raise ValueError('Wrong preparation protocol')
    if (run / 'CACHE_COMPLETION.json').exists():
        return CachedData(run).completion
    started = time.perf_counter()
    snapshot_ref = protocol['input_snapshot']
    snapshot_path = Path(snapshot_ref['path']).resolve()
    if sha(snapshot_path) != snapshot_ref['sha256']:
        raise ValueError('Frozen source snapshot changed')
    snapshot = read(snapshot_path)
    if not snapshot['complete'] or not snapshot['PASS'] or len(snapshot['sha256']) != 6471:
        raise ValueError('Expected completed 6471-input source audit')
    bound = BoundInputs(snapshot)
    source = Path(protocol['input_run']).resolve()
    manifest = read(bound.bind(source / 'MANIFEST.json'))
    cached = read(bound.bind(source / 'CACHE_RECORDS.json'))
    if not manifest['complete'] or not cached['complete'] or manifest['real_GT_read'] is not False:
        raise ValueError('Source cache provenance is incomplete')
    records = cached['records']
    if len(records) != 2879 or len(manifest['records']) != len(records):
        raise ValueError('Expected all2879 source frames')
    populations = split_indices(records, protocol['split'])
    arrays, descriptions = {}, {}
    for group, specs in [('inputs', INPUT_SPECS), ('targets', TARGET_SPECS)]:
        arrays[group], descriptions[group] = {}, {}
        for key, (shape, dtype) in specs.items():
            path = run / 'cache' / group / (key + '.npy')
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays[group][key] = np.lib.format.open_memmap(path, mode='w+', shape=(len(records), *shape), dtype=dtype)
            descriptions[group][key] = dict(path=str(path), shape=[len(records), *shape], dtype=dtype)
    for index, (record, src) in enumerate(zip(records, manifest['records'])):
        for key in ('index', 'id', 'population', 'image', 'image_sha256', 'width', 'height', 'session_id'):
            if record[key] != src[key]:
                raise ValueError(f'Original metadata changed: {index} {key}')
        frame_path = bound.bind(source / 'frames' / f'{index:06d}.npz')
        evidence_path = bound.bind(record['candidate_evidence_npz'])
        with np.load(frame_path, allow_pickle=False) as frame, np.load(evidence_path, allow_pickle=False) as evidence:
            frame_record = json.loads(str(frame['record_json']))
            if frame_record['baseline'] != record['baseline'] or frame_record['id'] != record['id']:
                raise ValueError('Prediction record/frame binding mismatch')
            values = pack_observation(record, frame, evidence)
        if src['population'] != 'real_dev':
            bound.bind(src['source_record']['label'], src['source_record']['label_sha256'])
        targets = synthetic_target(src, record)
        for key, value in values.items():
            arrays['inputs'][key][index] = value
        for key, value in targets.items():
            arrays['targets'][key][index] = value
        if src['population'] == 'real_dev' and any(np.any(x) for x in targets.values()):
            raise ValueError('Real targets must be zero/false')
        if (index + 1) % 200 == 0:
            print(f'Packed frozen observations {index + 1}/{len(records)}', flush=True)
    array_hashes = {}
    for group in arrays:
        for key, array in arrays[group].items():
            array.flush()
            path = descriptions[group][key]['path']
            array_hashes[path] = sha(path)
    record_path = run / 'CACHE_RECORDS.json'
    write(record_path, dict(schema=SCHEMA, complete=True, records=records, populations=populations,
        source_record_order_preserved=True, real_GT_read=False,
        source_cache_records_sha256=bound.actual[str(source / 'CACHE_RECORDS.json')]))
    array_manifest_path = run / 'CACHE_ARRAYS.json'
    write(array_manifest_path, dict(schema=SCHEMA, complete=True, arrays=descriptions,
        model_input_keys=list(INPUT_SPECS), targets_are_separate=True,
        p4_padding='Bottom/right zero to40x40; original stride16 and actual input shape retained.',
        coordinates='Original pixels; floating model inputs FP32. Original double-precision metadata retained.',
        full_synthetic_GT_mask_preserved=True, real_GT_read=False))
    bound.verify()
    if sha(run / 'PROTOCOL.json') != protocol_digest or source_sha256() != code_digests:
        raise ValueError('Preparation protocol/source changed during packing')
    if sha(snapshot_path) != snapshot_ref['sha256']:
        raise ValueError('Source snapshot changed during packing')
    bindings = dict(protocol_sha256=protocol_digest, snapshot_sha256=snapshot_ref['sha256'],
        source_sha256=code_digests, input_sha256=bound.actual,
        cache_records_sha256=sha(record_path), cache_arrays_sha256=sha(array_manifest_path))
    completion = dict(schema=SCHEMA, complete=True, PASS=True,
        generated_at_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings,
        n_frames=len(records), populations={k: len(v) for k, v in populations.items()},
        array_sha256=array_hashes, real_GT_read=False, real_targets_zero=True,
        full_synthetic_GT_mask_preserved=True, loss_mask_requires_original_prediction_match=True,
        old_source_snapshot_n_inputs=6471, packing_inputs_verified=len(bound.actual),
        real_annotation_files_opened=0, no_new_CNN_forward=True, no_training=True,
        input_files_unmodified=True, elapsed_seconds=time.perf_counter() - started)
    write(run / 'CACHE_COMPLETION.json', completion)
    print(json.dumps({k: completion[k] for k in ('complete', 'PASS', 'n_frames', 'populations', 'elapsed_seconds')}))
    return completion


class CachedData:
    """Memory-mapped observations with explicit, separate synthetic targets."""
    def __init__(self, run_dir, *, verify=True):
        self.run_dir = Path(run_dir).resolve()
        self.completion = read(self.run_dir / 'CACHE_COMPLETION.json')
        c = self.completion
        if c.get('schema') != SCHEMA or not c.get('complete') or not c.get('PASS'):
            raise ValueError('A completed prepared cache is required')
        binding = c['bindings']
        for name, expected in [('PROTOCOL.json', binding['protocol_sha256']),
                ('CACHE_RECORDS.json', binding['cache_records_sha256']),
                ('CACHE_ARRAYS.json', binding['cache_arrays_sha256'])]:
            if sha(self.run_dir / name) != expected:
                raise ValueError(f'Cache binding changed: {name}')
        if source_sha256() != binding['source_sha256']:
            raise ValueError('Cache preparation source changed')
        if verify:
            for path, expected in c['array_sha256'].items():
                if sha(path) != expected:
                    raise ValueError(f'Packed array changed: {path}')
        metadata = read(self.run_dir / 'CACHE_RECORDS.json')
        self.records, self.populations = metadata['records'], metadata['populations']
        descriptions = read(self.run_dir / 'CACHE_ARRAYS.json')['arrays']
        loaded = {}
        for group, specs in [('inputs', INPUT_SPECS), ('targets', TARGET_SPECS)]:
            loaded[group] = {}
            if set(descriptions[group]) != set(specs):
                raise ValueError(f'Packed array keys changed: {group}')
            for key, (shape, dtype) in specs.items():
                array = np.load(descriptions[group][key]['path'], mmap_mode='r', allow_pickle=False)
                if array.shape != (len(self.records), *shape) or array.dtype != np.dtype(dtype):
                    raise ValueError(f'Packed array shape/type changed: {group}/{key}')
                loaded[group][key] = array
        self.inputs, self.targets = loaded['inputs'], loaded['targets']
        real = self.populations['real_dev']
        if any(np.any(value[real]) for value in self.targets.values()):
            raise ValueError('Real targets are not zero/false')

    def batch(self, indices, device='cpu'):
        import torch
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.records)):
            raise ValueError('Expected a valid one-dimensional source-index list')
        def tensors(arrays):
            return {key: torch.from_numpy(np.array(value[indices], copy=True)).to(device)
                    for key, value in arrays.items()}
        return tensors(self.inputs), tensors(self.targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check'), default='prepare')
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run_dir)
    else:
        data = CachedData(args.run_dir)
        print(json.dumps(dict(complete=True, PASS=True, n_frames=len(data.records),
            populations={k: len(v) for k, v in data.populations.items()}, real_GT_read=False)))
