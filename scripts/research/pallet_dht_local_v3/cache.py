"""Exact rectangular image pixels for local refinement, with separate targets.

Preserves all 2,879 original records and the v2 split. No detector forward or
annotation file is opened. Gray is computed AFTER the canonical BGR resize and
LetterBox; extra square storage is zero-padded only on the bottom/right.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.research.pallet_line_pose_v1 import source_data as SD


SCHEMA = 'pallet_dht_local_cache_v3'
INPUT_KEYS = ('baseline_points', 'point_conf', 'point_valid', 'diagonal',
              'detected', 'raw_to_input_affine', 'input_shape_hw')
TARGET_KEYS = ('points', 'valid', 'loss_valid', 'matched')


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


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing frozen artifact differs: {path}')
    else:
        write(path, value)


def sources():
    return {str(Path(p).resolve()): sha(p) for p in (__file__, SD.__file__)}


def verify_hashes(values):
    for path, digest in values.items():
        if sha(path) != digest:
            raise ValueError(f'Frozen cache input changed: {path}')


def gray_observation(record):
    """Read image bytes only and reproduce the frozen predictor's image raster."""
    import cv2
    content = Path(record['image']).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != record['image_sha256']:
        raise ValueError(f'Source image SHA differs: {record["id"]}')
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Image decode failed: {record["id"]}')
    raw = image[100:-100, 100:-100].copy() if record['prepared_image'] else image
    if raw.shape[:2] != (record['height'], record['width']):
        raise ValueError('Original image shape differs')
    reflected = cv2.copyMakeBorder(raw, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
    if record['prepared_image'] and not np.array_equal(reflected, image):
        raise ValueError('Synthetic PNG is not exact original plus reflect101100')
    chw, transform = SD.prepare_image(reflected, imgsz=640, auto=True, stride=32)
    if list(transform.input_shape_hw) != record['input_shape_hw']:
        raise ValueError('Gray input shape differs from actual frozen backbone input')
    affine = np.array([[transform.gain, 0., 100*transform.gain + transform.pad_ltrb[0]],
                       [0., transform.gain, 100*transform.gain + transform.pad_ltrb[1]]])
    if not np.array_equal(affine, np.asarray(record['raw_to_input_affine'])):
        raise ValueError('Actual frozen coordinate affine differs')
    # prepare_image uses uint8 BGR resize/pad before float RGB conversion.
    # The rounded /255 inverse exactly recovers those raster bytes.
    rgb = np.rint(chw.transpose(1, 2, 0)*255.).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h > 640 or w > 640:
        raise ValueError('Canonical rectangular raster exceeds640')
    canvas = np.zeros((1, 640, 640), np.uint8)
    canvas[0, :h, :w] = gray
    return canvas, digest, hashlib.sha256(gray.tobytes()).hexdigest()


def content_mask(affine, raw_shape_hw, input_shape_hw):
    """Original raw footprint at input-cell centres; no GT/visibility semantics.

    The saved axis-aligned affine is unchanged. Reflection, LetterBox and outer
    storage padding are excluded. The model must erode this mask for Sobel's
    3x3 support; this cache mask is deliberately not already eroded.
    """
    import torch
    if affine.ndim != 3 or affine.shape[1:] != (2, 3):
        raise ValueError('Expected affine[B,2,3]')
    b = len(affine)
    if raw_shape_hw.shape != (b, 2) or input_shape_hw.shape != (b, 2):
        raise ValueError('Expected raw/input H,W per frame')
    if bool((affine[:, 0, 1] != 0).any() or (affine[:, 1, 0] != 0).any()
            or (affine[:, [0, 1], [0, 1]] <= 0).any()):
        raise ValueError('Only canonical positive axis-aligned affines are allowed')
    axis = torch.arange(640, device=affine.device, dtype=affine.dtype) + .5
    raw_x = (axis[None] - affine[:, 0, 2, None]) / affine[:, 0, 0, None]
    raw_y = (axis[None] - affine[:, 1, 2, None]) / affine[:, 1, 1, None]
    x_valid = (raw_x >= 0) & (raw_x < raw_shape_hw[:, 1, None]) & (axis[None] < input_shape_hw[:, 1, None])
    y_valid = (raw_y >= 0) & (raw_y < raw_shape_hw[:, 0, None]) & (axis[None] < input_shape_hw[:, 0, None])
    return (y_valid[:, :, None] & x_valid[:, None, :])[:, None]


def prepare(run_dir, workers=4):
    import cv2
    run = Path(run_dir).resolve()
    if not (run/'PURPOSE.md').is_file():
        raise ValueError('Result PURPOSE.md must exist before preparation')
    protocol = read(run/'PROTOCOL.json')
    if protocol['schema'] != 'pallet_dht_local_v3_protocol' or not protocol['split']['reuse_exact_populations']:
        raise ValueError('Registered local v3 protocol required')
    if protocol['image_cache']['canvas_hw'] != [640, 640] or protocol['image_cache']['gray_dtype'] != 'uint8':
        raise ValueError('Registered uint8 square storage contract differs')
    base = Path(protocol['input_run']).resolve()
    bindings = {str(run/'PROTOCOL.json'): sha(run/'PROTOCOL.json'),
                str(base/'CACHE_COMPLETION.json'): protocol['input_cache_completion_sha256'],
                str(base/'PROTOCOL.json'): protocol['input_protocol_sha256']}
    verify_hashes(bindings)
    old = read(base/'CACHE_COMPLETION.json')
    if not (old['complete'] and old['PASS'] and old['real_targets_zero']):
        raise ValueError('Completed, separated v2 source cache required')
    for name, key in [('CACHE_RECORDS.json', 'cache_records_sha256'),
                      ('CACHE_ARRAYS.json', 'cache_arrays_sha256')]:
        bindings[str(base/name)] = old['bindings'][key]
    verify_hashes(bindings)
    verify_hashes(old['bindings']['source_sha256'])
    bindings.update(old['bindings']['source_sha256'])
    metadata = read(base/'CACHE_RECORDS.json')
    records, populations = metadata['records'], metadata['populations']
    if len(records) != 2879 or [r['index'] for r in records] != list(range(2879)):
        raise ValueError('Original2879 contiguous records required')
    expected = {k: protocol['split'][k] for k in ('train', 'calibration', 'synth_val', 'real_dev')}
    if {k: len(v) for k, v in populations.items()} != expected:
        raise ValueError('Original population membership/count contract differs')
    if sorted(sum(populations.values(), [])) != list(range(len(records))):
        raise ValueError('Original populations must partition all2879 frames')
    descriptions = read(base/'CACHE_ARRAYS.json')['arrays']
    for group, keys in [('inputs', INPUT_KEYS), ('targets', TARGET_KEYS)]:
        for key in keys:
            path = descriptions[group][key]['path']
            bindings[path] = old['array_sha256'][path]
    verify_hashes(bindings)
    code = sources()
    freeze(run/'provenance/cache_preparation/SOURCE_BEFORE_PACK.json',
        dict(schema=SCHEMA, source_sha256=code, input_sha256=bindings,
             protocol_sha256=bindings[str(run/'PROTOCOL.json')]))
    if (run/'CACHE_COMPLETION.json').exists():
        data = LocalData(run)
        return data.completion
    started = time.perf_counter()
    cv2.setNumThreads(1)
    folder = run/'cache'
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder/'image_gray.pending.npy'
    gray_path = folder/'image_gray.npy'
    if gray_path.exists():
        raise ValueError('Uncompleted gray file exists; preserve it before explicitly rebuilding')
    gray = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8,
                                    shape=(len(records), 1, 640, 640))
    image_sha, raster_sha = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for first in range(0, len(records), 32):
            chunk = records[first:first+32]
            for row, (canvas, digest, raster_digest) in zip(chunk, pool.map(gray_observation, chunk)):
                gray[row['index']] = canvas
                image_sha[str(Path(row['image']).resolve())] = digest
                raster_sha[row['id']] = raster_digest
            if (first//32) % 16 == 0:
                print(f'Gray raster preparation {min(first+32, len(records))}/{len(records)}', flush=True)
    gray.flush()
    del gray
    partial.replace(gray_path)
    arrays = {'inputs': {}, 'targets': {}}
    array_sha = {}
    def describe(group, key, path):
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        arrays[group][key] = dict(path=str(path), shape=list(data.shape), dtype=str(data.dtype))
        array_sha[str(path)] = sha(path)
    describe('inputs', 'image_gray', gray_path)
    for group, keys in [('inputs', INPUT_KEYS), ('targets', TARGET_KEYS)]:
        target = folder/group
        target.mkdir(exist_ok=True)
        for key in keys:
            source = Path(descriptions[group][key]['path'])
            path = target/f'{key}.npy'
            path.write_bytes(source.read_bytes())
            if sha(path) != bindings[str(source)]:
                raise ValueError('Copied source array differs')
            describe(group, key, path)
    raw_shape = np.array([[r['height'], r['width']] for r in records], np.int64)
    path = folder/'inputs/raw_shape_hw.npy'
    np.save(path, raw_shape, allow_pickle=False)
    describe('inputs', 'raw_shape_hw', path)
    (run/'CACHE_RECORDS.json').write_bytes((base/'CACHE_RECORDS.json').read_bytes())
    write(run/'CACHE_ARRAYS.json', dict(schema=SCHEMA, arrays=arrays,
        gray_pixels='Canonical resized gray before extra bottom/right storage zeros',
        content_mask='Computed at batch time from unchanged affine and raw/input shape; no mask file'))
    for key in TARGET_KEYS:
        value = np.load(arrays['targets'][key]['path'], mmap_mode='r')
        if np.any(value[populations['real_dev']]):
            raise ValueError('Real targets must remain all zero/false')
    verify_hashes(bindings)
    verify_hashes(code)
    verify_hashes(image_sha)
    receipt = dict(schema=SCHEMA, complete=True, PASS=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(), n_frames=len(records),
        populations=expected, source_sha256=code, input_sha256=bindings,
        image_sha256=image_sha, gray_rectangular_pixels_sha256=raster_sha,
        cache_records_sha256=sha(run/'CACHE_RECORDS.json'),
        cache_arrays_sha256=sha(run/'CACHE_ARRAYS.json'), array_sha256=array_sha,
        original_records_and_populations_byte_identical=True,
        targets_byte_identical=True, real_targets_zero=True,
        annotation_files_opened=0, real_GT_read=False, no_new_backbone_forward=True,
        P4_loaded=False, full_mask_stored=False, cv2_version=cv2.__version__,
        cv2_build_information_sha256=hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started)
    write(run/'CACHE_COMPLETION.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('complete', 'PASS', 'n_frames', 'populations', 'elapsed_seconds')}))
    return receipt


class LocalData:
    """Memory-mapped uint8 gray; selected inputs and targets stay separate."""
    def __init__(self, run_dir, *, verify=True):
        self.run = Path(run_dir).resolve()
        self.completion = read(self.run/'CACHE_COMPLETION.json')
        done = self.completion
        if not (done['schema'] == SCHEMA and done['complete'] and done['PASS']):
            raise ValueError('Completed local gray cache required')
        if sources() != done['source_sha256']:
            raise ValueError('Gray preparation code changed')
        verify_hashes(done['input_sha256'])
        verify_hashes({str(self.run/'CACHE_RECORDS.json'): done['cache_records_sha256'],
                       str(self.run/'CACHE_ARRAYS.json'): done['cache_arrays_sha256']})
        if verify:
            verify_hashes(done['array_sha256'])
        metadata = read(self.run/'CACHE_RECORDS.json')
        self.records, self.populations = metadata['records'], metadata['populations']
        desc = read(self.run/'CACHE_ARRAYS.json')['arrays']
        if set(desc['inputs']) != set(INPUT_KEYS) | {'image_gray', 'raw_shape_hw'} or set(desc['targets']) != set(TARGET_KEYS):
            raise ValueError('Unexpected model inputs or targets')
        def load(group):
            result = {}
            for key, row in desc[group].items():
                value = np.load(row['path'], mmap_mode='r', allow_pickle=False)
                if list(value.shape) != row['shape'] or str(value.dtype) != row['dtype'] or len(value) != len(self.records):
                    raise ValueError('Cached array shape/dtype differs')
                result[key] = value
            return result
        self.inputs, self.targets = load('inputs'), load('targets')
        if self.inputs['image_gray'].shape != (len(self.records), 1, 640, 640) or self.inputs['image_gray'].dtype != np.uint8:
            raise ValueError('Expected uint8 gray[B,1,640,640] storage')
        if any(np.any(value[self.populations['real_dev']]) for value in self.targets.values()):
            raise ValueError('Real target arrays are not zero')

    def batch(self, indices, device='cpu'):
        import torch
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self.records)).any():
            raise ValueError('Expected one-dimensional valid source indices')
        def convert(arrays):
            return {key: torch.from_numpy(np.array(value[indices], copy=True)).to(device)
                    for key, value in arrays.items()}
        inputs, targets = convert(self.inputs), convert(self.targets)
        inputs['image_gray'] = inputs['image_gray'].float().div_(255.)
        inputs['input_content_mask'] = content_mask(inputs['raw_to_input_affine'],
            inputs['raw_shape_hw'], inputs['input_shape_hw'])
        return inputs, targets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'check'), default='prepare')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.run_dir, workers=args.workers)
    else:
        data = LocalData(args.run_dir)
        print(json.dumps(dict(complete=True, PASS=True, n_frames=len(data.records),
            populations={k: len(v) for k, v in data.populations.items()}, P4_loaded=False)))
