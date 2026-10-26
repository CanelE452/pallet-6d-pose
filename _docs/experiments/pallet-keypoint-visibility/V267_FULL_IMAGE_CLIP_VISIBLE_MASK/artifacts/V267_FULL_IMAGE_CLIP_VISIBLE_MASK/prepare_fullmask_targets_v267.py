"""Original SYN M4 full-frame targets. No q/crop/M0/model/current inputs."""
from pathlib import Path
import hashlib
import io
import json
import time
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).absolute().parent


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def binding(path):
    p = safe(path)
    raw = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}


def checked(bound):
    assert binding(bound['path']) == bound
    return safe(bound['path'])


def write_json(path, data):
    p = safe(path)
    assert not p.exists()
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')


def main():
    started = time.time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = safe(HERE/'fullmask_targets_v1')
    assert not output.exists()
    complete_path = safe(HERE/'data_audit/COMPLETE.json')
    assert binding(complete_path)['sha256'] == 'ef2a2e0ac722df7a2f9cfc84d7333726107df20b1b39feba0db7ce673da2d483'
    complete = json.loads(complete_path.read_text())
    rows_bound = complete['outputs']['VISIBLE_MASK_TARGETS.jsonl']
    rows = [json.loads(line) for line in checked(rows_bound).read_text().splitlines()]
    assert len(rows) == 1003 and [r['frame_index'] for r in rows] == list(range(1003))
    output.mkdir()
    write_json(output/'PREPARATION_BEFORE_MASK_ARRAYS.json', {'completed': True,
        'code': binding(Path(__file__).absolute()), 'source_complete': binding(complete_path),
        'source_rows': rows_bound, 'target': 'Original user SYN visible M4 only, uint8>127',
        'transform': 'Full original binary M4 ->torch bilinear512x512 align_corners=False antialias=True ->avg_pool2d16stride16 ->1x32x32 soft target',
        'fit_split': 'Unchanged TRAIN768 / VAL235', 'no_crop_or_query_or_M0_supervision': True,
        'fixed_training_draws': 'PCG64(17), uniform TRAIN indices with replacement, shape1000x8',
        'positive_weight': '.5 / FIT soft foreground frequency',
        'negative_weight': '.5 / (1-FIT soft foreground frequency)',
        'model_forwards_or_optimizer_updates': 0, 'real_reads': 0})
    masks = np.zeros((1003, 1, 32, 32), np.float16)
    fit = np.asarray([r['split'] == 'train' for r in rows], bool)
    assert fit.sum() == 768 and (~fit).sum() == 235
    mask_bindings = []
    for i, r in enumerate(rows):
        p = checked(r['M4'])
        with Image.open(io.BytesIO(p.read_bytes())) as im:
            raw = np.array(im)
        assert raw.dtype == np.uint8 and int((raw > 127).sum()) == r['M4_area']
        x = torch.from_numpy((raw > 127).astype(np.float32))[None, None]
        full512 = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False, antialias=True)
        reduced = F.avg_pool2d(full512, kernel_size=16, stride=16)
        assert reduced.shape == (1, 1, 32, 32) and torch.isfinite(reduced).all()
        assert float(reduced.min()) >= -1e-6 and float(reduced.max()) <= 1+5e-5
        masks[i] = reduced.clamp(0, 1).numpy()[0].astype(np.float16)
        mask_bindings.append(r['M4'])
    ids = np.arange(1003, dtype=np.int32)
    np.savez_compressed(safe(output/'MASK_TARGETS.npz'), mask=masks, fit=fit, frame_ids=ids)
    draws = np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit), size=(1000, 8), replace=True).astype(np.int64)
    assert fit[draws].all()
    np.save(safe(output/'FIT_DRAWS1000x8.npy'), draws)
    freq = float(masks[fit].astype(np.float64).mean())
    assert 0 < freq < 1
    weights = [.5/freq, .5/(1-freq)]
    write_json(output/'M4_SOURCE_BINDINGS.json', {'M4': mask_bindings, 'M0_reads': 0, 'RGB_reads': 0})
    names = ['PREPARATION_BEFORE_MASK_ARRAYS.json', 'MASK_TARGETS.npz', 'FIT_DRAWS1000x8.npy', 'M4_SOURCE_BINDINGS.json']
    result = {'completed': True, 'PASS': True, 'schema': 'V267_FULL_FRAME_M4_SOFT_TARGETS32_V1',
        'code': binding(Path(__file__).absolute()), 'source_complete': binding(complete_path),
        'outputs': {name: binding(output/name) for name in names},
        'arrays': {'mask': {'shape': [1003,1,32,32], 'dtype': 'float16'}, 'fit': {'shape': [1003], 'dtype': 'bool'},
                   'frame_ids': {'shape': [1003], 'dtype': 'int32'}},
        'FIT_DEV': [768,235], 'FIT_foreground_frequency': freq,
        'positive_weight': weights[0], 'negative_weight': weights[1],
        'mask_range': [float(masks.min()), float(masks.max())],
        'foreground_frequency_FIT_DEV': [freq, float(masks[~fit].astype(np.float64).mean())],
        'raw_masks_decoded': 1003, 'M0_reads': 0, 'RGB_reads': 0, 'query_or_box_reads_for_transform': 0,
        'current_or_real_reads': 0, 'model_forwards': 0, 'optimizer_updates': 0,
        'elapsed_seconds': time.time()-started, 'runtime': {'torch': torch.__version__, 'numpy': np.__version__}}
    write_json(output/'COMPLETE.json', result)
    print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(output/'COMPLETE.json'),
                     'FIT_foreground_frequency': freq, 'positive_weight': weights[0], 'negative_weight': weights[1]}), flush=True)


if __name__ == '__main__':
    main()
