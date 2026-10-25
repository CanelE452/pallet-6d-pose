"""Replay only the small frozen SOURCE head using explicit saved artifacts.

No encoder, images, GT references, training, downloads or metadata pointers.
Run package integrity verification first. CPU/GPU FP32 tolerance is prospective.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import types
import numpy as np
import torch


def safe(p):
    p = Path(p)
    text = str(p).lower()
    assert not any(x in text for x in ('v112', '__pycache__', '.pyc', 'current223', 'publicheld', 'private_evaluation', 'oldval', 'v177', 'v178', 'source_holdout', 'v182'))
    assert not p.is_symlink()
    return p


def file_binding(p):
    p = safe(p); h = hashlib.sha256(); size = 0
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk); size += len(chunk)
    return {'path': str(p), 'sha256': h.hexdigest(), 'bytes': size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--features-dir', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--csv', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    module_bytes = safe(args.module).read_bytes()
    assert hashlib.sha256(module_bytes).hexdigest() == 'dfd26ab7ccfa7037f0c3dca548d59c8574aa8b78c50699f70f01e03bf48935a9'
    module = types.ModuleType('v238_replay_module')
    exec(compile(module_bytes, str(safe(args.module)), 'exec'), module.__dict__)
    checkpoint = torch.load(safe(args.checkpoint), weights_only=True, map_location='cpu')
    assert checkpoint['updates'] == 1000 and checkpoint['head_parameters'] == 86054
    assert checkpoint['residual_parameters'] == 9956 and checkpoint['use_lora'] is False
    model = module.RelationalQueryHead().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['head_state_dict'], strict=True)
    directory = safe(args.features_dir)
    names = ['ORIGINAL_ALL_FEATURES.npy', 'NODE_CLS.npy', 'CONTEXT_CLS.npy',
             'SAMPLE_GROUP_INDEX.npy', 'SAMPLE_NODE_INDEX.npy', 'RELATIVE_XY8.npy']
    arrays = {name: np.load(safe(directory/name), mmap_mode='r', allow_pickle=False) for name in names}
    x = arrays[names[0]]; node = arrays[names[1]]; context = arrays[names[2]]
    group = arrays[names[3]]; slot = arrays[names[4]]; relative = arrays[names[5]]
    assert x.shape == (7617, 1186) and x.dtype == np.float32
    assert node.shape == (1767, 8, 384) and node.dtype == np.float32
    assert context.shape == (1767, 384) and context.dtype == np.float32
    assert group.shape == slot.shape == (7617,) and group.dtype == slot.dtype == np.int64
    assert relative.shape == (7617, 8, 2) and relative.dtype == np.float32
    assert ((group >= 0) & (group < 1767)).all() and ((slot >= 0) & (slot < 8)).all()
    with np.load(safe(args.geometry), allow_pickle=False) as z:
        original = z['original_available8']; supported = z['node_available']; cf = z['context_fraction']
    assert original.shape == supported.shape == (1767, 8) and original.dtype == supported.dtype == bool
    assert cf.shape == (1767,) and cf.dtype == np.float32
    with safe(args.csv).open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 7617
    expected = np.asarray([[float(r['logit0']), float(r['logit1'])] for r in rows], np.float32)
    family = np.asarray([int(r['family']) for r in rows], np.int64)
    assert np.isin(family, [0, 1]).all()
    assert all(int(r['sample_index']) == i and r['available'] == 'True' for i, r in enumerate(rows))
    actual = np.zeros_like(expected); max_active_difference = 0.; attention_checks = 0
    tensor = lambda a: torch.from_numpy(np.array(a, copy=True))
    with torch.no_grad():
        for start in range(0, 7617, 64):
            ix = np.arange(start, min(start+64, 7617)); gi = group[ix]; si = slot[ix]
            xx, nn = tensor(x[ix]), tensor(node[gi])
            alignment = module.verify_active_node_alignment(xx, nn, tensor(si), tensor(original[gi]), tensor(supported[gi]), atol=1e-6)
            assert alignment['checked'] == len(ix)
            max_active_difference = max(max_active_difference, alignment['max_abs'])
            logits, detail = model(xx, nn, tensor(context[gi]), tensor(si), tensor(original[gi]), tensor(supported[gi]), tensor(cf[gi] > 0), tensor(relative[ix]))
            values = logits.numpy(); assert np.isfinite(values).all(); actual[ix] = values
            attention = detail['attention_other7'].numpy()
            other = original[gi] & supported[gi]; other[np.arange(len(ix)), si] = False
            assert not np.count_nonzero(attention[~other])
            assert np.allclose(attention.sum(1), other.any(1).astype(np.float32), rtol=0, atol=1e-6)
            attention_checks += len(ix)
    # Fixed before the first actual replay. Never relax tolerance after a result.
    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5), float(np.max(np.abs(actual-expected)))
    predicted = np.where(actual[np.arange(7617), family] >= 0, 'E', 'D')
    assert predicted.tolist() == [r['prediction'] for r in rows]
    files = [args.module, args.checkpoint, args.geometry, args.csv] + [directory/n for n in names]
    print(json.dumps({'completed': True, 'PASS': True, 'rows': 7617, 'both_logits_checked': 15234,
        'CPU_GPU_FP32_max_abs': float(np.max(np.abs(actual-expected))), 'atol': 1e-5, 'rtol': 1e-5,
        'predictions_exact': 7617, 'active_node_QA_rows': 7617, 'active_node_max_abs': max_active_difference,
        'masked_attention_rows_checked': attention_checks, 'files': [file_binding(p) for p in files],
        'limits': 'Saved frozen feature/head replay only. Does not recompute image features, raw annotations or training. No new labels/model/threshold selection, encoder, images, current/held, downloads or pointer following.'}))


if __name__ == '__main__':
    main()
