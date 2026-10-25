"""Portable tiny-head replay on saved V244 old-CF features. No DINO/images."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def safe(value):
    p = Path(value).absolute()
    assert '..' not in p.parts
    assert not any(s in str(p).casefold() for s in ('v112', '__pycache__', '.pyc'))
    for a in reversed((p, *p.parents)):
        assert not a.is_symlink()
    return p


def checked(path, digest):
    p = safe(path)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    assert h.hexdigest() == digest
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-json')
    args = parser.parse_args()
    directory = safe(args.saved_dir)
    features = np.load(checked(directory / 'SUCCESSFUL_PAIR_FEATURES.npy',
        '445f565731bda748db16b64a8d6fdd415b52c901aae8d1b6e9214dca54ada554'), mmap_mode='r', allow_pickle=False)
    attempts = np.load(checked(directory / 'SUCCESSFUL_ATTEMPT_INDICES.npy',
        '38118633fe3f16a5306e0cd764fbefe781b5978f6afb80ee2f5bfb958b583fc8'), allow_pickle=False)
    rows_path = checked(directory / 'ALL5406_OLD_PAIR_BEHAVIOR.csv',
        'ed3bef4f0d4b526b62f171a3b6b3820fd9c4b5ec71c6844aa7eb1d18b8f3e19c')
    checkpoint = checked(args.checkpoint, 'e5cdd7f4f43c01218fa87a7be787c7a8b290426fc70a3a07470f8deea8e822e3')
    with rows_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert features.shape == (5367, 2, 1186) and features.dtype == np.float32
    assert attempts.shape == (5367,) and attempts.dtype == np.int64
    assert len(rows) == 5406
    assert attempts.tolist() == [i for i, r in enumerate(rows) if r['success'] == 'True']
    expected = np.asarray([[[float(rows[i][role + '_logit' + str(f)]) for f in (0, 1)]
                             for role in ('COVER', 'SHAM')] for i in attempts], np.float32)
    torch.set_num_threads(1)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert cp['arm'] == 'TREAT_CF_DATA_ONLY' and cp['updates'] == 1000
    assert cp['head_parameters'] == 76098 and cp['adapter_parameters'] == 0 and cp['use_lora'] is False
    # Exact CropFamilyHead forward: Linear1186->64, ReLU, Linear64->2.
    head = torch.nn.Sequential(torch.nn.Linear(1186, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    state = cp['head_state_dict']
    assert set(state) == {'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias'}
    head.load_state_dict({k.removeprefix('head.'): v for k, v in state.items()}, strict=True)
    head.eval().requires_grad_(False)
    flat = features.reshape(-1, 1186)
    actual = np.empty((len(flat), 2), np.float32)
    with torch.no_grad():
        for start in range(0, len(flat), 32):
            x = torch.from_numpy(np.array(flat[start:start+32], copy=True))
            assert np.isfinite(x.numpy()).all()
            actual[start:start+len(x)] = head(x).numpy()
    actual = actual.reshape(5367, 2, 2)
    # Prospective CPU/GPU FP32 tolerance, same as the existing saved-head replay.
    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5), float(np.max(np.abs(actual-expected)))
    family = np.asarray([int(rows[i]['family']) for i in attempts])
    values = actual[np.arange(5367)[:, None], np.arange(2)[None, :], family[:, None]]
    signs = np.where(values >= 0, 'E', 'D')
    expected_signs = [[rows[i][r + '_prediction'] for r in ('COVER', 'SHAM')] for i in attempts]
    assert signs.tolist() == expected_signs
    result = {'completed': True, 'PASS': True, 'both_family_logits_checked': 21468,
              'role_predictions_exact': 10734, 'CPU_GPU_max_abs': float(np.max(np.abs(actual-expected))),
              'atol': 1e-5, 'rtol': 1e-5, 'new_training_or_DINO_or_original_pixels': 0,
              'scope': 'Saved SOURCE FIT CF feature and frozen tiny-head replay only'}
    if args.output_json:
        with safe(args.output_json).open('x') as f:
            json.dump(result, f, indent=2)
            f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
