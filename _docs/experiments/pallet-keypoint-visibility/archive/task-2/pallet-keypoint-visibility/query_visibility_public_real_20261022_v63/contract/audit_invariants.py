"""Read-only proof checks for evaluation exclusion, baseline preservation, labels.

Manifest mode verifies source RGB hashes against the 223-image denylist. JSON,
JSONL or CSV records must contain image_sha256 (or rgb_sha256), image/rgb_path,
and split. It reports what was checked; it cannot certify unseen execution.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
from label_contract import selfcheck

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as f:
        if '.csv' in path.suffixes:
            return list(csv.DictReader(f))
        if '.jsonl' in path.suffixes:
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ('records', 'frames', 'rows', 'images'):
        if key in data:
            return data[key]
    raise ValueError('Manifest needs a list or records/frames/rows/images list')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-manifest', action='append', default=[])
    p.add_argument('--source-cache', action='append', default=[])
    p.add_argument('--verify-source-image-files', action='store_true')
    p.add_argument('--output', default='INVARIANT_AUDIT.json')
    args = p.parse_args()
    out = HERE / args.output
    assert out.resolve().is_relative_to(HERE), 'Output must stay in contract directory'
    deny = set(json.loads((HERE / 'EVALUATION_SHA256_DENYLIST.json').read_text())['image_sha256'])
    bindings = json.loads((HERE / 'BASELINE_IMMUTABILITY_BINDINGS.json').read_text())
    checked = []
    for r in bindings:
        actual = sha(r['path'])
        assert actual == r['sha256'], ('Baseline modified', r['path'])
        checked.append({'path': r['path'], 'sha256': actual})
    a = records(ROOT / 'evaluation_preparation_20261022' / 'PREPARED_QUERY_MANIFEST.csv')
    assert len(a) == 1784 and len({r['id'] for r in a}) == 223
    assert len({(r['id'], r['native_corner']) for r in a}) == 1784
    assert {r['image_sha256'] for r in a} == deny
    assert Counter(r['human_state'] for r in a) == Counter(DIRECT_VISIBLE=1168, EXTERNAL_OCCLUDED=92, SELF_OCCLUDED=136, UNANNOTATED=356, UNKNOWN_MAPPING=8, OUT_OF_FRAME=24)
    assert {r['id'] for r in a if r['human_state'] == 'UNKNOWN_MAPPING'} == {'SQUARE_YOLO:026048'}
    manifests = []
    split_images = {}
    for path in args.source_manifest:
        rr = records(path)
        checked_images = {}
        for r in rr:
            s = r.get('image_sha256') or r.get('rgb_sha256')
            im = r.get('image') or r.get('rgb_path') or r.get('image_path')
            split = r.get('split') or r.get('partition')
            assert s and len(s) == 64, ('Missing RGB SHA256', path)
            assert s not in deny, ('Current eval image used in source', path, r.get('id'))
            assert split and str(split).lower() not in {'eval', 'evaluation', 'test', 'current_eval', 'private_eval'}, ('Non-source split', split)
            if args.verify_source_image_files:
                assert im and sha(im) == s, ('Source image SHA mismatch', im)
            # Pixel-query rows can repeat one image; cross-split image reuse cannot.
            if s in split_images:
                assert split_images[s] == split, ('Source image appears in two splits', s)
            split_images[s] = split
            checked_images[s] = split
        manifests.append({'path': str(Path(path).resolve()), 'sha256': sha(path), 'records': len(rr), 'unique_images': len(checked_images), 'eval_image_sha_overlap': 0, 'split_images': dict(Counter(checked_images.values()))})
    caches = []
    cache_split_ids = {}
    for path in args.source_cache:
        import numpy as np
        with np.load(path, allow_pickle=False) as z:
            hashes = z['image_sha256'].reshape(-1).astype(str)
            ids = z['id'].reshape(-1).astype(str) if 'id' in z.files else z['sample_id'].reshape(-1).astype(str)
            sources = z['source'].reshape(-1).astype(str)
            y = z['y'].reshape(-1)
            assert len(hashes) == len(y) and set(np.unique(y)) <= {0, 1}
            assert not (set(hashes) & deny), ('Current evaluation image in source cache', path)
            assert all(len(h) == 64 for h in hashes)
            label_checks = {}
            if 'dense_y' in z.files:
                d = z['dense_y'].reshape(-1, 9, 9)
                assert np.array_equal(d[:, 4, 4], y), ('Center target mismatch', path)
                label_checks['dense_center_exact_scalar_match'] = True
            for h in set(hashes):
                assert h not in cache_split_ids or cache_split_ids[h] == str(path), ('Source cache image overlap', path, cache_split_ids.get(h))
                cache_split_ids[h] = str(path)
            caches.append({'path': str(Path(path).resolve()), 'sha256': sha(path), 'queries': len(y), 'unique_images': len(set(hashes)), 'frame_ids': len(set(ids)), 'positive': int(y.sum()), 'sources': sorted(set(sources)), 'eval_image_sha_overlap': 0, **label_checks})
    result = {'schema': 'V63_INVARIANT_AUDIT_V1', 'label_selfcheck': selfcheck(), 'baseline_bindings_verified': checked,
              'current_eval_frames': 223, 'current_eval_slots': 1784, 'current_eval_sha_denylist': len(deny),
              'human_state_counts': dict(Counter(r['human_state'] for r in a)), 'source_manifests_checked': manifests, 'source_caches_checked': caches,
              'source_image_bytes_rehashed': args.verify_source_image_files, 'PnP_code_imported_or_executed_by_audit': False,
              'scope_limit': 'Checks registered files and supplied manifests only; does not prove behavior of uninspected training/inference code.'}
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'baseline_bindings_verified': len(checked), 'source_manifests_verified': len(manifests), 'out': str(out)}))


if __name__ == '__main__':
    main()
