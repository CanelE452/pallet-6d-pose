"""Assemble fixed same-query SOURCE FIT counterfactuals from an exact entry."""
import argparse
import collections
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import types
import traceback

import numpy as np
from PIL import Image


def safe(path):
    text = str(path)
    for word in ('v112', 'current223', 'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', '__pycache__'):
        if word in text.lower():
            raise ValueError('Forbidden path before FS: ' + word)
    p = Path(path)
    for q in (p, *p.parents):
        if q.is_symlink():
            raise ValueError('Symlink')
    return p


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def bound(binding):
    p = safe(binding['path'])
    assert sha(p) == binding['sha256'], str(p)
    return p


def bind(p):
    return dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size)


def clean(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write(p, obj):
    p.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run(entry_path):
    e = json.loads(safe(entry_path).read_text())
    assert e['approved_exact_FIT_donor_pixels_and_pair_build'] is True
    assert sha(__file__) == e['code']['sha256']
    protocol = json.loads(bound(e['protocol']).read_text())
    assert protocol['version'] == 'V235'
    module = types.ModuleType('v235_pinned_compositor')
    module.__file__ = str(bound(e['compositor']))
    exec(compile(bound(e['compositor']).read_bytes(), module.__file__, 'exec'), module.__dict__)
    donor_manifest = json.loads(bound(e['donor_manifest']).read_text())
    assert donor_manifest['protocol']['sha256'] == e['protocol']['sha256']
    declared = {(b['path'], b['sha256']) for b in donor_manifest['exact_future_PNG_whitelist']}
    admitted = {(b['path'], b['sha256']) for b in e['exact_PNG_whitelist']}
    assert declared == admitted and len(admitted) == 125
    out = safe(e['output_directory'])
    out.mkdir(exist_ok=False)
    write(out / 'STARTED.json', dict(entry=bind(Path(entry_path)), code=bind(Path(__file__)), new_fit=0, automatic_retry=False))
    try:
        # Exact declared public FIT PNGs only; original depth/scene GT is never followed.
        encoded = {b['path']: bound(b).read_bytes() for b in e['exact_PNG_whitelist']}
        donors = []
        for d in donor_manifest['donors']:
            assert d['internal_source_role'] == 'INTERNAL_SOURCE_FIT' and d['original_associated_r_state'] == 'D'
            assert d['scene_id'] in ('000000', '000001', '000002', '000003', '000004')
            rgb = np.array(Image.open(io.BytesIO(encoded[d['RGB']['path']])).convert('RGB'))
            mask = np.array(Image.open(io.BytesIO(encoded[d['visible_mask']['path']])))
            assert rgb.shape == (480, 640, 3) and mask.shape == (480, 640)
            assert set(np.unique(mask)).issubset({0, 255}) and (mask > 0).any()
            donors.append((d, rgb, mask > 0))
        del encoded
        data_bindings = e['source_bindings']
        for b in data_bindings.values():
            bound(b)
        with np.load(data_bindings['samples']['path'], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        raw = np.load(data_bindings['rgb96']['path'], mmap_mode='r', allow_pickle=False)
        valid = np.load(data_bindings['valid96']['path'], mmap_mode='r', allow_pickle=False)
        diag = {r['sample_index']: r for r in (json.loads(line) for line in Path(data_bindings['diagnostics']['path']).read_text().splitlines())}
        with Path(data_bindings['YCB_reference_diagnostic']['path']).open() as f:
            ycb = {r['row_id']: r for r in csv.DictReader(f)}
        parents = np.flatnonzero(data['fit'] & data['available'] & (data['target'] == 0)).astype(np.int64)
        assert len(parents) == 5406 and len(data['target']) == 7617
        assert np.count_nonzero(data['family'][parents] == 0) == 4565
        assert np.count_nonzero(data['family'][parents] == 1) == 841
        np.save(out / 'parent_indices.npy', parents, allow_pickle=False)
        images = np.lib.format.open_memmap(out / 'PAIR_RGB96.npy', mode='w+', dtype=np.uint8, shape=(len(parents), 2, 3, 96, 96))
        masks = np.lib.format.open_memmap(out / 'PAIR_ALPHA_PACKBITS.npy', mode='w+', dtype=np.uint8, shape=(len(parents), 2, 1152))
        success = np.zeros(len(parents), bool)
        counts = collections.Counter()
        with (out / 'PAIR_METADATA.jsonl').open('x') as log:
            for j, index in enumerate(parents):
                index = int(index)
                row = diag[index]
                assert row['fit'] and row['target'] == 0 and row['available']
                assert row['row_id'] == str(data['row_id'][index])
                family = int(data['family'][index])
                y = ycb[row['row_id']] if family == 1 else None
                obj = int(y['object_id']) if y else None
                ref = [float(y['r_x_GT_diagnostic_only']), float(y['r_y_GT_diagnostic_only'])] if y else row['reference_xy_GT_diagnostic_only']
                pool = [d for d in donors if obj is None or d[0]['object_id'] != obj]
                choice = int.from_bytes(hashlib.sha256(('V235:pick:' + row['row_id']).encode()).digest(), 'big') % len(pool)
                donor, drgb, dmask = pool[choice]
                base = np.array(raw[index], copy=True)
                assert hashlib.sha256(base.tobytes()).hexdigest() == row['pixel_sha256']
                original_valid = np.array(valid[index, 0], copy=True)
                result = module.compose_counterfactual_pair(base, original_valid, drgb, dmask,
                    np.asarray(row['query_xy'], np.float64), np.float64(row['L']), np.asarray(ref, np.float64), row['row_id'])
                success[j] = bool(result['success'])
                record = {k: v for k, v in result.items() if k not in ('COVER', 'SHAM', 'alpha_COVER', 'alpha_SHAM', 'resized_donor_RGB', 'resized_donor_mask')}
                record.update(pair_index=j, parent_sample_index=index, family=family,
                    base_original_state='D', original_role='FIT', base_object_id_sampling_only=obj,
                    donor_id=donor['donor_id'], donor_object_id_sampling_only=donor['object_id'],
                    donor_RGB=donor['RGB'], donor_mask=donor['visible_mask'],
                    original_valid_sha256=row['valid_sha256'], original_pixel_sha256=row['pixel_sha256'])
                if result['success']:
                    for k, role in enumerate(('COVER', 'SHAM')):
                        im, alpha = result[role], result['alpha_' + role]
                        assert im.dtype == np.uint8 and im.shape == (3, 96, 96)
                        assert alpha.dtype == bool and alpha.shape == (96, 96)
                        assert not alpha[~original_valid].any()
                        assert np.array_equal(im[:, ~alpha], base[:, ~alpha])
                        images[j, k] = im
                        masks[j, k] = np.packbits(alpha.reshape(-1))
                    record['pair_RGB_sha256_CHW'] = [hashlib.sha256(images[j, k].tobytes()).hexdigest() for k in (0, 1)]
                    record['pair_alpha_sha256_bool'] = [hashlib.sha256(result['alpha_' + role].tobytes()).hexdigest() for role in ('COVER', 'SHAM')]
                    counts[f'family{family}_success'] += 1
                else:
                    images[j] = 0
                    masks[j] = 0
                    counts[f'family{family}_failed:' + str(result.get('failure_reason'))] += 1
                log.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + '\n')
                if j == 0 or (j + 1) % 1000 == 0 or j + 1 == len(parents):
                    print(json.dumps({'pair_parents': j + 1, 'total': len(parents), 'success': int(success[:j+1].sum())}), flush=True)
        images.flush(); masks.flush()
        del images, masks
        np.save(out / 'success.npy', success, allow_pickle=False)
        write(out / 'SUMMARY.json', dict(PASS=True, original_rows_preserved=7617,
            pairs_attempted=len(parents), successful=int(success.sum()), counts=dict(counts),
            raw_donor_PNGs=len(admitted), donor_instances=len(donors), new_fit=0,
            original_q_L_native_condition_valid_unchanged=True, current_or_held_payload=0,
            role_order=['COVER_E', 'SHAM_D'], failed_pair_not_E=True))
        names = ('parent_indices.npy', 'PAIR_RGB96.npy', 'PAIR_ALPHA_PACKBITS.npy', 'PAIR_METADATA.jsonl', 'success.npy', 'SUMMARY.json')
        write(out / 'COMPLETE.json', dict(PASS=True, completed=True, entry=bind(Path(entry_path)),
            code=bind(Path(__file__)), compositor=e['compositor'], protocol=e['protocol'],
            donor_manifest=e['donor_manifest'], source_bindings=data_bindings,
            outputs={n: bind(out / n) for n in names}, current_or_held_payload=0, new_fit=0))
        print(json.dumps({'PASS': True, 'COMPLETE': bind(out / 'COMPLETE.json'), 'counts': dict(counts)}), flush=True)
    except BaseException:
        write(out / 'FAILED_EXECUTION.json', dict(PASS=False, traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--entry', required=True)
    run(p.parse_args().entry)
