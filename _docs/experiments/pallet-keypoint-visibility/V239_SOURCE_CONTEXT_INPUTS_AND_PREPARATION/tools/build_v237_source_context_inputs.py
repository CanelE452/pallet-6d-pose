"""SOURCE-only original q8 RGB input reconstruction; explicit root entry required.

No models, GT image members, new counterfactuals, or query modifications.
"""
import argparse
import ast
from collections import OrderedDict, Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import traceback
import numpy as np


def safe(value):
    value = str(value)
    if any(x in value.casefold() for x in ('v112', '__pycache__', '.pyc', 'current223',
            'publicheld', 'oldval', 'v177', 'v178', 'source_holdout', 'v182', 'private_evaluation')):
        raise ValueError('Lexically excluded path before filesystem access')
    p = Path(os.path.abspath(value))
    for part in reversed((p, *p.parents)):
        if part.is_symlink():
            raise ValueError('Symlink excluded')
    return p


def digest(b):
    return hashlib.sha256(b).hexdigest()


def read_bound(binding):
    p = safe(binding['path'])
    with p.open('rb') as f:
        b = f.read()
    if digest(b) != binding['sha256'] or ('size_bytes' in binding and binding['size_bytes'] is not None and len(b) != binding['size_bytes']):
        raise ValueError('Changed bound bytes: ' + str(p))
    return b


def binding(p):
    p = safe(p); h = hashlib.sha256(); size = 0
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b); size += len(b)
    return {'path': str(p), 'sha256': h.hexdigest(), 'size_bytes': size}


def write_json(p, d):
    with safe(p).open('x') as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def load_functions(b, names):
    ns = {'np': np}
    nodes = [n for n in ast.parse(read_bound(b)).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), b['path'] + '::pure_functions', 'exec'), ns)
    return ns


def run(entry_path):
    with safe(entry_path).open() as f:
        e = json.load(f)
    assert e['approved_actual_SOURCE_RGB_input_build'] is True
    assert e['training_allowed'] is False and e['model_forward_allowed'] is False
    assert e['new_counterfactual_pairs_allowed'] is False and e['current_or_held_allowed'] is False
    assert binding(__file__) == e['code']
    direct = json.loads(read_bound(e['direct_manifest']))
    manifest = json.loads(read_bound(e['raw_whitelist_manifest']))
    assert direct['completed'] and direct['PASS'] and manifest['completed'] and manifest['PASS']
    assert direct['original_manifest'] == e['raw_whitelist_manifest']
    groups = [json.loads(x) for x in read_bound(direct['groups']).splitlines() if x]
    rows = [json.loads(x) for x in read_bound(direct['all_rows']).splitlines() if x]
    diagnostics = [json.loads(x) for x in read_bound(e['old_known_diagnostics']).splitlines() if x]
    assert len(groups) == 1767 and len(rows) == 15040 and len(diagnostics) == 7617
    assert all(g['group_index'] == i for i, g in enumerate(groups))
    assert all(r['full_row_index'] == i for i, r in enumerate(rows))
    whitelist = {}
    for name, typ in [('SYN440_RAW_NPZ_MEMBER_ONLY', 'npz'), ('YCB700_RGB_ONLY', 'png')]:
        entries = manifest['whitelists'][name]['entries']
        assert len(entries) == (440 if typ == 'npz' else 700)
        for b in entries:
            safe(b['path'])
            if typ == 'npz':
                assert b['allowed_member'] == 'RGB_uint8'
            assert b['path'] not in whitelist
            whitelist[b['path']] = {**b, 'format': typ}
    assert len(whitelist) == 1140
    for g in groups:
        b = whitelist[g['image']['path']]
        assert b['sha256'] == g['image']['sha256'] and b['format'] == g['image']['format']
        if b['format'] == 'npz':
            assert g['image']['member'] == 'RGB_uint8'
    crop = load_functions(e['local_crop_CODE'], ['crop'])['crop']
    ctx = load_functions(e['context_CODE'], ['query_extent', 'valid_box', 'target_context_grid', 'sample_target_context'])
    out = safe(e['output'])
    assert not out.exists()
    out.mkdir()
    n = len(groups)
    specifications = [
        ('NODE_RGB96.npy', np.uint8, (n, 8, 3, 96, 96)),
        ('NODE_VALID96.npy', bool, (n, 8, 96, 96)),
        ('CONTEXT_RGB224_F32.npy', np.float32, (n, 3, 224, 224)),
        ('CONTEXT_VALID224.npy', bool, (n, 224, 224)),
    ]
    arrays = {name: np.lib.format.open_memmap(safe(out / name), mode='w+', dtype=dtype, shape=shape)
              for name, dtype, shape in specifications}
    node_available = np.zeros((n, 8), bool)
    node_fraction = np.zeros((n, 8), np.float32)
    context_fraction = np.zeros(n, np.float32)
    rows_by_group = [[] for _ in groups]
    group_roles = [set() for _ in groups]
    for row in rows:
        gi, slot = row['group_index'], row['active_slot']; g = groups[gi]
        assert 0 <= slot < 8 and row['group_id'] == g['group_id']
        assert row['q'] == g['q8'][slot] and row['L'] == g['L8'][slot]
        assert row['original_query_available'] == g['original_available8'][slot]
        group_roles[gi].add(row['fit']); rows_by_group[gi].append(row)
    assert all(len(roles) == 1 for roles in group_roles), 'Original FIT/DEV input overlap'
    known_verified = np.zeros(7617, bool)
    cache = OrderedDict(); accessed = {}; views = []
    for gi, g in enumerate(groups):
        image = g['image']; path = image['path']
        if path in cache:
            rgb = cache.pop(path); cache[path] = rgb
        else:
            b = whitelist[path]
            raw = read_bound(b)
            if image['format'] == 'npz':
                # Central directory is needed by NPZ, but no other array member is accessed.
                with np.load(io.BytesIO(raw), allow_pickle=False) as packet:
                    rgb = np.array(packet['RGB_uint8'], copy=True)
            else:
                from PIL import Image
                with Image.open(io.BytesIO(raw)) as im:
                    rgb = np.array(im.convert('RGB'), dtype=np.uint8, copy=True)
            assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            if g['raw_hw'] is not None:
                assert list(rgb.shape[:2]) == g['raw_hw']
            accessed[path] = {'path': path, 'sha256': b['sha256'], 'container_bytes': len(raw),
                              'decoded_shape': list(rgb.shape), 'RGB_sha256_HWC': digest(rgb.tobytes()),
                              'member_decoded': 'RGB_uint8' if image['format'] == 'npz' else 'PNG_RGB',
                              'other_NPZ_members_decoded': 0}
            cache[path] = rgb
            while len(cache) > 8:
                cache.popitem(last=False)
        q8 = np.asarray(g['q8'], np.float64)
        assert np.array_equal(np.asarray(ctx['query_extent'](q8)), np.asarray(g['box_xyxy']))
        for slot in range(8):
            q, L = q8[slot], g['L8'][slot]
            if np.isfinite(q).all() and math.isfinite(L) and L > 0:
                local, valid = crop(rgb, q, L, 96)
            else:
                local, valid = np.zeros((3, 96, 96), np.uint8), np.zeros((1, 96, 96), bool)
            arrays['NODE_RGB96.npy'][gi, slot] = local
            arrays['NODE_VALID96.npy'][gi, slot] = valid[0]
            node_fraction[gi, slot] = valid.mean()
            node_available[gi, slot] = bool(g['original_available8'][slot] and valid[:, 47:49, 47:49].all() and valid.any())
        context, valid_context = ctx['sample_target_context'](rgb, g['box_xyxy'])
        assert context.dtype == np.float32 and context.shape == (224, 224, 3)
        arrays['CONTEXT_RGB224_F32.npy'][gi] = context.transpose(2, 0, 1)
        arrays['CONTEXT_VALID224.npy'][gi] = valid_context
        context_fraction[gi] = valid_context.mean()
        for row in rows_by_group[gi]:
            slot = row['active_slot']
            assert bool(node_available[gi, slot]) == row['crop_available']
            si = row['sample_index']
            if si is not None:
                old = diagnostics[si]
                assert old['sample_index'] == si and old['row_id'] == row['row_id']
                assert old['query_xy'] == row['q'] and old['L'] == row['L']
                assert digest(arrays['NODE_RGB96.npy'][gi, slot].tobytes()) == old['pixel_sha256'], ('RGB96 replay mismatch', si, row['row_id'])
                assert digest(arrays['NODE_VALID96.npy'][gi, slot].tobytes()) == old['valid_sha256'], ('valid96 replay mismatch', si)
                assert not known_verified[si]
                known_verified[si] = True
        views.append({'group_index': gi, 'group_id': g['group_id'], 'raw_hw': list(rgb.shape[:2]),
                      'context_RGB_F32_CHW_sha256': digest(arrays['CONTEXT_RGB224_F32.npy'][gi].tobytes()),
                      'context_valid_sha256': digest(valid_context.tobytes()),
                      'node_RGB_CHW_sha256': [digest(arrays['NODE_RGB96.npy'][gi, k].tobytes()) for k in range(8)],
                      'node_valid_sha256': [digest(arrays['NODE_VALID96.npy'][gi, k].tobytes()) for k in range(8)]})
        if (gi + 1) % 100 == 0 or gi + 1 == n:
            print(json.dumps({'groups_complete': gi + 1, 'groups': n, 'known_crop_hashes_replayed': int(known_verified.sum())}), flush=True)
    assert known_verified.all() and len(accessed) == 1140
    for array in arrays.values():
        array.flush()
    np.savez(safe(out / 'INPUT_GEOMETRY.npz'), node_available=node_available, node_fraction=node_fraction,
             context_fraction=context_fraction, q8=np.asarray([g['q8'] for g in groups], np.float64),
             L8=np.asarray([g['L8'] for g in groups], np.float64),
             original_available8=np.asarray([g['original_available8'] for g in groups], bool),
             box_xyxy=np.asarray([g['box_xyxy'] for g in groups], np.float64))
    for name, records in [('ASSET_RGB_ACCESS.jsonl', list(accessed.values())), ('GROUP_PIXEL_BINDINGS.jsonl', views)]:
        with safe(out / name).open('x') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    products = [name for name, _, _ in specifications] + ['INPUT_GEOMETRY.npz', 'ASSET_RGB_ACCESS.jsonl', 'GROUP_PIXEL_BINDINGS.jsonl']
    result = {'completed': True, 'PASS': True, 'entry': binding(entry_path), 'code': binding(__file__),
              'groups': n, 'SYN_groups': 767, 'YCB_groups': 1000, 'original_rows_retained': 15040,
              'masked_target_minus1_retained': 7423, 'known7617_original_RGB96_and_valid_SHA_exact': int(known_verified.sum()),
              'all15040_original_input_and_crop_availability_exact': True, 'FIT_DEV_group_overlap': 0,
              'raw_assets_decoded': len(accessed), 'SYN_RGB_uint8_members_only': 440, 'YCB_PNG_RGB_only': 700,
              'other_NPZ_array_members_decoded': 0, 'reference_pose_ownership_or_target_in_forward_inputs': False,
              'new_V235_counterfactuals': 0, 'new_current_or_held_payload': 0, 'model_forward': 0, 'training': 0,
              'context_math': 'V107 F64 bilinear coords -> F32 RGB and all-positive-tap validity; original finite q8 minmax without availability masking.',
              'node_math': 'Original V90 crop uint8 half-up and invalid-zero; q/L unchanged.',
              'array_layout': {name: {'shape': list(shape), 'dtype': str(np.dtype(dtype))} for name, dtype, shape in specifications},
              'outputs': [binding(out / name) for name in products]}
    write_json(out / 'COMPLETE.json', result)
    print(json.dumps({'completed': True, 'PASS': True, 'known_hash_parity': 7617, 'complete': binding(out / 'COMPLETE.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--entry', required=True)
    args = parser.parse_args()
    try:
        run(args.entry)
    except Exception:
        traceback.print_exc()
        raise
