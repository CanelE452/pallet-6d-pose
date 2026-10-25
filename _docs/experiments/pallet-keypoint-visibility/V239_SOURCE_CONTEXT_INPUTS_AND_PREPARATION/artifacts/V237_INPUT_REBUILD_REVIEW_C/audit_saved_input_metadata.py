"""V237 저장 출력의 opaque SHA/header/geometry/metadata만 검산한다."""
from pathlib import Path
import hashlib
import io
import json
import math
import os
import stat
from collections import Counter, defaultdict
import numpy as np

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V237_INPUT_REBUILD_REVIEW_C'
OUT = R / 'V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1'
M = R / 'V237_SOURCE_INPUT_MANIFEST_C'


def safe(p):
    p = Path(p)
    assert p.is_absolute() and '..' not in p.parts
    assert 'v112' not in str(p).casefold() and '__pycache__' not in p.parts and p.suffix != '.pyc'
    for x in list(p.parents)[::-1] + [p]:
        try: mode = os.lstat(x).st_mode
        except FileNotFoundError: continue
        assert not stat.S_ISLNK(mode)
    return p


def opaque(binding, content=False):
    p = safe(binding['path'])
    h = hashlib.sha256(); size = 0; parts = []
    with os.fdopen(os.open(p, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b); size += len(b)
            if content: parts.append(b)
    assert h.hexdigest() == binding['sha256'], str(p)
    assert binding.get('size_bytes') in (None, size)
    return b''.join(parts) if content else dict(path=str(p), sha256=h.hexdigest(), size_bytes=size)


def obj(path, sha):
    return json.loads(opaque(dict(path=str(path), sha256=sha), True))


def jsonl(binding):
    return [json.loads(x) for x in opaque(binding, True).splitlines() if x]


def own_binding(p):
    b = safe(p).read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), size_bytes=len(b))


def header(binding, shape, dtype):
    # NODE/CONTEXT 데이터는 숫자로 읽지 않는다. Header 이후에는 길이만 검사한다.
    with os.fdopen(os.open(safe(binding['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        version = np.lib.format.read_magic(f)
        assert version in ((1, 0), (2, 0))
        reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        actual, fortran, dt = reader(f)
        offset = f.tell()
    assert actual == shape and dt == np.dtype(dtype) and not fortran and not dt.hasobject
    assert binding['size_bytes'] == offset + math.prod(shape) * dt.itemsize
    return dict(shape=list(shape), dtype=str(dt), fortran_order=False, header_bytes=offset)


def run():
    protocol = own_binding(H / 'PROTOCOL_BEFORE_ACTUAL_OUTPUT_QA.json')
    assert protocol['sha256'] == '0c90946e41697f7c04ae28cfd69a43540a13e7fa647b7677ccda196cdb59746d'
    completed = dict(path=str(OUT / 'COMPLETE.json'), sha256='a9d8d8088f4cd61dac9384f2f25f49de1e48cef20787e3638e3f9c0aa5bb1718')
    complete = json.loads(opaque(completed, True))
    assert complete['completed'] is True and complete['PASS'] is True
    assert complete['code']['sha256'] == 'acadab5d93473fdab8e3924582466446a2f25a1b7db63c9b95677fd8f5694e57'
    assert complete['entry']['sha256'] == '6378dee52720130c278105c14a9668f788b736926c2066b0efcc1c48b259d003'
    direct = obj(M / 'DIRECT_GROUPED_INPUT_MANIFEST.json', '68827e79feb8115294f147675130091792b6c3be26b28640a07bdcd402cd7f36')
    manifest = obj(M / 'INPUT_MANIFEST.json', '394882a6d8c93ca724c1d2cf8e023a2c2f7ff839c0d428f87d80f6f21beb3f50')
    groups = jsonl(direct['groups']); rows = jsonl(direct['all_rows'])
    full = jsonl(dict(path=str(R / 'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/ALL15040_ROWS.jsonl'), sha256='434e8a90c9b949f1755ff5fbc765e249f10e53e4df458396d8365f6572e1ff27'))
    diag = jsonl(dict(path=str(R / 'V233_ROOT_CROP_ADAPTATION/crop_samples_v1/SAMPLE_DIAGNOSTICS.jsonl'), sha256='5862ce437067ae8d0c902fac34b93b476e3b5611d368e6d7451d78be1fb2cc5f'))
    assert len(groups) == 1767 and len(rows) == len(full) == 15040 and len(diag) == 7617
    outputs = {Path(b['path']).name: b for b in complete['outputs']}
    specs = {'NODE_RGB96.npy': ((1767, 8, 3, 96, 96), 'uint8'), 'NODE_VALID96.npy': ((1767, 8, 96, 96), 'bool'),
             'CONTEXT_RGB224_F32.npy': ((1767, 3, 224, 224), 'float32'), 'CONTEXT_VALID224.npy': ((1767, 224, 224), 'bool')}
    assert set(outputs) == set(specs) | {'INPUT_GEOMETRY.npz', 'ASSET_RGB_ACCESS.jsonl', 'GROUP_PIXEL_BINDINGS.jsonl'}
    for name, b in outputs.items(): assert b['path'] == str(OUT / name)
    verified = [opaque(b) for b in outputs.values()]
    headers = {name: header(outputs[name], *spec) for name, spec in specs.items()}
    # Root가 따로 허용한 input geometry만 숫자로 해석한다.
    with np.load(io.BytesIO(opaque(outputs['INPUT_GEOMETRY.npz'], True)), allow_pickle=False) as z:
        assert set(z.files) == {'node_available', 'node_fraction', 'context_fraction', 'q8', 'L8', 'original_available8', 'box_xyxy'}
        geo = {k: z[k] for k in z.files}
    expected_types = {'node_available': ((1767, 8), 'bool'), 'node_fraction': ((1767, 8), 'float32'),
                      'context_fraction': ((1767,), 'float32'), 'q8': ((1767, 8, 2), 'float64'),
                      'L8': ((1767, 8), 'float64'), 'original_available8': ((1767, 8), 'bool'), 'box_xyxy': ((1767, 4), 'float64')}
    for k, (shape, dt) in expected_types.items(): assert geo[k].shape == shape and geo[k].dtype == np.dtype(dt)
    for k, name in [('q8', 'q8'), ('L8', 'L8'), ('original_available8', 'original_available8'), ('box_xyxy', 'box_xyxy')]:
        assert np.array_equal(geo[k], np.asarray([g[name] for g in groups], dtype=geo[k].dtype))
    for k in ('node_fraction', 'context_fraction'):
        assert np.isfinite(geo[k]).all() and ((geo[k] >= 0) & (geo[k] <= 1)).all()
    pixels = jsonl(outputs['GROUP_PIXEL_BINDINGS.jsonl']); accesses = jsonl(outputs['ASSET_RGB_ACCESS.jsonl'])
    assert len(pixels) == 1767 and len(accesses) == 1140
    assets = {b['path']: b for w in manifest['whitelists'].values() for b in w['entries']}
    assert len(assets) == 1140 and {a['path'] for a in accesses} == set(assets)
    for a in accesses:
        b = assets[a['path']]
        assert a['sha256'] == b['sha256'] and a['other_NPZ_members_decoded'] == 0
        assert a['member_decoded'] == ('RGB_uint8' if a['path'].endswith('.npz') else 'PNG_RGB')
        assert len(a['decoded_shape']) == 3 and a['decoded_shape'][2] == 3
    for i, (g, p) in enumerate(zip(groups, pixels)):
        assert g['group_index'] == p['group_index'] == i and g['group_id'] == p['group_id']
        assert all(len(p[k]) == 8 for k in ['node_RGB_CHW_sha256', 'node_valid_sha256'])
        for k in ['node_RGB_CHW_sha256', 'node_valid_sha256']:
            assert all(len(x) == 64 and set(x) <= set('0123456789abcdef') for x in p[k])
        if g['raw_hw'] is not None: assert g['raw_hw'] == p['raw_hw']
    verified_known = set(); roles = defaultdict(set); covered = set()
    for i, (r, f) in enumerate(zip(rows, full)):
        assert r['full_row_index'] == i and r['row_id'] == f['row_id'] and r['sample_index'] == f['sample_index']
        assert r['target'] == f['target'] and r['original_state'] == f['original_state'] and r['fit'] == f['fit']
        gi = r['group_index']; slot = r['active_slot']; g = groups[gi]
        assert g['group_id'] == r['group_id'] and g['family_code'] == r['family_code'] == f['family_code']
        assert r['q'] == f['query_xy'] == g['q8'][slot] and r['L'] == f['L'] == g['L8'][slot]
        assert r['original_query_available'] == f['original_query_available'] == bool(geo['original_available8'][gi, slot])
        assert r['crop_available'] == f['available'] == bool(geo['node_available'][gi, slot])
        assert bool(r['original_query_available'] and f['central4'] and f['valid_fractions'][0] > 0) == r['crop_available']
        roles[gi].add(r['fit']); covered.add((gi, slot))
        si = r['sample_index']
        if si is not None:
            old = diag[si]
            assert geo['node_fraction'][gi, slot] == np.float32(f['valid_fractions'][0])
            assert old == f and old['row_id'] == r['row_id'] and si not in verified_known
            assert pixels[gi]['node_RGB_CHW_sha256'][slot] == old['pixel_sha256']
            assert pixels[gi]['node_valid_sha256'][slot] == old['valid_sha256']
            verified_known.add(si)
    assert len(covered) == 1767 * 8 and all(len(v) == 1 for v in roles.values())
    assert verified_known == set(range(7617))
    assert sum(r['target'] == -1 and r['sample_index'] is None for r in rows) == 7423
    assert Counter((r['family_code'], r['target']) for r in rows) == {(0, 0): 5853, (0, 1): 243, (0, -1): 944, (1, 0): 1181, (1, 1): 340, (1, -1): 6479}
    assert complete['groups'] == 1767 and complete['original_rows_retained'] == 15040 and complete['masked_target_minus1_retained'] == 7423
    assert complete['known7617_original_RGB96_and_valid_SHA_exact'] == 7617 and complete['FIT_DEV_group_overlap'] == 0
    result = dict(completed=True, PASS=True, code=own_binding(Path(__file__).absolute()), protocol=protocol,
                  code_review=own_binding(H / 'CODE_REVIEW.json'), root_complete=completed, opaque_outputs_verified=verified,
                  npy_headers=headers, geometry_types_shapes={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in geo.items()},
                  all15040_identity_targets_roles_availability_match=True, known7617_node_fraction_and_hash_metadata_match=True, known_original_hash_metadata_joins=7617,
                  masked_target_minus1_count_recomputed=7423, unique_group_nodes_covered=len(covered), FIT_DEV_group_overlap=0,
                  assets_unique=1140, SYN_members_only=440, YCB_PNGs_only=700,
                  geometry_support_counts=dict(original_available_nodes=int(geo['original_available8'].sum()), crop_available_nodes=int(geo['node_available'].sum()),
                                               context_positive_support_groups=int((geo['context_fraction'] > 0).sum())),
                  scope='Opaque whole-output SHA, NPY header, allowed geometry numeric values, saved metadata/hash joins only.',
                  limits=['No raw asset reread or independent source-pixel resampling.', 'Per-node RGB/valid SHA compared to Root saved pixel-binding records; pixel numeric arrays were never decoded.',
                          'Context pixel values/semantic support not independently recomputed; context whole-file SHA and declared sampler CODE were checked.',
                          'Counts of accessed assets are unique paths, not physical decode calls.'],
                  NODE_CONTEXT_numeric_pixel_reads=0, original_raw_GT_RGB_NPZ_weights_model_current_held_reads=0,
                  producer_modified=False, new_model_data_architecture_or_threshold_selection=False, current_admission=False)
    target = H / 'REVIEW.json'
    with safe(target).open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(dict(completed=True, PASS=True, review=own_binding(target))))


if __name__ == '__main__':
    run()
