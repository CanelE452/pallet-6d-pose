"""Root-authorized fixed4 SOURCE artifact replay; no semantic adjudication."""
import ast
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import stat
import sys
import traceback

import numpy as np
from PIL import Image, __version__ as PIL_VERSION

sys.dont_write_bytecode = True
R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
HERE = R / 'V246_FIXED4_ARTIFACT_REPLAY_QA_A'
COMPLETE = {'path': str(R / 'V246_FIXED_PUBLIC_TRAIN_CROPS_C/gallery_v1/COMPLETE.json'),
            'sha256': '334f5b68dc635171ddfb67a2a238b85aca2d904001645f74169154279dea5109'}


def guard(path):
    s = str(path)
    if any(t in s.lower() for t in ('v112', '__pycache__', '.pyc', 'current', 'private', 'held', 'holdout', 'oldval', 'source_val')):
        raise ValueError('Forbidden path before filesystem')
    p = Path(s)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Traversal forbidden')
    for a in reversed((p, *p.parents)):
        try:
            z = os.lstat(a)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(z.st_mode):
            raise ValueError('Symlink forbidden')
    return p


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(b):
    raw = guard(b['path']).read_bytes()
    assert sha(raw) == b['sha256'], 'Encoded SHA mismatch'
    for k in ('bytes', 'size_bytes'):
        if k in b:
            assert b[k] == len(raw)
    return raw


def read(b):
    return json.loads(checked(b))


def array_sha(a):
    return sha(np.ascontiguousarray(a).tobytes())


def main():
    assert not guard(HERE / 'REVIEW.json').exists(), 'Immutable single execution'
    c = read(COMPLETE)
    assert c['completed'] is True and c['PASS'] is True
    e = read(c['entry'])
    assert e['approved_fixed4_TRAIN_RGB_gallery'] is True
    assert c['entry']['sha256'] == '284f173811dbc670179645ea59035a5d76b2b09f216e831c04480d56481b3c9e'
    scope = read(e['root_scope'])
    assert scope['approved_fixed4_TRAIN_RGB_gallery'] is True
    selection = read(c['fixed_selection'])
    assert selection['pairs'] == ['1.92m_lab:0000_0001', '0.55m_lab:0016_0017']
    assert selection['only_SOURCE_TRAIN'] is True
    plan = read(c['plan'])
    cases = read(c['outputs']['ALL32_CASE_INPUTS.json'])
    assert cases['completed'] is True and len(cases['rows']) == 32
    train_complete = read(c['TRAIN_metadata_complete'])
    assert train_complete['completed'] is True and train_complete['PASS'] is True
    fq = [json.loads(x) for x in checked(e['TRAIN_frames']).splitlines() if x]
    qq = [json.loads(x) for x in checked(e['TRAIN_queries']).splitlines() if x]
    assert len(fq) == 44 and len(qq) == 352
    qmap = {q['query_id']: q for q in qq}
    assert len(qmap) == 352
    frame_map = {f['frame_input_id']: f for f in fq}
    assert [x['frame'] for x in plan['cases']] == selection['frames']
    assert len(plan['cases']) == 4
    source = checked(c['sampler'])
    assert c['sampler']['sha256'] == '6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
    tree = ast.parse(source)
    crop = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'crop']
    assert len(crop) == 1
    namespace = {'np': np}
    exec(compile(ast.Module(body=crop, type_ignores=[]), c['sampler']['path'], 'exec'), namespace)
    identity = lambda b: (b['path'], b['sha256'])
    rgb_white = {identity(b) for b in e['RGB_whitelist']}
    assert rgb_white == {identity(b) for b in c['RGB_whitelist']} == {identity(b) for b in scope['RGB_whitelist']}
    assert len(rgb_white) == 4
    saved_by_id = {x['query_id']: x for x in cases['rows']}
    expected_ids = [q['query_id'] for case in plan['cases'] for q in case['queries']]
    assert len(set(expected_ids)) == 32 and [x['query_id'] for x in cases['rows']] == expected_ids
    proofs, frame_proofs = [], []
    for i, case in enumerate(plan['cases']):
        f = case['frame']
        assert f == frame_map[f['frame_input_id']] and f['split'] == 'TRAIN_LAB'
        raw_binding = {'path': f['image'], 'sha256': f['image_sha256']}
        assert identity(raw_binding) in rgb_white
        raw = checked(raw_binding)
        copied = checked(c['outputs'][f'FRAME_{i:02d}_SOURCE_RGB.png'])
        assert copied == raw, 'Original source PNG copy changed'
        image = Image.open(BytesIO(raw))
        assert image.format == 'PNG'
        rgb = np.asarray(image.convert('RGB'))
        assert rgb.dtype == np.uint8 and rgb.shape == (*f['raw_hw'], 3)
        frame_proofs.append({'frame_input_id': f['frame_input_id'], 'encoded_source_copy_byte_exact': True,
                             'encoded_sha256': sha(raw), 'decoded_RGB_HWC_sha256': array_sha(rgb), 'raw_hw': f['raw_hw']})
        assert [q['native_slot'] for q in case['queries']] == list(range(8))
        for q in case['queries']:
            assert q == qmap[q['query_id']], 'Original q row changed'
            saved = saved_by_id[q['query_id']]
            assert all(saved[k] == v for k, v in q.items()), 'Saved metadata differs from original input'
            assert saved['frame_position'] == i and saved['source_RGB_sha256'] == f['image_sha256']
            present = q['query_xy'] is not None and q['roi_side'] is not None
            if present:
                sampled, valid = namespace['crop'](rgb, np.asarray(q['query_xy'], np.float64), float(q['roi_side']), 96)
            else:
                sampled = np.zeros((3, 96, 96), np.uint8)
                valid = np.zeros((1, 96, 96), bool)
            assert sampled.dtype == np.uint8 and sampled.shape == (3, 96, 96)
            assert valid.dtype == np.bool_ and valid.shape == (1, 96, 96)
            name = f'FRAME_{i:02d}_N{q["native_slot"]}_CLEAN96.png'
            assert saved['clean96_path'] == name
            clean_image = Image.open(BytesIO(checked(c['outputs'][name])))
            assert clean_image.mode == 'RGB' and clean_image.size == (96, 96)
            clean = np.asarray(clean_image).transpose(2, 0, 1).copy()
            assert np.array_equal(clean, sampled), 'Exact V90 crop replay mismatch'
            assert array_sha(sampled) == saved['RGB96_CHW_sha256'] == array_sha(clean)
            assert array_sha(valid) == saved['valid96_bool_sha256'], 'Validity SHA mismatch'
            assert float(valid.mean()) == saved['valid_fraction']
            assert saved['input_present'] is present
            if not present:
                assert not clean.any() and not valid.any() and saved['absent_crop_is_blank'] is True
            assert saved['native_semantic_state'] == 'UNKNOWN' and saved['human_gold'] is False
            proofs.append({'query_id': q['query_id'], 'native_slot': q['native_slot'], 'q_L_flags_unchanged': True,
                           'crop_CHW_bytes_exact': True, 'valid_bool_SHA_exact': True,
                           'original_input_U': not q['original_query_continuous_valid'], 'absent': not present,
                           'RGB96_CHW_sha256': array_sha(sampled), 'valid96_bool_sha256': array_sha(valid)})
    assert len(proofs) == 32 and len(frame_proofs) == 4
    report = {'completed': True, 'PASS': True, 'scope': 'Actual fixed4/32 artifact replay QA using the pinned V90 crop function; not independent physical or semantic gold.',
              'gallery_complete': COMPLETE, 'entry': c['entry'], 'root_scope': e['root_scope'],
              'sampler': c['sampler'], 'fixed_selection': c['fixed_selection'], 'plan': c['plan'],
              'saved_cases': c['outputs']['ALL32_CASE_INPUTS.json'],
              'counts': {'frames': 4, 'fixed_original_queries': 32, 'encoded_source_copies_exact': 4,
                         'CLEAN96_decoded_CHW_byte_exact': 32, 'valid96_bool_SHA_exact': 32,
                         'actual_original_U': sum(x['original_input_U'] for x in proofs), 'actual_absent': sum(x['absent'] for x in proofs)},
              'frames': frame_proofs, 'rows': proofs,
              'limits': {'absent_handling_numeric_exercised': any(x['absent'] for x in proofs),
                         'valid_bitmap_saved_as_file': False, 'valid_bitmap_replay_compared_to_saved_hash': True,
                         'new_semantic_labels_or_instance_correspondence_adjudication': 0,
                         'GT_certificate_depth_mask_mesh_pose_arrays_models_current_held_reads': 0,
                         'other_source_RGB_reads': 0, 'actual_view_image_calls': 0,
                         'source_or_gallery_artifacts_modified': False},
              'runtime': {'numpy': np.__version__, 'Pillow': PIL_VERSION}, 'frozen_after_write': True}
    raw = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    target = guard(HERE / 'REVIEW.json'); target.write_text(raw)
    print(json.dumps({'completed': True, 'PASS': True, 'review': {'path': str(target), 'sha256': sha(raw.encode()), 'size_bytes': len(raw.encode())}, 'counts': report['counts']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        path = guard(HERE / 'FAILURE.txt')
        if not path.exists():
            path.write_text(traceback.format_exc())
        raise
