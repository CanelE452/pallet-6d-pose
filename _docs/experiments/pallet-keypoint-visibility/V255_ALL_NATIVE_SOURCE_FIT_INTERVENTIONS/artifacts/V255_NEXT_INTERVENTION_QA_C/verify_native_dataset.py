"""V255 saved FIT dataset QA; original teacher definitions, no producer import."""
import argparse
import collections
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import stat
import struct
import sys
import traceback
import types
import numpy as np

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
H = R / 'V255_NEXT_INTERVENTION_QA_C'
ROOT_PROTOCOL = {'path': str(R/'V255_ROOT_NATIVE_COVERAGE/PROTOCOL_BEFORE_NATIVE_3D_RENDER.json'), 'sha256': '340ec90f3389255763a99402aad77d587a321eb352dab39e1f4fef1b171f05c1', 'size_bytes': 8330}
STAGE1_QA = {'path': str(H/'actual_review_v1/REVIEW.json'), 'sha256': '6a96d2f2cc6b315ec52a56cc56da0fa7e49c1b58580104b1e0f65fecfbdee71f', 'size_bytes': 6085}
PRODUCER_CODE = {'path': str(R/'V255_NATIVE_BALANCED_PRODUCER_A/build_native_balanced_scenes.py'), 'sha256': '090f541d80b725f52e2969c43096f6d0c6a7e7f89ab3028bb93e62cd24b150a5', 'size_bytes': 29308}
PRODUCER_ENTRY = {'path': str(R/'V255_ROOT_NATIVE_COVERAGE/ACTUAL_NATIVE_3D_RENDER_ENTRY.json'), 'sha256': '47e7e9d47291148a2bcc424fa65e62b44be4b26bbd1c9d243b6055f802d4bc07', 'size_bytes': 2696}
ROLES = ('COVER', 'NEAR_SHAM', 'FAR_SHAM')
MESH_KEYS = ('vertices', 'triangles', 'part_owner', 'face_axis', 'face_side', 'native_anchor3D', 'native_primary_part_id', 'native_incident_part_ids', 'native_patch_quads', 'native_edge_samples', 'native_witness_points')


def safe(path, missing=False):
    p = Path(path)
    assert p.is_absolute() and '..' not in p.parts
    assert not any(t in str(p).casefold() for t in ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'oldval', 'holdout', 'private_evaluation'))
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            assert missing
            continue
        assert not stat.S_ISLNK(mode), str(a)
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def identity(b):
    return {k: b[k] for k in ('path', 'sha256', 'size_bytes')}


def read(binding):
    with os.fdopen(os.open(safe(binding['path']), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        b = f.read()
    assert len(b) == binding['size_bytes'] and sha(b) == binding['sha256']
    return b


def js(binding):
    return json.loads(read(binding))


def fp(x):
    if x is None:
        return b'N'
    if isinstance(x, bool):
        return b'B' + bytes([x])
    if isinstance(x, list):
        return b'L' + struct.pack('<Q', len(x)) + b''.join(map(fp, x))
    assert isinstance(x, (int, float))
    return b'F' + struct.pack('<d', x)


def plain(x):
    return json.loads(json.dumps(x, allow_nan=False, default=lambda z: z.tolist() if isinstance(z, np.ndarray) else z.item()))


def put(path, obj):
    p = safe(path, missing=True)
    b = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    with os.fdopen(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), 'wb') as f:
        f.write(b)
    return {'path': str(p), 'sha256': sha(b), 'size_bytes': len(b)}


def region_from_saved(record):
    a, b = record['target_witness_states'], record['combined_witness_states']
    assert len(a) == len(b) == 37 and all(x in 'DESU' and len(x) == 1 for x in a+b)
    active = record['active_witness_indices']
    assert active == sorted(set(active)) and 0 in active and all(type(i) is int and 0 <= i < 37 for i in active)
    geometric_s = all(x == 'S' for x in a)
    before = 'S' if geometric_s else ('D' if len(record['front_face_axes']) >= 2 and all(a[i] == 'D' for i in active) else 'U')
    if not record['all_region_witnesses_in_frame']:
        state = 'U'
    elif before == 'S':
        state = 'S'
    elif before != 'D':
        state = 'U'
    else:
        state = 'E' if all(b[i] == 'E' for i in active) else ('D' if all(b[i] == 'D' for i in active) else 'U')
    assert record['geometric_S'] == geometric_s and record['target_alone_region_state'] == before and record['state'] == state
    assert record['target_counts'] == {s: a.count(s) for s in 'DSEU'}
    assert record['combined_counts'] == {s: b.count(s) for s in 'DSEU'}
    return a[0], b[0]


def modules_from_definition(protocol):
    """Use only the five original frozen definitions, never new producer CODE."""
    modules = {}
    sys.dont_write_bytecode = True
    for key, binding in protocol['code_bindings'].items():
        b = read(binding)
        m = types.ModuleType('v255_independent_definition_' + key)
        m.__file__, m.__package__ = binding['path'], None
        sys.modules[m.__name__] = m
        exec(compile(b, binding['path'], 'exec'), m.__dict__)
        modules[key] = m
    return modules


def run(entry_path):
    entry = js({'path': str(safe(entry_path)), **file_identity(entry_path)})
    assert entry.get('approved_saved_native_dataset_QA') is True
    assert set(entry) == {'approved_saved_native_dataset_QA', 'code', 'root_protocol', 'stage1_QA', 'producer_code', 'producer_entry', 'dataset_complete', 'output_directory'}
    assert entry['root_protocol'] == ROOT_PROTOCOL and entry['stage1_QA'] == STAGE1_QA
    assert entry['producer_code'] == PRODUCER_CODE and entry['producer_entry'] == PRODUCER_ENTRY
    own = {'path': str(Path(__file__).absolute()), **file_identity(__file__)}
    assert entry['code'] == own
    out = safe(entry['output_directory'], missing=True)
    assert out == H/'actual_native_review_v1' and not out.exists()
    out.mkdir()
    try:
        protocol = js(ROOT_PROTOCOL)
        assert sys.executable == protocol['runtime']['executable']
        assert platform.python_version() == protocol['runtime']['python'] and np.__version__ == protocol['runtime']['NumPy']
        q1 = js(STAGE1_QA)
        assert q1['completed'] is True and q1['PASS'] is True
        producer_entry = js(PRODUCER_ENTRY)
        assert producer_entry['approved_actual_native_balanced_render'] is True
        assert producer_entry['code'] == PRODUCER_CODE and producer_entry['root_protocol'] == ROOT_PROTOCOL
        assert producer_entry['actual_payload_read_allowlist'] == protocol['mesh_payload_bindings']
        producer_scope = js(producer_entry['root_scope'])
        assert producer_scope['approved_actual_native_balanced_render'] is True
        assert producer_scope['entry_contract'] == {k: v for k, v in producer_entry.items() if k != 'root_scope'}
        complete = js(entry['dataset_complete'])
        assert complete['completed'] is True and complete['PASS'] is True
        assert complete['root_protocol'] == ROOT_PROTOCOL and complete['stage1_QA'] == STAGE1_QA
        assert complete['code'] == entry['producer_code'] and complete['entry'] == entry['producer_entry']
        assert complete['root_scope'] == producer_entry['root_scope']
        assert complete['mesh_payload_bindings'] == protocol['mesh_payload_bindings'] and complete['code_bindings'] == protocol['code_bindings']
        root = Path(protocol['output_directory'])
        assert Path(entry['dataset_complete']['path']) == root/'COMPLETE.json'
        stage1 = js(protocol['source_fit_selection_complete'])
        configs = js(protocol['selected_configs_manifest'])['rows']
        parents = js(protocol['before_input_manifest'])['parents']
        parent = {p['base_index']: p for p in parents}
        assert len(parent) == len(parents) == len(configs) == stage1['census']['selected_original_FIT_base_count']
        assert q1['stage1_complete'] == protocol['source_fit_selection_complete']
        assert set(complete['outputs']) == {'SCENES.jsonl', 'ALL_NATIVE_TRIALS.jsonl', 'SUMMARY.json', 'MANIFEST.json'}
        for name, binding in complete['outputs'].items():
            assert Path(binding['path']) == root/name
        summary = js(complete['outputs']['SUMMARY.json'])
        manifest = js(complete['outputs']['MANIFEST.json'])
        files = {}
        for binding in manifest['files']:
            p = Path(binding['path'])
            assert p.is_relative_to(root) and p != root and binding['path'] not in files
            files[binding['path']] = identity(binding)
        def declared(binding):
            assert files[binding['path']] == identity(binding)
            return binding
        scenes = [json.loads(line) for line in read(declared(complete['outputs']['SCENES.jsonl'])).splitlines() if line]
        trials = [json.loads(line) for line in read(declared(complete['outputs']['ALL_NATIVE_TRIALS.jsonl'])).splitlines() if line]
        by_base = collections.defaultdict(list)
        trial_map = {}
        for i, s in enumerate(scenes):
            assert s['scene_index'] == i and s['source_role'] == 'FIT' and s['training_admission'] is False
            by_base[s['base_index']].append(s)
        for t in trials:
            key = (t['base_index'], t['native_slot'])
            assert key not in trial_map
            trial_map[key] = t
        assert set(by_base) == set(parent) and len(trial_map) == len(parent)*8
        base_bindings = {Path(b['path']).name: b for b in complete['base_receipts']}
        assert len(base_bindings) == len(complete['base_receipts']) == len(configs)
        expected_paths = {complete['outputs'][k]['path'] for k in ('SCENES.jsonl', 'ALL_NATIVE_TRIALS.jsonl', 'SUMMARY.json')}
        coverage = [collections.Counter() for _ in range(8)]
        ray_cases = {}
        all_point0 = collections.Counter()
        all_region = collections.Counter()
        unavailable = masked = pixel_count = 0
        for cfg in sorted(configs, key=lambda c: c['base_index']):
            base = cfg['base_index']
            p = parent[base]
            assert p['identity_PASS'] is True and p['source_base_role'] == 'FIT' and p['all_original_parent_rows'] == 8
            qmeta = copy.deepcopy(p['before_input'])
            qmeta.update(source_role='FIT', base_index=base, base_id=cfg['base_id'], scene_index=4*base, variant='BEFORE',
                decoded_RGB_sha256=p['before_input']['expected_BEFORE_decoded_RGB_sha256'],
                source_base_config={'file': copy.deepcopy(stage1['inputs']['configs192_opaque']), 'row_index': p['config_row_index']},
                selected_config_span=copy.deepcopy(p['selected_config_span']))
            assert p['config_row_index'] == base and cfg['split'] == 'TRAIN'
            rb = base_bindings[cfg['base_id']+'_COMPLETE.json']
            assert Path(rb['path']) == root/'bases'/(cfg['base_id']+'_COMPLETE.json')
            receipt = js(declared(rb))
            expected_paths.add(rb['path'])
            assert receipt['completed'] is True and receipt['base_index'] == base and receipt['original_parent'] == p
            assert receipt['scenes'] == by_base[base] and receipt['trials'] == [trial_map[(base, k)] for k in range(8)]
            assert receipt['BEFORE_pixel_SHA_exact'] is True and receipt['original_q8_L8_U_unchanged'] is True
            assert receipt['training_admission'] is False and receipt['new_label_fallback'] is False
            md, before_rgb = {}, None
            for s in by_base[base]:
                assert s['base_id'] == cfg['base_id']
                image = declared(s['image'])
                meta_binding = declared(s['metadata'])
                assert Path(image['path']).parent == root/'scenes' and Path(meta_binding['path']).parent == root/'scenes'
                assert Path(image['path']).stem == Path(meta_binding['path']).stem == s['scene_id']
                expected_paths.update((image['path'], meta_binding['path']))
                m = js(meta_binding)
                assert m['image'] == image and m['original_before_input'] == qmeta
                for key in ('q8', 'L8', 'original_available8', 'box_xyxy', 'raw_hw'):
                    assert fp(m['original_before_input'][key]) == fp(p['before_input'][key])
                assert m['teacher_records_are_model_input'] is False and m['training_admission'] is False
                assert m['literal_q_or_wholepart_or_real_native_gold'] is False
                assert m['RGB_sha256_HWC'] == s['RGB_sha256_HWC']
                assert image['format'] == 'npz' and image['member'] == 'RGB_uint8_HWC'
                with np.load(io.BytesIO(read(image)), allow_pickle=False) as z:
                    assert z.files == ['RGB_uint8_HWC']
                    rgb = z['RGB_uint8_HWC']
                    assert rgb.dtype == np.uint8 and list(rgb.shape) == m['RGB_shape'] == cfg['raw_hw']+[3]
                    assert m['RGB_dtype'] == 'uint8' and sha(np.ascontiguousarray(rgb).tobytes()) == m['RGB_sha256_HWC']
                if m['variant'] == 'BEFORE':
                    assert before_rgb is None and m['target_native_slot'] is None
                    before_rgb = rgb.copy()
                else:
                    assert before_rgb is not None
                    assert m['native_RGB_changed_pixels'] == int(np.any(rgb != before_rgb, axis=2).sum())
                pixel_count += 1
                cf = m['fresh_records_CF']
                assert len(cf) == 8 and m['permCF_to_canonical'] == cfg['permCF_to_canonical']
                point = [region_from_saved(rec) for rec in cf]
                for k, rec in enumerate(cf):
                    assert rec['CF_index'] == k and rec['canonical_index'] == cfg['permCF_to_canonical'][k]
                    assert rec['native_canonical_index'] == rec['canonical_index']
                    all_region[(m['variant'], rec['state'])] += 1
                    all_point0[(m['variant'], point[k][1])] += 1
                assert m['point0_target_alone_states8'] == [v[0] for v in point]
                assert m['point0_combined_states8'] == s['point0_combined_states8'] == [v[1] for v in point]
                key = (m['target_native_slot'], m['variant'])
                assert key not in md
                md[key] = m
            before = md[(None, 'BEFORE')]
            assert before['RGB_sha256_HWC'] == p['before_input']['expected_BEFORE_decoded_RGB_sha256']
            assert before['point0_target_alone_states8'] == before['point0_combined_states8']
            assert all(c['target_witness_states'] == c['combined_witness_states'] for c in before['fresh_records_CF'])
            for k in range(8):
                t = trial_map[(base, k)]
                alone = before['point0_target_alone_states8'][k]
                assert t['target_alone_point0_state'] == alone and t['canonical_index'] == cfg['permCF_to_canonical'][k]
                assert t['original_query_available'] == qmeta['original_available8'][k] and t['training_admission'] is False
                coverage[k]['trials'] += 1
                coverage[k]['target_alone_'+alone] += 1
                if alone != 'D':
                    assert t['constructed'] is False and t['reason'] == 'TARGET_ALONE_POINT0_NOT_D'
                    assert t['retained_geometric_state'] == alone and t['rendered_variants'] == [] and t['constructed_target'] is None
                    assert all((k, role) not in md for role in ROLES)
                    coverage[k]['not_constructed'] += 1
                    continue
                assert t['constructed'] is True and t['reason'] is None and t['rendered_variants'] == list(ROLES)
                actual = {}
                for role in ROLES:
                    m = md[(k, role)]
                    assert m['target_canonical_index'] == cfg['permCF_to_canonical'][k]
                    assert m['target_layer_digest'] == before['target_layer_digest']
                    assert m['point0_target_alone_states8'] == before['point0_target_alone_states8']
                    assert [c['target_witness_states'] for c in m['fresh_records_CF']] == [c['target_witness_states'] for c in before['fresh_records_CF']]
                    assert [c['target_alone_region_state'] for c in m['fresh_records_CF']] == [c['target_alone_region_state'] for c in before['fresh_records_CF']]
                    actual[role] = m['point0_combined_states8'][k]
                    assert m['actual_target_point0_state'] == actual[role]
                proof = actual == {'COVER': 'E', 'NEAR_SHAM': 'D', 'FAR_SHAM': 'D'}
                assert t['actual_target_point0_states'] == actual and t['observed_D_E_D_D'] == proof
                coverage[k]['constructed'] += 1
                coverage[k]['observed_D_E_D_D' if proof else 'proof_not_D_E_D_D'] += 1
                ray_cases.setdefault((cfg['template_index'], k), {'base_index': base, 'template_index': cfg['template_index'], 'native_slot': k,
                    'scene_metadata': [next(s['metadata'] for s in by_base[base] if s['target_native_slot'] == slot and s['variant'] == role) for slot, role in [(None, 'BEFORE')]+[(k, role) for role in ROLES]]})
            assert len(md) == 1+3*sum(trial_map[(base, k)]['constructed'] for k in range(8))
            unavailable += p['unavailable_native_members_retained']
            masked += p['masked_native_members_retained']
        assert expected_paths == set(files)
        expected_summary = {'completed': True, 'selected_original_FIT_bases': len(parent), 'all_native_trials': len(trial_map), 'scenes': len(scenes),
            'per_native': {str(k): dict(c) for k, c in enumerate(coverage)}, 'original_unavailable_native_members_retained': unavailable,
            'original_masked_native_members_retained': masked, 'all_BEFORE_pixel_SHA_exact': True, 'original_q8_L8_U_unchanged': True,
            'variant_name_not_used_as_label': True, 'training_admission': False, 'literal_q_wholepart_real_native_gold': False,
            'raw_original_RGB_GT_JSON_BASE192_reads': 0, 'new_encoder_model_CP_fit_PnP': 0, 'current_or_held_access': False}
        assert summary == complete['counts'] == expected_summary
        assert identity(complete['outputs']['SUMMARY.json']) == files[complete['outputs']['SUMMARY.json']['path']]
        # Predeclared min-base case per template/native; never filter by proof success.
        modules = modules_from_definition(protocol)
        templates = []
        for binding in protocol['mesh_payload_bindings']:
            with np.load(io.BytesIO(read(binding)), allow_pickle=False) as z:
                assert set(MESH_KEYS) <= set(z.files)
                templates.append({k: z[k].copy() for k in MESH_KEYS})
        cfg_by_base = {c['base_index']: c for c in configs}
        ray_rows = []
        for template_index in range(len(templates)):
            for k in range(8):
                case = ray_cases.get((template_index, k))
                if case is None:
                    ray_rows.append({'template_index': template_index, 'native_slot': k, 'base_index': None, 'scenes_recomputed': 0, 'status': 'NO_CONSTRUCTED_CASE_RETAINED'})
                    continue
                cfg = copy.deepcopy(cfg_by_base[case['base_index']])
                data = modules['core'].scaled_data(templates[template_index], cfg)
                camera = modules['renderer'].camera_arrays(cfg['camera'])
                perm = cfg['permCF_to_canonical']
                xy = modules['raster'].project_camera(modules['raster'].world_to_camera(data['native_anchor3D'], camera), camera)
                assert np.array_equal(xy[perm], np.asarray(cfg['native8_CF_xy']))
                cfg['selected_CF_index'], cfg['selected_canonical_index'] = k, perm[k]
                foreign, _, placement = modules['core'].foreign_placements(data, cfg)
                for role, binding in zip(('BEFORE',)+ROLES, case['scene_metadata']):
                    observed = js(declared(binding))
                    records = modules['teacher'].region_teacher(data, camera, modules['raster'], None if role == 'BEFORE' else foreign[role])
                    expected = modules['core'].CF_records(records, perm, data, cfg['camera'])
                    assert plain(expected) == observed['fresh_records_CF']
                    if role != 'BEFORE':
                        assert plain(placement) == observed['placement']
                ray_rows.append({'template_index': template_index, 'native_slot': k, 'base_index': case['base_index'], 'scenes_recomputed': 4, 'status': 'ALL8_WITNESS_AND_REGION_EXACT'})
        ray_binding = put(out/'RAY_SUBSET.json', {'selection_policy': 'Each template/native min original base_index with constructed target-alone-D triplet, regardless of observed proof; 4 scenes, no full renderer replay.', 'rows': ray_rows})
        review = {'completed': True, 'PASS': True, 'code': own, 'dataset_complete': entry['dataset_complete'], 'producer_entry': entry['producer_entry'],
            'producer_code': entry['producer_code'], 'root_protocol': ROOT_PROTOCOL, 'stage1_QA': STAGE1_QA, 'summary': expected_summary,
            'all_saved_RGB_only_NPZ_and_decoded_SHA_verified': pixel_count, 'all_explicit_manifest_bindings_verified': len(files),
            'all_original_nullable_typed_q8_L8_U_box_raw_hw_exact': True, 'all_fresh_teacher_metadata_region_counts_and_point0_checks': True,
            'all_scene_point0_state_counts': {role: {s: all_point0[(role, s)] for s in 'DESU'} for role in ('BEFORE',)+ROLES},
            'all_scene_region_state_counts': {role: {s: all_region[(role, s)] for s in 'DESU'} for role in ('BEFORE',)+ROLES},
            'ray_subset': ray_binding, 'ray_subset_scene_recomputations': sum(r['scenes_recomputed'] for r in ray_rows),
            'ray_definition_CODE_reused': protocol['code_bindings'], 'mesh_payload_bindings': protocol['mesh_payload_bindings'],
            'producer_import_or_call': False, 'full_RGB_rerender': False, 'new_training_or_feature_or_model_calls': 0,
            'old_original_RGB_GT_archives_BASE192_current_or_held_access': 0, 'training_admission': False, 'model_adopted': False,
            'proof_limit': 'All pixels/metadata checked; geometric teacher definition recomputed only on the predeclared subset. Not physical native/literal-q/wholepart gold or current-pallet performance.',
            'temporal_limit': 'Preflight CODE plus observed exact bindings; saved artifacts alone do not retrospectively prove process read ordering.'}
        rb = put(out/'REVIEW.json', review)
        print(json.dumps({'completed': True, 'PASS': True, 'review': rb, 'ray_subset': ray_binding, 'scenes': pixel_count, 'native_trials': len(trial_map)}, sort_keys=True))
    except BaseException as exc:
        put(out/'FAILURE.json', {'completed': False, 'PASS': False, 'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(), 'code': own, 'automatic_retry': False})
        raise


def file_identity(path):
    with os.fdopen(os.open(safe(path), os.O_RDONLY | os.O_NOFOLLOW), 'rb') as f:
        b = f.read()
    return {'sha256': sha(b), 'size_bytes': len(b)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True)
    run(parser.parse_args().entry)
