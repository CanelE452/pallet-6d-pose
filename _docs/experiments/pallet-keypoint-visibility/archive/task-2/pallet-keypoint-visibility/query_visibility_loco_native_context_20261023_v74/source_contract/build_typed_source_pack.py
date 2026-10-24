"""Coordinates/labels-only source census. No RGB/features/weights/solver import.

All60000 existing source rows are exported, without target/class/error selection.
Cache q selection was already top1 confidence before GT matching. Old cache
shard hashes verify only the small arrays explicitly opened below, never P3/P4.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
from native_context_contract import (
    FIXED_ZUP_SIGNS, VERTICAL_FIXED_FACES, inverse_stored_permutation,
    normal_facing_permutation, convex_self_visible_proxy,
    source_reference_proxy, source_canvas_to_raw, outward_normals, PROXY_NAMES)

HERE = Path(__file__).resolve().parent
REPO = Path('/home/minjae/Documents/github/pallet-pose')
SRC = REPO / 'data/pallet/results/pallet_line_pose_v1'
P = HERE.parents[1]
AUDIT = REPO / 'data/pallet/results/pallet_type_selftrain_v1/selftrain_recovery_v1/renderer_front_visibility_audit'
OPEN_ARRAYS = ('points', 'boxes', 'point_valid', 'input_shape', 'gain', 'record_index',
               'detected', 'matched', 'score', 'selected_candidate_index')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bound(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def save_json(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def legacy_functions():
    source = REPO / 'scripts/annotate/convert_to_camera_facing_v4.py'
    tree = ast.parse(source.read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in ('compute_perm_v4', 'polyarea')], type_ignores=[])
    env = {'np': np}
    exec(compile(module, str(source), 'exec'), env)
    return env['compute_perm_v4']


def main():
    start = time.monotonic()
    source_path = SRC / 'SOURCE_MANIFEST.json'
    cache_path = SRC / 'cache/CACHE_MANIFEST.json'
    complete_path = SRC / 'cache/CACHE_COMPLETE.json'
    source = json.loads(source_path.read_text())
    cache = json.loads(cache_path.read_text())
    complete = json.loads(complete_path.read_text())
    assert sha(source_path) == cache['source_manifest_sha256']
    assert sha(cache_path) == complete['manifest_sha256']
    assert complete['PASS'] and complete['complete'] and complete['completed_rows'] == 60000
    records = source['records']; n = len(records)
    assert n == 60000 and [r['index'] for r in records] == list(range(n))
    arrays = {k: np.load(SRC / 'cache' / cache['arrays'][k]['file'], mmap_mode='r') for k in OPEN_ARRAYS}
    assert np.array_equal(arrays['record_index'], cache['record_indices'])
    assert cache['record_indices'] == list(range(n))
    for k, a in arrays.items():
        assert list(a.shape) == cache['arrays'][k]['shape'] and str(a.dtype) == cache['arrays'][k]['dtype']
    checked_slices = 0
    for chunk in sorted((SRC / 'cache/shards').glob('*.json')):
        marker = json.loads(chunk.read_text())
        assert marker['manifest_sha256'] == sha(cache_path)
        for k in OPEN_ARRAYS:
            data = np.asarray(arrays[k][marker['start']:marker['end']])
            assert hashlib.sha256(data.tobytes(order='C')).hexdigest() == marker['array_slice_sha256'][k], (chunk, k)
            checked_slices += 1
    old_rows_path = AUDIT / 'ROWS.json'
    old_rows = {r['id']: r for r in json.loads(old_rows_path.read_text())}
    assert len(old_rows) == n
    small_manifest_path = P / 'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_FRAME_MANIFEST.jsonl'
    small = {r['id']: r for r in map(json.loads, small_manifest_path.read_text().splitlines())}
    assert len(small) == 1003
    old_area = legacy_functions()
    q_raw = np.full((n, 9, 2), np.nan, np.float64)
    b_raw = np.full((n, 4), np.nan, np.float64)
    raw_hw = np.zeros((n, 2), np.int32)
    native_valid = np.zeros((n, 9), bool)
    nearest_valid = np.zeros((n, 9), bool)
    ref = np.zeros((n, 8, 2), np.float64)
    ref_center = np.zeros((n, 2), np.float64)
    corners = np.zeros((n, 8, 3), np.float64)
    pose = np.zeros((n, 4, 4), np.float64)
    intrinsics = np.zeros((n, 3, 3), np.float64)
    camera = np.zeros((n, 3), np.float64)
    perm = np.zeros((n, 8), np.int8)
    fraction = np.full((n, 9), np.nan, np.float64)
    self_visible = np.zeros((n, 8), bool)
    depth = np.zeros((n, 8), np.float64)
    ref_inframe = np.zeros((n, 8), bool)
    proxy = np.full((n, 8), 4, np.int8)
    sha_ann = []
    assets = []
    counts = Counter(); group_counts = defaultdict(Counter)
    normal_failures = []; facecos_errors = []
    projection_errors = []; fixed_sign_errors = []
    old_area_rows = []
    archive_handles = {}
    output_manifest = HERE / 'TYPED_SOURCE_ROW_MAP.jsonl'
    if output_manifest.exists():
        raise FileExistsError(output_manifest)
    with output_manifest.open('x') as output:
        for i, r in enumerate(records):
            loc = r['renderer_annotation_locator_provenance_only']
            if '::' in loc:
                archive, member = loc.split('::', 1)
                if archive not in archive_handles:
                    archive_handles[archive] = zipfile.ZipFile(archive)
                payload = archive_handles[archive].read(member)
                raw_locator = archive + '::' + member.replace('/labels/', '/rgb/').replace('_label.json', '_rgb.png')
            else:
                annotation = REPO / loc
                payload = annotation.read_bytes()
                raw_locator = str(annotation.parent.parent / 'rgb' / annotation.name.replace('_label.json', '_rgb.png'))
            digest = hashlib.sha256(payload).hexdigest()
            assert digest == old_rows[r['id']]['raw_annotation_sha256'], r['id']
            data = json.loads(payload); assert len(data['objects']) == 1
            o = data['objects'][0]; cd = data['camera_data']
            assert o['keypoint_convention'] == 'camera_dynamic_0123_v4'
            raw_hw[i] = r['raw_shape_hw']
            assert raw_hw[i].tolist() == [cd['height'], cd['width']]
            q_raw[i], b_raw[i], gain, _ = source_canvas_to_raw(
                arrays['points'][i], arrays['boxes'][i], r['prepared_shape_hw'], arrays['input_shape'][i], r['reflect_pad_px'])
            assert abs(gain - arrays['gain'][i]) < 1e-12
            finite = np.isfinite(q_raw[i]).all(1) & arrays['point_valid'][i] & arrays['detected'][i]
            h, w = raw_hw[i]
            native_valid[i] = finite & (q_raw[i, :, 0] >= 0) & (q_raw[i, :, 0] < w) & (q_raw[i, :, 1] >= 0) & (q_raw[i, :, 1] < h)
            nearest = np.floor(np.nan_to_num(q_raw[i], nan=-1e9) + .5)
            nearest_valid[i] = native_valid[i] & (nearest[:, 0] >= 0) & (nearest[:, 0] < w) & (nearest[:, 1] >= 0) & (nearest[:, 1] < h)
            ref[i] = o['projected_cuboid']; ref_center[i] = o['projected_cuboid_centroid']
            corners[i] = o['cuboid']; pose[i] = o['pose_transform']; camera[i] = cd['location_worldframe']
            perm[i] = o['perm_v4']; assert sorted(perm[i].tolist()) == list(range(8))
            cp = (np.c_[corners[i], np.ones(8)] @ pose[i].T)[:, :3]
            depth[i] = cp[:, 2]
            k = cd['intrinsics']; intrinsics[i] = [[k['fx'], 0, k['cx']], [0, k['fy'], k['cy']], [0, 0, 1]]
            reproj = cp @ intrinsics[i].T; reproj = reproj[:, :2] / reproj[:, 2:]
            projection_errors.append(float(np.max(np.linalg.norm(reproj - ref[i], axis=1))))
            fraction[i] = o['v2_labels']['occlusion_fraction']
            self_visible[i] = convex_self_visible_proxy(corners[i], camera[i])
            proxy[i], ref_inframe[i] = source_reference_proxy(ref[i], depth[i], raw_hw[i], self_visible[i], fraction[i, :8])
            fixed = inverse_stored_permutation(corners[i], perm[i])
            fixed_projected = inverse_stored_permutation(ref[i], perm[i])
            replay, cos = normal_facing_permutation(fixed, fixed_projected, camera[i])
            if not np.array_equal(replay, perm[i]):
                normal_failures.append({'id': r['id'], 'stored': perm[i].tolist(), 'replay': replay.tolist()})
            fixed_normalized = (fixed - fixed.min(0)) / np.ptp(fixed, axis=0) * 2 - 1
            fixed_sign_errors.append(float(np.max(np.abs(fixed_normalized - FIXED_ZUP_SIGNS))))
            nd, fc = outward_normals(corners[i], [[0, 1, 2, 3]])
            facecos = float(nd[0] @ (camera[i] - fc[0]) / np.linalg.norm(camera[i] - fc[0]))
            facecos_errors.append(abs(facecos - o['front_visibility_cos']))
            if r['id'] in small:
                lp = old_area(fixed, fixed_projected)
                old_area_rows.append({'id': r['id'], 'stored': perm[i].tolist(), 'legacy_area': lp[:8],
                                      'normal_replay': replay.tolist(), 'legacy_matches': lp[:8] == perm[i].tolist(),
                                      'source': r['source'], 'partition': r['partition']})
            assets.append(o.get('source_asset', 'UNKNOWN')); sha_ann.append(digest)
            c = group_counts[r['source'] + '/' + r['partition']]
            c['frames'] += 1; c['detected'] += int(arrays['detected'][i]); c['matched'] += int(arrays['matched'][i])
            c['query_native_valid8'] += int(native_valid[i, :8].sum())
            for s, name in enumerate(PROXY_NAMES):
                c[name] += int((proxy[i] == s).sum())
                c[name + '_query_valid'] += int(((proxy[i] == s) & native_valid[i, :8]).sum())
            counts['normal_permutation_exact'] += int(np.array_equal(replay, perm[i]))
            counts['frames'] += 1
            counts['front_vertical'] += int(np.ptp(corners[i, :4, 2]) > 0 and len(np.unique(perm[i, :4])) == 4)
            output.write(json.dumps({'index': i, 'id': r['id'], 'source': r['source'], 'partition': r['partition'],
                'scenario_id': r['scenario_id'], 'source_split': r['source_split'],
                'prepared_image': r['image'], 'prepared_image_sha256': r['image_sha256'],
                'raw_rgb_locator': raw_locator, 'raw_rgb_sha256': None,
                'annotation_locator_GT_only': loc, 'annotation_sha256': digest,
                'raw_hw': raw_hw[i].tolist(), 'prepared_hw': r['prepared_shape_hw'],
                'cache_row': i, 'asset_GT_only': assets[-1]}, separators=(',', ':')) + '\n')
            if (i + 1) % 10000 == 0:
                print('SOURCE_COORDS_AND_LABELS', i + 1, '/60000', flush=True)
    for z in archive_handles.values():
        z.close()
    input_path = HERE / 'TYPED_SOURCE_INPUT_COORDS_ONLY.npz'
    gt_path = HERE / 'TYPED_SOURCE_PACK_GT_ONLY.npz'
    for path in (input_path, gt_path):
        if path.exists(): raise FileExistsError(path)
    np.savez_compressed(input_path, record_index=np.arange(n, dtype=np.int32),
        initial_query9_raw=q_raw, predicted_box_xyxy_raw=b_raw, raw_hw=raw_hw,
        initial_point_valid9=np.asarray(arrays['point_valid']), query_native_valid9=native_valid,
        query_nearest_valid9=nearest_valid, detected=np.asarray(arrays['detected']),
        detection_score=np.asarray(arrays['score']), selected_candidate_index=np.asarray(arrays['selected_candidate_index']))
    np.savez_compressed(gt_path, record_index=np.arange(n, dtype=np.int32),
        sample_id=np.asarray([r['id'] for r in records]), source=np.asarray([r['source'] for r in records]),
        partition=np.asarray([r['partition'] for r in records]), source_asset=np.asarray(assets),
        annotation_sha256=np.asarray(sha_ann), original_saved_query9_raw=q_raw,
        reference_projected8=ref, reference_projected_centroid=ref_center,
        cuboid_world_dynamic8=corners, perm_dynamic_to_writer_fixed8=perm,
        source_world_to_camera=pose, K=intrinsics, camera_world=camera, reference_camera_depth8=depth,
        external_9ray_fraction9=fraction, convex_self_visible_proxy8=self_visible,
        reference_inframe8=ref_inframe, reference_proxy_state8=proxy,
        certified_semantic_state8=np.full((n, 8), -1, np.int8),
        target_pixel_support8=np.full((n, 8), -1, np.int8),
        target_pixel_support_available8=np.zeros((n, 8), bool),
        detector_matched_to_source_target=np.asarray(arrays['matched']))
    assert len(old_area_rows) == 1003
    save_json(HERE / 'LEGACY_AREA_VS_NORMAL_1003.json', {'rows': old_area_rows})
    report = {'schema': 'native_source_coords_proxy_pack_v1', 'completed': True,
        'scope': 'All existing60000 source rows, no RGB/P3/P4/weights/forward/fit/PnP/current data access',
        'counts': dict(counts), 'source_partition_counts': {k: dict(v) for k, v in group_counts.items()},
        'legacy_area_1003': {'exact': sum(r['legacy_matches'] for r in old_area_rows),
                             'mismatch': sum(not r['legacy_matches'] for r in old_area_rows)},
        'normal_rule_failures': normal_failures, 'normal_rule_is_empirical_reconstruction_not_writer_snapshot': True,
        'full_source_projection_max_error_px': max(projection_errors),
        'fixed_zup_sign_max_abs_error': max(fixed_sign_errors),
        'front_visibility_cos_is_face_center_cosine_max_error': max(facecos_errors),
        'center_cos_selects_front_face_face_center_cosine_is_stored_diagnostic': True,
        'opened_cache_arrays': list(OPEN_ARRAYS), 'cache_slice_hashes_checked': checked_slices,
        'no_P3_P4_files_opened': True,
        'query_selection': 'Stable highest confidence argmax before GT matching; all rows retained including missing/unmatched',
        'query_stage': 'RAW_YOLO26N_TOP1; unlike prior1003 N3_SUBPIX initial_points',
        'GT_contract': {'proxy_state_names': list(PROXY_NAMES), 'external_fraction_threshold': .5,
            'precedence': 'missing evidence U; else reference outside/behind F; else convex self-hidden S; else >=.5 external-ray E; else D',
            'certified_semantic_state8': '-1=UNKNOWN for every row; no human semantic gold manufactured',
            'target_pixel_support8': '-1=UNKNOWN; source renderer labels do not export dense per-target visible masks in these records',
            'virtual_cuboid_surface_membership_required': False,
            'perm_pose_camera_reference_and_proxy_fields_forbidden_model_inputs': True},
        'split_caveat': source['split_rule'],
        'source_files': [bound(x) for x in [source_path, cache_path, complete_path, old_rows_path,
            small_manifest_path, Path(__file__), HERE / 'native_context_contract.py',
            REPO / 'scripts/research/pallet_line_pose_v1/features.py',
            REPO / 'scripts/research/pallet_line_pose_v1/cache_features.py',
            REPO / 'scripts/annotate/convert_to_camera_facing_v4.py']],
        'coordinate_array_bindings': {k: bound(SRC / 'cache' / cache['arrays'][k]['file']) for k in OPEN_ARRAYS},
        'outputs': [bound(x) for x in [input_path, gt_path, output_manifest, HERE / 'LEGACY_AREA_VS_NORMAL_1003.json']],
        'elapsed_seconds': time.monotonic() - start}
    save_json(HERE / 'TYPED_SOURCE_PACK_SEAL.json', report)
    print(json.dumps({k: report[k] for k in ['counts', 'legacy_area_1003', 'full_source_projection_max_error_px',
           'fixed_zup_sign_max_abs_error', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
