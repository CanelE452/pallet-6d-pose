"""Build exactly the locked5203 paired-mask GT-only reference-support rows.

No RGB/P3/model/weights/forward/fit/current/PnP/depth-ray reads. Metadata index
is frozen before any maskteacher, followed by exact prior1case parity, then all
rows. Existing source/protocol/seal files are never modified.
"""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
P = ROOT.parent
PROTOCOL = ROOT / 'PAIRED_MASK_GT_CORPUS_PROTOCOL.json'
PROTOCOL_SHA = 'd972c3c4d65d6537091d37002c72b65c6b2b43d2f44ebae1ab2e9a179f1d7a28'
YCB = P / 'query_visibility_public_real_20261022_v63/ycb'
YCB_MANIFEST = YCB / 'instances.jsonl'
MODELS = P / 'query_visibility_loco_native_context_20261023_v74/source_audit/YCB_PUBLIC_NATIVE_FEASIBILITY/models_info_OFFICIAL.json'
NATIVE_HELPER = P / 'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
TEACHER_HELPER = ROOT / 'teacher_preparation/reference_local_support_teacher.py'
SYNTH_LINKS = ROOT / 'source_inventory/REFERENCE_AND_M0_M4_TEACHER_LINKS_GT_ONLY.jsonl'
PRIOR_CASE = ROOT / 'teacher_preparation/FIXED_YCB_ONE_INSTANCE.json'
PRIOR_REF = P / 'query_visibility_loco_native_context_20261023_v74/source_contract/public_ycb_native_proxy_feasibility/ONE_INSTANCE_V2/ONE_INSTANCE_NATIVE_REFERENCES_GT_ONLY.npz'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    p = Path(path)
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def clean(v):
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, (np.bool_, np.integer)): return v.item()
    if isinstance(v, (float, np.floating)) and not np.isfinite(v): return None
    if isinstance(v, np.floating): return float(v)
    return v


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def load_pure(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def ycb_geometry(row, model_info, H):
    info = model_info[str(row['object_id'])]
    lo = np.asarray([info['min_' + x] for x in 'xyz'], float)
    size = np.asarray([info['size_' + x] for x in 'xyz'], float)
    fixed = lo + (H.FIXED_ZUP_SIGNS + 1) / 2 * size
    R = np.asarray(row['cam_R_m2c'], float).reshape(3, 3)
    t = np.asarray(row['cam_t_m2c'], float)
    K = np.asarray(row['K'], float).reshape(3, 3)
    cam_model = np.linalg.solve(R, -t)
    camera_fixed = fixed @ R.T + t
    assert np.isfinite(camera_fixed).all() and (camera_fixed[:, 2] > 0).all()
    hp = camera_fixed @ K.T
    projected_fixed = hp[:, :2] / hp[:, 2:]
    perm, _ = H.normal_facing_permutation(fixed, projected_fixed, cam_model)
    dynamic = fixed[perm]
    r = projected_fixed[perm]
    self_flag = H.convex_self_visible_proxy(dynamic, cam_model).astype(np.int8)
    return r, perm, self_flag, (camera_fixed[perm, 2] > 0).astype(np.int8)


def read_pair(row):
    raw0 = np.asarray(Image.open(row['M0_path']))
    raw4 = np.asarray(Image.open(row['M4_path']))
    if raw0.ndim != 2 or raw4.shape != raw0.shape or list(raw0.shape) != row['raw_hw']:
        raise ValueError('Mask pair/native HW mismatch')
    threshold = 127 if row['source'] == 'SYNTHETIC' else 0
    M0, M4 = raw0 > threshold, raw4 > threshold
    if np.any(M4 & ~M0):
        raise ValueError('M4subsetM0 failed: ' + str(int((M4 & ~M0).sum())))
    b0, b4 = bound(row['M0_path']), bound(row['M4_path'])
    if row['source'] == 'SYNTHETIC':
        assert b0['sha256'] == row['M0_expected_sha256']
        assert b4['sha256'] == row['M4_expected_sha256']
    u0, n0 = np.unique(raw0, return_counts=True); u4, n4 = np.unique(raw4, return_counts=True)
    meta = {'M0': b0, 'M4': b4, 'threshold_strict_gt': threshold,
        'M0_raw_dtype': str(raw0.dtype), 'M4_raw_dtype': str(raw4.dtype),
        'M0_raw_values': u0.tolist(), 'M0_raw_counts': n0.tolist(),
        'M4_raw_values': u4.tolist(), 'M4_raw_counts': n4.tolist(),
        'M0_target_area': int(M0.sum()), 'M4_target_area': int(M4.sum()),
        'Mfull_equals_Mvisible': bool(np.array_equal(M0, M4))}
    return M0, M4, meta


def metadata_index():
    synth = [json.loads(x) for x in SYNTH_LINKS.read_text().splitlines()]
    ycb = [json.loads(x) for x in YCB_MANIFEST.read_text().splitlines()]
    if len(synth) != 1003 or len(ycb) != 4200:
        save('COUNT_FAILURE.json', {'expected': [1003, 4200], 'actual': [len(synth), len(ycb)]})
        raise ValueError('Source count differs from frozen corpus protocol')
    assert Counter(x['split'] for x in synth) == {'train': 768, 'val': 235}
    assert Counter(x['split'] for x in ycb) == {'train': 3400, 'val': 800}
    rows = []
    for line, s in enumerate(synth):
        rows.append({'row_index': len(rows), 'id': s['id'], 'source': 'SYNTHETIC', 'split': s['split'],
            'source_metadata_line': line, 'source_metadata_file': str(SYNTH_LINKS),
            'annotation_path': s['annotation_GT_only']['path'], 'annotation_expected_sha256': s['annotation_GT_only']['sha256'],
            'reference_dynamic8': s['reference_projected_dynamic8_GT_only'],
            'perm_dynamic_to_fixed8': s['stored_dynamic_to_writer_fixed_permutation_GT_only'],
            'M0_path': s['M0_GT_only']['path'], 'M0_expected_sha256': s['M0_GT_only']['sha256'],
            'M4_path': s['M4_GT_only']['path'], 'M4_expected_sha256': s['M4_GT_only']['sha256'],
            'raw_hw': s['raw_hw'], 'reference_convention': s['reference_convention']})
    for line, y in enumerate(ycb):
        expected_split = 'train' if y['scene_id'] < 8 else 'val'
        assert y['split'] == expected_split
        rows.append({'row_index': len(rows), 'id': f'YCB:{y["scene_id"]:06d}:{y["frame_id"]:06d}:{y["instance_id"]:06d}',
            'source': 'YCB_PUBLIC_REAL', 'split': y['split'], 'source_metadata_line': line,
            'source_metadata_file': str(YCB_MANIFEST), 'raw_hw': [y['height'], y['width']],
            'scene_id': y['scene_id'], 'frame_id': y['frame_id'], 'instance_id': y['instance_id'], 'object_id': y['object_id'],
            'image_path_lineage_only_NOT_READ': y['image'], 'image_sha256_lineage_only': y['image_sha256'],
            'cam_R_m2c_GT_only': y['cam_R_m2c'], 'cam_t_m2c_GT_only': y['cam_t_m2c'], 'K_GT_only': y['K'],
            'M0_path': y['mask_full'], 'M4_path': y['mask_vis'],
            'reference_convention': 'BOP_model_Zup_AABB_camera_dynamic_normal_front_depth_rear'})
    assert len(rows) == 5203 and len(set(r['id'] for r in rows)) == 5203
    assert [r['row_index'] for r in rows] == list(range(5203))
    index_path = OUT / 'ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
    with index_path.open('x') as f:
        for row in rows: f.write(json.dumps(row, separators=(',', ':')) + '\n')
    save('METADATA_INDEX_LOCK.json', {'completed': True, 'frozen_before_mask_teacher': True,
        'counts': {'SYNTHETIC': 1003, 'YCB_PUBLIC_REAL': 4200, 'rows': 5203, 'slots': 41624},
        'index': bound(index_path), 'metadata': [bound(x) for x in [SYNTH_LINKS, YCB_MANIFEST, MODELS]],
        'selection': 'All existing lines, source order SYNTHETIC thenYCB, no mask/state/area/score filtering',
        'protocol': bound(PROTOCOL), 'code': bound(Path(__file__))})
    return rows, synth, ycb


def main():
    start = time.monotonic()
    assert sha(PROTOCOL) == PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text()); assert len(protocol['bindings']) == 9
    for b in protocol['bindings']: assert sha(b['path']) == b['sha256'], b['path']
    assert not (OUT / 'ORDERED_METADATA_INDEX_GT_ONLY.jsonl').exists()
    save('PREFLIGHT_BINDINGS.json', {'PASS': True, 'protocol': bound(PROTOCOL),
        'nine_bindings_verified': protocol['bindings'], 'code': bound(Path(__file__)),
        'source_manifest_additional_binding': bound(YCB_MANIFEST)})
    rows, synth, ycb = metadata_index()
    T = load_pure(TEACHER_HELPER, 'bound_teacher')
    H = load_pure(NATIVE_HELPER, 'bound_native_math')
    models = json.loads(MODELS.read_text())
    fixed = next(y for y in ycb if (y['scene_id'], y['frame_id'], y['instance_id']) == (0, 1, 0))
    r, perm, self_flag, z_positive = ycb_geometry(fixed, models, H)
    with np.load(PRIOR_REF, allow_pickle=False) as a:
        np.testing.assert_array_equal(r, a['reference_projected8'])
        np.testing.assert_array_equal(perm, a['perm_dynamic_to_fixed'])
        np.testing.assert_array_equal(self_flag, a['convex_self_visible_proxy8'])
    fixed_index = next(x for x in rows if x['source'] == 'YCB_PUBLIC_REAL' and
                       (x['scene_id'], x['frame_id'], x['instance_id']) == (0, 1, 0))
    M0, M4, _ = read_pair(fixed_index)
    result = T.reference_local_support(M0, M4, r, self_proxy8=self_flag)
    previous = json.loads(PRIOR_CASE.read_text())['teacher']
    assert clean(result) == previous
    save('FIXED_ONE_CASE_EXACT_PARITY.json', {'PASS': True, 'reference_perm_projection_self_teacher_all_exact': True,
        'float_or_state_differences': 0, 'previous': [bound(PRIOR_REF), bound(PRIOR_CASE)],
        'new_result': result, 'metadata_lock': bound(OUT / 'METADATA_INDEX_LOCK.json')})
    print('METADATA_FROZEN_5203_AND_FIXED_CASE_EXACT_PASS', flush=True)
    simple_fields = ['label', 'rule', 'GTref_valid_xy', 'center_M0', 'center_M4',
        'strong_3x3_window_valid', 'strong_3x3_full_M0', 'strong_3x3_visible', 'strong_3x3_covered',
        'expected_support_pixels_A', 'visible_support_pixels_V', 'weighted_expected_mass_A',
        'weighted_visible_mass_V', 'visible_over_expected_ratio', 'footprint_support_adequate',
        'valid_footprint_pixels', 'full_disk_lattice_pixels', 'footprint_valid_fraction', 'near_boundary',
        'rho_px', 'sigma_px', 'reference_layout_finite']
    collected = {k: [] for k in simple_fields}
    refs = []; perms = []; selves = []; depths = []
    stats = defaultdict(lambda: {'rows': 0, 'slots': 0, 'labels': Counter(), 'reasons': Counter(),
        'self_overlap': Counter(), 'invalid_reference_slots': 0, 'near_boundary_slots': 0,
        'adequate_footprint_slots': 0, 'center_M0zero_slots': 0, 'center_M0zero_labels': Counter(),
        'zero_expected_footprint_slots': 0, 'zero_expected_footprint_labels': Counter(),
        'NaN_ratio_slots': 0, 'equal_mask_rows': 0})
    output_rows = OUT / 'PER_ROW_TEACHER_GT_ONLY.jsonl'
    current = None
    try:
        with output_rows.open('x') as f:
            for i, row in enumerate(rows):
                current = row
                if row['source'] == 'SYNTHETIC':
                    assert sha(row['annotation_path']) == row['annotation_expected_sha256']
                    annotation = json.loads(Path(row['annotation_path']).read_text())
                    o = annotation['objects'][0]
                    r = np.asarray(row['reference_dynamic8'], float)
                    np.testing.assert_array_equal(r, np.asarray(o['projected_cuboid']))
                    perm = np.asarray(row['perm_dynamic_to_fixed8'], np.int8)
                    np.testing.assert_array_equal(perm, o['perm_v4'])
                    if o.get('cuboid') is not None and annotation['camera_data'].get('location_worldframe') is not None:
                        cube = np.asarray(o['cuboid'], float)
                        self_flag = H.convex_self_visible_proxy(cube, annotation['camera_data']['location_worldframe']).astype(np.int8)
                        _, composed = H.source_pose_frames(cube, o['pose_transform'])
                        cp = np.c_[cube, np.ones(8)] @ composed.T
                        z_positive = (cp[:, 2] > 0).astype(np.int8)
                    else:
                        self_flag = np.full(8, -1, np.int8); z_positive = self_flag.copy()
                else:
                    y = ycb[row['source_metadata_line']]
                    r, perm, self_flag, z_positive = ycb_geometry(y, models, H)
                M0, M4, mask_meta = read_pair(row)
                teacher = T.reference_local_support(M0, M4, r, self_proxy8=self_flag)
                assert teacher['query_diagnostics_separate'] is None
                label = teacher['label']; adequate = teacher['footprint_support_adequate']
                zero = teacher['expected_support_pixels_A'] == 0
                invalid = ~teacher['GTref_valid_xy']
                assert (label[zero] == -1).all() and (label[invalid] == -1).all()
                for k in simple_fields: collected[k].append(teacher[k])
                refs.append(r); perms.append(perm); selves.append(self_flag); depths.append(z_positive)
                s = stats[row['source'] + '/' + row['split']]
                s['rows'] += 1; s['slots'] += 8
                s['labels'].update({str(v): int((label == v).sum()) for v in [-1, 0, 1]})
                s['reasons'].update(map(str, teacher['rule']))
                s['self_overlap'].update(f'self={int(sf)},local={int(lb)}' for sf, lb in zip(self_flag, label))
                s['invalid_reference_slots'] += int(invalid.sum())
                s['near_boundary_slots'] += int(teacher['near_boundary'].sum())
                s['adequate_footprint_slots'] += int(adequate.sum())
                center_zero = teacher['center_M0'] == 0
                s['center_M0zero_slots'] += int(center_zero.sum())
                s['center_M0zero_labels'].update(map(str, label[center_zero]))
                s['zero_expected_footprint_slots'] += int(zero.sum())
                s['zero_expected_footprint_labels'].update(map(str, label[zero]))
                s['NaN_ratio_slots'] += int(np.isnan(teacher['visible_over_expected_ratio']).sum())
                s['equal_mask_rows'] += int(mask_meta['Mfull_equals_Mvisible'])
                f.write(json.dumps(clean({'row_index': i, 'id': row['id'], 'source': row['source'], 'split': row['split'],
                    'reference_dynamic8': r, 'perm_dynamic_to_fixed_GT_only': perm,
                    'mask_pair': mask_meta, 'teacher': teacher, 'reference_camera_Z_positive8': z_positive,
                    'human_semantic_visibility8': [-1] * 8, 'query_pixel_presence8': [-1] * 8}),
                    separators=(',', ':'), allow_nan=False) + '\n')
                if (i + 1) % 250 == 0 or i + 1 in [1003, 5203]:
                    print('GT_ONLY_MASK_TEACHER', i + 1, '/5203', row['source'], flush=True)
    except Exception as error:
        save('BUILD_FAILURE.json', {'completed': False, 'offending_row': current,
            'error': type(error).__name__ + ': ' + str(error), 'rows_retained': True,
            'no_mask_repair_or_row_skip': True, 'new_model_fit_current_PnP': 0})
        raise
    assert len(refs) == 5203 and sum(v['slots'] for v in stats.values()) == 41624
    arrays = {k + ('8' if np.asarray(v).ndim == 2 else ''): np.asarray(v) for k, v in collected.items()}
    arrays.update(row_index=np.arange(5203, dtype=np.int32), id=np.asarray([r['id'] for r in rows]),
        source=np.asarray([r['source'] for r in rows]), split=np.asarray([r['split'] for r in rows]),
        native_GTref8=np.asarray(refs), perm_dynamic_to_fixed8=np.asarray(perms, np.int8),
        convex_self_proxy8_separate=np.asarray(selves, np.int8), reference_camera_Z_positive8=np.asarray(depths, np.int8),
        human_semantic_visibility8=np.full((5203, 8), -1, np.int8), query_pixel_presence8=np.full((5203, 8), -1, np.int8))
    npz_path = OUT / 'GT_ONLY_TEACHER_CORPUS.npz'
    np.savez_compressed(npz_path, **arrays)
    summary = {}
    for group, v in stats.items():
        v['label_coverage_fraction'] = (v['labels']['0'] + v['labels']['1']) / v['slots']
        v['U_fraction'] = v['labels']['-1'] / v['slots']
        summary[group] = v
    save('CORPUS_SUMMARY.json', {'completed': True, 'rows': 5203, 'native_slots': 41624,
        'label_codes': {'1': 'VISIBLE', '0': 'COVERED', '-1': 'U'},
        'by_source_split': summary,
        'all_rows_preserved': True, 'metadata_selection_changed': False,
        'ratio_nan_representation': 'NaN allowed in typedNPZ; null in PER_ROW_TEACHER_GT_ONLY.jsonl',
        'mask_thresholds': {'SYNTHETIC': 'raw>127; raw1/2background', 'YCB_PUBLIC_REAL': 'raw>0'},
        'M4_subset_M0_violations': 0, 'human_semantic_visibility': 'allUNKNOWN',
        'query_pixel_presence': 'allUNKNOWN; no q/inputproducer invented',
        'self_proxy_separate_and_not_merged': True,
        'reused_validation_scope': json.loads(PROTOCOL.read_text())['source_splits'],
        'new_RGB_P3_weights_forward_fit_current_PnP': 0,
        'elapsed_seconds': time.monotonic() - start})
    save('GT_ONLY_CORPUS_SEAL.json', {'completed': True, 'PASS': True, 'rows': 5203, 'native_slots': 41624,
        'protocol': bound(PROTOCOL), 'code': bound(Path(__file__)),
        'outputs': [bound(x) for x in [OUT / 'PREFLIGHT_BINDINGS.json', OUT / 'METADATA_INDEX_LOCK.json',
            OUT / 'ORDERED_METADATA_INDEX_GT_ONLY.jsonl', OUT / 'FIXED_ONE_CASE_EXACT_PARITY.json',
            output_rows, npz_path, OUT / 'CORPUS_SUMMARY.json']],
        'teacher_and_native_helper_imported_exactly_without_copy_or_rename': [bound(TEACHER_HELPER), bound(NATIVE_HELPER)],
        'inputproducer_fit_or_current_admission': False,
        'new_RGB_P3_weights_forward_fit_current_PnP': 0, 'old_artifacts_modified': False})
    print('CORPUS_COMPLETE', json.dumps(clean(summary)), flush=True)


if __name__ == '__main__': main()
