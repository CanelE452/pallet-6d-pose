"""Audit the user's original1003 SYN full RGB / M4, not internal V129 data.

Read only declared original metadata/RGB headers and M0/M4 label PNGs. Never
open ROI feature archives. No model, crop, image resize, fit, or real GT read.
"""
from pathlib import Path
import collections
import csv
import gzip
import hashlib
import io
import json
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).absolute().parent
TASK = Path('/home/minjae/Documents/Codex/2026-10-20/task-2')
SOURCE = TASK / 'external_ray_proxy_head_20261022_v60/PREPARED_INPUTS_SYNTH.json.gz'
V61 = TASK / 'target_presence_classifier_20261022_v61'
RAW_ROOT = Path('/home/minjae/Documents/github/pallet-pose/data/pallet/training_data/paper_release/v2_prod40k_clean_merged')
EXPECTED = {
    str(SOURCE): '17c24bcb3e3694d3d8bb40851a944468fd491719fb7a15ca1e485a5607eb799e',
    str(V61/'MASK_CONTRACT_AUDIT.csv'): 'f13c50ec0075c9adfd5f5934e27b90fb83a798e1866857583acb32dd32e018ad',
    str(V61/'PRESENCE_QUERY_LABELS.csv'): 'bbd312302191491ea8356d73aa0a2934fff06604b7d89cb7cb83cee17428054c',
    str(V61/'PROTOCOL.json'): '0464482217a49a8ccaf0fcbd6cd75addae23aba812320195a6605f3963afb6a9',
}


def safe(value):
    p = Path(value)
    if not p.is_absolute() or any(s in str(p).lower() for s in ('v112', 'cache', '.pyc', '..')):
        raise ValueError(str(p))
    for a in reversed((p, *p.parents)):
        if a.is_symlink():
            raise ValueError(str(a))
    return p


def read_bound(path, expected=None):
    p = safe(path)
    raw = p.read_bytes()
    b = {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)}
    if expected is not None:
        assert b['sha256'] == expected, b
    return raw, b


def binding(path):
    return read_bound(path)[1]


def json_write(path, value):
    p = safe(path)
    assert not p.exists(), str(p)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n')


def jsonl_write(path, rows):
    p = safe(path)
    assert not p.exists(), str(p)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for r in rows))


def main():
    started = time.time()
    out = safe(HERE / 'data_audit')
    assert not out.exists(), str(out)
    dependencies = {}
    loaded = {}
    for s, expected in EXPECTED.items():
        raw, b = read_bound(s, expected)
        dependencies[Path(s).name] = b
        loaded[Path(s).name] = raw
    frames = json.loads(gzip.decompress(loaded['PREPARED_INPUTS_SYNTH.json.gz']))
    previous = {r['id']: r for r in csv.DictReader(io.StringIO(loaded['MASK_CONTRACT_AUDIT.csv'].decode()))}
    old_queries = list(csv.DictReader(io.StringIO(loaded['PRESENCE_QUERY_LABELS.csv'].decode())))
    assert len(frames) == len(previous) == 1003 and len(old_queries) == 8024
    assert len({r['id'] for r in frames}) == 1003
    mask_values = {'M0': set(), 'M4': set()}
    counts = {fold: collections.Counter() for fold in ('train', 'validation')}
    rgb_hashes = {fold: set() for fold in counts}
    scene_ids = {fold: set() for fold in counts}
    frame_rows, image_inputs, target_rows, query_rows = [], [], [], []
    problems = []
    for i, x in enumerate(frames):
        fold = x['split']
        assert fold in counts
        prior = previous[x['id']]
        assert prior['split'] == fold
        image_path = safe(x['image'])
        assert image_path.parent == RAW_ROOT/'rgb'
        stem = image_path.name.removesuffix('_rgb.png')
        ann_path = safe(RAW_ROOT/'labels'/(stem+'_label.json'))
        m4_path = safe(RAW_ROOT/'mask_visible'/(stem+'.png'))
        m0_path = safe(RAW_ROOT/'mask_amodal'/(stem+'.png'))
        try:
            rgb_raw, rgb_bound = read_bound(image_path, x['image_sha256'])
            ann_raw, ann_bound = read_bound(ann_path, prior['annotation_sha256'])
            m4_raw, m4_bound = read_bound(m4_path, prior['visible_sha256'])
            m0_raw, m0_bound = read_bound(m0_path, prior['amodal_sha256'])
        except (OSError, AssertionError) as exc:
            problems.append({'source_index': i, 'id': x['id'], 'error': str(exc)})
            continue
        ann = json.loads(ann_raw)
        assert len(ann['objects']) == 1
        obj = ann['objects'][0]
        v2 = obj['v2_labels']
        with Image.open(io.BytesIO(rgb_raw)) as im:
            rgb_hw = (im.height, im.width)
            rgb_format = im.format
        del rgb_raw
        m4 = np.array(Image.open(io.BytesIO(m4_raw)))
        m0 = np.array(Image.open(io.BytesIO(m0_raw)))
        assert m4.dtype == m0.dtype == np.uint8
        assert m4.shape == m0.shape == rgb_hw == tuple(x['raw_hw'])
        values4 = set(np.unique(m4).tolist()); values0 = set(np.unique(m0).tolist())
        assert values4 <= {0, 1, 2, 254, 255} and values0 <= {0, 1, 2, 254, 255}
        mask_values['M4'].update(values4); mask_values['M0'].update(values0)
        visible = m4 > 127
        amodal = m0 > 127
        a4, a0 = int(visible.sum()), int(amodal.sum())
        assert a4 == int(prior['visible_area']) == int(v2['mask_area_visible'])
        assert a0 == int(prior['amodal_area']) == int(v2['mask_area_amodal'])
        assert not (visible & ~amodal).any()
        assert a0 > 0
        occluded = int((amodal & ~visible).sum())
        perm = np.asarray(obj['perm_v4'], np.int64)
        assert perm.shape == (8,) and sorted(perm.tolist()) == list(range(8))
        ray = np.asarray(v2['occlusion_fraction'], np.float64)
        assert ray.shape == (9,) and np.isfinite(ray).all() and (ray >= 0).all() and (ray <= 1).all()
        q8 = np.asarray(x['initial_points'], np.float64)[:8]
        assert q8.shape == (8, 2) and np.isfinite(q8).all()
        h, w = rgb_hw
        pix = np.floor(q8 + .5).astype(np.int64)
        valid = ((q8[:, 0] >= 0)&(q8[:, 0] < w)&(q8[:, 1] >= 0)&(q8[:, 1] < h)&
                 (pix[:, 0] >= 0)&(pix[:, 0] < w)&(pix[:, 1] >= 0)&(pix[:, 1] < h))
        y = [int(visible[pix[k, 1], pix[k, 0]]) if valid[k] else None for k in range(8)]
        c = counts[fold]
        c['frames'] += 1; c['pixels'] += h*w
        c['M0_target_alone_pixels'] += a0; c['M4_visible_target_pixels'] += a4
        c['M0_only_pixels'] += occluded; c['M4_non_target_pixels'] += h*w-a4
        c['frames_with_M0_only_pixels'] += int(occluded > 0)
        c['frames_with_visibility_ray_ge_half_any_native'] += int((ray[:8] >= .5).any())
        c['coordinate_valid_queries_no_ROI_rule'] += int(valid.sum())
        c['M4_query_positive_coordinate_valid'] += sum(a == 1 for a in y)
        c['M4_query_negative_coordinate_valid'] += sum(a == 0 for a in y)
        rgb_hashes[fold].add(rgb_bound['sha256']); scene_ids[fold].add(stem)
        input_row = {'frame_index': i, 'id': x['id'], 'split': fold, 'raw_hw': list(rgb_hw), 'RGB': rgb_bound,
                     'scope': 'UNCROPPED_FULL_FRAME; no query/box/pose mask input'}
        target = {'frame_index': i, 'id': x['id'], 'split': fold, 'M4': m4_bound,
                  'label': 'visible_target=(raw_uint8>127); binary target versus non-target',
                  'M0_auxiliary': m0_bound, 'M4_area': a4, 'M0_area': a0, 'M0_only_area': occluded,
                  'annotation': ann_bound, 'full_FOREIGN_or_background_class_GT': 'UNAVAILABLE'}
        query = {'frame_index': i, 'id': x['id'], 'split': fold, 'initial_native8_xy': q8.tolist(),
             'original_coordinate_valid8_no_ROI_rule': valid.tolist(), 'M4_target_at_query8': y,
             'perm_v4_dynamic_to_native': perm.tolist(), 'saved_dynamic9_external_ray_occlusion_fraction': ray.tolist(),
             'ray_target_warning': 'Virtual cuboid ray-blockage proxy, not direct native semantic visibility or segmentation class GT.'}
        frame_rows.append({'input': input_row, 'target': target, 'query_sidecar': query,
             'M4_fraction_image': a4/(h*w), 'M0_fraction_image': a0/(h*w), 'mask_raw_values_M4': sorted(values4),
             'source_asset': obj.get('source_asset'), 'pallet_type': v2.get('pallet_type'),
             'material_variant_actual': v2.get('material_variant_actual'),
             'mask_profile': v2.get('mask_profile'), 'occlusion_decomposition_available': v2.get('occlusion_decomposition_available'),
             'RGB_format': rgb_format})
        image_inputs.append(input_row); target_rows.append(target); query_rows.append(query)
        if (i+1) % 200 == 0:
            print(json.dumps({'original_SYN_audited': i+1, 'total': 1003}), flush=True)
    assert not problems, problems[:5]
    assert len(frame_rows) == 1003 and counts['train']['frames'] == 768 and counts['validation']['frames'] == 235
    assert not (rgb_hashes['train'] & rgb_hashes['validation'])
    assert not (scene_ids['train'] & scene_ids['validation'])
    queries_by_id = {(r['id'], int(r['native_corner'])): r for r in old_queries}
    for r in query_rows:
        for k in range(8):
            q = queries_by_id[(r['id'], k)]
            assert q['split'] == r['split']
            assert float(q['query_x']) == r['initial_native8_xy'][k][0]
            assert float(q['query_y']) == r['initial_native8_xy'][k][1]
            if r['original_coordinate_valid8_no_ROI_rule'][k]:
                assert int(q['presence_target']) == r['M4_target_at_query8'][k]
    out.mkdir()
    jsonl_write(out/'FRAME_MANIFEST.jsonl', frame_rows)
    jsonl_write(out/'FULL_FRAME_RGB_INPUTS.jsonl', image_inputs)
    jsonl_write(out/'VISIBLE_MASK_TARGETS.jsonl', target_rows)
    jsonl_write(out/'ORIGINAL_QUERY_SIDECAR.jsonl', query_rows)
    fractions = {fold: [r['M4_fraction_image'] for r in frame_rows if r['input']['split'] == fold] for fold in counts}
    summary = {'completed': True, 'PASS': True, 'schema': 'V267_USER_ORIGINAL_SYN1003_FULL_FRAME_VISIBLE_MASK_AUDIT_V1',
         'source': 'User original v61 SYN cohort, not internal V129440 corpus',
         'original_raw_root': str(RAW_ROOT), 'dependencies': dependencies, 'frames': 1003,
         'census': {k: dict(v) for k, v in counts.items()},
         'raw_HW_counts': dict(collections.Counter(str(r['input']['raw_hw']) for r in frame_rows)),
         'mask_values': {k: sorted(v) for k, v in mask_values.items()}, 'label_threshold': '>127 uint8',
         'M4_fraction_image_percentiles_0_10_50_90_100': {k: np.percentile(v, [0,10,50,90,100]).tolist() for k,v in fractions.items()},
         'missing_or_hash_contract_failures': problems, 'RGB_header_shape_mismatch': 0,
         'M4_not_subset_M0_pixels': 0, 'area_annotation_or_old_audit_mismatches': 0,
         'split_proof': {'TRAIN_VAL_scene_id_overlap': 0, 'TRAIN_VAL_RGB_sha_overlap': 0,
               'TRAIN_VAL_unique_RGB_hashes': [len(rgb_hashes['train']), len(rgb_hashes['validation'])],
               'family_asset_lineage_complete': False, 'repeated_VAL_and_upstream_detector_validation_exposure': 'Previously disclosed; preserve existing split; not new blind validation'},
         'supervision': {'M4': 'Final visible target binary silhouette. Valid dense target/non-target supervision.',
              'M0': 'Target-only silhouette, includes target self-occluding front surfaces; not hidden 3D surface ground truth.',
              'M0_only': 'Target-alone silhouette not visible finally. Does not label the complete occluding object elsewhere.',
              'M4_complement': 'Non-target, combines background, holes, foreign objects and other content; not all external occlusion.',
              'FOREIGN_instance_or_depth': 'Not supplied by this original public-profile mask set.',
              'ray_occlusion_fraction': 'Stored9ray blockage proxy in dynamic order; keep separate from pixel targets and human native D/E.'},
         'input_scope': 'Full RGB1003 frames only; original q8 retained as sidecar, not used to crop/recenter input.',
         'no_ROI_feature_archive_read': True, 'RGB_pixel_arrays_decoded': 0, 'RGB_files_hashed_and_headers_read': 1003,
         'mask_pixel_arrays_decoded': 2006, 'current_or_real_annotation_reads': 0,
         'new_models_or_optimizer_or_inference': 0, 'elapsed_seconds': time.time()-started}
    json_write(out/'AUDIT_SUMMARY.json', summary)
    code = binding(Path(__file__).absolute())
    names = ['FRAME_MANIFEST.jsonl', 'FULL_FRAME_RGB_INPUTS.jsonl', 'VISIBLE_MASK_TARGETS.jsonl', 'ORIGINAL_QUERY_SIDECAR.jsonl', 'AUDIT_SUMMARY.json']
    json_write(out/'COMPLETE.json', {'completed': True, 'PASS': True, 'schema': summary['schema'],
          'code': code, 'outputs': {n: binding(out/n) for n in names}, 'frames': 1003,
          'TRAIN_VAL': [768,235], 'full_frame_input': True, 'current_or_real_reads': 0,
          'ROI_feature_archive_reads': 0, 'new_fit_model_forward_or_download': 0})
    print(json.dumps({'completed': True, 'PASS': True, 'complete': binding(out/'COMPLETE.json'),
                     'census': summary['census'], 'fraction_percentiles': summary['M4_fraction_image_percentiles_0_10_50_90_100']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
