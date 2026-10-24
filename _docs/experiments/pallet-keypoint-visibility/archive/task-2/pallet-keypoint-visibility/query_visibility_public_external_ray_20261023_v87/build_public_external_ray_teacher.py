"""Public-only rasterized virtual-endpoint ray proxy. No RGB/model/PnP.

Clear-to-endpoint != visible structural cue. Own first-hit != semantic self.
M0 is not an argument to the ray decision. All invalid/ambiguous cases retain U.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import importlib.util
import json
import time
import traceback
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / 'SOURCE_EXTERNAL_RAY_PROTOCOL_BEFORE_DEPTH_PIXELS.json'
PROTOCOL_SHA = 'a5e7d481b7d72bf0799cd513c1f92c848714d5e921b9bc264beb3f8bd07463cd'


def bound(path):
    f = Path(path)
    return dict(path=str(f), sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)


def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)): return x.item()
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x


def save(out, name, value):
    with (out / name).open('x') as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def sample_ray(r, camera_xyz, K, depth, scale, owner_count, owner_id, target_id):
    """One teacher-only reference; exact frozen15mm *range* interval decision."""
    r, c, K = np.asarray(r, np.float64), np.asarray(camera_xyz, np.float64), np.asarray(K, np.float64)
    H, W = depth.shape
    result = dict(label=-1, reason='INVALID_REFERENCE_RASTER', relation='INVALID',
                  pixel_xy=[-1, -1], depth_raw=-1, owner_count=-1, unique_owner_id=-1,
                  owner_type='INVALID', own_front_hit=False, observed_Z_mm=np.nan,
                  gamma=np.nan, ref_range_mm=np.nan, observed_range_lo_mm=np.nan,
                  observed_range_hi_mm=np.nan, raster_offset_px=np.nan)
    if r.shape != (2,) or c.shape != (3,) or K.shape != (3, 3): return result
    if not np.isfinite(r).all() or not np.isfinite(c).all() or c[2] <= 0: return result
    if not (0 <= r[0] < W and 0 <= r[1] < H): return result
    px, py = np.floor(r + .5).astype(np.int64)
    if not (0 <= px < W and 0 <= py < H): return result
    result['pixel_xy'] = [int(px), int(py)]
    result['raster_offset_px'] = float(np.linalg.norm(np.array([px, py]) - r))
    raw, count, owner = int(depth[py, px]), int(owner_count[py, px]), int(owner_id[py, px])
    owner_type = ('UNASSIGNED' if count == 0 else 'OVERLAP' if count > 1 else 'OWN' if owner == target_id else 'OTHER')
    result.update(depth_raw=raw, owner_count=count, unique_owner_id=owner if count == 1 else -2 if count > 1 else -1, owner_type=owner_type)
    # Depth invalidity has priority over visible-mask membership.
    if raw <= 0 or not np.isfinite(scale) or scale <= 0:
        result['reason'] = 'INVALID_DEPTH'; return result
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        result['reason'] = 'INVALID_CAMERA'; return result
    try: ray = np.linalg.solve(K, [float(px), float(py), 1.])
    except np.linalg.LinAlgError:
        result['reason'] = 'INVALID_CAMERA'; return result
    if not np.isfinite(ray).all() or ray[2] <= 0:
        result['reason'] = 'INVALID_CAMERA'; return result
    gamma = float(np.linalg.norm(ray / ray[2]))
    ref_range = float(c[2] * gamma)
    lo = max(0., (raw - .5) * scale) * gamma
    hi = (raw + .5) * scale * gamma
    result.update(gamma=gamma, ref_range_mm=ref_range, observed_Z_mm=raw * scale,
                  observed_range_lo_mm=lo, observed_range_hi_mm=hi)
    if ref_range - hi > 15.:
        result['relation'] = 'FRONT'
        if owner_type == 'OTHER': result.update(label=1, reason='EXTERNAL_OTHER_FIRST_HIT_PROXY')
        elif owner_type == 'OWN': result.update(reason='FRONT_OWN_U', own_front_hit=True)
        elif owner_type == 'OVERLAP': result['reason'] = 'FRONT_OVERLAP_U'
        else: result['reason'] = 'FRONT_UNASSIGNED_U'
    elif lo - ref_range > 15.:
        result.update(relation='CLEAR', label=0, reason='CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')
    else:
        result.update(relation='BAND', reason='RANGE_BAND_U')
    return result


def generated_tests():
    dep = np.full((3, 3), 10000, np.uint16)
    cnt, own = np.ones((3, 3), np.uint8), np.ones((3, 3), np.int16)
    K = np.eye(3)
    def run(z=1020., d=dep, count=cnt, oid=own, k=K, r=(0., 0.), scale=.1):
        return sample_ray(r, [0., 0., z], k, d, scale, count, oid, 0)
    cases = {}
    for name, result, expected in [
        ('foreign_front', run(), (1, 'EXTERNAL_OTHER_FIRST_HIT_PROXY')),
        ('clear', run(z=980.), (0, 'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY')),
        ('range_band', run(z=1000.), (-1, 'RANGE_BAND_U')),
        ('missing_depth_even_own_positive', run(d=np.zeros_like(dep), oid=np.zeros_like(own)), (-1, 'INVALID_DEPTH')),
        ('own_front_is_unknown', run(oid=np.zeros_like(own)), (-1, 'FRONT_OWN_U')),
        ('unassigned_front_is_unknown', run(count=np.zeros_like(cnt), oid=-np.ones_like(own)), (-1, 'FRONT_UNASSIGNED_U')),
        ('overlap_front_is_unknown', run(count=np.full_like(cnt, 2), oid=-2*np.ones_like(own)), (-1, 'FRONT_OVERLAP_U')),
        ('reference_outside', run(r=(-.1, 0.)), (-1, 'INVALID_REFERENCE_RASTER')),
        ('rounding_outside', run(r=(2.8, 0.)), (-1, 'INVALID_REFERENCE_RASTER')),
        ('strict15mm_quantization_boundary', run(z=1016., d=np.full_like(dep, 500), scale=2.), (-1, 'RANGE_BAND_U')),
        ('above15mm_quantization_boundary', run(z=1016.000001, d=np.full_like(dep, 500), scale=2.), (1, 'EXTERNAL_OTHER_FIRST_HIT_PROXY')),
        ('range_not_constant_Z', run(z=1010., r=(1., 0.), k=np.diag([.5, 1., 1.])), (1, 'EXTERNAL_OTHER_FIRST_HIT_PROXY')),
    ]:
        assert (result['label'], result['reason']) == expected, (name, result)
        cases[name] = clean(result)
    assert cases['range_not_constant_Z']['gamma'] > 2.
    return {'PASS': True, 'tests': cases, 'M0_argument_absent': True,
            'scope': 'Pure fixtures only; no source pixels/model/current read.'}


def geometry(row, model, H):
    lo = np.array([model['min_' + k] for k in 'xyz'], np.float64)
    size = np.array([model['size_' + k] for k in 'xyz'], np.float64)
    fixed = lo + (H.FIXED_ZUP_SIGNS + 1) / 2 * size
    R, t = np.array(row['cam_R_m2c'], np.float64).reshape(3, 3), np.array(row['cam_t_m2c'], np.float64)
    K = np.array(row['K'], np.float64).reshape(3, 3)
    camera_model = np.linalg.solve(R, -t)
    xyz = fixed @ R.T + t
    hp = xyz @ K.T
    projected = hp[:, :2] / hp[:, 2:]
    perm, _ = H.normal_facing_permutation(fixed, projected, camera_model)
    sf = H.convex_self_visible_proxy(fixed[perm], camera_model).astype(np.int8)
    return projected[perm], xyz[perm], perm, sf, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tests-only', action='store_true')
    args = ap.parse_args()
    out = args.out; out.mkdir(exist_ok=False, parents=True)
    save(out, 'GENERATED_PURE_RAY_TESTS.json', generated_tests())
    if args.tests_only:
        print('PURE_RAY_TESTS_PASS_12', flush=True); return
    start = time.monotonic()
    assert bound(PROTOCOL)['sha256'] == PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text()); B = protocol['bindings']
    for item in B.values(): assert bound(item['path'])['sha256'] == item['sha256'], item['path']
    assert protocol['range_margin_mm'] == 15.
    rows = [json.loads(s) for s in Path(B['manifest']['path']).read_text().splitlines()]
    assert len(rows) == 4200 and Counter(r['split'] for r in rows) == {'train': 3400, 'val': 800}
    ids = [f'YCB:{r["scene_id"]:06d}:{r["frame_id"]:06d}:{r["instance_id"]:06d}' for r in rows]
    assert len(set(ids)) == 4200
    spec = importlib.util.spec_from_file_location('bound_pure_native', B['native_math']['path'])
    H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)
    models = json.loads(Path(B['models_info']['path']).read_text())
    with np.load(B['old_GT_corpus']['path'], allow_pickle=False) as old:
        select = np.flatnonzero(old['source'] == 'YCB_PUBLIC_REAL')
        assert len(select) == 4200 and old['id'][select].tolist() == ids
        oldr, oldperm = old['native_GTref8'][select], old['perm_dynamic_to_fixed8'][select]
        oldself = old['convex_self_proxy8_separate'][select]
        oldlabel, oldm0, oldm4 = old['label8'][select], old['center_M08'][select], old['center_M48'][select]
    groups = defaultdict(list)
    metadata = []
    scene_gt, scene_camera = {}, {}
    for i, row in enumerate(rows):
        scene, frame = row['scene_id'], row['frame_id']
        assert row['split'] == ('train' if scene < 8 else 'val')
        folder = Path(row['depth']).parent.parent
        if scene not in scene_gt:
            gp, kp = folder / 'scene_gt.json', folder / 'scene_camera.json'
            scene_gt[scene] = json.loads(gp.read_text()); scene_camera[scene] = json.loads(kp.read_text())
            metadata += [bound(gp), bound(kp)]
        gt, camera = scene_gt[scene][str(frame)][row['instance_id']], scene_camera[scene][str(frame)]
        assert gt['obj_id'] == row['object_id']
        for a, b in [('cam_R_m2c', 'cam_R_m2c'), ('cam_t_m2c', 'cam_t_m2c')]: np.testing.assert_array_equal(row[a], gt[b])
        np.testing.assert_array_equal(row['K'], camera['cam_K'])
        assert row['depth_scale'] == camera['depth_scale'] and row['depth_scale'] > 0
        groups[scene, frame].append(i)
    assert len(groups) == 1000
    for (scene, frame), indices in groups.items():
        assert sorted(rows[i]['instance_id'] for i in indices) == list(range(len(scene_gt[scene][str(frame)]))), 'Missing annotated owner'
        assert len({rows[i]['depth'] for i in indices}) == 1
    index = out / 'ORDERED_METADATA_INDEX_GT_ONLY.jsonl'
    with index.open('x') as f:
        for i, row in enumerate(rows): f.write(json.dumps({'row_index': i, 'id': ids[i], 'metadata': row}, separators=(',', ':')) + '\n')
    save(out, 'METADATA_LOCK_BEFORE_DEPTH_MASK_PIXELS.json', {'PASS': True, 'index': bound(index),
         'rows': 4200, 'frames': 1000, 'all_annotated_instance_ordinals_complete': True,
         'scene_metadata': metadata, 'protocol': bound(PROTOCOL), 'code': bound(__file__), 'source_bindings': B})
    print('METADATA_LOCKED_4200_INSTANCES_1000_FRAMES', flush=True)
    observations = [None] * 4200
    refs, cameras, perms, selves = [None]*4200, [None]*4200, [None]*4200, [None]*4200
    input_bindings = []
    offending = None
    try:
        for g, ((scene, frame), indices) in enumerate(groups.items()):
            offending = dict(scene=scene, frame=frame, row_indices=indices)
            first = rows[indices[0]]
            depth_bytes = Path(first['depth']).read_bytes()
            assert depth_bytes[:8] == b'\x89PNG\r\n\x1a\n' and list(depth_bytes[24:26]) == [16, 0], 'Expected16bitgraydepth'
            depth = np.asarray(Image.open(first['depth']))
            Hn, Wn = first['height'], first['width']
            assert depth.shape == (Hn, Wn) and np.issubdtype(depth.dtype, np.integer)
            assert depth.min() >= 0 and depth.max() <= 65535
            count = np.zeros((Hn, Wn), np.uint8); owner = np.full((Hn, Wn), -1, np.int16)
            input_bindings.append(bound(first['depth']))
            for i in indices:
                row = rows[i]
                raw = np.asarray(Image.open(row['mask_vis']))
                assert raw.shape == depth.shape
                mask = raw > 0
                count += mask.astype(np.uint8)
                owner[mask] = row['instance_id']
                input_bindings.append(bound(row['mask_vis']))
            owner[count > 1] = -2
            for i in indices:
                row = rows[i]
                r, xyz, perm, sf, K = geometry(row, models[str(row['object_id'])], H)
                np.testing.assert_array_equal(r, oldr[i]); np.testing.assert_array_equal(perm, oldperm[i]); np.testing.assert_array_equal(sf, oldself[i])
                records = [sample_ray(r[k], xyz[k], K, depth, row['depth_scale'], count, owner, row['instance_id']) for k in range(8)]
                assert len(records) == 8
                for rec in records:
                    if rec['depth_raw'] == 0: assert rec['label'] == -1
                    if rec['label'] == 1: assert rec['relation'] == 'FRONT' and rec['owner_type'] == 'OTHER' and rec['owner_count'] == 1
                observations[i] = records
                refs[i], cameras[i], perms[i], selves[i] = r, xyz, perm, sf
            if (g+1) % 100 == 0: print('DEPTH_OWNER_FRAMES', g+1, '/1000', flush=True)
    except Exception as error:
        save(out, 'BUILD_FAILURE_PRESERVED.json', {'completed': False, 'offending_frame': offending,
             'error': repr(error), 'traceback': traceback.format_exc(), 'code': bound(__file__),
             'no_rows_skipped_or_fallback_labels': True, 'new_forward_fit_current_PnP': 0})
        raise
    assert all(x is not None for x in observations)
    arrays = {'id': np.array(ids), 'split': np.array([r['split'] for r in rows]),
              'reference_xy8': np.array(refs), 'reference_camera_xyz_mm8': np.array(cameras),
              'perm_dynamic_to_fixed8': np.array(perms, np.int8), 'convex_self_visible_proxy8_separate': np.array(selves, np.int8),
              'old_local_support_label8_DIAGNOSTIC_ONLY': oldlabel,
              'old_center_M08_DIAGNOSTIC_ONLY': oldm0, 'old_center_M48_DIAGNOSTIC_ONLY': oldm4,
              'human_semantic_gold8': np.full((4200, 8), -1, np.int8)}
    for key in observations[0][0]: arrays[key+'8'] = np.array([[x[key] for x in group] for group in observations])
    npz = out / 'NATIVE8_EXTERNAL_RAY_GT_ONLY.npz'; np.savez_compressed(npz, **arrays)
    bysplit = defaultdict(lambda: {'labels': Counter(), 'reasons': Counter(), 'owner': Counter(), 'old_local_by_ray': Counter(), 'center_M0zero_by_ray': Counter()})
    candidates = {}
    jsonl = out / 'PER_INSTANCE_EXTERNAL_RAY_GT_ONLY.jsonl'
    with jsonl.open('x') as f:
        for i, group in enumerate(observations):
            split = rows[i]['split']; stat = bysplit[split]
            f.write(json.dumps(clean({'id': ids[i], 'split': split, 'reference_xy8': refs[i], 'camera_xyz_mm8': cameras[i], 'ray8': group}), separators=(',', ':'), allow_nan=False) + '\n')
            for k, rec in enumerate(group):
                stat['labels'][str(rec['label'])] += 1; stat['reasons'][rec['reason']] += 1; stat['owner'][rec['owner_type']] += 1
                stat['old_local_by_ray'][f'old={int(oldlabel[i,k])},ray={rec["label"]}'] += 1
                if oldm0[i,k] == 0: stat['center_M0zero_by_ray'][str(rec['label'])] += 1
                if split == 'train':
                    digest = hashlib.sha256(f'{ids[i]}:{k}'.encode()).hexdigest()
                    prev = candidates.get(rec['reason'])
                    if prev is None or digest < prev['selector_sha256']:
                        candidates[rec['reason']] = dict(id=ids[i], row_index=i, slot=k, selector_sha256=digest, record=rec,
                             reference_xy=refs[i][k], camera_xyz_mm=cameras[i][k], source_metadata=rows[i])
    summary = {'completed': True, 'rows': 4200, 'slots': 33600, 'depth_frames': 1000,
               'by_split': dict(bysplit), 'reference_perm_self_parity_mismatches': 0,
               'label_codes': {'1': 'EXTERNAL_OTHER_FIRST_HIT_PROXY', '0': 'CLEAR_TO_VIRTUAL_ENDPOINT_PROXY', '-1': 'UNKNOWN'},
               'not_human_semantic_D_E_S_or_physicalcue_visibility': True,
               'M0_not_read_or_used_in_label_decision': True,
               'old_M0_M4_center_values_from_sealed_arrays_diagnostic_only': True,
               'rayproxy_training_or_current_admission': False, 'new_RGB_model_fit_current_PnP': 0,
               'elapsed_seconds': time.monotonic()-start}
    save(out, 'CORPUS_SUMMARY_KO.json', summary)
    save(out, 'TRAIN_MIN_SHA_BY_REASON_DIAGNOSTIC_SELECTION.json', {'selection': 'minSHA(id,slot) per new typedreason onTRAIN only; descriptive diagnostic, no cohort/model selection', 'records': candidates})
    save(out, 'SOURCE_DEPTH_MASK_BYTE_BINDINGS.json', {'files': input_bindings, 'N': len(input_bindings), 'allRGB_pixels_unread': True})
    save(out, 'CORPUS_SEAL.json', {'completed': True, 'protocol': bound(PROTOCOL), 'code': bound(__file__),
         'outputs': [bound(f) for f in sorted(out.iterdir()) if f.is_file()],
         'rows': 4200, 'slots': 33600, 'new_forward_fit_current_PnP': 0})
    print('EXTERNAL_RAY_CORPUS_COMPLETE', json.dumps(clean(summary)), flush=True)


if __name__ == '__main__': main()
