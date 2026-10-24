"""Bounded one-public-instance native-cuboid/depth-proxy feasibility.

Reads one depth image, its instance masks and official annotation metadata.
No RGB, mesh, model, learned forward, fitting, solver or current data access.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YCB = ROOT.parent / 'query_visibility_public_real_20261022_v63/ycb'
OFFICIAL = ROOT / 'source_audit/YCB_PUBLIC_NATIVE_FEASIBILITY/models_info_OFFICIAL.json'
HELPER = HERE.parent / 'native_context_contract_v2.py'
OUT = HERE / 'ONE_INSTANCE_V2'
DEPTH_MARGIN_MM = 15.
spec = importlib.util.spec_from_file_location('native_math_only', HELPER)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bound(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def depth_proxy_samples(projected8, z8, depth_mm, target_visible, other_visible):
    """9 camera rays to a constant-Z plane, not exact source raycasting."""
    h, w = depth_mm.shape
    angle = np.arange(8) * (2 * np.pi / 8)
    offsets = np.r_[np.zeros((1, 2)), 6 * np.c_[np.cos(angle), np.sin(angle)]]
    rows = []; states = []; bounds = []
    for k in range(8):
        per = []
        for j, off in enumerate(offsets):
            xy = projected8[k] + off
            pixel = np.floor(xy + .5).astype(int)
            continuous = bool(np.isfinite(xy).all() and 0 <= xy[0] < w and 0 <= xy[1] < h)
            inside = bool(continuous and 0 <= pixel[0] < w and 0 <= pixel[1] < h)
            ref_ok = bool(np.isfinite(z8[k]) and z8[k] > 0)
            measured = float(depth_mm[pixel[1], pixel[0]]) if inside else None
            owner = 'OOB'; own = None; other_count = None
            if inside:
                own = bool(target_visible[pixel[1], pixel[0]])
                other_count = sum(bool(m[pixel[1], pixel[0]]) for m in other_visible)
                owner = 'TARGET_VISIBLE' if own else ('OTHER_VISIBLE_INSTANCE' if other_count else 'ENVIRONMENT_OR_UNASSIGNED')
            valid = bool(inside and ref_ok and measured is not None and np.isfinite(measured) and measured > 0)
            gap = float(z8[k] - measured) if valid else None
            blocked = -1
            reason = 'UNKNOWN_OOB_OR_INVALID_REFERENCE'
            if inside and ref_ok and own:
                blocked = 0
                reason = 'TARGET_VISIBLE_VETO_INDEPENDENT_OF_DEPTH_VALIDITY'
            elif valid:
                if gap > DEPTH_MARGIN_MM:
                    blocked = 1
                    reason = 'NON_TARGET_DEPTH_CLOSER_BY_MORE_THAN_15MM'
                elif gap < -DEPTH_MARGIN_MM:
                    blocked = 0
                    reason = 'NON_TARGET_DEPTH_FARTHER_BY_MORE_THAN_15MM'
                else:
                    reason = 'UNKNOWN_DEPTH_WITHIN_15MM_MARGIN'
            elif inside and ref_ok:
                reason = 'UNKNOWN_INVALID_SCENE_DEPTH'
            per.append(blocked)
            rows.append({'native_index': k, 'ray': j, 'sample_x': float(xy[0]), 'sample_y': float(xy[1]),
                'pixel_x': int(pixel[0]), 'pixel_y': int(pixel[1]), 'reference_Z_mm': float(z8[k]),
                'scene_Z_mm': measured, 'depth_gap_mm': gap, 'continuous_and_nearest_inbounds': inside,
                'target_visible_mask': own, 'other_visible_instance_count': other_count,
                'owner_category': owner, 'external_depth_proxy': blocked, 'reason': reason})
        per = np.asarray(per)
        blocked = int((per == 1).sum()); clear = int((per == 0).sum()); unknown = int((per < 0).sum())
        lower, upper = blocked / 9., (blocked + unknown) / 9.
        # A fixed total9 prevents missing rays from becoming unjustified clear.
        state = 'E_DEPTH_RAY_CERTAIN_PROXY' if lower >= .5 else (
            'UNBLOCKED_DEPTH_RAY_CERTAIN_PROXY' if upper < .5 else 'U_DEPTH_RAY_UNCERTAIN')
        states.append(state)
        bounds.append({'native_index': k, 'blocked': blocked, 'clear': clear, 'unknown': unknown,
            'fraction_valid_only_diagnostic': blocked / (blocked + clear) if blocked + clear else None,
            'fraction_interval': [lower, upper], 'external_proxy_certainty': state})
    return rows, states, bounds


def main():
    result_path = OUT / 'ONE_INSTANCE_FEASIBILITY.json'
    assert not OUT.exists()
    OUT.mkdir()
    manifest = YCB / 'train_instances.jsonl'
    records = [json.loads(x) for x in manifest.read_text().splitlines()]
    r = min(records, key=lambda x: (x['scene_id'], x['frame_id'], x['instance_id']))
    assert (r['scene_id'], r['frame_id'], r['instance_id'], r['object_id']) == (0, 1, 0, 8)
    models = json.loads(OFFICIAL.read_text()); info = models[str(r['object_id'])]
    minimum = np.asarray([info['min_' + x] for x in 'xyz'], float)
    size = np.asarray([info['size_' + x] for x in 'xyz'], float)
    assert (size > 0).all()
    fixed = minimum + (H.FIXED_ZUP_SIGNS + 1) / 2 * size
    R = np.asarray(r['cam_R_m2c'], float).reshape(3, 3)
    t = np.asarray(r['cam_t_m2c'], float)
    K = np.asarray(r['K'], float).reshape(3, 3)
    assert np.isfinite(R).all() and abs(np.linalg.det(R) - 1) < 1e-6
    camera_model = np.linalg.solve(R, -t)
    camera_fixed = fixed @ R.T + t
    assert (camera_fixed[:, 2] > 0).all()
    homogeneous = camera_fixed @ K.T
    projected_fixed = homogeneous[:, :2] / homogeneous[:, 2:]
    perm, normal_cos = H.normal_facing_permutation(fixed, projected_fixed, camera_model)
    dynamic = fixed[perm]; camera_dynamic = camera_fixed[perm]; projected = projected_fixed[perm]
    self_visible = H.convex_self_visible_proxy(dynamic, camera_model)
    direct = (np.c_[dynamic, np.ones(8)] @ np.c_[R, t].T)
    assert np.array_equal(direct, camera_dynamic) or np.max(np.abs(direct - camera_dynamic)) < 1e-12
    directory = Path(r['depth']).parent.parent
    camera_path = directory / 'scene_camera.json'; gt_path = directory / 'scene_gt.json'
    scene_camera = json.loads(camera_path.read_text())[str(r['frame_id'])]
    scene_gt = json.loads(gt_path.read_text())[str(r['frame_id'])]
    assert scene_gt[r['instance_id']]['obj_id'] == r['object_id']
    np.testing.assert_array_equal(scene_gt[r['instance_id']]['cam_R_m2c'], r['cam_R_m2c'])
    np.testing.assert_array_equal(scene_gt[r['instance_id']]['cam_t_m2c'], r['cam_t_m2c'])
    np.testing.assert_array_equal(scene_camera['cam_K'], r['K'])
    assert scene_camera['depth_scale'] == r['depth_scale'] == .1
    depth_raw = np.asarray(Image.open(r['depth']))
    assert depth_raw.dtype == np.uint16 and list(depth_raw.shape) == [480, 640]
    depth_mm = depth_raw.astype(np.float64) * r['depth_scale']
    visible = np.asarray(Image.open(r['mask_vis'])) > 0
    full = np.asarray(Image.open(r['mask_full'])) > 0
    assert visible.shape == full.shape == depth_raw.shape and not (visible & ~full).any()
    other_paths = [directory / 'mask_visib' / f'{r["frame_id"]:06d}_{j:06d}.png'
                   for j in range(len(scene_gt)) if j != r['instance_id']]
    other = [np.asarray(Image.open(p)) > 0 for p in other_paths]
    rows, external, certainty = depth_proxy_samples(projected, camera_dynamic[:, 2], depth_mm, visible, other)
    semantic_proxy = []
    for k in range(8):
        ref_valid = bool(0 <= projected[k, 0] < 640 and 0 <= projected[k, 1] < 480 and camera_dynamic[k, 2] > 0)
        semantic_proxy.append('F_REFERENCE_OUTSIDE_PROXY' if not ref_valid else
            'S_CONVEX_SELF_HIDDEN_PROXY' if not self_visible[k] else
            'E_EXTERNAL_DEPTH_PROXY' if external[k] == 'E_DEPTH_RAY_CERTAIN_PROXY' else
            'D_CONVEX_UNBLOCKED_DEPTH_PROXY' if external[k] == 'UNBLOCKED_DEPTH_RAY_CERTAIN_PROXY' else 'U_DEPTH_RAYS')
    v_y, v_x = np.nonzero(visible); f_y, f_x = np.nonzero(full)
    projected_min = projected.min(0); projected_max = projected.max(0)
    def inside_count(x, y):
        return int(((x >= projected_min[0]) & (x <= projected_max[0]) &
                    (y >= projected_min[1]) & (y <= projected_max[1])).sum())
    measured_target = depth_mm[visible & (depth_mm > 0)]
    zmin, zmax = camera_dynamic[:, 2].min(), camera_dynamic[:, 2].max()
    output_csv = OUT / 'ONE_INSTANCE_NINE_RAYS.csv'
    with output_csv.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    references_path = OUT / 'ONE_INSTANCE_NATIVE_REFERENCES_GT_ONLY.npz'
    np.savez_compressed(references_path, native_index=np.arange(8), model_AABB_dynamic=dynamic,
        camera_AABB_dynamic=camera_dynamic, reference_projected8=projected,
        perm_dynamic_to_fixed=perm, convex_self_visible_proxy8=self_visible,
        certified_semantic_visibility8=np.full(8, -1, np.int8), pixel_presence8=np.full(8, -1, np.int8))
    result = {'completed': True, 'feasibility': 'EXECUTABLE_BOUNDED_WEAK_PROXY_ONLY',
        'instance': {k: r[k] for k in ['scene_id', 'frame_id', 'instance_id', 'object_id', 'split', 'image_sha256']},
        'selection': 'Lexicographicallyfirst existingTRAINinstance, fixedbeforemodelsmetadata/proxycomputation',
        'models_info': info, 'units': 'allmodelcoordinates,t,depth inmm; rawdepth*.1',
        'K': K.tolist(), 'camera_model_mm': camera_model.tolist(),
        'R_orthogonality_max_error': float(np.max(np.abs(R @ R.T - np.eye(3)))),
        'fixed_AABB_model8_mm': fixed.tolist(), 'perm_dynamic_to_fixed': perm.tolist(),
        'vertical_normal_cosines': normal_cos.tolist(), 'native_reference8_xy': projected.tolist(),
        'native_reference8_camera_Z_mm': camera_dynamic[:, 2].tolist(),
        'convex_self_visible_proxy8': self_visible.tolist(), 'per_reference_depth_proxy': certainty,
        'combined_weak_reference_proxy8': semantic_proxy,
        'rays_count': len(rows), 'ray_summary': {str(v): sum(x['external_depth_proxy'] == v for x in rows) for v in [-1, 0, 1]},
        'owner_summary': {k: sum(x['owner_category'] == k for x in rows) for k in sorted(set(x['owner_category'] for x in rows))},
        'target_visible_veto_count': sum(x['reason'].startswith('TARGET_VISIBLE_VETO') for x in rows),
        'alignment_diagnostic': {'projected_AABB_minmax': [projected_min.tolist(), projected_max.tolist()],
            'visible_mask_pixels': int(visible.sum()), 'visible_pixels_with_centers_inside_projected_AABB': inside_count(v_x, v_y),
            'full_mask_pixels': int(full.sum()), 'full_pixels_with_centers_inside_projected_AABB': inside_count(f_x, f_y),
            'target_depth_valid': len(measured_target), 'target_depth_missing': int(visible.sum() - len(measured_target)),
            'camera_AABB_depth_range_mm': [float(zmin), float(zmax)],
            'target_observed_depth_quantiles_mm': np.quantile(measured_target, [0, .1, .5, .9, 1]).tolist(),
            'valid_target_depth_within_AABB_Z_range': int(((measured_target >= zmin) & (measured_target <= zmax)).sum()),
            'limit': 'bbox/depthrange sanityonly; virtualvertices neednotlieonmesh or matchmeasuredsurfaceZ'},
        'unknown_and_ownership_contract': {'invalid_depth_or_OOB': 'externalmeasurementU unlessinboundsGTtargetvisiblemask1providesveto; OOBalwaysU',
            'target_mask1': 'externalfalseevenwithinvaliddepth; ownervetoisthepositiveGTvisiblemaskobservation',
            'mask0_alone': 'neverE; requiresvaliddepthcloserby>15mm; depthfartherby>15mmclear; near/equalU',
            'non_target_environment': 'includedasweakexternaldepthblocker; notpretendedotherinstancegold',
            'fraction_uncertainty': '[blocked/9,(blocked+unknown)/9]; missingraysnotdroppedforcertainty',
            'depth_noise_margin_mm': 15, 'margin_limit': 'fixedsourceproposal: cameraZgap; inspiredbyBOP15mmvisibilitytol butnotidenticalraydistancealgorithm/sensorcalibration'},
        'not_claimed': ['humansemanticD/E/Sgold', 'querypixelpresence', 'exactSYNTHraycastparity', 'actualdeployedinitializerperformance'],
        'future_query_producer': 'GTjitter ifusedmustbeEMULATED_SOURCE_INITIALIZER; no qproducedinthisboundedcheck',
        'native_scope': 'virtualmodelAABB; exactmeshpointnotrequired; modelZupconvention, notworldgravitymeasurement',
        'bindings': [bound(x) for x in [Path(__file__), HERE / 'YCB_NATIVE_WEAK_PROXY_PROPOSAL_V2.json',
            HELPER, manifest, OFFICIAL, Path(r['depth']), Path(r['mask_vis']), Path(r['mask_full']),
            camera_path, gt_path] + other_paths],
        'outputs': [bound(output_csv), bound(references_path)],
        'model_forward': 0, 'fit': 0, 'PnP': 0, 'current_reads': 0,
        'RGB_body_reads': 0, 'new_download_in_this_script': 0, 'instances_generated': 1}
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['feasibility', 'perm_dynamic_to_fixed', 'ray_summary',
        'owner_summary', 'combined_weak_reference_proxy8', 'alignment_diagnostic']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
