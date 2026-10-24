"""Repair the explicitly rejected V1 GT frame, preserving all original data.

No RGB/features/models/solver. V1 verified source bytes and all small cache
array slices. This version independently composes the source pose frame,
replays all60000 permutations and compares all8024 prior reference proxies.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import native_context_contract_v2 as H

HERE = Path(__file__).resolve().parent
OUT = HERE / 'BUILD_V2'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bound(p):
    return {'path': str(p), 'sha256': sha(p), 'bytes': Path(p).stat().st_size}


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write('\n')


def main():
    start = time.monotonic()
    lock = json.loads((HERE / 'SOURCE_PACK_BUILD_LOCK_V2.json').read_text())
    for b in lock['bindings']:
        assert sha(b['path']) == b['sha256']
    OUT.mkdir(exist_ok=False)
    v1 = json.loads((HERE / 'TYPED_SOURCE_PACK_SEAL.json').read_text())
    for b in v1['outputs']:
        assert sha(b['path']) == b['sha256']
    old = np.load(HERE / 'TYPED_SOURCE_PACK_GT_ONLY.npz', allow_pickle=False)
    a = {k: old[k] for k in old.files}
    T = a.pop('source_world_to_camera')
    q = a['cuboid_world_dynamic8']; center = q.mean(1)
    A = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
    G = np.tile(np.eye(4), (len(q), 1, 1)); G[:, :3, :3] = A
    G[:, :3, 3] = -center @ A.T
    world_to_camera = T @ G
    centered_yup = np.einsum('ij,nkj->nki', A, q - center[:, None, :])
    camera_points = np.einsum('nij,nkj->nki', T[:, :3, :3], centered_yup) + T[:, None, :3, 3]
    alternative = np.einsum('nij,nkj->nki', world_to_camera[:, :3, :3], q) + world_to_camera[:, None, :3, 3]
    assert np.max(np.abs(camera_points - alternative)) < 1e-12
    projections = np.einsum('nij,nkj->nki', a['K'], camera_points)
    projections = projections[:, :, :2] / projections[:, :, 2:]
    error = np.linalg.norm(projections - a['reference_projected8'], axis=2)
    assert np.isfinite(error).all() and error.max() < .01
    assert (camera_points[:, :, 2] > 0).all()
    # Source R is rounded, not exactly orthonormal; recover center by inverse.
    cam_recovered = -np.linalg.solve(world_to_camera[:, :3, :3], world_to_camera[:, :3, 3, None]).squeeze(-1)
    camera_error = np.linalg.norm(cam_recovered - a['camera_world'], axis=1)
    assert camera_error.max() < 2e-6
    a['source_centered_yup_to_camera'] = T
    a['world_to_camera_composed'] = world_to_camera
    a['cuboid_centered_yup_dynamic8'] = centered_yup
    a['reference_camera_depth8'] = camera_points[:, :, 2]
    with np.load(HERE / 'TYPED_SOURCE_INPUT_COORDS_ONLY.npz', allow_pickle=False) as raw_file:
        raw = {k: raw_file[k] for k in raw_file.files}
    counts = defaultdict(Counter)
    normal = Counter(); minority_rear_x_reverse = []
    perm_counts = Counter(); minimum_cosine_margin = 1.
    for i in range(len(q)):
        p = a['perm_dynamic_to_writer_fixed8'][i]
        fixed = H.inverse_stored_permutation(q[i], p)
        fp = H.inverse_stored_permutation(a['reference_projected8'][i], p)
        reconstructed, cos = H.normal_facing_permutation(fixed, fp, a['camera_world'][i])
        assert np.array_equal(reconstructed, p), str(a['sample_id'][i])
        normal['full_permutation_exact'] += 1
        normal['front_side_is_cosine_maximum'] += 1
        minimum_cosine_margin = min(minimum_cosine_margin, float(np.sort(cos)[-1] - np.sort(cos)[-2]))
        if a['reference_projected8'][i, 4, 0] > a['reference_projected8'][i, 5, 0]:
            minority_rear_x_reverse.append(str(a['sample_id'][i]))
        a['reference_proxy_state8'][i], a['reference_inframe8'][i] = H.source_reference_proxy(
            a['reference_projected8'][i], a['reference_camera_depth8'][i], raw['raw_hw'][i],
            a['convex_self_visible_proxy8'][i], a['external_9ray_fraction9'][i, :8])
        group = str(a['source'][i]) + '/' + str(a['partition'][i])
        c = counts[group]; c['frames'] += 1
        c['detected'] += int(raw['detected'][i]); c['matched'] += int(a['detector_matched_to_source_target'][i])
        for state, name in enumerate(H.PROXY_NAMES):
            m = a['reference_proxy_state8'][i] == state
            c[name] += int(m.sum())
            c[name + '_valid_query'] += int((m & raw['query_native_valid9'][i, :8]).sum())
        perm_counts[','.join(map(str, p))] += 1
    inventory = HERE.parents[1] / 'query_visibility_visible_preservation_20261023_v67/source_audit/SYNTH_REFERENCE_ANCHOR_INVENTORY.csv'
    old_names = {'D_UNBLOCKED_RAY_CONVEX_PROXY': 0, 'E_RAY_GE_HALF_PROXY': 1,
                 'S_CONVEX_PROXY': 2, 'S_CONVEX_SELF_PROXY': 2,
                 'S_CONVEX_SELF_HIDDEN_PROXY': 2, 'F_REFERENCE_PROXY': 3}
    lookup = {str(s): i for i, s in enumerate(a['sample_id'])}
    parity = Counter(); unknown_old_names = set()
    for r in csv.DictReader(inventory.open()):
        i = lookup[r['id']]; k = int(r['native_corner'])
        state = r['diagnostic_conditional_proxy_state']
        if state not in old_names:
            unknown_old_names.add(state); continue
        assert old_names[state] == a['reference_proxy_state8'][i, k], (r['id'], k, state)
        np.testing.assert_allclose([float(r['reference_x']), float(r['reference_y'])],
            a['reference_projected8'][i, k], rtol=0, atol=0)
        parity['prior1003_reference_and_proxy_exact'] += 1
    assert not unknown_old_names, unknown_old_names
    assert parity['prior1003_reference_and_proxy_exact'] == 8024
    np.savez_compressed(OUT / 'TYPED_SOURCE_PACK_GT_ONLY.npz', **a)
    output = OUT / 'TYPED_SOURCE_PACK_GT_ONLY.npz'
    projection_by_source = {str(s): {'max_error_px': float(error[a['source'] == s].max()),
        'p99_error_px': float(np.quantile(error[a['source'] == s], .99))} for s in np.unique(a['source'])}
    result = {'schema': 'native_source_typed_pack_v2', 'completed': True, 'PASS': True,
        'all_source_rows': len(q), 'input_coords_unchanged': bound(HERE / 'TYPED_SOURCE_INPUT_COORDS_ONLY.npz'),
        'row_map': bound(HERE / 'TYPED_SOURCE_ROW_MAP.jsonl'), 'GT_pack': bound(output),
        'normal_rule': dict(normal), 'minimum_front_max_cosine_margin': minimum_cosine_margin,
        'rear_image_x_reversal_count': len(minority_rear_x_reverse),
        'rear_image_x_reversal_ids': minority_rear_x_reverse,
        'rear_rule': 'Preserve front0..3 correspondence through cuboid depth edges, NOT independent projected-X sorting',
        'perm_counts': dict(perm_counts), 'projection_by_source': projection_by_source,
        'camera_world_recovery_max_error_m': float(camera_error.max()),
        'source_pose_frame': {'raw_annotation_field': 'pose_transform',
            'acts_on': 'centered Y-up A @ (world-cuboid_mean), A=[[1,0,0],[0,0,1],[0,-1,0]]',
            'composed_world_to_camera': 'T_source @ [A,-A*center;0,0,0,1]',
            'PnP_calls': 0},
        'source_partition_counts': {k: dict(v) for k, v in counts.items()},
        'prior1003_parity': dict(parity),
        'proxy_names': list(H.PROXY_NAMES), 'external_fraction_threshold_fixed': .5,
        'pixel_support': 'UNKNOWN -1, separate from reference proxy; no dense target mask in source annotation',
        'semantic_gt': 'UNKNOWN -1; native cuboid proxy is trainable source weak supervision, not human semantic gold',
        'virtual_mesh_surface_membership_required': False,
        'all_inputs_preserved': True, 'new_forward': 0, 'new_fit': 0, 'PnP': 0,
        'private_reads': 0, 'RGB_reads': 0, 'P3_P4_reads': 0,
        'V1_rejected_and_preserved': bound(HERE / 'V1_PACK_REJECTION.json'),
        'bindings': lock['bindings'] + [bound(inventory)],
        'elapsed_seconds': time.monotonic() - start}
    write(OUT / 'TYPED_SOURCE_PACK_SEAL.json', result)
    print(json.dumps({k: result[k] for k in ['completed', 'PASS', 'normal_rule', 'rear_image_x_reversal_count',
        'projection_by_source', 'prior1003_parity', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
