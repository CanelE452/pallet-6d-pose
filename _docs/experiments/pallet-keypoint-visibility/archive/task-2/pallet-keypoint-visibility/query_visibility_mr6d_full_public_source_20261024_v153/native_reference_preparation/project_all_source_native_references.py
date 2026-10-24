"""Model-free provided-pose MR6D references; original V74 native math unchanged.

`project` reads sealed selection plus provided JSON metadata only.
`render` additionally requires completed full-source download and reads source RGB.
Neither command reads predictions, fits pose, changes coordinates, or labels D/E/S.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json, math
import numpy as np

H = Path(__file__).resolve().parent
V = H.parent
P = V.parent
SRC = P / 'query_visibility_mr6d_public_native_intake_20261024_v151'
D = SRC / 'provided_metadata'
F = V / 'full_source_v1'
O = H / 'provided_reference_v1'
V74 = P / 'query_visibility_loco_native_context_20261023_v74/source_contract/native_context_contract_v2.py'
POLICY = '46b3b077edfc78d113dcd0b38771bb3df96053a3741002ac774bc1c9866e2334'
SPLIT_SHA = '3b163ce06df9c437ceb464689046471a003c35e19289f3283a64cd397f52dca7'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def bind(p):
    return {'path': str(p), 'sha256': sha(p)}

def read(p):
    return json.loads(Path(p).read_text())

def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def write_rows(p, rows):
    with Path(p).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def ast_contract():
    wanted = {'FIXED_ZUP_SIGNS', 'VERTICAL_FIXED_FACES', 'ALL_DYNAMIC_FACES', 'NATIVE_NAMES'}
    funcs = {'outward_normals', 'normal_facing_permutation', 'convex_self_visible_proxy'}
    nodes = []
    for n in ast.parse(V74.read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted for t in n.targets):
            nodes.append(n)
        if isinstance(n, ast.FunctionDef) and n.name in funcs:
            nodes.append(n)
    module = ast.Module(body=nodes, type_ignores=[])
    ns = {'np': np}
    exec(compile(module, str(V74), 'exec'), ns)
    assert funcs.issubset(ns) and wanted.issubset(ns)
    return ns, {'original_code': bind(V74),
                'pure_selected_AST_sha256': hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest(),
                'functions': sorted(funcs), 'constants': sorted(wanted),
                'exact_original_AST_no_math_edits': True}

def project(points, R, t, K):
    cam = np.asarray(points, np.float64) @ R.T + t
    ip = cam @ K.T
    if not np.isfinite(cam).all() or np.any(ip[:, 2] == 0):
        raise ValueError('Nonfinite or zero-depth supplied projection')
    return ip[:, :2] / ip[:, 2:], cam

def authorize_selection():
    assert sha(V / 'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json') == POLICY
    split_path = SRC / 'ROOT_MR6D_SCENE_SPLIT_BEFORE_ANY_FIT_OR_EXTRA_PIXELS.json'
    assert sha(split_path) == SPLIT_SHA
    source_selection_path = F / 'FULL_SOURCE_SELECTION_BEFORE_NEW_PAYLOADS.json'
    expected = []
    gt_bindings = {}
    for scene in ['000000', '000001', '000002', '000005', '000006']:
        gt_path = D / 'val' / scene / 'scene_gt.json'
        gt = read(gt_path)
        gt_bindings[scene] = bind(gt_path)
        for frame_key in sorted(gt, key=int):
            indices = [ix for ix, g in enumerate(gt[frame_key]) if g['obj_id'] == 1]
            if indices:
                expected.append((scene, str(frame_key), indices))
    assert len(expected) == 230 and sum(len(x[2]) for x in expected) == 303
    independent_path = H / 'INDEPENDENT_ALL230_NUMERIC_FRAME_ORDER_FROM_PROVIDED_GT.json'
    derived = []
    for fi, (scene, frame_key, indices) in enumerate(expected):
        derived.append({'frame_index': fi, 'id': f'V153_MR6D_val_{scene}_{int(frame_key):06d}',
                        'source': 'MR6D', 'source_id': f'val/{scene}:{frame_key}',
                        'split': 'SOURCE_VALIDATION' if scene == '000001' else 'SOURCE_DEVELOPMENT',
                        'official_split': 'val', 'scene': scene, 'frame_key': frame_key,
                        'frame_filename_stem': f'{int(frame_key):06d}',
                        'obj1_gt_instance_indices': indices,
                        'source_files': {'RGB': f'val/{scene}/rgb/{int(frame_key):06d}.jpg'}})
    independent = {'completed': True, 'root_policy': bind(V / 'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json'),
                   'fixed_scene_split': bind(split_path), 'source_GT_metadata': gt_bindings,
                   'rule': 'Exhaustive obj1 GT frames; scene integer then original JSON frame-key integer; no query/visibility/score filtering',
                   'frames': derived, 'RGB_or_prediction_read': False,
                   'not_a_fake_download_or_initializer_seal': True}
    if independent_path.exists():
        assert read(independent_path) == independent
    else:
        write(independent_path, independent)
    if source_selection_path.exists():
        selection_path = source_selection_path
        frames = read(source_selection_path)['frames']
    else:
        selection_path = independent_path
        frames = derived
    assert len(frames) == 230
    for fi, (s, e) in enumerate(zip(frames, expected)):
        scene, frame_key, indices = e
        assert s['frame_index'] == fi
        assert s['id'] == f'V153_MR6D_val_{scene}_{int(frame_key):06d}'
        assert s['source'] == 'MR6D' and s['source_id'] == f'val/{scene}:{frame_key}'
        assert s['official_split'] == 'val' and s['scene'] == scene and str(s['frame_key']) == frame_key
        assert s['split'] == ('SOURCE_VALIDATION' if scene == '000001' else 'SOURCE_DEVELOPMENT')
        assert s['obj1_gt_instance_indices'] == indices
        assert s['source_files']['RGB'] == f'val/{scene}/rgb/{int(frame_key):06d}.jpg'
    return selection_path, frames

def run_project():
    selection_path, selected = authorize_selection()
    ns, proof = ast_contract()
    mi_path = D / 'models/models_info.json'
    mi = read(mi_path)['1']
    dims = np.array([mi['size_x'], mi['size_y'], mi['size_z']], np.float64)
    center = np.array([mi['min_x'], mi['min_y'], mi['min_z']], np.float64) + dims / 2
    fixed = ns['FIXED_ZUP_SIGNS'] * dims / 2 + center
    assert np.array_equal(dims, [800, 1200, 144]) and np.array_equal(center, np.zeros(3))
    cad_path = SRC / 'native_readiness_review/actual_tiny_audit/RESULTS.json'
    cad = read(cad_path)
    assert len(cad['mesh']['canonical8_geometry_support']) == 8
    assert all(x['nearest_triangle_surface_distance'] == 0 for x in cad['mesh']['canonical8_geometry_support'])
    O.mkdir(exist_ok=False)
    references, centroids, frames, instances_all, metadata_bindings = [], [], [], [], {}
    for s in selected:
        fi, scene, key = s['frame_index'], s['scene'], str(s['frame_key'])
        gp = D / 'val' / scene / 'scene_gt.json'
        cp = D / 'val' / scene / 'scene_camera.json'
        if scene not in metadata_bindings:
            metadata_bindings[scene] = {'GT': bind(gp), 'camera': bind(cp)}
        gt, ca = read(gp)[key], read(cp)[key]
        K = np.asarray(ca['cam_K'], np.float64).reshape(3, 3)
        assert np.isfinite(K).all() and np.array_equal(K[2], [0, 0, 1])
        # Prior fixed10 source RGB are 1280x720. This expected source image shape
        # is verified against every original RGB in the separate render stage.
        raw_hw = [720, 1280]
        Hh, W = raw_hw
        instances = []
        for ix in s['obj1_gt_instance_indices']:
            g = gt[ix]
            assert g['obj_id'] == 1
            R = np.asarray(g['cam_R_m2c'], np.float64).reshape(3, 3)
            t = np.asarray(g['cam_t_m2c'], np.float64).reshape(3)
            assert np.isfinite(R).all() and np.isfinite(t).all()
            xy, cam = project(fixed, R, t, K)
            cxy, ccam = project(center[None], R, t, K)
            camera_obj = np.linalg.solve(R, -t)
            perm, cos = ns['normal_facing_permutation'](fixed, xy, camera_obj)
            assert sorted(perm.tolist()) == list(range(8))
            dynamic, d3, dcam = xy[perm], fixed[perm], cam[perm]
            proxy = ns['convex_self_visible_proxy'](d3, camera_obj)
            margin = float(np.sort(cos)[-1] - np.sort(cos)[-2])
            inside = ((dcam[:, 2] > 0) & (dynamic[:, 0] >= 0) & (dynamic[:, 0] < W)
                      & (dynamic[:, 1] >= 0) & (dynamic[:, 1] < Hh))
            lr = float(dynamic[1, 0] - dynamic[0, 0])
            assert lr >= 0
            common = {k: s[k] for k in ['frame_index', 'id', 'source', 'source_id', 'split', 'official_split', 'scene', 'frame_key']}
            common.update(gt_instance_index=ix, obj_id=1, instance_id=f"{s['id']}:GT{ix}")
            uncertainty = ('Nominal CAD corner lies on source CAD. Actual pallet dimensions and provided pose are '
                           'low-centimeter approximate: deterministic reference is not certified physical pixel gold. '
                           'Source deck +Z is anatomical and retained for upright objects.')
            pose_diag = {'provided_R_finite': True, 'provided_t_finite': True,
                         'R_orthogonality_max_abs': float(np.max(np.abs(R.T @ R - np.eye(3)))),
                         'R_determinant': float(np.linalg.det(R)),
                         'camera_Rtranspose_inverse_difference_mm': float(np.linalg.norm(camera_obj - (-R.T @ t))),
                         'provided_values_unchanged': True}
            inst = dict(common, raw_hw=raw_hw, provided_R_m2c=R.tolist(), provided_t_m2c_mm=t.tolist(),
                        provided_K=K.tolist(), camera_in_object_exact_linear_inverse=camera_obj.tolist(),
                        pose_diagnostics=pose_diag, canonical_fixed_corners_mm=fixed.tolist(),
                        canonical_fixed_projected8=xy.tolist(), dynamic_to_fixed_permutation=perm.tolist(),
                        normal_face_scores=cos.tolist(), chosen_face_index=int(np.argmax(cos)),
                        chosen_face_fixed_indices=ns['VERTICAL_FIXED_FACES'][int(np.argmax(cos))].tolist(),
                        front_runner_margin=margin, front_top_imageX_separation_px=lr,
                        exact_tie_original_writer_policy_uncertified=bool(margin <= 1e-10),
                        dynamic_projected8=dynamic.tolist(), dynamic_camera_Z8_mm=dcam[:, 2].tolist(),
                        inframe8=inside.tolist(), convex_self_visible_proxy8_only=proxy.tolist(),
                        centroid_projected=cxy[0].tolist(), centroid_camera_Z_mm=float(ccam[0, 2]),
                        semantic_states_initial=['U_UNREVIEWED'] * 8,
                        structural_reference_uncertainty=uncertainty,
                        source_masks_not_exclusive_owner_truth=True)
            for k in range(8):
                references.append(dict(common, native_slot=k, native_name=ns['NATIVE_NAMES'][k],
                    fixed_canonical_index=int(perm[k]), reference_model_XYZ_mm=d3[k].tolist(),
                    reference_camera_XYZ_mm=dcam[k].tolist(), reference_xy=dynamic[k].tolist(),
                    continuous_inframe=bool(inside[k]), positive_camera_Z=bool(dcam[k, 2] > 0),
                    raw_hw=raw_hw, front_face_scores=cos.tolist(), front_runner_margin=margin,
                    front_top_imageX_separation_px=lr, dynamic_to_fixed_permutation=perm.tolist(),
                    convex_self_visible_proxy_only=bool(proxy[k]),
                    exact_tie_original_writer_policy_uncertified=bool(margin <= 1e-10),
                    semantic_state='U_UNREVIEWED', human_gold=False, train_ready=False,
                    reference_point_uncertainty=uncertainty, source_masks_not_exclusive_owner_truth=True,
                    pose_diagnostics=pose_diag))
            centroids.append(dict(common, centroid_model_XYZ_mm=center.tolist(), centroid_reference_xy=cxy[0].tolist(),
                                  camera_Z_mm=float(ccam[0, 2]), separately_projected_3D_centroid_not_mean_2D_corners=True))
            instances.append(inst)
            instances_all.append(inst)
        frames.append(dict(s, raw_hw=raw_hw, instances=instances))
    assert len(references) == 2424 and len(centroids) == 303 and len(frames) == 230
    assert sum(x['split'] == 'SOURCE_DEVELOPMENT' for x in frames) == 185
    assert sum(x['split'] == 'SOURCE_VALIDATION' for x in frames) == 45
    write_rows(O / 'ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl', references)
    write_rows(O / 'ALL303_PROJECTED_CENTROIDS_GT_ONLY.jsonl', centroids)
    write(O / 'FRAME230_INSTANCE303_GEOMETRY.json', {'completed': True, 'frames': frames})
    counts = {'frames': len(frames), 'GT_instance_observations_not_unique_physical_objects': len(instances_all),
              'native_references': len(references), 'centroids': len(centroids),
              'nominal_inframe': sum(x['continuous_inframe'] for x in references),
              'nominal_OOF': sum(not x['continuous_inframe'] for x in references),
              'all_semantic_U_UNREVIEWED': len(references),
              'source_development_frames': 185, 'source_validation_frames': 45,
              'development_references': sum(x['split'] == 'SOURCE_DEVELOPMENT' for x in references),
              'validation_references': sum(x['split'] == 'SOURCE_VALIDATION' for x in references)}
    result = {'completed': True, 'PASS': True, 'UTC': datetime.now(timezone.utc).isoformat(),
              'code': bind(Path(__file__)), 'root_policy': bind(V / 'ROOT_POLICY_BEFORE_NEW_SOURCE_BODIES.json'),
              'selection': bind(selection_path), 'fixed_scene_split': bind(SRC / 'ROOT_MR6D_SCENE_SPLIT_BEFORE_ANY_FIT_OR_EXTRA_PIXELS.json'),
              'source_metadata': metadata_bindings, 'models_info': bind(mi_path), 'CAD_evidence': bind(cad_path),
              'exact_original_V74_AST': proof,
              'references': bind(O / 'ALL2424_DYNAMIC_NATIVE_REFERENCE_GT_ONLY.jsonl'),
              'centroids': bind(O / 'ALL303_PROJECTED_CENTROIDS_GT_ONLY.jsonl'),
              'frames_geometry': bind(O / 'FRAME230_INSTANCE303_GEOMETRY.json'), 'counts': counts,
              'minimum_front_runner_margin': min(x['front_runner_margin'] for x in instances_all),
              'minimum_front_top_LR_px': min(x['front_top_imageX_separation_px'] for x in instances_all),
              'front_tie_count_at_1e_minus10': sum(x['exact_tie_original_writer_policy_uncertified'] for x in instances_all),
              'max_R_orthogonality_error': max(x['pose_diagnostics']['R_orthogonality_max_abs'] for x in instances_all),
              'provided_R_T_K_or_pixels_changed': False, 'RGB_depth_mask_body_read': False,
              'expected_raw_HW_basis': '1280x720 prior fixed10 public source images; all230 actual shapes checked in the separate source render stage',
              'query_or_model_outputs_read': False, 'semantic_labels_created': 0,
              'new_model_fit_PnP_download_private_access': 0,
              'limits': ['Provided pose and CAD define nominal roles, not <=3px physical gold.',
                         'Convex self visibility is a separate proxy, never automatic semantic S.',
                         'Source full/visible masks are nonexclusive and not native labels.',
                         'All native roles start U_UNREVIEWED; actual structural review follows source RGB completion.']}
    write(O / 'REFERENCE_PROJECTION_COMPLETE.json', result)
    print(json.dumps({'seal': bind(O / 'REFERENCE_PROJECTION_COMPLETE.json'), 'counts': counts}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['project'])
    args = parser.parse_args()
    run_project()
