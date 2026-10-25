"""Analytic invented-array tests only; no public frame/mesh/model reads."""
import hashlib
import json
from pathlib import Path
import types
import numpy as np

H = Path(__file__).absolute().parent
raw = (H/'rgbd_geometry.py').read_bytes()
m = types.ModuleType('toy_geometry_only')
m.__file__ = str(H/'rgbd_geometry.py')
exec(compile(raw, m.__file__, 'exec'), m.__dict__)
results = []


def test(name, call):
    call()
    results.append({'name': name, 'PASS': True})


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected explicit rejection')


def base_triangle(z=2):
    return np.array([[0, 0, z], [2*z, 0, z], [0, 2*z, z]], np.float64)


F = np.array([[0, 1, 2]], np.int64)
I = np.eye(4)
K = np.eye(3)


def parser():
    text = 'mtllib inert.mtl\nv 0 0 2\nv 4 0 2\nv 0 4 2\nf -3/8/9 -2/7/8 -1/6/7\n'
    v, f, r = m.parse_obj_text(text)
    assert np.array_equal(v, base_triangle()) and np.array_equal(f, F)
    assert r['material_declarations_inert'] == [{'tag': 'mtllib', 'value': 'inert.mtl', 'inert': True}]
    rejected(lambda: m.parse_obj_text(text+'f 1 2 3 1\n'))
    rejected(lambda: m.parse_obj_text(text+'f 0 2 3\n'))
    rejected(lambda: m.parse_obj_text('v nan 0 0\nf 1 1 1\n'))
test('triangular_OBJ_slash_negative_indices_inert_material_and_rejections', parser)


def constant_triangle():
    d, s, r = m.render_depth(base_triangle(), F, I, K, (4, 4))
    expected = np.zeros((4, 4), bool)
    expected[0, :3] = True; expected[1, :2] = True; expected[2, 0] = True
    assert np.array_equal(s, expected) and np.array_equal(d[s], np.full(6, 2.))
    assert r['supported_pixels'] == 6 and np.isnan(d[~s]).all()
    uv, front = m.project_points(base_triangle(), K)
    assert np.array_equal(uv, [[0, 0], [2, 0], [0, 2]]) and front.all()
test('known_triangle_integer_pixel_centers_constant_depth', constant_triangle)


def reciprocal():
    v = np.array([[0, 0, 1], [8, 0, 2], [0, 16, 4]], np.float64)
    d, s, _ = m.render_depth(v, F, I, K, (5, 5))
    assert s[1, 1] and abs(d[1, 1] - 16/11) < 1e-14
    assert abs(d[1, 1] - 2.) > .5  # Incorrect affine-z interpolation gives 2.
    assert d[0, 0] == 1 and d[0, 4] == 2 and d[4, 0] == 4
test('perspective_correct_reciprocal_depth_analytic_16_over11', reciprocal)


def occlusion():
    v = np.concatenate((base_triangle(4), base_triangle(2)))
    f = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    d, s, _ = m.render_depth(v, f, I, K, (4, 4))
    d2, s2, _ = m.render_depth(v, f[::-1], I, K, (4, 4))
    assert np.array_equal(s, s2) and np.array_equal(d[s], d2[s2])
    assert np.array_equal(d[s], np.full(6, 2.))
test('occlusion_min_positive_z_independent_of_near_far_face_order', occlusion)


def boundaries():
    outside = base_triangle()+np.array([20, 20, 0])
    d, s, r = m.render_depth(outside, F, I, K, (4, 4))
    assert not s.any() and np.isnan(d).all() and r['fully_out_of_frame_triangles'] == 1
    partial = np.array([[-4, 0, 2], [4, 0, 2], [0, 4, 2]], np.float64)
    d, s, _ = m.render_depth(partial, F, I, K, (3, 3))
    assert s[0, 0] and s[0, 2] and s[2, 0] and not s[2, 2]
test('out_of_frame_and_partial_boundary_without_coordinate_correction', boundaries)


def pose_inverse():
    v = base_triangle().copy(); v[:, 2] = 0
    T = np.eye(4); T[2, 3] = 2
    observed = np.full((4, 4), 2., np.float64); mask = np.ones((4, 4), bool)
    a, r = m.diagnostic_pair(v, F, T, K, (4, 4), observed, mask)
    assert a['T']['support'].sum() == 6 and a['inverse_T']['support'].sum() == 0
    assert r['winner_selected'] is None and r['pose_hypotheses'] == ['T', 'inverse_T']
    assert r['native_gold_or_visibility_labels_created'] == 0
test('T_and_inverse_T_both_reported_no_pose_selection', pose_inverse)


def near_plane():
    v = np.array([[0, 0, -1], [1, 0, 1], [0, 1, 1]], np.float64)
    d, s, r = m.render_depth(v, F, I, K, (4, 4))
    assert d is None and s is None and r['status'] == 'UNSUPPORTED_NEAR_PLANE_CROSSING'
    assert r['crossing_triangle_indices'] == [0] and not r['automatic_clipping_or_mesh_correction']
    v[:, 2] = -2
    d, s, r = m.render_depth(v, F, I, K, (4, 4))
    assert not s.any() and r['fully_behind_or_near_triangles_skipped'] == 1
test('crossing_near_plane_unsupported_no_partial_fabrication', near_plane)


def validation():
    k = K.copy(); k[0, 0] = 0
    rejected(lambda: m.validate_K_T(k, I))
    t = I.copy(); t[0, 0] = 2
    rejected(lambda: m.validate_K_T(K, t))
    t = I.copy(); t[0, 3] = np.nan
    rejected(lambda: m.validate_K_T(K, t))
    t = I.copy(); t[3, 0] = 1
    rejected(lambda: m.validate_K_T(K, t))
    rejected(lambda: m.render_depth(base_triangle(), F, I, K, (4, 4), near=.1))
test('finite_intrinsics_rigid_SE3_and_fixed_near_guard', validation)


def continuous_statistics():
    observed = np.array([[2.5, 3], [0, np.nan]], np.float64)
    rendered = np.array([[2, 2], [2, np.nan]], np.float64)
    mask = np.array([[True, False], [True, False]])
    r, delta = m.describe_observed_depth(observed, rendered, mask)
    assert np.array_equal(delta[0], [.5, 1]) and np.isnan(delta[1]).all()
    assert r['SAM_by_render_2x2'] == [[2, 0], [1, 1]]
    assert r['all_valid_overlap']['count'] == 2 and r['all_valid_overlap']['mean'] == .75
    assert r['all_valid_overlap']['quantiles']['0.5'] == .75
    assert r['valid_overlap_within_SAM']['count'] == 1 and r['mask_render_IoU'] == 2/3
    assert r['D_E_SELF_native_labels_created'] == 0 and not r['mask_negative_is_visibility_negative']
    rejected(lambda: m.describe_observed_depth(observed.astype(np.uint16), rendered, mask))
test('raw_continuous_depth_delta_overlap_and_invalids_no_class_threshold', continuous_statistics)


def repeat():
    a = m.render_depth(base_triangle(), F, I, K, (4, 4))
    b = m.render_depth(base_triangle(), F, I, K, (4, 4))
    assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert a[2] == b[2] and a[0].dtype == np.float64 and a[1].dtype == bool
test('deterministic_repeated_output_bytes_and_dtypes', repeat)

receipt = {'completed': True, 'PASS': True, 'groups': len(results), 'tests': results,
           'code': {'path': str(H/'rgbd_geometry.py'), 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw)},
           'numpy_version': np.__version__, 'inputs': 'Invented analytic triangles/depth/masks only',
           'public_frame_mesh_payload_or_model_reads': 0, 'training_or_visibility_labels': 0}
with (H/'TOY_PROOF.json').open('x') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2, sort_keys=True); f.write('\n')
print(json.dumps(receipt, sort_keys=True))
