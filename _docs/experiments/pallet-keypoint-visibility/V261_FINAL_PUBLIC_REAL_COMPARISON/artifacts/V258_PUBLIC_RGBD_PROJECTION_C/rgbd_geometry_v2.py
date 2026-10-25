"""Deterministic +Z public mesh/depth diagnostic; no training or visibility labels."""
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import numpy as np

NEAR_M = 1e-9
EDGE_EPS = 1e-12
AREA_EPS = 1e-14
RIGID_ATOL = 1e-5


def require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def checked_path(path):
    """Only an explicit absolute leaf; no discovery or referenced material reads."""
    p = Path(path)
    require(p.is_absolute() and '..' not in p.parts, 'Explicit absolute leaf required')
    require(not any(s in str(p).casefold() for s in
                    ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'holdout', 'oldval')),
            'Forbidden path before FS')
    for a in [*reversed(p.parents), p]:
        require(not stat.S_ISLNK(os.lstat(a).st_mode), 'Symlink artifact forbidden')
    require(stat.S_ISREG(os.lstat(p).st_mode), 'Regular explicit leaf required')
    return p


def read_explicit(path, binding=None):
    p = checked_path(path)
    if binding is not None:
        require(str(p) == binding['path'], 'Exact bound path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        data = f.read()
    if binding is not None:
        require(len(data) == binding['size_bytes'] and
                hashlib.sha256(data).hexdigest() == binding['sha256'], 'Bound SHA/size')
    return data


def parse_obj_text(text):
    """Read v/f only. Faces must already be triangles; no triangulation/repair."""
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='strict')
    require(isinstance(text, str), 'OBJ text or bytes required')
    vertices, faces, declarations = [], [], []
    tags = {}
    unsupported_faces = []
    for line_no, source in enumerate(text.splitlines(), 1):
        fields = source.partition('#')[0].split()
        if not fields:
            continue
        tag = fields[0]
        tags[tag] = tags.get(tag, 0) + 1
        if tag == 'v':
            require(len(fields) in (4, 5, 7), 'OBJ v must have xyz, xyz1, or xyzRGB; RGBA unsupported, line ' + str(line_no))
            values = [float(x) for x in fields[1:]]
            require(np.isfinite(values).all(), 'Finite OBJ vertex and optional color')
            if len(values) == 4:
                require(values[3] == 1.0, 'Nonunit homogeneous OBJ vertex unsupported')
            elif len(values) == 6:
                require(all(0.0 <= color <= 1.0 for color in values[3:]), 'OBJ RGB colors must be finite within0..1')
            vertices.append(values[:3])
        elif tag == 'f':
            if len(fields) != 4:
                unsupported_faces.append({'line': line_no, 'vertex_count': len(fields)-1})
                continue
            tri = []
            for token in fields[1:]:
                v = token.split('/')[0]
                require(v and (v.lstrip('-').isdigit()), 'Integer OBJ position index')
                index = int(v)
                require(index != 0, 'OBJ indices are 1-based; negative relative allowed')
                index = index-1 if index > 0 else len(vertices)+index
                require(0 <= index < len(vertices), 'OBJ face references existing vertex')
                tri.append(index)
            faces.append(tri)
        elif tag in ('mtllib', 'usemtl'):
            declarations.append({'tag': tag, 'value': ' '.join(fields[1:]), 'inert': True})
    require(not unsupported_faces,
            'Unsupported nontriangular OBJ faces; explicit triangulation contract required: ' + json.dumps(unsupported_faces[:8]))
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    require(len(v) > 0 and len(f) > 0, 'Nonempty triangular OBJ')
    report = {'vertices': len(v), 'triangles': len(f), 'tag_counts': tags,
              'position_dtype': 'float64', 'face_dtype': 'int64',
              'material_declarations_inert': declarations,
              'mesh_scale': 1.0, 'vertex_axis_origin_or_scale_correction': False,
              'triangulation_or_welding': False}
    return v, f, report


def parse_obj(path, binding=None):
    """Caller must admit the exact public non-frame asset before calling I/O."""
    return parse_obj_text(read_explicit(path, binding))


def validate_K_T(K, T):
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    require(K.shape == (3, 3) and T.shape == (4, 4), 'K3x3/T4x4 required')
    require(np.isfinite(K).all() and np.isfinite(T).all(), 'Finite K/T required')
    require(K[0, 0] > 0 and K[1, 1] > 0 and abs(np.linalg.det(K)) > 0, 'Positive focal lengths/nonsingular K')
    require(np.allclose(K[2], [0, 0, 1], atol=1e-12, rtol=0), 'Pinhole K last row [0,0,1]')
    require(np.allclose(T[3], [0, 0, 0, 1], atol=1e-12, rtol=0), 'SE3 homogeneous last row')
    rotation = T[:3, :3]
    ortho = float(np.max(np.abs(rotation.T@rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    require(ortho <= RIGID_ATOL and abs(determinant-1) <= RIGID_ATOL, 'Rigid proper rotation required; no correction')
    return K.copy(), T.copy(), {'rotation_orthogonality_maxabs': ortho, 'rotation_determinant': determinant,
                               'rigid_validation_atol': RIGID_ATOL, 'matrix_reorthogonalized': False}


def project_points(points_camera, K, near=NEAR_M):
    """OpenCV pinhole +Z. Integer (column,row) locations are pixel centers."""
    xyz = np.asarray(points_camera, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(), 'Finite camera points Nx3')
    require(K.shape == (3, 3) and np.isfinite(K).all(), 'Finite K')
    require(near == NEAR_M, 'Fixed numerical near guard, no sweep')
    positive = xyz[:, 2] > near
    uv = np.full((len(xyz), 2), np.nan, dtype=np.float64)
    hp = xyz[positive]@K.T
    uv[positive] = hp[:, :2]/hp[:, 2:3]
    require(np.isfinite(uv[positive]).all(), 'Finite projected front points')
    return uv, positive


def render_depth(vertices, faces, T, K, hw, near=NEAR_M):
    """Two-sided integer-center z buffer, reciprocal interpolation, no clipping.

    Returns depth_F64 (NaN background), support_bool, report. A near-crossing
    triangle makes the entire interpretation unsupported: (None,None,report).
    Fully-behind triangles are skipped and counted. Raster edge ties are inclusive;
    exact z ties retain the first face, with no depth tolerance or face culling.
    """
    require(near == NEAR_M, 'Fixed near guard, no search')
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    require(v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all(), 'Finite vertices Nx3')
    require(f.ndim == 2 and f.shape[1] == 3 and f.dtype.kind in 'iu', 'Integer triangular faces')
    require(f.size and f.min() >= 0 and f.max() < len(v), 'Face range')
    require(len(hw) == 2 and all(isinstance(x, (int, np.integer)) and x > 0 for x in hw), 'Positive integer H,W')
    h, w = map(int, hw)
    K, T, validation = validate_K_T(K, T)
    xyz = v@T[:3, :3].T + T[:3, 3]
    uv, front = project_points(xyz, K, near)
    front3 = front[f]
    crossing = np.flatnonzero(front3.any(axis=1) & ~front3.all(axis=1))
    behind = np.flatnonzero(~front3.any(axis=1))
    base = {'projection': 'OpenCV pinhole +Z; column,row integer pixel centers',
            'hw': [h, w], 'mesh_scale': 1.0, 'translation_unit': 'meters',
            'near_m_numerical_guard': near, 'triangle_count': len(f),
            'fully_behind_or_near_triangles_skipped': len(behind),
            'near_crossing_triangles': len(crossing),
            'perspective_depth': '1 / sum(screen_barycentric_i / camera_z_i)',
            'z_buffer_rule': 'minimum positive camera Z; two-sided; first face at exact depth tie',
            'edge_inclusion_barycentric_eps': EDGE_EPS, 'degenerate_area_eps_pixel2': AREA_EPS,
            'pose_validation': validation, 'automatic_clipping_or_mesh_correction': False}
    if len(crossing):
        return None, None, dict(base, completed=False, status='UNSUPPORTED_NEAR_PLANE_CROSSING',
                                crossing_triangle_indices=crossing.tolist(),
                                requirement='Explicit near-plane clipping contract needed; no partial fabricated render returned.')
    zbuf = np.full((h, w), np.inf, dtype=np.float64)
    covered = degenerate = outside = processed = 0
    for index in np.flatnonzero(front3.all(axis=1)):
        xy = uv[f[index]]
        zs = xyz[f[index], 2]
        xmin = max(0, int(np.ceil(xy[:, 0].min())))
        xmax = min(w-1, int(np.floor(xy[:, 0].max())))
        ymin = max(0, int(np.ceil(xy[:, 1].min())))
        ymax = min(h-1, int(np.floor(xy[:, 1].max())))
        if xmin > xmax or ymin > ymax:
            outside += 1
            continue
        x0, y0 = xy[0]; x1, y1 = xy[1]; x2, y2 = xy[2]
        denominator = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
        if abs(denominator) <= AREA_EPS:
            degenerate += 1
            continue
        yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
        a = ((y1-y2)*(xx-x2) + (x2-x1)*(yy-y2))/denominator
        b = ((y2-y0)*(xx-x2) + (x0-x2)*(yy-y2))/denominator
        c = 1-a-b
        inside = (a >= -EDGE_EPS) & (b >= -EDGE_EPS) & (c >= -EDGE_EPS)
        reciprocal = a/zs[0] + b/zs[1] + c/zs[2]
        require(np.isfinite(reciprocal[inside]).all() and (reciprocal[inside] > 0).all(), 'Positive finite interpolated reciprocal Z')
        candidate = np.full(inside.shape, np.inf, dtype=np.float64)
        candidate[inside] = 1/reciprocal[inside]
        patch = zbuf[ymin:ymax+1, xmin:xmax+1]
        update = inside & (candidate < patch)
        patch[update] = candidate[update]
        covered += int(inside.sum())
        processed += 1
    support = np.isfinite(zbuf)
    depth = zbuf.copy()
    depth[~support] = np.nan
    require(np.isfinite(depth[support]).all() and (depth[support] > near).all(), 'Positive finite final depth')
    report = dict(base, completed=True, status='RENDERED', supported_pixels=int(support.sum()),
                  fully_out_of_frame_triangles=outside, projected_degenerate_triangles=degenerate,
                  processed_in_frame_triangles=processed, triangle_pixel_visits=covered,
                  pixel_coverage_is_not_point_visibility_or_native_gold=True)
    return depth, support, report


def describe_values(values):
    v = np.asarray(values, dtype=np.float64)
    require(v.ndim == 1 and np.isfinite(v).all(), 'Finite continuous sample vector')
    if not len(v):
        return {'count': 0, 'min': None, 'max': None, 'mean': None, 'std': None,
                'quantiles': None, 'negative': 0, 'zero': 0, 'positive': 0}
    quantiles = (0, .01, .05, .25, .5, .75, .95, .99, 1)
    q = np.quantile(v, quantiles, method='linear')
    return {'count': len(v), 'min': float(v.min()), 'max': float(v.max()),
            'mean': float(v.mean()), 'std': float(v.std()),
            'quantiles': {str(k): float(x) for k, x in zip(quantiles, q)},
            'negative': int((v < 0).sum()), 'zero': int((v == 0).sum()), 'positive': int((v > 0).sum())}


def describe_observed_depth(depth_m, rendered_depth, mask):
    """Continuous observed-minus-rendered meters and SAM/render support only.

    Returns (JSON-ready stats, raw_difference_F64_map). No depth threshold,
    occlusion class, SELF/E label or native semantics are produced. Nonfinite and
    nonpositive observed depth are counted as invalid, never visibility-negative.
    """
    observed = np.asarray(depth_m)
    rendered = np.asarray(rendered_depth)
    mask = np.asarray(mask)
    require(observed.ndim == 2 and observed.dtype.kind == 'f', 'Depth already converted to meters float HxW by admitted decoder')
    require(rendered.shape == observed.shape and rendered.dtype.kind == 'f', 'Rendered depth shape/float')
    require(mask.shape == observed.shape and mask.dtype == bool, 'Explicit Boolean SAM mask')
    observed_valid = np.isfinite(observed) & (observed > 0)
    rendered_support = np.isfinite(rendered) & (rendered > 0)
    both_valid = observed_valid & rendered_support
    residual = np.full(observed.shape, np.nan, dtype=np.float64)
    residual[both_valid] = observed[both_valid].astype(np.float64)-rendered[both_valid].astype(np.float64)
    intersection = int((mask & rendered_support).sum())
    union = int((mask | rendered_support).sum())
    total = observed.size
    stats = {'depth_unit': 'meters', 'difference_direction': 'observed_minus_rendered',
             'pixels': total, 'observed_valid_pixels': int(observed_valid.sum()),
             'observed_invalid_nonfinite': int((~np.isfinite(observed)).sum()),
             'observed_invalid_nonpositive_finite': int((np.isfinite(observed) & (observed <= 0)).sum()),
             'rendered_support_pixels': int(rendered_support.sum()), 'SAM_mask_pixels': int(mask.sum()),
             'observed_and_rendered_valid_pixels': int(both_valid.sum()),
             'SAM_by_render_2x2': [[intersection, int((mask & ~rendered_support).sum())],
                                   [int((~mask & rendered_support).sum()), int((~mask & ~rendered_support).sum())]],
             'SAM_by_render_order': {'rows': ['SAM_true', 'SAM_false'], 'columns': ['render_support', 'render_background']},
             'mask_render_IoU': intersection/union if union else None,
             'mask_render_Dice': 2*intersection/(int(mask.sum())+int(rendered_support.sum())) if mask.any() or rendered_support.any() else None,
             'all_valid_overlap': describe_values(residual[both_valid]),
             'valid_overlap_within_SAM': describe_values(residual[both_valid & mask]),
             'valid_overlap_outside_SAM': describe_values(residual[both_valid & ~mask]),
             'D_E_SELF_native_labels_created': 0, 'depth_threshold_or_pose_optimization': False,
             'mask_negative_is_visibility_negative': False}
    require(sum(sum(row) for row in stats['SAM_by_render_2x2']) == total, 'Mask/render partition')
    return stats, residual


def diagnostic_pair(vertices, faces, T, K, hw, depth_m, mask):
    """Exactly T and inverse(T), scale1; retain both, no winner/pose fitting."""
    K, T, _ = validate_K_T(K, T)
    require(np.asarray(depth_m).shape == tuple(hw) and np.asarray(mask).shape == tuple(hw), 'Admitted image hw')
    inverse = np.linalg.inv(T)
    outputs, reports = {}, {}
    for name, transform in (('T', T), ('inverse_T', inverse)):
        depth, support, report = render_depth(vertices, faces, transform, K, hw)
        if depth is None:
            outputs[name] = {'depth': None, 'support': None, 'observed_minus_rendered': None}
            reports[name] = {'render': report, 'continuous_depth_and_overlap': None}
        else:
            stats, residual = describe_observed_depth(depth_m, depth, mask)
            outputs[name] = {'depth': depth, 'support': support, 'observed_minus_rendered': residual}
            reports[name] = {'render': report, 'continuous_depth_and_overlap': stats}
    return outputs, {'completed': True, 'interpretations': reports, 'mesh_scale': 1.0,
                     'translation_unit': 'meters', 'observed_depth_unit': 'meters',
                     'pixel_center': 'integer column,row; no +0.5 offset',
                     'pose_hypotheses': ['T', 'inverse_T'], 'winner_selected': None,
                     'search_optimization_PnP_coordinate_correction_training': False,
                     'native_gold_or_visibility_labels_created': 0,
                     'registration_or_pose_correctness_certified': False}


def run_bound_entry(entry_path):
    """Optional explicit NPY I/O. Root may instead decode PNG and call pure APIs.

    No PNG decoder is hidden here. This runner requires already-admitted F64/F32
    meter-depth NPY and bool-mask NPY, plus exact OBJ/K/T bindings. It never follows
    serialized material/texture/raw-source pointers or chooses another frame.
    """
    entry = json.loads(read_explicit(entry_path))
    require(entry.get('approved_actual_public_geometry_diagnostic') is True, 'Root actual geometry entry required')
    require(entry.get('mesh_scale') == 1 and entry.get('depth_unit') == 'meters' and
            entry.get('translation_unit') == 'meters', 'Fixed explicit units/scale, no sweep')
    require(entry.get('automatic_retry') is False and entry.get('current_or_held_access') is False, 'No retry/protected access')
    own = read_explicit(Path(__file__).absolute(), entry['code'])
    require(bool(own), 'Pinned diagnostic code')
    bs = entry['inputs']
    require(set(bs) == {'mesh_obj', 'K', 'T', 'depth_m_npy', 'sam_mask_bool_npy'}, 'Exact five diagnostic inputs')
    vertices, faces, mesh_report = parse_obj(bs['mesh_obj']['path'], bs['mesh_obj'])
    K = np.loadtxt(io.BytesIO(read_explicit(bs['K']['path'], bs['K'])), dtype=np.float64)
    T = np.loadtxt(io.BytesIO(read_explicit(bs['T']['path'], bs['T'])), dtype=np.float64)
    depth = np.load(io.BytesIO(read_explicit(bs['depth_m_npy']['path'], bs['depth_m_npy'])), allow_pickle=False)
    mask = np.load(io.BytesIO(read_explicit(bs['sam_mask_bool_npy']['path'], bs['sam_mask_bool_npy'])), allow_pickle=False)
    arrays, report = diagnostic_pair(vertices, faces, T, K, entry['hw'], depth, mask)
    report['mesh_parser'] = mesh_report
    report['entry'] = entry
    return arrays, report
