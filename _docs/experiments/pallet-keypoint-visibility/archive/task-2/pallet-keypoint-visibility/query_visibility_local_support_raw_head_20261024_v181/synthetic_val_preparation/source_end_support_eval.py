"""Pure source supervision operator. It never loads RGB, weights or source files.

The selected end is a fixed small geometric patch, not the entire primary board.
The two new support targets are non-exclusive and never replace native D/E/S.
"""
import numpy as np

CLASSES = ('OUTSIDE_END_REGION', 'END_VISIBLE', 'END_FOREIGN_FRONT',
           'UNKNOWN_SELF_HIDDEN', 'UNKNOWN_AMBIGUOUS',
           'UNKNOWN_REGISTRATION_OR_CHANGE', 'INVALID_SUPPORT')
NUMERICAL_DEPTH_ULPS = 64  # Source float64 arithmetic tolerance, not a pose/pixel bound.


def expected_patch_depth(quads_world, camera, raw_hw):
    """Ray/plane intersections at original integer pixel centres, bounded by quads.

    Quads are the already registered local face patches. Overlapping projected
    faces use the closest patch intersection. No source mask is used to choose
    or resize the region and no q/teacher outcome enters this operation.
    """
    q = np.asarray(quads_world, np.float64)
    h, w = map(int, raw_hw)
    assert q.ndim == 3 and q.shape[1:] == (4, 3) and 1 <= len(q) <= 3
    assert np.isfinite(q).all() and h > 0 and w > 0
    K, R, t = (np.asarray(camera[k], np.float64) for k in ('K', 'R', 't'))
    assert K.shape == R.shape == (3, 3) and t.shape == (3,)
    assert np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all()
    assert np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-13)
    near = float(camera.get('near', .001))
    pc = q @ R.T + t
    depth = np.full((h, w), np.inf, np.float64)
    if np.any(pc[..., 2] <= near):
        return dict(expected_depth=depth, region_mask=np.zeros((h, w), bool),
                    geometry_valid=False, projected_region_out_of_frame=True,
                    reason='PATCH_NEAR_PLANE_OR_BEHIND_CAMERA')
    homogeneous = pc @ K.T
    projected = homogeneous[..., :2] / homogeneous[..., 2, None]
    oof = bool(np.any(projected[..., 0] < 0) or np.any(projected[..., 0] > w-1)
               or np.any(projected[..., 1] < 0) or np.any(projected[..., 1] > h-1))
    invK = np.linalg.inv(K)
    for face, uv in zip(pc, projected):
        u, v = face[1]-face[0], face[3]-face[0]
        normal = np.cross(u, v)
        gram = np.array([[u@u, u@v], [u@v, v@v]], np.float64)
        assert np.linalg.det(gram) > 0, 'Degenerate registered face patch'
        lo = np.maximum(np.floor(uv.min(0)).astype(np.int64), [0, 0])
        hi = np.minimum(np.ceil(uv.max(0)).astype(np.int64), [w-1, h-1])
        if np.any(hi < lo):
            continue
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        rays = np.column_stack((xx.ravel(), yy.ravel(), np.ones(xx.size))) @ invK.T
        denominator = rays @ normal
        valid = denominator != 0
        scale = np.zeros(len(rays), np.float64)
        scale[valid] = (normal @ face[0]) / denominator[valid]
        hit = rays * scale[:, None]
        delta = hit-face[0]
        ab = np.column_stack((delta @ u, delta @ v)) @ np.linalg.inv(gram).T
        # Inclusive fixed patch boundary; no outcome-tuned pixel dilation.
        valid &= (hit[:, 2] > near) & (ab >= 0).all(1) & (ab <= 1).all(1)
        px, py = xx.ravel()[valid], yy.ravel()[valid]
        depth[py, px] = np.minimum(depth[py, px], hit[valid, 2])
    return dict(expected_depth=depth, region_mask=np.isfinite(depth), geometry_valid=True,
                projected_region_out_of_frame=oof, projected_quad_xy=projected.tolist(),
                reason='REGISTERED_PATCH_PROJECTED_AT_ORIGINAL_PIXEL_CENTRES')


def validate_buffer_metadata(buffer, camera, native8_CF_xy, permCF_to_canonical):
    """Mandatory future loading contract; no files are opened by this function."""
    for k in ('K', 'R', 't'):
        assert np.array_equal(buffer['camera_'+k], np.asarray(camera[k], np.float64))
    assert np.array_equal(buffer['native8_CF_xy'], np.asarray(native8_CF_xy, np.float64))
    assert np.array_equal(buffer['permCF_to_canonical'], np.asarray(permCF_to_canonical, np.int32))


def classify_end_pixels(before, scene, patch, primary_part_id):
    """BEFORE must be target-only; foreign999 is never a source target part."""
    expected = patch['expected_depth']
    shape = expected.shape
    assert isinstance(primary_part_id, (int, np.integer)) and 0 <= primary_part_id < 999
    for packet in (before, scene):
        for key, dtype in (('part_id', np.int32), ('camera_depth', np.float64),
                           ('global_triangle_id', np.int32), ('ambiguity', bool)):
            assert packet[key].shape == shape and packet[key].dtype == dtype
    bp, sp = before['part_id'], scene['part_id']
    bd, sd = before['camera_depth'], scene['camera_depth']
    region = patch['region_mask']
    out = np.zeros(shape, np.int8)
    out[region] = 5
    ambiguous = region & (before['ambiguity'] | scene['ambiguity'])
    out[ambiguous] = 4
    active = region & ~ambiguous
    finite = np.isfinite(bd) & (bd > 0)
    tolerance = NUMERICAL_DEPTH_ULPS * np.finfo(np.float64).eps * np.maximum(
        1., np.maximum(np.abs(np.where(finite, bd, 0.)), np.where(region, np.abs(expected), 0.)))
    equal_patch = finite & (np.abs(bd-expected) <= tolerance)
    before_target = (bp >= 0) & (bp != 999)
    self_hidden = active & before_target & finite & (bd < expected-tolerance)
    out[self_hidden] = 3
    supported_before = active & (bp == primary_part_id) & equal_patch
    visible = supported_before & (sp == bp) & (before['global_triangle_id'] == scene['global_triangle_id']) & (bd == sd)
    foreign = supported_before & (sp == 999) & np.isfinite(sd) & (sd > 0) & (sd < bd)
    out[visible] = 1
    out[foreign] = 2
    return out


def original_crop_positions(q, L, side=96):
    q = np.asarray(q, np.float64)
    assert q.shape == (2,) and np.isfinite(q).all() and np.isfinite(L) and L > 0 and side == 96
    offset = ((np.arange(side, dtype=np.float64)+.5)/side-.5)*L
    x, y = np.meshgrid(q[0]+offset, q[1]+offset)
    return np.column_stack((x.ravel(), y.ravel()))


def sample_classes(classes, points):
    """Exact original96 bilinear onehot; invalid support is retained separately."""
    h, w = classes.shape
    p = np.asarray(points, np.float64)
    assert p.ndim == 2 and p.shape[1] == 2 and np.isfinite(p).all()
    lo = np.floor(p).astype(np.int64)
    frac = p-lo
    out = np.zeros((len(p), len(CLASSES)), np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            xx, yy = lo[:, 0]+dx, lo[:, 1]+dy
            weight = (frac[:, 0] if dx else 1-frac[:, 0]) * (frac[:, 1] if dy else 1-frac[:, 1])
            inside = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            cls = np.full(len(p), 6, np.int8)
            cls[inside] = classes[yy[inside], xx[inside]]
            out[np.arange(len(p)), cls] += weight
    crop_valid = (p[:, 0] >= -.5) & (p[:, 0] < w-.5) & (p[:, 1] >= -.5) & (p[:, 1] < h-.5)
    out[~crop_valid] = 0
    out[~crop_valid, 6] = 1
    assert np.allclose(out.sum(1), 1., rtol=0, atol=2e-15)
    return out


def summarize(weights, patch, original_available=True):
    mass = np.asarray(weights, np.float64).mean(0)
    assert mass.shape == (len(CLASSES),) and np.all(mass >= 0)
    assert np.isclose(mass.sum(), 1., rtol=0, atol=2e-13)
    visible, foreign = float(mass[1]), float(mass[2])
    unknown = float(mass[3]+mass[4]+mass[5])
    end_mass = visible+foreign+unknown
    reason = []
    if not original_available:
        vt = et = -1;reason.append('ORIGINAL_INPUT_U_UNCHANGED')
    elif not patch['geometry_valid'] or patch['projected_region_out_of_frame']:
        vt = et = -1;reason.append('REGISTERED_PATCH_GEOMETRY_OR_OUT_OF_FRAME_UNKNOWN')
    elif end_mass == 0:
        vt, et = 0, -1;reason.append('NO_SAMPLED_END_SUPPORT_IN_ORIGINAL_CROP_NOT_EXTERNAL_E')
    else:
        # Existence has a positive witness even if a different part of the region is unknown.
        vt = 1 if visible > 0 else -1 if unknown > 0 else 0
        et = 1 if foreign > 0 else -1 if unknown > 0 else 0
        if unknown > 0:reason.append('SELF_AMBIGUOUS_OR_REGISTRATION_MASS_RETAINED_UNKNOWN')
        if visible > 0 and foreign > 0:reason.append('PARTIAL_END_VISIBLE_AND_FOREIGN_NONEXCLUSIVE')
        if not reason:reason.append('KNOWN_SOURCE_END_SUPPORT')
    return dict(class_order=list(CLASSES),full96_average_mass=mass.tolist(),
                sampled_end_region_mass=end_mass,visible_support_mass=visible,
                foreign_front_support_mass=foreign,unknown_end_mass=unknown,
                visible_support_target=vt,foreign_front_support_target=et,
                target_encoding='YES1_NO0_UNKNOWNminus1_NONEXCLUSIVE',reasons=reason,
                native_visibility_target=-1,point_quality_target=-1,training_admission=False)


def process_one(record, geometry, before, scene, precomputed_patch=None):
    """Concrete row operation for a future separately approved source-only runner.

    The caller verifies inherited source SHA/IDs before supplying arrays. This
    function has no file IO, model input modifications or classifier-U updates.
    """
    assert record['split'] in ('TRAIN', 'VALIDATION') and len(geometry['quads_world']) == 3
    assert record['base_index'] == geometry['base_index'] and record['native_slot'] == geometry['native_slot']
    assert record['declared_canonical_index'] == geometry['canonical_index']
    assert record['primary_part_id'] == geometry['primary_part_id']
    for packet in (before, scene):
        validate_buffer_metadata(packet, geometry['camera'], geometry['native8_CF_xy'], geometry['permCF_to_canonical'])
    q, L = record['query_xy'], record['L']
    identity={k:record[k] for k in ('original_global_index','original_occurrence_id','scene_index','base_index','native_slot','split','variant','query_xy','L','original_input_flags','original_semantic_state')}
    identity.update(geometry_key=geometry['geometry_key'],factors=record.get('factors',{}))
    if q is None or L is None or not np.isfinite(q).all() or not np.isfinite(L) or L <= 0 or np.array_equal(q, [-1., -1.]):
        return dict(**identity,visible_support_target=-1,foreign_front_support_target=-1,
                    full96_average_mass=None,reasons=['ORIGINAL_Q_OR_L_MISSING'],native_visibility_target=-1,
                    point_quality_target=-1,training_admission=False)
    if precomputed_patch is None:
        patch=expected_patch_depth(geometry['quads_world'],geometry['camera'],geometry['raw_hw'])
    else:
        patch=precomputed_patch
        assert patch['geometry_key']==geometry['geometry_key']
        assert patch['expected_depth'].dtype==np.float64 and patch['expected_depth'].shape==tuple(geometry['raw_hw'])
        assert patch['region_mask'].dtype==bool and patch['region_mask'].shape==tuple(geometry['raw_hw'])
    classes=classify_end_pixels(before,scene,patch,geometry['primary_part_id'])
    weights=sample_classes(classes,original_crop_positions(q,L))
    result=summarize(weights,patch,record['original_input_flags']['query_available'])
    return dict(**identity,**result,
                projected_region_out_of_frame=patch['projected_region_out_of_frame'],
                scope='New fixed-end support supervision only; not original native D/E/S or physical corner/pose gold.')
