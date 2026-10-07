"""Prediction-only, one optical-Z degree of freedom depth teacher.

The imported legacy functions are pure geometry; its GT-based main and reference
solver are never called.  This module reads no files and constructs no model.
"""
from __future__ import annotations

import cv2
import numpy as np

from scripts.self_training_yolo.depth_corrected.audit_sensor_geometry_validation import (
    cuboid, ray_surface_z, shrink,
)

FACES = (
    ('near', (0, 1, 2, 3), 2, -1),
    ('far', (4, 5, 6, 7), 2, +1),
    ('top', (0, 1, 5, 4), 1, -1),
    ('bottom', (3, 2, 6, 7), 1, +1),
    ('left', (0, 3, 7, 4), 0, -1),
    ('right', (1, 5, 6, 2), 0, +1),
)
ZERO_EPS_M = 1e-12


def settings():
    return dict(schema='depth_signal_pilot_teacher_v1', face_shrink_fraction=.15,
                minimum_absolute_ray_normal_cosine=.3, grid_shape=[32, 32],
                grid='prediction-only visible shrunk-face union bounding rectangle; cell centers rounded to native pixels; raster-order deduplication',
                tile_shape=[4, 4], split='checkerboard (tile_x+tile_y)%2: fit A=0, heldout B=1',
                location_estimator='median of nonempty tile medians; fit A supplies zD; B independently checks direction',
                minimum_nonempty_blocks_per_half=1,
                directional_rule='both independently bootstrapped half 95% intervals exclude z0 and half corrections have the same sign',
                bootstrap_replicates=200, bootstrap_seed=20261007, bootstrap_interval=[.025, .975],
                bootstrap_unit='nonempty spatial tile, stratified by spatial half; A/B/pooled use the same 200 replicate IDs, not individual depth pixels',
                center_z_range_factors=[.8, 1.2], range_rule='both half estimates must lie in range; no clipping or filtering pixel z estimates by range',
                no_change_numerical_tolerance_m=ZERO_EPS_M,
                heldout_check='median of B-tile median absolute optical-Z residuals on unchanged initial face/ray correspondences; no worsening',
                depth_units='input depth_m is optical-axis Z in metres; pairing/unit/registration verification is caller responsibility',
                invalid='nonfinite/nonpositive depth plus caller supplied previously known invalid codes in metres',
                projection='existing OpenCV undistorted pinhole/no-distortion contract; selected R_cf/cf_extents, camera-facing corners0..7',
                transfer='add projection delta only on existing supported finite non-sentinel corners0..7; center8 and unsupported slots unchanged',
                GT_inputs=False, model_calls=0, F_calls=0, parameter_search=0)


def _geometry(K, R_cf, t0, cf_extents):
    K, R, t, dims = (np.asarray(value, dtype=np.float64)
                      for value in (K, R_cf, t0, cf_extents))
    if K.shape != (3, 3) or R.shape != (3, 3) or t.shape != (3,) or dims.shape != (3,):
        raise ValueError('Expected K/R [3,3], t/extents [3]')
    if not all(np.isfinite(value).all() for value in (K, R, t, dims)) or np.any(dims <= 0) or t[2] <= 0:
        raise ValueError('Nonfinite/nonpositive pose, camera or extent')
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0., 0., 1.]) or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError('Legacy pinhole/slab contract requires positive focal lengths and zero skew')
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(R), 1., atol=1e-6, rtol=0):
        raise ValueError('R must be a proper rotation')
    model = cuboid(*dims)
    camera = model @ R.T + t
    if np.min(camera[:, 2]) <= 0:
        raise ValueError('Initial cuboid intersects the camera plane')
    projected = (camera @ K.T)[:, :2] / camera[:, 2, None]
    return K, R, t, dims, model, camera, projected


def sample_geometry(hw, K, R_cf, t0, cf_extents):
    """At most 1024 unique native pixels; no depth/GT-dependent sample selection."""
    h, w = map(int, hw)
    if min(h, w) <= 0:
        raise ValueError('Nonpositive native image dimensions')
    K, R, t, dims, model, camera, projected = _geometry(K, R_cf, t0, cf_extents)
    face_rows = []
    for name, indices, axis, sign in FACES:
        normal = sign * R[:, axis]
        offset = dims[axis] / 2
        if float(normal @ camera[list(indices)].mean(0)) >= 0:
            continue  # outward normal points away from the camera
        polygon = shrink(projected[list(indices)], .15)
        face_rows.append(dict(name=name, normal=normal, offset=offset, polygon=polygon))
    if not face_rows:
        return dict(pixels=np.empty((0, 2), int), tile=np.empty(0, int), face=np.empty(0, int),
                    faces=[], directions=np.empty((0, 3)), cosine=np.empty(0), grid_candidates=0, grazing_rejected=0)
    polygon_points = np.concatenate([f['polygon'] for f in face_rows])
    lo = np.maximum(polygon_points.min(0), [0., 0.])
    hi = np.minimum(polygon_points.max(0), [w - 1., h - 1.])
    if np.any(hi <= lo):
        return dict(pixels=np.empty((0, 2), int), tile=np.empty(0, int), face=np.empty(0, int),
                    faces=face_rows, directions=np.empty((0, 3)), cosine=np.empty(0), grid_candidates=0, grazing_rejected=0)
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    pixels = np.rint(lo + (grid + .5) / 32 * (hi - lo)).astype(np.int64)
    _, unique = np.unique(pixels, axis=0, return_index=True)
    unique = np.sort(unique)  # original raster order, never error/depth selected
    pixels, grid = pixels[unique], grid[unique]
    tile = (grid[:, 1] // 8) * 4 + grid[:, 0] // 8
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
    directions = homogeneous @ np.linalg.inv(K).T
    nearest_z, hit = ray_surface_z(pixels, K, R, t, dims)
    assigned = np.full(len(pixels), -1, dtype=int)
    cosine = np.full(len(pixels), np.nan)
    grazing = np.zeros(len(pixels), dtype=bool)
    for face_index, face in enumerate(face_rows):
        normal = face['normal']
        denominator = directions @ normal
        with np.errstate(divide='ignore', invalid='ignore'):
            plane_z = (face['offset'] + normal @ t) / denominator
        inside = np.array([cv2.pointPolygonTest(face['polygon'].astype(np.float32), tuple(map(float, p)), False) >= 0
                           for p in pixels], dtype=bool)
        corresponds = inside & hit & np.isfinite(plane_z) & np.isclose(plane_z, nearest_z, rtol=0, atol=1e-9)
        cos = np.abs(denominator) / np.linalg.norm(directions, axis=-1)
        grazing |= corresponds & (cos < .3)
        good = corresponds & (cos >= .3) & (assigned < 0)
        assigned[good] = face_index; cosine[good] = cos[good]
    use = assigned >= 0
    return dict(pixels=pixels[use], tile=tile[use], face=assigned[use], faces=face_rows,
                directions=directions[use], cosine=cosine[use],
                grid_candidates=len(pixels), grazing_rejected=int(grazing.sum()))


def _block_medians(values, tiles):
    ids = np.unique(tiles)
    return ids, np.asarray([np.median(values[tiles == tile]) for tile in ids], dtype=np.float64)


def _bootstrap(a, b):
    rng = np.random.default_rng(20261007)
    da = a[rng.integers(0, len(a), size=(200, len(a)))]
    db = b[rng.integers(0, len(b), size=(200, len(b)))]
    return [np.quantile(np.median(values, axis=-1), [.025, .975]).tolist()
            for values in (da, db, np.concatenate([da, db], axis=-1))]


def estimate(depth_m, K, R_cf, t0, cf_extents, invalid_codes_m=()):
    """Fixed correspondences + fit/check; returns fallback tD on every abstention."""
    depth = np.asarray(depth_m, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError('Native optical-Z depth_m must be a 2D array')
    K, R, t, dims, *_ = _geometry(K, R_cf, t0, cf_extents)
    z0 = float(t[2]); v = t / z0
    result = dict(status='ABSTAIN', accepted=False, changed=False, reason=None,
                  z0=z0, zD=z0, z=z0, tD=t.tolist(), z_A=None, z_B=None, samples=0, blocks=0,
                  range_m=[.8 * z0, 1.2 * z0],
                  bootstrap_interval_A_m=None, bootstrap_interval_B_m=None,
                  bootstrap_interval_pooled_m=None, fit_nonempty_blocks=0, check_nonempty_blocks=0,
                  fit_samples=0, check_samples=0, valid_depth_samples=0,
                  check_depth_residual_before_m=None, check_depth_residual_after_m=None)
    geometry = sample_geometry(depth.shape, K, R, t, dims)
    pixels = geometry['pixels']
    result.update(grid_candidates=geometry['grid_candidates'], eligible_geometric_samples=len(pixels),
                  grazing_rejected=geometry['grazing_rejected'], visible_faces=[f['name'] for f in geometry['faces']])
    if not len(pixels):
        result['reason'] = 'NO_ELIGIBLE_FACE_SAMPLES'; return result
    measured = depth[pixels[:, 1], pixels[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    for code in invalid_codes_m:
        valid &= measured != float(code)
    pixels = pixels[valid]
    measured = measured[valid]
    tile = geometry['tile'][valid]
    face = geometry['face'][valid]
    rays = geometry['directions'][valid]
    normals = np.asarray([geometry['faces'][i]['normal'] for i in face]).reshape(-1, 3)
    offsets = np.asarray([geometry['faces'][i]['offset'] for i in face])
    nv = normals @ v
    with np.errstate(divide='ignore', invalid='ignore'):
        centers = (measured * np.sum(normals * rays, axis=-1) - offsets) / nv
    finite = np.isfinite(centers) & (np.abs(nv) > 1e-12)
    pixels, measured, tile, face, rays, normals, offsets, centers = [a[finite] for a in (pixels, measured, tile, face, rays, normals, offsets, centers)]
    halves = ((tile // 4 + tile % 4) % 2).astype(int)
    ids, block_z = _block_medians(centers, tile)
    block_half = (ids // 4 + ids % 4) % 2
    a, b = block_z[block_half == 0], block_z[block_half == 1]
    result.update(valid_depth_samples=len(centers), samples=len(centers), blocks=len(block_z), fit_samples=int((halves == 0).sum()),
                  check_samples=int((halves == 1).sum()), fit_nonempty_blocks=len(a), check_nonempty_blocks=len(b),
                  block_ids=ids.tolist(), block_z_m=block_z.tolist(),
                  sample_pixels=pixels.tolist(), sample_tile_ids=tile.tolist(), sample_face_names=[geometry['faces'][i]['name'] for i in face],
                  sample_absolute_cosines=geometry['cosine'][valid][finite].tolist())
    if not len(a) or not len(b):
        result['reason'] = 'MISSING_SPATIAL_HALF'; return result
    za, zb = float(np.median(a)), float(np.median(b))
    result.update(z_A=za, z_B=zb)
    cia, cib, cip = _bootstrap(a, b)
    result.update(bootstrap_interval_A_m=cia, bootstrap_interval_B_m=cib,
                  bootstrap_interval_pooled_m=cip, bootstrap_replicates=200)
    dr = np.sum(normals * rays, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        initial_z = (offsets + normals @ t) / dr
        corrected_z = (offsets + normals @ (v * za)) / dr
    check = halves == 1
    _, rb = _block_medians(np.abs(measured[check] - initial_z[check]), tile[check])
    _, ra = _block_medians(np.abs(measured[check] - corrected_z[check]), tile[check])
    before, after = float(np.median(rb)), float(np.median(ra))
    result.update(check_depth_residual_before_m=before, check_depth_residual_after_m=after)
    if max(abs(za - z0), abs(zb - z0)) <= ZERO_EPS_M:
        result.update(status='NO_CHANGE', reason='NUMERIC_ZERO_CORRECTION'); return result
    if not (.8 * z0 <= za <= 1.2 * z0 and .8 * z0 <= zb <= 1.2 * z0):
        result['reason'] = 'OUTSIDE_CENTER_Z_RANGE'; return result
    if (za - z0) * (zb - z0) <= 0:
        result['reason'] = 'SPATIAL_HALVES_DISAGREE_DIRECTION'; return result
    excludes = lambda ci: ci[1] < z0 - ZERO_EPS_M or ci[0] > z0 + ZERO_EPS_M
    if not excludes(cia) or not excludes(cib):
        result['reason'] = 'BLOCK_VARIABILITY_UNRESOLVED_DIRECTION'; return result
    if not np.isfinite(after) or after > before + ZERO_EPS_M:
        result['reason'] = 'HELDOUT_DEPTH_RESIDUAL_WORSENED'; return result
    result.update(status='ACCEPTED', accepted=True, changed=True, reason='FIT_DIRECTION_AND_HELDOUT_CHECK_PASS',
                  zD=za, z=za, tD=(v * za).tolist())
    return result


def project(K, R_cf, translation, cf_extents):
    K, R, t, dims, *_ = _geometry(K, R_cf, translation, cf_extents)
    return cv2.projectPoints(cuboid(*dims), cv2.Rodrigues(R)[0], t, K, None)[0].reshape(8, 2)


def transfer(q0, support, K, R_cf, t0, t_new, cf_extents):
    """Preserve raw non-cuboid residuals and all unobserved/center slots."""
    raw = np.asarray(q0, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if raw.shape != (9, 2) or support.shape != (9,):
        raise ValueError('Raw points [9,2] and pre-existing support [9] required')
    result = raw.copy()
    if np.array_equal(np.asarray(t0), np.asarray(t_new)):
        return result  # exact no-change, without floating point cancellation
    delta = project(K, R_cf, t_new, cf_extents) - project(K, R_cf, t0, cf_extents)
    valid = support[:8] & np.isfinite(raw[:8]).all(-1) & ~np.all(raw[:8] == -1, axis=-1)
    result[:8][valid] = raw[:8][valid] + delta[valid]
    assert np.array_equal(result[8], raw[8], equal_nan=True)
    assert np.array_equal(result[np.r_[~valid, True]], raw[np.r_[~valid, True]], equal_nan=True)
    return result
