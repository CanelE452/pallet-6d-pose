"""Analytic depth/projection contracts, without real data, GT or pose solves."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import teacher as T


def _render(hw, K, R, t, dims):
    yy, xx = np.indices(hw)
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    z, hit = T.ray_surface_z(pixels, K, R, t, dims)
    return np.where(hit, z, np.nan).reshape(hw)


def _block_depth(hw, K, R, t, dims, center_by_tile):
    geometry = T.sample_geometry(hw, K, R, t, dims)
    depth = np.full(hw, np.nan)
    for pixel, ray, tile, face_index in zip(geometry['pixels'], geometry['directions'], geometry['tile'], geometry['face']):
        face = geometry['faces'][face_index]
        center_z = center_by_tile(int(tile))
        point_z = (face['offset'] + face['normal'] @ (t / t[2] * center_z)) / (face['normal'] @ ray)
        depth[pixel[1], pixel[0]] = point_z
    return depth


def run_tests():
    start = time.perf_counter()
    cases = []
    K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
    hw = (480, 640); R = np.eye(3); t = np.array([.0, .0, 3.3]); dims = np.array([1.3, .15, 1.1])
    counts = {'actual_F_calls': 0, 'actual_PnP_calls': 0, 'NN_calls': 0, 'optimizer_updates': 0,
              'analytic_slab_geometry_calls': 0}
    old_slab = T.ray_surface_z
    old_pnp = {name: getattr(cv2, name) for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')}
    def slab(*args, **kwargs):
        counts['analytic_slab_geometry_calls'] += 1
        return old_slab(*args, **kwargs)
    def forbidden(*args, **kwargs):
        counts['actual_PnP_calls'] += 1
        raise AssertionError('Analytic unit tests must not call PnP/F')
    T.ray_surface_z = slab
    for name in old_pnp:
        setattr(cv2, name, forbidden)
    try:
        model = T.cuboid(*dims)
        expected_sign = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
        assert np.array_equal(model, expected_sign * dims / 2)
        for _, indices, axis, sign in T.FACES:
            assert np.all(model[list(indices), axis] * sign == dims[axis] / 2)
        cases.append('exact_camera_facing_corner_order_and_outward_face_planes')

        true_t = t / t[2] * 3.
        depth = _render(hw, K, R, true_t, dims)
        answer = T.estimate(depth, K, R, t, dims)
        assert answer['accepted'] and answer['status'] == 'ACCEPTED', answer['reason']
        np.testing.assert_allclose(answer['zD'], 3., atol=1e-12, rtol=0)
        np.testing.assert_allclose(answer['tD'], true_t, atol=1e-12, rtol=0)
        assert answer['check_depth_residual_after_m'] < answer['check_depth_residual_before_m']
        assert answer['bootstrap_replicates'] == 200
        cases.append('perfect_surface_recovers_center_z_not_surface_depth')

        oblique = cv2.Rodrigues(np.array([.35, .45, .12]))[0]
        offaxis = np.array([.12, -.06, 3.3]); target = offaxis / offaxis[2] * 3.
        oblique_depth = _render(hw, K, oblique, target, dims)
        result = T.estimate(oblique_depth, K, oblique, offaxis, dims)
        assert result['accepted'], result['reason']
        np.testing.assert_allclose(result['zD'], 3., atol=1e-10, rtol=0)
        cases.append('oblique_face_fixed_center_ray_recovers_true_translation')

        no_change = T.estimate(_render(hw, K, R, t, dims), K, R, t, dims)
        assert no_change['status'] == 'NO_CHANGE' and not no_change['accepted']
        assert no_change['zD'] == t[2] and no_change['tD'] == t.tolist()
        cases.append('zero_depth_correction_is_no_change_not_accepted')

        outside = T.estimate(_render(hw, K, R, t * .7, dims), K, R, t, dims)
        assert not outside['accepted'] and outside['reason'] == 'OUTSIDE_CENTER_Z_RANGE'
        assert outside['zD'] == t[2] and outside['z_A'] < outside['range_m'][0]
        cases.append('outside_twenty_percent_abstains_without_clipping')

        half = lambda block: (block // 4 + block % 4) % 2
        opposite = _block_depth(hw, K, R, t, dims, lambda block: 3.1 if half(block) == 0 else 3.5)
        opposite_result = T.estimate(opposite, K, R, t, dims)
        assert not opposite_result['accepted'] and opposite_result['reason'] == 'SPATIAL_HALVES_DISAGREE_DIRECTION'
        cases.append('opposite_spatial_half_directions_abstain')

        variable = _block_depth(hw, K, R, t, dims, lambda block: 3.1 if block // 4 < 2 else 3.6)
        variable_result = T.estimate(variable, K, R, t, dims)
        assert not variable_result['accepted'] and variable_result['reason'] == 'BLOCK_VARIABILITY_UNRESOLVED_DIRECTION'
        cases.append('block_bootstrap_interval_crossing_initial_z_abstains')

        worsening = _block_depth(hw, K, R, t, dims, lambda block: 3. if half(block) == 0 else 3.29)
        worsened = T.estimate(worsening, K, R, t, dims)
        assert not worsened['accepted'] and worsened['reason'] == 'HELDOUT_DEPTH_RESIDUAL_WORSENED'
        assert worsened['check_depth_residual_after_m'] > worsened['check_depth_residual_before_m']
        cases.append('heldout_residual_worsening_abstains_even_with_agreeing_directions')

        grazing_dims = np.array([1.3, .15, .1])
        geometry = T.sample_geometry(hw, K, cv2.Rodrigues(np.array([0., 1.4, 0.]))[0], t, grazing_dims)
        assert len(geometry['pixels']) <= 1024 and geometry['grazing_rejected'] > 0
        assert np.all(geometry['cosine'] >= .3)
        assert len(np.unique(geometry['pixels'], axis=0)) == len(geometry['pixels'])
        assert all(face['normal'] @ (t + cv2.Rodrigues(np.array([0., 1.4, 0.]))[0] @ T.cuboid(*grazing_dims)[list(indices)].mean(0)) < 0
                   for face in geometry['faces'] for name, indices, _, _ in T.FACES if name == face['name'])
        cases.append('unique_max1024_prediction_only_samples_visible_face_cosine_gate')

        invalid = np.zeros(hw)
        invalid[0, 0] = np.nan
        rejected = T.estimate(invalid, K, R, t, dims)
        assert rejected['reason'] == 'MISSING_SPATIAL_HALF' and rejected['valid_depth_samples'] == 0
        coded = T.estimate(np.ones(hw), K, R, t, dims, invalid_codes_m=[1.])
        assert coded['valid_depth_samples'] == 0 and not coded['accepted']
        cases.append('invalid_nonpositive_and_known_bad_depth_codes_excluded')

        projected = T.project(K, R, t, dims)
        q = np.vstack([projected + [2., -3.], [777., 888.]])
        q[0] = [-1, -1]; q[1] = [np.nan, np.nan]
        support = np.ones(9, bool); support[0:2] = False
        untouched = T.transfer(q, support, K, R, t, t.copy(), dims)
        assert np.array_equal(untouched, q, equal_nan=True)
        moved = T.transfer(q, support, K, R, t, true_t, dims)
        assert np.array_equal(moved[np.r_[~support[:8], True]], q[np.r_[~support[:8], True]], equal_nan=True)
        np.testing.assert_allclose(moved[2:8], T.project(K, R, true_t, dims)[2:8] + [2., -3.], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(q[0], [-1., -1.])
        cases.append('projection_delta_preserves_raw_residual_center_missing_and_exact_nochange')

        signatures = [inspect.signature(fn) for fn in (T.estimate, T.transfer, T.sample_geometry)]
        assert all(not any('gt' in name.lower() or 'reference' in name.lower() for name in signature.parameters) for signature in signatures)
        before = T.estimate(depth, K, R, t, dims)
        irrelevant_gt = np.arange(18).reshape(9, 2) + 10000
        assert irrelevant_gt.shape == (9, 2)
        after = T.estimate(depth, K, R, t, dims)
        assert before == after
        assert all(value == 0 for key, value in counts.items() if key != 'analytic_slab_geometry_calls')
        cases.append('GT_absent_from_API_deterministic_bootstrap_and_no_F_PnP_NN')
    finally:
        T.ray_surface_z = old_slab
        for name, function in old_pnp.items():
            setattr(cv2, name, function)
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return dict(status='PASS', checks=len(cases), passed=cases, counts=counts,
                wall_seconds=time.perf_counter()-start, settings=T.settings(),
                teacher_code_sha256=sha(T.__file__), test_code_sha256=sha(__file__),
                pure_geometry_source_sha256=sha(inspect.getfile(T.cuboid)),
                actual_data_rows=0, actual_DEV_depth_outcomes=0,
                uncertainty_draw_definition='200 stratified spatial-tile replicates; same replicate indices generate A, B and pooled statistics')


if __name__ == '__main__':
    print(json.dumps(run_tests(), ensure_ascii=False, allow_nan=False))
