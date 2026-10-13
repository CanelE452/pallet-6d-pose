"""Derivative and solver parity checks with independent finite differences."""
import importlib
from dataclasses import replace
import unittest

import cv2
import numpy as np

from . import curriculum as baseline
from . import common as C
from .test_curriculum import fixture
from .test_live_boundary import production_access_canary


def finite_jacobian(function, params, step=2e-6):
    columns = []
    for column in range(6):
        delta = np.zeros(6)
        delta[column] = step
        columns.append((function(params + delta) - function(params - delta)) / (2 * step))
    return np.column_stack(columns)


class AnalyticFitTests(unittest.TestCase):
    def test_analytic_derivative_is_opt_in(self):
        self.assertFalse(baseline.CurriculumConfig().analytic_jacobian)
        first = C.read(C.DOC / 'LIVE_CONFIG_V1.json')
        fast = C.read(C.DOC / 'LIVE_CONFIG_FAST_V1.json')
        self.assertNotIn('analytic_jacobian', first['curriculum'])
        self.assertTrue(fast['curriculum']['analytic_jacobian'])
        self.assertEqual(first['visibility_model'], fast['visibility_model'])
        self.assertEqual(first['segmentation_weights'], fast['segmentation_weights'])
        self.assertEqual(first['curriculum'], {k: v for k, v in fast['curriculum'].items() if k != 'analytic_jacobian'})

    def test_full_K_local_rotation_and_normalized_translation_derivative(self):
        rng = np.random.default_rng(20261013)
        maximum_error = 0.
        for case in range(120):
            xyz = rng.uniform(-.45, .45, (rng.integers(4, 10), 3)) * 10 ** rng.uniform(-1, .4)
            initial_R = cv2.Rodrigues(rng.normal(size=3))[0]
            initial_t = np.r_[rng.uniform(-1., 1., 2), rng.uniform(4., 25.)]
            depth_scale = max(np.linalg.norm(initial_t), np.linalg.norm(np.ptp(xyz, axis=0)))
            K = np.array([[rng.uniform(200., 1800.), rng.uniform(-70., 70.), rng.uniform(50., 900.)],
                          [rng.uniform(-20., 20.), rng.uniform(200., 1800.), rng.uniform(50., 700.)],
                          [rng.uniform(-2e-4, 2e-4), rng.uniform(-2e-4, 2e-4), rng.uniform(.9, 1.1)]])
            angle = [0., 1e-9, .02, .25, 1.4, 3.0][case % 6]
            axis = rng.normal(size=3)
            params = np.r_[axis / np.linalg.norm(axis) * angle, rng.uniform(-.03, .03, 3)]
            X0 = xyz @ initial_R.T

            def function(p):
                R = cv2.Rodrigues(p[:3])[0] @ initial_R
                return baseline.project(xyz, R, initial_t + p[3:] * depth_scale, K).reshape(-1)

            exact = baseline._projection_jacobian(params, X0, K, initial_t, depth_scale)
            numeric = finite_jacobian(function, params)
            error = float(np.max(np.abs(exact - numeric)))
            maximum_error = max(maximum_error, error)
            self.assertLessEqual(error, 1e-6, f'case={case}, angle={angle}')
        print('JACOBIAN_MAX_ABSOLUTE_ERROR', maximum_error)

    def test_frozen_weights_zero_weights_and_prior_rows(self):
        xyz, K, _, _, initial_R, initial_t, points = fixture()
        cfg = baseline.CurriculumConfig(prior_strength_px=4.)
        scale = np.linalg.norm(initial_t)
        params = np.array([.08, -.11, .02, .01, -.005, .015])
        weights = np.array([0., .01, .10, .35, .65, .90, 1., .75])

        def function(p):
            R = cv2.Rodrigues(p[:3])[0] @ initial_R
            residual = (baseline.project(xyz, R, initial_t + p[3:] * scale, K) - points) * np.sqrt(weights[:, None])
            return np.r_[residual.reshape(-1), cfg.prior_strength_px * p[:3] / np.deg2rad(cfg.rotation_prior_deg),
                         cfg.prior_strength_px * p[3:] / cfg.translation_prior_fraction]

        exact = baseline._weighted_residual_jacobian(params, xyz @ initial_R.T, K, initial_t, scale, weights, cfg)
        np.testing.assert_allclose(exact, finite_jacobian(function, params), atol=1e-6, rtol=0.)
        np.testing.assert_array_equal(exact[:2], 0.)

    def test_robust_irls_solution_matches_without_changing_the_objective(self):
        rng = np.random.default_rng(901)
        xyz, K, _, _, initial_R, initial_t, truth = fixture()
        maximum_uv_difference = 0.
        for case in range(30):
            points = truth + rng.normal(0., .1 + case % 3, truth.shape)
            if case % 2:
                points[2] += [30., -25.]
            ids = [0, 1, 4, 5, 6] if case % 3 == 0 else list(range(8))
            weights = rng.uniform(.05, 1., len(ids))
            cfg = baseline.CurriculumConfig(prior_strength_px=[0., 1., 4.][case % 3])
            arguments = (xyz, points, K, initial_R, initial_t, initial_R, initial_t, ids, weights, cfg)
            old_R, old_t, old_info = baseline._fit(*arguments)
            new_R, new_t, new_info = baseline._fit(*arguments[:-1], replace(cfg, analytic_jacobian=True))
            old_uv = baseline.project(xyz, old_R, old_t, K)
            new_uv = baseline.project(xyz, new_R, new_t, K)
            delta = float(np.max(np.abs(new_uv - old_uv)))
            maximum_uv_difference = max(maximum_uv_difference, delta)
            self.assertLess(delta, 3e-5, f'case={case}')
            np.testing.assert_allclose(new_info['robust_weights'], old_info['robust_weights'], atol=2e-7, rtol=0.)
        print('SYNTHETIC_MAX_UV_DIFFERENCE_PX', maximum_uv_difference)

    def test_existing_curriculum_semantics_with_only_fit_replaced(self):
        module = importlib.import_module('scripts.research.pallet_visibility_curriculum_20261013_v1.test_curriculum')
        original = baseline._fit
        def analytic_fit(*args):
            return original(*args[:-1], replace(args[-1], analytic_jacobian=True))

        baseline._fit = analytic_fit
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.CurriculumTests)
            result = unittest.TestResult()
            suite.run(result)
        finally:
            baseline._fit = original
        self.assertEqual(result.testsRun, 13)
        self.assertTrue(result.wasSuccessful(), str(result.failures + result.errors))

    @unittest.skipUnless(all((C.DOC / p).exists() for p in (
        'PREDICTION_INPUTS.json', 'cal_segmentation_v2/PREDICTIONS.jsonl.gz',
        'test_v1/PREDICTIONS.jsonl.gz')), 'Optional sealed prediction-only replay fixtures unavailable')
    def test_frozen_dev173_prediction_only_parity(self):
        frames = C.read(C.DOC / 'PREDICTION_INPUTS.json')['frames']
        rows = {r['id']: r for name in ('cal_segmentation_v2', 'test_v1')
                for r in C.rows(C.DOC / name / 'PREDICTIONS.jsonl.gz')
                if r['method'] == 'SEGMENTED_RGB_CURRICULUM'}
        self.assertEqual(len(frames), 173)
        self.assertEqual({f['id'] for f in frames}, set(rows))
        config = C.read(C.DOC / 'LIVE_CONFIG_V1.json')['curriculum']
        for frame in frames:
            row = rows[frame['id']]
            pose = frame['initial_poses']['N3_SUBPIX']
            xyz = C.cuboid(*pose['cf_extents'])
            arguments = (row['original_points'], xyz, frame['K'], pose['R_cf'],
                         pose['centroid'], row['self_hidden'], row['appearance'])
            with production_access_canary():
                old = baseline.solve_curriculum(*arguments, config)
                new = baseline.solve_curriculum(*arguments, dict(config, analytic_jacobian=True))
            with self.subTest(frame=frame['id']):
                np.testing.assert_allclose(old['R'], row['solution']['R'], atol=1e-12, rtol=0.)
                np.testing.assert_allclose(old['t'], row['solution']['t'], atol=1e-12, rtol=0.)
                self.assertEqual(len(old['stages']), len(new['stages']))
                self.assertEqual(old['used_ids'], new['used_ids'])
                self.assertEqual(old['accepted'], new['accepted'])
                self.assertEqual(old['observed_points'], new['observed_points'])
                for a, b in zip(old['stages'], new['stages']):
                    for field in ('name', 'accepted', 'reason', 'used_ids', 'added_ids',
                                  'heldout_ids', 'protected_visible_ids'):
                        self.assertEqual(a.get(field), b.get(field))
                    np.testing.assert_allclose(a['R'], b['R'], atol=1e-7, rtol=0.)
                    np.testing.assert_allclose(a['t'], b['t'], atol=1e-7, rtol=0.)
                    np.testing.assert_allclose(a['projected'], b['projected'], atol=1e-4, rtol=0.)
                Q = np.asarray(pose['R_cf']).T @ np.asarray(pose['R_physical'])
                np.testing.assert_allclose(np.asarray(old['R']) @ Q,
                                           row['actual_pose']['R_physical'], atol=1e-12, rtol=0.)
                np.testing.assert_allclose(np.asarray(new['R']) @ Q,
                                           row['actual_pose']['R_physical'], atol=1e-7, rtol=0.)
                self.assertFalse(set(new['used_ids']) & set(np.flatnonzero(row['self_hidden'])))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
