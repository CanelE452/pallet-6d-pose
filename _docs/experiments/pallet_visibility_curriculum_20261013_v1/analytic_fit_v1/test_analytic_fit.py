"""Derivative and solver parity checks with independent finite differences."""
import importlib
import unittest

import cv2
import numpy as np

import analytic_fit as A
import curriculum_analytic as experimental
from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as baseline
from scripts.research.pallet_visibility_curriculum_20261013_v1.test_curriculum import fixture


def finite_jacobian(function, params, step=2e-6):
    columns = []
    for column in range(6):
        delta = np.zeros(6)
        delta[column] = step
        columns.append((function(params + delta) - function(params - delta)) / (2 * step))
    return np.column_stack(columns)


class AnalyticFitTests(unittest.TestCase):
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

            exact = A.projection_jacobian(params, X0, K, initial_t, depth_scale)
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

        exact = A.weighted_residual_jacobian(params, xyz @ initial_R.T, K, initial_t, scale, weights, cfg)
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
            new_R, new_t, new_info = experimental._fit(*arguments)
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
        baseline._fit = experimental._fit
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.CurriculumTests)
            result = unittest.TestResult()
            suite.run(result)
        finally:
            baseline._fit = original
        self.assertEqual(result.testsRun, 13)
        self.assertTrue(result.wasSuccessful(), str(result.failures + result.errors))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    unittest.main()
