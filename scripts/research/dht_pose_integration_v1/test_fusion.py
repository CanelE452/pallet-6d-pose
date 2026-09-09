"""CPU-only geometry/selection-metric contract checks; no models or real data."""
import unittest

import numpy as np

try:
    from .fusion import (INCIDENT_ROLES, SIDE_EDGES, corner_metrics,
                         corners_to_lines, fuse_corners, gt_line_support, line_metrics)
except ImportError:
    from fusion import (INCIDENT_ROLES, SIDE_EDGES, corner_metrics,
                        corners_to_lines, fuse_corners, gt_line_support, line_metrics)


class FusionGeometryTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[110, 150], [420, 150], [420, 230], [110, 230],
                               [180, 100], [490, 100], [490, 180], [180, 180]], float)
        self.valid = np.ones(8, bool)
        self.lines, _ = corners_to_lines(self.truth, self.valid)

    def test_topology_is_two_incident_roles_per_corner(self):
        self.assertEqual(SIDE_EDGES, ((1, 2), (3, 0), (5, 6), (7, 4),
                                     (0, 4), (1, 5), (2, 6), (3, 7)))
        self.assertEqual(INCIDENT_ROLES, ((1, 4), (0, 5), (0, 6), (1, 7),
                                        (3, 4), (2, 5), (2, 6), (3, 7)))

    def test_lambda_zero_exact_identity_and_preserves_center_missing(self):
        p = np.vstack([self.truth, [123.456, -12.75]])
        p[0] = np.nan
        p[1] = [-100, 800]
        valid = np.r_[self.valid, True]
        q, mask = fuse_corners(p, valid, self.lines, 0, 640, 480)
        np.testing.assert_equal(q, p)
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(np.shares_memory(q, p))

    def test_parallel_lines_unique_solution_preserves_tangent(self):
        # Both incident constraints x=2: anchored x=(p_x + 2*lambda*2)/(1+2*lambda).
        lines = np.tile(np.array([[2, -100], [2, 100.0]]), (8, 1, 1))
        p = np.tile([10.0, 27.0], (8, 1))
        q, _ = fuse_corners(p, self.valid, lines, 4, 640, 480)
        np.testing.assert_allclose(q, np.tile([26 / 9, 27], (8, 1)), atol=1e-12)

    def test_isotropic_scale_and_translation_equivariance(self):
        p = self.truth + np.array([19.0, -11])
        q, _ = fuse_corners(p, self.valid, self.lines, 2, 640, 480)
        scale, shift = 3.7, np.array([-302.0, 183])
        transformed, _ = fuse_corners(p * scale + shift, self.valid,
                                      self.lines * scale + shift, 2,
                                      640 * scale, 480 * scale)
        np.testing.assert_allclose(transformed, q * scale + shift, atol=1e-10)

    def test_endpoint_reversal_does_not_change_fusion(self):
        p = self.truth + [19, -11]
        q, _ = fuse_corners(p, self.valid, self.lines, 1, 640, 480)
        reversed_q, _ = fuse_corners(p, self.valid, self.lines[:, ::-1], 1, 640, 480)
        np.testing.assert_allclose(q, reversed_q, atol=1e-12)

    def test_solution_has_zero_objective_gradient(self):
        p = self.truth + np.random.default_rng(19).normal(0, 35, (8, 2))
        q, _ = fuse_corners(p, self.valid, self.lines, 0.5, 640, 480)
        for k, roles in enumerate(INCIDENT_ROLES):
            gradient = q[k] - p[k]
            for j in roles:
                a, b = self.lines[j]
                delta = b - a
                normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
                gradient += 0.5 * normal * np.dot(normal, q[k] - a)
            np.testing.assert_allclose(gradient, [0, 0], atol=1e-12)

    def test_missing_corners_and_center_never_reconstructed(self):
        p = np.vstack([self.truth + [19, -11], [333.125, 201.875]])
        p[2] = np.nan
        p[4] = [-1, -1]
        mask = self.valid.copy()
        mask[4] = False
        q, returned_mask = fuse_corners(p, mask, self.lines, 4, 640, 480)
        np.testing.assert_equal(q[[2, 4, 8]], p[[2, 4, 8]])
        self.assertEqual(returned_mask.shape, (8,))
        self.assertFalse(returned_mask[2])
        self.assertFalse(returned_mask[4])
        self.assertEqual(int(returned_mask.sum()), 6)

    def test_undefined_lines_are_ignored_without_gt_support_input(self):
        lines = np.full((8, 2, 2), np.nan)
        lines[0] = 0  # Finite zero-length line is equally undefined.
        p = self.truth + [19, -11]
        q, _ = fuse_corners(p, self.valid, lines, 4, 640, 480)
        np.testing.assert_equal(q, p)

    def test_correct_lines_help_but_wrong_lines_can_harm(self):
        p = self.truth + [19, -11]
        q, _ = fuse_corners(p, self.valid, self.lines, 4, 640, 480)
        self.assertLess(np.linalg.norm(q - self.truth), np.linalg.norm(p - self.truth))
        wrong_q, _ = fuse_corners(self.truth, self.valid, self.lines + 100, 4, 640, 480)
        self.assertGreater(np.linalg.norm(wrong_q - self.truth), 50)

    def test_eval_preserves_corner_identity_and_penalizes_missing(self):
        p = self.truth.copy()
        p[[0, 1]] = p[[1, 0]]
        m = corner_metrics(p, self.valid, self.truth, self.valid, 640, 480)
        self.assertEqual(m["mean_error_missing_diagonal_px"], 77.5)
        self.assertEqual(m["pck_10px"], 0.75)
        p = self.truth.copy()
        p[0] = np.nan
        m = corner_metrics(p, self.valid, self.truth, self.valid, 640, 480)
        self.assertEqual(m["coverage"], 7 / 8)
        self.assertEqual(m["mean_error_px"], 0)
        self.assertEqual(m["mean_error_missing_diagonal_px"], 100)
        self.assertEqual(m["mean_error_missing_diagonal_normalized"], 1 / 8)

    def test_selection_caps_finite_failures_but_reporting_does_not(self):
        p = self.truth.copy()
        p[0] += [1600, 0]
        m = corner_metrics(p, self.valid, self.truth, self.valid, 640, 480)
        self.assertEqual(m["mean_error_missing_diagonal_px"], 200)
        self.assertEqual(m["mean_error_missing_diagonal_normalized"], 0.25)
        self.assertEqual(m["mean_capped_diagonal_normalized_error"], 0.125)
        self.assertFalse(m["all8_at_10px"])
        m = corner_metrics(self.truth, self.valid, self.truth, self.valid, 640, 480)
        self.assertTrue(m["all8_at_10px"])
        gt_valid = self.valid.copy()
        gt_valid[0] = False
        m = corner_metrics(p, self.valid, self.truth, gt_valid, 640, 480)
        self.assertIsNone(m["all8_at_10px"])

    def test_eval_gt_changes_do_not_modify_predictions(self):
        p = self.truth + [19, -11]
        q, valid = fuse_corners(p, self.valid, self.lines, 1, 640, 480)
        frozen_q, frozen_p, frozen_lines = q.copy(), p.copy(), self.lines.copy()
        for gt in (self.truth, self.truth + [80, -30]):
            corner_metrics(q, valid, gt, self.valid, 640, 480)
            line_metrics(*corners_to_lines(q, valid), gt, self.valid, 640, 480)
        np.testing.assert_equal(q, frozen_q)
        np.testing.assert_equal(p, frozen_p)
        np.testing.assert_equal(self.lines, frozen_lines)

    def test_line_metrics_zero_and_missing_endpoint_coverage(self):
        m = line_metrics(self.lines, self.valid, self.truth, self.valid, 640, 480)
        np.testing.assert_allclose(m["distance_px"], 0, atol=1e-12)
        np.testing.assert_allclose(m["angle_deg"], 0, atol=1e-6)
        mask = self.valid.copy()
        mask[0] = False
        lines, valid = corners_to_lines(self.truth, mask)
        m = line_metrics(lines, valid, self.truth, self.valid, 640, 480)
        self.assertEqual(m["n_pred_on_gt"], 6)
        self.assertEqual(m["mean_distance_missing_diagonal_px"], 200)
        self.assertEqual(m["mean_angle_missing_90deg"], 22.5)

    def test_structural_gt_support_excludes_short_and_offframe(self):
        truth = self.truth + [1000, 0]
        self.assertFalse(gt_line_support(truth, self.valid, 640, 480).any())
        truth = self.truth.copy()
        truth[2] = truth[1] + [0, 1]
        self.assertFalse(gt_line_support(truth, self.valid, 640, 480)[0])

    def test_invalid_lambda_rejected(self):
        for lam in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fuse_corners(self.truth, self.valid, self.lines, lam, 640, 480)


if __name__ == "__main__":
    unittest.main(verbosity=2)
