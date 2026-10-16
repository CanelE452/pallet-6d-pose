"""Known-image geometry tests independent RGB coordinate observations."""
from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

try:
    from .boundary_refine import infer_rgb_boundary_points
except ImportError:
    from boundary_refine import infer_rgb_boundary_points


def fixture():
    image = np.full((128, 128, 3), 255, np.uint8)
    cv2.rectangle(image, (24, 24), (104, 104), (65, 160, 70), -1)
    points = np.array([[24, 24], [104, 24], [104, 104], [24, 104],
                       [45, 45], [80, 45], [80, 80], [45, 80]], float)
    appearance = {"states": ["VISIBLE_BOUNDARY"] * 4 + ["VISIBLE_INTERIOR"] * 4,
                  "visibility_score": [1.] * 8, "boundary_score": [1.] * 4 + [0.] * 4,
                  "local_accuracy_score": [.8] * 4 + [.18] * 4}
    return image, points, np.zeros(8, bool), appearance


class RGBBoundaryTests(unittest.TestCase):
    def test_128_independent_noisy_seed_scenes_recover_image_outline(self):
        rng = np.random.default_rng(20261013)
        before, after, moved = [], [], 0
        for _ in range(128):
            image, truth, hidden, appearance = fixture()
            seed = truth.copy()
            seed[:4] += rng.uniform(-3., 3., (4, 2))
            result = infer_rgb_boundary_points(image, seed, hidden, appearance)
            predicted = np.asarray(result["points"])
            before.append(np.linalg.norm(seed[:4] - truth[:4], axis=1).mean())
            after.append(np.linalg.norm(predicted[:4] - truth[:4], axis=1).mean())
            moved += len(result["observed_refined_ids"])
            np.testing.assert_array_equal(predicted[4:], seed[4:])
            self.assertTrue(all(np.linalg.norm(predicted[k] - seed[k]) <= 6 + 1e-9
                                for k in result["observed_refined_ids"]))
        self.assertGreater(moved, 450)
        self.assertLess(np.mean(after), .9)
        self.assertLess(np.mean(after), np.mean(before) * .45)

    def test_structured_background_gray_edges_do_not_replace_material_outline(self):
        image, truth, hidden, appearance = fixture()
        # Strong independent gray edges around the object do not satisfy the
        # green-material inward/background outward transition criterion.
        for x in range(5, 124, 8):
            cv2.line(image, (x, 2), (x, 16), (0, 0, 0), 2)
            cv2.line(image, (x, 112), (x, 125), (0, 0, 0), 2)
        for y in range(5, 124, 8):
            cv2.line(image, (2, y), (16, y), (0, 0, 0), 2)
            cv2.line(image, (112, y), (125, y), (0, 0, 0), 2)
        seed = truth.copy()
        seed[:4] += [[2, -2], [-2, 2], [2, 2], [-2, -2]]
        result = infer_rgb_boundary_points(image, seed, hidden, appearance)
        self.assertEqual(result["observed_refined_ids"], [0, 1, 2, 3])
        self.assertLess(np.linalg.norm(np.asarray(result["points"])[:4] - truth[:4], axis=1).max(), 1.2)

    def test_single_observed_line_cannot_create_corner_coordinates(self):
        image, points, hidden, appearance = fixture()
        image[:] = 255
        image[65:] = (65, 160, 70)
        points[:4] = [[30, 30], [95, 30], [95, 65], [30, 65]]
        result = infer_rgb_boundary_points(image, points, hidden, appearance)
        self.assertEqual(result["observed_refined_ids"], [])
        self.assertLessEqual(result["diagnostics"]["fitted_edge_count"], 1)
        np.testing.assert_array_equal(result["points"], points)

    def test_low_semantic_visibility_and_self_hidden_points_are_preserved(self):
        image, points, hidden, appearance = fixture()
        low = dict(appearance, visibility_score=[.2] * 8)
        result = infer_rgb_boundary_points(image, points, hidden, low)
        self.assertEqual(result["observed_refined_ids"], [])
        np.testing.assert_array_equal(result["points"], points)
        hidden[0] = True
        result = infer_rgb_boundary_points(image, points, hidden, appearance)
        self.assertNotIn(0, result["observed_refined_ids"])
        np.testing.assert_array_equal(result["points"][0], points[0])

    def test_shift_limit_center_preservation_and_shared_edge_diagnostics(self):
        image, truth, hidden, appearance = fixture()
        seed = np.vstack([truth, [64, 64]])
        seed[:4, 0] += 9
        result = infer_rgb_boundary_points(image, seed, hidden, appearance)
        self.assertEqual(result["observed_refined_ids"], [])
        np.testing.assert_array_equal(result["points"], seed)
        self.assertTrue(any(r["reason"] == "rgb_intersection_exceeds_shift_limit"
                            for r in result["corner_records"]))
        seed[:4, 0] -= 7
        result = infer_rgb_boundary_points(image, seed, hidden, appearance)
        self.assertEqual(result["observed_refined_ids"], [0, 1, 2, 3])
        np.testing.assert_array_equal(result["points"][8], seed[8])
        self.assertEqual(len(result["diagnostics"]["shared_edge_correlations"]), 4)
        self.assertFalse(result["reprojected_points_used"])
        self.assertIn("virtual_bbox_or_rounded_corner", result["diagnostics"]["limits"])
        json.dumps(result, allow_nan=False)

    def test_semantic_visible_hull_corner_requires_new_rgb_outline_evidence(self):
        image, truth, hidden, appearance = fixture()
        seed = truth.copy()
        seed[:4] += 2
        # A semantic interior state can still occupy a geometric outline vertex.
        # Search is permitted, but two observed RGB transitions remain required.
        appearance.update(states=["VISIBLE_INTERIOR"] * 8,
                          visibility_score=[.65] * 8, boundary_score=[0.] * 8,
                          local_accuracy_score=[0.] * 8)
        result = infer_rgb_boundary_points(image, seed, hidden, appearance)
        self.assertEqual(result["observed_refined_ids"], [0, 1, 2, 3])
        np.testing.assert_array_equal(result["points"][4:], seed[4:])
        self.assertTrue(result["diagnostics"]["appearance_boundary_used_as_prior"])

    def test_supplied_predicted_probability_mask_is_separate_from_metadata(self):
        image, truth, hidden, appearance = fixture()
        seed = truth.copy()
        seed[:4] += 2
        probability = np.full(image.shape[:2], .02, np.float32)
        probability[24:105, 24:105] = .98
        image[:] = 120
        result = infer_rgb_boundary_points(image, seed, hidden, appearance,
                    material_mask=probability, mask_source="synthetic_predicted_foreground")
        self.assertEqual(result["observed_refined_ids"], [0, 1, 2, 3])
        self.assertEqual(result["diagnostics"]["mask_source"], "synthetic_predicted_foreground")
        json.dumps(result, allow_nan=False)
        with self.assertRaises(ValueError):
            infer_rgb_boundary_points(image, seed, hidden, appearance, material_mask=probability * 2)


if __name__ == "__main__":
    unittest.main()
