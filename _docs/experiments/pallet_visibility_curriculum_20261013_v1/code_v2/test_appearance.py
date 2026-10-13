"""Analytic scenes test ambiguity, colour support and state precedence."""
from __future__ import annotations

import json
import unittest

import cv2
import numpy as np

try:
    from .appearance import build_appearance_mask, infer_appearance
except ImportError:
    from appearance import build_appearance_mask, infer_appearance


def scene():
    points = np.array([[30, 30], [110, 30], [110, 90], [30, 90],
                       [50, 45], [130, 45], [130, 105], [50, 105]], dtype=float)
    image = np.full((140, 170, 3), 255, np.uint8)
    cv2.fillConvexPoly(image, cv2.convexHull(points.astype(np.int32)), (70, 160, 60))
    return image, points, np.zeros(8, bool)


class AppearanceTests(unittest.TestCase):
    def test_boundary_and_interior_have_distinct_image_evidence(self):
        image, points, hidden = scene()
        result = infer_appearance(image, points, hidden)
        self.assertEqual(result["states"][0], "VISIBLE_BOUNDARY")
        self.assertEqual(result["states"][4], "VISIBLE_INTERIOR")
        self.assertGreater(result["boundary_score"][0], result["boundary_score"][4] + .3)
        # A uniform interior cannot precisely identify a semantic coordinate.
        self.assertGreater(result["local_accuracy_score"][0], result["local_accuracy_score"][4])
        self.assertGreater(result["visibility_score"][4], .65)

    def test_foreground_background_colour_overlap_remains_unknown(self):
        _, points, hidden = scene()
        image = np.full((140, 170, 3), (70, 160, 60), np.uint8)
        result = infer_appearance(image, points, hidden)
        self.assertEqual(result["states"], ["UNKNOWN"] * 8)
        self.assertEqual(result["diagnostics"]["palette"]["status"],
                         "no_material_background_separation")
        self.assertEqual(np.count_nonzero(build_appearance_mask(image, points, hidden)), 0)

    def test_rgb_mismatch_lowers_support_without_proving_occlusion(self):
        image, points, hidden = scene()
        clean = infer_appearance(image, points, hidden)
        cv2.circle(image, tuple(points[4].astype(int)), 8, (30, 30, 220), -1)
        changed = infer_appearance(image, points, hidden)
        self.assertIn(changed["states"][4], ("UNKNOWN", "OCCLUSION_SUSPECTED"))
        self.assertLess(changed["visibility_score"][4], clean["visibility_score"][4] - .3)
        self.assertLess(changed["local_accuracy_score"][4], clean["local_accuracy_score"][4])

    def test_background_holes_are_excluded_from_material_palette(self):
        image, points, hidden = scene()
        cv2.rectangle(image, (65, 52), (102, 77), (255, 255, 255), -1)
        mask = build_appearance_mask(image, points, hidden)
        self.assertEqual(int(mask[65, 80]), 0)
        self.assertEqual(int(mask[40, 75]), 255)

    def test_self_hidden_green_pixel_does_not_become_visible(self):
        image, points, hidden = scene()
        hidden[4] = True
        result = infer_appearance(image, points, hidden)
        self.assertEqual(result["states"][4], "SELF_OCCLUDED")
        self.assertEqual(result["visibility_score"][4], 0.)
        self.assertEqual(result["local_accuracy_score"][4], 0.)

    def test_missing_and_frame_priority_and_json_serialization(self):
        image, points, hidden = scene()
        points[0] = (-2, 20)
        points[1] = (-1, -1)
        points[2] = (np.nan, 30)
        hidden[:3] = True
        result = infer_appearance(image, np.vstack([points, (80, 60)]), hidden)
        self.assertEqual(result["states"][:3], ["OUT_OF_FRAME", "UNKNOWN", "UNKNOWN"])
        json.dumps(result, allow_nan=False)
        for key in ("states", "visibility_score", "boundary_score", "local_accuracy_score"):
            self.assertEqual(len(result[key]), 8)
        for key in ("visibility_score", "boundary_score", "local_accuracy_score"):
            self.assertTrue(all(0 <= value <= 1 for value in result[key]))

    def test_same_coloured_occluder_is_an_explicit_rgb_identifiability_limit(self):
        image, points, hidden = scene()
        clean = infer_appearance(image, points, hidden)
        # Covering the requested location with an opaque object of exactly the
        # same colour has identical RGB evidence despite different 3D visibility.
        covered = image.copy()
        cv2.circle(covered, tuple(points[4].astype(int)), 8, (70, 160, 60), -1)
        self.assertEqual(clean, infer_appearance(covered, points, hidden))
        self.assertIn("same_coloured_occluders", clean["diagnostics"]["limits"])

    def test_coordinate_error_is_not_a_proven_external_occlusion(self):
        image, points, hidden = scene()
        wrong = points.copy()
        wrong[0] = (20, 20)
        result = infer_appearance(image, wrong, hidden)
        self.assertIn(result["states"][0], ("UNKNOWN", "OCCLUSION_SUSPECTED"))
        self.assertLess(result["local_accuracy_score"][0], .35)

    def test_invalid_inputs_rejected_and_input_arrays_preserved(self):
        image, points, hidden = scene()
        copies = [a.copy() for a in (image, points, hidden)]
        infer_appearance(image, points, hidden)
        for original, copy in zip((image, points, hidden), copies):
            np.testing.assert_array_equal(original, copy)
        with self.assertRaises(ValueError):
            infer_appearance(image[..., 0], points, hidden)
        with self.assertRaises(ValueError):
            infer_appearance(image, points[:7], hidden)
        with self.assertRaises(ValueError):
            infer_appearance(image, points, hidden, {"lab_tolerance": [0, 20, 20]})


if __name__ == "__main__":
    unittest.main()
