"""CNN mask fusion contracts; hand masks do not certify learned ownership."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np


try:
    from .segmented_appearance import infer_segmented_appearance, mask_from_probability
    from .appearance import infer_appearance, build_appearance_mask
except ImportError:
    root = Path("/home/minjae/Documents/github/pallet-pose")
    sys.path.insert(0, str(root))
    package = "scripts.research.pallet_visibility_curriculum_20261013_v1"
    spec = importlib.util.spec_from_file_location(package + ".segmented_appearance", Path(__file__).with_name("segmented_appearance.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    infer_segmented_appearance, mask_from_probability = module.infer_segmented_appearance, module.mask_from_probability
    from scripts.research.pallet_visibility_curriculum_20261013_v1.appearance import infer_appearance, build_appearance_mask


def scene():
    image = np.full((120, 160, 3), 215, np.uint8)
    image[25:91, 25:116] = [45, 130, 55]
    points = np.array([[25, 25], [115, 25], [115, 90], [25, 90],
                       [42, 43], [90, 43], [90, 72], [42, 72]], float)
    probability = np.zeros(image.shape[:2], np.float32)
    probability[25:91, 25:116] = .9
    return image, points, np.zeros(8, bool), probability


class SegmentedAppearanceTests(unittest.TestCase):
    def test_mask_support_survives_colour_palette_ambiguity(self):
        image, points, hidden, probability = scene()
        image[:] = 80
        original = infer_appearance(image, points, hidden)
        fused = infer_segmented_appearance(image, points, hidden, probability)
        self.assertEqual(original["diagnostics"]["palette"]["centers_lab"], [])
        self.assertEqual(fused["diagnostics"]["palette"], original["diagnostics"]["palette"])
        self.assertGreater(fused["visibility_score"][4], .65)
        self.assertEqual(fused["states"][4], "VISIBLE_INTERIOR")
        self.assertEqual(fused["diagnostics"]["mask_source"], "frozen_source_RGB_CNN")

    def test_known_absence_in_supplied_mask_lowers_support(self):
        image, points, hidden, probability = scene()
        first = infer_segmented_appearance(image, points, hidden, probability)
        changed = probability.copy()
        changed[27:59, 26:59] = 0
        after = infer_segmented_appearance(image, points, hidden, changed)
        self.assertLess(after["visibility_score"][4], first["visibility_score"][4] - .5)
        self.assertNotIn(after["states"][4], ["VISIBLE_BOUNDARY", "VISIBLE_INTERIOR"])
        self.assertEqual(after["diagnostics"]["corners"][4]["segmentation_probability_patch_mean"], 0.)

    def test_colour_matching_distractor_can_be_excluded_by_independent_mask(self):
        image, points, hidden, probability = scene()
        image[40:75, 124:145] = [45, 130, 55]
        raw = build_appearance_mask(image, points, hidden)
        self.assertGreater(int(raw[60, 127]), 0)
        fused_mask = mask_from_probability(probability, points)
        self.assertEqual(int(fused_mask[60, 127]), 0)
        self.assertGreater(int(fused_mask[60, 80]), 0)
        # This validates the fusion when an independent mask is available;
        # it does not assert that the trained CNN detects this distractor.

    def test_same_colour_same_CNN_output_remains_unidentifiable(self):
        image, points, hidden, probability = scene()
        first = infer_segmented_appearance(image, points, hidden, probability)
        # A different ownership explanation with identical inputs cannot be
        # distinguished, including when the CNN misses a same-coloured occluder.
        self.assertEqual(first, infer_segmented_appearance(image.copy(), points.copy(), hidden.copy(), probability.copy()))

    def test_geometric_state_priorities_and_original_arrays(self):
        image, points, hidden, probability = scene()
        points[0] = [-5, 20]
        points[1] = [-1, -1]
        hidden[[0, 2, 4]] = True
        before = [a.copy() for a in [image, points, hidden, probability]]
        result = infer_segmented_appearance(image, points, hidden, probability)
        self.assertEqual(result["states"][0], "OUT_OF_FRAME")
        self.assertEqual(result["states"][1], "UNKNOWN")
        self.assertEqual(result["states"][2], "SELF_OCCLUDED")
        self.assertEqual(result["visibility_score"][4], 0.)
        for a, b in zip([image, points, hidden, probability], before):
            np.testing.assert_array_equal(a, b)
        json.dumps(result, allow_nan=False)

    def test_hull_ROI_excludes_far_background_CNN_positives(self):
        _, points, _, probability = scene()
        probability[:] = .9
        mask = mask_from_probability(probability, points)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(mask[115, 155]), 0)
        self.assertEqual(int(mask[50, 70]), 255)

    def test_native_probability_shape_and_range_are_required(self):
        image, points, hidden, probability = scene()
        for wrong in [probability[:-1], np.full_like(probability, 1.1), np.full_like(probability, np.nan)]:
            with self.assertRaises(ValueError):
                infer_segmented_appearance(image, points, hidden, wrong)

    def test_raw_Lab_features_keep_same_meanings(self):
        image, points, hidden, probability = scene()
        raw = infer_appearance(image, points, hidden)
        fused = infer_segmented_appearance(image, points, hidden, probability)
        for k in range(8):
            a, b = raw["diagnostics"]["corners"][k], fused["diagnostics"]["corners"][k]
            for feature in ["patch_lab_mean", "patch_lab_std", "inward_lab_mean", "patch_chroma_mean",
                            "soft_inward_material_support", "soft_nearby_inward_material_support"]:
                self.assertEqual(a[feature], b[feature])


if __name__ == "__main__":
    unittest.main()
