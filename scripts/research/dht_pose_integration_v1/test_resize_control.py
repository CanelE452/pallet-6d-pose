"""Affine inverse audit with arbitrary belief coordinates, without images/GT."""
import unittest

import numpy as np

try:
    from .resize_control import correct_legacy_resize_coordinates, resize_control_parameters
except ImportError:
    from resize_control import correct_legacy_resize_coordinates, resize_control_parameters


class ResizeControlTests(unittest.TestCase):
    def test_known_640_480_floor_ratio(self):
        params = resize_control_parameters(640, 480, [3, 400, 528])
        np.testing.assert_allclose(params["scale_xy"], [100 / 99, 1], atol=1e-15)

    def test_roundtrip_to_actual_axis_resize_for_arbitrary_belief_points(self):
        rng = np.random.default_rng(29)
        for width, height in ((640, 480), (960, 540), (720, 480), (560, 560),
                              (480, 640), (777, 431)):
            sc = 400 / min(width, height)
            nw, nh = [max(8, int(round(s * sc)) & ~7) for s in (width, height)]
            bw, bh = nw // 8, nh // 8
            belief = rng.uniform(-10, 80, (9, 2)) + 0.4395
            network_pixels = belief * [nw / bw, nh / bh]
            # Established implementation, divided by its nominal shared scale.
            legacy = network_pixels / sc * [(width + 200) / width,
                                            (height + 200) / height] - 100
            corrected, valid = correct_legacy_resize_coordinates(
                legacy, np.ones(9, bool), width, height, [3, nh, nw])
            # Independent direct inverse: undo the ACTUAL per-axis cv2 sizes.
            actual = network_pixels * [width / nw, height / nh]
            actual = actual * [(width + 200) / width, (height + 200) / height] - 100
            np.testing.assert_allclose(corrected, actual, rtol=1e-14, atol=1e-11)
            self.assertTrue(valid.all())

    def test_identity_when_no_floor_change(self):
        points = np.arange(18, dtype=float).reshape(9, 2) / 7
        points[1] = np.nan
        corrected, valid = correct_legacy_resize_coordinates(
            points, np.ones(9, bool), 560, 560, [3, 400, 400])
        np.testing.assert_equal(corrected, points)
        self.assertFalse(valid[1])

    def test_center_transforms_missing_and_confidence_mask_preserved(self):
        points = np.tile([400.0, 217.25], (9, 1))
        points[0] = np.nan
        mask = np.ones(9, bool)
        mask[1] = False
        corrected, valid = correct_legacy_resize_coordinates(
            points, mask, 640, 480, [3, 400, 528])
        self.assertFalse(valid[0])
        self.assertFalse(valid[1])
        self.assertTrue(valid[8])
        self.assertAlmostEqual(corrected[8, 0], 500 * 100 / 99 - 100)
        np.testing.assert_equal(corrected[:, 1], points[:, 1])
        np.testing.assert_equal(corrected[0], points[0])
        np.testing.assert_equal(corrected[1], corrected[8])

    def test_wrong_record_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            resize_control_parameters(640, 480, [3, 400, 400])

    def test_no_padding_variant_roundtrip(self):
        params = resize_control_parameters(640, 480, [3, 400, 528], pad=0)
        np.testing.assert_equal(params["translation_xy"], [0, 0])
        points = np.ones((8, 2))
        corrected, _ = correct_legacy_resize_coordinates(
            points, np.ones(8, bool), 640, 480, [3, 400, 528], pad=0)
        np.testing.assert_allclose(corrected, np.tile([100 / 99, 1], (8, 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
