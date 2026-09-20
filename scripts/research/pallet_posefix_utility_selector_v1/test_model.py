"""Synthetic CPU-only tests; never load training/evaluation data or models."""

import inspect
import unittest

import numpy as np
import torch

from . import model as M


class UtilitySelectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)
        yy, xx = np.mgrid[:240, :320]
        self.image[:, :, 0] = xx % 256
        self.image[:, :, 1] = yy % 256
        self.image[:, :, 2] = (xx + yy) % 256
        self.box = np.array([60., 40., 260., 200.])
        self.q = np.array([[80 + 20 * i, 70 + 9 * i] for i in range(9)], float)
        self.valid = np.ones(9, bool)
        self.heatmap = dict(mode_original_xy9=self.q.copy(), entropy_normalized9=np.full(9, .5),
                            peak_mass7x7_9=np.full(9, .3), peak_probability9=np.full(9, .03))

    def feature(self, n2=None, replay=None):
        return M.features(self.image, self.q, self.q if n2 is None else n2,
                          self.q if replay is None else replay, self.box, self.valid, self.heatmap)

    def test_feature_api_has_no_target_argument(self):
        self.assertEqual(tuple(inspect.signature(M.features).parameters),
                         ("bgr", "r0", "n2", "replay", "box", "valid", "heatmap"))
        values = self.feature()
        self.assertEqual(set(values), {"rgb", "patches", "numeric", "global_numeric"})
        self.assertEqual(values["rgb"].shape, (3, 96, 72))
        self.assertEqual(values["patches"].shape, (8, 6, 24, 24))
        self.assertEqual(values["numeric"].shape, (8, 26))
        self.assertEqual(values["global_numeric"].shape, (64,))
        self.assertEqual(values["patches"].dtype, np.uint8)
        self.assertEqual(values["numeric"].dtype, np.float32)
        self.assertEqual(len(M.NUMERIC_FIELDS), 26)
        self.assertEqual(len(M.GLOBAL_FIELDS), 64)

    def test_patch_center_rgb_order_and_candidate_consistency(self):
        values = self.feature()
        np.testing.assert_array_equal(values["patches"][:, :3], values["patches"][:, 3:])
        for i in range(8):
            x, y = self.q[i].astype(int)
            np.testing.assert_array_equal(values["patches"][i, :3, 12, 12], self.image[y, x, ::-1])
        np.testing.assert_array_equal(values["numeric"][:, 6:14], np.zeros((8, 8)))
        diag = np.linalg.norm(self.box[2:] - self.box[:2])
        matrix = M._patch_matrix(self.q[0], .12 * diag)
        np.testing.assert_allclose(matrix @ np.r_[self.q[0], 1], [12, 12], atol=1e-12)

    def test_coordinate_translation_invariance(self):
        base = self.feature()
        # Same pixels/candidates embedded into a larger prepared-image canvas.
        padded = np.pad(self.image, ((20, 20), (20, 20), (0, 0)))
        heat = dict(self.heatmap, mode_original_xy9=self.q + 20)
        shifted = M.features(padded, self.q + 20, self.q + 20, self.q + 20,
                              self.box + 20, self.valid, heat)
        np.testing.assert_array_equal(base["patches"], shifted["patches"])
        np.testing.assert_allclose(base["numeric"], shifted["numeric"], atol=1e-7)
        np.testing.assert_allclose(base["global_numeric"], shifted["global_numeric"], atol=1e-7)

    def test_targets_signed_and_out_of_crop_included(self):
        gt = self.q.copy()
        n2, replay = gt + [3., 4.], gt + [0., 1.]
        gt[0] = [10000, -10000]
        n2[0], replay[0] = gt[0] + [3, 4], gt[0] + [0, 1]
        result = M.utility_targets(n2, replay, gt, self.valid, [np.arange(9)], self.valid, 200.)
        np.testing.assert_allclose(result["target"], np.full(8, 2.))
        self.assertTrue(result["mask"].all())
        self.assertFalse(result["gt_valid"][8])
        self.assertEqual(result["branch"], 0)

    def test_one_whole_branch_held_for_both_candidates(self):
        gt = self.q.copy()
        permutation = np.array([5, 4, 7, 6, 1, 0, 3, 2, 8])
        n2 = gt[permutation].copy()
        replay = gt.copy()
        result = M.utility_targets(n2, replay, gt, self.valid,
                                    [np.arange(9), permutation], self.valid, 100.)
        self.assertEqual(result["branch"], 1)
        np.testing.assert_array_equal(result["gt_aligned"], gt[permutation])
        self.assertTrue((result["target"] < 0).all())
        # Per-corner nearest-target or separate Replay branch would yield zero,
        # wrongly rewarding an inconsistent replacement; neither is allowed.
        self.assertLess(float(result["target"].sum()), -100)

    def test_pairs_strictly_positive_and_exact_fallback(self):
        utility = np.ones(8)
        utility[0] = 0
        utility[1] = np.nan
        valid = self.valid.copy()
        valid[4] = False
        selected = M.choose_pairs(utility, valid)
        np.testing.assert_array_equal(selected, [False, False, False, True])
        base = self.q.astype(np.float32)
        base[0, 0] = -0.
        candidate = self.q + 100
        output = M.apply_pairs(base, candidate, selected)
        self.assertEqual(output.dtype, base.dtype)
        self.assertEqual(output[[0, 1, 2, 3, 4, 7, 8]].tobytes(), base[[0, 1, 2, 3, 4, 7, 8]].tobytes())
        np.testing.assert_array_equal(output[[5, 6]], candidate[[5, 6]])
        self.assertEqual(M.apply_pairs(base, candidate, [False] * 4).tobytes(), base.tobytes())

    def test_model_shapes_parameter_budget_and_gradient(self):
        torch.manual_seed(1)
        model = M.UtilitySelector(26, 64)
        self.assertLessEqual(sum(p.numel() for p in model.parameters()), 150000)
        arrays = self.feature()
        batch = {key: torch.as_tensor(np.stack([value, value])) for key, value in arrays.items()}
        output = model(batch)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.isfinite(output).all())
        torch.nn.functional.smooth_l1_loss(output, torch.full_like(output, 2.), beta=1.).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(model.local[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.global_image[0].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.predictor[-1].weight.grad.abs().sum()), 0.)

    def test_invalid_mask_and_bad_inputs(self):
        bad = self.q.copy()
        bad[0] = np.nan
        values = self.feature(n2=bad)
        self.assertTrue(np.isfinite(values["numeric"]).all())
        self.assertEqual(values["numeric"][0, 17], 0.)
        self.assertFalse(values["patches"][0].any())
        with self.assertRaises(ValueError):
            M.features(self.image.astype(float), self.q, self.q, self.q, self.box, self.valid, self.heatmap)
        with self.assertRaises(ValueError):
            M.features(self.image, self.q, self.q, self.q, self.box * np.nan, self.valid, self.heatmap)
        with self.assertRaises(ValueError):
            M.utility_targets(self.q, self.q, self.q, self.valid, [np.arange(9)], self.valid, 0)
        with self.assertRaises(ValueError):
            M.apply_pairs(self.q, np.full((9, 2), np.nan), [True] * 4)
        invalid = self.valid.copy()
        invalid[:8] = False
        target = M.utility_targets(self.q, self.q, self.q, self.valid, [np.arange(9)], invalid, 100.)
        self.assertFalse(target["mask"].any())
        self.assertTrue(np.isfinite(target["target"]).all())


if __name__ == "__main__":
    unittest.main()
