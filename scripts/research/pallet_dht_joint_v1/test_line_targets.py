"""CPU geometry and loss-contract checks, without images, training, or GPU."""
import math
import unittest

import torch

from scripts.research.deep_hough_side_v1.dht import SparseDHT
from scripts.research.pallet_dht_joint_v1.line_targets import (
    EDGES, _bilinear_indices, auxiliary_line_loss, build_line_targets,
    canonical_lines, normalized_to_feature, segment_intersects_input,
)


torch.set_num_threads(1)


def lattice(height=40, width=40):
    g = SparseDHT(height, width, 90, .5, True, 28.)
    theta = g.theta_radians.double()
    reach = theta.cos().abs() * width / 2 + theta.sin().abs() * height / 2
    physical = g.rho_values.double().abs()[None] <= reach[:, None] + 1e-6
    return dict(height=height, width=width, theta_bins=90, rho_bins=113,
                rho_step=.5, rho_max=28., theta_values=g.theta_radians,
                rho_values=g.rho_values, valid=physical, vote_valid=g.valid)


def cube():
    return torch.tensor([[[.2, .3, 2], [.65, .3, 2], [.65, .6, 2], [.2, .6, 2],
                          [.35, .2, 2], [.8, .2, 2], [.8, .5, 2], [.35, .5, 2],
                          [.5, .4, 2]]], dtype=torch.float64)


class LineTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.square = lattice()
        cls.rectangle = lattice(32, 40)

    def build(self, points=None, batch=None, geometry=None, size=1):
        points = cube() if points is None else points
        batch = torch.zeros(len(points), dtype=torch.long) if batch is None else batch
        geometry = self.square if geometry is None else geometry
        hw = (geometry['height'] * 16, geometry['width'] * 16)
        return build_line_targets(points, batch, size, hw, geometry, dtype=torch.float64)

    def test_twelve_semantic_roles_and_rectangular_coordinates(self):
        self.assertEqual(len(set(EDGES)), 12)
        self.assertEqual([sum(i in edge for edge in EDGES) for i in range(8)], [3] * 8)
        p = normalized_to_feature(torch.tensor([[0., 0.], [.5, .5], [1., 1.]]), 32, 40)
        torch.testing.assert_close(p, torch.tensor([[-20., -16.], [0., 0.], [20., 16.]]))
        # First cell centre corresponds exactly to normalized (0.5/W, 0.5/H).
        p = normalized_to_feature(torch.tensor([[.5 / 40, .5 / 32]]), 32, 40)
        torch.testing.assert_close(p, torch.tensor([[-19.5, -15.5]]))
        out = self.build(geometry=self.rectangle)
        self.assertEqual(tuple(out['target'].shape), (1, 12, 90, 113))
        self.assertAlmostEqual(float(out['theta'][0, 0]), math.pi / 2)
        self.assertAlmostEqual(float(out['rho'][0, 0]), -.2 * 32)
        self.assertAlmostEqual(float(out['rho'][0, 1]), .15 * 40)

    def test_endpoint_reversal_and_normal_sign(self):
        p = torch.tensor([[[4., -3.], [4., 7.]], [[-5., 2.], [8., 2.]],
                          [[-4., -6.], [7., 3.]]], dtype=torch.float64)
        theta, rho, valid = canonical_lines(p)
        tr, rr, vr = canonical_lines(p.flip(-2))
        torch.testing.assert_close(theta, tr, atol=1e-12, rtol=0)
        torch.testing.assert_close(rho, rr, atol=1e-12, rtol=0)
        self.assertTrue(bool((valid & vr).all()))
        self.assertAlmostEqual(float(theta[0]), 0.)
        self.assertAlmostEqual(float(rho[0]), 4.)
        self.assertAlmostEqual(float(theta[1]), math.pi / 2)
        self.assertAlmostEqual(float(rho[1]), 2.)

    def test_antipodal_seam_keeps_signed_rho_and_mass(self):
        theta = torch.tensor([math.pi - math.pi / 180], dtype=torch.float64)
        rho = torch.tensor([6.], dtype=torch.float64)
        index, weight, valid = _bilinear_indices(theta, rho, self.square, self.square['valid'])
        entries = {int(i): float(v) for i, v in zip(index[0], weight[0]) if v > 1e-8}
        self.assertEqual(set(entries), {89 * 113 + 68, 44})
        self.assertAlmostEqual(entries[89 * 113 + 68], .5)
        self.assertAlmostEqual(entries[44], .5)
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(weight.sum()), 1.)

    def test_axis_line_between_feature_centres_is_not_ignored(self):
        # x=6 is a valid line, although feature columns lie at half integers.
        self.assertFalse(bool(self.square['vote_valid'][0, 68]))
        self.assertTrue(bool(self.square['valid'][0, 68]))
        out = self.build()
        self.assertTrue(bool(out['role_valid'][0, 1]))
        self.assertEqual(float(out['target'][0, 1, 0, 68]), 1.)

    def test_out_of_rho_range_is_ignored_not_clamped(self):
        p = cube()
        p[0, 0, :2] = torch.tensor([.9975, 1.], dtype=torch.float64)
        p[0, 1, :2] = torch.tensor([1., .9975], dtype=torch.float64)
        out = self.build(p)
        self.assertGreater(float(out['rho'][0, 0]), 28.)
        self.assertFalse(bool(out['role_valid'][0, 0]))
        self.assertEqual(float(out['target'][0, 0].sum()), 0.)
        self.assertGreater(int(out['diagnostics']['out_of_lattice_line_instances']), 0)

    def test_multi_instance_union_does_not_choose_one_or_double_targets(self):
        a = cube()
        b = a.clone(); b[..., :2] += .075
        one = self.build(a)['target']
        other = self.build(b)['target']
        combined = self.build(torch.cat([a, b]))['target']
        duplicate = self.build(torch.cat([a, a]))['target']
        torch.testing.assert_close(combined, torch.maximum(one, other))
        torch.testing.assert_close(duplicate, one)
        self.assertGreater(int((combined[0, 0] > 0).sum()), int((one[0, 0] > 0).sum()))

    def test_coincident_different_roles_remain_separate_channels(self):
        p = cube(); p[:, 4:8] = p[:, :4]
        out = self.build(p)
        torch.testing.assert_close(out['target'][0, 0], out['target'][0, 4])
        self.assertTrue(bool(out['role_valid'][0, 0] & out['role_valid'][0, 4]))
        self.assertFalse(bool(out['role_valid'][0, 8]))  # Zero-length connector ignored.

    def test_unknown_object_role_ignores_whole_global_role(self):
        known = cube(); partial = cube(); partial[0, 0, 2] = 0
        out = self.build(torch.cat([known, partial]))
        expected = torch.ones(12, dtype=torch.bool); expected[[0, 3, 8]] = False
        torch.testing.assert_close(out['role_valid'][0], expected)
        self.assertEqual(float(out['target'][0, [0, 3, 8]].sum()), 0.)
        self.assertEqual(int(out['diagnostics']['ignored_image_roles']), 3)
        # v=1 is known amodal supervision, not a claim of physical visibility.
        amodal = cube(); amodal[..., 2] = 1
        torch.testing.assert_close(self.build(amodal)['target'], self.build()['target'])

    def test_segment_support_is_finite_segment_not_infinite_extension(self):
        p = torch.tensor([[[-30., 0.], [30., 0.]], [[22., -1.], [24., 1.]],
                          [[20., 20.], [25., 25.]], [[-21., -20.], [21., -20.]],
                          [[1., 1.], [1., 1.]], [[float('nan'), 0.], [1., 1.]]])
        got = segment_intersects_input(p, 40, 40)
        torch.testing.assert_close(got, torch.tensor([True, False, False, True, False, False]))
        outside = cube(); outside[:, 0, :2] = torch.tensor([1.1, .2]); outside[:, 1, :2] = torch.tensor([1.2, .2])
        out = self.build(outside)
        self.assertFalse(bool(out['role_valid'][0, 0]))
        self.assertGreater(int(out['diagnostics']['outside_segment_line_instances']), 0)

    def test_invalid_vote_bins_are_never_background_or_positive(self):
        out = self.build()
        invalid = ~self.square['valid']
        self.assertFalse(bool(out['mask'][..., invalid].any()))
        self.assertEqual(float(out['target'][..., invalid].sum()), 0.)

    def test_empty_negative_and_all_unknown_are_finite(self):
        logits = torch.zeros(1, 12, 90, 113, requires_grad=True)
        empty = torch.empty(0, 9, 3)
        loss, diagnostic = auxiliary_line_loss(logits, empty, torch.empty(0), (640, 640), self.square)
        self.assertAlmostEqual(float(loss), math.log(2), places=6)
        self.assertEqual(int(diagnostic['background_only_image_roles']), 12)
        loss.backward()
        self.assertGreater(float(logits.grad.sum()), 0.)
        unknown = cube(); unknown[..., 2] = 0
        logits2 = torch.randn(1, 12, 90, 113, requires_grad=True)
        loss2, diagnostic2 = auxiliary_line_loss(logits2, unknown, torch.tensor([0]), (640, 640), self.square)
        self.assertEqual(float(loss2), 0.)
        loss2.backward()
        self.assertEqual(float(logits2.grad.abs().sum()), 0.)
        self.assertEqual(int(diagnostic2['ignored_image_roles']), 12)

    def test_balanced_bce_and_batch_normalization(self):
        logits = torch.zeros(1, 12, 90, 113, dtype=torch.float64, requires_grad=True)
        loss, _ = auxiliary_line_loss(logits, cube(), torch.tensor([0]), (640, 640), self.square)
        self.assertAlmostEqual(float(loss), math.log(2), places=12)
        data = self.build()
        loss.backward()
        target = data['target']; mask = data['mask']
        positive = (target > 0) & mask; background = (target == 0) & mask
        # Dense background has total gradient .25 per valid role, not O(T*R).
        per_role_background = (logits.grad * background).sum((-1, -2))
        torch.testing.assert_close(per_role_background, torch.full_like(per_role_background, .25 / 12))
        self.assertLess(float(logits.grad[(target == 1) & positive].max()), 0.)
        batch = torch.cat([cube(), cube()]); batch[1, :, 2] = 0
        loss2, _ = auxiliary_line_loss(torch.zeros(2, 12, 90, 113, dtype=torch.float64),
                                       batch, torch.tensor([0, 1]), (640, 640), self.square)
        self.assertAlmostEqual(float(loss2), math.log(2) / 2, places=12)

    def test_no_label_gradient_center_dependency_or_input_mutation(self):
        p = cube().requires_grad_(); before = p.detach().clone()
        logits = torch.randn(1, 12, 90, 113, dtype=torch.float64, requires_grad=True)
        loss, _ = auxiliary_line_loss(logits, p, torch.tensor([0]), (640, 640), self.square)
        loss.backward()
        self.assertIsNone(p.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.)
        torch.testing.assert_close(p.detach(), before)
        changed = before.clone(); changed[:, 8] = torch.tensor([1000., -1000., 0.])
        torch.testing.assert_close(self.build(changed)['target'], self.build(before)['target'])

    def test_malformed_contract_fails(self):
        with self.assertRaises(ValueError):
            self.build(batch=torch.tensor([1]))
        with self.assertRaises(ValueError):
            self.build(batch=torch.tensor([.5]))
        with self.assertRaises(ValueError):
            build_line_targets(cube(), torch.tensor([0]), 1, (640, 512), self.square)
        bad = dict(self.square); bad.pop('valid')
        with self.assertRaises(ValueError):
            self.build(geometry=bad)


if __name__ == '__main__':
    unittest.main()
