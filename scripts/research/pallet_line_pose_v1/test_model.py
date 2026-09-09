"""CPU analytic, interface, and trainability checks; no dataset/GPU required."""
import inspect
import math
import unittest

import torch
from torch.nn import functional as F

from model import (PalletLinePoseHead, SIDE_EDGES, INCIDENT_ROLES, NULL_INDEX,
                   build_candidates, compute_loss, fuse_corners, make_targets,
                   pixel_to_grid, summarize_distribution)


torch.set_num_threads(1)


def fixture(batch=1, dtype=torch.float32):
    points = torch.tensor([[40., 40.], [120., 40.], [120., 100.], [40., 100.],
                           [60., 20.], [140., 20.], [140., 80.], [60., 80.], [90., 60.]], dtype=dtype)
    return (points[None].repeat(batch, 1, 1),
            torch.tensor([[30., 10., 150., 110.]], dtype=dtype).repeat(batch, 1),
            torch.ones(batch, 9, dtype=torch.bool),
            torch.tensor([[160., 192.]], dtype=dtype).repeat(batch, 1))


def bare_output(points=None, boxes=None, valid=None):
    p, b, v, shape = fixture(dtype=torch.float64)
    p, b, v = (p if points is None else points), (b if boxes is None else boxes), (v if valid is None else valid)
    out = build_candidates(p, b, v)
    out.update(points_raw=p, points=p, logits=torch.zeros(len(p), 8, NULL_INDEX + 1, dtype=p.dtype))
    return out


class GeometryTests(unittest.TestCase):
    def test_fixed_semantic_incidence(self):
        for corner, roles in enumerate(INCIDENT_ROLES):
            self.assertEqual(len(roles), 2)
            for role in roles:
                self.assertIn(corner, SIDE_EDGES[role])
        self.assertEqual(len(set(SIDE_EDGES)), 8)

    def test_cell_center_sampling_and_roundtrip(self):
        feature = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4)
        pixel = torch.tensor([[[[4., 4.], [28., 20.]]]], dtype=torch.float64)
        grid = pixel_to_grid(pixel, 3, 4, 8)
        value = F.grid_sample(feature, grid, align_corners=False)
        torch.testing.assert_close(value.flatten(), torch.tensor([0., 11.], dtype=torch.float64))
        inverse = (grid + 1) * torch.tensor([32., 24.]) / 2
        torch.testing.assert_close(inverse, pixel)

    def test_grid_center_and_pivot_geometry(self):
        out = bare_output()
        self.assertEqual(out["candidate_h"].shape, (1, 8, 221, 3))
        h = out["candidate_h"][:, :, 6 * 17 + 8]
        edge = torch.tensor(SIDE_EDGES)
        ends = out["points_raw"][:, edge]
        residual = (h[..., None, :2] * ends).sum(-1) + h[..., None, 2]
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)
        normal = out["candidate_h"][..., :2]
        torch.testing.assert_close(normal.norm(dim=-1), torch.ones_like(normal[..., 0]))
        # Angle changes at zero offset retain exactly the predicted midpoint.
        zero_r = out["candidate_h"][:, :, 8::17]
        residual = (zero_r[..., :2] * out["midpoint"][:, :, None]).sum(-1) + zero_r[..., 2]
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-12, rtol=0)

    def test_uniform_distribution_preserves_initial_line_and_moment(self):
        out = bare_output()
        summary = summarize_distribution(out["logits"], out["candidate_h"])
        initial = out["candidate_h"][:, :, 6 * 17 + 8]
        torch.testing.assert_close(summary["h_mean"], initial, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(summary["null_probability"], torch.full((1, 8), 1 / 222, dtype=torch.float64))
        self.assertGreaterEqual(torch.linalg.eigvalsh(summary["moment"]).min().item(), -1e-12)
        z = torch.cat([out["points_raw"][:, :8], torch.ones(1, 8, 1, dtype=torch.float64)], -1)
        residual_diff = ((out["candidate_h"] - summary["h_mean"][:, :, None]) * z[:, :, None]).sum(-1)
        direct = (summary["conditional_probability"] * residual_diff.square()).sum(-1)
        quadratic = torch.einsum("bri,brij,brj->br", z, summary["moment"], z)
        torch.testing.assert_close(quadratic, direct, atol=1e-10, rtol=1e-10)

    def test_analytic_orthogonal_parallel_null_and_uncertain(self):
        p, _, valid, _ = fixture(dtype=torch.float64)
        p[:, 0] = p.new_tensor([10., 20.])
        h = p.new_zeros(1, 8, 3)
        h[..., 0] = 1
        h[:, 1] = h.new_tensor([1., 0., -14.])
        h[:, 4] = h.new_tensor([0., 1., -24.])
        moment = p.new_zeros(1, 8, 3, 3)
        null = p.new_zeros(1, 8)
        lv = torch.zeros(1, 8, dtype=torch.bool)
        lv[:, [1, 4]] = True
        args = (p, valid, h, moment, null, lv, p.new_ones(1))
        q, _, _ = fuse_corners(*args, lam=1)
        torch.testing.assert_close(q[0, 0], p.new_tensor([12., 22.]))
        h[:, 4] = h.new_tensor([1., 0., -18.])
        q, _, _ = fuse_corners(*args, lam=1)
        torch.testing.assert_close(q[0, 0], p.new_tensor([14., 20.]))
        null.fill_(1)
        q, _, _ = fuse_corners(*args, lam=1)
        torch.testing.assert_close(q, p, atol=0, rtol=0)
        null.zero_()
        moment[..., 2, 2] = 1e12
        q, _, _ = fuse_corners(*args, lam=1)
        torch.testing.assert_close(q, p, atol=1e-8, rtol=0)

    def test_lambda_zero_missing_centroid_and_invalid_box(self):
        p, b, v, _ = fixture(dtype=torch.float64)
        p[0, 0] = float("nan")
        p[0, 4] = -1
        v[0, 1] = False
        out = bare_output(p, b, v)
        s = summarize_distribution(out["logits"], out["candidate_h"])
        for lam in [0., 1.]:
            q, _, _ = fuse_corners(p, out["point_valid"], s["h_mean"], s["moment"],
                                  s["null_probability"], out["line_valid"], out["anchor_std"], lam)
            torch.testing.assert_close(q[:, [0, 1, 4, 8]], p[:, [0, 1, 4, 8]], atol=0, rtol=0, equal_nan=True)
            self.assertTrue(torch.isfinite(q[:, [2, 3, 5, 6, 7]]).all())
        b[:] = float("nan")
        bad = build_candidates(p, b, v)
        self.assertFalse(bad["line_valid"].any())
        self.assertTrue(torch.isfinite(bad["candidate_h"]).all())

    def test_local_sign_seam_and_target_grid(self):
        out = bare_output()
        gt = out["points_raw"].clone()
        # Exact +2-degree/+0.02D candidate for semantic role1, endpoints3 and0.
        role = 1
        h = out["candidate_h"][0, role, 7 * 17 + 10]
        n = h[:2]
        tangent = torch.stack([n[1], -n[0]])
        mid = out["midpoint"][0, role]
        mid = mid + (-h[2] - (n * mid).sum()) * n
        half = out["length"][0, role] / 2
        a, b = SIDE_EDGES[role]
        gt[0, a], gt[0, b] = mid - half * tangent, mid + half * tangent
        target = make_targets(out, gt, torch.ones(1, 9, dtype=torch.bool))
        self.assertEqual(target["distribution"][0, role].argmax().item(), 7 * 17 + 10)
        torch.testing.assert_close(target["delta_theta_deg"][0, role], torch.tensor(2., dtype=gt.dtype), atol=1e-10, rtol=0)
        # Reversing GT endpoint direction represents the identical undirected line.
        swapped = gt.clone()
        swapped[0, a], swapped[0, b] = gt[0, b].clone(), gt[0, a].clone()
        other = make_targets(out, swapped, torch.ones(1, 9, dtype=torch.bool))
        torch.testing.assert_close(other["distribution"][0, role], target["distribution"][0, role], atol=1e-10, rtol=0)

    def test_out_of_range_null_missing_and_degenerate_targets(self):
        out = bare_output()
        gt = out["points_raw"].clone()
        gt[:, [3, 0], 0] += .10 * out["box_diagonal"][:, None]
        v = torch.ones(1, 9, dtype=torch.bool)
        target = make_targets(out, gt, v)
        self.assertEqual(target["distribution"][0, 1, NULL_INDEX].item(), 1)
        self.assertTrue(target["null_target"][0, 1])
        v[:, 3] = False
        target = make_targets(out, gt, v)
        self.assertEqual(target["distribution"][0, 1].sum().item(), 0)
        v.fill_(True)
        gt[:, 3] = gt[:, 0]
        target = make_targets(out, gt, v)
        self.assertFalse(target["support"][0, 1])
        mask = torch.zeros(1, 8, dtype=torch.bool)
        self.assertFalse(make_targets(out, gt, v, mask)["support"].any())

    def test_isotropic_affine_geometry(self):
        out = bare_output()
        p, b = out["points_raw"], torch.tensor([[30., 10., 150., 110.]], dtype=torch.float64)
        logits = torch.linspace(-1, 1, 222, dtype=torch.float64)[None, None].expand(1, 8, -1)
        def solve(data, pts):
            s = summarize_distribution(logits, data["candidate_h"])
            return fuse_corners(pts, data["point_valid"], s["h_mean"], s["moment"], s["null_probability"],
                                data["line_valid"], data["anchor_std"], .25)[0]
        q = solve(out, p)
        scale, shift = 2., torch.tensor([17., -12.], dtype=p.dtype)
        pp, bb = scale * p + shift, scale * b + shift.repeat(2)
        transformed = build_candidates(pp, bb, out["point_valid"])
        qq = solve(transformed, pp)
        torch.testing.assert_close(qq, scale * q + shift, atol=1e-9, rtol=1e-9)


class HeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PalletLinePoseHead(4, 6, hidden=4)
        self.p, self.b, self.v, self.shape = fixture()
        self.p3, self.p4 = torch.randn(1, 4, 20, 24), torch.randn(1, 6, 10, 12)

    def call(self, **kwargs):
        return self.model(self.p3, self.p4, self.p, self.b, self.v, self.shape, **kwargs)

    def test_no_gt_forward_signature_baseline_and_output_shapes(self):
        self.assertFalse(any("gt" in key for key in inspect.signature(self.model.forward).parameters))
        out = self.call(lam=0)
        torch.testing.assert_close(out["points"], self.p, rtol=0, atol=0)
        self.assertEqual(out["logits"].shape, (1, 8, 222))
        self.assertEqual(out["line_variance"].shape, (1, 8, 2))
        self.assertEqual(out["moment"].shape, (1, 8, 3, 3))
        torch.testing.assert_close(out["points"][:, 8], self.p[:, 8], atol=0, rtol=0)
        torch.testing.assert_close(out["boxes"], self.b, atol=0, rtol=0)
        old = out["points"].clone()
        gt = self.p + 30
        _ = compute_loss(out, gt, self.v)
        torch.testing.assert_close(out["points"], old, atol=0, rtol=0)

    def test_corner_gradient_reaches_spatial_adapters_and_scorer(self):
        out = self.call(lam=1)
        gt = self.p.clone()
        gt[:, :8] += torch.tensor([3., -2.])
        losses = compute_loss(out, gt, self.v)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["corner_loss"].backward()
        for name in ["adapt3.0.weight", "adapt4.0.weight", "line_body.0.weight", "scorer.0.weight", "null_scorer.0.weight"]:
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(grad.abs().sum().item(), 0., name)

    def test_actual_image_and_geometry_only_controls(self):
        with torch.no_grad():
            image_a = self.call()["logits"]
            geometry_a = self.call(geometry_only=True)["logits"]
            self.p3 *= -17
            self.p4 += 10
            image_b = self.call()["logits"]
            geometry_b = self.call(geometry_only=True)["logits"]
        self.assertGreater((image_a - image_b).abs().max().item(), 1e-4)
        torch.testing.assert_close(geometry_a, geometry_b, atol=0, rtol=0)
        out = self.call()
        losses = compute_loss(out, self.p + 1, self.v, corner_weight=0)
        torch.testing.assert_close(losses["loss"], losses["line_loss"], atol=0, rtol=0)

    def test_padding_cannot_supply_image_evidence(self):
        self.shape[:] = torch.tensor([128., 160.])
        with torch.no_grad():
            a = self.call()["logits"]
            self.p3[:, :, 16:] = 1e6
            self.p3[:, :, :, 20:] = -1e6
            self.p4[:, :, 8:] = -1e6
            self.p4[:, :, :, 10:] = 1e6
            b = self.call()["logits"]
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_batch_degenerate_and_empty_loss(self):
        self.p[:, 0] = float("nan")
        self.v[:, 1] = False
        self.p[:, 4] = -1
        self.p[:, 5] = self.p[:, 6]
        out = self.call()
        self.assertFalse(out["line_valid"][0, 2])
        self.assertTrue(torch.isfinite(out["logits"]).all())
        torch.testing.assert_close(out["points"][:, [0, 1, 4, 8]], self.p[:, [0, 1, 4, 8]], atol=0, rtol=0, equal_nan=True)
        loss = compute_loss(out, torch.full_like(self.p, float("nan")), torch.zeros_like(self.v))
        self.assertEqual(loss["loss"].item(), 0.)
        loss["loss"].backward()
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_actual_channels_mixed_rectangles_and_training_step(self):
        model = PalletLinePoseHead(64, 128)
        p, b, v, shape = fixture(batch=2)
        shape[:] = torch.tensor([[384., 640.], [640., 448.]])
        p3, p4 = torch.randn(2, 64, 80, 80), torch.randn(2, 128, 40, 40)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        out = model(p3.half(), p4.half(), p, b, v, shape, lam=.25)
        gt = p + p.new_tensor([2., -1.])
        losses = compute_loss(out, gt, v)
        before = model.scorer[-1].weight.detach().clone()
        losses["loss"].backward()
        optimizer.step()
        self.assertGreater((model.scorer[-1].weight.detach() - before).abs().max().item(), 0)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(losses["corner_count"].item(), 16)
        self.assertEqual(out["points"].dtype, torch.float32)
        torch.testing.assert_close(out["points"][:, 8], p[:, 8], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
