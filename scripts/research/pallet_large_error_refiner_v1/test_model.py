import unittest
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.research.pallet_large_error_refiner_v1.model import WideRefiner, corrupt, objective, VERTICAL


def batch(n=4):
    return dict(p3=torch.randn(n, 64, 80, 80), p4=torch.randn(n, 128, 40, 40),
                points=torch.full((n, 9, 2), 100.), boxes=torch.tensor([[50., 50., 250., 250.]]).repeat(n, 1),
                point_valid=torch.ones(n, 9, dtype=torch.bool), input_shape=torch.tensor([[544, 640]]).repeat(n, 1),
                context=torch.zeros(n, 5), gt_points=torch.full((n, 9, 2), 130.),
                gt_valid=torch.ones(n, 9, dtype=torch.bool))


class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1)

    def test_identity_centroid_and_invalid(self):
        b = batch(); b['point_valid'][0, 1] = False
        m = WideRefiner()
        self.assertTrue(torch.equal(m(b), b['points']))
        with torch.no_grad(): m.output[-1].bias.fill_(10)
        q = m(b)
        self.assertTrue(torch.equal(q[:, 8], b['points'][:, 8]))
        self.assertTrue(torch.equal(q[0, 1], b['points'][0, 1]))
        size = (q[:, :8] - b['points'][:, :8]).norm(dim=-1)
        self.assertGreater(float(size.max()), 32)
        self.assertLessEqual(float(size.max()), .20 * 200 * 2 ** .5 + 1e-4)

    def test_no_target_input(self):
        b = batch(); m = WideRefiner()
        with torch.no_grad(): m.output[-1].weight.normal_()
        q = m(b)
        b['gt_points'].fill_(99999); b['gt_valid'].fill_(False)
        self.assertTrue(torch.equal(q, m(b)))

    def test_padded_features_ignored(self):
        b = batch(); m = WideRefiner()
        b['boxes'][:, 1] = 400; b['boxes'][:, 3] = 640
        with torch.no_grad(): m.output[-1].weight.normal_(std=.01)
        before = m(b)
        b['p3'][:, :, 68:] = 9999
        b['p4'][:, :, 34:] = 9999
        self.assertTrue(torch.equal(before, m(b)))

    def test_masked_gt_and_finite_gradients(self):
        b = batch(); m = WideRefiner()
        b['gt_valid'][:, 1:] = False
        q = m(b); a = objective(q, b)
        b['gt_points'][:, 1:] = float('nan')
        self.assertTrue(torch.equal(a, objective(q, b)))
        a.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(float(m.output[-1].weight.grad.abs().sum()), 0)

    def test_corruption_distribution_targets_and_pairs(self):
        b = batch(1)
        for k in ('p3', 'p4'): b[k] = b[k].expand(1000, -1, -1, -1)
        for k in set(b) - {'p3', 'p4'}: b[k] = b[k].repeat(1000, *([1] * (b[k].ndim - 1)))
        before = b['points'].clone(); gt = b['gt_points'].clone()
        a = corrupt(b, torch.Generator().manual_seed(4))
        z = corrupt(b, torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(a['points'], z['points']))
        self.assertTrue(torch.equal(b['points'], before)); self.assertTrue(torch.equal(a['gt_points'], gt))
        changed = (a['points'] != before).any(-1)
        self.assertFalse(changed[:, 8].any())
        counts = changed.sum(-1)
        for n, low, high in ((0, .45, .55), (1, .20, .30), (2, .20, .30)):
            self.assertTrue(low < float((counts == n).float().mean()) < high)
        for row in torch.where(counts == 2)[0]:
            pair = tuple(torch.where(changed[row])[0].tolist())
            self.assertIn(pair, VERTICAL)
            delta = a['points'][row] - before[row]
            self.assertTrue(torch.equal(delta[pair[0]], delta[pair[1]]))


if __name__ == '__main__': unittest.main()
