"""CPU helper tests only. Passing does not open the actual-model training gate."""
import unittest
import numpy as np
import torch
from contracts import ARM_WEIGHTS, backward_coefficients, exposure_plan, pck_counts, stream_seed


class ContractsTest(unittest.TestCase):
    def test_exposures(self):
        for arm, expected in zip(ARM_WEIGHTS, [(2400, 0), (2400, 0), (2400, 7200), (9600, 0)]):
            result = exposure_plan(arm)
            self.assertEqual((result['real'], result['synthetic']), expected)
            self.assertEqual(result['updates'], 300)

    def test_coefficients(self):
        self.assertEqual(backward_coefficients('T8_FULL'), {'target_base': 4.})
        self.assertEqual(backward_coefficients('T8_QUARTER'), {'target_base': 1.})
        for arm in ('REPLAY', 'T32_COMPUTE'):
            self.assertEqual(set(backward_coefficients(arm).values()), {1.})

    def test_toy_gradient_identities(self):
        w = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
        centers = [torch.tensor(t, dtype=w.dtype) for t in ((1., 2.), (-1., 0.), (0., -2.), (.5, .5))]
        losses = [(w - center).square().mean() for center in centers]  # toy C8
        def grad(loss):
            return torch.autograd.grad(loss, w, retain_graph=True)[0]
        full, quarter = grad(4 * losses[0]), grad(losses[0])
        replay = grad(sum(losses))
        source = sum(grad(v) for v in losses[1:])
        torch.testing.assert_close(quarter, full / 4)
        torch.testing.assert_close(replay - quarter, source)
        torch.testing.assert_close(grad(losses[0] + 0 * sum(losses[1:])), quarter)
        w.grad = None
        for loss in losses:
            loss.backward(retain_graph=True)
        torch.testing.assert_close(w.grad, replay)

    def test_fixed_denominator_and_boundary(self):
        gt = np.zeros((4, 2)); mask = [True, True, True, False]
        pred = np.array([[10., 0.], [10.01, 0.], [np.nan, 0.], [0., 0.]])
        counts = pck_counts(gt, mask, pred, True)
        self.assertEqual(counts, dict(denominator=3, numerators={'5.0': 0, '10.0': 1, '20.0': 2}))
        for xy, matched in ((None, False), (pred, False), (None, True)):
            value = pck_counts(gt, mask, xy, matched)
            self.assertEqual(value['denominator'], 3)
            self.assertEqual(set(value['numerators'].values()), {0})

    def test_invalid_roles_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            pck_counts(np.zeros((9, 2)), np.ones(9, bool), np.zeros((8, 2)), True)

    def test_seed_streams(self):
        self.assertEqual(stream_seed(1, 0, 'target_base', 0), stream_seed(1, 0, 'target_base', 0))
        self.assertNotEqual(stream_seed(1, 0, 'target_base', 0), stream_seed(1, 0, 'source_1', 0))


if __name__ == '__main__':
    unittest.main()
