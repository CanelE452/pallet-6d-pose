import unittest
import numpy as np
import torch
import screen as S


class Contracts(unittest.TestCase):
    def test_lr_bounded_and_end(self):
        for bias in (False, True):
            values = [S.learning_rate(i, bias) for i in range(1000)]
            self.assertTrue(all(np.isfinite(v) and 0 <= v <= .1 for v in values))
            self.assertAlmostEqual(values[-1], .0001)
        self.assertEqual(S.learning_rate(0, False), 0.)
        self.assertEqual(S.learning_rate(0, True), .1)

    def test_tensor_hash_includes_targets(self):
        a = {'img': torch.zeros(1), 'keypoints': torch.zeros(1)}
        b = {'img': torch.zeros(1), 'keypoints': torch.ones(1)}
        self.assertNotEqual(S.tensor_sha(a), S.tensor_sha(b))

    def test_geometry_uses_common_matched(self):
        n = {'a': dict(matched=True, errors_px=[1, 3]), 'b': dict(matched=True, errors_px=[100])}
        m = {'a': dict(matched=True, errors_px=[2, 4]), 'b': dict(matched=False, errors_px=[0])}
        result = S.paired_geometry(n, m)
        self.assertEqual(result['frames'], 1)
        self.assertEqual(result['n']['median_px'], 2)
        self.assertEqual(result['m']['median_px'], 3)

    def test_empty_geometry_is_undefined(self):
        r = S.paired_geometry({}, {})
        self.assertEqual(r['frames'], 0)
        self.assertIsNone(r['n']['median_px'])
        self.assertIsNone(r['m']['gross20'])

    def test_sampler_covers_subset_once(self):
        order = np.random.default_rng(101).permutation(S.BATCH * S.STEPS)
        self.assertEqual(len(set(order)), 4000)
        self.assertEqual(S.MILESTONES, (250, 500, 1000))

    def test_pretrained_initialization_replays_probe(self):
        expected = S.read(S.DOC / 'PROBE.json')['arms']
        for arm in ('n', 'm'):
            model, initial = S.model(arm)
            self.assertEqual(initial, expected[arm]['init'])
            self.assertEqual(model.model[-1].nc, 1)
            self.assertEqual(model.model[-1].nk, 27)
            self.assertTrue(model.model[-1].end2end)
            self.assertTrue(all(not p.requires_grad for p in model.model[-1].dfl.parameters()))
            self.assertTrue(any(mod.training for mod in model.modules()
                                if isinstance(mod, torch.nn.BatchNorm2d)))
            del model

    def test_shape_adapter_preserves_raw_predictions_and_checkpoint(self):
        from evaluate_compat import attach_shape

        def leaves(value):
            if torch.is_tensor(value):
                return [value.detach().clone()]
            if isinstance(value, dict):
                return [x for k in sorted(value) for x in leaves(value[k])]
            if isinstance(value, (list, tuple)):
                return [x for v in value for x in leaves(v)]
            return []

        S.seed_all()
        for arm in ('n', 'm'):
            path = S.RAW / 'runs' / arm / 'step0250.pt'
            original_file = S.sha(path)
            model = torch.load(path, map_location='cpu', weights_only=False)['model'].float().eval()
            image = torch.zeros(1, 3, 128, 128)
            with torch.no_grad():
                before = leaves(model(image))
                receipt = attach_shape(model)
                after = leaves(model(image))
            self.assertFalse(receipt['tensors_changed'])
            self.assertEqual(len(before), len(after))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
            self.assertEqual(original_file, S.sha(path))


if __name__ == '__main__':
    unittest.main()
