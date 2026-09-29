import copy
import random
import unittest

import numpy as np
import torch
from ultralytics.utils.instance import Instances

from .augmentation import PairedTransform, digest, random_plan, rng_state, restore_rng, state_seed
from .trainer import make_loader, loader_seed, protected_state_keys, assert_protected_state, batch_trace


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


class FakeBase:
    """A deterministic clipping fixture that consumes all three RNG streams."""
    def __call__(self, labels):
        values = [random.random(), np.random.random(), float(torch.rand(()))]
        keypoints = labels['instances'].keypoints.copy()
        keypoints[..., 0] += .1
        outside = (keypoints[..., 0] > 1.) | (keypoints[..., 1] > 1.)
        keypoints[..., 2][outside] = 0.
        keypoints[..., :2] = np.clip(keypoints[..., :2], 0, 1)
        return dict(img=torch.full((3, 64, 64), int(sum(values)*70), dtype=torch.uint8),
                    keypoints=torch.from_numpy(keypoints), bboxes=torch.tensor([[.5, .5, .8, .8]]),
                    cls=torch.zeros(1, 1), batch_idx=torch.zeros(1), im_file=labels['im_file'])


class SeedDataset(torch.utils.data.Dataset):
    """Python-only IPC avoids sandbox tensor-sharing sockets in this unit test."""
    def __len__(self): return 1024
    def __getitem__(self, index):
        return index, random.random(), float(np.random.random()), float(torch.rand(()))
    @staticmethod
    def collate_fn(batch): return batch


def fixture(target='RAW', source=False):
    coords = np.array([[.95, .3], [.99, .7], [.3, .3], [.6, .3], [.3, .6],
                       [.6, .6], [.45, .2], [.45, .7], [.45, .45]], np.float32)
    points = np.c_[coords, np.full(9, 2., np.float32)][None]
    points[0, 1, 2] = 1
    ref = points.copy(); ref[0, 0, 0] = .75; ref[0, 2, 0] += .03
    paired = {'real.png': {'RAW': points, 'REF': ref}}
    name = 'syn__example.png' if source else 'real.png'
    instance = Instances(np.array([[.5, .5, .8, .8]], np.float32),
                         keypoints=paired['real.png'][target].copy(), bbox_format='xywh', normalized=True)
    return dict(im_file=name, instances=instance), paired


class AugmentationTests(unittest.TestCase):
    def test_shared_post_affine_support_and_original_ignore(self):
        output = {}
        for arm in ('RAW', 'REF'):
            labels, paired = fixture(arm)
            seed(11)
            output[arm] = PairedTransform(FakeBase(), paired, arm, 'CLEAR')(labels)
        raw, ref = output['RAW'], output['REF']
        self.assertTrue(torch.equal(raw['img'], ref['img']))
        self.assertTrue(torch.equal(raw['keypoints'][..., 2], ref['keypoints'][..., 2]))
        self.assertEqual(raw['keypoints'][0, 0, 2], 1.)  # RAW outside, REF inside: both ignored.
        self.assertEqual(raw['keypoints'][0, 1, 2], 1.)  # Original ignore preserved after clipping.
        self.assertEqual(raw['transfer_info']['geometric_demotions'], 1)
        self.assertFalse(torch.equal(raw['keypoints'][..., :2], ref['keypoints'][..., :2]))
        self.assertEqual(raw['transfer_info']['plan'], ref['transfer_info']['plan'])

    def test_all_four_arms_plan_and_targets_exact_except_coordinates(self):
        applied = 0
        for trial in range(50):
            outputs = {}
            for target in ('RAW', 'REF'):
                for condition in ('CLEAR', 'OCC'):
                    labels, paired = fixture(target)
                    seed(trial)
                    outputs[target, condition] = PairedTransform(FakeBase(), paired, target, condition)(labels)
            for target in ('RAW', 'REF'):
                clear, occ = outputs[target, 'CLEAR'], outputs[target, 'OCC']
                for key in ('keypoints', 'bboxes', 'cls', 'batch_idx'):
                    self.assertTrue(torch.equal(clear[key], occ[key]))
                self.assertEqual(clear['transfer_info']['before_image'], occ['transfer_info']['before_image'])
                self.assertEqual(clear['transfer_info']['plan'], occ['transfer_info']['plan'])
                self.assertFalse(clear['transfer_info']['applied'])
            raw, ref = outputs['RAW', 'OCC'], outputs['REF', 'OCC']
            self.assertTrue(torch.equal(raw['img'], ref['img']))
            if ref['transfer_info']['applied']:
                applied += 1
                plan = ref['transfer_info']['plan']
                self.assertGreaterEqual(len(plan['covered']), 1)
                self.assertGreaterEqual(plan['remaining'], 2)
                clear = outputs['REF', 'CLEAR']['img'].clone()
                l, t, w, h = plan['rectangle']
                clear[:, t:t+h, l:l+w] = ref['img'][:, t:t+h, l:l+w]
                self.assertTrue(torch.equal(clear, ref['img']))
        self.assertGreater(applied, 0)

    def test_global_rng_is_single_transform_and_source_is_unchanged(self):
        for source in (False, True):
            labels, paired = fixture(source=source)
            seed(78); before = rng_state()
            stock = FakeBase()(copy.deepcopy(labels)); expected_rng = rng_state()
            for condition in ('CLEAR', 'OCC'):
                restore_rng(before)
                result = PairedTransform(FakeBase(), paired, 'RAW', condition)(copy.deepcopy(labels))
                self.assertEqual(state_seed(rng_state(), 'test'), state_seed(expected_rng, 'test'))
                if source:
                    for key in ('img', 'keypoints', 'bboxes'):
                        self.assertTrue(torch.equal(result[key], stock[key]))
                    self.assertFalse(result['transfer_info']['applied'])

    def test_plan_does_not_promote_ignore_or_center(self):
        points = np.full((9, 2), 32.)
        for trial in range(50):
            self.assertFalse(random_plan(points, np.zeros(9, bool), [0, 0, 64, 64], (64, 64), trial)['applied'])
            mask = np.zeros(9, bool); mask[[0, 1, 8]] = True
            self.assertFalse(random_plan(points, mask, [0, 0, 64, 64], (64, 64), trial)['applied'])

    def test_actual_collation_and_json_safe_trace(self):
        import json
        from ultralytics.data.dataset import YOLODataset
        samples = []
        for source in (False, True):
            labels, paired = fixture(source=source)
            seed(21)
            samples.append(PairedTransform(FakeBase(), paired, 'RAW', 'OCC', {'real.png': 'recording_fixture'})(labels))
        batch = YOLODataset.collate_fn(samples)
        trace = batch_trace(batch, batch.pop('transfer_info'))
        json.dumps(trace, allow_nan=False)
        self.assertEqual(trace['roles']['REAL']['images'], 1)
        self.assertEqual(trace['roles']['SOURCE']['images'], 1)
        self.assertEqual(trace['recording_exposure']['recording_fixture']['images'], 1)

    def test_loader_seed_is_supplied_before_iteration(self):
        from torch.utils.data import TensorDataset
        def stream(s):
            loader = make_loader(TensorDataset(torch.arange(128)), workers=0, training_seed=s)
            return [batch[0].tolist() for batch in loader]
        self.assertEqual(loader_seed(43)-loader_seed(42), 1)
        self.assertEqual(stream(42), stream(42))
        self.assertNotEqual(stream(42), stream(43))

    def test_worker2_order_and_rng_streams_change_reproducibly(self):
        def stream(value):
            loader = make_loader(SeedDataset(), workers=2, training_seed=value)
            rows = []
            try:
                for index, batch in enumerate(loader):
                    rows.append(batch)
                    if index == 7: break
            finally:
                loader.iterator._shutdown_workers()
            return rows
        a, b, repeat = stream(42), stream(43), stream(43)
        self.assertEqual(b, repeat)
        for left, right in zip(a, b):
            for field in range(4):
                self.assertNotEqual([row[field] for row in left], [row[field] for row in right])

    def test_protection_includes_flow_buffers(self):
        model = torch.nn.Module()
        model.pose = torch.nn.Linear(2, 2)
        model.fixed = torch.nn.Linear(2, 2)
        for p in model.fixed.parameters(): p.requires_grad_(False)
        model.register_buffer('flow_loc', torch.zeros(2))
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        protected = protected_state_keys(model)
        self.assertEqual(set(protected), {'flow_loc', 'fixed.weight', 'fixed.bias'})
        self.assertEqual(assert_protected_state(model.state_dict(), initial, protected), 3)
        model.flow_loc.add_(1)
        with self.assertRaises(AssertionError): assert_protected_state(model.state_dict(), initial, protected)


if __name__ == '__main__':
    unittest.main()
