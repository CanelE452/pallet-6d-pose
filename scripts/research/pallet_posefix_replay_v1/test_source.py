"""CPU contract tests; historical input and label artifacts remain read-only."""
from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.research.pallet_posefix_large_error_v1 import train as old_train
from scripts.research.pallet_posefix_replay_v1.source import C, SourceData


def tiny_dataset():
    points = np.tile(np.array([[15., 15.], [25., 15.], [25., 25.],
                              [15., 25.], [16., 16.], [24., 16.],
                              [24., 24.], [16., 24.], [20., 20.]]), (5, 1, 1))
    parts = np.array(['train', 'heldout', 'selection', 'calibration', 'heldout'])
    records = [dict(id=f'synthetic_{i}', partition=str(p), source_kind='synthetic',
                    image='fake.png', image_sha256='test-sha',
                    prepared_shape_hw=[40, 40]) for i, p in enumerate(parts)]
    return SimpleNamespace(
        partitions=parts, train_rows=np.array([0]), indices=np.arange(5),
        source={'records': records}, arrays=dict(
            points=points.copy() * 16, boxes=np.tile([160., 160., 480., 480.], (5, 1)),
            input_shape=np.tile([640, 640], (5, 1)), gain=np.full(5, 16.),
            point_valid=np.ones((5, 9), dtype=bool),
            gt_points=(points.copy() + 1.) * 16, gt_valid=np.ones((5, 9), dtype=bool),
            matched=np.array([True, True, True, True, False])))


class SourceContracts(unittest.TestCase):
    def setUp(self):
        self.data = tiny_dataset()
        self.reader = SourceData(self.data)
        self.image = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
        self.patchers = [
            mock.patch.object(C.E, 'bound', return_value=dict(
                path='fake.png', sha256='test-sha', bytes=4800)),
            mock.patch('scripts.research.pallet_posefix_replay_v1.source.cv2.imread',
                       return_value=self.image)]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_partitions_and_rgb_shape(self):
        self.assertEqual(self.reader.train_rows.tolist(), [0])
        self.assertEqual(self.reader.held_rows.tolist(), [1])
        row = self.reader.item(0)
        self.assertEqual(row['rgb'].shape, (3, 384, 288))
        self.assertEqual(row['rgb'].dtype, np.float32)
        self.assertEqual(row['partition'], 'train')
        self.assertFalse(row['target_valid'][8])
        for forbidden in (-1, 2, 3, 4, 5):
            with self.assertRaises(ValueError):
                self.reader.item(forbidden)
        with self.assertRaises(TypeError):
            self.reader.item(0.)

    def test_gt_does_not_change_inference_input_or_crop(self):
        before = self.reader.item(0)
        self.data.arrays['gt_points'][0] += 1000
        after = self.reader.item(0)
        for name in ('rgb', 'points', 'valid', 'matrix', 'original_points', 'box'):
            np.testing.assert_array_equal(before[name], after[name])
        self.assertFalse(after['target_valid'].any())

    def test_corruption_preserves_gt_and_original(self):
        item = self.reader.item(0)
        snapshot = copy.deepcopy(item)
        noisy = old_train.corrupted(item, np.random.default_rng(73), force=True)
        self.assertTrue(np.any(noisy['points'] != item['points']))
        for name in ('rgb', 'points', 'valid', 'target', 'target_valid',
                     'original_points', 'original_gt', 'matrix', 'box'):
            np.testing.assert_array_equal(item[name], snapshot[name])
        for name in ('target', 'target_valid', 'original_gt', 'matrix', 'box'):
            np.testing.assert_array_equal(noisy[name], item[name])
        np.testing.assert_array_equal(noisy['points'][8], item['points'][8])

    def test_changed_hash_gain_and_shape_fail(self):
        self.data.source['records'][0]['image_sha256'] = 'wrong'
        with self.assertRaisesRegex(AssertionError, 'RGB changed'):
            self.reader.item(0)
        self.data.source['records'][0]['image_sha256'] = 'test-sha'
        self.data.arrays['gain'][0] = .5
        with self.assertRaisesRegex(AssertionError, 'gain mismatch'):
            self.reader.item(0)
        self.data.arrays['gain'][0] = 16.
        self.data.source['records'][0]['prepared_shape_hw'] = [41, 40]
        with self.assertRaisesRegex(AssertionError, 'pad prepared RGB'):
            self.reader.item(0)


class ActualSourceParity(unittest.TestCase):
    def test_first_eight_match_historical_source_adapter(self):
        archive = C.PRIOR_RAW / 'actual_source8.npz'
        if not archive.exists():
            self.skipTest('Historical source-eight numeric artifact unavailable')
        source = SourceData()
        with np.load(archive) as expected:
            for i, row in enumerate(source.train_rows[:8]):
                actual = source.item(int(row))
                for name in ('rgb', 'points', 'valid', 'target', 'target_valid',
                             'original_points', 'matrix'):
                    np.testing.assert_array_equal(actual[name], expected[name][i],
                                                  err_msg=f'{row}: {name}')
                self.assertEqual(actual['partition'], 'train')
                self.assertFalse(actual['target_valid'][8])
        self.assertEqual(source.item(int(source.held_rows[0]))['partition'], 'heldout')


if __name__ == '__main__':
    unittest.main()
