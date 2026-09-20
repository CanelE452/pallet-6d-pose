import copy
import unittest
from unittest.mock import Mock
import numpy as np
import torch
import run as R


class Contracts(unittest.TestCase):
    def setUp(self):
        self.lock = dict(TAU_BOX=.85, keypoint_validity=dict(kp_conf_threshold=.5, min_valid_corners=6),
                         geometry_thresholds=dict(tau_remove=.05, tau_flip=.05))
        self.entry = dict(top1=dict(box_conf=.9, keypoints_xy=np.ones((9, 2)).tolist(), keypoints_conf=[.9]*9))

    def test_raw_failure_never_refines(self):
        refine = Mock()
        for value in (None, dict(self.entry['top1'], box_conf=.84), dict(self.entry['top1'], keypoints_conf=[.1]*9)):
            self.assertIsNone(R.filtered_call(dict(top1=value), self.lock, refine))
        refine.assert_not_called()

    def test_raw_pass_refines_once(self):
        refine = Mock(return_value='refined')
        self.assertEqual(R.filtered_call(self.entry, self.lock, refine), 'refined')
        refine.assert_called_once_with(self.entry)

    def test_second_filter_is_conjunction_and_rejects_nonfinite(self):
        self.assertTrue(R.stage2(dict(s_remove=.05, s_flip=.05), self.lock))
        for bad in (.051, float('nan'), float('inf'), None):
            self.assertFalse(R.stage2(dict(s_remove=bad, s_flip=.01), self.lock))
            self.assertFalse(R.stage2(dict(s_remove=.01, s_flip=bad), self.lock))

    def test_center_not_counted_as_sixth_corner(self):
        self.entry['top1']['keypoints_conf'] = [1]*5+[0]*3+[1]
        self.assertFalse(R.stage1(self.entry, self.lock))

    def test_nonfinite_corner_not_valid(self):
        self.entry['top1']['keypoints_xy'][:3] = [[float('nan'), 1]]*3
        self.assertFalse(R.stage1(self.entry, self.lock))

    def test_unflip_roundtrip(self):
        p = np.arange(18).reshape(9, 2).astype(float); c = np.arange(9)
        perm = [1, 0, 3, 2, 5, 4, 7, 6, 8]
        q, d = R.unflip(p, c, 640, perm)
        a, b = R.unflip(q, d, 640, perm)
        np.testing.assert_array_equal(a, p); np.testing.assert_array_equal(b, c)

    def test_jitter_leaves_target_center_invalid_and_inputs_untouched(self):
        p = torch.ones(2, 9, 2); v = torch.ones(2, 9, dtype=torch.bool); v[:, 0] = False
        b = dict(points=p, point_valid=v, boxes=torch.tensor([[0, 0, 100, 100]]*2), gt_points=p.clone())
        q = R.jitter(b, torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(q['points'][:, 0], p[:, 0]))
        self.assertTrue(torch.equal(q['points'][:, 8], p[:, 8]))
        self.assertTrue(torch.equal(q['gt_points'], p))
        self.assertTrue(torch.equal(b['points'], p))
        self.assertFalse(torch.equal(q['points'][:, 1:8], p[:, 1:8]))


if __name__ == '__main__':
    unittest.main()
