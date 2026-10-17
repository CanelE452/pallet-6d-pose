"""Small contract tests for noninterchangeable pose and paired-statistic rules."""
import unittest

import numpy as np

from .baseline_audit import metric
from .evaluation import _cluster_ci, paired


class EvaluationContractTests(unittest.TestCase):
    def test_rectangular_quarter_turn_is_failure_but_half_turn_is_equivalent(self):
        ref = {'R': np.eye(3).tolist(), 't': [0., 0., 1.], 'order': 2}
        quarter = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        quarter_pose = dict(available=True, centroid=[0., 0., 1.], R_physical=quarter.tolist())
        half_pose = {**quarter_pose, 'R_physical': (quarter @ quarter).tolist()}
        self.assertAlmostEqual(metric(quarter_pose, ref)['rotation_deg'], 90.)
        self.assertFalse(metric(quarter_pose, ref)['success'])
        self.assertTrue(metric(half_pose, ref)['success'])

    def test_no_pose_is_binary_failure_without_continuous_imputation(self):
        m = metric({'available': False}, {'R': np.eye(3).tolist(), 't': [0., 0., 1.], 'order': 2})
        self.assertFalse(m['success'])
        self.assertIsNone(m['translation_cm'])
        self.assertIsNone(m['rotation_deg'])

    def test_thresholds_apply_to_the_same_pose(self):
        ref = {'R': np.eye(3).tolist(), 't': [0., 0., 1.], 'order': 2}
        pose = dict(available=True, centroid=[.05, 0., 1.], R_physical=np.eye(3).tolist())
        self.assertTrue(metric(pose, ref)['success'])
        pose['centroid'][0] = .050001
        self.assertFalse(metric(pose, ref)['success'])

    def test_cluster_ci_keeps_whole_cluster_frame_weighting(self):
        rows = [{'cluster_id': 'A'}, {'cluster_id': 'A'}, {'cluster_id': 'B'}]
        ci = _cluster_ci([1., 1., -1.], rows, 'cluster_id')
        self.assertEqual(ci['cluster_sizes'], {'A': 2, 'B': 1})
        self.assertEqual(ci['CI95'], [-1., 1.])
        self.assertEqual(ci['valid_resamples'], 10000)


if __name__ == '__main__':
    unittest.main()
