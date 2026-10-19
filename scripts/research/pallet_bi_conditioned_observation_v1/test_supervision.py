"""Analytic fixtures for synthetic correspondence and label semantics."""
from copy import deepcopy
import math
import unittest

import numpy as np

from .supervision import align_reference, targets_for_frame


def fixture(order=2, matrix=None):
    xyz = np.array([[-1,-.2,-2],[1,-.2,-2],[1,.2,-2],[-1,.2,-2],
                    [-1,-.2,2],[1,-.2,2],[1,.2,2],[-1,.2,2]],float)
    K = np.array([[100.,0,100],[0,100,100],[0,0,1.]])
    R = np.eye(3) if matrix is None else matrix
    def projection(sym):
        cam = xyz@(R@sym).T+[0,0,10]
        h = cam@K.T
        return h[:,:2]/h[:,2:]
    q = projection(np.eye(3))
    upstream = dict(id='fixture',split='train',native_xyz=xyz.tolist(),K=K.tolist(),
        initial_pose=dict(R_cf=np.eye(3).tolist(),R_physical=np.eye(3).tolist()),
        initial_points=np.vstack([q,[100.,100.]]).tolist())
    gt = dict(id='fixture',split='train',K=K.tolist(),R_fixed=R.tolist(),t=[0,0,10],symmetry_order=order)
    features = dict(id='fixture',split='train',corner_rows=[dict(native_corner=k,
        candidate_ids=[f'k{k}:keep'],applied_xy=[q[k].tolist()],partial_replaces_full_likelihood=False)
        for k in range(8)])
    return upstream,features,gt,projection


class SupervisionTest(unittest.TestCase):
    def test_one_whole_object_symmetry_not_per_corner(self):
        up, features, gt, project = fixture()
        phase1 = project(np.diag([-1.,1.,-1.]))
        up['initial_points'][7] = phase1[7].tolist()
        features['corner_rows'][7]['applied_xy'] = [phase1[7].tolist()]
        row = targets_for_frame(up,features,gt)
        self.assertEqual(row['alignment']['chosen_symmetry_index'],0)
        self.assertGreater(row['corner_rows'][7]['target_log1p_error'][0],0.)
        self.assertEqual(len(row['alignment']['symmetry_costs']),2)
        up['initial_points'][:8] = phase1.tolist()
        alignment = align_reference(up,gt)
        self.assertEqual(alignment['chosen_symmetry_index'],1)
        np.testing.assert_array_equal(alignment['aligned_reference_xy8'],phase1)

    def test_renderer_matrix_is_preserved_without_orthogonalization(self):
        matrix = np.diag([1.+2e-7,1.,1.])
        up,features,gt,_ = fixture(order=1,matrix=matrix)
        row = targets_for_frame(up,features,gt)
        self.assertEqual(row['alignment']['original_renderer_R'],matrix.tolist())
        self.assertFalse(row['alignment']['raw_renderer_R_modified'])
        self.assertTrue(all(c['target_log1p_error']==[0.] for c in row['corner_rows']))

    def test_missing_reference_is_invalid_and_never_zero_error(self):
        up,features,gt,_ = fixture()
        gt['R_fixed'] = None
        row = targets_for_frame(up,features,gt)
        for corner in row['corner_rows']:
            self.assertEqual(corner['target_log1p_error'],[None])
            self.assertEqual(corner['valid_target'],[False])
        self.assertTrue(row['frame_retained_in_denominator'])

    def test_applied_coordinates_and_fixed_partial_semantics(self):
        up,features,gt,_ = fixture(order=1)
        feature_before = deepcopy(features)
        features['corner_rows'][0]['raw_intersection_xy'] = [[1000.,1000.]]
        features['corner_rows'][1]['partial_replaces_full_likelihood'] = True
        row = targets_for_frame(up,features,gt)
        self.assertEqual(row['corner_rows'][0]['target_log1p_error'],[0.])
        self.assertEqual(row['corner_rows'][1]['target_log1p_error'],[None])
        self.assertEqual(row['corner_rows'][1]['target_reasons'],['PARTIAL_PATH_FIXED'])
        self.assertEqual(features['corner_rows'][0]['applied_xy'],feature_before['corner_rows'][0]['applied_xy'])

    def test_extreme_finite_error_is_not_clipped(self):
        up,features,gt,_ = fixture(order=1)
        original = features['corner_rows'][0]['applied_xy'][0]
        features['corner_rows'][0]['applied_xy'] = [[original[0]+1e9,original[1]]]
        row = targets_for_frame(up,features,gt)
        self.assertEqual(row['corner_rows'][0]['target_log1p_error'],[math.log1p(1e9/2)])

    def test_unregistered_quarter_turn_symmetry_is_rejected(self):
        up,features,gt,_ = fixture()
        gt['symmetry_order'] = 4
        with self.assertRaisesRegex(ValueError,'C1/C2'):
            targets_for_frame(up,features,gt)


if __name__ == '__main__':
    unittest.main()
