import unittest
import numpy as np
import torch
from .cycle_pose import robust_score,choice
from .signal_diagnostic import batch_subset

class MechanismTests(unittest.TestCase):
    def test_small_residual_exact_original_score(self):
        for score in (3.,105.):
            self.assertAlmostEqual(robust_score(dict(residual9_px=list(range(9)),selector_score=score)),score)
    def test_only_rmse_term_changes(self):
        a=dict(residual9_px=[1.]*8+[100.],selector_score=100.)
        b=dict(a,selector_score=177.)
        self.assertAlmostEqual(robust_score(b)-robust_score(a),77.)
        self.assertLess(robust_score(a),100.)
    def test_frozen_scale_not_GT(self):
        self.assertEqual(robust_score.__defaults__,(12.,))
    def test_permutation_invariance_and_tie(self):
        cue=dict(residual9_px=[1.]*9,selector_score=3.)
        self.assertEqual(choice({'W':cue,'D':cue}),choice({'D':cue,'W':cue}))
        self.assertEqual(choice({'W':cue,'D':cue}),'D')
        self.assertIsNone(choice({}))
    def test_batch_subset_instance_reindex(self):
        b=dict(img=torch.arange(12).reshape(3,4),batch_idx=torch.tensor([0.,2.,2.]),
               keypoints=torch.arange(18).reshape(3,3,2),bboxes=torch.ones(3,4),cls=torch.ones(3,1),im_file=['a','b','c'])
        s=batch_subset(b,[2,1]);self.assertEqual(s['batch_idx'].tolist(),[0.,0.]);self.assertEqual(s['im_file'],['c','b'])
        self.assertTrue(torch.equal(s['keypoints'],b['keypoints'][1:]));self.assertEqual(len(s['img']),2)

if __name__=='__main__':unittest.main()
