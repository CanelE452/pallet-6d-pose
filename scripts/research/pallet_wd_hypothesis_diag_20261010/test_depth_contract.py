"""Meaningful boundary/coverage tests; no dataset, model or PnP calls."""
import unittest
from unittest.mock import patch
import numpy as np
from . import depth
from . import stage3


class DepthContract(unittest.TestCase):
    def test_five_percent_inclusive_and_missing_fails(self):
        primary=dict(frames=231,complete_three_seed_frames=231,
                     seed_mean_absolute_relative_error={'median':.05},unavailable_primary_ids=[])
        self.assertEqual(stage3.accuracy_gate(primary,0)['status'],'PASS')
        primary['seed_mean_absolute_relative_error']['median']=.050000001
        self.assertEqual(stage3.accuracy_gate(primary,0)['status'],'FAIL_STOP_S4')
        primary['seed_mean_absolute_relative_error']['median']=.01
        primary['complete_three_seed_frames']=230;primary['unavailable_primary_ids']=['missing']
        self.assertEqual(stage3.accuracy_gate(primary,0)['status'],'FAIL_STOP_S4')

    def test_positive_finite_roi_median_and_invalid_corner(self):
        image=np.full((40,40),2.,np.float32);image[20,20]=np.nan;image[20,21]=0.
        q=np.tile([20.,20.],(9,1));q[:4]=[[5,5],[35,5],[35,35],[5,35]]
        result=depth.front_depth_median(image,q)
        self.assertTrue(result['available']);self.assertEqual(result['median_m'],2.)
        q[2]=[-1,-1]
        self.assertFalse(depth.front_depth_median(image,q)['available'])

    def test_strict_gap_trigger_and_coordinates_preserved(self):
        names=('long-face-front','short-face-front')
        candidates={n:{'actual_pose':dict(available=True,R_cf=np.eye(3).tolist(),
             centroid=[0,0,z],cf_extents=[1,.1,1.3])} for n,z in zip(names,[1.,2.])}
        row=dict(qFinal=np.zeros((9,2)).tolist(),selection=dict(hyp=names[0],
             candidates=candidates,actual_pose=candidates[names[0]]['actual_pose'],formal_score_gap_px=1.))
        measurement=dict(available=True,median_m=2.)
        class FakePose:
            @staticmethod
            def cuboid(*args):return np.zeros((8,3))
        with patch.object(stage3.C,'pose_api',return_value=FakePose()):
            self.assertEqual(stage3._choose(row,measurement,1.)['hyp'],names[0])
            row['selection']['formal_score_gap_px']=.999
            selected=stage3._choose(row,measurement,1.)
            self.assertEqual(selected['hyp'],names[1])
            self.assertEqual(selected['qFinal'],row['qFinal'])
            self.assertEqual(row['selection']['hyp'],names[0])


if __name__=='__main__':unittest.main()
