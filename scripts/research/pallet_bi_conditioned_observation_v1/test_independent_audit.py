"""Constructed checks of the independent verifier; no references or updates."""
from copy import deepcopy
import math
import unittest
import numpy as np
from . import independent_audit as A


class IndependentAuditFixtures(unittest.TestCase):
    def test_center_units_and_exact_C2_but_not_rectangular_quarter_turn(self):
        ref=dict(R=np.eye(3).tolist(),t=[0.,0.,2.],order=2)
        pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[.03,.04,2.])
        value=A.raw_metric(pose,ref)
        self.assertAlmostEqual(value['translation_cm'],5.,12);self.assertTrue(value['success'])
        pose['R_physical']=np.diag([-1.,1.,-1.]).tolist()
        self.assertLess(A.raw_metric(pose,ref)['rotation_deg'],1e-10)
        pose['R_physical']=[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]
        self.assertAlmostEqual(A.raw_metric(pose,ref)['rotation_deg'],90.,12)
        self.assertFalse(A.raw_metric(pose,ref)['success'])

    def test_raw_reference_is_not_repaired_and_invalid_output_is_unavailable(self):
        ref=dict(R=[[1.,.002,0.],[0.,1.,0.],[0.,0.,1.]],t=[0.,0.,2.],order=2)
        original=deepcopy(ref);pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0.,0.,2.])
        self.assertGreater(A.raw_metric(pose,ref)['rotation_deg'],0.)
        self.assertEqual(ref,original)
        pose['R_physical']=[[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]]
        value=A.safe_metric(pose,ref);self.assertFalse(value['pose_available']);self.assertFalse(value['success'])
        self.assertIsNone(value['rotation_deg'])

    def test_missing_is_failure_without_zero_imputation(self):
        ref=dict(R=np.eye(3).tolist(),t=[0.,0.,2.],order=2)
        value=A.raw_metric(dict(available=False),ref)
        self.assertFalse(value['success']);self.assertIsNone(value['translation_cm'])
        stats=A.distribution([1.,None,2.,3.],4)
        self.assertEqual(stats['n'],3);self.assertEqual(stats['missing'],1)
        self.assertEqual(stats['mean'],2.);self.assertEqual(stats['sample_std'],1.)
        self.assertAlmostEqual(stats['P90'],2.8)

    def test_cluster_bootstrap_preserves_frame_weighting(self):
        records=[dict(cluster_id='a'),dict(cluster_id='a'),dict(cluster_id='b')]
        result=A.cluster_ci([0.,10.,20.],records,'cluster_id')
        self.assertEqual(result['CI95'],[5.,20.]);self.assertEqual(result['valid_resamples'],10000)
        self.assertEqual(result['empty_resamples'],0)
        result=A.cluster_ci([None,None,None],records,'cluster_id')
        self.assertIsNone(result['CI95']);self.assertEqual(result['empty_resamples'],10000)

    def test_half_turn_angle_and_boundary_translation(self):
        ref=dict(R=np.eye(3).tolist(),t=[0.,0.,2.],order=2)
        pose=dict(available=True,R_physical=np.diag([1.,-1.,-1.]).tolist(),centroid=[0.,0.,2.])
        self.assertAlmostEqual(A.raw_metric(pose,ref)['rotation_deg'],180.,12)
        pose['R_physical']=np.eye(3).tolist();pose['centroid']=[.050001,0.,2.]
        self.assertFalse(A.raw_metric(pose,ref)['success'])

    def test_numeric_MLP_replay_softplus(self):
        feature=dict(x_values=[[0.]*181,[0.]*181],x_validity=[[0.]*181,[0.]*181])
        norm=dict(mean=[0.]*181,standard_deviation=[1.]*181)
        weights={'layers.0.weight':np.zeros((64,364),np.float32),'layers.0.bias':np.zeros(64,np.float32),
            'layers.2.weight':np.zeros((64,64),np.float32),'layers.2.bias':np.zeros(64,np.float32),
            'layers.4.weight':np.zeros((1,64),np.float32),'layers.4.bias':np.array([2.],np.float32)}
        values=A._numpy_scores(feature,norm,'B','COMMON',weights)
        np.testing.assert_allclose(values,[math.log1p(math.exp(2.))]*2,atol=1e-6,rtol=0)
        np.testing.assert_array_equal(values,A._numpy_scores(feature,norm,'I','COMMON',weights))


if __name__=='__main__':unittest.main()
