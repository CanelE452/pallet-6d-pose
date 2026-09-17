import unittest
import numpy as np
import cv2
from pose import infer,metric,cuboid,rotations

class Pose(unittest.TestCase):
    def test_exact_known_square_and_group(self):
        xyz=np.array([1.1,.15,1.1]);K=np.array([[600.,0,320],[0,600,240],[0,0,1]]);rv=np.array([.05,.3,.01]);R=cv2.Rodrigues(rv)[0];t=np.array([.1,.2,3.]);x=cuboid(*xyz)
        p=cv2.projectPoints(x,rv,t,K,None)[0].reshape(8,2);p=np.r_[p,[[320.,240.]]];pred=infer(p,K,xyz)
        self.assertTrue(pred['available']);g=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=4);r=metric(('x',pred,g))
        self.assertLess(r['translation_cm'],1e-4);self.assertLess(r['rotation_deg'],1e-4);self.assertGreater(r['IoU3D'],.99999);self.assertLess(r['ADDsym_m'],1e-6)
    def test_quarter_turn_C2_rejected_C4_allowed(self):
        xyz=np.array([1.,.1,1.]);R=rotations(4)[1];p=dict(available=True,R_cf=R.tolist(),R_physical=R.tolist(),centroid=[0,0,3],cf_extents=xyz.tolist())
        gt=dict(R=np.eye(3),t=np.array([0,0,3]),xyz=xyz,body_R=np.eye(3),body_xyz=xyz,order=2)
        self.assertAlmostEqual(metric(('x',p,gt))['rotation_deg'],90);gt['order']=4;self.assertAlmostEqual(metric(('x',p,gt))['rotation_deg'],0)
    def test_missing_prediction_coverage(self):self.assertFalse(metric(('x',dict(available=False),{}))['available'])
if __name__=='__main__':unittest.main()
