import unittest
import numpy as np
from a_evaluate import iou,confidence_selected_match,summarize_all
from metrics import measure

class EvaluationTests(unittest.TestCase):
    def test_do_not_select_candidate_by_GT(self):
        wrong=dict(score=.9,box_xyxy=[100,100,110,110]);right=dict(score=.8,box_xyxy=[0,0,10,10])
        top,matched=confidence_selected_match([wrong,right],[0,0,10,10])
        self.assertIs(top,wrong);self.assertFalse(matched)
    def test_missing_detection_not_dropped(self):
        row=measure(np.full((9,2),np.nan),np.zeros((9,2)),np.ones(9,bool),np.arange(9)[None],(480,640))
        row.update(id='miss',fixed_errors=[800.]*8,detected=False,matched=False,outside_raw_corners=0)
        s=summarize_all([row]);self.assertEqual(s['total_frames'],1);self.assertEqual(s['primary_E_sym'],1)
        self.assertEqual(s['detection_coverage'],0);self.assertEqual(s['fixed_pooled_median_px'],800)
    def test_iou_boundary_and_degenerate(self):
        self.assertEqual(iou([0,0,10,10],[0,0,20,10]),.5)
        self.assertEqual(iou([0,0,0,0],[0,0,0,0]),0)
    def test_no_candidate(self):
        self.assertEqual(confidence_selected_match([],[0,0,10,10]),(None,False))
    def test_reflected_corner_basis_does_not_change_body_volume(self):
        from challenge.evaluation_v2.oriented_iou3d import oriented_iou_3d
        reflected=np.diag([1.,-1.,1.]);dims=[.9,.15,1.2];center=[0,0,3]
        self.assertAlmostEqual(oriented_iou_3d(reflected,center,dims,np.eye(3),center,dims),1.,places=7)
    def test_fixed_source_basis_all_C2_width_depth_phases(self):
        from a_source_orientation_correction import source_rotation,angle,SOURCE_BASIS
        from preflight import rotations
        R=rotations(4)[1]
        for k,Qyaw in enumerate(rotations(4)):
            Q=Qyaw@SOURCE_BASIS
            Rcf=R@Q
            old=Rcf@(np.eye(3) if k%2==0 else rotations(4)[1])
            self.assertLess(angle(source_rotation(old),R,2),1e-6)
            self.assertAlmostEqual(np.linalg.det(source_rotation(old)),1.)
    def test_square_pose_does_not_reject_equivalent_WD_tie(self):
        import cv2,importlib
        from a_pose import infer_square
        from preflight import rotations
        from audit_math import derive_permutations
        from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.constants import EDGES
        module=importlib.import_module('run_pose_evaluation');xyz=np.array([1.1,.15,1.1]);x=module.cuboid(*xyz)
        x=np.concatenate([x,np.zeros((1,3))]);perms=derive_permutations(x,rotations(4),np.array(EDGES))
        K=np.array([[600.,0,320],[0,600,240],[0,0,1]]);rvec=np.array([.05,.3,.01]);t=np.array([.1,.2,3.])
        p=cv2.projectPoints(x,rvec,t,K,None)[0].reshape(9,2);R=cv2.Rodrigues(rvec)[0]
        for perm in perms:
            result=infer_square(p[perm],K,xyz);self.assertTrue(result['available'])
            self.assertLess(np.linalg.norm(np.array(result['centroid'])-t),1e-6)
            angle=min(np.arccos(np.clip((np.trace(np.array(result['R_physical']).T@R@q)-1)/2,-1,1)) for q in rotations(4))
            self.assertLess(angle,1e-6)
if __name__=='__main__':unittest.main()
