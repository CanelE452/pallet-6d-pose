"""Physical-basis and independent-observation checks for W/D branch selection."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

_spec=importlib.util.spec_from_file_location("branch_candidate",Path(__file__).with_name("branch_curriculum.py"))
B=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=B;_spec.loader.exec_module(B)


def fixture(dimensions=(1.1,.11,1.3)):
    d=np.array(dimensions,float);K=np.array([[700.,0.,320.],[0.,700.,240.],[0.,0.,1.]])
    R=cv2.Rodrigues(np.array([.35,-.4,.1]))[0];t=np.array([.05,-.1,3.])
    points=B.C.project(B.C.cuboid(*d),R,t,K)
    appearance=dict(states=["VISIBLE_BOUNDARY"]*4+["VISIBLE_INTERIOR"]*2+["SELF_OCCLUDED","VISIBLE_INTERIOR"],
        visibility_score=[.95]*8,local_accuracy_score=[.95]*8,boundary_score=[.95]*4+[.1]*4)
    return d,K,R,t,points,appearance,np.arange(8)==6


def initial(points,K,dimensions,registry):
    start,reason=B._standard_initial(points,B.C.cuboid(*dimensions),K)
    assert start is not None,reason
    R,t=start;Q=B.physical_from_cf(dimensions,registry)
    return dict(available=True,R_cf=R.tolist(),R_physical=(R@Q).tolist(),centroid=t.tolist(),
                cf_extents=dimensions.tolist(),selected_hypothesis="FIXTURE_INITIAL")


class BranchTests(unittest.TestCase):
    def test_branch_selection_can_accept_a_scored_prediction_seed_without_curriculum_update(self):
        d,K,_,_,q,a,hidden=fixture();supplied=initial(q,K,d,d)
        def retain_seed(points,model,camera,R,t,H,appearance,config):
            return dict(R=R.tolist(),t=t.tolist(),accepted=False,used_ids=[],
                        stages=[dict(name="initial",used_ids=[])],reason="fixture_no_local_update")
        # The frozen predecessor accepts branch selection by common-pool cost,
        # independently of whether its local optimizer makes a further update.
        # This locks that inherited behavior rather than silently changing it.
        with patch.object(B,"solve_curriculum",side_effect=retain_seed), patch.object(B,"_huber",side_effect=[10.,0.]):
            result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a)
        self.assertTrue(result["branch_flip"])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["solution"]["accepted"])
        self.assertEqual(result["candidates"][1]["initializer"],"all_prediction_SQPNP_LM")
        self.assertNotIn(6,result["common_visible_ids"])

    def test_recover_registered_dimensions_and_physical_pose_despite_hidden_detector_outlier(self):
        d,K,R,t,q,a,hidden=fixture();q[6]+=[40.,-40.]
        supplied=initial(q,K,d[[2,1,0]],d)
        result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a)
        self.assertTrue(result["branch_flip"])
        np.testing.assert_allclose(result["actual_pose"]["cf_extents"],d)
        np.testing.assert_allclose(result["actual_pose"]["centroid"],t,atol=.001)
        self.assertLess(np.degrees(np.linalg.norm(cv2.Rodrigues(R.T@np.array(result["actual_pose"]["R_physical"]))[0])),.05)
        self.assertNotIn(6,result["common_visible_ids"])
        self.assertEqual(result["observed_points"],q.tolist())
        for c in result["candidates"]:
            self.assertEqual(c["self_hidden_used"],hidden.tolist())
            self.assertNotIn(6,c["curriculum"]["used_ids"])
            self.assertFalse(c["observations_generated_from_pose"])

    def test_retained_correct_family_keeps_its_physical_basis(self):
        d,K,R,t,q,a,hidden=fixture();supplied=initial(q,K,d,d)
        result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a)
        self.assertFalse(result["branch_flip"])
        np.testing.assert_allclose(result["actual_pose"]["R_physical"],R,atol=1e-9)
        self.assertLess(result["original_cost"],result["alternate_cost"])

    def test_visible_support_recovers_when_all_corner_fit_prefers_wrong_family(self):
        d,K,R,t,q,a,_=fixture();hidden=np.isin(np.arange(8),[3,6,7])
        # Detector hallucinations at hidden corners distort the apparent W/D
        # ratio. The unchanged five visible measurements still determine it.
        q[hidden]+=[[.678695093,-9.224484472], [63.543185415,27.579037839], [-40.289613029,-7.949905566]]
        a["states"]=["VISIBLE_BOUNDARY"]*3+["SELF_OCCLUDED"]+["VISIBLE_INTERIOR"]*2+["SELF_OCCLUDED"]*2
        correct=initial(q,K,d,d);wrong=initial(q,K,d[[2,1,0]],d)
        def all_corner_error(pose):
            p=B.C.project(B.C.cuboid(*pose["cf_extents"]),pose["R_cf"],pose["centroid"],K)
            return float(np.mean(np.linalg.norm(p-q,axis=1)**2))
        self.assertLess(all_corner_error(wrong),all_corner_error(correct))
        result=B.solve_branch_curriculum(q,K,d,wrong,hidden,a)
        self.assertTrue(result["branch_flip"])
        np.testing.assert_allclose(result["actual_pose"]["centroid"],t,atol=.001)
        self.assertLess(np.degrees(np.linalg.norm(cv2.Rodrigues(R.T@np.array(result["actual_pose"]["R_physical"]))[0])),.05)

    def test_ambiguity_margin_retains_supplied_family_without_truth_selection(self):
        d,K,_,_,q,a,hidden=fixture();supplied=initial(q,K,d[[2,1,0]],d)
        result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a,dict(selection_margin_px2=100.))
        self.assertFalse(result["branch_flip"])
        self.assertEqual(result["selection_reason"],"original_retained_near_tie")
        np.testing.assert_allclose(result["actual_pose"]["cf_extents"],supplied["cf_extents"])
        Q=np.array(supplied["R_cf"]).T@np.array(supplied["R_physical"])
        np.testing.assert_allclose(np.array(result["actual_pose"]["R_cf"]).T@np.array(result["actual_pose"]["R_physical"]),Q,atol=1e-12)

    def test_square_registry_deduplicates_gauge_hypotheses(self):
        d,K,_,_,q,a,hidden=fixture((1.1,.11,1.1));supplied=initial(q,K,d,d)
        result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a)
        self.assertEqual(len(result["candidates"]),1)
        self.assertFalse(result["branch_flip"])
        self.assertEqual(result["selection_reason"],"square_W_equals_D_deduplicated")

    def test_swapped_basis_projects_same_physical_box_with_explicit_corner_mapping(self):
        d,K,R,t,_,_,_=fixture();swapped=d[[2,1,0]];Q=B.physical_from_cf(swapped,d)
        # R_physical = R_cf @ Q, so physical object points are Q.T @ cf
        # points, or cf_points @ Q in row-vector notation.
        xyz=B.C.cuboid(*d);alternate=B.C.cuboid(*swapped);physical=alternate@Q
        mapping=np.argmin(np.linalg.norm(physical[:,None]-xyz[None],axis=-1),axis=1)
        self.assertEqual(len(set(mapping.tolist())),8)
        np.testing.assert_allclose(physical,xyz[mapping],atol=1e-12)
        np.testing.assert_allclose(B.C.project(alternate,R@Q.T,t,K),B.C.project(xyz,R,t,K)[mapping],atol=1e-9)

    def test_insufficient_independent_visibility_does_not_switch_family(self):
        d,K,_,_,q,a,hidden=fixture();a["states"]=["VISIBLE_BOUNDARY"]*3+["UNKNOWN"]*5
        supplied=initial(q,K,d[[2,1,0]],d)
        result=B.solve_branch_curriculum(q,K,d,supplied,hidden,a)
        self.assertFalse(result["branch_flip"])
        self.assertEqual(result["selection_reason"],"insufficient_shared_RGB_visible_support")


if __name__=="__main__":unittest.main()
