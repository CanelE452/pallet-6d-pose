"""Three observed corners may refine a known initial pose, with explicit prior."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as BASE
from scripts.research.pallet_visibility_curriculum_20261013_v1.test_curriculum import fixture, appearance

_spec=importlib.util.spec_from_file_location("seed3_candidate",Path(__file__).with_name("curriculum_seed3.py"))
M=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=M;_spec.loader.exec_module(M)


def evidence(ids):
    states=["OCCLUSION_SUSPECTED"]*8
    for index in ids:states[index]="VISIBLE_BOUNDARY"
    return appearance(states)


class Seed3Tests(unittest.TestCase):
    def test_three_accurate_observations_improve_pose_without_corrupted_fourth(self):
        xyz,K,_,_,ir,it,truth=fixture();points=truth.copy();points[3]+=[150,-100]
        a=evidence([0,1,4]);hidden=np.zeros(8,bool)
        baseline=BASE.solve_curriculum(points,xyz,K,ir,it,hidden,a)
        self.assertFalse(baseline["accepted"])
        for prior in (1.,4.):
            result=M.solve_curriculum(points,xyz,K,ir,it,hidden,a,dict(prior_strength_px=prior))
            self.assertTrue(result["accepted"])
            self.assertTrue(result["has_accepted_three_point_stage"])
            self.assertEqual(result["used_ids"],[0,1,4])
            self.assertLess(np.linalg.norm(np.array(result["projected"])-truth,axis=1).mean(),.2)
            self.assertEqual(result["observed_points"],points.tolist())
            g=result["stages"][1]["geometry"]
            self.assertEqual(g["augmented_prior_geometry"]["observation_jacobian_shape"],[6,6])
            self.assertEqual(g["augmented_prior_geometry"]["shape"],[12,6])
            self.assertTrue(g["P3P_ambiguity_requires_initial_prior"])
            self.assertTrue(g["prior_is_not_an_independent_observation"])

    def test_fewer_than_three_is_deferred_even_with_stronger_prior(self):
        xyz,K,_,_,ir,it,points=fixture()
        for prior in (1.,4.):
            result=M.solve_curriculum(points,xyz,K,ir,it,np.zeros(8,bool),evidence([0,4]),dict(prior_strength_px=prior))
            self.assertFalse(result["accepted"])
            np.testing.assert_array_equal(result["R"],ir)
            np.testing.assert_array_equal(result["t"],it)
            self.assertEqual(result["used_ids"],[])

    def test_thin_front_triangle_requires_sufficient_prior_condition(self):
        xyz,K,_,_,ir,it,truth=fixture();a=evidence([0,1,2])
        weak=M.solve_curriculum(truth,xyz,K,ir,it,np.zeros(8,bool),a,dict(prior_strength_px=1.))
        strong=M.solve_curriculum(truth,xyz,K,ir,it,np.zeros(8,bool),a,dict(prior_strength_px=4.))
        self.assertFalse(weak["accepted"])
        self.assertTrue(strong["accepted"])
        self.assertGreater(strong["stages"][1]["geometry"]["jacobian_condition"],15.)
        self.assertLess(strong["stages"][1]["geometry"]["augmented_prior_geometry"]["jacobian_condition"],15.)
        self.assertLess(np.linalg.norm(np.array(strong["projected"])-truth,axis=1).mean(),3.)

    def test_prior_does_not_promote_reprojected_or_self_hidden_point_to_third_observation(self):
        xyz,K,_,_,ir,it,points=fixture();a=evidence([0,1,4])
        a["independent"]=[True,True,True,True,False,True,True,True]
        result=M.solve_curriculum(points,xyz,K,ir,it,np.zeros(8,bool),a,dict(prior_strength_px=4.))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["visible_ids"],[0,1])
        self.assertFalse(result["reprojected_points_reused_as_observations"])

    def test_three_point_local_fit_cannot_jump_far_from_initial_pose(self):
        xyz,K,_,_,ir,it,points=fixture();points[[0,1,4]]+=[180,120]
        result=M.solve_curriculum(points,xyz,K,ir,it,np.zeros(8,bool),evidence([0,1,4]),dict(prior_strength_px=4.))
        self.assertFalse(result["accepted"])
        self.assertEqual(result["fallback_reason"],"exceeds_initial_pose_motion_limit")
        np.testing.assert_array_equal(result["R"],ir)


if __name__=="__main__":unittest.main()
