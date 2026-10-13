"""Analytic support/rollback invariants, without real data or annotations."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.research.pallet_visibility_curriculum_20261013_v1 import curriculum as M


def setup_scene():
    a,b,c=.6,.075,.45
    xyz=np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                  [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])
    R=cv2.Rodrigues(np.array([.2,-.35,.02]))[0]
    t=np.array([0.,0.,3.])
    K=np.array([[700.,0.,320.],[0.,700.,240.],[0.,0.,1.]])
    q=M.project(xyz,R,t,K)
    return xyz,R,t,K,q


def appearance(easy=(0,1,2,4),unknown=(),hidden=(),seed_accuracy=.9):
    states=['VISIBLE_BOUNDARY' if k in easy else 'VISIBLE_INTERIOR' for k in range(8)]
    for k in unknown:states[k]='UNKNOWN'
    for k in hidden:states[k]='SELF_OCCLUDED'
    return dict(states=states,visibility_score=[.9]*8,boundary_score=[1.]*8,
        local_accuracy_score=[seed_accuracy if k in easy else .9 for k in range(8)])


def fake_fit(poses):
    calls=[]
    def fit(xyz,q,K,R0,t0,R,t,ids,w,cfg):
        index=len(calls);calls.append(list(ids))
        if index>=len(poses):raise ValueError('Explicit unavailable next independent fit in this analytic case')
        target_R,target_t=poses[index]
        return target_R.copy(),target_t.copy(),dict(nfev=0,optimizer_success=True,robust_weights=w.tolist())
    return fit,calls


class ProgressiveGuardTests(unittest.TestCase):
    def test_frozen_weight_huber_sum_and_fixed_initial_prior(self):
        xyz,R,t,K,q=setup_scene();q=q.copy();q[0,0]+=3;q[1,1]+=12
        weights=np.array([.2,.7,1.,1.,1.,1.,1.,1.])
        delta=np.array([.04,-.01,.02]);candidate_R=cv2.Rodrigues(delta)[0]@R
        candidate_t=t+np.array([.01,-.02,.03]);cfg=M.CurriculumConfig()
        measured=M._aggregate_objective(xyz,q,K,candidate_R,candidate_t,R,t,[0,1],weights,cfg)
        residual=np.linalg.norm(M.project(xyz,candidate_R,candidate_t,K)[:2]-q[:2],axis=1)
        data=.2*(residual[0]**2 if residual[0]<=8 else 16*residual[0]-64)
        data+=.7*(residual[1]**2 if residual[1]<=8 else 16*residual[1]-64)
        scale=max(np.linalg.norm(t),np.linalg.norm(xyz.max(0)-xyz.min(0)))
        prior=np.dot(delta,delta)/np.deg2rad(8)**2+np.dot(candidate_t-t,candidate_t-t)/(scale*.08)**2
        self.assertAlmostEqual(measured['objective'],(data+prior)/.9,places=11)
        self.assertAlmostEqual(measured['frozen_weight_sum'],.9,places=12)

    def test_progressive_seed_can_advance_but_final_full_pool_restores_initial(self):
        xyz,R,t,K,q=setup_scene();seed=[0,1,2,4]
        target_t=t+np.array([.01,0,0]);q[seed]=M.project(xyz,R,target_t,K)[seed]
        heldout=[3,5,6,7];q[heldout]=M.project(xyz,R,t-np.array([.05,0,0]),K)[heldout]
        app=appearance(seed_accuracy=.3)
        fit,_=fake_fit([(R,target_t)])
        with patch.object(M,'_fit',fit):
            result=M.solve_curriculum(q,xyz,K,R,t,np.zeros(8,bool),app,dict(progressive_active_support=True,enable_unknown=False))
        self.assertTrue(result['stages'][1]['accepted'])
        self.assertFalse(result['accepted']);self.assertEqual(result['used_ids'],[])
        np.testing.assert_array_equal(result['R'],R);np.testing.assert_array_equal(result['t'],t)
        self.assertEqual(result['final_guard']['all_visible_ids'],list(range(8)))
        self.assertEqual(result['final_guard']['fallback_source'],'supplied_initial_pose')
        self.assertGreater(result['final_guard']['checkpoints'][0]['full_visible']['objective'],result['final_guard']['initial']['objective'])
        fit,_=fake_fit([(R,target_t)])
        with patch.object(M,'_fit',fit):
            legacy=M.solve_curriculum(q,xyz,K,R,t,np.zeros(8,bool),app,dict(enable_unknown=False))
        self.assertFalse(legacy['stages'][1]['accepted'])
        self.assertEqual(legacy['stages'][1]['reason'],'independent_visible_evidence_worsened')

    def test_late_unknown_rollback_restores_used_and_unknown_ids_together(self):
        xyz,R,t,K,q=setup_scene();q[7,0]+=1
        initial_t=t+np.array([1e-5,0,0]);late_t=t+np.array([2e-5,0,0])
        hidden=np.array([False,False,False,True,False,True,True,False])
        app=appearance(unknown=(7,),hidden=(3,5,6))
        fit,_=fake_fit([(R,t),(R,late_t)])
        with patch.object(M,'_fit',fit):
            result=M.solve_curriculum(q,xyz,K,R,initial_t,hidden,app,dict(progressive_active_support=True))
        unknown=[s for s in result['stages'] if s['name'].startswith('add_consistent_unknown_')]
        self.assertEqual(len(unknown),1);self.assertTrue(unknown[0]['accepted'])
        self.assertEqual(unknown[0]['validation_policy'],'legacy_unknown_visible_guards')
        self.assertEqual(result['used_ids'],[0,1,2,4]);self.assertEqual(result['unknown_added_ids'],[])
        np.testing.assert_array_equal(result['t'],t)
        self.assertEqual(result['final_guard']['fallback_source'],'earlier_passing_checkpoint')
        self.assertTrue(unknown[0]['intermediate_accept_discarded_by_final_guard'])
        self.assertTrue(all(not hidden[k] for s in result['stages'] for k in s['used_ids']))

    def test_latest_passing_checkpoint_is_retained_even_if_not_lowest_objective(self):
        xyz,R,t,K,q=setup_scene();q[3,0]+=8;q[[5,6,7],0]-=30
        initial_t=t+np.array([.03,0,0]);later_t=t+np.array([.001,0,0])
        fit,_=fake_fit([(R,t),(R,later_t)])
        with patch.object(M,'_fit',fit):
            result=M.solve_curriculum(q,xyz,K,R,initial_t,np.zeros(8,bool),appearance(),
                dict(progressive_active_support=True,enable_unknown=False))
        cp=result['final_guard']['checkpoints'];self.assertEqual(len(cp),2)
        self.assertTrue(all(c['passes_initial_all_visible_guard'] for c in cp))
        self.assertLess(cp[0]['full_visible']['objective'],cp[1]['full_visible']['objective'])
        self.assertEqual(result['final_guard']['retained_checkpoint_index'],cp[1]['stage_index'])
        np.testing.assert_array_equal(result['t'],later_t)
        self.assertEqual(result['final_omitted_visible_ids'],[5,6,7])
        self.assertEqual(result['final_guard']['rejected_additions_still_omitted_ids'],[5,6,7])

    def test_real_fit_under_four_defers_without_creating_coordinates(self):
        xyz,R,t,K,q=setup_scene();hidden=np.ones(8,bool);hidden[[0,1,4]]=False
        app=appearance(easy=(0,1,4),hidden=tuple(np.flatnonzero(hidden)))
        before=q.copy()
        result=M.solve_curriculum(q,xyz,K,R,t,hidden,app,dict(progressive_active_support=True))
        self.assertFalse(result['accepted']);self.assertEqual(result['used_ids'],[])
        np.testing.assert_array_equal(q,before)
        np.testing.assert_array_equal(result['R'],R);np.testing.assert_array_equal(result['t'],t)
        self.assertEqual(result['final_guard']['all_visible_ids'],[0,1,4])
        self.assertEqual(result['intermediate_accepted_count'],0)


if __name__=='__main__':
    unittest.main()
