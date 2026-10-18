from __future__ import annotations
import math,unittest
import numpy as np
import torch
from .independent_supervision import independent_errors,independent_supervision,analytic_order,analytic_continuous,numerical_bound
from . import supervision as S,risk_models as M
from .independent_checks import independent_actions


class IndependentSupervisionTests(unittest.TestCase):
    def compare(self,raw,success,stable=None):
        a=np.asarray(raw,float).reshape(-1,2);b=a.copy()if stable is None else np.asarray(stable,float).reshape(-1,2)
        x=independent_supervision(a,b,success);y=S.from_errors(a,b,success)
        self.assertEqual(x['counts'],y['counts']);self.assertEqual(x['numerical_ambiguous_pairs'],y['numerical_ambiguous_pairs'])
        self.assertTrue(all(np.array_equal(u,v)for u,v in zip(x['pair_groups'],y['pair_groups'])));return x
    def test_success_precedes_failed_tradeoff(self):
        r=self.compare([[100.,.2],[4.,3.]],[False,True]);self.assertEqual(r['pair_groups'][0].tolist(),[[1,0]])
    def test_all_failure_Pareto_gradient_direction(self):
        r=self.compare([[6.,2.],[30.,20.]],[False,False]);loss,grad=analytic_order([0.,0.],r['pair_groups'])
        self.assertEqual(grad.tolist(),[-.5,.5]);self.assertAlmostEqual(loss,math.log(2.));self.assertEqual(r['counts']['all_zero_with_Pareto_supervision'],1)
    def test_failure_tradeoff_has_no_licensed_order(self):
        r=self.compare([[6.,20.],[20.,6.]],[False,False]);self.assertEqual(r['counts']['usable_pairs'],0)
    def test_two_successes_no_order(self):self.assertEqual(self.compare([[1.,1.],[2.,2.]],[True,True])['counts']['usable_pairs'],0)
    def test_numerical_reversal_excluded(self):
        r=self.compare([[6.,2.],[7.,3.]],[False,False],[[8.,2.],[7.,3.]])
        self.assertEqual(r['counts']['NUMERICAL_ORDER_AMBIGUOUS'],1);self.assertEqual(r['counts']['usable_pairs'],0)
    def test_zero_actions_and_singleton_keep_no_loss(self):
        for raw,success in[([],[]),([[6.,2.]],[False])]:
            r=self.compare(raw,success);self.assertEqual(r['counts']['usable_pairs'],0)
            self.assertIsNone(analytic_order(np.zeros(len(raw)),r['pair_groups'])[0])
    def test_missing_axes_and_padding_gradient(self):
        r=self.compare([[math.nan,6.],[100.,math.nan],[math.nan,math.nan]],[False]*3)
        output=np.asarray([[20.,1.],[2.,30.],[99.,100.]],float)
        expected,g=analytic_continuous(output,r['continuous_target'],r['continuous_validity'])
        tensor=torch.tensor(output,dtype=torch.float64,requires_grad=True)
        loss=M.continuous_frame_loss(tensor,torch.tensor(r['continuous_target']),torch.tensor(r['continuous_validity']))
        loss.backward();self.assertAlmostEqual(float(loss),expected);self.assertTrue(np.allclose(tensor.grad.numpy(),g,atol=1e-15,rtol=0));self.assertTrue(np.array_equal(g[-1],[0.,0.]))
    def test_equal_type_and_frame_weighting(self):
        groups=(np.asarray([[0,1],[0,2]]),np.asarray([[1,2]]));scores=[0.,1.,2.]
        wanted,grad=analytic_order(scores,groups);x=torch.tensor(scores,dtype=torch.float64,requires_grad=True);loss=M.order_frame_loss(x,groups,chunk_size=1);loss.backward()
        self.assertLessEqual(abs(wanted-float(loss)),numerical_bound(wanted));self.assertTrue(np.allclose(x.grad.numpy(),grad,rtol=0,atol=1e-15))
        average,count=M.mean_frame_losses([loss,None,loss*3]);self.assertEqual(count,2);self.assertAlmostEqual(float(average),float(loss)*2)
    def test_6cm_100cm_continuous_and_order_distinguish(self):
        r=self.compare([[6.,2.],[100.,2.]],[False,False]);self.assertGreater(r['continuous_target'][1,0],r['continuous_target'][0,0]);self.assertEqual(r['pair_groups'][1].tolist(),[[0,1]])
    def test_original_C1_C2_without_90degree_collapse(self):
        pose=dict(available=True,R_physical=np.diag([-1.,1.,-1.]).tolist(),centroid=[0.,0.,2.])
        target=dict(R_fixed=np.eye(3).tolist(),t=[0.,0.,2.],symmetry_order=1)
        self.assertEqual(independent_errors(pose,target)[1],180.);target['symmetry_order']=2
        self.assertEqual(independent_errors(pose,target,True),[0.,0.])
        angle=np.pi/2;pose['R_physical']=[[math.cos(angle),0.,math.sin(angle)],[0.,1.,0.],[-math.sin(angle),0.,math.cos(angle)]]
        self.assertAlmostEqual(independent_errors(pose,target,True)[1],90.)
    def test_ID_rename_and_array_permutation_preserve_physical_pairs(self):
        def c(name,z):return dict(candidate_id=name,actual_pose=dict(available=True,R_physical=np.eye(3).tolist(),centroid=[0.,0.,z]),eligible=True,features=[0.]*62)
        frame=dict(candidates=[c('b',3.),c('a',2.)],fallback={'available':False},object_scale_m=1.)
        first=independent_actions(frame);frame['candidates'].reverse()
        for i,e in enumerate(frame['candidates']):e['candidate_id']='renamed'+str(i)
        second=independent_actions(frame)
        self.assertEqual([e['actual_pose']for e in first],[e['actual_pose']for e in second])
        self.assertEqual(self.compare([[6.,2.],[30.,20.]],[False,False])['pair_groups'][1].tolist(),[[0,1]])
    def test_negative_or_unavailable_axis_not_zero_error(self):
        r=self.compare([[-1.,math.nan]],[False]);self.assertFalse(r['continuous_validity'].any());self.assertIsNone(analytic_continuous([[0.,0.]],r['continuous_target'],r['continuous_validity'])[0])


if __name__=='__main__':unittest.main()
