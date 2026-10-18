"""Analytic gradients / constructed contracts only; optimizer creation/step=0."""
from __future__ import annotations
from copy import deepcopy
import math
import unittest
import numpy as np
import torch
from .risk_models import RiskModel,parameters,choose,continuous_frame_loss,order_frame_loss,mean_frame_losses
from . import supervision as S


class RiskModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)

    def test_parameter_counts(self):
        self.assertEqual(parameters('R_CONT_F0'),16386)
        self.assertEqual(parameters('R_ORDER_F0'),16321)
        self.assertEqual(parameters('R_ORDER_F1',288),34753)
        self.assertEqual(parameters('V_ORDER_RGB',288),57977)

    def test_thread_configuration_is_idempotent_before_next_optimizer(self):
        from .train import threads
        threads();threads()
        self.assertEqual(torch.get_num_threads(),1);self.assertEqual(torch.get_num_interop_threads(),1)

    def test_continuous_two_axes_and_minimax(self):
        model=RiskModel('R_CONT_F0');output=model(torch.zeros(3,188))
        self.assertEqual(tuple(output.shape),(3,2));self.assertTrue(bool((output>=0).all()))
        self.assertEqual(choose([[.1,3.],[2.,2.],[2.,2.]],'R_CONT_F0'),1)

    def test_exact_ties_preserve_canonical_first(self):
        self.assertEqual(choose([2.,2.],'R_ORDER_F0'),0)
        self.assertEqual(choose([[1.,1.],[1.,1.]],'R_CONT_F0'),0)
        self.assertIsNone(choose([],'R_ORDER_F0'))

    def test_masked_continuous_loss_hand_value_and_gradient(self):
        output=torch.tensor([[0.,0.],[0.,100.]],requires_grad=True)
        target=torch.tensor([[0.,0.],[1.,0.]])
        mask=torch.tensor([[True,True],[True,False]])
        loss=continuous_frame_loss(output,target,mask)
        self.assertAlmostEqual(float(loss),.25);loss.backward()
        self.assertAlmostEqual(float(output.grad[1,0]),-.5)
        self.assertEqual(float(output.grad[1,1]),0.)

    def test_empty_loss_does_not_manufacture_zero(self):
        self.assertIsNone(order_frame_loss(torch.tensor([]),([],[])))
        self.assertIsNone(continuous_frame_loss(torch.zeros(1,2),torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool)))
        self.assertEqual(mean_frame_losses([None,None]),(None,0))

    def test_Pareto_six_vs_hundred_gradient(self):
        sup=S.from_errors([[6.,2.],[100.,2.]],[[6.,2.],[100.,2.]],[False,False])
        self.assertEqual(sup['pair_groups'][1].tolist(),[[0,1]])
        score=torch.tensor([0.,0.],requires_grad=True);loss=order_frame_loss(score,sup['pair_groups']);loss.backward()
        self.assertAlmostEqual(float(loss),math.log(2),places=6)
        self.assertAlmostEqual(float(score.grad[0]),-.5);self.assertAlmostEqual(float(score.grad[1]),.5)
        self.assertEqual(sup['counts']['all_zero_with_Pareto_supervision'],1)

    def test_success_priority_over_better_rotation_failure(self):
        sup=S.from_errors([[100.,.2],[4.,3.]],[[100.,.2],[4.,3.]],[False,True])
        self.assertEqual(sup['pair_groups'][0].tolist(),[[1,0]])

    def test_two_successes_not_ranked(self):
        sup=S.from_errors([[1.,2.],[3.,1.]],[[1.,2.],[3.,1.]],[True,True])
        self.assertEqual(sup['counts']['both_success_no_order'],1);self.assertEqual(sup['counts']['usable_pairs'],0)

    def test_failure_tradeoffs_not_pseudo_ties(self):
        sup=S.from_errors([[6.,20.],[20.,6.]],[[6.,20.],[20.,6.]],[False,False])
        self.assertEqual(sup['counts']['usable_pairs'],0)
        self.assertEqual(sup['counts']['incomparable_or_equal_failure_pair'],1)
        self.assertEqual(sup['counts']['all_zero_without_order_supervision'],1)

    def test_ambiguous_exact_relation_disagreement_excluded(self):
        sup=S.from_errors([[6.,2.],[6.,3.]],[[6.000000000000001,2.],[6.,3.]],[False,False])
        self.assertEqual(sup['counts']['NUMERICAL_ORDER_AMBIGUOUS'],1);self.assertEqual(sup['counts']['usable_pairs'],0)

    def test_success_threshold_expression_difference_not_relabels(self):
        sup=S.from_errors([[5.,1.],[6.,2.]],[[math.nextafter(5.,math.inf),1.],[6.,2.]],[True,False])
        # Stable failure still Pareto-dominates the second failure: same
        # orientation, original success label retained, no synthetic relabel.
        self.assertTrue(sup['original_success'][0]);self.assertEqual(sup['pair_groups'][0].tolist(),[[0,1]])

    def test_relation_types_equal_weight_not_pair_count_weight(self):
        scores=torch.tensor([0.,1.,0.,0.,0.,0.])
        loss=order_frame_loss(scores,([[0,1]],[[2,3],[2,4],[2,5]]))
        expected=(math.log1p(math.e)+math.log(2))/2
        self.assertAlmostEqual(float(loss),expected,places=6)

    def test_frame_mean_not_action_count_weight(self):
        loss,n=mean_frame_losses([torch.tensor(1.),None,torch.tensor(3.)])
        self.assertEqual(n,2);self.assertEqual(float(loss),2.)

    def test_missing_axes_never_zero_error_labels(self):
        sup=S.from_errors([[math.nan,6.],[100.,math.inf]],[[math.nan,6.],[100.,math.inf]],[False,False])
        self.assertEqual(sup['continuous_validity'].tolist(),[[False,True],[True,False]])
        self.assertEqual(sup['counts']['usable_pairs'],0)
        self.assertEqual(sup['counts']['unavailable_or_nonfinite_pair'],1)

    def test_zero_actions_and_singleton(self):
        for n in(0,1):
            sup=S.from_errors(np.full((n,2),6.),np.full((n,2),6.),np.zeros(n,bool))
            self.assertEqual(sup['counts']['usable_pairs'],0);self.assertEqual(sup['counts']['zero_action_frame'],int(n==0))

    def test_deterministic_uniform_sampling_independent_of_arm(self):
        a=np.arange(600).reshape(-1,2);sup=dict(pair_groups=(a,a.copy()))
        first=S.sampled_pairs(sup,20261018,2,7);second=S.sampled_pairs(sup,20261018,2,7)
        self.assertEqual(S.pair_hash(first),S.pair_hash(second));self.assertEqual(len(first[0]),256)
        self.assertEqual(len(S.sampled_pairs(sup,20261018,2,7,training=False)[0]),300)

    def test_matched_F1_initial_state_and_ZERO_entire_extra(self):
        torch.manual_seed(20261018);zero=RiskModel('R_ORDER_F1_ZERO',288)
        torch.manual_seed(20261018);full=RiskModel('R_ORDER_F1',288)
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(zero.state_dict().values(),full.state_dict().values())))
        f0=torch.randn(4,188);extra=torch.randn(4,288)
        self.assertTrue(torch.equal(zero(f0,extra),full(f0,torch.zeros_like(extra))))
        self.assertTrue(torch.equal(zero(f0,extra),zero(f0,extra*17)))

    def test_visual_NULL_RGB_matched_initialization_and_mask_retained(self):
        torch.manual_seed(7);null=RiskModel('V_ORDER_NULL',288)
        torch.manual_seed(7);rgb=RiskModel('V_ORDER_RGB',288)
        self.assertTrue(all(torch.equal(a,b)for a,b in zip(null.state_dict().values(),rgb.state_dict().values())))
        f0=torch.zeros(2,188);extra=torch.zeros(2,288);patches=torch.randn(2,20,4,16,16);valid=torch.ones(2,20)
        neutral=patches.clone();neutral[:,:,:3]=0
        self.assertTrue(torch.equal(null(f0,extra,patches,valid),rgb(f0,extra,neutral,valid)))

    def test_nonfinite_output_rejected(self):
        with self.assertRaises(FloatingPointError):choose([math.nan],'R_ORDER_F0')

    def test_original_C1_C2_stable_symmetry(self):
        pose=dict(available=True,R_physical=np.diag([-1.,1.,-1.]).tolist(),centroid=[0.,0.,0.])
        target=dict(R_fixed=np.eye(3).tolist(),t=[0.,0.,0.],symmetry_order=1)
        self.assertAlmostEqual(S.stable_error(pose,target)[1],180.)
        target['symmetry_order']=2;self.assertAlmostEqual(S.stable_error(pose,target)[1],0.)

    def test_actual_H1_ID_array_order_alias_invariance(self):
        from scripts.research.pallet_boundary_action_repair_v1 import action_data as O
        def pose(x):return dict(available=True,R_physical=np.eye(3).tolist(),centroid=[x,0.,0.])
        frame=dict(id='toy',split='train',fallback=pose(.06),e0_regenerated=pose(.06),candidates=[
            dict(candidate_id='a',eligible=True,actual_pose=pose(.30),features=[0.]*62),
            dict(candidate_id='b',eligible=True,actual_pose=pose(.06),features=[0.]*62),
            dict(candidate_id='duplicate',eligible=True,actual_pose=pose(.06),features=[0.]*62)])
        target=dict(R_fixed=np.eye(3).tolist(),t=[0.,0.,0.],symmetry_order=2,split='train')
        norm=dict(mean=[0.]*31,scale=[1.]*31,valid_train_counts=[0]*31)
        first,entries=O.inputs(frame,norm,'H1');sup=S.prepare(entries,O.label_frame(frame,target),target)
        changed=deepcopy(frame);changed['candidates'].reverse()
        for i,c in enumerate(changed['candidates']):c['candidate_id']='renamed'+str(i)
        second,other=O.inputs(changed,norm,'H1');again=S.prepare(other,O.label_frame(changed,target),target)
        self.assertTrue(np.array_equal(first,second));self.assertEqual(len(entries),2)
        self.assertEqual(entries[0]['action_role'],'N0');self.assertIn('E0',entries[0]['source_baseline_aliases'])
        self.assertEqual(S.pair_hash(sup['pair_groups']),S.pair_hash(again['pair_groups']))
        self.assertEqual(choose([0.,0.],'R_ORDER_F0'),0)

    def test_unbound_extra_normalizer_rejected_before_feature_application(self):
        import gzip,json,sys
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from .train import attach_extra
        from scripts.research.pallet_boundary_action_repair_v1 import action_data as O
        with TemporaryDirectory()as directory:
            root=Path(directory);features=root/'F1.jsonl.gz';seal=root/'SEAL.json';normalizer=root/'NORM.json'
            with gzip.open(features,'wt')as stream:stream.write(json.dumps(dict(id='toy',actions=[]))+'\n')
            seal.write_text(json.dumps(dict(features_sha256=O.sha(features))))
            normalizer.write_text(json.dumps(dict(features_sha256='incorrect',feature_seal_sha256=O.sha(seal))))
            helper=SimpleNamespace(normalized_extra=lambda *args:(_ for _ in()).throw(AssertionError('applied unbound features')))
            with patch.dict(sys.modules,{'scripts.research.pallet_risk_information_v1.feature_support':helper}):
                with self.assertRaisesRegex(ValueError,'not bound'):
                    attach_extra(dict(train=[],validation=[]),features,seal,normalizer)


if __name__=='__main__':unittest.main()
