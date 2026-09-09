import unittest
import numpy as np
import torch
from scripts.research.pallet_dht_structured_v2.data_ops import fixed_states, prepare_batch, training_loss
from scripts.research.pallet_dht_structured_v2.model import LayoutVerifier
from scripts.research.pallet_dht_structured_v2.proposals import C4


def fixture():
    g = torch.Generator().manual_seed(123)
    points = torch.rand(2,9,2,generator=g)*200+100
    inputs = dict(p4=torch.rand(2,128,40,40,generator=g), baseline_points=points,
        point_conf=torch.ones(2,9), point_valid=torch.ones(2,9,dtype=torch.bool),
        diagonal=torch.full((2,),800.), raw_to_input_affine=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]]*2),
        input_shape_hw=torch.tensor([[640,640]]*2),line_h=torch.zeros(2,12,4,3),
        peak_logits=torch.zeros(2,12,4), peak_valid=torch.ones(2,12,4,dtype=torch.bool),
        intersections=points[:,:8,None].repeat(1,1,49,1),
        intersection_valid=torch.ones(2,8,49,dtype=torch.bool))
    inputs['line_h'][...,0]=1
    targets=dict(points=points.clone(),valid=torch.ones(2,9,dtype=torch.bool),
        loss_valid=torch.ones(2,9,dtype=torch.bool),matched=torch.ones(2,dtype=torch.bool))
    return inputs,targets


class DataOpsTests(unittest.TestCase):
    def test_target_independence_and_copy(self):
        x,y=fixture();states,variants=fixed_states([1,2],'point_c4')
        a,m,t=prepare_batch(x,y,states,variants)
        altered={k:torch.zeros_like(v) for k,v in y.items()}
        b,n,_=prepare_batch(x,altered,states,variants)
        self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        self.assertTrue(torch.equal(m,n));self.assertIs(t,y)
        self.assertTrue(torch.equal(a['layouts'][:,:,8],x['baseline_points'][:,None,8].expand(-1,64,-1)))
        self.assertTrue(torch.equal(x['baseline_points'],y['points']))

    def test_c4_bank_can_undo_all_three_corruptions(self):
        for quarter in range(3):
            x,y=fixture();variants=np.zeros((2,4));variants[:,0]=(quarter+.5)/3
            for state in ('point_c4','point_and_line_c4'):
                a,m,_=prepare_batch(x,y,[state]*2,variants)
                expected=x['baseline_points'][:,:8][:,C4[quarter+1]]
                self.assertTrue(torch.equal(a['baseline_points'][:,:8],expected))
                correct=(a['layouts'][:,:4,:8]==y['points'][:,None,:8]).all(-1).all(-1)
                self.assertTrue(bool(correct.any(-1).all()))
                self.assertTrue(bool(m[:,:4].all()))

    def test_missing_corner_identity_only(self):
        x,y=fixture();x['point_valid'][0,2]=False
        states,variants=fixed_states([1,2],'local_deformation')
        a,m,_=prepare_batch(x,y,states,variants)
        self.assertEqual(int(m[0].sum()),1)
        self.assertTrue(torch.equal(a['layouts'][0,0],x['baseline_points'][0]))

    def test_loss_finite_and_label_mask(self):
        torch.set_num_threads(1)
        x,y=fixture();y['loss_valid'][0,3]=False
        states,variants=fixed_states([1,2],'point_and_line_c4')
        a,m,t=prepare_batch(x,y,states,variants)
        model=LayoutVerifier();cost,d=model(a,return_diagnostics=True)
        cfg=dict(listwise_temperature=.1,regression_beta=.1,regression_weight=.5,corner_regression_weight=.25)
        loss,stats=training_loss(cost,d,a,m,t,cfg);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(stats['supervised_corners'],15)
        altered={k:v.clone() for k,v in t.items()};altered['points'][0,3]+=10000
        loss2,_=training_loss(cost,d,a,m,altered,cfg)
        self.assertEqual(float(loss),float(loss2))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_fixed_corruptions_stable_under_batch_partition(self):
        states,a=fixed_states([7,3,9],'local_deformation')
        _,b=fixed_states([9],'local_deformation')
        self.assertTrue(np.array_equal(a[2],b[0]))


if __name__=='__main__':unittest.main()
