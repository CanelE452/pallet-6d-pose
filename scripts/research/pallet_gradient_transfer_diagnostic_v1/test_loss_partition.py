import unittest
from types import SimpleNamespace
import torch
from ultralytics.utils.loss import KeypointLoss,RLELoss
from ultralytics.nn.modules.block import RealNVP
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26
from scripts.research.pallet_pose_objective_followup_v2.loss_signal import attach_signal_hooks
from .loss_partition import partition,actual_clamp_gate


def fixture(logit=0.,residual=1.,real_mask=(True,False)):
    torch.manual_seed(813)
    flow=RealNVP()
    branches=[];parameters=[]
    for name in ('cv4','one2one_cv4'):
        b=object.__new__(TrueIgnorePoseLoss26)
        b.keypoint_loss=KeypointLoss(torch.ones(9)/9)
        b.rle_loss=RLELoss(use_target_weight=True);b.flow_model=flow;b.target_weights=torch.ones(9)
        b.hyp=SimpleNamespace(pose=12.,kobj=1.,rle=1.)
        p=torch.zeros(2,2,9,5);p[...,:2]=residual;p[...,-2:]=logit;p.requires_grad_()
        parameters.append(('model.23.'+name+'.fixture',p));branches.append((b,p))
    parameters += [('model.23.flow_model.'+n,p) for n,p in flow.named_parameters()]
    model=SimpleNamespace(named_parameters=lambda:iter(parameters))
    criterion=SimpleNamespace(one2many=branches[0][0],one2one=branches[1][0],o2m=.8,o2o=.2)
    fg=torch.tensor([[True,False],[True,True]]) # Role anchor counts1:2, not image ratio1:1.
    gt=torch.zeros(2,9,3);gt[...,2]=2;gt[:,7:,2]=1;gt[1,6,2]=0
    args=(fg,torch.zeros(2,2,dtype=torch.long),gt,torch.arange(2)[:,None],torch.ones(2,1),
          torch.tensor([[[0.,0.,10.,10.]]*2]*2))
    with attach_signal_hooks(criterion) as hook:
        full=torch.zeros(6)
        for (b,p),w in zip(branches,[criterion.o2m,criterion.o2o]):
            loc,vis,rle=b.calculate_keypoints_loss(*args,p)
            full=full+torch.stack([loc*0+1,loc*12,vis,loc*0+.3,loc*0+.2,rle])*2*w
        hook.full_loss=full
        value=partition(model,criterion,hook,torch.tensor(real_mask))
    return value,full/2,parameters


class PartitionTests(unittest.TestCase):
    def compare_gradients(self,value,full,parameters):
        ps=[p for _,p in parameters]
        f=torch.autograd.grad(full.sum(),ps,allow_unused=True,retain_graph=True)
        r=torch.autograd.grad(value['real'],ps,allow_unused=True,retain_graph=True)
        s=torch.autograd.grad(value['source'],ps,allow_unused=True,retain_graph=True)
        for p,a,b,c in zip(ps,f,r,s):
            a=torch.zeros_like(p) if a is None else a;b=torch.zeros_like(p) if b is None else b;c=torch.zeros_like(p) if c is None else c
            self.assertTrue(torch.allclose(a,b+c,atol=3e-6,rtol=3e-5))
        self.assertTrue(torch.allclose(value['combined'],full.sum(),atol=3e-6,rtol=3e-5))
        return f,r,s

    def test_actual_module_value_and_all_parameter_gradients(self):
        v,f,params=fixture();grads,_,_=self.compare_gradients(v,f,params)
        self.assertTrue(any(float(g.abs().sum())>0 for (n,p),g in zip(params,grads) if 'flow_model' in n and g is not None))
        self.assertEqual(v['diagnostics']['branches']['one2many']['real_foreground_anchors'],1)
        self.assertEqual(v['diagnostics']['branches']['one2many']['source_foreground_anchors'],2)

    def test_shared_negative_clamp_and_ignored_zero(self):
        v,f,params=fixture(logit=-4.,residual=0.)
        grads,_,_=self.compare_gradients(v,f,params)
        for branch in v['diagnostics']['branches'].values():
            self.assertEqual(branch['RLE_shared_gate'],0.)
        for (name,p),g in zip(params,grads):
            if 'fixture' in name:self.assertEqual(float(g[...,7:,:].abs().max()),0.)
            elif g is not None:self.assertEqual(float(g.abs().max()),0.)

    def test_torch_clamp_boundary(self):
        for x,expected in [(-1.,0.),(0.,1.),(1.,1.)]:
            self.assertEqual(float(actual_clamp_gate(torch.tensor(x,requires_grad=True))),expected)

    def test_empty_role_preserves_direct_flow_graph(self):
        v,f,params=fixture(real_mask=(True,True));self.compare_gradients(v,f,params)
        self.assertEqual(float(v['source']),0.)


if __name__=='__main__':unittest.main()
