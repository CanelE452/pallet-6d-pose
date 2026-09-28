import unittest
import torch
from ultralytics.utils.loss import KeypointLoss
from .loss_signal import normalized_smooth_l1, SupplementLoss
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26


class LossSignalTests(unittest.TestCase):
    def test_ignore_all_channels(self):
        p=torch.randn(2,9,5,dtype=torch.float64,requires_grad=True)
        gt=torch.zeros(2,9,3,dtype=torch.float64);mask=torch.zeros(2,9,dtype=torch.bool);mask[:,0]=True
        value=normalized_smooth_l1(p,gt,mask,torch.ones(2,1,dtype=torch.float64)*100,torch.ones(9,dtype=torch.float64)/9)
        grad=torch.autograd.grad(value,p)[0]
        self.assertEqual(float(grad[:,1:].abs().max()),0)
        self.assertEqual(float(grad[...,2:].abs().max()),0)

    def test_large_error_saturation(self):
        p=torch.zeros(1,9,5,dtype=torch.float64);p[0,0,0]=100;p.requires_grad_()
        gt=torch.zeros(1,9,3,dtype=torch.float64);m=torch.zeros(1,9,dtype=torch.bool);m[0,0]=True
        a=torch.ones(1,1,dtype=torch.float64);s=torch.ones(9,dtype=torch.float64)/9
        loc=KeypointLoss(s)(p,gt,m,a);extra=normalized_smooth_l1(p,gt,m,a,s)
        self.assertEqual(float(torch.autograd.grad(loc,p,retain_graph=True)[0][0,0,0]),0)
        self.assertGreater(float(torch.autograd.grad(extra,p)[0][0,0,0]),0)

    def test_finite_difference_and_descent(self):
        p=torch.zeros(1,9,5,dtype=torch.float64);p[0,0,0]=2;p.requires_grad_()
        gt=torch.zeros(1,9,3,dtype=torch.float64);m=torch.zeros(1,9,dtype=torch.bool);m[0,0]=True
        area=torch.ones(1,1,dtype=torch.float64)*100;sig=torch.ones(9,dtype=torch.float64)/9
        f=lambda x:normalized_smooth_l1(x,gt,m,area,sig)
        grad=torch.autograd.grad(f(p),p)[0];eps=1e-5
        delta=torch.zeros_like(p);delta[0,0,0]=eps
        fd=(f(p+delta)-f(p-delta))/(2*eps)
        self.assertAlmostEqual(float(fd),float(grad[0,0,0]),places=8)
        self.assertLess(float(f(p-.01*grad)),float(f(p)))

    def test_ignored_zero_and_normalization_detached(self):
        p=torch.randn(1,9,5,dtype=torch.float64,requires_grad=True)
        gt=torch.zeros(1,9,3,dtype=torch.float64);m=torch.zeros(1,9,dtype=torch.bool)
        a=torch.ones(1,1,dtype=torch.float64,requires_grad=True);s=torch.ones(9,dtype=torch.float64,requires_grad=True)
        v=normalized_smooth_l1(p,gt,m,a,s);g=torch.autograd.grad(v,[p,a,s],allow_unused=True)
        self.assertEqual(float(g[0].abs().max()),0);self.assertIsNone(g[1]);self.assertIsNone(g[2])

    def test_actual_wrapper_baseline_and_ignore(self):
        from ultralytics.utils.loss import RLELoss
        from ultralytics.nn.modules.block import RealNVP
        c=object.__new__(SupplementLoss)
        c.keypoint_loss=KeypointLoss(torch.ones(9)/9)
        c.rle_loss=RLELoss(use_target_weight=True);c.flow_model=RealNVP();c.target_weights=torch.ones(9)
        p=torch.zeros(1,1,9,5);p[...,0]=3;p.requires_grad_()
        gt=torch.zeros(1,9,3);gt[...,2]=1;gt[:,0,2]=2;gt[:,1,2]=0
        args=(torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),gt,
              torch.zeros(1,1),torch.ones(1,1),torch.tensor([[[0.,0.,10.,10.]]]),p)
        baseline=TrueIgnorePoseLoss26.calculate_keypoints_loss(c,*args)
        c.coefficient=0.;zero=c.calculate_keypoints_loss(*args)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(baseline,zero)))
        c.coefficient=.16500387762358062;extra=c.calculate_keypoints_loss(*args)
        self.assertTrue(torch.equal(baseline[1],extra[1]) and torch.equal(baseline[2],extra[2]))
        g=torch.autograd.grad(sum(extra),p)[0]
        self.assertEqual(float(g[:,:,2:].abs().max()),0)
        self.assertGreater(float(g[0,0,1,2].abs()),0) # visibility0 remains a negative, not ignore


if __name__=='__main__':unittest.main()
