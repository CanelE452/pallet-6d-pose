"""CPU-only integration contracts; no training job, real image, or GPU use."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import integration as I


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_initial_forward_and_stock_weights_exact(self):
        torch.manual_seed(17)
        stock=I.build_model('point_only').eval()
        hough=I.build_model('hough_features').eval()
        a,b=stock.state_dict(),hough.state_dict()
        for key,value in a.items(): self.assertTrue(torch.equal(value,b[key]),key)
        x=torch.randn(1,3,96,128)
        with torch.no_grad():sa=stock(x);hb=hough(x)
        self.assertTrue(torch.equal(sa[0],hb[0]))
        for branch in ['one2many','one2one']:
            for key in ['boxes','scores','kpts']:
                self.assertTrue(torch.equal(sa[1][branch][key],hb[1][branch][key]),(branch,key))
        self.assertEqual(hb[1]['line_logits'].shape,(1,12,90,113))
        self.assertIsNone(hough.model[-1].hough.line_logits)
        clone=copy.deepcopy(hough)
        self.assertTrue(torch.equal(clone.model[0].conv.weight,hough.model[0].conv.weight))

    def test_loss_scale_update_and_gradient_reaches_trainable_neck(self):
        stock=I.build_model('point_only').train()
        hough=I.build_model('hough_features').train()
        for model in [stock,hough]:model.args.epochs=2
        torch.manual_seed(22)
        batch=dict(img=torch.randn(2,3,96,128),batch_idx=torch.tensor([0.,1.]),
            cls=torch.zeros(2,1),bboxes=torch.tensor([[.5,.5,.5,.5],[.5,.5,.5,.5]]),
            keypoints=torch.tensor([[[.25,.25,2],[.75,.25,2],[.75,.75,2],[.25,.75,2],
                [.35,.35,2],[.65,.35,2],[.65,.65,2],[.35,.65,2],[.5,.5,2]]]*2))
        a,items=stock(copy.deepcopy(batch));b,other=hough(copy.deepcopy(batch))
        self.assertTrue(torch.equal(a,b[:6]));self.assertEqual(float(b[6]),0.)
        self.assertTrue(torch.equal(items,other[:6]))
        b.sum().backward()
        self.assertGreater(float(hough.model[0].conv.weight.grad.abs().sum()),0.)
        self.assertGreater(sum(float(m.weight.grad.abs().sum()) for m in hough.model[-1].hough.outputs),0.)
        hough.criterion.update()
        self.assertEqual(hough.criterion.updates,1)
        self.assertAlmostEqual(hough.criterion.stock.o2m,.1)

    def test_disabled_aux_half_reduction_is_finite(self):
        criterion=I.JointCriterion.__new__(I.JointCriterion)
        criterion.weight=0.0
        criterion.stock=lambda predictions,batch:(torch.ones(6),torch.ones(6))
        logits=torch.full((2,12,90,113),-4.,dtype=torch.float16,requires_grad=True)
        loss,items=criterion({'line_logits':logits},{'img':torch.zeros(2,3,8,8)})
        self.assertTrue(torch.isfinite(loss).all());self.assertEqual(float(loss[-1]),0.)
        self.assertTrue(torch.isfinite(items).all())

    def test_target_api_joint_loss_finite(self):
        if not (HERE/'line_targets.py').exists(): self.skipTest('line target implementation pending')
        model=I.build_model('hough_joint').train();model.args.epochs=2
        batch=dict(img=torch.randn(2,3,96,128),batch_idx=torch.tensor([0.,1.]),cls=torch.zeros(2,1),
            bboxes=torch.tensor([[.5,.5,.5,.5]]*2),keypoints=torch.tensor([[[.25,.25,2],[.75,.25,2],
                [.75,.75,2],[.25,.75,2],[.35,.35,2],[.65,.35,2],[.65,.65,2],[.35,.65,2],[.5,.5,2]]]*2))
        loss,_=model(batch)
        self.assertEqual(len(loss),7);self.assertTrue(torch.isfinite(loss).all());self.assertGreater(float(loss[-1]),0.)
        loss.sum().backward()
        self.assertGreater(float(model.model[-1].hough.reduce[0].weight.grad.abs().sum()),0.)


if __name__=='__main__':unittest.main()
