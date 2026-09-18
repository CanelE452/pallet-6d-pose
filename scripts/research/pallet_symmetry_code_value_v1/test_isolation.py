import unittest
from types import SimpleNamespace
import numpy as np
import torch
import cv_env as E
from code_adapter import *
from inference import preservation
from point_inference import replace_selected
from eval_math import measure

class Isolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.paper=TrainingData('A');cls.mixed=TrainingData('B')
    def test_full_capacity_and_init(self):
        for seed in [1,2,3]:
            torch.manual_seed(seed);a=model();torch.manual_seed(seed);b=model()
            self.assertEqual(sum(p.numel() for p in a.parameters()),20307)
            self.assertEqual(E.state_sha(a.state_dict()),E.state_sha(b.state_dict()))
            for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
    def test_actual_input_step0_target_and_gradient_parity(self):
        for track,data in [('A',self.paper),('B',self.mixed)]:
            rows=data.order(1)[0];blind,aware=E.ARMS[track]
            a=data.batch(rows,blind,device='cpu');b=data.batch(rows,aware,device='cpu')
            for k in a:
                if k!='context':torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,equal_nan=True,msg=k)
            self.assertTrue(torch.equal(a['context'][:,:5],b['context'][:,:5]))
            self.assertTrue(torch.equal(a['context'][:,5:],torch.full((16,3),1/3)))
            self.assertTrue(torch.equal(b['context'][:,5:].sum(-1),torch.ones(16)))
            torch.manual_seed(1);m=model();torch.manual_seed(1);n=model()
            oa=forward(m,a);ob=forward(n,b);self.assertTrue(torch.equal(oa['logits'],ob['logits']))
            phase=lambda o,t:local_phase(o['points_raw'],o['point_valid'],t['gt_points'],t['gt_valid'],t['permutations'],t['group_valid'],o['box_diagonal'])
            pa=phase(oa,a);pb=phase(ob,b)
            for x,y in zip(pa,pb):torch.testing.assert_close(x,y,rtol=0,atol=0,equal_nan=True)
            qa=targets(oa,pa[0],pa[1]);qb=targets(ob,pb[0],pb[1]);self.assertTrue(torch.equal(qa['distribution'],qb['distribution']))
            la=train_loss(oa,a,True);lb=train_loss(ob,b,True);self.assertTrue(torch.equal(la,lb));la.backward();lb.backward()
            for k,p in m.named_parameters():
                q=dict(n.named_parameters())[k]
                if not k.startswith('metadata_'):self.assertTrue(torch.equal(p.grad,q.grad),k)
            self.assertGreater(float(m.metadata_scorer[-1].weight.grad.norm()),0)
            self.assertGreater(float(n.metadata_scorer[-1].weight.grad.norm()),0)
            self.assertEqual(float(m.metadata_encoder[0].weight.grad.norm()),0)
    def test_order_and_optimizer_schedule(self):
        opt=E.protocol()['optimizer']
        a=torch.optim.AdamW(model().parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
        b=torch.optim.AdamW(model().parameters(),lr=opt['lr'],betas=tuple(opt['betas']),weight_decay=opt['weight_decay'])
        self.assertEqual(a.defaults,b.defaults)
        for track,data in [('A',self.paper),('B',self.mixed)]:
            for seed in [1,2,3]:self.assertEqual(E.order_sha(data.order(seed)),E.protocol()['orders'][f'{track}_seed{seed}']['sequence_sha256'])
        self.assertEqual(E.protocol()['T'],1);self.assertEqual(E.protocol()['rule'],{'lam':1.,'max_move_image_diagonal_fraction':.01})
        lr=[E.D.old('train').learning_rate(i,dict(steps=6000,optimizer=opt)) for i in [1,100,3000,6000]]
        self.assertAlmostEqual(lr[-1],.0001);self.assertAlmostEqual(lr[1],.001)
    def test_inference_excludes_GT_arrays(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key in ['gt_points','gt_valid']:raise AssertionError('GT read by inference')
                return super().__getitem__(key)
        d=self.paper.base;original=d.arrays;d.arrays=Guard(original)
        try:
            for arm in E.ARMS['A']:
                batch=self.paper.batch(d.validation_rows[:2],arm,device='cpu',supervision=False)
                self.assertNotIn('gt_points',batch);self.assertNotIn('permutations',batch)
        finally:d.arrays=original
    def test_perturbations_preserve_dimensions(self):
        z=torch.cat([torch.randn(3,5),torch.eye(3)],-1)
        for mode in ['BLIND','CORRECT','NEUTRAL','ZERO','WRONG','SHUFFLED']:
            q=alter(z,mode,[2,4,1]);self.assertTrue(torch.equal(z[:,:5],q[:,:5]))
        self.assertTrue(torch.equal(alter(z,'WRONG')[:,5:],torch.eye(3)[[1,2,0]]))
        self.assertTrue(torch.equal(alter(z,'NEUTRAL'),alter(z,'BLIND')))
    def test_detector_and_evaluator_contract(self):
        p=np.arange(18,dtype=float).reshape(9,2);c=[dict(score=.9,box_xyxy=[0,0,100,100],keypoints_xy=p.tolist())]
        q=replace_selected(c,0,p,p+1,1,1);preservation(c,q,0)
        perms=E.read(E.D.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations']
        a=measure(p,p,np.ones(9,bool),perms,(480,640));b=measure(p,p,np.ones(9,bool),perms,(480,640));self.assertEqual(a,b)
        m=measure(p,p,np.ones(9,bool),perms,(480,640),matched=False,detected=False);self.assertEqual(m['E_sym'],1)
if __name__=='__main__':unittest.main()
