import unittest,copy
import numpy as np
import torch
import dcp_env as E
from refiner import *
from data import validate_group
from point_inference import replace_selected

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.config=dict(c3=2,c4=3,hidden=4,encoded=4,stencil_fraction=.13)
        cls.perms=E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations']
    def batch(self):
        torch.manual_seed(9)
        return dict(p3=torch.randn(1,2,8,8),p4=torch.randn(1,3,4,4),points=torch.ones(1,9,2)*30,
          boxes=torch.tensor([[5.,5.,55.,55.]]),point_valid=torch.ones(1,9,dtype=torch.bool),input_shape=torch.tensor([[64,64]]),context=torch.zeros(1,8))
    def test_zero_initial_exact_logits_and_base_state(self):
        torch.manual_seed(1);p=model('N0_BASE_REPLAY',self.config)
        torch.manual_seed(1);m=model('N4_META_SYM',self.config)
        for k,v in p.state_dict().items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
        b=self.batch();o=forward(p,b)
        for z in [torch.zeros(1,8),torch.ones(1,8),torch.arange(8)[None].float()]:
            b['context']=z;q=forward(m,b);self.assertTrue(torch.equal(o['logits'],q['logits']));self.assertEqual(q['metadata_residual'].abs().max(),0)
    def test_metadata_gradient_step1(self):
        m=model('N4_META_SYM',self.config);b=self.batch();o=forward(m,b);v=loss(o,b['points']+1,b['point_valid']);v.backward()
        self.assertGreater(float(m.metadata_scorer[-1].weight.grad.norm()),0)
        # Zero last weight implies encoder gradient0 on first backward, not broken wiring.
        self.assertEqual(float(m.metadata_encoder[0].weight.grad.norm()),0)
    def test_metadata_changes_only_residual_path(self):
        m=model('N4_META_SYM',self.config);torch.nn.init.normal_(m.metadata_scorer[-1].weight)
        b=self.batch();a=forward(m,b);b['context']+=2;q=forward(m,b)
        self.assertTrue(torch.equal(a['base_logits'],q['base_logits']));self.assertFalse(torch.equal(a['logits'],q['logits']))
    def test_missing_context_error(self):
        m=model('N4_META_SYM',self.config);b=self.batch();b['context'][0,0]=float('nan')
        with self.assertRaises(ValueError):forward(m,b)
    def test_missing_geometry_rejected_even_without_detection(self):
        from inference import predict_captured
        captured=dict(candidates=[],selected_index=None);norm=dict(mean=[0]*5,scale=[1]*5)
        with self.assertRaises(ValueError):predict_captured(None,'N4_META_SYM',captured,None,2,1,{},(480,640),norm)
        p,d=predict_captured(None,'N4_META_SYM',captured,[1.1,1.3,.11],2,1,{},(480,640),norm)
        self.assertEqual(p['candidates'],[]);self.assertIsNone(d)
    def test_invalid_dimensions_error(self):
        for dims in [[1,2,0],[1,float('nan'),2],[1,2]]:
            with self.assertRaises(ValueError):dimension_features(dims)
    def test_dimensions_not_sorted_or_swapped(self):
        a=dimension_features([[1.1,1.3,.11]]);self.assertAlmostEqual(a[0,0],np.log(1.1));self.assertAlmostEqual(a[0,1],np.log(1.3))
        self.assertFalse(np.array_equal(a,dimension_features([[1.3,1.1,.11]])))
    def test_camera_facing_WD_not_used_as_canonical_input(self):
        builder=E.C.module('dcp_test_fixed_dimension_provenance',E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/build_probe_metadata.py')
        fixed=[]
        for phase,p in enumerate(self.perms):
            # Same physical object, different camera-facing label encodings.
            w,d=(1.1,1.3) if phase%2==0 else (1.3,1.1)
            obj=dict(dimensions_m=dict(width=w,height=.11,depth=d),perm_v4=p[:8])
            camera,xyz,_,_=builder.fixed_dimensions(obj,'fixture');fixed.append(xyz)
        for xyz in fixed:self.assertEqual(xyz,[1.1,.11,1.3])
    def test_explicit_group_not_inferred(self):
        norm=dict(mean=[0]*5,scale=[1]*5)
        a=context([[1,1,.1]],[1],norm,True);self.assertEqual(a[0,5:].tolist(),[1,0,0])
        with self.assertRaises(ValueError):context([[1,1,.1]],[3],norm,True)
    def test_C1_C2_C4_topology(self):
        for order,p in [(1,[self.perms[0]]),(2,[self.perms[0],self.perms[2]]),(4,self.perms)]:self.assertTrue(validate_group(p,order))
        with self.assertRaises(ValueError):validate_group([self.perms[0],self.perms[1]],2)
    def test_reflection_rejected(self):
        with self.assertRaises(ValueError):validate_group([self.perms[0],[1,0,3,2,5,4,7,6,8]],2)
    def phase(self,gt,valid,pred,order=4):
        p=self.perms if order==4 else [self.perms[0]]
        return local_phase(pred,torch.ones(1,9,dtype=torch.bool),gt,valid,torch.tensor([p]),torch.ones(1,len(p),dtype=torch.bool),torch.tensor([100.]))
    def test_rotated_R0_local_and_mask_moves(self):
        gt=torch.arange(18).reshape(1,9,2).float();v=torch.ones(1,9,dtype=torch.bool);v[0,1]=False
        pred=gt[:,self.perms[1]];target,mask,g,_=self.phase(gt,v,pred)
        self.assertEqual(int(g),1);self.assertTrue(torch.equal(target,pred));self.assertTrue(torch.equal(mask,v[:,self.perms[1]]))
        self.assertTrue(torch.equal(target[:,8],gt[:,8]))
    def test_equivalent_relabeling_same_target(self):
        gt=torch.arange(18).reshape(1,9,2).float();v=torch.ones(1,9,dtype=torch.bool);pred=gt[:,self.perms[1]]
        a=self.phase(gt,v,pred)[0]
        for p in self.perms:self.assertTrue(torch.equal(a,self.phase(gt[:,p],v[:,p],pred)[0]))
    def test_identity_first_tie_and_missing(self):
        gt=torch.zeros(1,9,2);v=torch.ones(1,9,dtype=torch.bool)
        self.assertEqual(int(self.phase(gt,v,gt)[2]),0)
        self.assertTrue(torch.isfinite(self.phase(gt,v,torch.full_like(gt,float('nan')))[3]).all())
    def test_no_pointwise_assignment(self):
        gt=torch.arange(18).reshape(1,9,2).float();pred=gt.flip(1);v=torch.ones(1,9,dtype=torch.bool)
        a=self.phase(gt,v,pred)[0];self.assertTrue(any(torch.equal(a,gt[:,p]) for p in self.perms))
    def test_out_of_radius_finite(self):
        b=self.batch();o=forward(model('N0_BASE_REPLAY',self.config),b)
        self.assertTrue(torch.isfinite(loss(o,b['points']+1e4,b['point_valid'])))
    def test_center_and_lam0_exact(self):
        b=self.batch();o=forward(model('N4_META_SYM',self.config),b)
        self.assertTrue(torch.equal(decode(o)[:,8],b['points'][:,8]));self.assertTrue(torch.equal(decode(o,lam=0),b['points']))
    def test_candidate_replacement_only_selected_coordinates(self):
        c=[dict(score=.8,box_xyxy=[0,0,10,10],keypoints_xy=np.arange(18).reshape(9,2).tolist()),dict(score=.7,box_xyxy=[1,1,11,11],keypoints_xy=np.zeros((9,2)).tolist())]
        p=np.array(c[0]['keypoints_xy']);out=replace_selected(c,0,p,p+1,1,1)
        self.assertEqual(len(out),len(c));self.assertEqual(out[1],c[1]);self.assertEqual(out[0]['score'],c[0]['score']);self.assertEqual(out[0]['box_xyxy'],c[0]['box_xyxy']);self.assertTrue(np.array_equal(out[0]['keypoints_xy'][8],p[8]))
    def test_normalization_uses_saved_train_statistics(self):
        norm=dict(mean=[1]*5,scale=[2]*5);z=context([[1,1,1]],[2],norm,False)
        self.assertTrue(np.array_equal(z,np.full((1,5),-.5,np.float32)))

if __name__=='__main__':unittest.main()
