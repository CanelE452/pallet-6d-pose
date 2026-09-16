import math
import unittest
import numpy as np
import torch
from audit_math import (derive_permutations, validate_group, validate_edges,
                        edge_channel_permutations, symmetry_error,
                        conditional_line_distribution, line_candidate_evidence, rerank_logits)

torch.set_num_threads(1)
EDGES = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
def cuboid(a=1., b=1.):
    return np.array([[-a,.2,-b],[a,.2,-b],[a,-.2,-b],[-a,-.2,-b],
                     [-a,.2,b],[a,.2,b],[a,-.2,b],[-a,-.2,b],[0,0,0]])
def rotations(order):
    return np.array([[[math.cos(t),0,math.sin(t)],[0,1,0],[-math.sin(t),0,math.cos(t)]]
                    for t in np.arange(order)*2*math.pi/order])
def line_fixture():
    dtype=torch.float64
    locations=torch.tensor([[0.,0.],[4.,0.],[0.,4.],[4.,4.]],dtype=dtype)[None].repeat(8,1,1)
    lines=torch.tensor([[1,0,-4],[0,1,-4],[1,-1,0],[1,0,-8],[0,1,-8],[1,1,-20]],dtype=dtype)
    logits=torch.zeros(12,6,dtype=dtype)
    logits[0,0]=20;logits[3,1]=20;logits[8,2]=20
    valid=torch.ones_like(logits,dtype=torch.bool)
    sigma=torch.full((6,),.7,dtype=dtype)
    return locations,logits,valid,lines,sigma

def evidence(args,mode='three_ambiguity'):
    return line_candidate_evidence(*args,EDGES,mode)

class SymmetryTests(unittest.TestCase):
    def test_c4_derive_and_closure(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES)
        validate_group(p,EDGES);self.assertEqual(p.shape,(4,9))
        self.assertTrue(np.all(p[:,8]==8))
    def test_rectangle_rejects_c4(self):
        with self.assertRaises(ValueError):derive_permutations(cuboid(1,1.3),rotations(4),EDGES)
    def test_rectangle_accepts_c2(self):
        self.assertEqual(len(derive_permutations(cuboid(1,1.3),rotations(2),EDGES)),2)
    def test_reflection_not_group_rotation(self):
        with self.assertRaises(ValueError):derive_permutations(cuboid(),np.array([np.diag([-1,1,1])]),EDGES)
    def test_nine_permutation_bijection(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES);p[1,0]=p[1,1]
        with self.assertRaises(ValueError):validate_group(p,EDGES)
    def test_three_incident_edges(self):
        inc=validate_edges(EDGES);self.assertTrue(np.all(inc.sum(1)==3))
        e=EDGES.copy();e[-1]=[0,2]
        with self.assertRaises(ValueError):validate_edges(e)
    def test_mask_and_coordinates_permute_together(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES)
        y=np.arange(18,dtype=float).reshape(9,2)*3
        pred=y[p[1]].copy();v=np.array([1,0,1,1,0,1,1,0,1],bool);y[~v]=np.nan
        s=symmetry_error(pred,y,v,p,100)
        self.assertAlmostEqual(s['equivalent_mean'],0)
        self.assertGreater(s['fixed_mean'],0)
        self.assertEqual(s['n_corners'],5)
        self.assertTrue(np.array_equal(s['gt_mask'],v[p[1]]))
    def test_metric_invariant_to_valid_relabeling(self):
        rng=np.random.default_rng(902);p=derive_permutations(cuboid(),rotations(4),EDGES)
        for _ in range(50):
            y=rng.normal(size=(9,2))*100;pred=y[p[2]]+rng.normal(size=(9,2))
            v=rng.random(9)>.35;v[0]=True
            base=symmetry_error(pred,y,v,p,1000)['equivalent_mean']
            for r in p:
                test=symmetry_error(pred,y[r],v[r],p,1000)['equivalent_mean']
                self.assertAlmostEqual(base,test,places=10)
    def test_missing_prediction_stays_in_denominator(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES)
        y=np.arange(18,dtype=float).reshape(9,2);pred=np.full((9,2),np.nan)
        s=symmetry_error(pred,y,np.ones(9,bool),p,100)
        self.assertEqual(s['equivalent_mean'],100)
        self.assertEqual(s['n_corners'],8)
    def test_no_gt_is_not_perfect_prediction(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES)
        s=symmetry_error(np.zeros((9,2)),np.full((9,2),np.nan),np.zeros(9,bool),p,100)
        self.assertFalse(s['evaluable'])
    def test_edge_orbit_whole_object(self):
        p=derive_permutations(cuboid(),rotations(4),EDGES)
        m=edge_channel_permutations(p,EDGES)
        for row in m:self.assertTrue(np.array_equal(np.sort(row),np.arange(12)))
        self.assertTrue(np.array_equal(m[0],np.arange(12)))

class LineTests(unittest.TestCase):
    def test_uniform_posterior_has_maximum_ambiguity(self):
        _,l,v,_,_=line_fixture();l.zero_()
        q,a,c,ok,_=conditional_line_distribution(l,v)
        self.assertTrue(torch.equal(a,torch.ones_like(a)))
        self.assertTrue(torch.equal(c,torch.zeros_like(c)))
        self.assertTrue(torch.allclose(q.sum(-1),torch.ones(12,dtype=q.dtype)))
    def test_flat_hough_is_neutral(self):
        x,l,v,h,s=line_fixture();l.zero_()
        for mode in ['three_equal','three_ambiguity','one_ambiguity']:
            out=evidence((x,l,v,h,s),mode)
            self.assertTrue(torch.equal(out['bias'],torch.zeros_like(out['bias'])))
    def test_all_missing_is_neutral(self):
        x,l,v,h,s=line_fixture();v.zero_()
        out=evidence((x,l,v,h,s));self.assertTrue(torch.equal(out['bias'],torch.zeros_like(out['bias'])))
        self.assertTrue(torch.isfinite(out['bias']).all())
    def test_eta_zero_is_bit_exact(self):
        x,l,v,h,s=line_fixture();b=evidence((x,l,v,h,s))['bias']
        p=torch.randn_like(b);self.assertTrue(torch.equal(p,rerank_logits(p,b,.25,0)))
    def test_three_lines_rescore_existing_candidates(self):
        args=line_fixture();b=evidence(args)['bias'][0]
        self.assertEqual(int(b.argmax()),3)
        point=torch.zeros((8,4),dtype=torch.float64)
        prob=rerank_logits(point,evidence(args)['bias'],1,1).softmax(-1)[0]
        xy=(prob[:,None]*args[0][0]).sum(0)
        self.assertTrue((xy>2).all()) # baseline expectation is (2,2)
    def test_line_sign_invariance(self):
        x,l,v,h,s=line_fixture()
        self.assertTrue(torch.allclose(evidence((x,l,v,h,s))['bias'],evidence((x,l,v,-3*h,s))['bias'],atol=1e-12))
    def test_coordinate_scale_consistency(self):
        x,l,v,h,s=line_fixture();scaled=h.clone();scaled[:,:2]/=3
        self.assertTrue(torch.allclose(evidence((x,l,v,h,s))['bias'],evidence((x*3,l,v,scaled,s*3))['bias'],atol=1e-12))
    def test_semantic_channel_order_consistency(self):
        args=line_fixture();perm=np.random.default_rng(3).permutation(12)
        x,l,v,h,s=args
        expected=evidence(args)['bias']
        observed=line_candidate_evidence(x,l[perm],v[perm],h,s,EDGES[perm],'three_ambiguity')['bias']
        self.assertTrue(torch.allclose(expected,observed,atol=1e-12))
    def test_temperature_applied_once(self):
        args=line_fixture();b=evidence(args)['bias'];p=torch.randn_like(b);t=.4;eta=.25
        adjusted=rerank_logits(p,b,t,eta)/t
        self.assertTrue(torch.allclose(adjusted,p/t+eta*b))
    def test_invalid_line_geometry_and_nan_logits(self):
        x,l,v,h,s=line_fixture();h[0]=torch.tensor([float('nan'),0,0]);l[:,1]=float('nan')
        out=evidence((x,l,v,h,s));self.assertTrue(torch.isfinite(out['bias']).all())
    def test_parallel_lines_do_not_require_inverse(self):
        x,l,v,h,s=line_fixture();h[:,:2]=torch.tensor([1.,0.])
        out=evidence((x,l,v,h,s));self.assertTrue(torch.isfinite(out['bias']).all())
    def test_score_bound_and_log_odds_bound(self):
        args=line_fixture();b=evidence(args)['bias'];self.assertLessEqual(float(b.abs().max()),1)
        self.assertLessEqual(float((b.max(-1).values-b.min(-1).values).max()),2)
    def test_no_forced_motion_for_unavailable_edge(self):
        x,l,v,h,s=line_fixture();v[[0,3,8]]=False
        out=evidence((x,l,v,h,s));self.assertTrue(torch.equal(out['bias'][0],torch.zeros(4,dtype=x.dtype)))
    def test_confident_wrong_lines_can_harm_no_safety_claim(self):
        # A high-confidence wrong geometric cue CAN win. The gate is NOT a guarantee.
        x,l,v,h,s=line_fixture();l.zero_();l[0,3]=20;l[3,4]=20;l[8,5]=20
        shifted=x.clone();shifted[:,3]=torch.tensor([8.,8.])
        b=evidence((shifted,l,v,h,s))['bias'][0]
        self.assertGreater(float(b[3]),float(b[0]))

if __name__=='__main__':unittest.main(verbosity=2)
