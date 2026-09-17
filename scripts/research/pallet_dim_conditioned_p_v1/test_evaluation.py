import unittest
import numpy as np
from types import SimpleNamespace
import dcp_env as E
from eval_math import measure,damage,summary

class Evaluation(unittest.TestCase):
    def setUp(self):
        self.p=np.array(E.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'][-1]['permutations'])
        self.gt=np.arange(18).reshape(9,2)*20.;self.v=np.ones(9,bool)
    def test_missed_detection_full_denominator(self):
        r=measure(np.zeros((9,2)),self.gt,self.v,self.p,(480,640),matched=False,detected=False);r['id']='missing'
        s=summary([r]);self.assertEqual(s['E_sym'],1);self.assertEqual(s['missing'],1);self.assertEqual(s['corners'],8)
    def test_one_branch_all_metrics(self):
        r=measure(self.gt[self.p[1]],self.gt,self.v,self.p,(480,640))
        self.assertEqual(r['branch'],1);self.assertEqual(r['E_sym'],0);self.assertEqual(max(r['errors']),0);self.assertGreater(r['E_fixed'],0)
    def test_damage_canonical_alignment_with_mask(self):
        self.v[2]=False
        a=self.gt.copy();a[3]+=np.array([30,0]);b=self.gt[self.p[1]].copy();b[np.where(self.p[1]==3)[0][0]]+=np.array([30,0])
        r=measure(a,self.gt,self.v,self.p,(480,640));q=measure(b,self.gt,self.v,self.p,(480,640));r['id']=q['id']='x'
        self.assertNotEqual(r['branch'],q['branch']);d=damage([r],[q]);self.assertEqual(d['good5_to_bad10'],0);self.assertEqual(d['bad20_to_good10'],0)
        self.assertEqual(r['canonical_errors'],q['canonical_errors'])
    def test_cached_inference_never_reads_GT_arrays(self):
        from paper_evaluate import light_batch
        class NoGT(dict):
            def __getitem__(self,key):
                if key.startswith('gt_'):raise AssertionError('Inference accessed GT')
                return super().__getitem__(key)
        arrays=NoGT(points=self.gt[None].astype(np.float32),boxes=np.array([[0,0,300,300]],np.float32),point_valid=self.v[None])
        logs=dict(rows=np.array([0]),logits=np.zeros((1,8,222),np.float32),support=np.ones((1,8),bool))
        _,batch,output=light_batch(SimpleNamespace(arrays=arrays),logs,np.array([0]))
        self.assertEqual(set(batch),{'points','boxes','point_valid'})
        self.assertEqual(tuple(output['logits'].shape),(1,8,222))
if __name__=='__main__':unittest.main()
