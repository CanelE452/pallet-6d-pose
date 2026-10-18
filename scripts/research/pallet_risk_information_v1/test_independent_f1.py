from __future__ import annotations
import hashlib,math,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from . import independent_f1 as I
from .risk_models import RiskModel


class IndependentF1Tests(unittest.TestCase):
    def test_exact_canonical_72perfamily_schema(self):
        self.assertEqual(len(I.NAMES),72);self.assertEqual(len(set(I.NAMES)),72);self.assertEqual(I.NAMES[53],'B_fixed_profile_count');self.assertEqual(I.NAMES[56],'B_all_profiles_supported')
    def test_valid_only_signedlog_population_moments(self):
        raw=np.array([[0.,100.],[3.,999.]]);valid=np.array([[1,0],[1,0]],bool);m=I.stable_moments(raw,valid)
        self.assertEqual(m['valid_train_counts'],[2,0]);self.assertAlmostEqual(m['mean'][0],math.log(4.)/2);self.assertAlmostEqual(m['scale'][0],math.log(4.)/2);self.assertEqual(m['mean'][1],0.);self.assertEqual(m['scale'][1],1.)
    def test_missing_and_unseen_keep_validity_but_zero_scalar(self):
        normalizer=dict(mean=[0.]*144,scale=[1.]*144,valid_train_counts=[0]*144)
        row=dict(actions=[dict(extra_values=[10.]*144,extra_validity=[1.]*144)])
        result=I.independent_normalized(row,normalizer);self.assertTrue(np.array_equal(result[0,:144],np.zeros(144)));self.assertTrue(np.array_equal(result[0,144:],np.ones(144)))
    def test_no_profiles_are_not_a_confidence_one(self):
        name_to_index={n:i for i,n in enumerate(I.NAMES)}
        values=np.zeros(144);valid=np.ones(144);valid[name_to_index['B_supported_profile_fraction']]=0;valid[name_to_index['B_all_profiles_supported']]=0;values[name_to_index['B_no_fixed_observation']]=1
        self.assertEqual(values[name_to_index['B_no_fixed_observation']],1.);self.assertEqual(valid[name_to_index['B_all_profiles_supported']],0.)
    def test_independent_head_initialization_matches_both_arms(self):
        seed=20261018;expected,n=I.initialized_head_hash(seed,476);self.assertEqual(n,34753)
        for arm in('R_ORDER_F1_ZERO','R_ORDER_F1'):
            torch.manual_seed(seed);model=RiskModel(arm,288);h=hashlib.sha256()
            for name,value in model.state_dict().items():h.update(name.encode());h.update(value.numpy().tobytes())
            self.assertEqual(h.hexdigest(),expected)
    def test_no_GOnever_reads_F1_or_reference(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)
            with self.assertRaises(PermissionError):I.run(p,p,p,p,p,p,p/'output',GO=False)
            self.assertFalse((p/'output').exists())
    def test_two_by_two_finite_and_endpoint(self):
        query=dict(native_corner=0,native_neighbor=1,center_xy=[.5,-1.],outward_normal_xy=[0.,1.]);q=np.array([[0.,0.],[1.,0.]])
        status,coefficient=I.tiny_intersection(query,q,np.ones(2),(100,100),1.);self.assertEqual(status,'SUPPORTED_FINITE_SEGMENT');self.assertEqual(coefficient,.5)
        query['center_xy']=[2.,-1.];self.assertEqual(I.tiny_intersection(query,q,np.ones(2),(100,100),1.)[0],'AFTER_FINITE_ENDPOINT')
    def test_negative_slack_signed_log_not_clipped(self):
        raw=np.array([[-1000.],[-10.]]);m=I.stable_moments(raw,np.ones_like(raw,bool));self.assertLess(m['mean'][0],0.);self.assertGreater(m['scale'][0],0.)


if __name__=='__main__':unittest.main()
