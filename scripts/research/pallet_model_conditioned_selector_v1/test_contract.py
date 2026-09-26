import unittest
import numpy as np
from . import common as C
from scripts.research.pallet_selector_recovery_v1 import models as M,features as F

class ContractTests(unittest.TestCase):
    def test_94_exact_features(self):
        self.assertEqual(len(F.names()),94)
        self.assertEqual(F.names(),C.read(C.OLD/'SELECTOR_FEATURE_CONTRACT.json')['features'])
    def test_shared_candidate_swap_and_tie(self):
        s=np.array([[1.,2.],[2.,1.],[2.,2.]])
        np.testing.assert_array_equal(M.selection(s,C.HYP),1-M.selection(s[:,::-1],C.HYP[::-1]))
    def test_normalization_train_only(self):
        x=np.arange(3*2*4,dtype=np.float32).reshape(3,2,4);train=np.array([True,False,False]);y=x.copy();y[1:]+=10000
        _,a,b=M.normalize(x,train);_,c,d=M.normalize(y,train)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    def test_std_floor(self):
        _,_,s=M.normalize(np.ones((3,2,4),np.float32),np.ones(3,bool));np.testing.assert_allclose(s,1e-6)
    def test_category(self):
        r=dict(hypotheses=[dict(name=n,metric=dict(available=True,ADDsym_normalized=v)) for n,v in zip(C.HYP,(.1,.4))],oracle_name=C.HYP[0])
        self.assertEqual(C.category(r,C.HYP[0]),'SELECTOR_CORRECT');self.assertEqual(C.category(r,C.HYP[1]),'SELECTOR_RECOVERABLE')
        r['hypotheses'][1]['metric']['ADDsym_normalized']=.1+1e-13;self.assertEqual(C.category(r,C.HYP[1]),'ORACLE_TIE')
        r['hypotheses'][1]['metric']['available']=False;self.assertEqual(C.category(r,C.HYP[0]),'POSE_UNAVAILABLE')
    def test_oracle_direction(self):
        self.assertEqual(C.oracle_delta(.4,.1),'ORACLE_GAIN');self.assertEqual(C.oracle_delta(.1,.4),'ORACLE_LOSS')
        self.assertEqual(C.oracle_delta(.1,.1+1e-13),'ORACLE_TIE');self.assertEqual(C.oracle_delta(None,.1),'POSE_UNAVAILABLE')
    def run_decision(self,new,base=(.5,.5,.5),old=(.4,.4,.4),synthgain=True):
        groups={g:{'H_MANUAL_HMANSPEC_GEO':{'ADDsym_AUC':n},'S1_OLD_GEO':{'ADDsym_AUC':b},'H_MANUAL_OLD_GEO':{'ADDsym_AUC':o}} for g,n,b,o in zip(('CLEAN','MODERATE','SEVERE'),new,base,old)}
        return C.decision(groups,{'H_MANUAL_HMANSPEC_GEO':{'accuracy':.9 if synthgain else .5},'H_MANUAL_OLD_GEO':{'accuracy':.6}})
    def test_pipeline_beaten(self):self.assertEqual(self.run_decision((.5,.6,.5))['Q_PIPELINE'],'HMAN_PIPELINE_RECOVERS_AND_BEATS_BASE')
    def test_selector_gain_not_base(self):self.assertEqual(self.run_decision((.5,.45,.45))['Q_PIPELINE'],'HMAN_SELECTOR_GAIN_BUT_BASE_NOT_BEATEN')
    def test_mixed_no_routing(self):
        d=self.run_decision((.5,.6,.3));self.assertEqual(d['Q_SELECTOR_COMPATIBILITY'],'PARTIAL_SELECTOR_COMPATIBILITY_RECOVERY');self.assertEqual(d['current_deployment_candidate'],'S1_OLD_GEO')
    def test_tie_not_gain(self):self.assertEqual(self.run_decision((.5,.4,.4))['Q_SELECTOR_COMPATIBILITY'],'NO_SELECTOR_COMPATIBILITY_RECOVERY')
    def test_clean_safeguard(self):self.assertNotEqual(self.run_decision((.49,.6,.6))['Q_PIPELINE'],'HMAN_PIPELINE_RECOVERS_AND_BEATS_BASE')

if __name__=='__main__':unittest.main()
