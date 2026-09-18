import unittest
from unittest.mock import patch
import numpy as np
import torch
import cv_env as E
from evaluate import InferenceData
from code_adapter import PaperData,SquareData
from code_statistics import contrast

class Reporting(unittest.TestCase):
    def test_inference_constructor_does_not_open_target_arrays(self):
        original=np.load
        original_read=E.read
        def guarded(path,*args,**kwargs):
            self.assertNotIn(str(path).split('/')[-1],['gt_points.npy','gt_valid.npy','gt_support.npy','matched.npy','matched_iou.npy','matched_gt_index.npy'])
            return original(path,*args,**kwargs)
        def guarded_read(path):
            self.assertNotIn(str(path).split('/')[-1],['SOURCE_MANIFEST.json','square_annotation_target_view.json'])
            return original_read(path)
        for pop in ['SYNTH','SQUARE']:
            with patch('numpy.load',guarded),patch.object(E,'read',guarded_read):
                data=InferenceData(pop);batch=data.batch(data.rows[:2],device='cpu')
            self.assertEqual(set(batch),{'p3','p4','points','boxes','point_valid','input_shape','context'})
    def test_actual_inference_adapters_equal_original(self):
        for pop,old in [('SYNTH',PaperData()),('SQUARE',SquareData())]:
            new=InferenceData(pop);rows=new.rows[:3]
            a=new.batch(rows,device='cpu');b=old.batch(rows,'N4_META_SYM',device='cpu',supervision=False)
            for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0,equal_nan=True)
    def test_cluster_one_is_NA(self):
        x=contrast(np.ones((3,2)),np.zeros((3,2)),['s','s']);self.assertIsNone(x['CI95']);self.assertEqual(x['units'],1)
    def test_bootstrap_seed_and_direction(self):
        a=np.array([[1.,2.,3.]]*3);b=a+1;x=contrast(a,b);self.assertEqual(x['seed'],20260918);self.assertEqual(x['resamples'],10000)
        self.assertEqual(x['delta'],-1);self.assertEqual(x['improved_seeds'],3);self.assertEqual(x['CI95'],[-1.,-1.])
    def test_decision_requires_both_performance_and_perturbation(self):
        import copy
        from finish import decision
        null=dict(CI95=[-1.,1.],improved_seeds=1,per_seed_delta=[-1.,1.,1.])
        good=dict(CI95=[-2.,-1.],improved_seeds=3,per_seed_delta=[-1.,-1.,-1.])
        a={p:dict(contrast=copy.deepcopy(null),groups={'C1':copy.deepcopy(null),'C2':copy.deepcopy(null)},safety_pass=True) for p in ['SYNTH','DEV']}
        b=copy.deepcopy(a);b['SQUARE']=dict(contrast=copy.deepcopy(null),groups={'C4':copy.deepcopy(null)},safety_pass=True)
        mode=dict(applicable=True,contrast=copy.deepcopy(null),movement=dict(mean_coordinate_change_px=.1))
        c={'A1_CODE_AWARE':{'SYNTH':{'modes':{m:copy.deepcopy(mode) for m in ['NEUTRAL','WRONG']}}}}
        self.assertEqual(decision(a,b,c)['result'],'DROP_CODE')
        a['SYNTH']['contrast']=good;self.assertEqual(decision(a,b,c)['result'],'UNRESOLVED')
        for m in c['A1_CODE_AWARE']['SYNTH']['modes'].values():m['contrast']=dict(CI95=[1.,2.],improved_seeds=0,per_seed_delta=[1.,1.,1.])
        self.assertEqual(decision(a,b,c)['result'],'KEEP_CODE_DEV_EVIDENCE')
        b['DEV']['safety_pass']=False;self.assertEqual(decision(a,b,c)['result'],'UNRESOLVED')
if __name__=='__main__':unittest.main()
