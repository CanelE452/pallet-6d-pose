"""Meaningful synthetic decoder checks; no head, detector, or PnP invocation."""
import unittest
import numpy as np
from . import observations as O
from .calibration import choose_threshold,wilson_lower


class ObservationChecks(unittest.TestCase):
    def setUp(self):
        points=np.array([[100,150],[300,150],[300,220],[100,220],[120,130],[280,130],[280,200],[120,200],[200,170]],float)
        self.query=O.query_geometry(points); self.query['raw_hw']=[480,640]
        self.cal=dict(supported_edges=list(range(12)),confidence=dict(enabled=True,threshold=.9),
                      uncertainty=dict(query_scale=1.,corner_scale=1.),geometry=dict(enabled=True,max_extrapolation_ratio=.2))
        self.logits=np.full((84,66),-20.,np.float32); self.logits[:,65]=20.

    def enable(self,ids,bins=32):
        self.logits[ids,65]=-20; self.logits[ids,bins]=20

    def test_risk_threshold_least_restrictive_and_ties(self):
        scores=np.r_[np.full(100,.9),np.full(100,.7),np.full(50,.5)]
        correct=np.r_[np.ones(200,bool),np.zeros(50,bool)]
        chosen,scan=choose_threshold(scores,correct)
        self.assertEqual(chosen['threshold'],.7)
        self.assertEqual(chosen['accepted'],200)
        self.assertGreaterEqual(chosen['Wilson_lower'],.95)
        self.assertLess(scan[-1]['Wilson_lower'],.95)
        self.assertIsNone(choose_threshold([.9],[True])[0])
        self.assertGreater(wilson_lower(200,200),.95)

    def test_multimode_uncertainty_preserves_one_map(self):
        self.logits[:,0]=20; self.logits[:,64]=20
        _,best,_,sigma=O.logits_statistics(self.logits)
        self.assertTrue((best==0).all())
        self.assertTrue((sigma>40).all())

    def test_true_endpoint_extrapolation_is_allowed(self):
        self.enable(list(range(7))+list(range(56,63)))
        decoded=O.decode(self.query,self.logits,self.cal)
        corner=next(c for c in decoded['corners'] if c['id']==0)
        np.testing.assert_allclose(corner['xy'],[100,150],atol=1e-8)
        self.assertAlmostEqual(corner['max_extrapolation_ratio'],1/6)
        self.assertEqual(decoded['diagnostics']['line_pair_hypotheses'],42)
        self.assertEqual(len({c['id'] for c in decoded['corners']}),len(decoded['corners']))
        self.assertTrue(all(line['edge'] not in corner['edges'] for line in decoded['partial_lines']))

    def test_pair_consensus_rejects_coherent_line_outlier(self):
        self.enable(list(range(7)))
        self.logits[0,32]=-20; self.logits[0,64]=20
        decoded=O.decode(self.query,self.logits,self.cal)
        line=next(l for l in decoded['lines'] if l['edge']==0)
        self.assertNotIn(0,line['queries'])
        self.assertAlmostEqual(abs(line['offset']),150.)

    def test_two_queries_cannot_manufacture_line(self):
        self.enable([0,1])
        self.assertEqual(O.decode(self.query,self.logits,self.cal)['lines'],[])

    def test_partial_support_does_not_manufacture_endpoint(self):
        self.enable([2,3,4]+list(range(56,63)))
        decoded=O.decode(self.query,self.logits,self.cal)
        self.assertTrue(any(l['edge']==0 for l in decoded['lines']))
        self.assertFalse(any(c['id']==0 for c in decoded['corners']))
        self.assertTrue(any(c['reason']=='EXCESSIVE_SUPPORT_EXTRAPOLATION' for c in decoded['diagnostics']['corner_candidates']))

    def test_unsupported_does_not_mean_physically_absent(self):
        self.enable(list(range(7))); self.cal['supported_edges']=[]
        decoded=O.decode(self.query,self.logits,self.cal)
        self.assertEqual(decoded['lines'],[])
        self.assertTrue(all(q['reason']=='MODEL_CALIBRATION_UNSUPPORTED' and not q['physical_absence_inferred'] for q in decoded['queries']))

    def test_outside_image_and_missing_image_shape(self):
        self.enable(list(range(7))); self.query['raw_hw']=[100,640]
        self.assertEqual(O.decode(self.query,self.logits,self.cal)['lines'],[])
        del self.query['raw_hw']
        with self.assertRaises(ValueError): O.decode(self.query,self.logits,self.cal)


if __name__=='__main__': unittest.main()
