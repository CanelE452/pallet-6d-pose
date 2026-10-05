"""Conditional observations keep canonical identity and the global symmetry branch."""
import unittest

import numpy as np

from scripts.research.pallet_combined_closeout_20261003_v1.closeout_visibility_square import (
    canonical_points, family_rows, summarize_points)
from scripts.research.pallet_n3_completion_v3 import metrics as M


class VisibilityStratumTests(unittest.TestCase):
    def row(self, points, valid, permutations):
        return dict(id='synthetic:0',session='synthetic',hw=[480,640],
            gt=[[100.*i,0.] for i in range(9)],valid=valid,permutations=permutations,
            predictions={'base':points},matched={'base':True},detected={'base':True})

    def test_partial_detection_uses_one_global_branch_and_correct_canonical_stratum(self):
        pred=np.full((9,2),np.nan);pred[0]=[100.,0.]
        perm0=list(range(9));perm1=[1,0,*range(2,9)]
        row=self.row(pred.tolist(),[True,True]+[False]*7,[perm0,perm1])
        scored=M.score_corner_rows([row],'base')
        self.assertEqual(scored[0]['branch'],1)
        labels={('DEV319','synthetic:0',0):'DIRECT_VISIBLE',
                ('DEV319','synthetic:0',1):'EXTERNAL_OCCLUDED'}
        points=canonical_points([row],'base',scored,labels,'DEV319','dope','locked')
        visible=summarize_points([p for p in points if p['category']=='DIRECT_VISIBLE'],'DIRECT_VISIBLE')
        external=summarize_points([p for p in points if p['category']=='EXTERNAL_OCCLUDED'],'EXTERNAL_OCCLUDED')
        self.assertEqual(visible['reference_corners'],1)
        self.assertEqual(visible['observed_corners'],0)
        self.assertIsNone(visible['median_px'])
        self.assertEqual(visible['PCK10_percent'],0.)
        self.assertEqual(external['observed_corners'],1)
        self.assertEqual(external['median_px'],0.)
        self.assertEqual(external['PCK10_percent'],100.)
        self.assertTrue(all(p['whole_object_branch']==1 for p in points))

    def test_finite_prediction_error_equal_to_failure_penalty_remains_observed(self):
        # A legitimate finite error may equal the image diagonal. It must not be
        # removed by comparing its numeric value against the missing-point penalty.
        pred=np.full((9,2),np.nan);pred[0]=[800.,0.]
        row=self.row(pred.tolist(),[True]+[False]*8,[list(range(9))])
        scored=M.score_corner_rows([row],'base')
        points=canonical_points([row],'base',scored,
            {('DEV319','synthetic:0',0):'DIRECT_VISIBLE'},'DEV319','dope','locked')
        summary=summarize_points(points,'DIRECT_VISIBLE')
        self.assertTrue(points[0]['observed'])
        self.assertEqual(summary['observed_corners'],1)
        self.assertEqual(summary['median_px'],800.)
        self.assertEqual(summary['PCK10_percent'],0.)

    def test_empty_category_is_NA_and_measured_zero_remains_zero(self):
        empty=summarize_points([],'UNKNOWN')
        self.assertEqual(empty['metric_status'],'NA_EMPTY_CATEGORY')
        self.assertIsNone(empty['median_px'])
        self.assertIsNone(empty['PCK10_percent'])
        measured=summarize_points([dict(frame_id='synthetic',error_px=0.,observed=True)],'DIRECT_VISIBLE')
        self.assertEqual(measured['metric_status'],'MEASURED')
        self.assertEqual(measured['median_px'],0.)

    def test_family_uses_per_seed_statistics_without_averaging_predictions(self):
        rows=[dict(backbone='synthetic',mode='fixed',method=f'n3_seed{i}',seed=i,
                   reference_corners=602,median_px=float(value))
              for i,value in enumerate((2,8,20),1)]
        result=family_rows(rows,('backbone','mode'))[0]
        self.assertEqual(result['method'],'N3')
        self.assertEqual(result['seeds'],3)
        self.assertEqual(result['reference_corners'],602)
        self.assertEqual(result['median_px'],10.)


if __name__=='__main__':unittest.main()
