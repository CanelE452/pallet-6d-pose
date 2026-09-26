"""Regression checks for denominator, coordinate-only control, and scope honesty."""
import unittest
from pathlib import Path
import numpy as np
from . import common as C
from .evaluate import paired
from scripts.research.pallet_verified_anchor_v1.evaluate import metrics,point

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res=C.read(C.DOC/'CORE_RESULTS.json')['groups']
        cls.q=C.read(C.DOC/'PSEUDO_LABEL_QUALITY.json')
        cls.p=C.read(C.RAW/'PREDICTIONS.json')
    def test_three_denominators(self):
        self.assertEqual(self.res['ALL']['R0']['twoD']['corners'],985)
        self.assertEqual(self.res['ALL']['R0']['twoD']['observed_corners'],931)
        self.assertEqual(self.q['points'],66)
    def test_severity_partitions(self):
        self.assertEqual([self.res[g]['R0']['sixD']['frames'] for g in ('CLEAN','MODERATE','SEVERE')],[29,21,78])
    def test_all_historical_settings_reported(self):
        self.assertEqual(set(self.res['ALL']),set(C.ARMS));self.assertEqual(len(C.ARMS),13)
    def test_raw_corrected_only_coordinates(self):
        p=C.read(C.REC/'pose_only/PROTOCOL.json')
        x=[Path(t) for t in (C.ROOT/p['datasets']['RAW']['train_list']['path']).read_text().splitlines()]
        y=[Path(t) for t in (C.ROOT/p['datasets']['REF']['train_list']['path']).read_text().splitlines()]
        self.assertEqual([i.name for i in x],[i.name for i in y])
        for a,b in zip(x,y):
            aa=np.array((a.parent.parent/'labels'/a.with_suffix('.txt').name).read_text().split(),float)
            bb=np.array((b.parent.parent/'labels'/b.with_suffix('.txt').name).read_text().split(),float)
            np.testing.assert_array_equal(aa[:5],bb[:5]);np.testing.assert_array_equal(aa[7::3],bb[7::3])
    def test_detector_unchanged(self):
        for a in C.ARMS[1:]:
            for fid,p in self.p[a].items():
                base=self.p['R0'][fid]
                self.assertEqual(p['selected_index'],base['selected_index'])
                self.assertEqual([(c['score'],c['box_xyxy']) for c in p['candidates']],[(c['score'],c['box_xyxy']) for c in base['candidates']])
    def test_teacher_identity_is_not_clean19(self):
        self.assertEqual(self.q['teacher_checkpoint']['sha256'],'fc9b3d7b7f2e7c38b608a12461cf16a58b76669f12ef9be5c8dc35d81104a41d')
    def test_tail_worsening_is_retained(self):
        r=self.res['ALL'];self.assertGreater(r['REF_LR5']['twoD']['matched_pooled_corner8_P90_px'],r['RAW_LR5']['twoD']['matched_pooled_corner8_P90_px'])
    def test_visible_student_tie_is_not_hidden(self):
        q=self.q['groups']['ALL'];self.assertEqual(q['RAW_LR5']['PCK']['10']['correct'],43)
        self.assertEqual(q['REF_LR5']['PCK']['10']['correct'],43)
        self.assertIn('43/66',(C.PAPER/'manuscript.tex').read_text())
    def test_missing_point_not_dropped(self):
        self.assertIsNone(point({'selected_index':None},0))
        self.assertEqual(metrics([0,800])['PCK']['10']['total'],2)
        self.assertEqual(metrics([0,800])['PCK']['10']['correct'],1)
    def test_fixed_identity_no_per_point_matching(self):
        p=dict(selected_index=0,candidates=[dict(keypoints_xy=[[100,0],[0,0]])])
        self.assertEqual(float(np.linalg.norm(point(p,0)-[0,0])),100)
    def test_recording_disjoint(self):
        x=C.read(C.DOC/'CORE_COMPARABILITY_AUDIT.json');self.assertFalse(set(x['train_recordings'])&set(x['eval_recordings']))
        t=C.read(C.DOC/'TEACHER_RECORDING_AUDIT.json');self.assertFalse(set(t['teacher_groups'])&set(t['heldout_groups']))
    def test_no_independent_claim(self):
        self.assertFalse(C.read(C.DOC/'CORE_RESULTS.json')['independent_confirmation'])
        self.assertFalse(C.read(C.DOC/'INDEPENDENT_CONFIRMATION_AUDIT.json')['confirmation_executed'])
    def test_no_new_fit(self):
        self.assertEqual(C.read(C.DOC/'FINAL_DECISION.json')['new_fits'],0)
        self.assertEqual(C.read(C.DOC/'FINAL_DECISION.json')['method_development'],'STOP')

if __name__=='__main__':unittest.main()
