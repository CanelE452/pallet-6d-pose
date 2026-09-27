"""Bounded intervention and diagnostic aggregation regression checks."""
import unittest
import numpy as np
from . import common as C
from .diagnose import paired

class Contracts(unittest.TestCase):
    def test_original_binding_unchanged(self):
        for b in C.read(C.DOC/'INPUT_BINDINGS.json')['files']:C.verify(b)
    def test_teacher_identity(self):
        q=C.read(C.P.DOC/'PSEUDO_LABEL_QUALITY.json')
        self.assertEqual(q['teacher_checkpoint']['sha256'],'fc9b3d7b7f2e7c38b608a12461cf16a58b76669f12ef9be5c8dc35d81104a41d')
    def test_actual217_and_occurrences(self):
        r=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json')
        self.assertEqual(len(r),217);self.assertEqual(sum(x['occurrences_per_epoch'] for x in r),512)
        self.assertEqual(sum(sum(x['common_support'][:8]) for x in r),1627)
    def test_tie_does_not_mean_same_points(self):
        d=C.read(C.DOC/'REFERENCE_AND_THRESHOLD_SENSITIVITY.json')['paired']['ALL']
        self.assertEqual((d['gains10'],d['losses10'],d['correct_delta10']),(3,3,0))
        self.assertEqual((d['continuous_improve'],d['continuous_worsen']),(34,32))
    def test_boundary_inclusive(self):
        r=[{'errors':{'RAW_LR5':11,'REF_LR5':10,'TEACHER':10}}]
        p=paired(r);self.assertEqual(p['gains10'],1);self.assertEqual(p['losses10'],0)
    def test_shared_slots_and_mask(self):
        p=C.read(C.DOC/'NEW_PAIR_PREFLIGHT.json')['parity']
        for k in ('rgb_order','boxes','support','slots'):self.assertEqual(p['RAW'][k],p['REF'][k])
        self.assertNotEqual(p['RAW']['coordinates'],p['REF']['coordinates'])
    def test_ignore_gradient(self):
        t=C.read(C.DOC/'TRUE_IGNORE_TRAIN_FIXTURE_TEST.json');self.assertEqual(t['status'],'PASS')
        self.assertEqual(t['runs']['C_all_ignored']['keypoint_branch_grad'],0)
    def test_one_intervention(self):
        d=C.read(C.DOC/'DECISION_BEFORE_FIT.json');self.assertEqual(d['selected'],'A_budget');self.assertEqual(d['new_fits_planned'],2)
        lock=C.read(C.DOC/'INTERVENTION_LOCK.json');lr=lock['learning_rates_by_epoch'];self.assertEqual(lr[5:],[lr[4]]*5)
    def test_completed_pair(self):
        for a in ('RAW_NEW','REF_NEW'):
            f=C.read(C.DOC/f'FIT_{a}.json');self.assertEqual(f['optimizer_steps'],640);self.assertTrue(f['protected_state_exact'])
            p=C.read(C.DOC/f'PREFIX_{a}.json');self.assertTrue(p['first320_pass']);self.assertTrue(p['model_bit_exact'])
    def test_predictions_before_scoring(self):
        p=C.read(C.DOC/'PREDICTIONS_LOCK.json');s=C.read(C.DOC/'SCORING_START.json')
        self.assertLess(p['utc'],s['utc']);self.assertFalse(p['reference_coordinates_opened']);self.assertFalse(p['independent_test'])
        for b in p['files']:C.verify(b)
    def test_original_results_preserved(self):
        old=C.read(C.P.DOC/'CORE_RESULTS.json')['groups']['ALL'];new=C.read(C.DOC/'RESULTS.json')['full128']['ALL']
        for a in C.ARMS:
            self.assertEqual(old[a]['twoD'],new[a]['twoD']);self.assertEqual(old[a]['sixD'],new[a]['sixD'])
    def test_all_denominators(self):
        r=C.read(C.DOC/'RESULTS.json')
        for a in (*C.ARMS,'RAW_NEW','REF_NEW'):
            self.assertEqual(r['visible66']['ALL'][a]['n'],66)
            self.assertEqual(r['full128']['ALL'][a]['twoD']['corners'],985)
            self.assertEqual(r['full128']['ALL'][a]['sixD']['frames'],128)

if __name__=='__main__':unittest.main()
