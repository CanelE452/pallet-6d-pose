"""CPU tests for exact missing-control roles and fixed-name metric lookup."""
import subprocess
import sys
import unittest

import numpy as np

from . import controls as T


class ControlsContract(unittest.TestCase):
    def test_missing_controls_are_existing_models_only(self):
        self.assertEqual(T.MISSING,dict(R0='R0',OLD_REF='OLD_REF',
            RAW_CLEAR_S42='CLEAN_RAW_CLEAR',REF_CLEAR_S42='CLEAN_REF_CLEAR'))
        self.assertNotIn('RAW_CLEAR_S43',T.MISSING)
        self.assertNotIn('REF_CLEAR_S43',T.MISSING)

    def test_output_binding_never_targets_old_namespace(self):
        for alias in T.MISSING:
            value=T.configure(alias)
            self.assertTrue(value['private'].is_relative_to(T.C.RAW))
            self.assertTrue(value['lock'].is_relative_to(T.C.DOC))
            self.assertTrue(value['result'].is_relative_to(T.C.DOC))
            self.assertFalse(value['private'].is_relative_to(T.OLD.RAW))

    def test_old_output_write_rejected(self):
        with self.assertRaises(AssertionError):
            T.C.save(T.OLD.DOC/'NEVER_WRITE_TEST.json',{})

    def test_contrasts_unique_and_real_seed_scope(self):
        pairs=T.contrast_pairs()
        self.assertEqual(len(pairs),len(set(pairs)))
        self.assertIn(('R0_GEO','REF_CLEAR_S42_GEO'),pairs)
        self.assertIn(('REF_CLEAR_S42_GEO','REF_OCC_S42_GEO'),pairs)
        self.assertIn(('RAW_OCC_S43_GEO','REF_OCC_S43_GEO'),pairs)
        self.assertFalse(any('CLEAR_S43' in a or 'CLEAR_S43' in b for a,b in pairs))

    def test_fixed_name_not_best_metric(self):
        options=[dict(name='a',metric={'error':999}),dict(name='b',metric={'error':0})]
        self.assertEqual(T.candidate_metric_by_name(options,'a',None),{'error':999})
        self.assertEqual(T.candidate_metric_by_name(options,'b',None),{'error':0})
        self.assertEqual(T.candidate_metric_by_name(options,'absent',{'available':False}),{'available':False})
        with self.assertRaises(AssertionError):
            T.candidate_metric_by_name(options+options,'a',None)

    def test_same_frozen_selection_order_and_tie(self):
        from scripts.research.pallet_selector_recovery_v1.models import selection
        names=('long-face-front','short-face-front')
        score=np.array([[2.,1.],[0.,0.]])
        np.testing.assert_array_equal(selection(score,names),[1,0])
        np.testing.assert_array_equal(selection(score,names),1-selection(score[:,::-1],names[::-1]))

    def test_freeze_guard_rejects_metrics_before_filesystem_read(self):
        code="from scripts.research.pallet_clean_to_pose_transfer_v1.selector_compat import guard_reference_reads;guard_reference_reads();open('/tmp/POSE_METRICS.json')"
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('REFERENCE_READ_DENIED',result.stderr)


if __name__=='__main__':unittest.main()
