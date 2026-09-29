import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import report as R


class ReportTests(unittest.TestCase):
    def test_missing_decision_is_nonfinal_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(R.C, 'DOC', Path(folder)):
            result = R.main()
            self.assertEqual(result['status'], 'NOT_FINAL')
            self.assertFalse(result['passed'])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_pending_branch_prevents_final_even_if_declared_final(self):
        value = dict(status='FINAL', branches=dict(seed=dict(status='PENDING_SCORE')))
        self.assertEqual(R.decision_status(value), ('NOT_FINAL', ['seed']))

    def test_explicit_not_run_is_not_unfinished(self):
        value = dict(finalized=True, branches=dict(bridge=dict(status='NOT_RUN_PREDECLARED')))
        self.assertEqual(R.decision_status(value), ('FINAL', []))

    def test_metric_difference_not_paired_median(self):
        before = dict(conditional=dict(translation_cm=dict(median=4.), rotation_deg=dict(median=2.)))
        after = dict(conditional=dict(translation_cm=dict(median=3.), rotation_deg=dict(median=3.)))
        self.assertEqual(R.difference(before, after), '-1.0000/+1.0000')

    def test_private_input_rejected(self):
        with self.assertRaises(ValueError):
            R.public_input(R.C.RAW/'PRIVATE.json')

    def test_exactly_six_prespecified_contrasts(self):
        self.assertEqual(len(R.PAIRS), 6)
        self.assertEqual(R.PAIRS[0][:2], ('CLEAN_REF_CLEAR', 'CLEAN_REF_OCC'))
        self.assertEqual(R.PAIRS[-1][:2], ('R0', 'CLEAN_REF_OCC'))


if __name__ == '__main__':
    unittest.main()
