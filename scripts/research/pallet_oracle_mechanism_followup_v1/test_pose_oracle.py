"""Meaningful invariants for fixed-candidate pose diagnostics."""
import inspect
import unittest

import numpy as np
from . import pose_oracle as O


class PoseOracleContracts(unittest.TestCase):
    def test_oracle_order_and_tie_invariance(self):
        items = [dict(name='B', metric=dict(available=True, ADDsym_normalized=.04)),
                 dict(name='A', metric=dict(available=True, ADDsym_normalized=.04)),
                 dict(name='C', metric=dict(available=False))]
        self.assertEqual(O.oracle_choice(items)['name'], 'A')
        self.assertEqual(O.oracle_choice(items[::-1])['name'], 'A')
        self.assertIsNone(O.oracle_choice(items[-1:]))

    def test_failure_keeps_full_auc_denominator(self):
        full = O.D.Pose.pose_auc([0., float('inf')], 1.)
        self.assertAlmostEqual(full, .5, places=12)
        self.assertEqual(O.D.Pose.pose_auc([float('inf')], 1.), 0.)

    def test_auc_fraction_and_diameter_scale(self):
        fn = O.D.Pose.pose_auc
        self.assertEqual(fn([.10001], 1.), 0.)
        self.assertAlmostEqual(fn([.05], 1.), .5005, places=10)
        self.assertAlmostEqual(fn([.015, .05, .14], 1.), fn([.045, .15, .42], 3.), places=10)

    def test_inference_api_has_no_reference_argument(self):
        self.assertEqual(list(inspect.signature(O.candidate_record).parameters), ['prediction', 'metadata'])
        source = inspect.getsource(O.candidate_record)
        self.assertNotIn('Pose.metadata', source)
        self.assertNotIn('P.TRUTH', source)

    def test_native_production_parity_and_repeated_solve(self):
        for material in O.ARMS:
            rows, predictions, poses, _ = O.population(material)
            row = rows[0]; fid = row['id']
            current = O.candidate_record(predictions['R0'][fid], row)
            O.D.close(current['current'], poses['R0'][fid])
            O.D.close(current, O.candidate_record(predictions['R0'][fid], row))
            self.assertIn(current['selected_name'], [h['name'] for h in current['hypotheses']])

    def test_no_detection_does_not_generate_oracle_candidates(self):
        row = dict(K=np.eye(3), xyz=[1.1, .11, 1.3])
        result = O.candidate_record(dict(selected_index=None, candidates=[]), row)
        self.assertFalse(result['current']['available'])
        self.assertEqual(result['hypotheses'], [])

    def test_frozen_artifact_membership_parity_and_bound(self):
        path = O.RAW / 'POSE_ORACLE_RESULTS.json'
        if not path.exists():
            self.skipTest('Run frozen diagnostics first')
        result = O.read(path)
        lock = O.read(O.RAW / 'CANDIDATES_LOCK.json')
        for binding in lock['files'] + lock['sources']:
            O.verify(binding)
        for material, expected_n in [('PLASTIC', 128), ('WOOD', 45)]:
            metrics = O.read(O.RAW / f'{material}_METRICS.json')['arms']
            for arm in O.ARMS[material]:
                self.assertEqual(len(metrics[arm]), expected_n)
                summary = O.aggregate(list(metrics[arm].values()))
                O.D.close(summary, result['materials'][material]['groups']['ALL'][arm])
                self.assertGreaterEqual(summary['gap'], 0.)
                for record in metrics[arm].values():
                    self.assertEqual((O.oracle_choice(record['hypotheses']) or {}).get('name'), record['oracle_name'])


if __name__ == '__main__':
    unittest.main()
