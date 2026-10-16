"""Tests for denominator retention, threshold boundaries and tail accounting."""
import unittest

from .audit import distribution, normalize, paired, translation_tail


def metric(t, r=1.0, available=True):
    return normalize(dict(pose_available=available, translation_cm=t, rotation_deg=r,
        ADDsym_cm=t, success_le5=False if not available else t <= 5 and r <= 5,
        success_strict_lt5=False if not available else t < 5 and r < 5))


class AuditTests(unittest.TestCase):
    def test_sample_population_variance_and_untrimmed_tail(self):
        d = distribution([1., 2., 3., 100.])
        self.assertEqual(d['n'], 4)
        self.assertEqual(d['mean'], 26.5)
        self.assertAlmostEqual(d['variance_sample'], 2401.6666666666665)
        self.assertAlmostEqual(d['variance_population'], 1801.25)
        self.assertEqual(d['median'], 2.5)
        self.assertEqual(d['max'], 100.)
        self.assertAlmostEqual(d['P95'], 85.45)

    def test_threshold_boundary_is_reported_separately(self):
        m = metric(5., 5.)
        self.assertTrue(m['success_le5'])
        self.assertFalse(m['success_strict_lt5'])

    def test_saved_flag_disagreement_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            normalize(dict(pose_available=True, translation_cm=9., rotation_deg=1.,
                ADDsym_cm=9., success_le5=True, success_strict_lt5=True))

    def test_no_pose_failure_remains_in_paired_full_denominator(self):
        no_pose = normalize(dict(pose_available=False, translation_cm=None,
            rotation_deg=None, ADDsym_cm=None, success_le5=False, success_strict_lt5=False))
        rows = [dict(id='a', session='s', metrics={'N1': metric(1.), 'N0': no_pose}),
                dict(id='b', session='s', metrics={'N1': metric(9.), 'N0': metric(1.)})]
        p = paired(rows, 'N1', 'N0')
        self.assertEqual(p['full_n'], 2)
        self.assertEqual(p['transition']['recovery']['ids'], ['a'])
        self.assertEqual(p['transition']['damage']['ids'], ['b'])
        self.assertEqual(p['error_differences']['translation_cm']['n'], 1)

    def test_net_tail_sum_preserves_large_cancellation(self):
        rows = [dict(id='a', session='s', metrics={'N0': metric(1.), 'N1': metric(101.)}),
                dict(id='b', session='s', metrics={'N0': metric(100.), 'N1': metric(1.)}),
                dict(id='c', session='s', metrics={'N0': metric(2.), 'N1': metric(2.)})]
        t = translation_tail(rows)
        self.assertEqual(t['full_n'], 3)
        self.assertEqual(t['sum_delta_cm'], 1.)
        self.assertEqual(t['positive_delta_sum_cm'], 100.)
        self.assertEqual(t['negative_delta_sum_cm'], -99.)
        self.assertEqual(t['top10_worsened']['contribution_to_full_denominator_mean_cm'], 100/3)
        self.assertEqual(t['top10_improved']['sum_delta_cm'], -99.)


if __name__ == '__main__':
    unittest.main()
