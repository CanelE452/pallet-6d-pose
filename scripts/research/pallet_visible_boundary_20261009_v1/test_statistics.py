"""Tests of observation-weighted session resampling and strict damage thresholds."""
import hashlib
from types import SimpleNamespace
import unittest

import numpy as np

from . import statistics as S


class StatisticsTests(unittest.TestCase):
    def test_original_observation_denominator_and_empty_draws(self):
        # A has 3 corners at 1px; B has 1 at 9px; C contributes none.
        # The pooled contrast is 3px, not the equal-session mean 5px.
        bootstrap = SimpleNamespace(ids=['a', 'b', 'c'], inverse=np.array([0, 1, 2]),
            units=np.array(['A', 'B', 'C']), weights=np.array([[1., 1., 1.], [2., 0., 1.],
                [0., 2., 1.], [0., 0., 3.]]), draw_sha256='test', resamples=4)
        result = S.pooled_contrast([np.array([1., 1., 1.]), np.array([9.]), np.array([])], bootstrap, 'px')
        self.assertEqual(result['mean_paired_difference'], 3.)
        self.assertEqual(result['auxiliary_session_equal_weight_mean'], 5.)
        self.assertEqual(result['common_eligible_observations'], 4)
        self.assertEqual(result['common_eligible_frames'], 2)
        self.assertEqual(result['eligible_sessions'], 2)
        self.assertEqual(result['empty_resamples'], 1)
        self.assertEqual(result['valid_resamples'], 3)
        np.testing.assert_allclose(result['CI95'], np.quantile([3., 1., 9.], [.025, .975]))

    def test_existing_thirteen_session_draws(self):
        b = S.SharedBootstrap([f'f{i}' for i in range(13)], [f's{i:02}' for i in range(13)])
        expected = np.random.default_rng(20260917).multinomial(13, np.full(13, 1 / 13), size=10000).astype('uint16')
        self.assertEqual(b.draw_sha256, hashlib.sha256(expected.tobytes()).hexdigest())
        self.assertEqual(b.draw_sha256, '63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5')
        np.testing.assert_array_equal(b.weights.sum(axis=1), np.full(10000, 13))

    def test_sample_spread_and_centimeter_conversion(self):
        meters = S.distribution([.01, .02, .03], 'm')
        cm = S.distribution([1., 2., 3.], 'cm')
        self.assertEqual(cm['sample_variance'], 1.)
        self.assertEqual(cm['sample_std'], 1.)
        self.assertAlmostEqual(cm['sample_variance'], 10000 * meters['sample_variance'])
        self.assertAlmostEqual(cm['sample_std'], 100 * meters['sample_std'])
        self.assertIsNone(S.distribution([2.])['sample_std'])

    def test_damage_boundary_inclusivity(self):
        result = S.corner_damage([4., 5., 21., 21., 20., 3.], [10., 11., 5., 10., 5., 11.])
        self.assertEqual(result['good5_to_bad10'], 1)  # <5 and >10 are strict.
        self.assertEqual(result['bad20_to_good5'], 1)  # >20 then <=5.
        self.assertEqual(result['bad20_to_good10'], 2)
        self.assertEqual(result['worsened_by_over_1px'], 3)

    def test_visibility_pair_requires_both_supported_predictions(self):
        def row(errors, observed):
            valid = [True, True, True, False, False, False, False, False]
            return dict(id='frame', canonical_observed=observed,
                corner=dict(evaluable=True, canonical_valid=valid,
                    canonical_errors=errors + [None] * 5,
                    observed_errors=[errors[k] for k in range(3) if observed[k]]))
        before = row([2., 3., 4.], [True, False, True, False, False, False, False, False])
        after = row([1., 5., 6.], [True, True, False, False, False, False, False, False])
        eligible = {('frame', 0), ('frame', 2)}
        values = S.paired_values([after], [before], 'corner_px', eligible)
        np.testing.assert_array_equal(values[0], [-1.])
        # Corner2 has a valid GT and label but lacks the new observed prediction.
        empty = S.paired_values([after], [before], 'corner_px', {('frame', 2)})
        self.assertEqual(len(empty[0]), 0)


if __name__ == '__main__':
    unittest.main()
