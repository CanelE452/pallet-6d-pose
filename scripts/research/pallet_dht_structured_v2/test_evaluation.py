"""Generated metric/rule fixtures; no model forward or actual result access."""
import unittest

import numpy as np

from scripts.research.pallet_dht_structured_v2.evaluation import (
    select_indices, corner_metrics, choose_margin, rule_metrics,
    assess_advancement, clean_constraints,
)


def protocol():
    return dict(calibration=dict(margin_grid=[0., .05, .1, .25, .5, 1., 'identity_only'],
        state_weights=dict(clean=.5, point_c4=.2, point_and_line_c4=.2, local_deformation=.1),
        tie_tolerance=1e-12),
        synthetic_advancement=dict(c4_corruption_mean_error_reduction_min_fraction=.5))


def fixture():
    c = np.broadcast_to([3., 1., 2.], (4, 2, 3)).copy()
    e = np.full((4, 2, 3, 8), 30.)
    e[0, :, 0] = 8.
    e[0, :, 1] = 6.
    e[1:3, :, 0] = 100.
    e[1:3, :, 1] = 40.
    e[3, :, 0] = 25.
    e[3, :, 1] = 20.
    return dict(costs=c, corner_errors_px=e, candidate_valid=np.ones_like(c, bool),
                loss_valid=np.ones((2, 8), bool), diagonal=np.array([100., 200.]))


class TestEvaluation(unittest.TestCase):
    def test_gt_free_cost_rule_strict_margin_and_identity_ties(self):
        c = np.array([[1., 1., -3.], [1., .5, 4.]])
        valid = np.array([[True, True, False], [True, True, True]])
        np.testing.assert_array_equal(select_indices(c, valid, 0.), [0, 1])
        np.testing.assert_array_equal(select_indices(c, valid, .5), [0, 0])
        np.testing.assert_array_equal(select_indices(c, valid, 'identity_only'), [0, 0])

    def test_mask_and_diagonal_normalized_pooled_corner_denominator(self):
        e = np.array([[3., 4., 1000.], [10., 1000., 1000.]])
        mask = np.array([[True, True, False], [True, False, False]])
        metric = corner_metrics(e, mask, [10., 100.], e)
        self.assertEqual(metric['n_supervised_corners'], 3)
        self.assertAlmostEqual(metric['mean_px'], 17./3)
        self.assertAlmostEqual(metric['mean_diagonal_normalized'], (.3+.4+.1)/3)
        self.assertEqual(metric['median_px'], 4.)

    def test_good_crossing_uses_good_corner_denominator_and_strict_gt10(self):
        base = np.array([[10., 9., 50., 1.]])
        predicted = np.array([[10., 10.001, 60., 2.]])
        metric = corner_metrics(predicted, np.ones_like(base, bool), [100.], base)
        self.assertEqual(metric['baseline_good_le10_count'], 3)
        self.assertEqual(metric['good_to_bad_gt10_count'], 1)
        self.assertAlmostEqual(metric['good_to_bad_fraction'], 1/3)

    def test_synthetic_selection_uses_registered_state_weights_and_larger_margin_tie(self):
        data = fixture()
        selected = choose_margin(data, protocol())
        self.assertEqual(selected['margin'], 1.)
        expected = .5*(.06+.03)/2 + .4*(.4+.2)/2 + .1*(.2+.1)/2
        self.assertAlmostEqual(selected['objective'], expected)
        self.assertTrue(all(selected['constraints'].values()))

    def test_identity_only_wins_exact_objective_ties(self):
        data = fixture()
        data['corner_errors_px'][:] = 5.
        self.assertEqual(choose_margin(data, protocol())['margin'], 'identity_only')

    def test_clean_damage_blocks_gain_on_corrupted_states(self):
        data = fixture()
        data['corner_errors_px'][0, :, 1] = 11.
        selected = choose_margin(data, protocol())
        self.assertEqual(selected['margin'], 'identity_only')
        self.assertFalse(selected['grid'][0]['eligible'])

    def test_invalid_lower_cost_candidate_never_selected_even_if_gt_perfect(self):
        data = fixture()
        data['costs'][:, :, 2] = -100.
        data['corner_errors_px'][:, :, 2] = 0.
        data['candidate_valid'][:, :, 2] = False
        selected = choose_margin(data, protocol())
        self.assertEqual(selected['margin'], 1.)
        self.assertTrue(all(i == 1 for i in selected['metrics']['clean']['selected_indices']))

    def test_valid_GT_perfect_candidate_does_not_override_learned_cost(self):
        data = fixture()
        data['corner_errors_px'][:, :, 2] = 0.  # Valid GT-best candidate has higher learned cost.
        measured = rule_metrics(data, 0.)
        self.assertEqual(measured['clean']['selected_indices'], [1, 1])
        self.assertEqual(measured['clean']['mean_px'], 6.)
        self.assertGreater(measured['clean']['mean_px'], data['corner_errors_px'][0, :, 2].mean())

    def test_both_C4_states_must_improve_half_individually(self):
        data = fixture()
        metrics, baseline = rule_metrics(data, 1.), rule_metrics(data, 'identity_only')
        result = assess_advancement(metrics, baseline, 1., protocol())
        self.assertTrue(result['advance'])
        data['corner_errors_px'][2, :, 1] = 51.
        failed = assess_advancement(rule_metrics(data, 1.), baseline, 1., protocol())
        self.assertFalse(failed['advance'])
        self.assertTrue(failed['criteria']['point_c4_reduction_at_least_half'])
        self.assertFalse(failed['criteria']['point_and_line_c4_reduction_at_least_half'])

    def test_identity_only_is_not_advancement(self):
        data = fixture()
        baseline = rule_metrics(data, 'identity_only')
        self.assertFalse(assess_advancement(baseline, baseline, 'identity_only', protocol())['advance'])

    def test_clean_p90_is_independent_from_mean_constraint(self):
        base = dict(median_px=2., mean_px=10., p90_px=20.)
        x = dict(median_px=1., mean_px=9., p90_px=21., good_to_bad_fraction=0.)
        checks = clean_constraints(x, base)
        self.assertTrue(checks['mean_no_worse'])
        self.assertFalse(checks['p90_no_worse'])


if __name__ == '__main__':
    unittest.main()
