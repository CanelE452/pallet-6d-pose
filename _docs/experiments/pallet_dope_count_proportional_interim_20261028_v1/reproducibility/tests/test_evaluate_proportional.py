"""Pure scientific fixtures. No RGB, reference annotation or model is opened."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1] / 'code/evaluate_proportional.py'
spec = importlib.util.spec_from_file_location('test_proportional_evaluator_module', SOURCE)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


class EvaluationFixtures(unittest.TestCase):
    @staticmethod
    def training_receipt(real_only_updates=0):
        history = [dict(update=i + 1, n_synthetic=0 if i < real_only_updates else 8,
            n_real=16 if i < real_only_updates else 8) for i in range(828)]
        gradients = {'vgg.0.weight': 828, 'm6_2.weight': 828,
            'm6_1.weight': 828 - real_only_updates, 'm6_1.bias': 828 - real_only_updates}
        initial = {name: 828 for name in gradients}
        final = {name: initial[name] + count for name, count in gradients.items()}
        values = sorted(set(final.values()))
        return dict(optimizer_steps=values, stats=dict(complete=True,
            updates_completed=828, updates_required=828, global_optimizer_steps=1656,
            effective_batch_size=16, committed_synthetic_draws=sum(r['n_synthetic'] for r in history),
            committed_real_draws=sum(r['n_real'] for r in history), synthetic_unique_pool=110,
            real_unique_pool=550, parameter_step_validation_passed=True, history=history,
            per_parameter_gradient_updates=gradients, optimizer_parameter_initial_steps=initial,
            optimizer_parameter_steps=final, optimizer_parameter_step_values=values))

    @staticmethod
    def rows(correct=(1, 0), observed=(1, 1), references=(1, 2)):
        return [dict(id=str(i), n_reference=r, n_observed=o, correct5=c,
            correct10=c, correct20=c, manual_n_reference=r, manual_n_observed=o,
            manual_correct5=c, manual_correct10=c, manual_correct20=c)
            for i, (c, o, r) in enumerate(zip(correct, observed, references))]

    def test_seed_model_ids_do_not_collide(self):
        ids = {E.group_id(k, c, s) for k in E.KINDS for c in E.CONDITIONS for s in E.TRAINING_SEEDS}
        self.assertEqual(len(ids), 45)

    def test_all_mixed_batches_have_all_parameter_steps1656(self):
        receipt = self.training_receipt()
        E.validate_training_statistics(receipt, 'WOOD', 'A')
        self.assertEqual(receipt['optimizer_steps'], [1656])

    def test_real_only_batches_skip_only_affinity_adam_counters(self):
        receipt = self.training_receipt(42)
        E.validate_training_statistics(receipt, 'WOOD', 'A')
        self.assertEqual(receipt['optimizer_steps'], [1614, 1656])
        self.assertEqual(receipt['stats']['global_optimizer_steps'], 1656)
        self.assertEqual(receipt['stats']['optimizer_parameter_steps']['m6_2.weight'], 1656)

    def test_falsely_forced_all1656_affinity_counters_are_rejected(self):
        receipt = self.training_receipt(42)
        receipt['stats']['optimizer_parameter_steps']['m6_1.weight'] = 1656
        with self.assertRaises(RuntimeError):
            E.validate_training_statistics(receipt, 'WOOD', 'A')

    def test_unrelated_parameter_counter_cannot_be_decremented(self):
        receipt = self.training_receipt(42)
        receipt['stats']['per_parameter_gradient_updates']['m6_2.weight'] = 827
        receipt['stats']['optimizer_parameter_steps']['m6_2.weight'] = 1655
        receipt['stats']['optimizer_parameter_step_values'] = [1614, 1655, 1656]
        receipt['optimizer_steps'] = [1614, 1655, 1656]
        with self.assertRaises(RuntimeError):
            E.validate_training_statistics(receipt, 'WOOD', 'A')

    def test_partial_global_update_budget_is_rejected(self):
        receipt = self.training_receipt(42)
        receipt['stats']['global_optimizer_steps'] = 1655
        with self.assertRaises(RuntimeError):
            E.validate_training_statistics(receipt, 'WOOD', 'A')

    def test_unregistered_seed_is_rejected(self):
        with self.assertRaises(RuntimeError):
            E.group_id('WOOD', 'C', 1234)

    def test_root_go_requires_all_frozen_protocol_and_test_bindings(self):
        class FakeIO:
            root = E.EXPECTED_ROOT
            def __init__(self):
                self.frozen = {str(self.root / rel): 'a' * 64 for rel in (
                    'FIXED_PLAN.json', 'preflight/INPUTS.json', 'SUBSETS.json',
                    'preflight/TRAINING_GO.json', 'code/evaluate_proportional.py',
                    'tests/test_evaluate_proportional.py')}
                self.go = dict(passed=True, evaluation_authorized=True, canonical_FP32_authorized=True,
                    eval_allowlist_sha256='a' * 64, evaluator_sha256='a' * 64, frozen_files=self.frozen)
            def read(self, path):
                return self.go
            def hash(self, path, expected=None):
                actual = 'a' * 64
                E.require(expected is None or expected == actual, 'fixture frozen input changed')
                return actual
        io = FakeIO()
        self.assertIs(E.verify_go(io), io.go)
        del io.frozen[str(io.root / 'SUBSETS.json')]
        with self.assertRaises(RuntimeError):
            E.verify_go(io)

    def test_root_go_rejects_changed_frozen_file(self):
        class FakeIO:
            root = E.EXPECTED_ROOT
            def read(self, path):
                return dict(passed=True, evaluation_authorized=True, canonical_FP32_authorized=True,
                    eval_allowlist_sha256='a' * 64, evaluator_sha256='a' * 64,
                    frozen_files={str(self.root / rel): 'b' * 64 for rel in (
                        'FIXED_PLAN.json', 'preflight/INPUTS.json', 'SUBSETS.json',
                        'preflight/TRAINING_GO.json', 'code/evaluate_proportional.py',
                        'tests/test_evaluate_proportional.py')})
            def hash(self, path, expected=None):
                E.require(expected is None or expected == 'a' * 64, 'fixture frozen input changed')
                return 'a' * 64
        with self.assertRaises(RuntimeError):
            E.verify_go(FakeIO())

    def test_missing_points_fail_full_denominator_pck(self):
        import numpy as np
        point, draws = E.metric_draws(self.rows(), 'PCK10', np.asarray([[1, 1], [2, 0], [0, 2]]))
        self.assertAlmostEqual(point, 1 / 3)
        np.testing.assert_allclose(draws, [1 / 3, 1, 0])

    def test_coverage_has_reference_denominator(self):
        import numpy as np
        point, draws = E.metric_draws(self.rows(), 'coverage', np.asarray([[1, 1]]))
        self.assertAlmostEqual(point, 2 / 3)
        self.assertAlmostEqual(draws[0], 2 / 3)

    def test_conditional_precision_uses_observed_denominator(self):
        import numpy as np
        point, draws = E.metric_draws(self.rows(), 'conditional_precision10', np.asarray([[1, 1]]))
        self.assertAlmostEqual(point, .5)
        self.assertAlmostEqual(draws[0], .5)

    def test_undefined_draw_makes_entire_ci_na_without_dropping(self):
        import numpy as np
        rows = self.rows(observed=(1, 0))
        a = E.metric_draws(rows, 'conditional_precision10', np.asarray([[1, 1], [0, 2]]))
        result = E.summarize_difference([a], [a])
        self.assertEqual(result['point_difference'], 0)
        self.assertEqual(result['undefined_replicates'], 1)
        self.assertIsNone(result['ci95_low'])
        self.assertIsNone(result['ci95_high'])

    def test_identical_predictions_exact_zero_paired_ci(self):
        import numpy as np
        a = E.metric_draws(self.rows(), 'PCK10', np.asarray([[1, 1], [2, 0], [0, 2]]))
        result = E.summarize_difference([a], [a])
        self.assertEqual((result['point_difference'], result['ci95_low'], result['ci95_high']), (0, 0, 0))
        self.assertEqual(result['decision'], 'includes_zero')

    def test_candidate_minus_comparator_direction(self):
        import numpy as np
        result = E.summarize_difference([(.8, np.asarray([.8, .9, .7]))],
                                        [(.2, np.asarray([.2, .3, .1]))])
        self.assertAlmostEqual(result['point_difference'], .6)
        self.assertEqual(result['decision'], 'candidate_higher')

    def test_seed_mean_averages_ratios_without_pooling(self):
        import numpy as np
        candidate = [(.1, np.asarray([.1, .1])), (.9, np.asarray([.9, .9])), (.5, np.asarray([.5, .5]))]
        comparator = [(0, np.asarray([0., 0.]))] * 3
        result = E.summarize_difference(candidate, comparator)
        self.assertEqual(result['point_difference'], .5)
        self.assertEqual((result['ci95_low'], result['ci95_high']), (.5, .5))

    def test_seed_sample_sd_not_population_sd(self):
        summaries = []
        for seed, value in zip(E.TRAINING_SEEDS, (.2, .4, .6)):
            row = dict(kind='WOOD', condition='C', panel='all_annotated', training_seed=seed,
                       n_DEV=125, n_reference_points=963)
            row.update({m: value for m in (*E.QUALITY, 'conditional_pixel_error_mean', 'conditional_pixel_error_median')})
            summaries.append(row)
        result = E.seed_aggregate(summaries)[0]
        self.assertAlmostEqual(result['PCK10_seed_mean'], .4)
        self.assertAlmostEqual(result['PCK10_training_seed_sample_SD'], .2)
        self.assertEqual(result['training_seed_SD_ddof'], 1)

    def test_missing_training_seed_cannot_silently_be_selected(self):
        with self.assertRaises(RuntimeError):
            E.seed_aggregate([dict(kind='WOOD', condition='C', panel='all_annotated', training_seed=42)])

    def test_manual_uses_its_own_counts(self):
        import numpy as np
        rows = self.rows()
        rows[0]['manual_correct10'] = 0
        point, _ = E.metric_draws(rows, 'PCK10', np.asarray([[1, 1]]), manual=True)
        self.assertEqual(point, 0)

    def test_frozen_metric_missing_mask_and_center_exclusion(self):
        # Only frozen source text is read. No experiment cache or annotation.
        C = E.frozen_common()
        class SourceOnly:
            def hash(self, path, expected=None):
                digest = E.sha(path)
                if expected is not None:
                    E.require(digest == expected, 'frozen source mismatch')
                return digest
        helpers = C.metric_helpers(SourceOnly())
        pred = dict(points=[[0, 0], [8, 0]] + [None] * 6 + [[0, 0]],
                    valid=[True, False] + [False] * 6 + [True])
        obs = helpers['observed'](pred)
        self.assertEqual(set(obs), {0})
        group = dict(group='fixture', kind='WOOD', condition='C', round=2, model='fixture',
            status='available', available=True, checkpoint_sha256='fixture', predictions={'x': pred})
        frame = dict(id='x', image_sha256='fixture', session='fixture')
        refs = [dict(index=0, xy=[3, 4], verified_manual=True),
                dict(index=1, xy=[8, 0], verified_manual=False)]
        row = helpers['frame_2d'](group, frame, refs)
        summary = helpers['summary_2d'](group, [row])
        self.assertEqual(summary['PCK5'], .5)
        self.assertEqual(summary['coverage'], .5)
        self.assertEqual(summary['conditional_precision10'], 1)
        self.assertEqual(summary['conditional_pixel_error_n'], 1)
        self.assertEqual(summary['conditional_pixel_error_mean'], 5)
        self.assertEqual(summary['conditional_pixel_error_std'], 0)

    def test_bootstrap_requires_all_models(self):
        with self.assertRaises(RuntimeError):
            E.bootstrap_reports([], {}, replicates=10)

    def test_complete_three_seed_catalogue_ci_row_counts_and_pairing(self):
        old_expected = E.EXPECTED
        E.EXPECTED = {k: (2, 3) for k in E.KINDS}
        try:
            outputs, controls = [], {}
            for kind in E.KINDS:
                controls[kind] = self.rows()
                for condition in E.CONDITIONS:
                    for seed in E.TRAINING_SEEDS:
                        outputs.append(dict(summary=dict(kind=kind, condition=condition,
                            training_seed=seed), frames=self.rows()))
            per_seed, aggregate, receipts = E.bootstrap_reports(outputs, controls, replicates=30)
            self.assertEqual(len([r for r in per_seed if r['panel'] == 'all_annotated']), 405)
            self.assertEqual(len([r for r in per_seed if r['panel'] == 'GREEN_manual600']), 135)
            self.assertEqual(len([r for r in aggregate if r['panel'] == 'all_annotated']), 135)
            self.assertEqual(len([r for r in aggregate if r['panel'] == 'GREEN_manual600']), 45)
            self.assertEqual(len(receipts), 3)
            self.assertTrue(all(r['point_difference'] == 0 and r['ci95_low'] == 0 and
                r['ci95_high'] == 0 for r in per_seed + aggregate))
            outputs[-1]['frames'] = list(reversed(outputs[-1]['frames']))
            with self.assertRaises(RuntimeError):
                E.bootstrap_reports(outputs, controls, replicates=10)
        finally:
            E.EXPECTED = old_expected


if __name__ == '__main__':
    unittest.main()
