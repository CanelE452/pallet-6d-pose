"""Constructed H1 all-failure fixtures only; no actual experiment data read."""
from copy import deepcopy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import all_failure_analysis as A
from . import build_report as B

ARM, OTHER = A.H1_ARMS[:2]
POPS = dict(PRIMARY192=('TEST', 'REMAINDER'), TEST49=('TEST',),
    REMAINDER143=('REMAINDER',), CAL46=('CAL',))
COUNTS = dict(PRIMARY192=2, TEST49=1, REMAINDER143=1, CAL46=1)


def fixture(fid='a', partition='TEST', *, arms=(ARM, OTHER), all_failure=True):
    names = tuple(arms) + A.REFERENCES
    row = dict(id=fid, partition=partition, session='constructed', cluster_id='toy',
        poses={name: dict(available=True) for name in names},
        metrics={name: dict(pose_available=True, translation_cm=10., rotation_deg=10., success=False)
            for name in names},
        status={name: dict(fallback=False, status='TOY') for name in names},
        action_sources={name: dict(role='OTHER', aliases=[]) for name in names})
    stage = dict(id=fid, partition=partition, arms={name: dict(final_success=False,
        selected_candidate_id='toy-selected', after_constraints=dict(candidate_count=2,
            available_n=2, has_success=not all_failure)) for name in arms})
    return row, stage


def data():
    pairs = [fixture('a', 'TEST'), fixture('b', 'REMAINDER'), fixture('c', 'CAL')]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def analyze(scored, stages):
    return A.analyze(scored, stages, arms=(ARM, OTHER), populations=POPS, expected_n=COUNTS)


def summary(rows, *, population='PRIMARY192', arm=ARM, comparator='N0'):
    return next(row for row in rows if (row['population'], row['arm'], row['comparator']) ==
        (population, arm, comparator))


class AnalysisTests(unittest.TestCase):
    def test_all_fifteen_arms_three_references_are_registered(self):
        self.assertEqual(len(A.H1_ARMS), 15)
        self.assertEqual(A.REFERENCES, ('N0', 'E0', 'E0_REGENERATED'))

    def test_binary_all_fail_can_worsen_T_and_improve_R(self):
        scored, stages = data()
        scored[0]['metrics'][ARM].update(translation_cm=30., rotation_deg=8.)
        details, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['all_failure_n'], 2)
        self.assertEqual(value['T_worse_n'], 1)
        self.assertEqual(value['R_improved_n'], 1)
        self.assertEqual(value['worseEither_n'], 1)
        self.assertEqual(value['improvedEither_n'], 1)
        self.assertEqual(value['T_delta'], dict(n=2, sum=20., mean=10., min=0., max=20., aggregate_nonfinite=False))
        self.assertEqual(value['R_delta']['sum'], -2.)
        self.assertFalse(next(row for row in details if row['id']=='a')['selected_success'])

    def test_pair_arm_uses_own_eligible_actions(self):
        scored, stages = data()
        stages[0]['arms'][OTHER]['after_constraints']['has_success'] = True
        details, result = analyze(scored, stages)
        self.assertNotIn('a', result['all_failure_ids_by_arm'][OTHER])
        self.assertIn('a', result['all_failure_ids_by_arm'][ARM])
        self.assertEqual(len(details), 15)  # 5 eligible arm/frame identities × 3 refs

    def test_populations_not_duplicated_in_detail_csv(self):
        details, result = analyze(*data())
        self.assertEqual(len(details), 3*2*3)
        self.assertEqual(result['summary_rows'], 4*2*3)
        self.assertEqual(summary(result['rows'])['population_denominator'], 2)
        self.assertEqual(summary(result['rows'], population='TEST49')['all_failure_n'], 1)
        self.assertEqual(summary(result['rows'], population='CAL46')['all_failure_n'], 1)

    def test_missing_actual_pose_excluded_even_with_finite_saved_metrics(self):
        scored, stages = data()
        del scored[0]['poses'][ARM]
        details, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['selected_actual_pose_present_false_n'], 1)
        self.assertEqual(value['both_axes_comparable_n'], 1)
        self.assertEqual(value['T_selected_exclusion_reasons'], {'missing_actual_pose': 1})
        self.assertIsNone(next(r for r in details if r['id']=='a')['T_delta'])

    def test_unavailable_metric_null_and_nonfinite_not_zeroed(self):
        scored, stages = data()
        scored[0]['metrics'][ARM].update(translation_cm=None, rotation_deg=float('nan'), pose_available=False)
        scored[1]['metrics']['N0']['translation_cm'] = float('inf')
        details, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['T_comparable_n'], 0)
        self.assertIsNone(value['T_delta']['sum'])
        self.assertIsNone(value['T_delta']['mean'])
        self.assertEqual(value['T_comparator_exclusion_reasons'], {'nonfinite_error': 1})
        self.assertEqual(value['R_selected_exclusion_reasons'], {'metric_pose_unavailable': 1, 'nonfinite_error': 1})
        self.assertEqual(next(r for r in details if r['id']=='a')['selected_rotation_deg'], 'NaN')
        json.dumps(result, allow_nan=False)

    def test_strict_direction_uses_no_extra_threshold(self):
        scored, stages = data()
        scored[0]['metrics'][ARM]['translation_cm'] = 10.000000000000002
        _, result = analyze(scored, stages)
        self.assertEqual(summary(result['rows'])['T_worse_n'], 1)

    def test_missing_comparator_metric_explicit_excluded(self):
        scored, stages = data()
        del scored[0]['metrics']['N0']
        _, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['comparator_metric_present_false_n'], 1)
        self.assertEqual(value['T_comparable_n'], 1)

    def test_either_does_not_treat_one_missing_axis_as_complete_pair(self):
        scored, stages = data()
        scored[0]['metrics'][ARM].update(translation_cm=None, rotation_deg=20.)
        details, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['R_worse_n'], 1)
        self.assertEqual(value['both_axes_comparable_n'], 1)
        self.assertEqual(value['worseEither_n'], 0)
        self.assertIsNone(next(row for row in details if row['id']=='a')['worse_either'])

    def test_actual_source_fallback_and_unknown_preserved(self):
        scored, stages = data()
        scored[0]['status'][ARM]['fallback'] = True
        scored[1]['status'][ARM].pop('fallback')
        scored[0]['action_sources'][ARM] = dict(role='N0', aliases=['N0', 'E0_REGENERATED'])
        details, result = analyze(scored, stages)
        value = summary(result['rows'])
        self.assertEqual(value['selected_source_counts'], {'N0':1, 'OTHER':1})
        self.assertEqual(value['selected_fallback_counts'], {'true':1, 'unknown':1})
        self.assertEqual(json.loads(next(row for row in details if row['id']=='a')['selected_aliases_json']),
            ['N0', 'E0_REGENERATED'])

    def test_no_all_failure_keeps_zero_count_summary_with_null_moments(self):
        scored, stages = data()
        for row in stages:
            for stage in row['arms'].values(): stage['after_constraints']['has_success'] = True
        details, result = analyze(scored, stages)
        self.assertEqual(details, [])
        self.assertEqual(len(result['rows']), 24)
        self.assertEqual(summary(result['rows'])['all_failure_n'], 0)
        self.assertIsNone(summary(result['rows'])['T_delta']['max'])

    def test_inconsistent_success_or_missing_stage_flag_rejected(self):
        scored, stages = data()
        scored[0]['metrics'][ARM]['success'] = True
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            analyze(scored, stages)
        scored[0]['metrics'][ARM]['success'] = False
        stages[0]['arms'][ARM]['after_constraints']['has_success'] = None
        with self.assertRaisesRegex(ValueError, 'exact eligible'):
            analyze(scored, stages)

    def test_missing_duplicate_or_partition_changed_frames_rejected(self):
        scored, stages = data()
        for change in ('missing', 'duplicate', 'partition'):
            a, b = deepcopy(scored), deepcopy(stages)
            if change == 'missing': b.pop()
            elif change == 'duplicate': a.append(a[0])
            else: b[0]['partition'] = 'CAL'
            with self.assertRaises(ValueError): analyze(a, b)

    def test_denominator_drift_rejected(self):
        scored, stages = data()
        scored[0]['partition'] = stages[0]['partition'] = 'CAL'
        with self.assertRaisesRegex(ValueError, 'denominators'):
            analyze(scored, stages)

    def test_guard_runs_before_any_file_access(self):
        with patch.object(A, 'sha', side_effect=AssertionError('must not read')):
            with self.assertRaises(PermissionError): A.run('/absent', analysis_GO=False)


class SealedMetadataFixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.ev = self.root/'evaluation_v1'; self.ev.mkdir()
        self.output = self.root/'h1_all_failure_v1'
        self.scored, self.stages = [], []
        for i, part in enumerate(['TEST']*49 + ['REMAINDER']*143 + ['CAL']*46):
            row, stage = fixture('constructed_' + str(i), part, arms=A.H1_ARMS, all_failure=False)
            self.scored.append(row); self.stages.append(stage)
        self.stages[0]['arms'][ARM]['after_constraints']['has_success'] = False
        for name, rows in zip(A.JOURNALS, (self.scored, self.stages)):
            with gzip.open(self.ev/name, 'wt') as stream:
                for row in rows: stream.write(json.dumps(row) + '\n')
        A.write(self.root/'PLAN_LOCK.json', {'constructed': True})
        A.write(self.root/'PREDICTION_SEAL.json', dict(complete=True, before_scoring=True,
            reference_inputs=False, arms=list(A.ALL_ARMS), rows=238,
            ordered_ids=[row['id'] for row in self.scored], predictions_sha256='fake_saved_not_opened',
            plan_lock_sha256=A.sha(self.root/'PLAN_LOCK.json')))
        ps = A.sha(self.root/'PREDICTION_SEAL.json')
        A.write(self.root/'ALL_OUTPUTS_SCORING_GO.json', dict(all_registered_predictions_sealed=True,
            prediction_seal_sha256=ps, plan_sha256=A.sha(self.root/'PLAN_LOCK.json')))
        A.write(self.ev/'EVALUATION_SEAL.json', dict(schema='BOUNDARY_ACTION_EVALUATION_SEAL_V1',
            complete=True, all_failures_retained=True, prediction_seal_sha256=ps,
            input_predictions_sha256='fake_saved_not_opened', registry_sha256=A.sha(Path(A.__file__).with_name('registry.py')),
            artifacts={name:A.sha(self.ev/name) for name in A.JOURNALS}))

    def test_metadata_only_end_to_end_and_seal(self):
        posthoc = self.root/'posthoc_v1'; posthoc.mkdir(); marker = posthoc/'OLD_SEAL'; marker.write_bytes(b'preserve')
        seal = A.run(self.root, output=self.output, analysis_GO=True)
        self.assertEqual(seal['summary_rows'], 180)
        self.assertEqual(seal['detail_rows'], 3)
        self.assertEqual(seal['frame_n'], 238)
        self.assertEqual(marker.read_bytes(), b'preserve')
        self.assertFalse(seal['reference_payload_opened'])
        self.assertEqual(seal['new_pose_metric_evaluations'], 0)
        with (self.output/'H1_ALL_FAILURE_ROWS.csv').open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        for name, digest in seal['artifacts'].items(): self.assertEqual(A.sha(self.output/name), digest)
        with self.assertRaises(FileExistsError): A.run(self.root, output=self.output, analysis_GO=True)

    def test_journal_hash_tampering_rejected(self):
        with (self.ev/A.JOURNALS[1]).open('ab') as stream: stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            A.run(self.root, analysis_GO=True)
        self.assertFalse(self.output.exists())

    def test_all_output_gate_binding_tampering_rejected(self):
        path = self.root/'ALL_OUTPUTS_SCORING_GO.json'
        value = A.read(path); value['plan_sha256'] = 'wrong'; A.write(path, value)
        with self.assertRaisesRegex(ValueError, 'plan binding'):
            A.run(self.root, analysis_GO=True)

    def test_cannot_add_files_under_original_evaluation_or_posthoc_tree(self):
        for protected in (self.ev, self.root/'posthoc_v1'):
            with self.assertRaisesRegex(ValueError, 'separate H1'):
                A.run(self.root, output=protected/'newchild', analysis_GO=True)
            self.assertFalse((protected/'newchild').exists())

    def test_report_links_all_rows_and_verifies_separate_analysis_seal(self):
        A.run(self.root, analysis_GO=True)
        inputs = B.Inputs()
        summary = inputs.load('h1_all_failure_summary', self.output/'H1_ALL_FAILURE_SUMMARY.json')
        rows = inputs.load('h1_all_failure_rows', self.output/'H1_ALL_FAILURE_ROWS.csv', kind='csv')
        seal = inputs.load('h1_all_failure_seal', self.output/'ANALYSIS_SEAL.json')
        self.assertTrue(B.verify_h1_failure_report_inputs(inputs, summary, rows, seal, self.ev/'EVALUATION_SEAL.json'))
        tables = '\n'.join(B.h1_all_failure_tables(summary))
        for arm in A.H1_ARMS: self.assertIn(arm, tables)
        for name in A.REFERENCES + tuple(A.POPULATIONS): self.assertIn(name, tables)
        self.assertIn('H1_ALL_FAILURE_ROWS.csv', tables)
        self.assertIn('H1_ALL_FAILURE_ANALYSIS_SEAL.json', tables)
        seal['artifacts']['H1_ALL_FAILURE_ROWS.csv'] = 'tampered'
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            B.verify_h1_failure_report_inputs(inputs, summary, rows, seal, self.ev/'EVALUATION_SEAL.json')

    def test_report_missing_analysis_not_silent_zero(self):
        self.assertFalse(B.verify_h1_failure_report_inputs(B.Inputs(), {}, [], {}, self.ev/'EVALUATION_SEAL.json'))
        tables = '\n'.join(B.h1_all_failure_tables({}))
        self.assertIn('N/A', tables)
        self.assertIn('분석 완료나 연속손상0으로 기록하지 않습니다', tables)

    def test_report_independent_link_preserves_expression_qualification(self):
        _, value = analyze(*data())
        independent = dict(status='PASS_WITH_DOCUMENTED_REPRESENTATION_DIFFERENCES',
            failures=0, stage_membership_checks=6, summary_rows=24, detail_rows=18,
            detail_scalar_checks=54, representation_differences=3)
        tables = '\n'.join(B.h1_all_failure_tables(value, independent))
        self.assertIn('INDEPENDENT_H1_FAILURE_CHECK.json', tables)
        self.assertIn('PASS_WITH_DOCUMENTED_REPRESENTATION_DIFFERENCES', tables)
        self.assertIn('수치 차이 3건은 보존', tables)
        self.assertIn('표를 stable 값으로 고치지 않는다', tables)


if __name__ == '__main__':
    unittest.main()
