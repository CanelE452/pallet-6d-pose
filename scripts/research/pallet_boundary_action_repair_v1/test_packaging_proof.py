"""Small metadata-only proof/snapshot fixtures; no real results or image viewing."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from . import build_report as B
from . import requirement_evidence as E


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')


class PackagingProofTests(unittest.TestCase):
    def test_missing_attempt_receipts_preserve_unknown_counts(self):
        value = B.execution_attempt_summary({}, {}, {}, {})
        for key in ('root_timed_job_n', 'root_nonzero_exit_n', 'D1_unsealed_attempt_n',
                    'D1_partial_operations', 'D1_partial_CPU_seconds', 'failed_fit_n',
                    'pending_arm_n', 'L_D0_3fit_whole_process_wall_seconds'):
            self.assertIsNone(value[key], key)

    def test_nonzero_technical_exits_are_separate_from_successful_fits(self):
        cost = dict(root_analysis_batches=[dict(batch_id='OK', exit_code=0),
            dict(batch_id='IMPORT_FAIL', exit_code=1), dict(batch_id='RENDER_FAIL', exit_code=2),
            dict(batch_id='UNKNOWN', exit_code=None)])
        interruption = dict(status='INTERRUPTED_EXTERNAL_CAUSE_UNKNOWN',
            observed_tool_session_exit_code=143, concrete_termination_cause='UNKNOWN',
            unsealed_attempts=[{}, {}, {}, {}], interrupted_partial_operations=None,
            interrupted_partial_CPU_seconds=None, worker_exit_codes=None)
        value = B.execution_attempt_summary(cost, interruption,
            dict(failed_training_attempts=[]), dict(pending_arms={}))
        self.assertEqual(value['root_nonzero_exit_n'], 2)
        self.assertEqual(value['root_nonzero_exit_ids'], ['IMPORT_FAIL', 'RENDER_FAIL'])
        self.assertEqual(value['root_unknown_exit_n'], 1)
        self.assertEqual(value['failed_fit_n'], 0)
        self.assertEqual(value['pending_arm_n'], 0)
        self.assertEqual(value['D1_observed_exit_code'], 143)
        self.assertEqual(value['D1_unsealed_attempt_n'], 4)
        self.assertIsNone(value['D1_partial_operations'])
        self.assertIsNone(value['D1_worker_exit_codes'])

    def test_training_batch_timer_is_not_nested_fit_timer_sum(self):
        lcost = dict(distinct_execution_batches=[
            dict(execution_batch_id='L_D1_TRAINING15', wall_seconds=176.11),
            dict(execution_batch_id='L_D0_H1_TRAINING3', wall_seconds=None,
                known_nested_fit_wall_seconds=11.14)])
        value = B.execution_attempt_summary({}, {}, lcost, {})
        self.assertEqual(value['L_D1_15fit_batch_wall_seconds'], 176.11)
        self.assertIsNone(value['L_D0_3fit_whole_process_wall_seconds'])

    def test_result_lead_uses_fixed_representative_without_best_seed(self):
        name = 'D1_H1_MLP_s' + str(B.REPRESENTATIVE_SEED)
        summaries = dict(PRIMARY192={name: dict(successes=120, denominator=192, translation_cm=dict(mean=13.1)),
            'D1_H1_MLP_s20261020': dict(successes=999),
            'E0': dict(successes=114), 'N0': dict(translation_cm=dict(mean=11.3)),
            'B_FINITE_FULL': dict(successes=97), 'B_FINITE_LOCAL': dict(successes=109)})
        pairs = dict(PRIMARY192={'N0__TO__'+name: dict(counts=dict(RECOVERY=21, DAMAGE=9),
            deltas=dict(translation_cm=dict(mean=1.8)))})
        lead = B.descriptive_result_lead(summaries, pairs)
        for token in ('120/192', '114장', '회복 21장', '손상 9장', '+1.800cm', '97/109장'):
            self.assertIn(token, lead)
        self.assertNotIn('999', lead)
        self.assertIn('운영 기본값 교체는 보류', lead)

    def test_CI_tables_scale_only_success_rate_to_percentage_points(self):
        target = 'D1_H1_MLP_s' + str(B.REPRESENTATIVE_SEED)
        def metric(mean, interval, scale_n=192):
            return dict(mean=mean, common_available_n=scale_n,
                recording_cluster_CI=dict(CI95=interval, cluster_count=6),
                named_session_cluster_sensitivity_CI=dict(CI95=[-0.5,0.5], cluster_count=8))
        paired = dict(PRIMARY192={'E0__TO__'+target: dict(deltas=dict(
            success_rate=metric(.03125, [-.084746,.070588]),
            translation_cm=metric(-2.0395,[-3.2541,.70836]),
            rotation_deg=metric(-1.4523,[-3.7227,7.3033])))})
        sections = B.primary_confidence_tables(paired)
        text = '\n'.join(sections)
        self.assertIn('3.125 | [-8.475, 7.059]', text)
        self.assertIn('-2.039 | [-3.254, 0.708]', text)
        self.assertIn('-1.452 | [-3.723, 7.303]', text)
        self.assertIn('192/192/192 | 6/6/6', text)
        self.assertIn('192/192/192 | 8/8/8', text)
        self.assertEqual(text.count('| E0 |'), 4)  # two fixed targets × two units
        self.assertNotIn('s20261020', text)

    def test_missing_CIs_are_NA_and_all_ten_pairs_retained(self):
        sections = B.primary_confidence_tables({})
        text = '\n'.join(sections)
        self.assertIn('N/A', text)
        self.assertNotIn('[0.000, 0.000]', text)
        for comparator in B.COMPARATORS:
            self.assertEqual(text.count('| '+comparator+' |'), 4)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run, self.public, self.source = (self.root / n for n in ('run', 'public', 'source'))
        for path in (self.run, self.public, self.source): path.mkdir()
        self.image = self.public / 'image.png'
        self.image.write_bytes(b'\x89PNG\r\n\x1a\nFAKE_TEST_BYTES_NOT_RENDERED')
        self.review = dict(reviewer_type='model', actual_review_method='fake metadata fixture only',
            pixel_review_complete=True, findings=[], corrections=[], recheck_status='NOT_REQUIRED',
            reviewed_files=[dict(sha256=E.sha(self.image), actual_pixels_viewed=True, status='PASS')])

    def review_check(self, value):
        write(self.run / 'VISUAL_REVIEW.json', value)
        return E.Receipts(self.run, self.public, self.source).check(dict(claim='pixel',
            receipt='VISUAL_REVIEW.json', pointer='/', op='reviewed_image', file='public:image.png'))

    def snapshot(self):
        folder = self.run / 'publication_snapshots/result_A'
        manifest = folder / 'PUBLICATION_MANIFEST.json'
        write(manifest, dict(schema='FAKE_MANIFEST', immutable_result_A=True))
        check = folder / 'PUBLICATION_CHECK.json'
        write(check, dict(schema='OFFLINE_PUBLICATION_CHECK_V1', status='PASS', manifest_sha256=B.digest(manifest)))
        publication = dict(status='PUSHED_MAIN_VERIFIED', result_commit='a'*40,
            remote_main_contains_result_commit=True, remote_artifacts_verified=True,
            offline_check_snapshot=dict(result_commit='a'*40, status='PASS',
                path=str(check.relative_to(self.run)), sha256=B.digest(check),
                manifest_path=str(manifest.relative_to(self.run)), manifest_sha256=B.digest(manifest)))
        return publication, check, manifest

    def test_explicit_complete_review_passes(self):
        self.assertEqual(self.review_check(self.review)['status'], 'PASS')

    def test_global_viewed_flag_does_not_accept_missing_row_status(self):
        self.review['reviewed_files'][0].pop('status')
        self.assertEqual(self.review_check(self.review)['status'], 'FAIL')

    def test_failed_technical_review_does_not_pass(self):
        self.review['reviewed_files'][0]['status'] = 'FAILED_TECHNICAL'
        self.assertEqual(self.review_check(self.review)['status'], 'FAIL')

    def test_reviewer_and_method_are_required(self):
        for key in ('reviewer_type', 'actual_review_method'):
            value = deepcopy(self.review); value.pop(key)
            self.assertEqual(self.review_check(value)['status'], 'PENDING')

    def test_findings_corrections_and_recheck_are_required(self):
        for key in ('findings', 'corrections', 'recheck_status'):
            value = deepcopy(self.review); value.pop(key)
            self.assertEqual(self.review_check(value)['status'], 'PENDING')

    def test_unresolved_recheck_fails(self):
        self.review['recheck_status'] = 'PENDING'
        self.assertEqual(self.review_check(self.review)['status'], 'FAIL')

    def test_unviewed_matching_hash_does_not_pass(self):
        self.review['pixel_review_complete'] = False
        self.review['reviewed_files'][0]['actual_pixels_viewed'] = False
        self.assertEqual(self.review_check(self.review)['status'], 'FAIL')

    def test_not_applicable_cannot_waive_core_requirement(self):
        na = E.default_proofs()['PUB_09']['not_applicable_if']
        with self.assertRaisesRegex(ValueError, 'core requirement'):
            E.validate_not_applicable('B_01', na, 'PENDING')

    def test_not_applicable_cannot_hide_failure(self):
        with self.assertRaisesRegex(ValueError, 'failed evidence'):
            E.validate_not_applicable('PUB_09', E.default_proofs()['PUB_09']['not_applicable_if'], 'FAIL')

    def test_not_applicable_requires_verified_main_side_conditions(self):
        na = deepcopy(E.default_proofs()['PUB_09']['not_applicable_if'])
        E.validate_not_applicable('PUB_09', na, 'PENDING')
        na['checks'].pop()
        with self.assertRaisesRegex(ValueError, 'verified-main proof'):
            E.validate_not_applicable('PUB_09', na, 'PENDING')

    def test_honest_axis_states_and_59_inventory_remain_exact(self):
        defaults = E.default_proofs()
        execution = defaults['AXIS_EXECUTION_COMPLETENESS']['checks'][0]['expected']
        visual = defaults['AXIS_VISUAL_AND_PUBLICATION']['checks'][0]['expected']
        self.assertIn('PASS_COMPLETED_OUTPUTS_WITH_DOCUMENTED_TECHNICAL_LOG_GAPS', execution)
        self.assertEqual(visual, ['PUSHED_MAIN_VERIFIED_WITH_MODEL_PIXEL_REVIEW'])
        self.assertNotIn('PENDING', execution + visual)
        self.assertEqual(len(E.catalog()), 59)
        self.assertIn('not_applicable_if', E.template()['requirements']['PUB_09'])

    def test_default_publication_proof_binds_fixed_A_not_current_pack(self):
        defaults = E.default_proofs()
        for ident in ('PUB_03', 'PUB_04', 'PUB_05'):
            for rule in defaults[ident]['checks']:
                self.assertNotIn(rule['receipt'], ('public:PUBLICATION_CHECK.json', 'public:PUBLICATION_MANIFEST.json'))
        targets = [r.get('file') for r in defaults['PUB_04']['checks']]
        self.assertIn('publication_snapshots/result_A/PUBLICATION_MANIFEST.json', targets)

    def test_current_offline_check_is_not_read_without_external_snapshot(self):
        (self.public / 'PUBLICATION_CHECK.json').write_text('INVALID_JSON_MUST_NOT_BE_READ')
        inputs = B.Inputs()
        self.assertEqual(B.publication_snapshot(inputs, {}, self.run, self.public), {})
        self.assertEqual(inputs.bindings, [])

    def test_metadata_repacking_cannot_change_A_snapshot_bindings(self):
        publication, _, _ = self.snapshot()
        first_inputs = B.Inputs()
        first = B.publication_snapshot(first_inputs, publication, self.run, self.public)
        (self.public / 'PUBLICATION_CHECK.json').write_text('CURRENT_METADATA_CHECK_CHANGED')
        (self.public / 'PUBLICATION_MANIFEST.json').write_text('CURRENT_METADATA_MANIFEST_CHANGED')
        second_inputs = B.Inputs()
        second = B.publication_snapshot(second_inputs, publication, self.run, self.public)
        self.assertEqual(first, second)
        self.assertEqual(first_inputs.bindings, second_inputs.bindings)
        self.assertTrue(all(x['input'].startswith('result_commit_A_') for x in second_inputs.bindings))

    def test_changed_A_snapshot_manifest_is_rejected(self):
        publication, _, manifest = self.snapshot()
        write(manifest, dict(changed=True))
        with self.assertRaisesRegex(ValueError, 'binding changed'):
            B.publication_snapshot(B.Inputs(), publication, self.run, self.public)

    def test_wrong_result_commit_snapshot_is_rejected(self):
        publication, _, _ = self.snapshot()
        publication['offline_check_snapshot']['result_commit'] = 'b'*40
        with self.assertRaisesRegex(ValueError, 'verified result commit A'):
            B.publication_snapshot(B.Inputs(), publication, self.run, self.public)

    def test_current_check_override_is_rejected(self):
        publication, _, _ = self.snapshot()
        with self.assertRaisesRegex(ValueError, 'exact externally bound'):
            B.publication_snapshot(B.Inputs(), publication, self.run, self.public, self.public / 'PUBLICATION_CHECK.json')

    def test_final_matrix_or_map_cannot_hash_or_attest_itself(self):
        store = E.Receipts(self.run, self.public, self.source)
        output, proofmap = self.run / 'FINAL.json', self.run / 'PROOF.json'
        for rule in (dict(receipt='FINAL.json'), dict(receipt='PROOF.json'),
                     dict(receipt='other.json', file='FINAL.json')):
            with self.assertRaisesRegex(ValueError, 'cannot attest or hash itself'):
                E.reject_self_proof({'X': dict(checks=[rule])}, store, output, proofmap)


if __name__ == '__main__':
    unittest.main()
