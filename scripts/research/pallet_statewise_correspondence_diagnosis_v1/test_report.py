"""Generated result fixtures only; never read research predictions/references."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import analysis as A
from . import report as R


def generated_scored_rows():
    return [dict(partition='TEST' if i < 49 else 'REMAINDER',
                 metrics={a: dict(translation_cm=1., rotation_deg=1.) for a in ('E0', 'TYPED_SINGLE')})
            for i in range(192)]


def generated_observation_audit():
    return dict(rows=238, totals=dict(visible_archives=1252, complete=836, typed_partial=344,
        legacy_partial=72, e_owner_partial_to_post_full=381, e_owner_typed_to_post_full=310,
        e_owner_legacy_to_post_full=71, N3_to_e_same_channel_physical_owner_changed=304),
        partial_to_post_full_sources=dict(VISIBLE_BACKBONE_FALLBACK=260, B_TARGET_BACKGROUND_LINES=107,
            I_INTERNAL_RGB_JUNCTION=12, B_ACTUAL_BOUNDARY_RECOVERY=2),
        nonidentity_frames=38, nonidentity_vertices=304, semantic_owner_status='SEMANTIC_OWNER_UNVERIFIED',
        extraction_seal_sha256='a'*64)


def write_generated_observation_files(root):
    folder = root/'e_trace_v2'
    folder.mkdir()
    path = folder/'OBSERVATION_TYPE_AND_OWNER_AUDIT.csv'
    path.write_text('id,N3_to_e_same_channel_physical_owner_changed\n'+''.join(
        f'generated-frame-{i},{i < 38}\n' for i in range(238) for _ in range(8)))
    audit = generated_observation_audit()
    A.write(folder/'EXTRACTION_SEAL.json', dict(rows=238,
        ordered_ids=['generated-frame-'+str(i) for i in range(238)], totals=audit['totals'],
        partial_to_post_full_sources=audit['partial_to_post_full_sources'],
        semantic_owner_status=audit['semantic_owner_status'],
        files={path.name: dict(sha256=A.sha(path))}))


def generated_inputs():
    error = A.distribution([1., 2., 101., None])
    flags = {name: dict(true_n=0, false_n=0, unknown_n=192)
             for name in ('accepted', 'fallback', 'solver_rejected', 'technical_failure',
                          'timeout', 'ambiguous_but_accepted')}
    summary = dict(successes_le5=1, denominator=192, errors=dict(translation_cm=error, rotation_deg=error),
                   status_counts=flags, translation_tails={'50': {'count': 1}, '100': {'count': 1}},
                   no_pose_n=1)
    continuous = dict(mean_difference=-1.25, CI95=[-2.5, .5], group_contributions={
        'both_failure': dict(signed_delta_sum=-3., contribution_to_common_available_mean=-.25)})
    pair = dict(transition={'counts': dict(both_success=1, failure_to_success=2,
        success_to_failure=3, both_failure=186, reference_unresolved=0),
        'ids': dict(success_to_failure=['generated-id-1'])},
        success_rate=dict(mean_difference=-1/192, CI95=[-.02, .01]),
        continuous=dict(translation_cm=continuous, rotation_deg=continuous))
    pops = {}
    for pop, n in dict(A.PARTITIONS, PRIMARY=192).items():
        arms = {a: dict(deepcopy(summary), denominator=n) for a in A.ARMS}
        pops[pop] = dict(rows=n, arms=arms,
            comparisons={a+'-'+b: deepcopy(pair) for a, b in A.CONTRASTS})
    results = dict(arms=list(A.ARMS), populations=pops)
    common_arms = ['POSEFIX_RAW', *A.PF_SUBPIX_ARMS, 'N0', 'E0']
    common = dict(PRIMARY192_established=False, arms=common_arms, populations={
        pop: dict(arms={a: dict(deepcopy(summary), denominator=n) for a in common_arms})
        for pop, n in [('CAL', 46), ('TEST', 49), ('COMMON95', 95)]})
    diag = dict(CAL_point_stage_state_errors_px={'N3_INITIAL': {'B': error, 'S': A.distribution([None])}},
        PRIMARY_PoseFix_input_R0_good_point=dict(good_input_lt5_n=10, damaged_good_to_gt10_n=2,
            large_input_gt20_n=3, recovered_large_to_le10_n=1, good_input_movement_px=error),
        PRIMARY_PoseFix_vs_E0_patterns={'PF_FAIL_E0_SUCCESS': {'n': 2}},
        CAL_original_e_complete_good_input=dict(rows=192, good_input_lt5_n=57,
            damaged_good_to_gt10_n=9, large_input_gt20_n=21, recovered_large_to_le10_n=1),
        PRIMARY_E0_good_pose_diagnostic={a:dict(E0_good_input_n=114, damaged_ids=[],
            delta_translation_cm=error, delta_rotation_deg=error)for a in A.DIAGNOSTICS},
        error_support_status='COVERAGE_NOT_VERIFIED')
    stage = dict(joint_success_oracle=1, candidate_available_frames=2, no_candidates_frames=190,
                 after_support_unknown_frames=None)
    oldstage = dict(joint_success_oracle=1, candidate_available_frames=2)
    oracle = dict(original_bank_incomplete=True, populations={
        pop: dict(denominator=n, stages={'generated': deepcopy(stage), 'selected': deepcopy(stage)},
            original_stage_oracles={x: {'feasible': deepcopy(oldstage)} for x in ('G10', 'G13')},
            failure_flags={'INCOMPLETE': 190}, final_success_n=2, final_failure_n=n-2,
            final_reference_unresolved_n=0, failure_flags_final_failure_only={'INCOMPLETE': n-2},
            original_consumer_generated_complete_frames=2, original_consumer_generated_unknown_frames=n-2,
            original_consumer_generated_bank_complete=False)
        for pop, n in dict(A.PARTITIONS, PRIMARY=192).items()})
    seal = dict(new_training_updates=0, new_model_forwards=0, new_pose_solves=0,
                elapsed_seconds=1.2, process_cpu_seconds=.8)
    meta = dict(score_seal=deepcopy(seal), diagnostic_seal=deepcopy(seal),
        observation_headline=generated_observation_audit(),
        oracle_summary_text=R.oracle_text(oracle), independent_summary_text=R.independent_text(None),
        typed_numeric_audit=R.typed_numeric_audit(generated_scored_rows()),
        hashes={'fixture': 'a'*64}, artifact_status={'generated fixture': 'not an empirical result'})
    return results, common, diag, oracle, {}, [], meta


class ReportContracts(unittest.TestCase):
    def test_all_denominators_variance_availability_and_tail_remain_visible(self):
        args = generated_inputs()
        body = R.render(*args)
        for pop, n in dict(A.PARTITIONS, PRIMARY=192).items():
            self.assertIn(f'## {pop} {n}장 전체19경로', body)
        for arm in A.ARMS:
            self.assertEqual(body.count('| '+arm+' · '), 4)
        self.assertIn('위치 분산 cm²', body)
        self.assertIn('회전 분산 °²', body)
        self.assertIn('T/R available N', body)
        self.assertIn('| 101.000 |', body)
        self.assertIn('| 3/3 |', body)
        self.assertIn('| 1/1 | 1 | NA | NA | NA | NA | NA | NA |', body)
        self.assertIn('-1.250 [-2.500, 0.500]', body)
        self.assertIn('COVERAGE_NOT_VERIFIED', body)
        self.assertIn('POSEFIX_SUBPIX_FULL_COHORT_NOT_ESTABLISHED', body)
        self.assertNotIn('/home/', body)
        self.assertNotIn('FIXED_REFERENCES.json)', body)
        self.assertIn('B/I 1252개', body)
        self.assertIn('381개(typed 310+legacy 71)', body)
        self.assertIn('예측 fallback 260개', body)
        self.assertIn('38장/304꼭짓점', body)
        self.assertIn('순수 인과증명', body)
        self.assertIn('| N3/SubPix | 유지 |', body)
        self.assertIn('| E 후 B/I 재관측 | 운영 채택 보류 |', body)

    def test_wrong_full_denominator_and_cap_winner_are_rejected(self):
        args = list(generated_inputs())
        args[0]['populations']['TEST']['rows'] = 48
        with self.assertRaises(ValueError):
            R.render(*args)
        args = list(generated_inputs())
        args[1]['PRIMARY192_established'] = True
        with self.assertRaises(ValueError):
            R.render(*args)

    def test_oracle_coverage_original_packages_and_unknown_are_separate(self):
        oracle = generated_inputs()[3]
        text = R.oracle_text(oracle)
        self.assertIn('consumer 후보 bank에도 기록 공백', text)
        self.assertIn('| PRIMARY192 | G10 | feasible | 1 | 2 |', text)
        self.assertIn('| PRIMARY192 | G13 | feasible | 1 | 2 |', text)
        self.assertIn('| generated | 1 | 2 | 190 | NA |', text)
        self.assertIn('최종 실패 190장, 성공 2장', text)
        self.assertIn('PRIMARY 전체192 coverage/후보 flag', text)
        self.assertIn('PRIMARY 최종 실패 영상 안의 flag', text)
        self.assertIn('| PRIMARY192 | 2 | 190 | False |', text)
        self.assertIn('INCOMPLETE', R.oracle_text(None))
        self.assertIn('미확인', R.independent_text(None))
        self.assertIn('실패 수 1', R.independent_text(dict(checks=12, failures=['generated failure'])))

    def test_actual_cost_schema_and_independent_receipt_scope(self):
        args = generated_inputs()
        cost = dict(cpu_solver_analysis_stage_elapsed_seconds=100., gpu_process_elapsed_seconds=0.,
            whole_task_wall_seconds=500., worker_cpu_seconds=250., top_level_new_pose_slots=952,
            unique_slot_ID_arm_pairs=952, pilot_slots_reused_by_full_run=24,
            additional_original_e_logging_slots=0, original_e_logging_shared_with_TYPED_MULTI=238,
            per_arm={'TYPED_SINGLE': dict(new_slots=238, sum_slot_elapsed_seconds=40.,
                worker_cpu_seconds=39., median_slot_elapsed_seconds=.1, max_slot_elapsed_seconds=2.,
                SLSQP_candidate_records=None, seed_least_squares_attempts=0)})
        text = R.cost_text(cost, [], args[-1]['score_seal'], args[-1]['diagnostic_seal'])
        self.assertIn('| 실행 장부 cpu_solver_analysis_stage_elapsed_seconds | 100.000 |', text)
        self.assertIn('| 실행 장부 top_level_new_pose_slots | 952.000 |', text)
        self.assertIn('| TYPED_SINGLE | 238 | 40.000 | 39.000 | 0.100/2.000 | NA | 0 |', text)
        self.assertIn('원래 E0 상위 pipeline 비용은 추가', text)
        receipt = dict(arm_rows=4522, discrepancies=[], success_all_match=True,
                       tolerance=dict(translation_cm=1e-9, rotation_deg=1e-6))
        summary = dict(arm_cohort_rows=76, all_pass=True)
        text = R.independent_text(receipt, summary)
        self.assertIn('4522개 행별 오차', text)
        self.assertIn('discrepancy는 0개', text)
        self.assertIn('76개 집계', text)
        self.assertIn('bootstrap CI나 의미 owner 정답까지 독립 검증했다는 뜻이 아니다', text)

    def test_no_report_GO_means_no_result_IO(self):
        with mock.patch.object(A, 'read', side_effect=AssertionError('forbidden IO')):
            with self.assertRaises(PermissionError):
                R.generate('not-present', 'not-present')

    def test_sealed_full_owner_csv_derives_changed_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_generated_observation_files(root)
            audit = R.observation_headline(root)
            self.assertEqual(audit['nonidentity_frames'], 38)
            self.assertEqual(audit['nonidentity_vertices'], 304)
            self.assertEqual(audit['semantic_owner_status'], 'SEMANTIC_OWNER_UNVERIFIED')
            with (root/'e_trace_v2/OBSERVATION_TYPE_AND_OWNER_AUDIT.csv').open('a') as stream:
                stream.write('mutated,False\n')
            with self.assertRaises(ValueError):
                R.observation_headline(root)

    def test_tiny_raw_angle_signs_remain_counts_without_promoting_gain(self):
        rows = generated_scored_rows()
        rows[0]['metrics']['TYPED_SINGLE']['rotation_deg'] -= 1e-12
        rows[1]['metrics']['TYPED_SINGLE']['rotation_deg'] += 2e-12
        numeric = R.typed_numeric_audit(rows)
        self.assertEqual(numeric['rotation_deg']['raw_improved'], 1)
        self.assertEqual(numeric['rotation_deg']['raw_worsened'], 1)
        self.assertEqual(numeric['rotation_deg']['raw_tied'], 190)
        self.assertEqual(numeric['rotation_deg']['beyond_existing_1e_9_tolerance'], 0)
        self.assertEqual(numeric['translation_cm']['max_absolute_difference'], 0.)
        with self.assertRaises(ValueError):
            R.typed_numeric_audit(rows[:-1])

    def test_generation_verifies_hashes_and_preserves_prior_attempt(self):
        results, common, diag, *_ = generated_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_generated_observation_files(root)
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            write(root/'PREDICTION_SEAL.json', {'schema': 'generated fixture'})
            write(root/'CANDIDATE_SEAL.json', {'schema': 'generated fixture'})
            sf, df = root/'score_v2', root/'diagnostics_v1'
            write(sf/'RESULTS.json', results)
            write(sf/'POSEFIX_SUBPIX_COMMON95.json', common)
            write(sf/'SCORED_ROWS.json', {'rows':generated_scored_rows()})
            write(df/'DIAGNOSTICS.json', diag)
            shared = dict(prediction_seal_sha256=A.sha(root/'PREDICTION_SEAL.json'),
                          candidate_seal_sha256=A.sha(root/'CANDIDATE_SEAL.json'))
            ss = dict(generated_inputs()[-1]['score_seal'], **shared,
                source_sha256={'analysis.py': 'a'*64}, artifacts={
                    n: A.sha(sf/n) for n in ('RESULTS.json', 'POSEFIX_SUBPIX_COMMON95.json', 'SCORED_ROWS.json')})
            write(sf/'SCORE_SEAL.json', ss)
            ds = dict(generated_inputs()[-1]['diagnostic_seal'], score_seal_sha256=A.sha(sf/'SCORE_SEAL.json'),
                      artifacts={'DIAGNOSTICS.json': A.sha(df/'DIAGNOSTICS.json')})
            write(df/'DIAGNOSTIC_SEAL.json', ds)
            output = root/'generated-only-output'
            receipt = R.generate(root, output, root_report_GO=True)
            self.assertEqual(set(receipt), {'REPORT_KO.md', 'REPRODUCE.md'})
            self.assertIn('oracle 결과 파일 미확인', (output/'REPORT_KO.md').read_text())
            with self.assertRaises(FileExistsError):
                R.generate(root, output, root_report_GO=True)
            ds['score_seal_sha256'] = 'b'*64
            write(df/'DIAGNOSTIC_SEAL.json', ds)
            bridge = dict(old_score_seal_sha256='b'*64,
                new_score_seal_sha256=A.sha(sf/'SCORE_SEAL.json'), **shared,
                all_4522_metric_dicts_exact_equal=True,
                all_76_numeric_arm_aggregates_exact_equal=False,
                all_paired_comparisons_and_CIs_exact_equal=True)
            write(root/'SCORE_V1_V2_METADATA_BRIDGE.json', bridge)
            with self.assertRaises(ValueError):
                R.generate(root, root/'bad-metadata-bridge', root_report_GO=True)
            bridge['all_76_numeric_arm_aggregates_exact_equal'] = True
            write(root/'SCORE_V1_V2_METADATA_BRIDGE.json', bridge)
            R.generate(root, root/'valid-metadata-bridge', root_report_GO=True)
            self.assertIn('4,522개 metric dict', (root/'valid-metadata-bridge/REPORT_KO.md').read_text())
            write(df/'DIAGNOSTICS.json', {'fixture_tampered': True})
            with self.assertRaises(ValueError):
                R.generate(root, root/'new-output-after-tamper', root_report_GO=True)
            self.assertFalse((root/'new-output-after-tamper').exists())


if __name__ == '__main__':
    unittest.main()
