"""Small integrity/publication regression fixtures; no real score or RGB reads."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from . import finalize, publish, report
from .common import read, sha, write


class ReportPublicationTests(unittest.TestCase):
    def test_explicit_go_precedes_every_filesystem_read(self):
        with self.assertRaises(PermissionError):report.run('/does/not/exist')
        with self.assertRaises(PermissionError):publish.run('/does/not/exist')
        with self.assertRaises(PermissionError):finalize.run('/does/not/exist')

    def test_checker_must_bind_current_evaluation_not_merely_say_pass(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);evaluation=run/'evaluation';evaluation.mkdir()
            write(run/'PLAN_LOCK.json',{'frozen':True});(run/'PREDICTIONS.jsonl.gz').write_bytes(b'sealed journal')
            write(run/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,reference_inputs=False,
                  plan_lock_sha256=sha(run/'PLAN_LOCK.json'),predictions_sha256=sha(run/'PREDICTIONS.jsonl.gz')))
            write(evaluation/'RESULTS.json',{'fixture':True})
            write(evaluation/'EVALUATION_SEAL.json',dict(explicit_root_scoring_GO=True,
                  input_bindings=dict(plan_lock=sha(run/'PLAN_LOCK.json'),
                    new_prediction_seal=sha(run/'PREDICTION_SEAL.json'),new_predictions=sha(run/'PREDICTIONS.jsonl.gz')),
                  artifacts={'RESULTS.json':sha(evaluation/'RESULTS.json')}))
            check=run/'check.json';write(check,dict(status='PASS',failure_count=0,input_bindings={}))
            with self.assertRaises(ValueError):report.verify_report_inputs(run,evaluation,check,report_GO=True)
            write(check,dict(status='PASS',failure_count=0,input_bindings={
                  'evaluation':sha(evaluation/'EVALUATION_SEAL.json'),'results':sha(evaluation/'RESULTS.json')}),replace=True)
            report.verify_report_inputs(run,evaluation,check,report_GO=True)
            (evaluation/'RESULTS.json').write_text('{}')
            with self.assertRaises(ValueError):report.verify_report_inputs(run,evaluation,check,report_GO=True)

    def test_whole_inventory_and_na_rows_survive_four_population_tables(self):
        new=tuple('METHOD_'+str(i) for i in range(16))
        registry={'original_arms':list(report.BASELINES),'arms':[{'name':a} for a in new]}
        arms,actual_new=report._inventory(registry)
        self.assertEqual(len(arms),22);self.assertEqual(actual_new,new)
        dist=dict(n=1,mean=9.,sample_std=None,P90=9.,P95=9.,max=9.)
        summary=dict(successes=0,denominator=192,success_rate=0.,translation_cm=dist,rotation_deg=dist,
                     no_pose=191,T_gt50cm=0,T_gt100cm=0,R_gt45deg=0)
        results={'populations':{p:{a:summary for a in report.BASELINES} for p in report.POPULATIONS}}
        output='\n'.join(report.common_tables(results,arms,{a:{'status':'NOT_APPLICABLE'} for a in new}))
        for arm in arms:self.assertEqual(output.count('| '+arm+' |'),4)
        self.assertEqual(output.count('| NOT_APPLICABLE |'),64)
        broken=deepcopy(registry);broken['arms'].pop()
        with self.assertRaises(ValueError):report._inventory(broken)

    def test_success_gain_cannot_hide_tail_tradeoff(self):
        baseline=dict(successes=114,translation_cm={'mean':10.},rotation_deg={'mean':10.},
                      T_gt50cm=1,T_gt100cm=0,R_gt45deg=1,no_pose=0)
        primary={a:deepcopy(baseline) for a in report.COMPARATORS}
        primary['NEW']=dict(deepcopy(baseline),successes=115)
        self.assertEqual(report.classify('NEW',primary)['category'],'PROMISING_FOR_INDEPENDENT_VALIDATION')
        primary['NEW']['T_gt100cm']=1
        self.assertEqual(report.classify('NEW',primary)['category'],'SUCCESS_GAIN_WITH_TRADEOFF')
        self.assertEqual(report.classify('ABSENT',primary)['category'],'INCONCLUSIVE_DATA_OR_REFERENCE')

    def test_public_derivative_preserves_numbers_and_private_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.json';target=root/'public.json'
            original={'source':'/home/private/research/weights.pt','mean':1.234567890123,'failure_ids':['0001'],
                      'citation':'https://example.org/a/b','/home/private/research/receipt.json':'f'*64}
            write(source,original);digest=sha(source)
            receipt=publish._derive(source,target,{'/home/private/research':'$RUN_PRIVATE'},1024*1024)
            data=read(target)
            self.assertEqual(data['mean'],original['mean']);self.assertEqual(data['failure_ids'],['0001'])
            self.assertEqual(data['source'],'$RUN_PRIVATE/weights.pt')
            self.assertEqual(data['citation'],original['citation'])
            self.assertIn('$RUN_PRIVATE/receipt.json',data)
            self.assertEqual(sha(source),digest);self.assertEqual(receipt['private_sha256'],digest)

    def test_pose_journal_and_path_escape_are_not_publishable(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);bank=root/'PREDICTIONS.jsonl.gz';bank.write_bytes(b'not copied')
            with self.assertRaises(ValueError):publish._derive(bank,root/'out.json',{},1024)
            with self.assertRaises(ValueError):publish._safe_target(root,'../leak.json')
            image=root/'raw.png';image.write_bytes(b'private RGB')
            approval=root/'approval.json';write(approval,dict(root_publication_approval=True,
                 assets=[dict(source=str(image),target='image.png',sha256=sha(image),public_permission_confirmed=False)]))
            with self.assertRaises(PermissionError):publish._approved_assets(approval,root/'public',1024)

    def test_synthetic_head_small_result_seal_and_additional_forward_accounting(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            self.assertIsNone(report.verify_synthetic_head_evaluation(run))
            training=run/'head_training_v1/TRAINING_MANIFEST.json'
            write(training,dict(complete=True,total_training_runs=6))
            with self.assertRaises(ValueError):report.verify_synthetic_head_evaluation(run)

            required=dict(distribution_seal='head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json',
                pose_prediction_seal='learned_synthetic_v1/PREDICTION_SEAL.json',
                training_manifest='head_training_v1/TRAINING_MANIFEST.json',GT='synthetic_v1/SYNTH_GT_LABELS.jsonl.gz',
                labels='head_labels_v1/SYNTHETIC_HEAD_TARGETS.jsonl.gz',label_seal='head_labels_v1/LABEL_SEAL.json',
                split_lock='synthetic_v1/DATA_SPLIT_LOCK.json')
            for key,name in required.items():
                if key in ('distribution_seal','training_manifest'):continue
                p=run/name;p.parent.mkdir(parents=True,exist_ok=True)
                if key=='pose_prediction_seal':
                    write(p,dict(rows=1003,checkpoints={str(i):'fixed' for i in range(6)},cost={'CPU_seconds':2.}))
                elif p.suffix=='.json':write(p,{'unchanged_fixture_input':True})
                else:p.write_bytes(b'unchanged sealed fixture input')
            write(run/required['distribution_seal'],dict(complete=True,rows=1003,reference_reads=0,
                before_reference_decode=True,tiny_head_forward_calls=6018,new_training_updates=0,cost={'CPU_seconds':3.}))
            bindings={key:sha(run/name) for key,name in required.items()}
            folder=run/'synthetic_head_evaluation_v1';folder.mkdir()
            write(folder/'SYNTHETIC_LEARNED_EVALUATION.json',dict(complete=True,rows=1003,no_model_choice_or_fit=True,
                  first_representative_seed=20261018,input_bindings=bindings))
            (folder/'SYNTHETIC_LEARNED_EVALUATION_KO.md').write_text('fixture: frozen-head validation only\n')
            write(folder/'EVALUATION_SEAL.json',dict(schema='SYNTHETIC_HEAD_EVALUATION_SEAL_V1',complete=True,
                  explicit_root_GO=True,rows=1003,input_bindings=bindings,
                  result_sha256=sha(folder/'SYNTHETIC_LEARNED_EVALUATION.json'),
                  markdown_sha256=sha(folder/'SYNTHETIC_LEARNED_EVALUATION_KO.md'),
                  artifacts={p.name:sha(p) for p in folder.iterdir()},cost={'CPU_seconds':4.}))
            self.assertEqual(report.verify_synthetic_head_evaluation(run)['files'][0],'SYNTHETIC_LEARNED_EVALUATION.json')
            costs=report._cost(run,{'total_training_runs':6,'runs':[]})
            counts=costs['total_known_cost']['tiny_head_inference_counts']
            self.assertEqual(counts['learned_synthetic_v1/PREDICTION_SEAL.json']['tiny_head_forward_calls'],6018)
            self.assertEqual(counts['head_synthetic_distributions_v1/DISTRIBUTION_SEAL.json']['tiny_head_forward_calls'],6018)
            self.assertEqual(costs['total_known_cost']['tiny_head_forward_calls_recorded'],12036)
            self.assertEqual(costs['total_known_cost']['CPU_seconds_known_subtotal'],9.)
            (folder/'SYNTHETIC_LEARNED_EVALUATION_KO.md').write_text('changed')
            with self.assertRaises(ValueError):report.verify_synthetic_head_evaluation(run)

    def test_rounding_appendix_preserves_failed_attempt_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);failed=root/'independent_v1/INDEPENDENT_CHECK.json'
            write(failed,dict(status='FAIL',failure_count=86))
            current=root/'independent_v2';current.mkdir();audit=current/'ROUNDING_DIRECTION_AUDIT.json'
            write(audit,dict(status='PASS',zero_tolerance_changed=False,
                predictions_references_models_success_criteria_weights_or_reported_numbers_changed=False,
                previous_failure_artifacts={str(failed):sha(failed)},previous_failed_count_fields_exactly_explained=86,
                sign_switch_pair_metric_rows=37,unique_sign_switch_frame_count=5,original_expression_value_checks=10472))
            checker=dict(rounding_direction_audit={'file':audit.name,'sha256':sha(audit)})
            write(current/'INDEPENDENT_CHECK.json',checker)
            appendix=report.optional_appendices(root,current/'INDEPENDENT_CHECK.json',checker)
            self.assertEqual(appendix['summary']['rounding_direction']['previous_failed_fields'],86)
            self.assertIn('INDEPENDENT_CHECK_V1_FAILED.json',[v['name'] for v in appendix['files']])
            write(failed,dict(status='PASS'),replace=True)
            with self.assertRaises(ValueError):report.optional_appendices(root,current/'INDEPENDENT_CHECK.json',checker)

    def test_forensic_public_summary_keeps_evidence_without_private_pose_matrices(self):
        original={'selected_energy':2.4e-9,'poses':{'SELECTED':{'actual_pose':{'R_physical':[[1,0,0]],'centroid':[1,2,3]},
                    'finite_segment_inside':0,'valid_intersections':1,'t_min':-18.4}},
                  'optimizer_record':{'optimizer_success':True,'start_pose':{'centroid':[1,2,3]},'feasible':True}}
        public=report._without_pose_matrices(original)
        publish._reject_private_pose_payload(public)
        self.assertEqual(public['selected_energy'],2.4e-9)
        self.assertEqual(public['poses']['SELECTED']['t_min'],-18.4)
        self.assertTrue(public['optimizer_record']['feasible'])
        self.assertIn('actual_pose',original['poses']['SELECTED'])

    def test_finalize_default_and_custom_receipts_resume_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);run=root/'private';run.mkdir();source=root/'source';source.mkdir()
            public=source/'public';public.mkdir();evaluation=run/'evaluation_v1';evaluation.mkdir()
            write(run/'PLAN_LOCK.json',{'fixture':True});write(run/'CONFIG_FROZEN.json',{})
            new=['METHOD_'+str(i) for i in range(16)]
            write(run/'ARM_REGISTRY.json',dict(original_arms=list(report.BASELINES),arms=[dict(name=a) for a in new]))
            write(run/'ARM_EXECUTION_STATUS.json',{a:dict(status='NOT_APPLICABLE',reason='fixture no training') for a in new})
            (run/'PREDICTIONS.jsonl.gz').write_bytes(b'sealed fixture bytes, no pose or RGB')
            write(run/'PREDICTION_SEAL.json',dict(complete=True,before_scoring=True,reference_inputs=False,
                  arms=[],plan_lock_sha256=sha(run/'PLAN_LOCK.json'),predictions_sha256=sha(run/'PREDICTIONS.jsonl.gz')))
            populations={}
            stage={}
            for population,n in report.DENOMINATORS.items():
                dist=dict(n=n,mean=10.,sample_std=1.,P90=12.,P95=13.,max=14.)
                s=dict(denominator=n,successes=0,success_rate=0.,translation_cm=dist,rotation_deg=dist,
                       no_pose=0,T_gt50cm=0,T_gt100cm=0,R_gt45deg=0,status_counts={},failures=[])
                populations[population]={a:deepcopy(s) for a in report.BASELINES}
                stage[population]=dict(carried_oracle_successes=0,denominator=n,arms={})
            result=dict(arms=list(report.BASELINES),new_arms=[],populations=populations,
                        paired={p:{} for p in report.POPULATIONS},bootstrap_policy=dict(seed=20260917,draws=10000))
            write(evaluation/'RESULTS.json',result)
            write(evaluation/'CANDIDATE_STAGE_DIAGNOSIS.json',dict(populations=stage,rows=[]))
            for name in ('ROW_METRICS.csv','PAIRED_TRANSITIONS.csv','COMPARISON_ALL_ARMS.csv'):
                (evaluation/name).write_text('id,partition,arm,success\n')
            write(evaluation/'EVALUATION_SEAL.json',dict(explicit_root_scoring_GO=True,
                  input_bindings=dict(plan_lock=sha(run/'PLAN_LOCK.json'),new_prediction_seal=sha(run/'PREDICTION_SEAL.json'),
                                      new_predictions=sha(run/'PREDICTIONS.jsonl.gz')),
                  artifacts={p.name:sha(p) for p in evaluation.iterdir()}))
            check=run/'check.json';write(check,dict(status='PASS',failure_count=0,input_bindings={
                  'evaluation':sha(evaluation/'EVALUATION_SEAL.json'),'results':sha(evaluation/'RESULTS.json')}))
            gallery=run/'gallery.json';write(gallery,dict(explicit_postseal_GO=True,all_population_frames=238,
                  PRIMARY_frames=192,input_bindings={'prediction':sha(run/'PREDICTION_SEAL.json'),
                  'evaluation':sha(evaluation/'EVALUATION_SEAL.json')},outputs=[]))
            supplied=set(publish.REPORT_FILES)|set(publish.EVALUATION_FILES)|set(publish.PROTOCOL_FILES)
            for name in publish.REQUIRED:
                if name not in supplied:(public/name).write_text('fixture documentation\n')
            first=finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                               source_root=source,output_dir=public,final_GO=True)
            timestamp=(run/'report_v1/REPORT_SEAL.json').stat().st_mtime_ns
            again=finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                               source_root=source,output_dir=public,final_GO=True)
            self.assertEqual(first,again);self.assertEqual(timestamp,(run/'report_v1/REPORT_SEAL.json').stat().st_mtime_ns)
            self.assertTrue(again['complete']);self.assertFalse(again['git_commit_push_performed'])
            default_paths=(run/'PUBLICATION_RECEIPT.json',run/'FINALIZATION_RECEIPT.json')
            originals={str(p):p.read_bytes() for p in default_paths}
            custom_public=run/'PUBLICATION_RECEIPT_V2.json';custom_final=run/'FINALIZATION_RECEIPT_V2.json'
            second=finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                               source_root=source,output_dir=public,final_GO=True,
                               publication_receipt_path=custom_public,final_receipt_path=custom_final)
            second_again=finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                               source_root=source,output_dir=public,final_GO=True,
                               publication_receipt_path=custom_public,final_receipt_path=custom_final)
            self.assertEqual(second,second_again)
            self.assertEqual(second['publication_receipt_sha256'],sha(custom_public))
            self.assertEqual(second['receipt_paths']['publication'],str(custom_public.resolve()))
            self.assertEqual(timestamp,(run/'report_v1/REPORT_SEAL.json').stat().st_mtime_ns)
            for p in default_paths:self.assertEqual(originals[str(p)],p.read_bytes())
            default_again=finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                                      source_root=source,output_dir=public,final_GO=True)
            self.assertEqual(first,default_again)
            with self.assertRaises(ValueError):
                finalize.run(run,evaluation_dir=evaluation,independent_receipt=check,gallery_receipt=gallery,
                    source_root=source,output_dir=public,final_GO=True,final_receipt_path=custom_final)


if __name__=='__main__':unittest.main()
