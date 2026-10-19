"""Constructed metadata fixtures only; no campaign receipts are read."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from . import master_costs as M


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return M.sha(path)


def jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n'for r in rows));return M.sha(path)


def recovery(root,name,prior,new):
    base=root/name; inv=root/(name+'_RECOVERY');arm='A'
    lock=dict(workers=4,arms=[arm],row_helper_sha256='helper',numerical_kernel_source_binding={'kernel':'fixed'},
        static_ordered_row_ids={'real':['one','two']})
    locksha=put(base/'PRODUCER_LOCK.json',lock)
    oldjob=put(base/'row_jobs/real/0000.json',{'historical_parent':'intentionally_partial'})
    put(root/'GO.json',{'approved':True})
    rows=[]
    for i,calls in enumerate([prior,new]):
        folder=base/arm/'real'/f'{i:04d}'
        framesha=put(folder/'FRAME_SEAL.json',{'resource':{'actual_pose_optimizer_calls':999999},'complete':True})
        row=dict(complete=True,arm=arm,id=['one','two'][i],ordinal=i,domain='real',driver_sha256='helper',
            numerical_kernel_source_binding={'kernel':'fixed'},original_frame_seal_sha256=framesha,
            resource=dict(actual_new_pose_optimizer_calls=calls,current_execution_CPU_seconds=1.,current_execution_wall_seconds=2.))
        rowsha=put(folder/'ROW_EXECUTION.json',row)
        rows.append(dict(arm=arm,path=str(folder/'ROW_EXECUTION.json'),sha256=rowsha,receipt=row))
    before=dict(known_stored_new_optimizer_calls=prior,interrupted_prior_optimizer_calls=M.UNKNOWN,
        completed_artifacts_sha256={f'A/real/0000/{f}':M.sha(base/'A/real/0000'/f)for f in ('FRAME_SEAL.json','ROW_EXECUTION.json')},
        incomplete_arm_directories=[dict(prior_optimizer_progress=M.UNKNOWN,path='lost-active-row')])
    put(inv/'HISTORICAL_STATE.json',before)
    initial=dict(original_output=str(base),delegated_initializer_output=str(base),delegated_execute_output=str(inv),
        workers=4,original_producer_lock_sha256=locksha,approved_GO=str(root/'GO.json'),approved_GO_sha256=M.sha(root/'GO.json'),
        source_binding=dict(recovery_controller_sha256='recovery',delegated_D4_source_binding={
            'row_helper_sha256':'helper','numerical_kernel_source_binding':{'kernel':'fixed'}}),
        original_row_jobs_sha256={'real/0000.json':oldjob},total_jobs=2)
    put(inv/'INVOCATION_LOCK.json',initial);put(inv/'UNSEALED_HISTORY.json',{'files':[]})
    dispatch=dict(submitted_jobs=2,unsubmitted_jobs=0,checkpoint_stop_observed=False,active_rows_drained_before_exit=True)
    put(inv/'DISPATCH_RESULT.json',dispatch)
    jobhashes={}
    for i,record in enumerate(rows):
        job=dict(domain='real',ordinal=i,id=['one','two'][i],records=[record],
            actual_new_optimizer_calls_this_job=0 if i==0 else new,current_job_CPU_seconds=.1,current_job_wall_seconds=.2)
        jobhashes[f'real/{i:04d}.json']=put(inv/'row_jobs/real'/f'{i:04d}.json',job)
    summary=dict(status='COMPLETE_NEW',missing_rows=[],technical_failures=[],new_pose_optimizer_calls=prior+new,
        current_invocation_new_pose_optimizer_calls=new,producer_lock_sha256=locksha)
    put(base/'SUMMARY.json',summary)
    put(base/'A/real/FEATURE_SEAL.json',dict(complete=True,rows=2,producer_lock_sha256=locksha,
        row_execution_sha256={f'{i:04d}':r['sha256']for i,r in enumerate(rows)},
        frame_seal_sha256={f'{i:04d}':r['receipt']['original_frame_seal_sha256']for i,r in enumerate(rows)}))
    c=dict(complete=True,original_output=str(base),invocation=str(inv),recovery_controller_sha256='recovery',
        original_producer_lock_sha256=locksha,interrupted_historical_optimizer_calls=M.UNKNOWN,total_campaign_actual_calls=M.UNKNOWN,
        copied_historical_intrinsic_calls_added=0,prior_stored_known_calls=prior,job_envelopes_sha256=jobhashes,
        cost_identity_verified=True,technical_failures=[],original_completed_artifacts_preserved=True,
        original_parent_row_jobs_preserved=True,current_invocation_call_count_status='EXACT_FULL_ROW_IDENTITY',
        current_invocation_unattributed_optimizer_calls=0,dispatch=dispatch,
        actual_new_optimizer_calls_this_invocation=new,successful_returned_jobs_new_optimizer_calls_lower_bound=new,
        base_stored_calls_cumulative=prior+new,summary_sha256=M.sha(base/'SUMMARY.json'))
    for k,leaf in [('invocation_lock_sha256','INVOCATION_LOCK.json'),('historical_state_sha256','HISTORICAL_STATE.json'),
                   ('unsealed_history_sha256','UNSEALED_HISTORY.json'),('dispatch_result_sha256','DISPATCH_RESULT.json')]:c[k]=M.sha(inv/leaf)
    put(inv/'RECOVERY_COMPLETION.json',c)
    put(base/'DISPATCH_COMPLETION.json',dict(recovery_completion_sha256=M.sha(inv/'RECOVERY_COMPLETION.json'),summary_sha256=M.sha(base/'SUMMARY.json')))
    return dict(name=name,base=str(base),invocation=str(inv),arms=[arm],expected_jobs=2)


def closed(root):
    evidence=root/'old_closed_receipt.json';put(evidence,{'calls':479400})
    a=dict(name='closed',root=str(root/'closedbase'),actual_optimizer_calls=479400,known_optimizer_calls=479400,
        cost_complete=True,evidence={str(evidence):M.sha(evidence)},timing={})
    d=dict(schema='CLOSED_PRIOR_CORE_ATTEMPTS_METADATA_CHECK_V1',active_D4_totals_included=False,new_scientific_calls=0,
        expected_closed_call_sum=479400,report=dict(cost_complete=True,actual_optimizer_calls=479400,attempts=[a],preserved_prior_S2={'actual_optimizer_calls':62807}))
    p=root/'closed.json';put(p,d);return dict(path=str(p),sha256=M.sha(p),expected_calls=479400)


def head(root,arm='HEAD',seed=20261018,attempt=1):
    output=root/'training';folder=output/f'{arm}_s{seed}'
    for name in ('best.pt','last.pt','HISTORY.json'):put(folder/name,{'test_bytes':'not a model'})
    binding={'features_sha256':'features'}
    r=dict(complete=True,arm=arm,seed=seed,attempt=attempt,bindings=binding,actual_new_training_runs=1,
        optimizer_steps=4,training_frame_exposures=128,resumed_same_optimizer=True,
        checkpoint_sha256=M.sha(folder/'best.pt'),last_checkpoint_sha256=M.sha(folder/'last.pt'),history_sha256=M.sha(folder/'HISTORY.json'),
        cost=dict(CPU_seconds=2.,wall_seconds=3.,threads=1,device='CPU',GPU_seconds=0))
    put(folder/'RUN_RECEIPT.json',r)
    a=dict(arm=arm,seed=seed,attempt=attempt,bindings=binding,output=str(output),registered_before_optimizer=True)
    event=dict(event='COMPLETE',arm=arm,seed=seed,attempt=attempt,optimizer_steps=4,receipt_sha256=M.sha(folder/'RUN_RECEIPT.json'))
    return a,r,event


class MasterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def modify(self,p,**kwargs):
        d=json.loads(Path(p).read_text());d.update(kwargs);put(p,d)

    def test_cumulative_prefix_not_added_twice(self):
        c=M.closed_attempts(M.Reader(),closed(self.r))
        s=M.recovery_attempt(M.Reader(),recovery(self.r,'S2',55310,5))
        f=M.recovery_attempt(M.Reader(),recovery(self.r,'F',57077,7))
        self.assertEqual(c['known_calls']+s['base_calls_cumulative']+f['base_calls_cumulative'],591799)
        self.assertEqual(s['prior_stored_known_calls']+f['prior_stored_known_calls'],112387)
        self.assertEqual(s['copied_FRAME_intrinsic_calls_added'],0)
        self.assertEqual(s['interrupted_historical_optimizer_calls'],M.UNKNOWN)

    def test_original_partial_journal_is_valid(self):
        s=M.recovery_attempt(M.Reader(),recovery(self.r,'S',3,5))
        self.assertEqual((s['old_parent_job_envelopes'],s['new_parent_job_envelopes']),(1,2))

    def test_missing_new_job_rejected(self):
        spec=recovery(self.r,'S',3,5);p=Path(spec['invocation'])/'row_jobs/real/0001.json';p.unlink()
        with self.assertRaises(FileNotFoundError):M.recovery_attempt(M.Reader(),spec)

    def test_after_before_mismatch_rejected(self):
        spec=recovery(self.r,'S',3,5);p=Path(spec['invocation'])/'RECOVERY_COMPLETION.json'
        self.modify(p,base_stored_calls_cumulative=9)
        with self.assertRaises(ValueError):M.recovery_attempt(M.Reader(),spec)

    def test_old_parent_journal_changed_rejected(self):
        spec=recovery(self.r,'S',3,5);put(Path(spec['base'])/'row_jobs/real/0001.json',{})
        with self.assertRaises(ValueError):M.recovery_attempt(M.Reader(),spec)

    def test_old_completed_artifact_changed_rejected(self):
        spec=recovery(self.r,'S',3,5);put(Path(spec['base'])/'A/real/0000/FRAME_SEAL.json',{})
        with self.assertRaises(ValueError):M.recovery_attempt(M.Reader(),spec)

    def test_missing_completion_preserves_unknown(self):
        spec=recovery(self.r,'S',3,5);(Path(spec['invocation'])/'RECOVERY_COMPLETION.json').unlink()
        d=M.recovery_attempt(M.Reader(),spec)
        self.assertEqual(d['known_base_call_lower_bound'],3);self.assertEqual(d['current_invocation_calls'],M.UNKNOWN)

    def test_incomplete_invocation_lower_bound_and_unknown(self):
        spec=recovery(self.r,'S',3,5);self.modify(Path(spec['invocation'])/'RECOVERY_COMPLETION.json',complete=False,
            current_invocation_unattributed_optimizer_calls=M.UNKNOWN)
        d=M.recovery_attempt(M.Reader(),spec)
        self.assertEqual(d['known_base_call_lower_bound'],8);self.assertEqual(d['actual_optimizer_calls'],M.UNKNOWN)

    def test_interrupted_work_zero_rejected(self):
        spec=recovery(self.r,'S',3,5);self.modify(Path(spec['invocation'])/'HISTORICAL_STATE.json',interrupted_prior_optimizer_calls=0)
        with self.assertRaises(ValueError):M.recovery_attempt(M.Reader(),spec)

    def test_closed_digest_guard(self):
        spec=closed(self.r);self.modify(spec['path'],expected_closed_call_sum=0)
        with self.assertRaises(ValueError):M.closed_attempts(M.Reader(),spec)

    def test_closed_duplicate_roots_rejected(self):
        spec=closed(self.r);p=Path(spec['path']);d=json.loads(p.read_text());d['report']['attempts']*=2;put(p,d);spec['sha256']=M.sha(p)
        with self.assertRaises(ValueError):M.closed_attempts(M.Reader(),spec)

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):M.inside(self.r,'../elsewhere')

    def test_reader_stability_guard(self):
        p=self.r/'data.json';put(p,{});rr=M.Reader();rr.read(p);put(p,{'changed':True})
        with self.assertRaises(ValueError):rr.stable()

    def test_same_optimizer_resume_counted_once(self):
        a,r,e=head(self.r);jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',[a]);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',[e])
        d=M.head_accounts(M.Reader(),self.r,[],expected_fits=33)
        self.assertEqual(d['completed_new_fits'],1);self.assertEqual(d['internal_fit_CPU_seconds'],2)
        self.assertFalse(d['all_expected_fits_complete'])

    def test_unfinished_reservation_unknown_not_completed(self):
        a,r,e=head(self.r);(Path(a['output'])/f"{a['arm']}_s{a['seed']}"/'RUN_RECEIPT.json').unlink()
        jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',[a]);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',[])
        d=M.head_accounts(M.Reader(),self.r,[])
        self.assertEqual(d['completed_new_fits'],0);self.assertEqual(d['unreceipted_attempt_cost'],M.UNKNOWN)

    def test_duplicate_fit_event_rejected(self):
        a,r,e=head(self.r);jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',[a]);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',[e,e])
        with self.assertRaises(ValueError):M.head_accounts(M.Reader(),self.r,[])

    def test_all33_fits_count_receipts_once(self):
        reservations=[];events=[]
        for i in range(11):
            for seed in M.SEEDS:
                a,r,e=head(self.r,arm=f'ARM{i}',seed=seed,attempt=len(reservations)+1)
                reservations.append(a);events.append(e)
        jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',reservations);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',events)
        d=M.head_accounts(M.Reader(),self.r,[])
        self.assertEqual(d['completed_new_fits'],33);self.assertTrue(d['all_expected_fits_complete'])
        self.assertEqual(d['internal_fit_CPU_seconds'],66.);self.assertEqual(d['optimizer_steps'],132)

    def prediction_fixture(self):
        reservations=[];events=[];runs=[]
        for i,seed in enumerate(M.SEEDS):
            a,r,e=head(self.r,seed=seed,attempt=i+1);reservations.append(a);events.append(e);runs.append(r)
        jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',reservations);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',events)
        manifest=self.r/'training/TRAINING_MANIFEST.json';put(manifest,dict(complete=True,runs=runs))
        p=self.r/'prediction/PREDICTION_SEAL.json'
        put(p,dict(complete=True,rows=238,ordered_ids=[str(i)for i in range(238)],training_manifest_sha256=M.sha(manifest),
            checkpoints={f"HEAD_s{r['seed']}":dict(checkpoint_sha256=r['checkpoint_sha256'],training_binding=r['bindings'])for r in runs},
            tiny_head_forward_calls=714,new_backbone_forwards=0,cost=dict(CPU_seconds=2.,wall_seconds=3.)))
        return dict(path=str(p),arm='HEAD',training_manifest=str(manifest))

    def test_tiny_H1_forwards_separate_from_backbone(self):
        spec=self.prediction_fixture();d=M.head_accounts(M.Reader(),self.r,[spec])
        self.assertEqual(d['tiny_head_real_prediction_forwards'],714)
        self.assertEqual(d['new_backbone_prediction_forwards'],0)

    def test_prediction_duplicate_rejected(self):
        spec=self.prediction_fixture()
        with self.assertRaises(ValueError):M.head_accounts(M.Reader(),self.r,[spec,spec])

    def test_prediction_manifest_mismatch_rejected(self):
        spec=self.prediction_fixture();p=Path(spec['training_manifest']);d=json.loads(p.read_text());d['runs'][0]['cost']['CPU_seconds']=123.;put(p,d)
        pred=Path(spec['path']);self.modify(pred,training_manifest_sha256=M.sha(p))
        with self.assertRaises(ValueError):M.head_accounts(M.Reader(),self.r,[spec])

    def test_changed_checkpoint_rejected_without_decode(self):
        a,r,e=head(self.r);jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',[a]);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',[e])
        put(Path(a['output'])/f"{a['arm']}_s{a['seed']}"/'best.pt',{})
        with self.assertRaises(ValueError):M.head_accounts(M.Reader(),self.r,[])

    def test_ledger_failed_cost_and_open_unknown(self):
        p=self.r/'ledger.jsonl';jsonl(p,[dict(event='START',phase='a',argv=['x']),
            dict(event='TECHNICAL_FAILURE',phase='a',argv=['x'],wall_seconds=3.,child_CPU_seconds=2.),
            dict(event='START',phase='unfinished')])
        d=M.command_ledger(M.Reader(),dict(name='test',path=str(p),parent_scope='service'))
        self.assertEqual(d['failures'],1);self.assertEqual(d['additive_child_CPU_seconds'],2.)
        self.assertEqual(d['untracked_cost'],M.UNKNOWN)

    def test_ledger_orphan_terminal_rejected(self):
        p=self.r/'ledger';jsonl(p,[dict(event='COMPLETE',phase='a',wall_seconds=3.,child_CPU_seconds=2.)])
        with self.assertRaises(ValueError):M.command_ledger(M.Reader(),dict(name='test',path=str(p)))

    def service_fixture(self,restarts='0',substate='exited',status='0'):
        launch=self.r/'launch.json';put(launch,dict(services={'S':dict(unit='test',returncode=0,
            actual_argv=['systemd-run','--property=Restart=no','--property=CPUAccounting=yes'],scientific_argv=['child'])}))
        folder=self.r/'lifecycle';put(folder/'LOCK.json',dict(source_sha256='capture'))
        snap=dict(epoch=100.,units=[dict(unit='test.service',returncode=0,SubState=substate,NRestarts=restarts,
            CPUUsageNSec='2500000000',ExecMainStartTimestampMonotonic='1000000',ExecMainExitTimestampMonotonic='11000000',
            MemoryPeak='1000',Result='success',ExecMainStatus=status)])
        jsonl(folder/'SNAPSHOTS.jsonl',[snap]);put(folder/'COMPLETION.json',dict(complete=True,scientific_completion_claim=False,
            last_snapshot=snap,snapshot_sha256=M.sha(folder/'SNAPSHOTS.jsonl'),source_sha256='capture'))
        return dict(launches=[str(launch)],lifecycle_root=str(folder))

    def test_service_CPU_and_elapsed_units_not_child_sum(self):
        d=M.services(M.Reader(),self.service_fixture())
        self.assertEqual(d['service_CPU_seconds'],2.5);self.assertEqual(d['recovery_service_envelope_elapsed_seconds'],10.)
        self.assertEqual(d['campaign_elapsed_wall_seconds'],M.UNKNOWN)

    def test_service_failure_does_not_claim_scientific_success(self):
        d=M.services(M.Reader(),self.service_fixture(substate='failed',status='1'))
        self.assertFalse(d['scientific_completion_claim']);self.assertEqual(d['entries'][0]['ExecMainStatus'],'1')

    def test_service_restart_rejected(self):
        with self.assertRaises(ValueError):M.services(M.Reader(),self.service_fixture(restarts='1'))

    def test_elapsed_from_documented_start_not_child_wall(self):
        put(self.r/'START.json',dict(created_utc='2026-10-19T08:02:47.125825+00:00',head='dirty'))
        cat=dict(run=str(self.r),external_finalization=dict(path=str(self.r/'end.json')))
        d=M.campaign_elapsed(M.Reader(),cat);self.assertEqual(d['elapsed_wall_seconds'],M.UNKNOWN)
        put(self.r/'end.json',dict(campaign_complete=True,external_finalization_epoch=d['documented_start_epoch']+120.))
        d=M.campaign_elapsed(M.Reader(),cat);self.assertEqual(d['elapsed_wall_seconds'],120.)

    def test_elapsed_requires_campaign_completion(self):
        put(self.r/'START.json',dict(created_utc='2026-10-19T08:02:47.125825+00:00',head='dirty'))
        put(self.r/'end.json',dict(campaign_complete=False,external_finalization_epoch=9999999999.))
        with self.assertRaises(ValueError):M.campaign_elapsed(M.Reader(),dict(run=str(self.r),external_finalization=dict(path=str(self.r/'end.json'))))

    def test_unfinished_cutoff_is_labeled_not_final_elapsed(self):
        put(self.r/'START.json',dict(created_utc='2026-10-19T08:02:47.125825+00:00',head='dirty'))
        start=M.campaign_elapsed(M.Reader(),dict(run=str(self.r)))['documented_start_epoch']
        put(self.r/'end.json',dict(campaign_complete=False,measurement_cutoff_epoch=start+240.,
            scope='all scientific and derived complete; reporting and publication pending'))
        d=M.campaign_elapsed(M.Reader(),dict(run=str(self.r),elapsed_measurement=dict(path=str(self.r/'end.json'))))
        self.assertEqual(d['elapsed_at_measurement_cutoff_seconds'],240.)
        self.assertEqual(d['elapsed_wall_seconds'],M.UNKNOWN)

    def test_typed_counters_remain_separate_in_integration(self):
        cat=dict(schema=M.SCHEMA,run=str(self.r),closed=closed(self.r),
            recoveries=[recovery(self.r,'S2',55310,5),recovery(self.r,'F',57077,7)],
            predictions=[],services=dict(launches=[],lifecycle_root=str(self.r/'pending')),metadata=[],expected_old_D4_prefix=112387)
        put(self.r/'D_FIXED_COMMON_GRAPH/SUMMARY.json',dict(actual_new_family_optimizer_calls=786))
        put(self.r/'D_FIXED_COMMON_GRAPH_FIXED_GAMMA/SUMMARY.json',dict(actual_new_family_optimizer_calls=64,actual_new_family_dispatches=72))
        put(self.r/'s3_s4_postseal_analysis/ACTUAL_COST_INVENTORY.json',dict(complete=True,old_original_D1_or_PF_production_cost_charged_again=False,
            source_bindings={},instrumented_completed_generation_all_version_call_sum={'objective_calls':225575,'DIS_calls':15124},
            instrumented_completed_generation_all_version_resource_sum={'logical_render_calls':18749}))
        jsonl(self.r/'GLOBAL_H1_FIT_ATTEMPTS.jsonl',[]);jsonl(self.r/'H1_EXECUTION_EVENTS.jsonl',[])
        put(self.r/'START.json',dict(created_utc='2026-10-19T08:02:47.125825+00:00',head='dirty'))
        put(self.r/'S1_summary.json',dict(new_individual_cost_evaluations=1352))
        put(self.r/'S1_decomp.json',dict(totals=dict(new_unique_PF_cost_evaluations=1348,old_individual_cost_reconstruction_evaluations=4)))
        cat['metadata']=[dict(name='S1',path=str(self.r/'S1_summary.json'),fields=['new_individual_cost_evaluations'],scope='test'),
            dict(name='S1_COST_DECOMPOSITION',path=str(self.r/'S1_decomp.json'),
                fields=['totals.new_unique_PF_cost_evaluations','totals.old_individual_cost_reconstruction_evaluations'],scope='test')]
        d=M.audit(cat)
        self.assertEqual(d['typed_counters']['fullnormal_runtime_minimize_calls']['known_lower_bound'],591799)
        self.assertEqual(d['typed_counters']['D_diagnostic_runtime_minimize_calls']['sum_distinct_attempts'],850)
        self.assertEqual(d['typed_counters']['S3_S4']['objective_calls'],225575)
        self.assertEqual(d['typed_counters']['S1_candidate_cost_evaluations']['old_cost_fallback_reconstruction'],4)
        self.assertTrue(d['no_cross_type_grand_call_total']);self.assertEqual(d['new_pose_optimizer_calls'],0)
        d['private_pose_payload']={'R':[[123]],'frame_id':'PRIVATE_FRAME'}
        public=M.public_projection(d,'master_hash')
        encoded=json.dumps(public)
        self.assertNotIn('PRIVATE_FRAME',encoded);self.assertNotIn(str(self.r),encoded)
        self.assertFalse(public['complete_cost_measurement'])
        self.assertEqual(public['private_master_receipt_sha256'],'master_hash')
        put(self.r/'S1_decomp.json',dict(totals=dict(new_unique_PF_cost_evaluations=1348,old_individual_cost_reconstruction_evaluations=0)))
        with self.assertRaises(ValueError):M.audit(cat)


if __name__=='__main__':unittest.main()
