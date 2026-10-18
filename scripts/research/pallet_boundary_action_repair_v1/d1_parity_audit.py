"""Reference-free independent accounting of stored D1 executions and aliases.

This audit does not import the producer, feature builder, pose solver, or
accuracy evaluator. It checks the stored prediction-only stream directly.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,resource,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def equal(a,b,scale):
    return bool(a.get('available',True)and b.get('available',True)
        and np.max(np.abs(np.asarray(a['R_physical'])-b['R_physical']))<=1e-7
        and np.max(np.abs(np.asarray(a['centroid'])-b['centroid']))<=1e-7*scale)
def proper_rotation(matrix):
    a=np.asarray(matrix,float)
    return bool(a.shape==(3,3)and np.isfinite(a).all()and np.max(np.abs(a.T@a-np.eye(3)))<=1e-6 and abs(np.linalg.det(a)-1)<=1e-6)
def quantiles(values):
    if not values:return dict(min=None,mean=None,P50=None,P90=None,P95=None,max=None,sum=None)
    return dict(min=min(values),mean=float(np.mean(values)),P50=float(np.quantile(values,.5)),
        P90=float(np.quantile(values,.9)),P95=float(np.quantile(values,.95)),max=max(values),sum=sum(values))


def audit(private,old_run,output):
    started,cpu=time.monotonic(),time.process_time()
    root=Path(private);old=Path(old_run);results={};failures=[]
    def check(value,label,frame=None):
        if not value:failures.append(dict(check=label,id=frame))
    source_bindings={d:read(root/('d1_'+d+'_candidates_v1')/'FEATURE_SEAL.json')['source_sha256']for d in('real','synthetic')}
    check(source_bindings['real']==source_bindings['synthetic'],'both_domains_frozen_producer_source_closure_identical')
    repository=Path(__file__).resolve().parents[3]
    source_file_checks={path:sha(repository/path)==expected for path,expected in source_bindings['real'].items()}
    check(all(source_file_checks.values()),'actual_frozen_producer_source_files_still_match_SHA')
    modelbindings={d:read(root/('d1_'+d+'_observations_v1')/'OBSERVATION_SEAL.json')['model_bindings']for d in('real','synthetic')}
    check(modelbindings['real']==modelbindings['synthetic'],'both_domains_actual_model_descriptors_and_weights_identical')
    modelchecks={path:sha(path)==expected for path,expected in modelbindings['real'].items()}
    check(all(modelchecks.values()),'actual_loaded_model_files_still_match_frozen_SHA')
    amendment=read(root/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json')
    evidence={name:sha(root/name)==expected for name,expected in amendment['evidence_sha256'].items()}
    check(all(evidence.values()),'frozen_cache_evidence_bytes')
    stage=read(root/'D1_EXISTING_FIXED_E_STAGE_AUDIT.json');sourceaudit=read(root/'D1_REMAINDER_OLD_SOURCE_CLOSURE_AUDIT.json')
    equivalents=read(root/'D1_REMAINDER_CACHE_EQUIVALENCE_AUDIT.json');original_path=Path(next(iter(stage['source_bindings']))).parent
    typed=old.parent/'pallet-statewise-correspondence-diagnosis-20261016-v1';oldobs=original_path.parent/'fresh_observations_v4_14'
    check(all(sha(path)==expected for path,expected in stage['source_bindings'].items()),'original_prediction_archive_bytes')
    check(sha(original_path/'PREDICTION_SEAL.json')==stage['original_prediction_seal_sha256'],'original_prediction_seal_bytes')
    check(sha(original_path/'PROTOCOL.json')==sourceaudit['original_protocol_sha256'],'original_protocol_bytes')
    check(sha(oldobs/'OBSERVATION_SEAL.json')==equivalents['old_observation_seal_sha256'],'original_observation_seal_bytes')
    check(sha(oldobs/'OBSERVATIONS.jsonl.gz')==read(oldobs/'OBSERVATION_SEAL.json')['observations_sha256'],'original_observation_archive_bytes')
    extraction=read(typed/'e_trace_v2/EXTRACTION_SEAL.json');packetpath=typed/'e_trace_v2/TYPED_PACKETS.jsonl.gz'
    check(sha(packetpath)==extraction['files']['TYPED_PACKETS.jsonl.gz']['sha256'],'original_sealed_typed_packet_archive_bytes')
    expected_slots={r['id']:r['typed_slot_sha256']for r in stage['rows']}
    original_bindings={ident:dict(original_prediction_seal_sha256=stage['original_prediction_seal_sha256'],
        original_protocol_sha256=sourceaudit['original_protocol_sha256'],typed_slot_sha256=value)for ident,value in expected_slots.items()}
    for r in rows(oldobs/'OBSERVATIONS.jsonl.gz'):
        if r['id']in original_bindings:original_bindings[r['id']]['old_observation_row_sha256']=digest(r)
    for r in rows(packetpath):
        if r['id']in original_bindings:original_bindings[r['id']]['typed_packet_row_sha256']=digest(r)
    slotorder=read(typed/'REPLAY_PROTOCOL.json')['ordered_ids']
    for ident,value in expected_slots.items():
        slotpath=typed/'slots'/f'{slotorder.index(ident):03d}_TYPED_MULTI.json'
        check(sha(slotpath)==value,'original_fixed_MULTI_slot_bytes',ident)
        check(read(slotpath)['packet_sha256']==original_bindings[ident].get('typed_packet_row_sha256'),'frozen_original_MULTI_binds_actual_typed_packet_row',ident)
    journal={};journal_paths=[root/'D1_SHARED_RUNTIME_JOURNAL.jsonl'];active_runtime=root;recovery_checks={};journal_row_total=0
    if(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json').exists():
        recovery=read(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json');active_runtime=Path(recovery['recovery_private_root'])
        final_recovery=read(root/'D1_RECOVERY_FINALIZATION_RECEIPT.json');launch=read(root/'D1_RECOVERY_LAUNCH_LOCK.json')
        interruption=read(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json');supervisor=read(root/'D1_RECOVERY_SUPERVISOR_RECEIPT.json')
        original_runtime_lock=read(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json');new_runtime_lock=read(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json')
        new_runtime_receipt=read(active_runtime/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')
        recovery_checks=dict(original_lock_unchanged=sha(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json')==recovery['original_lock_sha256'],
            original_journal_unchanged=sha(root/'D1_SHARED_RUNTIME_JOURNAL.jsonl')==recovery['original_journal_sha256'],
            original_log_unchanged=sha(root/'D1_SHARED_FIXED_E_LIVE.log')==interruption['original_log_sha256'],
            original_interruption_receipt_bound=sha(root/'D1_INTERRUPTED_SHARED_EXECUTION_RECEIPT.json')==recovery['interrupted_execution_receipt_sha256'],
            original_source_exact_same_as_final=source_bindings['real']==recovery['frozen_original_source_sha256'],
            both_attempt_locks_same_frozen_source=original_runtime_lock['source_sha256']==new_runtime_lock['source_sha256']==recovery['frozen_original_source_sha256'],
            restart_unique_batch_id=original_runtime_lock['execution_batch_id']!=new_runtime_lock['execution_batch_id']==new_runtime_receipt['execution_batch_id']==final_recovery['recovery_execution_batch_id'],
            final_runtime_receipt_binds_own_lock=new_runtime_receipt['execution_lock_sha256']==sha(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json'),
            preserved_692_exact_cardinality=new_runtime_lock['domain_inputs']['real']['preserved_completed_frames']==238 and new_runtime_lock['domain_inputs']['synthetic']['preserved_completed_frames']==454,
            recovery_finalization_and_launch_bind_amendment=final_recovery['recovery_technical_amendment_lock_sha256']==launch['recovery_technical_amendment_lock_sha256']==sha(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json'),
            actual_recovery_exit_zero=supervisor['child_actual_exit_code']==0,
            recovery_finalization_complete=final_recovery['status']=='COMPLETE'and final_recovery['interrupted_lock_and_journal_unchanged'],
            canonical_runtime_receipt_exact_copy=sha(root/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')==sha(active_runtime/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')==final_recovery['recovery_runtime_receipt_sha256'],
            recovery_lock_and_journal_bound=sha(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json')==final_recovery['recovery_runtime_lock_sha256']and sha(active_runtime/'D1_SHARED_RUNTIME_JOURNAL.jsonl')==final_recovery['recovery_runtime_journal_sha256'],
            recovery_launcher_source_current=recovery['recovery_launcher_source_sha256']==sha(repository/'scripts/research/pallet_boundary_action_repair_v1/d1_recovery_batch.py'),
            detached_launcher_sources_current=all(sha(repository/'scripts/research/pallet_boundary_action_repair_v1'/name)==value for name,value in launch['source_sha256'].items()),
            partial_interrupted_numeric_counts_not_fabricated=all(r['optimizer_calls']is None and r['PnP_calls']is None and r['CPU_seconds']is None and r['wall_seconds']is None for r in interruption['unsealed_attempts']))
        for domain,items in recovery['completed_frames_reused'].items():
            for index,value in items.items():
                check(sha(root/('d1_'+domain+'_candidates_v1')/'frames'/index/'FRAME_SEAL.json')==value,'interrupted_completed_frame_bytes_preserved',domain+'/'+index)
        for attempt in recovery['unfinished_attempts_preserved']:
            directory=Path(attempt['archived_path'])
            check(directory.is_dir()and{str(p.relative_to(directory)):sha(p)for p in directory.rglob('*')if p.is_file()}==attempt['file_sha256'],'unsealed_interrupted_attempt_archive_preserved',str(attempt['index']))
        for label,value in recovery_checks.items():check(value,'technical_recovery/'+label)
        journal_paths.append(active_runtime/'D1_SHARED_RUNTIME_JOURNAL.jsonl')
    for path in journal_paths:
        own_lock=read(path.parent/'D1_SHARED_EXECUTION_BATCH_LOCK.json')
        own_pending={(r['domain'],r['index']):r for r in own_lock['ordered_pending']};own_keys=set()
        for line in path.read_text().splitlines():
            if line.strip():
                journal_row_total+=1
                r=json.loads(line);key=(r['domain'],r['index']);check(key not in journal,'unique_shared_journal_key',r['id']);journal[key]=r
                own_keys.add(key);plan=own_pending.get(key)
                check(plan is not None and plan['id']==r['id']and plan['kind']==r['execution_kind'],'journal_row_matches_own_batch_plan',r['id'])
        if path.parent!=root:check(own_keys==set(own_pending),'restarted_journal_complete_own_batch')
    pending=read(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json')['ordered_pending']
    check(set(journal)=={(r['domain'],r['index'])for r in pending},'complete_shared_pending_journal')
    if recovery_checks:
        recovery_checks['journal_union_disjoint']=len(journal)==journal_row_total
        recovery_checks['journal_union_exact_original_pending']=set(journal)=={(r['domain'],r['index'])for r in pending}
    preserved=set(amendment['validated_completed_original_frame_indices'])|{read(root/'D1_CACHE_PARITY_FIXED_SELECTION.json')['index']}
    global_input_and_archive_checks_passed=not failures
    for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        domain_failure_start=len(failures)
        op=root/('d1_real_observations_v1'if domain=='real'else'd1_synthetic_observations_v1')
        cp=root/('d1_real_candidates_v1'if domain=='real'else'd1_synthetic_candidates_v1')
        upstream=list(rows(root/'d1_upstream_v1'/(name+'_UPSTREAM.jsonl.gz')))
        seal=read(cp/'FEATURE_SEAL.json');oseal=read(op/'OBSERVATION_SEAL.json')
        features=list(rows(cp/'HEAD_FEATURES.jsonl.gz'))
        ids=[r['id']for r in upstream]
        check(seal['complete']and oseal['complete'],'complete_seals/'+domain)
        check(seal['execution_proof']['shared_runtime_lock_sha256']==sha(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json'),'actual_final_runtime_lock/'+domain)
        original_inputs=read(root/'D1_SHARED_EXECUTION_BATCH_LOCK.json')['domain_inputs'][domain]
        active_inputs=read(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json')['domain_inputs'][domain]
        check(seal['input_sha256']==sha(root/'d1_upstream_v1'/(name+'_UPSTREAM.jsonl.gz'))==oseal['input_sha256']==original_inputs['inputs_sha256']==active_inputs['inputs_sha256'],'original_and_restarted_input_bytes_identical/'+domain)
        check(seal['observation_seal_sha256']==sha(op/'OBSERVATION_SEAL.json')==original_inputs['observation_seal_sha256']==active_inputs['observation_seal_sha256'],'original_and_restarted_observation_bytes_identical/'+domain)
        check(seal['head_features_sha256']==sha(cp/'HEAD_FEATURES.jsonl.gz'),'feature_bytes/'+domain)
        check([f['id']for f in features]==ids==seal['ordered_ids']==oseal['ordered_ids'],'exact_ids/'+domain)
        counts=Counter();state=Counter();sources=Counter();calls=Counter();equality=Counter();ledger={};stage_outcomes=Counter();service=defaultdict(list)
        for i,(u,f)in enumerate(zip(upstream,features,strict=True)):
            obs=next(rows(op/'frames'/f'{i:04d}'/'OBSERVATION.jsonl.gz'))
            obsproof=read(op/'frames'/f'{i:04d}'/'OBSERVATION_SEAL.json')
            execute=next(rows(cp/'frames'/f'{i:04d}'/'E_RUN.jsonl.gz'))
            proof=read(cp/'frames'/f'{i:04d}'/'FRAME_SEAL.json')
            ident=u['id'];spec=execute['packet']['solver_input'];scale=float(f['object_scale_m'])
            ledger[f'frames/{i:04d}/FRAME_SEAL.json']=sha(cp/'frames'/f'{i:04d}'/'FRAME_SEAL.json')
            check(proof['e_run_sha256']==sha(cp/'frames'/f'{i:04d}'/'E_RUN.jsonl.gz'),'actual_e_bytes',ident)
            check(proof['features_sha256']==sha(cp/'frames'/f'{i:04d}'/'FEATURES.jsonl.gz'),'actual_feature_bytes',ident)
            check(f==next(rows(cp/'frames'/f'{i:04d}'/'FEATURES.jsonl.gz')),'aggregate_feature_row_equals_executed_frame',ident)
            check(proof['observation_seal_sha256']==sha(op/'frames'/f'{i:04d}'/'OBSERVATION_SEAL.json'),'actual_observation_receipt',ident)
            check(obsproof['observation_sha256']==sha(op/'frames'/f'{i:04d}'/'OBSERVATION.jsonl.gz'),'actual_observation_bytes',ident)
            check(obsproof['mask_sha256']==sha(op/'frames'/f'{i:04d}'/'MASK.npz'),'actual_RGB_mask_bytes',ident)
            check(f['id']==ident==obs['id']==execute['id'],'frame_ID',ident)
            check(proof['e0_independently_executed']and execute['proof']['e0_independently_executed'],'actual_e_execution',ident)
            frozen={k:v for k,v in seal['source_sha256'].items()if not k.endswith(('/d1_stage_memoization.py','/d1_memoized_batch.py'))}
            check(all(proof['source_sha256'].get(k)==v for k,v in frozen.items()),'unchanged_fixed_e_source_closure',ident)
            cached=execute.get('old_full_run_reused',False)
            service['exact_memo'if cached else'actual_new'].append(proof['resource'])
            runtime=journal.get((domain,i))
            if runtime:
                check(runtime.get('status')!='FAILED_TECHNICAL'and runtime['id']==ident,'shared_journal_successful_frame',ident)
                check(runtime.get('execution_kind')==('EXACT_ORIGINAL_STAGE_MEMOIZATION'if cached else'NEW_UNCHANGED_E_SOLVE'),'journal_execution_kind_matches_actual_proof',ident)
            else:check(domain=='real'and i in preserved and not cached,'previous_completed_actual_row_admitted',ident)
            if cached:
                memo=execute['proof'].get('exact_stage_memoization',{})
                check(memo.get('only_original_fixed_P1_WD0_G10_G13')is True and memo.get('history_only_union')is False,'one_original_fixed_stage_run',ident)
                check(memo.get('E0_previously_independently_executed')is True and bool(memo.get('original_execution_bindings')),'actual_old_execution_receipts',ident)
                check(bool(memo.get('input_checks'))and all(c['exact']for c in memo['input_checks']),'exact_stage_input_checks',ident)
                check(memo.get('original_execution_bindings')==original_bindings.get(ident),'original_stage_and_packet_observation_hashes_match_frozen_evidence',ident)
                check(all(v==0 for v in proof['operations'].values())and memo.get('new_optimizer_or_PnP_calls')==0,'memoization_no_new_numerical_calls',ident)
            counts['exact_original_stage_memoized_frames']+=bool(cached)
            counts['new_unchanged_e_execution_frames']+=not cached
            check(not execute['proof']['N0_copied_as_E0']and not execute['proof']['history_only_union'],'no_fake_E0_or_history_union',ident)
            check(spec['states']==obs['states'],'states_preserved',ident)
            check(equal(f['fallback'],u['initial_pose'],scale),'N0_value_parity',ident)
            check(proper_rotation(u['initial_pose']['R_cf'])and proper_rotation(u['initial_pose']['R_physical']),'actual_initial_rotations_proper',ident)
            check(proper_rotation(execute['e0_regenerated']['R_physical']),'actual_E0_rotation_proper',ident)
            ealias=[c for c in f['candidates']if'E0'in c['source_baseline_aliases']]
            nalias=[c for c in f['candidates']if'N0'in c['source_baseline_aliases']]
            check(len(ealias)==1 and equal(ealias[0]['actual_pose'],execute['e0_regenerated'],scale),'one_E0_alias_actual_pose',ident)
            check(len(nalias)==1 and equal(nalias[0]['actual_pose'],u['initial_pose'],scale),'one_N0_alias_actual_pose',ident)
            for c in f['candidates']:
                x=np.asarray(c['features']);check(x.shape==(62,)and np.isfinite(x).all()and np.isin(x[31:],[0.,1.]).all(),'old62feature_contract',ident)
            hidden=obs['_original_source']['initial_geometry']['initial_self_hidden']
            check(all((s=='S')==bool(hidden[k])for k,s in enumerate(obs['states'])),'S_geometry_preserved',ident)
            check(not any(hidden[m['native_corner']]for m in spec['visible']),'no_S_image_measurement',ident)
            check(not any(hidden[m['native_corner']]for m in spec['boundary_observations']),'no_S_partial',ident)
            check(not any(hidden[m['native_corner']]for m in spec['domains']+spec['u_regions']),'no_S_domain',ident)
            actual=obsproof['resource']['actual_calls']
            classifier=actual.get('state_classifier_head_calls')
            check(isinstance(classifier,int)and not isinstance(classifier,bool),'required_classifier_counter_present',ident)
            counts['missing_classifier_counter_frames']+=classifier is None
            counts['classifier_executed_frames']+=classifier is not None and classifier>0
            counts['gated_no_classifier_forward_frames']+=classifier==0
            counts['O_frames']+='O'in obs['states'];state.update(obs['states']);calls.update(proof['operations'])
            counts['frames']+=1;counts['candidates']+=len(f['candidates']);counts['eligible']+=sum(c['eligible']for c in f['candidates'])
            counts['raw_candidates']+=f['candidate_sources_before_dedup'];counts['projection_rejections']+=len(f['projection_rejections'])
            counts['logged_G10_G13_endpoint_records']+=len(execute['endpoint_journal'])
            counts['unsuccessful_logged_G10_G13_endpoints']+=sum(e['original_endpoint_record'].get('optimizer_success')is False for e in execute['endpoint_journal'])
            counts['actual_backend_return_diagnostic_records']+=sum(len(c['returned_original_solution'].get('candidates',[]))for c in execute['stage_execution_diagnostics']['P1_and_WD_backend_calls'])
            counts['actual_family_return_diagnostic_records']+=sum(len(c['solution'].get('candidates',[]))for stage in execute['stage_execution_diagnostics']['family_stage_calls']for c in stage['families'])
            for backend in execute['stage_execution_diagnostics']['P1_and_WD_backend_calls']:
                label=('P1'if backend['call_index']==0 else'WD0'if backend['call_index']in(1,2)else'ADDITIONAL_PARTIAL_BACKEND_RETURN_STAGE_UNATTRIBUTED')
                stage_outcomes[label]+=len(backend['returned_original_solution'].get('candidates',[]))
            for stage in execute['stage_execution_diagnostics']['family_stage_calls']:
                if stage['stage']!='WD0_SEEDS':
                    stage_outcomes[stage['stage']]+=sum(len(c['solution'].get('candidates',[]))for c in stage['families'])
            equality['E0_equals_N0']+=equal(execute['e0_regenerated'],u['initial_pose'],scale)
            equality['E0_distinct_from_N0']+=not equal(execute['e0_regenerated'],u['initial_pose'],scale)
            sources.update(c['source']for c in execute['candidates'])
        results[domain]=dict(complete=True,rows=len(features),statistics=dict(counts),states=dict(state),
            actual_CPU_operations=dict(calls),candidate_source_counts=dict(sources),N0_E0_pose_identity=dict(equality),
            head_features_sha256=sha(cp/'HEAD_FEATURES.jsonl.gz'),feature_seal_sha256=sha(cp/'FEATURE_SEAL.json'),
            observation_seal_sha256=sha(op/'OBSERVATION_SEAL.json'),real_or_synthetic_new_accuracy_used=False)
        execution_receipt=dict(schema='D1_ACTUAL_E_EXECUTION_RECEIPT_V1',complete=True,domain=domain,
            independent_execution_checks_passed=global_input_and_archive_checks_passed and len(failures)==domain_failure_start,
            ordered_ids=ids,rows=len(features),e0_independently_executed=True,
            input_sha256=seal['input_sha256'],source_sha256=seal['source_sha256'],
            BATCH_LOCK_sha256=sha(cp/'BATCH_LOCK.json'),feature_seal_sha256=sha(cp/'FEATURE_SEAL.json'),
            head_features_sha256=sha(cp/'HEAD_FEATURES.jsonl.gz'),observation_seal_sha256=sha(op/'OBSERVATION_SEAL.json'),
            per_frame_execution_seal_sha256=ledger,actual_operations=dict(calls),N0_E0_pose_identity=dict(equality),
            actual_new_vs_exact_memo_frames=dict(new_unchanged_e=counts['new_unchanged_e_execution_frames'],exact_original_memo=counts['exact_original_stage_memoized_frames']),
            original_returned_optimizer_outcome_records_by_stage=dict(stage_outcomes),
            producer_service_resource_by_execution_mode={mode:dict(rows=len(service[mode]),
                wall_seconds=quantiles([r['wall_seconds']for r in service[mode]]),
                CPU_seconds=quantiles([r['CPU_seconds']for r in service[mode]]))for mode in('actual_new','exact_memo')},
            producer_service_timer_scope='e + original 62-feature scoring + output serialization/provenance combined, after worker preload; optimizer-versus-feature stage time was not measured. Includes original partial2-worker, actual parity1-worker and shared4-worker contexts; not a fresh deployment latency benchmark.',
            diagnostic_count_scope='Returned stage outcome views, including prior executions reused by exact memoization; a nested partial backend return may repeat a family-return record, so views must not be summed as unique optimizer executions. First P1/two WD0 calls are identified by original call order; later partial returns are stage-unattributed. Actual new minimize/PnP call counts are separately hooked.',
            frozen_cache_evidence_checks=evidence,original_archive_bindings_checked=True,
            original_archive_bindings_checks_passed=global_input_and_archive_checks_passed,
            actual_model_bindings_identical_both_domains=modelbindings['real']==modelbindings['synthetic'],
            actual_model_bindings_sha256=digest(modelbindings['real']),actual_model_files_SHA_verified=all(modelchecks.values()),
            actual_producer_source_files_SHA_verified=all(source_file_checks.values()),actual_producer_source_file_count=len(source_file_checks),
            typed_packet_extraction_seal_sha256=sha(typed/'e_trace_v2/EXTRACTION_SEAL.json'),
            actual_visited_endpoint_records=counts['logged_G10_G13_endpoint_records'],
            original_failed_or_unsuccessful_records_preserved=True,reference_or_accuracy_reads=0,
            verifier_source_sha256=sha(__file__))
        if recovery_checks:
            execution_receipt['technical_interruption_recovery_checks']=recovery_checks
            execution_receipt['recovery_technical_amendment_lock_sha256']=sha(root/'D1_RECOVERY_TECHNICAL_AMENDMENT_LOCK.json')
            execution_receipt['domain_rows_across_batches_same_frozen_source_inputs_observations']=global_input_and_archive_checks_passed and len(failures)==domain_failure_start
        if(cp.parent/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json').exists():
            execution_receipt['shared_execution_batch_receipt_sha256']=sha(cp.parent/'D1_SHARED_EXECUTION_BATCH_RECEIPT.json')
            execution_receipt['shared_execution_batch_lock_sha256']=sha(active_runtime/'D1_SHARED_EXECUTION_BATCH_LOCK.json')
            execution_receipt['cache_technical_amendment_lock_sha256']=sha(cp.parent/'D1_CACHE_TECHNICAL_AMENDMENT_LOCK.json')
        receiptpath=cp/'D1_EXECUTION_RECEIPT.json'
        if receiptpath.exists():
            if read(receiptpath)!=execution_receipt:raise ValueError('Preserve changed independent receipt')
        else:receiptpath.write_text(json.dumps(execution_receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        results[domain]['D1_EXECUTION_RECEIPT_sha256']=sha(receiptpath)
    result=dict(schema='D1_REFERENCE_FREE_PARITY_AUDIT_V1',status='PASS'if not failures else'FAIL',
        independent_code_scope='stored schema, arithmetic counts, actual E0 source aliases, exact ID coverage and initial S exclusion; no producer imports',
        domains=results,failures=failures,common_real_bank='one D1 real238 HEAD_FEATURES byte stream for every 2x2 model',
        algorithm_procedure_matched=not failures,sim_to_real_image_distribution_solved=False,
        actual_model_bindings_identical_both_domains=modelbindings['real']==modelbindings['synthetic'],
        actual_model_bindings_sha256=digest(modelbindings['real']),actual_model_files_SHA_verified=all(modelchecks.values()),
        actual_model_file_count=len(modelchecks),
        actual_producer_source_files_SHA_verified=all(source_file_checks.values()),actual_producer_source_file_count=len(source_file_checks),
        technical_interruption_recovery_checks=recovery_checks,
        execution_journal_coverage=dict(attempt_journal_count=len(journal_paths),journal_rows=journal_row_total,
            distinct_domain_index_rows=len(journal),original_planned_pending_rows=len(pending),duplicate_rows=journal_row_total-len(journal)),
        all238_and1003_across_batches_same_frozen_source_inputs_observations=not failures,
        physical_owner='UNVERIFIED',physical_6D_GT='UNVERIFIED',source_sha256=sha(__file__),
        audit_wall_seconds=time.monotonic()-started,audit_CPU_seconds=time.process_time()-cpu,
        max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--old-run',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit(a.private,a.old_run,a.output),ensure_ascii=False))
