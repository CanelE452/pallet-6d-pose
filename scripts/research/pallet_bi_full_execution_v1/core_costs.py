"""Unbound read-only execution-cost accounting for S2 and unconditional F core.

Parallel current costs come only from ROW_EXECUTION resource fields. Copied
FRAME_SEAL intrinsic resource fields never enter these totals. Pilot FRAME
costs are read exactly once as their own original attempt. No pose/reference,
feature evaluator, producer or fit module is imported.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

UNKNOWN='UNKNOWN_NOT_ZERO'
FROZEN_BATCH='fccf54b807d804dba42a167715eb21647160e039c1ea0d4be07380fbddb779cc'
BATCH_KEY='scripts/research/pallet_bi_full_execution_v1/fullnormal_batch.py'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def number(value):return isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)and value>=0


def call_value(value):return int(value)if number(value)and value==int(value)else None


def metric(values):
    good=[float(v)for v in values if number(v)]
    return dict(known_n=len(good),unknown_n=len(values)-len(good),known_sum=sum(good),
        total=sum(good)if values and len(good)==len(values)else UNKNOWN,
        minimum=min(good)if good else None,maximum=max(good)if good else None,
        mean=statistics.fmean(good)if good else None,median=statistics.median(good)if good else None)


def catalog(campaign):
    r=Path(campaign)
    return [dict(name='S2_INITIAL_FAILED_PILOT',component='S2',mode='technical',root=str(r/'S2_FULL_NORMAL_PILOT'),prior_S2=True,
                 terminal=str(r/'S2_FULL_NORMAL_PILOT/TECHNICAL_FAILURE.json'),expected_calls=0),
        dict(name='S2_FIXED24_PILOT',component='S2',mode='pilot',root=str(r/'S2_FULL_NORMAL_PILOT_TECHNICAL_2'),prior_S2=True,expected_calls=7490),
        dict(name='S2_CONFOUNDED_2CPU',component='S2',mode='rows',root=str(r/'S2_FULL_NORMAL'),prior_S2=True,expected_calls=37820,
             terminal=str(r/'S2_UNIT_WEIGHT_QUIESCE_COMPLETE.json'),start=str(r/'FULL_S2_GO.json'),counts_source=str(r/'FULL_S2_GO.json')),
        dict(name='S2_CORRECTED_2CPU',component='S2',mode='rows',root=str(r/'S2_FULL_UNIT_WEIGHT'),prior_S2=True,expected_calls=17497,
             terminal=str(r/'S2_THREE_CPU_QUIESCE_COMPLETE.json'),start=str(r/'S2_UNIT_WEIGHT_FULL_GO.json'),counts_source=str(r/'S2_UNIT_WEIGHT_PREPRODUCTION_REVIEW.json')),
        dict(name='S2_CORRECTED_3CPU',component='S2',mode='rows',root=str(r/'S2_FULL_UNIT_WEIGHT_THREE_CPU'),
             counts_source=str(r/'S2_UNIT_WEIGHT_PREPRODUCTION_REVIEW.json'),gates=[str(r/'S2_THREE_CPU_SCHEDULING_ROOT_GO.json'),str(r/'S2_THREE_CPU_EXECUTION_LOCK.json')]),
        dict(name='F_UNCONDITIONAL_CORE_3CPU',component='F_UNCONDITIONAL_CORE',mode='rows',root=str(r/'F_CORE_THREE_CPU'),
             counts_source=str(r/'F_CORE_PRODUCTION_ROOT_GO.json'),resume_root=str(r/'F_CORE_RESUME_INVOCATIONS'),
             gates=[str(r/'F_CORE_PRODUCTION_ROOT_GO.json'),str(r/'F_CORE_CONTROLLER_LOCK.json'),str(r/'F_CORE_RESOURCE_GATE_PASSED.json')])]


def closed_quiesce(root,lock,terminal,records,jobs,load):
    """Authenticate new all-arm drains without relabelling a two-arm F bank.

    Historical manually audited three-arm receipts retain their own contract.
    Generic receipts additionally bind dispatch, every current ROW and its
    original FRAME seal; copied FRAME intrinsic resource fields are unused.
    """
    if terminal is None or terminal.get('unfinished_inmemory_rows')!=0 or terminal.get('unknown_interrupted_optimizer_calls')!=0:
        return False
    generic=terminal.get('all_started_rows_completed_all_arms')is True
    legacy=terminal.get('all_started_rows_completed_all3arms')is True
    arms=lock.get('arms',[])
    if not arms or len(arms)!=len(set(arms))or not(generic or(legacy and len(arms)==3)):
        return False
    if terminal.get('completed_arm_rows')!=len(records):raise ValueError('Quiesce row count differs')
    grouped=defaultdict_records(records)
    if any(len(rr)!=len(arms)or{r['arm']for r in rr}!=set(arms)for rr in grouped.values()):return False
    if terminal.get('completed_jobs')!=len(grouped):raise ValueError('Quiesce completed-job count differs')
    if not generic:return True
    if (terminal.get('active_rows_drained_before_exit')is not True or terminal.get('checkpoint_stop_observed')is not True
        or terminal.get('orphan_scientific_artifacts')!=[]or terminal.get('technical_failure_receipts')!=[]):return False
    dispatch=load(root/'DISPATCH_COMPLETION.json')
    if dispatch is None:return False
    for key,name in [('summary_sha256','SUMMARY.json'),('dispatch_completion_sha256','DISPATCH_COMPLETION.json'),('producer_lock_sha256','PRODUCER_LOCK.json')]:
        if not terminal.get(key):return False
        if terminal[key]!=sha(root/name):raise ValueError('Quiesce bound producer artifact changed: '+name)
    if (dispatch.get('active_rows_drained_before_exit')is not True or dispatch.get('checkpoint_stop_observed')is not True
        or dispatch.get('summary_sha256')!=terminal['summary_sha256']):return False
    jobmap={(j['domain'],j['ordinal']):j for j in jobs}
    expected_jobs=sum(len(ids)for ids in lock.get('static_ordered_row_ids',{}).values())
    submitted=terminal.get('submitted_jobs')
    if (len(jobmap)!=len(jobs)or set(jobmap)!=set(grouped)or submitted!=len(jobs)
        or dispatch.get('submitted_jobs')!=submitted or terminal.get('unsubmitted_jobs')!=dispatch.get('unsubmitted_jobs')
        or call_value(submitted)is None or call_value(dispatch.get('unsubmitted_jobs'))is None
        or submitted+dispatch['unsubmitted_jobs']!=expected_jobs):return False
    for key,rr in grouped.items():
        if any(r['calls']is None for r in rr)or jobmap[key].get('actual_new_optimizer_calls_this_job')!=sum(r['calls']for r in rr):
            return False
    bindings=terminal.get('row_execution_bindings',[])
    indexed={b['path']:b for b in bindings}
    if len(indexed)!=len(bindings)or set(indexed)!={r['receipt_path']for r in records}:return False
    for row in records:
        path=Path(row['receipt_path']);binding=indexed[str(path)]
        if (binding.get('sha256')!=sha(path)or binding.get('frame_seal_sha256')!=sha(path.parent/'FRAME_SEAL.json')
            or binding.get('current_optimizer_calls')!=row['calls']):raise ValueError('Quiesce row binding differs')
    return True


def audit_attempt(spec):
    root=Path(spec['root']);evidence={};issues=[]
    def load(path):
        path=Path(path)
        if not path.exists():return None
        value=read(path);evidence[str(path.resolve())]=sha(path);return value
    lock=load(root/'PRODUCER_LOCK.json')or{};summary=load(root/'SUMMARY.json')
    terminal=load(spec['terminal'])if spec.get('terminal')else None
    start=load(spec['start'])if spec.get('start')else None
    counts=(load(spec['counts_source'])or{})if spec.get('counts_source')else{}
    gates={str(path):load(path)for path in spec.get('gates',[])}
    source=lock.get('numerical_kernel_source_binding',lock.get('source_binding',{}))
    records=[];unreceipted=[];partial_artifacts=[];jobs=[];failures=[];history_unknown=[]
    if spec['mode']=='technical':
        calls=call_value((terminal or{}).get('actual_pose_optimizer_calls'))
        complete=terminal is not None and calls is not None and terminal.get('completed_rows')==0
        if complete and spec.get('expected_calls')is not None and calls!=spec['expected_calls']:raise ValueError('Technical attempt cost differs from terminal contract')
        if not complete:issues.append('technical-attempt terminal cost receipt unavailable')
        result=dict(cost_complete=complete,requested_population_complete=False,status='CLOSED_BEFORE_FIRST_ROW'if complete else'INCOMPLETE',
            known_optimizer_calls=calls if number(calls)else 0,actual_optimizer_calls=calls if complete else UNKNOWN,
            represented_arm_rows=0,fresh_completed_producer_rows=0,exact_reused_rows=0,
            unknown_cost_reasons=issues,timing=dict(imports_and_setup_wall_seconds=UNKNOWN,aggregate_wall_seconds=UNKNOWN),
            actual_family_dispatches=UNKNOWN,source_implied_completed_family_dispatches=0 if complete else UNKNOWN)
        result.update(name=spec['name'],component=spec['component'],mode=spec['mode'],root=str(root),evidence=evidence)
        return result
    if spec['mode']=='pilot':
        if summary is None or not isinstance(summary.get('manifest'),list):issues.append('pilot summary/manifest unavailable')
        for item in (summary or{}).get('manifest',[]):
            path=Path(item['seal']);seal=load(path)
            if seal is None:issues.append('missing pilot frame seal: '+str(path));continue
            if seal.get('complete')is not True or seal['id']!=item['id']:raise ValueError('Pilot frame identity/complete differs')
            if seal['resource']!=item['resource']or seal['exact_reused']!=item['exact_reused']:raise ValueError('Pilot manifest resource differs')
            resource=seal['resource'];records.append(dict(arm=item['arm'],domain=item['domain'],ordinal=item['ordinal'],id=item['id'],
                calls=call_value(resource.get('actual_pose_optimizer_calls')),CPU=resource.get('CPU_seconds'),wall=resource.get('wall_seconds'),peak=None,
                exact_reused=seal['exact_reused'],staged=False,signature=seal['cache_payload_sha256'],receipt_path=str(path)))
        requested=(summary or{}).get('rows',UNKNOWN)
    else:
        expected={(arm,domain,i):fid for domain,ids in lock.get('static_ordered_row_ids',{}).items()for arm in lock.get('arms',[])for i,fid in enumerate(ids)}
        requested=len(expected)if expected else UNKNOWN
        for path in sorted(root.glob('*/*/*/ROW_EXECUTION.json')):
            row=load(path);folder=path.parent;sp=folder/'FRAME_SEAL.json';seal=load(sp)
            if seal is None:issues.append('row envelope without frame seal: '+str(path));continue
            key=(row.get('arm'),row.get('domain'),row.get('ordinal'))
            if expected and(key not in expected or expected[key]!=row.get('id')):raise ValueError('ROW_EXECUTION canonical identity differs')
            if folder.parts[-3:]!=(str(row['arm']),str(row['domain']),f"{row['ordinal']:04d}"):raise ValueError('Row folder and requested identity differ')
            if row.get('complete')is not True or seal.get('complete')is not True:issues.append('incomplete row receipt: '+str(path))
            if row['original_frame_seal_sha256']!=sha(sp)or row['cache_payload_sha256']!=seal['cache_payload_sha256']:raise ValueError('Row seal/payload binding differs')
            if row.get('numerical_kernel_source_binding')!=source or seal.get('source_binding')!=source:raise ValueError('Row numerical source differs')
            if row.get('driver_sha256')!=lock.get('driver_sha256'):raise ValueError('Row helper source differs')
            # Original arm labels are intentionally preserved by exact copies;
            # cost identity uses the requested ROW_EXECUTION arm and origin seal.
            if any(row.get(k)!=seal.get(k)for k in ('id','domain','ordinal')):raise ValueError('Row and original frame identity differs')
            resource=row.get('resource')or{};calls=call_value(resource.get('actual_new_pose_optimizer_calls'));reused=row.get('exact_reused')
            if not isinstance(reused,bool):issues.append('missing exact reuse classification: '+str(path))
            if reused is True and calls is not None and calls!=0:raise ValueError('An exact copied row cannot claim current optimizer calls')
            if reused is True and row.get('reuse_origin')=='PRIOR_SEALED_CURRENT_ROW'and row.get('pilot_reuse_proof')is None:
                issues.append('new envelope may hide an older sealed frame without its original current-cost receipt: '+str(path))
            proof=row.get('pilot_reuse_proof')
            if proof is not None:
                origin=Path(proof['source_folder'])/'FRAME_SEAL.json';original=load(origin)
                if reused is not True or original is None or sha(origin)!=proof['source_frame_seal_sha256']or sha(sp)!=sha(origin):
                    raise ValueError('Copied prior execution origin differs')
            records.append(dict(arm=row['arm'],domain=row['domain'],ordinal=row['ordinal'],id=row['id'],calls=calls,
                CPU=resource.get('current_execution_CPU_seconds'),wall=resource.get('current_execution_wall_seconds'),
                peak=resource.get('worker_peak_RSS_bytes'),exact_reused=reused,staged=row.get('pilot_reuse_proof')is not None,
                signature=row['cache_payload_sha256'],receipt_path=str(path)))
        receipted={Path(row['receipt_path']).parent for row in records}
        for path in sorted(root.glob('*/*/*/FRAME_SEAL.json')):
            if path.parent not in receipted:unreceipted.append(str(path))
        for path in sorted(root.glob('*/*/FEATURE_SEAL.json')):
            feature_seal=load(path);arm,domain=path.parent.parts[-2:]
            if feature_seal.get('complete')is not True:continue
            if feature_seal.get('producer_lock_sha256')!=sha(root/'PRODUCER_LOCK.json')or feature_seal.get('source_binding')!=source:
                raise ValueError('Completed producer feature-seal source/lock differs')
            for ordinal,digest in feature_seal.get('row_execution_sha256',{}).items():
                rp=root/arm/domain/ordinal/'ROW_EXECUTION.json'
                if not rp.exists()or sha(rp)!=digest:raise ValueError('Current cost envelope differs from completed producer feature seal')
            for ordinal,digest in feature_seal.get('frame_seal_sha256',{}).items():
                if sha(root/arm/domain/ordinal/'FRAME_SEAL.json')!=digest:raise ValueError('Original frame seal differs from completed producer feature ancestry')
        for folder in root.glob('*/*/*'):
            if folder.is_dir()and not(folder/'FRAME_SEAL.json').exists():
                partial_artifacts.extend(str(p)for p in folder.iterdir()if p.is_file())
        for path in sorted((root/'row_jobs').glob('*/*.json')):
            job=load(path)
            if path.name.endswith('_FAILURE.json'):failures.append(dict(path=str(path),record=job))
            else:jobs.append(job)
        resume_root=Path(spec['resume_root'])if spec.get('resume_root')else None
        if resume_root:
            for path in sorted(resume_root.glob('*/*.json')):
                value=load(path)
                if isinstance(value,dict)and(value.get('historical_cost_unknown_possible')or value.get('current_failed_job_optimizer_costs_unknown_possible')
                    or value.get('sealed_frames_missing_execution_envelopes')or value.get('unsealed_partial_artifacts')or path.name=='INTERRUPTED.json'):
                    history_unknown.append(str(path))
        if not lock:issues.append('producer lock unavailable')
        if unreceipted:issues.append('sealed frames missing current execution receipt')
        if partial_artifacts:issues.append('unsealed artifacts may represent unrecorded work')
        if failures:issues.append('technical job failures may contain unrecorded work')
        if history_unknown:issues.append('resume history records unresolved unknown prior work')
    keys=[(x['arm'],x['domain'],x['ordinal'])for x in records]
    if len(set(keys))!=len(keys):raise ValueError('Repeated execution identity inside one attempt')
    call_stats=metric([x['calls']for x in records]);known_calls=int(call_stats['known_sum'])
    if call_stats['unknown_n']:issues.append('optimizer-call field missing/nonfinite in execution receipt')
    done_summary=summary is not None and summary.get('status')in ('COMPLETE_NEW','DIAGNOSTIC_ONLY_TECHNICAL_PILOT')
    population_complete=done_summary and requested==len(records)and not(summary or{}).get('missing_rows')and not(summary or{}).get('technical_failures')
    quiesce_complete=closed_quiesce(root,lock,terminal,records,jobs,load)
    closed=population_complete or quiesce_complete
    if not closed:issues.append('active/incomplete producer: current in-memory or unstarted work not fully receipted')
    if terminal is not None and not quiesce_complete:issues.append('quiesce evidence does not establish zero unknown interrupted calls')
    complete=closed and not issues
    if done_summary:
        declared=summary.get('new_pose_optimizer_calls',summary.get('new_optimizer_calls'))
        if number(declared)and declared!=known_calls:raise ValueError('Producer summary current-call total differs from receipts')
    if quiesce_complete:
        declared=terminal.get('current_attempt_actual_new_optimizer_calls',terminal.get('sealed_old_attempt_new_optimizer_calls'))
        if declared is not None and declared!=known_calls:raise ValueError('Quiesce current-call total differs from receipts')
    if complete and spec.get('expected_calls')is not None and spec['expected_calls']!=known_calls:raise ValueError('Preserved historical attempt total differs')
    staged_root=None;staging=None
    if lock.get('exact_reuse_binding'):
        reuse=lock['exact_reuse_binding'];staged_root=Path(reuse['root']);staging=load(staged_root/'REUSE_MANIFEST.json')
        if staging is None or sha(staged_root/'REUSE_MANIFEST.json')!=reuse['manifest_sha256']:raise ValueError('Exact staging manifest differs')
    elif lock.get('reuse_pilot'):
        staged_root=Path(lock['reuse_pilot']);staging=load(staged_root/'REUSE_MANIFEST.json')
        if staging and start and start.get('exact_prior_reuse_manifest_sha256')and sha(staged_root/'REUSE_MANIFEST.json')!=start['exact_prior_reuse_manifest_sha256']:
            raise ValueError('Preserved staging manifest differs from start GO')
        if staging is None:
            pilot=load(staged_root/'SUMMARY.json')
            if pilot is not None:
                if start and start.get('pilot_summary_sha256')and sha(staged_root/'SUMMARY.json')!=start['pilot_summary_sha256']:raise ValueError('Original pilot summary changed')
                staging=dict(eligible_n=pilot.get('rows',UNKNOWN),scope='fixed pilot available rows, not proof every row was consumed')
    gate_waits={path:value['wait_wall_seconds']for path,value in gates.items()if isinstance(value,dict)and number(value.get('wait_wall_seconds'))}
    interval=(terminal['epoch']-start['epoch'])if terminal and start and number(terminal.get('epoch'))and number(start.get('epoch'))else UNKNOWN
    byjob=defaultdict_records(records);outside=[];outside_unknown=0
    for job in jobs:
        rr=byjob.get((job['domain'],job['ordinal']),[])
        if rr and all(number(x['wall'])and x['calls']is not None for x in rr)and number(job.get('current_job_wall_seconds')):
            difference=job['current_job_wall_seconds']-sum(x['wall']for x in rr)
            if difference>=0 and job.get('actual_new_optimizer_calls_this_job')==sum(x['calls']for x in rr):outside.append(difference)
            else:outside_unknown+=1
        else:outside_unknown+=1
    new=[x for x in records if x['exact_reused']is False];packets={(x['domain'],x['ordinal'],x['signature'])for x in records}
    job_CPU=metric([x.get('current_job_CPU_seconds')for x in jobs]);job_wall=metric([x.get('current_job_wall_seconds')for x in jobs])
    missing_job_keys=set(byjob)-{(x['domain'],x['ordinal'])for x in jobs}
    if missing_job_keys:job_CPU['total']=job_wall['total']=UNKNOWN
    declared_unique=next((counts[key]for key in ('unique_numerical_rows','expected_unique_numerical_rows','unique_complete_packets')if key in counts),UNKNOWN)
    report=dict(name=spec['name'],component=spec['component'],mode=spec['mode'],root=str(root),
        status='COST_CLOSED'if complete else'INCOMPLETE_COST_LOWER_BOUND',cost_complete=complete,requested_population_complete=bool(population_complete),
        requested_arm_rows=requested,declared_requested_unique_packets=declared_unique,
        observed_complete_packet_keys=len(packets),packet_key_scope='per original image across requested arms; no cross-image merging',
        represented_arm_rows=len(records),fresh_completed_producer_rows=len(new),exact_reused_rows=sum(x['exact_reused']is True for x in records),
        staged_available_rows=staging.get('eligible_n')if staging else UNKNOWN,staged_consumed_rows=sum(x['staged']for x in records),
        staging_root=str(staged_root)if staged_root else None,legacy_SEALED_PILOT_label='may identify staged prior production; current cost comes only from ROW_EXECUTION fields',
        known_optimizer_calls=known_calls,actual_optimizer_calls=known_calls if complete else UNKNOWN,
        unknown_cost_reasons=issues,missing_current_receipt_frames=unreceipted,unsealed_partial_artifacts=partial_artifacts,
        unresolved_resume_history=history_unknown,technical_failure_count=len(failures),technical_failures=failures,
        actual_family_dispatches=UNKNOWN,
        source_implied_completed_family_dispatches=5*len(new)if source.get(BATCH_KEY)==FROZEN_BATCH else UNKNOWN,
        family_dispatch_scope='No runtime family counter exists. For the bound fullnormal_batch source only: one P1+two WD0+two G10 dispatches per fresh completed row; G13 exact alias excluded. Partial/failed work remains unknown.',
        optimizer_call_scope='recorded minimize calls; not family dispatches, fit calls or copied historical intrinsic costs',
        timing=dict(reported_run_wall_seconds=(summary or{}).get('wall_seconds',UNKNOWN),
            reported_timer_scope='producer timer scope; controller imports before run are not separately measured',
            authorization_to_quiesce_wall_seconds=interval,resource_gate_wait_wall_seconds=gate_waits,
            arm_CPU_seconds=metric([x['CPU']for x in records]),arm_wall_seconds=metric([x['wall']for x in records]),
            job_CPU_seconds=job_CPU,job_wall_seconds=job_wall,
            job_envelopes=len(jobs),completed_image_keys=len(byjob),job_wall_outside_arm_known_sum=sum(outside),job_wall_outside_arm_envelopes=len(outside),
            job_wall_outside_arm_unattributable_envelopes=outside_unknown,
            completed_image_keys_missing_job_envelope=[dict(domain=d,ordinal=i)for d,i in sorted(missing_job_keys)],
            imports_setup_CPU_seconds=UNKNOWN,imports_setup_wall_seconds=UNKNOWN,aggregate_CPU_seconds=UNKNOWN,aggregate_wall_seconds=UNKNOWN,
            measurement_scope='arm/job statistics cover recorded completed envelopes only; unreceipted/active work is excluded and not assigned zero',
            timing_limit='parallel job/arm wall sums overlap; resumed job envelopes may reference preserved prior ROW timings; no subtraction from run wall can isolate imports/setup/aggregation'),
        peak_worker_RSS_bytes=max([x['peak']for x in records if number(x['peak'])],default=UNKNOWN),
        peak_memory_scope='maximum recorded current ROW worker ru_maxrss; unreceipted startup/failed work may have a higher unknown peak',
        longest_completed_fresh_arm_rows=sorted(new,key=lambda x:-(x['wall']if number(x['wall'])else-1))[:10],
        longest_completed_jobs=sorted([{k:j.get(k)for k in ('id','domain','ordinal','current_job_wall_seconds','current_job_CPU_seconds','actual_new_optimizer_calls_this_job')}for j in jobs],
            key=lambda x:-(x['current_job_wall_seconds']if number(x['current_job_wall_seconds'])else-1))[:10],
        by_arm_domain={key:dict(rows=len(rr),known_optimizer_calls=int(metric([x['calls']for x in rr])['known_sum']),
            missing_call_fields=metric([x['calls']for x in rr])['unknown_n'],exact_reused_rows=sum(x['exact_reused']is True for x in rr))
            for key,rr in group_arm_domain(records).items()},evidence=evidence)
    return report


def defaultdict_records(records):
    grouped={}
    for row in records:grouped.setdefault((row['domain'],row['ordinal']),[]).append(row)
    return grouped


def group_arm_domain(records):
    grouped={}
    for row in records:grouped.setdefault(row['arm']+'/'+row['domain'],[]).append(row)
    return grouped


def aggregate(specs,*,expected_preserved_prior_S2=None):
    roots=[str(Path(s['root']).resolve())for s in specs]
    if len(roots)!=len(set(roots)):raise ValueError('Each execution root must be counted exactly once, even after resumes')
    attempts=[audit_attempt(spec)for spec in specs];prior=[a for a,s in zip(attempts,specs)if s.get('prior_S2')]
    prior_known=sum(a['known_optimizer_calls']for a in prior);prior_complete=all(a['cost_complete']for a in prior)
    if expected_preserved_prior_S2 is not None and prior_complete and prior_known!=expected_preserved_prior_S2:raise ValueError('Preserved prior S2 total must appear exactly once')
    complete=all(a['cost_complete']for a in attempts)
    return dict(schema='S2_AND_UNCONDITIONAL_F_CORE_COST_ACCOUNTING_V1',status='COMPLETE_CORE_COST_ACCOUNTING'if complete else'INCOMPLETE_KNOWN_LOWER_BOUND',
        cost_complete=complete,known_optimizer_call_lower_bound=sum(a['known_optimizer_calls']for a in attempts),
        actual_optimizer_calls=sum(a['known_optimizer_calls']for a in attempts)if complete else UNKNOWN,
        preserved_prior_S2=dict(known_optimizer_calls=prior_known,actual_optimizer_calls=prior_known if prior_complete else UNKNOWN,
            expected=expected_preserved_prior_S2,attempts=[a['name']for a in prior]),attempts=attempts,
        excluded_cost_scopes=['D diagnostics and D confound repair','S3/S4 material rendering/forwards','F frontend/CAL fitting','H1 fits/predictions','earlier campaigns'],
        no_double_count_rule='Each execution root once; pilot own FRAME receipt once; every later attempt uses current ROW_EXECUTION only. Reuse staging and intrinsic copied FRAME costs add no calls.',
        campaign_wall_seconds=UNKNOWN,wall_scope='S2 and F overlap; do not sum parallel producer wall into campaign elapsed time',
        new_pose_optimizer_calls_from_this_aggregator=0,new_fits_from_this_aggregator=0,reference_files_read=0,
        actual_family_dispatches=UNKNOWN,final_F_complete=False)


def run(campaign,output,*,specs=None,catalog_path=None):
    output=Path(output);input_catalog=None
    if catalog_path is not None:
        if specs is not None:raise ValueError('Use one explicit catalog or specs, not both')
        value=read(catalog_path);attempt_specs=value['specs'];expected=value['expected_preserved_prior_S2_optimizer_calls']
        input_catalog=dict(path=str(Path(catalog_path).resolve()),sha256=sha(catalog_path),expected_preserved_prior_S2=expected)
    else:
        attempt_specs=catalog(campaign)if specs is None else specs;expected=62807 if specs is None else None
    report=aggregate(attempt_specs,expected_preserved_prior_S2=expected)
    report['input_catalog']=input_catalog
    roots=[Path(s['root']).resolve()for s in attempt_specs]
    if any(output.resolve()==root or root in output.resolve().parents for root in roots):raise ValueError('Cost output must be outside immutable producer trees')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('CORE_COST_REPORT.json',report),('COST_INPUT_SPEC.json',attempt_specs)]:
        path=output/name
        if path.exists():raise FileExistsError('Preserve previous cost snapshot; choose a fresh output')
        path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    seal=dict(complete=True,accounting_complete=report['cost_complete'],snapshot_incomplete=not report['cost_complete'],source_sha256=sha(__file__),
        input_catalog=input_catalog,
        artifacts={name:sha(output/name)for name in ('CORE_COST_REPORT.json','COST_INPUT_SPEC.json')},
        new_pose_calls=0,new_fits=0,reference_reads=0,
        completeness_scope='optimizer-call accounting only; separately unmeasured setup/aggregate timing remains UNKNOWN_NOT_ZERO')
    path=output/'COST_REPORT_SEAL.json'
    if path.exists():raise FileExistsError('Preserve previous cost seal')
    path.write_text(json.dumps(seal,indent=2,sort_keys=True)+'\n');return seal


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path,help='Explicit immutable attempt catalog with preserved-prior62807 guard')
    a=p.parse_args();print(run(a.campaign,a.output,catalog_path=a.catalog))
