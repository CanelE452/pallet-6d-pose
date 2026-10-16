"""Own-frozen stdlib audit of the complete fresh V13 runtime records.

Stored clocks, full outputs, rank/weak predicates and actual dispatch ledgers
are checked without inference, GT, fits, new Jacobians, SVD or timing calls.
Saved parity is not independent pose accuracy or hardware authentication.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
PRIVATE=Path('/tmp/pallet-finite-edge-consensus-private-20261011-v13')
SOURCE=REPO/'scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime_check.py'
SOURCE_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
METHOD='FINITE_EDGE_CONSENSUS';TIMES=('full',*STAGES,'return');POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
LIMITS=dict(new_inference_RGB_GT_PnP_optimizer_Jacobian_SVD_timing_calls=0,recorded_clock_counter_and_rank_predicate_consistency_only=True,
    candidate_pose_J_and_SVD_not_recomputed=True,physical_ownership_certified=False,global_uniqueness_proven=False,
    hardware_authenticity_proven=False,quiet_snapshots_can_miss_transient_competition=True,
    handled_optimizer_branch_exceptions_are_not_complete_frame_failure=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input')
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):digest.update(b)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=digest.hexdigest())
def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(d):return {k:v for k,v in d.items() if v}
def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'exact frozen independent scalar clock/statistics source required')
    selected=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('require','Audit','describe')]
    require({n.name for n in selected}=={'require','Audit','describe'},'exact stdlib-only scalar whitelist')
    tree=ast.Module(body=selected,type_ignores=[])
    return tree,dict(source=binding(SOURCE),names=['require','Audit','describe'],AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())
def paths(p,t):
    names=dict(geometry='GEOMETRY_SEALED.jsonl.gz',core_protocol='PROTOCOL.json',seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',
        validation='REVIEW_VALIDATION_CHECKS.json',validation_protocol='REVIEW_VALIDATION_PROTOCOL.json',validation_started='REVIEW_VALIDATION_STARTED.json',
        original_validation='VALIDATION_CHECKS.json',original_validation_protocol='VALIDATION_PROTOCOL.json',original_validation_started='VALIDATION_STARTED.json',
        cusp_contract='CUSP_REVIEW_CONTRACT_KO.md',scoring_authority='SCORING_AUTHORITY_SUPPLEMENT.json',
        ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',scoring='SCORING_RECEIPT.json',
        scoring_started='SCORING_STARTED.json',predictions='PREDICTIONS.jsonl.gz')
    runtime=dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json')
    out={k:p/v for k,v in names.items()};out.update({k:t/v for k,v in runtime.items()});out.update(checker=Path(__file__).resolve(),scalar_source=SOURCE,
        fixed_geometry=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for name in ('runtime.py','deployment.py','pipeline.py','solver.py','evaluation.py','validation_checks.py','evaluation_review.py','validation_review.py'):out['code:'+name]=Path(__file__).with_name(name)
    return out
def metadata(f,b):
    proto,r,res=read(f['runtime_protocol']),read(f['receipt']),read(f['resources'])
    require(proto['schema']=='fresh_finite_edge_primary_runtime_protocol_v13' and proto['method']==METHOD and proto['configured_calls']==150 and
        proto['warmup']==20 and proto['measured']==130 and proto['repeats']==5 and proto['stages']==list(STAGES),'fixed runtime scope differs')
    require(proto['first_prediction_detector_forwards']==2 and proto['other_prediction_detector_forwards']==1 and proto['N3_forwards_per_prediction']==1 and
        proto['saved_geometry_used_only_after_return'] is True and proto['no_artificial_pre_forward'] is True,'fresh forward policy differs')
    require(r['schema']=='fresh_finite_edge_primary_runtime_receipt_v13' and r['complete'] is True and r['status']=='COMPLETE' and
        r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'complete successful fresh runtime required')
    for key,alias in (('protocol','runtime_protocol'),('accuracy_protocol','core_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[key],b[alias]),'runtime byte authority '+key)
    for key,alias in (('runtime:code','code:runtime.py'),('runtime:deployment','code:deployment.py'),('runtime:accuracy_protocol','core_protocol'),
        ('runtime:GEOMETRY_SEALED.jsonl.gz','geometry'),('runtime:VALIDATION_CHECKS.json','original_validation'),
        ('runtime:VALIDATION_PROTOCOL.json','original_validation_protocol'),('runtime:VALIDATION_STARTED.json','original_validation_started'),
        ('runtime:REVIEW_VALIDATION_CHECKS.json','validation'),('runtime:REVIEW_VALIDATION_PROTOCOL.json','validation_protocol'),
        ('runtime:REVIEW_VALIDATION_STARTED.json','validation_started'),('runtime:SCORING_AUTHORITY_SUPPLEMENT.json','scoring_authority'),
        ('runtime:cusp_review_contract','cusp_contract'),('runtime:geometry_gate','code:evaluation_review.py'),
        ('runtime:review_checker','code:validation_review.py')):
        require(same(proto['inputs'][key],b[alias]),'frozen runtime input '+key)
    for key in ('fixed_geometry','code:solver.py','code:pipeline.py','code:evaluation.py','code:validation_checks.py','code:evaluation_review.py','code:validation_review.py'):
        require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in proto['inputs'].values()),'unbound source '+key)
    core,seal,repair,check,own,score=[read(f[k]) for k in ('core_protocol','seal','repair_receipt','validation','validation_protocol','scoring')]
    failed,first,first_started,authority=[read(f[k]) for k in ('original_validation','original_validation_protocol','original_validation_started','scoring_authority')]
    require(failed['schema']=='finite_edge_consensus_scalar_checks_v13' and failed['complete'] is True and failed['passed'] is False
        and failed['checks']==713802 and failed['failure_count']==74 and failed['exception'] is None and failed['inputs']==first['inputs'],
        'preserve complete original 713802-check/74-failure inspector')
    require(first['schema']=='finite_edge_consensus_scalar_protocol_v13' and same(failed['own_protocol'],b['original_validation_protocol'])
        and same(failed['checker'],b['code:validation_checks.py']) and first_started['inputs']==first['inputs']
        and same(first_started['own_protocol'],b['original_validation_protocol']) and first_started['new_production_calls']==0,
        'preserved first inspector authority differs')
    for key,value in first['inputs'].items():require(key in own['inputs'] and same(value,own['inputs'][key]),'review changed original input '+key)
    for key,alias in (('review_checker','code:validation_review.py'),('review_contract','cusp_contract'),
        ('original_failure','original_validation'),('original_validation_protocol','original_validation_protocol')):
        require(same(own['inputs'][key],b[alias]),'review supplement bytes differ '+key)
    marker=check['review']
    require(marker['original_checker_changed'] is False and marker['numerical_core_or_saved_rows_changed'] is False
        and marker['performance_thresholds_changed'] is False and marker['original_complete_failed_audit_preserved'] is True
        and marker['original_checks']==713802 and marker['original_failure_count']==74 and marker['AST_patch_sites']==7
        and marker['actual_roundoff_branch_reconciliations']==4 and marker['one_sided_cusp_rank_is_not_two_sided_observability_proof'] is True,
        'actual full cusp review authority required')
    require(same(marker['AST_witness']['original'],b['code:validation_checks.py']) and marker['AST_witness']['AST_patch_sites']==7,'review original AST witness differs')
    events=check['cusp_reconciliations']
    require(len(events)==4 and all(e['id']=='plastic_night_01:035421' and e['edge']==2 and e['query_id']==20
        and e['phase'] in ('proposal','refit') and e['analytic_derivative_recomputed'] is True and e['performance_threshold_changed'] is False for e in events),
        'only original roundoff cusp may be reconciled')
    require(authority['schema']=='preserved_cusp_inspector_joined_review_authority_v13' and authority['original_complete_audit_passed'] is False
        and authority['original_checks']==713802 and authority['original_failure_count']==74 and authority['separate_full_review_PASS'] is True
        and authority['numerical_core_or_saved_rows_changed'] is False and authority['original_score_statistics_kernels_changed'] is False
        and authority['GT_access_only_after_complete_geometry_cleanup_and_full_review_PASS'] is True,'scoring authority supplement differs')
    keys={'wrapper':'code:evaluation_review.py','original_score_kernel':'code:evaluation.py','original_checker':'code:validation_checks.py',
        'review_checker':'code:validation_review.py','review_contract':'cusp_contract','VALIDATION_PROTOCOL.json':'original_validation_protocol',
        'VALIDATION_STARTED.json':'original_validation_started','VALIDATION_CHECKS.json':'original_validation',
        'REVIEW_VALIDATION_PROTOCOL.json':'validation_protocol','REVIEW_VALIDATION_STARTED.json':'validation_started','REVIEW_VALIDATION_CHECKS.json':'validation'}
    require(set(authority['inputs'])==set(keys),'complete scoring authority closure required')
    for key,alias in keys.items():require(same(authority['inputs'][key],b[alias]),'scoring authority byte join '+key)
    require(core['methods']==[METHOD] and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True and repair['complete'] is True and repair['cleanup_completed'] is True and repair['error'] is None and repair['cleanup_error'] is None,'full GT-free accuracy authority')
    for key,alias in (('protocol','core_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'accuracy seal byte join '+key)
    require(check['schema']=='finite_edge_consensus_scalar_checks_v13' and check['complete'] is True and check['passed'] is True and check['inputs']==own['inputs'] and
        same(check['own_protocol'],b['validation_protocol']) and same(check['checker'],b['code:validation_checks.py']),'actual independent V13 geometry PASS')
    for alias,key in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','core_protocol'),('geometry_seal','seal'),('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','code:validation_checks.py')):require(same(own['inputs'][alias],b[key]),'actual validated bytes '+alias)
    require(read(f['validation_started'])['inputs']==own['inputs'] and same(read(f['validation_started'])['own_protocol'],b['validation_protocol'])
        and read(f['validation_started'])['new_production_calls']==0,'actual reviewed geometry inspector STARTED authority')
    scored_started=read(f['scoring_started'])
    require(same(scored_started['independent_validation'],b['validation_protocol']) and same(scored_started['geometry_seal'],b['seal'])
        and scored_started['configured_scored_rows']==245 and scored_started['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,
        'scoring follows full reviewed geometry PASS')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['scored_rows']==score['frames']==245 and score['methods']==[METHOD] and
        same(score['protocol'],b['core_protocol']) and same(score['geometry_seal'],b['seal']) and same(score['predictions'],b['predictions']),
        'complete postseal245 scoring authority')
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['all_recorded_snapshots_quiet'] is True,'resource identity')
    pre,journal=read(f['preflight']),read(f['journal']);require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and journal['status']=='COMPLETE' and
        same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'quiet preflight/completed journal')
    return proto,r,res
def guard(args,stage):
    for p in (args.input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directory required')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V13 output only')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)
def freeze(args):
    guard(args,'freeze');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,info=extraction()
    write(args.output/FILES[0],dict(schema='independent_finite_edge_runtime_check_protocol_v13',inputs=b,helpers=info,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,info=extraction()
    require(own['schema']=='independent_finite_edge_runtime_check_protocol_v13' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    proto,r,res=metadata(f,b);write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));env=dict(math=math);exec(compile(tree,str(SOURCE)+':stdlib_AST_only','exec'),env)
    a=env['Audit']();totals={};timings={k:[] for k in TIMES};count=0;error=None
    try:
        panel=proto['panel'];a.eq(len(panel),26,'panel26');a.eq(len({v['session_id'] for v in panel}),13,'sessions13');a.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session')
        authority={v['id']:v for v in proto['normalization']['joined_frame_identities']};a.eq(len(authority),245,'normalization245');a.eq(r['normalization'],proto['normalization'],'same_normalization')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];a.eq(proto['schedule'],schedule,'fixed150schedule')
        ids={v['frame_id'] for v in panel};refs={};seen=set();fixed={}
        for v in rows(f['geometry']):
            require(v['method']==METHOD and v['id'] in authority and v['id'] not in seen,'unknown/duplicate accuracy row');seen.add(v['id'])
            if v['id'] in ids:
                pose=v['actual_pose'];refs[v['id']]={k:v[k] for k in ('native_points','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')}
                refs[v['id']]['pose']={k:pose[k] for k in ('available','selected_hypothesis',*POSE,*OPTIONAL) if k in pose}
        a.eq(seen,set(authority),'all245accuracy');a.eq(set(refs),ids,'panel26references')
        for v in rows(f['fixed_geometry']):
            if v['id'] in ids:require((v['method'],v['id']) not in fixed,'duplicate fixed reference');fixed[v['method'],v['id']]=v['native_points']
        a.eq(set(fixed),{(m,i) for m in ('BASE','N3_SUBPIX') for i in ids},'fixedBASE_N3panel')
        for i,v in enumerate(rows(f['rows'])):
            require(i<150,'extra fresh row');job=schedule[i];label='row.'+str(i)
            for k,value in job.items():a.eq(v[k],value,label+'.schedule.'+k)
            p=panel[job['image_index']];a.eq((v['id'],v['session'],v['method']),(p['frame_id'],p['session_id'],METHOD),label+'.identity')
            ticks=v['stage_ticks_ns'];a.eq([t[0] for t in ticks],list(STAGES),label+'.stages');ns=[v['t0_ns'],*[t[1] for t in ticks],v['t1_ns']]
            require(all(type(t) is int for t in ns) and all(y>x for x,y in zip(ns,ns[1:])),'nonmonotonic recorded clocks')
            for k,x,y in zip((*STAGES,'return'),ns,ns[1:]):a.near(v[k+'_ms'],(y-x)/1e6,label+'.clock.'+k)
            a.near(v['full_ms'],(ns[-1]-ns[0])/1e6,label+'.full_clock');a.near(v['full_ms'],math.fsum(v[k+'_ms'] for k in (*STAGES,'return')),label+'.stage_sum')
            for k in TIMES:
                require(type(v[k+'_ms']) in (int,float) and math.isfinite(v[k+'_ms']) and v[k+'_ms']>0,'nonpositive duration')
                if job['phase']=='measured':timings[k].append(v[k+'_ms'])
            o=v['full_fresh_output'];s=o['solver'];ledger=v['fresh_repair_ledger'];H=o['hidden_initial'];native=o['native_N3_points'];reference=refs[v['id']]
            for k,value in reference.items():
                if k!='pose':a.near(o[k],value,label+'.saved_parity.'+k,relative=0)
            pose,expected=o['actual_pose'],reference['pose'];a.eq(pose['available'],expected['available'],label+'.pose_availability');a.eq(pose.get('selected_hypothesis'),expected.get('selected_hypothesis'),label+'.dimension')
            if pose['available']:
                for k in OPTIONAL:a.eq(k in pose,k in expected,label+'.optional_pose_field.'+k)
                for k in (*POSE,*(k for k in OPTIONAL if k in expected)):a.near(pose[k],expected[k],label+'.saved_Rt.'+k,relative=0)
            a.near(o['original_base_points'],fixed['BASE',v['id']],label+'.original_BASE',relative=0);a.near(native,fixed['N3_SUBPIX',v['id']],label+'.original_N3',relative=0)
            a.eq(o['K'],authority[v['id']]['K'],label+'.K');a.eq(o['xyz'],authority[v['id']]['xyz'],label+'.dimensions')
            for k,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True)):a.eq(v[k],value,label+'.'+k)
            for k,value in (('saved_accuracy_coordinates_or_pose_as_input',False),('fresh_capture_input_only',True),('fresh_point_start_recomputed_inside_call',True),('fresh_endpoint_LOO_recomputed_inside_call',True),('cached_V10_endpoint_decisions_used',False),('reprojections_reused_as_observations',False)):
                a.eq(o[k],value,label+'.fresh.'+k)
            a.eq(s,ledger['full_solver'],label+'.one_actual_solver');a.eq(o['fresh_repair_ledger'],ledger,label+'.same_full_ledger')
            a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_finite_bank');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['endpoint_decision_source'],'FRESH_NATIVE_ENDPOINT_LOO',label+'.fresh_endpoint')
            records=ledger['observation_filter']['endpoint_records'];a.eq(ledger['endpoint_validation_requests'],len(records),label+'.endpoint_requests');a.eq(ledger['endpoint_decision_records'],len(records),label+'.endpoint_returns')
            for rec in records:
                blocked=sorted(set(H)|set(rec['endpoints']));a.eq(rec['validation_excluded'],blocked,label+'.LOO_H_union_endpoints');a.eq(rec['hidden'],H,label+'.LOO_separate_H')
                a.eq(rec['native_input_hash'],ledger['endpoint_native_input_hash'],label+'.one_shared_native_bank');a.eq(rec['observed_line_supplied_to_validation_solver'],False,label+'.LOO_independence')
                a.eq(bool(set(rec['solved'].get('used',[]))&set(blocked)),False,label+'.LOO_fit_exclusion')
            a.eq(bool(set(s['used'])&set(H)),False,label+'.whole_score_pool_no_H');a.eq(s['reprojected_points_reused_as_observations'],False,label+'.no_projected_refit')
            for c in [*s['proposal_solutions'],*s['all_candidate_solutions']]:
                a.eq(bool(set(c['fit_input_ids'])&set(H)),False,label+'.all_candidate_no_H');a.eq(all(type(d) in (int,float) and math.isfinite(d) and d>0 for d in c['model_camera_depths_m']),True,label+'.stored_positive_depth')
            for c in s['all_candidate_solutions']:
                g=c['point_line_geometry'];weak=False;rank=True
                for name in ('observed','modeled'):
                    w=g[name+'_normal_joint'];ss=w['singular_values'];recount=sum(x>ss[0]*1e-10 for x in ss) if ss else 0
                    a.eq(w['numerical_rank'],recount,label+'.stored_rank_recount');flag=len(ss)!=6 or ss[-1]<ss[0]*1e-4;a.eq(g[name+'_weak'],flag,label+'.stored_weak');rank&=recount==6;weak|=flag
                stable=set(c['fit_input_ids'])<=set(c['final_inliers']) and set(c['fit_line_edges'])<=set(c['final_line_inliers'])
                reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank else 'WEAK_OBSERVABILITY' if weak else None
                a.eq(c['accepted'],reason is None,label+'.candidate_gate');a.eq(c['rejected'],reason,label+'.typed_candidate_gate')
            if o['new_pose_estimated']:
                a.eq(bool(set(H)&set(s['fit_input_ids'])),False,label+'.H_fit_excluded');a.eq(pose,s,label+'.actual_Rt');a.eq(s['best_candidate']['accepted'],True,label+'.accepted_selected_candidate')
                for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projected_once.'+str(k),1e-12)
            else:a.eq(pose,o['initial_pose'],label+'.full_original_fallback')
            a.eq(o['native_points'][8],native[8],label+'.center');expected_prediction=copy.deepcopy(v['original_fresh_detector_prediction']);selected=expected_prediction['selected_index']
            if selected is not None:a.eq(expected_prediction['candidates'][selected]['keypoints_xy'][8],native[8],label+'.raw_center');expected_prediction['candidates'][selected]['keypoints_xy']=o['native_points']
            a.eq(o['prediction']['selected_index'],selected,label+'.selected_index');a.eq(o['prediction']['candidates'],expected_prediction['candidates'],label+'.all_conf_box_preserved')
            d=v['counter_deltas'];a.eq(set(d),{'cv','optimizer','head','base','fresh','local','native','model'},label+'.counter_groups');a.eq(d['head'],dict(entries=1,returns=1),label+'.head');a.eq(d['model'],dict(detector=2 if i==0 else 1,N3=1),label+'.actual_model_forwards')
            op=ledger['local_operation_counts'];a.eq(sparse(d['local']),sparse(op),label+'.finite_work');native_delta={k:ledger['endpoint_native_bank_after'].get(k,0)-ledger['endpoint_native_bank_before'].get(k,0) for k in set(ledger['endpoint_native_bank_after'])|set(ledger['endpoint_native_bank_before'])};a.eq(sparse(d['native']),sparse(native_delta),label+'.native_bank_work')
            require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'unknown CV primitive')
            for key,source in (('entries','local_optimizer_calls'),('returns','local_optimizer_completions'),('exceptions','local_optimizer_exceptions'),('residual_entries','optimizer_residual_calls'),('jacobian_entries','optimizer_jacobian_calls')):
                a.eq(d['optimizer'].get(key,0),op[source],label+'.actual_optimizer.'+key)
            for phase in ('proposal','refit'):a.eq(op[phase+'_optimizer_calls'],op[phase+'_optimizer_returns']+op[phase+'_optimizer_exceptions'],label+'.handled_branch_exceptions.'+phase)
            for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.CV_complete.'+p)
            a.eq(d['optimizer'].get('entries',0),d['optimizer'].get('returns',0)+d['optimizer'].get('exceptions',0),label+'.optimizer_complete');a.eq(v['fresh_point_bank_ledger']['coordinate_bank_count'],1,label+'.one_fresh_sparse_bank')
            for key,values in d.items():require(all(type(n) is int and n>=0 for n in values.values()),'invalid counter');totals.setdefault(key,Counter()).update(values)
            count+=1
        a.eq(count,150,'all150');a.eq(len(timings['full']),130,'measured130')
        for k in TIMES:a.near(env['describe'](timings[k]),r['summaries'][k],'summary.'+k)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('local','local_operation_counts'),('native','native_operation_counts')):a.eq(sparse(totals[key]),sparse(r[field]),'actual.sum.'+key)
        a.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'actual.loading_forwards');a.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');a.eq(r['model_forwards'],dict(detector=151,N3=150,detector_internal_initialization_calls=1),'actual.internal_init_accounting')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('image_decode_returns',26),('model_initializations_started',1),('model_initializations_complete',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'missing resource snapshots')
        for snap in res['snapshots']:a.eq(snap['quiet'] and snap['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0
    write(args.output/FILES[2],dict(schema='independent_finite_edge_runtime_checks_v13',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,
        checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True))
    print('FINITE_EDGE_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--runtime-input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
