"""Own-frozen stdlib audit of the V19 complete fresh PRIMARY runtime.

Stored clocks/counters/actual candidate predicates and parity only. The
separate full V19 geometry audit owns independent projection/J arithmetic.
No model, image, GT, PnP, optimizer, projection, J, SVD or timing is invoked.
The original complete V16 scalar audit remains FAIL, not reclassified.
"""
import argparse
import ast
from collections import Counter
import copy
import gzip
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import traceback

PRIVATE=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
METHOD='HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS'
SOLVER='OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS'
METHODS=(METHOD,'HYBRID_OBSERVED_FRAME_POINT_ONLY')
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
TIMES=('full',*STAGES,'return');POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
MATH_SOURCES={
 'clock_source':('pallet_endpoint_observability_repair_20261011_v11/runtime_check.py','0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11',('require','Audit','describe')),
 'predicate_source':('pallet_factor_subset_consensus_20261011_v15/runtime_check.py','afc21b2ec0c9f2c49b8f05669a9e697ea2a83767767f66d448e15085c9c4f478',('proposal_witnesses',))}
LIMITS=dict(new_GT_RGB_model_training_PnP_optimizer_projection_J_SVD_timing_calls=0,
 saved_clock_counter_candidate_predicate_and_output_consistency_only=True,
 candidate_projection_J_or_SVD_not_recomputed=True,stored_spectrum_recount_is_not_SVD=True,
 fixed_observed_frame_candidate_numeric_geometry_owned_by_separate_full_audit=True,
 runtime_unselected_candidate_residual_J_values_not_independently_recomputed=True,
 original_full_V16_scalar_validation_passed=False,original564FAIL_preserved=True,
 physical_ownership_or_global_uniqueness_certified=False,hardware_authenticity_proven=False,
 quiet_snapshots_can_miss_transient_contention=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,set):return [safe(x) for x in sorted(v,key=str)]
    return repr(v) if isinstance(v,float) and not math.isfinite(v) else v
def write(p,v):
    with Path(p).open('x') as f:json.dump(safe(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(d):return {k:v for k,v in d.items() if v}

def declared_path(args,key,v):
    """Only actual direct own-protocol declarations; no history recursion."""
    if v.get('origin')=='external_readonly_dependency':return None
    p=Path(v['path'])
    if v.get('origin')=='public_repository':
        require(not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public input '+key)
        return args.repo/p
    require(v.get('origin') is None and p.is_absolute() and (p.is_relative_to(args.repo) or p.is_relative_to(args.private)),
        'explicit private/research absolute binding '+key)
    return p

def paths(args):
    g,v,t=args.input,args.validation_input,args.runtime_input
    out={k:g/n for k,n in dict(geometry='GEOMETRY.jsonl.gz',ledgers='LEDGERS.jsonl.gz',geometry_protocol='GEOMETRY_PROTOCOL.json',geometry_started='GEOMETRY_STARTED.json',geometry_receipt='GEOMETRY_RECEIPT.json').items()}
    out.update({k:v/n for k,n in dict(validation='VALIDATION_CHECKS.json',validation_protocol='VALIDATION_PROTOCOL.json',validation_started='VALIDATION_STARTED.json').items()})
    out.update({k:t/n for k,n in dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json').items()})
    out.update(checker=Path(__file__).resolve(),runtime_source=args.private/'runtime.py',geometry_checker=args.private/'validation.py',ready_driver=args.private/'run_geometry_ready.py',original_driver=args.private/'run_geometry.py',serialization_ready_proof=args.private/'GEOMETRY_READY_PROOF.json',fixed_geometry=args.repo/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for n in ('GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json','GEOMETRY_RECEIPT.json'):out['first:'+n]=args.private/'geometry'/n
    for key,(relative,_,_) in MATH_SOURCES.items():out[key]=args.repo/'scripts/research'/relative
    for label,alias in (('runtime_input','runtime_protocol'),('validation_input','validation_protocol')):
        for key,value in read(out[alias])['inputs'].items():
            target=declared_path(args,key,value)
            if target is not None:out[label+':'+key]=target
    return out

def extraction(f):
    nodes=[];info=[]
    for key,(_,sha,names) in MATH_SOURCES.items():
        require(binding(f[key])['sha256']==sha,'unchanged independent source '+key)
        selected=[n for n in ast.parse(f[key].read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
        require({n.name for n in selected}==set(names),'exact independent AST whitelist '+key)
        tree=ast.Module(body=selected,type_ignores=[]);nodes+=selected
        info.append(dict(source=binding(f[key]),names=list(names),AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),formula_changes=0))
    return ast.Module(body=nodes,type_ignores=[]),info

def declarations(proto):return {k:v for k,v in proto['inputs'].items() if v.get('origin')=='external_readonly_dependency'}

def metadata(f,b):
    p,r,res=[read(f[k]) for k in ('runtime_protocol','receipt','resources')]
    require(p['schema']=='fresh_observed_frame_primary_runtime_protocol_v19' and p['method']==METHOD and p['configured_calls']==150 and p['warmup']==20 and p['measured']==130 and p['stages']==list(STAGES),'fixed fresh150 primary protocol')
    for key,value in (('first_predict_detector_forwards',2),('other_predict_detector_forwards',1),('N3_forwards_per_prediction',1),('geometry_used_only_after_return',True),('paired_point_only_accuracy_arm_not_timed',True),('no_artificial_pre_forward',True),('automatic_retry',False),('new_RGB',0),('new_training',0)):require(p[key]==value,'runtime policy '+key)
    require(r['schema']=='fresh_observed_frame_primary_runtime_receipt_v19' and r['complete'] is True and r['status']=='COMPLETE' and r['method']==METHOD and r['configured_calls']==150 and r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'actual full150 runtime required')
    for key,alias in (('protocol','runtime_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[key],b[alias]),'runtime binding '+key)
    require(p['inputs']==read(f['journal'])['inputs'],'durable initial inputs retained')
    for label,alias in (('runtime_input','runtime_protocol'),('validation_input','validation_protocol')):
        for key,value in read(f[alias])['inputs'].items():
            if label+':'+key in b:require(same(value,b[label+':'+key]),'current actual own-protocol input '+label+':'+key)
    require(same(p['inputs']['code'],b['runtime_source']) and same(p['inputs']['validator'],b['geometry_checker']),'actual source closure')
    require(isinstance(r['model_bindings'],list) and all(any(same(v,w) for w in p['inputs'].values()) for v in r['model_bindings']),'all live model/source declarations belong to frozen runtime closure')
    core,start,receipt=[read(f[k]) for k in ('geometry_protocol','geometry_started','geometry_receipt')]
    require(core['schema']=='observed_frame_repair_geometry_protocol_v19' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490 and core['limits']['GT_allowed'] is False,'fixed paired geometry authority')
    require(receipt['schema']=='observed_frame_repair_geometry_receipt_v19' and receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_errors']==[] and receipt['inputs_unchanged'] is True and receipt['environment_restored'] is True and receipt['inputs']==core['inputs']==start['inputs'] and receipt['actual_execution']==dict(frames_started=245,geometry_rows=490,ledger_rows=245,frames_complete=245) and receipt['protection_before']==receipt['protection_after']==core['protection'],'completed490/245 geometry cleanup')
    for key,alias in (('protocol','geometry_protocol'),('geometry','geometry'),('ledgers','ledgers')):require(same(receipt[key],b[alias]),'complete geometry bytes '+key)
    require(same(start['protocol'],b['geometry_protocol']),'actual geometry STARTED')
    own,cs,check=[read(f[k]) for k in ('validation_protocol','validation_started','validation')]
    require(own['schema']=='observed_frame_hybrid_scalar_protocol_v19' and check['schema']=='observed_frame_hybrid_scalar_checks_v19' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==0 and check['inputs']==own['inputs']==cs['inputs'] and cs['new_production_calls']==0 and same(check['own_protocol'],b['validation_protocol']) and same(cs['own_protocol'],b['validation_protocol']) and same(check['checker'],b['geometry_checker']),'actual full V19 independent geometry PASS')
    require(check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True and all(check['actual_counts'][k]==n for k,n in (('geometry_rows',490),('ledger_rows',245),('frames',245))),'new full V19 audit does not recertify full564 V16 FAIL')
    for key,alias in (('geometry','geometry'),('ledgers','ledgers'),('geometry_protocol','geometry_protocol'),('geometry_started','geometry_started'),('geometry_receipt','geometry_receipt')):require(same(own['inputs'][key],b[alias]),'independent completed bytes '+key)
    failed=read(f['first:GEOMETRY_RECEIPT.json']);proof=read(f['serialization_ready_proof'])
    require(failed['complete'] is False and failed['error']['type']=='TypeError' and failed['actual_execution'].get('geometry_rows',0)==0 and proof['schema']=='V19_serialization_only_ready_proof' and proof['inverse_whole_AST_equal'] is True and proof['changed_calls']==2 and proof['solver_point_line_observations_numeric_parameters_changed'] is False and proof['failed_frame_solve_entries']==2 and proof['failed_optimizer_entries']==46 and proof['failed_geometry_rows']==0,'first actual2solve/46optimizer/zero-row serialization failure preserved')
    for key,alias in (('original','original_driver'),('ready','ready_driver'),('original_first_failed_receipt','first:GEOMETRY_RECEIPT.json')):require(same(proof[key],b[alias]),'specific JSON-only repair '+key)
    pre,journal=read(f['preflight']),read(f['journal'])
    require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and journal['status']=='COMPLETE' and same(journal['protocol'],b['runtime_protocol']) and same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'complete runtime journal/preflight')
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['observed_snapshots_only'] is True and r['all_recorded_resource_snapshots_quiet'] is True,'recorded resource authority')
    return p,r,res

def observed_decision(o,a,label):
    """Recount stored count/SSE/tie and fixed-observed-block witnesses only."""
    s=o['solver'];accepted=[c for c in s['all_candidate_solutions'] if c['accepted']]
    ranked=sorted(accepted,key=lambda c:(-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index']));best=s.get('best_candidate');ties=non_eq=0
    a.eq(best,ranked[0] if ranked else None,label+'.count_SSE_best');a.eq(len(s['alternatives']),max(len(ranked)-1,0),label+'.all_relations')
    for c,w in zip(ranked[1:],s['alternatives']):
        a.eq({k:w[k] for k in c},c,label+'.whole_alternative');delta=w['truncated_sse_px2']-best['truncated_sse_px2'];tie=w['score_count']==best['score_count'] and abs(delta)<=1e-8
        a.eq(w['objective_difference_from_selected_px2'],delta,label+'.same_whole_U_delta');a.eq(w['numerical_objective_tie'],tie,label+'.inherited_true_tie');a.eq(w['same_actual8px_consensus_as_selected'],w['final_inliers']==best['final_inliers'] and w['final_line_inliers']==best['final_line_inliers'],label+'.same_support');a.eq(type(w['numerically_equivalent_to_selected']) is bool,True,label+'.reported_equivalence')
        if not w['numerically_equivalent_to_selected']:non_eq+=1;ties+=int(tie)
    a.eq(s['multiple_solutions'],bool(non_eq),label+'.multiplicity_preserved');a.eq(s['solver'],SOLVER,label+'.new_fixed_observed_solver')
    for c in (*s['proposal_solutions'],*s['all_candidate_solutions']):
        w=c['whole_pool_score'];ip=[k for k,e in zip(w['point_ids'],w['point_residual_norm_px']) if e<=8.];il=[k for k,e in zip(w['line_edges'],w['edge_block_norm_px']) if e is not None and e<=8.]
        a.eq(w['final_inliers'],ip,label+'.stored8px_point_support');a.eq(w['final_line_inliers'],il,label+'.stored8px_edge_support');a.eq(w['total_inlier_factor_count'],len(ip)+len(il),label+'.one_vote_per_edge');a.near(w['factor_truncated_sse_px2'],math.fsum(64. if e is None else min(e*e,64.) for e in w['point_residual_norm_px']+w['edge_block_norm_px']),label+'.same_fullpool_truncated_SSE')
        for block in w['edge_blocks']:
            if block['score_state']=='DEGENERATE_PROJECTED_EDGE_OUTLIER':continue
            for k,v in (('coordinate_frame','FIXED_OBSERVED_LINE'),('historical_projected_tangent_objective_reused',False),('containment_tangent_uses_observed_normal_even_when_modeled',True),('radius_gate_applied',False),('physical_semantic_line_correctness_certified',False)):a.eq(block[k],v,label+'.fixed_observed_block.'+k)
    if s['available']:a.eq(ties,0,label+'.true_tie_not_NEW');a.eq(s['unresolved_ambiguity'],False,label+'.resolved_count_SSE');a.eq(s['state'],'NEW_POSE',label+'.NEW_state')
    if ties:a.eq(s['available'],False,label+'.true_tie_veto_retained')
    for k,v in (('reprojected_points_reused_as_observations',False),('center_used_for_fit_or_score',False),('prior_used',False),('same_edge_point_line_double_count',False),('global_uniqueness_proven',False),('new_finite_objective_is_not_historical_scalar_failure_repair',True)):a.eq(s[k],v,label+'.numeric_policy.'+k)

def corner_selection(o,a,label):
    H=o['hidden_initial'];v=o['cornerwise_selection'];records=v['per_corner_decisions'];chosen=[r['id'] for r in records if r['accepted']]
    a.eq(v['actual_self_hidden_ids'],H,label+'.selection_H');a.eq(v['selected_boundary_corner_ids'],chosen,label+'.adopted_TLS_corners');a.eq(o['selected_corner_ids'],chosen,label+'.actual_adopted_q_ids');a.eq(v['LOO_solve_calls'],sum(r['heldout_pose_calls'] for r in records),label+'.corner_LOO_calls')
    for r in records:
        k=r['id'];blocked=sorted(set(H)|{k});a.eq(r['heldout_fit_requested_excluded_ids'],blocked,label+'.LOO_H_union_k');a.eq(r['actual_self_hidden_ids'],H,label+'.corner_same_H');a.eq(bool(set(r['fit_input_ids'])&set(blocked)),False,label+'.corner_fit_exclusion');a.eq(r['heldout_is_self_occlusion'],False,label+'.temporary_holdout_not_H');a.eq(r['projected_coordinate_used_as_observation'],False,label+'.LOO_projection_not_q');a.eq(r['numeric_validation_pose_prior_used'],False,label+'.no_heldout_prior')
        if r['accepted']:
            a.eq(k not in H and r['heldout_pose_calls']==1 and r['candidate_LOO_residual_px']<=8.,True,label+'.adoption_support');a.eq(not r['native_N3_available'] or r['native_minus_candidate_squared_residual_px2']>1e-8,True,label+'.strict_native_vs_boundary_gain');a.near(o['input_points'][k],r['candidate_xy'],label+'.actual_TLS_q',relative=0)
    for k in set(range(8))-set(chosen):a.near(o['input_points'][k],o['native_N3_points'][k],label+'.unadopted_native_q',relative=0)

def guard(args,stage):
    for p in (args.repo,args.private,args.input,args.validation_input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directories')
    require(args.output.is_relative_to(args.private) and all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in (args.repo,args.input,args.validation_input,args.runtime_input)),'new disjoint private checker directory')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)

def freeze(args):
    guard(args,'freeze');f=paths(args);b={k:binding(p) for k,p in f.items()};proto,_,_=metadata(f,b);_,info=extraction(f)
    write(args.output/FILES[0],dict(schema='independent_observed_frame_runtime_protocol_v19',inputs=b,helpers=info,limits=LIMITS,external_declared_metadata=declarations(proto),external_metadata_current_assets_not_verified=True,frozen_before_own_arithmetic=True))

def run(args):
    guard(args,'run');f=paths(args);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,info=extraction(f)
    proto,r,res=metadata(f,b);require(own['schema']=='independent_observed_frame_runtime_protocol_v19' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['external_declared_metadata']==declarations(proto) and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));env=dict(math=math,copy=copy,Counter=Counter,combinations=combinations);exec(compile(tree,str(f['clock_source'])+':stdlib_AST_only','exec'),env)
    a=env['Audit']();totals={};timings={k:[] for k in TIMES};count=0;error=None;packet_totals=Counter()
    try:
        panel=proto['panel'];a.eq(len(panel),26,'panel26');a.eq(len({v['session_id'] for v in panel}),13,'sessions13');a.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session')
        identities=proto['normalization']['joined_frame_identities'];authority={v['id']:v for v in identities};a.eq(len(authority),245,'normalization245');a.eq(len(identities),245,'no_duplicate_normalization');a.eq(r['normalization'],proto['normalization'],'same_normalization')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];a.eq(proto['schedule'],schedule,'fixed150schedule')
        ids={v['frame_id'] for v in panel};refs={};seen={m:set() for m in METHODS};fixed={}
        for v in rows(f['geometry']):
            require(v['method'] in seen and v['id'] in authority and v['id'] not in seen[v['method']],'duplicate/foreign paired accuracy row');seen[v['method']].add(v['id'])
            if v['method']==METHOD and v['id'] in ids:
                refs[v['id']]={k:v[k] for k in ('native_points','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')};pose=v['actual_pose'];refs[v['id']]['pose']={k:pose[k] for k in ('available','selected_hypothesis',*POSE,*OPTIONAL) if k in pose}
        a.eq(seen,{m:set(authority) for m in METHODS},'all490paired_accuracy');a.eq(set(refs),ids,'all26references')
        for v in rows(f['fixed_geometry']):
            if v['id'] in ids:require((v['method'],v['id']) not in fixed,'duplicate fixed reference');fixed[v['method'],v['id']]=v['native_points']
        a.eq(set(fixed),{(m,i) for m in ('BASE','N3_SUBPIX') for i in ids},'fixedBASE_N3panel')
        for i,v in enumerate(rows(f['rows'])):
            require(i<150,'extra runtime row');job=schedule[i];label='row.'+str(i)
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
            a.near(o['original_base_points'],fixed['BASE',v['id']],label+'.original_BASE',relative=0);a.near(native,fixed['N3_SUBPIX',v['id']],label+'.original_N3',relative=0);a.eq(o['K'],authority[v['id']]['K'],label+'.K');a.eq(o['xyz'],authority[v['id']]['xyz'],label+'.dimensions')
            for k,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True)):a.eq(v[k],value,label+'.'+k)
            for k,value in (('saved_accuracy_coordinates_or_pose_as_input',False),('cached_accuracy_row_used',False),('fresh_capture_input_only',True),('fresh_point_solver_recomputed_inside_call',True),('fresh_endpoint_LOO_recomputed_inside_call',True),('fresh_raw_selected_and_native_point_roots_reused_inside_call',True),('actual_fit_incidence_guard_applied_per_candidate',True),('cached_endpoint_decisions_used',False),('reprojections_reused_as_observations',False)):
                a.eq(o[k],value,label+'.fresh.'+k)
            a.eq(o['fresh_repair_ledger'],ledger,label+'.same_full_ledger');a.eq(o['pose_estimation_kind'],METHOD,label+'.runtime_method');a.eq(o['paired_point_accuracy_arm_executed_inside_primary_call'],False,label+'.no_accuracy_pair_inside_clock');a.eq(o['new_finite_objective_is_not_historical_scalar_failure_repair'],True,label+'.new_objective_scope');a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_subset_bank');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['endpoint_decision_source'],'FRESH_NATIVE_ENDPOINT_LOO',label+'.fresh_endpoint');success=env['proposal_witnesses'](o,ledger,a,label);observed_decision(o,a,label);corner_selection(o,a,label)
            filter_=ledger['observation_filter'];records=filter_['endpoint_records'];a.eq(ledger['endpoint_validation_requests'],len(records),label+'.endpoint_requests');a.eq(ledger['endpoint_decision_records'],len(records),label+'.endpoint_returns');a.eq(o['observation_filter'],filter_,label+'.same_filter')
            rejected=[rec['edge'] for rec in records if rec['decision']=='CONTRADICTED_REJECT'];a.eq(filter_['rejected_edge_ids'],rejected,label+'.reject_only_contradicted');a.eq(filter_['retained_unused_edge_ids'],[e for e in filter_['original_actual_unused_edge_ids'] if e not in rejected],label+'.retain_other_edges');a.eq(filter_['unknown_retained_edge_ids'],[rec['edge'] for rec in records if rec['decision']=='UNVERIFIED_RETAIN'],label+'.unknown_retained')
            for rec in records:
                blocked=sorted(set(H)|set(rec['endpoints']));a.eq(rec['validation_excluded'],blocked,label+'.LOO_H_union_endpoints');a.eq(rec['hidden'],H,label+'.LOO_separate_H');a.eq(rec['native_input_hash'],ledger['endpoint_native_input_hash'],label+'.native_shared_bank');a.eq(rec['observed_line_supplied_to_validation_solver'],False,label+'.LOO_independence');a.eq(bool(set(rec['solved'].get('used',[]))&set(blocked)),False,label+'.LOO_fit_exclusion')
            if o['new_pose_estimated']:
                a.eq(bool(set(H)&set(s['fit_input_ids'])),False,label+'.H_fit_excluded');a.eq(pose,s,label+'.actual_Rt');a.eq(s['best_candidate']['accepted'],True,label+'.accepted_selected');a.eq(s['best_candidate']['stable_incidence_guard']['accepted'],True,label+'.selected_normal_guard')
                for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projected_once.'+str(k),1e-12)
            else:a.eq(pose,o['initial_pose'],label+'.full_original_fallback');a.eq(o['native_points'],native,label+'.full_original_fallback_points');a.eq(o['reprojected_ids'],[],label+'.no_fallback_H_copies')
            a.eq(s['operation_counts']['hidden_reprojection_assignments'],len(H) if s['available'] else 0,label+'.H_assignment_once');a.eq(o['new_pose_estimated'],s['available'],label+'.NEW_outcome');a.eq(o['reprojected_ids'],H if s['available'] else [],label+'.approved_H_reprojection_ids');a.eq(o['native_points'][8],native[8],label+'.center');prediction=copy.deepcopy(v['original_fresh_detector_prediction']);selected=prediction['selected_index']
            if selected is not None:a.eq(prediction['candidates'][selected]['keypoints_xy'][8],native[8],label+'.raw_center');prediction['candidates'][selected]['keypoints_xy']=o['native_points']
            a.eq(o['prediction']['selected_index'],selected,label+'.selected_index');a.eq(o['prediction']['candidates'],prediction['candidates'],label+'.all_conf_box_preserved')
            d=v['counter_deltas'];a.eq(set(d),{'cv','optimizer','head','base','fresh','local','native','model','svd','direct','bank_capture','native_reuse'},label+'.all12counter_groups');a.eq(d['head'],dict(entries=1,returns=1),label+'.head');a.eq(d['model'],dict(detector=2 if i==0 else 1,N3=1),label+'.actual_model_forwards')
            op=ledger['local_operation_counts'];a.eq(sparse(d['local']),sparse(op),label+'.subset_work')
            reuse=v['fresh_point_bank_reuse'];a.eq(reuse,o['fresh_actual_hybrid_bank_reuse'],label+'.same_reuse_output');a.eq(reuse,ledger['fresh_actual_hybrid_bank_reuse'],label+'.same_reuse_ledger');a.eq(reuse['schema'],'fresh_hybrid_selected_native_point_bank_reuse_v16',label+'.reuse_schema')
            for k,value in (('same_coordinate_bank_reused',True),('constructor_restored',True),('endpoint_constructor_restored',True),('additional_selected_Generic_calls',0),('point_available_verdict_not_a_root_filter',True),('cache_reconstructed_from_serialized_Rt',False)):a.eq(reuse[k],value,label+'.reuse.'+k)
            captures=reuse['actual_constructor_count'];require(type(captures) is int and 1<=captures<=2,'one native/selected bank or two actual banks')
            samebank=reuse['selected_native_bank_is_same_object'];a.eq(type(samebank) is bool,True,label+'.samebank_typed');a.eq(captures,1 if samebank else 2,label+'.actual_distinct_banks')
            a.eq(reuse['selected_input_hash'],s['input_hash'],label+'.selected_digest');a.eq(reuse['native_input_hash'],ledger['endpoint_native_input_hash'],label+'.native_digest')
            direct=reuse['direct_bank_delta'];endpoint=reuse['endpoint_bank_delta'];a.eq(direct,ledger['direct_point_bank_operation_counts'],label+'.same_direct_ledger');a.eq(direct,s['hypothesis_bank_operation_counts'],label+'.same_direct_solver')
            a.eq(sparse(d['direct']),sparse(direct),label+'.actual_direct');a.eq(sparse(d['native']),sparse(endpoint),label+'.actual_endpoint');a.eq(endpoint,ledger['fresh_native_endpoint_phase_operation_counts'],label+'.endpoint_phase_counts')
            phase_after=records[-1]['bank_after'] if records else ledger['endpoint_native_bank_before'];a.eq(phase_after,ledger['fresh_native_endpoint_phase_after'],label+'.endpoint_phase_after')
            a.eq(endpoint,{k:phase_after[k]-ledger['endpoint_native_bank_before'][k] for k in phase_after},label+'.no_phase_double_count')
            a.eq(reuse['native_bank_after'],ledger['endpoint_native_bank_after'],label+'.native_end_ledger');a.eq(reuse['native_bank_before'],ledger['endpoint_native_bank_before'],label+'.native_begin_ledger')
            for prefix in ('selected','native'):
                before,after=reuse[prefix+'_bank_before'],reuse[prefix+'_bank_after'];delta={k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)}
                a.eq(delta,reuse[prefix+'_bank_delta'],label+'.'+prefix+'_delta')
                expected={k:(direct[k]+(endpoint[k] if samebank else 0)) if prefix=='selected' else (endpoint[k]+(direct[k] if samebank else 0)) for k in delta}
                a.eq(delta,expected,label+'.'+prefix+'_phase_partition')
            a.eq(direct['generic_calls'],0,label+'.no_extra_selected_Generic');a.eq(direct['generic_cache_misses'],0,label+'.no_extra_selected_four_ID_miss');a.eq(reuse['additional_native_Generic_calls'],endpoint['generic_calls'],label+'.new_native_Generic_recorded')
            cache=lambda items:{(v['dimension_index'],tuple(v['point_ids'])) for v in items}
            before,after,newkeys=(cache(reuse[k]) for k in ('native_numeric_cache_before','native_numeric_cache_after','newly_requested_native_numeric_problems'))
            a.eq(len(before),len(reuse['native_numeric_cache_before']),label+'.unique_native_before');a.eq(len(after),len(reuse['native_numeric_cache_after']),label+'.unique_native_after');a.eq(len(newkeys),len(reuse['newly_requested_native_numeric_problems']),label+'.unique_native_new')
            a.eq(before<=after,True,label+'.native_existing_cache_preserved');a.eq(newkeys,after-before,label+'.new_native_ID_problems');a.eq(endpoint['generic_cache_misses'],len(newkeys),label+'.actual_native_cache_misses')
            four=sum(len(k)==4 for _,k in newkeys);a.eq(reuse['additional_native_four_ID_numeric_problems'],four,label+'.four_ID_problem_count')
            if reuse['native_four_ID_bank_built_before_endpoint']:a.eq(four,0,label+'.existing_native_four_ID_reused')
            a.eq(d['bank_capture'],dict(observer_scopes_started=1,constructor_entries=captures,constructor_returns=captures,observer_scopes_restored=1),label+'.actual_constructor_lifecycle')
            a.eq(d['native_reuse'],dict(observer_scopes_started=1,constructor_requests=1,actual_native_bank_reuses=1,observer_scopes_restored=1),label+'.actual_native_reuse_lifecycle')
            pointbank=v['fresh_point_bank_ledger'];a.eq(pointbank,o['fresh_point_coordinate_bank_ledger'],label+'.same_point_bank_ledger');a.eq(pointbank['coordinate_bank_count'],captures,label+'.actual_point_bank_count')
            originals={v['input_hash']:v for v in pointbank['coordinate_banks']};a.eq(len(originals),captures,label+'.distinct_point_bank_hashes');a.eq(set(originals),{reuse['selected_input_hash'],reuse['native_input_hash']},label+'.actual_original_bank_identities')
            for prefix in ('selected','native'):
                for k,value in reuse[prefix+'_bank_before'].items():a.eq(originals[reuse[prefix+'_input_hash']][k],value,label+'.original_'+prefix+'_before.'+k)
            compat=o['packet_compatibility_review'];a.eq(compat['numerical_result_changed'],False,label+'.packet_numeric_preserved');a.eq(compat['original_frame_metadata_preserved'],True,label+'.packet_metadata_preserved');a.eq(compat['original_packet_function_used'],True,label+'.original_packet');packet_totals.update(calls=1,identical_fields_removed=len(compat['removed_identical_fields']))
            require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'unknown CV primitive');require(set(d['optimizer'])<={'entries','returns','exceptions','success_reports',*(p+'_'+k for p in ('residual','jacobian') for k in ('entries','returns','exceptions'))},'unknown optimizer counter');require(set(d['svd'])<={'entries','returns','exceptions'},'unknown SVD counter')
            for key,source in (('entries','local_optimizer_calls'),('returns','local_optimizer_completions'),('exceptions','local_optimizer_exceptions'),('residual_entries','optimizer_residual_calls'),('jacobian_entries','optimizer_jacobian_calls')):a.eq(d['optimizer'].get(key,0),op[source],label+'.actual_optimizer.'+key)
            a.eq(d['optimizer'].get('success_reports',0),success,label+'.actual_success_reports')
            for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.CV_complete.'+p)
            for p in ('','residual_','jacobian_'):a.eq(d['optimizer'].get(p+'entries',0),d['optimizer'].get(p+'returns',0)+d['optimizer'].get(p+'exceptions',0),label+'.handled_optimizer_or_callback_exceptions.'+p)
            a.eq(d['svd'].get('entries',0),d['svd'].get('returns',0)+d['svd'].get('exceptions',0),label+'.SVD_complete')
            for key,values in d.items():require(all(type(n) is int and n>=0 for n in values.values()),'invalid primitive/route counter');totals.setdefault(key,Counter()).update(values)
            count+=1
        a.eq(count,150,'all150');a.eq(len(timings['full']),130,'measured130')
        for k in TIMES:a.near(env['describe'](timings[k]),r['summaries'][k],'summary.'+k)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('local','local_operation_counts'),('native','native_operation_counts'),('svd','live_SVD_counts'),('direct','direct_point_operation_counts'),('bank_capture','point_bank_capture_counts'),('native_reuse','native_bank_reuse_counts')):a.eq(sparse(totals[key]),sparse(r[field]),'actual.sum.'+key)
        a.eq(sparse(packet_totals),sparse(r['packet_compatibility_counts']),'actual.packet_compatibility');a.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'loading_forwards');a.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');a.eq(r['model_forwards'],dict(detector=151,N3=150),'actual_internal_init')
        for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):a.eq(totals['base'][key],150,'actual.base_route.'+key)
        for key in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_repair_paths','fresh_filtered_local_paths'):
            for phase in ('started','complete'):a.eq(totals['fresh'][key+'_'+phase],150,'actual.fresh_route.'+key+'_'+phase)
        a.eq(totals['fresh'][METHOD+':started'],150,'actual.method_started');a.eq(totals['fresh'][METHOD+':complete'],150,'actual.method_complete');a.eq(totals['fresh']['reported_primary_views_from_single_filtered_local'],150,'actual.single_output_per_call');a.eq(totals['fresh']['fresh_native_endpoint_requests_started'],totals['fresh']['fresh_native_endpoint_requests_complete'],'actual.endpoint_completed')
        a.eq(totals['direct']['generic_calls'],0,'actual.sharedbank_Generic_reuse');a.eq(totals['direct']['generic_cache_misses'],0,'actual.sharedbank_no_miss');a.eq(totals['bank_capture']['observer_scopes_started'],150,'actual150constructor_scopes');a.eq(totals['bank_capture']['observer_scopes_restored'],150,'actual150constructor_restores');a.eq(totals['bank_capture']['constructor_entries'],totals['bank_capture']['constructor_returns'],'actual_bank_constructor_returns');a.eq(150<=totals['bank_capture']['constructor_entries']<=300,True,'actual150_to300_banks');a.eq(totals['native_reuse'],dict(observer_scopes_started=150,constructor_requests=150,actual_native_bank_reuses=150,observer_scopes_restored=150),'actual150native_reuse_lifecycles')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('model_initializations',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'missing resource snapshots')
        for snap in res['snapshots']:a.eq(snap['quiet'] and snap['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0
    write(args.output/FILES[2],dict(schema='independent_observed_frame_runtime_checks_v19',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,external_declared_metadata=declarations(proto),external_metadata_current_assets_not_verified=True,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True,new_production_calls=0,old_full_V16_validation_passed=False))
    print('OBSERVED_FRAME_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--private',type=Path,default=PRIVATE)
    p.add_argument('--input',type=Path,default=PRIVATE/'geometry_ready');p.add_argument('--validation-input',type=Path,default=PRIVATE/'validation')
    p.add_argument('--runtime-input',type=Path,default=PRIVATE/'runtime');p.add_argument('--output',type=Path,default=PRIVATE/'runtime_checks')
    args=p.parse_args()
    for key in ('repo','private','input','validation_input','runtime_input','output'):setattr(args,key,getattr(args,key).resolve())
    (freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
