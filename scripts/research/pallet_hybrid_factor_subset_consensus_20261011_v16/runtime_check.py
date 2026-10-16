"""Own-frozen stdlib audit of fresh V16 hybrid clocks and shared hypothesis banks.

No image, model, GT, optimizer, PnP, Jacobian, SVD or timing is invoked. Stored
candidate predicates/counts and final projection assignments are checked;
candidate geometry arithmetic belongs to the separate full geometry audit.
Stored clock consistency and resource snapshots do not authenticate hardware.
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

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
PRIVATE=Path('/tmp/pallet-hybrid-factor-subset-private-20261011-v16')
SOURCE=REPO/'scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime_check.py'
SOURCE_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
PREDICATE_SOURCE=REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/runtime_check.py'
PREDICATE_SHA='afc21b2ec0c9f2c49b8f05669a9e697ea2a83767767f66d448e15085c9c4f478'
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
METHOD='HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS';METHODS=(METHOD,'HYBRID_FACTOR_SUBSET_POINT_ONLY');TIMES=('full',*STAGES,'return');POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
LIMITS=dict(new_inference_RGB_GT_fit_PnP_Jacobian_SVD_timing_calls=0,
    saved_clock_summary_counter_and_candidate_predicate_consistency_only=True,
    fresh_candidate_pose_J_or_SVD_not_recomputed=True,accuracy_geometry_audit_not_repeated=True,
    same_four_ID_raw_root_cache_reuse_checked=True,mixed_LOCAL_roots_complete=False,
    handled_branch_exceptions_not_complete_frame_failure=True,physical_ownership_certified=False,
    hardware_authenticity_proven=False,quiet_snapshots_can_miss_transient_contention=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):d.update(chunk)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(d):return {k:v for k,v in d.items() if v}
def public(v):
    p=Path(v['path']);require(not p.is_absolute() and '..' not in p.parts,'public repository input');return REPO/p
def is_public(v):
    if v.get('origin')=='external_readonly_dependency':return False
    p=Path(v['path']);require(v.get('origin','public_repository')=='public_repository' and not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public repository binding');return True
def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'unchanged independent clock/statistics source')
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('require','Audit','describe')]
    require({n.name for n in nodes}=={'require','Audit','describe'},'exact stdlib whitelist');tree=ast.Module(body=nodes,type_ignores=[])
    require(binding(PREDICATE_SOURCE)['sha256']==PREDICATE_SHA,'unchanged independent candidate predicate source')
    old=[n for n in ast.parse(PREDICATE_SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='proposal_witnesses']
    new=[n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='proposal_witnesses']
    require(len(old)==len(new)==1 and ast.dump(old[0],include_attributes=False)==ast.dump(new[0],include_attributes=False),'complete candidate predicate AST unchanged')
    return tree,dict(source=binding(SOURCE),names=['require','Audit','describe'],AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),candidate_predicate_source=binding(PREDICATE_SOURCE),candidate_predicate_formula_changes=0)

def paths(p,t):
    names=dict(geometry='GEOMETRY_SEALED.jsonl.gz',core_protocol='PROTOCOL.json',seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',validation='VALIDATION_CHECKS.json',validation_protocol='VALIDATION_PROTOCOL.json',validation_started='VALIDATION_STARTED.json',scoring='SCORING_RECEIPT.json',scoring_started='SCORING_STARTED.json',predictions='PREDICTIONS.jsonl.gz',evaluation_protocol='EVALUATION_PROTOCOL.json',scoring_authority='SCORING_AUTHORITY.json')
    runtime=dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json')
    out={k:p/v for k,v in names.items()};out.update({k:t/v for k,v in runtime.items()});out.update(checker=Path(__file__).resolve(),scalar_source=SOURCE,predicate_source=PREDICATE_SOURCE,fixed_geometry=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for n in ('runtime.py','deployment.py','pipeline_ready.py','runner.py','evaluation.py','validation_checks.py'):out['code:'+n]=CODE/n
    for key,value in read(out['runtime_protocol'])['inputs'].items():
        if is_public(value):out['runtime_input:'+key]=public(value)
    for key,value in read(out['validation_protocol'])['inputs'].items():out['accuracy_input:'+key]=public(value)
    for key,value in read(out['evaluation_protocol'])['inputs'].items():
        if is_public(value):out['evaluation_input:'+key]=public(value)
    return out

def metadata(f,b):
    proto,r,res=read(f['runtime_protocol']),read(f['receipt']),read(f['resources'])
    require(proto['schema']=='fresh_hybrid_factor_subset_primary_runtime_protocol_v16' and proto['method']==METHOD and proto['configured_calls']==150 and proto['warmup']==20 and proto['measured']==130 and proto['repeats']==5 and proto['stages']==list(STAGES),'fixed150 fresh runtime')
    require(proto['first_prediction_detector_forwards']==2 and proto['other_prediction_detector_forwards']==1 and proto['N3_forwards_per_prediction']==1 and proto['saved_geometry_used_only_after_return'] is True and proto['no_artificial_pre_forward'] is True,'fresh forward policy')
    require(r['schema']=='fresh_hybrid_factor_subset_primary_runtime_receipt_v16' and r['method']==METHOD and r['complete'] is True and r['status']=='COMPLETE' and r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'complete fresh runtime')
    for key,alias in (('protocol','runtime_protocol'),('accuracy_protocol','core_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[key],b[alias]),'runtime byte authority '+key)
    for key,value in proto['inputs'].items():
        if 'runtime_input:'+key in b:require(same(value,b['runtime_input:'+key]),'current frozen runtime closure '+key)
    for key in ('code:'+n for n in ('runtime.py','deployment.py','pipeline_ready.py','runner.py','evaluation.py','validation_checks.py')):require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in proto['inputs'].values()),'unbound runtime source '+key)
    core,seal,repair,check,own,score=[read(f[k]) for k in ('core_protocol','seal','repair_receipt','validation','validation_protocol','scoring')]
    require(core['schema']=='fixed_hybrid_factor_subset_consensus_protocol_v16' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490 and seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and repair['complete'] is True and repair['cleanup_completed'] is True and repair['error'] is None and repair['cleanup_error'] is None,'complete restored geometry')
    for key,alias in (('protocol','core_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[key],b[alias]),'accuracy sealed bytes '+key)
    require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16' and check['schema']=='hybrid_factor_subset_scalar_checks_v16' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['inputs']==own['inputs'] and same(check['own_protocol'],b['validation_protocol']) and same(check['checker'],b['code:validation_checks.py']),'actual full geometry PASS')
    for key,value in own['inputs'].items():require(same(value,b['accuracy_input:'+key]),'current independently checked bytes '+key)
    start=read(f['validation_started']);require(start['inputs']==own['inputs'] and same(start['own_protocol'],b['validation_protocol']) and start['new_production_calls']==0,'actual scalar STARTED')
    evaluation,authority=read(f['evaluation_protocol']),read(f['scoring_authority']);marker=evaluation['score_dispatch']
    require(evaluation['schema']=='postseal_hybrid_factor_subset_evaluation_protocol_v16' and authority['schema']=='isolated_hybrid_proxy_score_authority_v16' and authority['methods']==list(METHODS) and authority['inputs']==evaluation['inputs'] and authority['score_dispatch']==marker and same(authority['own_evaluation_protocol'],b['evaluation_protocol']) and authority['original_module_globals_modified'] is False and authority['reference_or_error_formula_changed'] is False,'isolated scoring authority')
    require(marker['topology_AST_changes']==dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1) and marker['full_canonical_score_function_restore_proven'] is True and marker['score_reference_formula_AST_changes']==0 and marker['original_module_globals_modified'] is False and marker['immutable_parent_method_tuple_modified'] is False,'unchanged scoring formulas')
    for key,value in evaluation['inputs'].items():
        if is_public(value):require(same(value,b['evaluation_input:'+key]),'current scoring closure '+key)
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['scored_rows']==490 and score['frames']==245 and score['methods']==list(METHODS) and same(score['protocol'],b['core_protocol']) and same(score['geometry_seal'],b['seal']) and same(score['predictions'],b['predictions']),'actual postseal scoring')
    ss=read(f['scoring_started']);require(same(ss['independent_validation'],b['validation_protocol']) and same(ss['geometry_seal'],b['seal']) and ss['configured_scored_rows']==490 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'score after full independent PASS')
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['all_recorded_snapshots_quiet'] is True,'recorded resources')
    pre,journal=read(f['preflight']),read(f['journal']);require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and journal['status']=='COMPLETE' and same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'complete runtime journal/preflight')
    return proto,r,res

def proposal_witnesses(o,ledger,a,label):
    s=o['solver'];U=s['used'];H=o['hidden_initial'];edges=s['line_edges'];op=s['operation_counts'];ps=s['proposal_solutions'];cs=s['all_candidate_solutions'];attempts=s['optimizer_attempts'];D=len(s.get('registered_dimensions',[]));early=s['state'] in ('INVALID_OBSERVATION_CONTRACT','INSUFFICIENT_POINT_OBSERVATIONS','INSUFFICIENT_ACTUAL_INCIDENCE','UNSUPPORTED_CAMERA_CONTRACT')
    a.eq(s,ledger['full_solver'],label+'.fresh_solver_ledger');a.eq(bool(set(U)&set(H)),False,label+'.wholepool_no_H');a.eq(s['reprojected_points_reused_as_observations'],False,label+'.no_projected_refit')
    if not early:
        generated=s['direct_point_generation'];subsets=s['mixed_subset_enumeration'];a.eq([(g['dimension_index'],g['generator_ids']) for g in generated],[(d,list(k)) for d in range(D) for k in combinations(U,4)],label+'.all_direct_four_ID_requests')
        blocks=[('POINT',k) for k in U]+[('EDGE',e) for e in edges];mixed=[v for v in combinations(blocks,3) if any(k=='EDGE' for k,_ in v)];a.eq([v['factor_blocks'] for v in subsets],[[list(k) for k in v] for v in mixed],label+'.all_mixed_subsets')
        for sub in subsets:a.eq([v['dimension_index'] for v in sub['branches']],list(range(D)),label+'.all_dimensions')
        roots=sum(len(g['roots']) for g in generated);a.eq(op['direct_point_roots'],roots,label+'.all_raw_roots');a.eq(op['mixed_subset_requests'],len(mixed),label+'.mixed_count');a.eq(op['mixed_dimension_requests'],D*len(mixed),label+'.mixed_dimension_count');a.eq(s['call_bounds']['LOCAL_optimizer_calls_at_most'],2*D*len(mixed)+roots,label+'.optimizer_bound');a.eq(op['local_optimizer_calls']<=s['call_bounds']['LOCAL_optimizer_calls_at_most'],True,label+'.actual_bounded_calls')
        for g in generated:a.eq(g['returned_valid_roots'],len(g['roots']),label+'.returned_root_records')
    for c in (*ps,*cs):
        a.eq(bool(set(c['fit_input_ids'])&set(H)) or 8 in c['fit_input_ids'],False,label+'.candidate_no_H_center');a.eq(bool(set(c['fit_line_edges'])&set(s['consumed_edges'])),False,label+'.no_duplicate_edge');a.eq(all(type(z) in (int,float) and math.isfinite(z) and z>1e-9 for z in c['model_camera_depths_m']),True,label+'.stored_depth');a.eq(c['initial_pose_residual_prior'],False,label+'.no_prior')
    for c in cs:
        g=c['point_line_geometry'];weak=False;rank=True
        for name in ('observed','modeled'):
            w=g[name+'_normal_joint'];ss=w['singular_values'];recount=sum(v>ss[0]*1e-10 for v in ss) if ss else 0;flag=len(ss)!=6 or ss[-1]<ss[0]*1e-4;a.eq(w['numerical_rank'],recount,label+'.finite_rank_recount');a.eq(g[name+'_weak'],flag,label+'.finite_weak');rank&=recount==6;weak|=flag
        stable=set(c['fit_input_ids'])<=set(c['final_inliers']) and set(c['fit_line_edges'])<=set(c['final_line_inliers']);reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank else 'WEAK_OBSERVABILITY' if weak else None
        if reason is None:
            w=c['stable_incidence_guard'];a.eq(w['selected_candidate_reselected'],False,label+'.no_guard_reselection');a.eq(w['actual_fit_input_ids'],c['fit_input_ids'],label+'.actual_guard_points');a.eq(w['actual_fit_line_edges'],c['fit_line_edges'],label+'.actual_guard_lines');a.eq(w['incidence_scalar_rows'],2*(len(c['fit_input_ids'])+len(c['fit_line_edges'])),label+'.only_normal_incidence_rows');g=w.get('stable_incidence_geometry')
            if g is None:a.eq(w['guard_state'],'STABLE_INCIDENCE_NUMERICAL_FAILURE',label+'.guard_numerical');reason=w['guard_state']
            else:
                rank=True;weak=False
                for name in ('observed','modeled'):
                    ss=g[name+'_normal_joint']['singular_values'];recount=sum(v>ss[0]*1e-10 for v in ss) if ss else 0;flag=len(ss)!=6 or ss[-1]<ss[0]*1e-4;a.eq(g[name+'_normal_joint']['numerical_rank'],recount,label+'.normal_rank_recount');a.eq(w[name+'_weak'],flag,label+'.normal_weak');rank&=recount==6;weak|=flag
                state='PASSED_STABLE_INCIDENCE' if rank and not weak else 'STABLE_INCIDENCE_INSUFFICIENT' if not rank else 'STABLE_INCIDENCE_WEAK';a.eq(w['guard_state'],state,label+'.normal_route');a.eq(w['accepted'],state=='PASSED_STABLE_INCIDENCE',label+'.normal_acceptance')
                if not w['accepted']:reason=state
        a.eq(c['accepted'],reason is None,label+'.candidate_gate');a.eq(c['rejected'],reason,label+'.candidate_reason')
        if c['accepted']:a.eq(bool(c['fit_line_edges']) or len(c['fit_input_ids'])>=4,True,label+'.standard_four_point_fit')
    calls=Counter();success=0
    for sub in s['mixed_subset_enumeration']:
        for e in sub['branches']:calls['proposal']+=int(e['actual_optimizer_called']);success+=int(e['actual_optimizer_called'] and e.get('success',False))
    a.eq(len(attempts),len(ps),label+'.all_proposal_refit_records')
    for e,p in zip(attempts,ps):
        entry=e['refit'];calls['refit']+=int(entry['actual_optimizer_called']);success+=int(entry['actual_optimizer_called'] and entry.get('success',False));a.eq(entry['proposal_consensus_point_ids'],p['whole_pool_score']['final_inliers'],label+'.actual_refit_points');a.eq(entry['proposal_consensus_line_ids'],p['whole_pool_score']['final_line_inliers'],label+'.actual_refit_lines')
        if not entry['proposal_consensus_line_ids'] and len(entry['proposal_consensus_point_ids'])<4:a.eq(entry['rejected'],'INSUFFICIENT_POINT_INLIER_OBSERVATIONS',label+'.insufficient_point_refit');a.eq(entry['actual_optimizer_called'],False,label+'.no_three_point_optimizer')
    for phase in ('proposal','refit'):a.eq(op[phase+'_optimizer_calls'],calls[phase],label+'.actual_'+phase+'_calls');a.eq(op[phase+'_optimizer_calls'],op[phase+'_optimizer_returns']+op[phase+'_optimizer_exceptions'],label+'.handled_'+phase+'_exceptions')
    guards=[c['stable_incidence_guard'] for c in cs if 'stable_incidence_guard' in c];a.eq(op['stable_incidence_guard_calls'],len(guards),label+'.all_candidate_guards');a.eq(op['stable_incidence_geometry_calls'],sum(w['counts']['geometry_info_calls'] for w in guards),label+'.guard_geometry_calls');a.eq(op['stable_incidence_geometry_projection_entries'],sum(w['counts']['geometry_projection_entries'] for w in guards),label+'.guard_projection_calls')
    return success

def guard(args,stage):
    for p in (args.input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directories')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V16 output')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)

def declarations(proto):
    return {k:v for k,v in proto['inputs'].items() if not is_public(v)}

def freeze(args):
    guard(args,'freeze');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};proto,_,_=metadata(f,b);_,info=extraction()
    write(args.output/FILES[0],dict(schema='independent_hybrid_factor_subset_runtime_check_protocol_v16',inputs=b,helpers=info,limits=LIMITS,external_declared_metadata=declarations(proto),external_metadata_current_assets_not_verified=True,frozen_before_own_arithmetic=True))

def run(args):
    guard(args,'run');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,info=extraction()
    proto,r,res=metadata(f,b);require(own['schema']=='independent_hybrid_factor_subset_runtime_check_protocol_v16' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['external_declared_metadata']==declarations(proto) and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));env=dict(math=math);exec(compile(tree,str(SOURCE)+':stdlib_AST_only','exec'),env)
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
            for k,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True),('original_candidate_center_preservation_checked_inside_predict',True)):a.eq(v[k],value,label+'.'+k)
            for k,value in (('saved_accuracy_coordinates_or_pose_as_input',False),('cached_accuracy_row_used',False),('fresh_capture_input_only',True),('fresh_point_solver_recomputed_inside_call',True),('fresh_endpoint_LOO_recomputed_inside_call',True),('fresh_raw_selected_and_native_point_roots_reused_inside_call',True),('actual_fit_incidence_guard_applied_per_candidate',True),('cached_endpoint_decisions_used',False),('reprojections_reused_as_observations',False)):
                a.eq(o[k],value,label+'.fresh.'+k)
            a.eq(o['fresh_repair_ledger'],ledger,label+'.same_full_ledger');a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_subset_bank');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['endpoint_decision_source'],'FRESH_NATIVE_ENDPOINT_LOO',label+'.fresh_endpoint');success=proposal_witnesses(o,ledger,a,label)
            filter_=ledger['observation_filter'];records=filter_['endpoint_records'];a.eq(ledger['endpoint_validation_requests'],len(records),label+'.endpoint_requests');a.eq(ledger['endpoint_decision_records'],len(records),label+'.endpoint_returns');a.eq(o['observation_filter'],filter_,label+'.same_filter')
            rejected=[rec['edge'] for rec in records if rec['decision']=='CONTRADICTED_REJECT'];a.eq(filter_['rejected_edge_ids'],rejected,label+'.reject_only_contradicted');a.eq(filter_['retained_unused_edge_ids'],[e for e in filter_['original_actual_unused_edge_ids'] if e not in rejected],label+'.retain_other_edges');a.eq(filter_['unknown_retained_edge_ids'],[rec['edge'] for rec in records if rec['decision']=='UNVERIFIED_RETAIN'],label+'.unknown_retained')
            for rec in records:
                blocked=sorted(set(H)|set(rec['endpoints']));a.eq(rec['validation_excluded'],blocked,label+'.LOO_H_union_endpoints');a.eq(rec['hidden'],H,label+'.LOO_separate_H');a.eq(rec['native_input_hash'],ledger['endpoint_native_input_hash'],label+'.native_shared_bank');a.eq(rec['observed_line_supplied_to_validation_solver'],False,label+'.LOO_independence');a.eq(bool(set(rec['solved'].get('used',[]))&set(blocked)),False,label+'.LOO_fit_exclusion')
            if o['new_pose_estimated']:
                a.eq(bool(set(H)&set(s['fit_input_ids'])),False,label+'.H_fit_excluded');a.eq(pose,s,label+'.actual_Rt');a.eq(s['best_candidate']['accepted'],True,label+'.accepted_selected');a.eq(s['best_candidate']['stable_incidence_guard']['accepted'],True,label+'.selected_normal_guard')
                for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projected_once.'+str(k),1e-12)
            else:a.eq(pose,o['initial_pose'],label+'.full_original_fallback')
            a.eq(o['native_points'][8],native[8],label+'.center');prediction=copy.deepcopy(v['original_fresh_detector_prediction']);selected=prediction['selected_index']
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
        a.eq(sparse(packet_totals),sparse(r['packet_compatibility_counts']),'actual.packet_compatibility');a.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'loading_forwards');a.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');a.eq(r['model_forwards'],dict(detector=151,N3=150,detector_internal_initialization_calls=1),'actual_internal_init')
        for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):a.eq(totals['base'][key],150,'actual.base_route.'+key)
        for key in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_repair_paths','fresh_filtered_local_paths'):
            for phase in ('started','complete'):a.eq(totals['fresh'][key+'_'+phase],150,'actual.fresh_route.'+key+'_'+phase)
        a.eq(totals['fresh']['reported_primary_views_from_single_filtered_local'],150,'actual.single_output_per_call');a.eq(totals['fresh']['fresh_native_endpoint_requests_started'],totals['fresh']['fresh_native_endpoint_requests_complete'],'actual.endpoint_completed')
        a.eq(totals['direct']['generic_calls'],0,'actual.sharedbank_Generic_reuse');a.eq(totals['direct']['generic_cache_misses'],0,'actual.sharedbank_no_miss');a.eq(totals['bank_capture']['observer_scopes_started'],150,'actual150constructor_scopes');a.eq(totals['bank_capture']['observer_scopes_restored'],150,'actual150constructor_restores');a.eq(totals['bank_capture']['constructor_entries'],totals['bank_capture']['constructor_returns'],'actual_bank_constructor_returns');a.eq(150<=totals['bank_capture']['constructor_entries']<=300,True,'actual150_to300_banks');a.eq(totals['native_reuse'],dict(observer_scopes_started=150,constructor_requests=150,actual_native_bank_reuses=150,observer_scopes_restored=150),'actual150native_reuse_lifecycles')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('image_decode_returns',26),('model_initializations_started',1),('model_initializations_complete',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'missing resource snapshots')
        for snap in res['snapshots']:a.eq(snap['quiet'] and snap['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0
    write(args.output/FILES[2],dict(schema='independent_hybrid_factor_subset_runtime_checks_v16',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,external_declared_metadata=declarations(proto),external_metadata_current_assets_not_verified=True,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True))
    print('FACTOR_SUBSET_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--runtime-input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()

