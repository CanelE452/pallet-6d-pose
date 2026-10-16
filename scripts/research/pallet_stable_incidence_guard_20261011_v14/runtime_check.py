"""Own-frozen stdlib audit of fresh V14 clocks, counters and selected guard.

Reconstruct scalar incidence Jacobians; recount stored spectra without SVD.
No RGB, GT, models, optimizer, PnP, OpenCV or new timing is invoked.
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

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
PRIVATE=Path('/tmp/pallet-stable-incidence-guard-private-20261011-v14')
CLOCK=REPO/'scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime_check.py'
CLOCK_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
SCALAR=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
SCALAR_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
GEOMETRY_SHA='fe4c73e91094c09637769e3fcc7a814881dfc2bfed08da6af255ecadb767e2e5'
STAGES=('detector','N3_correction','initial_pose','boundary_observation','endpoint_validation_and_filtered_local','final_pose_reprojection_and_metadata')
METHOD='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE';TIMES=('full',*STAGES,'return');POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
LIMITS=dict(new_inference_RGB_GT_PnP_optimizer_OpenCV_SVD_timing_calls=0,independent_scalar_selected_incidence_J=True,
    stored_spectrum_recount_not_new_SVD=True,full_parent_candidate_geometry_audit_not_repeated=True,
    stored_clock_and_dispatch_consistency_not_hardware_authentication=True,physical_ownership_certified=False,
    global_uniqueness_proven=False,quiet_snapshots_can_miss_transient_competition=True,
    handled_optimizer_branch_exceptions_not_complete_frame_failure=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=digest.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(d):return {k:v for k,v in d.items() if v}
def public(value):
    path=Path(value['path']);require(not path.is_absolute() and '..' not in path.parts,'public repository binding required');return REPO/path
def extraction():
    require(binding(CLOCK)['sha256']==CLOCK_SHA and binding(SCALAR)['sha256']==SCALAR_SHA and binding(CODE/'validation_checks.py')['sha256']==GEOMETRY_SHA,'exact frozen independent sources')
    selected=[n for n in ast.parse(CLOCK.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('require','Audit','describe')]
    require({n.name for n in selected}=={'require','Audit','describe'},'clock whitelist')
    clock=ast.Module(body=selected,type_ignores=[]);node=[n for n in ast.parse((CODE/'validation_checks.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='extraction'];require(len(node)==1,'incidence extraction authority')
    env=dict(ast=ast,hashlib=hashlib,binding=binding,require=require,SOURCE=SCALAR,SOURCE_SHA=SCALAR_SHA,CODE=CODE)
    exec(compile(ast.Module(body=node,type_ignores=[]),'immutable V14 stdlib whitelist','exec'),env)
    scalar,policy,info=env['extraction']()
    return clock,scalar,policy,dict(clock_source=binding(CLOCK),clock_AST_sha256=hashlib.sha256(ast.dump(clock,include_attributes=False).encode()).hexdigest(),incidence=info,incidence_extractor_source=binding(CODE/'validation_checks.py'),Audit_ok_adapter='boolean equality alias only')
def paths(p,t):
    names=dict(geometry='GEOMETRY_SEALED.jsonl.gz',core_protocol='PROTOCOL.json',seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',validation='VALIDATION_CHECKS.json',validation_protocol='VALIDATION_PROTOCOL.json',validation_started='VALIDATION_STARTED.json',scoring='SCORING_RECEIPT.json',scoring_started='SCORING_STARTED.json',predictions='PREDICTIONS.jsonl.gz',evaluation_protocol='EVALUATION_PROTOCOL.json',original_score_authority='SCORING_AUTHORITY.json',first_score_failure='SCORING_DISPATCH_FIRST_FAILURE.json',score_review_protocol='SCORING_DISPATCH_REVIEW_PROTOCOL.json',score_review_started='SCORING_DISPATCH_REVIEW_STARTED.json',score_review_authority='SCORING_AUTHORITY_SUPPLEMENT.json',statistics_authority='STATISTICS_AUTHORITY_SUPPLEMENT.json')
    runtime=dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json')
    out={k:p/v for k,v in names.items()};out.update({k:t/v for k,v in runtime.items()});out.update(checker=Path(__file__).resolve(),clock_source=CLOCK,scalar_source=SCALAR,fixed_geometry=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for n in ('runtime.py','deployment.py','pipeline.py','runner.py','evaluation.py','evaluation_review.py','validation_checks.py'):out['code:'+n]=CODE/n
    for key in ('contract','panel','normalized_helper','unchanged_fresh_finite_deployment','unchanged_finite_solver'):out['runtime_source:'+key]=public(read(out['runtime_protocol'])['inputs']['runtime:'+key])
    for key,value in read(out['validation_protocol'])['inputs'].items():out['accuracy_input:'+key]=public(value)
    for key,value in read(out['score_review_protocol'])['inputs'].items():out['score_review_input:'+key]=public(value)
    return out
def metadata(f,b):
    proto,r,res=read(f['runtime_protocol']),read(f['receipt']),read(f['resources'])
    require(proto['schema']=='fresh_stable_incidence_primary_runtime_protocol_v14' and proto['method']==METHOD and proto['configured_calls']==150 and proto['warmup']==20 and proto['measured']==130 and proto['repeats']==5 and proto['stages']==list(STAGES),'fixed fresh scope')
    for k,v in dict(saved_geometry_used_only_after_return=True,one_finite_bank_call_supplies_single_output=True,stable_incidence_guard_inside_final_stage_and_full_interval=True,accepted_H_projection_reassigned_by_guard=False,guard_additional_optimizer_PnP_candidate_selection_calls=0,first_prediction_detector_forwards=2,other_prediction_detector_forwards=1,N3_forwards_per_prediction=1,no_artificial_pre_forward=True).items():require(proto[k]==v,'fresh policy '+k)
    require(r['schema']=='fresh_stable_incidence_primary_runtime_receipt_v14' and r['method']==METHOD and r['complete'] is True and r['status']=='COMPLETE' and r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'complete successful fresh runtime')
    for k,alias in (('protocol','runtime_protocol'),('accuracy_protocol','core_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[k],b[alias]),'runtime byte authority '+k)
    aliases={'code':'code:runtime.py','deployment':'code:deployment.py','accuracy_protocol':'core_protocol','geometry_gate':'code:evaluation_review.py','original_guard_adapter':'code:evaluation.py','geometry_checker':'code:validation_checks.py',**{name:alias for alias,name in names_for_runtime().items()}}
    for key,alias in aliases.items():require(same(proto['inputs']['runtime:'+key],b[alias]),'frozen runtime input '+key)
    for key in ('contract','panel','normalized_helper','unchanged_fresh_finite_deployment','unchanged_finite_solver'):require(same(proto['inputs']['runtime:'+key],b['runtime_source:'+key]),'unchanged timing source '+key)
    for key in ('fixed_geometry','code:pipeline.py','code:runner.py','scalar_source'):require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in proto['inputs'].values()),'unbound frozen source '+key)
    core,seal,repair,check,own,score=[read(f[k]) for k in ('core_protocol','seal','repair_receipt','validation','validation_protocol','scoring')]
    require(core['methods']==[METHOD] and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and repair['complete'] is True and repair['cleanup_completed'] is True and repair['error'] is None and repair['cleanup_error'] is None,'complete restored GT-free guard geometry')
    for k,alias in (('protocol','core_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[k],b[alias]),'accuracy seal '+k)
    require(own['schema']=='stable_incidence_scalar_protocol_v14' and check['schema']=='stable_incidence_scalar_checks_v14' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['inputs']==own['inputs'] and same(check['own_protocol'],b['validation_protocol']) and same(check['checker'],b['code:validation_checks.py']),'actual independent guard PASS')
    for k,value in own['inputs'].items():require(same(value,b['accuracy_input:'+k]),'current independently checked input '+k)
    for k,alias in (('geometry','geometry'),('ledgers','ledgers'),('repair_protocol','core_protocol'),('geometry_seal','seal'),('repair_receipt','repair_receipt'),('parent_population_checks','population'),('checker','code:validation_checks.py')):require(same(own['inputs'][k],b[alias]),'validated byte authority '+k)
    started=read(f['validation_started']);require(started['inputs']==own['inputs'] and same(started['own_protocol'],b['validation_protocol']) and started['new_production_calls']==0,'guard inspector STARTED')
    failure,old,review,dispatch,ss,stat=[read(f[k]) for k in ('first_score_failure','original_score_authority','score_review_protocol','score_review_authority','score_review_started','statistics_authority')]
    require(failure['complete'] is False and failure['failed_before_SCORING_STARTED'] is True and failure['failed_before_GT_kernel_or_score_rows'] is True and failure['GT_score_calls']==failure['scored_rows']==0 and failure['source_files_changed'] is False and same(failure['source'],b['code:evaluation.py']),'original pre-GT failed dispatch preserved')
    for k,alias in (('EVALUATION_PROTOCOL.json','evaluation_protocol'),('SCORING_AUTHORITY.json','original_score_authority')):require(same(failure['authority'][k],b[alias]),'preserved first authority '+k)
    require(old['schema']=='unchanged_V13_proxy_score_dispatch_for_guard_v14' and same(old['own_evaluation_protocol'],b['evaluation_protocol']),'original declaration is distinct from actual dispatch')
    marker=review['review'];require(review['schema']=='isolated_guard_score_dispatch_review_protocol_v14' and review['method']==METHOD and review['frames']==245 and review['original_failed_dispatch_preserved'] is True and review['frozen_before_own_GT_score_calls'] is True and marker['local_method_import_AST_changes']==1 and marker['full_canonical_score_function_restore_proven'] is True and marker['score_reference_formula_AST_changes']==0 and marker['immutable_parent_method_tuple_modified'] is False and marker['original_module_globals_modified'] is False and marker['original_module_package_context_preserved'] is True,'separate isolated successful dispatch policy')
    for k,value in review['inputs'].items():require(same(value,b['score_review_input:'+k]),'current dispatch input '+k)
    require(same(review['inputs']['review_code'],b['code:evaluation_review.py']) and same(review['inputs']['original_dispatch_failure'],b['first_score_failure']) and dispatch['inputs']==ss['inputs']==review['inputs'] and dispatch['review']==ss['review']==marker and same(dispatch['own_protocol'],b['score_review_protocol']) and same(dispatch['started'],b['score_review_started']) and same(ss['protocol'],b['score_review_protocol']) and ss['GT_score_calls_before_STARTED']==0 and ss['configured_scored_rows']==245 and dispatch['score_reference_formulas_changed'] is False,'actual local-import dispatch STARTED/authority')
    require(stat['inputs']==review['inputs'] and stat['original_full_statistics_function_changed'] is False and stat['frozen_before_original_STATISTICS_STARTED_and_own_arithmetic'] is True and same(stat['actual_score_receipt'],b['scoring']) and same(stat['own_review_protocol'],b['score_review_protocol']) and same(stat['successful_score_dispatch_authority'],b['score_review_authority']),'unchanged postscore statistics authority')
    require(score['complete'] is True and score['scored_rows']==score['frames']==245 and score['methods']==[METHOD] and same(score['protocol'],b['core_protocol']) and same(score['geometry_seal'],b['seal']) and same(score['predictions'],b['predictions']),'complete actual postseal scoring')
    scored=read(f['scoring_started']);require(same(scored['independent_validation'],b['validation_protocol']) and same(scored['geometry_seal'],b['seal']) and scored['configured_scored_rows']==245 and scored['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'scoring after actual guard PASS')
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['all_recorded_snapshots_quiet'] is True,'resource identity')
    pre,journal=read(f['preflight']),read(f['journal']);require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and journal['status']=='COMPLETE' and same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'completed preflight/journal')
    return proto,r,res
def names_for_runtime():
    return dict(geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',validation_protocol='VALIDATION_PROTOCOL.json',validation='VALIDATION_CHECKS.json',validation_started='VALIDATION_STARTED.json',evaluation_protocol='EVALUATION_PROTOCOL.json',original_score_authority='SCORING_AUTHORITY.json',first_score_failure='SCORING_DISPATCH_FIRST_FAILURE.json',score_review_protocol='SCORING_DISPATCH_REVIEW_PROTOCOL.json',score_review_started='SCORING_DISPATCH_REVIEW_STARTED.json',score_review_authority='SCORING_AUTHORITY_SUPPLEMENT.json',statistics_authority='STATISTICS_AUTHORITY_SUPPLEMENT.json',scoring='SCORING_RECEIPT.json',scoring_started='SCORING_STARTED.json',predictions='PREDICTIONS.jsonl.gz')
def selected_guard(o,ledger,h,a,label,policy):
    old=ledger['full_solver'];s=o['solver'];w=o['stable_incidence_guard'];gl=o['fresh_guard_ledger'];available=bool(old['available'])
    a.eq(w,gl['stable_incidence_guard'],label+'.guard_witness');a.eq(w,s['stable_incidence_guard'],label+'.solver_guard');a.eq(gl['guard_counts'],w['counts'],label+'.guard_counts');a.eq(w['policy'],policy,label+'.guard_policy');a.eq(w['original_solver_semantic_sha256'],semantic(old),label+'.one_parent_solver_SHA')
    a.eq(w['original_all_candidates_semantic_sha256'],semantic(old.get('all_candidate_solutions',[])),label+'.all_candidates_SHA')
    for k in ('all_candidate_solutions','best_candidate','optimizer_attempts','operation_counts'):
        if k in old:a.eq(s[k],old[k],label+'.immutable_'+k)
    for k,value in (('gate_applied',available),('original_numerical_pose_available',available),('selected_candidate_reselected',False),('global_uniqueness_proven',False),('physical_ownership_truth','UNKNOWN')):a.eq(w[k],value,label+'.'+k)
    accepted=False
    if available:
        c=old['best_candidate'];ids=c['fit_input_ids'];edges=c['fit_line_edges'];g=c['point_line_geometry'];lines=[dict(edge=e,normal=l['normal'],offset=l['offset']) for e in edges for l in old['actual_source_lines'] if l['edge']==e];require(len(lines)==len(edges),'selected actual line source')
        a.eq(w['selected_candidate'],c,label+'.selected_candidate');a.eq(w['actual_fit_input_ids'],ids,label+'.actual_points');a.eq(w['actual_fit_line_edges'],edges,label+'.actual_edges');a.eq(w['actual_normal_lines'],lines,label+'.actual_lines');a.eq(w['hidden'],o['hidden_initial'],label+'.H');a.eq(bool(set(ids)&set(o['hidden_initial'])) or 8 in ids,False,label+'.H_center_excluded');a.eq(bool(set(edges)&set(old['consumed_edges'])),False,label+'.no_point_line_double_vote')
        extents=h['dimensions'](o['xyz'])[c['dimension_index']]
        for k,value in (('selected_dimension_index',c['dimension_index']),('selected_extents',extents),('selected_final_z',c['final_z'])):a.eq(w[k],value,label+'.'+k)
        R,uv,J,depths=h['forward'](extents,c['final_z'],o['K']);a.near(c['R_cf'],R,label+'.selected_R');a.near(c['R_physical'],h['physical_rotation'](R,c['dimension_index']),label+'.physical_R');a.near(c['projected'],uv,label+'.selected_projection');a.eq(all(math.isfinite(d) and d>1e-9 for d in depths),True,label+'.selected_depth')
        _,observed=h['factor_rows'](uv,J,o['input_points'],ids,lines);_,modeled=h['factor_rows'](uv,J,o['input_points'],ids,lines,True);indices=list(range(2*len(ids)));end=2*len(ids)
        for block in g['edge_blocks']:a.eq(block['scalar_start'],end,label+'.finite_partition');indices.extend((end,end+1));end=block['scalar_end']
        a.eq([v['edge'] for v in g['edge_blocks']],edges,label+'.finite_edge_identity');a.eq(w['incidence_source_row_indices'],indices,label+'.normal_row_indices');a.eq(w['finite_scalar_rows'],end,label+'.finite_rows');a.eq(w['incidence_scalar_rows'],len(indices),label+'.normal_rows');a.eq(w['removed_longitudinal_rows'],end-len(indices),label+'.removed_hinges')
        for name,matrix in (('observed',observed),('modeled',modeled)):a.eq(w[name+'_normal_J'],[g[name+'_normal_J'][i] for i in indices],label+'.exact_'+name);a.near(w[name+'_normal_J'],matrix,label+'.independent_'+name)
        geometry=w.get('stable_incidence_geometry')
        if geometry is not None:
            split=2*len(ids);matrices=dict(observed_normal_joint=observed,modeled_normal_joint=modeled,point_only=modeled[:split],observed_normal_lines_only=observed[split:],modeled_normal_lines_only=modeled[split:]);ranks={k:h['rank_witness'](geometry[k],m,a,label+'.incidence.'+k) for k,m in matrices.items()}
            weak={name:len(geometry[name+'_normal_joint']['singular_values'])!=6 or geometry[name+'_normal_joint']['singular_values'][-1]<geometry[name+'_normal_joint']['singular_values'][0]*1e-4 for name in ('observed','modeled')};rank=all(ranks[k]==6 for k in ('observed_normal_joint','modeled_normal_joint'));state='PASSED_STABLE_INCIDENCE' if rank and not any(weak.values()) else 'STABLE_INCIDENCE_INSUFFICIENT' if not rank else 'STABLE_INCIDENCE_WEAK';accepted=state=='PASSED_STABLE_INCIDENCE'
            a.eq(w['guard_state'],state,label+'.incidence_route');a.eq(w['observed_weak'],weak['observed'],label+'.observed_weak');a.eq(w['modeled_weak'],weak['modeled'],label+'.modeled_weak');a.eq(w['complete_rank_info_calls_from_source'],5,label+'.source_rank_reports')
            a.eq(geometry['point_ids'],ids,label+'.geometry_points');a.eq(geometry['line_edges'],edges,label+'.geometry_edges');a.eq(geometry['modeled_normals_held_fixed_for_derivative'],True,label+'.held_fixed');a.eq(geometry['global_unique_pose_proven'],False,label+'.local_only')
        else:a.eq(w['guard_state'],'STABLE_INCIDENCE_NUMERICAL_FAILURE',label+'.typed_guard_numeric_failure');a.eq(isinstance(w.get('error'),dict),True,label+'.guard_error')
    else:a.eq(w['guard_state'],'PARENT_HAS_NO_NEW_POSE',label+'.no_new_noop')
    a.eq(w['accepted'],accepted,label+'.accepted');expected=copy.deepcopy(old)
    if available and not accepted:expected.update(available=False,pose_available=False,new_pose_estimated=False,no_pose=True,state=w['guard_state'],reason='selected actual fit lacks sufficient stable point/line incidence',output_status='NO_NEW_POSE',local_refinement_estimated=False,rejected_finite_pose={k:copy.deepcopy(old.get(k)) for k in ('state','R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry')},R_cf=None,R_physical=None,centroid=None,cf_extents=None,projected=None,fit_input_ids=[],fit_line_edges=[],input_inliers=[],final_inliers=[],inliers=[],final_line_inliers=[],point_inlier_count=0,unique_line_inlier_count=0,total_inlier_factor_count=0,points_final=o['input_points'],hidden_reprojected=False)
    expected['stable_incidence_guard']=w;a.eq(s,expected,label+'.single_guarded_solver')
    for k,value in (('guard_calls',1),('geometry_info_calls',int(available)),('new_optimizer_calls',0),('new_PnP_calls',0),('actual_rank_info_dispatches_independently_counted',False)):a.eq(w['counts'][k],value,label+'.guard_op.'+k)
    for k,value in (('saved_accuracy_row_or_guard_decision_used',False),('original_finite_optimizer_run_once',True),('new_guard_optimizer_PnP_candidate_selection_calls',0)):a.eq(gl[k],value,label+'.guard_ledger.'+k)
    return old,w
def guard(args,stage):
    for p in (args.input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directory')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V14 output')
    for n in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/n).exists(),'preserve '+n)
def freeze(args):
    guard(args,'freeze');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,_,policy,info=extraction();write(args.output/FILES[0],dict(schema='independent_stable_incidence_runtime_check_protocol_v14',inputs=b,helpers=info,policy=policy,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);clock,scalar,policy,info=extraction();require(own['schema']=='independent_stable_incidence_runtime_check_protocol_v14' and own['inputs']==b and own['helpers']==info and own['policy']==policy and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    proto,r,res=metadata(f,b);write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));env=dict(math=math);exec(compile(clock,str(CLOCK)+':stdlib_AST_only','exec'),env);h=dict(math=math);exec(compile(scalar,str(SCALAR)+':stdlib_AST_only','exec'),h)
    a=env['Audit']();a.ok=lambda value,label:a.eq(bool(value),True,label);totals={};timings={k:[] for k in TIMES};count=0;error=None
    try:
        panel=proto['panel'];a.eq(len(panel),26,'panel26');a.eq(len({v['session_id'] for v in panel}),13,'sessions13');a.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session');authority={v['id']:v for v in proto['normalization']['joined_frame_identities']};a.eq(len(authority),245,'normalization245');a.eq(r['normalization'],proto['normalization'],'same_normalization')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];a.eq(proto['schedule'],schedule,'fixed150schedule');ids={v['frame_id'] for v in panel};refs={};seen=set();fixed={}
        for v in rows(f['geometry']):
            require(v['method']==METHOD and v['id'] in authority and v['id'] not in seen,'foreign/duplicate accuracy row');seen.add(v['id'])
            if v['id'] in ids:refs[v['id']]={**{k:v[k] for k in ('native_points','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')},'pose':{k:v['actual_pose'][k] for k in ('available','selected_hypothesis',*POSE,*OPTIONAL) if k in v['actual_pose']}}
        a.eq(seen,set(authority),'all245accuracy');a.eq(set(refs),ids,'panel26references')
        for v in rows(f['fixed_geometry']):
            if v['id'] in ids:require((v['method'],v['id']) not in fixed,'duplicate fixed reference');fixed[v['method'],v['id']]=v['native_points']
        a.eq(set(fixed),{(m,i) for m in ('BASE','N3_SUBPIX') for i in ids},'fixedBASE_N3panel')
        for i,v in enumerate(rows(f['rows'])):
            require(i<150,'extra fresh row');job=schedule[i];label='row.'+str(i)
            for k,value in job.items():a.eq(v[k],value,label+'.schedule.'+k)
            p=panel[job['image_index']];a.eq((v['id'],v['session'],v['method']),(p['frame_id'],p['session_id'],METHOD),label+'.identity');ticks=v['stage_ticks_ns'];a.eq([t[0] for t in ticks],list(STAGES),label+'.stages');ns=[v['t0_ns'],*[t[1] for t in ticks],v['t1_ns']];require(all(type(t) is int for t in ns) and all(y>x for x,y in zip(ns,ns[1:])),'nonmonotonic clocks')
            for k,x,y in zip((*STAGES,'return'),ns,ns[1:]):a.near(v[k+'_ms'],(y-x)/1e6,label+'.clock.'+k)
            a.near(v['full_ms'],(ns[-1]-ns[0])/1e6,label+'.full_clock');a.near(v['full_ms'],math.fsum(v[k+'_ms'] for k in (*STAGES,'return')),label+'.stage_sum')
            for k in TIMES:
                require(type(v[k+'_ms']) in (int,float) and math.isfinite(v[k+'_ms']) and v[k+'_ms']>0,'nonpositive duration')
                if job['phase']=='measured':timings[k].append(v[k+'_ms'])
            o=v['full_fresh_output'];ledger=v['fresh_repair_ledger'];H=o['hidden_initial'];native=o['native_N3_points'];reference=refs[v['id']];s=o['solver'];pose=o['actual_pose']
            for k,value in reference.items():
                if k!='pose':a.near(o[k],value,label+'.saved_parity.'+k,relative=0)
            expected=reference['pose'];a.eq(pose['available'],expected['available'],label+'.pose_availability');a.eq(pose.get('selected_hypothesis'),expected.get('selected_hypothesis'),label+'.dimension')
            if pose['available']:
                for k in OPTIONAL:a.eq(k in pose,k in expected,label+'.optional_pose_field.'+k)
                for k in (*POSE,*(k for k in OPTIONAL if k in expected)):a.near(pose[k],expected[k],label+'.saved_Rt.'+k,relative=0)
            a.near(o['original_base_points'],fixed['BASE',v['id']],label+'.original_BASE',relative=0);a.near(native,fixed['N3_SUBPIX',v['id']],label+'.original_N3',relative=0);a.eq(o['K'],authority[v['id']]['K'],label+'.K');a.eq(o['xyz'],authority[v['id']]['xyz'],label+'.dimensions')
            for k,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True)):a.eq(v[k],value,label+'.'+k)
            for k,value in (('saved_accuracy_coordinates_or_pose_as_input',False),('fresh_capture_input_only',True),('fresh_point_start_recomputed_inside_call',True),('fresh_endpoint_LOO_recomputed_inside_call',True),('cached_V10_endpoint_decisions_used',False),('reprojections_reused_as_observations',False),('fresh_stable_incidence_recomputed_inside_call',True),('fresh_parent_finite_result_used',True),('parent_finite_solved_again_by_guard',False),('saved_accuracy_guard_or_pose_used',False),('final_stage_includes_stable_incidence_and_prediction_metadata',True),('accepted_H_projection_assigned_again',False)):a.eq(o[k],value,label+'.fresh.'+k)
            a.eq(o['fresh_repair_ledger'],ledger,label+'.finite_ledger');a.eq(o['fresh_guard_ledger'],v['fresh_guard_ledger'],label+'.guard_ledger');a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_finite_bank');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['endpoint_decision_source'],'FRESH_NATIVE_ENDPOINT_LOO',label+'.fresh_endpoint')
            records=ledger['observation_filter']['endpoint_records'];a.eq(ledger['endpoint_validation_requests'],len(records),label+'.endpoint_requests');a.eq(ledger['endpoint_decision_records'],len(records),label+'.endpoint_returns')
            for rec in records:
                blocked=sorted(set(H)|set(rec['endpoints']));a.eq(rec['validation_excluded'],blocked,label+'.LOO_H_union_endpoints');a.eq(rec['hidden'],H,label+'.LOO_H');a.eq(rec['native_input_hash'],ledger['endpoint_native_input_hash'],label+'.shared_native_bank');a.eq(rec['observed_line_supplied_to_validation_solver'],False,label+'.LOO_independence');a.eq(bool(set(rec['solved'].get('used',[]))&set(blocked)),False,label+'.LOO_fit_exclusion')
            old,w=selected_guard(o,ledger,h,a,label,policy);a.eq(bool(set(old['used'])&set(H)),False,label+'.whole_score_pool_no_H');a.eq(old['reprojected_points_reused_as_observations'],False,label+'.no_projected_refit')
            for c in [*old['proposal_solutions'],*old['all_candidate_solutions']]:a.eq(bool(set(c['fit_input_ids'])&set(H)),False,label+'.all_candidate_no_H');a.eq(all(type(d) in (int,float) and math.isfinite(d) and d>0 for d in c['model_camera_depths_m']),True,label+'.stored_positive_depth')
            for c in old['all_candidate_solutions']:
                geometry=c['point_line_geometry'];weak=False;rank=True
                for name in ('observed','modeled'):
                    witness=geometry[name+'_normal_joint'];ss=witness['singular_values'];recount=sum(x>ss[0]*1e-10 for x in ss) if ss else 0;flag=len(ss)!=6 or ss[-1]<ss[0]*1e-4
                    a.eq(witness['numerical_rank'],recount,label+'.finite_rank_recount');a.eq(geometry[name+'_weak'],flag,label+'.finite_weak');rank&=recount==6;weak|=flag
                stable=set(c['fit_input_ids'])<=set(c['final_inliers']) and set(c['fit_line_edges'])<=set(c['final_line_inliers']);reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank else 'WEAK_OBSERVABILITY' if weak else None
                a.eq(c['accepted'],reason is None,label+'.finite_candidate_gate');a.eq(c['rejected'],reason,label+'.finite_candidate_reason')
            if o['new_pose_estimated']:
                a.eq(w['accepted'],True,label+'.approved_NEW');a.eq(pose,s,label+'.actual_Rt');a.eq(s['best_candidate']['accepted'],True,label+'.accepted_finite_selected')
                for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projected_once.'+str(k),1e-12)
            else:a.eq(pose,o['initial_pose'],label+'.full_original_fallback');a.eq(o['native_points'],o['original_native_N3_points'],label+'.full_N3_fallback')
            a.eq(o['native_points'][8],native[8],label+'.center');prediction=copy.deepcopy(v['original_fresh_detector_prediction']);selected=prediction['selected_index']
            if selected is not None:a.eq(prediction['candidates'][selected]['keypoints_xy'][8],native[8],label+'.raw_center');prediction['candidates'][selected]['keypoints_xy']=o['native_points']
            a.eq(o['prediction']['selected_index'],selected,label+'.selected_index');a.eq(o['prediction']['candidates'],prediction['candidates'],label+'.all_conf_box_preserved')
            d=v['counter_deltas'];a.eq(set(d),{'cv','optimizer','head','base','fresh','local','native','model','svd','guard'},label+'.counter_groups');a.eq(d['head'],dict(entries=1,returns=1),label+'.head');a.eq(d['model'],dict(detector=2 if i==0 else 1,N3=1),label+'.actual_model_forwards');op=ledger['local_operation_counts'];a.eq(sparse(d['local']),sparse(op),label+'.finite_work');a.eq(sparse(d['guard']),{k:n for k,n in w['counts'].items() if type(n) is int and n},label+'.guard_work')
            native_delta={k:ledger['endpoint_native_bank_after'].get(k,0)-ledger['endpoint_native_bank_before'].get(k,0) for k in set(ledger['endpoint_native_bank_after'])|set(ledger['endpoint_native_bank_before'])};a.eq(sparse(d['native']),sparse(native_delta),label+'.native_work');require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'unknown CV primitive')
            for key,source in (('entries','local_optimizer_calls'),('returns','local_optimizer_completions'),('exceptions','local_optimizer_exceptions'),('residual_entries','optimizer_residual_calls'),('jacobian_entries','optimizer_jacobian_calls')):a.eq(d['optimizer'].get(key,0),op[source],label+'.actual_optimizer.'+key)
            for phase in ('proposal','refit'):a.eq(op[phase+'_optimizer_calls'],op[phase+'_optimizer_returns']+op[phase+'_optimizer_exceptions'],label+'.handled_branch_exceptions.'+phase)
            for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.CV_complete.'+p)
            for group in ('optimizer','svd'):a.eq(d[group].get('entries',0),d[group].get('returns',0)+d[group].get('exceptions',0),label+'.'+group+'_complete')
            a.eq(v['fresh_point_bank_ledger']['coordinate_bank_count'],1,label+'.one_fresh_sparse_bank')
            for key,values in d.items():require(all(type(n) is int and n>=0 for n in values.values()),'invalid counter');totals.setdefault(key,Counter()).update(values)
            count+=1
        a.eq(count,150,'all150');a.eq(len(timings['full']),130,'measured130')
        for k in TIMES:a.near(env['describe'](timings[k]),r['summaries'][k],'summary.'+k)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('local','local_operation_counts'),('native','native_operation_counts'),('svd','live_SVD_counts'),('guard','guard_operation_counts')):a.eq(sparse(totals[key]),sparse(r[field]),'actual.sum.'+key)
        a.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'actual.loading_forwards');a.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');a.eq(r['model_forwards'],dict(detector=151,N3=150,detector_internal_initialization_calls=1),'actual.internal_init_accounting')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('image_decode_returns',26),('model_initializations_started',1),('model_initializations_complete',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):a.eq(totals['base'][key],150,'actual.base.'+key)
        for key in ('fresh_predict_calls','fresh_sparse_point_paths','fresh_repair_paths','fresh_filtered_local_paths','fresh_stable_guard_paths',METHOD):
            for suffix in ('started','complete'):a.eq(totals['fresh'][key+':'+suffix] if key==METHOD else totals['fresh'][key+'_'+suffix],150,'actual.fresh.'+key+'.'+suffix)
        a.eq(totals['guard']['guard_calls'],150,'actual.guard150');a.eq(totals['guard']['new_optimizer_calls'],0,'actual.guard_no_optimizer');a.eq(totals['guard']['new_PnP_calls'],0,'actual.guard_no_PnP')
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'missing resource snapshots')
        for snap in res['snapshots']:a.eq(snap['quiet'] and snap['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0;write(args.output/FILES[2],dict(schema='independent_stable_incidence_runtime_checks_v14',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True))
    print('STABLE_INCIDENCE_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--runtime-input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
