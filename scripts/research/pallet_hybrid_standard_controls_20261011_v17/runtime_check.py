"""Own-frozen stdlib check of the complete fresh hybrid ordinary runtime.

Read only saved clocks, coordinates, cache/counter records and pose fields.
No GT, image, model, solver, optimizer, projection, Jacobian or SVD is called.
The V16 failed line audit stays FAIL and is not an input to ordinary PnP.
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
DOC=REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'
PRIVATE=Path('/tmp/pallet-hybrid-standard-controls-private-20261011-v17')
SOURCE=REPO/'scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime_check.py'
SOURCE_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
STAGES=('detector','N3_correction','initial_pose','boundary_observation','hybrid_selection_and_standard_PnP','final_pose_reprojection_and_metadata')
METHODS=('HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP');METHOD=METHODS[0];TIMES=('full',*STAGES,'return')
POSE=('R_cf','R_physical','centroid','cf_extents');OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
LIMITS=dict(new_inference_RGB_GT_fit_PnP_projection_Jacobian_SVD_timing_calls=0,
    saved_clock_summary_counter_and_candidate_predicate_consistency_only=True,
    fresh_candidate_pose_J_or_SVD_not_recomputed=True,accuracy_geometry_audit_not_repeated=True,
    numerical_cache_identity_is_saved_witness_not_hardware_authentication=True,
    ordinary_inliers_are_diagnostic_not_four_inlier_acceptance_gate=True,
    original_V16_full_line_scalar_validation_passed=False,
    whole_path_primitive_residual_not_attributed_to_pure_ordinary_solver=True,
    physical_ownership_certified=False,hardware_authenticity_proven=False,
    quiet_snapshots_can_miss_transient_contention=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(v):return {k:n for k,n in v.items() if n}
def public(v):
    if v.get('origin')=='external_readonly_dependency':return None
    p=Path(v['path']);require(v.get('origin','public_repository')=='public_repository' and not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public binding');return REPO/p
def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'unchanged independent timing/statistics source')
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('require','Audit','describe')]
    require({n.name for n in nodes}=={'require','Audit','describe'},'exact stdlib whitelist');tree=ast.Module(body=nodes,type_ignores=[])
    return tree,dict(source=binding(SOURCE),names=['require','Audit','describe'],AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),formula_changes=0)
def paths(p,t):
    names=dict(geometry='GEOMETRY_SEALED.jsonl.gz',core_protocol='PROTOCOL.json',seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',ledgers='CONTROL_LEDGERS.jsonl.gz',population='PARENT_POPULATION_CHECKS.json',validation='VALIDATION_CHECKS.json',validation_protocol='VALIDATION_PROTOCOL.json',validation_started='VALIDATION_STARTED.json',scoring='SCORING_RECEIPT.json',scoring_started='SCORING_STARTED.json',predictions='PREDICTIONS.jsonl.gz',evaluation_protocol='EVALUATION_PROTOCOL.json',scoring_authority='SCORING_AUTHORITY.json')
    runtime=dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json')
    f={k:p/v for k,v in names.items()};f.update({k:t/v for k,v in runtime.items()});f.update(checker=Path(__file__).resolve(),scalar_source=SOURCE,fixed_geometry=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for name in ('runtime.py','deployment.py','pipeline.py','runner.py','evaluation.py','validation_checks.py'):f['code:'+name]=CODE/name
    for label,key in (('runtime_input','runtime_protocol'),('accuracy_input','validation_protocol'),('evaluation_input','evaluation_protocol')):
        for name,v in read(f[key])['inputs'].items():
            target=public(v)
            if target is not None:f[label+':'+name]=target
    return f
def metadata(f,b):
    proto,r,res=[read(f[k]) for k in ('runtime_protocol','receipt','resources')]
    require(proto['schema']=='fresh_hybrid_standard_primary_runtime_protocol_v17' and proto['method']==METHOD and proto['configured_calls']==150 and proto['warmup']==20 and proto['measured']==130 and proto['repeats']==5 and proto['stages']==list(STAGES),'fixed150 fresh ordinary runtime')
    for k,v in (('first_prediction_detector_forwards',2),('other_prediction_detector_forwards',1),('N3_forwards_per_prediction',1),('saved_geometry_used_only_after_return',True),('no_artificial_pre_forward',True),('one_ordinary_solve_supplies_single_output',True),('native_paired_accuracy_arm_not_timed',True),('final_whole_U_SSE_and_rank',True),('final8px_inlier_count_is_diagnostic_only',True),('actual_LOCAL_optimizer_endpoint_line_calls',0)):require(proto[k]==v,'fresh fixed policy '+k)
    require(r['schema']=='fresh_hybrid_standard_primary_runtime_receipt_v17' and r['method']==METHOD and r['complete'] is True and r['status']=='COMPLETE' and r['statistics_official'] is True and r['serialized_rows']==150 and not r.get('error') and not r['cleanup_errors'],'complete fresh runtime')
    for k,alias in (('protocol','runtime_protocol'),('accuracy_protocol','core_protocol'),('raw_rows','rows'),('resources','resources')):require(same(r[k],b[alias]),'runtime byte authority '+k)
    for label,key in (('runtime_input','runtime_protocol'),('accuracy_input','validation_protocol'),('evaluation_input','evaluation_protocol')):
        for name,v in read(f[key])['inputs'].items():
            if label+':'+name in b:require(same(v,b[label+':'+name]),'current frozen '+label+' '+name)
    for key in ('code:'+n for n in ('runtime.py','deployment.py','pipeline.py','runner.py','evaluation.py','validation_checks.py')):require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in proto['inputs'].values()),'unbound runtime source '+key)
    core,seal,receipt,own,check,start=[read(f[k]) for k in ('core_protocol','seal','repair_receipt','validation_protocol','validation','validation_started')]
    require(core['schema']=='fixed_native_hybrid_standard_protocol_v17' and core['methods']==list(METHODS) and seal['schema']=='native_hybrid_standard_geometry_seal_v17' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,'complete paired ordinary geometry')
    for k,alias in (('protocol','core_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','population')):require(same(seal[k],b[alias]),'accuracy sealed bytes '+k)
    require(own['schema']=='native_hybrid_standard_scalar_protocol_v17' and check['schema']=='native_hybrid_standard_scalar_checks_v17' and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==0 and check['inputs']==own['inputs']==start['inputs'] and start['new_production_calls']==0 and same(check['own_protocol'],b['validation_protocol']) and same(start['own_protocol'],b['validation_protocol']) and same(check['checker'],b['code:validation_checks.py']) and same(own['checker'],b['code:validation_checks.py']),'actual complete ordinary geometry PASS')
    failed=read(f['accuracy_input:v16_ready:ready_checks']);require(failed['complete'] is True and failed['passed'] is False and failed['checks']==198134007 and failed['failure_count']==564 and failed['exception'] is None,'V16 complete line audit remains FAIL')
    evaluation,authority,score,ss=[read(f[k]) for k in ('evaluation_protocol','scoring_authority','scoring','scoring_started')];marker=evaluation['score_dispatch']
    require(evaluation['schema']=='postseal_native_hybrid_standard_evaluation_protocol_v17' and authority['schema']=='isolated_native_hybrid_standard_proxy_score_authority_v17' and authority['methods']==list(METHODS) and authority['inputs']==evaluation['inputs'] and authority['score_dispatch']==marker and same(authority['own_evaluation_protocol'],b['evaluation_protocol']) and authority['original_module_globals_modified'] is False and authority['reference_or_error_formula_changed'] is False,'isolated paired scoring authority')
    require(marker['topology_AST_changes']==dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1) and marker['full_canonical_score_function_restore_proven'] is True and marker['score_reference_formula_AST_changes']==0 and marker['original_module_globals_modified'] is False,'unchanged scoring formulas')
    require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True and score['scored_rows']==490 and score['frames']==245 and score['methods']==list(METHODS) and same(score['protocol'],b['core_protocol']) and same(score['geometry_seal'],b['seal']) and same(score['predictions'],b['predictions']),'actual postseal490 scoring')
    require(same(ss['independent_validation'],b['validation_protocol']) and same(ss['geometry_seal'],b['seal']) and ss['configured_scored_rows']==490 and ss['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'scoring after independent ordinary PASS')
    require(res['snapshots']==r['resource_snapshots'] and same(res['protocol'],b['runtime_protocol']) and res['all_recorded_snapshots_quiet'] is True,'recorded resources')
    pre,journal=read(f['preflight']),read(f['journal']);require(pre['passed'] is True and same(pre['protocol'],b['runtime_protocol']) and journal['status']=='COMPLETE' and same(journal['receipt'],b['receipt']) and journal['execution']==r['execution'],'complete journal/preflight')
    return proto,r,res
def ordinary(o,ledger,a,label):
    s=o['solver'];U,H=s['used'],o['hidden_initial'];a.eq(s,ledger['solver'],label+'.same_full_ordinary_solver')
    for k,v in (('solver','OBSERVATION_ONLY_STANDARD'),('prior_used',False),('known_dimension_constraint_used',False),('hidden_initial_excluded',True),('reprojected_points_reused_as_observations',False)):a.eq(s[k],v,label+'.ordinary.'+k)
    a.eq(s['hidden'],H,label+'.same_H');a.eq(bool(set(U)&set(H)) or 8 in U,False,label+'.U_no_H_center');a.eq(o['ordinary_inlier_count_is_diagnostic_only'],True,label+'.diagnostic8px');a.eq(s['operation_counts'],ledger['ordinary_operation_counts'],label+'.ordinary_ledger')
    for c in s['all_candidate_solutions']:
        a.eq(c['actual_fit_input_ids'],U,label+'.candidate_whole_U_fit');a.eq(c['score_count'],len(U),label+'.whole_U_score_count');a.near(c['objective_px2'],c['sse_px2'],label+'.whole_U_SSE');a.eq(bool(set(c['actual_fit_input_ids'])&set(H)),False,label+'.candidate_no_H')
    best=s.get('best_candidate');g=s['geometry'].get('jacobian')
    if best:
        a.near(best['sse_px2'],min(c['sse_px2'] for c in s['all_candidate_solutions']),label+'.best_whole_U_SSE');a.eq(s['policy']['numerical_score_tie_px2'],1e-8,label+'.inherited_tie_constant')
        for c in s['alternatives']:a.eq(c['numerical_objective_tie'],abs(c['sse_px2']-best['sse_px2'])<=1e-8,label+'.stored_alternative_tie')
        a.eq(s['unresolved_ambiguity'],any(c['numerical_objective_tie'] for c in s['alternatives']),label+'.tie_routing')
    if g:
        sv=g['singular_values'];a.eq(g['numerical_rank'],sum(v>sv[0]*1e-10 for v in sv) if sv else 0,label+'.stored_rank_recount');a.eq(o['stored_jacobian_fit_ids'],U,label+'.J_uses_all_U')
    else:a.eq(o['stored_jacobian_fit_ids'],[],label+'.no_J_witness')
    if s['available']:
        a.eq(len(U)>=4,True,label+'.standard_min4_U');a.eq(s['fit_input_ids'],U,label+'.accepted_full_U');a.eq(g['numerical_rank'],6,label+'.whole_U_rank6');a.eq(s['unresolved_ambiguity'],False,label+'.no_numeric_tie_veto');a.eq(o['actual_pose'],s,label+'.NEW_Rt_solver')
        a.eq(s['final_inliers'],best['inlier_ids'],label+'.diagnostic_inliers');a.eq(o['reprojected_ids'],H,label+'.H_once')
        for k in H:a.near(o['native_points'][k],s['projected'][k],label+'.H_projection.'+str(k),1e-12)
    else:
        a.eq(o['actual_pose'],o['initial_pose'],label+'.full_original_pose_fallback');a.eq(o['native_points'],o['original_native_N3_points'],label+'.full_original_N3_coordinates');a.eq(o['reprojected_ids'],[],label+'.no_new_projection')
    a.eq(o['new_pose_estimated'],s['available'],label+'.NEW_is_solver_output');a.eq(o['fallback_used'],not s['available'] and o['initial_pose']['available'],label+'.declared_original_fallback');a.eq(o['reprojections_reused_as_observations'],False,label+'.no_refit')

def guard(args,stage):
    for p in (args.input,args.runtime_input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink directories')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V17 output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)
def external(proto):return {k:v for k,v in proto['inputs'].items() if v.get('origin')=='external_readonly_dependency'}
def freeze(args):
    guard(args,'freeze');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};proto,_,_=metadata(f,b);_,info=extraction()
    write(args.output/FILES[0],dict(schema='independent_hybrid_standard_runtime_check_protocol_v17',inputs=b,helpers=info,limits=LIMITS,external_declared_metadata=external(proto),external_metadata_current_assets_not_verified=True,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=paths(args.input,args.runtime_input);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,info=extraction();proto,r,res=metadata(f,b)
    require(own['schema']=='independent_hybrid_standard_runtime_check_protocol_v17' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['external_declared_metadata']==external(proto) and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    write(args.output/FILES[1],dict(own_protocol=ob,inputs=b,new_production_calls=0));env=dict(math=math);exec(compile(tree,str(SOURCE)+':unchanged_stdlib_Audit_describe','exec'),env)
    a=env['Audit']();totals={};timings={k:[] for k in TIMES};count=0;error=None;packet_totals=Counter()
    try:
        panel=proto['panel'];a.eq(len(panel),26,'panel26');a.eq(len({v['session_id'] for v in panel}),13,'sessions13');a.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session')
        identities=proto['normalization']['joined_frame_identities'];authority={v['id']:v for v in identities};a.eq(len(authority),245,'normalization245');a.eq(len(identities),245,'no_duplicate_normalization');a.eq(r['normalization'],proto['normalization'],'same_normalization')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];a.eq(proto['schedule'],schedule,'fixed150schedule')
        ids={v['frame_id'] for v in panel};refs={};seen=set();fixed={}
        for v in rows(f['geometry']):
            require(v['method'] in METHODS and v['id'] in authority and (v['method'],v['id']) not in seen,'duplicate/foreign accuracy row');seen.add((v['method'],v['id']))
            if v['method']==METHOD and v['id'] in ids:
                refs[v['id']]={k:v[k] for k in ('native_points','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')};pose=v['actual_pose'];refs[v['id']]['pose']={k:pose[k] for k in ('available','selected_hypothesis',*POSE,*OPTIONAL) if k in pose}
        a.eq(seen,{(m,k) for m in METHODS for k in authority},'full245paired490');a.eq(set(refs),ids,'all26references')
        for v in rows(f['fixed_geometry']):
            if v['id'] in ids:require((v['method'],v['id']) not in fixed,'duplicate fixed reference');fixed[v['method'],v['id']]=v['native_points']
        a.eq(set(fixed),{(m,k) for m in ('BASE','N3_SUBPIX') for k in ids},'fixed_BASE_N3_panel')
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
            o=v['full_fresh_output'];s=o['solver'];ledger=v['fresh_standard_ledger'];native=o['native_N3_points'];reference=refs[v['id']]
            for k,value in reference.items():
                if k!='pose':a.near(o[k],value,label+'.saved_parity.'+k,relative=0)
            pose,expected=o['actual_pose'],reference['pose'];a.eq(pose['available'],expected['available'],label+'.pose_availability');a.eq(pose.get('selected_hypothesis'),expected.get('selected_hypothesis'),label+'.dimension')
            if pose['available']:
                for k in OPTIONAL:a.eq(k in pose,k in expected,label+'.optional_pose_field.'+k)
                for k in (*POSE,*(k for k in OPTIONAL if k in expected)):a.near(pose[k],expected[k],label+'.saved_Rt.'+k,relative=0)
            a.near(o['original_base_points'],fixed['BASE',v['id']],label+'.original_BASE',relative=0);a.near(native,fixed['N3_SUBPIX',v['id']],label+'.original_N3',relative=0);a.eq(o['K'],authority[v['id']]['K'],label+'.K');a.eq(o['xyz'],authority[v['id']]['xyz'],label+'.dimensions')
            for k,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True),('original_candidate_center_preservation_checked_inside_predict',True)):a.eq(v[k],value,label+'.'+k)
            for k,value in (('fresh_complete_capture',True),('fresh_capture_input_only',True),('fresh_hybrid_corner_selection_recomputed_inside_call',True),('fresh_point_solver_recomputed_inside_call',True),('fresh_standard_solver_recomputed_inside_call',True),('fresh_actual_selected_coordinate_bank_reused_inside_call',True),('native_paired_accuracy_arm_executed_inside_primary_call',False),('fresh_endpoint_LOO_recomputed_inside_call',False),('cached_accuracy_row_used',False),('saved_accuracy_coordinates_or_pose_as_input',False),('parent_completion_receipt_used_as_deployment_input',False),('line_factor_count',0)):a.eq(o[k],value,label+'.fresh.'+k)
            a.eq(o['fresh_standard_ledger'],ledger,label+'.same_full_ledger');a.eq(ledger['ordinary_solver_calls'],1,label+'.one_ordinary');a.eq(ledger['reported_primary_views'],1,label+'.one_view')
            for k in ('line_factors','endpoint_LOO_calls','LOCAL_optimizer_calls','native_paired_control_calls'):a.eq(ledger[k],0,label+'.no_extra_'+k)
            ordinary(o,ledger,a,label);a.eq(ledger['input_points'],o['input_points'],label+'.selected_q');a.eq(ledger['original_native_N3_points'],native,label+'.native_q');a.eq(o['native_points'][8],native[8],label+'.center')
            prediction=copy.deepcopy(v['original_fresh_detector_prediction']);selected=prediction['selected_index']
            if selected is not None:a.eq(prediction['candidates'][selected]['keypoints_xy'][8],native[8],label+'.raw_center');prediction['candidates'][selected]['keypoints_xy']=o['native_points']
            a.eq(o['prediction']['selected_index'],selected,label+'.selected_index');a.eq(o['prediction']['candidates'],prediction['candidates'],label+'.all_conf_box_preserved')
            d=v['counter_deltas'];a.eq(set(d),{'cv','optimizer','head','base','fresh','ordinary','svd','bank_capture','model'},label+'.all9counter_groups');a.eq(d['head'],dict(entries=1,returns=1),label+'.head');a.eq(d['model'],dict(detector=2 if i==0 else 1,N3=1),label+'.actual_model_forwards')
            reuse=v['fresh_point_bank_reuse'];a.eq(reuse,o['fresh_actual_selected_bank_reuse'],label+'.same_output_reuse');a.eq(reuse,ledger['actual_selected_point_bank_reuse'],label+'.same_ledger_reuse');a.eq(reuse['schema'],'fresh_hybrid_standard_selected_bank_reuse_v17',label+'.reuse_schema')
            for k,value in (('same_coordinate_bank_reused',True),('constructor_restored',True),('existing_numeric_keys_preserved',True),('cache_reconstructed_from_serialized_Rt',False)):a.eq(reuse[k],value,label+'.reuse.'+k)
            n=reuse['actual_constructor_count'];a.eq(1<=n<=2,True,label+'.captured_bank_count');a.eq(d['bank_capture'],dict(observer_scopes_started=1,constructor_entries=n,constructor_returns=n,observer_scopes_restored=1),label+'.constructor_lifecycle')
            point=v['fresh_point_bank_ledger'];a.eq(point,o['fresh_point_coordinate_bank_ledger'],label+'.same_original_point_ledger');a.eq(point['coordinate_bank_count'],len(point['coordinate_banks']),label+'.original_bank_count');a.eq(point['coordinate_bank_count'],n,label+'.same_actual_registry')
            banks=[w for w in point['coordinate_banks'] if w['input_hash']==s['input_hash']];a.eq(len(banks),1,label+'.one_selected_original_bank');a.eq(reuse['selected_input_hash'],s['input_hash'],label+'.selected_hash')
            before,after=reuse['selected_bank_before'],reuse['selected_bank_after'];delta={k:after.get(k,0)-before.get(k,0) for k in set(after)|set(before)};a.eq(delta,reuse['ordinary_bank_delta'],label+'.bank_after_minus_before');a.eq(delta,ledger['ordinary_operation_counts'],label+'.ordinary_ledger_delta');a.eq(sparse(d['ordinary']),sparse(delta),label+'.actual_ordinary_delta');a.eq(s['hypothesis_bank_counts'],after,label+'.solver_final_bank')
            for k,value in before.items():a.eq(banks[0][k],value,label+'.original_selected_bank_before.'+k)
            old={(w['dimension_index'],tuple(w['point_ids'])) for w in reuse['numeric_cache_before']};new={(w['dimension_index'],tuple(w['point_ids'])) for w in reuse['numeric_cache_after']};added={(w['dimension_index'],tuple(w['point_ids'])) for w in reuse['newly_requested_numeric_problems']}
            a.eq(len(old),len(reuse['numeric_cache_before']),label+'.no_duplicate_old_cache');a.eq(len(new),len(reuse['numeric_cache_after']),label+'.no_duplicate_new_cache');a.eq(old<=new,True,label+'.old_cache_preserved');a.eq(added,new-old,label+'.actual_new_numeric_keys');a.eq(delta['generic_cache_misses'],len(added),label+'.new_key_misses');a.eq(all(ids_==tuple(s['used']) for _,ids_ in added),True,label+'.new_problems_whole_U_only');a.eq(reuse['additional_whole_U_Generic_calls'],delta['generic_calls'],label+'.actual_new_Generic');a.eq(delta['generic_calls']<=len(added),True,label+'.no_repeat_numeric_key');a.eq(delta['standard_generic_calls'],delta['generic_calls'],label+'.ordinary_only_Generic_phase');a.eq(delta['lm_calls']<=1,True,label+'.one_ordinary_LM_start')
            captured_generic=sum(w['generic_calls'] for w in point['coordinate_banks'])+delta['generic_calls'];captured_lm=sum(w['lm_calls'] for w in point['coordinate_banks'])+delta['lm_calls'];a.eq(d['cv'].get('solvePnPGeneric:entries',0)>=captured_generic,True,label+'.captured_Generic_inside_full_path');a.eq(d['cv'].get('solvePnPRefineLM:entries',0)>=captured_lm,True,label+'.captured_LM_inside_full_path')
            compat=o['packet_compatibility_review'];a.eq(compat['numerical_result_changed'],False,label+'.packet_numeric_preserved');a.eq(compat['original_frame_metadata_preserved'],True,label+'.packet_metadata_preserved');a.eq(compat['original_packet_function_used'],True,label+'.original_packet');packet_totals.update(calls=1,identical_fields_removed=len(compat['removed_identical_fields']))
            require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'unknown CV primitive');require(set(d['svd'])<={'entries','returns','exceptions'},'unknown SVD counter');a.eq(any(d['optimizer'].values()),False,label+'.no_LOCAL_optimizer')
            for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.CV_complete.'+p)
            a.eq(d['svd'].get('entries',0),d['svd'].get('returns',0)+d['svd'].get('exceptions',0),label+'.SVD_complete')
            for key,values in d.items():require(all(type(n_) is int and n_>=0 for n_ in values.values()),'invalid primitive/route counter');totals.setdefault(key,Counter()).update(values)
            count+=1
        a.eq(count,150,'all150');a.eq(len(timings['full']),130,'measured130')
        for k in TIMES:a.near(env['describe'](timings[k]),r['summaries'][k],'summary.'+k)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('ordinary','ordinary_operation_counts'),('svd','live_SVD_counts'),('bank_capture','point_bank_capture_counts')):a.eq(sparse(totals[key]),sparse(r[field]),'actual.sum.'+key)
        a.eq(sparse(packet_totals),sparse(r['packet_compatibility_counts']),'actual.packet_compatibility');a.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'loading_forwards');a.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');a.eq(r['model_forwards'],dict(detector=151,N3=150,detector_internal_initialization_calls=1),'actual_internal_init')
        for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls','final_pose_paths','IMAGE_ROLE_CORNERWISE_HYBRID_final_pose_paths'):a.eq(totals['base'][key],150,'actual.base_route.'+key)
        a.eq(totals['base']['LOO_pose_paths'],totals['base']['IMAGE_ROLE_LOO_pose_paths'],'actual_original_LOO_paths')
        for key in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_standard_paths'):
            for phase in ('started','complete'):a.eq(totals['fresh'][key+'_'+phase],150,'actual.fresh_route.'+key+'_'+phase)
        a.eq(totals['fresh'][METHOD+':started'],150,'actual_method_started');a.eq(totals['fresh'][METHOD+':complete'],150,'actual_method_complete');a.eq(totals['fresh']['reported_primary_views_from_single_standard_solve'],150,'actual_single_output');a.eq(totals['bank_capture']['observer_scopes_started'],150,'actual_capture_scopes');a.eq(totals['bank_capture']['observer_scopes_restored'],150,'actual_capture_restore')
        a.eq(150<=totals['bank_capture']['constructor_entries']==totals['bank_capture']['constructor_returns']<=300,True,'actual_constructor_count');a.eq(totals['bank_capture'].get('constructor_exceptions',0),0,'actual_constructor_exceptions')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('image_decode_returns',26),('model_initializations_started',1),('model_initializations_complete',1)):a.eq(r['execution'][key],value,'actual.execution.'+key)
        for k in ('instruments_restored','opencv_threads_restored','head_hooks_removed','pipeline_closed','environment_restored'):a.eq(r['cleanup'][k],True,'recorded.cleanup.'+k)
        a.eq(r['protected_before'],r['protected_after'],'recorded.protection');require(bool(res['snapshots']),'missing resource snapshots')
        for snap in res['snapshots']:a.eq(snap['quiet'] and snap['temperature_under_80'],True,'recorded.quiet')
        a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and count==150 and a.failure_count==0
    write(args.output/FILES[2],dict(schema='independent_hybrid_standard_runtime_checks_v17',complete=error is None and count==150,passed=passed,inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,external_declared_metadata=external(proto),external_metadata_current_assets_not_verified=True,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,max_absolute_difference=a.max_difference,first_failure_preserved=True))
    print('HYBRID_STANDARD_RUNTIME_CHECK','PASS' if passed else 'FAIL',a.checks,flush=True)
    if not passed:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--runtime-input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)
if __name__=='__main__':main()
