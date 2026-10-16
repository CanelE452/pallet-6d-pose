"""Own-frozen stdlib audit of every saved point-only closure candidate.

ROOT alone runs freeze/run. Independent scalar Rodrigues, projection, point
Jacobians, costs and decisions are checked; stored spectra are recounted, not
recomputed by SVD. No production imports, fitting, GT, images or models run.
The old complete V16 564-failure audit remains failed.
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
import struct
import traceback

METHOD='HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
SOURCES={
 'projection':('pallet_sparse_local_line_20261010_v9/validation_checks.py','4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'),
 'ordinary':('pallet_same_observation_controls_20261010_v8/validation_checks.py','d9f8079d42515d6c72eac4c75676493766b95395c63e049414fb67fe48631121'),
 'incidence':('pallet_factor_subset_consensus_20261011_v15/validation_checks.py','ad945e8985b9ca19a576d2423e83b5b635a20b1df44495c2b683a2659c752f26'),
 'finite':('pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py','91d014543eba01c142cad9a0c69affd2eb16666913ce13a06a56857f854c98ef'),
 'guard':('pallet_stable_incidence_guard_20261011_v14/pipeline.py','17828a9f0ff7f895898cab761d6aee8e20533a7564c49353ecfe3d03ee7bd6ec')}
DRIVER_SHA='04ad5b052f4666415801b1d8b48fd038dc78a5dd6f4a2e1b571b5ff84f3680ed'
LIMITS=dict(new_production_calls=0,new_GT_RGB_model_training_calls=0,new_production_PnP_optimizer_J_SVD_calls=0,
 independent_scalar_point_J_reconstruction=True,
 all_saved_point_proposals_refits_and_whole_U_candidates_checked=True,point_only=True,
 stored_spectrum_recount_only=True,old_full_V16_564_FAIL_remains_FAIL=True,
 raw_roots_with_no_saved_pose_not_reconstructed=True,OpenCV_LM_iterative_convergence_not_exposed=True,
 local_rank_is_not_global_root_completeness_or_physical_accuracy=True,physical_ownership_truth='UNKNOWN',
 closure_manual_Rodrigues_counts_are_not_total_OpenCV_dispatch=True,
 original_nonfinite_coordinate_binary_hash_not_reconstructed_from_JSON_null=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank raw row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def canonical(n):return ast.dump(n,include_attributes=False)
def select(tree,names):
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require(len(nodes)==len(names) and {n.name for n in nodes}==set(names),'exact AST whitelist');return copy.deepcopy(nodes)

def input_paths(a):
    f={k:a.input/v for k,v in dict(geometry='GEOMETRY.jsonl.gz',ledgers='LEDGERS.jsonl.gz',geometry_protocol='GEOMETRY_PROTOCOL.json',geometry_started='GEOMETRY_STARTED.json',geometry_receipt='GEOMETRY_RECEIPT.json').items()}
    f.update(checker=Path(__file__).resolve(),driver=a.private/'run_geometry.py')
    # The driver input closure is explicit and nonrecursive; no historical
    # JSON inventory is reinterpreted as a new asset dependency.
    core=read(f['geometry_protocol'])
    public=a.repo/'_docs/experiments/pallet_three_head_observation_20261010_v7'
    for key in core['inputs']:
        if key.startswith('source:'):p=a.repo/key.removeprefix('source:')
        elif key.startswith('V7:'):p=public/key.removeprefix('V7:')
        elif key.startswith('CPU:'):p=a.private/'cpu'/key.removeprefix('CPU:')
        elif key in ('solver','pipeline','cpu_runner','tests','runner'):p=a.private/dict(solver='solver.py',pipeline='pipeline.py',cpu_runner='run_cpu.py',tests='test_solver.py',runner='run_geometry.py')[key]
        elif key=='cohort':p=a.repo/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
        elif key in ('completion','old_failed_validation'):p=a.repo/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'/dict(completion='PARENT_POPULATION_CHECKS.json',old_failed_validation='VALIDATION_READY_CHECKS.json')[key]
        elif key=='original_driver':p=a.private.parent/'pallet-finite-support-orthogonal-private-20261011-v19/run_geometry.py'
        else:
            require(key in ('PROTOCOL.json','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json','GEOMETRY_SEALED.jsonl.gz','OBSERVATIONS.jsonl.gz','FIXED_GEOMETRY_SEALED.jsonl.gz'),'undeclared driver input '+key);p=public/key
        f['geometry_input:'+key]=p
    for key,(rel,_) in SOURCES.items():f['scalar:'+key]=a.repo/'scripts/research'/rel
    return f

def extraction(f):
    trees={}
    for key,(_,sha) in SOURCES.items():require(binding(f['scalar:'+key])['sha256']==sha,'immutable scalar '+key);trees[key]=ast.parse(f['scalar:'+key].read_bytes())
    names=('require','safe','Audit','finite','dot','mm','eye','skew','rotation_and_right_J','dimensions','model','forward','physical_rotation','equivalent','eligible','factor_rows','rank_witness','semantic')
    constants={'EDGES','ATOL','RTOL','RANK','TIE','PRIMITIVES'}
    nodes=select(trees['projection'],names);found=set()
    for n in trees['projection'].body:
        if isinstance(n,ast.Assign):
            ids={t.id for target in n.targets for t in ast.walk(target) if isinstance(t,ast.Name)}
            if ids and ids<=constants:nodes.append(copy.deepcopy(n));found|=ids
    require(found==constants,'exact scalar constants');nodes+=select(trees['incidence'],('ranks','incidence'));nodes+=select(trees['finite'],('primary',))
    node=next(n.value for n in trees['guard'].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    require(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args,'literal normal guard policy')
    policy={k.arg:ast.literal_eval(k.value) for k in node.keywords};require(policy['numerical_rank_relative_threshold']==1e-10 and policy['weak_relative_threshold']==1e-4 and policy['hinge_rank_rows_retained']==0,'unchanged gates')
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]));ordinary=ast.Module(body=select(trees['ordinary'],('finite','dot','model','project','native_hash')),type_ignores=[])
    info=dict(exact_scalar_AST_sha256=hashlib.sha256(canonical(tree).encode()).hexdigest(),ordinary_AST_sha256=hashlib.sha256(canonical(ordinary).encode()).hexdigest(),source_bindings={k:binding(f['scalar:'+k]) for k in SOURCES},scalar_function_changes=0,production_imports=0)
    return tree,ordinary,policy,info

def metadata(f,b):
    core,start,r=[read(f[k]) for k in ('geometry_protocol','geometry_started','geometry_receipt')]
    require(core['schema']=='whole_U_closure_geometry_protocol_v21' and core['methods']==[METHOD] and core['frames']==core['rows']==core['ledger_rows']==245 and core['configured_before_geometry'] is True and core['limits']['GT_allowed'] is False,'fixed GT-free245 geometry')
    require(r['schema']=='whole_U_closure_geometry_receipt_v21' and r['complete'] is True and r['error'] is None and r['cleanup_errors']==[] and r['inputs_unchanged'] is r['environment_restored'] is r['counter_dispatch_closed'] is True and all(v is True for v in r['restoration'].values()),'actual complete clean geometry')
    require(core['inputs']==start['inputs']==r['inputs'] and core['protection']==start['protection']==r['protection_before']==r['protection_after'],'exact frozen source/protection')
    require(core['source_adapter_proof']==start['source_adapter_proof']==r['source_adapter_proof'] and core['environment']==start['environment'],'actual frozen environment and driver proof')
    for key,value in core['inputs'].items():require(same(value,b['geometry_input:'+key]),'current driver closure '+key)
    for value in (start['protocol'],r['protocol']):require(same(value,b['geometry_protocol']),'protocol bytes')
    for key in ('geometry','ledgers'):require(same(r[key],b[key]),'closed raw '+key)
    require(b['driver']['sha256']==DRIVER_SHA,'fixed driver source')
    # The whole metadata authority body is reused; no driver module imports.
    env=dict(require=require,read=read,binding=lambda p:{k:v for k,v in binding(p).items() if k!='path'})
    fn=select(ast.parse(f['driver'].read_bytes()),('source_authority',));exec(compile(ast.Module(body=fn,type_ignores=[]),str(f['driver'])+':metadata_only','exec'),env)
    p={k.removeprefix('geometry_input:'):v for k,v in f.items() if k.startswith('geometry_input:')};bb={k.removeprefix('geometry_input:'):v for k,v in b.items() if k.startswith('geometry_input:')}
    seal,cohort,completion=env['source_authority'](p,bb)
    require(core['ordered_ids']==cohort['ids'] and r['actual_execution']==dict(frames_started=245,frames_complete=245,geometry_rows=245,ledger_rows=245),'full ordered execution')
    return core,r,seal,cohort,completion

def candidate(c,row,U,H,h,a,policy,counts,label):
    d=c['dimension_index'];dims=h['dimensions'](row['xyz']);require(type(d) is int and 0<=d<len(dims),'registered candidate dimension')
    ids=c['fit_input_ids'];a.eq(ids,[k for k in U if k in ids],label+'.actual_fit_order');a.eq(c['fit_line_edges'],[],label+'.no_lines');a.ok(not set(ids)&set(H) and 8 not in ids,label+'.no_H_center')
    a.eq(c['cf_extents'],dims[d],label+'.dimensions');a.eq(c['selected_hypothesis'],('REGISTRY_WD','REGISTRY_DW')[d],label+'.dimension_name')
    R,uv,J,depths=h['forward'](dims[d],c['final_z'],row['K']);a.ok(all(math.isfinite(v) and v>1e-9 for v in depths),label+'.positive_depth')
    a.near(h['ordinary']['project'](dims[d],R,c['final_z'][3:],row['K']),uv,label+'.unchanged_V8_projection')
    for key,v in (('R_cf',R),('R_physical',h['physical_rotation'](R,d)),('centroid',c['final_z'][3:]),('projected',uv),('model_camera_depths_m',depths)):a.near(c[key],v,label+'.'+key)
    rr,matrix=h['factor_rows'](uv,J,row['input_points'],ids,[]);g=c['point_line_geometry']
    for key,v in (('point_ids',ids),('line_edges',[]),('edge_blocks',[]),('modeled_normals_held_fixed_for_derivative',True),('global_unique_pose_proven',False)):a.eq(g[key],v,label+'.geometry_'+key)
    for key,v in (('observed_normal_J',matrix),('modeled_normal_J',matrix),('actual_scalar_residuals_px',rr)):a.near(g[key],v,label+'.independent_'+key)
    rank,weak=h['ranks'](g,matrix,matrix,ids,h,a,label+'.point_geometry')
    for name,v in weak.items():a.eq(g[name+'_weak'],v,label+'.'+name+'_weak')
    cost=math.fsum(64.*(math.sqrt(1.+(v/8.)**2)-1.) for v in rr)
    for key,v in (('scalar_residuals_px',rr),('soft_l1_cost',cost),('sse_px2',math.fsum(v*v for v in rr))):a.near(c[key],v,label+'.'+key)
    norms=[math.dist(uv[k],row['input_points'][k]) for k in U];score=c['whole_pool_score'];a.near(score['point_residual_norm_px'],norms,label+'.independent_whole_U_norms')
    ip=[k for k,e in zip(U,score['point_residual_norm_px']) if e<=8.]
    expected=dict(point_ids=U,line_edges=[],point_residual_norm_px=norms,edge_block_norm_px=[],edge_blocks=[],final_inliers=ip,final_line_inliers=[],inlier_count=len(ip),unique_line_inlier_count=0,total_inlier_factor_count=len(ip),factor_sse_px2=math.fsum(v*v for v in norms),factor_truncated_sse_px2=math.fsum(min(v*v,64.) for v in norms),scalar_residuals_px=[uv[k][i]-row['input_points'][k][i] for k in U for i in range(2)],score_only_degenerate_edge_is_outlier=True,fixed_full_pool=True)
    a.near(score,expected,label+'.whole_pool_score');closure=c['phase']=='whole_U_standard_fit';direct=c['phase']=='direct_point_proposal'
    for key,v in (('initial_pose_residual_prior',False),('global_uniqueness_proven',False),('local_only',not direct),('optimizer_success',None if direct or closure else True)):a.eq(c[key],v,label+'.'+key)
    if direct:a.eq(len(ids),4,label+'.direct_four_ids');a.eq(c['actual_optimizer_called'],False,label+'.direct_not_optimized')
    elif not closure:a.near(c['reported_soft_l1_cost'],cost,label+'.reported_actual_cost')
    if not direct:
        a.ok(len(ids)>=4,label+'.standard_actual_fit_at_least_four');stable=set(ids)<=set(ip);rank_ok=rank['observed_normal_joint']==rank['modeled_normal_joint']==6
        reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank_ok else 'WEAK_OBSERVABILITY' if any(weak.values()) else None
        for key,v in (('whole_pool_point_ids',U),('whole_pool_line_edges',[]),('final_inliers',ip),('final_line_inliers',[]),('fit_pool_all_final8px_inliers',stable),('score_count',len(ip)),('actual_inlier_geometry_is_acceptance_gate',True),('whole_pool_scoring_factors_not_all_fit_observations',not closure)):a.eq(c[key],v,label+'.'+key)
        a.near(c['truncated_sse_px2'],score['factor_truncated_sse_px2'],label+'.truncated')
        if reason is None:
            counts['incidence_checks']+=1
            if not h['incidence'](c,row,U,H,[],h,a,policy,label+'.incidence'):reason=c['stable_incidence_guard']['guard_state']
        else:a.eq(c['stable_incidence_guard_not_called_reason'],'ORIGINAL_POINT_CANDIDATE_GATE_REJECTED' if closure else 'ORIGINAL_FINITE_CANDIDATE_GATE_REJECTED',label+'.typed_no_guard');a.ok('stable_incidence_guard' not in c,label+'.no_undeclared_guard')
        a.eq(c['accepted'],reason is None,label+'.accepted');a.eq(c['rejected'],reason,label+'.rejected')
    counts['candidate:'+c['phase']]+=1

def proposals_and_refits(s,row,U,H,h,a,counts,label):
    D=len(h['dimensions'](row['xyz']));generated=s['direct_point_generation'];proposals=s['proposal_solutions'];cs=s['all_candidate_solutions'];attempts=s['optimizer_attempts'];early=s['state'] in ('INVALID_OBSERVATION_CONTRACT','INSUFFICIENT_POINT_OBSERVATIONS','INSUFFICIENT_ACTUAL_INCIDENCE','UNSUPPORTED_CAMERA_CONTRACT')
    a.eq(s['mixed_subset_enumeration'],[],label+'.no_mixed_subsets');failures=[];matched=set();bank=Counter();roots=0
    if not early:
        a.eq([(g['dimension_index'],g['generator_ids']) for g in generated],[(d,list(ids)) for d in range(D) for ids in combinations(U,4)],label+'.all_four_ID_requests')
        for i,g in enumerate(generated):
            a.eq(g['returned_valid_roots'],len(g['roots']),label+'.root_count');a.eq(g['original_available_or_best_not_used'],True,label+'.no_best_filter');roots+=len(g['roots']);bank.update(g['bank_operation_counts'])
            gd=g['bank_operation_counts'];a.eq(gd['generic_cache_misses'],1,label+'.fresh_four_ID_key');a.eq(gd['generic_cache_hits'],0,label+'.no_duplicate_four_ID_request');a.eq(gd['generic_calls']+gd['geometry_rejected'],1,label+'.generation_or_layout_rejection');a.eq(gd['subset_generic_calls'],gd['generic_calls'],label+'.actual_subset_Generic')
            require(g['generation_state'] in ('VALID_GENERIC_ROOTS','DEGENERATE_POINT_SUBSET','NUMERICAL_GENERATION_FAILURE'),'typed generation failure');a.eq(g['generation_state']=='VALID_GENERIC_ROOTS',bool(g['roots']),label+'.root_state')
            for r in g['roots']:
                a.eq(r['actual_optimizer_called'],False,label+'.raw_root_no_optimizer')
                if r.get('rejected'):a.eq(r['rejected'],'DIRECT_POINT_GEOMETRY_UNAVAILABLE',label+'.direct_failure');failures.append(r['rejected']);continue
                matches=[p for p in proposals if (p['dimension_index'],p['point_generator_ids'],p['point_solution_index'],p['point_generator'])==(g['dimension_index'],g['generator_ids'],r['solution_index'],r['generator'])]
                require(len(matches)==1,'raw root retained exactly once');p=matches[0];matched.add(p['proposal_index']);a.eq(p['start_index'],r['start_index'],label+'.root_start');a.eq(p['fit_input_ids'],g['generator_ids'],label+'.root_fit')
        a.eq(matched,set(range(len(proposals))),label+'.no_dropped_roots');a.ok(len(generated)<=D*70 and D<=2,label+'.finite_generation_bound')
        for key,v in (('direct_four_ID_requests',len(generated)),('direct_point_roots',roots),('direct_geometry_calls',roots),('direct_geometry_failures',roots-len(proposals)),('mixed_subset_requests',0),('mixed_dimension_requests',0),('whole_pool_proposal_scores',len(proposals))):a.eq(s['operation_counts'][key],v,label+'.counts_'+key)
        bounds=dict(point_count=len(U),line_count=0,factor_blocks=len(U),registered_dimensions=D,four_ID_subsets=math.comb(len(U),4),maximum_direct_Generic_calls=D*math.comb(len(U),4),mixed_subsets=0,mixed_LOCAL_proposal_calls_at_most=0,universal_mixed_subsets=1140,universal_mixed_LOCAL_proposals=2280,direct_valid_roots=roots,LOCAL_optimizer_calls_at_most=roots,universal_LOCAL_optimizer_calls_at_most=4560+roots)
        a.eq(s['call_bounds'],bounds,label+'.inherited_actual_bound');a.eq(len(s['start_source_witnesses']),1,label+'.one_unused_start_diagnostic');a.eq(s['initial_pose_start_used'],False,label+'.no_initial_pose_start')
    else:a.eq(generated,[],label+'.early_no_generation');a.eq(proposals,[],label+'.early_no_proposals')
    for i,p in enumerate(proposals):
        for key,v in (('proposal_index',i),('logical_start_index',i),('proposal_kind','DIRECT_FINITE_FOUR_POINT_ROOT'),('phase','direct_point_proposal')):a.eq(p[key],v,label+'.proposal_'+key)
        candidate(p,row,U,H,h,a,h['guard_policy'],counts,label+'.proposal.'+str(i))
    a.eq(len(attempts),len(proposals),label+'.one_refit_record_per_proposal');seen={};indices=[];called=returned=exception=success=hits=0
    for i,attempt in enumerate(attempts):
        p=proposals[i];e=attempt['refit'];ip=p['whole_pool_score']['final_inliers'];a.eq(attempt['proposal_index'],i,label+'.attempt_index');a.eq(e['proposal_consensus_point_ids'],ip,label+'.actual_refit_pool');a.eq(e['proposal_consensus_line_ids'],[],label+'.no_refit_lines')
        a.eq(e['removed_point_ids'],[k for k in U if k not in ip],label+'.outlier_exclusion');a.eq(e['removed_line_ids'],[],label+'.no_edge_outliers')
        actual=e['actual_optimizer_called'];called+=int(actual);returned+=int(actual and 'scipy_status' in e);exception+=int(actual and 'exception' in e and 'scipy_status' not in e);success+=int(actual and e.get('success',False));hits+=int(e['cache_reused'])
        if len(ip)<4:a.eq(e['rejected'],'INSUFFICIENT_POINT_INLIER_OBSERVATIONS',label+'.minimum_four');a.eq(actual,False,label+'.no_insufficient_fit')
        else:
            key=(p['dimension_index'],struct.pack('<6d',*p['final_z']),tuple(ip));a.eq(e['cache_reused'],key in seen,label+'.exact_numeric_refit_cache')
            if key in seen:
                old=seen[key];a.eq(actual,False,label+'.cache_no_optimizer');a.eq(e['cache_source_start_index'],old['start_index'],label+'.cache_source')
                for field in ('initial_z','final_z','success','scipy_status','nfev','njev','reported_soft_l1_cost','exception'):a.eq(e.get(field),old.get(field),label+'.cache_'+field)
            else:seen[key]=e;a.eq(actual,True,label+'.fresh_optimizer')
            a.eq(e['initial_z'],p['final_z'],label+'.exact_start');a.eq(e['fit_point_ids'],ip,label+'.actual_numeric_fit_ids');a.eq(e['fit_line_edges'],[],label+'.actual_numeric_fit_no_lines')
            if 'scipy_status' in e:a.eq(e['success'],e['scipy_status']>0,label+'.SciPy_success_status');a.eq(e['success'] or e['rejected']=='OPTIMIZER_FAILED',True,label+'.typed_optimizer_failure')
            elif actual:a.eq(e['rejected'],'LOCAL_NUMERICAL_FAILURE',label+'.typed_numeric_exception');a.ok(isinstance(e.get('exception'),dict),label+'.saved_exception')
        if 'candidate_index' in e:
            c=cs[e['candidate_index']];indices.append(e['candidate_index'])
            for field,v in (('fit_input_ids',ip),('fit_line_edges',[]),('final_z',e['final_z']),('accepted',e['accepted']),('rejected',e['rejected'])):a.eq(c[field],v,label+'.returned_'+field)
        if e.get('rejected'):failures.append(e['rejected'])
    refits=[c for c in cs if c['phase']=='refit'];a.eq(indices,list(range(len(refits))),label+'.all_actual_refits_retained')
    op=s['operation_counts'];a.eq(called,returned+exception,label+'.closed_refit_dispatch');a.ok(called<=roots,label+'.refit_bound')
    for key,v in (('refit_optimizer_calls',called),('refit_optimizer_returns',returned),('refit_optimizer_exceptions',exception),('refit_cache_hits',hits),('local_optimizer_calls',called),('local_optimizer_completions',returned),('local_optimizer_exceptions',exception),('proposal_optimizer_calls',0),('proposal_optimizer_returns',0),('proposal_optimizer_exceptions',0),('proposal_cache_hits',0)) :a.eq(op[key],v,label+'.operation_'+key)
    for key,v in (('proposal_unique',0),('proposal_reuse',0),('refit_unique',len(seen)),('refit_reuse',hits)):
        if not early:a.eq(s['local_start_cache'][key],v,label+'.numeric_cache_'+key)
    return failures,bank,dict(entries=called,returns=returned,exceptions=exception,success_reports=success,residual_entries=op['optimizer_residual_calls'],jacobian_entries=op['optimizer_jacobian_calls'])

def closure(s,U,h,a,label,failures,bank):
    w=s['closure_ledger'];a.eq(w['used'],U,label+'.whole_U');a.eq(w['hidden'],s['hidden'],label+'.hidden');a.eq(w['SciPy_closure_refit_entries'],0,label+'.no_extra_SciPy')
    extra=[c for c in s['all_candidate_solutions'] if c['phase']=='whole_U_standard_fit'];D=len(s['registered_dimensions']) if 'registered_dimensions' in s else len(h['dimensions'](h['current_row']['xyz']))
    a.eq(w['maximum_extra_Generic_calls'],D,label+'.Generic_bound');a.eq(w['maximum_extra_LM_calls'],1,label+'.LM_bound');a.eq(w['extra_independent_observations'],0,label+'.no_new_observations')
    if w.get('skipped'):a.eq(w['candidates'],[],label+'.skipped_candidates');a.eq(extra,[],label+'.skipped_geometry');return
    delta={k:w['bank_after'][k]-w['bank_before'][k] for k in w['bank_after']};a.eq(w['bank_operation_counts'],delta,label+'.bank_delta');a.eq(w['bank_before'],dict(bank),label+'.raw4_before');bank.update(delta)
    a.ok(delta['generic_calls']<=D and delta['lm_calls']<=1,label+'.actual_extra_bound');a.eq(delta['subset_generic_calls'],0,label+'.not_subset');a.eq(delta['refit_generic_calls'],0,label+'.not_Generic_inlier_refit');a.eq(delta['subsets_considered'],0,label+'.no_subset_enumeration')
    a.eq(delta['generic_cache_hits']+delta['generic_cache_misses'],0 if w['standard_cache_hit'] else D,label+'.same_problem_cache_requests')
    # For U=4 every (dim,U) was already requested in the raw4 phase.
    if len(U)==4:a.eq(delta['generic_calls'],0,label+'.four_ID_whole_U_reuses_actual_raw_cache');a.eq(delta['generic_cache_misses'],0,label+'.no_four_ID_repeat')
    entries=w['candidates'];a.eq(w['closure_candidate_count'],len(entries),label+'.pool_count');a.eq(w['raw_valid_roots']+w['LM_valid_returns'],len(entries),label+'.raw_and_LM_union');a.ok(w['LM_valid_returns']<=1,label+'.one_LM_return');a.eq(w['LM_convergence_status'],'NOT_EXPOSED_BY_OPENCV',label+'.honest_LM_status')
    a.eq(delta['lm_calls'],int(w['raw_valid_roots']>0),label+'.one_actual_LM_if_raw_root');a.eq(w['original_standard_refit_count'],delta['lm_calls'],label+'.ordinary_actual_refit_entry')
    indices=[]
    for i,e in enumerate(entries):
        lm=e['proposal_kind']=='WHOLE_U_STANDARD_LM_REFIT';a.ok(e['proposal_kind'] in ('WHOLE_U_STANDARD_GENERIC_FIT','WHOLE_U_STANDARD_LM_REFIT'),label+'.typed_standard_kind')
        for key,v in (('index',i),('generator_ids',U),('actual_refit_performed',lm),('actual_OpenCV_LM_refit',lm),('mathematical_standard_Generic_fit',not lm),('actual_optimizer_called',False),('actual_optimizer_called_scope','SCIPY_LEAST_SQUARES_ONLY'),('actual_SciPy_least_squares_called',False),('actual_iterative_OpenCV_optimizer_called',lm),('optimizer_success',None),('scipy_status',None)):a.eq(e[key],v,label+'.entry_'+key)
        if e['candidate_index'] is None:a.ok(e['rejected'] in ('NONFINITE_OR_NONPOSITIVE_DEPTH','FINAL_GEOMETRY_UNAVAILABLE'),label+'.typed_unsaved_geometry');failures.append(e['rejected']);continue
        c=s['all_candidate_solutions'][e['candidate_index']];indices.append(e['candidate_index'])
        for key in ('proposal_kind','dimension_index','accepted','rejected','actual_refit_performed','actual_OpenCV_LM_refit','mathematical_standard_Generic_fit','actual_optimizer_called','actual_optimizer_called_scope','actual_SciPy_least_squares_called','actual_iterative_OpenCV_optimizer_called','optimizer_success','scipy_status'):a.eq(c[key],e[key],label+'.retained_'+key)
        for key,v in (('point_generator_ids',U),('fit_input_ids',U),('point_generator',e['generator']),('point_solution_index',e['solution_index']),('start_index',len(s['proposal_solutions'])+i),('logical_start_index',len(s['proposal_solutions'])+i),('source','LIVE_SAME_Q_WHOLE_U_STANDARD'),('soft_l1_cost_is_diagnostic_not_this_fit_objective',True),('standard_candidate_selection_objective','WHOLE_U_PIXEL_SSE'),('actual_fit_objective','WHOLE_U_L2_REPROJECTION_LM' if lm else 'OPENCV_GENERIC_STANDARD_PROBLEM'),('raw_generator_internal_objective_not_independently_certified',not lm),('LM_convergence_status','NOT_EXPOSED_BY_OPENCV')):a.eq(c[key],v,label+'.candidate_'+key)
        if e.get('rejected'):failures.append(e['rejected'])
    a.eq(indices,[c['candidate_index'] for c in extra],label+'.all_saved_standard_candidates');a.eq(w['accepted_candidates'],sum(e['accepted'] for e in entries),label+'.accepted');a.eq(w['rejected_candidates'],sum(not e['accepted'] for e in entries),label+'.rejected');a.eq(w['original_raw_four_ID_proposal_count'],len(s['proposal_solutions']),label+'.raw4_retained')

def decision(row,parent,obs,s,U,H,h,a,failures,label):
    cs=s['all_candidate_solutions'];viable=[c for c in cs if c['accepted']];ranked=sorted(viable,key=lambda c:(-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index']));best=ranked[0] if ranked else None
    ambiguity=False;alternatives=[];duplicates=ties=0
    if best:
        a.eq(s['best_candidate'],best,label+'.selected_best')
        for c in ranked[1:]:
            eq=h['equivalent'](best,c,row['xyz']);delta=c['truncated_sse_px2']-best['truncated_sse_px2'];tied=c['score_count']==best['score_count'] and abs(delta)<=1e-8;same_support=c['final_inliers']==best['final_inliers']
            alternatives.append(dict(c,numerically_equivalent_to_selected=eq,numerical_objective_tie=tied,objective_difference_from_selected_px2=delta,same_actual8px_consensus_as_selected=same_support));ambiguity|=not eq and tied;duplicates+=int(eq);ties+=int(tied)
        a.near(s['alternatives'],alternatives,label+'.true_tie_relations');a.eq(s['state'],'AMBIGUOUS_FINITE_CONSENSUS' if ambiguity else 'NEW_POSE',label+'.state')
        if ambiguity:a.eq(s['reason'],'NUMERICAL_CONSENSUS_TIE',label+'.only_true_tie_veto')
        a.eq(s['per_dimension_best'],[next(c for c in ranked if c['dimension_index']==d) for d in range(len(h['dimensions'](row['xyz']))) if any(c['dimension_index']==d for c in ranked)],label+'.dimension_best')
    else:
        a.eq(s.get('best_candidate'),None,label+'.no_best')
        if s['state'] not in ('INVALID_OBSERVATION_CONTRACT','INSUFFICIENT_POINT_OBSERVATIONS','INSUFFICIENT_ACTUAL_INCIDENCE','UNSUPPORTED_CAMERA_CONTRACT'):
            categories=dict(numeric=0,insufficient=0,weak=0,unstable=0,initialization=0,other=0)
            for reason in failures:categories['insufficient' if 'INSUFFICIENT' in reason else 'weak' if 'WEAK' in reason else 'unstable' if reason=='UNSTABLE_INLIER_REFIT' else 'initialization' if reason=='NO_LOCAL_INITIALIZATION' else 'numeric' if reason in ('DIRECT_POINT_GEOMETRY_UNAVAILABLE','LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED','STABLE_INCIDENCE_NUMERICAL_FAILURE') else 'other']+=1
            active=[k for k,v in categories.items() if v];states=dict(numeric='NUMERICAL_FAILURE',insufficient='INSUFFICIENT_ACTUAL_INLIER_GEOMETRY',weak='WEAK_OBSERVABILITY',unstable='UNSTABLE_INLIER_REFIT',initialization='NO_LOCAL_INITIALIZATION')
            state=('DEGENERATE_POINT_OBSERVATIONS' if s['direct_point_generation'] and all(g['generation_state']=='DEGENERATE_POINT_SUBSET' for g in s['direct_point_generation']) else 'NUMERICAL_GENERATION_FAILURE') if not s['proposal_solutions'] and len(U)>=4 else states.get(active[0],'MIXED_NO_VALID_CONSENSUS') if len(active)==1 else 'MIXED_NO_VALID_CONSENSUS'
            a.eq(s['state'],state,label+'.typed_no_best')
    if 'failure_reasons' in s:
        a.eq(s['failure_reasons'],failures,label+'.all_failed_branches')
        categories=Counter()
        for reason in failures:categories['insufficient' if 'INSUFFICIENT' in reason else 'weak' if 'WEAK' in reason else 'unstable' if reason=='UNSTABLE_INLIER_REFIT' else 'initialization' if reason=='NO_LOCAL_INITIALIZATION' else 'numeric' if reason in ('DIRECT_POINT_GEOMETRY_UNAVAILABLE','LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED','STABLE_INCIDENCE_NUMERICAL_FAILURE') else 'other']+=1
        a.eq(s['failure_category_counts'],{k:categories[k] for k in ('numeric','insufficient','weak','unstable','initialization','other')},label+'.failure_categories')
    new=bool(best and not ambiguity);fallback=not new and bool(parent['initial_pose'].get('available',False));native=copy.deepcopy(obs['native_N3_points']);q=copy.deepcopy(row['input_points'])
    if new:
        for key in ('R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry','soft_l1_cost','sse_px2','truncated_sse_px2','final_inliers','final_line_inliers','stable_incidence_guard'):a.near(s[key],best[key],label+'.chosen_'+key)
        for k in range(8):
            if k in H:native[k]=best['projected'][k];q[k]=best['projected'][k]
            elif h['finite'](q[k],(2,)) and q[k]!=[-1,-1]:native[k]=q[k]
        a.eq(row['actual_pose'],s,label+'.actual_NEW')
    else:
        a.eq(row['actual_pose'],parent['initial_pose'],label+'.entire_original_fallback')
        for key in ('R_cf','R_physical','centroid','cf_extents','projected'):a.eq(s[key],None,label+'.unavailable_'+key)
        a.eq(s['fit_input_ids'],[],label+'.no_unavailable_fit')
    for key,v in (('available',new),('pose_available',new),('new_pose_estimated',new),('no_pose',not new),('unresolved_ambiguity',ambiguity),('multiple_solutions',any(not h['equivalent'](best,c,row['xyz']) for c in ranked[1:]) if best else False),('numerical_duplicate_solution_count',duplicates),('numerical_tied_solution_count',ties),('hidden_reprojected',new and bool(H))):a.eq(s[key],v,label+'.solver_'+key)
    for key,v in (('new_pose_estimated',new),('fallback_used',fallback),('pose_available',new or fallback),('no_pose',not(new or fallback)),('output_status','NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),('reprojections_reused_as_observations',False),('reprojected_ids',H if new else []),('hidden_reprojected',new and bool(H))):a.eq(row[key],v,label+'.output_'+key)
    a.near(row['native_points'],native,label+'.native_output');a.near(s['points_final'],q,label+'.H_only_output');a.eq(row['native_points'][8],obs['native_N3_points'][8],label+'.center_preserved');a.eq(s['operation_counts']['hidden_reprojection_assignments'],len(H) if new else 0,label+'.H_once')
    for key,v in (('candidate_count',len(cs)),('proposal_candidate_count',len(s['proposal_solutions'])),('viable_candidate_count',len(viable))):
        if key in s:a.eq(s[key],v,label+'.'+key)
    if new:a.eq(s['local_refinement_estimated'],best['actual_refit_performed'] if best['phase']=='whole_U_standard_fit' else True,label+'.actual_refinement_kind')

def packet(row,ledger,parent,obs,completion,h,a,counts,label):
    q,H,s=parent['input_points'],parent['hidden_initial'],row['solver'];U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H];h['current_row']=row
    original=dict(parent_row=parent,native_N3_points=obs['native_N3_points'],parent_completion=completion,role_observation=obs);point_obs=copy.deepcopy(obs);point_obs['lines']=[]
    for key,v in (('method',METHOD),('input_points',q),('K',parent['K']),('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),('hidden_initial',H),('initial_pose',parent['initial_pose']),('original_native_N3_points',obs['native_N3_points']),('parent_completion',completion),('selected_corner_ids',parent['selected_corner_ids']),('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('point_only_observation_semantic_sha256',h['semantic'](point_obs))):a.eq(row[key],v,label+'.source_'+key)
    for key in ('selected_index','fixed_metadata','head_arm','head_mode','head_checkpoint_sha256','calibration_binding'):a.eq(row[key],parent[key],label+'.same_'+key)
    a.eq(ledger['input_packet_semantic_sha256'],h['semantic'](original),label+'.packet_SHA');a.eq(row['parent_method'],'IMAGE_ROLE_CORNERWISE_HYBRID',label+'.parent_method');a.eq(parent['selected_corner_ids'],parent['observation_contract']['hybrid_boundary_corner_ids'],label+'.original_adoption')
    for k in range(9):a.eq(q[k],{c['id']:c['xy'] for c in obs['corners']}[k] if k in row['selected_corner_ids'] else obs['native_N3_points'][k],label+'.actual_source_xy.'+str(k))
    for key,v in (('solver',METHOD),('used',U),('hidden',H),('excluded',H),('line_edges',[]),('consumed_edges',[]),('actual_source_lines',[])):a.eq(s[key],v,label+'.solver_'+key)
    for key in ('prior_used','initial_pose_residual_prior','initial_pose_start_used','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','center_used_for_fit_or_score','mixed_root_completeness_proven'):a.eq(s[key],False,label+'.absent_'+key)
    digest=h['ordinary']['native_hash'](q,row['K'],row['xyz'],row['raw_hw'])
    if digest is not None:a.eq(s['input_hash'],digest,label+'.coordinate_bank_hash')
    else:a.ok(not h['finite'](q,(9,2)),label+'.nonfinite_original_hash_unavailable')
    a.eq(ledger['point_bank_input_hash'],s['input_hash'],label+'.same_actual_bank');a.eq(ledger['point_solver'],s,label+'.full_actual_solver');a.eq(ledger['closure_ledger'],s['closure_ledger'],label+'.closure_ledger')
    a.eq(row['hybrid_consensus_policy'],ledger['policy'],label+'.policy');a.eq(s['policy'],ledger['policy']['solver'],label+'.solver_policy');a.eq(s['policy']['H_and_center_excluded'],True,label+'.H_center_policy');a.eq(s['operative_output_decision_policy']['numerical_score_tie_px2'],1e-8,label+'.fixed_tie');a.eq(s['operative_output_decision_policy']['same_support_different_cost_veto_removed'],True,label+'.no_extra_same_support_veto')
    proof=s['whole_U_closure_source_proof'];a.eq(proof['solve_coverage_insertions'],1,label+'.one_coverage_insertion')
    for key in ('solve_inverse_whole_AST_equal','original_true_tie_repair_whole_AST_equal','point_callbacks_delegate_frozen_S19_S13','all_original_four_ID_generation_refit_rank_score_H_body_unchanged'):a.eq(proof[key],True,label+'.source_proof_'+key)
    a.eq(proof['ordinary_fragment_AST_changes'],0,label+'.unchanged_ordinary_fragment');a.eq(proof['parent_globals_mutated'],False,label+'.no_parent_mutation')
    for name,v in proof['source_bindings'].items():a.ok(any(same(bound,v) and Path(bound['path']).name==Path(v['path']).name for key,bound in h['bindings'].items() if key.startswith('geometry_input:source:')),label+'.source_binding_'+name)
    pp=row['whole_U_closure_pipeline_source_proof'];a.eq(pp,ledger['source_proof'],label+'.same_pipeline_proof');a.eq(pp['solver_whole_AST_coverage_proof'],proof,label+'.same_nested_solver_proof');a.eq(pp['parent_globals_mutated'],False,label+'.pipeline_no_mutation');a.eq(pp['hybrid_packet_validator_is_exact_frozen_direct_delegate'],True,label+'.unchanged_packet_validator')
    for name,v in pp['source_bindings'].items():a.ok(any(same(bound,v) and Path(bound['path']).name==Path(v['path']).name for key,bound in h['bindings'].items() if key.startswith('geometry_input:')),label+'.pipeline_source_binding_'+name)
    failures,bank,opt=proposals_and_refits(s,row,U,H,h,a,counts,label);closure(s,U,h,a,label+'.closure',failures,bank)
    for i,c in enumerate(s['all_candidate_solutions']):a.eq(c['candidate_index'],i,label+'.candidate_index');candidate(c,row,U,H,h,a,h['guard_policy'],counts,label+'.candidate.'+str(i))
    decision(row,parent,obs,s,U,H,h,a,failures,label)
    op=s['operation_counts'];bank_counts=s['hypothesis_bank_operation_counts'];a.eq(bank_counts,{k:bank[k] for k in bank_counts},label+'.all_point_bank_work');a.eq(ledger['point_bank_operation_counts'],bank_counts,label+'.ledger_bank_work');a.eq(ledger['point_bank_after'],s['hypothesis_bank_counts'],label+'.bank_final');a.eq(ledger['point_solver_counts'],s['local_bank_counts'],label+'.solver_final_counts')
    a.eq({k:ledger['point_bank_after'][k]-ledger['point_bank_before'][k] for k in ledger['point_bank_after']},bank_counts,label+'.actual_total_bank_delta')
    a.eq(ledger['optimizer_dispatch'],opt,label+'.actual_SciPy');a.eq(ledger['actual_solver_dispatch'],dict(entries=1,returns=1,exceptions=0),label+'.one_live_solver');a.eq(ledger['actual_solver_calls'],1,label+'.one_path');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['used'],U,label+'.ledger_U');a.eq(ledger['initial_H'],H,label+'.ledger_H')
    a.eq(s['refit_count'],opt['entries'],label+'.actual_refit_count');a.eq(op['local_solve_calls'],1,label+'.one_bank_solve');a.eq(ledger['new_endpoint_validation_calls'],0,label+'.no_endpoint_replay')
    guard_w=[c['stable_incidence_guard'] for c in s['all_candidate_solutions'] if 'stable_incidence_guard' in c]
    for key,v in (('stable_incidence_guard_calls',len(guard_w)),('stable_incidence_geometry_calls',sum(w['counts']['geometry_info_calls'] for w in guard_w)),('stable_incidence_geometry_projection_entries',sum(w['counts']['geometry_projection_entries'] for w in guard_w))):a.eq(op[key],v,label+'.guard_'+key)
    for name in h['PRIMITIVES']:
        v={k:ledger[field][name] for k,field in (('entries','primitive_entry_calls'),('returns','primitive_return_calls'),('exceptions','primitive_exception_calls'))};a.eq(v['entries'],v['returns']+v['exceptions'],label+'.closed_CV_'+name)
        for k,n in v.items():counts['call:cv2.'+name+':'+k]+=n
    for name in ('solvePnP','cornerSubPix'):a.eq(ledger['primitive_entry_calls'][name],0,label+'.no_'+name)
    a.eq(ledger['primitive_entry_calls']['solvePnPGeneric'],bank_counts['generic_calls'],label+'.live_Generic_bank');a.eq(ledger['primitive_entry_calls']['solvePnPRefineLM'],bank_counts['lm_calls'],label+'.live_LM_bank')
    for field,name in (('svd_dispatch','numpy.linalg.svd'),('optimizer_dispatch','least_squares'),('actual_solver_dispatch','V21.solve')):
        v=ledger[field];a.eq(v['entries'],v['returns']+v['exceptions'],label+'.closed_'+name)
        for k,n in v.items():require(type(n) is int and n>=0,'nonnegative actual dispatch');counts['call:'+name+':'+k]+=n
    counts['frames']+=1;counts['status:'+METHOD+':'+row['output_status']]+=1;counts['saved_proposals']+=len(s['proposal_solutions']);counts['saved_candidates']+=len(s['all_candidate_solutions'])

def guard(a,stage):
    for p in (a.repo,a.private,a.input,a.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink folders')
    require(a.output!=a.input and a.output.is_relative_to(a.private),'new private checker output distinct from geometry')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(a.output/name).exists(),'preserve '+name)
def freeze(a):
    guard(a,'freeze');f=input_paths(a);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,_,_,info=extraction(f)
    write(a.output/FILES[0],dict(schema='whole_U_closure_point_scalar_protocol_v21',inputs=b,checker=b['checker'],helpers=info,methods=[METHOD],frames=245,rows=245,ledger_rows=245,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(a):
    guard(a,'run');f=input_paths(a);b={k:binding(p) for k,p in f.items()};own=read(a.output/FILES[0]);ob=binding(a.output/FILES[0]);tree,ordinary,policy,info=extraction(f)
    require(own['schema']=='whole_U_closure_point_scalar_protocol_v21' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'exact independent freeze')
    core,r,seal,cohort,completion=metadata(f,b);write(a.output/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['checker'],new_production_calls=0))
    h=dict(math=math,json=json,hashlib=hashlib,struct=struct,copy=copy,require=require,rows=rows);exec(compile(tree,'immutable independent point kernels','exec'),h)
    n=dict(math=math,json=json,hashlib=hashlib,struct=struct,require=require);exec(compile(ordinary,'immutable V8 projection','exec'),n);h.update(ordinary=n,guard_policy=policy,bindings=b)
    audit=h['Audit']();counts=Counter();error=None
    try:
        raw,ledgers=iter(rows(f['geometry'])),iter(rows(f['ledgers']));known=set(cohort['ids'])
        parents=iter(h['primary'](f['geometry_input:GEOMETRY_SEALED.jsonl.gz'],'method','IMAGE_ROLE_CORNERWISE_HYBRID',1960,known,audit,'V7_geometry'))
        observations=iter(h['primary'](f['geometry_input:OBSERVATIONS.jsonl.gz'],'head_arm','IMAGE_ROLE',735,known,audit,'V7_ROLE'))
        for i,fid in enumerate(cohort['ids']):
            row,ledger,parent,obs=[next(stream,None) for stream in (raw,ledgers,parents,observations)];require(all(isinstance(v,dict) and v['id']==fid and v['session']==cohort['frames'][i]['session'] for v in (row,ledger,parent,obs)),'same full ordered245 joins')
            packet(row,ledger,parent,obs,completion,h,audit,counts,'frame.'+str(i))
        for stream in (raw,ledgers,parents,observations):audit.eq(next(stream,None),None,'full source exhaustion')
        for name,value in r['actual_numerical_calls'].items():audit.eq(value,{k:counts['call:'+name+':'+k] for k in value},'actual.total.'+name)
        audit.eq(r['states'],{key.removeprefix('status:'):v for key,v in counts.items() if key.startswith('status:')},'actual.statuses')
        for snap in r['resource_snapshots']:audit.eq(snap['quiet'],True,'recorded_quiet');audit.eq(snap['temperature_under_80'],True,'recorded_temperature')
        audit.eq(counts['frames'],245,'all245');audit.eq(counts['candidate:direct_point_proposal'],counts['saved_proposals'],'all_direct_math');audit.eq(counts['candidate:refit']+counts['candidate:whole_U_standard_fit'],counts['saved_candidates'],'all_final_math')
        audit.eq({k:binding(p) for k,p in f.items()},b,'inputs_unchanged');audit.eq(binding(a.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames']==245;result=dict(schema='whole_U_closure_point_scalar_checks_v21',complete=complete,passed=complete and audit.failure_count==0,checks=audit.checks,failure_count=audit.failure_count,failures=audit.failures,exception=error,max_scalar_absolute_difference=audit.max_difference,inputs=b,own_protocol=ob,checker=b['checker'],counts=dict(counts),actual_counts=dict(frames=counts['frames'],geometry_rows=counts['frames'],ledger_rows=counts['frames'],successful_point_proposals_checked=counts['candidate:direct_point_proposal'],successful_refits_checked=counts['candidate:refit'],whole_U_standard_candidates_checked=counts['candidate:whole_U_standard_fit']),limits=LIMITS,new_production_calls=0,old_full_V16_validation_passed=False)
    write(a.output/FILES[2],result);print('WHOLE_U_POINT_SCALAR',result['passed'],audit.checks,audit.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.repo=a.repo.resolve();a.private=a.private.resolve();a.input=a.input.resolve();a.output=a.output.resolve();{'freeze':freeze,'run':run}[a.stage](a)
if __name__=='__main__':main()
