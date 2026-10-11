"""Own-frozen stdlib audit of all 150 fresh V21 point closure outputs.

ROOT alone executes freeze/run. Saved ticks, independent point projection/J,
all proposal/refit/whole-U decisions and dispatch ledgers are checked. No
production import, model, image, GT, fitting, SVD or measurement is invoked.
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

PRIVATE=Path('/tmp/pallet-whole-u-closure-private-20261011-v21')
HISTORY=Path('/tmp/pallet-finite-support-orthogonal-private-20261011-v19')
METHOD='HYBRID_POINT_CONSENSUS_WHOLE_U_CLOSURE'
STAGES=('detector','N3_correction','initial_pose','boundary_observation','hybrid_selection_and_point_consensus','final_pose_reprojection_and_metadata')
TIMES=('full',*STAGES,'return')
POSE=('R_cf','R_physical','centroid','cf_extents')
OPTIONAL=('projected','reprojection_px','sse_px2','soft_l1_cost','factor_sse_px2','truncated_sse_px2','residuals_used_px','line_residual_RMS_px')
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
GROUPS=('cv','optimizer','svd','head','base','fresh','local','direct','whole_U','bank_capture','model')
FILES=('RUNTIME_CHECK_PROTOCOL.json','RUNTIME_CHECK_STARTED.json','RUNTIME_CHECKS.json')
VALIDATOR_SHA='216dd1ef1015524244f2a20b4afbd6829b805abd0e1992d8e9ce7b44a87e4e88'
RUNTIME_SHA='d840c6963002c1f6e6776efdf34211a5e3b6183faccff8ccf3530db76f0fc31b'
CLOCK_SHA='0f024b6332ccb4c10c3369e3f4aee04fdd8b86f898fe805ba62ffd4dcffb5e11'
TICKS_SHA='729c3797b04e27d696fc5d343e1a7ea1908f0d08962215dc8e83e9baed800842'
LIMITS=dict(new_production_GT_RGB_model_training_PnP_optimizer_J_SVD_timing_calls=0,
 independent_scalar_point_projection_and_J_reconstruction=True,all_saved_point_candidates_checked=True,
 point_only=True,stored_spectrum_recount_only=True,old_full_V16_564_FAIL_remains_FAIL=True,
 historical_V19_runtime_cleanup_failure_remains_failed=True,hardware_authenticity_proven=False,
 quiet_snapshots_can_miss_transient_contention=True,physical_ownership_truth='UNKNOWN',
 local_rank_not_global_root_completeness=True,OpenCV_LM_convergence_not_exposed=True,
 manual_candidate_Rodrigues_counts_not_total_dispatch=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_bytes())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank saved row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:sorted(x) if isinstance(x,set) else str(x));f.write('\n');f.flush();os.fsync(f.fileno())
def sparse(v):return {k:n for k,n in v.items() if n}
def canonical(n):return ast.dump(n,include_attributes=False)
def select(tree,names):
    out=[copy.deepcopy(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require(len(out)==len(names) and {n.name for n in out}==set(names),'exact AST whitelist');return out

def declared_path(a,key,v):
    if v.get('origin')=='external_readonly_dependency':return None
    p=Path(v['path'])
    if v.get('origin')=='public_repository':
        require(not p.is_absolute() and '..' not in p.parts and p.parts and p.parts[0] in ('scripts','_docs'),'explicit public input '+key);return a.repo/p
    allowed=(a.repo,a.private)
    historical={'historical_failed_V19_runtime_receipt':HISTORY/'runtime/RUNTIME_RECEIPT.json','tested:original_driver':HISTORY/'run_geometry.py','geometry_input:original_driver':HISTORY/'run_geometry.py'}
    require(v.get('origin') is None and p.is_absolute() and (any(p.is_relative_to(root) for root in allowed) or p==historical.get(key)),'explicit current or exact historical input '+key)
    return p

def paths(a):
    out={k:a.input/v for k,v in dict(geometry='GEOMETRY.jsonl.gz',ledgers='LEDGERS.jsonl.gz',geometry_protocol='GEOMETRY_PROTOCOL.json',geometry_started='GEOMETRY_STARTED.json',geometry_receipt='GEOMETRY_RECEIPT.json').items()}
    out.update({k:a.validation_input/v for k,v in dict(validation='VALIDATION_CHECKS.json',validation_protocol='VALIDATION_PROTOCOL.json',validation_started='VALIDATION_STARTED.json').items()})
    out.update({k:a.runtime_input/v for k,v in dict(runtime_protocol='RUNTIME_PROTOCOL.json',receipt='RUNTIME_RECEIPT.json',resources='RUNTIME_RESOURCES.json',rows='RUNTIME_ROWS.jsonl.gz',preflight='RUNTIME_PREFLIGHT.json',journal='RUNTIME_STARTED.json').items()})
    out.update(checker=Path(__file__).resolve(),geometry_checker=a.private/'validation.py',runtime_source=a.private/'runtime.py',ticks_source=HISTORY/'runtime_check.py',clock_source=a.repo/'scripts/research/pallet_endpoint_observability_repair_20261011_v11/runtime_check.py',fixed_geometry=a.repo/'_docs/experiments/pallet_three_head_observation_20261010_v7/FIXED_GEOMETRY_SEALED.jsonl.gz')
    for label,alias in (('runtime_input','runtime_protocol'),('validation_input','validation_protocol')):
        for key,value in read(out[alias])['inputs'].items():
            target=declared_path(a,key,value)
            if target is not None:out[label+':'+key]=target
    return out

def extraction(f):
    require(binding(f['geometry_checker'])['sha256']==VALIDATOR_SHA and binding(f['clock_source'])['sha256']==CLOCK_SHA and binding(f['ticks_source'])['sha256']==TICKS_SHA,'frozen mathematical source bytes')
    original=ast.parse(f['geometry_checker'].read_bytes());names=('extraction','metadata','candidate','proposals_and_refits','closure','decision')
    functions=select(original,names);saved=copy.deepcopy(functions);changes=[]
    proposal=next(n for n in functions if n.name=='proposals_and_refits');sites=[]
    expected={'.fresh_four_ID_key':(1,0),'.no_duplicate_four_ID_request':(0,1),'.generation_or_layout_rejection':(1,0)}
    for n in ast.walk(proposal):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='eq' and len(n.args)==3 and isinstance(n.args[2],ast.BinOp) and isinstance(n.args[2].right,ast.Constant) and n.args[2].right.value in expected:
            suffix=n.args[2].right.value;old,new=expected[suffix];require(isinstance(n.args[1],ast.Constant) and n.args[1].value==old,'exact fresh-cache predicate');sites.append((n,n.args[1]));n.args[1]=ast.Constant(new);changes.append(suffix)
    require(set(changes)==set(expected) and len(changes)==3,'exact three cache-context predicates')
    close=next(n for n in functions if n.name=='closure');target=next(n for n in ast.walk(close) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='eq' and len(n.args)==3 and isinstance(n.args[2],ast.BinOp) and isinstance(n.args[2].right,ast.Constant) and n.args[2].right.value=='.raw4_before')
    old_call=copy.deepcopy(target);replacement=ast.parse("a.eq({k:w['bank_before'][k]-h['point_bank_before'][k] for k in w['bank_before']},{k:bank[k] for k in w['bank_before']},label+'.raw4_before')").body[0].value
    target.args=copy.deepcopy(replacement.args);patched=copy.deepcopy(functions)
    target.args=old_call.args
    for n,old in sites:n.args[1]=old
    require(canonical(ast.Module(body=functions,type_ignores=[]))==canonical(ast.Module(body=saved,type_ignores=[])),'inverse whole point-helper AST equality')
    constants=[copy.deepcopy(n) for n in original.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SOURCES','DRIVER_SHA') for t in n.targets)];require(len(constants)==2,'exact validation source constants')
    selected=ast.Module(body=constants+patched,type_ignores=[])
    source_env=dict(ast=ast,copy=copy,hashlib=hashlib,Path=Path,METHOD=METHOD,read=read,binding=binding,same=same,require=require,canonical=canonical,select=select)
    exec(compile(ast.fix_missing_locations(selected),'exact independent point source adapter','exec'),source_env)
    vf={k.removeprefix('validation_input:'):p for k,p in f.items() if k.startswith('validation_input:')}
    tree,ordinary,policy,source_info=source_env['extraction'](vf)
    clock=ast.Module(body=select(ast.parse(f['clock_source'].read_bytes()),('require','Audit','describe')),type_ignores=[])
    old_run=next(n for n in ast.parse(f['ticks_source'].read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loop=next(n for n in ast.walk(old_run) if isinstance(n,ast.For) and isinstance(n.target,ast.Tuple) and [x.id for x in n.target.elts if isinstance(x,ast.Name)]==['i','v'])
    begin=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ticks' for t in n.targets))
    end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='o' for t in n.targets))
    body=copy.deepcopy(loop.body[begin:end]);require(len(body)==8 and isinstance(body[-1],ast.For),'unchanged eight clock statement regions')
    fn=ast.parse('def clock_row(v,job,label,a,timings):\n    pass').body[0];fn.body=body;clock.body.append(fn)
    info=dict(point_source=binding(f['geometry_checker']),point_helpers=source_info,point_candidate_and_decision_AST_changes=0,raw_four_ID_cache_context_AST_predicates=3,closure_initial_bank_delta_AST_changes=1,inverse_whole_point_helper_AST_equal=True,clock_source=binding(f['clock_source']),ticks_source=binding(f['ticks_source']),clock_formula_AST_changes=0,clock_statements_AST_sha256=hashlib.sha256(canonical(ast.Module(body=body,type_ignores=[])).encode()).hexdigest(),point_adapter_AST_sha256=hashlib.sha256(canonical(selected).encode()).hexdigest())
    source_env['_point_function_AST']=selected
    return source_env,tree,ordinary,policy,clock,info

def metadata(f,b,env):
    p,r,j,res,pre=[read(f[k]) for k in ('runtime_protocol','receipt','journal','resources','preflight')]
    require(p['schema']=='fresh_whole_U_point_closure_runtime_protocol_v21' and p['method']==METHOD and p['stages']==list(STAGES) and p['warmup']==20 and p['measured']==130 and p['configured_calls']==150,'fixed fresh V21 protocol')
    for key,value in (('geometry_used_only_after_return',True),('one_point_only_closure_path_timed',True),('historical_V19_runtime_cleanup_failed_retained',True),('automatic_retry',False),('new_RGB',0),('new_training',0),('first_predict_detector_forwards',2),('other_predict_detector_forwards',1),('N3_forwards_per_prediction',1),('no_artificial_pre_forward',True)):require(p[key]==value,'fixed fresh policy '+key)
    require(r['schema']=='fresh_whole_U_point_closure_runtime_receipt_v21' and r['method']==METHOD and r['complete'] is True and r['status']=='COMPLETE' and r['statistics_official'] is True and r['configured_calls']==r['serialized_rows']==150 and r.get('error') is None and r['cleanup_errors']==[],'actual complete clean V21 runtime')
    require(j['inputs']==p['inputs'] and j['status']=='COMPLETE' and j['execution']==r['execution'] and same(j['receipt'],b['receipt']),'final durable complete journal')
    for record in (r,j,res,pre):require(same(record['protocol'],b['runtime_protocol']),'actual runtime protocol bytes')
    require(same(r['raw_rows'],b['rows']) and same(r['resources'],b['resources']) and pre['passed'] is True and pre['actual_model_fit_decode_calls']==0 and pre['resource']['quiet'] and pre['resource']['temperature_under_80'],'closed rows and quiet preflight')
    require(res['snapshots']==r['resource_snapshots'] and res['observed_snapshots_only'] is True and r['all_recorded_resource_snapshots_quiet'] is True,'same observed resource snapshots')
    for label,alias in (('runtime_input','runtime_protocol'),('validation_input','validation_protocol')):
        for key,value in read(f[alias])['inputs'].items():
            if label+':'+key in b:require(same(value,b[label+':'+key]),'current explicit own-input '+label+':'+key)
            else:require(value.get('origin')=='external_readonly_dependency','only external declarations remain metadata')
    require(b['runtime_source']['sha256']==RUNTIME_SHA and same(p['inputs']['code'],b['runtime_source']) and same(p['inputs']['validator'],b['geometry_checker']),'actual frozen runtime and scalar source')
    require(isinstance(r['model_bindings'],list) and all(any(same(x,v) for v in p['inputs'].values()) for x in r['model_bindings']),'models in declared frozen closure')
    vf={k.removeprefix('validation_input:'):v for k,v in f.items() if k.startswith('validation_input:')};vb={k.removeprefix('validation_input:'):v for k,v in b.items() if k.startswith('validation_input:')};core,gr,seal,cohort,completion=env['metadata'](vf,vb)
    own,start,check=[read(f[k]) for k in ('validation_protocol','validation_started','validation')]
    require(own['schema']=='whole_U_closure_point_scalar_protocol_v21' and check['schema']=='whole_U_closure_point_scalar_checks_v21' and check['complete'] is check['passed'] is True and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==start['new_production_calls']==0,'actual independent point scalar PASS')
    require(check['inputs']==own['inputs']==start['inputs'] and same(check['own_protocol'],b['validation_protocol']) and same(start['own_protocol'],b['validation_protocol']) and same(check['checker'],b['geometry_checker']) and same(start['checker'],b['geometry_checker']),'actual full own scalar envelope')
    require(check['old_full_V16_validation_passed'] is False and check['limits']['old_full_V16_564_FAIL_remains_FAIL'] is True and all(check['actual_counts'][k]==245 for k in ('frames','geometry_rows','ledger_rows')),'old full FAIL preserved and all245 new point audit')
    for key in ('geometry','ledgers','geometry_protocol','geometry_started','geometry_receipt'):require(same(own['inputs'][key],b[key]),'same independently checked geometry '+key)
    old=read(f['runtime_input:historical_failed_V19_runtime_receipt']);require(old['complete'] is False and old['status']=='FAILED_RUNTIME_CLEANUP' and old['statistics_official'] is False and old['serialized_rows']==old['execution']['calls_complete']==150,'historical failed V19 runtime remains failed')
    cleanup=r['cleanup']
    for key in ('instruments_entered','instruments_restored','opencv_threads_restored','pipeline_closed','pipeline_closed_before_outer_legacy_restore','head_hooks_removed','private_pipeline_registration_restored','original_opencv_threads_restored','original_torch_options_restored','environment_restored'):require(cleanup[key] is True,'actual LIFO cleanup '+key)
    require(cleanup['instrument_restoration_errors']==[] and set(cleanup['legacy_namespace_restoration'])=={'DOC','SCRATCH','read','iter_rows','binding','learned_FITS','research_path','environment'} and all(v is True for v in cleanup['legacy_namespace_restoration'].values()),'actual full legacy namespace restoration')
    require(r['protected_before']==r['protected_after']==p['protection'],'protected originals unchanged')
    return p,r,res,cohort

def corner_selection(o,a,label):
    H=o['hidden_initial'];v=o['cornerwise_selection'];records=v['per_corner_decisions'];chosen=[r['id'] for r in records if r['accepted']]
    a.eq(v['actual_self_hidden_ids'],H,label+'.selection_H');a.eq(v['selected_boundary_corner_ids'],chosen,label+'.adopted_TLS');a.eq(o['selected_corner_ids'],chosen,label+'.adopted_ids');a.eq(v['LOO_solve_calls'],sum(r['heldout_pose_calls'] for r in records),label+'.LOO_calls')
    for r in records:
        k=r['id'];a.eq(r['heldout_fit_requested_excluded_ids'],sorted(set(H)|{k}),label+'.H_union_k');a.eq(r['actual_self_hidden_ids'],H,label+'.same_H');a.eq(bool(set(r['fit_input_ids'])&(set(H)|{k})),False,label+'.no_self_test');a.eq(r['heldout_is_self_occlusion'],False,label+'.temporary_holdout');a.eq(r['projected_coordinate_used_as_observation'],False,label+'.no_reprojection_input');a.eq(r['numeric_validation_pose_prior_used'],False,label+'.no_LOO_prior')
        if r['accepted']:
            a.eq(k not in H and r['heldout_pose_calls']==1 and r['candidate_LOO_residual_px']<=8.,True,label+'.adoption_support');a.eq(not r['native_N3_available'] or r['native_minus_candidate_squared_residual_px2']>1e-8,True,label+'.strict_native_gain');a.near(o['input_points'][k],r['candidate_xy'],label+'.actual_TLS_q',relative=0)
    for k in set(range(8))-set(chosen):a.near(o['input_points'][k],o['native_N3_points'][k],label+'.retained_native_q',relative=0)

def point_row(v,h,a,counts,label):
    o=v['full_fresh_output'];s=o['solver'];ledger=v['fresh_repair_ledger'];H=o['hidden_initial'];q=o['input_points'];U=[k for k in h['eligible'](q,o['raw_hw']) if k not in H];h['current_row']=o;h['point_bank_before']=ledger['point_bank_before']
    for key,value in (('method',METHOD),('used',U),('hidden',H),('excluded',H),('line_edges',[]),('consumed_edges',[]),('actual_source_lines',[])):a.eq(o[key] if key=='method' else s[key],value,label+'.point_'+key)
    for key in ('prior_used','initial_pose_residual_prior','initial_pose_start_used','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','center_used_for_fit_or_score','mixed_root_completeness_proven'):a.eq(s[key],False,label+'.absent_'+key)
    a.eq(ledger['schema'],'fresh_hybrid_whole_U_closure_ledger_v21',label+'.ledger_schema');a.eq(ledger['point_solver'],s,label+'.full_solver');a.eq(ledger['closure_ledger'],s['closure_ledger'],label+'.closure');a.eq(o['fresh_repair_ledger'],ledger,label+'.same_output_ledger');a.eq(ledger['used'],U,label+'.same_U');a.eq(ledger['initial_H'],H,label+'.same_H')
    for key,value in (('actual_solver_calls',1),('reported_output_views',1),('new_endpoint_validation_calls',0),('new_line_solve_calls',0),('point_available_verdict_not_a_root_filter',True),('saved_accuracy_row_used',False),('extra_independent_observations',0)):a.eq(ledger[key],value,label+'.ledger_'+key)
    digest=h['ordinary']['native_hash'](q,o['K'],o['xyz'],o['raw_hw'])
    if digest is not None:a.eq(s['input_hash'],digest,label+'.selected_bank_hash')
    a.eq(s['operative_output_decision_policy']['numerical_score_tie_px2'],1e-8,label+'.fixed_true_tie');a.eq(s['operative_output_decision_policy']['same_support_different_cost_veto_removed'],True,label+'.no_extra_veto');a.eq(s['policy'],ledger['policy']['solver'],label+'.fixed_solver_policy');a.eq(o['hybrid_consensus_policy'],ledger['policy'],label+'.same_pipeline_policy')
    proof=s['whole_U_closure_source_proof'];a.eq(proof['solve_coverage_insertions'],1,label+'.one_coverage_insertion');a.eq(proof['ordinary_fragment_AST_changes'],0,label+'.ordinary_unchanged');a.eq(proof['parent_globals_mutated'],False,label+'.no_parent_mutation')
    for key in ('solve_inverse_whole_AST_equal','original_true_tie_repair_whole_AST_equal','point_callbacks_delegate_frozen_S19_S13','all_original_four_ID_generation_refit_rank_score_H_body_unchanged'):a.eq(proof[key],True,label+'.proof_'+key)
    pp=o['whole_U_closure_pipeline_source_proof'];a.eq(pp,ledger['source_proof'],label+'.pipeline_proof');a.eq(pp['solver_whole_AST_coverage_proof'],proof,label+'.nested_proof')
    for source in (proof,pp):
        for name,value in source['source_bindings'].items():a.eq(any(same(value,b) and Path(value['path']).name==Path(b['path']).name for b in h['bindings'].values()),True,label+'.current_source_'+name)
    failures,bank,opt=h['proposals_and_refits'](s,o,U,H,h,a,counts,label);h['closure'](s,U,h,a,label+'.closure',failures,bank)
    for i,c in enumerate(s['all_candidate_solutions']):a.eq(c['candidate_index'],i,label+'.candidate_index');h['candidate'](c,o,U,H,h,a,h['guard_policy'],counts,label+'.candidate.'+str(i))
    h['decision'](o,dict(initial_pose=o['initial_pose']),dict(native_N3_points=o['native_N3_points']),s,U,H,h,a,failures,label)
    delta={k:ledger['point_bank_after'][k]-ledger['point_bank_before'][k] for k in ledger['point_bank_after']};a.eq(delta,ledger['point_bank_operation_counts'],label+'.bank_delta');a.eq(delta,s['hypothesis_bank_operation_counts'],label+'.solver_delta');a.eq(delta,{k:bank[k] for k in delta},label+'.recounted_all_bank_work');a.eq(ledger['point_bank_after'],s['hypothesis_bank_counts'],label+'.bank_final');a.eq(ledger['point_solver_counts'],s['local_bank_counts'],label+'.solver_final_counts')
    a.eq(ledger['local_operation_counts'],s['operation_counts'],label+'.all_operation_counts');a.eq(s['refit_count'],opt['entries'],label+'.actual_refits');a.eq(s['operation_counts']['local_solve_calls'],1,label+'.one_solve')
    guards=[c['stable_incidence_guard'] for c in s['all_candidate_solutions'] if 'stable_incidence_guard' in c]
    for key,value in (('stable_incidence_guard_calls',len(guards)),('stable_incidence_geometry_calls',sum(w['counts']['geometry_info_calls'] for w in guards)),('stable_incidence_geometry_projection_entries',sum(w['counts']['geometry_projection_entries'] for w in guards))):a.eq(s['operation_counts'][key],value,label+'.guard_'+key)
    return opt

def counters(v,h,a,opt,label,totals,packet):
    o=v['full_fresh_output'];s=o['solver'];ledger=v['fresh_repair_ledger'];d=v['counter_deltas'];reuse=v['fresh_point_bank_reuse'];U=ledger['used'];D=len(h['dimensions'](o['xyz']))
    a.eq(set(d),set(GROUPS),label+'.eleven_groups');a.eq(sparse(d['local']),sparse(ledger['local_operation_counts']),label+'.local_counts');a.eq(reuse,o['fresh_actual_point_bank_reuse'],label+'.same_reuse_output');a.eq(reuse,ledger['fresh_actual_point_bank_reuse'],label+'.same_reuse_ledger');a.eq(reuse['schema'],'fresh_selected_point_closure_bank_reuse_v21',label+'.reuse_schema')
    for key,value in (('constructor_restored',True),('same_coordinate_bank_reused',True),('additional_raw_four_ID_Generic_calls',0),('new_endpoint_validation_calls',0),('new_line_solve_calls',0),('point_available_verdict_not_a_root_filter',True),('cache_reconstructed_from_serialized_Rt',False)):a.eq(reuse[key],value,label+'.reuse_'+key)
    captures=reuse['actual_constructor_count'];samebank=reuse['selected_native_bank_is_same_object'];a.eq(type(samebank) is bool,True,label+'.samebank_typed');a.eq(captures,1 if samebank else 2,label+'.actual_bank_count');a.eq(d['bank_capture'],dict(observer_scopes_started=1,constructor_entries=captures,constructor_returns=captures,observer_scopes_restored=1),label+'.constructor_lifecycle')
    a.eq(reuse['selected_input_hash'],s['input_hash'],label+'.selected_hash');a.eq(reuse['native_input_hash'],h['ordinary']['native_hash'](o['native_N3_points'],o['K'],o['xyz'],o['raw_hw']),label+'.native_hash')
    raw,whole=reuse['raw_four_ID_bank_delta'],reuse['whole_U_closure_bank_delta'];a.eq(raw,ledger['raw_four_ID_bank_operation_counts'],label+'.raw4_delta');a.eq(whole,ledger['whole_U_closure_bank_operation_counts'],label+'.whole_U_delta');a.eq(sparse(d['direct']),sparse(raw),label+'.actual_raw4');a.eq(sparse(d['whole_U']),sparse(whole),label+'.actual_whole_U')
    for key in ('generic_calls','generic_cache_misses','lm_calls','geometry_rejected','subset_generic_calls'):a.eq(raw[key],0,label+'.raw4_reuse_'+key)
    a.eq(whole['generic_calls']<=D and whole['lm_calls']<=1,True,label+'.extra_bound');a.eq(reuse['additional_whole_U_Generic_calls'],whole['generic_calls'],label+'.extra_Generic');a.eq(reuse['additional_whole_U_LM_calls'],whole['lm_calls'],label+'.extra_LM')
    for prefix in ('selected','native'):
        before,after=reuse[prefix+'_bank_before'],reuse[prefix+'_bank_after'];delta={k:after[k]-before[k] for k in after};a.eq(delta,reuse[prefix+'_bank_delta'],label+'.'+prefix+'_delta');a.eq(delta,ledger['point_bank_operation_counts'] if prefix=='selected' or samebank else {k:0 for k in after},label+'.'+prefix+'_partition')
    a.eq(reuse['selected_bank_before'],ledger['point_bank_before'],label+'.before');a.eq(reuse['selected_bank_after'],ledger['point_bank_after'],label+'.after');a.eq(reuse['selected_bank_delta'],{k:raw[k]+whole[k] for k in raw},label+'.exact_phase_partition')
    cache=lambda items:{(r['dimension_index'],tuple(r['point_ids'])) for r in items}
    keys=('selected_numeric_cache_before','selected_numeric_cache_after','newly_requested_selected_numeric_problems');before,after,new=(cache(reuse[k]) for k in keys)
    for key,value in zip(keys,(before,after,new)):a.eq(len(value),len(reuse[key]),label+'.unique_'+key)
    a.eq(before<=after,True,label+'.cache_preserved');a.eq(new,after-before,label+'.only_new_problems');a.eq(whole['generic_cache_misses'],len(new),label+'.cache_misses');a.eq(all(ids==tuple(U) and type(dim) is int and 0<=dim<D for dim,ids in new),True,label+'.whole_U_keys_only')
    point=v['fresh_point_bank_ledger'];a.eq(point,o['fresh_point_coordinate_bank_ledger'],label+'.same_original_bank_ledger');a.eq(point['coordinate_bank_count'],captures,label+'.capture_count');originals={r['input_hash']:r for r in point['coordinate_banks']};a.eq(len(originals),captures,label+'.unique_original_banks');a.eq(set(originals),{reuse['selected_input_hash'],reuse['native_input_hash']},label+'.actual_bank_identities')
    for prefix in ('selected','native'):
        for key,value in reuse[prefix+'_bank_before'].items():a.eq(originals[reuse[prefix+'_input_hash']][key],value,label+'.original_'+prefix+'_before.'+key)
    # Original N3/Base feature-pose frontends also dispatch OpenCV and have
    # no coordinate-bank ledger. The full live totals are joined below;
    # bank-accounted calls provide a lower bound, not an invented exact sum.
    original_generic=sum(r['generic_calls'] for r in originals.values());original_lm=sum(r['lm_calls'] for r in originals.values());a.eq(d['cv'].get('solvePnPGeneric:entries',0)>=original_generic+whole['generic_calls'],True,label+'.bank_accounted_Generic');a.eq(d['cv'].get('solvePnPRefineLM:entries',0)>=original_lm+whole['lm_calls'],True,label+'.bank_accounted_LM')
    require(set(d['cv'])<={p+':'+k for p in CV for k in ('entries','returns','exceptions')},'known CV counters');require(set(d['optimizer'])<={'entries','returns','exceptions','success_reports',*(p+'_'+k for p in ('residual','jacobian') for k in ('entries','returns','exceptions'))},'known optimizer counters');require(set(d['svd'])<={'entries','returns','exceptions'},'known SVD counters')
    for key,value in opt.items():a.eq(d['optimizer'].get(key,0),value,label+'.optimizer_'+key)
    for p in CV:a.eq(d['cv'].get(p+':entries',0),d['cv'].get(p+':returns',0)+d['cv'].get(p+':exceptions',0),label+'.closed_CV_'+p)
    for p in ('solvePnP','cornerSubPix'):a.eq(d['cv'].get(p+':entries',0),0,label+'.unused_'+p)
    for prefix in ('','residual_','jacobian_'):a.eq(d['optimizer'].get(prefix+'entries',0),d['optimizer'].get(prefix+'returns',0)+d['optimizer'].get(prefix+'exceptions',0),label+'.closed_callback_'+prefix)
    a.eq(d['svd'].get('entries',0),d['svd'].get('returns',0)+d['svd'].get('exceptions',0),label+'.closed_SVD')
    compat=o['packet_compatibility_review'];a.eq(compat['numerical_result_changed'],False,label+'.packet_numbers');a.eq(compat['original_frame_metadata_preserved'],True,label+'.packet_metadata');a.eq(compat['original_packet_function_used'],True,label+'.original_packet');packet.update(calls=1,identical_fields_removed=len(compat['removed_identical_fields']))
    for key,value in d.items():require(all(type(n) is int and n>=0 for n in value.values()),'nonnegative typed counters');totals[key].update(value)

def guard(a,stage):
    for p in (a.repo,a.private,a.input,a.validation_input,a.runtime_input,a.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink folders')
    require(a.output.is_relative_to(a.private) and all(not a.output.is_relative_to(p) and not p.is_relative_to(a.output) for p in (a.repo,a.input,a.validation_input,a.runtime_input)),'new disjoint checker output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(a.output/name).exists(),'preserve '+name)
def freeze(a):
    guard(a,'freeze');f=paths(a);b={k:binding(p) for k,p in f.items()};env,_,_,_,_,info=extraction(f);proto,_,_,_=metadata(f,b,env)
    write(a.output/FILES[0],dict(schema='independent_whole_U_point_closure_runtime_protocol_v21',inputs=b,checker=b['checker'],helpers=info,limits=LIMITS,external_declared_metadata={k:v for k,v in proto['inputs'].items() if v.get('origin')=='external_readonly_dependency'},external_metadata_current_assets_not_verified=True,frozen_before_own_arithmetic=True))
def run(a):
    guard(a,'run');f=paths(a);b={k:binding(p) for k,p in f.items()};own=read(a.output/FILES[0]);ob=binding(a.output/FILES[0]);env,tree,ordinary,policy,clock,info=extraction(f);proto,r,res,cohort=metadata(f,b,env)
    require(own['schema']=='independent_whole_U_point_closure_runtime_protocol_v21' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'exact own freeze')
    write(a.output/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['checker'],new_production_calls=0))
    h=dict(env,math=math,json=json,hashlib=hashlib,struct=struct,copy=copy,Counter=Counter,combinations=combinations,rows=rows);exec(compile(ast.fix_missing_locations(env['_point_function_AST']),'point helpers with complete scalar globals','exec'),h);exec(compile(tree,'unchanged independent point scalar kernels','exec'),h)
    n=dict(math=math,json=json,hashlib=hashlib,struct=struct,require=require);exec(compile(ordinary,'unchanged V8 projection','exec'),n);h.update(ordinary=n,guard_policy=policy,bindings=b)
    c=dict(math=math,require=require,STAGES=STAGES,TIMES=TIMES);exec(compile(ast.fix_missing_locations(clock),'unchanged independent clock kernels','exec'),c)
    audit=c['Audit']();point_audit=h['Audit']();counts=Counter();totals={k:Counter() for k in GROUPS};packet=Counter();timings={k:[] for k in TIMES};count=0;error=None
    try:
        panel=proto['panel'];audit.eq(len(panel),26,'panel26');audit.eq(len({v['frame_id'] for v in panel}),26,'unique_panel');audit.eq(Counter(v['session_id'] for v in panel),{s:2 for s in {v['session_id'] for v in panel}},'two_per_session');audit.eq(len({v['session_id'] for v in panel}),13,'sessions13')
        identities=proto['normalization']['joined_frame_identities'];authority={v['id']:v for v in identities};audit.eq(set(authority),set(cohort['ids']),'normalized245');audit.eq(len(identities),245,'unique_normalization');audit.eq(r['normalization'],proto['normalization'],'same_normalization')
        schedule=[dict(phase='warmup',index=i,image_index=i%26) for i in range(20)]+[dict(phase='measured',repeat=j,index=i,image_index=i) for j in range(5) for i in range(26)];audit.eq(proto['schedule'],schedule,'fixed150schedule');ids={v['frame_id'] for v in panel};refs={};seen=set();fixed={};fixed_seen={m:set() for m in ('BASE','N3_SUBPIX')}
        for row in rows(f['geometry']):
            require(row['method']==METHOD and row['id'] in authority and row['id'] not in seen,'full unique primary geometry');seen.add(row['id'])
            if row['id'] in ids:
                refs[row['id']]={k:row[k] for k in ('native_points','hidden_initial','output_status','new_pose_estimated','fallback_used','fixed_metadata')};pose=row['actual_pose'];refs[row['id']]['pose']={k:pose[k] for k in ('available','selected_hypothesis',*POSE,*OPTIONAL) if k in pose}
        audit.eq(seen,set(authority),'all245accuracy');audit.eq(set(refs),ids,'all26refs')
        for row in rows(f['fixed_geometry']):
            require(row['method'] in fixed_seen and row['id'] in authority and row['id'] not in fixed_seen[row['method']],'full fixed490 identity');fixed_seen[row['method']].add(row['id'])
            if row['id'] in ids:fixed[row['method'],row['id']]=row['native_points']
        audit.eq(fixed_seen,{m:set(authority) for m in fixed_seen},'all490fixed');audit.eq(set(fixed),{(m,i) for m in fixed_seen for i in ids},'allfixed26')
        for i,v in enumerate(rows(f['rows'])):
            require(i<150,'extra runtime row');job=schedule[i];label='row.'+str(i)
            for key,value in job.items():audit.eq(v[key],value,label+'.schedule_'+key)
            p=panel[job['image_index']];audit.eq((v['id'],v['session'],v['method']),(p['frame_id'],p['session_id'],METHOD),label+'.identity');audit.eq(authority[v['id']]['session'],v['session'],label+'.normalized_session');c['clock_row'](v,job,label,audit,timings)
            o=v['full_fresh_output'];s=o['solver'];native=o['native_N3_points'];reference=refs[v['id']]
            for key,value in reference.items():
                if key!='pose':audit.near(o[key],value,label+'.saved_'+key,relative=0)
            pose=o['actual_pose'];expected=reference['pose'];audit.eq(pose['available'],expected['available'],label+'.pose_available');audit.eq(pose.get('selected_hypothesis'),expected.get('selected_hypothesis'),label+'.dimension')
            if pose['available']:
                for key in OPTIONAL:audit.eq(key in pose,key in expected,label+'.optional_'+key)
                for key in (*POSE,*(k for k in OPTIONAL if k in expected)):audit.near(pose[key],expected[key],label+'.pose_'+key,relative=0)
            audit.near(o['original_base_points'],fixed['BASE',v['id']],label+'.BASE',relative=0);audit.near(native,fixed['N3_SUBPIX',v['id']],label+'.N3',relative=0);audit.eq(o['K'],authority[v['id']]['K'],label+'.K');audit.eq(o['xyz'],authority[v['id']]['xyz'],label+'.xyz')
            for key,value in (('parity_status','PASS'),('saved_geometry_used_only_after_return',True)):audit.eq(v[key],value,label+'.'+key)
            audit.eq(v['execution_events'],[dict(phase='whole_U_point_closure_call_started'),dict(phase='whole_U_point_closure_call_returned')],label+'.one_actual_closure_dispatch');audit.eq(o['predicted_initial_N3_hidden'],o['hidden_initial'],label+'.original_H');audit.eq(o['initial_H_applied'],True,label+'.H_applied')
            for key,value in (('fresh_complete_capture',True),('fresh_hybrid_corner_selection_recomputed_inside_call',True),('fresh_point_solver_recomputed_inside_call',True),('fresh_endpoint_LOO_recomputed_inside_call',False),('fresh_raw_selected_point_roots_reused_inside_call',True),('old_point_control_computed_by_original_selection',True),('paired_point_accuracy_arm_executed_inside_primary_call',False),('actual_fit_incidence_guard_applied_per_candidate',True),('parent_completion_receipt_used_as_deployment_input',False),('saved_accuracy_coordinates_or_pose_as_input',False),('cached_accuracy_row_used',False),('solver_replay_latency_claim',False),('runtime_only_fresh_metadata',True),('oracle',False),('fresh_capture_input_only',True)):audit.eq(o[key],value,label+'.fresh_'+key)
            audit.eq(o['native_points'][8],native[8],label+'.center');prediction=copy.deepcopy(v['original_fresh_detector_prediction']);index=prediction['selected_index']
            if index is not None:audit.eq(prediction['candidates'][index]['keypoints_xy'][8],native[8],label+'.raw_center');prediction['candidates'][index]['keypoints_xy']=o['native_points']
            audit.eq(o['prediction']['selected_index'],index,label+'.selected');audit.eq(o['prediction']['candidates'],prediction['candidates'],label+'.all_candidates_preserved');corner_selection(o,audit,label)
            opt=point_row(v,h,point_audit,counts,label);audit.eq(v['counter_deltas']['head'],dict(entries=1,returns=1),label+'.head');audit.eq(v['counter_deltas']['model'],dict(detector=2 if i==0 else 1,N3=1),label+'.forwards');counters(v,h,audit,opt,label,totals,packet);count+=1
        audit.eq(count,150,'all150');audit.eq(len(timings['full']),130,'measured130');audit.eq(set(r['summaries']),set(TIMES),'eight_summary_blocks')
        summaries={k:c['describe'](timings[k]) for k in TIMES}
        for key in TIMES:audit.near(summaries[key],r['summaries'][key],'summary.'+key)
        for key,field in (('cv','live_cv_counts'),('optimizer','live_optimizer_counts'),('svd','live_SVD_counts'),('head','live_head_counts'),('base','actual_base_counts'),('fresh','actual_fresh_counts'),('local','local_operation_counts'),('direct','direct_point_operation_counts'),('whole_U','whole_U_operation_counts'),('bank_capture','point_bank_capture_counts')):audit.eq(sparse(totals[key]),sparse(r[field]),'total.'+key)
        audit.eq(sparse(packet),sparse(r['packet_compatibility_counts']),'packet_total');audit.eq(r['model_forwards_after_loading'],dict(detector=0,N3=0),'loading0');audit.eq(totals['model'],dict(detector=151,N3=150),'actual151_150');audit.eq(r['model_forwards'],dict(detector=151,N3=150),'receipt151_150')
        for key in ('detector_calls','N3_route_calls','initial_pose_calls','feature_initial_pose_calls','IMAGE_ROLE_head_calls','head_calls'):audit.eq(totals['base'][key],150,'base.'+key)
        for key in ('fresh_predict_calls','fresh_hybrid_point_paths','fresh_repair_paths','fresh_point_closure_paths'):
            for phase in ('started','complete'):audit.eq(totals['fresh'][key+'_'+phase],150,'fresh.'+key+'_'+phase)
        audit.eq(totals['fresh'][METHOD+':started'],150,'method_started');audit.eq(totals['fresh'][METHOD+':complete'],150,'method_complete');audit.eq(totals['fresh']['reported_primary_views_from_single_point_closure'],150,'single_output')
        audit.eq(totals['direct']['generic_calls'],0,'raw4_Generic0');audit.eq(totals['direct']['generic_cache_misses'],0,'raw4_miss0');audit.eq(totals['direct']['lm_calls'],0,'raw4_LM0');audit.eq(totals['bank_capture']['observer_scopes_started'],150,'capture150');audit.eq(totals['bank_capture']['observer_scopes_restored'],150,'restore150')
        for key,value in (('calls_started',150),('calls_complete',150),('image_decode_calls',26),('model_initializations',1)):audit.eq(r['execution'][key],value,'execution.'+key)
        require(bool(res['snapshots']),'recorded resource snapshots')
        for snap in res['snapshots']:audit.eq(snap['quiet'] and snap['temperature_under_80'],True,'observed_quiet')
        audit.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');audit.eq(binding(a.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and count==150;failures=audit.failure_count+point_audit.failure_count;result=dict(schema='independent_whole_U_point_closure_runtime_checks_v21',complete=complete,passed=complete and failures==0,inputs=b,own_protocol=ob,checker=b['checker'],helpers=info,limits=LIMITS,new_production_calls=0,old_full_V16_validation_passed=False,historical_V19_runtime_cleanup_failure_preserved=True,checks=audit.checks+point_audit.checks,failure_count=failures,failures=audit.failures+point_audit.failures,exception=error,max_absolute_difference=max(audit.max_difference,point_audit.max_difference),rows=count,measured=len(timings['full']),summary_blocks=8,summary_slots=56,summaries=summaries if complete else None,point_candidate_counts=dict(counts),actual_counter_totals={k:dict(v) for k,v in totals.items()},external_declared_metadata=own['external_declared_metadata'],external_metadata_current_assets_not_verified=True)
    write(a.output/FILES[2],result);print('WHOLE_U_FRESH_RUNTIME_CHECK',result['passed'],result['checks'],failures,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--private',type=Path,default=PRIVATE);p.add_argument('--input',type=Path,default=PRIVATE/'geometry');p.add_argument('--validation-input',type=Path,default=PRIVATE/'validation');p.add_argument('--runtime-input',type=Path,default=PRIVATE/'runtime');p.add_argument('--output',type=Path,default=PRIVATE/'runtime_checks');a=p.parse_args()
    for key in ('repo','private','input','validation_input','runtime_input','output'):setattr(a,key,getattr(a,key).resolve())
    (freeze if a.stage=='freeze' else run)(a)
if __name__=='__main__':main()
