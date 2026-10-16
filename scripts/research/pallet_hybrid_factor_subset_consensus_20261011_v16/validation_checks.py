"""Own-frozen stdlib check of the two real same-hybrid S15 solver paths.

Reuse exact independent scalar AST bodies; no production imports, images,
GT, models, fitting, PnP or SVD. Native endpoint decisions are scalar checked
against their fresh full V4 witnesses. Stored spectra are only recounted.
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

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
PRIVATE=Path('/tmp/pallet-hybrid-factor-subset-private-20261011-v16')
PARENT=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
SUBSET=REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/validation_checks.py'
SUBSET_SHA='ad945e8985b9ca19a576d2423e83b5b635a20b1df44495c2b683a2659c752f26'
ENDPOINT=REPO/'scripts/research/pallet_unused_line_endpoint_audit_20261011_v10/validation_checks.py'
ENDPOINT_SHA='32ed14623a7e894168d0357f9275edf7eb6d4f0605f7e986ea4a4adba14ff52e'
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
FINITE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py'
GUARD=REPO/'scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py'
NATIVE=REPO/'scripts/research/pallet_three_head_observation_20261010_v7/validation_checks.py'
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(new_production_calls=0,all_saved_successful_proposals_and_refits_checked=True,
    fresh_endpoint_candidates_checked=True,stored_spectrum_recount_only=True,
    mixed_LOCAL_root_completeness_proven=False,physical_ownership_certified=False,
    hardware_authenticity_proven=False,quiet_snapshots_can_miss_transient_contention=True,
    inherited_64_coordinate_ULP_cusp_reconciliation_is_not_a_performance_gate=True,
    one_sided_hinge_J_is_not_observability_equality=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())

def ready_proof(f,cp,cpu,cs):
    old,op,os_=[read(f[k]) for k in ('cpu_original_checks','cpu_original_protocol','cpu_original_started')]
    require(op['schema']=='hybrid_factor_subset_CPU_protocol_v16' and old['schema']=='hybrid_factor_subset_CPU_checks_v16' and old['passed'] is False and old['tests_run']==0 and old['failures']==old['errors']==[] and old['error']['type']=='ImportError' and "cannot import name 'PoseBank'" in old['error']['message'],'preserved original zero-test import failure')
    require(old['inputs']==op['inputs'] and all(cp['inputs'][k]==v for k,v in op['inputs'].items()) and same(old['own_protocol'],binding(f['cpu_original_protocol'])) and same(os_['protocol'],old['own_protocol']),'preserved original CPU source/protocol')
    require(old['configured_environment']==op['configured_environment']==os_['configured_environment']==cp['configured_environment'],'same original CPU environment')
    for key in ('actual_OpenCV_entry_calls','actual_OpenCV_return_calls','actual_OpenCV_exception_calls','actual_SVD_dispatch','actual_instrumented_optimizer_dispatch'):require(not any(old[key].values()),'original numerical calls must be zero')
    marker=cp['import_only_ready_review'];require(marker==cpu['import_only_ready_review']==cs['import_only_ready_review'] and marker['original_failure_preserved'] is True and marker['original_test_groups_unchanged']==5 and marker['pipeline_import_location_changes']==marker['test_import_location_changes']==1 and marker['solver_math_threshold_or_test_expectation_changes']==marker['original_numerical_calls']==0 and marker['configured_environment']==cp['configured_environment'],'same-source import-only successful authority')
    tree=ast.parse(f['code:pipeline.py'].read_text());changes=0
    for i,node in enumerate(tree.body):
        if isinstance(node,ast.ImportFrom) and node.module=='pallet_factor_subset_consensus_20261011_v15.solver' and [v.name for v in node.names]==['FactorSubsetConsensusBank','PoseBank']:
            tree.body[i:i+1]=ast.parse('from ..pallet_factor_subset_consensus_20261011_v15.solver import FactorSubsetConsensusBank\nfrom ..pallet_cornerwise_independent_20261010_v4.pose import PoseBank').body;changes+=1;break
    require(changes==1 and ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(f['code:pipeline_ready.py'].read_text()),include_attributes=False),'one pipeline import-only full AST proof')
    tree=ast.parse(f['code:test_pipeline.py'].read_text());changes=0
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module is None and len(node.names)==1 and node.names[0].name=='pipeline':node.names[0].name='pipeline_ready';changes+=1
    require(changes==1 and ast.dump(tree,include_attributes=False)==ast.dump(ast.parse(f['code:test_pipeline_ready.py'].read_text()),include_attributes=False),'one test import-only full AST proof')

def selected(tree,names):
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),'exact helper function whitelist');return nodes
def extraction():
    require(binding(SUBSET)['sha256']==SUBSET_SHA and binding(ENDPOINT)['sha256']==ENDPOINT_SHA,'immutable independent scalar kernels')
    old=ast.parse(SUBSET.read_text());nodes=selected(old,('extraction','_branch','ranks','incidence','candidate'))
    packet=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='packet')
    start=next(i for i,n in enumerate(packet.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='op' for t in n.targets))
    end=next(i for i,n in enumerate(packet.body) if isinstance(n,ast.For) and any(isinstance(q,ast.Constant) and q.value=='actual_filtered_local_solve_calls' for q in ast.walk(n)))
    body=copy.deepcopy(packet.body[start:end]);guard=next(i for i,n in enumerate(packet.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='guard_w' for t in n.targets))
    body+=copy.deepcopy(packet.body[guard:guard+2])
    expected=next(n for n in packet.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_opt' for t in n.targets));body.append(copy.deepcopy(expected))
    pre=ast.parse("q,H,s=parent['input_points'],parent['hidden_initial'],row['solver']\nU=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]\nedges=[l['edge'] for l in lines]").body
    fn=ast.parse('def solver_core(row,parent,obs,endpoint,lines,h,a,policy,counts,label):\n pass').body[0]
    fn.body=pre+body+ast.parse('return expected_opt').body
    native_tree=ast.parse(NATIVE.read_text());names=('finite','ids','eligible','dot','project','determinant3','native_hash','dimensions','rotation_physical','equivalent','stored_rank','ledger','candidate','pose_packet')
    native=selected(native_tree,names);constants={'EDGES','FALSE_FLAGS','ATOL','RTOL','RADIUS'};found=set()
    for n in native_tree.body:
        if isinstance(n,ast.Assign):
            ids={q.id for t in n.targets for q in ast.walk(t) if isinstance(q,ast.Name)}
            if ids and ids<=constants:native.append(n);found.update(ids)
    require(found==constants and binding(NATIVE)['sha256']=='815fd15d8596fe6c8c85720e136627dcef8d4cdb22c23e00737b3f507a3fa4a6','immutable native scalar whitelist')
    native+=selected(ast.parse(ENDPOINT.read_text()),('native_problem','difference','record_check'))
    trees=[ast.fix_missing_locations(ast.Module(body=v,type_ignores=[])) for v in (nodes+[fn],native)]
    info=dict(subset_source=binding(SUBSET),endpoint_source=binding(ENDPOINT),native_source=binding(NATIVE),
        AST_sha256=[hashlib.sha256(ast.dump(t,include_attributes=False).encode()).hexdigest() for t in trees],
        unchanged_solver_math_body_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest(),
        source_packet_metadata_removed_only=True,scalar_formula_AST_changes=0)
    return trees,info

def helpers(trees):
    h=dict(math=math,json=json,hashlib=hashlib,struct=struct,copy=copy,Counter=Counter,combinations=combinations,
        require=require,binding=binding,rows=rows,SOURCE=SOURCE,SOURCE_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e',
        FINITE=FINITE,FINITE_SHA='91d014543eba01c142cad9a0c69affd2eb16666913ce13a06a56857f854c98ef',GUARD=GUARD,STATE=dict(context={},events=[]))
    exec(compile(trees[0],str(SUBSET)+':unchanged_scalar_math','exec'),h)
    tree,policy,_=h['extraction']();exec(compile(tree,str(SOURCE)+':independent_projection_J','exec'),h)
    n=dict(math=math,struct=struct,hashlib=hashlib,require=require)
    exec(compile(trees[1],str(ENDPOINT)+':native_endpoint_scalar_only','exec'),n);return h,n,policy

def source_checks(parent,obs,completion,h,a,label):
    q,H=parent['input_points'],parent['hidden_initial'];contract=parent['observation_contract'];adopted=contract['hybrid_boundary_corner_ids'];native=obs['native_N3_points']
    a.eq(parent['method'],'IMAGE_ROLE_CORNERWISE_HYBRID',label+'.parent_method');a.eq(parent['observation_supply'],'CORNERWISE_HYBRID',label+'.supply')
    a.eq(parent['predicted_initial_N3_hidden'],H,label+'.same_H');a.eq(obs['predicted_N3_hidden'],H,label+'.same_obs_H');a.eq(obs['GT_input'],False,label+'.GT_free')
    a.eq(obs['initial_N3_pose'],parent['initial_pose'],label+'.same_initial');a.eq(obs['raw_logits_sha256'],parent['observation_raw_logits_sha256'],label+'.same_ROLE_logits')
    a.eq(parent['selected_corner_ids'],adopted,label+'.original_selected');a.eq(contract['actual_self_hidden_ids'],H,label+'.contract_H');a.ok(not set(H)&set(adopted),label+'.H_not_adopted')
    a.eq(contract['native_N3_RGB_corner_ids'],[k for k in range(8) if k not in adopted],label+'.explicit_native_ids')
    for k in ('same_N3_bank_reused_across_heldouts',):a.eq(contract[k],True,label+'.'+k)
    for k in ('heldout_projection_is_observation','LOO_initial_prior_includes_heldout_influence'):a.eq(contract[k],False,label+'.'+k)
    corners={c['id']:c for c in obs['corners']};records={r['id']:r for r in contract['cornerwise_records']}
    a.eq(len(corners),len(obs['corners']),label+'.unique_corner_sources');a.eq(len(records),len(contract['cornerwise_records']),label+'.unique_old_LOO_records')
    for k in range(8):
        if k in adopted:
            c,r=corners[k],records[k];a.eq(q[k],c['xy'],label+'.actual_boundary.'+str(k));a.eq(q[k],r['candidate_xy'],label+'.old_accepted_xy.'+str(k));a.eq(r['accepted'],True,label+'.old_selection.'+str(k))
            a.eq(r['heldout_fit_requested_excluded_ids'],sorted(set(H)|{k}),label+'.old_LOO_exclusion.'+str(k));a.eq(r['fit_exclusion_verified'],True,label+'.old_LOO_verified');a.eq(r['projected_coordinate_used_as_observation'],False,label+'.old_LOO_no_projection')
        else:a.eq(q[k],native[k],label+'.actual_native.'+str(k))
    a.eq(q[8],native[8],label+'.center');a.eq(parent['head_checkpoint_sha256'],'882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d',label+'.corrected_IMAGE_ROLE')
    U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]
    lines,consumed=h['original_lines'](obs,q,U,adopted,a,label+'.source_lines');return U,adopted,lines,consumed

def packet(two,ledger,parent,obs,completion,h,n,a,policy,counts,native_counts,label):
    U,adopted,original,consumed=source_checks(parent,obs,completion,h,a,label);q,H=parent['input_points'],parent['hidden_initial'];raw={l['edge']:l for l in obs['lines']}
    local=dict(parent_row=parent,native_N3_points=obs['native_N3_points'],parent_completion=completion,role_observation=obs)
    for key,value in (('input_packet_semantic_sha256',h['semantic'](local)),('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('selected_input_semantic_sha256',h['semantic'](q)),('actual_filtered_solver_calls',2),('reported_output_views',2),('same_selected_coordinate_bank_shared',True),('actual_adopted_boundary_corner_ids',adopted),('initial_H',H)):a.eq(ledger[key],value,label+'.ledger.'+key)
    line_ledger=ledger['point_line_ledger'];f=line_ledger['observation_filter'];records=f['endpoint_records'];a.eq([r['edge'] for r in records],[l['edge'] for l in original],label+'.one_fresh_endpoint_request_each')
    before=line_ledger['endpoint_native_bank_before'];start=copy.deepcopy(before);native_row=dict(initial_H=H,original_native_N3_points=obs['native_N3_points'],K=parent['K'],xyz=parent['xyz'],raw_hw=parent['raw_hw'])
    for i,record in enumerate(records):before,_=n['record_check'](record,native_row,n,raw[record['edge']],before,a,native_counts,label+'.fresh_endpoint.'+str(i))
    a.eq(before,line_ledger['endpoint_native_bank_after'],label+'.fresh_native_after');delta=n['difference'](before,start,a,label+'.native_delta');a.eq(ledger['endpoint_native_bank_operation_counts'],delta,label+'.driver_native_delta')
    for k,v in delta.items():counts['native:'+k]+=v
    rejected=[r['edge'] for r in records if r['decision']=='CONTRADICTED_REJECT'];filtered=copy.deepcopy(obs);filtered['lines']=[l for l in obs['lines'] if l['edge'] not in rejected]
    retained=[l for l in original if l['edge'] not in rejected];pointobs=copy.deepcopy(obs);pointobs['lines']=[]
    for key,value in (('endpoint_decision_source','FRESH_NATIVE_ENDPOINT_LOO'),('original_observation_semantic_sha256',h['semantic'](obs)),('filtered_observation_semantic_sha256',h['semantic'](filtered)),('rejected_edge_ids',rejected),('retained_unused_edge_ids',[l['edge'] for l in retained]),('unknown_retained_edge_ids',[r['edge'] for r in records if r['decision']=='UNVERIFIED_RETAIN']),('consumed_edges',consumed),('physical_ownership_truth','UNKNOWN'),('corners_and_queries_unchanged',True),('no_endpoint_projection_as_observation',True)):a.eq(f[key],value,label+'.filter.'+key)
    a.eq(f['contract_error'],None,label+'.valid_source_line_contract');a.eq(line_ledger['endpoint_validation_requests'],len(records),label+'.fresh_requests');a.eq(line_ledger['endpoint_decision_records'],len(records),label+'.fresh_records')
    expected_dispatch=Counter();operation=Counter()
    for index,row in enumerate(two):
        method=METHODS[index];s=row['solver'];lines=retained if index==0 else [];usedobs=filtered if index==0 else pointobs
        for key,value in (('method',method),('parent_method',parent['method']),('observation_supply','CORNERWISE_HYBRID'),('input_points',q),('original_native_N3_points',obs['native_N3_points']),('hidden_initial',H),('initial_pose',parent['initial_pose']),('selected_corner_ids',adopted),('parent_completion',completion),('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('point_only_observation_semantic_sha256',h['semantic'](pointobs)),('observation_contract',parent['observation_contract'])):a.eq(row[key],value,label+'.view.'+str(index)+'.'+key)
        for key in ('K','xyz','raw_hw','selected_index','fixed_metadata','head_arm','head_mode','head_checkpoint_sha256','calibration_binding'):a.eq(row[key],parent[key],label+'.preserved.'+key)
        a.eq(row['hybrid_consensus_policy'],h['execution_policy'],label+'.policy');a.eq(s['policy'],h['execution_policy']['solver'],label+'.unchanged_S15')
        a.eq(s['used'],U,label+'.same_U');a.eq(s['hidden'],H,label+'.same_fit_H');a.eq(s['excluded'],H,label+'.excluded_H');a.eq(s['line_edges'],[l['edge'] for l in lines],label+'.actual_line_pool');a.eq(s['consumed_edges'],consumed if index==0 else [],label+'.consumed_once')
        for key in ('prior_used','initial_pose_residual_prior','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','center_used_for_fit_or_score','same_edge_point_line_double_count','mixed_root_completeness_proven'):a.eq(s[key],False,label+'.no_fake.'+key)
        a.eq(row['observation_filter'],f if index==0 else None,label+'.matched_line_ablation');a.eq(row['partial_lines_not_pose_inputs'],bool(index),label+'.point_only');a.eq(row['head_forward_calls'],0,label+'.no_new_model');a.eq(row['new_corner_selection_calls'],0,label+'.no_new_selection')
        stored={v['edge']:v for v in s.get('actual_source_lines',[])};a.eq(sorted(stored),[l['edge'] for l in lines],label+'.complete_source_lines')
        exact=[]
        for l in lines:
            w=stored[l['edge']];a.near(w['normal'],l['normal'],label+'.source_normal');a.near(w['offset'],l['offset'],label+'.source_offset');a.eq(w['raw'],raw[l['edge']],label+'.source_support')
            exact.append(dict(edge=l['edge'],normal=w['normal'],offset=w['offset'],support_query_ids=l['support_query_ids'],source_line=w['raw']))
        s_copy=s if index==0 else ledger['point_only_solver'];a.eq(s,s_copy,label+'.saved_solver_join')
        if index==0:a.eq(s,line_ledger['full_solver'],label+'.line_full_solver')
        dispatch=h['solver_core'](row,parent,obs,dict(contract_error=None),exact,h,a,policy,counts,label+'.'+method);expected_dispatch.update(dispatch);operation.update(s['operation_counts'])
        a.eq(s['observation_binding'],dict(sha256=hashlib.sha256(json.dumps(usedobs,sort_keys=True,separators=(',',':'),allow_nan=True).encode()).hexdigest(),bytes=len(json.dumps(usedobs,sort_keys=True,separators=(',',':'),allow_nan=True).encode()),definition='semantic sorted JSON; not file-byte provenance'),label+'.actual_observation_binding')
        for key in ('mixed_subsets','registered_dimensions','direct_valid_roots','LOCAL_optimizer_calls_at_most','maximum_direct_Generic_calls'):
            value=(s.get('call_bounds') or {}).get(key);counts[('bound:' if type(value) is int else 'bound_unavailable:')+method+':'+key]+=value if type(value) is int else 1
        counts['views']+=1;counts['status:'+method+':'+row['output_status']]+=1;counts['state:'+method+':'+s['state']]+=1
        counts['stored_proposals']+=len(s['proposal_solutions']);counts['stored_refits']+=len(s['all_candidate_solutions'])
    a.eq(two[1]['solver']['mixed_subset_enumeration'],[],label+'.point_only_no_mixed');a.eq(two[1]['solver']['hypothesis_bank_operation_counts']['generic_calls'],0,label+'.point_only_no_new_Generic');a.eq(two[1]['solver']['hypothesis_bank_operation_counts']['generic_cache_misses'],0,label+'.point_only_no_cache_miss')
    b0,b1,b2=[ledger[k] for k in ('shared_point_bank_before','shared_point_bank_after_line','shared_point_bank_after_point')]
    a.ok(all(v==0 for v in b0.values()),label+'.fresh_selected_bank');d1=n['difference'](b1,b0,a,label+'.line_bank');d2=n['difference'](b2,b1,a,label+'.point_bank');total=n['difference'](b2,b0,a,label+'.shared_bank')
    a.eq(d1,two[0]['solver']['hypothesis_bank_operation_counts'],label+'.line_direct_delta');a.eq(d2,two[1]['solver']['hypothesis_bank_operation_counts'],label+'.point_direct_delta');a.eq(total,ledger['shared_point_bank_operation_counts'],label+'.one_shared_bank_delta')
    a.eq(ledger['shared_point_bank_input_hash'],two[0]['solver']['input_hash'],label+'.shared_q_hash');a.eq(two[1]['solver']['input_hash'],two[0]['solver']['input_hash'],label+'.both_same_q_hash')
    a.eq(ledger['point_only_hypothesis_bank_operation_counts'],d2,label+'.point_only_ledger_delta');a.eq(ledger['point_only_bank_counts'],two[1]['solver']['local_bank_counts'],label+'.point_only_local_counts')
    for key,value in (('actual_filtered_local_solve_calls',1),('reported_output_views',1),('actual_direct_point_bank_reused',True),('local_operation_counts',two[0]['solver']['operation_counts']),('direct_point_bank_operation_counts',d1),('factor_subset_call_bounds',two[0]['solver'].get('call_bounds'))):a.eq(line_ledger[key],value,label+'.line_path_ledger.'+key)
    for field,prefix,keys in (('primitive_entry_calls','cv:entry:',h['PRIMITIVES']),('primitive_return_calls','cv:return:',h['PRIMITIVES']),('primitive_exception_calls','cv:exception:',h['PRIMITIVES']),('svd_dispatch','svd:',('entries','returns','exceptions')),('optimizer_dispatch','opt:',tuple(expected_dispatch))):
        require(set(ledger[field])==set(keys) and all(type(v) is int and v>=0 for v in ledger[field].values()),'complete actual nonnegative counters')
        for k,v in ledger[field].items():counts[prefix+k]+=v
    a.eq(ledger['optimizer_dispatch'],dict(expected_dispatch),label+'.both_solver_optimizer_dispatch')
    for k in h['PRIMITIVES']:a.eq(ledger['primitive_entry_calls'][k],ledger['primitive_return_calls'][k]+ledger['primitive_exception_calls'][k],label+'.CV_complete.'+k)
    a.eq(ledger['primitive_entry_calls']['solvePnPGeneric'],total['generic_calls']+delta['generic_calls'],label+'.direct_and_native_Generic');a.eq(ledger['primitive_entry_calls']['solvePnPRefineLM'],delta['lm_calls'],label+'.native_only_LM')
    for name in ('solvePnP','cornerSubPix'):a.eq(ledger['primitive_entry_calls'][name],0,label+'.no_additional_solver.'+name)
    dispatch=ledger['actual_solver_dispatch']
    for key,value in (('filtered_solver_entries',2),('filtered_solver_returns',2),('filtered_solver_exceptions',0),('endpoint_solver_entries',len(records)),('endpoint_solver_returns',len(records)),('endpoint_solver_exceptions',0)):a.eq(dispatch[key],value,label+'.actual_dispatch.'+key)
    a.eq(ledger['svd_dispatch']['entries'],ledger['svd_dispatch']['returns']+ledger['svd_dispatch']['exceptions'],label+'.SVD_complete')
    for k,v in operation.items():counts['operation:'+k]+=v
    for k,v in total.items():counts['shared:'+k]+=v
    counts['frames']+=1;counts['endpoint_requests']+=len(records)

def input_paths(args):
    p=args.input;out=dict(geometry=p/'GEOMETRY_SEALED.jsonl.gz',ledgers=p/'CONTROL_LEDGERS.jsonl.gz',repair_protocol=p/'PROTOCOL.json',geometry_seal=p/'GEOMETRY_SEAL.json',repair_receipt=p/'CONTROL_RECEIPT.json',parent_population_checks=p/'PARENT_POPULATION_CHECKS.json',checker=Path(__file__).resolve(),started=p/'CONTROL_STARTED.json',preflight=p/'PREFLIGHT.json',cpu_checks=p/'CPU_READY_CHECKS.json',cpu_protocol=p/'CPU_READY_PROTOCOL.json',cpu_started=p/'CPU_READY_STARTED.json',cpu_original_checks=p/'CPU_CHECKS.json',cpu_original_protocol=p/'CPU_PROTOCOL.json',cpu_original_started=p/'CPU_STARTED.json',cohort=COHORT,subset_scalar=SUBSET,endpoint_scalar=ENDPOINT,projection_scalar=SOURCE,finite_scalar=FINITE,normal_guard=GUARD,native_scalar=NATIVE,source_geometry=PARENT/'GEOMETRY_SEALED.jsonl.gz',source_observations=PARENT/'OBSERVATIONS.jsonl.gz')
    for name in ('pipeline.py','runner.py','test_pipeline.py','cpu_checks.py','pipeline_ready.py','test_pipeline_ready.py','cpu_ready.py'):out['code:'+name]=CODE/name
    parent=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
    for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):out['parent:'+name]=parent/name
    for i,_ in enumerate(read(out['repair_receipt'])['resource_snapshots']):out['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return out
def metadata(f,b):
    core,seal,r,pop=[read(f[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_hybrid_factor_subset_consensus_protocol_v16' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490,'fixed matched490 protocol')
    require(seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'complete GT-free paired seal')
    require(r['schema']=='hybrid_factor_subset_consensus_execution_v16' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'complete cleanup')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):require(same(seal[key],b[alias]),'sealed bytes '+key)
    require(same(r['protocol'],b['repair_protocol']) and all(same(r['raw_rows'][k],b[k]) for k in ('geometry','ledgers')),'actual receipt bytes')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and same(core['parent_population_checks'],b['parent_population_checks']),'full parent population')
    for key in b:
        if key in ('geometry','ledgers','repair_protocol','geometry_seal','repair_receipt','parent_population_checks','started','preflight') or key.startswith('resource:'):continue
        require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in core['inputs'].values()),'unbound scalar/public input '+key)
    cpu,cp,cs=[read(f[k]) for k in ('cpu_checks','cpu_protocol','cpu_started')]
    require(cpu['schema']=='hybrid_factor_subset_CPU_ready_checks_v16' and cp['schema']=='hybrid_factor_subset_CPU_ready_protocol_v16' and cpu['complete'] is True and cpu['passed'] is True and cpu['tests_run']==5 and not cpu['failures'] and not cpu['errors'] and cpu['error'] is None and cpu['inputs']==cp['inputs'] and same(cpu['own_protocol'],b['cpu_protocol']) and same(cs['protocol'],b['cpu_protocol']),'same-source CPU_READY5 PASS');ready_proof(f,cp,cpu,cs)
    for value in cp['inputs'].values():require(same(binding(REPO/value['path']),value),'CPU source closure')
    require(cpu['configured_environment']==cp['configured_environment']==cs['configured_environment']==dict(PALLET_SOURCE_ROOT=core['source_root'],PALLET_BASELINE_ROOT=core['baseline_root']),'CPU source environment')
    for key in ('source_unchanged','opencv_threads_restored','environment_restored','instruments_restored','asset_canary_restored','normal_package_import'):require(cpu[key] is True,'CPU restoration '+key)
    require(cpu['cleanup_errors']==[] and cpu['forbidden_real_asset_reads']==[] and cpu['real_frame_model_RGB_GT_dataset_reads']==0,'synthetic CPU no real assets')
    pc,po,ps,pr=[read(f['parent:'+name]) for name in ('VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json')]
    require(pc['schema']=='factor_subset_consensus_scalar_checks_v15' and pc['complete'] is True and pc['passed'] is True and pc['failure_count']==0 and pc['exception'] is None and pc['inputs']==po['inputs'] and same(pc['own_protocol'],b['parent:VALIDATION_PROTOCOL.json']) and same(pc['checker'],b['subset_scalar']),'immutable parent actual scalar PASS')
    require(ps['complete'] is True and ps['frames']==ps['rows']==ps['ledger_rows']==245 and ps['GT_read_allowed'] is False and pr['complete'] is True and pr['cleanup_completed'] is True and pr['error'] is pr['cleanup_error'] is None,'immutable parent full clean seal')
    for key,alias in (('repair_protocol','parent:PROTOCOL.json'),('geometry_seal','parent:GEOMETRY_SEAL.json'),('repair_receipt','parent:CONTROL_RECEIPT.json')):require(same(pc['inputs'][key],b[alias]),'parent checked bytes '+key)
    pst=read(f['parent:VALIDATION_STARTED.json']);require(pst['inputs']==po['inputs'] and same(pst['own_protocol'],b['parent:VALIDATION_PROTOCOL.json']) and pst['new_production_calls']==0,'parent own scalar STARTED')
    for key in ('started','preflight'):require(same(read(f[key])['protocol'],b['repair_protocol']),'source '+key+' protocol')
    require(read(f['preflight'])['passed'] is True,'source preflight PASS');return core,r,pop
def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing non-symlink folders')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive own output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (args.output/name).exists(),'preserve '+name)
def freeze(args):
    guard(args,'freeze');f=input_paths(args);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,info=extraction()
    write(args.output/FILES[0],dict(schema='hybrid_factor_subset_scalar_protocol_v16',inputs=b,checker=b['checker'],methods=list(METHODS),frames=245,rows=490,helpers=info,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=input_paths(args);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);trees,info=extraction()
    require(own['schema']=='hybrid_factor_subset_scalar_protocol_v16' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own frozen closure differs');core,r,pop=metadata(f,b)
    write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,new_production_calls=0));h,n,policy=helpers(trees);h['execution_policy']=core['policy'];a=h['Audit']();counts=Counter();native_counts=Counter();error=None
    try:
        cohort=read(f['cohort']);ordered=cohort['ids'];known=set(ordered);require(len(ordered)==len(known)==245 and [x['id'] for x in cohort['frames']]==ordered,'full ordered245');a.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'scope')
        geometry=iter(rows(f['geometry']));ledgers=iter(rows(f['ledgers']));parents=iter(h['primary'](f['source_geometry'],'method','IMAGE_ROLE_CORNERWISE_HYBRID',1960,known,a,'V7_geometry'));observations=iter(h['primary'](f['source_observations'],'head_arm','IMAGE_ROLE',735,known,a,'V7_ROLE'))
        for i,fid in enumerate(ordered):
            two=[next(geometry,None),next(geometry,None)];ledger,parent,obs=[next(stream,None) for stream in (ledgers,parents,observations)]
            require(all(isinstance(v,dict) and v['id']==fid and v['session']==cohort['frames'][i]['session'] for v in (*two,ledger,parent,obs)),'complete same-frame paired source joins')
            packet(two,ledger,parent,obs,pop['completion'],h,n,a,policy,counts,native_counts,'frame.'+str(i))
        for stream in (geometry,ledgers,parents,observations):a.eq(next(stream,None),None,'no_extra_rows')
        for key,value in (('frames_started',245),('frames_complete',245),('filtered_solver_entries',490),('filtered_solver_returns',490),('filtered_solver_exceptions',0),('geometry_rows',490),('ledger_rows',245),('endpoint_solver_entries',counts['endpoint_requests']),('endpoint_solver_returns',counts['endpoint_requests']),('endpoint_solver_exceptions',0)):a.eq(r['actual_execution'][key],value,'actual.'+key)
        for field,prefix in (('local_operation_counts','operation:'),('shared_point_bank_operation_counts','shared:'),('endpoint_native_bank_operation_counts','native:')):a.eq(r[field],{k.removeprefix(prefix):v for k,v in counts.items() if k.startswith(prefix)},'actual.'+field)
        for field,prefix in (('actual_OpenCV_entry_calls','cv:entry:'),('actual_OpenCV_return_calls','cv:return:'),('actual_OpenCV_exception_calls','cv:exception:')):a.eq(r[field],{k:counts[prefix+k] for k in h['PRIMITIVES']},'actual.'+field)
        a.eq(r['actual_optimizer'],{k:counts['opt:'+k] for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')},'actual.optimizer');a.eq(r['actual_SVD_dispatch'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'actual.SVD')
        for field,prefix in (('aggregate_declared_call_bounds','bound:'),('aggregate_bound_unavailable_frame_counts','bound_unavailable:'),('solver_state_counts','state:')):a.eq(r[field],{m:{k.removeprefix(prefix+m+':'):v for k,v in counts.items() if k.startswith(prefix+m+':')} for m in METHODS},'actual.'+field)
        a.eq(r['fresh_endpoint_validation_requests'],counts['endpoint_requests'],'actual.fresh_endpoint_requests')
        a.eq(r['output_status_counts'],{m:{s:counts['status:'+m+':'+s] for s in ('NEW_POSE','N3_BASELINE_FALLBACK','POSE_FAILURE') if counts['status:'+m+':'+s]} for m in METHODS},'actual.all_statuses')
        a.eq(r['forbidden_entries'],{},'no_forbidden_numeric_paths');a.eq(r['forbidden_asset_reads'],[],'no_truth_or_model_reads');a.eq(r['protection_before'],r['protection_after'],'protected_bytes')
        for key in ('primitive_and_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):a.eq(r['cleanup'][key],True,'cleanup.'+key)
        for i,snap in enumerate(r['resource_snapshots']):a.eq(read(f['resource:%03d'%i]),snap,'resource.identity');a.eq(snap['quiet'],True,'resource.recorded_quiet');a.eq(snap['temperature_under_80'],True,'resource.recorded_thermal')
        a.eq(counts['frames'],245,'all_frames');a.eq(counts['views'],490,'all_views');a.eq(counts['proposal_checks'],counts['stored_proposals'],'all_proposal_math');a.eq(counts['refit_checks'],counts['stored_refits'],'all_refit_math');a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result=dict(schema='hybrid_factor_subset_scalar_checks_v16',complete=error is None and counts['frames']==245,passed=error is None and counts['frames']==245 and a.failure_count==0,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,max_scalar_absolute_difference=a.max_difference,inputs=b,own_protocol=ob,checker=b['checker'],counts=dict(counts),native_counts=dict(native_counts),cusp_reconciliations=h['STATE']['events'],limits=LIMITS,new_production_calls=0,first_failure_preserved=True)
    write(args.output/FILES[2],result);print('HYBRID_SUBSET_SCALAR',result['passed'],a.checks,a.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();{'freeze':freeze,'run':run}[args.stage](args)
if __name__=='__main__':main()
