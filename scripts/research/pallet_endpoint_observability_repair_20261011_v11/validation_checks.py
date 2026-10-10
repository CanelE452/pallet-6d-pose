"""Own-frozen, stdlib scalar check of one filtered solve and its two views.

Unchanged V9 scalar candidate/J/residual/cost checks are AST-extracted, never
production-imported. No model, RGB, GT, PnP, optimizer or SVD calls. Stored
singular spectra are recounted; weak routing uses the original V1 expression.
"""
from __future__ import annotations
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
import struct
import time
import traceback

REPO=Path(__file__).resolve().parents[3]
DOC=REPO/'_docs/experiments/pallet_endpoint_observability_repair_20261011_v11'
PRIVATE=Path('/tmp/pallet-endpoint-observability-repair-private-20261011-v11')
V10=REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
V7=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
SOURCE_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
V1=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'
FUNCS=('require','read','rows','binding','same_binding','semantic','safe','write_new','finite','dot','mm','eye','skew',
    'rotation_and_right_J','dimensions','model','forward','physical_rotation','equivalent','eligible','factor_rows',
    'rank_witness','geometry_witness','original_lines','candidate')
CONSTANTS=('EDGES','PHYSICAL','CAL','ATOL','RTOL','RANK','TIE','LOCAL_KEYS','PRIMITIVES')
METHODS=('FILTERED_LOCAL','FILTERED_LOCAL_WEAK_SAFE')
OUTPUTS=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(own_freeze_after_complete_geometry_before_own_arithmetic=True,production_imports_model_RGB_GT_PnP_optimizer_SVD_calls=0,
    unchanged_V9_scalar_candidate_J_residual_cost_checks=True,stored_singular_rank_recount_only=True,
    exact_original_sparse_coordinates_no_native_numeric_fill=True,two_views_share_one_actual_filtered_solver=True,
    endpoint_UNKNOWN_retained=True,weak_selected_candidate_not_all_alternatives=True,
    weak_gate_is_new_operating_policy_from_existing_V1_threshold=True,physical_ownership_certified=False,
    global_uniqueness_proven=False,no_hardware_authenticity_claim=True,automatic_retry=False)

def require(value,message):
    if not value:raise ValueError(message)

def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()),'blank row');yield json.loads(line)

def binding(path):
    p=Path(path).resolve();require(p.is_file() and not any(x.is_symlink() for x in (Path(path),*Path(path).parents)),'missing/symlink input '+str(path))
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,sha256=digest.hexdigest(),bytes=p.stat().st_size)

def same(a,b):return all(a.get(k)==b.get(k) for k in ('sha256','bytes'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'unchanged V9 scalar source required')
    selected=[];names=set()
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in (*FUNCS,'Audit'):selected.append(n);names.add(n.name)
        elif isinstance(n,ast.Assign):
            assigned={x.id for t in n.targets for x in (t.elts if isinstance(t,ast.Tuple) else [t]) if isinstance(x,ast.Name)}
            if assigned and assigned<=set(CONSTANTS):selected.append(n);names.update(assigned)
    require(names==set(FUNCS)|set(CONSTANTS)|{'Audit'},'exact V9 scalar AST whitelist required')
    module=ast.Module(body=selected,type_ignores=[])
    cls=next(n for n in ast.parse(V1.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HypothesisBank')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_jacobian')
    expr=[n.value for n in ast.walk(fn) if isinstance(n,ast.keyword) and n.arg=='weak_condition']
    expected=ast.parse('bool(s[-1] < s[0] * 1e-4)',mode='eval').body
    require(len(expr)==1 and ast.dump(expr[0],include_attributes=False)==ast.dump(expected,include_attributes=False),'original weak expression differs')
    info=dict(source=binding(SOURCE),AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),
        weak_source=binding(V1),weak_AST_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
        weak_expression='bool(s[-1] < s[0] * 1e-4)')
    return module,info


def helpers(info):
    module,current=extraction();require(current==info,'own-frozen helper AST differs')
    env=dict(REPO=REPO,math=math,json=json,hashlib=hashlib,Path=Path,gzip=gzip,os=os,struct=struct)
    exec(compile(module,str(SOURCE)+':unchanged_scalar_AST_only','exec'),env);return env


def input_paths(args):
    p=args.input;paths=dict(geometry=p/'GEOMETRY_SEALED.jsonl.gz',ledgers=p/'CONTROL_LEDGERS.jsonl.gz',repair_protocol=args.protocol,
        geometry_seal=p/'GEOMETRY_SEAL.json',repair_receipt=p/'CONTROL_RECEIPT.json',parent_population_checks=p/'PARENT_POPULATION_CHECKS.json',
        checker=Path(__file__).resolve(),scalar_source=SOURCE,weak_source=V1,cohort=args.cohort,
        source_geometry=args.parent/'GEOMETRY_SEALED.jsonl.gz',source_observations=args.parent/'OBSERVATIONS.jsonl.gz',
        source_audit=args.audit/'AUDIT_ROWS.jsonl.gz',source_audit_protocol=args.audit/'AUDIT_PROTOCOL.json',
        source_audit_seal=args.audit/'AUDIT_SEAL.json',source_audit_validation=args.audit/'VALIDATION_CHECKS.json',
        source_audit_validation_protocol=args.audit/'VALIDATION_PROTOCOL.json',started=p/'CONTROL_STARTED.json',preflight=p/'PREFLIGHT.json')
    for name in ('solver.py','pipeline.py','runner.py','deployment.py'):
        q=Path(__file__).with_name(name)
        if q.is_file():paths['code:'+name]=q
    for i,_ in enumerate(read(paths['repair_receipt'])['resource_snapshots']):paths['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return paths


def metadata(paths,inputs):
    core,seal,receipt,pop=[read(paths[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['methods']==list(METHODS) and core['frames']==245,'fixed two-view cohort differs')
    require(seal['complete'] is True and seal['frames']==245 and seal['rows']==490 and seal['ledger_rows']==245 and
        seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'complete GT-free two-view seal required')
    require(receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,'complete successful cleanup required')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):
        require(same(seal[key],inputs[alias]),'sealed byte join differs: '+key)
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245,'full source population required')
    require(same(receipt['protocol'],inputs['repair_protocol']) and same(core['parent_population_checks'],inputs['parent_population_checks']) and
        all(same(receipt['raw_rows'][k],inputs[k]) for k in ('geometry','ledgers')),'receipt/core sealed inputs differ')
    started,preflight=read(paths['started']),read(paths['preflight'])
    require(same(started['protocol'],inputs['repair_protocol']) and preflight['passed'] is True and
        same(preflight['protocol'],inputs['repair_protocol']),'actual started/preflight protocol differs')
    audit_seal,audit_checks,audit_own=[read(paths[k]) for k in ('source_audit_seal','source_audit_validation','source_audit_validation_protocol')]
    require(audit_seal['complete'] is True and audit_seal['frames']==245 and audit_checks['complete'] is True and
        audit_checks['passed'] is True and audit_checks['inputs']==audit_own['inputs'],'frozen V10 independent PASS required')
    require(same(audit_seal['audit'],inputs['source_audit']) and same(audit_seal['protocol'],inputs['source_audit_protocol']) and
        same(audit_checks['own_protocol'],inputs['source_audit_validation_protocol']) and
        same(audit_checks['inputs']['audit_rows'],inputs['source_audit']),'checked original endpoint bytes differ')
    require(same(core['inputs']['code:validation_checks.py'],inputs['checker']),'repair bound checker differs')
    for key in inputs:
        if key.startswith('code:'):require(same(core['inputs'][key],inputs[key]),'repair source differs: '+key)
    for key in ('source_geometry','source_observations','source_audit','source_audit_protocol','source_audit_seal',
        'source_audit_validation','source_audit_validation_protocol','scalar_source','weak_source','cohort'):
        require(any(b['path']==inputs[key]['path'] and same(b,inputs[key]) for b in core['inputs'].values()),'unbound original source '+key)
    require(core['policy']['weak_relative_threshold']==1e-4,'inherited weak threshold differs')
    return core,seal,receipt,pop


def guard(args,stage):
    for p in (args.input,args.output,args.protocol,args.parent,args.audit,args.cohort):
        require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink ancestry')
    output=args.output.resolve();require(output==DOC.resolve() or output.is_relative_to(PRIVATE.resolve()),'new V11 output only')
    require(args.output.is_dir(),'existing exclusive output directory required')
    for name in OUTPUTS if stage=='freeze' else OUTPUTS[1:]:require(not (output/name).exists(),'preserve '+name)
    return output


def freeze(args):
    output=guard(args,'freeze');paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()};_module,info=extraction()
    metadata(paths,inputs);write(output/OUTPUTS[0],dict(schema='supplemental_endpoint_observability_scalar_protocol_v11',
        inputs=inputs,checker=inputs['checker'],helpers=info,methods=list(METHODS),frames=245,rows=490,limits=LIMITS,
        frozen_before_own_arithmetic=True));print('ENDPOINT_REPAIR_SCALAR_FROZEN',flush=True)


def primary(path,key,name,total,known,audit,label):
    seen=set()
    for value in rows(path):
        identity=(value[key],value['id']);require(value['id'] in known and identity not in seen,'unknown/duplicate '+label);seen.add(identity)
        if value[key]==name:yield value
    audit.eq(len(seen),total,label+'.full_source_rows')


def packet(rank,safe,ledger,parent,obs,endpoint,completion,h,audit,counts,label):
    q,H,s=parent['input_points'],parent['hidden_initial'],rank['solver'];U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]
    local_packet=dict(parent_row=parent,role_observation=obs,native_N3_points=obs['native_N3_points'],parent_completion=completion)
    audit.eq(ledger['input_packet_semantic_sha256'],h['semantic'](dict(local_packet=local_packet,audit_row=endpoint)),label+'.full_input_packet_SHA')
    rejected=sorted(r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='CONTRADICTED_REJECT')
    audit.eq(endpoint['parent_geometry_row_semantic_sha256'],h['semantic'](parent),label+'.endpoint_parent')
    audit.eq(endpoint['original_ROLE_observation_semantic_sha256'],h['semantic'](obs),label+'.endpoint_ROLE')
    for key,value in (('original_sparse_points',q),('original_native_N3_points',obs['native_N3_points']),('K',parent['K']),('xyz',parent['xyz']),('initial_H',H)):
        audit.eq(endpoint[key],value,label+'.endpoint_exact_'+key)
    filtered=copy.deepcopy(obs)
    if endpoint['contract_error'] is None:filtered['lines']=[r for r in filtered['lines'] if r['edge'] not in rejected]
    adopted=parent['observation_contract']['validated_boundary_corner_ids']
    if endpoint['contract_error'] is None:actual,consumed=h['original_lines'](filtered,q,U,adopted,audit,label+'.actual_filtered_source')
    else:
        actual,consumed=[],[];audit.eq(s['state'],'INVALID_OBSERVATION_CONTRACT',label+'.same_invalid_source_contract')
        audit.eq(s['reason'],endpoint['contract_error']['message'],label+'.same_typed_invalid_reason')
    audit.ok(not set(rejected)&set(consumed),label+'.rejected_not_consumed_by_point')
    lookup={r['edge']:r for r in endpoint['endpoint_records']};lines=[]
    for line in actual:
        source=lookup[line['edge']]['source_line'];hline={k:source[k] for k in ('edge','normal','offset','support_query_ids')}
        audit.near(hline,line,label+'.same_saved_normalized_factor.'+str(line['edge']));lines.append(hline)
    edges=[v['edge'] for v in lines];audit.eq(edges,[e for e in endpoint['original_actual_unused_edges'] if e not in rejected],label+'.reject_only_UNKNOWN_retained')
    for row,method in ((rank,METHODS[0]),(safe,METHODS[1])):
        audit.eq(row['method'],method,label+'.method.'+method)
        audit.eq(row['parent_completion'],completion,label+'.original_completion.'+method)
        for key,value in (('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),
            ('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('original_V9_geometry_row_semantic_sha256',endpoint['local_geometry_row_semantic_sha256'])):
            audit.eq(row[key],value,label+'.source_lineage.'+key+'.'+method)
        for key,value in (('input_points',q),('original_native_N3_points',obs['native_N3_points']),('K',parent['K']),
            ('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),('hidden_initial',H),('initial_pose',parent['initial_pose'])):
            audit.eq(row[key],value,label+'.original_'+key+'.'+method)
        audit.eq(row['fresh_filtered_solver_semantic_sha256'],h['semantic'](s),label+'.one_shared_solver_hash.'+method)
        f=row['observation_filter'];audit.eq(f['rejected_edge_ids'],rejected,label+'.rejected_ids.'+method)
        audit.eq(f['retained_unused_edge_ids'],edges,label+'.retained_ids.'+method)
        audit.eq(f['original_observation_semantic_sha256'],h['semantic'](obs),label+'.original_OBS_hash.'+method)
        audit.eq(f['filtered_observation_semantic_sha256'],h['semantic'](filtered),label+'.filtered_OBS_hash.'+method)
        audit.eq(f['endpoint_records'],endpoint['endpoint_records'],label+'.frozen_endpoint_records.'+method)
        audit.eq(f['unknown_retained_edge_ids'],[r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='UNVERIFIED_RETAIN'],label+'.UNKNOWN_retained.'+method)
        audit.eq(row['endpoint_audit_semantic_sha256'],h['semantic'](endpoint),label+'.same_frozen_audit_SHA.'+method)
    audit.eq(s['used'],U,label+'.whole_original_U');audit.eq(s['hidden'],H,label+'.H');audit.eq(s['excluded'],H,label+'.H_excluded')
    audit.eq(s['line_edges'],edges,label+'.filtered_line_pool');audit.eq(s['consumed_edges'],consumed,label+'.no_duplicate_factors')
    for key in ('prior_used','initial_pose_residual_prior','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations'):
        audit.eq(s[key],False,label+'.no_prior_or_refit.'+key)
    audit.eq(s['global_uniqueness_proven'],False,label+'.local_only')
    encoded=json.dumps(dict(points=[q[k] for k in U],point_ids=U,lines=lines),sort_keys=True,separators=(',',':'),allow_nan=True).encode()
    if s.get('factor_pool_binding') is not None:
        audit.eq(s['factor_pool_binding'],dict(sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded),definition='semantic sorted JSON; not file-byte provenance'),label+'.actual_factor_SHA')
    cs=s['all_candidate_solutions'];keys=[]
    for i,c in enumerate(cs):
        audit.eq(c['candidate_index'],i,label+'.candidate_index.'+str(i));keys.append(h['candidate'](c,rank,U,H,lines,audit,counts,label+'.candidate.'+str(i),'stored_candidate'))
    counts['stored_candidate_records']+=len(cs);valid=[i for i,c in enumerate(cs) if c['accepted']]
    best=None
    if valid:
        index=min(valid,key=lambda i:keys[i]);best=cs[index];audit.eq(s['best_candidate'],best,label+'.minimum_fixed_pool_softcost')
        h['candidate'](best,rank,U,H,lines,audit,counts,label+'.chosen','chosen_recheck')
        tied=any(abs(c['soft_l1_cost']-best['soft_l1_cost'])<=h['TIE'] and not h['equivalent'](c,best,rank['xyz']) for c in (cs[i] for i in valid))
        audit.eq(s['unresolved_ambiguity'],tied,label+'.all_local_branch_ties');audit.eq(s['available'],not tied,label+'.rank_only_availability')
        audit.eq(s['state'],'AMBIGUOUS_LOCAL_POINT_LINE' if tied else 'NEW_POSE',label+'.rank_state')
    else:audit.eq(s['available'],False,label+'.no_viable_candidate_unavailable');audit.eq(s['best_candidate'],None,label+'.no_viable_selected_candidate')
    weak={};gate=ledger['observability_gate']
    audit.eq(gate['relative_threshold'],1e-4,label+'.inherited_weak_cutoff');audit.eq(gate['selected_candidate_index'],best['candidate_index'] if best else None,label+'.selected_gate_candidate')
    for name,key in (('observed','observed_normal_joint'),('modeled','modeled_normal_joint')):
        spectrum=best['point_line_geometry'][key]['singular_values'] if best else []
        weak[key]=bool(spectrum and spectrum[-1]<spectrum[0]*1e-4)
        audit.eq(gate[name+'_weak'],weak[key],label+'.'+name+'_weak')
        audit.eq(gate[name+'_singular_values'],spectrum if best else None,label+'.'+name+'_stored_spectrum')
        audit.eq(gate[name+'_witness'],best['point_line_geometry'][key] if best else None,label+'.'+name+'_same_matrix_witness')
    applied=bool(s['available'] and any(weak.values()));safe_solver=safe['solver']
    audit.eq(gate['applied'],applied,label+'.gate_applied_only_to_rank_NEW');audit.eq(gate['rank_only_proposal_available'],s['available'],label+'.gate_rank_availability')
    audit.eq(gate['new_Jacobian_SVD_calls'],0,label+'.gate_no_new_J_or_SVD');audit.eq(gate['no_alternative_reselection'],True,label+'.no_new_choice')
    audit.eq(ledger['full_rank_only_solver'],s,label+'.full_actual_rank_solver_copy')
    audit.eq(ledger['observation_filter'],rank['observation_filter'],label+'.filter_ledger');audit.eq(safe['observation_filter'],rank['observation_filter'],label+'.same_filter_two_views')
    for row in (rank,safe):audit.eq(row['observability_gate'],dict(gate,output_view_applies_gate=row['method']==METHODS[1]),label+'.row_gate.'+row['method'])
    audit.eq(safe_solver['all_candidate_solutions'],cs,label+'.safe_no_second_candidate_pool')
    audit.eq(safe_solver['optimizer_attempts'],s['optimizer_attempts'],label+'.safe_no_second_optimization')
    audit.eq(safe_solver['operation_counts'],s['operation_counts'],label+'.one_actual_work_ledger')
    audit.eq(safe_solver['available'],s['available'] and not applied,label+'.weak_safe_availability')
    audit.eq(safe_solver['state'],'WEAK_OBSERVABILITY' if applied else s['state'],label+'.typed_weak_gate')
    if not applied:audit.eq(safe_solver,s,label+'.safe_same_solver_when_no_gate')
    else:
        fields=('R_cf','R_physical','centroid','cf_extents','projected','points_final','fit_input_ids','fit_line_edges','hidden_reprojected','state','reason')
        audit.eq(safe_solver['rejected_local_pose'],{k:s[k] for k in fields},label+'.declined_pose_preserved')
        for key in ('R_cf','R_physical','centroid','cf_extents','projected'):audit.eq(safe_solver[key],None,label+'.weak_no_accepted_pose.'+key)
        for key in ('fit_input_ids','fit_line_edges','input_inliers','final_inliers','inliers','final_line_inliers'):audit.eq(safe_solver[key],[],label+'.weak_no_accepted_fit.'+key)
        for key in ('point_inlier_count','unique_line_inlier_count','total_inlier_factor_count'):audit.eq(safe_solver[key],0,label+'.weak_no_accepted_count.'+key)
        audit.eq(safe_solver['points_final'],q,label+'.weak_solver_original_sparse_points')
    for solved in (s,safe_solver):
        audit.eq(solved['new_pose_estimated'],solved['available'],label+'.numeric_NEW_consistent')
        audit.eq(solved['pose_available'],solved['available'],label+'.numeric_availability_consistent')
        audit.eq(solved['no_pose'],not solved['available'],label+'.numeric_no_pose_consistent')
    for row in (rank,safe):
        solved=row['solver'];new=solved['available'];fallback=bool(not new and parent['initial_pose'].get('available',False))
        audit.eq(row['new_pose_estimated'],new,label+'.NEW.'+row['method']);audit.eq(row['fallback_used'],fallback,label+'.fallback.'+row['method'])
        audit.eq(row['output_status'],'NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',label+'.operational_status.'+row['method'])
        audit.eq(row['pose_available'],new or fallback,label+'.output_availability.'+row['method']);audit.eq(row['no_pose'],not(new or fallback),label+'.output_no_pose.'+row['method'])
        audit.eq(row['reprojections_reused_as_observations'],False,label+'.no_display_refit.'+row['method'])
        expected=copy.deepcopy(obs['native_N3_points']);sparse=copy.deepcopy(q)
        if new:
            for key in ('R_cf','R_physical','centroid','cf_extents','point_line_geometry','soft_l1_cost','sse_px2','final_inliers','final_line_inliers'):
                audit.near(solved[key],best[key],label+'.selected_actual_pose.'+key+'.'+row['method'])
            _,uv,_,_=h['forward'](solved['cf_extents'],best['final_z'],row['K'])
            for k in range(8):
                if k in H:expected[k]=uv[k];sparse[k]=uv[k]
                elif h['finite'](q[k],(2,)) and q[k]!=[-1,-1]:expected[k]=q[k]
            audit.near(solved['projected'],uv,label+'.final_projection.'+row['method'])
            audit.eq(solved['fit_input_ids'],U,label+'.accepted_fit_only_actual_U.'+row['method'])
            audit.eq(solved['fit_line_edges'],edges,label+'.accepted_actual_filtered_lines.'+row['method'])
            audit.eq(row['actual_pose'],solved,label+'.full_actual_pose_join.'+row['method'])
        else:audit.eq(row['actual_pose'],parent['initial_pose'],label+'.full_original_fallback_pose.'+row['method'])
        audit.near(row['native_points'],expected,label+'.display_H_only_projection.'+row['method'])
        audit.near(solved['points_final'],sparse,label+'.solver_H_only_projection.'+row['method']);audit.eq(solved['hidden_reprojected'],new and bool(H),label+'.solver_H_projection_flag.'+row['method'])
        audit.eq(row['native_points'][8],obs['native_N3_points'][8],label+'.unchanged_center.'+row['method'])
        audit.eq(row['reprojected_ids'],H if new else [],label+'.H_projection_ids.'+row['method']);audit.eq(row['hidden_reprojected'],new and bool(H),label+'.H_projection_flag.'+row['method'])
        counts['output:'+row['method']+':'+row['output_status']]+=1
    audit.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_filtered_solve');audit.eq(ledger['reported_output_views'],2,label+'.two_views')
    audit.eq(ledger['endpoint_validation_requests'],0,label+'.no_new_endpoint_pose_calls');audit.eq(ledger['endpoint_decision_source'],'SEALED_V10_ENDPOINT_DECISION_REPLAY',label+'.frozen_endpoint_replay')
    for key,value in s['operation_counts'].items():counts['local:'+key]+=value
    success_reports=sum(a['actual_optimizer_called'] and a['success'] for a in s['optimizer_attempts']);counts['optimizer_success_reports']+=success_reports
    op=s['operation_counts'];audit.eq(ledger['optimizer_dispatch'],dict(entries=op['local_optimizer_calls'],returns=op['local_optimizer_completions'],
        exceptions=op['local_optimizer_calls']-op['local_optimizer_completions'],success_reports=success_reports,
        residual_entries=op['optimizer_residual_calls'],jacobian_entries=op['optimizer_jacobian_calls']),label+'.actual_optimizer_per_frame')
    for field,prefix in (('primitive_entry_calls','cv:entry:'),('primitive_return_calls','cv:return:'),('primitive_exception_calls','cv:exception:'),('svd_dispatch','svd:')):
        require(set(ledger[field])==set(('entries','returns','exceptions') if field=='svd_dispatch' else h['PRIMITIVES']) and
            all(type(v) is int and v>=0 for v in ledger[field].values()),'invalid actual counter')
        for key,value in ledger[field].items():counts[prefix+key]+=value
    for name in h['PRIMITIVES']:audit.eq(ledger['primitive_entry_calls'][name],ledger['primitive_return_calls'][name]+ledger['primitive_exception_calls'][name],label+'.actual_primitive_complete.'+name)
    for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','cornerSubPix'):audit.eq(ledger['primitive_entry_calls'][name],0,label+'.no_new_point_fit_or_subpix.'+name)
    audit.eq(ledger['svd_dispatch']['entries'],ledger['svd_dispatch']['returns']+ledger['svd_dispatch']['exceptions'],label+'.actual_SVD_complete')
    counts['frames']+=1;counts['weak_gated_frames']+=int(applied)


def run(args):
    output=guard(args,'run');own_path=output/OUTPUTS[0];own=read(own_path);ob=binding(own_path)
    paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()};_module,info=extraction()
    require(own['schema']=='supplemental_endpoint_observability_scalar_protocol_v11' and own['inputs']==inputs and
        own['helpers']==info and own['methods']==list(METHODS) and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    core,seal,receipt,pop=metadata(paths,inputs);write(output/OUTPUTS[1],dict(own_protocol=ob,inputs=inputs,actual_production_calls=0))
    h=helpers(info);audit=h['Audit']();counts=Counter();error=None;began=time.monotonic()
    try:
        cohort=read(paths['cohort']);ordered=cohort['ids'];known=set(ordered)
        require(len(ordered)==len(known)==245 and [f['id'] for f in cohort['frames']]==ordered,'full ordered245 required')
        audit.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'cohort_scope')
        streams=[iter(rows(paths[k])) for k in ('geometry','ledgers','source_audit')]
        parents=iter(primary(paths['source_geometry'],'method','IMAGE_ROLE_BOUNDARY_ONLY',1960,known,audit,'parent'))
        observations=iter(primary(paths['source_observations'],'head_arm','IMAGE_ROLE',735,known,audit,'ROLE'))
        for i,fid in enumerate(ordered):
            group=[next(streams[0],None),next(streams[0],None),next(streams[1],None),next(parents,None),next(observations,None),next(streams[2],None)]
            require(all(isinstance(r,dict) and r['id']==fid and r['session']==cohort['frames'][i]['session'] for r in group),'incomplete/reordered joined rows')
            packet(*group,pop['completion'],h,audit,counts,'frame.'+str(i));del group
        for stream in [*streams,parents,observations]:audit.eq(next(stream,None),None,'no_extra_rows')
        execution=receipt['actual_execution']
        for key in ('frames_started','frames_complete','filtered_solver_entries','filtered_solver_returns','ledger_rows'):audit.eq(execution[key],245,'actual.'+key)
        audit.eq(execution['geometry_rows'],490,'actual.two_views490');audit.eq(execution['filtered_solver_exceptions'],0,'actual.no_incomplete_solver')
        audit.eq(receipt['local_operation_counts'],{k:counts['local:'+k] for k in h['LOCAL_KEYS']},'actual.one_solver_operation_sum')
        expected=dict(entries=counts['local:local_optimizer_calls'],returns=counts['local:local_optimizer_completions'],
            exceptions=counts['local:local_optimizer_calls']-counts['local:local_optimizer_completions'],success_reports=counts['optimizer_success_reports'],
            residual_entries=counts['local:optimizer_residual_calls'],jacobian_entries=counts['local:optimizer_jacobian_calls'])
        audit.eq(receipt['actual_optimizer'],expected,'actual.optimizer_sum')
        for field,prefix in (('actual_OpenCV_entry_calls','cv:entry:'),('actual_OpenCV_return_calls','cv:return:'),('actual_OpenCV_exception_calls','cv:exception:')):
            audit.eq(receipt[field],{k:counts[prefix+k] for k in h['PRIMITIVES']},'actual.'+field)
        audit.eq(receipt['actual_SVD_dispatch'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'actual.SVD_recorded_sum')
        audit.eq(receipt['output_status_counts'],{m:{s:counts['output:'+m+':'+s] for s in ('NEW_POSE','N3_BASELINE_FALLBACK','POSE_FAILURE') if counts['output:'+m+':'+s]} for m in METHODS},'actual.two_view_status_sum')
        for key in ('primitive_and_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):audit.eq(receipt['cleanup'][key],True,'recorded.cleanup.'+key)
        audit.eq(receipt['forbidden_entries'],{},'actual.no_forbidden_entries');audit.eq(receipt['forbidden_asset_reads'],[],'actual.no_forbidden_assets')
        audit.eq(receipt['protection_before'],receipt['protection_after'],'recorded.protection_unchanged')
        for i,snap in enumerate(receipt['resource_snapshots']):
            audit.eq(read(paths['resource:%03d'%i]),snap,'recorded.resource_identity.'+str(i));audit.eq(snap['quiet'],True,'recorded.quiet.'+str(i));audit.eq(snap['temperature_under_80'],True,'recorded.thermal.'+str(i))
        audit.eq(counts['frames'],245,'complete245');audit.eq(counts['stored_candidate_arithmetic_completed'],counts['stored_candidate_records'],'all_candidates')
        audit.eq({k:binding(p) for k,p in paths.items()},inputs,'complete.input_bytes_unchanged');audit.eq(binding(own_path),ob,'complete.own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    passed=error is None and counts['frames']==245 and audit.failure_count==0
    write(output/OUTPUTS[2],dict(schema='supplemental_endpoint_observability_scalar_checks_v11',complete=error is None and counts['frames']==245,
        passed=passed,own_protocol=ob,checker=inputs['checker'],inputs=inputs,limits=LIMITS,checks=audit.checks,failure_count=audit.failure_count,
        failures=audit.failures,exception=error,counts=dict(counts),max_scalar_absolute_difference=audit.max_difference,
        wall_seconds=time.monotonic()-began,new_model_RGB_GT_PnP_optimizer_SVD_calls=0,first_failure_preserved=True,automatic_retry=False))
    print('ENDPOINT_REPAIR_SCALAR','PASS' if passed else 'FAIL',audit.checks,dict(counts),flush=True)
    if not passed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name,default in (('input',DOC),('output',DOC),('protocol',DOC/'PROTOCOL.json'),('parent',V7),('audit',V10),('cohort',COHORT)):
        p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
