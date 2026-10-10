"""Own-frozen scalar audit of new native endpoint validation witnesses.

Standard library only. Borrow unchanged scalar functions from the byte-bound
V7 independent checker through an AST whitelist. No production imports, model,
RGB, GT, PnP, optimizer, ray, Jacobian or SVD calls. Stored V9 local conditioning
is checked by semantic copy, never recomputed. One frame's witnesses at a time.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
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
DOC=REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
PRIVATE=Path('/tmp/pallet-unused-line-endpoint-audit-private-20261011-v10')
V9=REPO/'_docs/experiments/pallet_sparse_local_line_20261010_v9'
V7=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
SCALAR=REPO/'scripts/research/pallet_three_head_observation_20261010_v7/validation_checks.py'
V6_SOLVER=REPO/'scripts/research/pallet_partial_line_heldout_20261010_v6/solver.py'
SCALAR_SHA='815fd15d8596fe6c8c85720e136627dcef8d4cdb22c23e00737b3f507a3fa4a6'
FUNCS=('finite','ids','eligible','dot','project','determinant3','native_hash','dimensions',
       'rotation_physical','equivalent','stored_rank','ledger','candidate','pose_packet')
CONSTANTS=('EDGES','FALSE_FLAGS','ATOL','RTOL','RADIUS')
PRIMITIVES=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
OUTPUTS=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
ATOL,RTOL=1e-7,1e-12
LIMITS=dict(supplemental_after_completed_GT_free_audit=True,
    new_production_import_model_RGB_GT_PnP_optimizer_ray_Jacobian_SVD_calls=0,
    independent_native_candidate_Rt_projection_residual_consensus_RMS=True,
    scalar_helpers_unchanged_AST_extraction=True,stored_SVD_rank_recount_only=True,
    stored_V9_LOCAL_conditioning_and_objective_witnesses_copy_only=True,
    candidate_and_chosen_arithmetic_counted_separately=True,
    threshold_membership_from_validated_stored_RMS=True,
    native_N3_is_exact_original_ROLE_observation_not_final_display=True,
    no_final_observation_filter_or_replacement_pose=True,
    no_accuracy_generalization_or_runtime_claim=True,
    physical_correspondence_ownership_independently_certified=False,
    global_uniqueness_proven=False,independent_hardware_authenticity_claim=False,
    quiet_snapshots_can_miss_transient_competition=True,
    protection_snapshots_recorded_not_independent_historical_hardware_proof=True,
    external_dependency_declarations_bound_by_protocol_not_all_rehashed=True,
    automatic_retry=False)
PARENT_METHODS=('IMAGE_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_BOUNDARY_ONLY',
    'IMAGE_NO_ROLE_BOUNDARY_ONLY','GEOMETRY_ONLY_CORNERWISE_HYBRID',
    'IMAGE_NO_ROLE_CORNERWISE_HYBRID','IMAGE_ROLE_CORNERWISE_HYBRID',
    'N3_INDEPENDENT_ROBUST_H','N3_INDEPENDENT_ROBUST_NO_MASK')
HEADS=('GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE')
TRUTH={'corner','pose','GT','ground_truth','reference_pose','human_states_native',
    'baseline_corner','baseline_pose','mask_audit'}
PHYSICAL={0,2,4,6,8,9,10,11}
CAL={0,2,4,8,9,10,11}
MATH_KEYS=tuple(name+':'+state for name in ('numpy_svd','native_pose_jacobian')
    for state in ('entries','returns','exceptions'))


def require(value,reason):
    if not value:raise ValueError(reason)


def read(path):
    with Path(path).open(encoding='utf-8') as f:return json.load(f)


def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            require(bool(line.strip()),'blank compressed row')
            yield json.loads(line)


def binding(path):
    p=Path(path)
    require(p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),'missing/symlink input '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    q=p.resolve()
    return dict(path=str(q.relative_to(REPO)) if q.is_relative_to(REPO) else p.name,sha256=h.hexdigest(),bytes=p.stat().st_size)


def same(a,b):return isinstance(b,dict) and all(a[k]==b.get(k) for k in ('sha256','bytes'))


def canonical(value):
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):
        return sorted((canonical(v) for v in value),key=lambda v:json.dumps(v,sort_keys=True))
    if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def semantic(value):
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def brief(value):
    if isinstance(value,dict) and len(value)>20:return dict(type='dict',keys=sorted(str(k) for k in value),payload_not_retained=True)
    if isinstance(value,(list,tuple,set)) and len(value)>24:return dict(type=type(value).__name__,length=len(value),payload_not_retained=True)
    if isinstance(value,dict):return {str(k):brief(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [brief(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):return repr(value)
    return value


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(canonical(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


class Audit:
    def __init__(self):self.checks=0;self.failure_count=0;self.failures=[];self.max_difference=0.
    def ok(self,value,label,actual=None,expected=None):
        self.checks+=1
        if not value:
            self.failure_count+=1
            if len(self.failures)<100:self.failures.append(dict(check=label,actual=brief(actual),expected=brief(expected)))
    def eq(self,a,e,label):self.ok(a==e,label,a,e)
    def near(self,a,e,label):
        if isinstance(e,dict):
            good=isinstance(a,dict) and set(a)==set(e);self.ok(good,label+'.keys',a,e)
            if good:
                for k in e:self.near(a[k],e[k],label+'.'+str(k))
        elif isinstance(e,(list,tuple)):
            good=isinstance(a,(list,tuple)) and len(a)==len(e);self.ok(good,label+'.shape',a,e)
            if good:
                for i,(x,y) in enumerate(zip(a,e)):self.near(x,y,label+'.'+str(i))
        elif e is None or isinstance(e,(str,bool)):self.eq(a,e,label)
        else:
            good=type(a) in (int,float) and type(e) in (int,float) and math.isfinite(a) and math.isfinite(e)
            error=abs(a-e) if good else None
            if good:self.max_difference=max(self.max_difference,error)
            self.ok(good and error<=ATOL+RTOL*abs(e),label,a,e)


def extraction():
    require(binding(SCALAR)['sha256']==SCALAR_SHA,'unchanged published V7 scalar source required')
    tree=ast.parse(SCALAR.read_text(encoding='utf-8')); selected=[]; names=set()
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name in FUNCS:selected.append(n);names.add(n.name)
        elif isinstance(n,ast.Assign):
            assigned=set()
            for t in n.targets:
                if isinstance(t,ast.Name):assigned.add(t.id)
                elif isinstance(t,ast.Tuple):assigned.update(x.id for x in t.elts if isinstance(x,ast.Name))
            if assigned and assigned<=set(CONSTANTS):selected.append(n);names.update(assigned)
    require(names==set(FUNCS)|set(CONSTANTS),'exact scalar extraction whitelist required')
    module=ast.Module(body=selected,type_ignores=[])
    audit_tree=ast.parse(Path(__file__).with_name('audit.py').read_text(encoding='utf-8'))
    copied=next(n for n in audit_tree.body if isinstance(n,ast.FunctionDef) and n.name=='stored_local_diagnosis')
    policy_node=next(n.value for n in audit_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    method_node=next(n.value for n in audit_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='METHOD' for t in n.targets))
    require(isinstance(policy_node,ast.Call) and isinstance(policy_node.func,ast.Name) and
        policy_node.func.id=='dict' and not policy_node.args and
        all(k.arg is not None for k in policy_node.keywords) and
        len({k.arg for k in policy_node.keywords})==len(policy_node.keywords),
        'POLICY must be a literal dict keyword call, without arbitrary evaluation')
    policy={k.arg:ast.literal_eval(k.value) for k in policy_node.keywords}
    method=ast.literal_eval(method_node)
    witness=next(n for n in ast.parse(V6_SOLVER.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='_witness_problem')
    hashes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in selected if isinstance(n,ast.FunctionDef)}
    info=dict(source=binding(SCALAR),function_AST_sha256=hashes,
        extraction_AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),
        stored_local_copy_AST_sha256=hashlib.sha256(ast.dump(copied,include_attributes=False).encode()).hexdigest(),
        original_V6_witness_AST_sha256=hashlib.sha256(ast.dump(witness,include_attributes=False).encode()).hexdigest(),
        policy=policy,method=method)
    return module,copied,info


def load_helpers(info):
    module,copied,current=extraction();require(current==info,'frozen scalar helper AST differs')
    namespace=dict(math=math,struct=struct,hashlib=hashlib,require=require)
    exec(compile(module,str(SCALAR)+':scalar_AST_only','exec'),namespace)
    copying=dict(copy=copy)
    exec(compile(ast.Module(body=[copied],type_ignores=[]),str(Path(__file__).with_name('audit.py'))+':stored_copy_only','exec'),copying)
    namespace['stored_local_diagnosis']=copying['stored_local_diagnosis']
    return namespace


def native_problem(s,blocked,h):
    """Independent stdlib form of the frozen V6 recorded-witness predicate."""
    if s.get('available') is not True:return 'VALIDATION_POSE_UNAVAILABLE:'+str(s.get('state','MISSING_STATE'))
    if s.get('state')!='NEW_POSE' or s.get('new_pose_estimated') is not True or s.get('pose_available') is not True or s.get('no_pose') is not False or s.get('fallback_used') is not False or s.get('unresolved_ambiguity') is not False:
        return 'VALIDATION_POSE_STATUS_INCONSISTENT'
    for key in h['FALSE_FLAGS']:
        if s.get(key) is not False:return 'MISSING_OR_INVALID_PRIOR_FREE_WITNESS:'+key
    try:
        if set(s['excluded'])!=set(blocked):return 'VALIDATION_EXCLUSION_WITNESS_MISMATCH'
        U=set(s['used'])
        if len(U)<4 or U&set(blocked):return 'VALIDATION_SCORING_POOL_WITNESS_INVALID'
        for key in ('fit_input_ids','final_inliers','generator_ids'):
            values=set(s[key])
            if len(values)<4 or not values<=U:return 'VALIDATION_SELECTED_ID_WITNESS_INVALID:'+key
        if not s['all_candidate_solutions']:return 'VALIDATION_ALL_CANDIDATE_WITNESS_MISSING'
        for c in s['all_candidate_solutions']:
            for key in ('generator_ids','actual_fit_input_ids','inlier_ids'):
                if not set(c[key])<=U:return 'VALIDATION_ACTIVE_CANDIDATE_USES_BLOCKED_ID:'+key
            if len(c['residuals_used_px'])!=len(s['used']):return 'VALIDATION_CANDIDATE_SCORING_POOL_WITNESS_INVALID'
        if not h['finite'](s['projected'],(8,2)):return 'VALIDATION_PROJECTED_WITNESS_INVALID'
        for key,shape in (('R_cf',(3,3)),('R_physical',(3,3)),('centroid',(3,)),('cf_extents',(3,))):
            if not h['finite'](s[key],shape):return 'VALIDATION_POSE_WITNESS_INVALID:'+key
    except (KeyError,TypeError,ValueError):return 'VALIDATION_POSE_WITNESS_MISSING_OR_MALFORMED'
    return None


def difference(after,before,audit,label):
    audit.eq(set(after),set(before),label+'.keys')
    require(set(after)==set(before),'bank ledger keys differ')
    result={k:after[k]-before[k] for k in after}
    audit.ok(all(type(v) is int and v>=0 for v in result.values()),label+'.nonnegative_actual_delta')
    return result


def record_check(record,row,h,expected_line,before,audit,counts,label):
    edge=record['edge'];ends=list(h['EDGES'][edge]);H=row['initial_H'];blocked=sorted(set(H)|set(ends));s=record['solved']
    for key,value in (('endpoints',ends),('hidden',H),('validator_temporary_excluded',ends),('validation_excluded',blocked),
        ('actual_self_hidden_separate_from_temporary_exclusion',True),('attempted_pose_calls',1),
        ('observed_line_supplied_to_validation_solver',False),('global_cached_generation_may_include_blocked_IDs',True),
        ('active_scoring_generators_refits_exclude_blocked_IDs',True),('physical_ownership_certified',False),
        ('validation_pose_may_be_wrong_consensus',True)):
        audit.eq(record[key],value,label+'.'+key)
    audit.eq(record['bank_before'],before,label+'.shared_bank_before')
    events={s['input_hash']:before} if any(before.values()) else {}
    after=h['pose_packet'](s,row['original_native_N3_points'],row['K'],row['xyz'],row['raw_hw'],H,ends,
        audit,counts,events,label+'.native_point_solver')
    audit.eq(record['native_input_hash'],s['input_hash'],label+'.native_bank_hash')
    audit.eq(record['bank_after'],after,label+'.actual_bank_after')
    change=difference(after,before,audit,label+'.delta')
    audit.eq(record['bank_delta'],change,label+'.bank_delta_join');audit.eq(s['operation_counts'],change,label+'.solver_delta_join')
    source=record['source_line'];audit.eq(source['edge'],edge,label+'.source_edge')
    audit.eq(source['endpoints'],ends,label+'.source_ends');audit.eq(source['raw'],expected_line,label+'.actual_original_line')
    normal,offset=source['normal'],source['offset']
    require(h['finite'](normal,(2,)) and type(offset) in (int,float) and math.isfinite(offset),'saved normalized observed line required')
    norm=math.sqrt(math.fsum(v*v for v in expected_line['normal']));require(norm>0,'original observed normal collapsed')
    audit.near(normal,[v/norm for v in expected_line['normal']],label+'.independent_normalization')
    audit.near(offset,expected_line['offset']/norm,label+'.independent_offset_normalization')
    audit.eq(source['support_query_ids'],expected_line['queries'],label+'.same_line_query_support')
    problem=native_problem(s,blocked,h)
    if problem is not None:
        audit.eq(record['endpoint_RMS_px'],None,label+'.unknown_no_RMS')
        audit.eq(record['signed_endpoint_distances_px'],None,label+'.unknown_no_endpoint_distance')
        decision,reason='UNVERIFIED_RETAIN',problem
    else:
        uv=h['project'](s['cf_extents'],s['R_cf'],s['centroid'],row['K'])
        signed=[]
        for k in ends:
            try:value=math.fsum(x*y for x,y in zip(uv[k],normal))-offset
            except (OverflowError,ValueError):value=None
            signed.append(value if value is not None and math.isfinite(value) else None)
        try:rms=math.sqrt(math.fsum(v*v for v in signed)/2) if None not in signed else None
        except (OverflowError,ValueError):rms=None
        rms=rms if rms is not None and math.isfinite(rms) else None
        audit.near(record['signed_endpoint_distances_px'],signed,label+'.independent_signed_distances')
        audit.near(record['endpoint_RMS_px'],rms,label+'.independent_endpoint_RMS')
        if rms is None:
            decision,reason='UNVERIFIED_RETAIN','VALIDATION_ENDPOINT_RMS_NONFINITE'
            counts['independent_nonfinite_endpoint_RMS']+=1
        else:
            stored=record['endpoint_RMS_px'];require(type(stored) in (int,float) and math.isfinite(stored),'finite available RMS required')
            decision='SUPPORTED_RETAIN' if stored<=8. else 'CONTRADICTED_REJECT'
            reason='AVAILABLE_NATIVE_N3_ENDPOINT_RMS_WITHIN_FIXED_8PX' if stored<=8. else 'AVAILABLE_NATIVE_N3_ENDPOINT_RMS_EXCEEDS_FIXED_8PX'
            counts['independent_checkable_endpoint_RMS']+=1
    audit.eq(record['decision'],decision,label+'.fixed_V6_decision');audit.eq(record['reason'],reason,label+'.typed_reason')
    counts['endpoint_validation_requests']+=1;counts['decision:'+decision]+=1
    return after,change


def source_pool(row,local,parent,obs,h,audit,counts,label):
    q=parent['input_points'];H=parent['hidden_initial']
    U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]
    contract=parent['observation_contract'];adopted=contract['validated_boundary_corner_ids']
    audit.eq(row['actual_sparse_used_ids'],U,label+'.actual_sparse_U')
    audit.eq(local['solver']['used'],U,label+'.LOCAL_same_U')
    audit.eq(row['actual_adopted_boundary_ids'],adopted,label+'.same_actual_adopted_ids')
    audit.ok(set(H).isdisjoint(local['solver']['fit_input_ids']),label+'.LOCAL_H_fit_excluded')
    corners={c['id']:c for c in obs['corners']};audit.eq(len(corners),len(obs['corners']),label+'.unique_source_corners')
    consumed=set()
    for k in adopted:
        if k not in U:continue
        c=corners[k];audit.eq(c['xy'],q[k],label+'.adopted_actual_xy.'+str(k))
        audit.ok(len(c['edges'])==2 and len(set(c['edges']))==2 and
            all(type(e) is int and 0<=e<12 and k in h['EDGES'][e] for e in c['edges']),label+'.incident_edges.'+str(k))
        consumed.update(c['edges'])
    if row['contract_error'] is not None:
        audit.eq(local['solver']['state'],'INVALID_OBSERVATION_CONTRACT',label+'.saved_invalid_contract')
        audit.eq(row['contract_error']['message'],local['solver']['reason'],label+'.same_typed_contract_error')
        audit.eq(row['endpoint_records'],[],label+'.invalid_contract_no_validation')
        audit.eq(row['original_actual_unused_edges'],[],label+'.invalid_contract_no_pool')
        counts['saved_invalid_factor_contracts_copy_only']+=1
        return {}
    audit.eq(row['consumed_edges'],sorted(consumed),label+'.independent_consumed_incident_edges')
    audit.eq(local['solver']['consumed_edges'],sorted(consumed),label+'.consumed_LOCAL_join')
    audit.eq(row['original_line_contract_checks'],local['solver']['line_contract_checks'],label+'.saved_line_checks_join')
    saved_checks=iter(row['original_line_contract_checks']);seen={};actual={}
    for raw in obs['lines']:
        e=raw['edge'];check=next(saved_checks,None);require(isinstance(check,dict),'missing original line contract check')
        require(type(e) is int and 0<=e<12,'invalid original semantic edge ID')
        audit.eq(check['edge'],e,label+'.line_check_edge')
        identity=json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False)
        if e in seen:
            audit.eq(identity,seen[e],label+'.consistent_duplicate_identity')
            audit.eq(check,dict(edge=e,accepted=False,reason='CONSISTENT_DUPLICATE_COLLAPSED'),label+'.one_factor_duplicate_collapse')
            continue
        seen[e]=identity;audit.eq(raw['endpoints'],list(h['EDGES'][e]),label+'.source_physical_endpoint_ids')
        if e not in PHYSICAL:
            audit.ok(e not in consumed,label+'.unphysical_edge_not_consumed')
            audit.eq(check,dict(edge=e,accepted=False,reason='SOURCE_REGISTRY_UNSUPPORTED_EDGE'),label+'.source_physical_registry')
            continue
        normal=raw['normal'];support=raw['support_points'];radii=raw['query_radii_px'];queries=raw['queries']
        require(h['finite'](normal,(2,)) and all(h['finite'](p,(2,)) for p in support) and
            len(radii)==len(support)==len(queries) and len(set(queries))>=3 and
            all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in radii),
            'original physical line support contract malformed')
        length=math.sqrt(math.fsum(v*v for v in normal));require(length>0,'collapsed original physical line normal')
        n=[v/length for v in normal];offset=raw['offset']/length
        errors=[abs(math.fsum(x*y for x,y in zip(p,n))-offset) for p in support]
        audit.near(check['absolute_support_residuals_px'],errors,label+'.independent_source_support_residuals.'+str(e))
        audit.eq(check['support_query_ids'],queries,label+'.source_query_ID_join.'+str(e))
        audit.eq(check['query_radii_px'],radii,label+'.source_CAL_radius_join.'+str(e))
        audit.eq(check['tolerance_px'],1e-9,label+'.unchanged_source_consensus_tolerance.'+str(e))
        for j,point in zip(queries,support):
            require(type(j) is int and 0<=j<84 and j//7==e,'source line query has wrong semantic edge')
            query=obs['queries'][j]
            audit.eq(query['edge'],e,label+'.query_edge.'+str(j))
            audit.eq(query['calibrated_model_coverage'],True,label+'.source_CAL_supported_query.'+str(j))
            audit.eq(query['selected_xy'],point,label+'.actual_observed_query_location.'+str(j))
            audit.eq(query['radius_px'],radii[queries.index(j)],label+'.actual_query_radius.'+str(j))
        # Membership uses numerically checked saved residuals, matching the
        # producer exactly at its 1e-9 support boundary.
        good=math.isfinite(offset) and all(v<=r+1e-9 for v,r in zip(check['absolute_support_residuals_px'],radii))
        reason='CONSUMED_BY_ACTUALLY_ADOPTED_BOUNDARY' if e in consumed else 'accepted' if good else 'FINAL_LINE_CONSENSUS_INCONSISTENT'
        audit.eq(check['accepted'],good and e not in consumed,label+'.fixed_source_line_membership.'+str(e))
        audit.eq(check['reason'],reason,label+'.fixed_source_line_reason.'+str(e))
        if e in consumed:audit.eq(good,True,label+'.consumed_source_line_consistent.'+str(e))
        if good and e not in consumed:
            audit.ok(e in CAL,label+'.actual_line_has_original_CAL_coverage.'+str(e));actual[e]=raw
    audit.eq(next(saved_checks,None),None,label+'.no_extra_line_checks')
    audit.ok(consumed<=set(seen),label+'.all_consumed_source_lines_present')
    expected=sorted(actual)
    audit.eq(row['original_actual_unused_edges'],expected,label+'.independent_original_unused_pool')
    audit.eq(local['solver']['line_edges'],expected,label+'.exact_LOCAL_line_pool')
    audit.ok(not consumed&set(expected),label+'.no_point_line_double_count')
    # The line coefficients hashed here are preserved producer witnesses,
    # after independent normalization checks in record_check. A fresh libm
    # normalization is not claimed byte-identical to NumPy's normalization.
    records=row['endpoint_records'];audit.eq([r['edge'] for r in records],expected,label+'.one_request_per_actual_unused_edge')
    pool=dict(points=[q[k] for k in U],point_ids=U,lines=[dict(edge=r['edge'],normal=r['source_line']['normal'],
        offset=r['source_line']['offset'],support_query_ids=r['source_line']['support_query_ids']) for r in records])
    encoded=json.dumps(pool,sort_keys=True,separators=(',',':'),allow_nan=True).encode()
    expected_binding=dict(sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded),definition='semantic sorted JSON; not file-byte provenance')
    audit.eq(row['factor_pool_binding'],expected_binding,label+'.semantic_factor_binding_from_validated_witnesses')
    audit.eq(local['solver']['factor_pool_binding'],expected_binding,label+'.same_LOCAL_factor_binding')
    counts['source_actual_unused_lines']+=len(actual);counts['source_consumed_edges']+=len(consumed)
    return actual


def counter_packet(value,keys,audit,label,counts,prefix):
    audit.eq(set(value),set(keys),label+'.exact_fields')
    require(set(value)==set(keys) and all(type(v) is int and v>=0 for v in value.values()),'invalid recorded counter packet '+label)
    for key in keys:counts[prefix+key]+=value[key]


def packet(row,ledger,local,old_ledger,parent,obs,completion,h,audit,counts,label):
    audit.eq(row['schema'],'frozen_sparse_ROLE_unused_line_endpoint_diagnosis_row_v10',label+'.row_schema')
    audit.eq(ledger['schema'],'frozen_sparse_ROLE_unused_line_endpoint_diagnosis_ledger_v10',label+'.ledger_schema')
    audit.eq(row['method'],h['method'],label+'.fixed_method');audit.eq(row['policy'],h['policy'],label+'.fixed_policy')
    source=dict(parent_row=parent,role_observation=obs,native_N3_points=obs['native_N3_points'],parent_completion=completion)
    whole=dict(local_packet=source,local_row=local)
    for key,value in (('local_geometry_row_semantic_sha256',semantic(local)),('parent_geometry_row_semantic_sha256',semantic(parent)),
        ('original_ROLE_observation_semantic_sha256',semantic(obs)),('original_native_N3_semantic_sha256',semantic(obs['native_N3_points']))):
        audit.eq(row[key],value,label+'.'+key)
    for key,value in (('parent_geometry_row_semantic_sha256',semantic(parent)),('parent_observation_row_semantic_sha256',semantic(obs)),
        ('parent_native_N3_semantic_sha256',semantic(obs['native_N3_points']))):audit.eq(local[key],value,label+'.LOCAL_source_'+key)
    audit.eq(old_ledger['parent_packet_semantic_sha256'],semantic(source),label+'.original_LOCAL_packet_binding')
    audit.eq(ledger['original_V9_ledger_semantic_sha256'],semantic(old_ledger),label+'.original_LOCAL_ledger_binding')
    audit.eq(ledger['input_packet_semantic_sha256'],semantic(whole),label+'.actual_input_packet_binding')
    for key,value in (('K',parent['K']),('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),
        ('original_native_N3_points',obs['native_N3_points']),('original_sparse_points',parent['input_points']),('initial_H',parent['hidden_initial'])):
        audit.eq(row[key],value,label+'.exact_original_'+key)
    for key,value in (('input_points',parent['input_points']),('original_native_N3_points',obs['native_N3_points']),
        ('K',parent['K']),('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),('hidden_initial',parent['hidden_initial'])):
        audit.eq(local[key],value,label+'.LOCAL_exact_original_'+key)
    audit.eq(obs['head_arm'],'IMAGE_ROLE',label+'.original_ROLE_head');audit.eq(obs['GT_input'],False,label+'.original_ROLE_GT_free')
    audit.eq(obs['predicted_N3_hidden'],parent['hidden_initial'],label+'.original_ROLE_H_identity')
    audit.eq(obs['initial_N3_pose'],parent['initial_pose'],label+'.original_ROLE_initial_pose_identity')
    audit.eq(local['solver']['solver'],'LOCAL_POINT_LINE',label+'.LOCAL_solver_identity')
    audit.eq(local['solver']['initial_pose_residual_prior'],False,label+'.LOCAL_initial_seed_no_residual_prior')
    audit.eq(local['solver']['global_uniqueness_proven'],False,label+'.LOCAL_no_global_uniqueness')
    audit.eq(local['solver']['reprojected_points_reused_as_observations'],False,label+'.LOCAL_H_projection_not_refit')
    actual=source_pool(row,local,parent,obs,h,audit,counts,label+'.source_pool')
    audit.eq(row['stored_local_diagnosis'],h['stored_local_diagnosis'](local['solver']),label+'.immutable_LOCAL_conditioning_copy')
    before=row['native_bank_before'];audit.ok(all(type(v) is int and v==0 for v in before.values()),label+'.fresh_native_bank_zero')
    expected_hash=h['native_hash'](obs['native_N3_points'],row['K'],row['xyz'],row['raw_hw'])
    if expected_hash is not None:audit.eq(ledger['native_coordinate_bank_hash'],expected_hash,label+'.native_numeric_bank_hash')
    else:counts['nonfinite_native_bank_hash_not_reconstructible']+=1
    previous=before;available=0;decisions=Counter()
    for i,r in enumerate(row['endpoint_records']):
        previous,_change=record_check(r,row,h,actual[r['edge']],previous,audit,counts,label+'.edge.'+str(i))
        available+=int(r['solved']['available'] is True);decisions[r['decision']]+=1
        audit.eq(r['native_input_hash'],ledger['native_coordinate_bank_hash'],label+'.one_shared_native_hash.'+str(i))
    audit.eq(row['native_bank_after'],previous,label+'.final_shared_bank_after')
    delta=difference(previous,before,audit,label+'.whole_bank_delta')
    audit.eq(row['native_bank_delta'],delta,label+'.row_bank_delta')
    for k in delta:counts['bank:'+k]+=delta[k]
    for key,value in (('native_bank_before',before),('native_bank_after',previous),('native_bank_delta',delta),
        ('endpoint_validation_requests',len(row['endpoint_records'])),('available_validation_poses',available),
        ('checkable_validation_poses',decisions['SUPPORTED_RETAIN']+decisions['CONTRADICTED_REJECT']),
        ('unverified_validation_poses',decisions['UNVERIFIED_RETAIN']),('native_coordinate_bank_count',1)):
        audit.eq(ledger[key],value,label+'.ledger_'+key)
    for decision,key in (('CONTRADICTED_REJECT','diagnostic_contradicted_edges'),('UNVERIFIED_RETAIN','diagnostic_unknown_edges'),('SUPPORTED_RETAIN','diagnostic_supported_edges')):
        audit.eq(row[key],[r['edge'] for r in row['endpoint_records'] if r['decision']==decision],label+'.same_'+key)
    for key,value in (('original_input_packet_unchanged',True),('no_final_observation_filter_applied',True),
        ('new_final_pose_computed',False),('new_LOCAL_optimizer_calls',0),('new_GT_scores',0),('new_model_image_calls',0),
        ('original_V9_pose_and_status_preserved',True),('GT_free',True),('physical_ownership_truth','UNKNOWN'),
        ('classification_mismatch_is_not_pose_failure',True)):
        audit.eq(row[key],value,label+'.row_'+key)
    for key,value in (('input_packet_unchanged',True),('new_final_pose_paths',0),('new_LOCAL_optimizer_calls',0),
        ('new_GT_scores',0),('new_detector_N3_head_decode_calls',0),('new_local_conditioning_Jacobian_or_SVD_calls',0),
        ('native_point_validation_may_compute_Jacobian_and_rank',True),('native_point_validation_SVD_entries_not_instrumented_here',True),
        ('candidate_conditioning_fields_copied_without_recompute',True),('endpoint_validation_hypotheses_are_actual_new_point_PnP_calls',True),
        ('latency_or_accuracy_improvement_claim',False)):
        audit.eq(ledger[key],value,label+'.ledger_'+key)
    for field in ('factor_validator_pose_or_J_operations','factor_validator_native_solver_operations'):
        audit.ok(bool(ledger[field]) and all(type(v) is int and v==0 for v in ledger[field].values()),label+'.factor_validator_zero_'+field)
    for field,prefix in (('primitive_entry_calls','primitive:entries:'),('primitive_return_calls','primitive:returns:'),('primitive_exception_calls','primitive:exceptions:')):
        counter_packet(ledger[field],PRIMITIVES,audit,label+'.'+field,counts,prefix)
    for name in PRIMITIVES:
        audit.eq(ledger['primitive_entry_calls'][name],ledger['primitive_return_calls'][name]+ledger['primitive_exception_calls'][name],label+'.primitive_complete_dispatch.'+name)
    for name in ('solvePnP','cornerSubPix'):audit.eq(ledger['primitive_entry_calls'][name],0,label+'.forbidden_primitive_zero.'+name)
    audit.eq(ledger['primitive_entry_calls']['solvePnPGeneric'],delta['generic_calls'],label+'.actual_Generic_vs_bank')
    audit.eq(ledger['primitive_entry_calls']['solvePnPRefineLM'],delta['lm_calls'],label+'.actual_LM_vs_bank')
    dispatch=ledger['native_validation_dispatch'];counter_packet(dispatch,('entries','returns','exceptions'),audit,label+'.native_dispatch',counts,'native_dispatch:')
    audit.eq(dispatch['entries'],len(row['endpoint_records']),label+'.actual_requests_vs_records')
    audit.eq(dispatch['returns'],dispatch['entries'],label+'.all_requested_validation_returned')
    audit.eq(dispatch['exceptions'],0,label+'.no_incomplete_validation_dispatch')
    counter_packet(ledger['actual_native_math_dispatch'],MATH_KEYS,audit,label+'.native_math',counts,'native_math:')
    for name in ('numpy_svd','native_pose_jacobian'):
        values=ledger['actual_native_math_dispatch']
        audit.eq(values[name+':entries'],values[name+':returns']+values[name+':exceptions'],label+'.complete_math_dispatch.'+name)
    counts['audit_frames']+=1;counts['native_coordinate_banks']+=1


def input_paths(args):
    folder,parent,local=args.input.resolve(),args.source_parent.resolve(),args.v9_input.resolve()
    paths=dict(audit_rows=folder/'AUDIT_ROWS.jsonl.gz',audit_ledgers=folder/'AUDIT_LEDGERS.jsonl.gz',
        audit_protocol=args.protocol.resolve(),audit_seal=folder/'AUDIT_SEAL.json',audit_receipt=folder/'AUDIT_RECEIPT.json',
        parent_population_checks=folder/'PARENT_POPULATION_CHECKS.json',checker=Path(__file__).resolve(),
        audit_started=folder/'AUDIT_STARTED.json',preflight=folder/'AUDIT_PREFLIGHT.json',
        protection=DOC/'PROTECTION_BEFORE.json',cohort=args.cohort.resolve(),
        scalar_source=SCALAR,original_V6_solver=V6_SOLVER,audit_code=Path(__file__).with_name('audit.py'),
        runner_code=Path(__file__).with_name('runner.py'))
    for name in ('PROTOCOL.json','GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','GEOMETRY_SEAL.json',
        'CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json'):
        paths['v9:'+name]=local/name
    for name in ('PROTOCOL.json','GEOMETRY_SEALED.jsonl.gz','OBSERVATIONS.jsonl.gz','GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json'):
        paths['parent:'+name]=parent/name
    receipt=read(paths['audit_receipt'])
    for i,_ in enumerate(receipt['resource_snapshots']):paths['resource:%03d'%i]=folder/('AUDIT_RESOURCE_%03d.json'%i)
    # Original declarations are themselves byte-bound by AUDIT_PROTOCOL. All
    # resolvable public declarations (including unused scoring files) are
    # rehashed without decoding them. External basename declarations are not
    # guessed: the recorded driver protection is checked separately.
    core=read(paths['audit_protocol'])
    for key,expected in core['inputs'].items():
        p=Path(expected['path'])
        if not p.is_absolute() and (REPO/p).is_file():paths['declared_public:'+key]=REPO/p
    return paths


def core_bindings(inputs,core):
    aliases=dict(checker='validation_checks.py',audit_code='audit.py',runner_code='runner.py',
        protection='protection',cohort='cohort',scalar_source='original_code:pallet_three_head_observation_20261010_v7/validation_checks.py',
        original_V6_solver='original_code:pallet_partial_line_heldout_20261010_v6/solver.py',
        **{'v9:PROTOCOL.json':'v9_protocol','v9:GEOMETRY_SEALED.jsonl.gz':'v9_geometry',
        'v9:CONTROL_LEDGERS.jsonl.gz':'v9_ledgers','v9:GEOMETRY_SEAL.json':'v9_seal',
        'v9:CONTROL_RECEIPT.json':'v9_receipt','v9:PARENT_POPULATION_CHECKS.json':'v9_population',
        'v9:VALIDATION_PROTOCOL.json':'v9_validation_protocol','v9:VALIDATION_CHECKS.json':'v9_validation',
        'parent:PROTOCOL.json':'reused_v9:parent:PROTOCOL.json',
        'parent:GEOMETRY_SEALED.jsonl.gz':'reused_v9:parent:GEOMETRY_SEALED.jsonl.gz',
        'parent:OBSERVATIONS.jsonl.gz':'reused_v9:parent:OBSERVATIONS.jsonl.gz',
        'parent:GEOMETRY_SEAL.json':'reused_v9:parent:GEOMETRY_SEAL.json',
        'parent:INFERENCE_RECEIPT.json':'reused_v9:parent:INFERENCE_RECEIPT.json'})
    aliases.update({key:key[len('declared_public:'):] for key in inputs if key.startswith('declared_public:')})
    for key,source in aliases.items():require(same(inputs[key],core['inputs'][source]),'original audit input binding differs: '+key)


def bound_receipts(paths,inputs,extracted):
    core,seal,receipt,population=[read(paths[k]) for k in ('audit_protocol','audit_seal','audit_receipt','parent_population_checks')]
    require(core['schema']=='fixed_frozen_unused_line_endpoint_audit_protocol_v10' and core['frames']==245 and
        core['method']==extracted['method'] and core['policy']==extracted['policy'],'fixed original diagnosis scope differs')
    core_bindings(inputs,core)
    require(seal['schema']=='frozen_unused_line_endpoint_audit_seal_v10' and seal['complete'] is True and
        seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and
        seal['cleanup_completed_before_seal'] is True,'complete GT-free245 audit seal required')
    require(receipt['schema']=='frozen_unused_line_endpoint_audit_receipt_v10' and receipt['complete'] is True and
        receipt['error'] is None and receipt['cleanup_error'] is None and receipt['cleanup_errors']==[] and
        receipt['environment_and_monkeypatch_cleanup_completed'] is True,'completed driver cleanup required')
    require(population['schema']=='full_GT_free_unused_line_audit_parent_population_checks_v10' and
        population['complete'] is True and population['passed'] is True and population['frames']==245 and
        population['v9_geometry_rows']==population['v9_ledger_rows']==245,'full parent population required')
    for key,own in (('protocol','audit_protocol'),('audit','audit_rows'),('ledgers','audit_ledgers'),
        ('receipt','audit_receipt'),('parent_population_checks','parent_population_checks')):
        require(same(seal[key],inputs[own]),'sealed diagnosis binding differs: '+key)
    require(same(receipt['protocol'],inputs['audit_protocol']) and
        same(receipt['parent_population_checks'],inputs['parent_population_checks']) and
        same(core['parent_population_checks'],inputs['parent_population_checks']),'protocol/proof/receipt join differs')
    for key,own in (('audit','audit_rows'),('ledgers','audit_ledgers')):
        require(same(receipt['raw_rows'][key],inputs[own]),'receipt completed stream binding differs: '+key)
    old_seal,old_receipt,old_checks,old_own=[read(paths['v9:'+n]) for n in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json')]
    require(old_seal['complete'] is True and old_seal['GT_read_allowed'] is False and
        old_seal['frames']==old_seal['rows']==old_seal['ledger_rows']==245 and
        old_seal['cleanup_completed_before_seal'] is True,'complete original V9 geometry required')
    require(old_receipt['complete'] is True and old_receipt['error'] is None and old_receipt['cleanup_error'] is None and
        old_receipt['environment_and_monkeypatch_cleanup_completed'] is True,'original V9 cleanup incomplete')
    require(old_checks['complete'] is True and old_checks['passed'] is True and
        old_checks['inputs']==old_own['inputs'] and same(old_checks['own_protocol'],inputs['v9:VALIDATION_PROTOCOL.json']),
        'original independent V9 scalar PASS authority differs')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        require(same(old_seal[key],inputs['v9:'+name]),'original V9 seal differs: '+key)
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('accuracy_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        require(same(old_checks['inputs'][key],inputs['v9:'+name]),'original checked V9 bytes differ: '+key)
    require(same(binding(REPO/old_checks['checker']['path']),old_checks['checker']),'original V9 scalar source differs')
    parent_seal,parent_receipt=[read(paths['parent:'+n]) for n in ('GEOMETRY_SEAL.json','INFERENCE_RECEIPT.json')]
    require(parent_seal['complete'] is True and parent_seal['GT_read_allowed'] is False and
        parent_seal['frames']==245 and parent_seal['rows']==1960 and parent_seal['fixed_rows']==490 and
        parent_receipt['complete'] is True and parent_receipt['inference_error'] is None and parent_receipt['cleanup_error'] is None,
        'complete GT-free V7 source with successful cleanup required')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('observations','OBSERVATIONS.jsonl.gz')):
        require(same(parent_seal[key],inputs['parent:'+name]),'original V7 source seal differs: '+key)
    return core,seal,receipt,population


def guard(args,stage):
    for p in (args.input,args.output,args.protocol,args.v9_input,args.source_parent,args.cohort):
        p=p.absolute();require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink input/output ancestry')
    output=args.output.resolve()
    require(output==DOC.resolve() or output.is_relative_to(PRIVATE.resolve()),'only exclusive new V10 DOC/private output')
    require(all(p.is_dir() for p in (args.input,args.output,args.v9_input,args.source_parent)),'existing input/output directories required')
    require(not output.is_relative_to(args.v9_input.resolve()) and not output.is_relative_to(args.source_parent.resolve()),'output overlaps immutable parent')
    for name in OUTPUTS if stage=='freeze' else OUTPUTS[1:]:require(not (output/name).exists(),'preserve '+name)
    if stage=='run':require((output/OUTPUTS[0]).is_file(),'own freeze before arithmetic required')
    return output


def freeze(args):
    output=guard(args,'freeze');paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()}
    _module,_copy,extracted=extraction();bound_receipts(paths,inputs,extracted)
    write_new(output/OUTPUTS[0],dict(schema='supplemental_unused_line_endpoint_scalar_protocol_v10',
        inputs=inputs,checker=inputs['checker'],frames=245,rows=245,ledger_rows=245,
        scalar_extraction=extracted,limits=LIMITS,scalar_absolute_tolerance=ATOL,scalar_relative_tolerance=RTOL,
        endpoint_RMS_threshold_px=8.,own_freeze_after_complete_diagnosis_before_own_arithmetic=True))
    print('UNUSED_LINE_SCALAR_FROZEN',binding(output/OUTPUTS[0])['sha256'],flush=True)


def population_stream(path,field,names,primary,total,known,audit,label):
    seen=set();n=Counter()
    for row in rows(path):
        key=(row[field],row['id'])
        require(row[field] in names and row['id'] in known and key not in seen,'unknown/duplicate source population '+label)
        audit.ok(not TRUTH.intersection(row),label+'.GT_free.'+row[field]);seen.add(key);n[row[field]]+=1
        if row[field]==primary:yield row
    audit.eq(len(seen),total,label+'.full_stream_rows')
    for name in names:
        audit.eq(n[name],245,label+'.each245.'+name)
        audit.eq({i for arm,i in seen if arm==name},known,label+'.same_cohort.'+name)


def recorded_receipts(paths,inputs,core,seal,receipt,population,counts,audit):
    for key in ('new_final_pose_paths','new_LOCAL_optimizer_calls','new_GT_scores','new_training_RGB'):
        audit.eq(core[key],0,'scope.protocol.'+key)
    for key,value in (('existing_V9_accuracy_known',True),('evaluation_is_unseen',False),('no_accuracy_or_runtime_claim',True),
        ('no_automatic_retry',True),('own_freeze_after_V9_publication_before_own_validation_PnP',True)):
        audit.eq(core[key],value,'scope.protocol.'+key)
    for key in ('new_final_pose_paths','new_GT_scores'):audit.eq(seal[key],0,'scope.seal.'+key)
    audit.eq(receipt['new_model_RGB_LOCAL_final_pose_GT_score_calls'],0,'scope.no_new_model_LOCAL_GT')
    audit.eq(receipt['GT_read_allowed'],False,'scope.no_GT_decode');audit.eq(receipt['replay_wall_is_deployment_latency'],False,'scope.not_latency')
    audit.eq(receipt['automatic_retry'],False,'scope.no_auto_retry');audit.eq(receipt['forbidden_entries_or_asset_reads'],{},'scope.canaries_empty')
    audit.ok(type(receipt['elapsed_seconds']) in (int,float) and math.isfinite(receipt['elapsed_seconds']) and receipt['elapsed_seconds']>0,'driver.elapsed_record_positive')
    execution=receipt['execution']
    for key in ('frames_started','frames_complete','audit_rows','ledger_rows'):audit.eq(execution[key],245,'driver.complete245.'+key)
    for key in ('endpoint_validation_requests','endpoint_validation_records_complete'):
        audit.eq(execution[key],counts['endpoint_validation_requests'],'driver.actual_vs_all_records.'+key)
    audit.eq(receipt['new_validation_point_PnP_requests'],counts['endpoint_validation_requests'],'driver.native_requests')
    audit.eq(receipt['decision_counts'],{k:counts['decision:'+k] for k in ('SUPPORTED_RETAIN','CONTRADICTED_REJECT','UNVERIFIED_RETAIN')},'driver.all_record_decisions')
    audit.eq(receipt['native_bank_operation_counts'],{k[len('bank:'):]:v for k,v in counts.items() if k.startswith('bank:')},'driver.full_native_bank_sum')
    for field,prefix in (('actual_OpenCV_entry_calls','primitive:entries:'),('actual_OpenCV_return_calls','primitive:returns:'),('actual_OpenCV_exception_calls','primitive:exceptions:')):
        audit.eq(receipt[field],{k:counts[prefix+k] for k in PRIMITIVES},'driver.whole_actual_counter_sum.'+field)
    audit.eq(receipt['actual_OpenCV_primitive_names'],list(PRIMITIVES),'driver.full_six_primitive_names')
    audit.eq(receipt['actual_native_validation_dispatch'],{k:counts['native_dispatch:'+k] for k in ('entries','returns','exceptions')},'driver.actual_native_dispatch_sum')
    audit.eq(receipt['actual_native_math_dispatch'],{k:counts['native_math:'+k] for k in MATH_KEYS},'driver.actual_new_native_J_SVD_dispatch_sum')
    for key in ('numerical_scope_started','numerical_scope_restored','native_math_dispatch_restored','primitive_scope_started',
        'primitive_scope_restored','opencv_thread_count_restored','split_environment_restored'):
        audit.eq(receipt['primitive_restoration_checks'][key],True,'driver.cleanup.'+key)
    audit.eq(receipt['library_environment']['opencv_threads'],1,'driver.recorded_single_OpenCV_thread')
    audit.eq(receipt['library_environment']['actual_primitive_dispatch_instrumentation'],True,'driver.recorded_primitive_instrumentation')
    audit.eq(receipt['explicit_split_environment'],dict(PALLET_SOURCE_ROOT=core['source_root'],
        PALLET_BASELINE_ROOT=core['baseline_root']),'driver.recorded_explicit_split_roots')
    audit.eq(receipt['protection_before'],receipt['protection_after'],'driver.recorded_protection_unchanged')
    audit.eq(receipt['protection_before'],core['protected_before'],'driver.protection_protocol_join')
    snapshot=read(paths['protection']);audit.eq(receipt['protection_before']['published_parent_HEAD'],snapshot['tracked']['original_research_head'],'driver.current_published_parent_HEAD')
    for key in ('own_latest_HEAD_snapshot','published_V7_V8_V9_protected'):
        audit.eq(receipt['protection_before'][key],True,'driver.recorded_protection.'+key)
    protector=receipt['protection_before']['unchanged_original_verifier']
    for key in ('passed','source_checkout_unchanged'):audit.eq(protector[key],True,'driver.recorded_original_verifier.'+key)
    audit.eq(protector['changed_protected_paths'],[],'driver.no_changed_protected_paths')
    for name in ('audit','ledgers'):
        audit.eq(receipt['stream_counts'][name],245,'driver.stream_count.'+name)
        value=receipt['stream_diagnostics'][name]
        audit.eq(value['rows'],245,'driver.stream_rows.'+name);audit.eq(value['published'],True,'driver.stream_published.'+name)
        audit.eq(value['close_errors'],[],'driver.stream_fsync_close.'+name);audit.eq(value['preservation_errors'],[],'driver.stream_preservation.'+name)
    for key,value in (('full_parent_population_rows_retained',0),('one_frame_parent_ROLE_point_control_local_candidate_rows_kept',True),
        ('GT_scored_or_human_inputs_decoded',False),('new_model_image_PnP_LOCAL_GT_calls',0),('reused_original_schema_retained',True)):
        audit.eq(population[key],value,'driver.parent_proof.'+key)
    audit.eq(population['completion'],core['parent_completion'],'driver.original_parent_completion')
    audit.eq(population['ordered_ids'],read(paths['cohort'])['ids'],'driver.population_order')
    started=read(paths['audit_started']);audit.eq(started['status'],'STARTED','driver.started_status')
    audit.ok(same(started['protocol'],inputs['audit_protocol']),'driver.started_protocol')
    audit.eq(started['configured_frames'],245,'driver.started245');audit.eq(started['new_model_LOCAL_GT_calls'],0,'driver.started_no_model_LOCAL_GT')
    pre=read(paths['preflight']);audit.eq(pre['passed'],True,'driver.preflight_PASS')
    audit.ok(same(pre['protocol'],inputs['audit_protocol']),'driver.preflight_protocol')
    audit.eq(pre['actual_validation_PnP_requests'],0,'driver.independent_preflight_before_PnP')
    audit.eq(pre['resource_snapshot']['quiet'],True,'driver.preflight_quiet');audit.eq(pre['resource_snapshot']['temperature_under_80'],True,'driver.preflight_thermal')
    require(len(receipt['resource_snapshots'])>=2,'before and after resource snapshots required')
    audit.eq(receipt['resource_snapshots'][0]['phase'],'before_validation','driver.before_snapshot')
    audit.eq(receipt['resource_snapshots'][-1]['phase'],'after_validation','driver.after_snapshot')
    previous=0
    for i,declared in enumerate(receipt['resource_snapshots']):
        snap=read(paths['resource:%03d'%i]);audit.eq(snap,declared,'driver.resource_identity.'+str(i))
        audit.eq(snap['quiet'],True,'driver.recorded_quiet.'+str(i));audit.eq(snap['temperature_under_80'],True,'driver.recorded_thermal.'+str(i))
        audit.ok(snap['phase'] in ('before_validation','during_validation','after_validation'),'driver.resource_phase.'+str(i))
        audit.ok(type(snap['frames_complete']) is int and previous<=snap['frames_complete']<=245,'driver.resource_progress.'+str(i));previous=snap['frames_complete']
    audit.eq(previous,245,'driver.final_resource245')


def run(args):
    output=guard(args,'run');own_path=output/OUTPUTS[0];own=read(own_path);own_binding=binding(own_path)
    paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()};_module,_copy,extracted=extraction()
    require(own['schema']=='supplemental_unused_line_endpoint_scalar_protocol_v10' and own['inputs']==inputs and
        own['checker']==inputs['checker'] and own['scalar_extraction']==extracted and own['limits']==LIMITS and
        own['frames']==own['rows']==own['ledger_rows']==245 and own['scalar_absolute_tolerance']==ATOL and
        own['scalar_relative_tolerance']==RTOL and own['endpoint_RMS_threshold_px']==8. and
        own['own_freeze_after_complete_diagnosis_before_own_arithmetic'] is True,'own frozen checker/input/scope differs')
    core,seal,receipt,population=bound_receipts(paths,inputs,extracted)
    write_new(output/OUTPUTS[1],dict(schema='supplemental_unused_line_endpoint_scalar_started_v10',
        own_protocol=own_binding,inputs=inputs,checker=inputs['checker'],new_production_calls=0,automatic_retry=False))
    audit,counts,failure=Audit(),Counter(),None;began=time.monotonic()
    try:
        h=load_helpers(extracted);h.update(policy=extracted['policy'],method=extracted['method'])
        cohort=read(paths['cohort']);ordered=cohort['ids'];known=set(ordered)
        audit.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'cohort.easy_medium245')
        require(len(ordered)==len(known)==245 and [f['id'] for f in cohort['frames']]==ordered,'cohort order/population differs')
        audit.ok(all(f['label'] in ('clean','moderate') for f in cohort['frames']),'cohort.no_severe')
        parent=iter(population_stream(paths['parent:GEOMETRY_SEALED.jsonl.gz'],'method',PARENT_METHODS,'IMAGE_ROLE_BOUNDARY_ONLY',1960,known,audit,'original_geometry'))
        obs=iter(population_stream(paths['parent:OBSERVATIONS.jsonl.gz'],'head_arm',HEADS,'IMAGE_ROLE',735,known,audit,'original_ROLE'))
        streams=[iter(rows(paths[k])) for k in ('audit_rows','audit_ledgers','v9:GEOMETRY_SEALED.jsonl.gz','v9:CONTROL_LEDGERS.jsonl.gz')]+[parent,obs]
        for i,fid in enumerate(ordered):
            joined=[next(s,None) for s in streams];require(all(isinstance(r,dict) for r in joined),'incomplete source/audit joined frame')
            row,ledger,local,old_ledger,p,o=joined
            require(all(r['id']==fid and r['session']==p['session'] for r in joined),'source/audit reordered or mismatched frame')
            audit.eq(p['session'],cohort['frames'][i]['session'],'cohort.session.'+str(i))
            for name,value in (('audit',row),('LOCAL',local),('ROLE',o)):
                audit.ok(not TRUTH.intersection(value),'frame.GT_free.'+str(i)+'.'+name)
            packet(row,ledger,local,old_ledger,p,o,population['completion'],h,audit,counts,'frame.'+str(i))
            del joined,row,ledger,local,old_ledger,p,o
        for name,stream in zip(('audit','ledgers','LOCAL','old_ledgers','parent','ROLE'),streams):audit.eq(next(stream,None),None,'complete.no_extra.'+name)
        recorded_receipts(paths,inputs,core,seal,receipt,population,counts,audit)
        audit.eq(counts['audit_frames'],245,'complete.all245')
        audit.eq(counts['native_coordinate_banks'],245,'complete.one_shared_native_bank_per_frame')
        audit.eq(counts['stored_candidate_records_checked'],counts['stored_candidate_records_in_packets'],'complete.all_stored_candidates_checked')
        audit.eq({k:binding(p) for k,p in paths.items()},inputs,'complete.all_input_bytes_unchanged_after_arithmetic')
        audit.eq(binding(own_path),own_binding,'complete.own_protocol_unchanged_after_arithmetic')
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=failure is None and counts['audit_frames']==245;passed=complete and audit.failure_count==0
    write_new(output/OUTPUTS[2],dict(schema='supplemental_unused_line_endpoint_scalar_checks_v10',
        complete=complete,passed=passed,own_protocol=own_binding,checker=inputs['checker'],inputs=inputs,limits=LIMITS,
        checks=audit.checks,failure_count=audit.failure_count,failures=audit.failures,exception=failure,
        counts=dict(counts),max_scalar_absolute_difference=audit.max_difference,wall_seconds=time.monotonic()-began,
        new_model_RGB_GT_PnP_LOCAL_Jacobian_SVD_calls=0,full_population_rows_retained=0,
        full_native_candidate_witnesses_preserved_in_original_raw=True,original_V9_unchanged=True if passed else 'final binding test only if reached',
        first_failure_preserved=True,automatic_retry=False))
    print('UNUSED_LINE_SCALAR','PASS' if passed else 'FAIL',audit.checks,dict(counts),flush=True)
    if not passed:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC)
    p.add_argument('--protocol',type=Path,default=DOC/'AUDIT_PROTOCOL.json')
    p.add_argument('--v9-input',type=Path,default=V9);p.add_argument('--source-parent',type=Path,default=V7)
    p.add_argument('--cohort',type=Path,default=COHORT);args=p.parse_args()
    (freeze if args.stage=='freeze' else run)(args)


if __name__=='__main__':main()
