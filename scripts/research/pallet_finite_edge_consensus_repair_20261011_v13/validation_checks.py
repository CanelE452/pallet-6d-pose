"""Own-frozen stdlib check of finite-edge actual-inlier consensus.

No production imports or new optimizer, image, model, GT or SVD calls. Scalar
forward R/projection/J helpers are byte-bound unchanged V9 audit functions;
finite support hinges and their derivatives are independently reconstructed.
Stored singular spectra are recounted, not independently regenerated.
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
DOC=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
PRIVATE=Path('/tmp/pallet-finite-edge-consensus-private-20261011-v13')
PARENT=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
AUDIT=REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
SOURCE_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
V1=REPO/'scripts/research/pallet_observation_refiner_20261009_v1/solver.py'
FUNCS=('require','read','rows','binding','same_binding','semantic','safe','write_new','finite','dot','mm','eye','skew',
       'rotation_and_right_J','dimensions','model','forward','physical_rotation','equivalent','eligible','rank_witness','original_lines')
CONSTANTS=('EDGES','PHYSICAL','CAL','ATOL','RTOL','RANK','TIE','PRIMITIVES')
METHOD='FINITE_EDGE_CONSENSUS'
OUTPUTS=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(own_freeze_after_complete_geometry_before_own_arithmetic=True,production_imports_model_RGB_GT_PnP_optimizer_SVD_calls=0,
    scalar_R_projection_J_and_finite_hinge_derivatives_independent=True,stored_singular_rank_recount_only=True,
    actual_inlier_refit_pool_rank_and_existing_weak_gate_checked=True,full_original_pool_score_checked=True,
    one_edge_vote_not_query_votes=True,source_radius_not_a_new_gate=True,endpoint_UNKNOWN_retained=True,
    exact_original_sparse_coordinates_no_native_fill=True,H_fit_excluded_NEW_projection_once_without_refit=True,
    physical_ownership_certified=False,global_uniqueness_proven=False,hardware_authenticity_proven=False,
    thresholds_and_ties_follow_stored_scalar_after_independent_near_check=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)

def read(p):
    with Path(p).open(encoding='utf-8') as f:return json.load(f)

def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:
        for line in f:require(bool(line.strip()),'blank raw row');yield json.loads(line)

def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p))
    digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=digest.hexdigest())

def same(a,b):return all(a.get(k)==b.get(k) for k in ('bytes','sha256'))

def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'unchanged scalar source required')
    selected=[];names=set()
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in (*FUNCS,'Audit'):selected.append(n);names.add(n.name)
        elif isinstance(n,ast.Assign):
            ids={x.id for t in n.targets for x in (t.elts if isinstance(t,ast.Tuple) else [t]) if isinstance(x,ast.Name)}
            if ids and ids<=set(CONSTANTS):selected.append(n);names.update(ids)
    require(names==set(FUNCS)|set(CONSTANTS)|{'Audit'},'exact stdlib helper whitelist')
    module=ast.Module(body=selected,type_ignores=[])
    cls=next(n for n in ast.parse(V1.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='HypothesisBank')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_jacobian')
    values=[n.value for n in ast.walk(fn) if isinstance(n,ast.keyword) and n.arg=='weak_condition']
    require(len(values)==1 and ast.dump(values[0],include_attributes=False)==ast.dump(ast.parse('bool(s[-1] < s[0] * 1e-4)',mode='eval').body,include_attributes=False),'inherited weak expression differs')
    return module,dict(source=binding(SOURCE),AST_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest(),
        weak_source=binding(V1),weak_AST_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest())

def helpers(info):
    module,current=extraction();require(info==current,'frozen helper AST differs')
    env=dict(REPO=REPO,math=math,json=json,hashlib=hashlib,Path=Path,gzip=gzip,os=os,struct=struct)
    exec(compile(module,str(SOURCE)+':stdlib_whitelist_only','exec'),env);return env

def input_paths(args):
    p=args.input;paths=dict(geometry=p/'GEOMETRY_SEALED.jsonl.gz',ledgers=p/'CONTROL_LEDGERS.jsonl.gz',repair_protocol=args.protocol,
        geometry_seal=p/'GEOMETRY_SEAL.json',repair_receipt=p/'CONTROL_RECEIPT.json',parent_population_checks=p/'PARENT_POPULATION_CHECKS.json',
        checker=Path(__file__).resolve(),scalar_source=SOURCE,weak_source=V1,cohort=args.cohort,
        source_geometry=args.parent/'GEOMETRY_SEALED.jsonl.gz',source_observations=args.parent/'OBSERVATIONS.jsonl.gz',
        source_audit=args.audit/'AUDIT_ROWS.jsonl.gz',source_audit_protocol=args.audit/'AUDIT_PROTOCOL.json',source_audit_seal=args.audit/'AUDIT_SEAL.json',
        source_audit_validation=args.audit/'VALIDATION_CHECKS.json',source_audit_validation_protocol=args.audit/'VALIDATION_PROTOCOL.json',
        started=p/'CONTROL_STARTED.json',preflight=p/'PREFLIGHT.json',cpu_checks=p/'CPU_CHECKS.json',cpu_protocol=p/'CPU_PROTOCOL.json',
        cpu_review_checks=p/'CPU_REVIEW_CHECKS.json',cpu_review_protocol=p/'CPU_REVIEW_PROTOCOL.json',cpu_review_source=Path(__file__).with_name('cpu_review.py'),
        cpu_original_started=p/'CPU_STARTED.json',cpu_review_started=p/'CPU_REVIEW_STARTED.json')
    for name in ('solver.py','pipeline.py','runner.py','test_solver.py'):paths['code:'+name]=Path(__file__).with_name(name)
    for i,_ in enumerate(read(paths['repair_receipt'])['resource_snapshots']):paths['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return paths

def metadata(paths,inputs):
    core,seal,receipt,pop=[read(paths[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_endpoint_observability_repair_protocol_v13' and core['methods']==[METHOD] and core['frames']==245,'fixed method/cohort')
    require(seal['schema']=='endpoint_observability_geometry_seal_v13' and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and
        seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'complete GT-free geometry seal')
    require(receipt['schema']=='endpoint_observability_repair_execution_v13' and receipt['complete'] is True and receipt['cleanup_completed'] is True and
        receipt['error'] is None and receipt['cleanup_error'] is None,'complete cleanup receipt')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):
        require(same(seal[key],inputs[alias]),'seal byte join '+key)
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and same(core['parent_population_checks'],inputs['parent_population_checks']),'parent population proof')
    require(same(receipt['protocol'],inputs['repair_protocol']) and all(same(receipt['raw_rows'][k],inputs[k]) for k in ('geometry','ledgers')),'receipt raw/protocol byte join')
    for key in ('checker','scalar_source','weak_source','cohort','source_geometry','source_observations','source_audit','source_audit_protocol','source_audit_seal','source_audit_validation','source_audit_validation_protocol','cpu_checks','cpu_protocol','cpu_review_checks','cpu_review_protocol','cpu_review_source',*('code:'+n for n in ('solver.py','pipeline.py','runner.py','test_solver.py'))):
        require(any(b['path']==inputs[key]['path'] and same(b,inputs[key]) for b in core['inputs'].values()),'unbound core source '+key)
    policy=core['policy']['solver']
    for key,value in (('f_scale_px',8.),('max_nfev',50),('refit_iterations',1),('maximum_actual_optimizer_dispatches',8),('numerical_rank_relative_threshold',1e-10),('weak_relative_threshold',1e-4),('numerical_score_tie_px2',1e-8),('point_inlier_threshold_px',8.),('edge_block_inlier_threshold_px',8.),('truncated_factor_squared_cap',64.)):
        require(policy[key]==value,'fixed solver policy '+key)
    start,pre=read(paths['started']),read(paths['preflight'])
    require(same(start['protocol'],inputs['repair_protocol']) and pre['passed'] is True and same(pre['protocol'],inputs['repair_protocol']),'started/preflight binding')
    old,cpu,review,rp,old_started,new_started=[read(paths[k]) for k in ('cpu_checks','cpu_protocol','cpu_review_checks','cpu_review_protocol','cpu_original_started','cpu_review_started')]
    require(old['schema']=='finite_edge_CPU_checks_v13' and old['complete'] is True and old['passed'] is False and old['tests_run']==15 and len(old['failures'])==1 and
        old['failures'][0]['test'].startswith('test_natural_wrong_point_removed_remaining_pose (') and 'AssertionError: 6 != 7\n' in old['failures'][0]['traceback'] and
        old['errors']==[] and old['error'] is None and old['source_unchanged'] is True and old['opencv_threads_restored'] is True and old['cleanup_errors']==[],
        'preserve original 14PASS/one overstrict correspondence-count assertion failure')
    require(old['inputs']==cpu['inputs'] and same(old['own_protocol'],inputs['cpu_protocol']) and same(old_started['protocol'],inputs['cpu_protocol']),'original CPU authority')
    expected=dict(old['inputs'])
    for key in ('cpu_protocol','cpu_original_started','cpu_checks','cpu_review_source'):expected[inputs[key]['path']]=inputs[key]
    require(review['schema']=='finite_edge_CPU_review_checks_v13' and rp['schema']=='finite_edge_CPU_review_protocol_v13' and review['complete'] is True and review['passed'] is True and
        review['tests_run']==15 and review['failures']==review['errors']==[] and review['error'] is None and review['source_unchanged'] is True and
        review['opencv_threads_restored'] is True and review['cleanup_errors']==[] and review['inputs']==rp['inputs']==expected and
        same(review['own_protocol'],inputs['cpu_review_protocol']) and same(new_started['protocol'],inputs['cpu_review_protocol']) and
        same(new_started['original_failure'],inputs['cpu_checks']),'actual same-source corrected CPU assertion PASS')
    require(rp['AST_assertion_changes']==1 and rp['old_assertion']=="self.assertEqual(len(solved['fit_input_ids']),7)" and
        rp['new_assertion']=="self.assertGreaterEqual(len(solved['fit_input_ids']),4)" and rp['solver_or_observation_policy_changes']==0,'one CPU expectation correction only')
    for key,value in (('AST_assertion_changes',1),('original_tests_changed',False),('solver_or_observation_policy_changed',False),('original_failed_attempt_preserved',True),('actual_review_groups_rerun',15),('wrong_point_exclusion_and_centroid_check_unchanged',True)):
        require(review['review'][key]==value,'reviewed CPU scope '+key)
    for item in cpu['inputs'].values():require(same(binding(REPO/item['path']),item),'original CPU dependency changed')
    aseal,acheck,aown=[read(paths[k]) for k in ('source_audit_seal','source_audit_validation','source_audit_validation_protocol')]
    require(aseal['complete'] is True and aseal['frames']==245 and acheck['complete'] is True and acheck['passed'] is True and acheck['inputs']==aown['inputs'],'frozen endpoint audit PASS')
    require(same(aseal['audit'],inputs['source_audit']) and same(aseal['protocol'],inputs['source_audit_protocol']) and
        same(acheck['own_protocol'],inputs['source_audit_validation_protocol']) and same(acheck['inputs']['audit_rows'],inputs['source_audit']),'checked original endpoint bytes')
    return core,receipt,pop

def finite_blocks(uv,J,q,ids,lines,h,modeled=False,scoring=False):
    residual=[];matrix=[];blocks=[]
    for k in ids:residual.extend(uv[k][i]-q[k][i] for i in range(2));matrix.extend(J[k])
    for line in lines:
        a,b=h['EDGES'][line['edge']];A,B=uv[a],uv[b];delta=[B[i]-A[i] for i in range(2)];L=math.hypot(*delta)
        support=line['source_line']['support_points'];N=len(support);start=len(residual)
        base=dict(edge=line['edge'],endpoints=[a,b],scalar_start=start,scalar_end=start+2+N,support_query_ids=line['support_query_ids'],
            actual_support_xy=support,query_radii_px=line['source_line']['query_radii_px'],radius_gate_applied=False,block_factor_count=1,
            physical_ownership_truth='UNKNOWN',modeled_normals_held_fixed=modeled)
        if scoring and L<=1e-12:
            residual.extend([None]*(2+N));base.update(score_state='DEGENERATE_PROJECTED_EDGE_OUTLIER',normal_signed_px=None,
                outside_hinges_px=None,longitudinal_support_px=None,projected_segment_length_px=L);blocks.append(base);continue
        require(L>1e-12 and math.isfinite(L),'typed fitted edge collapse');t=[x/L for x in delta];n=[-t[1],t[0]] if modeled else line['normal']
        signed=[h['dot'](n,uv[k])-line['offset'] for k in (a,b)]
        residual.extend(x/math.sqrt(2) for x in signed);matrix.extend([[h['dot'](n,[J[k][i][j] for i in range(2)])/math.sqrt(2) for j in range(6)] for k in (a,b)])
        ddelta=[[J[b][i][j]-J[a][i][j] for j in range(6)] for i in range(2)]
        dL=[sum(t[i]*ddelta[i][j] for i in range(2)) for j in range(6)]
        dt=[[(ddelta[i][j]-t[i]*dL[j])/L for j in range(6)] for i in range(2)]
        along=[];hinges=[];branches=[]
        for xy in support:
            v=[xy[i]-A[i] for i in range(2)];u=h['dot'](v,t);du=[sum(-t[i]*J[a][i][j]+v[i]*dt[i][j] for i in range(2)) for j in range(6)]
            if u<0:hinge,row,branch=u,du,'BEFORE_FIRST_ENDPOINT'
            elif u>L:hinge,row,branch=u-L,[du[j]-dL[j] for j in range(6)],'AFTER_SECOND_ENDPOINT'
            else:hinge,row,branch=0.,[0.]*6,'INSIDE_CLOSED_SEGMENT'
            along.append(u);hinges.append(hinge);branches.append(branch);residual.append(hinge/math.sqrt(N));matrix.append([x/math.sqrt(N) for x in row])
        base.update(normal_signed_px=signed,outside_hinges_px=hinges,longitudinal_support_px=along,projected_segment_length_px=L)
        if scoring:base['score_state']='FINITE_EDGE_BLOCK'
        else:base['hinge_branches']=branches
        blocks.append(base)
    return residual,matrix,blocks

def score_check(w,uv,J,row,U,lines,h,a,label):
    rr,_,blocks=finite_blocks(uv,J,row['input_points'],U,lines,h,scoring=True)
    point=[math.dist(uv[k],row['input_points'][k]) for k in U]
    edge=[None if b['score_state']=='DEGENERATE_PROJECTED_EDGE_OUTLIER' else math.sqrt(math.fsum(x*x for x in rr[b['scalar_start']:b['scalar_end']])) for b in blocks]
    a.near(w['point_residual_norm_px'],point,label+'.point_norms');a.near(w['edge_block_norm_px'],edge,label+'.edge_norms')
    a.near(w['edge_blocks'],blocks,label+'.closed_segment_blocks');a.near(w['scalar_residuals_px'],rr,label+'.scalar_residuals')
    # Preserve production boundary predicates after independently checking their scalars.
    ip=[k for k,e in zip(U,w['point_residual_norm_px']) if e<=8.];il=[l['edge'] for l,e in zip(lines,w['edge_block_norm_px']) if e is not None and e<=8.]
    norms=point+edge
    expected=dict(point_ids=U,line_edges=[l['edge'] for l in lines],final_inliers=ip,final_line_inliers=il,inlier_count=len(ip),
        unique_line_inlier_count=len(il),total_inlier_factor_count=len(ip)+len(il),factor_sse_px2=None if None in norms else math.fsum(v*v for v in norms),
        factor_truncated_sse_px2=math.fsum(64. if v is None else min(v*v,64.) for v in norms),score_only_degenerate_edge_is_outlier=True,fixed_full_pool=True)
    for key,value in expected.items():a.near(w[key],value,label+'.'+key)
    return ip,il

def candidate_check(c,row,U,H,lines,h,a,counts,label,phase):
    dims=h['dimensions'](row['xyz']);d=c['dimension_index'];require(type(d) is int and 0<=d<len(dims),'candidate dimension')
    ids=c['fit_input_ids'];edges=c['fit_line_edges'];selected=[l for l in lines if l['edge'] in edges]
    require(ids==[k for k in U if k in ids] and edges==[l['edge'] for l in selected],'fit subset order/identity')
    a.eq(c['phase'],phase,label+'.phase');a.eq(c['cf_extents'],dims[d],label+'.registry_dims');a.eq(c['selected_hypothesis'],('REGISTRY_WD','REGISTRY_DW')[d],label+'.registry_name')
    a.ok(not set(ids)&set(H),label+'.no_H_fit');a.ok(8 not in ids,label+'.no_center_fit')
    R,uv,J,depths=h['forward'](dims[d],c['final_z'],row['K'])
    a.ok(all(math.isfinite(v) and v>1e-9 for v in depths),label+'.independent_all_eight_positive_depths')
    for key,value in (('R_cf',R),('R_physical',h['physical_rotation'](R,d)),('centroid',c['final_z'][3:]),('projected',uv),('model_camera_depths_m',depths)):
        a.near(c[key],value,label+'.'+key)
    rr,observed,blocks=finite_blocks(uv,J,row['input_points'],ids,selected,h);_,modeled,_=finite_blocks(uv,J,row['input_points'],ids,selected,h,modeled=True)
    g=c['point_line_geometry'];a.eq(g['point_ids'],ids,label+'.fit_geometry_ids');a.eq(g['line_edges'],edges,label+'.fit_geometry_edges')
    a.near(g['observed_normal_J'],observed,label+'.independent_observed_J');a.near(g['modeled_normal_J'],modeled,label+'.independent_modeled_J')
    a.near(g['actual_scalar_residuals_px'],rr,label+'.geometry_residuals');a.near(g['edge_blocks'],blocks,label+'.finite_geometry_blocks')
    split=2*len(ids);ranks={}
    for name,matrix in (('observed_normal_joint',observed),('modeled_normal_joint',modeled),('point_only',modeled[:split]),('observed_normal_lines_only',observed[split:]),('modeled_normal_lines_only',modeled[split:])):
        ranks[name]=h['rank_witness'](g[name],matrix,a,label+'.'+name)
    weak={}
    for name,key in (('observed','observed_normal_joint'),('modeled','modeled_normal_joint')):
        spectrum=g[key]['singular_values'];weak[name]=len(spectrum)!=6 or spectrum[-1]<spectrum[0]*1e-4;a.eq(g[name+'_weak'],weak[name],label+'.'+name+'_weak')
    a.eq(g['modeled_normals_held_fixed_for_derivative'],True,label+'.held_fixed_modeled_normal');a.eq(g['global_unique_pose_proven'],False,label+'.local_rank_only')
    a.near(c['scalar_residuals_px'],rr,label+'.fit_residuals');cost=math.fsum(64.*(math.sqrt(1.+(v/8.)**2)-1.) for v in rr)
    a.near(c['soft_l1_cost'],cost,label+'.fit_softloss');a.near(c['reported_soft_l1_cost'],cost,label+'.actual_reported_softloss');a.near(c['sse_px2'],math.fsum(v*v for v in rr),label+'.fit_SSE')
    for key,value in (('optimizer_success',True),('initial_pose_residual_prior',False),('local_only',True),('global_uniqueness_proven',False)):a.eq(c[key],value,label+'.'+key)
    ip,il=score_check(c['whole_pool_score'],uv,J,row,U,lines,h,a,label+'.whole_pool')
    if phase=='proposal':a.eq(ids,U,label+'.proposal_whole_U');a.eq(edges,[l['edge'] for l in lines],label+'.proposal_whole_lines')
    else:
        stable=set(ids)<=set(ip) and set(edges)<=set(il);rank_ok=ranks['observed_normal_joint']==ranks['modeled_normal_joint']==6
        reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank_ok else 'WEAK_OBSERVABILITY' if any(weak.values()) else None
        for key,value in (('whole_pool_point_ids',U),('whole_pool_line_edges',[l['edge'] for l in lines]),('final_inliers',ip),('final_line_inliers',il),('fit_pool_all_final8px_inliers',stable),('accepted',reason is None),('rejected',reason),('score_count',len(ip)+len(il)),('actual_inlier_geometry_is_acceptance_gate',True),('whole_pool_scoring_factors_not_all_fit_observations',True)):
            a.eq(c[key],value,label+'.'+key)
        a.near(c['truncated_sse_px2'],c['whole_pool_score']['factor_truncated_sse_px2'],label+'.full_pool_truncated')
    counts[phase+'_candidate_checks']+=1

def primary(path,key,name,total,known,a,label):
    seen=set()
    for row in rows(path):
        identity=(row[key],row['id']);require(row['id'] in known and identity not in seen,'unknown/duplicate source');seen.add(identity)
        if row[key]==name:yield row
    a.eq(len(seen),total,label+'.full_population')

def packet(row,ledger,parent,obs,endpoint,completion,h,a,counts,label):
    q,H,s=parent['input_points'],parent['hidden_initial'],row['solver'];U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]
    local_packet=dict(parent_row=parent,role_observation=obs,native_N3_points=obs['native_N3_points'],parent_completion=completion)
    a.eq(ledger['input_packet_semantic_sha256'],h['semantic'](dict(local_packet=local_packet,audit_row=endpoint)),label+'.packet_SHA')
    for key,value in (('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('original_ROLE_observation_semantic_sha256',h['semantic'](obs)),('original_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points']))):a.eq(endpoint[key],value,label+'.endpoint_source.'+key)
    rejected=[r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='CONTRADICTED_REJECT'];filtered=copy.deepcopy(obs)
    if endpoint['contract_error'] is None:filtered['lines']=[l for l in filtered['lines'] if l['edge'] not in rejected]
    adopted=parent['observation_contract']['validated_boundary_corner_ids']
    if endpoint['contract_error'] is None:rawlines,consumed=h['original_lines'](filtered,q,U,adopted,a,label+'.source_factors')
    else:rawlines,consumed=[],[];a.eq(s['state'],'INVALID_OBSERVATION_CONTRACT',label+'.typed_invalid')
    source={l['edge']:l for l in filtered['lines']};stored={l['edge']:l for l in s.get('actual_source_lines',[])};lines=[]
    for l in rawlines:
        w=stored[l['edge']];a.near(w['normal'],l['normal'],label+'.independent_normal');a.near(w['offset'],l['offset'],label+'.independent_offset')
        a.eq(w['raw'],source[l['edge']],label+'.unchanged_support_source')
        lines.append(dict(edge=l['edge'],normal=w['normal'],offset=w['offset'],support_query_ids=l['support_query_ids'],source_line=w['raw']))
    edges=[l['edge'] for l in lines];a.eq(sorted(stored),edges,label+'.complete_saved_line_sources')
    for key,value in (('method',METHOD),('input_points',q),('original_native_N3_points',obs['native_N3_points']),('K',parent['K']),('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),('hidden_initial',H),('initial_pose',parent['initial_pose']),('parent_completion',completion),('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('endpoint_audit_semantic_sha256',h['semantic'](endpoint))):a.eq(row[key],value,label+'.original.'+key)
    f=row['observation_filter'];a.eq(f,ledger['observation_filter'],label+'.filter_ledger')
    a.eq(f['original_factor_pool_binding'],endpoint['factor_pool_binding'],label+'.original_factor_byte_witness')
    for key,value in (('original_observation_semantic_sha256',h['semantic'](obs)),('filtered_observation_semantic_sha256',h['semantic'](filtered)),('endpoint_records',endpoint['endpoint_records']),('rejected_edge_ids',rejected),('retained_unused_edge_ids',edges),('unknown_retained_edge_ids',[r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='UNVERIFIED_RETAIN']),('consumed_edges',consumed)):
        a.eq(f[key],value,label+'.filter.'+key)
    a.eq(edges,[e for e in endpoint['original_actual_unused_edges'] if e not in rejected],label+'.reject_only_filter');a.eq(s,ledger['full_solver'],label+'.actual_solver_ledger')
    for key,value in (('used',U),('hidden',H),('excluded',H),('line_edges',edges),('consumed_edges',consumed)):a.eq(s[key],value,label+'.solver.'+key)
    for key in ('prior_used','initial_pose_residual_prior','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','center_used_for_fit_or_score','same_edge_point_line_double_count'):
        a.eq(s[key],False,label+'.no_fake_observations.'+key)
    if endpoint['contract_error'] is None:
        encoded=json.dumps(dict(points=[q[k] for k in U],point_ids=U,lines=lines),sort_keys=True,separators=(',',':'),allow_nan=True).encode()
        a.eq(s['factor_pool_binding'],dict(sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded),definition='semantic sorted JSON; not file-byte provenance'),label+'.actual_full_factor_SHA')
        a.eq(s['factor_pool'],dict(point_ids=U,line_edges=edges,unique_line_factors=len(edges),actual_point_factors=len(U),scalar_residuals=2*len(U)+sum(2+len(l['support_query_ids']) for l in lines)),label+'.actual_factor_rows')
    proposals=s['proposal_solutions'];candidates=s['all_candidate_solutions'];attempts=s['optimizer_attempts'];require(len(attempts)<=4 and len(proposals)<=4 and len(candidates)<=4,'fixed four existing branches')
    for i,p in enumerate(proposals):a.eq(p['proposal_index'],i,label+'.proposal_index');candidate_check(p,row,U,H,lines,h,a,counts,label+'.proposal.'+str(i),'proposal')
    for i,c in enumerate(candidates):a.eq(c['candidate_index'],i,label+'.candidate_index');candidate_check(c,row,U,H,lines,h,a,counts,label+'.refit.'+str(i),'refit')
    viable=[c for c in candidates if c['accepted']];ranked=sorted(viable,key=lambda c:(-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index']));best=ranked[0] if ranked else None
    ambiguity=[];duplicates=ties=0;alternatives=[]
    if best:
        a.eq(s.get('best_candidate'),best,label+'.full_pool_selection')
        for c in ranked[1:]:
            eq=h['equivalent'](best,c,row['xyz']);delta=c['truncated_sse_px2']-best['truncated_sse_px2'];tie=c['score_count']==best['score_count'] and abs(delta)<=1e-8
            same_support=c['final_inliers']==best['final_inliers'] and c['final_line_inliers']==best['final_line_inliers'];duplicates+=int(eq);ties+=int(tie)
            alternatives.append(dict(c,numerically_equivalent_to_selected=eq,numerical_objective_tie=tie,objective_difference_from_selected_px2=delta,same_actual8px_consensus_as_selected=same_support))
            if not eq and (same_support or tie):ambiguity.append('OBSERVATION_CONSENSUS_AMBIGUITY' if same_support else 'NUMERICAL_CONSENSUS_TIE')
        a.near(s['alternatives'],alternatives,label+'.all_branch_relations');a.eq(s['per_dimension_best'],[next(c for c in ranked if c['dimension_index']==d) for d in range(len(h['dimensions'](row['xyz']))) if any(c['dimension_index']==d for c in ranked)],label+'.per_dimension_best')
        a.eq(s['state'],'OBSERVATION_CONSENSUS_AMBIGUITY' if 'OBSERVATION_CONSENSUS_AMBIGUITY' in ambiguity else 'NUMERICAL_CONSENSUS_TIE' if ambiguity else 'NEW_POSE',label+'.selected_branch_state')
    else:a.eq(s.get('best_candidate'),None,label+'.no_selected_branch')
    new=bool(best and not ambiguity);a.eq(s['available'],new,label+'.numerical_availability');a.eq(s['unresolved_ambiguity'],bool(ambiguity),label+'.ambiguity')
    a.eq(s['numerical_duplicate_solution_count'],duplicates,label+'.equivalent_branches');a.eq(s['numerical_tied_solution_count'],ties,label+'.numerical_ties')
    success=Counter();calls=Counter();cache=Counter();failures=[];seen_starts={'proposal':{},'refit':{}}
    for i,e in enumerate(attempts):
        a.eq(e['start_index'],i,label+'.logical_start_index');a.eq(e['initial_pose_residual_prior'],False,label+'.start_no_prior')
        for phase in ('proposal','refit'):
            entry=e[phase]
            if entry is None:continue
            actual=entry['actual_optimizer_called'];calls[phase]+=int(actual);success[phase]+=int(actual and entry.get('success',False));cache[phase]+=int(entry['cache_reused'])
            if phase=='refit':
                p=proposals[entry['proposal_index']];a.eq(entry['proposal_consensus_point_ids'],p['whole_pool_score']['final_inliers'],label+'.refit_actual_point_consensus')
                a.eq(entry['proposal_consensus_line_ids'],p['whole_pool_score']['final_line_inliers'],label+'.refit_actual_edge_consensus')
                if 'candidate_index' in entry:
                    c=candidates[entry['candidate_index']];a.eq(c['fit_input_ids'],entry['proposal_consensus_point_ids'],label+'.fit_only_consensus_points');a.eq(c['fit_line_edges'],entry['proposal_consensus_line_ids'],label+'.fit_only_consensus_lines')
            if 'initial_z' in entry:
                key=(entry['dimension_index'],struct.pack('<6d',*entry['initial_z']))
                if phase=='refit':key+=(tuple(entry['proposal_consensus_point_ids']),tuple(entry['proposal_consensus_line_ids']))
                a.eq(entry['cache_reused'],key in seen_starts[phase],label+'.exact_start_cache.'+phase)
                if key in seen_starts[phase]:
                    previous=seen_starts[phase][key];a.eq(entry['cache_source_start_index'],previous['start_index'],label+'.cache_provenance.'+phase)
                    a.eq(actual,False,label+'.cache_no_optimizer.'+phase)
                    for field in ('initial_z','final_z','success','scipy_status','nfev','njev','reported_soft_l1_cost','exception'):
                        a.eq(entry.get(field),previous.get(field),label+'.cache_same_numeric_witness.'+phase+'.'+field)
                else:seen_starts[phase][key]=entry;a.eq(actual,True,label+'.fresh_optimizer_dispatch.'+phase)
            if entry.get('rejected'):failures.append(entry['rejected'])
    op=s['operation_counts'];a.ok(sum(calls.values())<=8,label+'.optimizer_budget');a.eq(op['local_optimizer_calls'],sum(calls.values()),label+'.actual_attempt_count')
    for phase in ('proposal','refit'):a.eq(op[phase+'_optimizer_calls'],calls[phase],label+'.'+phase+'_calls');a.eq(op[phase+'_cache_hits'],cache[phase],label+'.'+phase+'_cache_hits')
    for key,value in (('candidate_count',len(candidates)),('proposal_candidate_count',len(proposals)),('viable_candidate_count',len(viable)),('refit_count',calls['refit'])):
        if key in s:a.eq(s[key],value,label+'.'+key)
    if attempts:
        categories=dict(numeric=0,insufficient=0,weak=0,unstable=0,other=0)
        for reason in failures:
            key='insufficient' if reason.startswith('INSUFFICIENT_ACTUAL_INLIER') else 'weak' if reason=='WEAK_OBSERVABILITY' else 'unstable' if reason=='UNSTABLE_INLIER_REFIT' else 'numeric' if reason in ('LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED') else 'other'
            categories[key]+=1
        a.eq(s['failure_reasons'],failures,label+'.all_branch_failure_reasons');a.eq(s['failure_category_counts'],categories,label+'.typed_failure_counts')
        if not viable:
            active=[key for key,count in categories.items() if count];states=dict(numeric='NUMERICAL_FAILURE',insufficient='INSUFFICIENT_ACTUAL_INLIER_GEOMETRY',weak='WEAK_OBSERVABILITY',unstable='UNSTABLE_INLIER_REFIT')
            state=states.get(active[0],'MIXED_NO_VALID_CONSENSUS') if len(active)==1 else 'MIXED_NO_VALID_CONSENSUS';a.eq(s['state'],state,label+'.typed_no_viable_state')
    fallback=not new and parent['initial_pose'].get('available',False)
    for key,value in (('new_pose_estimated',new),('fallback_used',fallback),('pose_available',new or fallback),('no_pose',not(new or fallback)),('output_status','NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),('reprojections_reused_as_observations',False),('reprojected_ids',H if new else []),('hidden_reprojected',new and bool(H))):a.eq(row[key],value,label+'.output.'+key)
    expected=copy.deepcopy(obs['native_N3_points']);sparse=copy.deepcopy(q)
    if new:
        for key in ('R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry','whole_pool_score','soft_l1_cost','sse_px2','truncated_sse_px2','final_inliers','final_line_inliers'):a.near(s[key],best[key],label+'.chosen.'+key)
        for k in range(8):
            if k in H:expected[k]=best['projected'][k];sparse[k]=best['projected'][k]
            elif h['finite'](q[k],(2,)) and q[k]!=[-1,-1]:expected[k]=q[k]
        a.eq(row['actual_pose'],s,label+'.actual_selected_solver')
    else:
        a.eq(row['actual_pose'],parent['initial_pose'],label+'.full_original_fallback_pose')
        for key in ('R_cf','R_physical','centroid','cf_extents','projected'):a.eq(s[key],None,label+'.unavailable_pose_cleared.'+key)
        a.eq(s['fit_input_ids'],[],label+'.unavailable_fit_ids');a.eq(s['fit_line_edges'],[],label+'.unavailable_fit_edges')
    a.near(row['native_points'],expected,label+'.full_display_coordinates');a.near(s['points_final'],sparse,label+'.H_only_solver_reprojection')
    a.eq(row['native_points'][8],obs['native_N3_points'][8],label+'.center_preserved');a.eq(s['hidden_reprojected'],new and bool(H),label+'.solver_H_reprojection')
    for key,value in (('new_pose_estimated',new),('pose_available',new),('no_pose',not new),('global_uniqueness_proven',False)):a.eq(s[key],value,label+'.solver_flags.'+key)
    a.eq(op['hidden_reprojection_assignments'],len(H) if new else 0,label+'.one_H_assignment_no_refit')
    a.eq(ledger['actual_filtered_local_solve_calls'],1,label+'.one_finite_solver');a.eq(ledger['reported_output_views'],1,label+'.one_view');a.eq(ledger['endpoint_validation_requests'],0,label+'.no_endpoint_pose_replay_calls')
    a.eq(ledger['endpoint_decision_source'],'SEALED_V10_ENDPOINT_DECISION_REPLAY',label+'.frozen_endpoint_decisions')
    expected_opt=dict(entries=op['local_optimizer_calls'],returns=op['local_optimizer_completions'],exceptions=op['local_optimizer_exceptions'],
        success_reports=sum(success.values()),residual_entries=op['optimizer_residual_calls'],jacobian_entries=op['optimizer_jacobian_calls'])
    a.eq(ledger['optimizer_dispatch'],expected_opt,label+'.actual_optimizer_dispatch');a.eq(expected_opt['entries'],expected_opt['returns']+expected_opt['exceptions'],label+'.optimizer_completions')
    for field,prefix,keys in (('primitive_entry_calls','cv:entry:',h['PRIMITIVES']),('primitive_return_calls','cv:return:',h['PRIMITIVES']),('primitive_exception_calls','cv:exception:',h['PRIMITIVES']),('svd_dispatch','svd:',('entries','returns','exceptions')),('optimizer_dispatch','opt:',tuple(expected_opt))):
        require(set(ledger[field])==set(keys) and all(type(v) is int and v>=0 for v in ledger[field].values()),'invalid dispatch counters')
        for key,value in ledger[field].items():counts[prefix+key]+=value
    for n in h['PRIMITIVES']:a.eq(ledger['primitive_entry_calls'][n],ledger['primitive_return_calls'][n]+ledger['primitive_exception_calls'][n],label+'.CV_complete.'+n)
    for n in ('solvePnP','solvePnPGeneric','solvePnPRefineLM','cornerSubPix'):a.eq(ledger['primitive_entry_calls'][n],0,label+'.no_extra_point_solver.'+n)
    a.eq(ledger['svd_dispatch']['entries'],ledger['svd_dispatch']['returns']+ledger['svd_dispatch']['exceptions'],label+'.SVD_complete')
    for key,value in op.items():counts['operation:'+key]+=value
    counts['frames']+=1;counts['status:'+row['output_status']]+=1;counts['stored_proposals']+=len(proposals);counts['stored_refits']+=len(candidates)

def guard(args,stage):
    for p in (args.input,args.output,args.protocol,args.parent,args.audit,args.cohort):require(not any(q.is_symlink() for q in (p,*p.parents)),'symlink ancestry')
    out=args.output.resolve();require(out==DOC.resolve() or out.is_relative_to(PRIVATE.resolve()),'exclusive V13 output only');require(out.is_dir(),'existing output directory')
    for n in OUTPUTS if stage=='freeze' else OUTPUTS[1:]:require(not (out/n).exists(),'preserve '+n)
    return out

def freeze(args):
    out=guard(args,'freeze');paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()};_,info=extraction();metadata(paths,inputs)
    write(out/OUTPUTS[0],dict(schema='finite_edge_consensus_scalar_protocol_v13',inputs=inputs,checker=inputs['checker'],helpers=info,
        method=METHOD,frames=245,limits=LIMITS,frozen_before_own_arithmetic=True));print('FINITE_EDGE_SCALAR_FROZEN',flush=True)

def run(args):
    out=guard(args,'run');own_path=out/OUTPUTS[0];own=read(own_path);ob=binding(own_path);paths=input_paths(args);inputs={k:binding(p) for k,p in paths.items()};_,info=extraction()
    require(own['schema']=='finite_edge_consensus_scalar_protocol_v13' and own['inputs']==inputs and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    core,receipt,pop=metadata(paths,inputs);write(out/OUTPUTS[1],dict(own_protocol=ob,inputs=inputs,new_production_calls=0));h=helpers(info);a=h['Audit']();counts=Counter();error=None;began=time.monotonic()
    try:
        cohort=read(paths['cohort']);ordered=cohort['ids'];known=set(ordered);require(len(ordered)==len(known)==245 and [f['id'] for f in cohort['frames']]==ordered,'full ordered245 cohort')
        a.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'scope')
        streams=[iter(rows(paths[k])) for k in ('geometry','ledgers','source_audit')]
        parents=iter(primary(paths['source_geometry'],'method','IMAGE_ROLE_BOUNDARY_ONLY',1960,known,a,'V7_geometry'));observations=iter(primary(paths['source_observations'],'head_arm','IMAGE_ROLE',735,known,a,'V7_ROLE'))
        for i,fid in enumerate(ordered):
            group=[next(stream,None) for stream in streams]+[next(parents,None),next(observations,None)]
            require(all(isinstance(r,dict) and r['id']==fid and r['session']==cohort['frames'][i]['session'] for r in group),'incomplete/reordered joined frame')
            row,ledger,endpoint,parent,obs=group;packet(row,ledger,parent,obs,endpoint,pop['completion'],h,a,counts,'frame.'+str(i));del group
        for stream in [*streams,parents,observations]:a.eq(next(stream,None),None,'no_extra_rows')
        for key in ('frames_started','frames_complete','filtered_solver_entries','filtered_solver_returns','geometry_rows','ledger_rows'):a.eq(receipt['actual_execution'][key],245,'actual.'+key)
        a.eq(receipt['actual_execution']['filtered_solver_exceptions'],0,'actual.no_incomplete_solver')
        a.eq(receipt['local_operation_counts'],{k.removeprefix('operation:'):v for k,v in counts.items() if k.startswith('operation:')},'actual.operation_sums')
        for field,prefix in (('actual_OpenCV_entry_calls','cv:entry:'),('actual_OpenCV_return_calls','cv:return:'),('actual_OpenCV_exception_calls','cv:exception:')):a.eq(receipt[field],{n:counts[prefix+n] for n in h['PRIMITIVES']},'actual.'+field)
        a.eq(receipt['actual_optimizer'],{k:counts['opt:'+k] for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')},'actual.optimizer_sum')
        a.eq(receipt['actual_SVD_dispatch'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'actual.SVD_sum')
        a.eq(receipt['output_status_counts'],{METHOD:{s:counts['status:'+s] for s in ('NEW_POSE','N3_BASELINE_FALLBACK','POSE_FAILURE') if counts['status:'+s]}},'actual.all245_operating_status')
        for key in ('primitive_and_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):a.eq(receipt['cleanup'][key],True,'recorded.cleanup.'+key)
        a.eq(receipt['forbidden_entries'],{},'no_forbidden_entries');a.eq(receipt['forbidden_asset_reads'],[],'no_forbidden_assets');a.eq(receipt['protection_before'],receipt['protection_after'],'protected_bytes')
        for i,snap in enumerate(receipt['resource_snapshots']):a.eq(read(paths['resource:%03d'%i]),snap,'resource_identity');a.eq(snap['quiet'],True,'recorded_quiet');a.eq(snap['temperature_under_80'],True,'recorded_thermal')
        a.eq(counts['frames'],245,'all245');a.eq(counts['stored_proposals'],counts['proposal_candidate_checks'],'all_proposal_math');a.eq(counts['stored_refits'],counts['refit_candidate_checks'],'all_refit_math')
        a.eq({k:binding(p) for k,p in paths.items()},inputs,'input_bytes_unchanged');a.eq(binding(own_path),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames']==245;passed=complete and a.failure_count==0
    write(out/OUTPUTS[2],dict(schema='finite_edge_consensus_scalar_checks_v13',complete=complete,passed=passed,own_protocol=ob,checker=inputs['checker'],
        inputs=inputs,limits=LIMITS,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,counts=dict(counts),
        max_scalar_absolute_difference=a.max_difference,wall_seconds=time.monotonic()-began,new_production_calls=0,first_failure_preserved=True,automatic_retry=False))
    print('FINITE_EDGE_SCALAR','PASS' if passed else 'FAIL',a.checks,dict(counts),flush=True)
    if not passed:raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'))
    for name,default in (('input',DOC),('output',DOC),('protocol',DOC/'PROTOCOL.json'),('parent',PARENT),('audit',AUDIT),('cohort',COHORT)):p.add_argument('--'+name,type=Path,default=default)
    args=p.parse_args();(freeze if args.stage=='freeze' else run)(args)

if __name__=='__main__':main()
