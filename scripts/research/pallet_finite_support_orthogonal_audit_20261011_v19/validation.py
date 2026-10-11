"""Independent, own-frozen scalar audit of the observed-frame repair.

Only stdlib is imported. Exact old independent projection/normal-incidence
AST bodies check every saved proposal/refit; the new fixed observed tangent
and its endpoint/support hinges are rederived here. No solver, GT, image,
model, PnP, optimization or SVD runs. Stored spectra are only recounted.
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

FAMILY='pallet_finite_support_orthogonal_audit_20261011_v19'
METHODS=('HYBRID_OBSERVED_FRAME_FACTOR_SUBSET_CONSENSUS','HYBRID_OBSERVED_FRAME_POINT_ONLY')
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
MATH_SOURCES={
 'projection_scalar':('pallet_sparse_local_line_20261010_v9/validation_checks.py','4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'),
 'subset_scalar':('pallet_factor_subset_consensus_20261011_v15/validation_checks.py','ad945e8985b9ca19a576d2423e83b5b635a20b1df44495c2b683a2659c752f26'),
 'paired_scalar':('pallet_hybrid_factor_subset_consensus_20261011_v16/validation_checks.py','1567db06b27e0394e95a539c943eac19eb57b5d257ebccf8d24a65866bbf697c'),
 'endpoint_scalar':('pallet_unused_line_endpoint_audit_20261011_v10/validation_checks.py','32ed14623a7e894168d0357f9275edf7eb6d4f0605f7e986ea4a4adba14ff52e'),
 'native_scalar':('pallet_three_head_observation_20261010_v7/validation_checks.py','815fd15d8596fe6c8c85720e136627dcef8d4cdb22c23e00737b3f507a3fa4a6'),
 'finite_scalar':('pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py','91d014543eba01c142cad9a0c69affd2eb16666913ce13a06a56857f854c98ef'),
 'normal_guard':('pallet_stable_incidence_guard_20261011_v14/pipeline.py','17828a9f0ff7f895898cab761d6aee8e20533a7564c49353ecfe3d03ee7bd6ec')}
LIMITS=dict(new_production_calls=0,new_GT_RGB_model_training_calls=0,new_optimizer_PnP_SVD_calls=0,
 all_saved_successful_proposals_and_refits_checked=True,fresh_endpoint_candidates_checked=True,
 stored_spectrum_recount_only=True,old_full_V16_564_FAIL_remains_FAIL=True,
 physical_ownership_certified=False,unseen_accuracy_certified=False,global_root_completeness_proven=False,
 cusp_allowance_coordinate_ULPs=64,cusp_allowance_is_not_a_performance_threshold=True,
 one_sided_cusp_derivative_is_not_smooth_observability=True,
 quiet_snapshots_can_miss_transient_contention=True,hardware_authenticity_proven=False,
 failed_root_pose_not_saved_cannot_be_reconstructed=True,automatic_retry=False)
STATE=dict(context={},events=[])

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
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())
def selected(tree,names):
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require({n.name for n in nodes}==set(names),'exact AST function whitelist');return nodes
def canonical(node):return ast.dump(node,include_attributes=False)

def extract(f):
    """Source-only extraction; no scalar/helper or production function calls."""
    for key,(_,sha) in MATH_SOURCES.items():require(binding(f[key])['sha256']==sha,'immutable independent source '+key)
    env=dict(ast=ast,copy=copy,hashlib=hashlib,require=require,binding=binding,selected=selected,
        SUBSET=f['subset_scalar'],SUBSET_SHA=MATH_SOURCES['subset_scalar'][1],ENDPOINT=f['endpoint_scalar'],
        ENDPOINT_SHA=MATH_SOURCES['endpoint_scalar'][1],NATIVE=f['native_scalar'])
    old=ast.parse(f['paired_scalar'].read_bytes());fn=selected(old,('extraction',))[0]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(f['paired_scalar'])+':AST_extractor_only','exec'),env)
    trees,info=env['extraction']();core=next(n for n in trees[0].body if isinstance(n,ast.FunctionDef) and n.name=='solver_core')
    original=copy.deepcopy(core);old_test=ast.parse('not eq and (same_support or tie)',mode='eval').body
    branches=[n for n in ast.walk(core) if isinstance(n,ast.If) and canonical(n.test)==canonical(old_test)]
    require(len(branches)==1,'one inherited any-cost support ambiguity condition')
    branch=branches[0];saved=copy.deepcopy(branch.test);branch.test=ast.parse('not eq and tie',mode='eval').body
    changed=canonical(core);branch.test=saved;require(canonical(core)==canonical(original),'whole scalar core inverse proof')
    branch.test=ast.parse('not eq and tie',mode='eval').body
    trees[0].body+=copy.deepcopy(selected(old,('source_checks','packet')))
    trees=[ast.fix_missing_locations(t) for t in trees]
    info.update(new_decision_AST_predicate_changes=1,whole_scalar_core_inverse_equal=True,
        repaired_core_AST_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        paired_packet_and_source_checks_whole_AST_unchanged=True,
        fixed_observed_frame_residual_and_score_functions='new independent validation.py definitions',
        production_modules_imported=False)
    return trees,info

def helpers(trees,f):
    h=dict(ast=ast,math=math,json=json,hashlib=hashlib,struct=struct,copy=copy,Counter=Counter,
        combinations=combinations,Path=Path,require=require,binding=binding,rows=rows,
        SOURCE=f['projection_scalar'],SOURCE_SHA=MATH_SOURCES['projection_scalar'][1],
        FINITE=f['finite_scalar'],FINITE_SHA=MATH_SOURCES['finite_scalar'][1],GUARD=f['normal_guard'],
        STATE=STATE,METHODS=METHODS)
    exec(compile(trees[0],str(f['subset_scalar'])+':independent_math','exec'),h)
    tree,policy,_=h['extraction']();exec(compile(tree,str(f['projection_scalar'])+':scalar_projection','exec'),h)
    h.update(finite_blocks=finite_blocks,score_check=score_check)
    n=dict(math=math,struct=struct,hashlib=hashlib,require=require)
    exec(compile(trees[1],str(f['endpoint_scalar'])+':native_endpoint_scalar','exec'),n)
    return h,n,policy

def observed_choice(eu,su,A,B,support,line,modeled,witness,h):
    """Strict saved one-sided branch only after independent scalar agreement."""
    low,high=sorted(range(2),key=lambda k:(eu[k],k));seu,ssu=list(eu),list(su)
    block=next((b for b in witness if b['edge']==line['edge']),None) if witness is not None else None
    if block is not None:
        seu,ssu=block['endpoint_u'],block['support_u']
        require(h['finite'](seu,(2,)) and h['finite'](ssu,(len(su),)),'finite stored observed interval')
        require(all(abs(v-w)<=h['ATOL']+h['RTOL']*abs(v) for v,w in zip(seu+ssu,eu+su)),'independent observed interval coordinates')
    slow,shigh=sorted(range(2),key=lambda k:(seu[k],k));scale=max(1.,*(abs(v) for p in (A,B,*support) for v in p));bound=64.*math.ulp(scale)
    if (low,high)!=(slow,shigh):
        require(max(abs(eu[0]-eu[1]),abs(seu[0]-seu[1]),*(abs(v-w) for v,w in zip(eu,seu)))<=bound,'endpoint order differs outside coordinate roundoff')
        STATE['events'].append(dict(**STATE['context'],edge=line['edge'],modeled=modeled,kind='OBSERVED_INTERVAL_ORDER_CUSP',bound_px=bound,independent_endpoint_u=eu,stored_endpoint_u=seu))
    strict=lambda u,v,l,r:'BEFORE_OBSERVED_INTERVAL' if u<v[l] else 'AFTER_OBSERVED_INTERVAL' if u>v[r] else 'INSIDE_OBSERVED_INTERVAL'
    branches=[]
    for i,(u,stored) in enumerate(zip(su,ssu)):
        expected=strict(u,eu,low,high);choice=strict(stored,seu,slow,shigh)
        if block is not None:require(block['hinge_branches'][i]==choice,'stored hinge string matches strict observed-coordinate predicate')
        if choice!=expected:
            before={choice,expected}=={'BEFORE_OBSERVED_INTERVAL','INSIDE_OBSERVED_INTERVAL'}
            after={choice,expected}=={'AFTER_OBSERVED_INTERVAL','INSIDE_OBSERVED_INTERVAL'}
            which=slow if before else shigh;ind=low if before else high
            require((before or after) and max(abs(stored-seu[which]),abs(u-eu[ind]),abs(stored-u),abs(seu[which]-eu[ind]))<=bound,'observed hinge branch mismatch outside coordinate roundoff')
            STATE['events'].append(dict(**STATE['context'],edge=line['edge'],query_id=line['support_query_ids'][i],modeled=modeled,kind='OBSERVED_SUPPORT_ENDPOINT_CUSP',independent_u_px=u,stored_u_px=stored,roundoff_bound_px=bound,stored_one_sided_branch=choice))
        branches.append(choice)
    if block is not None:
        require(block['interval_endpoint_indices']==[slow,shigh] and block['interval_order_cusp']==(seu[0]==seu[1]) and block['endpoint_hit_cusps']==[u in seu for u in ssu],'strict stored cusp metadata')
    return slow,shigh,branches,block

def finite_blocks(uv,J,q,ids,lines,h,modeled=False,scoring=False,branch_witness=None):
    residual=[];matrix=[];blocks=[]
    for k in ids:residual.extend(uv[k][i]-q[k][i] for i in range(2));matrix.extend(J[k])
    for line in lines:
        a,b=h['EDGES'][line['edge']];A,B=uv[a],uv[b];L=math.dist(A,B);support=line['source_line']['support_points'];N=len(support);start=len(residual)
        base=dict(edge=line['edge'],endpoints=[a,b],scalar_start=start,scalar_end=start+2+N,support_query_ids=line['support_query_ids'],
            actual_support_xy=support,query_radii_px=line['source_line']['query_radii_px'],radius_gate_applied=False,
            physical_ownership_truth='UNKNOWN',block_factor_count=1,modeled_normals_held_fixed=modeled)
        if scoring and L<=1e-12:
            residual.extend([None]*(2+N));base.update(score_state='DEGENERATE_PROJECTED_EDGE_OUTLIER',normal_signed_px=None,
                outside_hinges_px=None,longitudinal_support_px=None,projected_segment_length_px=L);blocks.append(base);continue
        require(math.isfinite(L) and L>1e-12 and N>=3 and h['finite'](support,(N,2)),'finite actual edge support')
        normal=line['normal'];require(h['finite'](normal,(2,)) and abs(math.hypot(*normal)-1.)<=1e-10,'unit observed normal')
        tangent=[-normal[1],normal[0]];eu=[h['dot'](p,tangent) for p in (A,B)];su=[h['dot'](p,tangent) for p in support]
        low,high,branches,stored=observed_choice(eu,su,A,B,support,line,modeled,branch_witness,h)
        n=[-(B[1]-A[1])/L,(B[0]-A[0])/L] if modeled else normal
        signed=[h['dot'](n,p)-line['offset'] for p in (A,B)]
        residual.extend(v/math.sqrt(2.) for v in signed)
        matrix.extend([[h['dot'](n,[J[k][i][j] for i in range(2)])/math.sqrt(2.) for j in range(6)] for k in (a,b)])
        du=[[h['dot'](tangent,[J[k][i][j] for i in range(2)]) for j in range(6)] for k in (a,b)]
        hinges=[]
        for u,branch in zip(su,branches):
            endpoint=low if branch=='BEFORE_OBSERVED_INTERVAL' else high
            outside=branch!='INSIDE_OBSERVED_INTERVAL';value=u-eu[endpoint] if outside else 0.
            hinges.append(value);residual.append(value/math.sqrt(N));matrix.append([-v/math.sqrt(N) for v in du[endpoint]] if outside else [0.]*6)
        # Discrete cusp metadata follows independently checked stored coordinates,
        # while all residual/J values above are independently reconstructed.
        seu=stored['endpoint_u'] if stored is not None else eu;ssu=stored['support_u'] if stored is not None else su
        base.update(observed_tangent=tangent,endpoint_u=eu,interval_endpoint_indices=[low,high],
            interval_order_cusp=seu[0]==seu[1],support_u=su,outside_hinges_px=hinges,normal_signed_px=signed,
            hinge_branches=branches,endpoint_hit_cusps=[u in seu for u in ssu],new_independent_point_correspondences=0,
            physical_semantic_line_correctness_certified=False,coordinate_frame='FIXED_OBSERVED_LINE',
            historical_projected_tangent_objective_reused=False,projected_segment_length_px=L,
            observed_interval_length_px=abs(eu[1]-eu[0]),longitudinal_support_px=[u-min(eu) for u in su],
            longitudinal_coordinate_frame='FIXED_OBSERVED_TANGENT_RELATIVE_TO_LOWER_PROJECTED_ENDPOINT',
            containment_tangent_uses_observed_normal_even_when_modeled=True)
        if scoring:base['score_state']='FINITE_EDGE_BLOCK'
        blocks.append(base)
    return residual,matrix,blocks

def score_check(w,uv,J,row,U,lines,h,a,label):
    rr,_,blocks=finite_blocks(uv,J,row['input_points'],U,lines,h,scoring=True,branch_witness=w['edge_blocks'])
    point=[math.dist(uv[k],row['input_points'][k]) for k in U]
    edge=[None if b['score_state']=='DEGENERATE_PROJECTED_EDGE_OUTLIER' else math.sqrt(math.fsum(v*v for v in rr[b['scalar_start']:b['scalar_end']])) for b in blocks]
    for key,value in (('point_residual_norm_px',point),('edge_block_norm_px',edge),('edge_blocks',blocks),('scalar_residuals_px',rr)):a.near(w[key],value,label+'.'+key)
    ip=[k for k,e in zip(U,w['point_residual_norm_px']) if e<=8.];il=[l['edge'] for l,e in zip(lines,w['edge_block_norm_px']) if e is not None and e<=8.]
    norms=point+edge
    expected=dict(point_ids=U,line_edges=[l['edge'] for l in lines],final_inliers=ip,final_line_inliers=il,inlier_count=len(ip),
        unique_line_inlier_count=len(il),total_inlier_factor_count=len(ip)+len(il),factor_sse_px2=None if None in norms else math.fsum(v*v for v in norms),
        factor_truncated_sse_px2=math.fsum(64. if v is None else min(v*v,64.) for v in norms),score_only_degenerate_edge_is_outlier=True,fixed_full_pool=True)
    for key,value in expected.items():a.near(w[key],value,label+'.'+key)
    return ip,il

def input_paths(args):
    repo,p,private=args.repo,args.input,args.private;doc=repo/'_docs/experiments';parent=doc/'pallet_three_head_observation_20261010_v7'
    f=dict(geometry=p/'GEOMETRY.jsonl.gz',ledgers=p/'LEDGERS.jsonl.gz',geometry_protocol=p/'GEOMETRY_PROTOCOL.json',
        geometry_started=p/'GEOMETRY_STARTED.json',geometry_receipt=p/'GEOMETRY_RECEIPT.json',checker=Path(__file__).resolve(),
        cohort=doc/'pallet_kp_corrected_supervision_20261010_v1/COHORT.json',parent_protocol=parent/'PROTOCOL.json',
        parent_seal=parent/'GEOMETRY_SEAL.json',parent_receipt=parent/'INFERENCE_RECEIPT.json',parent_geometry=parent/'GEOMETRY_SEALED.jsonl.gz',
        parent_observations=parent/'OBSERVATIONS.jsonl.gz',parent_fixed_geometry=parent/'FIXED_GEOMETRY_SEALED.jsonl.gz',
        parent_completion=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/PARENT_POPULATION_CHECKS.json',
        old_failed_validation=doc/'pallet_hybrid_factor_subset_consensus_20261011_v16/VALIDATION_READY_CHECKS.json',
        driver=private/'run_geometry_ready.py',original_driver=private/'run_geometry.py',serialization_ready_proof=private/'GEOMETRY_READY_PROOF.json')
    for key,(relative,_) in MATH_SOURCES.items():f[key]=repo/'scripts/research'/relative
    for name in ('solver.py','pipeline.py','geometry.py'):f['engine:'+name]=repo/'scripts/research'/FAMILY/name
    for name in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json'):f['cpu:'+name]=private/'cpu'/name
    for name in ('GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json','GEOMETRY_RECEIPT.json'):f['original:'+name]=private/'geometry'/name
    # The original input closure includes byte-only declarations; exact aliases
    # map them to public research source and fixed saved materials, never GT.
    protocol=read(f['geometry_protocol'])
    for key in protocol['inputs']:
        if key.startswith('source:'):f['geometry_input:'+key]=repo/key.removeprefix('source:')
    f['point_control']=doc/'pallet_consensus_decision_repair_20261011_v18/GEOMETRY_SEALED.jsonl.gz'
    return f

def literal_policy_checks(path,policy):
    tree=ast.parse(path.read_bytes());node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    require(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args,'declared policy dict')
    for keyword in node.keywords:
        require(keyword.arg in policy,'complete policy keyword '+keyword.arg)
        try:value=ast.literal_eval(keyword.value)
        except (ValueError,TypeError):continue
        require(policy[keyword.arg]==value,'source literal policy '+keyword.arg)
    require(set(policy)=={k.arg for k in node.keywords},'exact source policy keys')

def metadata(f,b):
    core,r,start,cohort,pop,seal,old=[read(f[k]) for k in ('geometry_protocol','geometry_receipt','geometry_started','cohort','parent_completion','parent_seal','old_failed_validation')]
    require(core['schema']=='observed_frame_repair_geometry_protocol_v19' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490 and core['configured_before_geometry'] is True,'fixed paired geometry protocol')
    require(r['schema']=='observed_frame_repair_geometry_receipt_v19' and r['complete'] is True and r['error'] is None and r['cleanup_errors']==[] and r['inputs_unchanged'] is r['environment_restored'] is True,'complete sealed geometry/cleanup')
    require(r['inputs']==core['inputs']==start['inputs'] and same(r['protocol'],b['geometry_protocol']) and same(start['protocol'],b['geometry_protocol']) and same(r['geometry'],b['geometry']) and same(r['ledgers'],b['ledgers']),'full current seal bytes')
    require(r['protection_before']==r['protection_after']==core['protection']==start['protection'] and core['environment']==start['environment'],'source protection and environment')
    require(r['limits']==core['limits']==start['limits'] and core['limits']['GT_allowed'] is False and core['limits']['old564_scalar_FAIL_retained'] is True and core['limits']['score_only_after_independent_full_geometry_PASS'] is True,'GT-free run scope')
    mapping={'PROTOCOL.json':'parent_protocol','GEOMETRY_SEAL.json':'parent_seal','INFERENCE_RECEIPT.json':'parent_receipt','GEOMETRY_SEALED.jsonl.gz':'parent_geometry','OBSERVATIONS.jsonl.gz':'parent_observations','FIXED_GEOMETRY_SEALED.jsonl.gz':'parent_fixed_geometry','cohort':'cohort','completion':'parent_completion','old_failed_validation':'old_failed_validation','point_control':'point_control','runner':'driver'}
    for name in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json'):mapping['CPU:'+name]='cpu:'+name
    for key,declared in core['inputs'].items():
        alias='geometry_input:'+key if key.startswith('source:') else mapping[key]
        require(same(declared,b[alias]),'every frozen geometry input '+key)
    require(cohort['counts']==dict(clean=153,moderate=92,severe_excluded=74,original=319) and len(cohort['ids'])==len(set(cohort['ids']))==245 and core['ordered_ids']==cohort['ids'],'fixed original319/excluded74/paired245')
    require(seal['complete'] is True and seal['frames']==245 and seal['rows']==1960 and seal['fixed_rows']==490 and seal['GT_read_allowed'] is False,'original full GT-free source seal')
    for key,alias in (('protocol','parent_protocol'),('geometry','parent_geometry'),('observations','parent_observations'),('fixed_geometry','parent_fixed_geometry')):require(same(seal[key],b[alias]),'original sealed source '+key)
    pr=read(f['parent_receipt']);require(pr['complete'] is True and pr['inference_error'] is pr['cleanup_error'] is None and pr['GT_access_during_inference'] is False,'clean parent completion')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and pop['ordered_ids']==cohort['ids'] and pop['completion']['complete'] is True,'full parent population')
    done=pop['completion'];require(done['protocol_sha256']==b['parent_protocol']['sha256'] and done['geometry_sha256']==b['parent_geometry']['sha256'] and done['inference_receipt_sha256']==b['parent_receipt']['sha256'],'complete source identity')
    require(old['complete'] is True and old['passed'] is False and old['checks']==198134007 and old['failure_count']==564 and old['exception'] is None,'retain original completed full FAIL')
    cpu,cp,cs=[read(f['cpu:'+name]) for name in ('CPU_CHECKS.json','CPU_PROTOCOL.json','CPU_STARTED.json')]
    require(cpu['complete'] is True and cpu['passed'] is True and cpu['tests_run']==6 and cpu['failures']==cpu['errors']==[] and cpu['exception'] is None and cpu['source_unchanged'] is True,'actual synthetic CPU6 PASS')
    require(cp['schema']=='observed_frame_actual_CPU_protocol_v19' and cpu['schema']=='observed_frame_actual_CPU_checks_v19' and same(cpu['protocol'],b['cpu:CPU_PROTOCOL.json']) and same(cs['protocol'],b['cpu:CPU_PROTOCOL.json']) and cpu['source_bindings']==cp['inputs']==cs['inputs'],'CPU same frozen source')
    for key,v in cpu['source_bindings'].items():require(same(v,b['geometry_input:source:'+key]),'CPU-current geometry source '+key)
    first,fp,fs=[read(f['original:'+name]) for name in ('GEOMETRY_RECEIPT.json','GEOMETRY_PROTOCOL.json','GEOMETRY_STARTED.json')]
    require(first['complete'] is False and first['error']['type']=='TypeError' and 'ndarray' in first['error']['message'] and first['actual_execution'].get('geometry_rows',0)==first['actual_execution'].get('ledger_rows',0)==0 and first['actual_execution']['frames_started']==1,'preserved first ndarray serialization failure')
    require(first['inputs']==fp['inputs']==fs['inputs'] and same(fp['inputs']['runner'],b['original_driver']) and same(first['protocol'],b['original:GEOMETRY_PROTOCOL.json']) and same(fs['protocol'],b['original:GEOMETRY_PROTOCOL.json']),'original zero-serialized failure identity')
    require(first['cleanup_errors']==[] and first['environment_restored'] is first['inputs_unchanged'] is True and first['protection_before']==first['protection_after']==fp['protection'],'original failed attempt clean restoration')
    require(first['actual_numerical_calls']['V19.solve']==dict(entries=2,returns=2,exceptions=0) and first['actual_numerical_calls']['least_squares']['entries']==46,'preserved first attempt actual workload; not zero math')
    proof=read(f['serialization_ready_proof']);require(proof['schema']=='V19_serialization_only_ready_proof' and same(proof['original'],b['original_driver']) and same(proof['ready'],b['driver']) and same(proof['original_first_failed_receipt'],b['original:GEOMETRY_RECEIPT.json']),'explicit original/ready/failed receipt byte proof')
    require(proof['inverse_whole_AST_equal'] is True and proof['changed_calls']==2 and proof['solver_point_line_observations_numeric_parameters_changed'] is False and proof['failed_frame_solve_entries']==2 and proof['failed_optimizer_entries']==46 and proof['failed_geometry_rows']==0 and proof['automatic_retry'] is False and proof['manual_specific_infrastructure_repair'] is True,'explicit manual serialization-only scope')
    original,ready=ast.parse(f['original_driver'].read_bytes()),ast.parse(f['driver'].read_bytes());changes=0
    for node in ast.walk(ready):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='common' and node.func.attr=='finite' and len(node.args)==1 and isinstance(node.args[0],ast.Name) and node.args[0].id in ('row','ledger'):
            replacement=node.args[0]
            for parent in ast.walk(ready):
                for field,value in ast.iter_fields(parent):
                    if isinstance(value,list):
                        for i,item in enumerate(value):
                            if item is node:value[i]=replacement;changes+=1
                    elif value is node:setattr(parent,field,replacement);changes+=1
    require(changes==2 and canonical(ready)==canonical(original),'only two finite JSON serialization wrappers; whole driver inverse equal')
    return core,r,pop,cohort,seal

def new_source_checks(two,ledger,f,h,a,label):
    policy=two[0]['hybrid_consensus_policy'];literal_policy_checks(f['engine:pipeline.py'],policy);literal_policy_checks(f['engine:solver.py'],policy['solver'])
    a.eq(policy['methods'],list(METHODS),label+'.policy_methods');a.eq(policy['primary'],METHODS[0],label+'.policy_primary')
    decision=policy['decision_policy_both_views'];a.eq(decision['numerical_score_tie_px2'],1e-8,label+'.inherited_true_tie')
    a.eq(decision['same_support_different_cost_veto_removed'],True,label+'.same_support_is_not_extra_veto')
    for row in two:
        a.eq(row['finite_observation_policy'],policy,label+'.finite_policy');a.eq(row['solver']['operative_output_decision_policy'],policy['decision_policy_both_views'],label+'.decision_policy')
        proof=row['solver']['observed_frame_source_proof'];a.eq(proof['parent_globals_mutated'],False,label+'.no_parent_mutation')
        for key in ('finite_callback_inverse_whole_AST_equal','geometry_and_optimizer_method_bodies_whole_AST_unchanged','score_inverse_whole_AST_equal','solve_inverse_whole_AST_equal','point_only_callbacks_delegate_frozen_parent'):a.eq(proof[key],True,label+'.source_proof.'+key)
        a.eq(row['solver']['new_finite_objective_is_not_historical_scalar_failure_repair'],True,label+'.new_objective')
        if row['solver']['state']=='AMBIGUOUS_FINITE_CONSENSUS':a.eq(row['solver']['reason'],'NUMERICAL_CONSENSUS_TIE',label+'.only_true_numeric_tie_reason')
        for key,v in proof['source_bindings'].items():
            public=[bound for name,bound in h['frozen_source_bindings'].items() if name.startswith('geometry_input:source:') and same(bound,v) and Path(bound['path']).name==Path(v['path']).name]
            require(len(public)==1,'solver source proof in exact frozen source closure '+key)
        pp=row['observed_frame_pipeline_source_proof'];a.eq(pp['parent_globals_mutated'],False,label+'.pipeline_parent_mutation')
        for key in ('V15_repair_whole_function_AST_unchanged','V16_paired_solve_whole_function_AST_unchanged','V16_hybrid_packet_validation_direct_delegate','fresh_native_endpoint_body_unchanged','same_q_H_and_shared_raw4_bank_body_unchanged'):a.eq(pp[key],True,label+'.pipeline_proof.'+key)
        for key,v in pp['source_bindings'].items():
            public=[bound for name,bound in h['frozen_source_bindings'].items() if name.startswith('geometry_input:source:') and same(bound,v) and Path(bound['path']).name==Path(v['path']).name]
            require(len(public)==1,'pipeline source proof in exact frozen source closure '+key)
    a.eq(two[1]['hybrid_consensus_policy'],policy,label+'.same_policy');a.eq(ledger['finite_observation_policy'],policy,label+'.ledger_policy')
    a.eq(ledger['old_full_V16_564_FAIL_remains_FAIL'],True,label+'.old_FAIL_preserved');a.eq(ledger['schema'],'observed_frame_hybrid_factor_subset_paired_ledger_v19',label+'.new_ledger_schema')
    a.eq(ledger['point_line_ledger']['schema'],'observed_frame_endpoint_filtered_solver_ledger_v19',label+'.line_ledger_schema')
    h['execution_policy']=policy

def guard(args,stage):
    for p in (args.repo,args.input,args.output,args.private):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink folders')
    require(args.output!=args.input,'own checker output folder separate from geometry')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)
def freeze(args):
    guard(args,'freeze');f=input_paths(args);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,info=extract(f)
    write(args.output/FILES[0],dict(schema='observed_frame_hybrid_scalar_protocol_v19',inputs=b,checker=b['checker'],helpers=info,methods=list(METHODS),frames=245,rows=490,ledger_rows=245,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=input_paths(args);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);trees,info=extract(f)
    require(own['schema']=='observed_frame_hybrid_scalar_protocol_v19' and own['inputs']==b and own['helpers']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own exact frozen closure')
    core,r,pop,cohort,seal=metadata(f,b);write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['checker'],new_production_calls=0))
    h,n,policy=helpers(trees,f);h['frozen_source_bindings']=b;a=h['Audit']();counts=Counter();native_counts=Counter();error=None
    try:
        ordered=cohort['ids'];a.eq([x['id'] for x in cohort['frames']],ordered,'ordered_cohort')
        geometry,ledgers=iter(rows(f['geometry'])),iter(rows(f['ledgers']))
        parents=iter(h['primary'](f['parent_geometry'],'method','IMAGE_ROLE_CORNERWISE_HYBRID',1960,set(ordered),a,'V7_geometry'))
        observations=iter(h['primary'](f['parent_observations'],'head_arm','IMAGE_ROLE',735,set(ordered),a,'V7_ROLE'))
        for i,fid in enumerate(ordered):
            two=[next(geometry,None),next(geometry,None)];ledger,parent,obs=[next(s,None) for s in (ledgers,parents,observations)]
            require(all(isinstance(v,dict) and v['id']==fid and v['session']==cohort['frames'][i]['session'] for v in (*two,ledger,parent,obs)),'complete same-frame paired joins')
            label='frame.'+str(i);new_source_checks(two,ledger,f,h,a,label)
            h['packet'](two,ledger,parent,obs,pop['completion'],h,n,a,policy,counts,native_counts,label)
        for stream in (geometry,ledgers,parents,observations):a.eq(next(stream,None),None,'all_populations_exhausted')
        for k,v in dict(frames_started=245,frames_complete=245,geometry_rows=490,ledger_rows=245).items():a.eq(r['actual_execution'][k],v,'actual.'+k)
        calls=r['actual_numerical_calls']
        for name in h['PRIMITIVES']:
            a.eq(calls['cv2.'+name],{k:counts['cv:'+p+':'+name] for k,p in (('entries','entry'),('returns','return'),('exceptions','exception'))},'actual.CV.'+name)
        a.eq(calls['numpy.linalg.svd'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'actual.SVD')
        a.eq(calls['least_squares'],{k:counts['opt:'+k] for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')},'actual.optimizer')
        a.eq(calls['V19.solve'],dict(entries=490,returns=490,exceptions=0),'actual.two_solver_paths')
        a.eq(calls['PoseBank.solve'],dict(entries=counts['endpoint_requests'],returns=counts['endpoint_requests'],exceptions=0),'actual.native_endpoint_paths')
        a.eq(r['states'],{m+':'+s:counts['status:'+m+':'+s] for m in METHODS for s in ('NEW_POSE','N3_BASELINE_FALLBACK','POSE_FAILURE') if counts['status:'+m+':'+s]},'actual.states')
        for snapshot in r['resource_snapshots']:a.eq(snapshot['quiet'],True,'recorded_quiet');a.eq(snapshot['temperature_under_80'],True,'recorded_thermal')
        for k,v in (('frames',245),('views',490)):a.eq(counts[k],v,'full.'+k)
        a.eq(counts['proposal_checks'],counts['stored_proposals'],'all_proposal_math');a.eq(counts['refit_checks'],counts['stored_refits'],'all_refit_math')
        a.eq({k:binding(p) for k,p in f.items()},b,'inputs_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    complete=error is None and counts['frames']==245 and counts['views']==490
    result=dict(schema='observed_frame_hybrid_scalar_checks_v19',complete=complete,passed=complete and a.failure_count==0,
        checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,max_scalar_absolute_difference=a.max_difference,
        inputs=b,own_protocol=ob,checker=b['checker'],counts=dict(counts),native_counts=dict(native_counts),
        actual_counts=dict(frames=counts['frames'],geometry_rows=counts['views'],ledger_rows=counts['frames'],successful_proposals_checked=counts['proposal_checks'],successful_refits_checked=counts['refit_checks']),
        cusp_reconciliations=STATE['events'],limits=LIMITS,new_production_calls=0,old_full_V16_validation_passed=False,
        preserved_first_geometry_attempt=dict(complete=False,actual_solver_entries=2,actual_optimizer_entries=46,serialized_geometry_rows=0,serialization_ready_proof=b['serialization_ready_proof']))
    write(args.output/FILES[2],result);print('OBSERVED_FRAME_SCALAR',result['passed'],a.checks,a.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--repo',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path)
    args=p.parse_args();args.repo=args.repo.resolve();args.input=args.input.resolve();args.output=args.output.resolve();args.private=(args.private or args.input.parent).resolve();{'freeze':freeze,'run':run}[args.stage](args)
if __name__=='__main__':main()
