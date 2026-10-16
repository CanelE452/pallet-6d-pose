"""Own-frozen stdlib scalar check of exhaustive factor-subset consensus.

No production imports, inference, image/GT reads, PnP, optimization or SVD.
Independent projection/J and finite hinges are reconstructed for every saved
proposal/refit. Stored spectra are recounted, never regenerated. Near a hinge
cusp only the inherited 64-coordinate-ulp reconciliation is permitted; its
one-sided derivative is not an observability equality. Normal incidence is
checked independently for every candidate submitted to the V14 guard.
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
DOC=REPO/'_docs/experiments/pallet_factor_subset_consensus_20261011_v15'
PRIVATE=Path('/tmp/pallet-factor-subset-consensus-private-20261011-v15')
PARENT=REPO/'_docs/experiments/pallet_three_head_observation_20261010_v7'
AUDIT=REPO/'_docs/experiments/pallet_unused_line_endpoint_audit_20261011_v10'
COHORT=REPO/'_docs/experiments/pallet_kp_corrected_supervision_20261010_v1/COHORT.json'
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
SOURCE_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
FINITE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/validation_checks.py'
FINITE_SHA='91d014543eba01c142cad9a0c69affd2eb16666913ce13a06a56857f854c98ef'
GUARD=REPO/'scripts/research/pallet_stable_incidence_guard_20261011_v14/pipeline.py'
METHOD='FACTOR_SUBSET_FINITE_EDGE_CONSENSUS'
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(own_freeze_after_geometry_before_own_arithmetic=True,new_production_calls=0,
    all_saved_successful_proposal_and_refit_scalar_checks=True,stored_spectrum_recount_only=True,
    failed_direct_root_geometry_not_reproducible_without_saved_Rt=True,
    mixed_LOCAL_root_completeness_proven=False,physical_ownership_certified=False,
    hardware_authenticity_proven=False,quiet_snapshots_can_miss_transient_contention=True,
    cusp_allowance_coordinate_ulps=64,cusp_reconciliation_not_a_performance_gate=True,
    one_sided_hinge_J_not_observability_equality=True,automatic_retry=False)
STATE=dict(context={},events=[])

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank raw row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));d=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):d.update(block)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=d.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())

def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA and binding(FINITE)['sha256']==FINITE_SHA,'immutable independent scalar sources')
    names={'require','safe','Audit','finite','dot','mm','eye','skew','rotation_and_right_J','dimensions','model','forward','physical_rotation','equivalent','eligible','factor_rows','rank_witness','original_lines','semantic'}
    constants={'EDGES','PHYSICAL','CAL','ATOL','RTOL','RANK','TIE','PRIMITIVES'};nodes=[];found=set();assigned=set()
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n);found.add(n.name)
        elif isinstance(n,ast.Assign):
            ids={q.id for t in n.targets for q in ast.walk(t) if isinstance(q,ast.Name)}
            if ids and ids<=constants:nodes.append(n);assigned.update(ids)
    require(found==names and assigned==constants,'exact independent helper whitelist')
    old=ast.parse(FINITE.read_text());selected=[n for n in old.body if isinstance(n,ast.FunctionDef) and n.name in ('finite_blocks','score_check','primary')]
    require({n.name for n in selected}=={'finite_blocks','score_check','primary'},'exact finite helper whitelist')
    fn=next(n for n in selected if n.name=='finite_blocks');original=copy.deepcopy(fn)
    branch=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.dump(n.test)==ast.dump(ast.parse('u<0',mode='eval').body)]
    require(len(branch)==1 and ast.dump(branch[0].orelse[0].test)==ast.dump(ast.parse('u>L',mode='eval').body),'unique strict hinge predicates')
    branch=branch[0];after=branch.orelse[0];bt,at=copy.deepcopy(branch.test),copy.deepcopy(after.test)
    parent=next(n for n in ast.walk(fn) if hasattr(n,'body') and isinstance(n.body,list) and any(x is branch for x in n.body));pos=parent.body.index(branch)
    stmt=ast.parse('_choice=_branch(u,L,xy,A,B,branch_witness,len(along),line,modeled)').body[0]
    def patch():
        fn.args.args.append(ast.arg(arg='branch_witness'));fn.args.defaults.append(ast.Constant(None));parent.body.insert(pos,stmt)
        branch.test=ast.parse("_choice=='BEFORE_FIRST_ENDPOINT'",mode='eval').body;after.test=ast.parse("_choice=='AFTER_SECOND_ENDPOINT'",mode='eval').body
    patch();patched=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
    fn.args.args.pop();fn.args.defaults.pop();parent.body.remove(stmt);branch.test=bt;after.test=at
    require(ast.dump(fn,include_attributes=False)==ast.dump(original,include_attributes=False),'undeclared finite AST changes');patch()
    tree=ast.fix_missing_locations(ast.Module(body=nodes+selected,type_ignores=[]))
    node=next(n.value for n in ast.parse(GUARD.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    require(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args and all(k.arg is not None for k in node.keywords),'literal unchanged guard policy')
    policy={k.arg:ast.literal_eval(k.value) for k in node.keywords}
    require(policy['numerical_rank_relative_threshold']==1e-10 and policy['weak_relative_threshold']==1e-4 and policy['hinge_rank_rows_retained']==0,'unchanged normal guard thresholds')
    return tree,policy,dict(source=binding(SOURCE),finite_source=binding(FINITE),guard_source=binding(GUARD),
        exact_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),finite_patch_sites=4,finite_patched_AST_sha256=patched)

def _branch(u,L,xy,A,B,witness,index,line,modeled):
    strict=lambda v,l:'BEFORE_FIRST_ENDPOINT' if v<0 else 'AFTER_SECOND_ENDPOINT' if v>l else 'INSIDE_CLOSED_SEGMENT'
    calculated=strict(u,L)
    if witness is None:return calculated
    block=next(b for b in witness if b['edge']==line['edge']);su=block['longitudinal_support_px'][index];sl=block['projected_segment_length_px'];choice=block['hinge_branches'][index]
    require(all(type(v) in (int,float) and math.isfinite(v) for v in (u,L,su,sl)) and min(L,sl)>1e-12,'finite cusp values')
    require(abs(su-u)<=1e-7+1e-12*abs(su) and abs(sl-L)<=1e-7+1e-12*abs(sl) and choice==strict(su,sl),'independent u/L and strict stored branch')
    if calculated==choice:return choice
    scale=max(1.,L,sl,*(abs(v) for p in (xy,A,B) for v in p));bound=64.*math.ulp(scale)*(1.+math.dist(xy,A)/L)
    before={calculated,choice}=={'BEFORE_FIRST_ENDPOINT','INSIDE_CLOSED_SEGMENT'};after={calculated,choice}=={'AFTER_SECOND_ENDPOINT','INSIDE_CLOSED_SEGMENT'}
    require(abs(sl-L)<=bound and ((before and max(abs(u),abs(su))<=bound) or (after and max(abs(u-L),abs(su-sl))<=bound)),'cusp branch mismatch outside coordinate rounding bound')
    STATE['events'].append(dict(**STATE['context'],edge=line['edge'],query_id=line['support_query_ids'][index],modeled=modeled,independent_u_px=u,stored_u_px=su,independent_L_px=L,stored_L_px=sl,roundoff_bound_px=bound,stored_one_sided_branch=choice))
    return choice

def ranks(g,observed,modeled,ids,h,a,label):
    split=2*len(ids);matrices=dict(observed_normal_joint=observed,modeled_normal_joint=modeled,point_only=modeled[:split],observed_normal_lines_only=observed[split:],modeled_normal_lines_only=modeled[split:])
    rank={k:h['rank_witness'](g[k],m,a,label+'.'+k) for k,m in matrices.items()};weak={}
    for name in ('observed','modeled'):
        s=g[name+'_normal_joint']['singular_values'];weak[name]=len(s)!=6 or s[-1]<s[0]*1e-4
    return rank,weak

def incidence(c,row,U,H,lines,h,a,policy,label):
    w=c['stable_incidence_guard'];before=copy.deepcopy(c);before.pop('stable_incidence_guard');before.update(accepted=True,rejected=None)
    ids,edges=c['fit_input_ids'],c['fit_line_edges'];selected=[dict(edge=l['edge'],normal=l['normal'],offset=l['offset']) for l in lines if l['edge'] in edges]
    temporary=dict(available=True,state='CANDIDATE_FOR_STABLE_INCIDENCE',used=U,fit_input_ids=ids,fit_line_edges=edges,consumed_edges=row['solver']['consumed_edges'],line_edges=[l['edge'] for l in lines],actual_source_lines=[dict(edge=l['edge'],normal=l['normal'],offset=l['offset']) for l in lines],centroid=c['centroid'],cf_extents=c['cf_extents'],R_cf=c['R_cf'],R_physical=c['R_physical'],projected=c['projected'],best_candidate=before,all_candidate_solutions=[before])
    for key,value in (('policy',policy),('selected_candidate',before),('original_solver_semantic_sha256',h['semantic'](temporary)),('original_all_candidates_semantic_sha256',h['semantic']([before])),('actual_fit_input_ids',ids),('actual_fit_line_edges',edges),('hidden',H),('actual_normal_lines',selected),('gate_applied',True),('original_numerical_pose_available',True),('selected_candidate_reselected',False),('global_uniqueness_proven',False),('physical_ownership_truth','UNKNOWN')):a.eq(w[key],value,label+'.'+key)
    _,uv,J,_=h['forward'](c['cf_extents'],c['final_z'],row['K']);_,obs=h['factor_rows'](uv,J,row['input_points'],ids,selected);_,mod=h['factor_rows'](uv,J,row['input_points'],ids,selected,True)
    indices=list(range(2*len(ids)));end=2*len(ids)
    for block in c['point_line_geometry']['edge_blocks']:a.eq(block['scalar_start'],end,label+'.partition');indices.extend((end,end+1));end=block['scalar_end']
    for key,value in (('incidence_source_row_indices',indices),('finite_scalar_rows',end),('incidence_scalar_rows',len(indices)),('removed_longitudinal_rows',end-len(indices)),('selected_dimension_index',c['dimension_index']),('selected_extents',c['cf_extents']),('selected_final_z',c['final_z'])):a.eq(w[key],value,label+'.'+key)
    for key,value in (('guard_calls',1),('geometry_info_calls',1),('new_optimizer_calls',0),('new_PnP_calls',0),('actual_rank_info_dispatches_independently_counted',False)):a.eq(w['counts'][key],value,label+'.counts_'+key)
    for key,m in (('observed_normal_J',obs),('modeled_normal_J',mod)):
        a.near(w[key],m,label+'.independent_'+key);a.eq(w[key],[c['point_line_geometry'][key][k] for k in indices],label+'.saved_normal_rows')
    g=w.get('stable_incidence_geometry')
    if g is None:a.eq(w['guard_state'],'STABLE_INCIDENCE_NUMERICAL_FAILURE',label+'.typed_numerical');a.eq(w['accepted'],False,label+'.numerical_unavailable');require(isinstance(w['error'],dict),'numerical guard error');return False
    rank,weak=ranks(g,obs,mod,ids,h,a,label+'.normal_geometry');rank_ok=rank['observed_normal_joint']==rank['modeled_normal_joint']==6
    state='PASSED_STABLE_INCIDENCE' if rank_ok and not any(weak.values()) else 'STABLE_INCIDENCE_INSUFFICIENT' if not rank_ok else 'STABLE_INCIDENCE_WEAK'
    for key,value in (('guard_state',state),('accepted',state=='PASSED_STABLE_INCIDENCE'),('observed_weak',weak['observed']),('modeled_weak',weak['modeled'])):a.eq(w[key],value,label+'.'+key)
    return w['accepted']

def candidate(c,row,U,H,lines,h,a,policy,counts,label):
    STATE['context']=dict(id=row['id'],phase=c['phase'],label=label);d=c['dimension_index'];dims=h['dimensions'](row['xyz']);require(type(d) is int and 0<=d<len(dims),'registered candidate dimension')
    ids,edges=c['fit_input_ids'],c['fit_line_edges'];selected=[l for l in lines if l['edge'] in edges]
    a.eq(ids,[k for k in U if k in ids],label+'.ordered_actual_ids');a.eq(edges,[l['edge'] for l in selected],label+'.ordered_actual_edges');a.ok(not set(ids)&set(H) and 8 not in ids,label+'.no_H_center');a.eq(c['cf_extents'],dims[d],label+'.dimension');a.eq(c['selected_hypothesis'],('REGISTRY_WD','REGISTRY_DW')[d],label+'.registry_name')
    R,uv,J,depths=h['forward'](dims[d],c['final_z'],row['K'])
    for key,value in (('R_cf',R),('R_physical',h['physical_rotation'](R,d)),('centroid',c['final_z'][3:]),('projected',uv),('model_camera_depths_m',depths)):a.near(c[key],value,label+'.'+key)
    g=c['point_line_geometry'];rr,observed,blocks=h['finite_blocks'](uv,J,row['input_points'],ids,selected,h,branch_witness=g['edge_blocks']);_,modeled,_=h['finite_blocks'](uv,J,row['input_points'],ids,selected,h,modeled=True,branch_witness=g['edge_blocks'])
    for key,value in (('point_ids',ids),('line_edges',edges),('modeled_normals_held_fixed_for_derivative',True),('global_unique_pose_proven',False)):a.eq(g[key],value,label+'.geometry_'+key)
    for key,value in (('observed_normal_J',observed),('modeled_normal_J',modeled),('actual_scalar_residuals_px',rr),('edge_blocks',blocks)):a.near(g[key],value,label+'.independent_'+key)
    rank,weak=ranks(g,observed,modeled,ids,h,a,label+'.finite_geometry')
    for name in weak:a.eq(g[name+'_weak'],weak[name],label+'.'+name+'_weak')
    cost=math.fsum(64.*(math.sqrt(1.+(v/8.)**2)-1.) for v in rr)
    for key,value in (('scalar_residuals_px',rr),('soft_l1_cost',cost),('sse_px2',math.fsum(v*v for v in rr))):a.near(c[key],value,label+'.'+key)
    direct=c['phase']=='direct_point_proposal'
    for key,value in (('initial_pose_residual_prior',False),('global_uniqueness_proven',False),('local_only',not direct),('optimizer_success',None if direct else True)):a.eq(c[key],value,label+'.'+key)
    if direct:a.eq(c['actual_optimizer_called'],False,label+'.no_direct_optimizer');a.eq(len(ids),4,label+'.four_actual_ids');a.eq(edges,[],label+'.point_root_no_lines')
    else:a.near(c['reported_soft_l1_cost'],cost,label+'.actual_reported_cost')
    ip,il=h['score_check'](c['whole_pool_score'],uv,J,row,U,lines,h,a,label+'.whole_pool')
    if c['phase']=='refit':
        stable=set(ids)<=set(ip) and set(edges)<=set(il);rank_ok=rank['observed_normal_joint']==rank['modeled_normal_joint']==6
        reason='UNSTABLE_INLIER_REFIT' if not stable else 'INSUFFICIENT_ACTUAL_INLIER_GEOMETRY' if not rank_ok else 'WEAK_OBSERVABILITY' if any(weak.values()) else None
        for key,value in (('whole_pool_point_ids',U),('whole_pool_line_edges',[l['edge'] for l in lines]),('final_inliers',ip),('final_line_inliers',il),('fit_pool_all_final8px_inliers',stable),('score_count',len(ip)+len(il)),('actual_inlier_geometry_is_acceptance_gate',True),('whole_pool_scoring_factors_not_all_fit_observations',True)):a.eq(c[key],value,label+'.'+key)
        a.near(c['truncated_sse_px2'],c['whole_pool_score']['factor_truncated_sse_px2'],label+'.truncated')
        if reason is None:
            counts['incidence_checks']+=1
            if not incidence(c,row,U,H,lines,h,a,policy,label+'.normal_incidence'):reason=c['stable_incidence_guard']['guard_state']
        else:a.eq(c['stable_incidence_guard_not_called_reason'],'ORIGINAL_FINITE_CANDIDATE_GATE_REJECTED',label+'.no_guard_after_finite_reject');a.ok('stable_incidence_guard' not in c,label+'.no_undeclared_guard')
        a.eq(c['accepted'],reason is None,label+'.accepted');a.eq(c['rejected'],reason,label+'.rejected')
    counts['proposal_checks' if c['phase']!='refit' else 'refit_checks']+=1

def packet(row,ledger,parent,obs,endpoint,completion,h,a,policy,counts,label):
    q,H,s=parent['input_points'],parent['hidden_initial'],row['solver'];U=[k for k in h['eligible'](q,parent['raw_hw']) if k not in H]
    local_packet=dict(parent_row=parent,role_observation=obs,native_N3_points=obs['native_N3_points'],parent_completion=completion)
    a.eq(ledger['input_packet_semantic_sha256'],h['semantic'](dict(local_packet=local_packet,audit_row=endpoint)),label+'.packet_SHA')
    for key,value in (('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('original_ROLE_observation_semantic_sha256',h['semantic'](obs)),('original_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points']))):a.eq(endpoint[key],value,label+'.endpoint_source.'+key)
    rejected=[r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='CONTRADICTED_REJECT'];filtered=copy.deepcopy(obs)
    if endpoint['contract_error'] is None:filtered['lines']=[l for l in filtered['lines'] if l['edge'] not in rejected]
    adopted=parent['observation_contract']['validated_boundary_corner_ids'];rawlines,consumed=(h['original_lines'](filtered,q,U,adopted,a,label+'.source') if endpoint['contract_error'] is None else ([],[]))
    stored={l['edge']:l for l in s.get('actual_source_lines',[])};source={l['edge']:l for l in filtered['lines']};lines=[]
    for l in rawlines:
        w=stored[l['edge']];a.near(w['normal'],l['normal'],label+'.normal');a.near(w['offset'],l['offset'],label+'.offset');a.eq(w['raw'],source[l['edge']],label+'.actual_support_source')
        lines.append(dict(edge=l['edge'],normal=w['normal'],offset=w['offset'],support_query_ids=l['support_query_ids'],source_line=w['raw']))
    edges=[l['edge'] for l in lines];a.eq(sorted(stored),edges,label+'.complete_line_sources');a.eq(s,ledger['full_solver'],label+'.actual_solver')
    for key,value in (('method',METHOD),('input_points',q),('original_native_N3_points',obs['native_N3_points']),('K',parent['K']),('xyz',parent['xyz']),('raw_hw',parent['raw_hw']),('hidden_initial',H),('initial_pose',parent['initial_pose']),('parent_completion',completion),('parent_geometry_row_semantic_sha256',h['semantic'](parent)),('parent_observation_row_semantic_sha256',h['semantic'](obs)),('parent_native_N3_semantic_sha256',h['semantic'](obs['native_N3_points'])),('endpoint_audit_semantic_sha256',h['semantic'](endpoint))):a.eq(row[key],value,label+'.original.'+key)
    for key in ('selected_index','fixed_metadata','head_arm','head_mode','head_checkpoint_sha256','calibration_binding'):
        if key in parent:a.eq(row[key],parent[key],label+'.source_provenance.'+key)
    a.eq(row['factor_subset_consensus_policy'],h['execution_policy'],label+'.execution_policy');a.eq(s['policy'],h['execution_policy']['solver'],label+'.solver_policy');a.eq(row['selected_corner_ids'],adopted,label+'.original_adopted_ids')
    f=row['observation_filter'];a.eq(f,ledger['observation_filter'],label+'.filter_ledger');a.eq(f['original_factor_pool_binding'],endpoint['factor_pool_binding'],label+'.original_factor_binding')
    for key,value in (('original_observation_semantic_sha256',h['semantic'](obs)),('filtered_observation_semantic_sha256',h['semantic'](filtered)),('endpoint_records',endpoint['endpoint_records']),('rejected_edge_ids',rejected),('retained_unused_edge_ids',edges),('unknown_retained_edge_ids',[r['edge'] for r in endpoint['endpoint_records'] if r['decision']=='UNVERIFIED_RETAIN']),('consumed_edges',consumed)):a.eq(f[key],value,label+'.filter.'+key)
    a.eq(edges,[e for e in endpoint['original_actual_unused_edges'] if e not in rejected],label+'.reject_only')
    for key,value in (('used',U),('hidden',H),('excluded',H),('line_edges',edges),('consumed_edges',consumed)):a.eq(s[key],value,label+'.pool.'+key)
    for key in ('prior_used','initial_pose_residual_prior','initial_dimension_prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','center_used_for_fit_or_score','same_edge_point_line_double_count','mixed_root_completeness_proven'):a.eq(s[key],False,label+'.no_fake.'+key)
    op=s['operation_counts'];proposals=s['proposal_solutions'];cs=s['all_candidate_solutions'];attempts=s['optimizer_attempts'];dims=h['dimensions'](row['xyz']);D=len(dims);F=len(U)+len(lines);M=math.comb(F,3)-math.comb(len(U),3) if F>=3 else 0
    if endpoint['contract_error'] is None:
        encoded=json.dumps(dict(points=[q[k] for k in U],point_ids=U,lines=lines),sort_keys=True,separators=(',',':'),allow_nan=True).encode()
        a.eq(s['factor_pool_binding'],dict(sha256=hashlib.sha256(encoded).hexdigest(),bytes=len(encoded),definition='semantic sorted JSON; not file-byte provenance'),label+'.actual_factor_binding')
        a.eq(s['factor_pool'],dict(point_ids=U,line_edges=edges,actual_point_factors=len(U),unique_line_factors=len(lines),normal_incidence_scalar_rows=2*F,scalar_residuals=2*len(U)+sum(2+len(l['support_query_ids']) for l in lines)),label+'.factor_rows')
    direct=[p for p in proposals if p['proposal_kind']=='DIRECT_FINITE_FOUR_POINT_ROOT'];mixed=[p for p in proposals if p['proposal_kind']=='MIXED_THREE_FACTOR_LOCAL_SUBSET']
    generated=s['direct_point_generation'];enumerated=s['mixed_subset_enumeration'];early=s['state'] in ('INVALID_OBSERVATION_CONTRACT','INSUFFICIENT_POINT_OBSERVATIONS','INSUFFICIENT_ACTUAL_INCIDENCE','UNSUPPORTED_CAMERA_CONTRACT')
    failures=[];calls=Counter();returns=Counter();exceptions=Counter();success=Counter();caches=Counter()
    if not early:
        expected=[(d,list(ids)) for d in range(D) for ids in combinations(U,4)];a.eq([(g['dimension_index'],g['generator_ids']) for g in generated],expected,label+'.all_four_ID_requests')
        roots=0;matched=set()
        for i,g in enumerate(generated):
            a.eq(g['returned_valid_roots'],len(g['roots']),label+'.all_returned_roots');a.eq(g['original_available_or_best_not_used'],True,label+'.no_availability_filter');roots+=len(g['roots'])
            require(g['generation_state'] in ('VALID_GENERIC_ROOTS','DEGENERATE_POINT_SUBSET','NUMERICAL_GENERATION_FAILURE'),'typed Generic generation state')
            a.eq(g['generation_state']=='VALID_GENERIC_ROOTS',bool(g['roots']),label+'.valid_roots_state')
            for r in g['roots']:
                a.eq(r['actual_optimizer_called'],False,label+'.Generic_root_no_optimizer')
                if r.get('rejected'):a.eq(r['rejected'],'DIRECT_POINT_GEOMETRY_UNAVAILABLE',label+'.typed_direct_failure');failures.append(r['rejected']);continue
                found=[p for p in direct if (p['dimension_index'],p['point_generator_ids'],p['point_solution_index'],p['point_generator'])==(g['dimension_index'],g['generator_ids'],r['solution_index'],r['generator'])]
                require(len(found)==1,'each successful raw root retained exactly once');p=found[0];matched.add(p['proposal_index']);a.eq(p['start_index'],r['start_index'],label+'.root_start');a.eq(p['fit_input_ids'],g['generator_ids'],label+'.root_actual_four_ids')
        a.eq(matched,{p['proposal_index'] for p in direct},label+'.all_direct_proposals_joined')
        blocks=[('POINT',k) for k in U]+[('EDGE',e) for e in edges];subsets=[v for v in combinations(blocks,3) if any(k=='EDGE' for k,_ in v)]
        a.eq([sub['factor_blocks'] for sub in enumerated],[[list(v) for v in sub] for sub in subsets],label+'.exhaustive_mixed_subsets');a.eq(len(enumerated),M,label+'.mixed_cardinality')
        seed=s['start_source_witnesses'];a.eq(len(seed),1,label+'.one_original_seed_source');a.eq(seed[0]['source'],'ORIGINAL_INITIAL_N3_ONLY',label+'.original_N3_seed');a.eq(seed[0]['original_pose'],parent['initial_pose'],label+'.seed_not_point_result')
        for i,sub in enumerate(enumerated):
            ids=[v for k,v in subsets[i] if k=='POINT'];es=[v for k,v in subsets[i] if k=='EDGE']
            for key,value in (('index',i),('point_ids',ids),('line_edges',es),('normal_incidence_scalar_rows',6),('necessary_not_sufficient',True)):a.eq(sub[key],value,label+'.subset.'+str(i)+'.'+key)
            a.eq([b['dimension_index'] for b in sub['branches']],list(range(D)),label+'.all_registered_dimensions')
            for entry in sub['branches']:
                actual=entry['actual_optimizer_called'];calls['proposal']+=int(actual);returns['proposal']+=int(actual and 'scipy_status' in entry);exceptions['proposal']+=int(actual and 'exception' in entry and 'scipy_status' not in entry);success['proposal']+=int(actual and entry.get('success',False));caches['proposal']+=int(entry['cache_reused'])
                a.eq(entry['cache_reused'],False,label+'.unique_mixed_start_pool');a.eq(entry['actual_optimizer_called'],seed[0]['accepted'],label+'.seed_dispatch')
                found=[p for p in mixed if p['subset_index']==i and p['dimension_index']==entry['dimension_index']]
                if entry.get('rejected'):failures.append(entry['rejected']);a.eq(found,[],label+'.failed_mixed_has_no_proposal')
                else:
                    require(len(found)==1,'successful mixed branch retained once');p=found[0];a.eq(p['fit_input_ids'],ids,label+'.mixed_fit_points');a.eq(p['fit_line_edges'],es,label+'.mixed_fit_edges');a.eq(p['subset_factor_blocks'],sub['factor_blocks'],label+'.mixed_source_subset');a.eq(p['final_z'],entry['final_z'],label+'.mixed_returned_pose')
                if actual:
                    R,_=h['rotation_and_right_J'](entry['initial_z'][:3]);a.near(h['physical_rotation'](R,entry['dimension_index']),parent['initial_pose']['R_physical'],label+'.seed_physical_remap');a.eq(entry['initial_z'][3:],parent['initial_pose']['centroid'],label+'.seed_centroid')
        bounds=dict(point_count=len(U),line_count=len(lines),factor_blocks=F,registered_dimensions=D,four_ID_subsets=math.comb(len(U),4) if len(U)>=4 else 0,maximum_direct_Generic_calls=D*(math.comb(len(U),4) if len(U)>=4 else 0),mixed_subsets=M,mixed_LOCAL_proposal_calls_at_most=D*M,universal_mixed_subsets=1140,universal_mixed_LOCAL_proposals=2280,direct_valid_roots=roots,LOCAL_optimizer_calls_at_most=2*D*M+roots,universal_LOCAL_optimizer_calls_at_most=4560+roots)
        a.eq(s['call_bounds'],bounds,label+'.actual_work_bound');a.ok(F<=20 and D<=2 and M<=1140 and calls['proposal']<=D*M,label+'.finite_subset_upper_bounds')
        for key,value in (('direct_four_ID_requests',len(generated)),('direct_point_roots',roots),('direct_geometry_calls',roots),('direct_geometry_failures',roots-len(direct)),('mixed_subset_requests',M),('mixed_dimension_requests',D*M),('whole_pool_proposal_scores',len(proposals))):a.eq(op[key],value,label+'.counts.'+key)
    else:a.eq(generated,[],label+'.early_no_point_generation');a.eq(enumerated,[],label+'.early_no_mixed_generation');a.eq(proposals,[],label+'.early_no_proposals');a.eq(cs,[],label+'.early_no_refits')
    for i,p in enumerate(proposals):a.eq(p['proposal_index'],i,label+'.proposal_index');a.eq(p['logical_start_index'],i,label+'.proposal_logical_index');candidate(p,row,U,H,lines,h,a,policy,counts,label+'.proposal.'+str(i))
    a.eq(len(attempts),len(proposals),label+'.all_proposals_have_refit_record');seen={};candidate_indices=[]
    for i,attempt in enumerate(attempts):
        p=proposals[i];e=attempt['refit'];ip=p['whole_pool_score']['final_inliers'];il=p['whole_pool_score']['final_line_inliers'];a.eq(attempt['proposal_index'],i,label+'.attempt_proposal');a.eq(e['proposal_consensus_point_ids'],ip,label+'.refit_inlier_points');a.eq(e['proposal_consensus_line_ids'],il,label+'.refit_inlier_lines')
        a.eq(e['removed_point_ids'],[k for k in U if k not in ip],label+'.excluded_point_outliers');a.eq(e['removed_line_ids'],[k for k in edges if k not in il],label+'.excluded_edge_outliers')
        actual=e['actual_optimizer_called'];calls['refit']+=int(actual);returns['refit']+=int(actual and 'scipy_status' in e);exceptions['refit']+=int(actual and 'exception' in e and 'scipy_status' not in e);success['refit']+=int(actual and e.get('success',False));caches['refit']+=int(e['cache_reused'])
        insufficient=not il and len(ip)<4 or 2*(len(ip)+len(il))<6
        if insufficient:a.eq(e['rejected'],'INSUFFICIENT_POINT_INLIER_OBSERVATIONS' if not il and len(ip)<4 else 'INSUFFICIENT_ACTUAL_INLIER_INCIDENCE',label+'.insufficient_refit');a.eq(actual,False,label+'.no_insufficient_optimizer')
        else:
            key=(p['dimension_index'],struct.pack('<6d',*p['final_z']),tuple(ip),tuple(il));a.eq(e['cache_reused'],key in seen,label+'.exact_pose_and_fit_cache')
            if key in seen:
                previous=seen[key];a.eq(actual,False,label+'.cache_no_optimizer');a.eq(e['cache_source_start_index'],previous['start_index'],label+'.cache_source')
                for f in ('initial_z','final_z','success','scipy_status','nfev','njev','reported_soft_l1_cost','exception'):a.eq(e.get(f),previous.get(f),label+'.cached_numeric_'+f)
            else:seen[key]=e;a.eq(actual,True,label+'.fresh_refit_optimizer')
            a.eq(e['initial_z'],p['final_z'],label+'.refit_start_exact_proposal')
        if 'candidate_index' in e:
            c=cs[e['candidate_index']];candidate_indices.append(e['candidate_index']);a.eq(c['fit_input_ids'],ip,label+'.candidate_fit_actual_inliers');a.eq(c['fit_line_edges'],il,label+'.candidate_fit_actual_lines');a.eq(c['final_z'],e['final_z'],label+'.candidate_returned_pose');a.eq(c['accepted'],e['accepted'],label+'.entry_acceptance');a.eq(c['rejected'],e['rejected'],label+'.entry_rejection')
        if e.get('rejected'):failures.append(e['rejected'])
    a.eq(candidate_indices,list(range(len(cs))),label+'.all_refit_candidates_retained')
    for i,c in enumerate(cs):a.eq(c['candidate_index'],i,label+'.candidate_index');candidate(c,row,U,H,lines,h,a,policy,counts,label+'.refit.'+str(i))
    viable=[c for c in cs if c['accepted']];ranked=sorted(viable,key=lambda c:(-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index']));best=ranked[0] if ranked else None;ambiguity=[];alternatives=[];duplicates=ties=0
    if best:
        a.eq(s['best_candidate'],best,label+'.selected_candidate')
        for c in ranked[1:]:
            eq=h['equivalent'](best,c,row['xyz']);delta=c['truncated_sse_px2']-best['truncated_sse_px2'];tie=c['score_count']==best['score_count'] and abs(delta)<=1e-8;same_support=c['final_inliers']==best['final_inliers'] and c['final_line_inliers']==best['final_line_inliers'];duplicates+=int(eq);ties+=int(tie)
            alternatives.append(dict(c,numerically_equivalent_to_selected=eq,numerical_objective_tie=tie,objective_difference_from_selected_px2=delta,same_actual8px_consensus_as_selected=same_support))
            if not eq and (same_support or tie):ambiguity.append('OBSERVATION_CONSENSUS_AMBIGUITY' if same_support else 'NUMERICAL_CONSENSUS_TIE')
        a.near(s['alternatives'],alternatives,label+'.all_branch_relations');a.eq(s['state'],'AMBIGUOUS_FINITE_CONSENSUS' if ambiguity else 'NEW_POSE',label+'.selected_state');a.eq(s['per_dimension_best'],[next(c for c in ranked if c['dimension_index']==d) for d in range(D) if any(c['dimension_index']==d for c in ranked)],label+'.dimension_best')
    else:a.eq(s.get('best_candidate'),None,label+'.no_best')
    if not early:
        categories=dict(numeric=0,insufficient=0,weak=0,unstable=0,initialization=0,other=0)
        for reason in failures:
            cat='insufficient' if 'INSUFFICIENT' in reason else 'weak' if 'WEAK' in reason else 'unstable' if reason=='UNSTABLE_INLIER_REFIT' else 'initialization' if reason=='NO_LOCAL_INITIALIZATION' else 'numeric' if reason in ('DIRECT_POINT_GEOMETRY_UNAVAILABLE','LOCAL_NUMERICAL_FAILURE','FINAL_GEOMETRY_UNAVAILABLE','NONFINITE_OR_NONPOSITIVE_DEPTH','OPTIMIZER_FAILED','STABLE_INCIDENCE_NUMERICAL_FAILURE') else 'other';categories[cat]+=1
        a.eq(s['failure_reasons'],failures,label+'.all_failures');a.eq(s['failure_category_counts'],categories,label+'.typed_failures')
        if not best:
            if not proposals and M==0 and len(U)>=4:state='DEGENERATE_POINT_OBSERVATIONS' if generated and all(g['generation_state']=='DEGENERATE_POINT_SUBSET' for g in generated) else 'NUMERICAL_GENERATION_FAILURE'
            else:
                active=[k for k,v in categories.items() if v];states=dict(numeric='NUMERICAL_FAILURE',insufficient='INSUFFICIENT_ACTUAL_INLIER_GEOMETRY',weak='WEAK_OBSERVABILITY',unstable='UNSTABLE_INLIER_REFIT',initialization='NO_LOCAL_INITIALIZATION');state=states.get(active[0],'MIXED_NO_VALID_CONSENSUS') if len(active)==1 else 'MIXED_NO_VALID_CONSENSUS'
            a.eq(s['state'],state,label+'.typed_no_viable_state')
    new=bool(best and not ambiguity);fallback=not new and bool(parent['initial_pose'].get('available',False));expected=copy.deepcopy(obs['native_N3_points']);sparse=copy.deepcopy(q)
    if new:
        for key in ('R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry','soft_l1_cost','sse_px2','truncated_sse_px2','final_inliers','final_line_inliers','stable_incidence_guard'):a.near(s[key],best[key],label+'.chosen.'+key)
        for k in range(8):
            if k in H:expected[k]=best['projected'][k];sparse[k]=best['projected'][k]
            elif h['finite'](q[k],(2,)) and q[k]!=[-1,-1]:expected[k]=q[k]
        a.eq(row['actual_pose'],s,label+'.actual_NEW_pose')
    else:
        a.eq(row['actual_pose'],parent['initial_pose'],label+'.full_original_pose_fallback')
        for key in ('R_cf','R_physical','centroid','cf_extents','projected'):a.eq(s[key],None,label+'.unavailable_'+key)
        a.eq(s['fit_input_ids'],[],label+'.unavailable_fitIDs');a.eq(s['fit_line_edges'],[],label+'.unavailable_fitEdges')
    for key,value in (('new_pose_estimated',new),('fallback_used',fallback),('pose_available',new or fallback),('no_pose',not(new or fallback)),('output_status','NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),('reprojections_reused_as_observations',False),('reprojected_ids',H if new else []),('hidden_reprojected',new and bool(H))):a.eq(row[key],value,label+'.output_'+key)
    for key,value in (('available',new),('pose_available',new),('new_pose_estimated',new),('no_pose',not new),('unresolved_ambiguity',bool(ambiguity)),('numerical_duplicate_solution_count',duplicates),('numerical_tied_solution_count',ties),('hidden_reprojected',new and bool(H))):a.eq(s[key],value,label+'.solver_'+key)
    a.near(row['native_points'],expected,label+'.display');a.near(s['points_final'],sparse,label+'.H_only_projection');a.eq(row['native_points'][8],obs['native_N3_points'][8],label+'.center_preserved');a.eq(op['hidden_reprojection_assignments'],len(H) if new else 0,label+'.H_once')
    for phase in ('proposal','refit'):
        for suffix,counter in (('calls',calls),('returns',returns),('exceptions',exceptions)):a.eq(op[phase+'_optimizer_'+suffix],counter[phase],label+'.'+phase+'_'+suffix)
        a.eq(op[phase+'_cache_hits'],caches[phase],label+'.'+phase+'_cache_hits')
    a.eq(op['local_optimizer_calls'],sum(calls.values()),label+'.all_optimizer_calls');a.eq(op['local_optimizer_completions'],sum(returns.values()),label+'.all_returns');a.eq(op['local_optimizer_exceptions'],sum(exceptions.values()),label+'.all_exceptions')
    for key,value in (('candidate_count',len(cs)),('proposal_candidate_count',len(proposals)),('viable_candidate_count',len(viable)),('refit_count',calls['refit'])):
        if key in s:a.eq(s[key],value,label+'.'+key)
    for key,value in (('actual_filtered_local_solve_calls',1),('reported_output_views',1),('endpoint_validation_requests',0),('endpoint_decision_source','SEALED_V10_ENDPOINT_DECISION_REPLAY'),('actual_direct_point_proposal_count',len(direct)),('actual_mixed_subset_proposal_count',len(mixed)),('local_operation_counts',op),('direct_point_bank_operation_counts',s['hypothesis_bank_operation_counts'])):a.eq(ledger[key],value,label+'.ledger_'+key)
    guard_w=[c['stable_incidence_guard'] for c in cs if 'stable_incidence_guard' in c]
    for key,value in (('stable_incidence_guard_calls',len(guard_w)),('stable_incidence_geometry_calls',sum(w['counts']['geometry_info_calls'] for w in guard_w)),('stable_incidence_geometry_projection_entries',sum(w['counts']['geometry_projection_entries'] for w in guard_w))):a.eq(op[key],value,label+'.guard_count_'+key)
    a.eq(ledger['actual_finite_Generic_dispatch_matches_point_bank'],True,label+'.actual_Generic_bank_join');a.eq(ledger['primitive_entry_calls']['solvePnPGeneric'],s['hypothesis_bank_operation_counts']['generic_calls'],label+'.actual_Generic_calls')
    a.eq(ledger['factor_subset_call_bounds'],s.get('call_bounds'),label+'.ledger_actual_bounds')
    for key,value in s['hypothesis_bank_operation_counts'].items():counts['point_operation:'+key]+=value
    for key in ('mixed_subsets','registered_dimensions','direct_valid_roots','LOCAL_optimizer_calls_at_most','maximum_direct_Generic_calls'):
        bounds=s.get('call_bounds') or {}
        if type(bounds.get(key)) is int:counts['bound:'+key]+=bounds[key]
        else:counts['bound_unavailable:'+key]+=1
    if not early:a.ok(sum(calls.values())<=s['call_bounds']['LOCAL_optimizer_calls_at_most'],label+'.optimizer_bound')
    expected_opt=dict(entries=sum(calls.values()),returns=sum(returns.values()),exceptions=sum(exceptions.values()),success_reports=sum(success.values()),residual_entries=op['optimizer_residual_calls'],jacobian_entries=op['optimizer_jacobian_calls']);a.eq(ledger['optimizer_dispatch'],expected_opt,label+'.optimizer_live_join')
    for field,prefix,keys in (('primitive_entry_calls','cv:entry:',h['PRIMITIVES']),('primitive_return_calls','cv:return:',h['PRIMITIVES']),('primitive_exception_calls','cv:exception:',h['PRIMITIVES']),('svd_dispatch','svd:',('entries','returns','exceptions')),('optimizer_dispatch','opt:',tuple(expected_opt))):
        require(set(ledger[field])==set(keys) and all(type(v) is int and v>=0 for v in ledger[field].values()),'nonnegative complete live counters')
        for key,value in ledger[field].items():counts[prefix+key]+=value
    for name in h['PRIMITIVES']:a.eq(ledger['primitive_entry_calls'][name],ledger['primitive_return_calls'][name]+ledger['primitive_exception_calls'][name],label+'.CV_complete.'+name)
    for name in ('solvePnP','solvePnPRefineLM','cornerSubPix'):a.eq(ledger['primitive_entry_calls'][name],0,label+'.no_additional_solver.'+name)
    a.eq(ledger['svd_dispatch']['entries'],ledger['svd_dispatch']['returns']+ledger['svd_dispatch']['exceptions'],label+'.SVD_complete')
    for key,value in op.items():counts['operation:'+key]+=value
    counts['frames']+=1;counts['status:'+row['output_status']]+=1;counts['state:'+s['state']]+=1;counts['stored_proposals']+=len(proposals);counts['stored_refits']+=len(cs)

def input_paths(args):
    p=args.input;out=dict(geometry=p/'GEOMETRY_SEALED.jsonl.gz',ledgers=p/'CONTROL_LEDGERS.jsonl.gz',repair_protocol=p/'PROTOCOL.json',geometry_seal=p/'GEOMETRY_SEAL.json',repair_receipt=p/'CONTROL_RECEIPT.json',parent_population_checks=p/'PARENT_POPULATION_CHECKS.json',checker=Path(__file__).resolve(),started=p/'CONTROL_STARTED.json',preflight=p/'PREFLIGHT.json',cpu_checks=p/'CPU_CHECKS.json',cpu_protocol=p/'CPU_PROTOCOL.json',cpu_started=p/'CPU_STARTED.json',cohort=COHORT,scalar_source=SOURCE,finite_scalar_source=FINITE,normal_guard_source=GUARD,source_geometry=PARENT/'GEOMETRY_SEALED.jsonl.gz',source_observations=PARENT/'OBSERVATIONS.jsonl.gz',source_audit=AUDIT/'AUDIT_ROWS.jsonl.gz',source_audit_protocol=AUDIT/'AUDIT_PROTOCOL.json',source_audit_seal=AUDIT/'AUDIT_SEAL.json',source_audit_validation=AUDIT/'VALIDATION_CHECKS.json',source_audit_validation_protocol=AUDIT/'VALIDATION_PROTOCOL.json')
    for name in ('solver.py','pipeline.py','runner.py','test_solver.py','cpu_checks.py'):out['code:'+name]=CODE/name
    parent=REPO/'_docs/experiments/pallet_stable_incidence_guard_20261011_v14'
    for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json'):out['parent:'+name]=parent/name
    for i,_ in enumerate(read(out['repair_receipt'])['resource_snapshots']):out['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return out

def metadata(f,b):
    core,seal,r,pop=[read(f[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_factor_subset_consensus_protocol_v15' and core['methods']==[METHOD] and core['frames']==core['rows']==core['ledger_rows']==245,'fixed full245 factor subset protocol')
    require(seal['schema']=='factor_subset_consensus_geometry_seal_v15' and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'completed GT-free geometry seal')
    require(r['schema']=='factor_subset_consensus_execution_v15' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'completed cleanup and geometry')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):require(same(seal[key],b[alias]),'sealed bytes '+key)
    require(same(r['protocol'],b['repair_protocol']) and all(same(r['raw_rows'][k],b[k]) for k in ('geometry','ledgers')),'receipt source bytes')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and same(core['parent_population_checks'],b['parent_population_checks']),'full parent population proof')
    for key in b:
        if key in ('repair_protocol','geometry_seal','repair_receipt','geometry','ledgers','started','preflight','parent_population_checks') or key.startswith('resource:'):continue
        require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in core['inputs'].values()),'unbound core input '+key)
    _,guard_policy,_=extraction();policy=core['policy']['solver'];finite=policy['frozen_source_finite_policy']
    require(policy['stable_incidence']==guard_policy and policy['direct_point_roots_conditioned_on_available_or_selected'] is False and policy['mixed_subset_exhaustive'] is True and policy['three_point_only_subsets'] is False and policy['selected_point_pose_is_additional_LOCAL_start'] is False and policy['stable_incidence_required_per_accepted_candidate'] is True,'fixed structural policy')
    for key,value in (('f_scale_px',8.),('max_nfev',50),('refit_iterations',1),('numerical_rank_relative_threshold',1e-10),('weak_relative_threshold',1e-4),('numerical_score_tie_px2',1e-8),('point_inlier_threshold_px',8.),('edge_block_inlier_threshold_px',8.),('truncated_factor_squared_cap',64.)):require(finite[key]==value,'unchanged numerical policy '+key)
    cpu,cp,cs=[read(f[k]) for k in ('cpu_checks','cpu_protocol','cpu_started')]
    require(cpu['schema']=='factor_subset_CPU_checks_v15' and cp['schema']=='factor_subset_CPU_protocol_v15' and cpu['complete'] is True and cpu['passed'] is True and cpu['tests_run']==12 and not cpu['failures'] and not cpu['errors'] and cpu['error'] is None and cpu['inputs']==cp['inputs'] and same(cpu['own_protocol'],b['cpu_protocol']) and same(cs['protocol'],b['cpu_protocol']),'actual same-source CPU12 PASS')
    require(cpu['configured_environment']==cp['configured_environment']==cs['configured_environment']==dict(PALLET_SOURCE_ROOT=core['source_root'],PALLET_BASELINE_ROOT=core['baseline_root']),'explicit CPU environment')
    for key in ('source_unchanged','opencv_threads_restored','environment_restored','instruments_restored','asset_canary_restored','normal_package_import'):require(cpu[key] is True,'CPU restoration '+key)
    require(not cpu['cleanup_errors'] and not cpu['forbidden_real_asset_reads'] and cpu['real_frame_model_RGB_GT_dataset_reads']==0,'synthetic CPU only')
    for item in cp['inputs'].values():require(same(binding(REPO/item['path']),item),'CPU input bytes unchanged')
    pg,po,ps,pr=[read(f['parent:'+n]) for n in ('VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json')]
    require(pg['complete'] is True and pg['passed'] is True and pg['failure_count']==0 and pg['exception'] is None and pg['inputs']==po['inputs'] and same(pg['own_protocol'],b['parent:VALIDATION_PROTOCOL.json']) and ps['complete'] is True and ps['frames']==245 and pr['complete'] is True and pr['cleanup_completed'] is True,'immutable normal-only parent full245 PASS')
    require(same(pg['inputs']['repair_protocol'],b['parent:PROTOCOL.json']) and same(pg['inputs']['geometry_seal'],b['parent:GEOMETRY_SEAL.json']) and same(pg['inputs']['repair_receipt'],b['parent:CONTROL_RECEIPT.json']),'parent checked raw authority')
    ps_=read(f['parent:VALIDATION_STARTED.json']);require(ps_['inputs']==po['inputs'] and same(ps_['own_protocol'],b['parent:VALIDATION_PROTOCOL.json']) and ps_['new_production_calls']==0 and same(pg['inputs']['code:pipeline.py'],b['normal_guard_source']),'parent exact scalar STARTED and normal guard source')
    ach,ao,aseal=[read(f[k]) for k in ('source_audit_validation','source_audit_validation_protocol','source_audit_seal')]
    require(ach['complete'] is True and ach['passed'] is True and ach['failure_count']==0 and ach['exception'] is None and ach['inputs']==ao['inputs'] and same(ach['own_protocol'],b['source_audit_validation_protocol']) and same(aseal['audit'],b['source_audit']) and same(aseal['protocol'],b['source_audit_protocol']),'checked frozen endpoint decisions')
    require(same(ach['inputs']['audit_rows'],b['source_audit']),'endpoint full245 raw bytes')
    for key in ('started','preflight'):require(same(read(f[key])['protocol'],b['repair_protocol']),'own '+key+' protocol')
    require(read(f['preflight'])['passed'] is True,'passed preflight')
    return core,r,pop

def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink input/output')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V15 output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not (args.output/name).exists(),'preserve '+name)
def freeze(args):
    guard(args,'freeze');f=input_paths(args);b={k:binding(p) for k,p in f.items()};metadata(f,b);_,policy,info=extraction()
    write(args.output/FILES[0],dict(schema='factor_subset_consensus_scalar_protocol_v15',inputs=b,checker=b['checker'],method=METHOD,frames=245,helpers=info,normal_policy=policy,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=input_paths(args);b={k:binding(p) for k,p in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,policy,info=extraction()
    require(own['schema']=='factor_subset_consensus_scalar_protocol_v15' and own['inputs']==b and own['helpers']==info and own['normal_policy']==policy and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs');core,r,pop=metadata(f,b)
    write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,new_production_calls=0));h=dict(math=math,json=json,hashlib=hashlib,struct=struct,copy=copy,_branch=_branch,rows=rows,require=require,execution_policy=core['policy']);exec(compile(tree,str(SOURCE)+':independent_stdlib_helpers','exec'),h);a=h['Audit']();counts=Counter();error=None
    try:
        cohort=read(f['cohort']);ordered=cohort['ids'];known=set(ordered);require(len(ordered)==len(known)==245 and [x['id'] for x in cohort['frames']]==ordered,'full245 cohort');a.eq(cohort['counts'],dict(clean=153,moderate=92,severe_excluded=74,original=319),'scope')
        streams=[iter(rows(f[k])) for k in ('geometry','ledgers','source_audit')];parents=iter(h['primary'](f['source_geometry'],'method','IMAGE_ROLE_BOUNDARY_ONLY',1960,known,a,'V7_geometry'));observations=iter(h['primary'](f['source_observations'],'head_arm','IMAGE_ROLE',735,known,a,'V7_ROLE'))
        for i,fid in enumerate(ordered):
            group=[next(stream,None) for stream in streams]+[next(parents,None),next(observations,None)];require(all(isinstance(x,dict) and x['id']==fid and x['session']==cohort['frames'][i]['session'] for x in group),'complete ordered same-frame stream joins')
            row,ledger,endpoint,parent,obs=group;packet(row,ledger,parent,obs,endpoint,pop['completion'],h,a,policy,counts,'frame.'+str(i));del group
        for stream in (*streams,parents,observations):a.eq(next(stream,None),None,'no_extra_rows')
        for key in ('frames_started','frames_complete','filtered_solver_entries','filtered_solver_returns','geometry_rows','ledger_rows'):a.eq(r['actual_execution'][key],245,'actual.'+key)
        a.eq(r['actual_execution']['filtered_solver_exceptions'],0,'actual.no_incomplete_solver');a.eq(r['local_operation_counts'],{k.removeprefix('operation:'):v for k,v in counts.items() if k.startswith('operation:')},'actual.operation_sums')
        for field,prefix in (('actual_OpenCV_entry_calls','cv:entry:'),('actual_OpenCV_return_calls','cv:return:'),('actual_OpenCV_exception_calls','cv:exception:')):a.eq(r[field],{n:counts[prefix+n] for n in h['PRIMITIVES']},'actual.'+field)
        a.eq(r['actual_optimizer'],{k:counts['opt:'+k] for k in ('entries','returns','exceptions','success_reports','residual_entries','jacobian_entries')},'actual.optimizer_sum');a.eq(r['actual_SVD_dispatch'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'actual.SVD_sum')
        a.eq(r['direct_point_bank_operation_counts'],{k.removeprefix('point_operation:'):v for k,v in counts.items() if k.startswith('point_operation:')},'actual.direct_bank_sums');a.eq(r['aggregate_declared_call_bounds'],{k.removeprefix('bound:'):v for k,v in counts.items() if k.startswith('bound:')},'actual.aggregate_bounds')
        a.eq(r['aggregate_bound_unavailable_frame_counts'],{k.removeprefix('bound_unavailable:'):v for k,v in counts.items() if k.startswith('bound_unavailable:')},'actual.unavailable_bounds');a.eq(r['solver_state_counts'],{k.removeprefix('state:'):v for k,v in counts.items() if k.startswith('state:')},'actual.typed_states')
        a.eq(r['output_status_counts'],{METHOD:{s:counts['status:'+s] for s in ('NEW_POSE','N3_BASELINE_FALLBACK','POSE_FAILURE') if counts['status:'+s]}},'actual.full245_statuses')
        for key in ('primitive_and_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):a.eq(r['cleanup'][key],True,'recorded.cleanup.'+key)
        a.eq(r['forbidden_entries'],{},'no_forbidden_entries');a.eq(r['forbidden_asset_reads'],[],'no_forbidden_assets');a.eq(r['protection_before'],r['protection_after'],'protected_bytes')
        for i,snap in enumerate(r['resource_snapshots']):a.eq(read(f['resource:%03d'%i]),snap,'resource_identity');a.eq(snap['quiet'],True,'recorded_quiet');a.eq(snap['temperature_under_80'],True,'recorded_thermal')
        a.eq(counts['frames'],245,'all245');a.eq(counts['stored_proposals'],counts['proposal_checks'],'all_proposal_math');a.eq(counts['stored_refits'],counts['refit_checks'],'all_refit_math');a.eq({k:binding(p) for k,p in f.items()},b,'input_bytes_unchanged');a.eq(binding(args.output/FILES[0]),ob,'own_protocol_unchanged')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    result=dict(schema='factor_subset_consensus_scalar_checks_v15',complete=error is None and counts['frames']==245,passed=error is None and counts['frames']==245 and a.failure_count==0,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,max_scalar_absolute_difference=a.max_difference,inputs=b,own_protocol=ob,checker=b['checker'],counts=dict(counts),cusp_reconciliations=STATE['events'],limits=LIMITS,new_production_calls=0,first_failure_preserved=True)
    write(args.output/FILES[2],result);print('FACTOR_SUBSET_SCALAR_CHECKS',result['passed'],a.checks,a.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();{'freeze':freeze,'run':run}[args.stage](args)
if __name__=='__main__':main()

