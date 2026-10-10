"""Own-frozen independent scalar audit of the selected V14 incidence guard.

Reconstruct only selected point/endpoint-normal Jacobians. No inference, GT,
fit, OpenCV, NumPy, new SVD or timing is invoked. Parent V13 candidates are
immutable reviewed inputs, rather than another full candidate audit here.
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
PARENT=REPO/'_docs/experiments/pallet_finite_edge_consensus_repair_20261011_v13'
PRIVATE=Path('/tmp/pallet-stable-incidence-guard-private-20261011-v14')
SOURCE=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/validation_checks.py'
SOURCE_SHA='4319477832a599c70c015fdb2a8538a51cb0f19077591e352c75e15d7860ec1e'
METHOD='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE';PARENT_METHOD='FINITE_EDGE_CONSENSUS'
CV=('solvePnP','solvePnPGeneric','solvePnPRefineLM','projectPoints','Rodrigues','cornerSubPix')
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
LIMITS=dict(new_model_RGB_GT_fit_PnP_OpenCV_Jacobian_SVD_timing_calls=0,
    independent_scalar_projection_and_incidence_J=True,stored_SVD_spectrum_recount_only=True,
    parent_full_candidate_audit_not_repeated=True,local_rank_is_not_global_uniqueness=True,
    physical_ownership_certified=False,hardware_authenticity_proven=False,
    quiet_snapshots_can_miss_transient_competition=True,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink input '+str(p));digest=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):digest.update(block)
    p=p.resolve();return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=digest.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False,default=sorted);f.write('\n');f.flush();os.fsync(f.fileno())

def extraction():
    require(binding(SOURCE)['sha256']==SOURCE_SHA,'exact frozen independent scalar source required')
    names={'require','safe','Audit','finite','dot','mm','eye','skew','rotation_and_right_J','dimensions','model','forward','physical_rotation','factor_rows','rank_witness'}
    constants={'EDGES','ATOL','RTOL','RANK'};nodes=[];found=set();assigned=set()
    for n in ast.parse(SOURCE.read_text()).body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names:nodes.append(n);found.add(n.name)
        elif isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) or isinstance(t,ast.Tuple) for t in n.targets):
            ids={q.id for t in n.targets for q in ast.walk(t) if isinstance(q,ast.Name)}
            if ids & constants:nodes.append(n);assigned.update(ids)
    require(found==names and constants<=assigned,'exact stdlib whitelist');tree=ast.Module(body=nodes,type_ignores=[])
    gt=ast.parse((CODE/'pipeline.py').read_text());policy=None
    for n in gt.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets):
            call=n.value;require(isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='dict' and not call.args and all(k.arg is not None for k in call.keywords),'literal guard policy only')
            policy={k.arg:ast.literal_eval(k.value) for k in call.keywords}
    require(policy is not None and policy['numerical_rank_relative_threshold']==1e-10 and policy['weak_relative_threshold']==1e-4 and policy['hinge_rank_rows_retained']==0,'existing rank/weak and no-hinge approval policy')
    return tree,policy,dict(source=binding(SOURCE),names=sorted(names),AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())

def paths(p):
    names=dict(geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',repair_protocol='PROTOCOL.json',geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',parent_population_checks='PARENT_POPULATION_CHECKS.json',started='CONTROL_STARTED.json',preflight='PREFLIGHT.json',cpu_protocol='CPU_PROTOCOL.json',cpu_started='CPU_STARTED.json',cpu_checks='CPU_CHECKS.json',cpu_review_protocol='REVIEW_CPU_PROTOCOL.json',cpu_review_started='REVIEW_CPU_STARTED.json',cpu_review_checks='REVIEW_CPU_CHECKS.json',cpu_package_protocol='PACKAGE_CPU_PROTOCOL.json',cpu_package_started='PACKAGE_CPU_STARTED.json',cpu_package_checks='PACKAGE_CPU_CHECKS.json')
    out={k:p/v for k,v in names.items()};out.update(checker=Path(__file__).resolve(),scalar_source=SOURCE)
    for n in ('pipeline.py','runner.py','test_guard.py','cpu_checks.py','cpu_environment_review.py','cpu_package_review.py'):out['code:'+n]=CODE/n
    for n in ('GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json'):
        out['parent:'+n]=PARENT/n
    for i,_ in enumerate(read(out['repair_receipt'])['resource_snapshots']):out['resource:%03d'%i]=p/('RESOURCE_%03d.json'%i)
    return out

def metadata(f,b):
    core,seal,r,pop=[read(f[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_selected_stable_incidence_protocol_v14' and core['methods']==[METHOD] and core['frames']==core['rows']==core['ledger_rows']==245,'fixed full245 guard policy')
    require(seal['schema']=='selected_stable_incidence_geometry_seal_v14' and seal['complete'] is True and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'complete GT-free seal')
    require(r['schema']=='selected_stable_incidence_execution_v14' and r['complete'] is True and r['cleanup_completed'] is True and r['error'] is None and r['cleanup_error'] is None,'complete guard and cleanup')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):require(same(seal[key],b[alias]),'seal byte join '+key)
    require(same(r['protocol'],b['repair_protocol']) and all(same(r['raw_rows'][k],b[k]) for k in ('geometry','ledgers')),'receipt byte join')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==pop['rows']==245 and same(core['parent_population_checks'],b['parent_population_checks']) and same(pop['parent_geometry'],b['parent:GEOMETRY_SEALED.jsonl.gz']),'parent population proof')
    for key in ('checker','scalar_source',*('code:'+n for n in ('pipeline.py','runner.py','test_guard.py','cpu_checks.py','cpu_environment_review.py','cpu_package_review.py')),*('parent:'+n for n in ('GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json')),*('cpu_'+n for n in ('protocol','started','checks','review_protocol','review_started','review_checks','package_protocol','package_started','package_checks'))):
        require(any(v['path']==b[key]['path'] and same(v,b[key]) for v in core['inputs'].values()),'unbound frozen source '+key)
    parent,review,first=[read(f['parent:'+n]) for n in ('REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json','VALIDATION_CHECKS.json')]
    require(parent['complete'] is True and parent['passed'] is True and parent['checks']==713803 and parent['failure_count']==0 and parent['exception'] is None and parent['inputs']==review['inputs'] and same(parent['own_protocol'],b['parent:REVIEW_VALIDATION_PROTOCOL.json']),'immutable parent full713803 review PASS')
    require(first['complete'] is True and first['passed'] is False and first['checks']==713802 and first['failure_count']==74 and first['exception'] is None,'preserve original parent roundoff inspector FAIL')
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),('original_failure','VALIDATION_CHECKS.json'),('original_validation_protocol','VALIDATION_PROTOCOL.json')):require(same(review['inputs'][key],b['parent:'+name]),'reviewed immutable parent bytes '+key)
    ps=read(f['parent:REVIEW_VALIDATION_STARTED.json']);require(ps['inputs']==review['inputs'] and same(ps['own_protocol'],b['parent:REVIEW_VALIDATION_PROTOCOL.json']) and ps['new_production_calls']==0,'parent review STARTED')
    old,op,os_,prior,pp,ps_,new,np_,ns=[read(f[k]) for k in ('cpu_checks','cpu_protocol','cpu_started','cpu_review_checks','cpu_review_protocol','cpu_review_started','cpu_package_checks','cpu_package_protocol','cpu_package_started')]
    require(old['schema']=='stable_incidence_CPU_checks_v14' and old['complete'] is True and old['passed'] is False and old['tests_run']==7 and old['failures']==[] and len(old['errors'])==3 and all("KeyError: 'PALLET_SOURCE_ROOT'\n" in e['traceback'] for e in old['errors']) and old['error'] is None and old['inputs']==op['inputs'] and same(old['own_protocol'],b['cpu_protocol']) and same(os_['protocol'],b['cpu_protocol']),'original fourPASS/three environment-errors CPU preserved')
    prior_expected=dict(op['inputs'])
    for key in ('cpu_protocol','cpu_started','cpu_checks','code:cpu_environment_review.py'):prior_expected[b[key]['path']]=b[key]
    require(prior['schema']=='stable_incidence_CPU_checks_v14' and prior['complete'] is True and prior['passed'] is False and prior['tests_run']==0 and prior['failures']==prior['errors']==[] and prior['error']['type']=='ImportError' and prior['error']['message']=='attempted relative import with no known parent package' and prior['source_unchanged'] is True and prior['inputs']==pp['inputs']==prior_expected and same(prior['own_protocol'],b['cpu_review_protocol']) and same(ps_['protocol'],b['cpu_review_protocol']),'preserve zero-group failed package setup review')
    expected=dict(op['inputs'])
    for key in ('cpu_protocol','cpu_started','cpu_checks','cpu_review_protocol','cpu_review_started','cpu_review_checks','code:cpu_environment_review.py','code:cpu_package_review.py'):expected[b[key]['path']]=b[key]
    require(new['schema']=='stable_incidence_CPU_checks_v14' and new['complete'] is True and new['passed'] is True and new['tests_run']==7 and new['failures']==new['errors']==[] and new['error'] is None and new['source_unchanged'] is True and new['opencv_threads_restored'] is True and not new['cleanup_errors'] and new['inputs']==np_['inputs']==expected and same(new['own_protocol'],b['cpu_package_protocol']) and same(ns['protocol'],b['cpu_package_protocol']),'actual unchanged seven-group package/environment CPU PASS')
    marker=new['package_review'];require(marker['original_failed_CPU_attempt_preserved'] is True and marker['failed_environment_review_preserved'] is True and marker['original_tests_and_guard_changed'] is False and marker['Python_package_and_environment_setup_only'] is True and marker['normal_original_package_import'] is True and marker['configured_actual_groups']==7 and marker['prior_environment_review_tests_run']==0,'package/environment-only CPU correction')
    require(marker==np_['package_review']==ns['package_review'] and marker['configured_environment']==dict(PALLET_SOURCE_ROOT=core['source_root'],PALLET_BASELINE_ROOT=core['baseline_root']),'same actual CPU package marker and split environment')
    for value in op['inputs'].values():require(same(binding(REPO/value['path']),value),'CPU dependency bytes changed')
    for key in ('primitive_optimizer_restored','opencv_threads_restored','split_environment_restored','asset_canary_open_functions_restored'):require(r['cleanup'][key] is True,'restoration '+key)
    require(not r['forbidden_entries'] and not r['forbidden_asset_reads'] and r['new_training_RGB_models_GT_optimizer_PnP_calls']==0 and r['original_finite_fit_repeated'] is False,'no new fit/model/GT work')
    require(r['protection_before']==r['protection_after'],'protected source changed')
    for i,value in enumerate(r['resource_snapshots']):require(read(f['resource:%03d'%i])==value and value['quiet'] is True and value['temperature_under_80'] is True,'recorded quiet resource '+str(i))
    for key in ('started','preflight'):require(same(read(f[key])['protocol'],b['repair_protocol']),'own '+key+' protocol')
    require(read(f['preflight'])['passed'] is True,'quiet preflight')
    return core,r,pop

def packet(row,ledger,parent,h,a,policy,counts,label):
    old=parent['solver'];s=row['solver'];w=row['stable_incidence_guard'];available=bool(old['available'])
    a.eq(parent['method'],PARENT_METHOD,label+'.parent_method');a.eq(row['method'],METHOD,label+'.method');a.eq(row['parent_method'],PARENT_METHOD,label+'.declared_parent')
    for key in ('input_points','K','xyz','hidden_initial','original_native_N3_points','initial_pose','observation_contract','observation_filter','fixed_metadata','id','session'):a.eq(row[key],parent[key],label+'.unchanged_'+key)
    a.eq(row['parent_V13_row_semantic_sha256'],semantic(parent),label+'.parent_SHA');a.eq(ledger['parent_V13_row_semantic_sha256'],semantic(parent),label+'.ledger_parent_SHA');a.eq(ledger['guarded_row_semantic_sha256'],semantic(row),label+'.row_SHA')
    for actual in (ledger['stable_incidence_guard'],s['stable_incidence_guard']):a.eq(actual,w,label+'.same_guard_witness')
    a.eq(ledger['guard_counts'],w['counts'],label+'.counts_witness');a.eq(w['policy'],policy,label+'.policy');a.eq(row['stable_incidence_policy'],policy,label+'.row_policy')
    a.eq(w['original_solver_semantic_sha256'],semantic(old),label+'.solver_SHA');a.eq(w['original_all_candidates_semantic_sha256'],semantic(old.get('all_candidate_solutions',[])),label+'.all_candidates_SHA')
    for k in ('all_candidate_solutions','best_candidate','optimizer_attempts','operation_counts'):
        if k in old:a.eq(s[k],old[k],label+'.immutable_'+k)
    for k,value in (('selected_candidate_reselected',False),('global_uniqueness_proven',False),('physical_ownership_truth','UNKNOWN'),('original_numerical_pose_available',available),('gate_applied',available)):a.eq(w[k],value,label+'.'+k)
    counts['guard_calls']+=1;counts['parent_NEW']+=available;counts['parent_unavailable_noop']+=not available
    accepted=False
    if available:
        c=old['best_candidate'];ids=c['fit_input_ids'];edges=c['fit_line_edges'];H=parent['hidden_initial'];d=c['dimension_index'];extents=h['dimensions'](parent['xyz'])[d]
        a.eq(c['accepted'],True,label+'.original_accepted');a.eq(ids,old['fit_input_ids'],label+'.fitIDs');a.eq(edges,old['fit_line_edges'],label+'.fitEdges');a.ok(not set(ids)&set(H) and 8 not in ids,label+'.H_center_excluded');a.ok(not set(edges)&set(old['consumed_edges']),label+'.no_point_line_double_count')
        lines=[dict(edge=e,normal=l['normal'],offset=l['offset']) for e in edges for l in old['actual_source_lines'] if l['edge']==e];require(len(lines)==len(edges),'selected actual lines missing')
        for key,value in (('actual_fit_input_ids',ids),('actual_fit_line_edges',edges),('hidden',H),('selected_dimension_index',d),('selected_extents',extents),('selected_final_z',c['final_z']),('selected_candidate',c),('actual_normal_lines',lines)):a.eq(w[key],value,label+'.'+key)
        R,uv,J,depths=h['forward'](extents,c['final_z'],parent['K']);a.near(c['R_cf'],R,label+'.independent_R');a.near(c['projected'],uv,label+'.independent_projected');a.near(c['R_physical'],h['physical_rotation'](R,d),label+'.physicalR')
        _,observed=h['factor_rows'](uv,J,parent['input_points'],ids,lines);_,modeled=h['factor_rows'](uv,J,parent['input_points'],ids,lines,True)
        g=c['point_line_geometry'];indices=list(range(2*len(ids)));end=2*len(ids)
        for b in g['edge_blocks']:a.eq(b['scalar_start'],end,label+'.finite_partition');indices.extend((end,end+1));end=b['scalar_end']
        a.eq([b['edge'] for b in g['edge_blocks']],edges,label+'.block_identity');a.eq(w['incidence_source_row_indices'],indices,label+'.source_rows');a.eq(w['finite_scalar_rows'],end,label+'.finite_rows');a.eq(w['incidence_scalar_rows'],len(indices),label+'.normal_rows');a.eq(w['removed_longitudinal_rows'],end-len(indices),label+'.removed_hinges')
        a.eq(w['observed_normal_J'],[g['observed_normal_J'][i] for i in indices],label+'.exact_extracted_observed');a.eq(w['modeled_normal_J'],[g['modeled_normal_J'][i] for i in indices],label+'.exact_extracted_modeled');a.near(w['observed_normal_J'],observed,label+'.independent_observed');a.near(w['modeled_normal_J'],modeled,label+'.independent_modeled')
        geometry=w.get('stable_incidence_geometry');state=w['guard_state']
        if geometry is not None:
            split=2*len(ids);matrices=dict(observed_normal_joint=observed,modeled_normal_joint=modeled,point_only=modeled[:split],observed_normal_lines_only=observed[split:],modeled_normal_lines_only=modeled[split:]);ranks={}
            for key,matrix in matrices.items():ranks[key]=h['rank_witness'](geometry[key],matrix,a,label+'.'+key)
            a.eq(geometry['point_ids'],ids,label+'.incidence_ids');a.eq(geometry['line_edges'],edges,label+'.incidence_edges');a.eq(geometry['modeled_normals_held_fixed_for_derivative'],True,label+'.held_normal');a.eq(geometry['global_unique_pose_proven'],False,label+'.local_only')
            weak={name:len(geometry[name+'_normal_joint']['singular_values'])!=6 or geometry[name+'_normal_joint']['singular_values'][-1]<geometry[name+'_normal_joint']['singular_values'][0]*1e-4 for name in ('observed','modeled')}
            rank=all(ranks[k]==6 for k in ('observed_normal_joint','modeled_normal_joint'));expected='PASSED_STABLE_INCIDENCE' if rank and not any(weak.values()) else 'STABLE_INCIDENCE_INSUFFICIENT' if not rank else 'STABLE_INCIDENCE_WEAK'
            a.eq(state,expected,label+'.rank_weak_routing');a.eq(w['observed_weak'],weak['observed'],label+'.observed_weak');a.eq(w['modeled_weak'],weak['modeled'],label+'.modeled_weak');a.eq(w['complete_rank_info_calls_from_source'],5,label+'.five_rank_reports')
            counts['expected_SVD_entries']+=sum(bool(m) for m in matrices.values())
            accepted=expected=='PASSED_STABLE_INCIDENCE'
        else:a.eq(state,'STABLE_INCIDENCE_NUMERICAL_FAILURE',label+'.typed_numeric_failure');a.ok(isinstance(w.get('error'),dict),label+'.recorded_numeric_error')
        counts['selected_scalar_forward_checks']+=1
    else:a.eq(w['guard_state'],'PARENT_HAS_NO_NEW_POSE',label+'.no_new_noop')
    a.eq(w['accepted'],accepted,label+'.accepted');counts['guard_accepted']+=accepted
    expected_solver=copy.deepcopy(old)
    if available and not accepted:
        expected_solver.update(available=False,pose_available=False,new_pose_estimated=False,no_pose=True,state=w['guard_state'],reason='selected actual fit lacks sufficient stable point/line incidence',output_status='NO_NEW_POSE',local_refinement_estimated=False,
            rejected_finite_pose={k:copy.deepcopy(old.get(k)) for k in ('state','R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry')},R_cf=None,R_physical=None,centroid=None,cf_extents=None,projected=None,fit_input_ids=[],fit_line_edges=[],input_inliers=[],final_inliers=[],inliers=[],final_line_inliers=[],point_inlier_count=0,unique_line_inlier_count=0,total_inlier_factor_count=0,points_final=parent['input_points'],hidden_reprojected=False)
    expected_solver['stable_incidence_guard']=w;a.eq(s,expected_solver,label+'.single_guarded_solver')
    if available and not accepted:
        initial=parent['initial_pose'];fallback=bool(initial.get('available',False));a.eq(row['actual_pose'],initial,label+'.full_initial_pose');a.eq(row['native_points'],parent['original_native_N3_points'],label+'.full_N3_coordinates');a.eq(row['output_status'],'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',label+'.decline_status')
        for key,value in (('new_pose_estimated',False),('fallback_used',fallback),('pose_available',fallback),('no_pose',not fallback),('hidden_reprojected',False),('reprojected_ids',[]),('hidden_after',[]),('hidden_set_changed',False),('native_N3_used_as_independent_RGB_observations',False)):a.eq(row[key],value,label+'.decline_'+key)
    else:
        for key in ('native_points','output_status','new_pose_estimated','fallback_used','pose_available','no_pose','hidden_reprojected','reprojected_ids','hidden_after','hidden_set_changed','output_coordinate_sources'):a.eq(row[key],parent[key],label+'.retained_'+key)
        a.eq(row['actual_pose'],s if available else parent['actual_pose'],label+'.retained_actual_pose')
    for key,value in (('reprojections_reused_as_observations',False),('actual_existing_finite_optimizer_refit_repeated',False),('accepted_H_projection_assigned_again',False),('numerical_NEW_is_not_accuracy_success',True),('global_uniqueness_proven',False)):a.eq(row[key],value,label+'.'+key)
    a.eq(row['native_points'][8],parent['original_native_N3_points'][8],label+'.center_preserved')
    if accepted:
        for k in parent['hidden_initial']:a.near(row['native_points'][k],uv[k],label+'.H_projection.'+str(k))
    for key,value in (('guard_calls',1),('geometry_info_calls',int(available)),('new_optimizer_calls',0),('new_PnP_calls',0),('actual_rank_info_dispatches_independently_counted',False)):a.eq(w['counts'][key],value,label+'.op_'+key)
    for key,value in w['counts'].items():
        if type(value) is int:counts['op:'+key]+=value
    for group in ('primitive_entry_calls','primitive_return_calls','primitive_exception_calls'):
        a.eq(set(ledger[group]),set(CV),label+'.'+group+'_keys')
        for key,value in ledger[group].items():a.ok(type(value) is int and value>=0,label+'.nonnegative_'+group+'.'+key);counts[group+':'+key]+=value
    for key in CV:a.eq(ledger['primitive_entry_calls'][key],ledger['primitive_return_calls'][key]+ledger['primitive_exception_calls'][key],label+'.primitive_join.'+key)
    a.eq(ledger['primitive_entry_calls']['projectPoints'],w['counts']['geometry_projection_entries'],label+'.project_dispatch');a.eq({k:v for k,v in ledger['primitive_entry_calls'].items() if k!='projectPoints'},{k:0 for k in CV if k!='projectPoints'},label+'.no_other_primitives')
    svd=ledger['svd_dispatch'];a.eq(set(svd),{'entries','returns','exceptions'},label+'.SVD_keys');a.eq(svd['entries'],svd['returns']+svd['exceptions'],label+'.SVD_join')
    if w.get('stable_incidence_geometry') is not None:a.eq(svd['entries'],sum(bool(m) for m in matrices.values()),label+'.actual_nonempty_rank_SVD_dispatch')
    elif not available:a.eq(svd,dict(entries=0,returns=0,exceptions=0),label+'.noop_SVD')
    for key,value in svd.items():a.ok(type(value) is int and value>=0,label+'.SVD_nonnegative');counts['svd:'+key]+=value

def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing nonsymlink input/output')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive V14 output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)
def freeze(args):
    guard(args,'freeze');f=paths(args.input);b={k:binding(v) for k,v in f.items()};core,_,_=metadata(f,b);_,policy,witness=extraction();require(core['policy']==policy,'frozen source policy')
    write(args.output/FILES[0],dict(schema='stable_incidence_scalar_protocol_v14',inputs=b,checker=b['checker'],helpers=witness,policy=policy,limits=LIMITS,frozen_before_own_arithmetic=True))
def run(args):
    guard(args,'run');f=paths(args.input);b={k:binding(v) for k,v in f.items()};own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);tree,policy,witness=extraction()
    require(own['schema']=='stable_incidence_scalar_protocol_v14' and own['inputs']==b and own['helpers']==witness and own['policy']==policy and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'own freeze differs')
    core,r,pop=metadata(f,b);require(core['policy']==policy,'core source policy differs');write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,new_production_calls=0));h=dict(math=math,json=json);exec(compile(tree,str(SOURCE)+':stdlib_scalar_AST','exec'),h);a=h['Audit']();counts=Counter();error=None
    try:
        parent_rows=rows(f['parent:GEOMETRY_SEALED.jsonl.gz']);ledger_rows=rows(f['ledgers']);ids=pop['ordered_ids'];require(len(ids)==len(set(ids))==245,'fixed245 ordered IDs')
        for i,row in enumerate(rows(f['geometry'])):
            require(i<245,'extra guard row');parent=next(parent_rows);ledger=next(ledger_rows);require(row['id']==parent['id']==ledger['id']==ids[i] and row['session']==parent['session']==ledger['session'],'ordered complete same-frame joins');packet(row,ledger,parent,h,a,policy,counts,'frame.'+str(i));counts['frames']+=1
        require(next(parent_rows,None) is None and next(ledger_rows,None) is None,'extra parent or ledger row');a.eq(counts['frames'],245,'full245');a.eq(counts['parent_NEW'],75,'fixed_parent75selected');a.eq(counts['selected_scalar_forward_checks'],75,'75selected_independent_scalar');a.eq(counts['parent_unavailable_noop'],170,'170noops')
        expected=dict(frames_started=245,frames_complete=245,geometry_rows=245,ledger_rows=245,parent_NEW=counts['parent_NEW'],guard_applied=counts['parent_NEW'],guard_accepted=counts['guard_accepted'],parent_unavailable_noop=counts['parent_unavailable_noop'])
        for key,value in expected.items():a.eq(r['actual_execution'].get(key,0),value,'receipt.execution.'+key)
        a.eq(r['guard_operation_counts'],{key[3:]:value for key,value in counts.items() if key.startswith('op:')},'receipt.all_guard_operation_counts')
        for raw,receipt in (('primitive_entry_calls','actual_OpenCV_entry_calls'),('primitive_return_calls','actual_OpenCV_return_calls'),('primitive_exception_calls','actual_OpenCV_exception_calls')):a.eq(r[receipt],{k:counts[raw+':'+k] for k in CV},'receipt.'+receipt)
        a.eq(r['actual_SVD_dispatch'],{k:counts['svd:'+k] for k in ('entries','returns','exceptions')},'receipt.actual_SVD_dispatch')
        statuses=Counter(row['output_status'] for row in rows(f['geometry']));a.eq(r['output_status_counts'],{METHOD:dict(statuses)},'receipt.status_counts')
    except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    unchanged={k:binding(v) for k,v in f.items()}==b
    result=dict(schema='stable_incidence_scalar_checks_v14',complete=error is None and counts['frames']==245,passed=error is None and a.failure_count==0 and unchanged,checks=a.checks,failure_count=a.failure_count,failures=a.failures,exception=error,max_difference=a.max_difference,counts=dict(counts),inputs=b,own_protocol=ob,checker=b['checker'],limits=LIMITS,source_unchanged=unchanged)
    write(args.output/FILES[2],result);print('STABLE_INCIDENCE_SCALAR_CHECKS',result['passed'],a.checks,a.failure_count,flush=True)
    if not result['passed']:raise SystemExit(1)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();{'freeze':freeze,'run':run}[args.stage](args)
if __name__=='__main__':main()
