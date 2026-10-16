"""Own-frozen independent saved point-consensus decision/output checks.

The completed V16 paired scalar audit remains FAIL. Its complete failure log
must prove that none of its failed predicates belongs to POINT_ONLY. This
checker does not invent the number of original point-only predicates, rerun
their Jacobians, or turn the full parent audit into a PASS.
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
import re
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_consensus_decision_repair_20261011_v18'
PARENT=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
PARENT_CODE=CODE.parent/'pallet_hybrid_factor_subset_consensus_20261011_v16'
PRIVATE=Path('/tmp/pallet-consensus-decision-repair-private-20261011-v18')
METHODS=('HYBRID_POINT_CONSENSUS_ORIGINAL','HYBRID_POINT_CONSENSUS_RESTORED')
OLD_METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')
S15=CODE.parent/'pallet_factor_subset_consensus_20261011_v15/solver.py'
S15_SHA='4f447f8fa21ae11d90be19f7423b6c42b8dbde4cd8bacb32450b54dfe4c495a9'
S13=CODE.parent/'pallet_finite_edge_consensus_repair_20261011_v13/solver.py'
S13_SHA='3b8cdeb5bc189cb196be4066d095ead081f8a5c3de1b6ee62c28e9402621e4d7'
P4=CODE.parent/'pallet_cornerwise_independent_20261010_v4/pose.py'
P4_SHA='ef827022bef6807c5f7e4016d7ebefa1838c5f013f6d48be6f4932e0c9ad4a40'
FAILURE=re.compile(r'^frame\.(\d+)\.(HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS|HYBRID_FACTOR_SUBSET_POINT_ONLY)\.(proposal|refit)\.(\d+)\.(.+)$')
LIMITS=dict(new_models_GT_PnP_optimizer_projection_J_SVD_timing_seed_calls=0,
    original_full_geometry_scalar_validation_passed=False,original_point_check_count_not_invented=True,
    original_point_geometry_is_scoped_from_complete_parent_audit=True,
    physical_equivalence_or_geometry_not_recomputed=True,stored_spectra_recount_is_not_new_SVD=True,
    physical_ownership_or_global_uniqueness_certified=False,threshold_changes=0,
    saved_replay_latency_claim=False,pose_accuracy_success_not_certified=True)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [safe(x) for x in v]
    if isinstance(v,set):return [safe(x) for x in sorted(v,key=str)]
    return repr(v) if isinstance(v,float) and not math.isfinite(v) else v
def write(p,v):
    with Path(p).open('x') as f:json.dump(safe(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:yield json.loads(line)
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def ids(v):return isinstance(v,list) and v==sorted(set(v)) and all(type(k) is int and 0<=k<8 for k in v)
def sorted_key(c):return (-c['score_count'],c['truncated_sse_px2'],c['dimension_index'],c['logical_start_index'])

class Audit:
    def __init__(self):self.checks=0;self.failures=[];self.max_difference=0.
    def ok(self,v,label,actual=None,expected=None):
        self.checks+=1
        if not v:self.failures.append(dict(label=label,actual=actual,expected=expected))
    def eq(self,x,y,label):self.ok(x==y,label,x,y)
    def near(self,x,y,label):
        if isinstance(x,list) and isinstance(y,list):
            self.eq(len(x),len(y),label+'.arity')
            for i,(u,v) in enumerate(zip(x,y)):self.near(u,v,label+'.'+str(i))
            return
        if not finite(x) or not finite(y):self.ok(False,label,x,y);return
        d=abs(x-y);self.max_difference=max(self.max_difference,d)
        self.ok(d<=1e-7+1e-12*abs(y),label,x,y)

def literal(node,names):
    if isinstance(node,ast.Name):require(node.id in names,'undeclared source constant '+node.id);return names[node.id]
    if isinstance(node,(ast.List,ast.Tuple)):return [literal(v,names) for v in node.elts]
    if isinstance(node,ast.Call):
        if isinstance(node.func,ast.Name) and node.func.id=='list':
            require(len(node.args)==1 and not node.keywords,'safe literal list call');return list(literal(node.args[0],names))
        require(isinstance(node.func,ast.Name) and node.func.id=='dict' and not node.args and all(k.arg for k in node.keywords),'safe literal dict call')
        return {k.arg:literal(k.value,names) for k in node.keywords}
    return ast.literal_eval(node)
def source_witness():
    require(binding(S15)['sha256']==S15_SHA and binding(S13)['sha256']==S13_SHA and binding(P4)['sha256']==P4_SHA,'unchanged original numerical source')
    old=ast.parse(S15.read_text());method=next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='FactorSubsetConsensusBank')
    solve=next(n for n in method.body if isinstance(n,ast.FunctionDef) and n.name=='solve')
    branch=next(n for n in ast.walk(solve) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='same_support')
    require(ast.unparse(branch.body[0])=="ambiguity.append('OBSERVATION_CONSENSUS_AMBIGUITY')",'exact removed any-cost veto authority')
    src13=ast.parse(S13.read_text());p=next(n.value for n in src13.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    inherited=next(k.value for k in p.keywords if k.arg=='numerical_score_tie_px2')
    require(ast.dump(inherited,include_attributes=False)==ast.dump(ast.parse("P.POLICY['numerical_score_tie_px2']",mode='eval').body,include_attributes=False),'unchanged V13 inherited tie authority')
    p4=next(n.value for n in ast.parse(P4.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    tie=next(ast.literal_eval(k.value) for k in p4.keywords if k.arg=='numerical_score_tie_px2');require(tie==1e-8,'unchanged numerical score tie')
    tree=ast.parse((CODE/'solver.py').read_text());names={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('ORIGINAL','METHOD','METHODS','PRIMARY','PARENT_METHOD','TIE_PX2','POLICY'):
            names[n.targets[0].id]=literal(n.value,names)
    require(names['PARENT_METHOD']==OLD_METHODS[1] and names['METHODS']==list(METHODS) and names['TIE_PX2']==tie and names['POLICY']['same_support_different_cost_veto_removed'] is True,'one fixed count/SSE decision restoration')
    return dict(original_solver=binding(S15),original_finite_source=binding(S13),original_point_source=binding(P4),tie_px2=tie,
        original_any_cost_veto_AST=ast.dump(branch,include_attributes=False),new_solver=binding(CODE/'solver.py'),policy=names['POLICY'])

def rank(report,a,label):
    sv=report['singular_values'];a.ok(isinstance(sv,list) and all(finite(v) and v>=0 for v in sv),label+'.finite_spectrum')
    a.ok(all(x>=y for x,y in zip(sv,sv[1:])),label+'.ordered_spectrum')
    n=sum(v>sv[0]*1e-10 for v in sv) if sv and sv[0]>0 else 0;a.eq(report['numerical_rank'],n,label+'.stored_rank_recount');a.eq(n,6,label+'.rank6')
    weak=len(sv)!=6 or sv[-1]<sv[0]*1e-4;a.eq(weak,False,label+'.inherited_scaled_weak_gate');return weak

def candidate(c,U,H,a,label):
    fit,final=c['fit_input_ids'],c['final_inliers'];a.ok(ids(fit) and ids(final) and len(fit)>=4 and set(fit)<=set(final)<=set(U) and not set(fit)&set(H),label+'.actual_min4_no_H_fit')
    for key in ('fit_line_edges','final_line_inliers','whole_pool_line_edges'):a.eq(c[key],[],label+'.no_line.'+key)
    a.eq(c['whole_pool_point_ids'],U,label+'.same_whole_U');a.eq(c['rejected'],None,label+'.no_rejection');a.eq(c['fit_pool_all_final8px_inliers'],True,label+'.stable_actual_refit')
    g=c['point_line_geometry'];a.eq(g['point_ids'],fit,label+'.geometry_fit_ids');a.eq(g['line_edges'],[],label+'.point_only_geometry')
    for name in ('observed','modeled'):
        weak=rank(g[name+'_normal_joint'],a,label+'.finite.'+name);a.eq(g[name+'_weak'],weak,label+'.finite_weak.'+name)
    guard=c['stable_incidence_guard'];a.eq(guard['accepted'],True,label+'.incidence_accepted');a.eq(guard['guard_state'],'PASSED_STABLE_INCIDENCE',label+'.incidence_state')
    a.eq(guard['actual_fit_input_ids'],fit,label+'.same_incidence_fit');a.eq(guard['actual_fit_line_edges'],[],label+'.incidence_no_edges')
    for name in ('observed','modeled'):
        weak=rank(guard['stable_incidence_geometry'][name+'_normal_joint'],a,label+'.incidence.'+name);a.eq(guard[name+'_weak'],weak,label+'.incidence_weak.'+name)
    a.ok(len(c['model_camera_depths_m'])==8 and all(finite(z) and z>1e-9 for z in c['model_camera_depths_m']),label+'.inherited_positive_depth')
    a.ok(len(c['projected'])==8 and all(len(p)==2 and all(finite(v) for v in p) for p in c['projected']),label+'.stored_finite_projection')
    score=c['whole_pool_score'];r=score['point_residual_norm_px'];a.eq(score['point_ids'],U,label+'.score_U');a.eq(score['line_edges'],[],label+'.score_no_lines')
    a.ok(len(r)==len(U) and all(finite(v) and v>=0 for v in r),label+'.finite_saved_point_errors')
    inliers=[k for k,v in zip(U,r) if v<=8.];a.eq(final,inliers,label+'.8px_vote');a.eq(c['score_count'],len(inliers),label+'.one_point_one_vote')
    a.eq(score['final_inliers'],inliers,label+'.whole_pool_inlier_ids');a.eq(score['final_line_inliers'],[],label+'.whole_pool_no_line_inliers')
    a.eq(score['total_inlier_factor_count'],len(inliers),label+'.whole_pool_count')
    trunc=math.fsum(min(v*v,64.) for v in r);sse=math.fsum(v*v for v in r)
    a.near(c['truncated_sse_px2'],trunc,label+'.independent_whole_U_truncated_SSE');a.near(score['factor_truncated_sse_px2'],trunc,label+'.score_truncated_SSE');a.near(score['factor_sse_px2'],sse,label+'.whole_U_SSE')
    rr=c['scalar_residuals_px'];a.ok(len(rr)==2*len(fit) and all(finite(v) for v in rr),label+'.stored_actual_fit_residuals')
    a.near(c['sse_px2'],math.fsum(v*v for v in rr),label+'.actual_fit_SSE')
    a.near(c['soft_l1_cost'],math.fsum(64.*(math.sqrt(1.+v*v/64.)-1.) for v in rr),label+'.actual_fit_cost_not_ranking_cost')

def decision(s,H,a,label):
    U=s['used'];a.ok(ids(U) and ids(H) and not(set(U)&set(H)),label+'.same_actual_U_no_H_center');a.eq(s['hidden'],H,label+'.original_H')
    for key in ('line_edges','fit_line_edges','final_line_inliers','consumed_edges'):a.eq(s[key],[],label+'.point_only.'+key)
    for key in ('prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','same_edge_point_line_double_count','center_used_for_fit_or_score'):a.eq(s[key],False,label+'.no_fake.'+key)
    accepted=[c for c in s['all_candidate_solutions'] if c['accepted'] is True]
    for i,c in enumerate(accepted):candidate(c,U,H,a,label+'.accepted.'+str(i))
    best=s.get('best_candidate');relations=s['alternatives'];extra=ties=non_eq=0
    if best is None:
        a.eq(accepted,[],label+'.no_best_no_accepted');a.eq(relations,[],label+'.no_best_no_relations');a.eq(s['available'],False,label+'.insufficient_stays_unavailable')
    else:
        ranked=sorted(accepted,key=sorted_key);a.ok(bool(ranked) and best==ranked[0],label+'.unchanged_count_then_SSE_best');a.eq(s['viable_candidate_count'],len(accepted),label+'.viable_count');a.eq(len(relations),len(accepted)-1,label+'.all_alternative_relations')
        for i,(c,w) in enumerate(zip(ranked[1:],relations)):
            a.ok(all(w[k]==v for k,v in c.items()),label+'.alternative_original_witness.'+str(i))
            delta=w['truncated_sse_px2']-best['truncated_sse_px2'];tie=w['score_count']==best['score_count'] and abs(delta)<=1e-8
            support=w['final_inliers']==best['final_inliers'] and w['final_line_inliers']==best['final_line_inliers'];eq=w['numerically_equivalent_to_selected'];a.ok(type(eq) is bool,label+'.stored_equivalence_boolean')
            a.eq(w['objective_difference_from_selected_px2'],delta,label+'.same_whole_U_cost_delta');a.eq(w['numerical_objective_tie'],tie,label+'.fixed_numerical_tie');a.eq(w['same_actual8px_consensus_as_selected'],support,label+'.same_support')
            if not eq:non_eq+=1;ties+=int(tie);extra+=int(support and not tie)
        a.eq(s['multiple_solutions'],bool(non_eq),label+'.multiplicity_preserved')
    restored=s['state']=='AMBIGUOUS_FINITE_CONSENSUS' and ties==0 and extra>0
    if s['state']=='AMBIGUOUS_FINITE_CONSENSUS':
        reasons=set(s['reason'].split(';'));a.ok(reasons and reasons<={'OBSERVATION_CONSENSUS_AMBIGUITY','NUMERICAL_CONSENSUS_TIE'},label+'.declared_old_ambiguity')
        a.eq(s['available'],False,label+'.old_ambiguity_unavailable');a.eq(s['unresolved_ambiguity'],True,label+'.old_veto_recorded')
        if restored:a.eq(reasons,{'OBSERVATION_CONSENSUS_AMBIGUITY'},label+'.only_extra_veto_removed')
    elif s['available']:a.eq(ties,0,label+'.true_tie_cannot_already_be_NEW')
    return dict(accepted_candidates=len(accepted),extra_veto_alternatives=extra,true_tie_alternatives=ties,non_equivalent_alternatives=non_eq),restored

def input_paths(args):
    f={k:args.input/n for k,n in dict(geometry='GEOMETRY_SEALED.jsonl.gz',ledgers='CONTROL_LEDGERS.jsonl.gz',repair_protocol='PROTOCOL.json',
        geometry_seal='GEOMETRY_SEAL.json',repair_receipt='CONTROL_RECEIPT.json',parent_population_checks='PARENT_POPULATION_CHECKS.json',started='CONTROL_STARTED.json').items()}
    f.update(checker=Path(__file__),original_solver=S15,original_finite_source=S13,original_point_source=P4,parent_geometry=PARENT/'GEOMETRY_SEALED.jsonl.gz',parent_ledgers=PARENT/'CONTROL_LEDGERS.jsonl.gz')
    for name in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json',
        'FIRST_SCALAR_ENVIRONMENT_FAILURE.json','VALIDATION_READY_PROTOCOL.json','VALIDATION_READY_STARTED.json','VALIDATION_READY_CHECKS.json',
        'TRACE_VALIDATION_PROTOCOL.json','TRACE_VALIDATION_STARTED.json','TRACE_VALIDATION_CHECKS.json',
        'FAILURE_WITNESS_PROTOCOL.json','FAILURE_WITNESS_STARTED.json','FAILURE_WITNESS_CHECKS.json','FAILURE_WITNESSES.json',
        'FAILURE_SUMMARY_PROTOCOL.json','FAILURE_SUMMARY_STARTED.json','FAILURE_SUMMARY_CHECKS.json','FAILURE_WITNESS_SUMMARY.json',
        'AMBIGUITY_DIAGNOSTIC_PROTOCOL.json','AMBIGUITY_DIAGNOSTIC_STARTED.json','AMBIGUITY_DIAGNOSTIC_CHECKS.json','AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz'):
        f['parent:'+name]=PARENT/name
    for name in ('validation_checks.py','validation_authority_ready.py','validation_environment_ready.py','validation_failure_trace.py','failure_witness.py','failure_witness_summary.py','saved_ambiguity_diagnostic.py'):
        f['parent_code:'+name]=PARENT_CODE/name
    for name in ('solver.py','pipeline.py','runner.py'):f['code:'+name]=CODE/name
    return f

def scoped_parent_authority(f,b):
    def p(name):return read(f['parent:'+name])
    ready,rp,rs=[p('VALIDATION_READY_'+x+'.json') for x in ('CHECKS','PROTOCOL','STARTED')]
    trace,tp,ts=[p('TRACE_VALIDATION_'+x+'.json') for x in ('CHECKS','PROTOCOL','STARTED')]
    require(ready['complete'] is True and ready['passed'] is False and ready['checks']==198134007 and ready['failure_count']==564 and ready['exception'] is None and len(ready['failures'])==100,'preserved full V16 FAIL; not a geometry PASS')
    require(ready['inputs']==rp['inputs']==rs['inputs'] and same(ready['own_protocol'],b['parent:VALIDATION_READY_PROTOCOL.json']) and same(rs['own_protocol'],b['parent:VALIDATION_READY_PROTOCOL.json'])
        and ready['new_production_calls']==0 and same(ready['inputs']['geometry'],b['parent_geometry']) and same(ready['inputs']['ledgers'],b['parent_ledgers'])
        and same(ready['ready_checker'],b['parent_code:validation_environment_ready.py']) and same(ready['authority_code'],b['parent_code:validation_authority_ready.py']),'original failed READY byte authority')
    require(trace['complete'] is True and trace['passed'] is False and trace['trace_complete'] is True and trace['trace_passed'] is True and trace['scalar_validation_passed'] is False
        and trace['checks']==198134007 and trace['failure_count']==564 and trace['exception'] is None and trace['all_failed_predicates_retained'] is True and len(trace['failures'])==564,'complete unchanged all-failure trace; full geometry stays FAIL')
    require(trace['failures'][:100]==ready['failures'] and trace['max_scalar_absolute_difference']==ready['max_scalar_absolute_difference'] and trace['counts']==ready['counts'] and trace['native_counts']==ready['native_counts'] and trace['cusp_reconciliations']==ready['cusp_reconciliations'],'exact original trace prefix/counts/scalars')
    require(trace['inputs']==tp['inputs']==ts['inputs'] and same(trace['own_protocol'],b['parent:TRACE_VALIDATION_PROTOCOL.json']) and same(ts['own_protocol'],b['parent:TRACE_VALIDATION_PROTOCOL.json'])
        and same(trace['trace_checker'],b['parent_code:validation_failure_trace.py']) and same(trace['inputs']['geometry'],b['parent_geometry']),'same complete trace geometry/source bytes')
    marker=trace['failure_trace'];require(marker==tp['failure_trace']==ts['failure_trace'] and marker['logging_guard_AST_changes']==1 and marker['scalar_formula_threshold_or_acceptance_AST_changes']==0 and marker['whole_original_scalar_source_restore_proven'] is True and marker['geometry_solver_or_operating_rows_modified'] is False,'logging-only authority; no scalar repair')
    for v in trace['failures']:
        m=FAILURE.fullmatch(v['check']);require(m is not None and int(m.group(1))<245 and m.group(2)==OLD_METHODS[0],'fail closed: every retained failure must be PRIMARY, not POINT_ONLY')
    saved,wc,wp,ws=[p(x) for x in ('FAILURE_WITNESSES.json','FAILURE_WITNESS_CHECKS.json','FAILURE_WITNESS_PROTOCOL.json','FAILURE_WITNESS_STARTED.json')]
    require(wc['complete'] is True and wc['passed'] is True and wc['error'] is None and wc['retained_failures_joined']==564 and wc['rows_streamed']==490 and same(wc['witnesses'],b['parent:FAILURE_WITNESSES.json']),'actual all564 stored-witness join PASS')
    require(wc['inputs']==wp['inputs']==ws['inputs'] and same(wc['own_protocol'],b['parent:FAILURE_WITNESS_PROTOCOL.json']) and same(wp['inputs']['chosen_failure_checks'],b['parent:TRACE_VALIDATION_CHECKS.json']) and same(wp['inputs']['script'],b['parent_code:failure_witness.py']),'actual witness freeze and source')
    require([w['stored_failure'] for w in saved['witnesses']]==trace['failures'] and all(c['method']==OLD_METHODS[0] for c in saved['contexts'].values()),'all original failure contexts disjoint from point-only')
    summary,sc,sp,ss=[p(x) for x in ('FAILURE_WITNESS_SUMMARY.json','FAILURE_SUMMARY_CHECKS.json','FAILURE_SUMMARY_PROTOCOL.json','FAILURE_SUMMARY_STARTED.json')]
    require(sc['complete'] is True and sc['passed'] is True and sc['error'] is None and same(sc['summary'],b['parent:FAILURE_WITNESS_SUMMARY.json']) and sc['inputs']==sp['inputs']==ss['inputs'] and same(sc['own_protocol'],b['parent:FAILURE_SUMMARY_PROTOCOL.json']) and same(sp['inputs']['checker'],b['parent_code:failure_witness_summary.py']),'actual independent failure summary PASS')
    require(summary['complete'] is True and summary['original_scalar_validation_passed'] is False and summary['original_scalar_checks']==198134007 and summary['original_scalar_failures']==564
        and summary['methods'][OLD_METHODS[1]].get('failed_predicates',0)==0 and summary['methods'][OLD_METHODS[1]]['unique_candidate_contexts']==0 and summary['methods'][OLD_METHODS[1]]['unique_frames']==0,'explicit point-only no-failed-predicate scope')
    require([w['stored_failure'] for w in summary['failed_predicates']]==trace['failures'] and all(w['method']==OLD_METHODS[0] for w in summary['failed_predicates']) and same(sp['inputs']['TRACE_VALIDATION_CHECKS.json'],b['parent:TRACE_VALIDATION_CHECKS.json']) and same(sp['inputs']['FAILURE_WITNESSES.json'],b['parent:FAILURE_WITNESSES.json']),'same all564 summary authority')
    ac,ap,astart=[p('AMBIGUITY_DIAGNOSTIC_'+x+'.json') for x in ('CHECKS','PROTOCOL','STARTED')]
    require(ac['complete'] is True and ac['passed'] is True and ac['rows_complete']==490 and ac['error'] is None and ac['cleanup_errors']==[] and ac['original_full_geometry_scalar_passed'] is False
        and ac['inputs']==ap['inputs']==astart['inputs'] and same(ac['own_protocol'],b['parent:AMBIGUITY_DIAGNOSTIC_PROTOCOL.json']) and same(astart['own_protocol'],b['parent:AMBIGUITY_DIAGNOSTIC_PROTOCOL.json'])
        and same(ac['rows'],b['parent:AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz']) and same(ap['inputs']['checker'],b['parent_code:saved_ambiguity_diagnostic.py'])
        and same(ap['inputs']['GEOMETRY_SEALED.jsonl.gz'],b['parent_geometry']),'same actual all490 stored ambiguity diagnosis')
    return dict(original_full_scalar_validation_passed=False,original_full_scalar_checks=198134007,original_full_scalar_failures=564,
        all_failed_predicates_retained=True,point_only_original_failed_predicates=0,independent_parent_point_check_count_not_claimed=True,
        certified_scope='unchanged POINT_ONLY candidates and fields covered by the completed original paired audit; no failed predicate in this arm',
        full_geometry_PASS_not_claimed=True,new_geometry_calls=0)

def expected_solver(old,q,H,checks,restored,policy):
    s=copy.deepcopy(old)
    if old['state']=='AMBIGUOUS_FINITE_CONSENSUS':
        if checks['true_tie_alternatives']:s['reason']='NUMERICAL_CONSENSUS_TIE'
        elif restored:
            best=old['best_candidate'];points=copy.deepcopy(q)
            for k in H:points[k]=copy.deepcopy(best['projected'][k])
            s.update(available=True,pose_available=True,new_pose_estimated=True,no_pose=False,unresolved_ambiguity=False,
                state='NEW_POSE',reason='restored_original_count_SSE_point_consensus_decision',output_status='NEW_POSE',
                local_refinement_estimated=True,dimension_index=best['dimension_index'],selected_hypothesis=best['selected_hypothesis'],
                fit_input_ids=best['fit_input_ids'],input_inliers=best['fit_input_ids'],final_inliers=best['final_inliers'],inliers=best['final_inliers'],
                fit_line_edges=[],final_line_inliers=[],point_inlier_count=len(best['final_inliers']),unique_line_inlier_count=0,
                total_inlier_factor_count=best['score_count'],points_final=points,hidden_reprojected=bool(H),
                factor_sse_px2=best['whole_pool_score']['factor_sse_px2'],residuals_used_px=best['whole_pool_score']['point_residual_norm_px'],
                line_residual_RMS_px=best['whole_pool_score']['edge_block_norm_px'])
            for key in ('R_cf','R_physical','centroid','cf_extents','projected','point_line_geometry','stable_incidence_guard','scalar_residuals_px','soft_l1_cost','sse_px2','truncated_sse_px2'):s[key]=best[key]
    s.update(global_uniqueness_proven=False,decision_restoration_policy=policy,
        original_same_support_veto_state=old['state'],original_same_support_veto_reason=old['reason'],
        inherited_solver_policy_is_candidate_generation_history=True,operative_output_decision_policy=policy)
    return s

def packet(parent,pair,ledger,a,policy,counts,label):
    a.eq(parent['method'],OLD_METHODS[1],label+'.parent_point_only');a.eq(parent['oracle'],False,label+'.nonoracle')
    a.ok(not set(parent)&{'evaluation_reference','pose','human','GT','visibility_truth','metric_targets'},label+'.no_truth_input')
    H,q,native=parent['hidden_initial'],parent['input_points'],parent['original_native_N3_points'];s=parent['solver']
    checks,restored=decision(s,H,a,label+'.decision');expected=expected_solver(s,q,H,checks,restored,policy)
    d=dict(schema='point_only_decision_restoration_receipt_v18',original_solver_semantic_sha256=semantic(s),execution_kind='SAVED_DECISION_REPLAY',
        checks=checks,original_state=s['state'],new_state=expected['state'],restored=restored,true_non_equivalent_numeric_tie_retained=checks['true_tie_alternatives']>0,
        new_display_H_assignments=len(H) if restored else 0,best_candidate_reselected=False,all_candidate_witnesses_changed=False,
        new_solver_optimizer_projection_J_SVD_calls=0,saved_replay_new_solver_calls=0,inherited_geometry_validity_independent_audit_required=True,
        latency_claim=False,cached_replay_latency_claim=False)
    original=copy.deepcopy(parent);original.update(method=METHODS[0],parent_method=OLD_METHODS[1],original_point_only_row_semantic_sha256=semantic(parent),
        decision_execution_kind='SAVED_ORIGINAL_DECISION_COPY',actual_new_solver_calls=0,saved_decision_replay=True,cached_accuracy_row_used=True,solver_replay_latency_claim=False)
    a.ok(pair[0]==original,label+'.ORIGINAL_full_row_copy');r=pair[1];a.ok(r['solver']==expected,label+'.RESTORED_pure_solver_decision')
    a.ok(r['solver']['all_candidate_solutions']==s['all_candidate_solutions'] and r['solver']['alternatives']==s['alternatives'] and r['solver'].get('best_candidate')==s.get('best_candidate'),label+'.immutable_best_and_full_candidates')
    new=bool(expected['available']);initial=parent['initial_pose'];fallback=not new and bool(initial['available']);out=copy.deepcopy(native)
    if new:
        for k in range(8):
            if k in H:out[k]=expected['projected'][k]
            elif all(finite(v) for v in q[k]) and q[k]!=[-1,-1]:out[k]=q[k]
    adopted=set(parent['selected_corner_ids']);sources=['FINAL_POSE_REPROJECTION' if new and k in H else 'VALIDATED_BOUNDARY_INTERSECTION' if new and k in adopted else 'NATIVE_N3_RGB' for k in range(8)]+['UNCHANGED_DETECTOR_CENTER']
    fields=dict(method=METHODS[1],parent_method=OLD_METHODS[1],solver=expected,native_points=out,actual_pose=expected if new else initial,
        new_pose_estimated=new,pose_available=new or bool(initial['available']),fallback_used=fallback,no_pose=not(new or bool(initial['available'])),
        output_status='NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',hidden_reprojected=bool(new and H),
        output_coordinate_sources=sources,reprojected_ids=H if new else [],initial_H_applied=True,original_self_hidden_initial=H,
        reprojections_reused_as_observations=False,partial_lines_not_pose_inputs=True,local_point_line_refinement=False,global_uniqueness_proven=False,
        decision_restoration=d,decision_restoration_policy=policy,decision_execution_kind='SAVED_DECISION_REPLAY',original_point_only_row_semantic_sha256=semantic(parent),
        selected_output_pose_was_actually_fit_in_parent=bool(new and expected.get('best_candidate') is not None),saved_decision_replay=True,cached_accuracy_row_used=True,
        solver_replay_latency_claim=False,numerical_NEW_is_not_pose_accuracy_success=True)
    for key,value in fields.items():a.ok(r[key]==value,label+'.restored_field.'+key)
    changed=set(fields)|{'hidden_after','hidden_set_changed','restored_pose_mask_diagnostic'}
    a.eq(set(r),set(parent)|changed,label+'.all_row_fields_declared');a.ok(all(r[k]==v for k,v in parent.items() if k not in changed),label+'.other_parent_fields_preserved')
    if restored:
        mask=r['restored_pose_mask_diagnostic'];a.eq(mask['margin_deg'],2.,label+'.unchanged_mask_margin');a.eq(mask['used_for_acceptance_or_pose_selection'],False,label+'.mask_is_diagnostic')
        R,t,e=expected['R_cf'],expected['centroid'],expected['cf_extents'];camera=[-math.fsum(R[j][i]*t[j] for j in range(3)) for i in range(3)];cosines=[]
        for signs in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)):
            ray=[camera[i]-signs[i]*e[i]/2 for i in range(3)];length=math.sqrt(math.fsum(v*v for v in ray));require(length>0,'nonzero display mask ray')
            cosines.append(max(signs[i]*ray[i]/length for i in range(3)))
        a.near(mask['cosine'],cosines,label+'.independent_display_mask_cosines')
        after=[i for i,v in enumerate(mask['cosine']) if v < -math.sin(math.radians(2.))];a.eq(r['hidden_after'],after,label+'.recorded_mask_predicate')
    else:a.eq(r['hidden_after'],parent['hidden_after'],label+'.unrestored_mask_unchanged');a.eq(r['restored_pose_mask_diagnostic'],None,label+'.no_extra_mask_recalculation')
    a.eq(r['hidden_set_changed'],bool(new and set(r['hidden_after'])!=set(H)),label+'.mask_change_recorded_without_veto')
    a.eq(out[8],native[8],label+'.center_preserved');a.ok(not(set(expected['fit_input_ids'])&set(H)),label+'.H_never_fit')
    if not new:a.eq(out,native,label+'.full_N3_fallback_coordinates');a.eq(r['actual_pose'],initial,label+'.full_original_initial_pose')
    values=dict(schema='saved_point_only_decision_pair_ledger_v18',id=parent['id'],session=parent['session'],methods=list(METHODS),parent_method=OLD_METHODS[1],
        parent_row_semantic_sha256=semantic(parent),parent_solver_semantic_sha256=semantic(s),original_row_semantic_sha256=semantic(pair[0]),restored_row_semantic_sha256=semantic(r),
        decision=d,inherited_solver_operation_counts=s['operation_counts'],actual_new_solver_calls=0,actual_new_optimizer_projection_J_SVD_calls=0,
        original_output_view_copies=1,restored_output_view_decisions=1,replay_display_H_assignments=d['new_display_H_assignments'],scalar_pose_to_mask_diagnostic_calls=int(restored),
        original_H=H,input_points=q,original_native_N3_points=native,independent_point_only_scalar_authority_is_driver_requirement=True,latency_claim=False,cached_replay_latency_claim=False,
        narrow_point_only_scope_passed=True,original_full_paired_scalar_passed=False)
    for key,value in values.items():a.ok(ledger[key]==value,label+'.ledger.'+key)
    a.eq(ledger['source_ambiguity_classification'],'ONLY_DIFFERENT_COST_SAME_SUPPORT_VETO' if restored else 'NUMERICAL_TIE_PRESENT' if checks['true_tie_alternatives'] else 'NO_ACCEPTED_BEST' if s.get('best_candidate') is None else 'NO_REPORTED_VETO_WITH_ACCEPTED_BEST',label+'.source_classification')
    counts.update(frames_complete=1,geometry_rows=2,ledger_rows=1,parent_geometry_rows_visited=2,ambiguity_rows_visited=2,
        original_point_only_rows_selected=1,pure_restoration_decisions=1,restored_rows=int(restored),true_tie_rows_retained=int(checks['true_tie_alternatives']>0),
        replay_display_H_assignments=d['new_display_H_assignments'],scalar_pose_to_mask_diagnostic_calls=int(restored))

def metadata(f,b,info):
    core,seal,receipt,pop=[read(f[k]) for k in ('repair_protocol','geometry_seal','repair_receipt','parent_population_checks')]
    require(core['schema']=='fixed_point_consensus_decision_restoration_protocol_v18' and core['methods']==list(METHODS) and core['frames']==core['ledger_rows']==245 and core['rows']==490 and core['policy']==info['policy'],'fixed paired decision protocol')
    require(seal['schema']=='point_consensus_decision_restoration_geometry_seal_v18' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['methods']==list(METHODS) and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True and seal['original_full_paired_scalar_passed'] is False and seal['narrow_point_only_scope_passed'] is True,'complete paired seal; parent full FAIL preserved')
    for key,alias in (('protocol','repair_protocol'),('geometry','geometry'),('ledgers','ledgers'),('control_receipt','repair_receipt'),('parent_population_checks','parent_population_checks')):require(same(seal[key],b[alias]),'sealed bytes '+key)
    require(receipt['schema']=='point_consensus_decision_restoration_execution_v18' and receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None and receipt['cleanup_errors']==[] and receipt['source_inputs_unchanged'] is True and receipt['new_production_calls']==receipt['new_numerical_pose_fit_calls']==0 and receipt['latency_claim'] is False,'complete zero-fit saved receipt')
    require(same(receipt['protocol'],b['repair_protocol']) and same(receipt['published_outputs']['GEOMETRY_SEALED.jsonl.gz'],b['geometry']) and same(receipt['published_outputs']['CONTROL_LEDGERS.jsonl.gz'],b['ledgers']),'receipt raw byte joins')
    require(pop['schema']=='point_consensus_decision_parent_population_v18' and pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and pop['parent_rows']==490 and pop['parent_point_only_rows']==245 and pop['scored_or_human_input'] is False and len(pop['ordered_ids'])==len(set(pop['ordered_ids']))==245 and same(core['parent_population_checks'],b['parent_population_checks']),'full ordered point parent population')
    require(read(f['started'])['inputs']==core['inputs'] and same(read(f['started'])['protocol'],b['repair_protocol']) and read(f['started'])['new_production_calls']==0,'source ownfreeze before saved replay')
    for key in ('checker','code:solver.py','code:pipeline.py','code:runner.py','parent_geometry','parent_ledgers'):
        require(any(same(v,b[key]) and v['path']==b[key]['path'] for v in core['inputs'].values()),'main-bound current code/raw '+key)
    ps,pr=[read(f['parent:'+x]) for x in ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json')]
    require(ps['complete'] is True and ps['frames']==ps['ledger_rows']==245 and ps['rows']==490 and ps['GT_read_allowed'] is False and ps['cleanup_completed_before_seal'] is True
        and pr['complete'] is True and pr['cleanup_completed'] is True and pr['error'] is pr['cleanup_error'] is None and same(ps['geometry'],b['parent_geometry']) and same(ps['ledgers'],b['parent_ledgers']),'immutable complete GT-free V16 geometry; no full scalar PASS asserted')
    scope=scoped_parent_authority(f,b);return core,receipt,pop,scope

def guard(args,stage):
    for p in (args.input,args.output):require(p.is_dir() and not any(q.is_symlink() for q in (p,*p.parents)),'existing non-symlink folders')
    require(args.output.resolve()==DOC or args.output.resolve().is_relative_to(PRIVATE),'exclusive own output')
    for name in FILES if stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args();guard(args,args.stage)
    f=input_paths(args);b={k:binding(v) for k,v in f.items()};info=source_witness()
    if args.stage=='freeze':write(args.output/FILES[0],dict(schema='point_consensus_decision_scalar_protocol_v18',inputs=b,checker=b['checker'],methods=list(METHODS),frames=245,rows=490,source_witness=info,limits=LIMITS,frozen_before_own_arithmetic=True));return
    own=read(args.output/FILES[0]);ob=binding(args.output/FILES[0]);require(own['schema']=='point_consensus_decision_scalar_protocol_v18' and own['inputs']==b and own['checker']==b['checker'] and own['source_witness']==info and own['limits']==LIMITS and own['frozen_before_own_arithmetic'] is True,'unchanged own freeze')
    write(args.output/FILES[1],dict(inputs=b,own_protocol=ob,checker=b['checker'],new_production_calls=0));a=Audit();counts=Counter();statuses={m:Counter() for m in METHODS};scope=None;error=None
    try:
        core,receipt,pop,scope=metadata(f,b,info);parent=iter(rows(f['parent_geometry']));geometry=iter(rows(f['geometry']));ledgers=iter(rows(f['ledgers']));diagnoses=iter(rows(f['parent:AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz']))
        for i,fid in enumerate(pop['ordered_ids']):
            primary=next(parent);point=next(parent);pair=[next(geometry),next(geometry)];ledger=next(ledgers);diagnostic_pair=[next(diagnoses),next(diagnoses)]
            require((primary['id'],primary['method'])==(fid,OLD_METHODS[0]) and (point['id'],point['method'])==(fid,OLD_METHODS[1]) and primary['session']==point['session'],'ordered paired source population');del primary
            require(all((r['id'],r['method'],r['session'])==(fid,m,point['session']) for r,m in zip(pair,METHODS)) and (ledger['id'],ledger['session'])==(fid,point['session']),'ordered490/245 output join')
            require(all((d['id'],d['method'],d['session'])==(fid,m,point['session']) for d,m in zip(diagnostic_pair,OLD_METHODS)) and diagnostic_pair[1]['original_solver_semantic_sha256']==semantic(point['solver']) and diagnostic_pair[1]['classification']==ledger['source_ambiguity_classification'],'same diagnosed source decision')
            packet(point,pair,ledger,a,info['policy'],counts,'frame.'+str(i))
            for r in pair:statuses[r['method']][r['output_status']]+=1
            require(same(ledger['parent_ledger_binding'],b['parent_ledgers']) and ledger['parent_ledger_binding']['path']==b['parent_ledgers']['path'],'main-bound parent ledger provenance')
            del point,pair,ledger,diagnostic_pair
        require(next(parent,None) is next(geometry,None) is next(ledgers,None) is next(diagnoses,None) is None,'all490/245 complete; no dropped rows')
        for key,value in counts.items():a.eq(receipt['actual_execution'].get(key,0),value,'receipt.actual_execution.'+key)
        a.eq(receipt['actual_execution']['parent_ledger_rows_visited'],245,'receipt.recorded_parent_ledger_rows_not_redecoded_this_pass')
        for key in ('actual_new_solver_calls','actual_new_optimizer_calls','actual_new_projection_calls','actual_new_J_calls','actual_new_SVD_calls','actual_new_model_GT_RGB_training_calls'):a.eq(receipt['actual_execution'][key],0,'receipt.zero_new_production.'+key)
        a.eq(dict(statuses),receipt['output_status_counts'],'receipt.all_operating_statuses');a.eq(counts['frames_complete'],245,'full245');a.eq(counts['geometry_rows'],490,'paired490');a.eq(counts['restored_rows'],161,'exact_original_extra_veto_population');a.eq(counts['true_tie_rows_retained'],1,'true_numerical_tie_retained')
        a.eq(statuses[METHODS[0]]['NEW_POSE'],79,'original_NEW79_preserved');a.eq(statuses[METHODS[1]]['NEW_POSE'],240,'restored240_same_best_outputs')
        require({k:binding(v) for k,v in f.items()}==b,'all inputs unchanged after own checks')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    result=dict(schema='point_consensus_decision_scalar_checks_v18',complete=error is None,passed=error is None and not a.failures,inputs=b,own_protocol=ob,checker=b['checker'],
        checks=a.checks,failure_count=len(a.failures),failures=a.failures,exception=error,max_scalar_absolute_difference=a.max_difference,source_witness=info,
        scope=scope,limits=LIMITS,counts=dict(counts),output_status_counts={m:dict(v) for m,v in statuses.items()},new_production_calls=0,first_failure_preserved=True)
    write(args.output/FILES[2],result);require(result['passed'],'point-only decision/output validation failed; preserve original audit and first failure')

if __name__=='__main__':main()
