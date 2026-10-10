"""Classify the frozen S15 ambiguity veto from saved, GT-free scalar fields.

No new equivalence, projection, Jacobian, SVD, solver, score or model is run.
The original full scalar audit remains complete-but-FAIL; this narrower audit
can check the reported decision without certifying its geometry or accuracy.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import traceback

REPO=Path(__file__).resolve().parents[3];CODE=Path(__file__).resolve().parent
DOC=REPO/'_docs/experiments/pallet_hybrid_factor_subset_consensus_20261011_v16'
SOLVER=REPO/'scripts/research/pallet_factor_subset_consensus_20261011_v15/solver.py'
SOLVER_SHA='4f447f8fa21ae11d90be19f7423b6c42b8dbde4cd8bacb32450b54dfe4c495a9'
FINITE=REPO/'scripts/research/pallet_finite_edge_consensus_repair_20261011_v13/solver.py'
FINITE_SHA='3b8cdeb5bc189cb196be4066d095ead081f8a5c3de1b6ee62c28e9402621e4d7'
ORIGINAL_SHA='1567db06b27e0394e95a539c943eac19eb57b5d257ebccf8d24a65866bbf697c'
METHODS=('HYBRID_FACTOR_SUBSET_FINITE_EDGE_CONSENSUS','HYBRID_FACTOR_SUBSET_POINT_ONLY')
FILES=('AMBIGUITY_DIAGNOSTIC_PROTOCOL.json','AMBIGUITY_DIAGNOSTIC_STARTED.json',
    'AMBIGUITY_DIAGNOSTIC_ROWS.jsonl.gz','AMBIGUITY_DIAGNOSTIC_CHECKS.json')
LIMITS=dict(new_GT_scores_model_projection_solver_optimizer_J_SVD_calls=0,
    equivalence_is_stored_reported_predicate_not_independently_recomputed=True,
    rank_and_acceptance_are_stored_flags_not_new_geometry_validation=True,
    original_full_geometry_scalar_validation_passed=False,
    same_binary_consensus_is_not_equal_objective_or_physical_truth=True,
    different_fit_pool_soft_costs_not_compared=True,threshold_changes=0,
    physical_ownership_certified=False,global_root_completeness_proven=False,
    performance_cause_or_counterfactual_pose_accuracy_established=False,
    original_pose_outputs_changed=False,automatic_retry=False)

def require(v,s):
    if not v:raise ValueError(s)
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with gzip.open(p,'rt') as f:
        for line in f:require(bool(line.strip()),'blank saved row');yield json.loads(line)
def binding(p):
    p=Path(p).absolute();require(p.is_file() and not any(q.is_symlink() for q in (p,*p.parents)),'missing/symlink '+str(p));h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return dict(path=str(p.relative_to(REPO)) if p.is_relative_to(REPO) else p.name,bytes=p.stat().st_size,sha256=h.hexdigest())
def same(a,b):return isinstance(a,dict) and isinstance(b,dict) and all(a.get(k)==b.get(k) for k in ('bytes','sha256'))
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def paths(folder):
    f={n:folder/n for n in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json',
        'GEOMETRY_SEALED.jsonl.gz','CONTROL_LEDGERS.jsonl.gz','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json',
        'FIRST_SCALAR_ENVIRONMENT_FAILURE.json','VALIDATION_READY_PROTOCOL.json','VALIDATION_READY_STARTED.json','VALIDATION_READY_CHECKS.json')}
    f.update(checker=Path(__file__),solver=SOLVER,finite=FINITE,original_checker=CODE/'validation_checks.py',
        ready_checker=CODE/'validation_environment_ready.py',authority_code=CODE/'validation_authority_ready.py',pipeline=CODE/'pipeline_ready.py')
    return f
def source_witness(b):
    require(b['solver']['sha256']==SOLVER_SHA and b['finite']['sha256']==FINITE_SHA and b['original_checker']['sha256']==ORIGINAL_SHA,'immutable numerical and failed scalar sources')
    tree=ast.parse(SOLVER.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='solve')
    targets={n.targets[0].id:n.value for n in ast.walk(fn) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('tied','same_support')}
    expected=ast.parse("tied=w['score_count']==best['score_count'] and abs(delta)<=S13.POLICY['numerical_score_tie_px2']\nsame_support=w['final_inliers']==best['final_inliers'] and w['final_line_inliers']==best['final_line_inliers']").body
    require(set(targets)=={'tied','same_support'} and all(ast.dump(targets[n.targets[0].id],include_attributes=False)==ast.dump(n.value,include_attributes=False) for n in expected),'exact stored tie/support predicates')
    decisions=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==ast.dump(ast.parse('not equivalent',mode='eval').body,include_attributes=False)]
    expected_decision=ast.parse("if not equivalent:\n result['multiple_solutions']=True\n if same_support:ambiguity.append('OBSERVATION_CONSENSUS_AMBIGUITY')\n elif tied:ambiguity.append('NUMERICAL_CONSENSUS_TIE')").body[0]
    require(len(decisions)==1 and ast.dump(decisions[0],include_attributes=False)==ast.dump(expected_decision,include_attributes=False),'exact same-support veto independent of cost')
    p4=REPO/'scripts/research/pallet_cornerwise_independent_20261010_v4/pose.py';p4tree=ast.parse(p4.read_text())
    policy=next(n.value for n in p4tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='POLICY' for t in n.targets))
    tie=next(ast.literal_eval(k.value) for k in policy.keywords if k.arg=='numerical_score_tie_px2')
    require(tie==1e-8,'unchanged inherited numerical tie constant')
    return dict(numerical_score_tie_px2=tie,solver=b['solver'],finite=b['finite'],point_equivalence_source=binding(p4),
        decision_AST_sha256=hashlib.sha256(ast.dump(expected_decision,include_attributes=False).encode()).hexdigest(),
        existing_policy_only=True,new_equivalence_calls=0)
def authority(f,b):
    own,seal,receipt,pop=[read(f[k]) for k in ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','PARENT_POPULATION_CHECKS.json')]
    require(own['schema']=='fixed_hybrid_factor_subset_consensus_protocol_v16' and own['methods']==list(METHODS) and own['frames']==245 and own['rows']==490,'original paired protocol')
    require(seal['schema']=='hybrid_factor_subset_geometry_seal_v16' and seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['methods']==list(METHODS) and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,'completed GT-free geometry')
    for key,alias in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):require(same(seal[key],b[alias]),'sealed '+key)
    require(receipt['schema']=='hybrid_factor_subset_consensus_execution_v16' and receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['cleanup_error'] is None and receipt['error'] is None and same(receipt['protocol'],b['PROTOCOL.json']),'original actual execution/cleanup')
    x=receipt['actual_execution'];require(x['frames_complete']==x['ledger_rows']==245 and x['geometry_rows']==x['filtered_solver_entries']==x['filtered_solver_returns']==490 and x['filtered_solver_exceptions']==0,'all245/490 original paths')
    require(pop['complete'] is True and pop['passed'] is True and pop['frames']==245 and len(pop['ordered_ids'])==len(set(pop['ordered_ids']))==245,'ordered full source population')
    original,start,failure=[read(f[k]) for k in ('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','FIRST_SCALAR_ENVIRONMENT_FAILURE.json')]
    require(original['inputs']==start['inputs'] and same(start['own_protocol'],b['VALIDATION_PROTOCOL.json']) and same(original['checker'],b['original_checker']),'preserved original scalar freeze')
    require(failure['schema']=='preserved_hybrid_scalar_environment_failure_v16' and failure['error_type']=='NameError' and failure['error_message']=="name 'ast' is not defined" and failure['own_checks_written'] is False and failure['independent_checks_completed']==0,'preserved original zero-arithmetic failure')
    ready,rs,rc=[read(f[k]) for k in ('VALIDATION_READY_PROTOCOL.json','VALIDATION_READY_STARTED.json','VALIDATION_READY_CHECKS.json')]
    require(rc['schema']=='hybrid_factor_subset_scalar_checks_v16' and rc['complete'] is True and rc['passed'] is False and rc['checks']==198134007 and rc['failure_count']==564 and rc['exception'] is None and rc['new_production_calls']==0,'completed original 564-failure audit remains FAIL')
    require(ready['inputs']==rs['inputs']==rc['inputs'] and same(rc['own_protocol'],b['VALIDATION_READY_PROTOCOL.json']) and same(rs['own_protocol'],b['VALIDATION_READY_PROTOCOL.json']),'actual failed ready authority')
    extras=dict(ready_checker='ready_checker',authority_code='authority_code',original_validation_protocol='VALIDATION_PROTOCOL.json',original_validation_started='VALIDATION_STARTED.json',original_failure='FIRST_SCALAR_ENVIRONMENT_FAILURE.json')
    require(set(ready['inputs'])==set(original['inputs'])|set(extras) and all(ready['inputs'][k]==v for k,v in original['inputs'].items()) and all(same(ready['inputs'][k],b[v]) for k,v in extras.items()),'same original inputs plus environment-only closure')
    marker=ready['environment_ready'];require(marker==rs['environment_ready']==rc['environment_ready'] and marker['AST_environment_keyword_insertions']==1 and marker['scalar_formula_or_threshold_changes']==0 and marker['original_checker_changed'] is False,'environment-only correction recorded')
    for v in (ready,rs,rc):require(same(v['ready_checker'],b['ready_checker']) and same(v['authority_code'],b['authority_code']),'current ready source identities')
    for k,alias in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):require(same(original['inputs'][k],b[alias]),'same completed failed-audit '+k)
    return pop,source_witness(b)
def number(v,s):require(type(v) in (int,float) and math.isfinite(v),s);return v
def candidate(v,H):
    require(v['accepted'] is True and v['rejected'] is None and not (set(v['fit_input_ids'])|set(v['final_inliers']))&set(H),'stored accepted fit/consensus excludes H')
    g=v['point_line_geometry'];guard=v['stable_incidence_guard']
    require(guard['accepted'] is True and all(g[k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint')) and g['observed_weak'] is False and g['modeled_weak'] is False,'stored finite/normal acceptance flags')
    count=v['score_count'];require(type(count) is int and count==len(v['final_inliers'])+len(v['final_line_inliers']),'one point/edge vote count')
    cost=number(v['truncated_sse_px2'],'stored finite whole-pool SSE');require(cost>=0,'nonnegative score')
    return dict(candidate_index=v['candidate_index'],proposal_index=v['proposal_index'],proposal_kind=v['proposal_kind'],dimension_index=v['dimension_index'],
        fit_point_ids=v['fit_input_ids'],fit_line_edges=v['fit_line_edges'],final_inlier_ids=v['final_inliers'],final_line_inliers=v['final_line_inliers'],
        score_count=count,truncated_sse_px2=cost,accepted=v['accepted'],rejected=v['rejected'],
        stored_finite_ranks={k:g[k]['numerical_rank'] for k in ('observed_normal_joint','modeled_normal_joint')},
        stored_finite_weak={k:g[k] for k in ('observed_weak','modeled_weak')},stored_normal_guard_state=guard['guard_state'],stored_normal_guard_accepted=guard['accepted'])
def classify(row,tie):
    s=row['solver'];best=s.get('best_candidate');H=row['hidden_initial'];alternatives=[];reasons=set();extra=ties=0
    require(row['method'] in METHODS and row['reprojections_reused_as_observations'] is False,'known unchanged method/no projection refit')
    if best is None:
        require(not s['alternatives'] and s.get('viable_candidate_count',0)==0 and s['available'] is False,'no accepted best branch')
        classification='NO_ACCEPTED_BEST';saved_best=None
    else:
        saved_best=candidate(best,H);require(any(v==best for v in s['all_candidate_solutions']),'best is preserved full accepted candidate')
        for v in s['alternatives']:
            compact=candidate(v,H);same_count=v['score_count']==best['score_count'];delta=v['truncated_sse_px2']-best['truncated_sse_px2']
            support=v['final_inliers']==best['final_inliers'] and v['final_line_inliers']==best['final_line_inliers'];tied=same_count and abs(delta)<=tie
            eq=v['numerically_equivalent_to_selected'];require(type(eq) is bool,'reported equivalence boolean')
            require(v['same_actual8px_consensus_as_selected'] is support and v['numerical_objective_tie'] is tied and v['objective_difference_from_selected_px2']==delta,'stored scalar support/count/cost decision identity')
            veto=None
            if not eq:
                if support:veto='OBSERVATION_CONSENSUS_AMBIGUITY'
                elif tied:veto='NUMERICAL_CONSENSUS_TIE'
                if veto:reasons.add(veto)
                extra+=int(support and not tied);ties+=int(tied)
            compact.update(stored_non_equivalent_to_selected=not eq,same_final_support=support,same_score_count=same_count,
                numerical_objective_tie=tied,cost_difference_from_best_px2=delta,reported_veto_reason=veto,
                different_cost_same_support_veto=not eq and support and not tied)
            alternatives.append(compact)
        require(s['viable_candidate_count']==len(s['alternatives'])+1==sum(v['accepted'] is True for v in s['all_candidate_solutions']) and s['numerical_tied_solution_count']==sum(v['numerical_objective_tie'] for v in s['alternatives']) and s['numerical_duplicate_solution_count']==sum(v['numerically_equivalent_to_selected'] for v in s['alternatives']),'stored accepted/alternative/tie counts')
        require(s['multiple_solutions'] is any(not v['numerically_equivalent_to_selected'] for v in s['alternatives']),'reported physical multiplicity flag')
        ambiguous=bool(reasons);require((s['state']=='AMBIGUOUS_FINITE_CONSENSUS') is ambiguous and s['unresolved_ambiguity'] is ambiguous and s['available'] is (not ambiguous),'stored ambiguity/output routing')
        if ambiguous:require(s['reason']==';'.join(sorted(reasons)),'exact reported ambiguity reasons')
        classification='ONLY_DIFFERENT_COST_SAME_SUPPORT_VETO' if ambiguous and extra and not ties else 'NUMERICAL_TIE_AND_DIFFERENT_COST_SAME_SUPPORT' if ties and extra else 'NUMERICAL_TIE_PRESENT' if ties else 'NO_REPORTED_VETO_WITH_ACCEPTED_BEST'
    return dict(id=row['id'],session=row['session'],method=row['method'],output_status=row['output_status'],solver_state=s['state'],solver_reason=s['reason'],
        new_pose_estimated=row['new_pose_estimated'],fallback_used=row['fallback_used'],H=H,classification=classification,
        stored_best=saved_best,stored_accepted_candidate_count=sum(v['accepted'] is True for v in s['all_candidate_solutions']),
        candidate_count=s['candidate_count'],viable_candidate_count=s.get('viable_candidate_count',0),failure_category_counts=s.get('failure_category_counts'),
        actual_alternative_relations=alternatives,different_cost_same_support_veto_alternatives=extra,non_equivalent_numerical_tie_alternatives=ties,
        actual_reported_ambiguity_reasons=sorted(reasons),original_solver_semantic_sha256=semantic(s),
        equivalence_and_rank_independently_recomputed=False,pose_accuracy_or_policy_counterfactual_checked=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('freeze','run'));p.add_argument('--input',type=Path,default=DOC);p.add_argument('--output',type=Path,default=DOC);args=p.parse_args()
    require(args.output.is_dir(),'existing explicit output directory required')
    for name in FILES if args.stage=='freeze' else FILES[1:]:require(not(args.output/name).exists(),'preserve '+name)
    f=paths(args.input);b={k:binding(v) for k,v in f.items()};pop,witness=authority(f,b)
    if args.stage=='freeze':write(args.output/FILES[0],dict(schema='saved_ambiguity_diagnostic_protocol_v16',inputs=b,limits=LIMITS,source_witness=witness,frames=245,rows=490,methods=list(METHODS),frozen_before_own_saved_scalar_comparisons=True));return
    own=read(args.output/FILES[0]);require(own['inputs']==b and own['limits']==LIMITS and own['source_witness']==witness,'unchanged own freeze')
    write(args.output/FILES[1],dict(own_protocol=binding(args.output/FILES[0]),inputs=b,new_production_calls=0))
    counts={m:Counter() for m in METHODS};groups={m:{} for m in METHODS};error=None;cleanup=[];pending=args.output/('PENDING_'+FILES[2]);raw=stream=None;completed=0
    try:
        geometry=iter(rows(f['GEOMETRY_SEALED.jsonl.gz']));raw=pending.open('xb');stream=gzip.GzipFile(fileobj=raw,mode='wb',mtime=0)
        for fid in pop['ordered_ids']:
            for method in METHODS:
                row=next(geometry);require((row['id'],row['method'])==(fid,method),'full ordered245/490 identity')
                v=classify(row,witness['numerical_score_tie_px2']);del row;counts[method]['rows']+=1
                for key in (v['classification'],'STATE:'+v['solver_state'],*['REASON:'+r for r in v['actual_reported_ambiguity_reasons']]):groups[method].setdefault(key,[]).append(fid)
                counts[method]['different_cost_same_support_veto_alternatives']+=v['different_cost_same_support_veto_alternatives'];counts[method]['non_equivalent_numerical_tie_alternatives']+=v['non_equivalent_numerical_tie_alternatives']
                stream.write((json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n').encode());completed+=1
        require(next(geometry,None) is None and completed==490 and all(v['rows']==245 for v in counts.values()),'complete unchanged paired population')
        require({k:binding(v) for k,v in f.items()}==b,'all saved inputs unchanged during diagnostic')
    except BaseException as e:error=dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
    finally:
        if stream:
            try:stream.close()
            except BaseException as e:cleanup.append(dict(type=type(e).__name__,message=str(e),phase='gzip_close'))
        if raw:
            for phase,op in (('raw_flush',raw.flush),('raw_fsync',lambda:os.fsync(raw.fileno())),('raw_close',raw.close)):
                try:op()
                except BaseException as e:cleanup.append(dict(type=type(e).__name__,message=str(e),phase=phase))
        error=error or (cleanup[0] if cleanup else None)
        if error is None:
            try:os.link(pending,args.output/FILES[2]);pending.unlink()
            except BaseException as e:error=dict(type=type(e).__name__,message=str(e),phase='exclusive_publish')
        write(args.output/FILES[3],dict(schema='saved_ambiguity_diagnostic_checks_v16',complete=error is None,passed=error is None,own_protocol=binding(args.output/FILES[0]),inputs=b,
            limits=LIMITS,source_witness=witness,original_full_geometry_scalar_passed=False,original_full_geometry_scalar_failure_count=564,rows_complete=completed,
            counts={m:dict(v) for m,v in counts.items()},groups={m:{k:dict(count=len(v),ids=v) for k,v in g.items()} for m,g in groups.items()},
            error=error,cleanup_errors=cleanup,rows=binding(args.output/FILES[2]) if (args.output/FILES[2]).is_file() else None,
            preserved_prefix=binding(pending) if pending.is_file() else None,new_GT_scores_model_solver_optimizer_projection_J_SVD_calls=0))
    require(error is None,'saved ambiguity diagnostic failed; preserve first failure and prefix')
if __name__=='__main__':main()
