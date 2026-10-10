"""Postseal fixed five-method scoring; parent modules are never mutated.

The score body changes only one local method-name import in an isolated normal
package namespace. References, error formulas and statistical kernels remain
unchanged. This is known DEV geometric-proxy evaluation, not physical GT.
"""
import ast
from collections import Counter
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from . import runner as R
from . import pipeline as P
from ..pallet_finite_edge_consensus_repair_20261011_v13 import evaluation as E13
from ..pallet_finite_edge_consensus_repair_20261011_v13 import pipeline as P13
from ..pallet_stable_incidence_guard_20261011_v14 import pipeline as G14
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9.statistics import outcome
from ..pallet_three_head_observation_20261010_v7 import statistics as S

METHODS=('BASE','N3_SUBPIX',C.COMPARATOR,G14.METHOD,P.PRIMARY)
CONTRASTS=((P.PRIMARY,'N3_SUBPIX'),(P.PRIMARY,C.COMPARATOR),(P.PRIMARY,G14.METHOD))

def geometry_gate(args):
    core=R.frozen(args);seal,receipt,check,own,start=[C.read(R.DOC/n) for n in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    C.require(seal['schema']=='factor_subset_consensus_geometry_seal_v15' and seal['complete'] is True
        and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True,'complete GT-free245 subset seal required')
    C.require(receipt['schema']=='factor_subset_consensus_execution_v15' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'complete restored subset execution required')
    C.require(check['schema']=='factor_subset_consensus_scalar_checks_v15' and own['schema']=='factor_subset_consensus_scalar_protocol_v15'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['inputs']==own['inputs'] and check['new_production_calls']==0,'actual independent subset scalar PASS required')
    C.require(start['inputs']==own['inputs'] and start['new_production_calls']==0,'actual independent STARTED required')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'independent subset protocol')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',start['own_protocol'],'independent STARTED protocol')
    C.bound(R.CODE/'validation_checks.py',check['checker'],'current independent checker source')
    C.require(C.same_binding(own['checker'],check['checker']),'own checker binding differs')
    for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        C.bound(R.DOC/name,own['inputs'][alias],'actual independently checked '+alias)
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        C.bound(R.DOC/name,seal[key],'sealed '+key)
    return core,seal,receipt,check

def isolated_score():
    source=ast.parse(Path(E13.__file__).read_text());found=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='score']
    C.require(len(found)==1,'one unchanged V13 score function required')
    original=found[0];canonical=ast.dump(original,include_attributes=False)
    target=ast.parse('from .pipeline import METHODS').body[0]
    replacement=ast.parse('METHODS = _subset_METHODS').body[0];changes=[]
    class Replace(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(target,include_attributes=False):
                changes.append(1);return ast.copy_location(copy.deepcopy(replacement),node)
            return self.generic_visit(node)
    changed=Replace().visit(copy.deepcopy(original));C.require(len(changes)==1,'exact one local method dispatch change')
    reversed_changes=[]
    class Undo(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(replacement,include_attributes=False):
                reversed_changes.append(1);return ast.copy_location(copy.deepcopy(target),node)
            return self.generic_visit(node)
    C.require(ast.dump(Undo().visit(copy.deepcopy(changed)),include_attributes=False)==canonical and len(reversed_changes)==1,
        'full canonical score-function restoration proof')
    facade=SimpleNamespace(**vars(R));facade.A=SimpleNamespace(protect=R.protect);facade.audit_args=lambda args:args
    namespace=dict(vars(E13));namespace.update(R=facade,geometry_gate=geometry_gate,_subset_METHODS=P.METHODS)
    tree=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
    exec(compile(tree,str(E13.__file__)+' [isolated subset dispatch]','exec'),namespace)
    witness=dict(original_score_source=C.binding(Path(E13.__file__)),local_method_import_AST_changes=1,
        original_score_function_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        isolated_score_function_AST_sha256=hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest(),
        full_canonical_score_function_restore_proven=True,score_reference_formula_AST_changes=0,
        original_module_globals_modified=False,immutable_parent_method_tuple_modified=False,
        original_module_package_context_preserved=True)
    return namespace['score'],witness

def evaluation_inputs(args):
    values={'geometry_input:'+key:value for key,value in R.inputs(args).items()}
    paths=dict(adapter=Path(__file__),unchanged_score_dispatch=Path(E13.__file__),
        unchanged_scalar_statistics=Path(S.__file__),cohort=Path(args.cohort),protocol=Path(args.protocol),
        geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry=R.DOC/'GEOMETRY_SEALED.jsonl.gz',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',
        validation_started=R.DOC/'VALIDATION_STARTED.json',checker=R.CODE/'validation_checks.py',
        control_receipt=R.DOC/'CONTROL_RECEIPT.json',ledgers=R.DOC/'CONTROL_LEDGERS.jsonl.gz',
        parent_population_checks=R.DOC/'PARENT_POPULATION_CHECKS.json',same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
        stable_scores=R.G.DOC/'PREDICTIONS.jsonl.gz',stable_scoring=R.G.DOC/'SCORING_RECEIPT.json',
        stable_evaluation=R.G.DOC/'EVALUATION_PROTOCOL.json',stable_dispatch=R.G.DOC/'SCORING_AUTHORITY_SUPPLEMENT.json',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',point_scores=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',
        bootstrap=S.BOOTSTRAP)
    values.update({k:C.binding(p) for k,p in paths.items()});return values

def freeze(args):
    geometry_gate(args);bindings=evaluation_inputs(args);_,dispatch=isolated_score()
    C.write_new(R.output(args,'EVALUATION_PROTOCOL.json'),dict(schema='postseal_factor_subset_evaluation_protocol_v15',
        inputs=bindings,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],frames=245,scored_rows=1225,
        new_scored_rows=245,moment_distributions=135,moment_scalars=810,paired_groups=27,metric_CIs=81,
        score_dispatch=dispatch,reference='unchanged N3-phase GEOMETRIC_PROXY known DEV; not independent physical pose GT',
        original_frozen_sources_and_failures_preserved=True,unchanged_proxy_score_kernel=True,
        unchanged_summary_paired_bootstrap_kernels=True,new_seed_draws=0,new_models_fits_RGB=0,
        inherited_score_receipt_schema_is_v13=True,frozen_before_own_GT_access_or_statistic_arithmetic=True))

def frozen(args):
    geometry_gate(args);own=C.read(R.DOC/'EVALUATION_PROTOCOL.json');_,dispatch=isolated_score()
    C.require(own['schema']=='postseal_factor_subset_evaluation_protocol_v15' and own['inputs']==evaluation_inputs(args)
        and own['methods']==list(METHODS) and own['contrasts']==[list(v) for v in CONTRASTS] and own['score_dispatch']==dispatch,
        'own postseal evaluation source/dispatch differs');return own

def score(args):
    own=frozen(args);fn,dispatch=isolated_score();before=(P13.METHODS,E13.R,E13.geometry_gate)
    C.write_new(R.output(args,'SCORING_AUTHORITY.json'),dict(schema='isolated_subset_proxy_score_authority_v15',
        own_evaluation_protocol=C.binding(R.DOC/'EVALUATION_PROTOCOL.json'),inputs=own['inputs'],methods=list(P.METHODS),
        score_dispatch=dispatch,reference_or_error_formula_changed=False,original_module_globals_modified=False,
        new_independent_geometry_check_PASS=True))
    try:fn(args)
    finally:C.require(before==(P13.METHODS,E13.R,E13.geometry_gate),'immutable parent globals changed during isolated score dispatch')
    frozen(args)

def score_gate(args):
    own=frozen(args);receipt=C.read(R.DOC/'SCORING_RECEIPT.json');authority=C.read(R.DOC/'SCORING_AUTHORITY.json')
    C.require(receipt['schema']=='repair_postseal_proxy_scoring_v13' and receipt['complete'] is True
        and receipt['scored_rows']==245 and receipt['methods']==list(P.METHODS),'complete actual subset245 scores required')
    C.require(authority['schema']=='isolated_subset_proxy_score_authority_v15' and authority['inputs']==own['inputs']
        and authority['score_dispatch']==own['score_dispatch'] and authority['methods']==list(P.METHODS)
        and authority['original_module_globals_modified'] is False and authority['reference_or_error_formula_changed'] is False,
        'actual isolated scoring authority differs')
    C.bound(R.DOC/'EVALUATION_PROTOCOL.json',authority['own_evaluation_protocol'],'actual scoring protocol')
    C.bound(R.DOC/'PREDICTIONS.jsonl.gz',receipt['predictions'],'actual scored rows')
    C.bound(R.DOC/'PROTOCOL.json',receipt['protocol'],'actual score geometry protocol')
    C.bound(R.DOC/'GEOMETRY_SEAL.json',receipt['geometry_seal'],'actual score geometry seal')
    start=C.read(R.DOC/'SCORING_STARTED.json')
    C.require(C.same_binding(start['protocol'],receipt['protocol']) and start['configured_scored_rows']==245
        and start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,'actual score STARTED differs')
    return own,receipt

def compact(value):
    # Only unused copies in RAM are removed; full original gzip rows stay bound.
    unused={'all_candidate_solutions','alternatives','proposal_solutions','optimizer_attempts',
        'direct_point_generation','mixed_subset_enumeration'}
    if isinstance(value,dict):return {key:compact(v) for key,v in value.items() if key not in unused}
    if isinstance(value,list):return [compact(v) for v in value]
    return value

def posthoc(row,n3,point,stable,observation):
    result=S.posthoc(dict(row,method=C.PARENT_PRIMARY),n3,observation)
    solver=row['solver'];new=bool(row['new_pose_estimated']);best=solver.get('best_candidate');geometry=solver.get('point_line_geometry') if new else None
    guard=best.get('stable_incidence_guard') if best else None
    fitted_points=list(solver['fit_input_ids']) if new else [];fitted_edges=list(solver['fit_line_edges']) if new else []
    if new:
        C.require(best['accepted'] is True and guard['accepted'] is True and geometry['point_ids']==fitted_points
            and geometry['line_edges']==fitted_edges and (bool(fitted_edges) or len(fitted_points)>=4),
            'accepted actual fit/normal-incidence/four-point standard contract differs')
    result.update(method=P.PRIMARY,posthoc_kernel_label_adapter_only=True,
        factor_subset_diagnostics=dict(solver_state=solver['state'],supplied_point_ids=solver['used'],supplied_unused_edges=solver['line_edges'],
            accepted_fit_point_ids=fitted_points,accepted_fit_line_edges=fitted_edges,
            accepted_final_point_inliers=solver['final_inliers'] if new else [],accepted_final_line_inliers=solver['final_line_inliers'] if new else [],
            direct_point_roots=solver['operation_counts']['direct_point_roots'],mixed_subset_requests=solver['operation_counts']['mixed_subset_requests'],
            point_bank_reused=solver['direct_point_coordinate_bank_reused'],operation_counts=solver['operation_counts'],
            direct_bank_operation_counts=solver['hypothesis_bank_operation_counts'],call_bounds=solver.get('call_bounds'),
            proposal_count=solver.get('proposal_candidate_count',0),refit_count=solver.get('refit_count',0),
            viable_candidate_count=solver.get('viable_candidate_count',0),failure_categories=solver.get('failure_category_counts'),
            selected_proposal_kind=best['proposal_kind'] if best else None,
            selected_proposal_is_operational_NEW=new,selected_proposal_guard_state=guard['guard_state'] if guard else None,
            actual_fit_normal_incidence_rank6=bool(new and guard['accepted']),
            accepted_fit_has_fewer_than4_points=bool(new and len(fitted_points)<4),accepted_fit_has_lines=bool(new and fitted_edges),
            point_only_refit=bool(new and not fitted_edges),numeric_rescue_over_point=bool(new and not point['new_pose_estimated']),
            numeric_rescue_over_stable=bool(new and not stable['new_pose_estimated']),
            original_N3_supplied_at_least_one_mixed_LOCAL_start=solver['initial_pose_start_used'],
            initial_pose_residual_prior=False,point_result_available_was_not_root_filter=True,
            actual_pose_vs_N3=outcome(row,n3),actual_pose_vs_point=outcome(row,point),actual_pose_vs_stable=outcome(row,stable),
            actual_inlier_geometry_is_acceptance_gate=True,finite_hinges_are_not_observability_equalities=True,
            point_line_double_count=False,physical_ownership_certified=False,global_uniqueness_proven=False,
            mixed_LOCAL_root_completeness_proven=False,numerical_NEW_is_not_goal_success=True))
    return result

def stats(args):
    import numpy as np
    own,receipt=score_gate(args);parent=C.read(R.G.DOC/'SCORING_RECEIPT.json')
    C.require(parent['complete'] is True and parent['scored_rows']==245 and parent['methods']==[G14.METHOD],
        'complete unchanged stable comparator scores required')
    C.bound(R.G.DOC/'PREDICTIONS.jsonl.gz',parent['predictions'],'unchanged stable comparator')
    paths=dict(new_scores=R.DOC/'PREDICTIONS.jsonl.gz',stable_scores=R.G.DOC/'PREDICTIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',point_scores=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',
        cohort=Path(args.cohort),bootstrap=S.BOOTSTRAP,statistics_kernel=Path(S.__file__),own_code=Path(__file__),
        scoring_receipt=R.DOC/'SCORING_RECEIPT.json',scoring_started=R.DOC/'SCORING_STARTED.json',
        protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry_validation=R.DOC/'VALIDATION_CHECKS.json',
        validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',scoring_authority=R.DOC/'SCORING_AUTHORITY.json',
        same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz')
    bindings={key:C.binding(path) for key,path in paths.items()}
    for name in ('STATISTICS_STARTED.json','METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz'):R.output(args,name)
    C.write_new(R.output(args,'STATISTICS_STARTED.json'),dict(inputs=bindings,new_seed_draws=0,configured_new_rows=245,
        configured_saved_score_rows=1225,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],known_DEV_only=True))
    groups={method:{} for method in METHODS}
    for key in ('fixed_scores','point_scores','stable_scores','new_scores'):
        for full in C.rows(paths[key]):
            row=compact(full);del full
            C.require(row['method'] in groups and row['id'] not in groups[row['method']],'unique saved method/ID required')
            groups[row['method']][row['id']]=row
    cohort=C.read(args.cohort);ids=cohort['ids'];labels={frame['id']:frame['label'] for frame in cohort['frames']}
    C.require(len(ids)==245 and all(set(group)==set(ids) for group in groups.values()),'five methods retain full245')
    with gzip.open(S.BOOTSTRAP,'rt') as stream:bootstrap=json.load(stream)
    draws=np.asarray(bootstrap['counts'],np.int64);sessions=bootstrap['sessions']
    C.require(draws.shape==(10000,13) and len(set(sessions))==13 and (draws>=0).all() and (draws.sum(1)==13).all(),
        'reuse original13session10000draws')
    index={session:i for i,session in enumerate(sessions)}
    observations={row['id']:row for row in C.rows(paths['same_role_observations']) if row['head_arm']=='IMAGE_ROLE'}
    C.require(set(observations)==set(ids),'full sameROLE observations required')
    post=[posthoc(groups[P.PRIMARY][fid],groups['N3_SUBPIX'][fid],groups[C.COMPARATOR][fid],groups[G14.METHOD][fid],observations[fid]) for fid in ids]
    strata={}
    for name,selected in (('combined',ids),('easy',[fid for fid in ids if labels[fid]=='clean']),('medium',[fid for fid in ids if labels[fid]=='moderate'])):
        strata[name]=dict(frames=len(selected),ids=selected,
            methods={method:S.summaries([groups[method][fid] for fid in selected]) for method in METHODS},
            contrasts={a+'_minus_'+b:{scope:S.paired(groups[a],groups[b],selected,index,draws,scope) for scope in S.SCOPES} for a,b in CONTRASTS},
            diagnostics={P.PRIMARY:S.diagnostic_summary([row for row in post if row['id'] in set(selected)])},
            subset_solver_states=dict(Counter(row['factor_subset_diagnostics']['solver_state'] for row in post if row['id'] in set(selected))))
    primary=strata['combined']['contrasts'][P.PRIMARY+'_minus_N3_SUBPIX']['common_operational']
    improved=primary['common_frames']==245 and all(primary['metrics'][metric]['mean_delta']<0 for metric in ('translation_cm','rotation_deg'))
    C.require({key:C.binding(path) for key,path in paths.items()}==bindings,'fixed statistics bytes changed')
    S.C.save_rows(R.output(args,'POSTHOC_ROWS.jsonl.gz'),post)
    result=dict(schema='factor_subset_raw_row_metrics_v15',complete=True,methods=list(METHODS),strata=strata,
        contrasts=[list(v) for v in CONTRASTS],scopes=list(S.SCOPES),inputs=bindings,posthoc_rows=C.binding(R.DOC/'POSTHOC_ROWS.jsonl.gz'),
        actual_arithmetic=dict(scored_rows=1225,new_rows=245,posthoc_rows=245,methods=5,strata=3,moment_distributions=135,
            moment_scalars=810,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0),
        primary_both_T_R_means_better_than_N3_on_full245=improved,reference=own['reference'],
        new_model_image_GT_score_PnP_optimizer_training_RGB_calls=0,statistics_does_not_change_proposals_or_operating_status=True)
    C.write_new(R.output(args,'METRICS.json'),result)
    with R.output(args,'METRICS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('stratum','method','population','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'));writer.writeheader()
        for stratum,value in strata.items():
            for method,summary in value['methods'].items():
                for population,metrics in summary['metrics'].items():
                    for metric,moment in metrics.items():writer.writerow(dict(stratum=stratum,method=method,population=population,metric=metric,**moment))
    print('FACTOR_SUBSET_METRICS',improved,flush=True)

def main():
    p=R.parser();next(action for action in p._actions if action.dest=='stage').choices=('freeze','score','stats');args=p.parse_args()
    {'freeze':freeze,'score':score,'stats':stats}[args.stage](args)

if __name__=='__main__':main()
