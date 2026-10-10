"""Postseal paired point-consensus decision comparison; formulas are unchanged.

Only two saved point-only decision views are newly scored. V17 ordinary and
Base/N3 scores stay byte-identical. The complete V16 scalar FAIL is preserved;
a separate point-only decision/output PASS does not certify its failed lines.
Known DEV cuboid GEOMETRIC_PROXY is not independent physical pose truth.
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
from ..pallet_factor_subset_consensus_20261011_v15 import evaluation as E15
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_three_head_observation_20261010_v7 import common as C7
from ..pallet_three_head_observation_20261010_v7 import statistics as S

ORDINARY_METHODS=('HYBRID_STANDARD_PNP','NATIVE_STANDARD_PNP')
METHODS=('BASE','N3_SUBPIX',*ORDINARY_METHODS,*P.METHODS)
CONTRASTS=((P.PRIMARY,P.METHODS[0]),(P.PRIMARY,'N3_SUBPIX'),
    (P.PRIMARY,'NATIVE_STANDARD_PNP'),(P.PRIMARY,'HYBRID_STANDARD_PNP'),
    (P.METHODS[0],'N3_SUBPIX'))
ARITHMETIC=dict(scored_rows=1470,new_rows=490,posthoc_rows=490,methods=6,strata=3,
    moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,new_bootstrap_draws=0)
ORDINARY_DOC=C.REPO/'_docs/experiments/pallet_hybrid_standard_controls_20261011_v17'


def ordinary_paths(args):
    folder=Path(args.ordinary_parent)
    names=('PROTOCOL.json','GEOMETRY_SEAL.json','GEOMETRY_SEALED.jsonl.gz',
        'CONTROL_RECEIPT.json','CONTROL_LEDGERS.jsonl.gz','PARENT_POPULATION_CHECKS.json',
        'VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json',
        'EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json','SCORING_STARTED.json',
        'SCORING_RECEIPT.json','PREDICTIONS.jsonl.gz','STATISTICS_STARTED.json','METRICS.json',
        'STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')
    return {name:folder/name for name in names}


def ordinary_authority(args):
    """Bind completed V17 without decoding or rescoring its saved raw rows."""
    paths=ordinary_paths(args);data={name:C.read(paths[name]) for name in
        ('PROTOCOL.json','GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_PROTOCOL.json',
        'VALIDATION_STARTED.json','VALIDATION_CHECKS.json','EVALUATION_PROTOCOL.json',
        'SCORING_AUTHORITY.json','SCORING_STARTED.json','SCORING_RECEIPT.json',
        'STATISTICS_STARTED.json','METRICS.json','STATISTICS_VERIFICATION_PROTOCOL.json',
        'STATISTICS_VERIFICATION_STARTED.json','VERIFICATION.json')}
    core,seal,receipt,check,own,start=[data[n] for n in ('PROTOCOL.json','GEOMETRY_SEAL.json',
        'CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    C.require(core['schema']=='fixed_native_hybrid_standard_protocol_v17' and core['methods']==list(ORDINARY_METHODS)
        and seal['schema']=='native_hybrid_standard_geometry_seal_v17' and seal['complete'] is True
        and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'actual complete V17 ordinary490 and cleanup required')
    C.require(check['schema']=='native_hybrid_standard_scalar_checks_v17' and own['schema']=='native_hybrid_standard_scalar_protocol_v17'
        and check['complete'] is True and check['passed'] is True and check['checks']==394552
        and check['failure_count']==0 and check['exception'] is None and check['new_production_calls']==0
        and check['inputs']==own['inputs']==start['inputs'] and start['new_production_calls']==0,
        'actual independent V17 geometry394552 PASS required')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),
        ('ledgers','CONTROL_LEDGERS.jsonl.gz'),('control_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):C.bound(paths[name],seal[key],'V17 sealed '+key)
    for key,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),
        ('repair_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        C.bound(paths[name],own['inputs'][key],'V17 checked '+key)
    for item in (check['own_protocol'],start['own_protocol']):C.bound(paths['VALIDATION_PROTOCOL.json'],item,'V17 check protocol')
    C.bound(C.REPO/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/validation_checks.py',check['checker'],'V17 checker')
    C.require(C.same_binding(check['checker'],own['checker']),'V17 own checker differs')
    score,evaluation,authority,score_start=[data[n] for n in ('SCORING_RECEIPT.json','EVALUATION_PROTOCOL.json','SCORING_AUTHORITY.json','SCORING_STARTED.json')]
    C.require(score['schema']=='repair_postseal_proxy_scoring_v13' and score['complete'] is True
        and score['frames']==245 and score['scored_rows']==490 and score['methods']==list(ORDINARY_METHODS)
        and score['GT_access_only_after_complete_geometry_and_cleanup'] is True,
        'actual immutable V17 ordinary490 score receipt required')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEAL.json','geometry_seal'),('PREDICTIONS.jsonl.gz','predictions')):
        C.bound(paths[name],score[key],'V17 score '+key)
    C.require(evaluation['schema']=='postseal_native_hybrid_standard_evaluation_protocol_v17'
        and evaluation['new_methods']==list(ORDINARY_METHODS) and evaluation['scored_rows']==980
        and authority['schema']=='isolated_native_hybrid_standard_proxy_score_authority_v17'
        and authority['inputs']==evaluation['inputs'] and authority['score_dispatch']==evaluation['score_dispatch']
        and score_start['configured_scored_rows']==490 and score_start['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True,
        'original V17 isolated scorer authority required')
    C.bound(paths['EVALUATION_PROTOCOL.json'],authority['own_evaluation_protocol'],'V17 evaluation authority')
    C.bound(paths['VALIDATION_PROTOCOL.json'],score_start['independent_validation'],'V17 actual score gate')
    stat,stat_start,verified,vprotocol,vstart=[data[n] for n in ('METRICS.json','STATISTICS_STARTED.json',
        'VERIFICATION.json','STATISTICS_VERIFICATION_PROTOCOL.json','STATISTICS_VERIFICATION_STARTED.json')]
    C.require(stat['schema']=='native_hybrid_standard_raw_row_metrics_v17' and stat['complete'] is True
        and stat['methods']==['BASE','N3_SUBPIX',*ORDINARY_METHODS] and stat['inputs']==stat_start['inputs']
        and stat['actual_arithmetic']['scored_rows']==980 and stat['actual_arithmetic']['moment_scalars']==648,
        'completed immutable V17 statistics980/648 required')
    C.require(verified['schema']=='independent_native_hybrid_standard_statistics_checks_v17' and verified['complete'] is True
        and verified['passed'] is True and verified['failure_count']==0 and verified['error'] is None
        and verified['actual_counts']['saved_scored_rows_streamed']==980
        and verified['actual_counts']['moment_scalars_checked']==648 and verified['actual_counts']['metric_CI_slots_checked']==81
        and verified['inputs']==vprotocol['inputs']==vstart['inputs'],'actual V17 independent statistics980/648/81 PASS required')
    for value in (verified['protocol'],vstart['protocol']):
        C.bound(paths['STATISTICS_VERIFICATION_PROTOCOL.json'],value,'V17 statistics checker protocol')
    C.bound(C.REPO/'scripts/research/pallet_hybrid_standard_controls_20261011_v17/verify.py',verified['checker'],'V17 unchanged statistics verifier')
    C.bound(paths['METRICS.json'],verified['inputs']['METRICS'],'V17 independently checked statistics bytes')
    C.bound(paths['PREDICTIONS.jsonl.gz'],verified['inputs']['new_scores'],'V17 independently checked ordinary score bytes')
    for key,name in (('new_scores','PREDICTIONS.jsonl.gz'),('cohort',None)):
        C.bound(Path(args.cohort) if name is None else paths[name],stat['inputs'][key],'same V17 score/cohort '+key)
    C.bound(Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',stat['inputs']['fixed_scores'],'same immutable Base/N3 bytes')
    C.bound(Path(args.parent)/'OBSERVATIONS.jsonl.gz',stat['inputs']['same_role_observations'],'same ROLE observations')
    return dict(complete=True,passed=True,geometry_checks=394552,score_rows=980,moment_scalars=648,
        source_streams_not_rescored=True,bindings={k:C.binding(p) for k,p in paths.items()})


def geometry_gate(args):
    core=R.frozen(args);seal,receipt,check,own,start=[C.read(R.DOC/n) for n in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    C.require(seal['schema']=='point_consensus_decision_restoration_geometry_seal_v18' and seal['complete'] is True
        and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True,'complete245/490 GT-free paired decision seal required')
    C.require(receipt['schema']=='point_consensus_decision_restoration_execution_v18' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'complete paired decision execution required')
    C.require(check['schema']=='point_consensus_decision_scalar_checks_v18' and own['schema']=='point_consensus_decision_scalar_protocol_v18'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0 and check['exception'] is None
        and check['inputs']==own['inputs'] and check['new_production_calls']==0,'actual paired point-only decision/output PASS required')
    C.require(start['inputs']==own['inputs'] and start['new_production_calls']==0,'actual checker STARTED required')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'checker protocol')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',start['own_protocol'],'STARTED protocol')
    C.bound(R.CODE/'validation_checks.py',check['checker'],'current point-only checker source')
    C.require(C.same_binding(own['checker'],check['checker']),'own checker differs')
    for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):C.bound(R.DOC/name,own['inputs'][alias],'checked '+alias)
    C.bound(R.CODE/'validation_checks.py',own['inputs']['checker'],'checked checker alias')
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):C.bound(R.DOC/name,seal[key],'sealed '+key)
    scope=check['scope']
    C.require(scope['original_full_scalar_validation_passed'] is False and scope['point_only_original_failed_predicates']==0
        and scope['independent_parent_point_check_count_not_claimed'] is True,'point-only evidence must not convert full V16 FAIL to PASS')
    parent=C.read(Path(args.decision_parent)/'VALIDATION_READY_CHECKS.json')
    C.require(parent['complete'] is True and parent['passed'] is False and parent['checks']==198134007
        and parent['failure_count']==564 and parent['exception'] is None,'actual complete V16 scalar FAIL564 must remain preserved')
    ordinary_authority(args)
    return core,seal,receipt,check

def isolated_score():
    nodes=[n for n in ast.parse(Path(E13.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='score']
    C.require(len(nodes)==1,'one immutable full score function');original=nodes[0];canonical=ast.dump(original,include_attributes=False)
    target=ast.parse('from .pipeline import METHODS').body[0];replacement=ast.parse('METHODS = _hybrid_METHODS').body[0]
    changes=Counter()
    class Replace(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(target,include_attributes=False):changes['local_method_import']+=1;return copy.deepcopy(replacement)
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg in ('configured_scored_rows','scored_rows') and isinstance(node.value,ast.Constant) and node.value.value==245:
                changes[node.arg]+=1;return ast.keyword(arg=node.arg,value=ast.Constant(490))
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('count==245',mode='eval').body,include_attributes=False):
                changes['completion_count']+=1;return ast.parse('count==490',mode='eval').body
            return self.generic_visit(node)
    changed=Replace().visit(copy.deepcopy(original));expected=dict(local_method_import=1,configured_scored_rows=1,scored_rows=1,completion_count=1)
    C.require(dict(changes)==expected,'exact four method/population topology sites')
    undo=Counter()
    class Undo(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(replacement,include_attributes=False):undo['local_method_import']+=1;return copy.deepcopy(target)
            return self.generic_visit(node)
        def visit_keyword(self,node):
            if node.arg in ('configured_scored_rows','scored_rows') and isinstance(node.value,ast.Constant) and node.value.value==490:
                undo[node.arg]+=1;return ast.keyword(arg=node.arg,value=ast.Constant(245))
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('count==490',mode='eval').body,include_attributes=False):
                undo['completion_count']+=1;return ast.parse('count==245',mode='eval').body
            return self.generic_visit(node)
    C.require(ast.dump(Undo().visit(copy.deepcopy(changed)),include_attributes=False)==canonical and dict(undo)==expected,
        'whole canonical score function restoration')
    facade=SimpleNamespace(**vars(R));facade.A=SimpleNamespace(protect=R.protect);facade.audit_args=lambda args:args
    env=dict(vars(E13));env.update(R=facade,geometry_gate=geometry_gate,_hybrid_METHODS=P.METHODS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),str(E13.__file__)+' [paired topology only]','exec'),env)
    return env['score'],dict(original_score_source=C.binding(Path(E13.__file__)),topology_AST_changes=expected,
        original_score_function_AST_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
        isolated_score_function_AST_sha256=hashlib.sha256(ast.dump(changed,include_attributes=False).encode()).hexdigest(),
        full_canonical_score_function_restore_proven=True,score_reference_formula_AST_changes=0,
        original_module_globals_modified=False,immutable_parent_method_tuple_modified=False,original_module_package_context_preserved=True)

def statistics_witness():
    names=('moments','metric','summaries','paired');nodes=[n for n in ast.parse(Path(S.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.require({n.name for n in nodes}==set(names),'whole unchanged four statistical functions')
    return dict(source=C.binding(Path(S.__file__)),names=list(names),whole_functions_AST_sha256=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest(),formula_changes=0,new_draws=0)

def evaluation_inputs(args):
    values={'geometry_input:'+k:v for k,v in R.inputs(args).items()}
    paths=dict(adapter=Path(__file__),unchanged_score_dispatch=Path(E13.__file__),unchanged_scalar_statistics=Path(S.__file__),
        cohort=Path(args.cohort),protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry=R.DOC/'GEOMETRY_SEALED.jsonl.gz',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',validation_started=R.DOC/'VALIDATION_STARTED.json',
        checker=R.CODE/'validation_checks.py',control_receipt=R.DOC/'CONTROL_RECEIPT.json',ledgers=R.DOC/'CONTROL_LEDGERS.jsonl.gz',
        parent_population_checks=R.DOC/'PARENT_POPULATION_CHECKS.json',same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',fixed_scoring=Path(args.v9_input)/'SCORING_RECEIPT.json',bootstrap=S.BOOTSTRAP,
        preserved_V16_full_FAIL=Path(args.decision_parent)/'VALIDATION_READY_CHECKS.json')
    values.update({k:C.binding(p) for k,p in paths.items()})
    values.update({'ordinary:'+k:C.binding(p) for k,p in ordinary_paths(args).items()});return values


def freeze(args):
    geometry_gate(args);bindings=evaluation_inputs(args);_,dispatch=isolated_score()
    C.write_new(R.output(args,'EVALUATION_PROTOCOL.json'),dict(schema='postseal_point_consensus_decision_evaluation_protocol_v18',inputs=bindings,
        methods=list(METHODS),new_methods=list(P.METHODS),contrasts=[list(v) for v in CONTRASTS],frames=245,scored_rows=1470,new_scored_rows=490,
        moment_distributions=162,moment_scalars=972,paired_groups=45,metric_CIs=135,score_dispatch=dispatch,statistics_dispatch=statistics_witness(),
        reference='unchanged N3-phase GEOMETRIC_PROXY known DEV; not independent physical pose GT',
        original_frozen_sources_and_failures_preserved=True,V16_full_scalar_validation_passed=False,
        score_gate_scope='paired point-only decision/output PASS; independent parent point check count not claimed',
        new_seed_draws=0,new_models_fits_RGB=0,inherited_score_receipt_schema_is_v13=True,
        frozen_before_own_GT_access_or_statistic_arithmetic=True))


def frozen(args):
    geometry_gate(args);own=C.read(R.DOC/'EVALUATION_PROTOCOL.json');_,dispatch=isolated_score()
    C.require(own['schema']=='postseal_point_consensus_decision_evaluation_protocol_v18' and own['inputs']==evaluation_inputs(args)
        and own['methods']==list(METHODS) and own['new_methods']==list(P.METHODS) and own['contrasts']==[list(v) for v in CONTRASTS]
        and own['score_dispatch']==dispatch and own['statistics_dispatch']==statistics_witness(),'frozen paired decision evaluation differs');return own


def score(args):
    own=frozen(args);fn,dispatch=isolated_score();before=(P13.METHODS,E13.R,E13.geometry_gate)
    C.write_new(R.output(args,'SCORING_AUTHORITY.json'),dict(schema='isolated_point_consensus_decision_proxy_score_authority_v18',
        own_evaluation_protocol=C.binding(R.DOC/'EVALUATION_PROTOCOL.json'),inputs=own['inputs'],methods=list(P.METHODS),
        score_dispatch=dispatch,reference_or_error_formula_changed=False,original_module_globals_modified=False,
        new_independent_point_only_decision_check_PASS=True,V16_full_scalar_validation_passed=False))
    try:fn(args)
    finally:C.require(before==(P13.METHODS,E13.R,E13.geometry_gate),'parent globals changed')
    frozen(args)


def score_gate(args):
    own=frozen(args);r=C.read(R.DOC/'SCORING_RECEIPT.json');a=C.read(R.DOC/'SCORING_AUTHORITY.json');s=C.read(R.DOC/'SCORING_STARTED.json')
    C.require(r['schema']=='repair_postseal_proxy_scoring_v13' and r['complete'] is True and r['scored_rows']==490 and r['frames']==245
        and r['methods']==list(P.METHODS) and r['GT_access_only_after_complete_geometry_and_cleanup'] is True,'actual paired490 scores required')
    C.require(a['schema']=='isolated_point_consensus_decision_proxy_score_authority_v18' and a['inputs']==own['inputs'] and a['score_dispatch']==own['score_dispatch']
        and a['methods']==list(P.METHODS) and a['original_module_globals_modified'] is False and a['reference_or_error_formula_changed'] is False
        and a['new_independent_point_only_decision_check_PASS'] is True and a['V16_full_scalar_validation_passed'] is False,'isolated paired authority differs')
    for path,item in ((R.DOC/'EVALUATION_PROTOCOL.json',a['own_evaluation_protocol']),(R.DOC/'PREDICTIONS.jsonl.gz',r['predictions']),
        (R.DOC/'PROTOCOL.json',r['protocol']),(R.DOC/'GEOMETRY_SEAL.json',r['geometry_seal'])):C.bound(path,item,'actual score bytes')
    C.require(s['configured_scored_rows']==490 and s['GT_only_after_full_seal_cleanup_and_independent_PASS'] is True
        and C.same_binding(s['protocol'],r['protocol']) and C.same_binding(s['independent_validation'],C.binding(R.DOC/'VALIDATION_PROTOCOL.json')),'score STARTED differs')
    return own,r


def posthoc(row,n3,observation):
    # Both views retain the same actual ROLE hybrid observations and selected q.
    result=S.posthoc(dict(row,method='IMAGE_ROLE_CORNERWISE_HYBRID'),n3,observation)
    result['method']=row['method'];result['posthoc_kernel_label_adapter_only']=True
    semantic=row['mask_audit'];legacy=row.get('legacy_mask_audit_before_known_state_correction')
    states=semantic['human_states_native'];known=sorted(k for k,state in enumerate(states) if state not in ('UNKNOWN','UNANNOTATED'))
    C.require(semantic['known_ids']==known and semantic.get('UNKNOWN_and_UNANNOTATED_are_unknown_visibility') is True,
        'score semantic known must exclude UNKNOWN/UNANNOTATED')
    result['mask_definition_provenance']=dict(legacy_known_definition_can_include_UNKNOWN=True,
        legacy_mask_audit_preserved=legacy,score_semantic_known_excludes_UNKNOWN_and_UNANNOTATED=True,
        score_semantic_known_ids=known,mask_not_a_success_gate=True,
        predicted_H_excluded_from_experimental_attempt=True,
        approved_NEW_final_fit_excludes_H=bool(row['new_pose_estimated']),
        fallback_keeps_immutable_original_all8_N3=bool(row['fallback_used']),
        scored_mask_applied_marker_is_attempt_contract_not_fallback_refit=True)
    result['decision_control_diagnostics']=dict(solver_state=row['solver']['state'],whole_U=row['solver']['used'],
        actual_fit_ids=row['solver']['fit_input_ids'] if row['new_pose_estimated'] else [],
        final_inliers=row['solver']['final_inliers'] if row['new_pose_estimated'] else [],
        same_saved_q_H_and_original_candidate_pool=True,saved_decision_view=row['decision_execution_kind'],
        candidate_generator_refit_geometry_threshold_changes=0,partial_line_factor_count=0,
        actual_new_solver_calls=0,physical_ownership_certified=False,global_uniqueness_proven=False,
        numerical_NEW_is_not_goal_success=True)
    return result


def stats(args):
    import numpy as np
    own,_=score_gate(args);ordinary_authority(args);fixed=C.read(Path(args.v9_input)/'SCORING_RECEIPT.json')
    C.require(fixed['complete'] is True and fixed['frames']==245 and fixed['fixed_control_rows']==490,'immutable Base/N3 fixed scores required')
    C.bound(Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',fixed['fixed_predictions'],'fixed original Base/N3 scores')
    paths=dict(new_scores=R.DOC/'PREDICTIONS.jsonl.gz',ordinary_scores=Path(args.ordinary_parent)/'PREDICTIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',cohort=Path(args.cohort),bootstrap=S.BOOTSTRAP,
        statistics_kernel=Path(S.__file__),own_code=Path(__file__),scoring_receipt=R.DOC/'SCORING_RECEIPT.json',
        scoring_started=R.DOC/'SCORING_STARTED.json',protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',
        validation_started=R.DOC/'VALIDATION_STARTED.json',evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',
        scoring_authority=R.DOC/'SCORING_AUTHORITY.json',same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz')
    bindings={k:C.binding(p) for k,p in paths.items()}
    for n in ('STATISTICS_STARTED.json','METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz'):R.output(args,n)
    C.write_new(R.output(args,'STATISTICS_STARTED.json'),dict(inputs=bindings,new_seed_draws=0,configured_new_rows=490,configured_saved_score_rows=1470,
        methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],known_DEV_only=True))
    groups={m:{} for m in METHODS}
    for key,allowed in (('fixed_scores',METHODS[:2]),('ordinary_scores',ORDINARY_METHODS),('new_scores',P.METHODS)):
        count=0
        for full in C.rows(paths[key]):
            row=E15.compact(full);del full
            C.require(row['method'] in allowed and row['id'] not in groups[row['method']],'unique six-method/ID')
            groups[row['method']][row['id']]=row;count+=1
        C.require(count==490,'each immutable fixed/ordinary/new paired stream490')
    cohort=C.read(args.cohort);ids=cohort['ids'];labels={f['id']:f['label'] for f in cohort['frames']}
    C.require(len(ids)==245 and all(set(g)==set(ids) for g in groups.values()),'six methods retain245')
    for fid in ids:
        original,restored,hybrid,native=[groups[m][fid] for m in (*P.METHODS,*ORDINARY_METHODS)]
        for key in ('input_points','hidden_initial','original_native_N3_points','selected_corner_ids',
            'head_checkpoint_sha256','calibration_binding','head_arm','observation_supply'):
            C.require(original[key]==restored[key]==hybrid[key],'same frozen hybrid q/H/head/CAL '+key)
        C.require(original['head_arm']=='IMAGE_ROLE' and original['observation_supply']=='CORNERWISE_HYBRID'
            and original['original_native_N3_points']==native['original_native_N3_points']==groups['N3_SUBPIX'][fid]['native_points']
            and native['hidden_initial']==original['hidden_initial'],'same native/H source control')
    with gzip.open(S.BOOTSTRAP,'rt') as stream:bootstrap=json.load(stream)
    draws=np.asarray(bootstrap['counts'],np.int64);sessions=bootstrap['sessions']
    C.require(draws.shape==(10000,13) and len(set(sessions))==13 and (draws>=0).all() and (draws.sum(1)==13).all(),'reuse original13session10000draws')
    index={session:i for i,session in enumerate(sessions)};observations={r['id']:r for r in C.rows(paths['same_role_observations']) if r['head_arm']=='IMAGE_ROLE'}
    C.require(set(observations)==set(ids),'full sameROLE245 observations')
    post=[posthoc(groups[method][fid],groups['N3_SUBPIX'][fid],observations[fid]) for method in P.METHODS for fid in ids];strata={}
    for name,selected in (('combined',ids),('easy',[fid for fid in ids if labels[fid]=='clean']),('medium',[fid for fid in ids if labels[fid]=='moderate'])):
        strata[name]=dict(frames=len(selected),ids=selected,methods={m:S.summaries([groups[m][fid] for fid in selected]) for m in METHODS},
            contrasts={a+'_minus_'+b:{scope:S.paired(groups[a],groups[b],selected,index,draws,scope) for scope in S.SCOPES} for a,b in CONTRASTS},
            diagnostics={m:S.diagnostic_summary([r for r in post if r['method']==m and r['id'] in set(selected)]) for m in P.METHODS})
    def both_better(comparator):
        value=strata['combined']['contrasts'][P.PRIMARY+'_minus_'+comparator]['common_operational']
        return value['common_frames']==245 and all(value['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg'))
    C.require({k:C.binding(p) for k,p in paths.items()}==bindings,'statistics bytes changed')
    S.C.save_rows(R.output(args,'POSTHOC_ROWS.jsonl.gz'),post)
    C.write_new(R.output(args,'METRICS.json'),dict(schema='point_consensus_decision_raw_row_metrics_v18',complete=True,methods=list(METHODS),strata=strata,
        contrasts=[list(v) for v in CONTRASTS],scopes=list(S.SCOPES),inputs=bindings,posthoc_rows=C.binding(R.DOC/'POSTHOC_ROWS.jsonl.gz'),actual_arithmetic=ARITHMETIC,
        primary_both_T_R_means_better_than_N3_on_full245=both_better('N3_SUBPIX'),
        primary_both_T_R_means_better_than_original_on_full245=both_better(P.METHODS[0]),
        primary_both_T_R_means_better_than_native_standard_on_full245=both_better('NATIVE_STANDARD_PNP'),
        primary_both_T_R_means_better_than_hybrid_standard_on_full245=both_better('HYBRID_STANDARD_PNP'),reference=own['reference'],
        new_model_image_GT_score_PnP_optimizer_training_RGB_calls=0,statistics_does_not_change_proposals_or_operating_status=True,
        V16_full_scalar_validation_passed=False,same_observation_pair_source_join_verified=True,
        comparison_is_existing_DEV_geometric_proxy_not_independent_physical_GT=True))
    with R.output(args,'METRICS.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('stratum','method','population','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'));writer.writeheader()
        for name,value in strata.items():
            for method,summary in value['methods'].items():
                for scope,metrics in summary['metrics'].items():
                    for metric,moment in metrics.items():writer.writerow(dict(stratum=name,method=method,population=scope,metric=metric,**moment))
    print('POINT_CONSENSUS_DECISION_METRICS',both_better('N3_SUBPIX'),both_better(P.METHODS[0]),flush=True)


def parser():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','score','stats')
    if not any(a.dest=='v9_input' for a in p._actions):p.add_argument('--v9-input',default=str(C.DOC))
    p.add_argument('--ordinary-parent',default=str(ORDINARY_DOC))
    return p


def main():
    args=parser().parse_args();{'freeze':freeze,'score':score,'stats':stats}[args.stage](args)

if __name__=='__main__':main()
