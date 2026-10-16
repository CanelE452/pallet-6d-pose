"""Postseal six-method evaluation, using unchanged proxy score/statistic kernels.

The V14 operating choice is already frozen. This adapter changes method/output
dispatch and truthful incidence diagnostics, not references or scalar metrics.
"""
from collections import Counter
from contextlib import contextmanager
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from . import runner as R
from . import pipeline as G
from ..pallet_finite_edge_consensus_repair_20261011_v13 import evaluation as E13
from ..pallet_finite_edge_consensus_repair_20261011_v13 import pipeline as P13
from ..pallet_finite_edge_consensus_repair_20261011_v13 import statistics_review as D13
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_three_head_observation_20261010_v7 import statistics as S

METHODS=('BASE','N3_SUBPIX',C.COMPARATOR,C.PRIMARY,G.PARENT_METHOD,G.METHOD)
CONTRASTS=((G.METHOD,'N3_SUBPIX'),(G.METHOD,G.PARENT_METHOD),(G.METHOD,C.COMPARATOR))

def geometry_gate(args):
    core=R.frozen(args);seal,receipt,check,own,started=[C.read(R.DOC/n) for n in
        ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json')]
    C.require(seal['schema']=='selected_stable_incidence_geometry_seal_v14' and seal['complete'] is True
        and seal['frames']==seal['rows']==seal['ledger_rows']==245 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True,'complete full245 guard seal required')
    C.require(receipt['schema']=='selected_stable_incidence_execution_v14' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'complete restored guard execution required')
    C.require(check['schema']=='stable_incidence_scalar_checks_v14' and own['schema']=='stable_incidence_scalar_protocol_v14'
        and check['complete'] is True and check['passed'] is True and check['failure_count']==0
        and check['exception'] is None and check['source_unchanged'] is True and check['inputs']==own['inputs'],
        'actual independent incidence scalar PASS required')
    C.require(started['inputs']==own['inputs'] and started['new_production_calls']==0,'actual independent STARTED authority required')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',check['own_protocol'],'actual guard checker protocol')
    C.bound(R.DOC/'VALIDATION_PROTOCOL.json',started['own_protocol'],'actual checker STARTED protocol')
    C.bound(R.CODE/'validation_checks.py',check['checker'],'current successful independent checker')
    for alias,name in (('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol','PROTOCOL.json'),('geometry_seal','GEOMETRY_SEAL.json'),('repair_receipt','CONTROL_RECEIPT.json'),
        ('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        C.bound(R.DOC/name,own['inputs'][alias],'actual checked '+alias)
    for key,name in (('protocol','PROTOCOL.json'),('geometry','GEOMETRY_SEALED.jsonl.gz'),('ledgers','CONTROL_LEDGERS.jsonl.gz'),
        ('control_receipt','CONTROL_RECEIPT.json'),('parent_population_checks','PARENT_POPULATION_CHECKS.json')):
        C.bound(R.DOC/name,seal[key],'actual sealed '+key)
    return core,seal,receipt,check

def evaluation_inputs(args):
    paths=dict(adapter=Path(__file__),guard_runner=Path(R.__file__),unchanged_score_dispatch=Path(E13.__file__),
        unchanged_scalar_statistics=Path(S.__file__),finite_diagnostics=Path(D13.__file__),cohort=Path(args.cohort),
        protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry=R.DOC/'GEOMETRY_SEALED.jsonl.gz',
        geometry_validation=R.DOC/'VALIDATION_CHECKS.json',validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',
        validation_started=R.DOC/'VALIDATION_STARTED.json',checker=R.CODE/'validation_checks.py',
        control_receipt=R.DOC/'CONTROL_RECEIPT.json',ledgers=R.DOC/'CONTROL_LEDGERS.jsonl.gz',
        parent_population_checks=R.DOC/'PARENT_POPULATION_CHECKS.json',
        same_role_observations=Path(args.parent)/'OBSERVATIONS.jsonl.gz',
        parent_finite_scores=R.P.DOC/'PREDICTIONS.jsonl.gz',parent_finite_scoring=R.P.DOC/'SCORING_RECEIPT.json',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',old_local_scores=Path(args.v9_input)/'PREDICTIONS.jsonl.gz',
        point_scores=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',bootstrap=S.BOOTSTRAP)
    return {k:C.binding(p) for k,p in paths.items()}

def freeze(args):
    geometry_gate(args);bindings=evaluation_inputs(args)
    C.write_new(R.output(args,'EVALUATION_PROTOCOL.json'),dict(schema='postseal_stable_incidence_evaluation_protocol_v14',
        inputs=bindings,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],frames=245,scored_rows=1470,
        new_scored_rows=245,moment_distributions=162,moment_scalars=972,paired_groups=27,metric_CIs=81,
        reference='unchanged N3-phase GEOMETRIC_PROXY known DEV, not independent physical pose GT',
        original_V13_failures_and_full_review_preserved=True,unchanged_proxy_score_kernel=True,
        unchanged_summary_paired_bootstrap_kernels=True,new_seed_draws=0,new_models_fits_RGB=0,
        inherited_score_receipt_schema_is_v13=True,frozen_before_own_GT_access_or_statistic_arithmetic=True))

def frozen(args):
    geometry_gate(args);own=C.read(R.DOC/'EVALUATION_PROTOCOL.json')
    C.require(own['schema']=='postseal_stable_incidence_evaluation_protocol_v14' and own['inputs']==evaluation_inputs(args)
        and own['methods']==list(METHODS) and own['contrasts']==[list(v) for v in CONTRASTS],
        'own postseal evaluation freeze differs');return own

@contextmanager
def score_dispatch():
    # Normal package imports keep relative-import semantics intact. The
    # frozen score body still imports its METHODS tuple from its own module.
    old=E13.R,E13.geometry_gate,P13.METHODS
    facade=SimpleNamespace(**vars(R));facade.A=SimpleNamespace(protect=R.protect);facade.audit_args=lambda a:a
    E13.R,E13.geometry_gate,P13.METHODS=facade,geometry_gate,G.METHODS
    try:yield
    finally:E13.R,E13.geometry_gate,P13.METHODS=old

def score(args):
    own=frozen(args)
    C.write_new(R.output(args,'SCORING_AUTHORITY.json'),dict(schema='unchanged_V13_proxy_score_dispatch_for_guard_v14',
        own_evaluation_protocol=C.binding(R.DOC/'EVALUATION_PROTOCOL.json'),inputs=own['inputs'],methods=list(G.METHODS),
        original_V13_source_changed=False,method_and_output_dispatch_only=True,reference_or_error_formula_changed=False,
        old_failed_CPU_and_cusp_inspectors_preserved=True,new_independent_incidence_check_PASS=True))
    with score_dispatch():E13.score(args)
    frozen(args)

def posthoc(row,n3,point,observation):
    adapted=dict(row,method=C.PARENT_PRIMARY);result=S.posthoc(adapted,n3,observation)
    finite=D13.finite_fields(row,n3,point);w=row['stable_incidence_guard'];geometry=w.get('stable_incidence_geometry')
    result.update(method=G.METHOD,posthoc_kernel_label_adapter_only=True,finite_edge_diagnostics=finite,
        stable_incidence_diagnostics=dict(parent_finite_NEW=w['original_numerical_pose_available'],guard_applied=w['gate_applied'],
            guard_state=w['guard_state'],guard_accepted=w['accepted'],selected_candidate_reselected=False,
            original_selected_fit_point_ids=w.get('actual_fit_input_ids',[]),original_selected_fit_line_edges=w.get('actual_fit_line_edges',[]),
            accepted_fit_point_ids=row['solver']['fit_input_ids'] if row['new_pose_estimated'] else [],
            accepted_fit_line_edges=row['solver']['fit_line_edges'] if row['new_pose_estimated'] else [],
            finite_scalar_rows=w.get('finite_scalar_rows'),normal_incidence_scalar_rows=w.get('incidence_scalar_rows'),
            longitudinal_rows_removed_only_from_observability=w.get('removed_longitudinal_rows'),
            observed_incidence_rank=geometry['observed_normal_joint']['numerical_rank'] if geometry else None,
            modeled_incidence_rank=geometry['modeled_normal_joint']['numerical_rank'] if geometry else None,
            observed_weak=w.get('observed_weak'),modeled_weak=w.get('modeled_weak'),guard_counts=w['counts'],
            additional_optimizer_PnP_or_candidate_selection_calls=0,original_finite_optimizer_counts_are_copied_history=True,
            accepted_H_projection_reassigned=False,physical_ownership_certified=False,global_unique_pose_proven=False,
            parent_one_sided_hinge_rank_is_not_stable_incidence_proof=True),
        finite_hinges_still_used_by_original_optimizer_and_consensus=True,
        stable_observability_approval_uses_actual_points_and_line_normals_only=True)
    return result

def stats(args):
    import numpy as np
    own=frozen(args);receipt=C.read(R.DOC/'SCORING_RECEIPT.json')
    C.require(receipt['complete'] is True and receipt['scored_rows']==245 and receipt['methods']==list(G.METHODS),'complete actual guard245 scores required')
    C.bound(R.DOC/'PREDICTIONS.jsonl.gz',receipt['predictions'],'actual guard scores')
    parent=C.read(R.P.DOC/'SCORING_RECEIPT.json');C.require(parent['complete'] is True and parent['scored_rows']==245,'complete immutable finite comparator')
    C.bound(R.P.DOC/'PREDICTIONS.jsonl.gz',parent['predictions'],'immutable parent finite scores')
    paths=dict(new_scores=R.DOC/'PREDICTIONS.jsonl.gz',finite_scores=R.P.DOC/'PREDICTIONS.jsonl.gz',
        fixed_scores=Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',old_local_scores=Path(args.v9_input)/'PREDICTIONS.jsonl.gz',
        point_scores=Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',cohort=Path(args.cohort),bootstrap=S.BOOTSTRAP,
        statistics_kernel=Path(S.__file__),own_code=Path(__file__),scoring_receipt=R.DOC/'SCORING_RECEIPT.json',
        protocol=Path(args.protocol),geometry_seal=R.DOC/'GEOMETRY_SEAL.json',geometry_validation=R.DOC/'VALIDATION_CHECKS.json',
        validation_protocol=R.DOC/'VALIDATION_PROTOCOL.json',evaluation_protocol=R.DOC/'EVALUATION_PROTOCOL.json',scoring_authority=R.DOC/'SCORING_AUTHORITY.json')
    paths['same_role_observations']=Path(args.parent)/'OBSERVATIONS.jsonl.gz'
    bindings={k:C.binding(p) for k,p in paths.items()}
    for n in ('STATISTICS_STARTED.json','METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz'):R.output(args,n)
    C.write_new(R.output(args,'STATISTICS_STARTED.json'),dict(inputs=bindings,new_seed_draws=0,configured_new_rows=245,
        configured_saved_score_rows=1470,methods=list(METHODS),contrasts=[list(v) for v in CONTRASTS],known_DEV_only=True))
    groups={m:{} for m in METHODS}
    for key in ('fixed_scores','old_local_scores','point_scores','finite_scores','new_scores'):
        for row in S.compact_rows(paths[key]):
            C.require(row['method'] in groups and row['id'] not in groups[row['method']],'unique saved method/ID required');groups[row['method']][row['id']]=row
    cohort=C.read(args.cohort);ids=cohort['ids'];labels={f['id']:f['label'] for f in cohort['frames']}
    C.require(len(ids)==245 and all(set(g)==set(ids) for g in groups.values()),'full six methods retain245 frames')
    with gzip.open(S.BOOTSTRAP,'rt') as f:bootstrap=json.load(f)
    draws=np.asarray(bootstrap['counts'],np.int64);sessions=bootstrap['sessions']
    C.require(draws.shape==(10000,13) and len(set(sessions))==13 and (draws>=0).all() and (draws.sum(1)==13).all(),'same original13session10000 draws')
    index={s:i for i,s in enumerate(sessions)};observations={r['id']:r for r in C.rows(paths['same_role_observations']) if r['head_arm']=='IMAGE_ROLE'}
    C.require(set(observations)==set(ids),'full original sameROLE observations required')
    post=[posthoc(groups[G.METHOD][fid],groups['N3_SUBPIX'][fid],groups[C.COMPARATOR][fid],observations[fid]) for fid in ids]
    strata={}
    for name,selected in (('combined',ids),('easy',[i for i in ids if labels[i]=='clean']),('medium',[i for i in ids if labels[i]=='moderate'])):
        strata[name]=dict(frames=len(selected),ids=selected,methods={m:S.summaries([groups[m][fid] for fid in selected]) for m in METHODS},
            contrasts={a+'_minus_'+b:{scope:S.paired(groups[a],groups[b],selected,index,draws,scope) for scope in S.SCOPES} for a,b in CONTRASTS},
            diagnostics={G.METHOD:S.diagnostic_summary([r for r in post if r['id'] in set(selected)])},
            incidence_guard_states=dict(Counter(r['stable_incidence_diagnostics']['guard_state'] for r in post if r['id'] in set(selected))))
    primary=strata['combined']['contrasts'][G.METHOD+'_minus_N3_SUBPIX']['common_operational']
    improved=primary['common_frames']==245 and all(primary['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg'))
    C.require({k:C.binding(p) for k,p in paths.items()}==bindings,'fixed statistics bytes changed')
    S.C.save_rows(R.output(args,'POSTHOC_ROWS.jsonl.gz'),post)
    result=dict(schema='stable_incidence_raw_row_metrics_v14',complete=True,methods=list(METHODS),strata=strata,
        contrasts=[list(v) for v in CONTRASTS],scopes=list(S.SCOPES),inputs=bindings,posthoc_rows=C.binding(R.DOC/'POSTHOC_ROWS.jsonl.gz'),
        actual_arithmetic=dict(scored_rows=1470,new_rows=245,posthoc_rows=245,methods=6,strata=3,moment_distributions=162,
            moment_scalars=972,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0),
        primary_both_T_R_means_better_than_N3_on_full245=improved,reference=own['reference'],
        new_model_image_GT_score_PnP_optimizer_training_RGB_calls=0,guard_changes_only_same_selected_pose_operating_availability=True)
    C.write_new(R.output(args,'METRICS.json'),result)
    with R.output(args,'METRICS.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('stratum','method','population','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'));writer.writeheader()
        for stratum,value in strata.items():
            for method,summary in value['methods'].items():
                for population,metrics in summary['metrics'].items():
                    for metric,moment in metrics.items():writer.writerow(dict(stratum=stratum,method=method,population=population,metric=metric,**moment))
    print('STABLE_INCIDENCE_METRICS',improved,flush=True)

def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('freeze','score','stats');args=p.parse_args()
    {'freeze':freeze,'score':score,'stats':stats}[args.stage](args)
if __name__=='__main__':main()
