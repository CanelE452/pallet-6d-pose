"""Postseal scoring and fixed raw-row statistics for the one finite edge consensus repair.

Reference poses and human labels are available only after complete geometry,
restored fitting guards and independent geometry PASS. They never choose the
filtered observation set, weak operating gate or the optimizer's branch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import gzip
import json
from pathlib import Path
import time
import traceback

from . import runner as R
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9 import evaluator as E9
from ..pallet_same_observation_controls_20261010_v8 import evaluator as E8


def geometry_gate(args):
    own=R.frozen(args);folder=R.DOC
    seal,receipt,validation=[C.read(folder/n) for n in ('GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json')]
    C.require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==245
              and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,
              'complete full245 GT-free geometry seal required')
    C.require(receipt['complete'] is True and receipt['error'] is None and receipt['cleanup_error'] is None
              and receipt['cleanup_completed'] is True,'complete restored numerical scope required')
    C.require(validation['schema']=='finite_edge_consensus_scalar_checks_v13'
              and validation['complete'] is True and validation['passed'] is True,'independent repair geometry PASS required')
    checker_protocol=C.read(folder/'VALIDATION_PROTOCOL.json')
    C.require(validation['inputs']==checker_protocol['inputs'],'successful checker input protocol differs')
    for alias,path in (('geometry',folder/'GEOMETRY_SEALED.jsonl.gz'),('ledgers',folder/'CONTROL_LEDGERS.jsonl.gz'),
                       ('repair_protocol',Path(args.protocol)),('geometry_seal',folder/'GEOMETRY_SEAL.json'),
                       ('repair_receipt',folder/'CONTROL_RECEIPT.json')):
        C.bound(path,checker_protocol['inputs'][alias],'actual checked repair '+alias)
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
                     ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('CONTROL_RECEIPT.json','control_receipt')):
        C.bound(folder/name,seal[key],'new sealed geometry '+name)
    C.bound(folder/'VALIDATION_PROTOCOL.json',validation['own_protocol'],'repair checker protocol')
    C.bound(R.CODE/'validation_checks.py',validation['checker'],'repair checker source')
    return own,seal,receipt,validation


def score(args):
    from ..pallet_three_head_observation_20261010_v7.run import RowWriter
    from ..pallet_three_head_observation_20261010_v7 import common as V7
    from .pipeline import METHODS
    own,seal,receipt,validation=geometry_gate(args)
    for name in ('SCORING_STARTED.json','SCORING_RECEIPT.json','SCORING_FAILURE.json',
                 'PREDICTIONS.jsonl.gz','INTERRUPTED_PREDICTIONS.jsonl.gz'):
        R.output(args,name)
    C.write_new(R.output(args,'SCORING_STARTED.json'),dict(protocol=C.binding(args.protocol),
        geometry_seal=C.binding(R.DOC/'GEOMETRY_SEAL.json'),independent_validation=validation['own_protocol'],
        configured_scored_rows=245,GT_only_after_full_seal_cleanup_and_independent_PASS=True))
    began=time.monotonic();writer=None;count=0;error=None
    try:
        authority={f['id']:f for f in C.read(V7.OLD/'INPUTS.json')['frames']}
        ids=C.read(R.DOC/'PARENT_POPULATION_CHECKS.json')['ordered_ids']
        scope=dict(ids=ids,authority=authority,count=245)
        fixed={(r['method'],r['id']):r for r in C.rows(Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz')}
        C.require(len(fixed)==490 and all(set(fid for method,fid in fixed if method==m)==set(ids)
                  for m in ('BASE','N3_SUBPIX')),'full immutable fixed reference controls required')
        writer=RowWriter(R.output(args,'PREDICTIONS.jsonl.gz'),interrupted_path=R.output(args,'INTERRUPTED_PREDICTIONS.jsonl.gz'))
        with C.legacy_context(args) as (old,_,_),E9.forbid_models_fits_and_local_optimizers() as (local_guard,fit_guard):
            from ..pallet_kp_corrected_supervision_20261010_v1.scoring_resume import references
            from ..pallet_observation_refiner_20261009_v1.evaluate import score as frozen_score
            eval_module,frames,targets,reference=references(old,scope)
            label_path=Path(args.source_root)/('_docs/experiments/pallet_combined_closeout_20261003_v1/'
                'closeout_20261006_v1/visibility_square/STATIC_VISIBILITY_MERGE_AUDIT.json')
            labels={(r['frame_id'],r['corner_id']):r['category'] for r in old.read(label_path)['rows']
                    if r['population']=='DEV319' and r['frame_id'] in set(ids)}
            for row in C.rows(R.DOC/'GEOMETRY_SEALED.jsonl.gz'):
                C.require(row['method'] in METHODS and row['id'] in ids,'new scored method/population differs')
                scored=frozen_score(eval_module,frames[row['id']],targets[row['id']],row)
                # The unchanged V8 annotator recognizes its own method names.
                # Map the label only so actual H application is represented.
                method=scored['method'];adapted=dict(scored,method=C.COMPARATOR)
                scored=E8.annotate(adapted,fixed,targets[row['id']],labels)
                legacy=copy.deepcopy(scored['mask_audit']);states=legacy['human_states_native']
                known={k for k,state in enumerate(states) if state not in ('UNKNOWN','UNANNOTATED')}
                humanH={k for k,state in enumerate(states) if state=='SELF_OCCLUDED'}
                H=set(scored['hidden_initial'])
                scored['legacy_mask_audit_before_known_state_correction']=legacy
                scored['mask_audit'].update(known_ids=sorted(known),mask_applied=True,
                    mask_wrong_on_known=bool((H^humanH)&known),
                    excluded_set_differs_from_human_self_on_known=bool((H^humanH)&known),
                    UNKNOWN_and_UNANNOTATED_are_unknown_visibility=True)
                scored.update(method=method,annotation_method_label_adapter_only=True,
                    mask_applied_to_final_fit=True,oracle=False,
                    actual_pose_reference='unchanged N3-phase GEOMETRIC_PROXY known DEV')
                writer.write(scored);count+=1
            C.require(not local_guard and not fit_guard,'model/fit attempted during postseal scoring')
        C.require(count==245,'complete245 scored repair rows required')
        R.frozen(args);protected=R.A.protect(R.audit_args(args));writer.close()
        C.require(not writer.close_errors,'score stream close failed');writer.promote()
        C.write_new(R.output(args,'SCORING_RECEIPT.json'),dict(schema='repair_postseal_proxy_scoring_v13',complete=True,
            scored_rows=245,frames=245,methods=list(METHODS),protocol=C.binding(args.protocol),
            geometry_seal=C.binding(R.DOC/'GEOMETRY_SEAL.json'),predictions=C.binding(R.DOC/'PREDICTIONS.jsonl.gz'),
            reference=C.finite(reference),unchanged_scorer=C.binding(Path(frozen_score.__code__.co_filename)),
            human_visibility=C.binding(label_path),protected=protected,GT_access_only_after_complete_geometry_and_cleanup=True,
            new_model_fit_optimizer_training_RGB_calls=0,elapsed_seconds=time.monotonic()-began,
            elapsed_is_deployment_latency=False,automatic_retry=False))
        print('REPAIR_SCORED',count,flush=True)
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if writer is not None:writer.preserve_interrupted()
        C.write_new(R.output(args,'SCORING_FAILURE.json'),dict(complete=False,error=error,scored_prefix=count,
            interrupted=C.binding(writer.interrupted_path) if writer and writer.interrupted_path.is_file() else None,
            new_model_fit_training_RGB_calls=0,automatic_retry=False))
        raise


def stats(args):
    import numpy as np
    from ..pallet_three_head_observation_20261010_v7 import statistics as S
    from ..pallet_sparse_local_line_20261010_v9 import statistics as S9
    from .pipeline import METHODS
    geometry_gate(args);receipt=C.read(R.DOC/'SCORING_RECEIPT.json')
    C.require(receipt['complete'] is True and receipt['scored_rows']==245,'complete new scoring receipt required')
    C.bound(R.DOC/'PREDICTIONS.jsonl.gz',receipt['predictions'],'new scored rows')
    for name in ('STATISTICS_STARTED.json','METRICS.json','METRICS.csv','POSTHOC_ROWS.jsonl.gz'):
        R.output(args,name)
    input_paths={'new_scores':R.DOC/'PREDICTIONS.jsonl.gz','fixed_scores':Path(args.v9_input)/'FIXED_PREDICTIONS.jsonl.gz',
        'old_local_scores':Path(args.v9_input)/'PREDICTIONS.jsonl.gz',
        'point_scores':Path(args.v9_input)/'COMPARATOR_PREDICTIONS.jsonl.gz',
        'cohort':Path(args.cohort),'bootstrap':S.BOOTSTRAP,'statistics_kernel':Path(S.__file__),
        'own_code':Path(__file__),'scoring_receipt':R.DOC/'SCORING_RECEIPT.json','protocol':Path(args.protocol)}
    inputs={k:C.binding(p) for k,p in input_paths.items()}
    C.write_new(R.output(args,'STATISTICS_STARTED.json'),dict(inputs=inputs,new_seed_draws=0,
        configured_new_rows=245,fixed_strata=('combined','easy','medium'),known_DEV_only=True))
    began=time.monotonic();all_methods=('BASE','N3_SUBPIX',C.COMPARATOR,C.PRIMARY,*METHODS)
    groups={m:{} for m in all_methods}
    for key in ('fixed_scores','old_local_scores','point_scores','new_scores'):
        for row in S.compact_rows(input_paths[key]):
            C.require(row['method'] in groups and row['id'] not in groups[row['method']],'unique original method/ID required')
            groups[row['method']][row['id']]=row
    cohort=C.read(args.cohort);ids=cohort['ids'];labels={f['id']:f['label'] for f in cohort['frames']}
    C.require(len(ids)==245 and all(set(g)==set(ids) for g in groups.values()),'all five methods retain245 frames')
    with gzip.open(S.BOOTSTRAP,'rt',encoding='utf-8') as stream:bootstrap=json.load(stream)
    draws=np.asarray(bootstrap['counts'],np.int64);sessions=bootstrap['sessions']
    C.require(draws.shape==(10000,13) and len(set(sessions))==13 and (draws>=0).all() and (draws.sum(1)==13).all(),
              'reuse original13session10000 draws')
    index={s:i for i,s in enumerate(sessions)}
    contrasts=((METHODS[0],'N3_SUBPIX'),(METHODS[0],C.PRIMARY),(METHODS[0],C.COMPARATOR))
    observations={r['id']:r for r in C.rows(Path(args.parent)/'OBSERVATIONS.jsonl.gz') if r['head_arm']=='IMAGE_ROLE'}
    C.require(set(observations)==set(ids),'full sameROLE observations required for posthoc')
    posthoc=[]
    for method in METHODS:
        for fid in ids:
            row=groups[method][fid]
            posthoc.append(S9.posthoc(row,groups['N3_SUBPIX'][fid],groups[C.COMPARATOR][fid],observations[fid]))
    strata={}
    for name,selected in (('combined',ids),('easy',[i for i in ids if labels[i]=='clean']),
                          ('medium',[i for i in ids if labels[i]=='moderate'])):
        strata[name]=dict(frames=len(selected),ids=selected,
            methods={m:S.summaries([groups[m][fid] for fid in selected]) for m in all_methods},
            contrasts={a+'_minus_'+b:{scope:S.paired(groups[a],groups[b],selected,index,draws,scope)
                                     for scope in S.SCOPES} for a,b in contrasts},
            diagnostics={m:S.diagnostic_summary([r for r in posthoc if r['method']==m and r['id'] in set(selected)])
                         for m in METHODS})
    primary=strata['combined']['contrasts'][METHODS[0]+'_minus_N3_SUBPIX']['common_operational']
    improved=primary['common_frames']==245 and all(primary['metrics'][m]['mean_delta']<0 for m in ('translation_cm','rotation_deg'))
    C.require({k:C.binding(p) for k,p in input_paths.items()}==inputs,'stats input bytes changed')
    S.C.save_rows(R.output(args,'POSTHOC_ROWS.jsonl.gz'),posthoc)
    result=dict(schema='fixed_repair_raw_row_metrics_v13',complete=True,methods=list(all_methods),strata=strata,
        contrasts=[list(p) for p in contrasts],scopes=list(S.SCOPES),inputs=inputs,
        actual_arithmetic=dict(scored_rows=1225,new_rows=245,posthoc_rows=245,methods=5,strata=3,
            moment_distributions=135,moment_scalars=810,paired_groups=27,metric_CIs=81,new_bootstrap_draws=0),
        posthoc_rows=C.binding(R.DOC/'POSTHOC_ROWS.jsonl.gz'),
        primary_both_T_R_means_better_than_N3_on_full245=improved,
        reference='GEOMETRIC_PROXY known DEV; not independently measured physical pose',
        new_model_image_GT_score_PnP_optimizer_training_RGB_calls=0,elapsed_seconds=time.monotonic()-began)
    C.write_new(R.output(args,'METRICS.json'),result)
    with R.output(args,'METRICS.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=('stratum','method','population','metric','n','mean','sample_variance','sample_std','median','P90','max','unit'))
        writer.writeheader()
        for stratum,value in strata.items():
            for method,summary in value['methods'].items():
                for population,metrics in summary['metrics'].items():
                    for metric,moment in metrics.items():writer.writerow(dict(stratum=stratum,method=method,population=population,metric=metric,**moment))
    print('REPAIR_METRICS',improved,{m:strata['combined']['methods'][METHODS[0]]['metrics']['operational'][m]['mean']
                                 for m in S.METRICS},flush=True)


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('score','stats');args=p.parse_args()
    {'score':score,'stats':stats}[args.stage](args)


if __name__=='__main__':main()
