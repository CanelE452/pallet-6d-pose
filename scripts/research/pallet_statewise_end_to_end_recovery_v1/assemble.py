"""Seal all actual automatic outputs before the fixed real reference is read."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time

from .common import read,rows,save_rows,sha,write,PredictionFirewall


def run(run_dir):
    run_dir=Path(run_dir);started=time.monotonic()
    config=read(run_dir/'CONFIG_FROZEN.json');plan=read(run_dir/'PLAN_LOCK.json')
    if sha(run_dir/'CONFIG_FROZEN.json')!=plan['config_sha256']:
        raise ValueError('Scientific configuration changed')
    if sha(run_dir/'ARM_REGISTRY.json')!=plan['arm_registry_sha256']:
        raise ValueError('Predeclared arm registry changed')
    guard=PredictionFirewall(config['forbidden_prediction_paths']).install()
    batch=run_dir/'real_batch_all_available_v1';bseal=read(batch/'BATCH_SEAL.json')
    if not bseal['complete'] or bseal['rows']!=238 or bseal['reference_inputs']is not False:
        raise ValueError('Complete real238 prediction-only batch required')
    if sha(batch/'PREDICTIONS.jsonl.gz')!=bseal['predictions_sha256']:
        raise ValueError('Real numerical prediction bank changed')
    input_path=run_dir/'inputs_all_available_v1/PREDICTION_INPUTS238.jsonl.gz'
    if sha(input_path)!=bseal['input_sha256']:
        raise ValueError('All available original candidate input bytes changed')
    inputs={f['id']:f for f in rows(input_path)}
    bridge=run_dir/'real_features_v1';fs=read(bridge/'FEATURE_SEAL.json')
    if sha(bridge/'HEAD_FEATURES.jsonl.gz')!=fs['head_features_sha256']:
        raise ValueError('Real feature alias bridge changed')
    if fs['reference_reads']!=0 or fs['feature_scalars_changed']!=0 or fs['actual_poses_changed']!=0:
        raise ValueError('Feature bridge is not lossless and reference-free')
    features={f['id']:f for f in rows(bridge/'HEAD_FEATURES.jsonl.gz')}
    learned={};learning_status={};learned_seal=None
    if (run_dir/'head_training_v1/TRAINING_NOT_RUN.json').exists():
        reason=read(run_dir/'head_training_v1/TRAINING_NOT_RUN.json')
        learning_status=dict(status='NOT_APPLICABLE',reason=reason)
    else:
        ld=run_dir/'learned_real_v1';learned_seal=read(ld/'PREDICTION_SEAL.json')
        if sha(ld/'PREDICTIONS.jsonl.gz')!=learned_seal['predictions_sha256'] or learned_seal['reference_reads']!=0:
            raise ValueError('Learned actual prediction seal changed')
        learned={f['id']:f for f in rows(ld/'PREDICTIONS.jsonl.gz')}
        if set(learned)!=set(inputs):
            raise ValueError('Learned exact238 population differs')
        learning_status=dict(status='EXECUTED',training_manifest_sha256=learned_seal['training_manifest_sha256'])
    ledger=[]
    for row in rows(batch/'PREDICTIONS.jsonl.gz'):
        frame=inputs[row['id']];feature=features[row['id']]
        if frame['partition']!=row['partition']:
            raise ValueError('Immutable real population changed')
        row['archived_initial_candidates']=frame['candidate_archive_all']
        row['initial_candidate_source_counts']=frame['candidate_source_counts']
        row['initial_positive_depth_admission_count']=len(frame['candidates'])
        row['original_candidate_archive_counts']=dict(
            available_source_pose_roles=len(frame['candidate_archive_all']),
            positive_depth_admitted_source_pose_roles=len(frame['candidates']),
            projection_ineligible_preserved_source_pose_roles=len(frame['candidate_archive_all'])-len(frame['candidates']),
            original_outcome_Rt_unavailable=sum(v for k,v in frame['candidate_source_counts'].items()if k.endswith('_Rt_unavailable')))
        row['image_sha256']=frame['image_sha256']
        if learned:
            preds=learned[row['id']]['predictions']
            if set(preds)&set(row['predictions']):
                raise ValueError('Learned prediction overwrites an existing arm')
            row['predictions'].update(preds)
            eligible={c['candidate_id']for c in feature['candidates']if c['eligible']}
            for arm in preds:
                row['selection_banks'][arm]=dict(candidates=row['learning_bank'],
                    eligible_ids=sorted(eligible),family_stage='NOT_APPLICABLE_GLOBAL_FINITE_CANDIDATE_HEAD',
                    source='common union of original carried + actual full-solver candidate outputs',
                    abstention_baseline='N0',same_feature_scalars_as_generation=True)
        row['exploratory_development_evaluation']=True
        row['reference_inputs']=False
        ledger.append(row)
    if [r['id']for r in ledger]!=bseal['ordered_ids'] or len(ledger)!=238:
        raise ValueError('Original real prediction ordering changed')
    arms=list(ledger[0]['predictions'])
    registry=read(run_dir/'ARM_REGISTRY.json')
    execution={r['name']:dict(status='EXECUTED'if r['name']in arms else 'NOT_APPLICABLE',
        reason=None if r['name']in arms else learning_status.get('reason','No actual output produced')) for r in registry['arms']}
    write(run_dir/'ARM_EXECUTION_STATUS.json',execution)
    save_rows(run_dir/'PREDICTIONS.jsonl.gz',ledger)
    receipt=dict(schema='ALL_AUTOMATIC_STATEWISE_PREDICTIONS_SEAL_V1',complete=True,before_scoring=True,
        reference_inputs=False,rows=238,ordered_ids=[r['id']for r in ledger],arms=arms,
        predictions_sha256=sha(run_dir/'PREDICTIONS.jsonl.gz'),plan_lock_sha256=sha(run_dir/'PLAN_LOCK.json'),
        source_sha256=sha(__file__),scientific_config_sha256=sha(run_dir/'CONFIG_FROZEN.json'),
        batch_seal_sha256=sha(batch/'BATCH_SEAL.json'),feature_seal_sha256=sha(bridge/'FEATURE_SEAL.json'),
        learned_prediction_seal_sha256=sha(run_dir/'learned_real_v1/PREDICTION_SEAL.json')if learned else None,
        learning_status=learning_status,all238_frames_and_actual_failures_kept=True,
        technical_amendments={n:sha(run_dir/n)for n in [
            'INPUT_CONTRACT_REPAIR_V1.json','inputs_all_available_v1/INPUT_COVERAGE_INDEPENDENT.json',
            'TECHNICAL_FEATURE_PACKAGING_AMENDMENT.json',
            'SOURCE_SYMMETRY_PROSE_CORRECTION.json',
            'L_PRELABEL_SYMMETRY_DESCRIPTION_AMENDMENT.json']},
        original_reference_sha256='d488564817a8cac8d0103350f03ac61d78bdeb56163c6a5a4daffe414f0f2af1',
        reference_numeric_document_not_opened=True,firewall=guard.receipt(),elapsed_seconds=time.monotonic()-started)
    write(run_dir/'PREDICTION_SEAL.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args()
    r=run(a.run_dir);print(dict(complete=r['complete'],rows=r['rows'],arms=r['arms'],predictions_sha256=r['predictions_sha256']))
