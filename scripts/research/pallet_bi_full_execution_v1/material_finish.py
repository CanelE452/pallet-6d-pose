"""Sequential sealed S3/S4 feature unions, own-distribution H1 and predictions.

This controller may wait for the full producer seal. It starts no extra CPU
dataset worker until production has completed, and never reads real references.
Each child command owns its dependency validation and row/checkpoint resume.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

from . import image_batch as I,material_assets as A


def execute(key,argv,output):
    output=Path(output);started=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    I.append(output/'POSTPROCESS_LEDGER.jsonl',dict(event='START',phase=key,argv=list(map(str,argv)),time_unix=time.time()))
    with open(output/(key+'.log'),'a')as stream:
        code=subprocess.call(list(map(str,argv)),stdout=stream,stderr=subprocess.STDOUT)
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt=dict(event='COMPLETE'if code==0 else'TECHNICAL_FAILURE',phase=key,return_code=code,
        wall_seconds=time.monotonic()-started,child_CPU_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        argv=list(map(str,argv)),time_unix=time.time())
    I.append(output/'POSTPROCESS_LEDGER.jsonl',receipt)
    print(I.canonical(receipt),flush=True)
    if code:raise RuntimeError(f'{key} failed; log and completed checkpoints preserved')


def run(run,d1,old_run,output,wait=False):
    run,d1,old,out=map(Path,(run,d1,old_run,output));out.mkdir(parents=True,exist_ok=True)
    folder=Path(__file__).parent
    source_bindings={name:A.sha(folder/name)for name in ('material_finish.py','material_report.py',
        'material_review.py','extra_joint.py','extra_sources.py','s1_extra.py','heads.py')}
    I.write(out/'POSTPROCESS_SOURCE_LOCK.json',dict(sources=source_bindings,
        current_real_reference_reads=0,scientific_producer_modules_modified=False))
    producer=run/'s3_s4_production';last_notice=0.
    while not(producer/'FULL_RECEIPT.json').exists():
        if not wait:raise ValueError('Full producer seal required before feature preparation')
        if time.monotonic()-last_notice>60:
            print(I.canonical(dict(stage='WAIT_FOR_COMPLETE_S3_S4_PRODUCER',new_scientific_calls=0)),flush=True)
            last_notice=time.monotonic()
        time.sleep(10)
    seal=I.read(producer/'FULL_RECEIPT.json')
    if not seal.get('complete')or seal.get('rows')!=1241 or not seal.get('all1241_completed'):
        raise ValueError('Complete full1241 producer seal required')
    if seal['method_lock_sha256']!=A.sha(producer/'METHOD_LOCK_RESOURCE_LP4.json'):
        raise ValueError('Producer active source/environment binding differs')
    completed=out/'FINISH_RECEIPT.json'
    if completed.exists():
        receipt=I.read(completed)
        if not receipt['complete']or receipt['producer_seal_sha256']!=A.sha(producer/'FULL_RECEIPT.json'):
            raise ValueError('Completed postprocessing producer binding differs')
        if receipt['postprocess_source_lock_sha256']!=A.sha(out/'POSTPROCESS_SOURCE_LOCK.json'):
            raise ValueError('Completed postprocessing source binding differs')
        for path,digest in receipt['sealed_artifact_paths'].items():
            if A.sha(path)!=digest:raise ValueError('Completed postprocessing artifact changed: '+path)
        return dict(complete=True,H1_fits=15,arms=list(I.ARMS),exact_completed_receipt_reused=True,
            receipt_sha256=A.sha(completed),new_scientific_calls=0)
    started=time.monotonic();python=Path(sys.executable)
    execute('archive_report',[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.material_report',
        '--production',producer,'--output',run/'s3_s4_archive_report'],out)
    execute('structure_review',[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.material_review','render',
        '--inputs',d1/'d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz','--production',producer,
        '--selection',producer/'STRUCTURAL_REVIEW_SELECTION24.json','--output',run/'s3_s4_structure_review'],out)
    for domain,name in [('real','REAL238'),('synthetic','SYNTHETIC1003')]:
        arms={arm:dict(extras=str(producer/f'EXTRAS_{arm}_{domain}.jsonl.gz'),
            output=str(run/(arm.lower()+'_'+domain+'_union')),seal=str(producer/'FULL_RECEIPT.json'))for arm in I.ARMS}
        spec=out/(domain+'_ARMS.json');I.write(spec,arms)
        execute('joint_'+domain,[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.extra_joint',
            '--inputs-path',d1/'d1_upstream_v1'/f'{name}_UPSTREAM.jsonl.gz',
            '--base-features-path',run/f's1_{domain}_union/HEAD_FEATURES.jsonl.gz',
            '--base-frame-dir',d1/f'd1_{domain}_candidates_v1','--arm-paths',spec,
            '--config-path',old/'CONFIG_FROZEN.json','--run',run,
            '--joint-output',out/('joint_'+domain),'--root-go'],out)
    validation={};artifacts={}
    for arm in I.ARMS:
        train_features=run/(arm.lower()+'_synthetic_union')
        real_features=run/(arm.lower()+'_real_union')
        labels=run/'s3_s4_labels'/arm;training=run/'s3_s4_training'/arm;prediction=run/'s3_s4_predictions'/arm
        execute(arm+'_labels',[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.heads','prepare',
            '--features',train_features/'HEAD_FEATURES.jsonl.gz','--feature-seal',train_features/'FEATURE_SEAL.json',
            '--targets',old/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz','--target-seal',old/'synthetic_v1/PROVISION_SEAL.json',
            '--split-lock',old/'synthetic_v1/DATA_SPLIT_LOCK.json','--output',labels,'--run',run],out)
        execute(arm+'_train',[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.heads','train',
            '--features',train_features/'HEAD_FEATURES.jsonl.gz','--feature-seal',train_features/'FEATURE_SEAL.json',
            '--labels-dir',labels,'--output',training,'--arm',arm,'--run',run],out)
        execute(arm+'_predict',[python,'-B','-m','scripts.research.pallet_bi_full_execution_v1.heads','predict',
            '--features',real_features/'HEAD_FEATURES.jsonl.gz','--feature-seal',real_features/'FEATURE_SEAL.json',
            '--training-dir',training,'--output',prediction,'--arm',arm,'--run',run],out)
        manifest=I.read(training/'TRAINING_MANIFEST.json');predseal=I.read(prediction/'PREDICTION_SEAL.json')
        runs=manifest['runs']
        if len(runs)!=3 or not manifest['complete']or predseal['rows']!=238:raise ValueError('Three seeds and all238 predictions required')
        byseed={str(r['seed']):dict(joint_success=r['best_metrics']['validation']['success_5cm5deg'],
            validation_loss=r['best_validation_loss'],best_epoch=r['best_epoch'],
            checkpoint_sha256=r['checkpoint_sha256'])for r in runs}
        validation[arm]=dict(by_seed=byseed,validation_frames=235,
            median_joint_success=float(np.median([r['joint_success']for r in byseed.values()])),
            median_validation_loss=float(np.median([r['validation_loss']for r in byseed.values()])),
            own_validation_is_development_not_independent_test=True,current_real_reference_reads=0)
        artifacts[arm]=dict(training_manifest=str(training/'TRAINING_MANIFEST.json'),
            training_manifest_sha256=A.sha(training/'TRAINING_MANIFEST.json'),
            real_prediction_seal=str(prediction/'PREDICTION_SEAL.json'),prediction_seal_sha256=A.sha(prediction/'PREDICTION_SEAL.json'),
            real_features=str(real_features/'HEAD_FEATURES.jsonl.gz'),synthetic_features=str(train_features/'HEAD_FEATURES.jsonl.gz'))
        # Atomic overwrite of a progress file is explicitly distinguished from
        # immutable scientific outputs. A restart still validates every child.
        progress=out/'VALIDATION_PROGRESS.json';temp=progress.with_suffix('.pending')
        temp.write_text(I.canonical(validation)+'\n');temp.replace(progress)
    I.write(out/'VALIDATION_SELECTION_INPUTS.json',validation)
    sealed_paths=[out/'VALIDATION_SELECTION_INPUTS.json',
        run/'s3_s4_archive_report/ROW_ARCHIVE_SEAL.json',run/'s3_s4_structure_review/REVIEW_RECEIPT.json']
    for arm in I.ARMS:
        sealed_paths.extend([run/'s3_s4_training'/arm/'TRAINING_MANIFEST.json',
            run/'s3_s4_predictions'/arm/'PREDICTION_SEAL.json',run/'s3_s4_predictions'/arm/'PREDICTIONS.jsonl.gz'])
        for domain in ('real','synthetic'):
            sealed_paths.extend([run/(arm.lower()+'_'+domain+'_union')/'FEATURE_SEAL.json',
                run/(arm.lower()+'_'+domain+'_union')/'HEAD_FEATURES.jsonl.gz'])
    receipt=dict(complete=True,arms=list(I.ARMS),H1_fits=15,seeds=[20261018,20261019,20261020],
        producer_seal_sha256=A.sha(producer/'FULL_RECEIPT.json'),artifacts=artifacts,
        all_real_predictions_sealed_before_reference_decode=True,current_real_reference_reads=0,
        wall_seconds=time.monotonic()-started,feature_and_H1_cost_ledger=str(out/'POSTPROCESS_LEDGER.jsonl'),
        sealed_artifact_paths={str(path):A.sha(path)for path in sealed_paths},
        postprocess_source_lock_sha256=A.sha(out/'POSTPROCESS_SOURCE_LOCK.json'),
        validation_selection_inputs_sha256=A.sha(out/'VALIDATION_SELECTION_INPUTS.json'))
    I.write(out/'FINISH_RECEIPT.json',receipt)
    return dict(complete=True,H1_fits=15,arms=list(I.ARMS),artifacts=artifacts)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','d1','old-run','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wait',action='store_true');a=p.parse_args()
    print(I.canonical(run(a.run,a.d1,a.old_run,a.output,a.wait)),flush=True)
