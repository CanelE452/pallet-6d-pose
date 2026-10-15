"""Capture the frozen mask provider for boundary-recovery diagnostics; no fitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as V
from . import evaluation_interval_v4_2b as E
from .statewise_masks_v2 import StatewiseMaskProvider

OUT=V.FIX/'boundary_recovery_v1/image_evidence_v1'
PROVIDER=None


def initialize(paths):
    global PROVIDER
    PROVIDER=StatewiseMaskProvider.from_configs(C.read(paths['mask_descriptor']),paths['boundary_mask_config'],'cuda')


def capture(task):
    row,output=task;image_path=C.ROOT/row['image']
    if C.sha(image_path)!=row['image_sha256']:raise ValueError('RGB input changed')
    image=cv2.imread(str(image_path))
    audit=row['runtime']['mask_audit']
    hidden={k for k,s in enumerate(row['runtime']['initial_geometry']['initial_self_hidden']) if s}
    if hidden.intersection(audit['prompt_native_ids']):raise ValueError('InitialS prompt forbidden')
    result=PROVIDER(image,prompt_points=audit['prompt_points'],prompt_native_ids=audit['prompt_native_ids'],
                    prompt_box_xyxy=audit['prompt_box_xyxy'],initial_projection_xy=None,K=row['K'])
    provenance=result['provenance'];prior=audit['provider_provenance']
    for field in ('prompt_native_ids','prompt_points','prompt_box_xyxy','selected_mask_index','model_sha256'):
        if provenance.get(field)!=prior.get(field):raise ValueError('Frozen provider decision changed: '+field)
    stem=row['id'].replace(':','__');folder=Path(output)
    np.savez_compressed(folder/(stem+'.npz'),pallet_mask=result['pallet_mask'],state_pallet_mask=result['state_pallet_mask'])
    record={'id':row['id'],'partition':row['partition'],'image':row['image'],'image_sha256':row['image_sha256'],
            'mask_npz':stem+'.npz','mask_file_sha256':C.sha(folder/(stem+'.npz')),
            'provider_provenance':provenance,'initial_S_ids':sorted(hidden),'GT_or_pose_labels_read':False}
    C.write(folder/(stem+'.json'),record)
    return record


def main():
    if OUT.exists():raise FileExistsError('Preserve prior capture')
    rows,_,_=V._verify_prediction_seal(E.OUTPUT)
    rows=[r for r in rows if r['cohort']=='PRIMARY_NONSEVERE']
    if len(rows)!=95:raise ValueError('All46CAL+49TEST required')
    _,paths,model_bindings=V._runtime_models()
    source={str(Path(__file__).relative_to(C.ROOT)):C.sha(__file__)}
    for name in ('statewise_masks_v2.py','segmentation.py'):
        path=Path(__file__).parent/name;source[str(path.relative_to(C.ROOT))]=C.sha(path)
    inputs={str((E.OUTPUT/'PREDICTIONS.jsonl.gz').relative_to(C.ROOT)):C.sha(E.OUTPUT/'PREDICTIONS.jsonl.gz'),**model_bindings}
    V._verify_sources(source);V._verify_sources(inputs)
    OUT.mkdir(parents=True)
    C.write(OUT/'PROTOCOL.json',{'actual_frozen_mask_provider':True,'CAL':46,'TEST':49,'severe':0,
                'states_and_initial_pose_frozen':True,'initial_S_prompts':0,'new_pose_fits':0,'new_state_classifier_forward':0,
                'GT_or_pose_labels_used':False,'source_sha256':source,'input_sha256':inputs,'workers':3})
    started=time.monotonic();records={}
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn'),initializer=initialize,initargs=(paths,)) as pool:
        tasks={pool.submit(capture,(r,str(OUT))):r['id'] for r in rows}
        for task in as_completed(tasks):
            rec=task.result();records[rec['id']]=rec
            C.write(OUT/'LIVE_PROGRESS.json',{'completed':len(records),'total':95,'last_id':rec['id']})
            print(f'{len(records)}/95 {rec["id"]}',flush=True)
    V._verify_sources(source);V._verify_sources(inputs)
    ordered=[records[r['id']] for r in rows];C.write(OUT/'RECORDS.json',ordered)
    C.write(OUT/'SEAL.json',{'records':95,'records_sha256':C.sha(OUT/'RECORDS.json'),
            'protocol_sha256':C.sha(OUT/'PROTOCOL.json'),'source_sha256':source,'input_sha256':inputs,
            'GT_or_pose_labels_read':False,'new_fits':0,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
