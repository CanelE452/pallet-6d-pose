"""Bounded, authenticated current-R0 and frozen-PoseFix source inference.

No source targets, real references, optimizer, or changed detector selection.
Per-frame files permit explicit resume without repeating completed forwards.
"""
from . import common as C
READS=C.source_guard()
import argparse
import copy
import gc
from pathlib import Path
import time

import cv2
import numpy as np
import torch

ATTEMPT_COUNTS={}


def rows():
    p=C.protocol()
    for name in ('input','reusable','source_contract'):C.verify(p[name])
    data=C.read(C.RAW/'SOURCE_INPUTS.json')
    assert len(data)==len({r['id'] for r in data})==5120
    assert sum(r['split']=='TRAIN' for r in data)==4096 and sum(r['split']=='VAL' for r in data)==1024
    return data


def verified_receipt(path,model):
    p=C.protocol();r=C.read(path)
    assert r['complete'] and r['model']==model and r['frames']==5120
    assert r['protocol']==C.bind(C.DOC/'SOURCE_PROTOCOL.json') and r['checkpoint']==p['checkpoints'][model]
    assert len(r['files'])==5120
    for b in r['files']:C.verify(b)
    return r


def attempt(model,index,smoke=False):
    p=C.protocol()
    path=C.RAW/'attempts'/model/(f'smoke_{index:05d}.json' if smoke else f'{index:05d}.json')
    assert not path.exists(), ('Incomplete forward attempt; stop rather than silently repeat',str(path))
    cap=p['budget']['R0_missing']+p['budget']['R0_smoke'] if model=='R0' else 5120
    if model not in ATTEMPT_COUNTS:
        ATTEMPT_COUNTS[model]=sum(1 for _ in path.parent.glob('*.json')) if path.parent.exists() else 0
    # Completion records live in a separate directory and are not double counted.
    assert ATTEMPT_COUNTS[model]<cap,(model,ATTEMPT_COUNTS[model],cap)
    C.save(path,dict(created_at=C.now(),model=model,index=index,smoke=smoke,
        protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json'),checkpoint=p['checkpoints'][model]))
    ATTEMPT_COUNTS[model]+=1
    return path


def complete_attempt(path,output):
    target=C.RAW/'completed_attempts'/path.parent.name/path.name
    C.save(target,dict(attempt=C.bind(path),output=C.bind(output)))


def completed_frame(model,index,row):
    path=frame_path(model,index)
    if not path.exists():return False
    saved=C.read(path);p=C.protocol()
    assert saved['id']==row['id'] and saved['protocol_sha']==C.sha(C.DOC/'SOURCE_PROTOCOL.json')
    assert saved['model']==model and saved['checkpoint_sha']==p['checkpoints'][model]['sha256']
    if saved['forward']:
        receipt=C.read(C.RAW/'completed_attempts'/model/f'{index:05d}.json')
        C.verify(receipt['attempt']);C.verify(receipt['output'])
        assert receipt['output']==C.bind(path)
    elif model=='R0':
        expected=adapt_cached(C.read(C.RAW/'REUSABLE_R0.json')[row['id']])
        assert saved['prediction']==expected
    return True


def state_digest(model):
    import hashlib
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare():
    assert not (C.DOC/'SOURCE_PROTOCOL.json').exists()
    contract=C.read(C.DOC/'SOURCE_CONTRACT.json')
    assert contract.get('complete') is True and contract.get('status')=='PASS', 'Source contract audit unfinished'
    source_lock=C.read(C.ROOT/'_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json')
    # Authenticate input-only metadata against its historical split lock.
    def nested(x):
        if isinstance(x,dict):
            if 'path' in x and 'sha256' in x:yield x
            else:
                for value in x.values():yield from nested(value)
        elif isinstance(x,list):
            for value in x:yield from nested(value)
    input_path=C.SOURCE/'SYNTH_INPUTS.json'
    matches=[b for b in nested(source_lock) if b['path']==str(input_path.relative_to(C.ROOT))]
    assert len(matches)==1;C.verify(matches[0])
    data=[r for r in C.read(input_path) if r['split'] in ('TRAIN','VAL')]
    assert len(data)==5120 and sum(r['split']=='TRAIN' for r in data)==4096
    assert len({r['id'] for r in data})==5120
    audit_path=C.ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1/SYNTH_DETECTION_AUDIT.json'
    audit=C.read(audit_path);assert audit['complete'] and audit['frames']==1985
    ids={r['id'] for r in data};reusable={}
    for binding in audit['files']:
        C.verify(binding);p=C.read(C.ROOT/binding['path'])
        if p['id'] in ids:reusable[p['id']]=binding
    assert len(reusable)==178
    for r in data:C.verify(r['image'])
    old=C.read(C.PREV/'INFERENCE_INPUT_LOCK.json')
    checkpoints={'R0':old['checkpoints']['R0']}
    old_predictions=C.read(C.PREV/'PREDICTIONS_LOCK.json')
    for model in C.MODELS[1:]:checkpoints[model]=old_predictions['checkpoints'][model]
    for binding in checkpoints.values():C.verify(binding)
    transitive=C.read(C.PREV/'TRAIN_TRANSITIVE_CODE_LOCK.json')['files']
    for binding in transitive:C.verify(binding)
    C.save(C.RAW/'SOURCE_INPUTS.json',data)
    C.save(C.RAW/'REUSABLE_R0.json',reusable)
    smoke=contract['smoke_selection']['selected'];assert len(smoke)==8 and all(r['id'] in reusable for r in smoke)
    protocol=dict(created_at=C.now(),objective='Learn whether a source-only whole-pose union can preserve R0 while using successful PoseFix corrections; full stable joint T/R goal unchanged.',
        phase='Bounded source-pool generation and TRAIN-only feasibility; no fit authorized by this phase contract.',
        models=list(C.MODELS),input=C.bind(C.RAW/'SOURCE_INPUTS.json'),reusable=C.bind(C.RAW/'REUSABLE_R0.json'),
        source_contract=C.bind(C.DOC/'SOURCE_CONTRACT.json'),historical_split=C.bind(C.ROOT/'_docs/experiments/pallet_selector_recovery_v1/stage2_synth_scorer/SYNTHETIC_SPLIT_LOCK.json'),
        detector_cache_audit=C.bind(audit_path),checkpoints=checkpoints,smoke_ids=[r['id'] for r in smoke],
        budget=dict(R0_missing=4942,R0_smoke=8,refiner_image_forwards_max=15360,new_fits=0,TEST_forwards=0,real_image_forwards=0),
        R0_runtime=dict(batch=1,FP32=True,cudnn_TF32=True,matmul_TF32=False,conf=.001,imgsz=640,rect=True,augment=False,half=False,already_padded=True),
        refiner_runtime=dict(batch=1,FP32=True,TF32=False,cudnn_benchmark=False,cudnn_deterministic=True,cap_fraction=None,crop_expansion=1.25),
        smoke_tolerance=dict(xy_box_absolute_px=1e-4,score_confidence_absolute=1e-6,rtol=0.,same_count_order_selected_index=True),
        stop='On smoke/input/geometry mismatch stop before large inference; no relaxed tolerance, image replacement, or unlogged restart.',
        training_followup='Separate protocol after TRAIN pool feasibility, before any fit or real routing. Source VAL quality remains unscored until fitting is frozen.',
        provenance='R0 trained broader synthetic population. Refiner TRAIN overlaps source selectorTRAIN/VAL105/26. No independent unseen-source claim.',
        source_label_scope='All 5120 original source images remain in inference. Subsequent fit/VAL eligibility is declared C2 AND a proper rigid canonical-cuboid-to-Xcf map, applied uniformly to all source metadata before inference: TRAIN2598/VAL1024. C1 and improper geometry remain recorded but are not fit labels. Original FAIL and metadata-only amendment are preserved. Real173 unchanged.',
        codes=[C.bind(C.HERE/'common.py'),C.bind(Path(__file__)),
            C.bind(C.ROOT/'scripts/research/pallet_line_pose_v1/features.py'),
            C.bind(C.ROOT/'scripts/research/pallet_pose_stable_improvement_20261001_v1/infer.py'),
            C.bind(C.ROOT/'scripts/research/pallet_posefix_large_error_v1/core.py')]+transitive)
    C.save(C.DOC/'SOURCE_PROTOCOL.json',protocol)
    C.save(C.DOC/'SOURCE_PROTOCOL_SHA.json',C.bind(C.DOC/'SOURCE_PROTOCOL.json'))
    print('SOURCE_PROTOCOL_FROZEN',protocol['budget'],flush=True)


def rgb(row):
    C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']))
    assert image is not None and list(image.shape[:2])==row['hw']
    return image


def adapt_cached(binding):
    C.verify(binding);pred=copy.deepcopy(C.read(C.ROOT/binding['path']))
    for c in pred['candidates']:
        c['keypoints_xy']=(np.asarray(c['keypoints_xy'],float)+100.).tolist()
        c['box_xyxy']=(np.asarray(c['box_xyxy'],float)+100.).tolist()
    return dict(candidates=pred['candidates'],selected_index=pred['selected_index'])


def frame_path(model,index):return C.RAW/'source_predictions'/model/f'{index:05d}.json'


def detector():
    from scripts.research.pallet_line_pose_v1.features import FrozenYoloFeatures
    p=C.protocol()
    for b in p['codes']:C.verify(b)
    data=rows()
    if (C.DOC/'R0_SOURCE_COMPLETE.json').exists():
        verified_receipt(C.DOC/'R0_SOURCE_COMPLETE.json','R0')
        print('R0_ALREADY_COMPLETE',flush=True);return
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    model=FrozenYoloFeatures(C.ROOT/p['checkpoints']['R0']['path'])
    # AutoBackend otherwise fuses on the first predict call. Perform the same
    # GPU fusion before the immutability baseline; fusion is not training.
    model.yolo.model.to('cuda').fuse(verbose=False)
    by_id={r['id']:r for r in data};reusable=C.read(C.RAW/'REUSABLE_R0.json')
    initial_state=state_digest(model.yolo.model)
    smoke_path=C.DOC/'R0_SMOKE.json'
    if not smoke_path.exists():
        C.save(C.DOC/'R0_SMOKE_START.json',dict(created_at=C.now(),protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json'),IDs=p['smoke_ids']))
        checked=[]
        for n,fid in enumerate(p['smoke_ids']):
            pending=attempt('R0',n,smoke=True)
            captured=model.predict(rgb(by_id[fid]),already_padded=True)
            expected=adapt_cached(reusable[fid])
            assert captured['selected_index']==expected['selected_index']
            assert len(captured['candidates'])==len(expected['candidates'])
            error={k:0. for k in ('keypoints_xy','box_xyxy','score','keypoints_conf')}
            for a,b in zip(captured['candidates'],expected['candidates']):
                assert a['candidate_index']==b['candidate_index']
                for key in error:
                    aa,bb=np.asarray(a[key]),np.asarray(b[key]);delta=float(np.abs(aa-bb).max())
                    error[key]=max(error[key],delta)
                    tolerance=1e-4 if key in ('keypoints_xy','box_xyxy') else 1e-6
                    assert np.allclose(aa,bb,atol=tolerance,rtol=0.),(fid,key,delta)
            checked.append(dict(id=fid,maximum_absolute_errors=error))
            proof=C.RAW/'smoke'/f'{n:05d}.json';C.save(proof,dict(id=fid,maximum_absolute_errors=error))
            complete_attempt(pending,proof)
        C.save(smoke_path,dict(complete=True,checked=checked,image_forwards=8,protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json')))
        print('R0_SMOKE_PASS',checked,flush=True)
    else:
        smoke=C.read(smoke_path);assert smoke['complete'] and smoke['protocol']==C.bind(C.DOC/'SOURCE_PROTOCOL.json')
    computed=0;reused=0;start=time.monotonic()
    for j,row in enumerate(data):
        path=frame_path('R0',j)
        if completed_frame('R0',j,row):continue
        pending=None
        if row['id'] in reusable:
            prediction=adapt_cached(reusable[row['id']]);origin='authenticated_cache';reused+=1
        else:
            pending=attempt('R0',j)
            capture=model.predict(rgb(row),already_padded=True)
            prediction=C.clean(dict(candidates=capture['candidates'],selected_index=capture['selected_index']))
            origin='new_frozen_R0';computed+=1
        C.save(path,dict(id=row['id'],prediction=prediction,origin=origin,forward=pending is not None,model='R0',
            checkpoint_sha=p['checkpoints']['R0']['sha256'],protocol_sha=C.sha(C.DOC/'SOURCE_PROTOCOL.json')))
        if pending is not None:complete_attempt(pending,path)
        if (j+1)%256==0:print('R0_SOURCE',j+1,len(data),'new',computed,flush=True)
    final_state=state_digest(model.yolo.model);assert final_state==initial_state
    model.close();del model;gc.collect();torch.cuda.empty_cache()
    files=[C.bind(frame_path('R0',j)) for j in range(len(data))]
    attempts=list((C.RAW/'attempts/R0').glob('*.json'));assert len(attempts)==4950
    C.save(C.DOC/'R0_SOURCE_COMPLETE.json',dict(complete=True,model='R0',checkpoint=p['checkpoints']['R0'],frames=len(data),
        new_forwards_this_process=computed,reused_this_process=reused,total_forward_attempts_including_smoke=len(attempts),
        model_state_before=initial_state,model_state_after=final_state,
        protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json'),files=files,wall_seconds=time.monotonic()-start))
    print('R0_SOURCE_COMPLETE',computed,reused,flush=True)


def refiners():
    from scripts.research.pallet_pose_stable_improvement_20261001_v1.infer import load_refiner,preserved
    from scripts.research.pallet_posefix_large_error_v1 import core as CORE
    p=C.protocol()
    for b in p['codes']:C.verify(b)
    data=rows();baseline=verified_receipt(C.DOC/'R0_SOURCE_COMPLETE.json','R0')
    final_lock=C.DOC/'SOURCE_PREDICTIONS_LOCK.json'
    if final_lock.exists():
        done=C.read(final_lock);assert done['complete'] and done['protocol']==C.bind(C.DOC/'SOURCE_PROTOCOL.json')
        for b in done['receipts'].values():C.verify(b)
        for model_name in C.MODELS[1:]:verified_receipt(C.DOC/f'{model_name}_SOURCE_COMPLETE.json',model_name)
        print('ALL_SOURCE_ALREADY_COMPLETE',flush=True);return
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    for model_name in C.MODELS[1:]:
        complete=C.DOC/f'{model_name}_SOURCE_COMPLETE.json'
        if complete.exists():
            verified_receipt(complete,model_name)
            continue
        model=load_refiner(model_name,p['checkpoints'][model_name],'cuda')
        initial_state=state_digest(model)
        start=time.monotonic();forwards=0
        for j,row in enumerate(data):
            path=frame_path(model_name,j)
            if completed_frame(model_name,j,row):
                preserved(C.read(frame_path('R0',j))['prediction'],C.read(path)['prediction'])
                continue
            base=C.read(frame_path('R0',j))['prediction']
            image=rgb(row);inp=CORE.prepare_input(image,base)
            pending=attempt(model_name,j) if inp is not None else None
            pred=CORE.predict(model,image,base,cap_fraction=None)
            forwards+=int(inp is not None);preserved(base,pred)
            C.save(path,dict(id=row['id'],prediction=pred,origin='frozen_refiner',forward=inp is not None,model=model_name,
                checkpoint_sha=p['checkpoints'][model_name]['sha256'],protocol_sha=C.sha(C.DOC/'SOURCE_PROTOCOL.json')))
            if pending is not None:complete_attempt(pending,path)
            if (j+1)%256==0:print('REFINER_SOURCE',model_name,j+1,len(data),flush=True)
        final_state=state_digest(model);assert initial_state==final_state
        del model;gc.collect();torch.cuda.empty_cache()
        attempts=list((C.RAW/'attempts'/model_name).glob('*.json'))
        assert len(attempts)<=5120
        C.save(complete,dict(complete=True,model=model_name,frames=len(data),image_forwards_this_process=forwards,
            total_forward_attempts=len(attempts),model_state_before=initial_state,model_state_after=final_state,
            protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json'),checkpoint=p['checkpoints'][model_name],
            files=[C.bind(frame_path(model_name,j)) for j in range(len(data))],wall_seconds=time.monotonic()-start))
        print('REFINER_SOURCE_COMPLETE',model_name,forwards,flush=True)
    C.save(C.DOC/'SOURCE_PREDICTIONS_LOCK.json',dict(complete=True,created_at=C.now(),models=list(C.MODELS),frames=len(data),
        protocol=C.bind(C.DOC/'SOURCE_PROTOCOL.json'),metadata=C.bind(C.RAW/'SOURCE_INPUTS.json'),
        receipts={m:C.bind(C.DOC/('R0_SOURCE_COMPLETE.json' if m=='R0' else f'{m}_SOURCE_COMPLETE.json')) for m in C.MODELS},
        real_GT_reads=0,source_target_reads=0,reference_guard_active=True,read_paths=sorted(set(READS))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','detector','refiners']);args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if args.stage!='prepare':
        assert torch.cuda.is_available()
        import subprocess
        line=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
        print('GPU_PREFLIGHT',line,flush=True);assert int(line.split(',')[1].strip())<80
    globals()[args.stage]()


if __name__=='__main__':main()
