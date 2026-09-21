"""User-approved CAD8 pseudo adaptation; all CAD excluded from current eval."""
import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_type_selftrain_v1 import train as T, common as C
from scripts.research.pallet_type_selftrain_v1.pseudo import top
from scripts.research.pallet_cad_r0_gt_gallery_v1 import build as G

RAW=ROOT/'data/pallet/results/pallet_cad8_selftrain_v1'
DOC=ROOT/'_docs/experiments/pallet_cad8_selftrain_v1'
BASE=ROOT/'outputs/pallet_cad_refiner_comparison_v1'
ARMS={'N3_PNP':'N3_DIM_SYM_seed1','REPLAY_PNP':'REPLAY_RAW'}
INDICES=[2,6,7,8,9,10,11,12]
read=G.read


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);G.write(p,value)


def gpu():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    foreign=[x for x in proc.splitlines() if x.split(',')[0].strip()!=str(os.getpid()) and 'rustdesk' not in x.lower()]
    assert not foreign,('Preserve other GPU jobs',foreign)
    assert float(raw.split(',')[3])<80,('Thermal stop',raw)
    return dict(gpu=raw,processes=proc)


def prepare():
    from scripts.paper.paired_uncertainty_and_tails import workspace_conditions
    from scripts.research.pallet_type_selftrain_v1.test_contract import test_pseudo_label_padding_and_no_mutation,test_uncertain_keypoint_true_ignore
    test_pseudo_label_padding_and_no_mutation();test_uncertain_keypoint_true_ignore()
    inputs=read(BASE/'INPUTS.json');selected=[inputs[i-1] for i in INDICES]
    payload=read(BASE/'self_occlusion/PREDICTIONS.json')
    conditions=workspace_conditions()
    for r in selected:
        G.verify(r['image']);m=conditions[r['image']['path']]
        assert m['occlusion']=='none' and m['truncation']=='none'
    oldeval=read(C.DOC/'EVAL_PROTOCOL.json')
    evalrows=[copy.deepcopy(r) for r in oldeval['records'] if r['kind']=='PLASTIC' and r['session']!='eval_cad']
    teacher_sessions={i.split(':')[0] for i in read(ROOT/'_docs/experiments/pallet_posefix_replay_v1/INPUT_LOCK.json')['real_ids']}
    train_hashes={r['image']['sha256'] for r in selected}
    for r in evalrows:
        assert r['image']['sha256'] not in train_hashes
        G.verify(r['image']);G.verify(r['annotation'])
        m=conditions[r['image']['path']]
        r.update(occlusion=m['occlusion'],truncation=m['truncation'],teacher_session_disjoint=r['session'] not in teacher_sessions)
    assert len(evalrows)==176
    primary=[r['id'] for r in evalrows if r['teacher_session_disjoint'] and r['occlusion'] not in ['none','unknown','']]
    assert len(primary)==96
    sources=[BASE/'self_occlusion/PREDICTIONS.json',BASE/'self_occlusion/PROTOCOL.json',BASE/'INPUTS.json',
        C.DOC/'TRAIN_PROTOCOL.json',C.DOC/'EVAL_PROTOCOL.json',ROOT/'data/evaluation/pallet_eval_v1/manifests/frames.csv',
        ROOT/'_docs/experiments/pallet_posefix_replay_v1/INPUT_LOCK.json',Path(__file__),Path(T.__file__),
        ROOT/'scripts/self_training_yolo/v3/true_ignore_trainer.py',ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py']
    bindings=[G.binding(p) for p in sources]
    parent=read(C.DOC/'TRAIN_PROTOCOL.json');assert parent['args']==T.ARGS
    source_slots=(ROOT/parent['datasets']['SYN_ONLY']['train_list']['path']).read_text().splitlines()[:512]
    assert len(source_slots)==len(set(source_slots))==512
    val=(C.RAW/'dataset/val.txt').read_text()
    all_bindings=[];datasets={}
    for arm,source_arm in ARMS.items():
        folder=RAW/'dataset'/arm;images=[]
        for r in selected:
            fid=r['id'];p=payload['predictions'][source_arm][fid];c=top(p)
            info=next(d for d in payload['decisions'] if d['arm']==source_arm and d['id']==fid)
            assert info['applied']
            im=cv2.imread(str(ROOT/r['image']['path']));assert im.shape[:2]==(480,640)
            dest=folder/'images'/(fid.replace(':','__')+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
            G.image_write(dest,cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101))
            label=folder/'labels'/(dest.stem+'.txt');text=T.label(c,[480,640]);assert text is not None
            write(label,text)
            arr=np.array(text.split(),float)[5:].reshape(9,3);trusted=arr[:,2]==2
            np.testing.assert_allclose(arr[trusted,:2]*[840,680]-100,np.array(c['keypoints_xy'])[trusted],atol=5e-7)
            images.append(str(dest));all_bindings += [G.binding(dest),G.binding(label),r['image']]
        # Balanced 64 exposures of each image per epoch; same ordering in both arms.
        real_order=np.random.default_rng(9021).permutation(np.repeat(np.arange(8),64))
        slots=source_slots+[images[int(i)] for i in real_order]
        write(folder/'train.txt','\n'.join(slots)+'\n');write(folder/'val.txt',val)
        write(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        datasets[arm]=dict(data=G.binding(folder/'data.yaml'),train_list=G.binding(folder/'train.txt'),real_unique=8,real_slots=512,synthetic_slots=512)
        all_bindings += [G.binding(folder/'val.txt')]
    for p in source_slots+val.splitlines():
        im=Path(p);lab=im.parent.parent/'labels'/im.with_suffix('.txt').name
        if im.parent.name=='val':lab=im.parent.parent.parent/'labels/val'/im.with_suffix('.txt').name
        all_bindings.extend([G.binding(im),G.binding(lab)])
    write(DOC/'PROTOCOL.json',dict(arms=ARMS,args=T.ARGS,initialization=G.binding(C.N.E.R0),datasets=datasets,
        train_indices=INDICES,train_records=selected,eval_records=evalrows,primary_occlusion_ids=primary,
        split='Only this experiment: CAD8 promoted to training; entire CAD session excluded from eval; original global manifests and prior results preserved.',
        labels='Frozen N3/Replay self-occlusion PnP pseudo coordinates only, R0 boxes/confidence/center. No CAD GT coordinates used for training.',
        budget='2 arms, each fixed5epochs/320optimizer updates, last.pt only; same R0 initialization, source replay512, CAD8 repeated512 slots,1:1 ratio.',
        primary='96 externally occluded plastic frames in sessions not used to train Replay teacher; no CAD and no confidence/geometry filter at evaluation.',
        secondary='All nonCAD176 and occluded107, with Replay teacher-overlap session explicitly flagged.',
        causal_limit='Does not isolate PnP contribution from other pseudo correction; no uncorrected-pseudo student in this bounded two-arm run.',
        inference='Student only; no N3/Replay/PnP at evaluation. Same reflect101 pad100,640,conf.001,top1,IoU.5 matching.',
        metrics='Whole-object C2 corner8 PCK10/20 with misses penalized; matched median/P90 and coverage; paired frame changes.',
        selection='No real eval checkpoint selection, early stopping or hyperparameter sweep. Single seed42, reused DEV, not independent confirmation.',
        system_changes=False,temperature_limit=80,auto_promote=False,annotations_modified=False,
        sources=bindings,inputs=all_bindings))
    print('PREPARED',dict(train=8,eval=176,primary_occlusion=96,all_occlusion=sum(r['occlusion'] not in ['none','unknown',''] for r in evalrows)),flush=True)


def check():
    p=read(DOC/'PROTOCOL.json')
    for b in p['sources']+p['inputs']+[p['initialization']]:G.verify(b)
    for d in p['datasets'].values():G.verify(d['data']);G.verify(d['train_list'])
    return p


def train(arm):
    from scripts.self_training_yolo.v3.true_ignore_trainer import TrueIgnorePoseTrainer
    p=check();fit=DOC/f'FIT_{arm}.json'
    if fit.exists():G.verify(read(fit)['checkpoint']);return
    C.N.setup();assert torch.cuda.is_available();print('GPU_START',gpu(),flush=True)
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))//1024
    assert available>=6000
    run_dir=RAW/'runs'/arm;assert not run_dir.exists(),'Preserve incomplete run; do not overwrite'
    trainer=TrueIgnorePoseTrainer(overrides=dict(p['args'],model=str(ROOT/p['initialization']['path']),
        data=str(ROOT/p['datasets'][arm]['data']['path']),project=str(RAW/'runs'),name=arm,exist_ok=False))
    steps=[];epochs=[];start=time.monotonic()
    def on_start(t):
        src=torch.load(C.N.E.R0,map_location='cpu',weights_only=False)['model'].float().state_dict();dst=t.model.state_dict()
        assert src.keys()==dst.keys()
        assert all(torch.equal(v,dst[k].detach().cpu()) for k,v in src.items())
        t.optimizer.register_step_post_hook(lambda opt,args,kwargs:steps.append(len(steps)+1))
        print('R0_INITIALIZATION_EXACT',arm,flush=True)
    def epoch(t):
        status=gpu();epochs.append(dict(epoch=t.epoch+1,steps=len(steps),status=status))
        print('CAD8_PROGRESS',arm,t.epoch+1,len(steps),status,flush=True)
    def batch(t):
        if len(steps)%32==0:gpu()
    trainer.add_callback('on_train_start',on_start);trainer.add_callback('on_train_epoch_end',epoch);trainer.add_callback('on_train_batch_end',batch)
    trainer.train();ck=run_dir/'weights/last.pt'
    rows=list(csv.DictReader((run_dir/'results.csv').open()))
    assert len(rows)==5 and len(steps)==320 and ck.exists(),(len(rows),len(steps))
    write(fit,dict(complete=True,checkpoint=G.binding(ck),protocol=G.binding(DOC/'PROTOCOL.json'),steps=len(steps),epochs=epochs,seconds=time.monotonic()-start))
    del trainer;gc.collect();torch.cuda.empty_cache();check()
    print('CAD8_TRAIN_COMPLETE',arm,flush=True)


@torch.no_grad()
def infer(arm):
    from ultralytics import YOLO
    p=check();dest=RAW/f'PREDICTIONS_{arm}.json';fit=read(DOC/f'FIT_{arm}.json');G.verify(fit['checkpoint'])
    if dest.exists():assert read(dest)['checkpoint']==fit['checkpoint'];return
    C.N.setup();gpu();assert torch.cuda.is_available()
    model=YOLO(str(ROOT/fit['checkpoint']['path']),task='pose');rows=[]
    for i,r in enumerate(p['eval_records']):
        im=cv2.imread(str(ROOT/r['image']['path']));canvas=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
        torch.backends.cudnn.allow_tf32=True
        out=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
        cc=[]
        if out.boxes is not None and len(out.boxes):
            for j,(box,score,xy,conf) in enumerate(zip(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy(),out.keypoints.xy.cpu().numpy(),out.keypoints.conf.cpu().numpy())):
                cc.append(dict(candidate_index=j,score=float(score),box_xyxy=(box-100).astype(float).tolist(),keypoints_xy=(xy-100).astype(float).tolist(),keypoints_conf=conf.astype(float).tolist()))
        rows.append(dict(id=r['id'],raw_hw=list(im.shape[:2]),prediction=dict(candidates=cc,selected_index=int(np.argmax([c['score'] for c in cc])) if cc else None)))
        if (i+1)%50==0:print('CAD8_EVAL',arm,i+1,gpu(),flush=True)
    write(dest,dict(checkpoint=fit['checkpoint'],records=rows,GT_input=False,no_refinement=True))
    del model;gc.collect();torch.cuda.empty_cache()


def score():
    from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
    from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M
    p=check();ids={r['id'] for r in p['eval_records']};meta={r['id']:r for r in p['eval_records']}
    files={'R0':C.RAW/'EVAL_PREDICTIONS_R0.json',**{a:RAW/f'PREDICTIONS_{a}.json' for a in ARMS}}
    write(DOC/'PREDICTIONS_LOCK.json',{a:G.binding(f) for a,f in files.items()})
    pe,pop=O.population_metadata();targets={}
    for item,_ in pop:
        if item.frame_id in ids:
            t=pe.E._legacy_forbidden_target(item)
            targets[item.frame_id]=(np.array(t.keypoints_xy),np.array(t.box_xyxy),np.array(t.keypoint_supervision_mask))
    assert set(targets)==ids
    perms=next(o['permutations'] for o in read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if o['object_type']==C.TYPES['PLASTIC'])
    allmetrics={}
    for arm,path in files.items():
        payload=read(path);G.verify(payload['checkpoint']);rows=[]
        byid={r['id']:r for r in payload['records']}
        for r in p['eval_records']:
            fid=r['id'];raw=byid[fid];c=top(raw['prediction']);gt,box,valid=targets[fid]
            matched=c is not None and O.iou(c['box_xyxy'],box)>=.5
            q=c['keypoints_xy'] if c else np.full((9,2),np.nan)
            rows.append(dict(id=fid,session=r['session'],**M.measure(q,gt,valid,perms,raw['raw_hw'],matched,c is not None)))
        allmetrics[arm]=rows
    old={r['id']:r for r in read(C.RAW/'EVAL_METRICS.json')['R0']}
    for r in allmetrics['R0']:
        assert r['matched']==old[r['id']]['matched'];np.testing.assert_allclose(r['errors'],old[r['id']]['errors'],atol=1e-8)
    subsets=dict(PRIMARY_OCC96=set(p['primary_occlusion_ids']),ALL_OCC107={r['id'] for r in p['eval_records'] if r['occlusion'] not in ['none','unknown','']},ALL_NONCAD176=ids,
        CLEAN69={r['id'] for r in p['eval_records'] if r['occlusion']=='none'})
    report={};damage={}
    for name,subset in subsets.items():
        selected={arm:[r for r in rows if r['id'] in subset] for arm,rows in allmetrics.items()}
        report[name]={a:M.summary(rows) for a,rows in selected.items()}
        damage[name]={a:M.damage(selected['R0'],selected[a]) for a in ARMS}
    write(RAW/'METRICS.json',allmetrics)
    write(DOC/'RESULTS.json',dict(summary=report,damage=damage,baseline_parity=True,no_eval_refinement=True,
        caveat='Single seed,8 training images, reused DEV, not independent confirmation; no causal isolation of PnP.',fits={a:read(DOC/f'FIT_{a}.json') for a in ARMS}))
    lines=['# CAD8 pseudo fine-tuning → non-CAD occlusion','',
        'Training: CAD 2,6,7,8,9,10,11,12. Frozen pseudo labels only; no manual CAD targets. Entire CAD excluded from evaluation.',
        'Two arms from R0; 512 synthetic replay +512 balanced CAD slots,5epochs,320updates,seed42. Standalone student at inference.',
        'Primary excludes the Replay teacher training session (plastic_night_01). Condition tags locked before training. Reused DEV; not independent confirmation.','']
    for name,arms in report.items():
        lines += ['## '+name,'','| Arm | Median px | P90 px | PCK10 | PCK20 | Matched |','|---|---:|---:|---:|---:|---:|']
        for a,s in arms.items():lines.append(f'| {a} | {s["matched_pooled_corner8_median_px"]:.3f} | {s["matched_pooled_corner8_P90_px"]:.3f} | {100*s["PCK"]["10"]:.2f}% | {100*s["PCK"]["20"]:.2f}% | {s["matched"]}/{s["total_frames"]} |')
        lines+=['']
    write(DOC/'RESULTS.md','\n'.join(lines)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','all','score',*ARMS]);args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='score':score()
    elif args.stage=='all':
        for arm in ARMS:train(arm);infer(arm)
        score()
    else:train(args.stage);infer(args.stage)
