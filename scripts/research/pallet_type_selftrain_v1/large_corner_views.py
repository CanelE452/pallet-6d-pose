"""Large-error candidate recovery: frozen R0, GT-free whole-pose view medoid.

Inference and selection never open annotation contents. GT is available only
after all194 predictions/decisions are frozen; oracle results are diagnostics.
"""
import argparse
import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import recovery_common as R
from .pseudo import top,PERM
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as EM
from scripts.research.pallet_posefix_large_error_v1.evaluate import recovery_damage
from scripts.self_training_yolo.v3 import true_ignore_trainer  # checkpoint class

C=R.C
DOC=R.BASE_DOC/'large_corner_recovery_v1'
RAW=R.BASE_RAW/'large_corner_recovery_v1'
PHASE='views'
VARIANTS=['R0','FULL960','FULL1280','CROP125','CROP175','FLIP640']
ARMS=VARIANTS+['CONSENSUS']


@contextmanager
def roots():
    with patch.object(R,'DOC',DOC),patch.object(R,'RAW',RAW):yield


def transform(image,box,name):
    h,w=image.shape[:2]
    if name=='FLIP640':return cv2.flip(image,1),dict(offset=[0,0],flip=True,original_width=w),640
    if name.startswith('CROP'):
        factor={'CROP125':1.25,'CROP175':1.75}[name]
        box=np.asarray(box,float);center=(box[:2]+box[2:])/2;size=np.maximum(box[2:]-box[:2],16)*factor
        lo=np.maximum(np.floor(center-size/2),[0,0]).astype(int)
        hi=np.minimum(np.ceil(center+size/2),[w,h]).astype(int)
        assert np.all(hi>lo)
        return image[lo[1]:hi[1],lo[0]:hi[0]].copy(),dict(offset=lo.tolist(),flip=False,original_width=w),640
    return image.copy(),dict(offset=[0,0],flip=False,original_width=w),{'R0':640,'FULL960':960,'FULL1280':1280}[name]


def restore(candidate,meta):
    out=copy.deepcopy(candidate);q=np.array(out['keypoints_xy'],float);box=np.array(out['box_xyxy'],float)
    if meta['flip']:
        q[:,0]=meta['original_width']-1-q[:,0];q=q[PERM]
        box=box[[2,1,0,3]];box[[0,2]]=meta['original_width']-1-box[[0,2]]
        out['keypoints_conf']=np.asarray(out['keypoints_conf'])[PERM].tolist()
    offset=np.array(meta['offset']);q+=offset;box+=np.tile(offset,2)
    out['keypoints_xy']=q.tolist();out['box_xyxy']=box.tolist();return out


def pose_distance(a,b,perms,diagonal):
    a=np.asarray(a,float);b=np.asarray(b,float)
    assert a.shape==b.shape==(9,2) and diagonal>0
    return min(float(np.sqrt(np.mean(np.sum((a[:8]-b[np.array(p)[:8]])**2,axis=1))))/diagonal for p in perms)


def choose_medoid(candidates,perms,box):
    names=[v for v in VARIANTS if candidates.get(v) is not None]
    assert names and names[0]=='R0'
    diag=float(np.linalg.norm(np.asarray(box)[2:]-np.asarray(box)[:2]));assert diag>0
    matrix=np.zeros((len(names),len(names)),float)
    for i,a in enumerate(names):
        for j,b in enumerate(names[:i]):
            matrix[i,j]=matrix[j,i]=pose_distance(candidates[a]['keypoints_xy'],candidates[b]['keypoints_xy'],perms,diag)
    # Whole pose from an actual model prediction. No coordinate averaging or C4 alignment.
    costs=matrix.mean(axis=1);index=int(np.flatnonzero(costs<=costs.min()+1e-12)[0])
    return names[index],dict(names=names,costs=costs.tolist(),distance_matrix=matrix.tolist(),tie_break='fixed variant order within normalized1e-12, R0 first')


def replace_corners(original,candidate):
    out=copy.deepcopy(original)
    if candidate is not None:
        selected=top(out);assert selected is not None
        selected['keypoints_xy'][:8]=copy.deepcopy(candidate['keypoints_xy'][:8])
    return out


def prepare():
    sources=[C.bound(__file__),C.bound(Path(__file__).with_name('test_large_corner_views.py')),
        C.bound(R.__file__),C.bound(R.BASE_DOC/'EVAL_PROTOCOL.json'),C.bound(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json'),
        C.bound(C.N.E.R0),C.bound(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')]
    protocol=dict(objective='Recover genuinely large corner errors, not just improve a median.',variants=VARIANTS,
        population='All194 ordinary-plastic existing DEV images; no frame filtering; R0 baseline boxes/scores/confidence/center are preserved.',
        generator='Same frozen R0 at640/960/1280; predicted-box-centered crops1.25/1.75 at640; horizontal flip at640. Reflect101 border100 on each transformed input.',
        matching='Select highest-IoU candidate to original predicted box, require IoU>=.5 and score>=.25; missing candidate falls back to original. No GT box used.',
        flip='Inverse pixel-center x=original_width-1-x followed by fixed native reflection permutation. No learned reassignment or quarter-turn alignment.',
        decision='Primary CONSENSUS is unweighted whole-pose medoid over available views, C2 quotient RMS8 distance normalized by original predicted box diagonal; fixed-order tie break within1e-12 numerical tolerance. Emit native prediction of selected view unchanged except original center/box/confidence retained.',
        scope='A candidate-generation/selection stage, not yet new self-training. Individual views are predeclared secondary controls, not best-on-eval selection.',
        primary_metrics='Canonical-identity aligned baseline>20px to corrected<=10px recovery, baseline<5px to corrected>10px damage; matched-only and all-frame counts. Report full194 PCK20,median/P90,pose.',
        followup_gate='Primary consensus must recover at least5 previously>20px corners in at least3 images, damage no more than1percent of baseline<5px corners, and not reduce full194 PCK20. This only permits a follow-up training experiment, not goal completion.',
        oracle='GT best complete view per frame by mean symmetry8 error is diagnostic only. No oracle per-corner splicing, no oracle output training or deployment.',
        evaluation='Official plastic C2 remains unchanged. Reused DEV, not independent confirmation. No eval annotations passed to inference or selection. No GT-derived indices used at inference.',
        constraints='No depth/CAD, no new geometry/shape/X filters, no new real GT training, no replacement of existing final models, no GPU/system changes.',
        sources=sources)
    for b in sources:C.verify(b)
    C.freeze(DOC/PHASE/'PROTOCOL.json',protocol)
    with roots():R.evaluation_protocol(PHASE,ARMS,sources+[C.bound(DOC/PHASE/'PROTOCOL.json')])
    print('LARGE_VIEWS_PROTOCOL_LOCKED',flush=True)


@torch.no_grad()
def infer():
    protocol=C.read(DOC/PHASE/'PROTOCOL.json')
    for binding in protocol['sources']:C.verify(binding)
    C.N.setup();torch.set_num_interop_threads(1);assert torch.cuda.is_available();gpu=C.N.E.gpu()
    model=YOLO(str(C.N.E.R0),task='pose');torch.backends.cudnn.allow_tf32=True
    records=C.read(DOC/PHASE/'EVAL_PROTOCOL.json')['records'];assert len(records)==194
    saved={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    groups={r['object_type']:r for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    perms=groups[C.TYPES['PLASTIC']]['permutations'];assert len(perms)==2
    frames=[];start=time.monotonic()
    for i,record in enumerate(records):
        destination=RAW/PHASE/'frames'/f'{i:03d}.json'
        if destination.exists():
            row=C.read(destination);assert row['id']==record['id'] and row['protocol_sha256']==C.sha(DOC/PHASE/'PROTOCOL.json')
        else:
            original=saved[record['id']];anchor=top(original['prediction']);assert anchor is not None
            C.verify(record['image']);im=cv2.imread(str(C.ROOT/record['image']['path']));assert im is not None
            views={};metadata={}
            for name in VARIANTS:
                view,meta,size=transform(im,anchor['box_xyxy'],name)
                padded=cv2.copyMakeBorder(view,100,100,100,100,cv2.BORDER_REFLECT_101)
                result=model.predict(padded,conf=.001,imgsz=size,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
                candidates=[]
                if result.boxes is not None and len(result.boxes):
                    boxes=result.boxes.xyxy.cpu().numpy()-100;scores=result.boxes.conf.cpu().numpy()
                    coords=result.keypoints.xy.cpu().numpy()-100;conf=result.keypoints.conf.cpu().numpy()
                    for j in range(len(scores)):
                        candidate=dict(candidate_index=j,score=float(scores[j]),box_xyxy=boxes[j].tolist(),keypoints_xy=coords[j].tolist(),keypoints_conf=conf[j].tolist())
                        candidates.append(restore(candidate,meta))
                if name=='R0':
                    assert candidates
                    candidate=max(candidates,key=lambda p:p['score'])
                    delta=float(np.max(np.abs(np.array(candidate['keypoints_xy'])-anchor['keypoints_xy'])))
                    assert delta<=.001 and abs(candidate['score']-anchor['score'])<=1e-6,(record['id'],delta)
                    # Preserve exact original identity view after parity check.
                    candidate=copy.deepcopy(anchor);overlap=1.
                else:
                    valid=[p for p in candidates if p['score']>=.25 and np.isfinite(p['keypoints_xy']).all()]
                    candidate=max(valid,key=lambda p:R.E.O.iou(p['box_xyxy'],anchor['box_xyxy'])) if valid else None
                    overlap=R.E.O.iou(candidate['box_xyxy'],anchor['box_xyxy']) if candidate is not None else None
                    if overlap is None or overlap<.5:candidate=None
                views[name]=candidate;metadata[name]=dict(transform=meta,imgsz=size,matched_predicted_box_iou=overlap)
            selected,consensus=choose_medoid(views,perms,anchor['box_xyxy'])
            row=dict(id=record['id'],image=record['image'],raw_hw=original['raw_hw'],views=views,metadata=metadata,
                selected=selected,consensus=consensus,identity_parity_px=delta,
                protocol_sha256=C.sha(DOC/PHASE/'PROTOCOL.json'))
            C.freeze(destination,row)
        frames.append(row)
        if (i+1)%25==0 or i+1==194:print('LARGE_VIEWS',i+1,'/194',round(time.monotonic()-start,1),'sec',C.N.E.gpu(),flush=True)
    for arm in ARMS:
        rows=[]
        for f in frames:
            original=saved[f['id']];view=f['views'][f['selected'] if arm=='CONSENSUS' else arm]
            rows.append(dict(id=f['id'],kind='PLASTIC',raw_hw=f['raw_hw'],prediction=replace_corners(original['prediction'],view)))
        C.freeze(RAW/PHASE/f'EVAL_PREDICTIONS_{arm}.json',dict(arm=arm,complete=True,checkpoint=C.bound(C.N.E.R0),
            protocol=C.bound(DOC/PHASE/'PROTOCOL.json'),records=rows,GT_free_decisions=True))
    for b in protocol['sources']:C.verify(b)
    C.freeze(RAW/PHASE/'INFERENCE.json',frames)
    C.freeze(DOC/PHASE/'OUTPUTS_LOCK.json',dict(complete=True,all194=True,GT_read_during_inference=False,initial_gpu=gpu,
        artifacts=[C.bound(RAW/PHASE/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+[C.bound(RAW/PHASE/'INFERENCE.json')]))
    print('LARGE_VIEWS_ALL_OUTPUTS_LOCKED',flush=True)


def score():
    lock=C.read(DOC/PHASE/'OUTPUTS_LOCK.json')
    for b in lock['artifacts']:C.verify(b)
    results={}
    with roots():
        for arm in ARMS:
            path=RAW/PHASE/f'SCREEN_{arm}.json'
            results[arm]=C.read(path) if path.exists() else R.score(PHASE,arm,False)
    baseline=results['R0']['metrics'];summary={}
    for arm,r in results.items():
        summary[arm]=dict(symmetry=r['symmetry'],pose=r['pose'],damage=EM.damage(baseline,r['metrics']),
            hard_recovery_all=recovery_damage(baseline,r['metrics']),hard_recovery_matched=recovery_damage(baseline,r['metrics'],True))
    oracle=[]
    for i,row in enumerate(baseline):
        candidates=[(a,results[a]['metrics'][i]) for a in VARIANTS]
        assert all(r['id']==row['id'] for a,r in candidates)
        arm,best=min(candidates,key=lambda ar:ar[1]['frame_mean_px'])
        oracle.append(dict(**best,diagnostic_view=arm))
    r=results['CONSENSUS'];d=summary['CONSENSUS']['hard_recovery_matched']
    recovered_frames=sum(any(b is not None and b>20 and n<=10 for b,n in zip(old['canonical_errors'],new['canonical_errors']))
        for old,new in zip(baseline,r['metrics']) if old['matched'])
    flags=dict(at_least5_large_corner_recoveries=d['recovered']>=5,at_least3_recovered_frames=recovered_frames>=3,
        damage_rate_at_most1percent=d['damage_rate']<=.01,full194_PCK20_preserved=r['symmetry']['PCK']['20']>=results['R0']['symmetry']['PCK']['20'])
    output=dict(arms=summary,predeclared_followup_flags=flags,followup_permitted=all(flags.values()),recovered_frames=recovered_frames,
        oracle_diagnostic_only=dict(symmetry=EM.summary(oracle),hard_recovery=recovery_damage(baseline,oracle,True),rows=oracle),
        goal_complete=False,independent_confirmation=False,new_selftraining_done=False,
        sources=[C.bound(DOC/PHASE/'PROTOCOL.json'),C.bound(DOC/PHASE/'OUTPUTS_LOCK.json')])
    C.freeze(RAW/PHASE/'SUMMARY.json',output)
    lines=['# Large-corner recovery: multi-view R0 candidates','',
        'All194 ordinary-plastic DEV images, official C2 unchanged. Only first8corner predictions can change; original boxes/scores/confidences/center preserved. No evaluation-image rejection. Primary CONSENSUS selected before GT by whole-pose medoid, not oracle.','',
        '| Arm | PCK20 % | median8 px | P90_8 px | matched hard recovered / hard | good corners damaged / good | IoU3D |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for a,m in summary.items():
        d=m['hard_recovery_matched'];s=m['symmetry']
        lines.append(f"| {a} | {100*s['PCK']['20']:.3f} | {s['matched_pooled_corner8_median_px']:.3f} | {s['matched_pooled_corner8_P90_px']:.3f} | {d['recovered']}/{d['hard']} | {d['damaged']}/{d['good']} | {m['pose']['iou3d_median']:.5f} |")
    lines+=['','Follow-up gate (not goal completion): '+json.dumps(flags),'',
        'GT best-whole-view diagnostic only: '+json.dumps(output['oracle_diagnostic_only']['hard_recovery']),
        '','GT best view is never used for model output, pseudo-label generation, training, selection-rule tuning, or a performance claim. C4 is never treated as valid ordinary-plastic symmetry. Existing repeated DEV is not an independent test.','']
    C.write_text(RAW/PHASE/'REPORT.md','\n'.join(lines));print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','infer','score']);a=p.parse_args();globals()[a.action]()
