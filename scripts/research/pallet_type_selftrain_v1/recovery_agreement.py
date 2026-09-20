"""Bounded pseudo-only corner-mask test; frozen refined coordinates, no new GT."""
import argparse
import hashlib
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import recovery_common as R
from . import recovery_pose as P
from . import train as T
from .pseudo import top

C=R.C
PHASE='pose_agreement'
DOC=R.DOC/PHASE
RAW=R.RAW/PHASE
TEACHERS=['RAW_LR5','REF_LR5']
C2=[list(range(9)),[5,4,7,6,1,0,3,2,8]]
TAU=.02


def gate(refined, original_valid, teachers):
    """Only a mask; whole-object C2 alignment, never per-corner reassignment."""
    target=np.asarray(refined['keypoints_xy'],float)
    valid=np.asarray(original_valid,bool).copy()
    out=valid.copy();details=[]
    diagonal=max(float(np.linalg.norm(np.asarray(refined['box_xyxy'])[2:]-np.asarray(refined['box_xyxy'])[:2])),1.)
    for pred in teachers:
        if pred is None:
            out[:8]=False;details.append(dict(missing=True));continue
        points=np.asarray(pred['keypoints_xy'],float);conf=np.asarray(pred['keypoints_conf'],float)
        eligible=valid[:8]&np.isfinite(target[:8]).all(-1)
        distances=[]
        for perm in C2:
            q=points[perm]
            d=np.linalg.norm(q[:8]-target[:8],axis=1)
            distances.append(float(np.mean(d[eligible])) if eligible.any() and np.isfinite(d[eligible]).all() else float('inf'))
        branch=int(np.argmin(distances));perm=C2[branch]
        q=points[perm];confidence=conf[perm]
        delta=np.linalg.norm(q[:8]-target[:8],axis=1)/diagonal
        out[:8]&=np.isfinite(q[:8]).all(-1)&np.isfinite(confidence[:8])&(confidence[:8]>=.5)&(delta<=TAU)
        details.append(dict(missing=False,branch=branch,normalized_shift=[float(d) if np.isfinite(d) else None for d in delta]))
    return out,details


def count_control(original_valid, selected, identity):
    original_valid=np.asarray(original_valid,bool);selected=np.asarray(selected,bool)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'big'))
    out=original_valid.copy();out[:8]=False
    candidates=np.flatnonzero(original_valid[:8]);n=int(selected[:8].sum())
    out[rng.choice(candidates,n,replace=False)]=True
    assert out[:8].sum()==selected[:8].sum()
    return out


def masked_label(original,mask):
    # Preserve exact text tokens for every remaining coordinate and box.
    tokens=original.split();assert len(tokens)==32
    for i,keep in enumerate(mask):
        if not keep:tokens[5+3*i:8+3*i]=['0.500000000','0.500000000','1.000000000']
    return ' '.join(tokens)+'\n'


def prepare():
    parent_path=R.DOC/'pose_only/PROTOCOL.json';parent=C.read(parent_path)
    for b in parent['inputs']+parent['sources']+[parent['initialization']]:C.verify(b)
    checkpoints={k:C.read(R.DOC/'pose_only'/f'FIT_{k}.json')['checkpoint'] for k in TEACHERS}
    for b in checkpoints.values():C.verify(b)
    rows=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    assert len(rows)==249
    eval_hash={r['image']['sha256'] for r in C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert not {r['image']['sha256'] for r in rows}&eval_hash
    protocol=dict(teachers=checkpoints,tau_normalized_box_diagonal=TAU,
        sources=[C.bound(parent_path),C.bound(R.BASE_RAW/'PSEUDO_ACCEPTED.json'),C.bound(R.BASE_DOC/'EVAL_PROTOCOL.json'),
                 C.bound(__file__),C.bound(Path(__file__).with_name('test_recovery_agreement.py'))],
        mask='Both fixed RAW_LR5 and REF_LR5 confident>=.5 and within.02 original R0 box diagonal of unchanged Replay target. Whole-object C2 aligned independently, center unchanged. No frame rejection, new geometry/flip/LOO rule, or coordinate replacement.',
        controls='AGREE versus deterministic COUNT mask: same per-image trusted-corner count, random identities; REF_LR5 original unmasked control. Same217 real images, original order/multiplicities and512 synthetic replay.',
        training='Two fresh R0 students; existing pose-only recipe LR1e-5,5epochs,320updates. No checkpoint/threshold sweep or teacher refresh.',
        goal_test='Full194 existing DEV: matched R0>20 to <=10 at least5 corners across3frames, <=1percent R0<5 to >10 damage, PCK20 no worse than R0. Also report versus original REF and COUNT; satisfying this screen is NOT independent confirmation.',
        population='Keep full194 for historical comparability; additionally report previously frozen retained159, never tune to either.',
        limitations='Teachers are correlated and trained on this pseudo pool: agreement is not truth. Existing Replay has historical manual supervision; NO new real annotation or AprilTag input. Known teacher/eval overlap remains exploratory.',
        new_real_labels=False,auto_promote=False)
    C.freeze(DOC/'MASK_PROTOCOL.json',protocol)
    print('MASK_PROTOCOL_FROZEN',flush=True)


@torch.no_grad()
def infer_masks():
    protocol=C.read(DOC/'MASK_PROTOCOL.json')
    for b in protocol['sources']+list(protocol['teachers'].values()):C.verify(b)
    C.N.setup();torch.set_num_interop_threads(1);assert torch.cuda.is_available();C.N.E.gpu()
    torch.backends.cudnn.allow_tf32=True
    models={k:YOLO(str(C.ROOT/b['path']),task='pose') for k,b in protocol['teachers'].items()}
    rows=[r for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    done=[];start=time.monotonic()
    for i,r in enumerate(rows):
        dest=RAW/'masks'/f'{r["id"]}.json'
        if dest.exists():
            entry=C.read(dest);assert entry['protocol_sha256']==C.sha(DOC/'MASK_PROTOCOL.json')
        else:
            C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            canvas=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
            teachers=[]
            for model in models.values():
                p=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
                if p.boxes is None or not len(p.boxes):teachers.append(None);continue
                j=int(p.boxes.conf.argmax())
                box=p.boxes.xyxy[j].cpu().numpy()-100
                # Detector branches are frozen in both students: same physical candidate.
                np.testing.assert_allclose(box,top(r['raw'])['box_xyxy'],atol=.002,rtol=0)
                teachers.append(dict(keypoints_xy=(p.keypoints.xy[j].cpu().numpy()-100).tolist(),keypoints_conf=p.keypoints.conf[j].cpu().numpy().tolist()))
            labels,_=P.paired_labels(r);v=np.array(labels['REF'].split(),float)[5:].reshape(9,3)[:,2]==2
            mask,detail=gate(top(r['refined']),v,teachers);control=count_control(v,mask,r['id'])
            shift=np.linalg.norm(np.asarray(top(r['refined'])['keypoints_xy'])-np.asarray(top(r['raw'])['keypoints_xy']),axis=1)
            entry=dict(id=r['id'],original_valid=v.tolist(),agreement=mask.tolist(),count_control=control.tolist(),
                raw_to_refined_shift_px=shift.tolist(),teachers=teachers,details=detail,protocol_sha256=C.sha(DOC/'MASK_PROTOCOL.json'))
            C.freeze(dest,entry)
        done.append(entry)
        if (i+1)%50==0 or i+1==len(rows):print('AGREEMENT_MASKS',i+1,'/249',round(time.monotonic()-start,1),C.N.E.gpu(),flush=True)
    totals={}
    for key in ['original_valid','agreement','count_control']:
        totals[key]=dict(corners=sum(sum(r[key][:8]) for r in done),
            frames_with_any=sum(any(r[key][:8]) for r in done),
            corrections_over20=sum(sum(v and d>20 for v,d in zip(r[key][:8],r['raw_to_refined_shift_px'][:8])) for r in done))
    C.freeze(DOC/'MASK_COMPLETE.json',dict(totals=totals,records=[C.bound(RAW/'masks'/f'{r["id"]}.json') for r in done],
        protocol=C.bound(DOC/'MASK_PROTOCOL.json'),no_GT_read=True))
    print('MASK_COUNTS',totals,flush=True)


def datasets():
    parent=C.read(R.DOC/'pose_only/PROTOCOL.json');masks=C.read(DOC/'MASK_COMPLETE.json')
    for b in masks['records']+[masks['protocol']]:C.verify(b)
    labels={r['id']:P.paired_labels(r)[0]['REF'] for r in C.read(R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC'}
    order=(C.ROOT/parent['datasets']['REF']['train_list']['path']).read_text().splitlines()
    inputs=[];data={}
    for arm,key in [('AGREE','agreement'),('COUNT','count_control')]:
        folder=RAW/'dataset'/arm;dest_order=[]
        for original in order:
            p=Path(original);dest=folder/'images'/p.name;T.link(p,dest);dest_order.append(str(dest))
            lp=folder/'labels'/p.with_suffix('.txt').name
            if p.stem in labels:
                entry=C.read(RAW/'masks'/f'{p.stem}.json')
                C.write_text(lp,masked_label(labels[p.stem],entry[key]))
            else:T.link(p.parent.parent/'labels'/p.with_suffix('.txt').name,lp)
            inputs.extend([C.bound(dest),C.bound(lp)])
        C.write_text(folder/'train.txt','\n'.join(dest_order)+'\n')
        C.write_text(folder/'val.txt',(R.BASE_RAW/'dataset/val.txt').read_text())
        C.write_text(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        data[arm]=dict(data=C.bound(folder/'data.yaml'),train_list=C.bound(folder/'train.txt'))
    protocol=dict(args=parent['args'],initialization=parent['initialization'],
        arms={k:dict(target=k,lr=1e-5) for k in data},datasets=data,inputs=inputs,
        sources=parent['sources']+[C.bound(DOC/'MASK_PROTOCOL.json'),C.bound(DOC/'MASK_COMPLETE.json'),C.bound(__file__)],
        source_negatives=0,new_real_annotations=False,auto_promote=False)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    R.evaluation_protocol(PHASE,list(data),protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('AGREEMENT_DATASETS_FROZEN',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','masks','datasets','AGREE','COUNT']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='masks':infer_masks()
    elif a.stage=='datasets':datasets()
    else:
        with patch.object(P,'PHASE',PHASE):P.train(a.stage)
