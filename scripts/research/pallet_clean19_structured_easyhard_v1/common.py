"""Immutable experiment namespace; no historical writes."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import cv2
import numpy as np
import torch
from scripts.research import pallet_clean19_student_v1 as H

ROOT=H.ROOT
HERE=Path(__file__).resolve().parent
NAME='pallet_clean19_structured_easyhard_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
read=H.P.read
bind=H.C.bound
verify=H.C.verify
save=H.save
MATERIALS=('PLASTIC','WOOD')
ARMS=('S0','S1','S2')


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def setup():
    H.P.N.setup();torch.set_num_interop_threads(1)


def guard():
    g=H.P.N.E.gpu()
    assert not g['foreign_compute'],g
    line=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True)
    assert int(line.strip())<80,g
    assert torch.cuda.is_available()
    return g


def immutable():
    for b in read(DOC/'INPUT_BINDINGS.json')['files']:verify(b)


@torch.no_grad()
def predict(model,image,padding=100):
    canvas=cv2.copyMakeBorder(image,padding,padding,padding,padding,cv2.BORDER_REFLECT_101) if padding else image
    torch.backends.cudnn.allow_tf32=True
    p=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False)[0]
    rows=[]
    if p.boxes is not None:
        for i in range(len(p.boxes)):
            rows.append(dict(candidate_index=i,score=float(p.boxes.conf[i]),box_xyxy=(p.boxes.xyxy[i].cpu().numpy()-padding).tolist(),
                keypoints_xy=(p.keypoints.xy[i].cpu().numpy()-padding).tolist(),keypoints_conf=p.keypoints.conf[i].cpu().tolist()))
    return dict(candidates=rows,selected_index=int(np.argmax([r['score'] for r in rows])) if rows else None)


def summary(rows):
    e=np.array([v for r in rows for v in r['errors']],float)
    return dict(points=len(e),median=float(np.median(e)) if len(e) else None,P90=float(np.quantile(e,.9)) if len(e) else None,
        PCK={str(t):dict(correct=int((e<=t).sum()),total=len(e),fraction=float((e<=t).mean()) if len(e) else None) for t in (5,10,20)},
        greater={str(t):int((e>t).sum()) for t in (10,20,40)})


def native(pred,target,mask,hw):
    c=H.P.C.selected(pred);q=np.full((9,2),np.nan) if c is None else np.array(c['keypoints_xy'])
    err=np.linalg.norm(q-np.array(target),axis=1);err[~np.isfinite(err)]=np.hypot(*hw)
    return dict(errors=err[np.array(mask,bool)].tolist(),detected=c is not None)
