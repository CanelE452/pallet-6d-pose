"""V16 isolated candidate-specific line evidence experiment."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_line_evidence_v16'))
import context16 as P
from context16 import V
from context16 import ROOT,C,read,write,sha,digest,setup,subset,SEEDS,CONDITIONS,TRAIN_CONDITIONS,NEW_CONDITIONS,KEYS,PREP,E,COMPARE,manifest,targets,repcache,candidates
import torch
import numpy as np
import json
import time
RUN=ROOT/'data/pallet/results/pallet_detail_v17_fixed'
ARMS=('low_none','high_none','low_proxy','high_proxy','low_mask','high_mask')

def save(split,condition,name,m,O,points,extra=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True)
    det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)

def load(split,condition,device='cpu'):
    m,O=P.load(split,condition,device)
    x=torch.load(RUN/'rgb'/split/f'{condition}.pt',map_location='cpu',weights_only=True)
    assert x['ids']==[r['id'] for r in m['rows'] if r['detected']]
    O['rgb']=x['rgb'].to(device).float()/255.-.5
    return m,O

def training(seed,device='cuda:0'):
    O,rep,gt,valid,ids,kind=P.training(seed,device);parts=[];rgbids=[]
    for c in TRAIN_CONDITIONS:
        x=torch.load(RUN/'rgb/heldout'/f'{c}.pt',weights_only=True);parts.append(x['rgb']);rgbids.extend([i+'::'+c for i in x['ids']])
    assert rgbids==ids;O['rgb']=torch.cat(parts).to(device).float()/255.-.5
    return O,rep,gt,valid,ids,kind
