"""V15 stability experiment; all earlier artifacts remain immutable."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT.parent/'pallet_final_fusion_v14'))
import context14 as C
from context14 import read,write,sha,digest,setup,subset,import_file,SEEDS,KEYS,PREP,E,COMPARE
from network14 import FinalPointLineNetwork
RUN=ROOT/'data/pallet/results/pallet_stability_v15'
V14=C.RUN;V11=C.V11;V12=C.V12
TRAIN_CONDITIONS=C.CONDITIONS
NEW_CONDITIONS=('crop10','crop30')
CONDITIONS=(*TRAIN_CONDITIONS,*NEW_CONDITIONS)
ARMS=('median_old','risk_old','balanced_old','balanced_rich')
REP_KEYS=('base','amodal','context','probabilities','line_delta','line_descriptor','line_weights','lines')

def load(split,condition,device='cpu'):
    if condition not in NEW_CONDITIONS:return C.load(split,condition,device)
    d=RUN/'cache'/split/condition;m=read(d/'manifest.json')
    chunks=[torch.load(d/p,map_location='cpu',weights_only=True) for p in m['chunks']]
    return m,{k:torch.cat([x[k] for x in chunks]).to(device) for k in KEYS}

def manifest(split,condition):
    root=RUN if condition in NEW_CONDITIONS else V11
    return read(root/'cache'/split/condition/'manifest.json')

def targets(split,condition):
    return read(RUN/'targets'/f'{split}_{condition}.json')['rows'] if condition in NEW_CONDITIONS else C.target_rows(split,condition)

def repcache(split,condition,seed,variant='actual',device='cpu'):
    if condition not in NEW_CONDITIONS:r=C.repcache(split,condition,seed,variant,device)
    else:r={k:v.to(device) for k,v in torch.load(RUN/'parents'/split/condition/f'seed{seed}_{variant}.pt',map_location='cpu',weights_only=True).items()}
    return {k:r[k] for k in REP_KEYS}

def candidates(O,rep):
    b=rep['base'];a=rep['amodal'];d=rep['line_delta'];old=torch.stack((b,(b+a)*.5,a,b+.5*d,b+d),2)
    ix=rep['line_weights'][...,:-1].argsort(dim=-1,descending=True,stable=True)[...,:3]
    lines=rep['lines'][:,None].expand(-1,8,-1,-1).gather(2,ix[...,None].expand(-1,-1,-1,3))
    norm=lines[...,:2].norm(dim=-1).clamp_min(1e-8);n=lines[...,:2]/norm[...,None];c=lines[...,2]/norm
    distance=(b[:,:,None]*n).sum(-1)+c;radius=.05*(O['box'][:,2:]-O['box'][:,:2]).norm(dim=-1).clamp_min(1)
    displacement=-distance.clamp(-radius[:,None,None],radius[:,None,None])[...,None]*n
    return torch.cat((old,b[:,:,None]+displacement),2)

def training(seed,device='cuda:0'):
    O,r,gt,valid,ids=C.training('heldout',seed,device);kind=[]
    for condition in TRAIN_CONDITIONS:
        m,_=C.load('heldout',condition);t=C.target_rows('heldout',condition)
        kind.extend([x for x,row in zip(t,m['rows']) if row['detected']])
    return O,{k:r[k] for k in REP_KEYS},gt,valid,ids,C.P.target_type(kind).to(device)

def save(split,condition,name,m,O,points,extra=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
