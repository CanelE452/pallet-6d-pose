"""Immutable parent observations and isolated candidate-scoring experiment."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT.parent/'pallet_observability_v12'))
import context12 as P
from context12 import read,write,sha,digest,setup,subset,import_file,local,base_points,KEYS,CONDITIONS,SEEDS
from router12 import Router
RUN=ROOT/'data/pallet/results/pallet_candidate_gain_v13'
V12=P.RUN
V11=P.V11
ARMS=('constant','geometry','image')
ALPHAS=(0.,.5,1.)

def training_observations(device='cpu'):
    old,O=P.observations('train','cpu');parts=[{k:O[k] for k in KEYS}];ids=[r['id']+'::clean' for r in old.records]
    for condition in CONDITIONS[1:]:
        m,o=P.load_corner_train() if condition=='corner20' else P.C.load_observations('train',condition)
        parts.append(o);ids.extend([r['id']+'::'+condition for r in m['rows'] if r['detected']])
    O={k:torch.cat([p[k] for p in parts]).to(device) for k in KEYS}
    assert len(O['points'])==len(ids)
    return ids,O

def training_targets(device='cpu'):
    rows=read(P.V10/'targets_train.json')['rows'];allrows=list(rows)
    for condition in CONDITIONS[1:]:
        if condition=='corner20':m=read(V12/'cache'/'train_corner20'/'manifest.json');path=V12/'targets_train_corner20.json'
        else:m=read(V11/'cache'/'train'/condition/'manifest.json');path=V11/'targets'/f'train_{condition}.json'
        t=read(path)['rows'];assert [r['id'] for r in m['rows']]==[r['id'] for r in t]
        allrows.extend([r for r,c in zip(t,m['rows']) if c['detected']])
    return torch.tensor([r['points'] for r in allrows],dtype=torch.float32,device=device),torch.tensor([r['valid'] for r in allrows],device=device)

def parent_model(seed,arm,device='cuda:0'):
    model=Router(seed);ck=V12/'models'/f'{arm}_seed{seed}'/'final.pt'
    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict'])
    return model.to(device).eval().requires_grad_(False)

def candidates(rep):
    alpha=rep['base'].new_tensor(ALPHAS)
    return rep['base'][:,:,None]+alpha[None,None,:,None]*(rep['full']-rep['base'])[:,:,None]

def parent_cache(split,condition,seed,device='cpu'):
    return {k:v.to(device) for k,v in torch.load(RUN/'parents'/split/condition/f'seed{seed}.pt',map_location='cpu',weights_only=True).items()}

def save_predictions(split,condition,name,manifest,O,points,extras=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True)
    det=np.array([r['detected'] for r in manifest['rows']]);data=dict(ids=np.array([r['id'] for r in manifest['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extras or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
