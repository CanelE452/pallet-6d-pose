"""Frozen V18 proposal, learned per-point retention; isolated artifacts."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_vector_v18'))
import context18 as Q
from context18 import ROOT,C,V,P,read,write,sha,digest,setup,subset,SEEDS,CONDITIONS,TRAIN_CONDITIONS,KEYS,PREP,E,COMPARE,manifest,targets,repcache,load
from model18 import VectorRefiner
import numpy as np
import torch
import json,time
RUN=ROOT/'data/pallet/results/pallet_retention_v19'
ARMS=('local_gate','area_gate','line_gate')

def proposal_training(seed,device='cuda:0'):
    O,rep,gt,valid,ids,kind=Q.training(seed,device)
    model=VectorRefiner(seed).to(device).eval().requires_grad_(False);model.load_state_dict(torch.load(Q.RUN/'models'/f'line_vector_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict']);parts=[]
    with torch.no_grad():
        for first in range(0,len(gt),32):
            sl=slice(first,first+32);parts.append(model(subset(O,sl),subset(rep,sl),'line_vector')['points'])
    rep['proposal_delta']=torch.cat(parts)-rep['base'];del model
    return O,rep,gt,valid,ids,kind

def eval_inputs(split,condition,seed,device='cpu'):
    m,O=load(split,condition,device);rep=repcache(split,condition,seed,device=device);p=np.load(Q.RUN/'predictions'/split/condition/f'line_vector_seed{seed}.npz');det=p['detected'];assert p['ids'].tolist()==[r['id'] for r in m['rows']]
    assert np.array_equal(p['base'][det],rep['base'].cpu().numpy());rep['proposal_delta']=torch.tensor(p['points'][det]-p['base'][det],device=device)
    return m,O,rep

def save(split,condition,name,m,O,out):
    folder=RUN/'predictions'/split/condition;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    out=dict(out,original_points=O['points'])
    for k in ('original_points','points','base','proposal_delta','alpha','coverage'):
        v=out[k].detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(folder/f'{name}.npz',**data)
