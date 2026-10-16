"""V14: immutable parents, separated candidate-evaluator training and fixed evaluation."""
from pathlib import Path
import sys
import time
import numpy as np
import torch
import json
KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT.parent/'pallet_candidate_gain_v13'))
import context13 as C13
from context13 import P,read,write,sha,digest,setup,subset,import_file,local,base_points,KEYS,SEEDS,CONDITIONS
from optional_model import OptionalRefiner,predict_lines,donor_lines
RUN=ROOT/'data/pallet/results/pallet_final_fusion_v14'
V11=C13.V11;V12=C13.V12;V13=C13.RUN;V10=P.V10
ARMS=('insample_fusion','heldout_points','heldout_fusion')
E=import_file('v14_eval_compat',KIT.parent/'pallet_optional_lines_v11/evaluate.py')
COMPARE=import_file('v14_compare_compat',KIT.parent/'pallet_observability_v12/eval12.py').compare
PREP=import_file('v14_prep_compat',KIT.parent/'pallet_optional_lines_v11/prepare.py')

def load(split,condition,device='cpu'):
    if split in ('real','synth_val'):return P.C.load_observations(split,condition,device)
    d=RUN/'cache'/split/condition;m=read(d/'manifest.json')
    parts=[torch.load(d/p,map_location='cpu',weights_only=True) for p in m['chunks']]
    return m,{k:torch.cat([p[k] for p in parts]).to(device) for k in KEYS}

def target_rows(split,condition):
    d=V11 if split in ('real','synth_val') else RUN
    return read(d/'targets'/f'{split}_{condition}.json')['rows']

def repcache(split,condition,seed,variant='actual',device='cpu'):
    return {k:v.to(device) for k,v in torch.load(RUN/'parents'/split/condition/f'seed{seed}_{variant}.pt',map_location='cpu',weights_only=True).items()}

def candidates(rep):
    b=rep['base'];a=rep['amodal'];l=rep['line_delta']
    return torch.stack((b,(b+a)*.5,a,b+.5*l,b+l),2)

def save(split,condition,name,m,O,points,extra=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True)
    det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)

def training(split,seed,device='cuda:0'):
    obs=[];reps=[];targets=[];ids=[]
    for c in CONDITIONS:
        m,O=load(split,c);obs.append(O);reps.append(repcache(split,c,seed))
        t=target_rows(split,c);assert [r['id'] for r in t]==[r['id'] for r in m['rows']]
        targets.extend([r for r,rec in zip(t,m['rows']) if rec['detected']]);ids.extend([r['id']+'::'+c for r in m['rows'] if r['detected']])
    O={k:torch.cat([p[k] for p in obs]).to(device) for k in KEYS}
    rep={k:torch.cat([p[k] for p in reps]).to(device) for k in reps[0]}
    gt=torch.tensor([r['points'] for r in targets],dtype=torch.float32,device=device);valid=torch.tensor([r['valid'] for r in targets],device=device)
    return O,rep,gt,valid,ids
