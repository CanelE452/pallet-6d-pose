"""Independent residual-training population; immutable V20 and earlier parents."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_cause_v20'))
import context20 as OLD
from context20 import *
KIT=Path(__file__).resolve().parent
RUN=ROOT/'data/pallet/results/pallet_residual_v21'
ARMS=('unit_old_sum','unit_old_pcgrad','unit_new_sum','unit_new_pcgrad','unit_new_sum_local')

def new_load(split,condition,device='cpu'):
    d=RUN/'cache'/split/condition;m=read(d/'manifest.json')
    parts=[torch.load(d/p,map_location='cpu',weights_only=True) for p in m['chunks']]
    return m,{k:torch.cat([p[k] for p in parts]).to(device) for k in KEYS}

def new_targets(split,condition):return read(RUN/'targets'/f'{split}_{condition}.json')['rows']

def new_rep(split,condition,seed,device='cpu'):
    return {k:v.to(device) for k,v in torch.load(RUN/'parents'/split/condition/f'seed{seed}.pt',map_location='cpu',weights_only=True).items()}

def new_training(seed,device='cuda:0'):
    obs=[];reps=[];ts=[];ids=[]
    for c in TRAIN_CONDITIONS:
        m,o=new_load('new_train',c);obs.append(o);reps.append(new_rep('new_train',c,seed));t=new_targets('new_train',c)
        assert [x['id'] for x in t]==[x['id'] for x in m['rows']]
        ts.extend(x for x,r in zip(t,m['rows']) if r['detected']);ids.extend(r['id']+'::'+c for r in m['rows'] if r['detected'])
    O={k:torch.cat([x[k] for x in obs]).to(device) for k in KEYS};rep={k:torch.cat([x[k] for x in reps]).to(device) for k in reps[0]}
    return O,rep,torch.tensor([x['points'] for x in ts],dtype=torch.float32,device=device),torch.tensor([x['valid'] for x in ts],device=device),ids,C.P.target_type(ts).to(device)

def save21(split,c,name,m,O,out):
    d=RUN/'predictions'/split/c;d.mkdir(parents=True,exist_ok=True);det=np.array([x['detected'] for x in m['rows']]);data=dict(ids=np.array([x['id'] for x in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**{k:out[k] for k in ('points','base','proposal_delta','alpha','coverage')}).items():
        z=v.detach().cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,np.float32);a[det]=z;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
