"""V20 isolated 2x2 causal repair plus matched area/local controls."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_retention_v19'))
import context19 as T
from context19 import ROOT,C,V,P,Q,read,write,sha,digest,setup,subset,SEEDS,CONDITIONS,TRAIN_CONDITIONS,KEYS,PREP,E,COMPARE,manifest,targets,repcache,load,proposal_training,eval_inputs
import torch,numpy as np,json,time
RUN=ROOT/'data/pallet/results/pallet_cause_v20'
ARMS=('raw_weighted','raw_equal','unit_weighted','unit_equal','unit_equal_area','unit_equal_local')

def save(split,condition,name,m,O,out):
    folder=RUN/'predictions'/split/condition;folder.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(original_points=O['points'],**{k:out[k] for k in ('points','base','proposal_delta','alpha','coverage')}).items():
        z=v.detach().cpu().numpy();a=np.full((len(det),)+z.shape[1:],np.nan,dtype=np.float32);a[det]=z;data[k]=a
    np.savez_compressed(folder/f'{name}.npz',**data)
