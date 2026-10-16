"""V12 isolated experiment, reusing immutable V10/V11 observations."""
from pathlib import Path
import sys
import importlib.util
import json
import numpy as np
import torch

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT.parent/'pallet_optional_lines_v11'))
import common as C
from common import read,write,sha,digest,setup,subset,local,V10,EXPORT,observations
RUN=ROOT/'data/pallet/results/pallet_observability_v12'
V11=C.RUN
CONDITIONS=(*C.CONDITIONS,'corner20')
ARMS=('scalar_gate','learned_gate','gated_amodal','always_amodal')
SEEDS=(1,2,3)
KEYS=('features','content','box','K','dims','image_hw','points')

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def load_corner_train(device='cpu'):
    d=RUN/'cache'/'train_corner20';m=read(d/'manifest.json');chunks=[torch.load(d/p,map_location='cpu',weights_only=True) for p in m['chunks']]
    return m,{k:torch.cat([c[k] for c in chunks]).to(device) for k in KEYS}

def target_type(rows):
    result=[]
    for r in rows:
        result.append(np.where(r.get('outside',[False]*8),2,np.where(r.get('hidden',[False]*8),1,0)))
    return torch.tensor(np.array(result),dtype=torch.long)

def base_points(model,O):
    # V10 local mode ignores line context. Dummy lines provide shape only.
    lines=O['points'].new_tensor([[[1.,0.,0.],[0.,1.,0.]]]).expand(len(O['points']),-1,-1)
    mass=torch.ones(len(lines),2,device=lines.device)
    return model(O,lines,mass,'local')['points']
