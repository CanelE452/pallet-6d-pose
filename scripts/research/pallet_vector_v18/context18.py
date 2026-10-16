"""Direct point-vector test; immutable V15/V16 parents and populations."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_line_evidence_v16'))
import context16 as P
from context16 import ROOT,C,V,read,write,sha,digest,setup,subset,SEEDS,CONDITIONS,TRAIN_CONDITIONS,NEW_CONDITIONS,KEYS,PREP,E,COMPARE,manifest,targets,repcache,candidates,load,training
import torch
import numpy as np
import json
import time
RUN=ROOT/'data/pallet/results/pallet_vector_v18'
ARMS=('point_vector','line_vector','point_risk','line_risk')
RADIUS=.15

def save(split,condition,name,m,O,points,extra=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True);det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
