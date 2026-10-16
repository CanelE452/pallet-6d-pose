"""V16 isolated candidate-specific line evidence experiment."""
from pathlib import Path
import sys
KIT=Path(__file__).resolve().parent
sys.path.insert(0,str(KIT.parent/'pallet_stability_v15'))
import context15 as V
from context15 import ROOT,C,read,write,sha,digest,setup,subset,SEEDS,CONDITIONS,TRAIN_CONDITIONS,NEW_CONDITIONS,KEYS,PREP,E,COMPARE,load,manifest,targets,repcache,candidates,training
import torch
import numpy as np
import json
import time
RUN=ROOT/'data/pallet/results/pallet_line_evidence_v16'
ARMS=('control','strip','role','balanced_role','median_role')

def save(split,condition,name,m,O,points,extra=None):
    d=RUN/'predictions'/split/condition;d.mkdir(parents=True,exist_ok=True)
    det=np.array([r['detected'] for r in m['rows']]);data=dict(ids=np.array([r['id'] for r in m['rows']]),detected=det)
    for k,v in dict(points=points,original_points=O['points'],**(extra or {})).items():
        v=v.detach().cpu().numpy();a=np.full((len(det),)+v.shape[1:],np.nan,dtype=np.float32);a[det]=v;data[k]=a
    np.savez_compressed(d/f'{name}.npz',**data)
