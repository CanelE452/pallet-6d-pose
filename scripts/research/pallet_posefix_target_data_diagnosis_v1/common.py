"""Read-only diagnosis. No fit, optimizer, target or checkpoint mutation."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from scripts.research.pallet_posefix_crop_completion_v2 import common as P
from scripts.research.pallet_posefix_crop_completion_v2 import data as D

ROOT=P.ROOT; B=P.B; NAME='pallet_posefix_target_data_diagnosis_v1'
HERE=Path(__file__).resolve().parent; DOC=ROOT/'_docs/experiments'/NAME; RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
read=P.read; bind=P.bind; verify=P.verify; state_hash=P.state_hash
ARMS=['PRIOR1','FULL125','FULL_PRESERVE','FULL150']

def save(path,obj):
    path=Path(path);assert any(path.resolve().is_relative_to(p) for p in (DOC,RAW,OUT))
    path.parent.mkdir(parents=True,exist_ok=True)
    def scalar(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    s=obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=scalar)+'\n'
    with path.open('x') as f:f.write(s)

def band(x):return int(np.searchsorted([5.,10.,20.,40.],x,side='left'))
def strict(row):
    q=B.E.P.top(row['refined']); vals=[row.get('stage1',{}).get(k) for k in ('s_remove','s_flip')]+[row.get('stage2',{}).get('s_remove')]
    return q is not None and np.isfinite(q['keypoints_xy'][:8]).all() and all(v is not None and np.isfinite(v) and v<=.025 for v in vals)

@lru_cache(maxsize=1)
def real_records():return read(P.DOC/'TRAIN_INPUT_AUDIT.json')['real']
@lru_cache(maxsize=6)
def pair(i,exp=1.25):
    r=real_records()[int(i)];b=r['old' if exp==1.25 else 'pair'];verify(b)
    return torch.load(ROOT/b['path'],map_location='cpu',weights_only=False)

def runs(indices):
    ii=sorted(set(int(i) for i in indices));rr=[]
    for i in ii:
        if not rr or i!=rr[-1][-1]+1:rr.append([i])
        else:rr[-1].append(i)
    return dict(count=len(rr),longest=max(map(len,rr),default=0),runs=[dict(start=x[0],end=x[-1],length=len(x)) for x in rr])

def summary(errors):
    a=np.array(errors,float)
    return dict(n=len(a),median=float(np.median(a)) if len(a) else None,
        quantiles={str(q):float(np.quantile(a,q)) if len(a) else None for q in (.75,.9,.95)},
        correct10=int((a<=10).sum()),correct20=int((a<=20).sum()),
        PCK10=float((a<=10).mean()) if len(a) else None,PCK20=float((a<=20).mean()) if len(a) else None,
        bands=[int((np.searchsorted([5,10,20,40],a,side='left')==j).sum()) for j in range(5)])
