from pathlib import Path
import hashlib
import json
import numpy as np
from scripts.research.pallet_clean19_pose_sensitive_diag_v1 import common as G
C=G.C;P=C.H.P;V=C.H.V;D=G.D
ROOT=C.ROOT;NAME='pallet_existing_data_transfer_v1'
DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME
ARMS=('T0_EASY_PSEUDO','T1_HARD_PSEUDO','T2_HARD_MANUAL')
read=C.read;bind=C.bind;verify=C.verify
def save(p,obj):
    assert p.is_relative_to(DOC) or p.is_relative_to(RAW)
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(obj if isinstance(obj,str) else json.dumps(D.clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def key(fid):return hashlib.sha256(('existing-hard-v1'+fid).encode()).hexdigest()
def manual(r):
    ann=read(ROOT/r['annotation']['path']);q,mask=P.C.R.manual_target(ann)
    entries=ann['objects'][0]['keypoint_annotations']
    for i,e in enumerate(entries):
        if any('identity' in k.lower() and str(v).lower() in ('unknown','unverified','unresolved') for k,v in e.items()):mask[i]=False
    mask[8]=False
    return q,mask
def immutable():
    lock=read(DOC/'INPUT_LOCK.json')
    for b in lock['files']:verify(b)
    verify(lock['split']);verify(lock['candidates'])
