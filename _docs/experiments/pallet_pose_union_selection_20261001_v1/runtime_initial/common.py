"""Local immutable artifacts; no evaluation-reference imports."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_pose_union_selection_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
HERE=Path(__file__).resolve().parent
PREV=ROOT/'_docs/experiments/pallet_pose_stable_improvement_20261001_v1'
PREV_RAW=ROOT/'data/pallet/results/pallet_pose_stable_improvement_20261001_v1'
SOURCE=ROOT/'data/pallet/results/pallet_selector_recovery_v1/stage2_synth_scorer'
MODELS=('R0','DIVERSE251_s1','DIVERSE251_s2','DIVERSE251_s3')


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def verify(b):
    p=ROOT/b['path'];assert sha(p)==b['sha256'],b['path']
    if 'bytes' in b:assert p.stat().st_size==b['bytes'],b['path']
def clean(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x
def save(p,x):
    p=Path(p).resolve();assert p.is_relative_to(DOC) or p.is_relative_to(RAW)
    p.parent.mkdir(parents=True,exist_ok=True)
    text=x if isinstance(x,str) else json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x') as f:f.write(text)
def protocol():
    verify(read(DOC/'SOURCE_PROTOCOL_SHA.json'))
    return read(DOC/'SOURCE_PROTOCOL.json')
def source_guard(allow_source_targets=False):
    """Applied before inference/training imports; permit only source references."""
    seen=[]
    forbidden=('/data/evaluation/','GEOMETRY_RESOLVED_POSE_GT','TRUTH_FOR_DISPLAY','AXIS_REVIEW_MANIFEST',
        '/real_gt_v2/annotations/','/pallet_pose_stable_improvement_20261001_v1/POSE_METRICS',
        '/pallet_pose_diagnosis_20260930_v1/E1_POSE_METRICS')
    if not allow_source_targets:
        forbidden+=('/SOURCE_MANIFEST.json','GEOMETRY_SIDETABLE.npz','DIMENSION_SIDECAR.json',
            '/SYNTH_RECORDS.json','/SYNTH_LABELS.npz','/labels/',
            'gt_points.npy','gt_valid.npy','gt_support.npy','matched_gt_index.npy')
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]);assert not any(t in p for t in forbidden),('REAL_REFERENCE_ACCESS_DENIED',p)
            if str(ROOT) in p:seen.append(p)
    sys.addaudithook(hook)
    return seen
