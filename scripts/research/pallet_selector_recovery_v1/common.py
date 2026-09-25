"""Lightweight paths/utilities. No real GT module imports."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
NAME='pallet_selector_recovery_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
PREV_DOC=ROOT/'_docs/experiments/pallet_recording_disjoint_transfer_v1'
PREV_RAW=ROOT/'data/pallet/results/pallet_recording_disjoint_transfer_v1'
STRUCT=ROOT/'_docs/experiments/pallet_clean19_structured_easyhard_v1'
STAGES={1:'stage1_diagnostic',2:'stage2_synth_scorer',3:'stage3_real_recovery',4:'stage4_clean_preservation'}
ARMS=('S0','S1')
HYP=('long-face-front','short-face-front')
os.environ.setdefault('MPLCONFIGDIR','/tmp/pallet-selector-mpl')
os.environ.setdefault('OMP_NUM_THREADS','2')

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def bind(p):
    p=Path(p).absolute()
    return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def verify(b):assert sha(ROOT/b['path'])==b['sha256'],b['path']
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    p=Path(p);assert any(p.resolve().is_relative_to(r) for r in (DOC,RAW,OUT)),p
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(x if isinstance(x,str) else json.dumps(clean(x),ensure_ascii=False,indent=2)+'\n')
def freeze(p,x):
    assert not Path(p).exists(),f'Immutable artifact exists: {p}'
    save(p,x)
def sdoc(n):return DOC/STAGES[n]
def sraw(n):return RAW/STAGES[n]
def setup():
    import torch
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
    import cv2
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def gpu():
    import torch
    assert torch.cuda.is_available(),'CUDA unavailable in this execution context; check host before concluding'
    line=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    foreign=[r for r in processes.splitlines() if str(os.getpid())!=r.split(',')[0].strip() and 'rustdesk' not in r.lower()]
    assert not foreign,foreign
    assert int(line.split(',')[1].strip())<80,line
    return dict(time=now(),gpu=line,processes=processes)
def synth_guard():
    """Runtime file-access deny guard before importing any training dependencies."""
    accessed=[]
    forbidden=('pallet_eval','/data/evaluation/','verified_anchor','TRUTH_FOR_DISPLAY','GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW','pallet_recording_disjoint_transfer','pallet_existing_data_transfer','REAL_SCORER_RESULTS','REAL_ROUTER_RESULTS','stage1_diagnostic','stage3_real_recovery')
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0])
            assert not any(t in p for t in forbidden),f'SYNTH_ONLY_READ_VIOLATION: {p}'
            if str(ROOT) in p or p.startswith(('data/','_docs/','challenge/','scripts/')):accessed.append(p)
    sys.addaudithook(hook)
    return accessed
def selected(p):
    i=p.get('selected_index');return None if i is None else p['candidates'][i]
def key(s,seed=20260925):return hashlib.sha256(f'{seed}|{s}'.encode()).hexdigest()
