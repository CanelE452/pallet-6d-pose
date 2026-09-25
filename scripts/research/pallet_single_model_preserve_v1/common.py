import sys
import os
import hashlib
from pathlib import Path
from scripts.research.pallet_selector_recovery_v1 import common as P
ROOT=P.ROOT;NAME='pallet_single_model_preserve_v1';DOC=ROOT/'_docs/experiments'/NAME;RAW=ROOT/'data/pallet/results'/NAME;OUT=ROOT/'outputs'/NAME
read=P.read;bind=P.bind;verify=P.verify;sha=P.sha;now=P.now;clean=P.clean;setup=P.setup;gpu=P.gpu;selected=P.selected;state_hash=P.state_hash
STRUCT=P.STRUCT;PREV_DOC=P.PREV_DOC;PREV_RAW=P.PREV_RAW
def save(p,x):
    import json
    p=Path(p);assert any(p.resolve().is_relative_to(r) for r in (DOC,RAW,OUT))
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(x if isinstance(x,str) else json.dumps(clean(x),ensure_ascii=False,indent=2)+'\n')
def freeze(p,x):assert not Path(p).exists(),str(p);save(p,x)
def key(s):return hashlib.sha256(('preserve-v1:'+s).encode()).hexdigest()
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def train_guard():
    reads=[]
    bad=('verified_anchor','TRUTH_FOR_DISPLAY','GEOMETRY_RESOLVED_POSE_GT','POSE_METRICS.json','FRAME_METRICS.json','REAL_SCORER_RESULTS','BASELINE_LOCK.json','pallet_eval','/data/evaluation/')
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=os.fsdecode(args[0]);assert not any(s in p for s in bad),'TRAIN_REFERENCE_READ_FORBIDDEN: '+p
            if str(ROOT) in p:reads.append(p)
    sys.addaudithook(hook);return reads
