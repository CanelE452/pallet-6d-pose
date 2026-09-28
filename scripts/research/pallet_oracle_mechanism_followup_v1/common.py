from pathlib import Path
import json
import os
import time
from scripts.research.pallet_selftraining_paper_closure_v1 import common as P
from scripts.research.pallet_material_selftrain_closure_v1 import common as M

ROOT=P.ROOT
NAME='pallet_oracle_mechanism_followup_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
read,sha,bind,verify,clean,now,table=P.read,P.sha,P.bind,P.verify,P.clean,P.now,P.table
ARMS=('R0','RAW_LR5','REF_LR5')

def save(path,value,freeze=False):
    path=Path(path).resolve()
    assert any(path.is_relative_to(r) for r in (DOC,RAW)),path
    value=value if isinstance(value,str) else json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if freeze and path.exists():
        assert path.read_text()==value,path
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(value);os.replace(tmp,path)

def checkpoint(material,arm):
    if material=='WOOD':return M.checkpoint(arm if arm in ('R0','SYN_LR5','TEACHER') else 'WOOD_'+arm)
    if arm=='TEACHER':return M.checkpoint(arm)
    if arm=='R0':return M.checkpoint('R0')
    return read(P.REC/'pose_only'/f'FIT_{arm}.json')['checkpoint']

def resource(event,seconds=0,gpu=False,fits=0,updates=0,details=None):
    ledger=read(DOC/'RESOURCE_LEDGER.json')
    ledger['events'].append(dict(event=event,utc=now(),wall_seconds=seconds,gpu_seconds=seconds if gpu else 0,
                                 fits=fits,optimizer_updates=updates,details=details))
    ledger['totals']={k:sum(e[k] for e in ledger['events']) for k in ('gpu_seconds','fits','optimizer_updates')}
    ledger['totals']['elapsed_wall_seconds']=time.time()-ledger['start_unix']
    for k,cap in [('gpu_seconds',21600),('fits',12),('optimizer_updates',7680),('elapsed_wall_seconds',36000)]:
        assert ledger['totals'][k]<=cap,(k,ledger['totals'][k],cap)
    save(DOC/'RESOURCE_LEDGER.json',ledger)

def state(stage,completed,next_action):
    previous=read(DOC/'STATE.json') if (DOC/'STATE.json').exists() else {}
    previous.update(stage=stage,updated_utc=now(),completed=completed,next_action=next_action)
    save(DOC/'STATE.json',previous)
