from pathlib import Path
import json
from scripts.research.pallet_selftraining_paper_closure_v1 import common as P

ROOT=P.ROOT
NAME='pallet_material_selftrain_closure_v1'
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
PAPER=P.PAPER
read,sha,bind,verify,clean,now,table=P.read,P.sha,P.bind,P.verify,P.clean,P.now,P.table
OLD_START='f45d9c0849e545270a3a7639a2ace3174bc32527'
ARMS=('R0','SYN_LR5','WOOD_RAW_LR5','WOOD_REF_LR5')
TEACHER=ROOT/'_docs/experiments/pallet_posefix_replay_v1'

def save(path,value,freeze=False):
    path=Path(path).resolve();assert any(path.is_relative_to(r) for r in (DOC,RAW)),path
    txt=value if isinstance(value,str) else json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if freeze and path.exists():assert path.read_text()==txt,path;return
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(txt)

def checkpoint(arm):
    if arm=='R0':return read(P.REC/'pose_only/PROTOCOL.json')['initialization']
    if arm=='TEACHER':return read(TEACHER/'FIT.json')['checkpoint']
    if arm=='SYN_LR5':return read(P.REC/'pose_only/FIT_SYN_LR5.json')['checkpoint']
    return read(DOC/f'FIT_{arm}.json')['checkpoint']
