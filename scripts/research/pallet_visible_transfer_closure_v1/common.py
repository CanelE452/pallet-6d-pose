from pathlib import Path
import json
from scripts.research.pallet_selftraining_paper_closure_v1 import common as P
ROOT=P.ROOT
DOC=ROOT/'_docs/experiments/pallet_visible_transfer_closure_v1'
RAW=ROOT/'data/pallet/results/pallet_visible_transfer_closure_v1'
read,sha,bind,verify,now,clean,table=P.read,P.sha,P.bind,P.verify,P.now,P.clean,P.table
ARMS=('R0','RAW_LR5','REF_LR5')
def save(path,obj,freeze=False):
    path=Path(path).resolve();assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    text=obj if isinstance(obj,str) else json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if freeze and path.exists():assert path.read_text()==text,path;return
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
def checkpoint(arm):
    return read(P.REC/'pose_only/PROTOCOL.json')['initialization'] if arm=='R0' else read(P.REC/'pose_only'/f'FIT_{arm}.json')['checkpoint']
