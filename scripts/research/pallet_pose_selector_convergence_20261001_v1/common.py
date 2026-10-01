"""Separate artifacts; original union method and failed gates stay immutable."""
from pathlib import Path
import json
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as U

ROOT=U.ROOT
NAME='pallet_pose_selector_convergence_20261001_v1'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
HERE=Path(__file__).resolve().parent
PARENT_DOC=U.DOC
PARENT_RAW=U.RAW
MODEL_NAMES=('R0_ONLY','UNION_s1','UNION_s2','UNION_s3')
read=U.read
bind=U.bind
sha=U.sha
verify=U.verify
now=U.now


def save(path,value):
    path=Path(path).resolve();assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(U.clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def protocol():
    verify(read(DOC/'PROTOCOL_SHA.json'))
    p=read(DOC/'PROTOCOL.json')
    assert p['schema']=='pallet_pose_convex_convergence_v1'
    assert p['models']==list(MODEL_NAMES)
    for b in p['codes']:verify(b)
    for b in p['inputs'].values():verify(b)
    return p
