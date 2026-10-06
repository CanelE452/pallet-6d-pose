"""A-only read-only legacy adapters and bounded artifact I/O."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DOC=ROOT/'_docs/experiments/pallet_joint_action_handoff_20261006_v1'
DIM_CODE=ROOT/'scripts/research/pallet_dim_conditioned_p_v1'
GENERIC_CODE=ROOT/'scripts/research/pallet_final_ml_contribution_test_v1'
sys.path.insert(0,str(DIM_CODE));sys.path.insert(1,str(GENERIC_CODE));sys.path.insert(2,str(ROOT))
import dcp_env
from refiner import DimensionConditionedPointRefiner, context, local_phase
from generic_point_refiner import finite, sample
spec=importlib.util.spec_from_file_location('handoff_existing_pose',DIM_CODE/'pose.py')
POSE=importlib.util.module_from_spec(spec);spec.loader.exec_module(POSE)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.pending');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def hash_value(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
