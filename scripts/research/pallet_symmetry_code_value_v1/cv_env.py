"""Only this experiment may be written. DCP modules and artifacts stay read-only."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_symmetry_code_value_v1'
RAW=ROOT/'data/pallet/results/pallet_symmetry_code_value_v1'
DCP_CODE=ROOT/'scripts/research/pallet_dim_conditioned_p_v1'
sys.path.insert(1,str(DCP_CODE))
import dcp_env as D
read=D.read;sha=D.sha;bound=D.bound;verify=D.verify;now=D.now;gpu=D.gpu;state_sha=D.state_sha
ARMS={'A':['A0_CODE_BLIND','A1_CODE_AWARE'],'B':['B0_CODE_BLIND','B1_CODE_AWARE']}
POPS={'A':['SYNTH','DEV'],'B':['SYNTH','DEV','SQUARE']}
def write(path,value):
    path=Path(path).resolve();assert path.is_relative_to(DOC) or path.is_relative_to(RAW),path
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def freeze(path,value):
    if Path(path).exists():assert read(path)==value,('Lock changed',path)
    else:write(path,value)
def order_sha(a):return hashlib.sha256(a.astype('<i8').tobytes()).hexdigest()
def protocol():return read(DOC/'PROTOCOL_LOCK.json')
