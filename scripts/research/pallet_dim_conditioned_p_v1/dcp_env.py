"""New-root-only experiment I/O; historical P/source modules remain read-only."""
from pathlib import Path
import sys,json,hashlib,os,subprocess
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_dim_conditioned_p_v1'
RAW=ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1'
OLD_CODE=ROOT/'scripts/research/pallet_final_ml_contribution_test_v1'
SYM_DOC=ROOT/'_docs/experiments/pallet_symmetry_three_line_v1'
SYM_RAW=ROOT/'data/pallet/results/pallet_symmetry_three_line_v1'
sys.path.insert(1,str(ROOT));sys.path.insert(2,str(OLD_CODE))
import common as C
read=C.read;sha=C.sha;old=C.old;LINE=C.LINE;R0=C.R0;R0_SHA=C.R0_SHA
ARMS=['N0_BASE_REPLAY','N1_SYM_ONLY','N2_DIM_ONLY','N3_DIM_SYM','N4_META_SYM']
SQUARE_ARMS=['S0_FIXED','S1_SYM','S2_META_SYM']
MIXED_ARMS=['M0_DIM_SYM','M1_META_SYM']
def now():return datetime.now(timezone.utc).isoformat()
def bound(path):
    p=Path(path).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def write(path,value):
    path=Path(path).resolve();assert any(path.is_relative_to(p) for p in (DOC,RAW))
    path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.pending');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(path)
def freeze(path,value):
    if Path(path).exists():assert read(path)==value,('Frozen artifact differs',path)
    else:write(path,value)
def verify(bindings):
    for b in bindings:assert sha(ROOT/b['path'])==b['sha256'],b['path']
def state_sha(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def gpu():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    foreign=[r for r in p.splitlines() if r.split(',')[0].strip()!=str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    assert not foreign,('Foreign compute; preserve and stop',foreign)
    assert float(s.split(',')[4])<80,('Thermal guard',s)
    return dict(time=now(),gpu=s,compute=p,foreign_compute=foreign,rustdesk_preserved=True)
