"""Read-only historical imports; all generated outputs restricted to new roots."""
from pathlib import Path
import sys, json, hashlib, subprocess, os
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_sensors_refinement_closeout_v1'
RAW=ROOT/'data/pallet/results/pallet_sensors_refinement_closeout_v1'
PAPER=ROOT/'_docs/paper/sensors_refinement_closeout_v1'
PCODE=ROOT/'scripts/research/pallet_final_ml_contribution_test_v1'
sys.path.insert(0,str(PCODE));sys.path.insert(1,str(ROOT))
sys.path.append(str(ROOT/'scripts/paper/pose_metric_closure_v1'))
import common as C
read=C.read;sha=C.sha;old=C.old
R0=C.R0;LINE=C.LINE;B=C.B;BRAW=C.BRAW
def write(path,value):
    path=Path(path).resolve()
    assert any(path.is_relative_to(r) for r in (DOC,RAW,PAPER)),path
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def freeze(path,value):
    if Path(path).exists():assert read(path)==value,('Frozen value differs',path)
    else:write(path,value)
def bound(path):
    path=Path(path);s=path.stat()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns)
def dataset():return old('train').FeatureDataset(LINE,LINE/'cache')
def gpu():
    fields='name,driver_version,temperature.gpu,memory.used,utilization.gpu,power.draw,clocks.sm,clocks.mem'
    status=subprocess.check_output(['nvidia-smi',f'--query-gpu={fields}','--format=csv,noheader,nounits'],text=True).strip()
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
    foreign=[r for r in proc.splitlines() if r.split(',')[0].strip()!=str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    return dict(timestamp=datetime.now(timezone.utc).isoformat(),fields=fields,gpu=status,processes=proc,foreign_compute=foreign,display_remote_retained=True)
def verify():
    lock=read(DOC/'SOURCE_BINDING.json')
    for r in lock['files']:
        p=ROOT/r['path'];assert sha(p)==r['sha256'],p
    for r in lock['cache_arrays']:
        s=(ROOT/r['path']).stat();assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'],r['path']
def fwd(model,b):return model(*(b[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0)
