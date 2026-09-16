"""New-experiment-only writes. Historical modules are read-only dependencies."""
from pathlib import Path
import sys, json, hashlib, os, subprocess
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
DOC=ROOT/'_docs/experiments/pallet_symmetry_three_line_v1'
RAW=ROOT/'data/pallet/results/pallet_symmetry_three_line_v1'
sys.path.insert(1,str(ROOT))
sys.path.insert(2,str(ROOT/'scripts/research/pallet_final_ml_contribution_test_v1'))
import common as C
R0=C.R0; R0_SHA=C.R0_SHA
DHT=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v2_wls_correction/heads/hough_seed1/checkpoint_final.pt'
DHT_SHA='50406286a135dc7ff5991af0940fb8f9cdd07048dc5fb6296cc4936dd46e54d5'
EXPORT=ROOT/'data/pallet/results/pallet_symmetry_dht_local_v1/export'
SQUARE=ROOT/'challenge/yolo_pose_one_model/datasets/live_gt_v6_clean'
TRACK=ROOT/'challenge/yolo_pose_one_model/challenge_c4_track'
read=C.read; sha=C.sha
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path).resolve();assert any(path.is_relative_to(p) for p in (DOC,RAW))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(path)
def freeze(path,value):
    if Path(path).exists():assert read(path)==value,('Frozen artifact differs',path)
    else:write(path,value)
def bound(path):
    p=Path(path).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size)
def gpu():
    s=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    other=[r for r in p.splitlines() if r.split(',')[0].strip()!=str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    assert not other,('Foreign compute; do not stop it',other)
    assert float(s.split(',')[4])<80
    return dict(time=now(),gpu=s,compute=p,foreign_compute=other)
