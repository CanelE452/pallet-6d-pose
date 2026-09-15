"""Submission-only paths, atomic receipts and read-only historical bindings."""
from pathlib import Path
import sys, os, json, hashlib, subprocess, importlib.util
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments/pallet_sensors_submission_v1'
RAW = ROOT / 'data/pallet/results/pallet_sensors_submission_v1'
PAPER = ROOT / '_docs/paper/sensors_submission_v1'
OLD_DOC = ROOT / '_docs/experiments/pallet_sensors_refinement_closeout_v1'
OLD_RAW = ROOT / 'data/pallet/results/pallet_sensors_refinement_closeout_v1'
OFFICIAL = OLD_RAW / 'external/PoseFix_RELEASE'
OLD_CODE = ROOT / 'scripts/research/pallet_sensors_refinement_closeout_v1'
sys.path.insert(1, str(ROOT/'scripts/research/pallet_final_ml_contribution_test_v1'))
sys.path.insert(2, str(ROOT))
sys.path.append(str(ROOT/'scripts/paper/pose_metric_closure_v1'))
sys.path.append(str(OLD_CODE))
import common as C
read = C.read
sha = C.sha
old = C.old
LINE = C.LINE
B = C.B
BRAW = C.BRAW
R0 = C.R0
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, obj):
    path = Path(path).resolve()
    assert any(path.is_relative_to(r) for r in (DOC, RAW, PAPER)), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.pending')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)
def bound(path):
    p=Path(path).resolve(); s=p.stat()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=s.st_size)
def freeze(path, obj):
    if Path(path).exists(): assert read(path)==obj, ('Frozen record differs', str(path))
    else: write(path,obj)
def import_old(name):
    # Historical modules resolve `env` to this submission-only write guard.
    spec=importlib.util.spec_from_file_location('submission_reuse_'+name, OLD_CODE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def gpu():
    q=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader'],text=True).strip()
    p=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    foreign=[r for r in p.splitlines() if r.split(',')[0].strip()!=str(os.getpid()) and '/usr/share/rustdesk/rustdesk' not in r]
    return dict(time=now(),gpu=q,compute=p,foreign_compute=foreign,display_retained=True)
def verify():
    binding=read(DOC/'SOURCE_BINDING.json')
    for r in binding['files']: assert sha(ROOT/r['path'])==r['sha256'],r['path']
    for r in binding['cache_arrays']:
        s=(ROOT/r['path']).stat(); assert s.st_size==r['bytes'] and s.st_mtime_ns==r['mtime_ns'],r['path']
def dataset(): return old('train').FeatureDataset(LINE,LINE/'cache')
def receipt(name, inputs, outputs, start, **extra):
    write(DOC/(name+'.json'),dict(complete=True,start=start,end=now(),command=sys.argv,inputs=[bound(p) for p in inputs],outputs=[bound(p) for p in outputs],**extra))
def complete(name):
    p=DOC/(name+'.json')
    if not p.exists(): return False
    r=read(p)
    if not r.get('complete'): return False
    for b in r.get('inputs',[])+r.get('outputs',[]):
        p=ROOT/b['path'];assert p.exists() and sha(p)==b['sha256'],('Stale receipt',name,p)
    return True
