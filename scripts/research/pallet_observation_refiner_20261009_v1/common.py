"""Read-only source bindings; experiment outputs belong to this worktree."""
from pathlib import Path
import contextlib
import gzip
import hashlib
import json
import os
import subprocess
import sys
import numpy as np

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ['PALLET_SOURCE_ROOT']).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_observation_refiner_20261009_v1'
SCRATCH = Path(os.environ.get('PALLET_OBSERVATION_SCRATCH', '/dev/shm/pallet-observation-private-20261009'))
CONTROLS = ('BASE', 'SUBPIX', 'N3', 'N3_SUBPIX')
SUFFIXES = ('NO_MASK_STANDARD', 'NO_MASK_ROBUST', 'GEOM_NOSELF_STANDARD',
            'GEOM_NOSELF_ROBUST', 'ORACLE_NOSELF_ROBUST', 'ORACLE_VISIBLE_ROBUST')

def finite(x):
    if isinstance(x, dict): return {str(k): finite(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [finite(v) for v in x]
    if hasattr(x, 'tolist'): return finite(x.tolist())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x

def read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt',encoding='utf-8') as f:
        return json.load(f)

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(finite(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def digest(x):
    return hashlib.sha256(json.dumps(finite(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def binding(path):
    path=Path(path).resolve()
    for root,origin in ((WORKTREE,'experiment'),(ROOT,'source')):
        if path.is_relative_to(root):return dict(path=str(path.relative_to(root)),origin=origin,sha256=sha(path),bytes=path.stat().st_size)
    return dict(path=path.name,origin='external_readonly_dependency',sha256=sha(path),bytes=path.stat().st_size)

def iter_rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            if line.strip():yield json.loads(line)

def save_rows(path,rows):
    assert not Path(path).exists(), 'Preserve completed raw rows'
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as f:
        for row in rows:f.write(json.dumps(finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def source_modules():
    import scripts.research
    source=str(ROOT/'scripts/research')
    scripts.research.__path__=[p for p in scripts.research.__path__ if p!=source]+[source]

def source_state():
    status=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=ROOT)
    diff=subprocess.check_output(['git','diff','--binary'],cwd=ROOT)
    names=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).decode().split('\0')
    return dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                tracked_file_sha256={p:sha(ROOT/p) for p in names if p},
                tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),status_sha256=hashlib.sha256(status).hexdigest())

@contextlib.contextmanager
def no_truth_reads():
    """Fail if deployable inference tries to open reference/visibility evidence."""
    import builtins,io
    blocked=('TARGETS.json','GEOMETRY_RESOLVED_POSE_GT','AXIS_REVIEW_MANIFEST',
             'STATIC_VISIBILITY_MERGE_AUDIT','ANNOTATION_PROGRESS','severity_snapshot')
    original=(builtins.open,io.open,Path.open)
    def wrap(fn):
        def guarded(file,*a,**kw):
            if any(t in str(file) for t in blocked):raise AssertionError('GT_CANARY:'+Path(str(file)).name)
            return fn(file,*a,**kw)
        return guarded
    builtins.open,io.open,Path.open=map(wrap,original)
    try:yield
    finally:builtins.open,io.open,Path.open=original
