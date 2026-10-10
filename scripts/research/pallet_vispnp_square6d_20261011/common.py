"""New-namespace outputs; all historical inputs remain read only."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = WORKTREE = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PALLET_SOURCE_ROOT', str(ROOT))).resolve()
DOC = Path(os.environ.get('PALLET_VIS_OUTPUT', str(ROOT / '_docs/experiments/pallet_vispnp_square6d_20261011'))).resolve()
PRIVATE = Path(os.environ.get('PALLET_PRIVATE_OUTPUT', '/tmp/pallet-vispnp-square6d-20261011-private'))
OLD_DOC = ROOT / '_docs/experiments/pallet_feature_gradient_joint_20261010'
OLD_FINAL = ROOT / '_docs/experiments/pallet_n3_subpix_final_20261010'
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
SEEDS = (1, 2, 3)

def finite(v):
    if isinstance(v, dict): return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [finite(x) for x in v]
    if hasattr(v, 'tolist'): return finite(v.tolist())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def read_json(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt', encoding='utf-8') as f: return json.load(f)
read = read_json

def write_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(finite(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
write = write_json

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(v):return hashlib.sha256(json.dumps(finite(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def rows(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)
iter_rows=rows

def write_rows(path, records):
    assert not Path(path).exists(), 'Preserve completed evidence'
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as f:
        for r in records:f.write(json.dumps(finite(r),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

def existing():
    from scripts.research.pallet_n3_subpix_final_20261010 import common as previous
    previous.ROOT=SOURCE
    return previous.existing()

def historical_rows():
    return [r for r in rows(OLD_DOC/'PREDICTIONS.jsonl.gz') if r['method'] in METHODS]

def core_bindings():
    names=['common.py','visibility.py','adapter.py','preflight.py','real.py','summarize_real.py','statistics.py','verdict.py','synth.py']
    return [dict(path=str((Path(__file__).parent/name).relative_to(ROOT)),sha256=sha(Path(__file__).parent/name)) for name in names]

def lock_sources():
    assert not (DOC/'SOURCE_LOCK.json').exists()
    write(DOC/'SOURCE_LOCK.json',dict(status='LOCKED_BEFORE_A1_A2_RESULTS',new_core=core_bindings(),
        original_core=read(DOC/'METHOD_LOCK.json')['source_bindings'],
        a0_core_preexecution_hash_captured='new_core_pre_A0' in read(DOC/'METHOD_LOCK.json'),
        note='Published original run A0 passed exact parity; its new core pre-execution hash was not captured. Source lock pins all A1/A2 code before any new VIS result.'))
