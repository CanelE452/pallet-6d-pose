"""New outputs only; authenticated historical inputs remain read only."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = WORKTREE = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PALLET_SOURCE_ROOT', str(ROOT))).resolve()
DOC = Path(os.environ.get('PALLET_WD_OUTPUT', str(ROOT/'_docs/experiments/pallet_wd_hypothesis_diag_20261010'))).resolve()
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
SEEDS = (1, 2, 3)
REAL = ROOT/'_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz'
SYNTH = ROOT/'_docs/experiments/pallet_vispnp_square6d_20261011/SYNTH_ALL.jsonl.gz'
GT = 'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json'
GEOMETRY = 'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
FINAL4 = ('eval_pallet07', 'eval_pallet09', 'eval_night08', 'eval_night09')

def finite(v):
    if isinstance(v, dict): return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [finite(x) for x in v]
    if hasattr(v, 'tolist'): return finite(v.tolist())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def read(path):
    path=Path(path)
    with (gzip.open if str(path).endswith('.gz') else open)(path,'rt',encoding='utf-8') as f:return json.load(f)
read_json=read

def write(path, value):
    path=Path(path).resolve()
    assert path.is_relative_to(DOC), 'Outputs belong to the new namespace'
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(finite(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)
write_json=write

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
    path=Path(path).resolve()
    assert path.is_relative_to(DOC) and not path.exists(), 'Preserve completed evidence'
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=6) as f:
        for r in records:f.write(json.dumps(finite(r),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(path)

def binding(path, owner=None):
    path=Path(path)
    owner=owner or (ROOT if path.is_relative_to(ROOT) else SOURCE)
    return dict(path=str(path.relative_to(owner)),sha256=sha(path),bytes=path.stat().st_size,
                owner='published_worktree' if owner==ROOT else 'historical_source')

_POSE=None
def pose_api():
    global _POSE
    if _POSE is None:
        from scripts.research.pallet_n3_subpix_final_20261010 import common as previous
        previous.ROOT=SOURCE
        previous.existing()  # resolve original source modules, no load_real()
        from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
        E,_=legacy()
        _POSE=E.POSE
    return _POSE

def core_bindings():
    return [binding(Path(__file__).parent/name,ROOT) for name in
            ('common.py','inputs.py','solver.py','rules.py','preflight.py','stage1.py')]

def source_lock_path():
    final=DOC/'STAGE1_FINAL_SOURCE_LOCK.json'
    return final if final.exists() else DOC/'STAGE1_SOURCE_LOCK.json'

def amend_before_calls(reason):
    """Preserve the initial audit lock; pin a corrected pre-execution version."""
    old_path=DOC/'STAGE1_SOURCE_LOCK.json';old=read(old_path)
    assert not (DOC/'STAGE1_SELECTIONS_REAL.jsonl.gz').exists(), 'No amendment after solver evidence'
    final=DOC/'STAGE1_FINAL_SOURCE_LOCK.json'
    assert not final.exists(), 'Preserve amendment history'
    write(final,dict(status='FINAL_LOCKED_BEFORE_ANY_STAGE0_STAGE1_SOLVER_CALL',
        solver_calls_before_amendment=0,reason=reason,previous_lock_path=old_path.name,
        previous_lock_sha256=sha(old_path),new_core=core_bindings(),original_core=old['original_core'],
        changes='Execution ordering/receipt linking only; no thresholds, metric definitions, frozen coordinates or inputs changed.'))
    return read(final)

def pin_core():
    p=DOC/'STAGE1_SOURCE_LOCK.json'
    assert not p.exists(), 'Preserve the original execution lock'
    pose=pose_api()
    source=[SOURCE/'scripts/research/pallet_dim_conditioned_p_v1/pose.py',
            SOURCE/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
            SOURCE/'challenge/evaluation_v2/pnp_selector.py',
            SOURCE/'scripts/annotate/pallet_geometry.py']
    write(p,dict(status='LOCKED_BEFORE_STAGE0_AND_STAGE1_CANDIDATE_CALLS',new_core=core_bindings(),
                 original_core=[binding(x,SOURCE) for x in source],
                 note='Original infer/solve/metric are reused; only selector return values are instrumented.'))
    return read(p)

def verify_core():
    lock=read(source_lock_path())
    for b in lock['new_core']+lock['original_core']:
        owner=ROOT if b['owner']=='published_worktree' else SOURCE
        assert sha(owner/b['path'])==b['sha256'], ('Frozen source changed',b['path'])
    return lock
