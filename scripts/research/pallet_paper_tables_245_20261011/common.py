"""Read-only historical inputs and exclusive new paper-table outputs."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PALLET_SOURCE_ROOT', str(ROOT))).resolve()
DOC = Path(os.environ.get('PALLET_PAPER_OUTPUT', ROOT / '_docs/experiments/pallet_paper_tables_245_20261011')).resolve()
JOINT = ROOT / '_docs/experiments/pallet_feature_gradient_joint_20261010/PREDICTIONS.jsonl.gz'
SQUARE = ROOT / '_docs/experiments/pallet_square6d_manualpnp_20261010/PREDICTIONS.jsonl.gz'
REUSE = ROOT / '_docs/experiments/pallet_n3_completion_v3/REUSE_RESULTS.json'
OLD_RAW = SOURCE / 'data/pallet/results/pallet_n3_completion_v3/reuse/PER_FRAME_SCORES.json'
SEEDS = (1, 2, 3)
BACKBONES = ('YOLO', 'DOPE', 'ResNet-18')
CORE_METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')
FINAL4 = ('eval_pallet07', 'eval_pallet09', 'eval_night08', 'eval_night09')

def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite(v) for v in value]
    if hasattr(value, 'tolist'): return finite(value.tolist())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as f:
        return json.load(f)

def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f: yield json.loads(line)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(finite(value), ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(name, value):
    path = DOC / name
    assert path.resolve().is_relative_to(DOC.resolve()) and not path.exists(), ('Preserve output', name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(finite(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def write_rows(name, values):
    path = DOC / name
    assert not path.exists(), ('Preserve sealed evidence', name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode='xb', mtime=0) as f:
        for value in values:
            f.write((json.dumps(finite(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode())

def binding(path):
    path = Path(path).resolve()
    root = ROOT if path.is_relative_to(ROOT) else SOURCE
    assert path.is_relative_to(root)
    return dict(path=str(path.relative_to(root)), owner='published' if root == ROOT else 'historical',
                sha256=sha(path), bytes=path.stat().st_size)

def verify_binding(b):
    p = (ROOT if b['owner'] == 'published' else SOURCE) / b['path']
    assert p.is_file() and sha(p) == b['sha256'], ('Frozen input changed', b['path'])
    return p

def verify_inputs():
    for b in read(DOC / 'INPUT_AUDIT.json')['bindings']: verify_binding(b)

def method_lock():
    p = DOC / 'METHOD_LOCK.json'
    value = read(p)
    assert value['status'].startswith('LOCKED'), value['status']
    return value

def pose_api():
    from scripts.research.pallet_wd_hypothesis_diag_20261010 import common as previous
    previous.SOURCE = SOURCE
    return previous.pose_api()

def math_api():
    pose_api()
    from scripts.research.pallet_training_free_compare_20261007_v1.common import legacy
    E, _ = legacy()
    return E.M

def prediction_only(row):
    keys = ('id', 'session', 'grade', 'method', 'seed', 'q0', 'qFinal', 'prediction_support', 'raw_hw', 'fixed_metadata')
    output = {k: row[k] for k in keys}
    output.update(backbone='YOLO', population='RECT', source_flag=False, reference_inputs=False,
                  detected=True, coordinate_provenance='unchanged authenticated existing cached coordinates')
    return output

def original_manifest():
    return read(ROOT / '_docs/experiments/pallet_n3_subpix_final_20261010/INPUT_AUDIT.json')['inputs']

