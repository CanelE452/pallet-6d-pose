"""Public output helpers; private existing inputs are read only."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_n3_subpix_final_20261010'
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX')


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def iter_rows(path=None):
    with gzip.open(path or DOC / 'PREDICTIONS.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def existing():
    """Resolve original immutable scorer/data dependencies, never training main()."""
    import scripts.research
    source = str(ROOT / 'scripts/research')
    scripts.research.__path__ = [source] + [p for p in scripts.research.__path__ if p != source]
    from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
    from scripts.research.pallet_training_free_compare_20261007_v1.methods import correct, cap_points
    return load_real, correct, cap_points
