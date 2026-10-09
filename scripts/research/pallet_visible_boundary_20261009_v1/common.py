"""Separate output namespace; existing source data and models are read-only."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_visible_boundary_20261009_v1'
SCRATCH = Path(os.environ.get('PALLET_BOUNDARY_SCRATCH', '/dev/shm/pallet-visible-boundary-20261009'))
CONTROLS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')
NEW_ARMS = tuple(f'{start}_{method}_{limit}' for start in ('BASE', 'N3')
                 for method in ('WIDE', 'BOUNDARY') for limit in ('NATIVE', 'CAP1'))
ARMS = CONTROLS + NEW_ARMS


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


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    for directory, origin in ((WORKTREE, 'experiment'), (ROOT, 'source')):
        if path.is_relative_to(directory):
            return dict(path=str(path.relative_to(directory)), origin=origin,
                        sha256=sha(path), bytes=path.stat().st_size)
    return dict(path=path.name, origin='external_reference', sha256=sha(path), bytes=path.stat().st_size)


def source_modules():
    import scripts.research
    source = str(ROOT / 'scripts/research')
    scripts.research.__path__ = [source] + [p for p in scripts.research.__path__ if p != source]


def iter_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def source_state():
    changed = subprocess.check_output(['git', 'diff', '--name-only', '-z'], cwd=ROOT).decode().split('\0')
    status = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z'], cwd=ROOT)
    diff = subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT)
    return dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip(),
                tracked_file_sha256={name: sha(ROOT / name) for name in changed if name},
                tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                status_sha256=hashlib.sha256(status).hexdigest())


def cap_points(q0, native, support, raw_hw):
    q0, native = np.asarray(q0, np.float64), np.asarray(native, np.float64)
    support = np.asarray(support, bool)
    assert q0.shape == native.shape == (9, 2) and support.shape == (9,)
    valid = support[:8] & np.isfinite(q0[:8]).all(1) & ~(q0[:8] == -1).all(1)
    result = q0.copy()
    assert np.isfinite(native[:8][valid]).all()
    delta = native[:8][valid] - q0[:8][valid]
    lengths = np.linalg.norm(delta, axis=1)
    cap = .01 * float(np.hypot(*raw_hw))
    factor = np.ones_like(lengths)
    np.divide(cap, lengths, out=factor, where=lengths > 0)
    result[np.flatnonzero(valid)] = q0[:8][valid] + delta * np.minimum(1., factor)[:, None]
    return result
