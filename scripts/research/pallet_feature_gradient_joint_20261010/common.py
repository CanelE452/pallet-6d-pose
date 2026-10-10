"""Bounded new-output I/O; original private inputs and frozen results are read only."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = Path(os.environ.get('PALLET_JOINT_OUTPUT', str(WORKTREE / '_docs/experiments/pallet_feature_gradient_joint_20261010'))).resolve()
PRIVATE = Path(os.environ.get('PALLET_PRIVATE_OUTPUT', '/tmp/pallet-feature-gradient-joint-private-20261010'))
BASELINE_DOC = WORKTREE / '_docs/experiments/pallet_n3_subpix_final_20261010'
NEW_METHODS = ('JOINT_FIXED_ISOTROPIC', 'FG_JOINT_POSTERIOR')
METHODS = ('BASE', 'N3_DIM_SYM', 'SUBPIX', 'N3_THEN_SUBPIX', *NEW_METHODS)


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
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def source_bindings(names):
    directory = WORKTREE / 'scripts/research/pallet_feature_gradient_joint_20261010'
    return [{'path': str((directory / name).relative_to(WORKTREE)), 'sha256': sha(directory / name)} for name in names]


def verify_bindings(bindings):
    for binding in bindings:
        assert sha(WORKTREE / binding['path']) == binding['sha256'], binding['path']
