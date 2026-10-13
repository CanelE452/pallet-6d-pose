"""New output namespace, with existing modules and inputs used read-only."""
from pathlib import Path
import gzip
import os
import sys

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_n3_subpix_20261008_v1'
OUTPUT = Path(os.environ.get('PALLET_N3_SUBPIX_SCRATCH', '/dev/shm/pallet-n3-subpix-20261008'))
ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX')

# Namespace packages let the new runner retain its own output location while
# old modules resolve their original read-only data/configuration paths.
import scripts.research
source = str(ROOT / 'scripts/research')
scripts.research.__path__ = [source] + [p for p in scripts.research.__path__ if p != source]
from scripts.research.pallet_training_free_compare_20261007_v1.common import (
    read, write, sha, digest, finite, legacy, load_real)
from scripts.research.pallet_training_free_compare_20261007_v1.methods import correct, cap_points


def iter_rows(path=None):
    import json
    with gzip.open(path or DOC / 'PREDICTIONS.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def combine(gray, q0, qN, prediction_support):
    """Inference only: no reference pose, labels, visibility or target box."""
    import numpy as np
    qS, diagnostics = correct(gray, qN.copy(), prediction_support.copy(), 'SUBPIX')
    height, width = gray.shape
    qFinal = cap_points(q0.copy(), qS, width, height, prediction_support.copy())
    return qS, qFinal, diagnostics
