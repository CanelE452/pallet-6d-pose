"""Read-only prior adapters and this experiment's isolated output namespace."""
from pathlib import Path
import gzip
import json
import os
import sys

sys.dont_write_bytecode = True
WORKTREE = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get('PALLET_SOURCE_ROOT', str(WORKTREE))).resolve()
DOC = WORKTREE / '_docs/experiments/pallet_subpix_order_20261009_v1'
OUTPUT = Path(os.environ.get('PALLET_SUBPIX_ORDER_SCRATCH', '/dev/shm/pallet-subpix-order-20261009'))
PRIOR = ROOT / '_docs/experiments/pallet_n3_subpix_20261008_v1'
ARMS = ('BASE', 'N3', 'SUBPIX', 'N3_SUBPIX', 'SUBPIX_N3')
GRADE_PATH = ROOT / '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json'

import scripts.research
source = str(ROOT / 'scripts/research')
scripts.research.__path__ = [source] + [p for p in scripts.research.__path__ if p != source]
from scripts.research.pallet_training_free_compare_20261007_v1.common import (
    read, write, sha, digest, finite, legacy, load_real)
from scripts.research.pallet_training_free_compare_20261007_v1.methods import correct, cap_points


def iter_rows(path=None):
    with gzip.open(path or DOC / 'PREDICTIONS.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def grade_map():
    """Existing image grades, exclusively for analysis and timing stratification."""
    rows = [r for r in read(GRADE_PATH)['rows'] if r['population'] == 'DEV319']
    result = {r['id']: r['severity'] for r in rows}
    assert len(rows) == len(result) == 319
    assert {k: sum(g == k for g in result.values()) for k in ('clean','moderate','severe')} == {'clean':153,'moderate':92,'severe':74}
    return result


def binding(path):
    path = Path(path).resolve()
    if path.is_relative_to(WORKTREE):
        label = str(path.relative_to(WORKTREE)); origin = 'experiment'
    elif path.is_relative_to(ROOT):
        label = str(path.relative_to(ROOT)); origin = 'source'
    else:
        label = path.name; origin = 'immutable_baseline_dependency'
    return dict(path=label, origin=origin, sha256=sha(path), bytes=path.stat().st_size)
