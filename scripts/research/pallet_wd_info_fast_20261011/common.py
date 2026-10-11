"""New-namespace outputs, explicit source roots, and counted solver operations."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import gzip
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_wd_info_fast_20261011'
SOURCE = Path(os.environ['PALLET_SOURCE_ROOT']).resolve()
PUBLIC = Path(os.environ['PALLET_PUBLIC_ROOT']).resolve()
PRIVATE = Path(os.environ['PALLET_FAST_PRIVATE']).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as f:
        return json.load(f)


def rows(path):
    with gzip.open(path, 'rt') as f:
        for line in f:
            yield json.loads(line)


def clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, 'tolist'):
        return clean(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(path, value, *, replace=False):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE), path.name
    assert replace or not path.exists(), ('Preserve existing output', path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                  allow_nan=False) + '\n')
    pending.replace(path)


def binding(path):
    path = Path(path).resolve()
    for label, base in [('new_worktree', ROOT), ('historical_source', SOURCE),
                        ('published_source', PUBLIC), ('private_new', PRIVATE)]:
        if path.is_relative_to(base):
            return dict(owner=label, path=str(path.relative_to(base)),
                        sha256=sha(path), bytes=path.stat().st_size)
    raise ValueError('Unregistered source root')


@contextmanager
def pnp_counter(job, limit):
    """Count each OpenCV solver/refinement invocation, including failed calls.

    Job reservations in PLAN_LOCK sum to <=50,000; independent workers append
    receipts under a file lock. No fitting is hidden behind a single F count.
    """
    import cv2
    lock = read(DOC / 'PLAN_LOCK.json')
    assert lock['pnp_reservations'][job] == limit
    original = {}; calls = {}; started = time.monotonic()
    for name in ('solvePnP', 'solvePnPGeneric', 'solvePnPRansac',
                 'solvePnPRefineLM', 'solvePnPRefineVVS'):
        if not hasattr(cv2, name):
            continue
        fn = getattr(cv2, name); original[name] = fn; calls[name] = 0
        def counted(*args, _name=name, _fn=fn, **kwargs):
            assert sum(calls.values()) < limit, ('PnP budget exhausted', job)
            calls[_name] += 1
            return _fn(*args, **kwargs)
        setattr(cv2, name, counted)
    try:
        yield calls
    finally:
        for name, fn in original.items():
            setattr(cv2, name, fn)
        receipt = dict(job=job, limit=limit, calls=calls,
                       total=sum(calls.values()), wall_seconds=time.monotonic()-started)
        with (PRIVATE / 'PNP_RECEIPTS.jsonl').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(receipt) + '\n')
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


def lock_ready():
    lock = read(DOC / 'PLAN_LOCK.json')
    assert lock['status'] == 'LOCKED_BEFORE_NEW_FITS_AND_SOLVERS'
    for b in lock['new_code_bindings']:
        assert sha(ROOT / b['path']) == b['sha256'], b['path']
    return lock
