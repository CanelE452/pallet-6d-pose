"""Immutable IO and accounting for fixed-data correspondence diagnostics."""
from pathlib import Path
import gzip
import hashlib
import json
import time

import numpy as np


ARMS = ('REOBS_MULTI', 'REOBS_SINGLE', 'TYPED_MULTI', 'TYPED_SINGLE')
POPULATIONS = {'CAL': 46, 'TEST': 49, 'REMAINDER': 143}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(finite(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as stream:
        stream.write(value if isinstance(value, str) else
                     json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_rows(path, values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for row in values:
                stream.write((canonical(row)+'\n').encode())


def budget_elapsed(folder):
    p = Path(folder)/'RESOURCE_LEDGER.jsonl'
    if not p.exists():
        return {'cpu_solver_analysis': 0., 'gpu_process': 0.}
    result = {'cpu_solver_analysis': 0., 'gpu_process': 0.}
    for line in p.read_text().splitlines():
        row = json.loads(line)
        result[row['kind']] += row['elapsed_seconds']
    return result


def record_stage(folder, name, started, *, kind='cpu_solver_analysis', **details):
    p = Path(folder)/'RESOURCE_LEDGER.jsonl'
    row = finite(dict(stage=name, kind=kind, elapsed_seconds=time.monotonic()-started, **details))
    with p.open('a') as stream:
        stream.write(canonical(row)+'\n')
    return row

