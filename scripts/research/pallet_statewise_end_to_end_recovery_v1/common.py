"""Small immutable IO, accounting, and explicit prediction/reference boundary."""
from __future__ import annotations

from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

SCHEMA = 'STATEWISE_END_TO_END_RECOVERY_V1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False, default=convert)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not replace:
        raise FileExistsError(p)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    tmp.write_text(canonical(value) + '\n')
    os.replace(tmp, p)


def rows(path):
    p = Path(path)
    with (gzip.open(p, 'rt') if p.suffix == '.gz' else p.open()) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_rows(path, values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    tmp = p.with_name(p.name + f'.tmp.{os.getpid()}')
    with gzip.open(tmp, 'wt') as f:
        for r in values:
            f.write(canonical(r) + '\n')
    os.replace(tmp, p)


@contextmanager
def measured(run_dir, stage, **details):
    start, cpu = time.monotonic(), time.process_time()
    receipt = dict(stage=stage, start_epoch=time.time(), **details)
    try:
        yield receipt
        receipt['execution_status'] = 'DONE'
    except BaseException as exc:
        receipt.update(execution_status='FAILED_TECHNICAL', error=repr(exc))
        raise
    finally:
        receipt.update(elapsed_seconds=time.monotonic()-start,
                       process_cpu_seconds=time.process_time()-cpu,
                       process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       peak_rss_scope='process lifetime high-water mark, not additive stage delta')
        p = Path(run_dir)/'RESOURCE_LEDGER.jsonl'
        with p.open('a') as f:
            f.write(canonical(receipt)+'\n')


class PredictionFirewall:
    """Process-scoped audit hook. Install before decoding prediction inputs.

    It prevents new primary/reference/oracle/manual-six inputs, not historical
    knowledge. CAL/synthetic supervision must use a separate process.
    """
    def __init__(self, forbidden_paths):
        self.forbidden = {str(Path(p).resolve()) for p in forbidden_paths}
        self.denied = []
        self.active = False

    def install(self):
        self.active = True
        sys.addaudithook(self._hook)
        return self

    def _hook(self, event, args):
        if self.active and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(Path(os.fsdecode(args[0])).resolve())
            if path in self.forbidden:
                self.denied.append(path)
                raise PermissionError('Prediction process forbids reference/manual input: ' + path)

    def receipt(self):
        return dict(active=self.active, forbidden_paths=sorted(self.forbidden),
                    denied_reads=self.denied, allowed_reference_reads=0,
                    scope='new process file reads; historical development exposure remains')
