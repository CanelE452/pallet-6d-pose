"""Small I/O and execution records; no reference, model, or solver imports."""
from __future__ import annotations

from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with open(path) as stream:
        return json.load(stream)


def write(path, value, *, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if overwrite else "x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "xt") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def csv_write(path, records):
    records = list(records)
    if not records:
        raise ValueError("An empty result needs an explicit status, not an empty CSV")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in records for key in row))
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "xt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def append_log(path, event):
    """Append under an OS file lock; concurrent workers retain original events."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(dict(epoch=time.time(), **event),
                                ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def measured(log_path, operation, **metadata):
    start, cpu = time.monotonic(), time.process_time()
    append_log(log_path, dict(event="START", operation=operation, **metadata))
    try:
        yield
    except BaseException as exc:
        append_log(log_path, dict(event="FAILED", operation=operation,
            wall_seconds=time.monotonic() - start,
            CPU_seconds=time.process_time() - cpu,
            error_type=type(exc).__name__, error=str(exc), **metadata))
        raise
    else:
        append_log(log_path, dict(event="COMPLETED", operation=operation,
            wall_seconds=time.monotonic() - start,
            CPU_seconds=time.process_time() - cpu,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            **metadata))
