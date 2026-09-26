import contextlib
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_min_hard_ab_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OUT = ROOT / 'outputs' / NAME
TAGS = ('CLEAN', 'MODERATE', 'SEVERE', 'UNCERTAIN', 'INVALID')

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)

def verify(b):
    assert sha(ROOT / b['path']) == b['sha256'], b['path']

def save(path, value, immutable=False):
    p = Path(path)
    assert any(p.resolve().is_relative_to(root.resolve()) for root in (DOC, RAW, OUT))
    p.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable and p.exists():
        assert p.read_text() == data, f'Immutable conflict: {p}'
        return
    fd, tmp = tempfile.mkstemp(prefix='.' + p.name, dir=p.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

@contextlib.contextmanager
def exclusive(name):
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / (name + '.lck')).open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('Already open/running: ' + name) from e
        yield

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()

def key(prefix, frame_id):
    return hashlib.sha256((prefix + frame_id).encode()).hexdigest()

def queue():
    lock = read(DOC / 'DIFFICULTY_QUEUE_LOCK.json')
    verify(lock['private_queue'])
    return read(RAW / 'DIFFICULTY_QUEUE_PRIVATE.json')['rows']

def state():
    return read(RAW / 'STATE.json')

def set_state(status, **kwargs):
    value = dict(status=status, updated_at=now(), **kwargs)
    save(RAW / 'STATE.json', value)
    save(DOC / 'STATUS.json', value)

def annotation_labels_path():
    return RAW / 'HARD_LABELS_PRIVATE.json'
