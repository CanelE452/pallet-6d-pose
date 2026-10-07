"""Small namespace-local receipts; existing inputs are strictly read-only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_depth_signal_pilot_20261007_v1'
DOC = ROOT / '_docs/experiments' / NAME
PRIVATE = Path('/dev/shm/pallet-depth-signal-pilot-20261007')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                sha256=sha(path), bytes=path.stat().st_size)

def verify(binding):
    path = ROOT / binding['path']
    assert sha(path) == binding['sha256'], binding['path']

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def save(path, value, freeze=True):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(PRIVATE), path
    text = value if isinstance(value, str) else json.dumps(clean(value), ensure_ascii=False,
                                                          indent=2, allow_nan=False) + '\n'
    if freeze and path.exists():
        assert path.read_text() == text, ('Refusing to overwrite', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)
