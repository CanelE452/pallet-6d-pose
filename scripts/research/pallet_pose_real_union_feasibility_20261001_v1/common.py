"""Immutable cache-only diagnostics; never a learned runtime selector."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_real_union_feasibility_20261001_v1'
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def verify(binding):
    actual = bind(ROOT / binding['path'])
    assert actual['sha256'] == binding['sha256'], binding['path']
    if 'bytes' in binding:
        assert actual['bytes'] == binding['bytes'], binding['path']


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW), path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value, str) else
                json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def protocol():
    verify(read(DOC / 'PROTOCOL_SHA.json'))
    value = read(DOC / 'PROTOCOL.json')
    assert value['schema'] == 'pallet_pose_real_union_feasibility_v1'
    for binding in [*value['inputs'].values(), *value['codes']]:
        verify(binding)
    return value
