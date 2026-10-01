"""Immutable artifacts for TRAIN signed physical T/R regression."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as U

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_pose_signed_axes_20261001_v1'
HERE = Path(__file__).resolve().parent
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
PARENT_DOC = U.DOC
PARENT_RAW = U.RAW
CONV_DOC = ROOT / '_docs/experiments/pallet_pose_selector_convergence_20261001_v1'
CONV_RAW = ROOT / 'data/pallet/results/pallet_pose_selector_convergence_20261001_v1'
ANCHOR_DOC = ROOT / '_docs/experiments/pallet_pose_pareto_anchor_20261001_v1'
ANCHOR_HERE = ROOT / 'scripts/research/pallet_pose_pareto_anchor_20261001_v1'
ANCHOR_RAW = ROOT / 'data/pallet/results/pallet_pose_pareto_anchor_20261001_v1'
CONTEXT_DOC = ROOT / '_docs/experiments/pallet_pose_anchor_context_20261001_v1'
CONTEXT_HERE = ROOT / 'scripts/research/pallet_pose_anchor_context_20261001_v1'
CONTEXT_RAW = ROOT / 'data/pallet/results/pallet_pose_anchor_context_20261001_v1'
RISK_DOC = ROOT / '_docs/experiments/pallet_pose_anchor_risk_20261001_v1'
RISK_HERE = ROOT / 'scripts/research/pallet_pose_anchor_risk_20261001_v1'
RISK_RAW = ROOT / 'data/pallet/results/pallet_pose_anchor_risk_20261001_v1'
RBF_DOC = ROOT / '_docs/experiments/pallet_pose_anchor_rbf_20261001_v1'
RBF_HERE = ROOT / 'scripts/research/pallet_pose_anchor_rbf_20261001_v1'
RBF_RAW = ROOT / 'data/pallet/results/pallet_pose_anchor_rbf_20261001_v1'
MODEL_NAMES = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def protocol(name='PROTOCOL'):
    verify(read(DOC / f'{name}_SHA.json'))
    value = read(DOC / f'{name}.json')
    assert value['schema'].startswith('pallet_pose_signed_axes_')
    for binding in [*value['inputs'].values(), *value['codes']]:
        verify(binding)
    return value
