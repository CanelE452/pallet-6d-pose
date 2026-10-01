"""Separate immutable artifacts for pairwise supervision of frozen candidates."""
from pathlib import Path
import json
from scripts.research.pallet_pose_union_selection_20261001_v1 import common as U

ROOT = U.ROOT
NAME = 'pallet_pose_selector_pairwise_20261001_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
HERE = Path(__file__).resolve().parent
PARENT_DOC = U.DOC
PARENT_RAW = U.RAW
CONV_DOC = ROOT / '_docs/experiments/pallet_pose_selector_convergence_20261001_v1'
CONV_RAW = ROOT / 'data/pallet/results/pallet_pose_selector_convergence_20261001_v1'
MODEL_NAMES = ('R0_ONLY', 'UNION_s1', 'UNION_s2', 'UNION_s3')
NEW_MODELS = MODEL_NAMES[1:]
read, bind, sha, verify, now = U.read, U.bind, U.sha, U.verify, U.now


def save(path, value):
    path = Path(path).resolve()
    assert path.is_relative_to(DOC) or path.is_relative_to(RAW)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(value if isinstance(value, str) else
                json.dumps(U.clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def protocol():
    verify(read(DOC / 'PROTOCOL_SHA.json'))
    p = read(DOC / 'PROTOCOL.json')
    assert p['schema'] == 'pallet_pose_pairwise_supervision_v1'
    assert p['models'] == list(MODEL_NAMES)
    for binding in p['codes']:
        verify(binding)
    for binding in p['inputs'].values():
        verify(binding)
    return p
