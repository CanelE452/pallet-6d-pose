import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pallet-recording-transfer-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from scripts.research.pallet_existing_data_transfer_v1 import common as E

ROOT = E.ROOT
NAME = 'pallet_recording_disjoint_transfer_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OUT = ROOT / 'outputs' / NAME
ANCHOR = ROOT / '_docs/experiments/pallet_verified_anchor_v1'
FINAL = ROOT / 'data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json'
CONTRACT = ROOT / '_docs/experiments/pallet_011067_corner_contract_v1'
ARMS = ('R0', 'S0', 'S1')
SEVS = {'CLEAN': 'CLEAN', 'MODERATE': 'MODERATE_OCCLUSION', 'SEVERE': 'SEVERE_OCCLUSION'}
read = E.read

def save(path, value):
    """Write generated artifacts only in this experiment, never historical inputs."""
    p=Path(path).resolve()
    assert any(p.is_relative_to(root) for root in (DOC, RAW, OUT)), p
    p.parent.mkdir(parents=True,exist_ok=True)
    text=value if isinstance(value,str) else json.dumps(E.D.clean(value),ensure_ascii=False,indent=2)+'\n'
    p.write_text(text)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def bind(path):
    p = Path(path).resolve()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)

def verify(b):
    p = ROOT / b['path']
    assert p.is_file() and sha(p) == b['sha256'], f'Changed input: {p}'

def immutable():
    bindings = read(DOC / 'INPUT_BINDINGS.json')
    for b in bindings['files']:
        verify(b)
    return len(bindings['files'])

def records():
    return read(E.DOC / 'SPLIT_LOCK.json')['heldout']

def groups(rr):
    return {'ALL': [r['id'] for r in rr], **{k: [r['id'] for r in rr if r['severity'] == s] for k, s in SEVS.items()},
            **{rec: [r['id'] for r in rr if r['recording_group'] == rec] for rec in sorted({r['recording_group'] for r in rr})}}
