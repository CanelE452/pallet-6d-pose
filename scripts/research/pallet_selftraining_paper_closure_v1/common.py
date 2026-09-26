import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = 'pallet_selftraining_paper_closure_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
PAPER = ROOT / '_docs/paper/selftraining_submission_v1'
OLD = ROOT / '_docs/experiments/pallet_type_selftrain_v1'
CACHE = ROOT / 'data/pallet/results/pallet_type_selftrain_v1'
REC = OLD / 'selftrain_recovery_v1'
SPLIT = ROOT / '_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json'
FINAL = ROOT / 'data/pallet/results/pallet_verified_anchor_v1/metadata_conflict_qa/VERIFIED_LABELS_FINAL_PRIVATE.json'
TRUTH = ROOT / 'data/pallet/results/pallet_replay_clean19_v1/TRUTH_FOR_DISPLAY_ONLY.json'
META = ROOT / 'data/pallet/results/pallet_visible_refine_hidden_pnp_v1/INFERENCE_METADATA.json'
SUFFIXES = ('LR4', 'LR5', 'ORDER43', 'ORDER44')
ARMS = ['R0'] + [f'{t}_{s}' for s in SUFFIXES for t in ('SYN', 'RAW', 'REF')]

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def bind(p):
    p = Path(p).resolve()
    return dict(path=str(p.relative_to(ROOT)), sha256=sha(p), bytes=p.stat().st_size)
def verify(b): assert sha(ROOT/b['path']) == b['sha256'], b['path']
def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def save(p, value, freeze=False):
    p = Path(p).resolve()
    assert any(p.is_relative_to(r) for r in (DOC, RAW, PAPER)), p
    value = value if isinstance(value, str) else json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if freeze and p.exists(): assert p.read_text()==value, p; return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value)
def phase(arm): return 'pose_repeat' if 'ORDER' in arm else 'pose_only'
def prediction_path(arm):
    return CACHE/f'EVAL_PREDICTIONS_{arm}.json' if arm=='R0' else CACHE/'selftrain_recovery_v1'/phase(arm)/f'EVAL_PREDICTIONS_{arm}.json'
def records(): return read(SPLIT)['heldout']
def selected(p):
    i=p.get('selected_index')
    return None if i is None else p['candidates'][i]
def now(): return datetime.now(timezone.utc).isoformat()
def table(headers, rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
