from pathlib import Path
import json
import os
import time
from scripts.research.pallet_oracle_mechanism_followup_v1 import common as OLD

ROOT, P, M = OLD.ROOT, OLD.P, OLD.M
NAME = 'pallet_pose_objective_followup_v2'
DOC, RAW = ROOT / '_docs/experiments' / NAME, ROOT / 'data/pallet/results' / NAME
read, sha, bind, verify, clean, now, table = OLD.read, OLD.sha, OLD.bind, OLD.verify, OLD.clean, OLD.now, OLD.table
checkpoint = OLD.checkpoint

def save(path, value, freeze=False):
    path = Path(path).resolve()
    assert any(path.is_relative_to(r) for r in (DOC, RAW)), path
    content = value if isinstance(value, str) else json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if freeze and path.exists():
        assert path.read_text() == content, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(content); os.replace(tmp, path)

def resource(event, seconds=0, gpu=False, fits=0, updates=0, details=None):
    ledger = read(DOC / 'RESOURCE_LEDGER.json')
    assert event not in [e['event'] for e in ledger['events']], event
    ledger['events'].append(dict(event=event, utc=now(), wall_seconds=seconds,
        gpu_seconds=seconds if gpu else 0, fits=fits, optimizer_updates=updates, details=details))
    ledger['totals'] = {k: sum(e[k] for e in ledger['events']) for k in ('gpu_seconds','fits','optimizer_updates')}
    ledger['totals']['elapsed_wall_seconds'] = time.time() - ledger['start_unix']
    for k, cap in [('gpu_seconds',21600),('fits',12),('optimizer_updates',7680),('elapsed_wall_seconds',36000)]:
        assert ledger['totals'][k] <= cap, (k, ledger['totals'][k], cap)
    save(DOC / 'RESOURCE_LEDGER.json', ledger)

def state(stage, completed, next_action):
    previous = read(DOC / 'STATE.json') if (DOC / 'STATE.json').exists() else {}
    previous.update(stage=stage, updated_utc=now(), completed=completed, next_action=next_action,
        input_bindings=bind(DOC / 'INPUT_BINDINGS.json'), resource_ledger=bind(DOC / 'RESOURCE_LEDGER.json'))
    save(DOC / 'STATE.json', previous)
