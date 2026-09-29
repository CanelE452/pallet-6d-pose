"""New outputs only; the completed transfer batch remains immutable."""
from __future__ import annotations

from pathlib import Path
import json
import os
from scripts.research.pallet_clean_to_pose_transfer_v1 import common as OLD

ROOT = OLD.ROOT
NAME = 'pallet_clean_pose_minimal_v1'
DOC = ROOT / '_docs/experiments' / NAME
RAW = ROOT / 'data/pallet/results' / NAME
OUT = ROOT / 'outputs' / NAME
read, bind, verify, now, sha = OLD.read, OLD.bind, OLD.verify, OLD.now, OLD.sha


def save(path, value, immutable=False):
    path = Path(path).resolve()
    assert any(path.is_relative_to(p) for p in (DOC, RAW, OUT)), path
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if immutable and path.exists():
        assert path.read_text() == text, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def resource(event, seconds=0, fits=0, updates=0, selector_fits=0, details=None):
    """Carry past cost forward exactly; never mutate the previous final ledger."""
    path = DOC / 'RESOURCE_LEDGER.json'
    value = read(path)
    if any(e['event'] == event for e in value['new_events']):
        raise AssertionError('Resource event already charged: ' + event)
    increments = dict(student_fits=fits, selector_fits=selector_fits,
                      GPU_training_seconds=seconds, optimizer_updates=updates)
    totals = {k: value['totals'][k] + v for k, v in increments.items()}
    assert all(totals[k] <= cap for k, cap in value['caps'].items())
    value['new_events'].append(dict(event=event, at=now(), **increments, details=details))
    value['totals'] = totals
    save(path, value)
