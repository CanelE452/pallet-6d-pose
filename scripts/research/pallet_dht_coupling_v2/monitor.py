"""Read-only concise progress for the owned experiment; performs no inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def last_complete_trace(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 16384))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            pass
    return None


def snapshot(root):
    root = Path(root)
    status_path = root / 'STATUS.json'
    result = dict(stage='not_started', complete=False)
    if status_path.exists():
        status = read(status_path)
        result.update(status)
        for role in ('driver', 'child'):
            if status.get(role + '_pid'):
                result[role + '_alive'] = Path('/proc', str(status[role + '_pid'])).exists()
        if status.get('log'):
            result['log_age_seconds'] = round(time.time() - Path(status['log']).stat().st_mtime, 1)
    for kind, directory in (('training', 'runs'), ('evaluation', 'evaluation')):
        completed = []
        for marker in sorted((root / directory).glob('*/COMPLETION.json')):
            value = read(marker)
            if value.get('complete') and value.get('PASS'):
                completed.append(marker.parent.name)
        result['completed_' + kind + '_cells'] = completed
        result['n_completed_' + kind + '_cells'] = len(completed)
    stage = result['stage']
    if stage.startswith('train_'):
        name = stage[len('train_'):]
        trace = last_complete_trace(root / 'runs' / name / 'BATCH_TRACE.jsonl')
        if trace is not None:
            started = (trace['epoch'] - 1) * 3499 + trace['batch'] + 1
            result['current_training'] = dict(name=name, epoch=trace['epoch'],
                epoch_batch_started=trace['batch'] + 1, batches_started=started,
                expected_batches=6998, percent_batches_started=round(100 * started / 6998, 2),
                interpretation='Input batches started; completed optimizer updates are certified only by completion audits')
    if (root / 'DRIVER_FAILURE.json').exists():
        result['recorded_failure'] = read(root / 'DRIVER_FAILURE.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    print(json.dumps(snapshot(parser.parse_args().run_dir), ensure_ascii=False))
