"""Run the frozen experiment through training, evaluation, report and notification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
STAGES = (
    ('cache', 'cache_features.py', 'cache/CACHE_COMPLETE.json'),
    ('train', 'train.py', 'LOGITS_MANIFEST.json'),
    ('select', 'select_synthetic.py', 'SYNTHETIC_EVALUATION.json'),
    ('real_evaluation', 'evaluate_real.py', 'REAL_EVALUATION_COMPLETE.json'),
    ('aggregate', 'aggregate_results.py', 'VERDICT.json'),
    ('report', 'report.py', 'REPORT_RENDER.json'),
    ('finalize', 'finalize.py', 'COMPLETION.json'),
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    if not (run_dir/'PURPOSE.md').exists():
        raise ValueError('The run directory needs its declared purpose')
    lock_handle = (run_dir/'DRIVER.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(run_dir/'TRAIN_PROTOCOL.json')
    for source, digest in protocol['source_sha256'].items():
        if sha(source) != digest:
            raise ValueError(f'Frozen training source changed: {source}')
    required = [HERE/script for _, script, _ in STAGES]
    required += [HERE/name for name in ('driver.py', 'model.py', 'features.py', 'source_data.py',
                                        'readout.py', 'inference.py', 'paper_evaluation.py')]
    # All downstream actions must exist before the first long stage starts.
    current = {str(path): sha(path) for path in required}
    contract_path = run_dir/'DRIVER_CONTRACT.json'
    if contract_path.exists():
        contract = read(contract_path)
        if contract['source_sha256'] != current or contract['train_protocol_sha256'] != sha(run_dir/'TRAIN_PROTOCOL.json'):
            raise ValueError('Driver sources changed; preserve history and audit a repair before resuming')
    else:
        contract = dict(complete=True, created_at_utc=now(), source_sha256=current,
                        train_protocol_sha256=sha(run_dir/'TRAIN_PROTOCOL.json'),
                        stages=[dict(name=n, script=s, completion=c) for n,s,c in STAGES])
        write(contract_path, contract)
    logs = run_dir/'logs'
    logs.mkdir(exist_ok=True)
    history = []
    current_stage = 'preflight'
    child = None
    try:
        for name, script, artifact in STAGES:
            current_stage = name
            if sha(run_dir/'TRAIN_PROTOCOL.json') != contract['train_protocol_sha256']:
                raise ValueError('Training protocol changed during execution')
            for source, digest in current.items():
                if sha(source) != digest:
                    raise ValueError(f'Experiment source changed during execution: {source}')
            command = [sys.executable, str(HERE/script), '--run-dir', str(run_dir)]
            start = time.monotonic()
            print(f'[{now()}] Start {name}: {script}', flush=True)
            log_path = logs/f'{name}.log'
            with log_path.open('a') as handle:
                handle.write(f'\n[{now()}] {json.dumps(command)}\n')
                handle.flush()
                child = subprocess.Popen(command, cwd=run_dir, stdout=handle,
                                         stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONUNBUFFERED':'1'})
                while child.poll() is None:
                    write(run_dir/'DRIVER_PROGRESS.json', dict(complete=False, status='running',
                        stage=name, pid=os.getpid(), child_pid=child.pid, updated_at_utc=now(),
                        elapsed_stage_seconds=time.monotonic()-start, log=str(log_path),
                        completed_stages=history, driver_contract_sha256=sha(contract_path)))
                    time.sleep(2)
                returncode = child.returncode
            if returncode != 0:
                raise RuntimeError(f'{name} exited {returncode}; see {log_path}')
            marker = run_dir/artifact
            result = read(marker)
            if result.get('complete') is not True or result.get('PASS', True) is not True:
                raise ValueError(f'{name} did not prove successful completion: {marker}')
            history.append(dict(stage=name, elapsed_seconds=time.monotonic()-start,
                                artifact=str(marker), sha256=sha(marker)))
            print(f'[{now()}] Complete {name}', flush=True)
        write(run_dir/'DRIVER_PROGRESS.json', dict(complete=True, status='complete',
              completed_at_utc=now(), pid=os.getpid(), completed_stages=history,
              driver_contract_sha256=sha(contract_path)))
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        failure = dict(complete=False, PASS=False, status='failed', stage=current_stage,
                       failed_at_utc=now(), error=f'{type(exc).__name__}: {exc}',
                       completed_stages=history, traceback=traceback.format_exc())
        write(run_dir/'DRIVER_FAILURE.json', failure)
        write(run_dir/'DRIVER_PROGRESS.json', failure)
        print(f'Failure recorded: {run_dir / "DRIVER_FAILURE.json"}. '
              'The completion notification has not been sent.', flush=True)
        raise
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir)
