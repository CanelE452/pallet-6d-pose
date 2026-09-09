"""Continue the audited CSV-precision repair without repeating completed experiments."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from driver import read, write, sha, now

HERE = Path(__file__).resolve().parent


def run(run_dir):
    run_dir = Path(run_dir).resolve()
    failure = read(run_dir/'DRIVER_FAILURE.json')
    original = read(run_dir/'DRIVER_CONTRACT.json')
    repair = read(run_dir/'repairs/csv_precision_001/REPAIR_MANIFEST.json')
    if failure['stage'] != 'aggregate' or not repair['complete'] or not repair['PASS']:
        raise ValueError('This continuation is restricted to the audited reporting repair')
    if sha(run_dir/'DRIVER_CONTRACT.json') != repair['original_driver_contract_sha256']:
        raise ValueError('Original execution contract changed')
    if sha(run_dir/'TRAIN_PROTOCOL.json') != original['train_protocol_sha256']:
        raise ValueError('Training protocol changed')
    for path, previous in original['source_sha256'].items():
        expected = repair['source_after_sha256'].get(path, previous)
        if sha(path) != expected:
            raise ValueError(f'Unregistered source modification: {path}')
    history = failure['completed_stages']
    if [r['stage'] for r in history] != ['cache', 'train', 'select', 'real_evaluation']:
        raise ValueError('Training, selection and actual evaluation must already be complete')
    for row in history:
        if sha(row['artifact']) != row['sha256'] or not read(row['artifact'])['complete']:
            raise ValueError('A completed pre-repair experiment artifact changed')
    lock = (run_dir/'DRIVER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    current = 'aggregate'
    try:
        for current, script, marker in [('aggregate', 'aggregate_results.py', 'VERDICT.json'),
                ('report', 'report.py', 'REPORT_RENDER.json'), ('finalize', 'finalize.py', 'COMPLETION.json')]:
            for path, previous in original['source_sha256'].items():
                if sha(path) != repair['source_after_sha256'].get(path, previous):
                    raise ValueError(f'Source changed during reporting continuation: {path}')
            started = time.monotonic()
            log_path = run_dir/'logs'/f'{current}_resume.log'
            with log_path.open('a') as handle:
                handle.write(f'\n[{now()}] Audited reporting continuation\n'); handle.flush()
                child = subprocess.Popen([sys.executable, str(HERE/script), '--run-dir', str(run_dir)],
                    cwd=run_dir, stdout=handle, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONUNBUFFERED':'1'})
                while child.poll() is None:
                    write(run_dir/'DRIVER_PROGRESS.json', dict(complete=False, status='running_repair',
                        stage=current, pid=os.getpid(), child_pid=child.pid, updated_at_utc=now(),
                        completed_stages=history, log=str(log_path),
                        repair_manifest_sha256=sha(run_dir/'repairs/csv_precision_001/REPAIR_MANIFEST.json')))
                    time.sleep(2)
            if child.returncode:
                raise RuntimeError(f'{current} exited {child.returncode}; see {log_path}')
            data = read(run_dir/marker)
            if not data.get('complete') or not data.get('PASS', True):
                raise ValueError(f'{current} did not complete')
            history.append(dict(stage=current, artifact=str(run_dir/marker), sha256=sha(run_dir/marker),
                                elapsed_seconds=time.monotonic()-started))
            print(f'[{now()}] Completed reporting continuation: {current}', flush=True)
        write(run_dir/'DRIVER_PROGRESS.json', dict(complete=True, status='complete_with_audited_repair',
            completed_at_utc=now(), completed_stages=history,
            original_failure_sha256=sha(run_dir/'DRIVER_FAILURE.json'),
            repair_manifest_sha256=sha(run_dir/'repairs/csv_precision_001/REPAIR_MANIFEST.json'),
            resume_source_sha256=sha(__file__), training_or_inference_repeated=False))
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=10)
        write(run_dir/'REPORTING_RESUME_FAILURE.json', dict(complete=False, PASS=False, stage=current,
            failed_at_utc=now(), traceback=traceback.format_exc(), completed_stages=history))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    run(parser.parse_args().run_dir)
