"""Run calibration, frozen evaluation, visualization and verified delivery."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from .preflight import REPO, read, sha, write

def main(root):
    root = root.resolve()
    assert (root / 'PURPOSE.md').exists()
    assert not (root / 'COMPLETION.json').exists(), 'Pilot already complete'
    assert read(root / 'INPUT_SNAPSHOT.json')['PASS']
    freeze = read(root / 'SOURCE_FREEZE.json')
    for path, digest in freeze['sha256'].items():
        assert sha(path) == digest, f'Source freeze mismatch {path}'
    def status(stage):
        write(root / 'STATUS.json', dict(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat()))
        print(stage, flush=True)
    def run(module, name, extra=()):
        status(name)
        with (root / (name + '.log')).open('a') as log:
            subprocess.run([sys.executable, '-m', f'scripts.research.pallet_dht_global_layout_v1.{module}',
                            '--run-dir', str(root), *extra], cwd=REPO,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    if not (root / 'RESULTS.json').exists():
        run('runner', 'COMPUTE_RUN', ('--phase', 'all'))
    if not (root / 'REPORT_RENDER.json').exists():
        run('report', 'REPORT_RUN')
    if not (root / 'ACTUAL_VISUAL_QA.json').exists():
        run('visual_qa', 'VISUAL_QA_RUN')
    status('awaiting_independent_audit_and_actual_visual_review')
    for _ in range(3600):
        names = ('INDEPENDENT_AUDIT.json', 'ROOT_VISUAL_REVIEW.json', 'CONCLUSION.json')
        if all((root / name).is_file() for name in names):
            break
        time.sleep(2)
    else:
        raise TimeoutError('Review receipts not ready; computation artifacts remain available')
    run('finalize', 'FINALIZE_RUN')
    status('complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=REPO / 'data/pallet/results/pallet_dht_global_layout_v1')
    main(parser.parse_args().run_dir)
