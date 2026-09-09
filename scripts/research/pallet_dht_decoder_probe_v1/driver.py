"""Run the approved small decoder pilot through evaluation and reviewed delivery."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

from scripts.research.pallet_dht_coupling_v2.prepare import read, write, sha, check
from .freeze import verify


def now():
    return datetime.now(timezone.utc).isoformat()


def status(root, stage, **extra):
    write(root / 'STATUS.json', dict(stage=stage, updated_at_utc=now(), complete=False, **extra))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    check(not (root / 'COMPLETION.json').exists(), 'Already complete; do not repeat notification')
    verify(root)
    stages = [('diagnostic', 'DIAGNOSTIC.json'), ('cache', 'CACHE_COMPLETION.json'),
              ('train', 'TRAINING_COMPLETION.json'), ('evaluate', 'EVALUATION_COMPLETION.json'),
              ('report', 'REPORT_RENDER.json'), ('visual_qa', 'ACTUAL_VISUAL_QA.json')]
    try:
        for stage, marker in stages:
            verify(root)
            status(root, stage)
            if not (root / marker).exists():
                command = [sys.executable, '-m', f'scripts.research.pallet_dht_decoder_probe_v1.{stage}',
                           '--run-dir', str(root)]
                print(f'{now()} Starting {stage}', flush=True)
                with (root / f'{stage.upper()}_RUN.log').open('a') as stream:
                    subprocess.run(command, cwd=Path(__file__).resolve().parents[3],
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)
            marker_value = read(root / marker)
            check(marker_value.get('complete') is True and marker_value.get('PASS', True) is True,
                  f'Incomplete actual {stage} artifact')
            print(f'{now()} Verified {marker}', flush=True)
        status(root, 'actual_screenshot_review', html=str(root / 'index.html'))
        print('Actual screenshots ready; waiting for agent inspection receipt, not user approval', flush=True)
        while not (root / 'VISUAL_REVIEW.json').exists():
            time.sleep(1)
        review = read(root / 'VISUAL_REVIEW.json')
        check(review.get('actual_screenshots_inspected') is True
              and review['html_sha256'] == sha(root / 'index.html'), 'Current screenshots must be inspected')
        verify(root)
        from .finalize import finalize
        finalize(root)
    except Exception as error:
        status(root, 'failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
