"""Run the frozen integration verification and open its actual-image report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from discord_notify import notify


def invoke(name, root):
    subprocess.run([sys.executable, str(HERE/name), '--run-dir', str(root)], cwd=root, check=True)


def completion_message(result):
    lines = ['DOPE·YOLO + DHT 검증 완료 (실사 개발셋52장).',
             '검출 코너 평균 오차(px), seed별 수치 평균:']
    for model, label in [('DOPE_exact_resize', 'DOPE(좌표 보정)'), ('YOLO', 'YOLO')]:
        values = []
        for arm in ('baseline', 'selected'):
            matches = [row for row in result['seed_summary']
                       if (row['model'], row['arm'], row['population'], row['group']) ==
                       (model, arm, 'real_dev', 'all')]
            if len(matches) != 1:
                raise ValueError(f'Missing or ambiguous completed real metric: {model}/{arm}')
            value = matches[0]['metrics']['corner_mean_px']['mean']
            if value is None or not math.isfinite(value):
                raise ValueError(f'Nonfinite completed real metric: {model}/{arm}')
            values.append(value)
        lam = result['selection'][model]['lambda']
        lines.append(f'{label}: {values[0]:.2f} → {values[1]:.2f}px, λ={lam:g}'
                     + (' (DHT 미사용)' if lam == 0 else ''))
    lines.append('실험 완료는 성능 개선을 뜻하지 않습니다. 상세 결과: live/integration_report.html')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'data/pallet/results/dht_pose_integration_v1')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    for name in ['PURPOSE.md', 'CONFIG.json', 'manifest.json', 'AUXILIARY_PROTOCOL.json']:
        if not (root/name).is_file():
            raise ValueError(f'Missing frozen input: {name}')
    live = root/'live'
    if not args.report_only:
        invoke('dope_baseline.py', root)
        invoke('yolo_baseline.py', root)
        invoke('dht_lines.py', root)
        invoke('evaluate.py', root)
        # Preserve the measured parent runtime whose SHA defines LIVE_PROTOCOL.
        # A fresh output directory is required for an independent timing rerun.
        if not (root/'RUNTIME.json').exists():
            invoke('combined_benchmark.py', root)
        invoke('live_lines.py', root)
        invoke('evaluate.py', live)
        if not (live/'RUNTIME.json').exists():
            invoke('combined_benchmark.py', live)
        for name in ['DATA_REVALIDATION.json', 'DOPE_RESIZE_AUDIT.json']:
            if (root/name).exists() and not (live/name).exists():
                shutil.copyfile(root/name, live/name)
    result = json.loads((live/'RESULTS.json').read_text())
    runtime = json.loads((live/'RUNTIME.json').read_text())
    if not result.get('complete') or not runtime.get('complete') or not runtime.get('parity_PASS'):
        raise ValueError('Actual-image accuracy and timed-pipeline verification must complete before final reporting')
    command = [sys.executable, str(HERE/'report.py'), '--run-dir', str(live)]
    if args.no_open:
        command.append('--no-open')
    subprocess.run(command, cwd=live, check=True)
    report = live/'integration_report.html'
    if not report.is_file() or report.stat().st_size == 0:
        raise ValueError('Completed report artifact is missing or empty')
    if not args.report_only:
        notify(completion_message(result), live, event='complete', evidence=live/'RESULTS.json')
    print('Integration verification complete; accuracy verdict is in live/RESULTS.json', flush=True)


if __name__ == '__main__':
    main()
