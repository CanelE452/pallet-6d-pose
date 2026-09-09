#!/usr/bin/env python3
"""Prepare the experiment plan; --execute explicitly starts configured training.

Default: create missing metadata and refresh the offline dashboard. --execute:
prepare/reuse features, execute all configured runs, verify and update dashboard.
The browser opens once at the end unless --no-open is set.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from report import combinations, open_dashboard, read, write


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from discord_notify import notify


def run(script, args, run_dir):
    print(f"Running {script} (cwd={run_dir})", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=run_dir, check=True)


def notify_main_completion(cfg, matrix, run_dir):
    completion = matrix['completion']
    expected = completion['expected_runs']
    if (cfg['stage'] != 'main' or matrix['stage'] != 'main' or matrix['main_status'] != 'COMPLETE'
            or expected <= 0 or expected != len(list(combinations(cfg)))
            or completion['complete_runs'] != expected or completion['complete_main_runs'] != expected
            or completion['invalid_runs'] != 0 or completion['incomplete_runs'] != 0):
        return
    dashboard = run_dir/'experiment_dashboard.html'
    if not dashboard.is_file() or dashboard.stat().st_size == 0:
        raise ValueError('Completed main dashboard artifact is missing or empty')
    lines = [f'DHT 패딩·시점 본실험 {expected}/{expected}개 실행 검증 완료.',
             f'실행당 {cfg["training"]["steps"]}단계, 최종 체크포인트·평가 검증 통과.']
    summaries = {(row['regime'], row['condition']): row for row in matrix.get('seed_summary', [])
                 if row['population'] == 'real_dev'}
    details = []
    for regime in cfg['training']['regimes']:
        for condition in cfg['conditions']:
            row = summaries.get((regime, condition))
            if row is None or row.get('seeds') != cfg['training']['seeds']:
                continue
            median = row['metrics'].get('distance_px_median')
            p90 = row['metrics'].get('distance_px_p90')
            if (median and p90 and all(value is not None and math.isfinite(value)
                                     for value in (median['mean'], p90['mean']))):
                details.append(f'{regime}/{condition}: {median["mean"]:.2f}/{p90["mean"]:.2f}px')
    if details:
        lines.append('실사 선 위치 중앙값/P90 (seed별 통계 평균):')
        lines.extend(details)
    lines.append('상세 비교: experiment_dashboard.html')
    notify('\n'.join(lines), run_dir, event='complete', evidence=run_dir/'MATRIX.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/pallet/results/dht_padding_view_v1")
    parser.add_argument("--execute", action="store_true", help="Prepare features and execute every configured training run")
    parser.add_argument("--no-open", action="store_true", help="Do not open the local dashboard automatically")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if not all((run_dir / name).is_file() for name in ("CONFIG.json", "manifest.json")):
        run("prepare.py", ["--run-dir", run_dir, "--phase", "metadata"], run_dir)
    cfg = read(run_dir / "CONFIG.json")
    failure = None
    if args.execute:
        try:
            run("prepare.py", ["--run-dir", run_dir, "--phase", "cache"], run_dir)
            for regime, condition, seed in combinations(cfg):
                run("train_eval.py", ["--run-dir", run_dir, "--regime", regime,
                                      "--condition", condition, "--seed", seed, "--phase", "all"], run_dir)
        except subprocess.CalledProcessError as exc:
            failure = exc
    run("report.py", ["--run-dir", run_dir, "--no-open"], run_dir)
    matrix = read(run_dir / "MATRIX.json")
    if args.execute and failure is None and matrix["completion"]["complete_runs"] != matrix["completion"]["expected_runs"]:
        failure = RuntimeError("One or more configured runs failed final completion verification; see MATRIX.json")
    state = {"mode": "execute" if args.execute else "plan_only", "stage": cfg["stage"],
             "main_status": matrix["main_status"], "completion": matrix["completion"],
             "dashboard": str(run_dir / "experiment_dashboard.html"),
             "execution_error": str(failure) if failure else None}
    write(run_dir / "DRIVER_STATUS.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    if args.execute and failure is None:
        notify_main_completion(cfg, matrix, run_dir)
    if not args.no_open:
        open_dashboard(run_dir / "experiment_dashboard.html")
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
