"""Descriptive reporting after the fixed six runs; never selects a model."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import write_json


def analyze(out):
    result = json.loads((out / 'RESULTS.json').read_text())
    seeds = result['config']['seeds']
    rows = list(csv.DictReader((out / 'PER_ROLE.csv').open()))
    summary = {}
    for population in result['populations']:
        summary[population] = {}
        for arm in ('A', 'B'):
            reports = [result['by_run'][f'{arm}_seed{s}'][population] for s in seeds]
            values = {}
            for key in ('angle_deg', 'distance_px', 'distance_diagonal', 'foreground_mass'):
                if key not in reports[0]:
                    continue
                for stat in ('median', 'p90'):
                    v = [r[key][stat] for r in reports]
                    values[f'{key}_{stat}_seed_mean'] = float(np.mean(v))
                    values[f'{key}_{stat}_seed_range'] = [min(v), max(v)]
            summary[population][arm] = values
    real = [r for r in rows if r['population'] == 'real_dev']
    groups = {}
    for group in sorted({r['group'] for r in real}):
        selected = [r for r in real if r['group'] == group]
        groups[group] = {'n_frames': len({r['id'] for r in selected})}
        for arm in ('A', 'B'):
            groups[group][arm] = {}
            for metric in ('angle_deg', 'distance_px'):
                v = [float(np.median([float(r[metric]) for r in selected
                                      if r['arm'] == arm and int(r['seed']) == s])) for s in seeds]
                groups[group][arm][f'{metric}_median_seed_mean'] = float(np.mean(v))
    report = {
        'aggregation': 'mean of per-seed role medians (and mean of per-seed p90), not pooled median or confidence interval',
        'population_summary': summary, 'real_by_open_group': groups,
        'verdict': 'MIXED_NO_CLEAR_REAL_LINE_IMPROVEMENT',
        'conclusion_ko': '합성의 팔레트 foreground attention 지도는 학습되었다. 실사 위치 중앙값은 소폭 줄었지만 방향과 큰 위치 실패는 개선되지 않았고, outdoor 그룹에서는 악화되었다. 팔레트 영역 집중만으로 실사 선 추정이 해결되었다고 판단하지 않는다.',
        'scope': 'Three paired seeds, fixed 3000 steps, frozen synthetic DOPE backbone, existing real DEV52 only; no causal claim from attention weights.',
    }
    write_json(out / 'ANALYSIS.json', report)
    a, b = summary['real_dev']['A'], summary['real_dev']['B']
    sa, sb = summary['synth_test']['A'], summary['synth_test']['B']
    addition = [
        '', '## 결과 해석', '', report['conclusion_ko'], '',
        '아래 수치는 3회 실행에서 각각 계산한 role 중앙값(또는 90백분위)의 평균이다. 신뢰구간이나 전체 pooled 중앙값이 아니다.', '',
        f"- 합성 test256 foreground attention 질량: {sa['foreground_mass_median_seed_mean']*100:.1f}% → {sb['foreground_mass_median_seed_mean']*100:.1f}%.",
        f"- 실사 DEV52 선 위치 중앙값: {a['distance_px_median_seed_mean']:.2f}px → {b['distance_px_median_seed_mean']:.2f}px. 세 seed 모두 소폭 감소(평균 약3%).",
        f"- 실사 선 방향 중앙값: {a['angle_deg_median_seed_mean']:.2f}° → {b['angle_deg_median_seed_mean']:.2f}°.",
        f"- 실사 위치 90백분위: {a['distance_px_p90_seed_mean']:.2f}px → {b['distance_px_p90_seed_mean']:.2f}px. 큰 실패는 오히려 늘었다.",
        '', '| 실사 그룹 | 장수 | A 방향° | B 방향° | A 위치px | B 위치px |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name, g in groups.items():
        addition.append(f"| {name} | {g['n_frames']} | {g['A']['angle_deg_median_seed_mean']:.2f} | {g['B']['angle_deg_median_seed_mean']:.2f} | {g['A']['distance_px_median_seed_mean']:.2f} | {g['B']['distance_px_median_seed_mean']:.2f} |")
    addition += ['', '시각적으로 B는 여러 실사에서 팔레트 모서리 주변에 더 집중하지만, 가림이 있는 outdoor 프레임에서는 건물·적재물·반사 패딩 영역을 참조하는 실패도 남는다. 전체 attention 평균에 가려진 차이는 갤러리의 12개 role 선택으로 확인한다.',
                 'Real pixel mask GT가 없으므로 실사 attention의 정확한 foreground 비율은 주장하지 않는다.',
                 '기존 DH_full의 과거 약39° 기록과는 백본·데이터·전처리·정답 검증 경로가 달라 직접 개선율을 비교하지 않는다.', '']
    original = (out / 'REPORT.md').read_text().split('\n## 결과 해석')[0]
    (out / 'REPORT.md').write_text(original + '\n'.join(addition))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)
