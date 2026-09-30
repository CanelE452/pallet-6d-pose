"""Evidence plots and a complete natural99 gallery from frozen real inputs.

``composition`` is reference-free and may run during training. ``all`` requires
the complete prediction lock and scored RESULTS; it performs zero forwards.
Gallery seed 1 is declared by plan() before results and is never selected by
performance. Every natural99 image is shown, including failures and regressions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import numpy as np

from . import common as C
from scripts.research.pallet_pose_diagnosis_20260930_v1.review_figures import photo


FIG = C.DOC / 'figures'
KEYS = ('translation_cm', 'rotation_deg')
UNITS = ('T (cm)', 'R (°)')
FIXED = ('R0', 'PRIOR1', 'FULL125')
SEED = 1
COLORS = {'R0': '#64748b', 'PRIOR1': '#537fc4', 'FULL125': '#8a6ab0',
          'SINGLE': '#d98a25', 'DIVERSE': '#1e9677'}
plt.rcParams.update({'font.family': [FontProperties(fname='/usr/share/fonts/truetype/nanum/NanumGothic.ttf').get_name(), 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})


def savefig(fig, filename, dpi=130):
    FIG.mkdir(parents=True, exist_ok=True)
    kwargs = {'pil_kwargs': {'quality': 90}} if filename.endswith('.jpg') else {}
    fig.savefig(FIG / filename, dpi=dpi, bbox_inches='tight', **kwargs)
    plt.close(fig)


def plan():
    """Freeze complete membership/order/seed independently of outcomes."""
    C.protocol()
    rows = C.read(C.RAW / 'EVAL_METADATA.json')
    original = {r['id'] for r in C.D.metadata()}
    natural = sorted((r for r in rows if r['id'] in original and r['severity'] != 'CLEAN'),
                     key=lambda r: (r['recording'], r['id']))
    assert len(natural) == 99
    pages = []
    for rec in sorted({r['recording'] for r in natural}):
        group = [r for r in natural if r['recording'] == rec]
        for start in range(0, len(group), 6):
            pages.append(dict(recording=rec, recording_N=len(group),
                              first=start + 1, last=min(start + 6, len(group)),
                              IDs=[r['id'] for r in group[start:start + 6]],
                              figure=f'gallery_{rec}_{start // 6 + 1:02d}.jpg'))
    output = dict(gallery_seed=SEED, seed_selection='Fixed seed1 before new evaluation, never best seed',
                  gallery_models=['R0', *(f'{a}_s{SEED}' for a in C.ARMS)],
                  membership_rule='Every natural99 frame; recording then ID; six rows and three models per page',
                  gallery_frames=99, gallery_pages=pages,
                  image_bindings=[r['image'] for r in natural],
                  outcome_based_frame_selection=False,
                  summary_plots_use_all_training_seeds=list(C.SEEDS),
                  reference_legend='Cyan stored 2D reference; T/R from K/dimension-derived geometry, not independent physical pose measurement',
                  inference_forwards=0)
    C.save(C.DOC / 'FIGURE_CASE_SELECTION.json', output)
    return output


def composition():
    planned = plan()
    prepared = C.read(C.DOC / 'DATA_PREPARATION_FINAL.json')
    C.verify(prepared['inputs'])
    inp = C.read(C.ROOT / prepared['inputs']['path'])
    counts = {arm: Counter(r['recording_id'] for r in inp['arms'][arm]) for arm in C.ARMS}
    assert all(sum(v.values()) == C.protocol()['sample_size'] for v in counts.values())
    recordings = sorted(set().union(*(set(v) for v in counts.values())))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)
    limit = max(sum(v.values()) for v in counts.values()) * 1.12
    for axis, arm, color in zip(axes, C.ARMS, (COLORS['SINGLE'], COLORS['DIVERSE'])):
        values = [counts[arm][r] for r in recordings]
        bars = axis.barh(recordings, values, color=color)
        axis.bar_label(bars, labels=[str(v) if v else '' for v in values], padding=3)
        axis.set(xlim=(0, limit), xlabel='고유 학습 영상 수',
                 title=f'{arm}: {sum(values)}장 / {sum(v > 0 for v in values)} recording')
        axis.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('학습 영상 수는 같고 촬영 recording 구성이 다름', fontsize=15)
    fig.text(.5, .005, '같은 의사 타깃 계보·300 update·source 예산 / 사람 검토 또는 독립 물리 GT를 새로 추가하지 않음', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    savefig(fig, '01_training_composition.png')
    print('COMPOSITION_FIGURE_READY', dict(counts), 'gallery_seed', planned['gallery_seed'], flush=True)


def model_names():
    return [*FIXED, *(f'{a}_s{s}' for a in C.ARMS for s in C.SEEDS)]


def color(model):
    return COLORS[model] if model in FIXED else COLORS['SINGLE' if model.startswith('SINGLE') else 'DIVERSE']


def load_scored():
    from . import evaluate as E
    lock, rows, predictions, _ = E.locked_inputs()
    results_path = C.DOC / 'RESULTS.json'
    assert results_path.is_file(), 'RESULTS.json must exist before outcome graphics'
    results = C.read(results_path)
    metrics = C.read(C.RAW / 'POSE_METRICS.json')
    details = C.read(C.DOC / 'DETAILED_COMPARISONS.json')
    assert set(metrics) == set(model_names())
    assert all(set(v) == set(lock['IDs']) for v in metrics.values())
    return rows, predictions, metrics, results, details


def aggregate_plots(results):
    names = model_names()
    for statistic, filename in [('median', '02_pose_medians.png'), ('P90', '03_pose_P90.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8.3))
        for row, pop in enumerate(('NATURAL99', 'CLEAN29')):
            for col, key in enumerate(KEYS):
                ax = axes[row, col]
                summaries = [results['summaries'][pop][name] for name in names]
                values = [v['conditional'][key][statistic] for v in summaries]
                shown = [np.nan if v is None else v for v in values]
                bars = ax.bar(np.arange(len(names)), shown, color=[color(n) for n in names], alpha=.92)
                ax.bar_label(bars, labels=['NA' if v is None else f'{v:.2f}' for v in values], padding=3, fontsize=8)
                labels = [f'{n}\n{s["valid_pose"]}/{s["frames"]} valid' for n, s in zip(names, summaries)]
                ax.set_xticks(np.arange(len(names)), labels, rotation=28, ha='right', fontsize=8)
                finite = [v for v in values if v is not None and np.isfinite(v)]
                if finite:
                    ax.set_ylim(0, max(finite) * 1.19 if max(finite) > 0 else 1)
                ax.set(title=f'{pop}: {UNITS[col]} {statistic}', ylabel=UNITS[col])
                ax.grid(axis='y', alpha=.2)
        fig.suptitle(f'같은 GEO·T/R 계약의 {statistic} — 모든 학습 seed 표시, 낮을수록 좋음', fontsize=14)
        fig.text(.5, .005, '유효 pose 조건부 값. 전체 분모와 실패 수를 함께 표시하며 실패는 별도 비악화 판정에 포함. 재사용 DEV.', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, 1))
        savefig(fig, filename)


def changes(before, after, ids):
    values = []
    counts = Counter()
    for fid in ids:
        a, b = after[fid], before[fid]
        if not a['available'] or not b['available']:
            counts['실패/비교불가'] += 1
            continue
        dt, dr = (a[k] - b[k] for k in KEYS)
        values.append((fid, dt, dr))
        st = -1 if dt < -1e-7 else 1 if dt > 1e-7 else 0
        sr = -1 if dr < -1e-7 else 1 if dr > 1e-7 else 0
        label = ('동시 개선' if st == sr == -1 else '동시 악화' if st == sr == 1 else
                 'T만 개선' if st == -1 and sr == 1 else 'R만 개선' if st == 1 and sr == -1 else '동률 포함')
        counts[label] += 1
    assert sum(counts.values()) == len(ids)
    return values, counts


def paired_plots(rows, metrics):
    planned = plan()
    ids = [fid for page in planned['gallery_pages'] for fid in page['IDs']]
    rec = {r['id']: r['recording'] for r in rows}
    recordings = sorted({rec[i] for i in ids})
    palette = {r: plt.get_cmap('tab10')(j) for j, r in enumerate(recordings)}
    fig, axes = plt.subplots(2, len(C.SEEDS), figsize=(15, 8), sharex=True, sharey=True)
    outcome = {}
    for row, arm in enumerate(C.ARMS):
        for col, seed in enumerate(C.SEEDS):
            name = f'{arm}_s{seed}'
            values, counts = changes(metrics['R0'], metrics[name], ids)
            outcome[name] = counts
            ax = axes[row, col]
            for r in recordings:
                subset = [v for v in values if rec[v[0]] == r]
                ax.scatter([v[1] for v in subset], [v[2] for v in subset],
                           color=palette[r], alpha=.75, s=24,
                           label=f'{r} (N={sum(rec[i] == r for i in ids)})')
            ax.axhline(0, color='black', lw=.7)
            ax.axvline(0, color='black', lw=.7)
            ax.set_xscale('symlog', linthresh=1)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.set(title=f'{name}\n동시 개선 {counts["동시 개선"]} / 동시 악화 {counts["동시 악화"]}',
                   xlabel='ΔT (cm), symlog 축', ylabel='ΔR (°)')
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, -.012))
    fig.suptitle('natural99 프레임별 변화 = 새 모델 − R0 / 왼쪽 아래가 T·R 동시 개선', fontsize=14)
    fig.tight_layout(rect=(0, .07, 1, 1))
    savefig(fig, '04_all_seed_paired_changes.png')
    categories = ['동시 개선', 'T만 개선', 'R만 개선', '동시 악화', '동률 포함', '실패/비교불가']
    colors = ['#1e9677', '#d98a25', '#8a6ab0', '#c95b5b', '#aab0b8', '#30343a']
    names = list(outcome)
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(names))
    for category, col in zip(categories, colors):
        v = np.array([outcome[n][category] for n in names])
        bars = ax.bar(names, v, bottom=bottom, label=category, color=col)
        ax.bar_label(bars, labels=[str(n) if n else '' for n in v], label_type='center', fontsize=9)
        bottom += v
    assert np.all(bottom == 99)
    ax.set(ylim=(0, 106), ylabel='자연 가림 프레임 수', title='각 seed에서 같은 natural99 전체 분모의 T/R 변화 방향')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(.5, -.12), fontsize=9)
    ax.tick_params(axis='x', rotation=10)
    fig.text(.5, -.025, '세 seed는 같은99장을 반복 평가한 것. 297개의 독립 영상으로 세지 않는다. 방향 동률 허용치 1e-7.', ha='center', fontsize=10)
    fig.tight_layout()
    savefig(fig, '05_outcome_directions.png')


def recording_plot(details):
    by_recording = details['by_recording']
    labels, entries = [], []
    for pop in ('NATURAL99', 'CLEAN29'):
        for rec, record in sorted(by_recording[pop].items()):
            labels.append(f'{pop} / {rec} / N={record["frames"]}')
            entries.append(record['models'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    y = np.arange(len(labels))
    for col, key in enumerate(KEYS):
        ax = axes[col]
        for arm, shift, colr in zip(C.ARMS, (-.19, .19), (COLORS['SINGLE'], COLORS['DIVERSE'])):
            means, lower, upper = [], [], []
            for models in entries:
                base = models['R0']['conditional'][key]['median']
                new = [models[f'{arm}_s{s}']['conditional'][key]['median'] for s in C.SEEDS]
                if base is None or any(v is None for v in new):
                    means.append(np.nan); lower.append(0); upper.append(0)
                    continue
                delta = np.asarray(new) - base
                means.append(delta.mean()); lower.append(delta.mean() - delta.min()); upper.append(delta.max() - delta.mean())
            ax.barh(y + shift, means, height=.34, color=colr, label=arm,
                    xerr=np.asarray([lower, upper]), capsize=2, error_kw={'elinewidth': .9})
        ax.axvline(0, color='black', lw=.8)
        ax.set_yticks(y, labels, fontsize=9)
        ax.set(xlabel=f'평균 seed 중앙값 − R0 중앙값 / {UNITS[col]}', title=UNITS[col])
        ax.grid(axis='x', alpha=.2)
        ax.legend(fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('recording별 변화: 막대=세 seed 평균, 가로선=seed 최솟값~최댓값', fontsize=14)
    fig.text(.5, .005, '오차막대는 신뢰구간이 아니다. Natural/Clean을 분리하며 음수가 개선. 작은 recording의 N을 함께 표시.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, 1))
    savefig(fig, '06_recording_changes.png')


def learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    styles = ('-', '--', ':')
    for arm in C.ARMS:
        for seed, style in zip(C.SEEDS, styles):
            name = f'{arm}_s{seed}'
            fit = C.read(C.DOC / f'FIT_{name}.json')
            C.verify(fit['trace'])
            trace = [json.loads(line) for line in (C.ROOT / fit['trace']['path']).read_text().splitlines()]
            assert [r['step'] for r in trace] == list(range(1, 301))
            for row, branch in enumerate(('real', 'source')):
                for col, key in enumerate(('heatmap', 'coordinate')):
                    values = np.array([r['loss'][branch][key] for r in trace])
                    assert np.isfinite(values).all()
                    ax = axes[row, col]
                    ax.plot(np.arange(1, 301), values, color=color(name), alpha=.12, lw=.7)
                    mean = np.convolve(values, np.ones(25) / 25, mode='valid')
                    ax.plot(np.arange(25, 301), mean, color=color(name), ls=style, lw=1.4, label=name)
                    ax.set(title=f'{branch} / {key} loss', xlabel='optimizer update', ylabel='학습 목적함수 값')
                    ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, .015))
    fig.suptitle('모든 6회 학습의 손실 — 진한 선=25 update 이동평균, 옅은 선=원래 값', fontsize=14)
    fig.text(.5, -.012, '학습 손실 감소는 평가 T/R 개선 또는 일반화의 증거가 아니다. 모든 모델은 고정 마지막300 update를 사용.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, 1))
    savefig(fig, '07_learning_curves.png')


def metric_text(row):
    if not row['available']:
        return 'POSE FAILURE (전체 분모 유지)'
    return f'T {row["translation_cm"]:.2f} cm / R {row["rotation_deg"]:.2f}°'


def gallery(rows, predictions, metrics):
    # All references are deliberately first opened only after load_scored().
    from scripts.research.pallet_pose_diagnosis_20260930_v1.scoring import references
    refs = references()
    planned = plan()
    meta = {r['id']: r for r in rows}
    names = planned['gallery_models']
    text = ['# 자연 가림99장 전체 비교', '',
            '**고정 seed1의 R0 → SINGLE251 → DIVERSE251을 같은 이미지에서 비교한다.** Seed1은 결과를 보기 전에 정했으며 최고 성능 seed를 선택한 것이 아니다. 전체 통계·판정에는 seed1/2/3을 모두 사용한다.', '',
            '99장 전체를 recording·ID 순으로 표시한다. 주황은 해당 모델의 원래 코너 출력, 청록은 저장된 2D 평가 참조, 흰 점선은 동일한 R0 검출 박스다. T/R은 K·치수와 2D 좌표에서 유도된 같은 GEO 자세 계약의 결과이며 독립 장비로 실측한 물리 pose 정답을 뜻하지 않는다.', '',
            'T와 R은 낮을수록 좋다. 검출 매칭/IoU로 나쁜 pose를 제외하지 않았다. 이미지 바깥 좌표는 화면에서만 잘리며 점수 계산에는 그대로 남는다. 그림의 원래 인덱스를 프레임별 정답 최적 순열로 재배열하지 않았다. 실패는 실패로 표시한다.', '',
            '[전체 결과](REPORT_KO.md) · [사전 고정한 사례 선택 규칙](FIGURE_CASE_SELECTION.json) · [모든 seed의 수치](RESULTS.json)', '']
    displayed = []
    for page in planned['gallery_pages']:
        ids = page['IDs']
        fig, axes = plt.subplots(len(ids), 3, figsize=(15, 3.05 * len(ids)), squeeze=False)
        for row, fid in enumerate(ids):
            r = meta[fid]
            C.verify(r['image'])
            image = cv2.imread(str(C.ROOT / r['image']['path']))
            assert image is not None
            for col, name in enumerate(names):
                title = f'{fid}\n{name}: {metric_text(metrics[name][fid])}'
                photo(axes[row, col], image, predictions[name][fid], refs[fid]['gt'], title)
            displayed.append(fid)
        fig.suptitle(f'{page["recording"]} — 주황=모델 출력 / 청록=저장 2D 참조 / 흰 점선=고정 검출 박스', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .98))
        savefig(fig, page['figure'])
        text += [f'## {page["recording"]}: {page["first"]}–{page["last"]} / {page["recording_N"]}', '',
                 f'![{page["recording"]} 전체 프레임 비교](figures/{page["figure"]})', '']
    assert len(displayed) == len(set(displayed)) == 99
    C.save(C.DOC / 'GALLERY_NATURAL99.md', '\n'.join(text))
    print('NATURAL_GALLERY_COMPLETE', len(displayed), len(planned['gallery_pages']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['composition', 'all'])
    args = parser.parse_args()
    composition()
    if args.stage == 'all':
        rows, predictions, metrics, results, details = load_scored()
        aggregate_plots(results)
        paired_plots(rows, metrics)
        recording_plot(details)
        learning_curves()
        gallery(rows, predictions, metrics)
        print('FIGURES_COMPLETE', len(list(FIG.glob('*'))), flush=True)


if __name__ == '__main__':
    main()
