"""Render fixed seed-1 best/worst 0918 dimension-transfer examples."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from scripts.evaluation import final_dimension_release as F
from scripts.evaluation import green0918_square_audit_v1 as Q


DOC = F.ROOT / '_docs/experiments/pallet_green0918_dimension_audit_v1'
RAW = F.ROOT / 'data/pallet/results/pallet_green0918_dimension_audit_v1'
FIGURES = DOC / 'figures'
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))


def read(path):
    return json.loads(Path(path).read_text())


def selected_points(row):
    index = row['selected_index']
    return None if index is None else np.asarray(row['candidates'][index]['keypoints_xy'], np.float64)


def draw_prediction(axis, points, color):
    if points is None:
        return
    for first, second in EDGES:
        axis.plot(points[[first, second], 0], points[[first, second], 1], color=color,
                  linewidth=1.8, alpha=.9)
    axis.scatter(points[:8, 0], points[:8, 1], s=22, facecolors='none', edgecolors=color,
                 linewidths=1.4)
    axis.scatter(points[8, 0], points[8, 1], s=28, marker='x', color=color, linewidths=1.5)


def draw_annotation(axis, annotation):
    entries = annotation['objects'][0]['keypoint_annotations']
    for index, entry in enumerate(entries[:8]):
        if entry.get('xy') is None or entry.get('visibility', 0) == 0:
            continue
        point = entry['xy']
        manual = entry.get('source') == 'manual_click'
        axis.scatter([point[0]], [point[1]], s=40 if manual else 22,
                     c='#ff36d1' if manual else '#eeeeee', marker='o' if manual else '+',
                     linewidths=1.2, zorder=10)
        if manual:
            axis.text(point[0] + 3, point[1] - 3, str(index), color='#ff36d1', fontsize=8,
                      weight='bold', zorder=11)


def main():
    dataset = read(Q.DATASET)
    metadata = {row['id']: row for row in dataset['records']}
    predictions = read(RAW / 'PREDICTIONS.json')['predictions']
    metrics = read(RAW / 'METRICS.json')['modes']['manual_declared']['rows']
    models = ('R0', 'N0_BASE_REPLAY_seed1', 'N2_DIM_ONLY_seed1')
    prediction_rows = {name: {row['id']: row for row in predictions[name]} for name in models}
    metric_rows = {name: {row['id']: row for row in metrics[name]} for name in models}
    eligible = []
    for frame_id in metadata:
        before = metric_rows['N0_BASE_REPLAY_seed1'][frame_id]
        after = metric_rows['N2_DIM_ONLY_seed1'][frame_id]
        if before['matched'] and after['matched'] and before['frame_mean_px'] is not None and after['frame_mean_px'] is not None:
            eligible.append((after['frame_mean_px'] - before['frame_mean_px'], frame_id))
    eligible.sort()
    selection = [('largest_improvement_1', *eligible[0]), ('largest_improvement_2', *eligible[1]),
                 ('largest_worsening_1', *eligible[-1]), ('largest_worsening_2', *eligible[-2])]
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = []
    colors = {'R0': '#47a8ff', 'N0_BASE_REPLAY_seed1': '#ff9f32', 'N2_DIM_ONLY_seed1': '#36e06f'}
    labels = {'R0': 'R0', 'N0_BASE_REPLAY_seed1': 'N0: image/control',
              'N2_DIM_ONLY_seed1': 'N2: image + W,D,H'}
    for tag, delta, frame_id in selection:
        row = metadata[frame_id]
        image = cv2.cvtColor(cv2.imread(str(F.ROOT / row['image']['path'])), cv2.COLOR_BGR2RGB)
        annotation = read(F.ROOT / row['annotation']['path'])
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for axis, name in zip(axes, models):
            axis.imshow(image)
            draw_annotation(axis, annotation)
            draw_prediction(axis, selected_points(prediction_rows[name][frame_id]), colors[name])
            metric = metric_rows[name][frame_id]
            axis.set_title(f"{labels[name]}\nmean={metric['frame_mean_px']:.2f}px, matched={metric['matched']}",
                           color=colors[name], fontsize=11)
            axis.set_xlim(0, image.shape[1]); axis.set_ylim(image.shape[0], 0); axis.axis('off')
        figure.suptitle(f'{frame_id} | W x D x H = 1.10 x 1.10 x 0.15 m | N2-N0 = {delta:+.2f}px\n'
                       'magenta: manual click, white: PnP-derived annotation', fontsize=12)
        destination = FIGURES / f'{tag}_{frame_id}.png'
        figure.savefig(destination, dpi=150, facecolor='#111111')
        plt.close(figure)
        manifest.append(dict(tag=tag, id=frame_id, delta_frame_mean_px=float(delta),
            figure=F.binding(destination), N0=metric_rows['N0_BASE_REPLAY_seed1'][frame_id],
            N2=metric_rows['N2_DIM_ONLY_seed1'][frame_id]))
    F.freeze(DOC / 'GALLERY_MANIFEST.json', dict(selection='Seed1 N2 minus N0 frame_mean_px among jointly matched frames; two extrema each; outcome-selected diagnostic.',
        eligible_frames=len(eligible), records=manifest))
    lines = ['# 0918 정사각형 치수 입력 사례', '',
        '고정 seed1의 N2−N0 프레임 평균 오차 차이에서 개선 2장과 악화 2장을 함께 선택했다. '
        '결과를 보고 고른 진단용 그림이며 전체 성능 근거는 집계표다. 자홍색은 직접 클릭, 흰색은 PnP 생성 주석이다.', '']
    for record in manifest:
        label = '개선' if record['delta_frame_mean_px'] < 0 else '악화'
        path = (F.ROOT / record['figure']['path']).relative_to(DOC)
        lines += [f"## {label}: {record['id']} (N2−N0 {record['delta_frame_mean_px']:+.2f}px)", '',
                  f'![{record["id"]}]({path.as_posix()})', '']
    report = DOC / 'GALLERY.md'
    text = '\n'.join(lines).rstrip() + '\n'
    if report.exists() and report.read_text() != text:
        raise ValueError('Immutable gallery report differs')
    if not report.exists():
        report.write_text(text)
    print(report, flush=True)


if __name__ == '__main__':
    main()
