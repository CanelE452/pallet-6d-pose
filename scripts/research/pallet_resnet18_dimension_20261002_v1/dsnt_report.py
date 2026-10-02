"""Build the locked Korean report for the three-arm DSNT ResNet experiment.

This script performs no inference or training.  It requires completed
CONSTANT, SHAPE and FULL training receipts plus their frozen real/pose results,
verifies every referenced binding, and then writes two figures, a detailed
Korean report and an input/output manifest.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pallet-pose-matplotlib')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dsnt_diagnostic as D
import dsnt_train as T


ARMS = ('CONSTANT', 'SHAPE', 'FULL')
COLORS = {'CONSTANT': '#6b7280', 'SHAPE': '#2563eb', 'FULL': '#dc2626'}
OBJECT_LABELS = {
    'plastic_standard_110x130x11': 'Plastic 110x130x11 cm',
    'wood_small_80x59x14': 'Wood 80x59x14 cm',
    'plastic_standard_110x110x15': 'Square 110x110x15 cm',
}
COHORTS = (
    ('DEV319', 'DEV319 전체'),
    ('DEV319_OBJECT_plastic_standard_110x130x11', '직사각형 플라스틱 110×130×11 cm'),
    ('DEV319_OBJECT_wood_small_80x59x14', '직사각형 목재 80×59×14 cm'),
    ('GREEN150_MANUAL_DECLARED', 'GREEN150 정사각형·declared'),
    ('GREEN150_MANUAL_IN_FRAME', 'GREEN150 정사각형·in-frame'),
    ('GREEN0918_119_MANUAL_DECLARED', '0918 정사각형 119장·declared'),
    ('GREEN0918_119_MANUAL_IN_FRAME', '0918 정사각형 119장·in-frame'),
)
PRIMARY_FIGURE_COHORTS = (
    'DEV319',
    'DEV319_OBJECT_plastic_standard_110x130x11',
    'DEV319_OBJECT_wood_small_80x59x14',
    'GREEN150_MANUAL_DECLARED',
    'GREEN0918_119_MANUAL_DECLARED',
)
FIGURE_LABELS = {
    'DEV319': 'DEV319',
    'DEV319_OBJECT_plastic_standard_110x130x11': 'Plastic\n110x130',
    'DEV319_OBJECT_wood_small_80x59x14': 'Wood\n80x59',
    'GREEN150_MANUAL_DECLARED': 'GREEN150\nsquare',
    'GREEN0918_119_MANUAL_DECLARED': '0918-119\nsquare',
}
CALIBRATION_FIGURE = D.DOC / 'DSNT_CALIBRATION_CURVES.png'
REAL_FIGURE = D.DOC / 'DSNT_REAL_COMPARISON_V2.png'
REPORT = D.DOC / 'REPORT_KO_V2.md'
MANIFEST = D.DOC / 'DSNT_REPORT_MANIFEST_V2.json'


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f'{description} is required: {path}')


def same_binding(left: dict, right: dict, description: str) -> None:
    keys = ('path', 'sha256', 'bytes')
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError(f'Binding mismatch for {description}: {left} != {right}')


def finite(value, description: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f'Nonfinite value for {description}: {value}')
    return number


def verify_summary(summary: dict, description: str) -> None:
    required = ('total_frames', 'evaluable_frames', 'E_sym',
                'matched_pooled_corner8_median_px', 'matched_pooled_corner8_P90_px',
                'PCK', 'matched', 'coverage')
    missing = [key for key in required if key not in summary]
    if missing:
        raise ValueError(f'Missing summary keys for {description}: {missing}')
    total = int(summary['total_frames'])
    evaluable = int(summary['evaluable_frames'])
    if total <= 0 or not 0 < evaluable <= total:
        raise ValueError(f'Invalid population for {description}')
    for key in ('E_sym', 'coverage'):
        finite(summary[key], f'{description}.{key}')
    matched = int(summary['matched'])
    if not 0 <= matched <= total:
        raise ValueError(f'Invalid matched count for {description}')
    for key in ('matched_pooled_corner8_median_px', 'matched_pooled_corner8_P90_px'):
        value = summary[key]
        if matched:
            finite(value, f'{description}.{key}')
        elif value is not None:
            raise ValueError(f'Expected an empty matched-only statistic for {description}.{key}')
    for threshold in ('5', '10', '20'):
        finite(summary['PCK'][threshold], f'{description}.PCK{threshold}')


def cohort_entry(real: dict, cohort: str) -> dict:
    results = real['results']
    prefix = 'DEV319_OBJECT_'
    if cohort.startswith(prefix):
        object_type = cohort[len(prefix):]
        try:
            return results['DEV319_BY_OBJECT'][object_type]
        except KeyError as exc:
            raise ValueError(f'Missing DEV object cohort: {object_type}') from exc
    try:
        return results[cohort]
    except KeyError as exc:
        raise ValueError(f'Missing real cohort: {cohort}') from exc


def verify_curve(arm: str, curves: list[dict]) -> None:
    if len(curves) != T.EPOCHS or [row.get('epoch') for row in curves] != list(range(1, T.EPOCHS + 1)):
        raise ValueError(f'{arm} does not contain exactly epochs 1..{T.EPOCHS}')
    previous_step = 0
    for row in curves:
        if row.get('arm') != arm or int(row['step']) <= previous_step:
            raise ValueError(f'Invalid calibration curve ordering for {arm}')
        previous_step = int(row['step'])
        calibration = row['calibration']
        if int(calibration['frames']) != 1004 or int(calibration['supervised_points']) <= 0:
            raise ValueError(f'Unexpected calibration population for {arm}')
        for key in ('loss', 'matched_fraction'):
            finite(calibration[key], f'{arm}.calibration.{key}')
        for name in ('softargmax_network_px', 'argmax_network_px'):
            for key in ('median', 'p90'):
                finite(calibration[name][key], f'{arm}.{name}.{key}')


def load_arm(arm: str) -> dict:
    training_path = T.complete_path(arm)
    prediction_path = D.RAW / f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json'
    real_path = D.DOC / f'DSNT_FULL_{arm}_REAL_RESULTS.json'
    pose_path = D.DOC / f'DSNT_FULL_{arm}_POSE_RESULTS.json'
    for path, description in ((training_path, f'{arm} training receipt'),
                              (prediction_path, f'{arm} real predictions'),
                              (real_path, f'{arm} real result'),
                              (pose_path, f'{arm} pose result')):
        require_file(path, description)

    training = read(training_path)
    prediction = read(prediction_path)
    real = read(real_path)
    pose = read(pose_path)
    for payload, description in ((training, 'training'), (prediction, 'prediction'),
                                 (real, 'real'), (pose, 'pose')):
        if not payload.get('complete') or payload.get('arm') != arm:
            raise ValueError(f'Incomplete or wrong-arm {description} payload for {arm}')

    if not training.get('final_epoch_fixed') or int(training['epochs']) != T.EPOCHS:
        raise ValueError(f'{arm} is not a fixed final-epoch training receipt')
    if int(training['updates']) <= 0 or int(training['images_seen']) <= 0:
        raise ValueError(f'Invalid training counts for {arm}')
    for key in ('protocol', 'final_checkpoint', 'resume_checkpoint'):
        D.verify(training[key])
    same_binding(training['protocol'], D.bound(T.PROTOCOL), f'{arm} current protocol')
    verify_curve(arm, training['curves'])

    for key in ('checkpoint', 'training'):
        D.verify(prediction[key])
    for binding in prediction['datasets'].values():
        D.verify(binding)
    for binding in prediction['contracts'].values():
        D.verify(binding)
    same_binding(prediction['checkpoint'], training['final_checkpoint'], f'{arm} final checkpoint')
    same_binding(prediction['training'], D.bound(training_path), f'{arm} training receipt')
    expected_counts = {'DEV319': 319, 'GREEN150': 150, 'GREEN0918_119': 119}
    if prediction.get('inputs') != expected_counts:
        raise ValueError(f'Unexpected prediction population declaration for {arm}')
    for dataset, count in expected_counts.items():
        rows = prediction['predictions'].get(dataset)
        if rows is None or len(rows) != count or len({row['id'] for row in rows}) != count:
            raise ValueError(f'Incomplete prediction rows for {arm}/{dataset}')

    D.verify(real['checkpoint'])
    D.verify(real['predictions'])
    same_binding(real['checkpoint'], training['final_checkpoint'], f'{arm} scored checkpoint')
    same_binding(real['predictions'], D.bound(prediction_path), f'{arm} scored predictions')
    for binding in real['datasets'].values():
        D.verify(binding)
    support = real['dimension_support']
    for key in ('source_manifest', 'dimension_sidecar', 'normalization'):
        D.verify(support[key])
    if int(support['train_rows']) != 55980:
        raise ValueError(f'Unexpected dimension-support population for {arm}')
    for cohort, _ in COHORTS:
        entry = cohort_entry(real, cohort)
        verify_summary(entry['summary'], f'{arm}/{cohort}')
        if len(entry['rows']) != int(entry['summary']['total_frames']):
            raise ValueError(f'Row/summary count mismatch for {arm}/{cohort}')

    D.verify(pose['predictions'])
    same_binding(pose['predictions'], D.bound(prediction_path), f'{arm} pose predictions')
    if pose.get('physical_GT') is not False or pose.get('forced_single_pallet_output') is not True:
        raise ValueError(f'Unexpected pose interpretation contract for {arm}')
    if len(pose['rows']) != 319 or int(pose['summary']['frames']) != 319:
        raise ValueError(f'Unexpected pose population for {arm}')
    available = int(pose['summary']['available'])
    if not 0 <= available <= 319:
        raise ValueError(f'Invalid pose availability for {arm}')
    finite(pose['summary']['coverage'], f'{arm}.pose.coverage')
    finite(pose['summary']['ADDsym_AUC_full'], f'{arm}.pose.ADDsym_AUC_full')
    for metric in ('translation_cm', 'rotation_deg', 'yaw_deg', 'IoU3D'):
        values = pose['summary'][metric]
        if available:
            finite(values['median'], f'{arm}.pose.{metric}.median')
            finite(values['p90'], f'{arm}.pose.{metric}.p90')
        elif values != {'median': None, 'p90': None}:
            raise ValueError(f'Unexpected empty-pose summary for {arm}/{metric}')

    return dict(arm=arm, training_path=training_path, prediction_path=prediction_path,
        real_path=real_path, pose_path=pose_path, training=training,
        prediction=prediction, real=real, pose=pose)


def load_all() -> dict[str, dict]:
    arms = {arm: load_arm(arm) for arm in ARMS}
    reference_support = arms['CONSTANT']['real']['dimension_support']
    reference_ids = {}
    for cohort, _ in COHORTS:
        reference_ids[cohort] = [row['id'] for row in cohort_entry(arms['CONSTANT']['real'], cohort)['rows']]
    for arm in ARMS[1:]:
        if arms[arm]['real']['dimension_support'] != reference_support:
            raise ValueError(f'Dimension-support diagnostic differs for {arm}')
        for cohort, _ in COHORTS:
            ids = [row['id'] for row in cohort_entry(arms[arm]['real'], cohort)['rows']]
            if ids != reference_ids[cohort]:
                raise ValueError(f'Paired row order differs for {arm}/{cohort}')
    return arms


def paired(full_rows: list[dict], control_rows: list[dict]) -> dict:
    if [row['id'] for row in full_rows] != [row['id'] for row in control_rows]:
        raise ValueError('Paired comparison ID order differs')
    differences = []
    for full, control in zip(full_rows, control_rows):
        if not full.get('evaluable') or not control.get('evaluable'):
            continue
        differences.append(finite(full['frame_mean_px'], 'FULL frame mean') -
                           finite(control['frame_mean_px'], 'control frame mean'))
    values = np.asarray(differences, np.float64)
    if not len(values):
        raise ValueError('No evaluable paired rows')
    epsilon = 1e-9
    return dict(pairs=len(values), improved=int((values < -epsilon).sum()),
        harmed=int((values > epsilon).sum()), tied=int((np.abs(values) <= epsilon).sum()),
        mean_delta_px=float(values.mean()), median_delta_px=float(np.median(values)))


def paired_tables(arms: dict[str, dict]) -> dict:
    comparisons = {}
    for control in ('CONSTANT', 'SHAPE'):
        comparisons[control] = {}
        for cohort, _ in COHORTS:
            full_rows = cohort_entry(arms['FULL']['real'], cohort)['rows']
            control_rows = cohort_entry(arms[control]['real'], cohort)['rows']
            comparisons[control][cohort] = paired(full_rows, control_rows)
    return comparisons


def immutable_text(path: Path, value: str) -> None:
    if path.exists():
        if path.read_text() != value:
            raise ValueError(f'Immutable report differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(value)
    temporary.replace(path)


def immutable_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    figure.savefig(temporary, format='png', dpi=180, bbox_inches='tight',
                   metadata={'Software': 'pallet-pose dsnt_report.py'})
    plt.close(figure)
    if path.exists():
        if path.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError(f'Immutable figure differs: {path}')
        temporary.unlink()
    else:
        temporary.replace(path)


def calibration_figure(arms: dict[str, dict]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    specifications = (
        ('loss', 'Calibration loss', lambda row: row['calibration']['loss']),
        ('median', 'Softargmax median (network px)',
         lambda row: row['calibration']['softargmax_network_px']['median']),
        ('p90', 'Softargmax P90 (network px)',
         lambda row: row['calibration']['softargmax_network_px']['p90']),
        ('matched', 'Predicted-hull IoU>=0.5 (%)',
         lambda row: 100 * row['calibration']['matched_fraction']),
    )
    for axis, (_, title, getter) in zip(axes.flat, specifications):
        for arm in ARMS:
            curves = arms[arm]['training']['curves']
            axis.plot([row['epoch'] for row in curves], [getter(row) for row in curves],
                      marker='o', linewidth=2, markersize=4, color=COLORS[arm], label=arm)
        axis.set_title(title)
        axis.set_xlabel('Fixed epoch')
        axis.grid(alpha=.25)
        axis.set_xticks(range(1, T.EPOCHS + 1))
    axes[0, 0].legend(frameon=False, ncol=3)
    figure.suptitle('Source calibration (1,004 synthetic frames; epoch 10 fixed)', fontsize=14)
    immutable_figure(CALIBRATION_FIGURE, figure)


def grouped_bars(axis, categories: list[str], values: dict[str, list[float]], title: str,
                 ylabel: str) -> None:
    x = np.arange(len(categories), dtype=np.float64)
    width = .24
    for offset, arm in zip((-1, 0, 1), ARMS):
        axis.bar(x + offset * width, values[arm], width, color=COLORS[arm], label=arm)
    axis.set_xticks(x, categories)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis='y', alpha=.25)


def real_figure(arms: dict[str, dict]) -> None:
    labels = [FIGURE_LABELS[key] for key in PRIMARY_FIGURE_COHORTS]
    plot_value = lambda value: np.nan if value is None else float(value)
    medians = {arm: [plot_value(cohort_entry(arms[arm]['real'], key)['summary']
                     ['matched_pooled_corner8_median_px']) for key in PRIMARY_FIGURE_COHORTS]
               for arm in ARMS}
    p90 = {arm: [plot_value(cohort_entry(arms[arm]['real'], key)['summary']
                 ['matched_pooled_corner8_P90_px']) for key in PRIMARY_FIGURE_COHORTS]
           for arm in ARMS}
    translation = {arm: [plot_value(arms[arm]['pose']['summary']['translation_cm']['median']),
                         plot_value(arms[arm]['pose']['summary']['translation_cm']['p90'])] for arm in ARMS}
    rotation = {arm: [plot_value(arms[arm]['pose']['summary']['rotation_deg']['median']),
                      plot_value(arms[arm]['pose']['summary']['rotation_deg']['p90'])] for arm in ARMS}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    grouped_bars(axes[0, 0], labels, medians, 'Real 2D matched-corner median', 'pixels')
    grouped_bars(axes[0, 1], labels, p90, 'Real 2D matched-corner P90', 'pixels')
    grouped_bars(axes[1, 0], ['Median', 'P90'], translation,
                 'DEV319 reconstructed translation', 'cm')
    grouped_bars(axes[1, 1], ['Median', 'P90'], rotation,
                 'DEV319 reconstructed rotation', 'degrees')
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    figure.suptitle('Frozen real-development comparison (lower is better)', fontsize=14, y=.985)
    figure.legend(handles, labels_legend, loc='upper center', ncol=3, frameon=False,
                  bbox_to_anchor=(.5, .955))
    figure.subplots_adjust(left=.07, right=.99, bottom=.08, top=.88, hspace=.34, wspace=.18)
    immutable_figure(REAL_FIGURE, figure)


def number(value, digits: int = 3) -> str:
    if value is None:
        return '—'
    return f'{float(value):.{digits}f}'


def percent(value, digits: int = 2) -> str:
    if value is None:
        return '—'
    return f'{100 * float(value):.{digits}f}%'


def axis_names(indices: list[int]) -> str:
    names = ('W', 'D', 'H')
    return ', '.join(names[index] for index in indices) if indices else '없음'


def comparison_sentence(arms: dict[str, dict], control: str) -> str:
    fragments = []
    for cohort in PRIMARY_FIGURE_COHORTS:
        full = cohort_entry(arms['FULL']['real'], cohort)['summary']
        base = cohort_entry(arms[control]['real'], cohort)['summary']
        left = full['matched_pooled_corner8_median_px']
        right = base['matched_pooled_corner8_median_px']
        delta = None if left is None or right is None else left - right
        formatted = '계산 불가' if delta is None else f'{delta:+.3f}px'
        fragments.append(f"{FIGURE_LABELS[cohort].replace(chr(10), ' ')} {formatted}")
    pose_full = arms['FULL']['pose']['summary']
    pose_base = arms[control]['pose']['summary']
    def delta_text(metric: str, suffix: str) -> str:
        left = pose_full[metric]['median']; right = pose_base[metric]['median']
        return '계산 불가' if left is None or right is None else f'{left - right:+.3f}{suffix}'
    return (f"FULL−{control} 2D median: " + ', '.join(fragments) +
            f"; DEV319 pose median ΔT={delta_text('translation_cm', 'cm')}, "
            f"ΔR={delta_text('rotation_deg', '°')}.")


def stable_improvement(arms: dict[str, dict], control: str) -> bool:
    for cohort in PRIMARY_FIGURE_COHORTS:
        full = cohort_entry(arms['FULL']['real'], cohort)['summary']
        base = cohort_entry(arms[control]['real'], cohort)['summary']
        for metric in ('matched_pooled_corner8_median_px', 'matched_pooled_corner8_P90_px'):
            if full[metric] is None or base[metric] is None or not full[metric] < base[metric]:
                return False
    for metric in ('translation_cm', 'rotation_deg'):
        for statistic in ('median', 'p90'):
            left = arms['FULL']['pose']['summary'][metric][statistic]
            right = arms[control]['pose']['summary'][metric][statistic]
            if left is None or right is None or not left < right:
                return False
    return True


def report_text(arms: dict[str, dict], comparisons: dict) -> str:
    lines = [
        '# 치수 조건부 ResNet18 DSNT 3-arm 결과', '',
        '이 보고서는 동일한 ResNet18 구조와 학습 순서에서 입력만 `CONSTANT`, `SHAPE`, `FULL`로 바꾼 '
        '고정 epoch 10 결과를 비교한다. `FULL`은 이미지와 canonical W·D·H 및 두 비율을 함께 받고, '
        '`SHAPE`은 두 비율만, `CONSTANT`는 치수 문맥을 0으로 받는다. 세 arm의 학습 완료 영수증, '
        '전수 예측, 2D 결과와 pose 결과의 SHA-256 바인딩을 이 보고서 생성 시 다시 검증했다.', '',
        '![합성 calibration 곡선](DSNT_CALIBRATION_CURVES.png)', '',
        '![실사 2D 및 pose 비교](DSNT_REAL_COMPARISON_V2.png)', '',
        '## 실행 및 고정 상태', '',
        '| arm | epochs | updates | images seen | elapsed min | final checkpoint SHA-256 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for arm in ARMS:
        receipt = arms[arm]['training']
        lines.append(f"| {arm} | {receipt['epochs']} | {receipt['updates']} | {receipt['images_seen']} | "
                     f"{receipt['elapsed_seconds'] / 60:.2f} | `{receipt['final_checkpoint']['sha256']}` |")

    lines += ['', '실사 이미지는 학습 또는 checkpoint 선택에 사용하지 않았다. 세 arm 모두 동일한 합성 '
              'TRAIN 55,980장과 calibration 1,004장을 사용했고 최종 epoch를 사전에 고정했다.', '',
              '## 최종 합성 calibration', '',
              '| arm | train loss | calibration loss | soft median px | soft P90 px | argmax median px | matched/1004 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        curve = arms[arm]['training']['curves'][-1]
        calibration = curve['calibration']
        lines.append(f"| {arm} | {number(curve['training_loss'], 5)} | {number(calibration['loss'], 5)} | "
                     f"{number(calibration['softargmax_network_px']['median'])} | "
                     f"{number(calibration['softargmax_network_px']['p90'])} | "
                     f"{number(calibration['argmax_network_px']['median'])} | "
                     f"{calibration['matched_frames']}/1004 ({percent(calibration['matched_fraction'])}) |")

    lines += ['', '## 실사 2D 결과', '',
              '중앙값과 P90은 predicted-hull IoU≥0.5로 matched 된 관측 corner를 모아 계산한다. PCK와 '
              '`E_sym`에는 미검출·unmatched penalty가 포함된다. 이 ResNet은 한 팔레트가 있다고 가정해 항상 '
              '9점을 출력하므로 여기의 coverage는 detector recall이 아니라 predicted-hull match 비율이다.', '',
              '| 집단 | arm | frames | median px | P90 px | PCK5 | PCK10 | PCK20 | matched | E_sym |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cohort, label in COHORTS:
        for arm in ARMS:
            summary = cohort_entry(arms[arm]['real'], cohort)['summary']
            lines.append(f"| {label} | {arm} | {summary['total_frames']} | "
                         f"{number(summary['matched_pooled_corner8_median_px'])} | "
                         f"{number(summary['matched_pooled_corner8_P90_px'])} | "
                         f"{percent(summary['PCK']['5'])} | {percent(summary['PCK']['10'])} | "
                         f"{percent(summary['PCK']['20'])} | {summary['matched']}/{summary['total_frames']} | "
                         f"{number(summary['E_sym'], 5)} |")

    lines += ['', 'GREEN의 `declared` 행은 visible로 선언된 수동점을 유지하며 이미지 밖 수동점도 오차 '
              '분모에 포함한다. `in-frame` 행은 그 점을 제외한 민감도 결과다. 두 경우 모두 matching hull은 '
              '기존 계약과 같이 all-known in-image annotation point로 만든다.', '',
              '## FULL 대조군 paired 결과', '',
              '각 프레임의 symmetry-aware `frame_mean_px`를 같은 ID끼리 뺀 결과다. 개선은 FULL 오차가 '
              '대조군보다 작은 프레임, 악화는 큰 프레임이다. 통계적 신뢰구간이나 독립 반복 seed는 없다.', '',
              '| 대조군 | 집단 | pairs | 개선 | 악화 | 동률 | 평균 Δpx | 중앙 Δpx |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for control in ('CONSTANT', 'SHAPE'):
        for cohort, label in COHORTS:
            row = comparisons[control][cohort]
            lines.append(f"| {control} | {label} | {row['pairs']} | {row['improved']} | "
                         f"{row['harmed']} | {row['tied']} | {row['mean_delta_px']:+.3f} | "
                         f"{row['median_delta_px']:+.3f} |")

    lines += ['', '## DEV319 pose 결과', '',
              'T·R은 기존 DEV319의 재구성 reference와 등록 치수, 동일 SQPnP/LM 계약으로 계산했다. '
              '물리 계측 GT가 아니다. 모든 점을 강제로 출력하므로 coverage는 PnP solver가 유효 해를 반환한 '
              '비율이며 YOLO·DOPE의 detector coverage와 직접 같은 뜻이 아니다.', '',
              '| arm | PnP available | ADDsym AUC full | T median cm | T P90 cm | R median ° | R P90 ° | yaw median ° | IoU3D median |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        summary = arms[arm]['pose']['summary']
        lines.append(f"| {arm} | {summary['available']}/319 ({percent(summary['coverage'])}) | "
                     f"{number(summary['ADDsym_AUC_full'], 5)} | "
                     f"{number(summary['translation_cm']['median'])} | {number(summary['translation_cm']['p90'])} | "
                     f"{number(summary['rotation_deg']['median'])} | {number(summary['rotation_deg']['p90'])} | "
                     f"{number(summary['yaw_deg']['median'])} | {number(summary['IoU3D']['median'], 4)} |")

    support = arms['FULL']['real']['dimension_support']
    minimum = support['canonical_WDH_m']['minimum']
    maximum = support['canonical_WDH_m']['maximum']
    lines += ['', '## 치수 입력 지원 범위', '',
              f"합성 TRAIN 55,980장의 axis-aligned W·D·H 범위는 "
              f"W {minimum[0]:.3f}–{maximum[0]:.3f}m, D {minimum[1]:.3f}–{maximum[1]:.3f}m, "
              f"H {minimum[2]:.3f}–{maximum[2]:.3f}m이다. 범위 안이라는 사실만으로 분포가 같다는 뜻은 아니다.", '',
              '| 물체 | canonical W,D,H m | normalized 5D context | TRAIN WDH 범위 밖 축 |',
              '|---|---|---|---|']
    for row in support['real']:
        dims = ', '.join(f'{value:.3f}' for value in row['canonical_WDH_m'])
        context = ', '.join(f'{value:+.3f}' for value in row['normalized_context'])
        lines.append(f"| {OBJECT_LABELS.get(row['object_type'], row['object_type'])} | {dims} | "
                     f"{context} | {axis_names(row['outside_TRAIN_canonical_axes'])} |")

    lines += ['', '목재 80×59×14cm의 D=0.59m가 합성 TRAIN의 D 최소값보다 작으면 FULL의 목재 결과는 '
              '외삽 결과다. 이 경우 직사각형 전체 평균만으로 치수 입력의 일반적 효과를 주장해서는 안 된다.', '',
              '## 보수적 해석', '',
              f"- {comparison_sentence(arms, 'CONSTANT')}",
              f"- {comparison_sentence(arms, 'SHAPE')}",
    ]
    constant_pass = stable_improvement(arms, 'CONSTANT')
    shape_pass = stable_improvement(arms, 'SHAPE')
    if constant_pass and shape_pass:
        lines.append('- 이번 고정 지표 집합에서는 FULL이 두 대조군보다 모든 주요 2D median/P90과 T·R '
                     'median/P90에서 낮다. 다만 단일 seed와 재사용 DEV 결과이므로 안정적 일반화나 인과 효과로 확대하지 않는다.')
    else:
        failed = ', '.join(control for control, passed in
                           (('CONSTANT', constant_pass), ('SHAPE', shape_pass)) if not passed)
        lines.append(f'- FULL은 {failed} 대비 모든 주요 2D 및 T·R 지표를 동시에 개선하지 못했다. 따라서 '
                     '이 결과만으로 치수 입력이 T·R을 안정적으로 함께 개선한다고 주장할 수 없다.')
    lines += [
        '- GREEN150과 0918 119장은 각 집단 안에서 W·D·H가 모두 1.10×1.10×0.15m로 일정하다. '
        '따라서 정사각형 결과는 모델 전체의 전이를 보여주지만 프레임별 치수 변화의 인과 효과를 식별하지 못한다.',
        '- 이 실험은 full-image 직접 ResNet 추정기다. 기존 YOLO/DOPE 출력 뒤의 동일 보정 모듈을 '
        '검증한 실험이 아니므로 backbone-agnostic 보정기 근거와 구분해야 한다.',
        '- 실사 세 집단은 독립 test가 아니라 재사용 development 자료다. GREEN 주석의 QA 상태와 '
        'DEV pose reference의 재구성 한계를 유지한다.',
        '- 이 실행에는 latency 측정이 없다. 속도·정확도 비교는 같은 장치, batch, warm-up, 전처리·후처리 '
        '범위를 고정한 별도 측정이 필요하다.', '',
        '## 산출물과 재현 경로', '',
        '- [Calibration figure](DSNT_CALIBRATION_CURVES.png)',
        '- [Real comparison figure](DSNT_REAL_COMPARISON_V2.png)',
        '- [Report manifest](DSNT_REPORT_MANIFEST_V2.json)',
        '- 각 arm의 전수 2D 행과 pose 행은 같은 폴더의 `DSNT_FULL_<ARM>_REAL_RESULTS.json` 및 '
        '`DSNT_FULL_<ARM>_POSE_RESULTS.json`에 있다.', '',
    ]
    return '\n'.join(lines)


def generate() -> None:
    arms = load_all()
    comparisons = paired_tables(arms)
    calibration_figure(arms)
    real_figure(arms)
    immutable_text(REPORT, report_text(arms, comparisons))
    manifest = dict(schema='resnet18_dimension_dsnt_three_arm_report_v2', complete=True,
        arms=list(ARMS), generator=D.bound(Path(__file__)), protocol=D.bound(T.PROTOCOL),
        inputs={arm: dict(training=D.bound(arms[arm]['training_path']),
                          predictions=D.bound(arms[arm]['prediction_path']),
                          real_results=D.bound(arms[arm]['real_path']),
                          pose_results=D.bound(arms[arm]['pose_path'])) for arm in ARMS},
        paired_FULL_vs_controls=comparisons,
        outputs=dict(report=D.bound(REPORT), calibration_figure=D.bound(CALIBRATION_FIGURE),
                     real_figure=D.bound(REAL_FIGURE)),
        interpretation=dict(independent_confirmation=False, physical_pose_GT=False,
            square_dimension_effect_identifiable=False, latency_measured=False,
            forced_single_pallet_output=True, one_training_seed_per_arm=True))
    D.write(MANIFEST, manifest)
    print('DSNT_THREE_ARM_REPORT_COMPLETE', REPORT, flush=True)


if __name__ == '__main__':
    generate()
