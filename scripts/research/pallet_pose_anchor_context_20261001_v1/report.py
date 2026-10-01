"""Publish the fixed189 ablation using completed, frozen result artifacts only.

This module never imports an evaluator, label loader, or trainer. It exports
cached error arrays and traces, copies checkpoint bytes, and plots summaries.
"""
from . import common as C
import csv
import io
import json
import numpy as np


def csv_export(name, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    C.save(C.DOC / name, buffer.getvalue())
    return len(rows)


def plot_results(source, previous, training, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.DOC / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    labels = ['R0_ONLY', 'UNION s1', 'UNION s2', 'UNION s3']
    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    specs = [('translation_cm', 'median', 'T median (cm)'), ('rotation_deg', 'median', 'R median (deg)'),
             ('translation_cm', 'P90', 'T P90 (cm)'), ('rotation_deg', 'P90', 'R P90 (deg)')]
    values = {}
    for ax, (metric, q, title) in zip(axes.flat, specs):
        old = [previous['summaries'][m]['full_population'][metric][q] for m in C.MODEL_NAMES]
        new = [source['summaries'][m]['full_population'][metric][q] for m in C.MODEL_NAMES]
        values[f'{metric}_{q}'] = dict(previous94=old, current189=new)
        colors = ['#0d9488'] * 4
        if metric == 'translation_cm' and q == 'P90':
            colors[-1] = '#dc2626'
        ax.bar(x - .2, old, .4, label='Previous anchored94', color='#64748b')
        ax.bar(x + .2, new, .4, label='Fixed anchor-context189', color=colors)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylim(0, max(old + new) * 1.24)
        ax.grid(axis='y', alpha=.2)
        for j in range(4):
            ax.text(j - .2, old[j] + max(old + new)*.01, f'{old[j]:.3f}', ha='center', fontsize=8)
            ax.text(j + .2, new[j] + max(old + new)*.01, f'{new[j]:.3f}', ha='center', fontsize=8)
        if metric == 'translation_cm' and q == 'P90':
            limit = source['summaries']['R0_GEO']['full_population'][metric][q] * 1.05
            ax.axhline(limit, color='#dc2626', linestyle='--', linewidth=1.2,
                       label=f'R0 GEO x 1.05 = {limit:.3f} cm')
            ax.legend(loc='upper left', fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Actual learned selectors | source VAL1024 | same anchored targets\n43/45 source checks passed; learned real evaluation not run')
    path = folder / 'source_val_feature_comparison.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    artifacts = [C.bind(path)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, model in zip(axes.flat, C.MODEL_NAMES):
        rows = [r for r in traces if r['model'] == model]
        ax.plot([r['call'] for r in rows], [r['objective'] for r in rows], color='#0d9488')
        ax.set_title(model)
        ax.set_xlabel('Objective evaluations (solver calls)')
        ax.set_ylabel('Full TRAIN CE + ridge')
        ax.grid(alpha=.2)
    fig.suptitle('Four actual deterministic fits | fixed189 input map\nNumerical convergence certifies this objective, not T/R improvement')
    path = folder / 'training_objectives.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    artifacts.append(C.bind(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    before = [training['models'][m]['previous_converged_same_target']['statistics'] for m in C.MODEL_NAMES]
    after = [training['models'][m]['statistics'] for m in C.MODEL_NAMES]
    accuracy_old = [s['target_accuracy_full_population'] * 100 for s in before]
    accuracy_new = [s['target_accuracy_full_population'] * 100 for s in after]
    unsafe_old = [s['anchor_violations']['either']['count'] for s in before]
    unsafe_new = [s['anchor_violations']['either']['count'] for s in after]
    for ax, old, new, title, unit in (
        (axes[0], accuracy_old, accuracy_new, 'Exact TRAIN target selected', 'Percent of all2598 frames'),
        (axes[1], unsafe_old, unsafe_new, 'Either T or R exceeds the R0 anchor', 'Number of frames')):
        ax.bar(x - .2, old, .4, label='Previous anchored94', color='#64748b')
        ax.bar(x + .2, new, .4, label='Fixed anchor-context189', color='#0d9488')
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(0, 104 if ax is axes[0] else max(old + new) * 1.2)
        for j in range(4):
            fmt = '.2f' if ax is axes[0] else '.0f'
            ax.text(j - .2, old[j] + max(old + new)*.01, format(old[j], fmt), ha='center', fontsize=8)
            ax.text(j + .2, new[j] + max(old + new)*.01, format(new[j], fmt), ha='center', fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle('TRAIN diagnosis only | same targets, original runtime candidates\n2598 frames retained; one all-invalid frame is a separate failure')
    path = folder / 'train_target_and_anchor_comparison.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    artifacts.append(C.bind(path))
    values['train_target_accuracy_percent'] = dict(previous94=accuracy_old, current189=accuracy_new, denominator=2598)
    values['train_any_axis_anchor_violation_count'] = dict(previous94=unsafe_old, current189=unsafe_new,
                                                          frames=2598, available_anchor_rows=2597, failed_rows=1)
    return artifacts, values


def main():
    protocol = C.protocol('TRAIN_PROTOCOL')
    source = C.read(C.DOC / 'SOURCE_VAL_GATE.json')
    complete = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    training = C.read(C.DOC / 'TRAIN_CONVERGENCE.json')
    risk = C.read(C.DOC / 'TRAIN_RISK_DIAGNOSTIC.json')
    prefit = C.read(C.DOC / 'PREFIT_REVIEW.json')
    source_review = C.read(C.DOC / 'SOURCE_VAL_VERIFICATION.json')
    not_run = C.read(C.DOC / 'REAL_EVALUATION_NOT_RUN.json')
    previous = C.read(C.ANCHOR_DOC / 'SOURCE_VAL_GATE.json')
    assert training['complete'] and training['PASS'] and training['independent_check_PASS']
    assert risk['complete'] and risk['PASS'] and risk['source_TRAIN_only']
    assert not risk['VAL_quality_read'] and not risk['real_targets_read'] and risk['new_fits'] == 0
    assert source_review['complete'] and source_review['PASS']
    assert source_review['source_gate'] == C.bind(C.DOC / 'SOURCE_VAL_GATE.json')
    assert prefit['complete'] and prefit['PASS']
    assert complete['complete'] and complete['all_certified'] and complete['fit_count'] == 4
    assert source['complete'] and not source['PASS'] and source['checks_passed'] == 43 and source['checks_total'] == 45
    assert previous['checks_passed'] == 38 and previous['checks_total'] == 45
    assert source['failed_checks'] == ['UNION_s3/R0_ONLY/translation_cm_P90_guard', 'UNION_s3/R0_GEO/translation_cm_P90_guard']
    assert not_run['learned_real_routes'] == not_run['real_metric_calls'] == not_run['raw_real_reference_reads'] == 0
    for path, absent in not_run['absence_checks'].items():
        assert absent and not (C.ROOT / path).exists(), path
    assert protocol['feature_dim'] == 189 and protocol['raw_feature_dim'] == 94
    assert protocol['feature_map'] == 'normalized94_abs_anchor_delta94_identity1'
    # Check the qualitative comparisons used below directly against frozen
    # summaries; equal displayed medians must not be described as regressions.
    for model in C.MODEL_NAMES[1:]:
        old_train = training['models'][model]['previous_converged_same_target']['statistics']
        new_train = training['models'][model]['statistics']
        for axis in ('translation_cm', 'rotation_deg'):
            assert new_train['selected_error_summary']['full_population'][axis]['median'] > old_train['selected_error_summary']['full_population'][axis]['median']
        old_val, new_val = previous['summaries'][model]['full_population'], source['summaries'][model]['full_population']
        assert new_val['rotation_deg']['P90'] < old_val['rotation_deg']['P90']
        if model == 'UNION_s3':
            assert new_val['translation_cm']['median'] == old_val['translation_cm']['median']
        else:
            assert new_val['translation_cm']['median'] > old_val['translation_cm']['median']
        assert risk['models'][model]['target']['classes']['safe_equal']['count'] == 0
    for model in ('UNION_s1', 'UNION_s3'):
        assert source['summaries'][model]['full_population']['translation_cm']['P90'] > previous['summaries'][model]['full_population']['translation_cm']['P90']
    for name in ('TRAIN_CONVERGENCE_KO.md', 'SOURCE_VAL_VERIFICATION_KO.md', 'REAL_EVALUATION_NOT_RUN_KO.md',
                 'TRAIN_RISK_DIAGNOSTIC_KO.md'):
        assert (C.DOC / name).exists(), name

    C.verify(source['metrics'])
    with np.load(C.ROOT / source['metrics']['path'], allow_pickle=False) as stored:
        ids, models = stored['ids'].tolist(), stored['models'].tolist()
        errors = {m: stored[m].copy() for m in models}
    route = C.read(C.DOC / 'SOURCE_VAL_ROUTING_LOCK.json')
    C.verify(route['choices'])
    choices = C.read(C.ROOT / route['choices']['path'])
    val_rows = []
    for model in models:
        assert errors[model].shape == (1024, 2) and np.isfinite(errors[model]).all()
        for j, fid in enumerate(ids):
            choice = choices['records'].get(model, {}).get(fid, {})
            val_rows.append(dict(model=model, id=fid, split='VAL', available=True,
                                 T_cm=float(errors[model][j, 0]), R_deg=float(errors[model][j, 1]),
                                 candidate=choice.get('candidate_name', 'fixed_GEO'), fallback=choice.get('fallback', False)))
    checks = [dict(model=m, baseline=b, criterion=k, PASS=v) for m, group in source['comparisons'].items()
              for b, comp in group.items() for k, v in comp['checks'].items()]
    receipts, traces, exports = {}, [], {}
    for model in C.MODEL_NAMES:
        receipt = C.read(C.DOC / f'FIT_{model}.json')
        receipts[model] = receipt
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        assert receipt['certificate']['PASS'] and receipt['fits_executed'] == 1
        log = [json.loads(line) for line in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        traces.extend(dict(model=model, **row) for row in log if row['event'] == 'objective')
        path = C.DOC / 'model_parameters' / f'{model}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        original = C.ROOT / receipt['checkpoint']['path']
        with path.open('xb') as handle:
            handle.write(original.read_bytes())
        assert path.read_bytes() == original.read_bytes()
        exports[model] = dict(local=receipt['checkpoint'], published=C.bind(path))
    assert len(traces) == complete['total_objective_calls'] == 1495
    iterations = sum(r['iterations'] for r in receipts.values())
    assert iterations == 1350
    csv_counts = {name: csv_export(name, rows) for name, rows in (
        ('SOURCE_VAL_FRAME_RESULTS.csv', val_rows), ('SOURCE_VAL_CHECKS.csv', checks),
        ('TRAINING_OBJECTIVE_LOG.csv', traces))}
    assert csv_counts == {'SOURCE_VAL_FRAME_RESULTS.csv': 8192, 'SOURCE_VAL_CHECKS.csv': 45,
                          'TRAINING_OBJECTIVE_LOG.csv': 1495}
    artifacts, figure_values = plot_results(source, previous, training, traces)
    prior_report = C.read(C.ANCHOR_DOC / 'REPORT_DATA.json')
    reused = [b for b in prior_report['artifacts'] if '/real_rgb_dimensions_' in b['path']]
    assert len(reused) == 3 and prior_report['actual_RGB_images'] == 6
    for binding in reused:
        C.verify(binding)
    seed3_t90 = source['summaries']['UNION_s3']['full_population']['translation_cm']['P90']
    r0_t90 = source['summaries']['R0_GEO']['full_population']['translation_cm']['P90']
    assert r0_t90 == source['summaries']['R0_ONLY']['full_population']['translation_cm']['P90']
    limit, ratio = 1.05 * r0_t90, seed3_t90 / r0_t90
    assert seed3_t90 > limit
    C.save(C.DOC / 'REPORT_DATA.json', dict(complete=True, code=C.bind(C.HERE / 'report.py'),
        train_protocol=C.bind(C.DOC / 'TRAIN_PROTOCOL.json'), source_gate=C.bind(C.DOC / 'SOURCE_VAL_GATE.json'),
        previous_source_gate=C.bind(C.ANCHOR_DOC / 'SOURCE_VAL_GATE.json'),
        training_verification=C.bind(C.DOC / 'TRAIN_CONVERGENCE.json'),
        training_risk_diagnostic=C.bind(C.DOC / 'TRAIN_RISK_DIAGNOSTIC.json'),
        source_verification=C.bind(C.DOC / 'SOURCE_VAL_VERIFICATION.json'),
        prefit_verification=C.bind(C.DOC / 'PREFIT_REVIEW.json'),
        real_not_run=C.bind(C.DOC / 'REAL_EVALUATION_NOT_RUN.json'),
        actual_new_fits=4, objective_calls=len(traces), iterations=iterations,
        fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        source_VAL_checks_passed=43, source_VAL_checks_total=45, previous_source_VAL_checks_passed=38,
        failed_checks=source['failed_checks'], learned_real_evaluated=False,
        real_oracle_diagnostic_evaluated=False, stable_joint_improvement_achieved=False,
        method_success=False, goal_complete=False,
        feature_map=protocol['feature_map'], raw_feature_dim=94, feature_dim=189,
        csv_rows=csv_counts, exports=exports, artifacts=artifacts, reused_figures=reused,
        reused_image_report=C.bind(C.ANCHOR_DOC / 'REPORT_DATA.json'), reused_actual_RGB_images=6,
        reused_image_scope='Previous reference-derived oracle gallery; not this learned189 method on real images.',
        figure_values=figure_values,
        seed3_T_P90_failure=dict(value_cm=seed3_t90, R0_GEO_cm=r0_t90, R0_ONLY_cm=r0_t90,
            limit_cm=limit, ratio_to_baseline=ratio, percent_above_baseline=100*(ratio-1), permitted_percent=5),
        target_and_mask_changed=False, new_reference_metric_calculations_by_report=0,
        new_fits_by_report=0, new_image_forwards_by_report=0))

    lines = ['# R0 기준 후보 관계를 입력에 추가한 189차원 선택기', '',
        '**실제 모델의 안정적인 T·R 동시 개선은 아직 달성하지 못했습니다.**', '',
        '같은 학습 정답과 후보를 유지하고, 각 후보와 기존 R0 운영 후보의 관계를 입력에 추가했습니다. 네 선택기를 실제 학습한 뒤 합성 source VAL1024장에서 확인한 결과는 **43/45 통과·2개 실패**입니다. 직전94차원 방법의38/45보다 통과 수는 늘었지만, seed3의 T P90 비회귀 조건이 두 비교에서 실패해 이 방법의 learned 실사 평가는 실행하지 않았습니다.', '',
        'T는 위치 오차(cm), R은 회전 오차(°)이며 작을수록 좋습니다. 입력은 **RGB 이미지 한 장 + 팔레트 치수**, 기존 카메라 보정 K입니다. 새 영상·센서·실사 정답을 추가하지 않았습니다.', '',
        '## 이번에 바꾼 입력', '',
        '각 후보의 기존94개 특징을 같은 TRAIN 평균·표준편차로 FP32 정규화하고 FP64로 바꾼 값을 `z`라 할 때, 새 입력은 다음189개입니다.', '',
        '```text', '[ z(94), abs(z - z_R0운영후보)(94), 이 후보가 R0운영후보인가(1) ]', '```', '',
        'anchor는 이미지 입력으로 정해진 기존 R0 GEO 후보입니다. 학습된 R0_ONLY가 고른 후보나 정답 오차로 고른 후보를 anchor로 사용하지 않습니다. 절댓값 차이 뒤에 별도 정규화는 없습니다. 유효 후보가 있는데 R0 anchor가 없으면 계약 오류로 중단하고, 모든 후보가 실패한 행은 그대로 유지합니다.', '',
        '직전 [양축 보존 타깃94 방법](../pallet_pose_pareto_anchor_20261001_v1/REPORT_KO.md)과 **학습 정답·safe mask·원래 유효 후보·정규화·TRAIN2598행·CE+ridge(λ=1e−4)·초깃값0·solver·판정 기준을 동일하게 유지**했습니다. 변경한 것은 입력 표현94→189 하나입니다. 기존94개 가중치 뒤에95개의0을 붙이는 동등성을 검산했으며, 합성 prefit fixture의 점수 차이 최대는1.78e−15였습니다. 합산 순서 차이가 있으므로 score의 byte 일치를 주장하지 않습니다.', '',
        '학습 정답에는 기존 `T≤R0 anchor T AND R≤R0 anchor R` 조건과 고정 TRAIN 비용을 유지했습니다. **이 조건은 추론 시 후보를 제거하는 마스크가 아닙니다.** 추론은 정답 오차를 모르며, 원래 유효 후보 전체에서 점수가 가장 작은 전체 pose 하나를 고릅니다. 따라서 새 입력도 양축 비회귀를 보장하지 않습니다.', '',
        '## 실제 네 학습과 TRAIN 검산', '',
        '네 모델 모두 목적함수 수렴 인증을 통과했습니다. 각 fit의 한계1000 iterations·2000 objective calls와 `||gradient||²/(2λ)≤1e−6` 조건을 변경하지 않았습니다. PyTorch FP64 autograd와 별도 행별189개 특징 구현으로 검산했습니다.', '',
        '| 모델 | optimizer iterations | objective calls | 수렴 gap 상한 |', '|---|---:|---:|---:|']
    for model, receipt in receipts.items():
        lines.append(f"| {model} | {receipt['iterations']} | {receipt['objective_calls']} | {receipt['certificate']['gradient_l2_squared_over_2lambda']:.3e} |")
    lines += ['', f'총{len(traces)}회 목적함수 계산과 {iterations}회 optimizer iteration을 실행했습니다. 이 인증은 CE+ridge의 수치 수렴을 뜻하며 T/R 개선 인증이 아닙니다.', '',
        '![네 모델의 실제 학습 목적함수](figures/training_objectives.png)', '',
        '| 동일 TRAIN에서94→189 비교 | 타깃 선택 정확도 | R0보다 한 축 이상 악화한 수 |', '|---|---:|---:|']
    for model in C.MODEL_NAMES:
        record = training['models'][model]
        old, new = record['previous_converged_same_target']['statistics'], record['statistics']
        assert old['frames'] == new['frames'] == 2598 and old['failed_rows'] == new['failed_rows'] == 1
        assert old['available_anchor_rows'] == new['available_anchor_rows'] == 2597
        lines.append(f"| {model} | {old['target_accuracy_full_population']*100:.2f}% → {new['target_accuracy_full_population']*100:.2f}% | {old['anchor_violations']['either']['count']} → {new['anchor_violations']['either']['count']} |")
    lines += ['', '정확도 비율의 분모는 전체2598장입니다. 후보 실패1장은 정답으로 세지 않으며, 위반 여부도 평가할 수 없는 별도 실패로 남깁니다. 위반 수는 유효 anchor2597장 중에서 센 값이고, 전체 T/R 집계에는 실패1개를 +∞로 유지합니다. 조건부2597장 기준 비율도 [TRAIN 검산 JSON](TRAIN_CONVERGENCE.json)에 따로 제공합니다.', '',
        '![TRAIN 타깃 정확도와 anchor 악화 선택](figures/train_target_and_anchor_comparison.png)', '',
        '타깃 선택 정확도는 높아지고 R0보다 악화하는 선택은 줄었습니다. 그러나 UNION 세 모델의 TRAIN T·R 중앙값은 직전94 방법보다 모두 커졌습니다. 타깃 정답률이나 비회귀 선택 수 개선을 연속적인 위치·회전 오차 개선과 동일시하지 않습니다.', '',
        '## 합성 source VAL1024: 43/45 통과, 두 조건 실패', '',
        '학습 종료 후 네 모델의4×1024개 선택을 source 정답 조회 전에 고정했습니다. 모든1024행을 유지해 같은 T/C2 회전 오차를 계산했고, 8개 모델의 pose 실패는 모두0입니다. 실패가0이라는 사실은 pose가 정확하다는 뜻이 아닙니다.', '',
        '| 모델 | T 중앙값(cm) | R 중앙값(°) | T P90(cm) | R P90(°) | pose 실패 |', '|---|---:|---:|---:|---:|---:|']
    for model, summary in source['summaries'].items():
        population = summary['full_population']
        numbers = [population[m][q] for m, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        lines.append('| ' + model + ' | ' + ' | '.join(f'{n:.6f}' for n in numbers) + f" | {summary['failed_pose']} |")
    lines += ['', '| 같은 source VAL의94→189 비교 | T 중앙값(cm) | R 중앙값(°) | T P90(cm) | R P90(°) |', '|---|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES:
        old, new = previous['summaries'][model]['full_population'], source['summaries'][model]['full_population']
        pairs = [f'{old[m][q]:.6f} → {new[m][q]:.6f}' for m, q in
                 [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        lines.append('| ' + model + ' | ' + ' | '.join(pairs) + ' |')
    lines += ['', '![같은 source VAL의94와189 입력 비교](figures/source_val_feature_comparison.png)', '',
        '38/45→43/45는 고정된 source 검사의 통과 수 비교입니다. 모든 T/R 지표가 일괄 개선됐다는 뜻은 아닙니다. UNION 세 모델의 R P90은 직전94보다 낮아졌지만, T 중앙값은 seed1·2에서 커졌고 seed3은 동일했습니다. T P90은 seed1·3에서 커졌습니다. R0_ONLY도 이번 표현으로 다시 학습했으므로 그 대조 결과가 바뀐 점을 표와 그래프에 함께 남깁니다. 고정 R0_GEO 비교 역시 원래45개 조건에 포함됩니다.', '',
        '| 실패 모델 | 비교 대상 | 실패 조건 | 실제 T P90(cm) | 허용 상한(cm) |', '|---|---|---|---:|---:|']
    for failure in source['failed_checks']:
        model, baseline, condition = failure.split('/')
        cap = source['summaries'][baseline]['full_population']['translation_cm']['P90'] * 1.05
        lines.append(f'| {model} | {baseline} | {condition} | {seed3_t90:.6f} | {cap:.9f} |')
    lines += ['', f'seed3의 T P90은 {seed3_t90:.6f}cm로, 동일한 두 기준값 {r0_t90:.6f}cm보다 **{100*(ratio-1):.2f}% 증가**했습니다. 허용치는5% 증가인 {limit:.9f}cm입니다. 두 실패를 남겨두고 일부 seed만 선택하거나 한계를 완화하지 않았습니다.', '',
        '## 실사 실행 여부와 이미지', '',
        '**이번189 선택기의 learned 실사 routing·오차 계산은0회입니다.** source45개 조건을 모두 통과해야 실사 평가에 들어가도록 정했으므로 여기서 중단했습니다. 실사 개선 목표의 완료 상태도 false로 유지합니다.', '',
        '기존 후보에 양축 비회귀 전체 pose 선택 조합이 있는지를 확인한 [이전 실사 oracle 가능성 진단](../pallet_pose_pareto_anchor_20261001_v1/REAL_VERIFICATION_KO.md)은 재계산하지 않았습니다. 그 원래5개 기준 통과는 참조를 보는 진단이며 이번 모델의 실사 실적이 아닙니다.', '',
        '**아래 세 이미지는 이전 참조 oracle 갤러리를 그대로 연결한 것입니다. 이번 새 모델의 실사 결과가 아닙니다.** 실제 RGB6장과 원래 입력 치수를 보여주며, 왼쪽은 기존 R0, 오른쪽은 참조 기반 보존 oracle입니다. 기존 전체 pose를 K로 투영한 선이며 정답 윤곽을 그린 것이 아닙니다. 각 자연 촬영에서 기존 R0 T 오차가 가장 큰 한 장을 선택한 진단 예시입니다.', '']
    for page in range(1, 4):
        lines += [f'![이전 참조 oracle의 RGB와 치수 {page}, 이번 모델 실사 결과 아님](../pallet_pose_pareto_anchor_20261001_v1/figures/real_rgb_dimensions_{page}.jpg)', '']
    lines += ['## 남은 TRAIN 선택 위험', '',
        '현재 동결된 선택만 다시 분류한 [TRAIN 위험 진단](TRAIN_RISK_DIAGNOSTIC_KO.md)입니다. 새 점수·정책·학습을 시험한 결과는 아닙니다. 안전 개선은 R0보다 한 축 이상이 엄격히 좋아지고 다른 축이 나빠지지 않는 경우입니다.', '',
        '| 모델 | target의 non-anchor 안전 개선 | 실제 R0 유지 | 실제 안전 개선 | 실제 악화 | 최대 정규화 초과 |', '|---|---:|---:|---:|---:|---:|']
    for model in C.MODEL_NAMES[1:]:
        rr = risk['models'][model]
        target_classes, selected_classes = rr['target']['classes'], rr['selected']['classes']
        maximum = rr['selected']['normalized_max_excess_all_available']['maximum']
        lines.append(f"| {model} | {target_classes['safe_improvement']['count']} | {selected_classes['anchor']['count']} | {selected_classes['safe_improvement']['count']} | {selected_classes['unsafe']['count']} | {maximum:.2f} |")
    lines += ['', '정규화 초과는 `max((T−T_anchor)/sT, (R−R_anchor)/sR, 0)`이며 고정 TRAIN 척도를 사용합니다. cm나 degree 단위의 오차가 아닙니다. 유효2597장과 별도 실패1장을 유지했습니다. 선택은 이미 R0 유지에 치우쳐 있지만 남은 소수 악화의 크기는 클 수 있습니다.', '',
        '후속 검토안은 TRAIN의 `log1p(risk)`를 이용하는 margin CE이며 **이번 보고서에서는 구현·학습·평가하지 않았습니다**. 단순히 R0 선호를 더 높이면 필요한 엄격 개선까지 줄일 수 있어, 이를 자동적인 해결책으로 보지 않습니다.', '',
        '## 확인할 파일과 해석의 범위', '',
        '- [학습 전 독립 검토](PREFIT_REVIEW_KO.md), [입력·타깃·solver 계약](TRAIN_PROTOCOL.json), [TRAIN 독립 수렴·성능 검산](TRAIN_CONVERGENCE_KO.md)',
        '- [source VAL 독립 검산](SOURCE_VAL_VERIFICATION_KO.md), [전체8192행 CSV](SOURCE_VAL_FRAME_RESULTS.csv), [45개 판정 CSV](SOURCE_VAL_CHECKS.csv)',
        '- [학습1495행 CSV](TRAINING_OBJECTIVE_LOG.csv), [실제 네 checkpoint JSON](model_parameters/), [실사 미실행 기록](REAL_EVALUATION_NOT_RUN_KO.md)',
        '- [공개 자료 검증](PUBLIC_REVIEW_KO.md), [파일 SHA 목록](PUBLICATION_MANIFEST.json), [그림·원자료 연결 JSON](REPORT_DATA.json)', '',
        '기존 상대 관계 특징·상대 이득 선택기의 선행 실패가 있으므로, 이189개 입력만으로 새로운 방법의 유효성을 주장하지 않습니다. 이번 작업은 하나의 고정 입력 표현을 비교한 실험입니다. TRAIN 선택 변화는 확인했지만 현재 결과만으로 Linear189 전체가 불가능하다거나 RGB 정보 부족이 원인이라고 확정할 수 없습니다.', '',
        'source VAL과 실사 DEV는 여러 방법에서 반복 사용했습니다. refiner의 기존 source 노출과 selector TRAIN/VAL/TEST 중복105/26/26건을 제거한 독립 평가라고 주장하지 않습니다. 전체 교사 계보에는 기존 수동 코너38개/이미지9장이 포함되며 이번 학습에 새 실사 정답을 넣지는 않았습니다. 이전 실사 참조도2D 주석·K·치수에서 만든 pose이며 독립 장비 실측 정답이 아닙니다.', '',
        'R0_ONLY는 결정론적 대조 fit 하나이고 UNION_s1/2/3은 각각 기존 frozen refiner seed를 사용합니다. 네 fit을 네 독립 반복 실험으로 해석하지 않습니다. 정확한 재실행에는 SHA로 연결한 로컬 원본 이미지·특징·pose 캐시가 필요하며, 공개 자료에는 검토용 CSV·checkpoint·실행 코드·그림·판정 기록을 포함합니다.', '']
    C.save(C.DOC / 'REPORT_KO.md', '\n'.join(lines))
    print('ANCHOR_CONTEXT_REPORT_COMPLETE', dict(csv_rows=csv_counts, figures=3, reused_figures=3,
          actual_new_fits=4, source_passed=43, source_total=45, learned_real_evaluated=False), flush=True)


if __name__ == '__main__':
    main()
