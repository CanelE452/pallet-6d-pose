"""Actual target-only fits and reference-only feasibility, kept distinct."""
from . import common as C
import csv
import io
import itertools
import json
import numpy as np


def write_csv(name, rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    C.save(C.DOC / name, out.getvalue())
    return len(rows)


def plot_metrics(real, previous, source, old_source, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.DOC / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    specs = [('translation_cm', 'median', 'T median (cm)'), ('rotation_deg', 'median', 'R median (deg)'),
             ('translation_cm', 'P90', 'T P90 (cm)'), ('rotation_deg', 'P90', 'R P90 (deg)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, q, title) in zip(axes.flat, specs):
        vals = [real['summaries']['NATURAL99']['R0']['conditional'][q][metric]['value'],
                previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['summaries']['NATURAL99']['DIVERSE251']['conditional'][q][metric]['value'],
                real['summaries']['NATURAL99']['DIVERSE251']['conditional'][q][metric]['value']]
        ax.bar(['R0 actual', 'Prior cost\noracle*', 'Anchored\noracle*'], vals, color=['#64748b', '#8b5cf6', '#0d9488'])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2)
        for j, v in enumerate(vals):
            ax.text(j, v + max(vals) * .025, f'{v:.3f}', ha='center')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Natural99 | whole-pose feasibility on reused DEV\n* Reference-derived diagnostics, not learned-model improvements')
    p = folder / 'real_oracle_constraints.png'
    fig.savefig(p, dpi=140)
    plt.close(fig)
    artifacts = [C.bind(p)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = np.arange(4)
    for ax, (metric, q, title) in zip(axes.flat, specs):
        old = [old_source['summaries'][m]['full_population'][metric][q] for m in C.MODEL_NAMES]
        new = [source['summaries'][m]['full_population'][metric][q] for m in C.MODEL_NAMES]
        ax.bar(x - .2, old, .4, label='Previous converged target', color='#64748b')
        ax.bar(x + .2, new, .4, label='R0-anchored target', color='#d97706')
        ax.set_xticks(x, ['R0_ONLY', 'UNION s1', 'UNION s2', 'UNION s3'])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(0, max(old + new) * 1.18)
        for j in range(4):
            ax.text(j - .2, old[j], f'{old[j]:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(j + .2, new[j], f'{new[j]:.3f}', ha='center', va='bottom', fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Actual learned selectors | source VAL1024, all final fits\nTarget-only change; 38/45 source checks passed, method stopped')
    p = folder / 'source_val_target_comparison.png'
    fig.savefig(p, dpi=140)
    plt.close(fig)
    artifacts.append(C.bind(p))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, model in zip(axes.flat, C.MODEL_NAMES):
        rows = [r for r in traces if r['model'] == model]
        ax.plot([r['call'] for r in rows], [r['objective'] for r in rows], color='#0f766e')
        ax.set_title(model)
        ax.set_xlabel('Objective evaluations (solver calls)')
        ax.set_ylabel('Full TRAIN CE + ridge')
        ax.grid(alpha=.2)
    fig.suptitle('Four actual fits | frozen features, changed target rule only')
    p = folder / 'training_objectives.png'
    fig.savefig(p, dpi=140)
    plt.close(fig)
    artifacts.append(C.bind(p))
    return artifacts


def gallery(metadata, metrics, lookup, populations, candidates):
    import matplotlib.pyplot as plt
    from PIL import Image
    meta = {r['id']: r for r in metadata}
    ids = populations['NATURAL99']
    recordings = sorted({meta[i]['recording'] for i in ids})
    chosen = [min((i for i in ids if meta[i]['recording'] == rec),
                  key=lambda i: (-metrics['R0'][i]['translation_cm'], i)) for rec in recordings]
    assert len(chosen) == 6
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.count_nonzero(signs[i] != signs[j]) == 1]
    details, artifacts = [], []
    for page in range(3):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=False)
        fig.subplots_adjust(top=.84, bottom=.03, left=.025, right=.975, wspace=.08, hspace=.48)
        for line, fid in enumerate(chosen[2 * page:2 * page + 2]):
            row = meta[fid]
            C.verify(row['image'])
            rgb = np.asarray(Image.open(C.ROOT / row['image']['path']).convert('RGB'))
            oracle = lookup[('PARETO_ANCHOR_ORACLE', 1, fid)]
            pair = [('R0', candidates['R0'][fid]['GEO_name'], metrics['R0'][fid]['translation_cm'],
                     metrics['R0'][fid]['rotation_deg'], 'R0 operational', '#ef4444'),
                    (oracle['selected_model'], oracle['selected_hypothesis'], float(oracle['T_cm']),
                     float(oracle['R_deg']), 'Anchor oracle s1*', '#a855f7')]
            detail = dict(id=fid, recording=row['recording'], dimensions_m=row['xyz'], image=row['image'],
                          oracle_seed=1, display_selection='Largest operational R0 T error per natural recording; ID breaks ties.', panels=[])
            for col, (model, hyp, t, r, label, color) in enumerate(pair):
                ax = axes[line, col]
                pose = next(h['pose'] for h in candidates[model][fid]['hypotheses'] if h['name'] == hyp)
                corners = signs * np.asarray(pose['cf_extents']) / 2
                xyz = corners @ np.asarray(pose['R_cf']).T + np.asarray(pose['centroid'])
                camera = xyz @ np.asarray(row['K']).T
                uv = camera[:, :2] / camera[:, 2:3]
                ax.imshow(rgb)
                drawn = 0
                for i, j in edges:
                    if xyz[[i, j], 2].min() > 0:
                        ax.plot(uv[[i, j], 0], uv[[i, j], 1], color=color, lw=2.2)
                        drawn += 1
                ax.set_xlim(-.5, rgb.shape[1] - .5)
                ax.set_ylim(rgb.shape[0] - .5, -.5)
                dims = ' x '.join(f'{100 * d:g}' for d in row['xyz'])
                ax.set_title(f"{row['recording']} | {label}\nT={t:.2f} cm, R={r:.2f} deg\nDimensions: {dims} cm\n{model} / {hyp}", fontsize=9.5)
                ax.axis('off')
                detail['panels'].append(dict(model=model, hypothesis=hyp, T_cm=t, R_deg=r, drawn_edges=drawn,
                                             pose_source='Frozen camera-facing pose and extents; direct K projection, no new PnP.'))
            details.append(detail)
        fig.suptitle('Actual RGB + pallet dimensions | projected existing candidate boxes\n* Reference-derived oracle is for diagnosis only; no ground-truth outline is displayed.', fontsize=12, y=.98)
        path = C.DOC / 'figures' / f'real_rgb_dimensions_{page + 1}.jpg'
        fig.savefig(path, dpi=125)
        plt.close(fig)
        artifacts.append(C.bind(path))
    return artifacts, details


def main():
    real_protocol = C.protocol()
    train_protocol = C.protocol('TRAIN_PROTOCOL')
    real = C.read(C.DOC / 'REAL_FEASIBILITY.json')
    source_feas = C.read(C.DOC / 'SOURCE_FEASIBILITY.json')
    source = C.read(C.DOC / 'SOURCE_VAL_GATE.json')
    trained = C.read(C.DOC / 'TRAINING_COMPLETE.json')
    verified = C.read(C.DOC / 'TRAIN_CONVERGENCE.json')
    not_run = C.read(C.DOC / 'REAL_EVALUATION_NOT_RUN.json')
    assert real['PASS'] and source_feas['PASS'] and verified['PASS']
    assert source['checks_passed'] == 38 and source['checks_total'] == 45 and not source['PASS']
    assert trained['fit_count'] == 4 and trained['all_certified']
    previous = C.read(C.ROOT / real_protocol['inputs']['previous_feasibility']['path'])
    old_source = C.read(C.CONV_DOC / 'SOURCE_VAL_GATE.json')
    C.verify(source['metrics'])
    with np.load(C.ROOT / source['metrics']['path'], allow_pickle=False) as z:
        ids, models = z['ids'].tolist(), z['models'].tolist()
        errors = {m: z[m] for m in models}
    choices = C.read(C.RAW / 'SOURCE_VAL_CHOICES.json')
    val_rows = []
    for model in models:
        assert errors[model].shape == (1024, 2) and np.isfinite(errors[model]).all()
        for j, fid in enumerate(ids):
            pick = choices['records'].get(model, {}).get(fid, {})
            val_rows.append(dict(model=model, id=fid, split='VAL', available=True,
                T_cm=float(errors[model][j, 0]), R_deg=float(errors[model][j, 1]),
                candidate=pick.get('candidate_name', 'fixed_GEO'), fallback=pick.get('fallback', False)))
    checks = [dict(model=m, baseline=b, criterion=k, PASS=v)
              for m, cc in source['comparisons'].items() for b, g in cc.items() for k, v in g['checks'].items()]
    receipts, traces, exports = {}, [], {}
    for model in C.MODEL_NAMES:
        receipt = C.read(C.DOC / f'FIT_{model}.json')
        receipts[model] = receipt
        for key in ('START', 'trace', 'checkpoint'):
            C.verify(receipt[key])
        log = [json.loads(s) for s in (C.ROOT / receipt['trace']['path']).read_text().splitlines()]
        traces += [dict(model=model, **r) for r in log if r['event'] == 'objective']
        path = C.DOC / 'model_parameters' / f'{model}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        original = C.ROOT / receipt['checkpoint']['path']
        with path.open('xb') as f:
            f.write(original.read_bytes())
        assert path.read_bytes() == original.read_bytes()
        exports[model] = dict(local=receipt['checkpoint'], published=C.bind(path))
    csv_counts = {name: write_csv(name, rows) for name, rows in (
        ('SOURCE_VAL_FRAME_RESULTS.csv', val_rows), ('SOURCE_VAL_CHECKS.csv', checks),
        ('TRAINING_OBJECTIVE_LOG.csv', traces))}
    assert csv_counts == {'SOURCE_VAL_FRAME_RESULTS.csv': 8192, 'SOURCE_VAL_CHECKS.csv': 45, 'TRAINING_OBJECTIVE_LOG.csv': 1143}
    plots = plot_metrics(real, previous, source, old_source, traces)
    with (C.DOC / 'REAL_FEASIBILITY_ROWS.csv').open(newline='') as f:
        real_rows = list(csv.DictReader(f))
    lookup = {(r['diagnostic'], int(r['seed']), r['id']): r for r in real_rows}
    assert len(lookup) == 519
    inp = real_protocol['inputs']
    meta = C.read(C.ROOT / inp['eval_metadata']['path'])
    metrics = C.read(C.ROOT / inp['operational_metrics']['path'])
    groups = C.read(C.ROOT / inp['eval_groups']['path'])
    candidates = C.read(C.ROOT / inp['stable_pose_candidates']['path'])
    images, details = gallery(meta, metrics, lookup, groups, candidates)
    C.save(C.DOC / 'REPORT_DATA.json', dict(complete=True, actual_new_fits=4, source_VAL_checks_passed=38,
        source_VAL_checks_total=45, learned_real_evaluated=False, real_oracle_diagnostic_evaluated=True,
        stable_joint_improvement_achieved=False, goal_complete=False,
        code=C.bind(C.HERE / 'report.py'), real_protocol=C.bind(C.DOC / 'PROTOCOL.json'),
        train_protocol=C.bind(C.DOC / 'TRAIN_PROTOCOL.json'), real_feasibility=C.bind(C.DOC / 'REAL_FEASIBILITY.json'),
        source_feasibility=C.bind(C.DOC / 'SOURCE_FEASIBILITY.json'), source_gate=C.bind(C.DOC / 'SOURCE_VAL_GATE.json'),
        training_verification=C.bind(C.DOC / 'TRAIN_CONVERGENCE.json'), real_not_run=C.bind(C.DOC / 'REAL_EVALUATION_NOT_RUN.json'),
        csv_rows=csv_counts, real_diagnostic_csv_rows=519, exports=exports,
        objective_calls=len(traces), iterations=sum(r['iterations'] for r in receipts.values()),
        fit_wall_seconds=sum(r['wall_seconds'] for r in receipts.values()),
        artifacts=plots + images, actual_RGB_images=6, illustrations=details))
    lines = ['# R0 양축 보존 타깃: 가능성 진단과 실제 학습 결과', '',
        '**실제 모델의 안정적인 T·R 동시 개선은 아직 달성하지 못했습니다.**', '',
        'R0보다 T와 R을 모두 악화시키지 않는 후보로 학습 정답을 제한했습니다. 참조를 이용하는 실사 가능성 진단은 원래5개 안정성 조건을 모두 통과했고, 합성 TRAIN 진단도45/45를 통과했습니다. 그러나 실제 선택기4개를 학습해 합성 VAL1024장에서 검증하니 **38/45 통과·7개 실패**였습니다. 이 방법의 learned 실사 평가·적용은 중단했습니다.', '',
        '입력은 **RGB 이미지 한 장 + 팔레트 치수**, 기존 카메라 보정 K입니다. 원래 이미지·치수·후보·참조는 유지했으며 동영상이나 새 센서를 추가하지 않았습니다.', '',
        '## 변경한 것과 실제 추론의 차이', '',
        '기준 anchor는 같은 이미지의 기존 R0 운영 GEO 전체 pose입니다. 학습 데이터의 참조 오차로 `T(candidate) ≤ T(R0)` 및 `R(candidate) ≤ R(R0)`를 만족하는 후보만 정답 후보로 허용하고, 그 안에서 기존 `max(T/2.4636887551191258, R/1.113474019956766)` 비용과 정확한 동률 규칙으로 전체 pose 하나를 고릅니다.', '',
        '**이 제약은 학습 정답 생성에만 적용됩니다.** CE의 경쟁 후보와 실제 추론 후보는 원래 유효 후보 전체입니다. 실제 추론은 참조 T/R를 모르므로 이 제약으로 후보를 지울 수 없습니다. 따라서 안전한 학습 타깃이 학습된 선택기의 안전성을 보장하지 않습니다.', '',
        '직전 수렴 통제와 비교해 바꾼 요인은 target 적격조건 하나입니다. 기존94개 기하 특징·선형 점수·FP32 정규화 후 FP64 계산·전체2598행 CE+L2(λ=1e−4)·초깃값0·solver·후보·TRAIN 척도를 유지했습니다. R0_ONLY의 타깃도 바뀔 수 있으므로 대조 모델까지4개 모두 새로 학습했습니다.', '',
        '## 실사 후보 가능성: 정답을 보는 진단', '',
        '전체173장(자연 가림99·clean29·wood45)과 세 frozen refiner seed를 그대로 사용했습니다. 원래5개 기준, SINGLE251/R0/PRIOR1/FULL125 비교, recording×seed bootstrap2,000회, 촬영6개 제외 검사, 자연·clean1.05배 비회귀 한계와 실패 수 기준을 유지했습니다. 새 학습 모델을 실사에 실행한 결과가 아닙니다.', '',
        '| 진단 | 원래 기준 | 자연 T 중앙값(cm) | R 중앙값(°) | T P90(cm) | R P90(°) |', '|---|---|---:|---:|---:|---:|']
    prev = previous['diagnostics']['FIXED_TRAIN_COST_ORACLE']['summaries']['NATURAL99']['DIVERSE251']['conditional']
    cur = real['summaries']['NATURAL99']['DIVERSE251']['conditional']
    for name, status, values in [('직전 비용 oracle', '4/5', prev), ('R0 양축 보존 oracle', '5/5', cur)]:
        nums = [values[q][k]['value'] for k, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        lines.append('| ' + name + ' | ' + status + ' | ' + ' | '.join(f'{v:.6f}' for v in nums) + ' |')
    lines += ['', '표는 seed별 모집단 중앙값/P90을 구한 뒤 세 seed를 평균한 값입니다. **T 꼬리를 보존하는 대신 직전 비용 oracle보다 R P90은 증가했습니다.** 그래도 원래 R0와 비교하는 전체 기준은 통과합니다. 두 oracle 모두 배포 가능한 개선 수치가 아닙니다.', '',
        f"519개 frame×seed에서 양축 비증가를 확인했습니다. 그중 {real['fallback_counts']['anchor_returned']}개는 R0를 그대로 유지했습니다. 독립 검산으로 후보 선택·원래5개 기준을 재현했습니다.", '',
        '![실사 후보 가능성의 제약과 절충](figures/real_oracle_constraints.png)', '',
        '## 합성 TRAIN과 실제 학습', '',
        '선언된 C2 대칭·정상 rigid pose 조건의 TRAIN2598장을 그대로 사용했습니다. 유효 anchor2597개와 원래 후보 실패1개를 유지했고, 실패 행은 전체2598행 CE 분모에서0 loss로 남겼습니다. 제약된 TRAIN oracle은45/45를 통과했으며, 별도 scalar-loop 구현으로 모든 target·적격 후보 mask를 대조했습니다.', '',
        '| 모델 | optimizer iterations | objective calls | 수렴 gap 상한 |', '|---|---:|---:|---:|']
    for model, receipt in receipts.items():
        lines.append(f"| {model} | {receipt['iterations']} | {receipt['objective_calls']} | {receipt['certificate']['gradient_l2_squared_over_2lambda']:.3e} |")
    lines += ['', f"네 fit은 총{len(traces)}회 목적함수 계산과 {sum(r['iterations'] for r in receipts.values())}회 optimizer iteration을 실행했습니다. 수치 수렴은 별도 PyTorch autograd로 검산했습니다. 이는 해당 CE+ridge 목적함수의 인증이며 T/R 개선 인증이 아닙니다.", '',
        '![실제 네 학습의 목적함수 기록](figures/training_objectives.png)', '',
        '| 동일 새 TRAIN 타깃에서 비교 | 이전→새 선택 정확도 | 이전→새 R0 양축 중 하나 이상 악화 |', '|---|---:|---:|']
    for model in C.MODEL_NAMES[1:]:
        record = verified['models'][model]
        old, new = record['previous_converged_same_target']['statistics'], record['statistics']
        lines.append(f"| {model} | {old['target_accuracy_available_rows']*100:.2f}% → {new['target_accuracy_available_rows']*100:.2f}% | {old['anchor_violations']['either']['count']} → {new['anchor_violations']['either']['count']} / 2597 |")
    lines += ['', '정답 후보를 고르는 비율은 올라갔고 악화 선택은 줄었지만, TRAIN에서도 제약 위반이 남았습니다. 정확도와 악화율의 분모는 유효2597장이며 실패1개를 제외했다고 숨기지 않습니다. 전체 T/R 집계에는2598행을 모두 유지합니다.', '',
        '## 합성 VAL1024: 실제 학습한 선택기의 결과', '',
        '학습 종료·checkpoint 동결 후4×1024개 선택을 정답 조회 전에 고정했습니다. 이후 동일한 물리 좌표계 T/C2 회전 오차로 전체8모델을 평가했습니다. 모든 모델의 pose 실패는0입니다. T 단위 cm, R 단위 °입니다.', '',
        '| 모델 | T 중앙값 | R 중앙값 | T P90 | R P90 | 실패 |', '|---|---:|---:|---:|---:|---:|']
    for model, s in source['summaries'].items():
        values = s['full_population']
        nums = [values[k][q] for k, q in [('translation_cm', 'median'), ('rotation_deg', 'median'), ('translation_cm', 'P90'), ('rotation_deg', 'P90')]]
        lines.append('| ' + model + ' | ' + ' | '.join(f'{v:.6f}' for v in nums) + f" | {s['failed_pose']} |")
    lines += ['', '![동일 source VAL에서 타깃 변경 전후](figures/source_val_target_comparison.png)', '',
        '**실패한7개 조건을 모두 남깁니다.** 회전 P90 비회귀6개와 seed3의 회전 중앙값 엄격 개선1개입니다.', '',
        '| 모델 | 비교 대상 | 실패 조건 |', '|---|---|---|']
    for failure in source['failed_checks']:
        lines.append('| ' + ' | '.join(failure.split('/')) + ' |')
    lines += ['', '좋은 seed나 성공한 지표만 채택하지 않았습니다. source 기준 실패로 이 방법의 learned 실사 routing·실사 오차 계산은0회입니다. 앞의 실사 oracle 진단과 이 미실행을 혼동하지 않아야 합니다.', '',
        '## 실제 이미지와 입력 치수', '',
        '각 자연 촬영에서 기존 R0 T 오류가 가장 큰 한 장을 고정해 보여줍니다. 오른쪽은 고정 seed1의 **참조 기반 보존 oracle**이며 새로 학습한 선택기의 실사 결과가 아닙니다. 치수는 원본 입력값(cm)이고, 기존 전체 pose를 K로 투영했습니다. 전체 판정에는 모든173장을 사용합니다.', '']
    for page in range(1, 4):
        lines += [f'![RGB·치수·보존 oracle 비교 {page}](figures/real_rgb_dimensions_{page}.jpg)', '']
    lines += ['## 확인한 것과 남은 문제', '',
        '기존 후보 안에 원래 안정성 기준을 만족하는 전체 pose 선택 조합이 존재함을 확인했습니다. 타깃의 양축 보존 조건도 source TRAIN에 구현하고 네 모델을 실제 학습했습니다. 다만 현재94개 특징과 선형 점수의 CE 학습은 그 조건을 정확히 재현하지 못했고, source VAL의 전체 기준도 통과하지 못했습니다.', '',
        '이 결과만으로 Linear94의 모든 방법이 불가능하다거나 RGB 정보 부족이 원인이라고 확정할 수 없습니다. 다음에는 기준 R0와의 관계가 현재 점수 함수에 어떻게 전달되는지와 기존 상대 이득 선택기의 실패 전례를 확인한 뒤 입력 표현 또는 선택 구조의 최소 변경을 정해야 합니다. 단순히 같은 특징에서 anchor 값을 빼는 선형 변경은 argmin에서 상쇄될 수 있으므로 새 정보가 되는지도 확인해야 합니다.', '',
        '## 검증·원자료·한계', '',
        '- [실사 가능성 전체519행](REAL_FEASIBILITY_ROWS.csv), [실사 독립 검산](REAL_VERIFICATION_KO.md), [유효 입력과 결측 경계 검토](REAL_METHOD_REVIEW_KO.md)',
        '- [source TRAIN 가능성](SOURCE_FEASIBILITY.json), [실제 TRAIN 수렴·악화 선택 검산](TRAIN_CONVERGENCE_KO.md)',
        '- [VAL 전체8192행](SOURCE_VAL_FRAME_RESULTS.csv), [45개 판정](SOURCE_VAL_CHECKS.csv), [source VAL 독립 검산](SOURCE_VAL_VERIFICATION_KO.md)',
        '- [학습 log CSV](TRAINING_OBJECTIVE_LOG.csv), [실제 네 checkpoint JSON](model_parameters/), [실사 미실행 증거](REAL_EVALUATION_NOT_RUN_KO.md)',
        '- [선행 실패와 이번 변경의 구분](PRIOR_METHOD_AUDIT_KO.md), [공개 검증](PUBLIC_REVIEW_KO.md), [파일 해시 목록](PUBLICATION_MANIFEST.json)',
        '- [기존 자연 가림99장 전체 이미지](../pallet_pose_stable_improvement_20261001_v1/GALLERY_NATURAL99.md)', '',
        '실사 참조는 기존2D 주석·K·치수에서 만든 pose이며 독립 장비 실측 정답이 아닙니다. DEV와 source VAL은 여러 방법에서 반복 사용했습니다. 기존 refiner source 노출과 selector TRAIN/VAL/TEST의 중복105/26/26건을 제거한 새로운 평가라고 주장하지 않습니다. 전체 교사 계보에는 기존 수동 코너38개/이미지9장이 포함되며 이번 학습에는 새 실사 정답을 사용하지 않았습니다.', '',
        'R0_ONLY는 결정론적 대조 모델 하나이고 UNION_s1/2/3은 기존 refiner seed를 각각 사용합니다. 이를 네 개 독립 반복 실험이나 새 촬영 일반화로 해석하지 않습니다. 동일 입력·정규화·solver를 유지한 target 변경의 효과와 실제 성능 판정을 구분했습니다.', '',
        '정확한 재실행에는 프로토콜에 SHA로 연결한 로컬 원본 이미지·feature·pose 캐시가 필요합니다. 공개 커밋에는 전체 원본 데이터셋 대신 검토용 CSV·체크포인트·실행 코드·이미지·판정 자료를 포함했습니다.', '']
    C.save(C.DOC / 'REPORT_KO.md', '\n'.join(lines))
    print('ANCHOR_REPORT_COMPLETE', dict(csv_rows=csv_counts, real_diagnostic_rows=519, figures=len(plots + images), fits=4))


if __name__ == '__main__':
    main()
