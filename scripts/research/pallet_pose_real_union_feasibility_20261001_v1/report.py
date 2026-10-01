"""Publish cached diagnostic values and actual RGB/dimensions illustrations."""
from . import common as C
import csv
import io
import itertools
import numpy as np

LABELS = {'AXIS_LOWER_BOUND': 'Axis lower bound (not a pose)',
          'FIXED_TRAIN_COST_ORACLE': 'Fixed TRAIN-cost whole-pose oracle'}
POPS = ('NATURAL99', 'CLEAN29', 'WOOD45')


def stat(array):
    array = np.asarray(array, float)
    assert array.ndim == 3 and array.shape[-1] == 2
    assert np.isfinite(array).all(), 'This report expects the independently verified zero-failure cache.'
    return dict(T_median_cm=float(np.median(array[:, :, 0], axis=1).mean()),
        R_median_deg=float(np.median(array[:, :, 1], axis=1).mean()),
        T_P90_cm=float(np.quantile(array[:, :, 0], .9, axis=1).mean()),
        R_P90_deg=float(np.quantile(array[:, :, 1], .9, axis=1).mean()),
        frames_per_seed=array.shape[1], seeds=array.shape[0], failures=0)


def baseline_array(metrics, model, ids):
    names = [f'{model}_s{s}' for s in (1, 2, 3)] if model in ('SINGLE251', 'DIVERSE251') else [model]
    return np.asarray([[[metrics[m][i]['translation_cm'], metrics[m][i]['rotation_deg']]
                        for i in ids] for m in names])


def detailed_findings(results):
    oracle = results['diagnostics']['FIXED_TRAIN_COST_ORACLE']
    assert results['diagnostics']['AXIS_LOWER_BOUND']['stability']['PASS']
    assert [k for k, v in oracle['stability']['gates'].items() if not v['PASS']] == ['natural_tails']
    lines = ['## 실패한 조건과 다음 변경의 근거', '',
        '전체 pose oracle은 seed별 중앙값 개선·bootstrap·촬영 제외·clean 보존을 통과했습니다. 실패는 자연 가림 **T P90의 R0 대비 비회귀 제한 한 가지**입니다. 평균 T/R 개선과 큰 T 오류의 억제를 동시에 만족시키지 못했습니다.', '',
        '| 자연 가림 T P90 | cm |', '|---|---:|']
    tail = oracle['stability']['gates']['natural_tails']['comparisons']['R0']['checks']['P90:translation_cm']
    for label, key in [('R0', 'before'), ('원래 허용 한계 (R0×1.05)', 'limit'), ('고정 비용 전체 pose oracle', 'after')]:
        lines.append(f"| {label} | {tail[key]['value']:.6f} |")
    lines += ['', '자연 가림 중앙값 차이의95% 구간은 다음과 같습니다. 단위는 T cm / R °이며, 같은 recording과 seed를 짝지어2,000회 재표집했습니다. 전부0 미만이어도 위의 T 꼬리 조건을 대신할 수 없습니다.', '',
        '| Oracle − 기준 | T 차이95% 구간 | R 차이95% 구간 |', '|---|---:|---:|']
    for baseline in ('SINGLE251', 'R0'):
        ci = oracle['hierarchy']['NATURAL99'][f'DIVERSE251-minus-{baseline}']['hierarchical_bootstrap']['metrics']
        cells = [f"[{ci[m]['CI95'][0]:.6f}, {ci[m]['CI95'][1]:.6f}]" for m in ('translation_cm', 'rotation_deg')]
        lines.append(f'| {baseline} | ' + ' | '.join(cells) + ' |')
    lines += ['', '**확인:** 현재 TRAIN 척도의 scalar 비용을 정확하게 최소화하는 선택도 이 DEV의 전체 안정성 기준을 만족하지 않습니다. 같은 정답 후보를 맞히는 학습 정확도만 높여도 전체 목표가 자동 달성되지는 않습니다.', '',
        '**다음 개입:** R0의 큰 T 오류를 더 키우지 않는 제약을 후보 선택의 학습 목표에 반영할 수 있는지 먼저 검토합니다. 기존 R0를 보존하는 제약 아래 전체 pose 개선 후보가 존재하는지 별도 고정 진단으로 확인한 뒤, 실사 정답 없이 학습·추론할 수 있는 신호로 연결해야 합니다. 이번 결과에서 새 제약이나 임계값을 탐색하거나 새 개선 성능으로 보고하지 않았습니다.', '',
        '축별 하한은 모든 조건을 통과했으므로 전체 후보 집합이 불가능하다고 결론 내릴 수 없습니다. 다만 서로 다른 후보의 최소 T/R를 합친 하한은 실제 pose가 아니므로, 전체 pose로 가능한지도 추가 증명이 필요합니다.', '']
    return lines


def figures(summary, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.DOC / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    models = ['R0', 'PRIOR1', 'FULL125', 'SINGLE251', 'DIVERSE251', *LABELS]
    names = ['R0', 'PRIOR1', 'FULL125', 'SINGLE mean3', 'DIVERSE mean3', 'Axis lower bound*', 'Whole-pose oracle*']
    colors = ['#64748b'] * 5 + ['#0d9488', '#7c3aed']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, key, title in zip(axes.flat, ('T_median_cm', 'R_median_deg', 'T_P90_cm', 'R_P90_deg'),
            ('T median (cm)', 'R median (deg)', 'T P90 (cm)', 'R P90 (deg)')):
        vals = [summary['NATURAL99'][m][key] for m in models]
        ax.barh(names, vals, color=colors)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis='x', alpha=.2)
        ax.set_xlim(0, max(vals) * 1.2)
        for i, val in enumerate(vals):
            ax.text(val + max(vals) * .01, i, f'{val:.3f}', va='center', fontsize=9)
    fig.suptitle('Natural99 | mean of per-seed quantiles, lower is better\n* Reference-derived diagnostics; not measured model improvements', fontsize=14)
    path = folder / 'natural_candidate_feasibility.png'
    fig.savefig(path, dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 3.7), constrained_layout=True)
    rows = list(LABELS)
    gates = list(results['diagnostics'][rows[0]]['stability']['gates'])
    matrix = np.asarray([[bool(results['diagnostics'][d]['stability']['gates'][g]['PASS']) for g in gates] for d in rows])
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix, cmap=ListedColormap(['#fca5a5', '#99f6e4']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(gates)), ['All 3 seeds\njoint medians', 'Paired bootstrap\n2000 draws', 'Leave each\nrecording out', 'Natural99\nP90 / failures', 'Clean29\nguards'])
    ax.set_yticks(range(2), ['Axis lower bound\n(not a pose)', 'Whole-pose\nreference oracle'])
    for j in range(2):
        for i in range(len(gates)):
            ax.text(i, j, 'PASS' if matrix[j, i] else 'FAIL', ha='center', va='center', weight='bold')
    ax.set_title('Original five stability criteria applied to diagnostics only\nA diagnostic PASS is not a deployable-method PASS', pad=18)
    path2 = folder / 'diagnostic_gates.png'
    fig.savefig(path2, dpi=140)
    plt.close(fig)
    return [C.bind(path), C.bind(path2)]


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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), constrained_layout=True)
        for line, fid in enumerate(chosen[2 * page:2 * page + 2]):
            row = meta[fid]
            C.verify(row['image'])
            rgb = np.asarray(Image.open(C.ROOT / row['image']['path']).convert('RGB'))
            oracle = lookup[('FIXED_TRAIN_COST_ORACLE', 1, fid)]
            pair = [('R0', candidates['R0'][fid]['GEO_name'], metrics['R0'][fid]['translation_cm'],
                     metrics['R0'][fid]['rotation_deg'], 'R0 operational', '#ef4444'),
                    (oracle['selected_model'], oracle['selected_hypothesis'], float(oracle['T_cm']),
                     float(oracle['R_deg']), 'Reference oracle s1 (diagnostic)', '#a855f7')]
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
                ax.set_title(f"{row['recording']} | {label}\nT={t:.2f} cm, R={r:.2f} deg | {dims} cm\n{model} / {hyp}", fontsize=10)
                ax.axis('off')
                detail['panels'].append(dict(model=model, hypothesis=hyp, T_cm=t, R_deg=r, drawn_edges=drawn,
                                             pose_source='Frozen camera-facing pose and extents; direct K projection, no new PnP.'))
            details.append(detail)
        fig.suptitle('Actual RGB + pallet dimensions | projected existing candidate boxes\nReference-derived oracle is for diagnosis only; no ground-truth outline is displayed.', fontsize=13)
        path = C.DOC / 'figures' / f'real_rgb_dimensions_{page + 1}.jpg'
        fig.savefig(path, dpi=125)
        plt.close(fig)
        artifacts.append(C.bind(path))
    return artifacts, details


def main():
    protocol = C.protocol()
    results = C.read(C.DOC / 'RESULTS.json')
    assert results['complete'] and not results['method_success'] and not results['goal_complete']
    groups = C.read(C.ROOT / protocol['inputs']['eval_groups']['path'])
    meta = C.read(C.ROOT / protocol['inputs']['eval_metadata']['path'])
    metrics = C.read(C.ROOT / protocol['inputs']['operational_metrics']['path'])
    candidates = C.read(C.ROOT / protocol['inputs']['stable_pose_candidates']['path'])
    with (C.DOC / 'FRAME_RESULTS.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['diagnostic'], int(r['seed']), r['id']): r for r in rows}
    assert len(rows) == len(lookup) == 1038
    summary = {}
    for pop in POPS:
        ids = groups[pop]
        summary[pop] = {m: stat(baseline_array(metrics, m, ids)) for m in ('R0', 'PRIOR1', 'FULL125', 'SINGLE251', 'DIVERSE251')}
        for diag in LABELS:
            a = np.asarray([[[float(lookup[(diag, s, i)]['T_cm']), float(lookup[(diag, s, i)]['R_deg'])]
                             for i in ids] for s in (1, 2, 3)])
            summary[pop][diag] = stat(a)
            for s in (1, 2, 3):
                summary[pop][f'{diag}_s{s}'] = stat(a[s - 1:s])
    plots = figures(summary, results)
    illustrations, details = gallery(meta, metrics, lookup, groups, candidates)
    C.save(C.DOC / 'REPORT_DATA.json', dict(complete=True, diagnostic_only=True, method_success=False,
        goal_complete=False, protocol=C.bind(C.DOC / 'PROTOCOL.json'), results=C.bind(C.DOC / 'RESULTS.json'),
        report_code=C.bind(C.HERE / 'report.py'), summary=summary, illustrations=details,
        artifacts=plots + illustrations, CSV_rows=len(rows), actual_RGB_images=6,
        image_forwards=0, new_PnP=0, new_reference_metric_calls=0))
    axis = results['diagnostics']['AXIS_LOWER_BOUND']['stability']['PASS']
    oracle = results['diagnostics']['FIXED_TRAIN_COST_ORACLE']['stability']['PASS']
    if not axis:
        conclusion = '기존 후보의 축별 최솟값을 사용하는 낙관적 하한도 원래 안정성 조건을 통과하지 못했습니다. 검증된 실패0·동일 분모 조건에서 이 후보 집합만 재선택하는 방법으로는 원래 전체 기준을 충족할 수 없습니다.'
    elif oracle:
        conclusion = '기존 후보 중에서 원래 안정성 조건을 만족하는 전체 pose 선택 조합은 존재합니다. 평가 참조를 사용한 oracle의 존재 증명이며, 실제 선택기가 그 후보를 알아낼 수 있다는 증거는 아닙니다.'
    else:
        conclusion = '축별 낙관적 하한은 통과했지만, 사전에 고정한 TRAIN 비용으로 고르는 전체 pose oracle은 실패했습니다. 이 고정 선택 규칙의 실패이며 다른 전체 pose 선택 조합까지 불가능하다고 단정할 수 없습니다.'
    lines = ['# 기존 실사 후보의 T·R 개선 가능성 진단', '',
        '**실제로 사용할 수 있는 모델의 안정적인 T·R 동시 개선은 아직 달성하지 못했습니다.**', '', conclusion, '',
        '## 입력과 이번 진단이 답하는 질문', '',
        '입력은 **RGB 이미지 한 장과 팔레트 치수**이며 기존 카메라 내부 보정 K를 유지합니다. 동영상·여러 시점·새 센서를 추가하지 않았습니다. 최신 선택기의 합성 검증 실패 이후, 같은 후보를 더 학습하기 전에 후보 집합 자체에 원래 목표를 만족할 여지가 있는지 확인했습니다.', '',
        '원래 R0와 DIVERSE251의 세 seed를 각각 짝지어 W/D 전체 pose 후보를 합칩니다. 기존 실사173장 = 자연 가림99 + 깨끗한29 + 목재45를 모두 유지하며 새 학습·추론·PnP·참조 오차 재계산은 각각0회입니다. 이전에 계산하고 공개한 후보별 T/R 오류를 사용했습니다.', '',
        '## 두 진단의 차이', '',
        '| 진단 | 선택 규칙 | 해석 |', '|---|---|---|',
        '| 축별 낙관적 하한 | 같은 후보 집합에서 최소 T와 최소 R을 각각 취함 | 서로 다른 후보일 수 있는 수학적 하한. 실제 pose나 동시 성능이 아님 |',
        '| 고정 TRAIN 비용 oracle | max(T/2.4636887551191258, R/1.113474019956766)를 최소화하는 전체 pose 하나 | 참조를 보고 고르는 진단. 실제 추론 선택기가 아님 |', '',
        '비용 척도는 기존 합성 TRAIN에서 고정한 값입니다. 실사에서 재조정하지 않았습니다. 정확한 비용 동률은 Pareto 지배 후보 제거 → R0 우선 → 가설 이름 → 모델 이름 순으로 처리했습니다. 모델당 가설이 정확히2개이므로 기존 T-best/R-best 두 행의 합집합이 필요한 비지배 후보를 보존합니다.', '',
        '## 원래 안정성 기준', '',
        'seed 세 개 각각의 자연 가림 T/R 중앙값을 paired SINGLE251 및 고정 R0/PRIOR1/FULL125와 비교합니다. SINGLE251/R0 대비 recording×seed paired bootstrap2,000회(seed20261001)의 양축95% 상단이0 미만이어야 하고, 자연 촬영6개를 각각 제외해도 양축 개선이 유지되어야 합니다. 자연 P90과 clean 중앙값/P90에는 원래1.05배 제한 및 실패 수 비증가 기준을 유지했습니다. 3seed를 합쳐 중앙값 하나로 바꾸지 않았습니다.', '',
        f'- 축별 낙관적 하한: **{"PASS" if axis else "FAIL"}**',
        f'- 고정 TRAIN 비용 전체 pose oracle: **{"PASS" if oracle else "FAIL"}**',
        '- 배포 가능한 방법의 성능 성공: **확인되지 않음**', '',
        '![원래 안정성 기준 진단 판정](figures/diagnostic_gates.png)', '',
        *detailed_findings(results),
        '## 전체 수치', '',
        'T 단위는 cm, R은 현재 물리 좌표계 C2 대칭 회전 오차(°)입니다. SINGLE/DIVERSE 및 진단 평균은 **seed별 모집단 중앙값 또는 P90을 구한 뒤 세 seed를 평균**한 값입니다. R0/PRIOR1/FULL125는 고정 모델 하나입니다. 원래 운영 결과와 참조를 이용한 진단 값을 명시적으로 구분합니다.', '']
    names = {'SINGLE251': 'SINGLE251 (3seed 평균)', 'DIVERSE251': 'DIVERSE251 운영 (3seed 평균)',
             'AXIS_LOWER_BOUND': '축별 하한 (진단, 3seed 평균)', 'FIXED_TRAIN_COST_ORACLE': '전체 pose oracle (진단, 3seed 평균)'}
    for pop in POPS:
        lines += [f'### {pop}', '', '| 모델/진단 | T 중앙값 | R 중앙값 | T P90 | R P90 | 실패 |', '|---|---:|---:|---:|---:|---:|']
        for model, values in summary[pop].items():
            label = names.get(model, model.replace('AXIS_LOWER_BOUND_', '축별 하한 ').replace('FIXED_TRAIN_COST_ORACLE_', '전체 pose oracle '))
            lines.append('| ' + label + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in ('T_median_cm', 'R_median_deg', 'T_P90_cm', 'R_P90_deg')) + ' | 0 |')
        lines += ['']
    lines += ['![자연 가림99의 후보 가능성](figures/natural_candidate_feasibility.png)', '',
        '## 실제 RGB·치수·후보 투영', '',
        '각 자연 촬영에서 **기존 R0 T 오류가 가장 큰 한 장**을 표시했습니다. 같은 값이면 ID 순으로 고릅니다. 모든 수치 판정에는 전체173장을 사용합니다. 오른쪽은 미리 고정한 seed1의 oracle 예시이며 좋은 seed를 고른 결과가 아닙니다. 그림에는 각 이미지의 실제 입력 치수(cm)가 들어 있습니다. 다른 seed의 결과는 전체 CSV에 남겼습니다.', '',
        '왼쪽 빨강은 R0 운영 pose, 오른쪽 보라는 평가 참조를 이용해 고른 기존 전체 pose입니다. 저장된 camera-facing R·중심·치수를 원래 K로 직접 투영했으며 새 PnP나 정답 외곽선은 사용하지 않았습니다. 영상의 다른 물체를 잡은 후보가 남을 수 있습니다.', '']
    for page in range(1, 4):
        lines += [f'![실제 이미지와 치수 비교 {page}](figures/real_rgb_dimensions_{page}.jpg)', '']
    lines += ['## 검증 자료와 한계', '',
        '- [사전 동결 프로토콜](PROTOCOL.json), [입력·후보 구조 감사](INPUT_AUDIT_KO.md)',
        '- [전체 프레임1038행 CSV](FRAME_RESULTS.csv), [모든 gate·bootstrap·촬영별 결과 JSON](RESULTS.json)',
        '- [독립 수치 검증](VERIFICATION_KO.md), [공개 파일 해시 목록](PUBLICATION_MANIFEST.json)',
        '- [직전 실제 선택기 학습·검증 실패](../pallet_pose_selector_pairwise_20261001_v1/REPORT_KO.md)',
        '- [기존 자연 가림99장 전체 비교 갤러리](../pallet_pose_stable_improvement_20261001_v1/GALLERY_NATURAL99.md)', '',
        '현재 진단은 원래 안정성 계약의 SINGLE251/R0/PRIOR1/FULL125 비교를 재현합니다. 후속 UNION 실험의 learned R0_ONLY는 source 검증 실패로 실사 routing을 실행하지 않았으므로, 그 확장 비교 계약 전체를 충족했다는 주장을 하지 않습니다.', '',
        '기존2D 주석·K·치수로 만든 참조 pose를 반복 사용한 DEV 진단입니다. 독립 장비 실측 정답이나 새 촬영 일반화 검증이 아닙니다. 기존 교사 모델 계보에는 수동 코너38개/이미지9장이 포함됩니다. 이번 계산에서 새 실사 정답을 학습에 사용하지 않았습니다.', '',
        '계산 코드는 같은 이름의 scripts/research 경로에 공개했습니다. 정확한 로컬 재실행에는 프로토콜에 SHA로 연결한 기존 이미지·예측·메타데이터 캐시가 필요합니다. 이 공개 커밋은 전체 원본 데이터셋을 복제하지 않으며, 검토용 CSV·판정·6장의 실제 이미지 비교를 제공합니다.', '',
        '이 결과는 다음 변경의 범위를 결정하는 근거입니다. 그림·oracle·수학적 하한을 실제 모델의 개선 수치로 바꾸어 보고하지 않습니다.', '']
    C.save(C.DOC / 'REPORT_KO.md', '\n'.join(lines))
    print('REPORT_COMPLETE', {'CSV_rows': len(rows), 'figures': len(plots) + len(illustrations), 'actual_RGB_images': 6})


if __name__ == '__main__':
    main()
