"""CPU-only aggregate report for the locked no-fit directional diagnostic.

Public output contains no frame identifiers, native coordinates or target arrays.
This module neither imports the model probe nor constructs an optimizer.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C


SIGNS = ('TOWARD', 'AWAY', 'NEAR_ZERO', 'EPS_SENSITIVE', 'NO_RESIDUAL')
RESOLVED = set(SIGNS[:3])
DIRECTIONS = ('real', 'source', 'combined')
SCOPES = ('CORNERS_0_7', 'ALL_9', 'CENTER_8')


def stable_sign(values, floor):
    if min(values) > floor:
        return 'TOWARD'
    if max(values) < -floor:
        return 'AWAY'
    if max(map(abs, values)) <= floor:
        return 'NEAR_ZERO'
    return 'EPS_SENSITIVE'


def summary(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p10=float(np.percentile(a, 10)) if len(a) else None,
                p90=float(np.percentile(a, 90)) if len(a) else None,
                minimum=float(a.min()) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)


def point_scope(points, role, scope):
    return [p for p in points if p['role'] == role and
            (scope == 'ALL_9' or (p['corner'] < 8 if scope == 'CORNERS_0_7' else p['corner'] == 8))]


def aggregate(points, own_direction):
    counts = {d: {s: sum(p['signs'][d] == s for p in points) for s in SIGNS}
              for d in DIRECTIONS}
    own_positive = [p for p in points if p['signs'][own_direction] == 'TOWARD']
    eligible = [p for p in own_positive if p['signs']['combined'] in RESOLVED]
    cancelled = [p for p in eligible if p['signs']['combined'] == 'AWAY']
    suppressed = [p for p in eligible if p['signs']['combined'] == 'NEAR_ZERO']
    retained = [p for p in eligible if p['signs']['combined'] == 'TOWARD']
    rescued = [p for p in points if p['signs'][own_direction] == 'AWAY' and p['signs']['combined'] == 'TOWARD']
    activated = [p for p in points if p['signs'][own_direction] == 'NEAR_ZERO' and p['signs']['combined'] == 'TOWARD']
    pairs = Counter((p['signs'][own_direction], p['signs']['combined']) for p in points)
    per_image = {}
    for p in points:
        row = per_image.setdefault(p['image_id'], dict(points=0, own_positive=0,
            eligible=0, cancellation=0, suppression=0))
        row['points'] += 1
        if p['signs'][own_direction] == 'TOWARD':
            row['own_positive'] += 1
            if p['signs']['combined'] in RESOLVED:
                row['eligible'] += 1
                row['cancellation'] += p['signs']['combined'] == 'AWAY'
                row['suppression'] += p['signs']['combined'] == 'NEAR_ZERO'
    assert len(cancelled)+len(suppressed)+len(retained) == len(eligible)
    return dict(points=len(points), images=len(per_image), own_direction=own_direction,
        sign_counts=counts,
        sign_transition_counts={f'{a}_TO_{b}': pairs[(a,b)] for a in SIGNS for b in SIGNS},
        residual_px=summary([p['residual_px'] for p in points]),
        response_px_at_common_middle_step={d: summary([
            p['response_px_at_common_reference_step'][d][1] for p in points]) for d in DIRECTIONS},
        own_toward=len(own_positive), resolved_own_toward_denominator=len(eligible),
        own_toward_combined_unresolved=len(own_positive)-len(eligible),
        own_toward_combined_eps_sensitive=sum(p['signs']['combined'] == 'EPS_SENSITIVE' for p in own_positive),
        own_toward_combined_no_residual=sum(p['signs']['combined'] == 'NO_RESIDUAL' for p in own_positive),
        any_pair_eps_sensitive=sum('EPS_SENSITIVE' in (p['signs'][own_direction], p['signs']['combined']) for p in points),
        any_pair_no_residual=sum('NO_RESIDUAL' in (p['signs'][own_direction], p['signs']['combined']) for p in points),
        cancellation=len(cancelled), suppression=len(suppressed), retained_toward=len(retained),
        away_to_toward=len(rescued), nearzero_to_toward=len(activated),
        net_toward_count=counts['combined']['TOWARD']-counts[own_direction]['TOWARD'],
        cancellation_fraction=len(cancelled)/len(eligible) if eligible else None,
        suppression_fraction=len(suppressed)/len(eligible) if eligible else None,
        images_with_eligible=sum(r['eligible'] > 0 for r in per_image.values()),
        images_with_cancellation=sum(r['cancellation'] > 0 for r in per_image.values()),
        images_with_suppression=sum(r['suppression'] > 0 for r in per_image.values()),
        per_image_cancellation_count_histogram={str(k): v for k,v in sorted(Counter(
            r['cancellation'] for r in per_image.values()).items())},
        per_image_suppression_count_histogram={str(k): v for k,v in sorted(Counter(
            r['suppression'] for r in per_image.values()).items())})


def prior_context():
    paths = [C.ROOT/'_docs/experiments/pallet_oracle_mechanism_followup_v1'/f'SIGNAL_DIAGNOSTIC_{m}.json'
             for m in ('PLASTIC', 'WOOD')]
    rows = []
    for path in paths:
        old = C.read(path)
        for arm, modes in old['results'].items():
            for mode, values in modes.items():
                batches = values['gradient_batches']
                rows.append(dict(material=old['material'], arm=arm, augmentation=mode,
                    batch_cosines=[b['cosine'] for b in batches],
                    source_over_real_norm=[b['source_over_real_norm'] for b in batches],
                    source_projection_onto_real=[b['source_projection_onto_real'] for b in batches]))
    return dict(bindings=[C.bind(p) for p in paths], rows=rows,
        non_equivalence='Prior probe independently normalized separate real/source sub-batches and advanced REF criterion five times; this diagnostic partitions one full mixed graph and uses four advances. Values are contextual, not exact replication.')


def build_public(protocol, private):
    assert private['optimizer_constructed'] is False
    assert private['optimizer_steps'] == private['new_fits'] == private['checkpoint_writes'] == 0
    assert private['evaluation_reference_read'] is False
    assert all(x['exact'] and x['all_parameter_grad_buffers_none'] for x in private['state_checks'].values())
    expected = {(arm, batch) for arm in protocol['checkpoints'] for batch in range(protocol['batches'])}
    assert {(r['arm'], r['batch']) for r in private['records']} == expected
    assert len(private['records']) == len(expected)
    output = []
    for row in private['records']:
        for p in row['points']:
            assert p['role'] in ('REAL', 'SOURCE') and 0 <= p['corner'] <= 8
            for d in DIRECTIONS:
                values = p['response_px_at_common_reference_step'][d]
                assert len(values) == 3 and np.isfinite(values).all()
                expected_sign = stable_sign(values, protocol['numerical_floor_px']) if p['residual_px'] > 1e-8 else 'NO_RESIDUAL'
                assert p['signs'][d] == expected_sign
        out = {k: row[k] for k in ('arm', 'batch', 'branch_weights', 'gradients', 'gradient_cosine',
            'partition', 'max_gradient_partition_error', 'theta_norm', 'common_gradient_norm',
            'median_coordinate_additivity_residual_px', 'max_coordinate_additivity_residual_px')}
        out['groups'] = {role: {scope: aggregate(point_scope(row['points'], role, scope), role.lower())
            for scope in SCOPES} for role in ('REAL', 'SOURCE')}
        out['combined_vs_real_first_order_REAL_objective_descent_factor'] = (
            1 + row['gradient_cosine']*row['gradients']['source']['norm']/row['gradients']['real']['norm'])
        output.append(out)
    pooled = {}
    for arm in protocol['checkpoints']:
        points = [p for row in private['records'] if row['arm'] == arm for p in row['points']]
        pooled[arm] = {role: {scope: aggregate(point_scope(points, role, scope), role.lower())
            for scope in SCOPES} for role in ('REAL', 'SOURCE')}
    decisions = {}
    for arm in pooled:
        rows = [r for r in output if r['arm'] == arm]
        decisions[arm] = {}
        for scope in SCOPES:
            counts = {k: [r['groups']['REAL'][scope][k] for r in rows] for k in ('cancellation', 'suppression')}
            decisions[arm][scope] = dict(
                cancellation_observed_in_both_batches=all(n > 0 for n in counts['cancellation']),
                suppression_observed_in_both_batches=all(n > 0 for n in counts['suppression']),
                counts_by_batch=counts,
                interpretation='Observed local pointwise cancellation is not proof of dominant interference, target correctness, training improvement, or population prevalence.')
    result = dict(schema_version=1, scope='PLASTIC fixed TRAIN 16 real +16 source; no evaluation labels',
        protocol=C.bind(C.DOC/'PROTOCOL.json'), measurement=C.bind(C.RAW/'RESULTS_PRIVATE.json'),
        report_code=C.bind(Path(__file__)), measurement_bindings=private['bindings'],
        state_checks=private['state_checks'], resources={k: private[k] for k in (
            'GPU_seconds', 'optimizer_constructed', 'optimizer_steps', 'new_fits', 'evaluation_reference_read', 'checkpoint_writes')},
        sign_definitions=dict(TOWARD='All three radial-response estimates > numerical floor',
            AWAY='All three estimates < negative numerical floor', NEAR_ZERO='All three absolute estimates <= floor',
            EPS_SENSITIVE='Any other combination', NO_RESIDUAL='Initial residual <=1e-8 input pixels'),
        denominator='Own component TOWARD and COMBINED in TOWARD/AWAY/NEAR_ZERO; unresolved counts reported separately',
        display_priority='CORNERS_0_7 first, ALL_9 and CENTER_8 separately disclosed; descriptive reporting order, not a predeclared primary outcome',
        points_are_not_independent=True, results=output, pooled=pooled, local_observation=decisions,
        prior_context=prior_context(), new_training_recommendation='NONE; no automatic fit or parameter selection')
    note = C.DOC/'EXECUTION_NOTES.md'
    if note.exists():
        result['execution_notes'] = C.bind(note)
    return result


def figures(private, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = C.DOC/'figures'
    folder.mkdir(exist_ok=True)
    arms = list(protocol['checkpoints'])
    floor = protocol['numerical_floor_px']
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, sorted(private['records'], key=lambda r: (arms.index(r['arm']), r['batch']))):
        points = point_scope(row['points'], 'REAL', 'CORNERS_0_7')
        for sign, color, marker in [('TOWARD','#3274a1','o'),('AWAY','#d4473d','o'),
                                    ('NEAR_ZERO','#b18b2f','s'),('EPS_SENSITIVE','#929292','x'),('NO_RESIDUAL','#929292','+')]:
            pp = [p for p in points if p['signs']['combined'] == sign]
            if pp:
                ax.scatter([p['response_px_at_common_reference_step']['real'][1] for p in pp],
                           [p['response_px_at_common_reference_step']['combined'][1] for p in pp],
                           s=22, alpha=.75, color=color, marker=marker, label=f'{sign} (n={len(pp)})')
        values = [abs(p['response_px_at_common_reference_step'][d][1]) for p in points for d in ('real','combined')]
        limit = max(max(values, default=floor)*1.06, floor*3)
        ax.plot([-limit,limit],[-limit,limit],color='gray',lw=.8,ls='--')
        ax.axhline(0,color='black',lw=.7); ax.axvline(0,color='black',lw=.7)
        ax.axhspan(-floor,floor,color='gray',alpha=.12)
        ax.set(xlim=(-limit,limit),ylim=(-limit,limit),title=f"{row['arm']} / batch {row['batch']}",
               xlabel='REAL-component radial response (input px)', ylabel='COMBINED radial response (input px)')
        ax.legend(fontsize=7, loc='best'); ax.grid(alpha=.15)
    fig.suptitle('Frozen TRAIN corner response; positive = toward pseudo target\nCentral differences at common scale 0.0003; no optimizer step')
    fig.savefig(folder/'functional_response.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12,4.8), constrained_layout=True)
    colors = dict(cancellation='#d4473d',suppression='#b18b2f',away_to_toward='#3274a1',nearzero_to_toward='#489780')
    names = dict(cancellation='TOWARD to AWAY', suppression='TOWARD to NEAR_ZERO',
                 away_to_toward='AWAY to TOWARD', nearzero_to_toward='NEAR_ZERO to TOWARD')
    for ax, role in zip(axes, ('REAL','SOURCE')):
        labels=[]; groups=[]
        for row in sorted(private['records'],key=lambda r:(arms.index(r['arm']),r['batch'])):
            g=aggregate(point_scope(row['points'],role,'CORNERS_0_7'),role.lower())
            groups.append(g)
            labels.append(f"{row['arm']} / batch {row['batch']}\nTOWARD {g['own_toward']} to {g['sign_counts']['combined']['TOWARD']}")
        left=np.zeros(len(groups)); right=np.zeros(len(groups))
        for key,color in colors.items():
            counts=np.array([g[key] for g in groups])
            losing=key in ('cancellation','suppression')
            base=left if losing else right
            width=-counts if losing else counts
            ax.barh(range(len(groups)),width,left=base,color=color,label=names[key])
            for i,count in enumerate(counts):
                if count: ax.text(base[i]+width[i]/2,i,str(count),ha='center',va='center',fontsize=8,color='white')
            if losing: left+=width
            else: right+=width
        limit=max(max(abs(left)),max(right),1)+2
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlim=(-limit,limit),
               xlabel='Point counts: loss of TOWARD ← 0 → gain of TOWARD',
               title=f'{role} corners: own component vs COMBINED')
        ax.axvline(0,color='black',lw=.8)
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Both opposing and assisting directional effects; corners 0–7\nSign changes are not GT accuracy; SOURCE is collateral motion')
    fig.savefig(folder/'sign_transitions.png',dpi=180); plt.close(fig)


def fmt(value, digits=4):
    return 'NA' if value is None else f'{value:.{digits}f}'


def markdown(public, protocol):
    ref_corners = public['pooled']['REF_LR5']['REAL']['CORNERS_0_7']
    ref_balance = (f"실제로 두 checkpoint·두 batch 모두 개별점 상쇄가 있었지만 반대 방향의 이득도 있었다. **REF의 코너 {ref_corners['points']}점에서는 개선 방향을 잃은 {ref_corners['cancellation']+ref_corners['suppression']}점과 얻은 {ref_corners['away_to_toward']+ref_corners['nearzero_to_toward']}점이 같고, TOWARD는 {ref_corners['own_toward']}→{ref_corners['sign_counts']['combined']['TOWARD']}로 순개수 변화가 {ref_corners['net_toward_count']}이다.** 모든 batch의 전역 gradient cosine은 양수다. 따라서 국소 간섭은 관측됐지만 replay 제거 또는 source 가중치 감소가 전체 성능을 개선한다는 결론은 아니다.")
    lines=['# Gradient transfer 진단 — fit 0', '',
        '## 결론의 범위', '',
        '동일한 고정 TRAIN 입력에서 실제 혼합 loss의 REAL·SOURCE gradient 성분이 **공유 pose/flow 파라미터를 통해 감독 좌표를 어느 방향으로 움직이는지** 측정했다. 양수는 기존 타깃을 향한 국소 반응이며 물리 GT 정확도나 6D 개선을 뜻하지 않는다. 새 학습·optimizer·checkpoint 저장·DEV 참조 읽기는 모두 0이다.', '',
        ref_balance, '',
        '아래의 `상쇄`는 REAL 성분만으로 TOWARD인 점이 COMBINED에서는 AWAY인 경우, `억제`는 NEAR_ZERO인 경우다. 두 batch에서 관측되어도 이 소표본의 국소 현상일 뿐 주된 학습 실패 원인이나 후속 학습의 성공을 확정하지 않는다.', '',
        '| checkpoint / scope | REAL TOWARD | 판정 가능 분모 | 상쇄 | 억제 | 계속 TOWARD | 미해결 | 상쇄 이미지 / 전체 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, roles in public['pooled'].items():
        for scope, g in roles['REAL'].items():
            lines.append(f"| {arm} / {scope} | {g['own_toward']} | {g['resolved_own_toward_denominator']} | {g['cancellation']} | {g['suppression']} | {g['retained_toward']} | {g['own_toward_combined_unresolved']} | {g['images_with_cancellation']}/{g['images']} |")
    lines += ['', '## 반대 방향의 이득도 함께 세기', '',
        '| checkpoint / scope | REAL TOWARD→COMBINED AWAY | TOWARD→NEAR_ZERO | AWAY→TOWARD | NEAR_ZERO→TOWARD | TOWARD 전체 REAL→COMBINED | 순개수 변화 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, roles in public['pooled'].items():
        for scope,g in roles['REAL'].items():
            lines.append(f"| {arm} / {scope} | {g['cancellation']} | {g['suppression']} | {g['away_to_toward']} | {g['nearzero_to_toward']} | {g['own_toward']}→{g['sign_counts']['combined']['TOWARD']} | {g['net_toward_count']:+d} |")
    lines += ['', '이는 부호를 넘는 점의 개수일 뿐 개선·악화의 크기를 상계한 총효용이 아니다. 같은 입력·checkpoint에서 해로운 점과 이로운 점이 함께 있으므로 선택적으로 상쇄점만 보고 source의 전체 역할을 판단하지 않는다. ALL_9와 코너 전용 표의 차이는 center를 포함하는지의 차이다.']
    lines += ['', '## 실제 입력과 계산 계약', '',
        '- PLASTIC 기존 REF TRAIN 217장 중 정렬된 unique path의 midpoint 16장, 기존 합성 512장 중 16장이다. 점수·오차·DEV로 고르지 않았다. 두 batch는 각각 REAL 8 + SOURCE 8이며 checkpoint 간 RGB·box·target·support를 재사용한다.',
        '- 기존 affine/HSV를 유지했고 새 occlusion은 없다. 실제 one2one TAL-assigned TRAIN anchor의 코너 0..7과 center 8을 기록한다. 배포 시 최고 confidence로 선택한 detection의 좌표를 측정한 것이 아니다. 코너 전용을 먼저 설명하고 ALL_9/support와 center를 별도 공개한다. 이는 기술적 보고 순서이며 protocol에 없는 사전 primary outcome을 만들지 않는다.',
        '- 실제 full mixed graph의 global 분모와 RLE aggregate clamp 미분 gate를 보존한 성분 분해다. REAL 성분은 별도 real-only 학습을 재정규화한 loss가 아니다.',
        '- R0는 E2E 시작 .8/.2, REF는 마지막 epoch .1/.9이며 저장된 frozen EMA이다. 원래 online 상태·optimizer moments·weight decay·clipping·EMA step을 복원한 것이 아니다.',
        '- `d_role = -g_role / max(||g_real||, ||g_source||, ||g_combined||)`. 각 방향에 같은 분모를 써 상대 크기와 가산성을 보존한다. `theta ± eps·||theta||·d_role`의 중앙 유한차분 방향 반응을 eps=0.0003에 맞춰 표시한다. 단위는 input640 px이며 실제 한 optimizer step의 이동량이 아니다.',
        '- 공통 scale은 **같은 checkpoint·batch 내부**의 세 방향에 공통이다. checkpoint 간에는 gradient norm과 E2E branch 가중치가 다르므로 response 절댓값 증가를 학습 능력 증가나 실제 학습률 효과로 해석하지 않는다.',
        '- eps는 사전 고정한 0.0001 / 0.0003 / 0.001이다. 세 추정이 모두 +0.001px 초과일 때 TOWARD, 모두 −0.001px 미만일 때 AWAY, 모두 절댓값 0.001px 이하일 때 NEAR_ZERO, 나머지는 EPS_SENSITIVE다. 초기 잔차 ≤1e−8px는 NO_RESIDUAL이다.',
        '- 작은 대칭 perturbation의 반올림 영향을 줄이기 위해 matmul/cuDNN TF32를 모두 끈 FP32 진단이다. 원래 학습 전체를 같은 정밀도 설정으로 재실행한 것이 아니다.',
        '- 판정 분모는 REAL=TOWARD 중 COMBINED가 TOWARD/AWAY/NEAR_ZERO인 점이다. EPS_SENSITIVE·NO_RESIDUAL을 분모에서 제외하되 개수는 숨기지 않는다. 같은 이미지의 점은 독립 표본이 아니다.', '',
        '## checkpoint / batch별 결과', '',
        '| checkpoint / batch | REAL 점 | REAL TOWARD / 분모 | 상쇄 / 억제 / 미해결 | REAL 방향 반응 중앙값 | COMBINED 중앙값 | gradient cosine |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in public['results']:
        g=r['groups']['REAL']['ALL_9']; responses=g['response_px_at_common_middle_step']
        lines.append(f"| {r['arm']} / {r['batch']} | {g['points']} | {g['own_toward']}/{g['resolved_own_toward_denominator']} | {g['cancellation']} / {g['suppression']} / {g['own_toward_combined_unresolved']} | {fmt(responses['real']['median'])} | {fmt(responses['combined']['median'])} | {fmt(r['gradient_cosine'])} |")
    lines += ['', '중앙값은 전체 해당 REAL 감독점의 중간 epsilon 반응이다. 상쇄 비율의 판정 가능 subset과 분모가 다르다. 위 미해결 열은 REAL=TOWARD 조건 이후의 COMBINED 미해결 수다. REAL 방향 자체의 EPS_SENSITIVE도 제외되는지 확인할 전체 부호 수는 아래와 같다.', '',
        '| checkpoint / batch | REAL 방향 TOWARD / AWAY / NEAR_ZERO / EPS / NO_RESIDUAL | COMBINED 같은 순서 | 둘 중 EPS인 점 / 전체 |',
        '|---|---:|---:|---:|']
    for r in public['results']:
        g=r['groups']['REAL']['ALL_9']
        values=[' / '.join(str(g['sign_counts'][d][s]) for s in SIGNS) for d in ('real','combined')]
        lines.append(f"| {r['arm']} / {r['batch']} | {values[0]} | {values[1]} | {g['any_pair_eps_sensitive']}/{g['points']} |")
    lines += ['', '아래 산점도는 코너 0..7만 그리며 점 색은 COMBINED의 세 epsilon 부호 안정성 기준이다. 회색으로 표시되지 않은 점도 REAL 방향이 epsilon-sensitive일 수 있으므로 산점도 한 장만으로 상쇄 수를 세지 않는다. 세 epsilon의 일치는 이 유한차분 검사에서의 수치 안정성이지 통계적 확신도가 아니다.', '',
        '![Functional response](figures/functional_response.png)', '',
        '## SOURCE 좌표의 부수 반응', '',
        '| checkpoint | SOURCE 점 | SOURCE TOWARD / 분모 | TOWARD→AWAY | TOWARD→NEAR_ZERO | AWAY→TOWARD | NEAR_ZERO→TOWARD | 순 TOWARD 변화 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, roles in public['pooled'].items():
        g=roles['SOURCE']['ALL_9']
        lines.append(f"| {arm} | {g['points']} | {g['own_toward']}/{g['resolved_own_toward_denominator']} | {g['cancellation']} | {g['suppression']} | {g['away_to_toward']} | {g['nearzero_to_toward']} | {g['net_toward_count']:+d} |")
    lines += ['', 'SOURCE 표는 해당 감독 좌표의 국소 이동이다. 합성 validation·retention 성능을 새로 측정한 것이 아니다.', '',
        '아래 그림은 코너의 TOWARD 상실과 획득을 양쪽에 함께 보이고 각 행에 전체 TOWARD 점수를 표시한다. ALL_9 수치를 코너 수치와 혼합하지 않는다.', '',
        '![Sign transitions](figures/sign_transitions.png)', '',
        '## gradient와 수치 검사', '',
        '| checkpoint / batch | REAL norm | SOURCE norm | COMBINED norm | 분해 max abs error | 좌표 가산 잔차 median / max px |',
        '|---|---:|---:|---:|---:|---:|']
    for r in public['results']:
        gg=r['gradients']
        lines.append(f"| {r['arm']} / {r['batch']} | {fmt(gg['real']['norm'])} | {fmt(gg['source']['norm'])} | {fmt(gg['combined']['norm'])} | {r['max_gradient_partition_error']:.3g} | {r['median_coordinate_additivity_residual_px']:.3g} / {r['max_coordinate_additivity_residual_px']:.3g} |")
    lines += ['', 'head/flow norm과 전체 부호 전이표는 [RESULTS.json](RESULTS.json)에 있다. 좌표 가산 잔차는 중간 epsilon의 벡터 중앙차분 `COMBINED − REAL − SOURCE`이며 gradient 분해 오차와 다른 검사다. 유한 perturbation·부동소수점 오차 때문에 정확히 0일 필요는 없다. 이번 최대 좌표 가산 잔차 0.000138px는 사전 부호 floor 0.001px보다 작았다.', '',
        'flow gradient norm은 네 조건에서 모두 0이었다. 실제 두 branch의 RLE pre-clamp 값이 모두 음수이고 shared derivative gate가 0인 원 criterion의 동작이다. flow 경로를 누락하거나 detach해서 0으로 만든 것이 아니며, 전체 full-loss gradient와 성분 합을 flow 포함 허용 파라미터 전부에서 대조했다. 위치/visibility를 통한 head gradient는 남아 있었다.', '',
        '전역 cosine이 네 조건 모두 양수라는 사실은 이 동일 normalization 아래 source 성분이 전체 REAL 성분 loss의 일차 감소 방향에 기여함을 뜻한다. 개별 좌표의 radial response는 그 전체 목적함수와 같지 않으므로 일부점 역전과 모순되지 않는다. 어느 지표도 물리 정확도를 직접 측정하지 않는다.', '',
        'COMBINED 방향의 REAL 목적함수 일차 감소량을 REAL 방향의 감소량으로 나눈 값은 `1 + cosine·||g_SOURCE||/||g_REAL||`다. 이는 같은 scale에서의 미분값 비율이며 AdamW 또는 실제 유한 step의 성능 비율이 아니다: '+', '.join(f"{r['arm']}/batch{r['batch']}={r['combined_vs_real_first_order_REAL_objective_descent_factor']:.5f}" for r in public['results'])+'.', '',
        '각 ± perturbation에서 assignment·target·support 동일성과 복원 후 prediction bit-exact를 실행 코드가 단언했다. 아래 879개 state tensor exactness도 계측 당시의 runtime assertion 기록이다. 전체 ± 좌표와 state snapshot을 별도로 보존하지 않았으므로 사후 독립 두 번째 전체-state 감사라고 부르지 않는다. 저장 상태 tensor와 grad buffer 상태:', '']
    for arm, state in public['state_checks'].items():
        lines.append(f"- {arm}: {state['tensors']} state tensors exact={state['exact']}; parameter `.grad` buffers all None={state['all_parameter_grad_buffers_none']}.")
    lines += ['', f"실제 계측 GPU 구간 {public['resources']['GPU_seconds']:.3f}s. optimizer 생성 {int(public['resources']['optimizer_constructed'])}, optimizer updates {public['resources']['optimizer_steps']}, fits {public['resources']['new_fits']}, checkpoint writes {public['resources']['checkpoint_writes']}. CPU 보고서 렌더 시간은 이 GPU 구간에 포함하지 않는다.", '',
        '첫 시도는 symlink 경로 해석으로 합성 RGB의 `syn__` alias가 사라져 label 연결에 실패했고, 모델 로딩 전 assertion으로 중단했다. 실패 fixture/cache를 보존하고 같은 32개 image/label hash와 seed로 새 fixture의 alias만 복원했다. 두 번째 시도는 설치 패키지 코드의 외부 경로 hash 기록에서 중단했고 새 namespace의 binding 함수만 수정했다. 둘 다 모델·gradient 결과 관측 전 오류이며 입력 선택이나 epsilon을 사후 조정하지 않았다. [실행 정정 기록](EXECUTION_NOTES.md)을 함께 보존한다.', '',
        '현재 설치된 Albumentations의 `ImageCompression quality_range` 인자 경고가 있었다. 이 진단에서 환경을 변경하지 않았고 checkpoint 간에는 같은 실제 tensor를 사용했지만, 과거 학습 당시의 증강 tensor를 bit-exact 재생했다고 주장하지 않는다.', '',
        '## 과거 진단과 이번에 추가한 정보', '',
        '이전 original-affine Plastic REF의 전역 cosine은 +.0108, −.0229, −.0586, −.4096으로 혼합이었다. 마지막 batch의 source projection은 −.4551이어서 전역 real-gradient 성분을 부분 상쇄했지만 역전하지는 않았다. Wood R0는 4/4 양수였고 Wood REF는 2양·2음이었다. 음의 cosine만으로 source를 실패 원인으로 확정하지 않았던 이유다.', '',
        '이전 진단은 real/source 소배치를 따로 정규화하고 REF criterion을 5회 advance했다. 이번은 같은 mixed16 graph의 분모/gate를 유지하고 마지막 epoch에 맞춰 4회 advance한다. 따라서 수치의 exact replication 비교가 아니다. 이번에는 추가로 실제 공유 파라미터 방향이 개별 감독 좌표에 미치는 반응을 측정한다. 과거 coordinate-leaf descent와도 다른 질문이다.', '',
        '## 판정과 남은 질문', '']
    for arm, scopes in public['local_observation'].items():
        d=scopes['ALL_9']
        lines.append(f"- {arm}: 두 batch의 상쇄 개수 {d['counts_by_batch']['cancellation']}, 억제 개수 {d['counts_by_batch']['suppression']}. 상쇄 양 batch 관측={d['cancellation_observed_in_both_batches']}, 억제 양 batch 관측={d['suppression_observed_in_both_batches']}.")
    lines += ['', 'REAL 타깃 방향이 COMBINED에서 사라지거나 역전하는 점이 양 batch에 존재하면 **그 고정 입력·checkpoint에서의 국소적인 source 성분 간섭**을 지지한다. 몇 점의 관측과 전체 타깃 개선을 막는 지배적 원인은 구분한다. 관측되지 않거나 epsilon 민감성이 크면 이 검사만으로 간섭을 채택하지 않는다. 두 방향 모두 개선해도 표현 능력·학습 궤적·일반화가 해결됐다는 뜻은 아니다.', '',
        '기존 REF pseudo와 synthetic target 추종만 검사했다. 물리 signed-axis/corner 정답, 6D 정확도, 새로운 데이터에서의 일반화, 장기 AdamW 업데이트 또는 source 가중치 변경의 효과는 이 결과로 확정할 수 없다. 16 TRAIN 프레임·두 frozen checkpoint의 작은 결정론적 진단이며 DEV 기반 선택·새 학습 선택·자동 fit은 하지 않는다.', '',
        '재현: [README.md](README.md). 고정 사전 명세: [PROTOCOL.json](PROTOCOL.json). 원자료의 hash binding과 공개 집계: [RESULTS.json](RESULTS.json).']
    return '\n'.join(lines)+'\n'


def self_test():
    def p(own, combined, image='synthetic_a', corner=0):
        values={'TOWARD':[.01,.02,.03],'AWAY':[-.01,-.02,-.03], 'NEAR_ZERO':[0.,0.,0.],
                'EPS_SENSITIVE':[-.01,.02,.03]}
        return dict(image_id=image, role='REAL', corner=corner, residual_px=1.,
            response_px_at_common_reference_step=dict(real=values[own],source=values['NEAR_ZERO'],combined=values[combined]),
            signs=dict(real=own,source='NEAR_ZERO',combined=combined))
    points=[p('TOWARD','TOWARD'),p('TOWARD','AWAY'),p('TOWARD','NEAR_ZERO'),p('TOWARD','EPS_SENSITIVE'),p('AWAY','AWAY','synthetic_b',8)]
    g=aggregate(points,'real')
    assert (g['own_toward'],g['resolved_own_toward_denominator'],g['cancellation'],g['suppression'],g['retained_toward'],g['own_toward_combined_unresolved'])==(4,3,1,1,1,1)
    assert g['images']==2 and g['images_with_cancellation']==1
    assert len(point_scope(points,'REAL','CORNERS_0_7'))==4
    assert len(point_scope(points,'REAL','CENTER_8'))==1
    assert stable_sign([.001,.001,.001],.001)=='NEAR_ZERO'
    assert stable_sign([.001,.002,.003],.001)=='EPS_SENSITIVE'
    assert stable_sign([-.002,-.003,-.004],.001)=='AWAY'
    assert summary([])['median'] is None
    assert aggregate([],'real')['cancellation_fraction'] is None
    balanced=aggregate(points+[p('AWAY','TOWARD'),p('NEAR_ZERO','TOWARD')],'real')
    assert balanced['away_to_toward']==balanced['nearzero_to_toward']==1
    assert balanced['net_toward_count']==-1  # One own-TOWARD point remains unresolved.
    print('REPORT_SELF_TEST_PASS')


def main():
    protocol=C.read(C.DOC/'PROTOCOL.json')
    private=C.read(C.RAW/'RESULTS_PRIVATE.json')
    for binding in protocol['inputs']+private['bindings']:
        C.verify(binding)
    public=build_public(protocol,private)
    figures(private,protocol)
    public['figures']=[C.bind(C.DOC/'figures'/name) for name in ('functional_response.png','sign_transitions.png')]
    C.save(C.DOC/'RESULTS.json',public)
    C.save(C.DOC/'REPORT_KO.md',markdown(public,protocol))
    C.save(C.DOC/'README.md', '''# Gradient transfer diagnostic v1

이미 승인된 no-fit 진단의 공개 보고서다. 새 fit, optimizer 생성/step, checkpoint write, DEV reference read는 허용하지 않는다. 원래 이미지·target 배열·frame ID는 private namespace에 유지한다.

## CPU 보고서 재현

저장된 private 결과가 있는 동일 환경에서 실행한다. GPU 측정은 다시 실행하지 않는다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_gradient_transfer_diagnostic_v1.report --self-test
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -m scripts.research.pallet_gradient_transfer_diagnostic_v1.report
```

입력: `PROTOCOL.json`, `data/pallet/results/pallet_gradient_transfer_diagnostic_v1/RESULTS_PRIVATE.json`, 기존 source bindings. 모든 binding을 검증한 뒤 집계·그림을 재생성한다. Private 데이터가 없으면 재계측이나 데이터 교체로 우회하지 않고 중단한다.

출력: `RESULTS.json`, `REPORT_KO.md`, `figures/functional_response.png`, `figures/sign_transitions.png`. 그림은 실제 측정 데이터의 집계/응답이며 원래 이미지나 좌표를 그리지 않는다. 공개 JSON에는 frame ID·bbox·intrinsics·target/predicted coordinate 배열을 넣지 않는다.

## 이미 실행한 계측 경로

`probe.prepare`가 기존 TRAIN midpoint 입력과 사전 protocol을 잠갔고, `probe.run`이 functional ±epsilon으로 측정했다. 측정 실행·host GPU 접근은 부모가 별도로 관리했다. 이 README는 재실행 허가가 아니다.

`RESULTS.json`의 raw measurement hash·protocol/code bindings·state audit를 먼저 확인한다. 양수는 저장 타깃 방향이지 물리 정답 방향이 아니다. REAL 성분은 full mixed loss의 성분이며 standalone real-only training과 같지 않다.
''')
    print('REPORT_COMPLETE',C.DOC/'REPORT_KO.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    self_test() if args.self_test else main()
