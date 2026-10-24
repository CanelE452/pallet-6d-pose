#!/usr/bin/env python3
"""Saved-summary reporting only. No model, packet, ledger or source-input reads."""
import csv
import hashlib
import html
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

V = Path(__file__).parent.parent
PREP = Path(__file__).parent
OUT = V / 'numeric_readout_v1'
SCOPE = V / 'ROOT_POST_AUDIT_SAVED_NUMERICAL_READOUT_SCOPE_C.json'
SCOPE_SHA = '6a7d3f007cdffd6ff40c4979de33b765f9db679d9da14cd3a4efa71fec463ad5'
MODELS = ['v61', 'legacy', 'C52', 'C53', 'C54', 'C55', 'C56', 'C57', 'C58', 'C59']
PUBLIC_MODELS = [m for m in MODELS if m != 'legacy']


def safe(p):
    p = Path(p)
    assert not any(re.search(r'(^|_)v112($|_|\.)', x, re.I) for x in p.parts), str(p)
    for q in [p, *p.parents]:
        assert not q.is_symlink(), str(q)
    return p


def sha(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()


def binding(p):
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': safe(p).stat().st_size}


def read_bound(b):
    p = safe(b['path'])
    assert p.suffix == '.json', 'Only explicit saved JSON summaries are read'
    assert sha(p) == b['sha256'], str(p)
    return json.loads(p.read_text())


def write_json(name, obj, folder=OUT):
    p = folder / name
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(name, rows):
    assert rows
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def ratio(a, b):
    return a / b if b else None


def from_cm(cm):
    d, e = cm
    dn, en = sum(d), sum(e)
    n = dn + en
    u = d[2] + e[2]
    return dict(N=n, CM_D_E_by_D_E_U=cm, D_N=dn, E_N=en,
                D_precision=ratio(d[0], d[0] + e[0]), D_recall=ratio(d[0], dn),
                E_precision=ratio(e[1], d[1] + e[1]), E_recall=ratio(e[1], en), U=u, U_rate=ratio(u, n))


def cm_from_metric(r):
    return [[r['TP'], r['FN_decided'], sum(r[x] for x in ('U_visible', 'out_of_task_S_visible', 'other_unresolved_visible'))],
            [r['FP'], r['TN'], sum(r[x] for x in ('U_occluded', 'out_of_task_S_occluded', 'other_unresolved_occluded'))]]


def add_cm(cms):
    return [[sum(c[i][j] for c in cms) for j in range(3)] for i in range(2)]


def close(a, b):
    return a is b if a is None or b is None else math.isclose(a, b, rel_tol=0, abs_tol=1e-14)


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def cmtext(cm):
    return f"[{','.join(map(str, cm[0]))}]; [{','.join(map(str, cm[1]))}]"


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join('| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)


def html_table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def link(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def run():
    assert sha(SCOPE) == SCOPE_SHA
    scope = json.loads(SCOPE.read_text())
    assert scope['approved_saved_output_reporting_only'] is True
    assert [str(OUT), str(PREP)] == scope['allowed_new_directories']
    audit = read_bound(scope['audit_review'])
    complete = read_bound(scope['saved_evaluation_complete'])
    source = read_bound(scope['source_report'])
    assert audit['completed'] and audit['PASS'] and complete['completed'] and complete['PASS']
    current_b = complete['outputs']['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json']
    public_b = complete['outputs']['publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json']
    current, public = read_bound(current_b), read_bound(public_b)
    primary = audit['current']['primary']['ALL']
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    check('all10 fixed order', list(primary) == MODELS and complete['models'] == MODELS)
    check('fixed primary1237 D1147 E90', current['primary_D_E_N'] == 1237 and current['primary_D'] == 1147 and current['primary_E'] == 90)
    for m in MODELS:
        r = from_cm(primary[m]['CM_D_E_by_D_E_U'])
        for k in r:
            check(f'primary {m} {k}', r[k] == primary[m][k] if isinstance(r[k], (int, list)) else close(r[k], primary[m][k]))
    material_rows, same_slot = [], {}
    for cohort in ('paired_same_query_RGB', 'same_slot_pipeline'):
        for m in MODELS:
            rr = [r for r in current['tables']['metrics'] if r['cohort'] == cohort and r['task'] == 'D_vs_E' and r['model'] == m]
            check(f'{cohort} {m} two materials', {r['group'] for r in rr} == {'WOOD_PRIMARY105', 'SQUARE118'})
            aggregate = from_cm(add_cm([cm_from_metric(r) for r in rr]))
            if cohort == 'paired_same_query_RGB':
                check(f'primary material sum {m}', aggregate == primary[m])
                for r in rr:
                    material_rows.append(dict(model=m, material=r['group'], frames=r['frames'], **from_cm(cm_from_metric(r))))
            else:
                check(f'same slot {m} N1248', aggregate['N'] == 1248)
                same_slot[m] = aggregate
    native = current['tables']['primary_native8_D_E_with_U']
    check('native240 all10 all8 all3groups', len(native) == 240 and {r['native_slot'] for r in native} == set(range(8)) and {r['model'] for r in native} == set(MODELS))
    for m in MODELS:
        rr = [r for r in native if r['group'] == 'ALL' and r['model'] == m]
        check(f'native sum {m}', len(rr) == 8 and add_cm([cm_from_metric(r) for r in rr]) == primary[m]['CM_D_E_by_D_E_U'])
    self_rows = current['tables']['human_self_binary_response']
    self_aggregate = {}
    for cohort, n in [('same_slot_pipeline', 136), ('paired_same_query_RGB', 21)]:
        self_aggregate[cohort] = {}
        for m in MODELS:
            rr = [r for r in self_rows if r['cohort'] == cohort and r['model'] == m]
            row = dict(N=sum(r['human_S_support'] for r in rr), D=sum(r['visible_response'] for r in rr), E=sum(r['binary_hidden_response'] for r in rr), U=sum(r['unresolved'] for r in rr))
            check(f'self {cohort} {m}', row['N'] == n and row['D'] + row['E'] + row['U'] == n and all(not r['learned_self_prediction'] and not r['external_vs_self_identified'] for r in rr))
            self_aggregate[cohort][m] = row
    ref = audit['current']['all_registered_humanD_existing_reference_LE3_association']
    manual = {}
    for m in MODELS:
        rr = [r for r in current['tables']['manual_near_D'] if r['model'] == m]
        manual[m] = dict(N=sum(r['N'] for r in rr), D=sum(r['pred_visible'] for r in rr), E=sum(r['pred_binary_hidden'] for r in rr), U=sum(r['unresolved'] for r in rr))
        check(f'reference461 manual376 {m}', ref[m]['N'] == sum(ref[m][k] for k in ['D', 'E', 'U']) == 461 and manual[m]['N'] == sum(manual[m][k] for k in ['D', 'E', 'U']) == 376)
    pub = public['cohorts']['FULL360']['ALL']
    check('public9 and legacy missing', set(pub['models']) == set(PUBLIC_MODELS) and public['legacy_baseline']['status'] == 'MISSING_IN_REVIEWED_LINEAGE')
    public_rows, owner_rows, public_native_rows = [], [], []
    for group in ['ALL', *map(str, range(8))]:
        rec = public['cohorts']['FULL360'][group]
        for m in PUBLIC_MODELS:
            b = rec['models'][m]['binary_T_nonT']
            cm = b['known_truth_confusion']
            check(f'public known/unknown {group} {m}', sum(map(sum, cm)) == b['known_truth'] and b['known_truth'] + b['unknown_truth'] == b['N'] and b['precision_excludes_unknown_truth'] and b['weak_reference_only'])
            for i, cls in enumerate(['T', 'NON_T']):
                c = b['per_class'][cls]
                check(f'public ratios {group} {m} {cls}', close(c['precision_on_known_truth'], ratio(cm[i][i], cm[0][i] + cm[1][i])) and close(c['recall_including_prediction_U'], ratio(cm[i][i], sum(cm[i]))))
            row = dict(group=group, model=m, N=b['N'], known=b['known_truth'], truth_UNKNOWN=b['unknown_truth'], CM_T_NON_T_by_T_NON_T_U=cm, prediction_U_all=b['prediction_U_all'], **{f'{c}_{k}': b['per_class'][c][k] for c in ['T', 'NON_T'] for k in ['precision_on_known_truth', 'recall_including_prediction_U']})
            (public_rows if group == 'ALL' else public_native_rows).append(row)
            if group == 'ALL' and 'owner3' in rec['models'][m]:
                o = rec['models'][m]['owner3']
                owner_rows.append(dict(model=m, **o))
    check('public204known156unknown', all(r['N'] == 360 and r['known'] == 204 and r['truth_UNKNOWN'] == 156 for r in public_rows))
    src_rows = []
    for arm, x in source['models'].items():
        m = arm.split('_')[0]
        check(f'source{m} TRAIN327 perfect', x['TRAIN']['known'] == 327 and x['TRAIN']['CM_D_E_by_D_E_U'] == [[218, 0, 0], [0, 109, 0]])
        check(f'source{m} H45 Utruth315', x['HOLDOUT']['known'] == 45 and x['HOLDOUT']['truth_UNKNOWN'] == 315 and x['by_authority']['SOURCE_RGB_SEMANTIC_END_WEAK']['HOLDOUT']['known'] == 0)
        for role in ['TRAIN', 'HOLDOUT']:
            src_rows.append(dict(model=m, role=role, **x[role]))
    check('source six synthetic bases', source['HOLDOUT_bases'] == [42, 50, 74, 75, 83, 104] and source['HOLDOUT_direct_known_synthetic_only_real_end0'])
    check('source native6/7 known0', all(x['by_native'][str(k)]['ALL']['known'] == 0 for x in source['models'].values() for k in [6, 7]))
    check('inputU is separate45/1784', current['original_input_U'] == 45 and current['original_slots'] == 1784 and current['tables']['availability']['C58_direct_total_U'] == 45 and current['tables']['availability']['C59_direct_total_U'] == 45)
    check('C58/C59 attempt no improvement vs C57 two class correct', all(primary[m]['CM_D_E_by_D_E_U'][0][0] < primary['C57']['CM_D_E_by_D_E_U'][0][0] and primary[m]['CM_D_E_by_D_E_U'][1][1] < primary['C57']['CM_D_E_by_D_E_U'][1][1] for m in ['C58', 'C59']))
    OUT.mkdir(exist_ok=False)
    write_csv('PRIMARY_ALL10.csv', [dict(model=m, **primary[m]) for m in MODELS])
    write_csv('SAME_SLOT_ALL10.csv', [dict(model=m, **same_slot[m]) for m in MODELS])
    write_csv('PRIMARY_MATERIAL_ALL10.csv', material_rows)
    for k, rr in current['tables'].items():
        if isinstance(rr, list) and rr:
            write_csv('SAVED_' + k.upper() + '.csv', rr)
    write_csv('SELF_TWO_COHORTS_ALL10.csv', [dict(cohort=c, model=m, **x) for c, mm in self_aggregate.items() for m, x in mm.items()])
    write_csv('REFERENCE461_AND_MANUAL376_ALL10.csv', [dict(cohort=c, model=m, **r) for c, mm in [('existing_reference461', ref), ('manual376', manual)] for m, r in mm.items()])
    write_csv('PUBLIC_WEAK_BINARY9.csv', public_rows)
    write_csv('PUBLIC_WEAK_OWNER3_8.csv', owner_rows)
    write_csv('PUBLIC_WEAK_NATIVE9.csv', public_native_rows)
    write_csv('SOURCE_TRAIN_AND_HOLDOUT_C58_C59.csv', src_rows)
    inputs = dict(scope=binding(SCOPE), audit_review=scope['audit_review'], saved_evaluation_complete=scope['saved_evaluation_complete'], current_summary=current_b, public_summary=public_b, source_report=scope['source_report'])
    data = dict(completed=True, PASS=True, report_kind='SAVED_TABLE_NUMERICAL_READOUT_ONLY', created_UTC=datetime.now(timezone.utc).isoformat(), inputs=inputs,
                models=MODELS, primary=primary, same_slot=same_slot, material=material_rows, native240=native,
                self_two_cohorts=self_aggregate, existing_reference461=ref, manual376=manual, public_weak9=public_rows, public_weak_owner8=owner_rows,
                public_coverage=public['coverage'], public_legacy_missing=public['legacy_baseline'], source_train_holdout=src_rows,
                source_HOLDOUT_bases=source['HOLDOUT_bases'], source_known_native6_7=0,
                original_input_U=dict(N=1784, U=45, rate=45/1784),
                conclusion='이번 native onehot 시도 C59와 matched zero8 C58은 고정 실사 주평가에서 이전 C57보다 D·E 정답 수가 모두 적다. 실사 개선을 확인하지 못했으며 어떤 모델도 선택·채택하지 않는다.',
                limits=['10모델은 같은 역할·학습 이력의 교환 가능한 비교군이 아니다. v61은 historical pixel presence, legacy는 historical realCAL 노출을 포함한다.',
                        'current 인간 semantic authority는 유효한 평가 기준이다. 반복 노출 cohort이므로 fresh/blind 검증이 아니다.',
                        'SELF는 binary D/E/U 반응이며 self 원인 분류·external 식별 성능이 아니다. 136과21의 차이115에 대한 세부 제외원인은 저장표에 없으므로 추정하지 않는다.',
                        'reference461과 manual376은 별개 고정 association이다. 주평가 필터·좌표 보정·physical native gold가 아니다.',
                        'public weak는 point/crop owner T/F/BG의 별도 task다. native D/E/S 및 direct-end 정답으로 승격하지 않는다.',
                        'source TRAIN327은 apparent fit이며 HOLD45는6 synthetic bases의 약한 직접부위 감독이다.315 UNKNOWN과 real weak HOLD0을 보존한다. native6/7 known0이다.',
                        'stored parameter count116407은 같아도 zero8와 onehot의 유효 함수공간은 같지 않고 shared backbone 영향이 있다. 실패의 다른 원인을 확정하지 않는다.',
                        '원 q/L/native/초기 crop을 유지한 저장 결과만 읽었다. 새 source/실사 pixel·배열·weights·모델·forward·fit·PnP·SAM·threshold fit은 없다.'],
                selected_model=None, adopted=False, fully_solved=False)
    write_json('RESULTS_KO.json', data)

    # Both panels use all10 fixed order and show exact predicted counts for every row.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(20, 10.6), gridspec_kw={'wspace': .16})
    colors = ['#2464a4', '#d87721', '#929aa4']
    for i, (ax, dn, title) in enumerate(zip(axes, [1147, 90], ['Human D / visible: N = 1147', 'Human E / external: N = 90'])):
        for y, m in enumerate(MODELS):
            counts = primary[m]['CM_D_E_by_D_E_U'][i]
            left = 0
            for j, n in enumerate(counts):
                ax.barh(y, n, left=left, height=.66, color=colors[j], label=['Pred D', 'Pred E', 'Pred U'][j] if y == 0 and i == 0 else None)
                if n >= dn * .105:
                    ax.text(left + n / 2, y, str(n), ha='center', va='center', color='white', weight='bold', fontsize=12)
                left += n
            ax.text(dn * 1.025, y, f'D {counts[0]:4}   E {counts[1]:4}   U {counts[2]:3}', va='center', family='DejaVu Sans Mono', fontsize=10.8)
        ax.set_yticks(range(10), MODELS)
        ax.invert_yaxis()
        ax.set_xlim(0, dn * 1.68)
        ax.set_xticks([0, dn / 4, dn / 2, dn * 3 / 4, dn], [f'{x:g}' for x in [0, dn / 4, dn / 2, dn * 3 / 4, dn]])
        ax.set_xlabel('Saved prediction count (all truth rows, including U)')
        ax.set_title(title, weight='bold', pad=16)
        ax.axvline(dn, color='#777', lw=.7, linestyle=':')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.13)
        ax.set_axisbelow(True)
    fig.suptitle('V200 | Fixed real primary cohort: 1237 queries = D1147 + E90', fontsize=20, weight='bold', y=.967)
    fig.text(.5, .913, 'C58 zero8 control / C59 native onehot: both RAW, frozen before evaluation. No model or threshold selection.', ha='center', fontsize=12)
    fig.legend(loc='lower center', bbox_to_anchor=(.5, .091), ncol=3, frameon=False, fontsize=12)
    fig.text(.5, .064, 'CM: truth rows D,E; prediction columns D,E,U. Primary U differs from original input U45/1784 (2.52%).', ha='center', fontsize=11)
    fig.text(.5, .039, 'Repeatedly exposed human-semantic cohort; not fresh/blind. v61 pixel-presence and legacy realCAL histories differ.', ha='center', fontsize=11)
    fig.text(.5, .016, 'C58/C59 have fewer correct D and E than fixed prior C57. No improvement, causal conclusion or adoption.', ha='center', fontsize=11, weight='bold')
    fig.subplots_adjust(top=.86, bottom=.17, left=.06, right=.99)
    fig.savefig(OUT / 'PRIMARY_ALL10_COUNTS.png', dpi=140)
    fig.savefig(OUT / 'PRIMARY_ALL10_COUNTS.svg')
    plt.close(fig)

    sections = []
    def add(title, text='', headers=None, rows=None):
        sections.append((title, text, headers, rows))
    add('고정 실사 주평가: 이번 번호 조건 시도에서 개선 확인 없음',
        'C58(zero8 control)은 D 650/1147·E 29/90, C59(native onehot)는 D 603/1147·E 39/90이다. 이전 C57은 D 729/1147·E 52/90으로, 새 두 팔보다 두 클래스의 정답 수가 모두 많다. 이 고정 결과는 이번 시도의 개선을 뒷받침하지 않는다. C57을 새로 선택하거나 어떤 모델을 채택하는 절차도 수행하지 않았다. 실패 원인·일반적 불가능성·통계적 우열을 확정하지 않는다.')
    add('주평가 10모델: N1237 = D1147 + E90',
        'D는 인간 semantic visible, E는 인간 semantic external이다. CM은 [truth D: pred D,E,U]; [truth E: pred D,E,U]. Precision은 known D/E의 해당 예측 열을 분모로, recall은 U를 포함한 해당 truth 전체를 분모로 쓴다. U/N도 명시했다. 초기 품질·owner·reference error로 주평가를 거르지 않았다.',
        ['모델', 'CM [D,E,U] 두 행', 'D P', 'D R', 'E P', 'E R', 'U/N'],
        [[m, cmtext(primary[m]['CM_D_E_by_D_E_U']), pct(primary[m]['D_precision']), pct(primary[m]['D_recall']), pct(primary[m]['E_precision']), pct(primary[m]['E_recall']), f"{primary[m]['U']}/1237 ({pct(primary[m]['U_rate'])})"] for m in MODELS])
    add('동일 슬롯 cohort는 별도: N1248', '위1237은 paired_same_query_RGB 주평가다. 아래 same_slot_pipeline1248은 분모가 달라 합치지 않는다.',
        ['모델', 'CM [D,E,U] 두 행', 'D P/R', 'E P/R', 'U/N'], [[m, cmtext(same_slot[m]['CM_D_E_by_D_E_U']), pct(same_slot[m]['D_precision'])+' / '+pct(same_slot[m]['D_recall']), pct(same_slot[m]['E_precision'])+' / '+pct(same_slot[m]['E_recall']), f"{same_slot[m]['U']}/1248 ({pct(same_slot[m]['U_rate'])})"] for m in MODELS])
    add('WOOD·SQUARE: 주평가 material별10모델', '등록 frame은 WOOD105·SQUARE118이다. 주평가 슬롯은 WOOD645(D616/E29), SQUARE592(D531/E61)이다. 두 material CM의 합은 각 모델 주평가 CM과 일치한다.',
        ['material', '모델', 'CM [D,E,U]', 'D P/R', 'E P/R', 'U/N'], [[r['material'], r['model'], cmtext(r['CM_D_E_by_D_E_U']), pct(r['D_precision'])+' / '+pct(r['D_recall']), pct(r['E_precision'])+' / '+pct(r['E_recall']), f"{r['U']}/{r['N']}"] for r in material_rows])
    nrows = []
    for k in range(8):
        rr = {r['model']: from_cm(cm_from_metric(r)) for r in native if r['group'] == 'ALL' and r['native_slot'] == k}
        nrows.append([k, f"{rr['C58']['N']} ({rr['C58']['D_N']}/{rr['C58']['E_N']})", *[f"{rr[m]['CM_D_E_by_D_E_U'][0][0]}/{rr[m]['D_N']}; {rr[m]['CM_D_E_by_D_E_U'][1][1]}/{rr[m]['E_N']}; U{rr[m]['U']}" for m in ['C57', 'C58', 'C59']]])
    add('native0..7 전체 유지', '아래는 C57·새 두 팔의 정답 수를 표시한 축약표다. 모든10모델×8native×ALL/WOOD/SQUARE=240행의 CM/P/R/U는 연결 CSV에 그대로 보존했다. source 직접 감독이 없던 native6·7도 실사 평가에서 제외하지 않았다.',
        ['native', 'N (D/E)', 'C57 D정답; E정답; U', 'C58 D정답; E정답; U', 'C59 D정답; E정답; U'], nrows)
    add('SELF136과 paired21: 별개 binary response',
        'all SELF136은 same_slot_pipeline이며 모두 WOOD다. paired_same_query_RGB는21이다. 차이115는 기존 paired_same_query_RGB_D_vs_ES eligibility 밖이지만, 저장 summary에 세부 제외사유가 없어 원인을 추정하지 않는다. 두 표의 E 반응은 learned self 분류나 external/self 식별 정답이 아니다.',
        ['모델', 'all136 D/E/U', 'paired21 D/E/U'], [[m, '/'.join(str(self_aggregate['same_slot_pipeline'][m][k]) for k in ['D','E','U']), '/'.join(str(self_aggregate['paired_same_query_RGB'][m][k]) for k in ['D','E','U'])] for m in MODELS])
    add('기존 reference≤3 461과 manual376은 별개',
        '461은 등록된 humanD 중 기존 general reference error≤3 association이다.376은 manual-click direct valid-query error≤3 표의 합(WOOD153+SQUARE223)이다. 동일 집합으로 가정하지 않으며 primary 필터·원좌표 오차 인증·좌표 보정으로 쓰지 않았다.',
        ['모델', 'reference461 D/E/U', 'manual376 D/E/U'], [[m, '/'.join(str(ref[m][k]) for k in ['D','E','U']), '/'.join(str(manual[m][k]) for k in ['D','E','U'])] for m in MODELS])
    add('publicheld360 weak owner: 별도9모델 task',
        '45frame의360슬롯 중 known center204(T24/nonT180=F96+BG84), truth UNKNOWN156이다. 원 inputU108/available252, crop 미구성24, center4 invalid112를 보존한다. v61은 자체 available251/U109이며 다른8 owner출력은 U108이다. 아래 known precision은 UNKNOWN truth를 제외하고, recall은 known truth의 predictionU를 포함한다. legacy는 reviewed lineage에 없어 보간·새 forward하지 않았다. public weak point/crop owner를 native D/E/S 또는 direct-end 정답으로 승격하지 않는다.',
        ['모델', 'known CM T/nonT→T/nonT/U', 'T P/R', 'nonT P/R', '전체 prediction U/360'], [[r['model'], cmtext(r['CM_T_NON_T_by_T_NON_T_U']), pct(r['T_precision_on_known_truth'])+' / '+pct(r['T_recall_including_prediction_U']), pct(r['NON_T_precision_on_known_truth'])+' / '+pct(r['NON_T_recall_including_prediction_U']), f"{r['prediction_U_all']}/360"] for r in public_rows])
    add('source fit와 합성 HOLDOUT은 실사와 분리',
        '두 팔 모두 TRAIN327(D218/E109)을 모두 맞혔고109 known D-E-D triplet을 모두 맞혔다. 이는 apparent fit이다. source HOLDOUT은6 synthetic bases [42,50,74,75,83,104]의45known(D30/E15),315 truthUNKNOWN이다. C58/C59 모두 D28/30, E는5/15와4/15, E precision5/7과4/6, paired all3-correct4/15와3/15다. real weak direct HOLDOUT known은0, native6·7의 direct known도0이다. TRAIN327은 synthetic246+publicreal weak81이며 strict synthetic-only recipe가 아니다. 이미 재사용된 source design이므로 fresh/blind가 아니다. TRAIN exact RAW327→312unique identity/no D-E conflict는 독립 frame 수가 아니다.168 incomplete attempt-native groups와868 complete UNKNOWN/othertruth groups는 추가 정답으로 세지 않는다.',
        ['모델', '역할', 'rows/known/UNKNOWN', 'CM [D,E,U]'], [[r['model'], r['role'], f"{r['rows']}/{r['known']}/{r['truth_UNKNOWN']}", cmtext(r['CM_D_E_by_D_E_U'])] for r in src_rows])
    add('비교 역할·실행 범위와 한계',
        'v61은 historical pixel-presence baseline으로 native semantic과 task가 다르다. legacy는 historical realCAL 및 반복 노출 이력이 있어 새 source-only head와 fresh transfer 대조군으로 해석할 수 없다. C52/C53의 D/E/U는 regional two-support proxy, C54..C59는 direct-end logit decoder이며 support2/owner3 보조출력과 구분한다. C58과 C59는 모두 RAW이고 각각 zero8/native onehot이다.116407 stored parameters는 같아도 유효 함수공간이 같다는 뜻은 아니며 shared backbone도 영향을 받을 수 있다. 원 q/L/native·초기 crop·고정 decoder를 유지했다. C54..C59는 finite direct logit≥0→E, <0→D이며 원 input unavailable/nonfinite는 U다. 새 모델/threshold/data/architecture 선택, 원 crop·source 배열·weights 읽기, 모델 forward/fit/PnP/SAM은 하지 않았다. 반복 평가된 인간 semantic authority를 유지하며 fresh/blind·개선·채택·목표 완료를 주장하지 않는다.')
    add('고정68 gallery는 별도 시각 근거',
        'root의 고정6개 실제 card 관찰 기록은 CASE0840(humanD, 기존 reference error 2.7108px)에서 C58/C59가 E, CASE0996(humanE,1.13822px)에서 두 팔이 D였음을 기록한다. 이는 이번 fixed case의 실패 예시이며 원인이나 전체 실패빈도를 확정하지 않는다. 인접 재료·외부물체 단서와 native 목표를 구분하는 문제가 남아 있다. 이 numeric 작성자는 원 crop/card를 새로 열지 않았고, 아래 별도 gallery와 root 관찰 영수증을 연결한다.68개 전체를 실제 도구로 보았다는 주장이 아니다.')
    artifact_links = [(f, f) for f in ['PRIMARY_ALL10.csv', 'SAME_SLOT_ALL10.csv', 'PRIMARY_MATERIAL_ALL10.csv', 'SAVED_PRIMARY_NATIVE8_D_E_WITH_U.csv', 'SELF_TWO_COHORTS_ALL10.csv', 'REFERENCE461_AND_MANUAL376_ALL10.csv', 'PUBLIC_WEAK_BINARY9.csv', 'PUBLIC_WEAK_OWNER3_8.csv', 'PUBLIC_WEAK_NATIVE9.csv', 'SOURCE_TRAIN_AND_HOLDOUT_C58_C59.csv', 'RESULTS_KO.json', 'PRIMARY_ALL10_COUNTS.svg']]
    provenance_links = [('독립 실제 수치감사 PASS', link(scope['audit_review']['path'])), ('저장 current summary', link(current_b['path'])), ('저장 public weak summary', link(public_b['path'])), ('저장 평가 COMPLETE', link(scope['saved_evaluation_complete']['path'])), ('V199 source 보고서 JSON', link(scope['source_report']['path']))]
    provenance_links += [('고정68 portable gallery', '../fixed_native_real_visual_v1/PORTABLE_GALLERY_KO.md'), ('고정68 gallery HTML', '../fixed_native_real_visual_v1/index.html'), ('root 고정6 실제 card 관찰 기록', '../ROOT_ACTUAL_FIXED6_CARD_VISUAL_QA.json')]
    md = '# V200 저장 수치 readout\n\n![10모델 고정 주평가 count chart](PRIMARY_ALL10_COUNTS.png)\n\n'
    for title, text, headers, rows in sections:
        md += '## ' + title + '\n\n' + text + '\n\n'
        if headers:
            md += table(headers, rows) + '\n'
    md += '## 수치표·근거\n\n' + ''.join(f'- [{label}]({url})\n' for label, url in artifact_links + provenance_links)
    (OUT / 'REPORT_KO.md').write_text(md)
    body = '<h1>V200 저장 수치 readout</h1><img class="chart" src="PRIMARY_ALL10_COUNTS.png" alt="10모델 고정 주평가 count chart">'
    for title, text, headers, rows in sections:
        body += '<section><h2>' + html.escape(title) + '</h2><p>' + html.escape(text) + '</p>'
        if headers:
            body += html_table(headers, rows)
        body += '</section>'
    body += '<h2>수치표·근거</h2><ul>' + ''.join('<li><a href="' + html.escape(url, quote=True) + '">' + html.escape(label) + '</a></li>' for label, url in artifact_links + provenance_links) + '</ul>'
    (OUT / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V200 저장 수치 readout</title><style>body{font:16px/1.65 sans-serif;max-width:1450px;margin:32px auto;padding:0 20px;color:#172333}h1{font-size:30px}h2{font-size:23px;margin-top:38px}p{max-width:1150px}.chart{width:100%;height:auto}.scroll{overflow:auto}table{border-collapse:collapse;min-width:720px;margin:18px 0}td,th{border:1px solid #cbd3db;padding:8px 11px;text-align:left;white-space:nowrap}th{background:#edf2f7}tr:nth-child(even){background:#f7f9fb}a{color:#175c99}</style><body>' + body + '</body></html>')
    relative_refs = []
    for page in ['REPORT_KO.md', 'index.html']:
        txt = (OUT / page).read_text()
        refs = re.findall(r'\]\(([^)]+)\)', txt) if page.endswith('.md') else re.findall(r'(?:href|src)="([^"]+)"', txt)
        for ref in refs:
            assert not ref.startswith(('/', 'http:', 'https:', 'file:'))
            target = safe(OUT / ref)
            check(f'link {page} {ref}', target.is_file())
            relative_refs.append(dict(page=page, reference=ref))
    write_json('SCALAR_AND_LINK_QA.json', dict(completed=True, PASS=True, saved_summary_arithmetic_only=True, check_count=len(checks), checks=checks, relative_references=relative_refs, no_original_ledger_or_arrays_or_model_reads=True, figure_tool_view_pending=True))
    outputs = {p.name: binding(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json('BUILD_COMPLETE.json', dict(completed=True, PASS=True, code=binding(Path(__file__)), scope=binding(SCOPE), inputs=inputs, outputs=outputs, checks=len(checks), original_arrays_weights_models_forwards_fits=0, figure_tool_view_pending=True), PREP)
    print(json.dumps(dict(completed=True, PASS=True, files=len(outputs), scalar_link_checks=len(checks), report=binding(OUT/'REPORT_KO.md'), results=binding(OUT/'RESULTS_KO.json'), chart=binding(OUT/'PRIMARY_ALL10_COUNTS.png'), build_complete=binding(PREP/'BUILD_COMPLETE.json')), ensure_ascii=False))


if __name__ == '__main__':
    run()
