"""V256 saved SOURCE readout. CSV arithmetic only; no model or feature reads."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import stat

CONTROL = 'CONTROL_ORIGINAL_FIT'
TREAT = 'TREAT_ORIGINAL_PLUS_NATIVE_CHILDREN'
ARMS = (CONTROL, TREAT)
ROLES = ('BEFORE', 'COVER', 'NEAR_SHAM', 'FAR_SHAM')
LABELS = ('D', 'E', 'U')


def need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def safe(path, exists=True):
    p = Path(path)
    need(p.is_absolute() and '..' not in p.parts, 'Exact absolute leaf required')
    need(not any(x in str(p).casefold() for x in
                 ('v112', 'cache', '.pyc', 'current223', 'publicheld', 'holdout', 'oldval')),
         'Forbidden path before FS')
    for a in [*reversed(p.parents), p]:
        try:
            mode = os.lstat(a).st_mode
        except FileNotFoundError:
            need(not exists, 'Exact leaf absent')
            continue
        need(not stat.S_ISLNK(mode), 'No symlink artifact')
    return p


def raw(binding):
    p = safe(binding['path'])
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as f:
        data = f.read()
    need(len(data) == binding['size_bytes'] and hashlib.sha256(data).hexdigest() == binding['sha256'],
         'Exact SHA/size: ' + p.name)
    return data


def ident(p):
    p = safe(p)
    data = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'size_bytes': len(data)}


def dump(p, value):
    p = safe(p, False)
    with p.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return ident(p)


def boolean(s):
    need(s in ('True', 'False'), 'Explicit CSV bool')
    return s == 'True'


def csv_rows(data):
    out = []
    for i, r in enumerate(csv.DictReader(io.StringIO(data.decode('utf-8')))):
        for k in ('sample_index', 'family', 'target'):
            r[k] = int(r[k])
        r['available'] = boolean(r['available'])
        r['original_native_address'] = int(r['original_native_address']) if r['original_native_address'] else None
        for k in ('logit0', 'logit1', 'selected_family_E_logit'):
            r[k] = float(r[k])
            need(math.isfinite(r[k]), 'Finite saved logit')
        need(r['sample_index'] == i and r['target'] in (0, 1) and r['family'] in (0, 1), 'Saved row order/known labels')
        need(r['role'] in ('FIT', 'DEV') and r['prediction'] in LABELS, 'Saved role/prediction')
        need(r['selected_family_E_logit'] == r['logit' + str(r['family'])], 'Family-selected scalar')
        expected = ('E' if r['selected_family_E_logit'] >= 0 else 'D') if r['available'] else 'U'
        need(r['prediction'] == expected, 'Fixed raw-logit0 sign/support')
        out.append(r)
    need(len({r['row_id'] for r in out}) == len(out), 'Unique saved row IDs')
    return out


def metric(rows):
    cm = [[sum(r['target'] == y and r['prediction'] == p for r in rows) for p in LABELS] for y in (0, 1)]
    v = {'rows': len(rows), 'CM_D_E_by_D_E_U': cm,
         'original_input_U': sum(not r['available'] for r in rows),
         'model_U': cm[0][2] + cm[1][2],
         'U_rate': (cm[0][2] + cm[1][2])/len(rows) if rows else None}
    for c, j in (('D', 0), ('E', 1)):
        truth, predicted, tp = sum(cm[j]), cm[0][j] + cm[1][j], cm[j][j]
        v[c] = {'truth': truth, 'predicted': predicted, 'precision': tp/predicted if predicted else None,
                'recall': tp/truth if truth else None}
    recalls = [v[c]['recall'] for c in ('D', 'E')]
    v['balanced_recall'] = sum(recalls)/2 if all(x is not None for x in recalls) else None
    return v


def reductions(rows):
    return {f'family{f}_{role}': metric([r for r in rows if r['family'] == f and r['role'] == role])
            for f in (0, 1) for role in ('FIT', 'DEV')}


def original_reduction(by_arm):
    a, b = [by_arm[x] for x in ARMS]
    keys = ('sample_index', 'row_id', 'family', 'role', 'target', 'available', 'original_native_address')
    need(len(a) == len(b) and all(all(x[k] == y[k] for k in keys) for x, y in zip(a, b)), 'Original same denominator')
    metrics = {arm: reductions(rows) for arm, rows in by_arm.items()}
    paired = {}
    for f in (0, 1):
        for role in ('FIT', 'DEV'):
            rr = [(x, y) for x, y in zip(a, b) if x['family'] == f and x['role'] == role]
            recovered = sum(x['prediction'] != LABELS[x['target']] and y['prediction'] == LABELS[x['target']] for x, y in rr)
            damaged = sum(x['prediction'] == LABELS[x['target']] and y['prediction'] != LABELS[x['target']] for x, y in rr)
            paired[f'family{f}_{role}'] = {'rows': len(rr), 'recovered': recovered, 'damaged': damaged, 'net': recovered-damaged}
    four = {arm: [metrics[arm][f'family{f}_DEV'][c]['recall'] for f in (0, 1) for c in ('D', 'E')] for arm in ARMS}
    need(all(x is not None for v in four.values() for x in v), 'All original DEV classes present')
    mean = {arm: sum(v)/4 for arm, v in four.items()}
    drops = [x-y for x, y in zip(four[CONTROL], four[TREAT])]
    native = {arm: {str(k): metric([r for r in rows if r['family'] == 0 and r['role'] == 'DEV' and
                                   r['original_native_address'] == k]) for k in range(8)} for arm, rows in by_arm.items()}
    return {'family_metrics_original': metrics, 'paired_original_changes': paired,
            'DEV_recalls_D_E_by_SYN_YCB': four, 'DEV_mean_balanced_recall': mean,
            'mean_balanced_recall_gain_pp': 100*(mean[TREAT]-mean[CONTROL]),
            'worst_DEV_recall_drop': max(drops), 'original_native_DEV': native}


def assert_producer_summary(result, producer):
    for arm in ARMS:
        for key, m in result['family_metrics_original'][arm].items():
            pm = producer['family_metrics_original'][arm][key]
            for k in ('rows', 'CM_D_E_by_D_E_U', 'original_input_U', 'D', 'E'):
                need(m[k] == pm[k], 'Producer arithmetic mismatch ' + arm + '/' + key + '/' + k)
    for k in ('paired_original_changes', 'DEV_recalls_D_E_by_SYN_YCB', 'DEV_mean_balanced_recall', 'worst_DEV_recall_drop'):
        need(result[k] == producer[k], 'Producer saved reduction mismatch: ' + k)


def cf_reduction(full, child):
    known = {r['sample_index']: r for r in full if r['sample_index'] is not None}
    need(len(known) == len(child[CONTROL]) == len(child[TREAT]), 'New full-map known join')
    for arm in ARMS:
        for i, r in enumerate(child[arm]):
            m = known[i]
            need(r['row_id'] == m['row_id'] and r['target'] == m['target'] and r['family'] == 0 and r['role'] == 'FIT',
                 'New known FIT row identity')
    before, children = {}, {}
    for m in full:
        need(m['source_role'] == 'FIT' and m['fit'] is True and m['family_code'] == 0, 'New SOURCE FIT-only')
        j = int(m['active_slot'])
        need(j == int(m['native_slot']) and 0 <= j < 8, 'Original native address')
        if m['variant'] == 'BEFORE':
            key = (int(m['base_index']), j)
            need(key not in before, 'Unique BEFORE base/native')
            before[key] = m
        elif j == m['intervention_target_native_slot']:
            key = (int(m['base_index']), j)
            roles = children.setdefault(key, {})
            need(m['variant'] in ROLES[1:] and m['variant'] not in roles, 'Unique child intervention k=query j')
            roles[m['variant']] = m
    need(set(children) <= set(before), 'Every constructed trial has its own BEFORE')
    detail = []
    for key, base in sorted(before.items()):
        roles = dict(children.get(key, {}))
        need(not roles or set(roles) == set(ROLES[1:]), 'Keep complete construction or absent construction')
        constructed = bool(roles)
        roles['BEFORE'] = base
        for r in roles.values():
            need(r['original_query_xy'] == base['original_query_xy'] and r['original_L'] == base['original_L'] and
                 r['original_query_available'] == base['original_query_available'], 'Same active original q/L/U across scenes')
        allknown = constructed and all(roles[v]['sample_index'] is not None and roles[v]['target'] in (0, 1) for v in ROLES)
        item = {'base_index': key[0], 'native_k': key[1], 'target_alone_state': base['point0_target_alone_state'],
                'constructed': constructed, 'all_four_known': allknown,
                'fresh_point0_states': [roles[v]['point0_combined_state'] for v in ROLES] if constructed else None,
                'actual_targets': [roles[v]['target'] for v in ROLES] if constructed else None,
                'sample_indices': [roles[v]['sample_index'] for v in ROLES] if constructed else None,
                'row_ids': [roles[v]['row_id'] for v in ROLES] if constructed else None}
        item['fresh_point0_D_E_D_D'] = constructed and item['fresh_point0_states'] == ['D', 'E', 'D', 'D']
        item['known_targets_D_E_D_D'] = allknown and item['actual_targets'] == [0, 1, 0, 0]
        for arm in ARMS:
            rr = [child[arm][roles[v]['sample_index']] for v in ROLES] if allknown else None
            item[arm] = {'predictions': [r['prediction'] for r in rr] if rr else None,
                         'model_D_E_D_D': bool(rr) and [r['prediction'] for r in rr] == ['D', 'E', 'D', 'D'],
                         'all_actual_targets_correct': bool(rr) and all(r['prediction'] == LABELS[r['target']] for r in rr),
                         'cover_minus_before_positive': rr[1]['selected_family_E_logit'] > rr[0]['selected_family_E_logit'] if rr else None,
                         'cover_minus_near_positive': rr[1]['selected_family_E_logit'] > rr[2]['selected_family_E_logit'] if rr else None,
                         'cover_minus_far_positive': rr[1]['selected_family_E_logit'] > rr[3]['selected_family_E_logit'] if rr else None}
        detail.append(item)
    tables = []
    for native in [*range(8), 'ALL']:
        rr = detail if native == 'ALL' else [r for r in detail if r['native_k'] == native]
        census = {'native_k': native, 'candidates': len(rr),
                  'target_alone_D': sum(r['target_alone_state'] == 'D' for r in rr),
                  'target_alone_S': sum(r['target_alone_state'] == 'S' for r in rr),
                  'target_alone_other': sum(r['target_alone_state'] not in ('D', 'S') for r in rr),
                  'constructed': sum(r['constructed'] for r in rr),
                  'all_four_known': sum(r['all_four_known'] for r in rr),
                  'fresh_point0_D_E_D_D': sum(r['fresh_point0_D_E_D_D'] for r in rr),
                  'known_targets_D_E_D_D': sum(r['known_targets_D_E_D_D'] for r in rr)}
        for arm in ARMS:
            table = dict(census, arm=arm)
            for k in ('model_D_E_D_D', 'all_actual_targets_correct', 'cover_minus_before_positive',
                      'cover_minus_near_positive', 'cover_minus_far_positive'):
                table[k] = sum(r[arm][k] is True for r in rr)
            tables.append(table)
    return tables, detail


def csv_write(path, rows):
    need(bool(rows), 'Nonempty table')
    with safe(path, False).open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return ident(path)


def metric_table(summary):
    rows = []
    for arm in ARMS:
        for f in (0, 1):
            for role in ('FIT', 'DEV'):
                m = summary['family_metrics_original'][arm][f'family{f}_{role}']
                for c, y in (('D', 0), ('E', 1)):
                    rows.append({'arm': arm, 'family': f, 'role': role, 'class': c, 'truth': m[c]['truth'],
                                 'pred_D': m['CM_D_E_by_D_E_U'][y][0], 'pred_E': m['CM_D_E_by_D_E_U'][y][1],
                                 'pred_U': m['CM_D_E_by_D_E_U'][y][2], 'precision': m[c]['precision'],
                                 'recall': m[c]['recall'], 'U_rate': m['U_rate'], 'balanced_recall': m['balanced_recall']})
    return rows


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows])


def rel_link(out, b):
    return os.path.relpath(b['path'], out)


def reports(out, summary, cfg, complete):
    mets = summary['family_metrics_original']
    tab = []
    for f, family in ((0, 'SYN 등록 witness'), (1, 'YCB associated-r AUX')):
        for c in ('D', 'E'):
            a, b = [mets[arm][f'family{f}_DEV'][c] for arm in ARMS]
            tab.append([family, c, a['truth'], f"{mets[CONTROL][f'family{f}_DEV']['CM_D_E_by_D_E_U'][0 if c=='D' else 1][0 if c=='D' else 1]}/{a['truth']}",
                        f"{mets[TREAT][f'family{f}_DEV']['CM_D_E_by_D_E_U'][0 if c=='D' else 1][0 if c=='D' else 1]}/{b['truth']}",
                        pct(a['recall']), pct(b['recall']), pct(a['precision']), pct(b['precision'])])
    cf = summary['selected_k_CF_tables']
    native_tab = []
    for k in range(8):
        a, b = [next(r for r in cf if r['arm'] == arm and r['native_k'] == k) for arm in ARMS]
        native_tab.append([k, a['candidates'], a['target_alone_S'], a['constructed'], a['all_four_known'],
                           a['model_D_E_D_D'], b['model_D_E_D_D']])
    all_cf = next(r for r in cf if r['arm'] == CONTROL and r['native_k'] == 'ALL')
    admission = summary['fit_admission']
    syn0, syn1 = [mets[arm]['family0_DEV'] for arm in ARMS]
    y0, y1 = [mets[arm]['family1_DEV'] for arm in ARMS]
    links = []
    for arm in ARMS:
        links += [(arm+' 원 SOURCE CSV', 'artifacts/V256_ROOT_NATIVE_DATA_FIT/matched_existing_crop_data_pair_v1/'+Path(complete['outputs'][arm]['path']).name),
                  (arm+' 새 FIT CSV', 'artifacts/V256_ROOT_NATIVE_DATA_FIT/matched_existing_crop_data_pair_v1/'+Path(complete['child_fit_outputs'][arm]['path']).name)]
    links += [('학습 COMPLETE', 'artifacts/V256_ROOT_NATIVE_DATA_FIT/matched_existing_crop_data_pair_v1/COMPLETE.json'), ('독립 결과 QA', 'artifacts/V256_ACTUAL_RESULT_QA_B/actual_review_v1/REVIEW.json'),
              ('최종 barrier', 'artifacts/V256_ROOT_NATIVE_DATA_FIT/matched_existing_crop_data_pair_v1/'+Path(complete['freeze']['path']).name),
              ('고정 사전 recipe', 'artifacts/V256_ROOT_NATIVE_DATA_FIT/PROSPECTIVE_DATA_ONLY_PAIR_POLICY.json')]
    if cfg.get('historical_gallery_relative_link'):
        links.append(('V255 고정 16-case / 64-scene 자료 갤러리', cfg['historical_gallery_relative_link']))
    text = f'''# V256: 같은 head에서 새 coherent SYN 자료를 추가한 SOURCE 비교

같은 76098-parameter head를 같은 seed17 초기값에서 각각 1000회 학습했습니다. CONTROL은 원 SOURCE FIT, TREAT는 여기에 새 own-scene non-BEFORE known FIT {admission['new_nonBEFORE_all_admitted_children']:,}행을 추가했습니다. encoder·구조·문턱을 바꾸지 않았습니다. 원 YCB 학습 자료와 1000회 draw는 두 팔에서 동일하며, 공유 head 학습 결과의 YCB logit까지 같다는 뜻은 아닙니다.

원 SYN DEV E recall은 {pct(syn0['E']['recall'])}에서 {pct(syn1['E']['recall'])}로 올랐고, D recall은 {pct(syn0['D']['recall'])}에서 {pct(syn1['D']['recall'])}로 낮아졌습니다. E precision도 {pct(syn0['E']['precision'])}에서 {pct(syn1['E']['precision'])}로 낮아졌습니다. YCB DEV E는 두 팔 모두 {y0['CM_D_E_by_D_E_U'][1][1]}/{y0['E']['truth']}입니다. 부분적인 SYN 클래스 변화이며 공개 실사 전이 문제나 최종 pallet visibility가 해결됐다는 증거는 없습니다. 모델 미채택, current/held 새 실행 0입니다.

## 원 SOURCE DEV: 같은 분모

CM 행은 실제 D/E, 열은 예측 D/E/U입니다. D0/E1, 선택한 가족 E-logit ≥ 0이면 E이며 문턱 탐색은 없습니다.

{markdown_table(['교사 namespace','클래스','원 분모','CONTROL 정답/분모','TREAT 정답/분모','CONTROL recall','TREAT recall','CONTROL precision','TREAT precision'], tab)}

SYN CM은 CONTROL `{syn0['CM_D_E_by_D_E_U']}`, TREAT `{syn1['CM_D_E_by_D_E_U']}`입니다. YCB CM은 CONTROL `{y0['CM_D_E_by_D_E_U']}`, TREAT `{y1['CM_D_E_by_D_E_U']}`입니다. 네 클래스 recall의 산술평균은 {pct(summary['DEV_mean_balanced_recall'][CONTROL])} → {pct(summary['DEV_mean_balanced_recall'][TREAT])}, 차이는 {summary['mean_balanced_recall_gain_pp']:+.6f} pp이며 최악의 클래스 recall 감소는 {100*summary['worst_DEV_recall_drop']:.6f} pp입니다. 두 namespace를 하나의 물리적 과업 정확도로 합친 수치가 아닙니다.

원 known SOURCE {summary['original_known_rows']:,}행에서 모델 U는 {summary['original_model_U_by_arm'][CONTROL]}/{summary['original_known_rows']}과 {summary['original_model_U_by_arm'][TREAT]}/{summary['original_known_rows']}입니다. 원 full15040의 S/U 등 masked7423은 known7617에서의 모델 abstention 분모에 들어가지 않습니다. 원 DEV 1741행은 반복 사용된 exposed SOURCE DEV이며 fresh blind 검증이 아닙니다. FIT 지표는 [METRICS.csv](tables/METRICS.csv), 개별 회복·악화는 [PAIRED_CHANGES.csv](tables/PAIRED_CHANGES.csv)에 별도 기록했습니다.

![원 DEV 클래스별 recall](figures/DEV_RECALL.png)

## 새 SOURCE FIT counterfactual: 별도 분모

후보 {all_cf['candidates']}개에서 target-alone S {all_cf['target_alone_S']}개를 보존했고 {all_cf['constructed']}개를 구성했습니다. 이 중 네 장 모두 원 input-support/locality 조건으로 known인 것은 {all_cf['all_four_known']}개입니다. 아래 모델 D/E/D/D 패턴은 이 all-four-known 분모에서만 셉니다. 변형 이름으로 정답을 추론하지 않고 각 장의 새 point0 teacher/known target을 직접 확인했습니다. BEFORE는 `(base, native k)`로 결합하고 child의 intervention k와 query j가 같은 경우만 봅니다.

{markdown_table(['등록 주소 k','후보','target-alone S','구성','4장 known','CONTROL D/E/D/D','TREAT D/E/D/D'], native_tab)}

이 표는 학습용 FIT 분할 소속 자료에 대한 저장 출력 진단입니다. TREAT의 loss 후보로는 모든 known non-BEFORE {admission['new_nonBEFORE_all_admitted_children']:,}행을 넣었고, 개별행 실제 draw 노출은 별도 ledger/QA에 기록됐습니다. 모든 장·모든 주소가 실제 loss에 뽑혔다고 이 표만으로 주장하지 않습니다. 새 BEFORE known {admission['new_BEFORE_excluded_from_augmentation']}행은 중복 augmentation loss에서 제외했습니다. 새 full-map {admission['new_map_census']['all_new_scene_node_rows']:,}행, known {admission['new_map_census']['known_rows']:,}행, masked {admission['new_map_census']['masked_rows_retained_in_full_map']:,}행과 원 U를 보존했습니다. 원 q는 저장된 SYN initializer 좌표이며 fresh v61 추론이나 실사 keypoint annotation이 아닙니다. 좌표 보정도 하지 않았습니다. V254의 strict FIT87과는 후보·교사·known 조건이 달라 직접적인 성공률 증가로 비교하지 않습니다.

![새 FIT selected-k 패턴](figures/NEW_FIT_CF_PATTERN.png)

## 해석과 재현 범위

SYN 가족0은 등록된 whole-target의 intended native-address first witness point0 교사입니다. physical native corner·전체 영역·literal query pixel gold로 격상하지 않습니다. YCB 가족1은 associated-r AUX이며 native/SELF 의미는 UNKNOWN입니다. 모든 native 주소가 후보에 포함됐어도 새 실제 known 수와 교사 기하 분모는 주소별로 다릅니다. 원 q/L/U와 input 조건은 보존됐고 teacher/reference/variant 정보는 모델 forward 입력이 아닙니다.

두 final1000 checkpoint/log/ledger를 봉인한 뒤 DEV 출력을 계산한 sealed-code 순서와 저장 barrier를 독립 QA가 확인했습니다. 파일 시각만으로 실행 chronology를 독립 증명했다는 뜻은 아닙니다. 이 보고서는 승인된 저장 CSV/JSON 산술을 재계산하며 checkpoint·feature·encoder·RGB를 읽거나 forward를 재실행하지 않습니다. 새 SOURCE training/독립 QA 완료는 런타임·저장자료 검증이고 모델 채택이나 current/v61 비교 결과가 아닙니다. 사전 recipe는 자동 deployment gate를 정의하지 않았으며 네 recall/평균/최악 감소를 그대로 보고합니다. 군집 불확실성은 원 group 메타데이터가 이 readout의 최소 입력에 없어 산출하지 않았습니다.

이 보고서 CODE는 승인된 4개 CSV와 새 full-map/summary에 대해 직접 CM·P/R·paired 변화·selected-k 패턴을 산술 재계산하고 생산자 SUMMARY 및 독립 QA와 대조했습니다. [상세 수치](tables/SUMMARY.json), [selected-k 전체 join](tables/SELECTED_K_PAIRS.jsonl), [주소별 DEV](tables/NATIVE_DEV.csv)와 입력/출력 바인딩을 함께 보존합니다. 게시 bundle에는 별도의 독립 QA/replay 경로를 포함합니다.

''' + '\n'.join(f'- [{title}]({link})' for title, link in links) + '\n'
    (out/'PORTABLE_REPORT_KO.md').write_text(text, encoding='utf-8')
    htables = '<h2>원 SOURCE DEV</h2>' + html_table(['교사','클래스','분모','CONTROL 정답/분모','TREAT 정답/분모','C recall','T recall','C precision','T precision'], tab)
    htables += '<h2>새 SOURCE FIT selected-k</h2>' + html_table(['k','후보','S','구성','4 known','C D/E/D/D','T D/E/D/D'], native_tab)
    hbody = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V256 SOURCE data-only 비교</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border:1px solid #ccc;padding:7px}}th{{background:#eef2f6}}img{{max-width:100%;height:auto}}.note{{border-left:4px solid #aa3b28;padding:12px;background:#fff4ef}}a{{color:#174e9b}}</style>
<h1>V256: 같은 head, coherent SYN 자료 추가</h1>
<p class="note">부분적인 SYN 변화입니다. YCB DEV E 두 팔 모두 0/57. SOURCE DEV는 반복 노출됐으며 fresh blind 검증이 아닙니다. 모델 미채택/current·held 새 실행 0. 현재 pallet 실사 개선 결과가 아닙니다.</p>
<p>각 76098 parameters, 같은 seed17 초기값, 각1000 updates. 원 YCB 자료·draw 동일; 새 non-BEFORE known FIT {admission['new_nonBEFORE_all_admitted_children']:,}행 추가. 원 threshold0·q/L/U 보존.</p>
{htables}<p>CM/P/R/U·FIT 별도: <a href="tables/METRICS.csv">METRICS.csv</a>. D/E/D/D는 실제 fresh targets와 all-four-known {all_cf['all_four_known']}개에서 확인. FITmembership과 실제 draw 노출 구분, V254 strict87과 직접 % 비교 금지.</p>
<img src="figures/DEV_RECALL.png" alt="원 DEV D/E recall"><img src="figures/NEW_FIT_CF_PATTERN.png" alt="새 FIT selected k pattern">
<p>SYN intended registered witness와 YCB associated-r AUX는 별도 namespace입니다. physical native/literal q/SELF gold로 주장하지 않습니다. 네 클래스 평균 BR {pct(summary['DEV_mean_balanced_recall'][CONTROL])} → {pct(summary['DEV_mean_balanced_recall'][TREAT])} ({summary['mean_balanced_recall_gain_pp']:+.6f} pp), 최악 recall 감소 {100*summary['worst_DEV_recall_drop']:.6f} pp. 자동 deployment gate/채택 없음.</p>
<p><a href="README_KO.md">상세 보고서·한계</a> · <a href="tables/SUMMARY.json">산술 요약</a> · <a href="artifacts/V256_SOURCE_PAIR_READOUT_C/results_v1/COMPLETE.json">입력/출력 바인딩</a> · <a href="tables/SELECTED_K_PAIRS.jsonl">selected-k full join</a></p>
</html>'''
    (out/'ROOT_READOUT_KO.html').write_text(hbody+'\n', encoding='utf-8')


def html_table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def plots(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir()
    colors = ('#577b9c', '#d77932')
    four = summary['DEV_recalls_D_E_by_SYN_YCB']
    fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
    for i, arm in enumerate(ARMS):
        xs = [x+(i-.5)*.34 for x in range(4)]
        bars = ax.bar(xs, [100*x for x in four[arm]], width=.34, color=colors[i], label=('CONTROL original FIT' if i == 0 else 'TREAT + native children'))
        for bar, value in zip(bars, four[arm]):
            ax.text(bar.get_x()+bar.get_width()/2, 100*value+1, f'{100*value:.2f}', ha='center', fontsize=9)
    denominators = [summary['family_metrics_original'][CONTROL][f'family{f}_DEV'][c]['truth'] for f in (0, 1) for c in ('D', 'E')]
    ax.set(xticks=range(4), xticklabels=[f'{name} (n={n})' for name, n in zip(('SYN D', 'SYN E', 'YCB D', 'YCB E'), denominators)],
           ylabel='Recall (%) at fixed logit 0', ylim=(0, 112), title='Original SOURCE DEV — same denominator, exposed not fresh blind')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(out/'figures/DEV_RECALL.png', dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for i, arm in enumerate(ARMS):
        rr = [next(r for r in summary['selected_k_CF_tables'] if r['arm'] == arm and r['native_k'] == k) for k in range(8)]
        values = [100*r['model_D_E_D_D']/r['all_four_known'] if r['all_four_known'] else 0 for r in rr]
        bars = ax.bar([k+(i-.5)*.34 for k in range(8)], values, .34, color=colors[i], label=('CONTROL' if i == 0 else 'TREAT'))
        for bar, r in zip(bars, rr):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, f"{r['model_D_E_D_D']}/{r['all_four_known']}", ha='center', fontsize=8, rotation=90)
    ax.set(xticks=range(8), xlabel='Registered native address k (not physical native gold)', ylabel='Model D/E/D/D / four-known (%)',
           ylim=(0, 120), title='New SOURCE FIT selected k — not held-out performance')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, frameon=False)
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.savefig(out/'figures/NEW_FIT_CF_PATTERN.png', dpi=170); plt.close(fig)


def run(entry):
    cfg = json.loads(safe(entry).read_bytes())
    need(cfg.get('approved_actual_saved_SOURCE_readout') is True, 'Root actual report authorization')
    need(cfg['code'] == ident(Path(__file__).absolute()), 'Pinned report code')
    scope = json.loads(raw(cfg['read_scope']))
    need(scope.get('approved_actual_saved_SOURCE_readout') is True and scope['entry_contract'] ==
         {k: v for k, v in cfg.items() if k != 'read_scope'}, 'Exact root report scope')
    fit = json.loads(raw(cfg['fit_complete']))
    need(fit['completed'] is True and fit['PASS'] is True and fit['selected_arm'] is None and
         fit['current_or_held_admission'] is False, 'Completed unadopted SOURCE pair')
    freeze = json.loads(raw(fit['freeze']))
    need(freeze['completed'] is True and freeze['PASS'] is True and freeze['DEV_prediction_calls_before_both_finals'] == 0,
         'Both-final barrier before original DEV predictions')
    need(set(freeze['finals']) == set(ARMS) and all(freeze['new_optimizer_updates'][a] == 1000 for a in ARMS), 'Both1000')
    for arm in ARMS:
        need(freeze['finals'][arm]['checkpoint'] == fit['checkpoints'][arm] and
             freeze['finals'][arm]['log'] == fit['logs'][arm], 'Saved final identity, no checkpoint read')
    qa = json.loads(raw(cfg['result_QA']))
    need(qa['completed'] is True and qa['PASS'] is True and qa['training_complete'] == cfg['fit_complete'], 'Independent result QA exact pair')
    policy = json.loads(raw(cfg['policy']))
    need(fit['policy'] == cfg['policy'] and policy['optimizer']['updates'] == 1000, 'Fixed recipe')
    producer = json.loads(raw(fit['summary']))
    admission = json.loads(raw(fit['fit_admission']))
    rows = {arm: csv_rows(raw(fit['outputs'][arm])) for arm in ARMS}
    child = {arm: csv_rows(raw(fit['child_fit_outputs'][arm])) for arm in ARMS}
    full_raw = raw(fit['inputs']['new_sample_map'])
    full = [json.loads(line) for line in full_raw.splitlines() if line]
    coordinate_census = json.loads(raw(cfg['coordinate_census']))
    need(coordinate_census['completed'] is True and coordinate_census['PASS'] is True and
         coordinate_census['feature_complete'] == fit['new_feature_complete'] and
         coordinate_census['sample_map_binding'] == fit['inputs']['new_sample_map'], 'SOURCE coordinate census exact packet')
    result = original_reduction(rows)
    assert_producer_summary(result, producer)
    for k in ('family_metrics_original', 'paired_original_changes', 'DEV_recalls_D_E_by_SYN_YCB'):
        if k == 'family_metrics_original':
            for arm in ARMS:
                for key, item in result[k][arm].items():
                    need(all(item[x] == qa[k][arm][key][x] for x in ('rows', 'CM_D_E_by_D_E_U', 'D', 'E')), 'QA CM/P/R match')
        else:
            need(result[k] == qa[k], 'Independent result QA arithmetic agreement')
    cf, detail = cf_reduction(full, child)
    census = {'all_new_scene_node_rows': len(full), 'known_rows': len(child[CONTROL]),
              'masked_rows_retained_in_full_map': sum(x['sample_index'] is None for x in full),
              'variant_all_rows': dict(Counter(x['variant'] for x in full)),
              'variant_known_rows': dict(Counter(x['variant'] for x in full if x['sample_index'] is not None))}
    need(census == admission['new_map_census'], 'Full new known/masked census')
    need(admission['YCB_pools_and_all_draws_byteexact'] is True, 'Unchanged YCB draws confirmed by separate QA')
    result.update({'completed': True, 'PASS': True, 'fit_complete': cfg['fit_complete'], 'result_QA': cfg['result_QA'],
                   'fit_admission': admission, 'selected_k_CF_tables': cf, 'new_trial_candidates': len(detail),
                   'original_known_rows': len(rows[CONTROL]), 'original_model_U_by_arm': {a: sum(r['prediction']=='U' for r in rows[a]) for a in ARMS},
                   'new_full_census': census, 'threshold_E_logit_ge': 0, 'current_admission': False,
                   'coordinate_census': cfg['coordinate_census'],
                   'model_adopted': False, 'selected_arm': None, 'source_DEV_reused_exposed_not_fresh_blind': True,
                   'gate_policy': 'No automatic deployment gate in the V256 prospective recipe; all four recalls, meanBR and worst drop reported.',
                   'teacher_namespaces': {'0': 'SYN intended registered whole-target first witness point0', '1': 'YCB associated-r AUX/nativeUNKNOWN'},
                   'no_new_model_encoder_feature_array_RGB_or_current_payload_reads': True})
    out = safe(cfg['output_directory'], False)
    need(not out.exists(), 'Fresh report output; no automatic retry')
    out.mkdir()
    (out/'tables').mkdir()
    table = out/'tables'
    try:
        dump(table/'SUMMARY.json', result)
        csv_write(table/'METRICS.csv', metric_table(result))
        csv_write(table/'PAIRED_CHANGES.csv', [dict(group=k, **v) for k, v in result['paired_original_changes'].items()])
        csv_write(table/'SELECTED_K_CF.csv', cf)
        with (table/'SELECTED_K_PAIRS.jsonl').open('x', encoding='utf-8') as f:
            for d in detail:
                f.write(json.dumps(d, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        csv_write(table/'NATIVE_DEV.csv', [dict(arm=arm, native=k, **{key: val for key, val in m.items() if key not in ('D', 'E', 'CM_D_E_by_D_E_U')},
                  D_recall=m['D']['recall'], E_recall=m['E']['recall'], CM=json.dumps(m['CM_D_E_by_D_E_U']))
                  for arm in ARMS for k, m in result['original_native_DEV'][arm].items()])
        plots(out, result)
        reports(out, result, cfg, fit)
        paths = [table/x for x in ('SUMMARY.json', 'METRICS.csv', 'PAIRED_CHANGES.csv', 'SELECTED_K_CF.csv', 'SELECTED_K_PAIRS.jsonl', 'NATIVE_DEV.csv')]
        paths += [out/x for x in ('PORTABLE_REPORT_KO.md', 'ROOT_READOUT_KO.html', 'figures/DEV_RECALL.png', 'figures/NEW_FIT_CF_PATTERN.png')]
        manifest = dump(out/'MANIFEST.json', {'files': [ident(p) for p in paths], 'explicit_leaf_only': True})
        receipt = dump(out/'COMPLETE.json', {'completed': True, 'PASS': True, 'code': cfg['code'], 'entry': ident(entry),
                        'fit_complete': cfg['fit_complete'], 'result_QA': cfg['result_QA'], 'freeze': fit['freeze'],
                        'manifest': manifest, 'summary': ident(table/'SUMMARY.json'), 'current_admission': False,
                        'model_adopted': False, 'selected_arm': None, 'arrays_models_RGB_reads': 0,
                        'outputs': {p.name: ident(p) for p in paths}})
        print(json.dumps({'completed': True, 'PASS': True, 'COMPLETE': receipt,
                          'mean_balanced_recall_gain_pp': result['mean_balanced_recall_gain_pp'],
                          'CF_all': [r for r in cf if r['native_k'] == 'ALL']}, sort_keys=True))
        return receipt
    except BaseException as e:
        dump(out/'FAILURE.json', {'completed': False, 'PASS': False, 'error': repr(e), 'automatic_retry': False})
        raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry', required=True)
    args = ap.parse_args()
    run(args.entry)
