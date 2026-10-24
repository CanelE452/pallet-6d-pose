"""Compact Korean report from completed saved JSON/CSV and existing figures only."""
from pathlib import Path
import csv
import hashlib
import html
import json

V = Path(__file__).resolve().parent
E = V / 'current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT'
S = V / 'runs/C1_NATURAL_KNOWNSET_SOURCE_V1'
MODELS = ['v61', 'legacy', 'V91_RAW', 'V94_C1_RAW']
NAMES = {'v61': 'v61', 'legacy': '기존 가림 모델', 'V91_RAW': 'V91 raw', 'V94_C1_RAW': 'C1 raw',
         'V91_PLUS_OLD_GEOMETRY': 'V91 + 기존 geometry', 'V94_C1_PLUS_OLD_GEOMETRY': 'C1 + 기존 geometry'}
GROUPS = {'WOOD_PRIMARY105': '목재', 'SQUARE118': '정사각형'}


def read(p):
    return json.loads(p.read_text())


def bind(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def rows(name):
    return list(csv.DictReader((E / name).open()))


def pct(x):
    return f'{100 * float(x):.1f}%'


def escape(x):
    return html.escape(str(x))


def table(headers, body):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(x)}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in body) + '</tbody></table></div>'


def figure(rel, caption):
    return f'<figure><a href="{escape(rel)}"><img loading="lazy" src="{escape(rel)}" alt="{escape(caption)}"></a><figcaption>{escape(caption)}</figcaption></figure>'


def main():
    source = read(S / 'SOURCE_RESULTS.json')
    gates = read(S / 'SOURCE_GATE_RESULTS.json')
    current = read(E / 'CURRENT_PRIMARY_RESULTS.json')
    metrics, near, self_rows = rows('METRICS.csv'), rows('MANUAL_NEAR_D.csv'), rows('HUMAN_SELF_BINARY_RESPONSE.csv')
    chosen = [r for r in metrics if r['cohort'] == 'paired_same_query_RGB' and r['task'] == 'D_vs_E' and r['model'] in MODELS]
    assert len(chosen) == 8
    for r in chosen:
        assert int(r['TP']) + int(r['FN_decided']) + int(r['U_visible']) == int(r['visible_support'])
        assert int(r['FP']) + int(r['TN']) + int(r['U_occluded']) == int(r['occluded_support'])
    c1 = {r['group']: r for r in chosen if r['model'] == 'V94_C1_RAW'}
    assert [(int(c1[g]['TN']), int(c1[g]['occluded_support'])) for g in GROUPS] == [(12, 29), (32, 61)]
    assert source['selected_update'] == 100 and gates['all_gates_pass']
    assert source['metrics']['VAL_HALL']['matrix'] == [[78, 3, 0], [0, 11, 0]]
    near_map = {(r['group'], r['model']): r for r in near}
    assert [(int(near_map[g, 'V94_C1_RAW']['pred_visible']), int(near_map[g, 'V94_C1_RAW']['N'])) for g in GROUPS] == [(149, 153), (216, 223)]
    figure_specs = [
        ('root_conditional_source_crop_review/FIRST8_SAVED_SOURCE_QUERY_CROPS_GT_ONLY.png', '공개 source 고정 8개 원 query crop. D는 기록된 목재 구조 단서, E는 해당 단서 위의 cardboard로 검토되었습니다. 조건부 AI cue 라벨이며 native8 대응이나 인간 gold가 아닙니다. 재질 shortcut 가능성은 남습니다.'),
        ('root_conditional_source_crop_review/ALL3_SOURCE_VAL_VISIBLE_MISSES_GT_ONLY.png', '공개 source VAL에서 남은 D→hidden 오류 3개 전부. 기록된 목재 support가 crop 오른쪽 경계에 보이고 q에는 바닥이 있으며, 일부 아래 padding이 있습니다. q에서 단서까지의 거리는 native keypoint 오차가 아닙니다. 동일 scalar 연산자만으로 cue grounding이 입증되지 않습니다.'),
        ('current_nearD_failure_crop_diagnosis/CONTACT_01_ALL8_V2.png', '현재 manual-click D·초기 좌표 오차≤3px에서 C1이 hidden으로 답한 11개 중 원 native 출력 순서 첫 8개입니다. 저장 RGB96과 원 q/r를 그대로 표시했습니다. 첫 사례 오른쪽 검은 영역은 validity 0인 padding이며 외부 가림 물체가 아닙니다.'),
        ('current_nearD_failure_crop_diagnosis/CONTACT_02_REMAINING3_V2.png', '동일한 고정 11개 오류 중 나머지 3개입니다. 합계 목재4개·정사각형7개를 모두 보존합니다. 사례10은 이 crop만으로 뚜렷한 팔레트 cue를 독립 확인하기 어려우나 원 human D/수동 r을 그대로 유지합니다. target identity·재질·semantic reference 중 단독 원인을 확정하지 않습니다.')
    ]
    for rel, _ in figure_specs:
        assert (V / rel).is_file(), f'Wait for completed figure: {rel}'
    a_meta = V / 'current_nearD_failure_crop_diagnosis/ALL11_CASES_AND_PROVENANCE.json'
    assert a_meta.is_file(), 'Wait for completed all11 provenance'
    cases = read(a_meta)['all11']
    assert len(cases) == 11
    old_D_counts = {m: sum(r['saved_models'][m]['decision'] == 'D' for r in cases) for m in ['v61', 'V91']}
    assert old_D_counts == {'v61': 9, 'V91': 5}
    observation_path = V / 'current_nearD_failure_crop_diagnosis/ACTUAL_ALL11_VISUAL_OBSERVATIONS_KO_V2.json'
    root_view_path = V / 'current_nearD_failure_crop_diagnosis/ROOT_ACTUAL_ALL11_VIEW.json'
    audit_path = V / 'independent_actual_C1_current_evaluation_audit/REVIEW.json'
    assert read(audit_path)['PASS'] and read(root_view_path)['completed']
    availability = current['tables']['availability']
    result = {
        'schema': 'V94_C1_COMPACT_KOREAN_PHASE_RESULTS_V1', 'completed_report': True,
        'goal_complete': False, 'goal_status': 'active', 'primary': 'V94_C1_RAW',
        'conclusion_KO': '공개 Pallox 조건부 source gate는 통과했으나 현재 외부 가림 recall은 목재12/29·정사각형32/61로 부족합니다. manual≤3px D 유지율 개선은 관찰됐지만 전체 목표 해결이나 단일 원인의 증명이 아닙니다.',
        'head_fit': {'source': 'Pallox natural conditional known-cue-set only', 'physical_object_count': 1,
                     'natural_pair_same_q_L_count': 0,
                     'TRAIN_D': 56, 'TRAIN_E': 9, 'VAL_D': 81, 'VAL_E': 11,
                     'updates': 2000, 'selected_update': 100, 'parameter_count': 26573,
                     'synthetic_or_YCB_head_training_rows': 0, 'controlled_V3_included': False,
                     'native_expected_part': 'UNKNOWN', 'human_gold': False,
                     'source_results': source, 'source_gates': gates},
        'operator': {'forward': ['p3_roi', 'local_rgb96', 'local_valid96'],
                     'q_L_slot_candidate_changed': False, 'scalar': 'z<0 D; z>=0 auxiliary hidden; unavailable U',
                     'localization_or_pose_correction': False, 'PnP': 0, 'learned_self': False,
                     'native_expected_part_association': 'UNKNOWN'},
        'paired_same_query_RGB_D_vs_E': chosen,
        'manual_near_D_le3': [r for r in near if r['model'] in MODELS],
        'human_S_binary_hidden_response': [r for r in self_rows if r['group'] == 'WOOD_PRIMARY105'],
        'availability_all1784': availability, 'availability_by_group': rows('AVAILABILITY_BY_GROUP.csv'),
        'limitations_KO': [
            '현재223은 반복 개발 중 노출된 평가 집합이며 새로운 blind test가 아닙니다. C1 fit/문턱/체크포인트 선택에는 사용하지 않았습니다.',
            'V91과 architecture·source q/L 분포·학습 readout/operator·supervision이 함께 달라져 개선 원인을 하나로 분리할 수 없습니다.',
            '공개 source는 동일 물리 bin의 LAB/HALL 분리와 약한 조건부 cue 라벨입니다. FULL 포함은 native expected-cue association 증명이 아닙니다.',
            'C1은 sourceGT region/mosaic readout을 현재 q중심 scalar로 바꾸었으나 target identity, semantic correspondence, 자연 외부 가림 전이 모두 해결됐다고 할 수 없습니다.',
            'U는 입력/연산자 가용성 보류이며 의미적 불확실성이나 confidence calibration을 학습한 결과가 아닙니다.',
            '합성/YCB/controlledV3는 이번 C1 head fit에 섞지 않았습니다. 기존 frozen backbone과 저장 입력은 유지했습니다.'
        ],
        'figures': [dict(binding=bind(V / rel), caption_KO=cap) for rel, cap in figure_specs],
        'all11_diagnostic_metadata': bind(a_meta),
        'all11_visual_observations': bind(observation_path),
        'root_all11_actual_view': bind(root_view_path),
        'independent_current_saved_audit': bind(audit_path),
        'prior_D_among_C1_nearD_misses': old_D_counts,
        'source_files': [bind(S / 'SOURCE_RESULTS.json'), bind(S / 'SOURCE_GATE_RESULTS.json'), bind(E / 'CURRENT_PRIMARY_RESULTS.json'), bind(E / 'METRICS.csv'), bind(E / 'MANUAL_NEAR_D.csv'), bind(E / 'HUMAN_SELF_BINARY_RESPONSE.csv')],
        'report_creation_new_model_inference_training_RGB_decode': 0,
        'final_root_QA_and_seal': 'pending_root'
    }
    raw_body = []
    for g in GROUPS:
        for m in MODELS:
            r = next(r for r in chosen if r['group'] == g and r['model'] == m)
            matrix = f"[{r['TP']}, {r['FN_decided']}, {r['U_visible']}]<br>[{r['FP']}, {r['TN']}, {r['U_occluded']}]"
            raw_body.append([GROUPS[g], NAMES[m], f'<span class="matrix">{matrix}</span>', pct(r['visible_precision']), pct(r['visible_recall_all_GT']), pct(r['occluded_precision']), pct(r['occluded_recall_all_GT']), pct(r['abstention_rate'])])
    near_body = []
    for m in MODELS:
        vals = []
        for g in GROUPS:
            r = near_map[g, m]
            vals += [f"{r['pred_visible']}/{r['N']} · {pct(r['semantic_D_recall'])}", r['pred_binary_hidden'], r['unresolved']]
        near_body.append([NAMES[m]] + vals)
    self_body = []
    for m in ['v61', 'legacy', 'V91_RAW', 'V94_C1_RAW', 'V91_PLUS_OLD_GEOMETRY', 'V94_C1_PLUS_OLD_GEOMETRY']:
        pair = next(r for r in self_rows if r['group'] == 'WOOD_PRIMARY105' and r['cohort'] == 'paired_same_query_RGB' and r['model'] == m)
        pipe = next(r for r in self_rows if r['group'] == 'WOOD_PRIMARY105' and r['cohort'] == 'same_slot_pipeline' and r['model'] == m)
        self_body.append([NAMES[m], f"{pair['binary_hidden_response']}/21", pair['unresolved'], f"{pipe['binary_hidden_response']}/136", pipe['unresolved']])
    style = '''<style>*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#18222d;font:16px/1.65 system-ui,-apple-system,"Noto Sans CJK KR",sans-serif}main{max-width:1200px;margin:auto;padding:40px 28px 80px}h1{font-size:32px;line-height:1.3;margin:8px 0 18px}h2{font-size:23px;margin:36px 0 12px}p{margin:12px 0}.eyebrow{color:#526474;font-weight:650}.lead{font-size:20px}.notice{background:#fff4dd;border-left:5px solid #c58a14;padding:16px 20px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}.card{background:white;border:1px solid #dbe2e8;border-radius:8px;padding:16px}.card strong{font-size:27px;display:block}.small,figcaption{font-size:14px;color:#4d5c69}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px;margin:12px 0}th,td{border:1px solid #dbe2e8;padding:9px 10px;text-align:right;vertical-align:middle}th{background:#eaf0f5;white-space:nowrap}td:first-child,td:nth-child(2){text-align:left}.matrix{font-family:ui-monospace,monospace;white-space:nowrap;font-size:13px}a{color:#135c91}figure{margin:22px 0;background:white;padding:12px;border:1px solid #dbe2e8}img{display:block;width:100%;height:auto}figcaption{margin:9px 4px}details{background:#fff;border:1px solid #dbe2e8;padding:10px 16px;margin:16px 0}summary{cursor:pointer;font-weight:650}code{font-size:14px;background:#eaf0f5;padding:2px 5px}@media(max-width:750px){main{padding:24px 14px}.cards{grid-template-columns:1fr}h1{font-size:26px}}</style>'''
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V94 C1 — 초기 query crop 가림 판별</title>{style}<main>
    <div class="eyebrow">V94 · C1 natural known-set · phase 결과</div><h1>초기 query crop을 그대로 읽는 scalar 판별기</h1>
    <p class="lead">공개 source 검증은 통과했습니다. 현재 데이터에서는 보이는 점을 유지하는 성능이 높아졌지만, 외부 가림을 놓치는 문제가 남아 있습니다.</p>
    <div class="notice"><b>목표는 아직 진행 중입니다.</b> 현재 외부 가림 recall은 목재 <b>12/29 (41.4%)</b>, 정사각형 <b>32/61 (52.5%)</b>입니다. source 통과를 실제 팔레트 문제 해결로 해석하지 않습니다.</div>
    <div class="cards"><div class="card"><strong>149/153 · 216/223</strong>좌표 오차≤3px인 manual D 유지<br><span class="small">목재 97.4% · 정사각형 96.9%</span></div><div class="card"><strong>1739 valid · 45 U</strong>원래 1784 query 입력 보존<br><span class="small">추가 입력 U 0 · 좌표 이동/보정/PnP 0</span></div><div class="card"><strong>65 TRAIN · 92 VAL</strong>공개 source 조건부 known 행<br><span class="small">물리 bin 1개 · human/native gold 아님</span></div></div>
    <h2>무엇을 학습하고 평가했나</h2>
    <p>C1은 원래 q/L crop의 <code>P3 ROI + RGB96 + valid mask</code>를 받아 <b>가림 scalar logit 하나</b>를 냅니다. 위치를 찾거나 키포인트를 보정하지 않습니다. <code>z&lt;0 → D</code>, <code>z≥0 → auxiliary hidden</code>이며 입력을 읽을 수 없으면 U입니다. 외부/자기 가림 원인을 학습해 구분하는 모델은 아닙니다.</p>
    <p>새 head의 학습행은 공개 <b>Pallox 목재 bin의 자연 before/after conditional known-cue-set</b>뿐입니다. TRAIN_LAB 56D+9E, VAL_HALL 81D+11E를 사용하고 나머지 643행은 unknown으로 보존했습니다. 자연 촬영쌍이지만 각 프레임의 실제 initializer q/L이 달라 동일 q/L counterfactual 짝 학습은 아닙니다. 합성·YCB·controlledV3 학습행은 이번 fit에 포함하지 않았습니다. 기존 frozen backbone/저장 입력은 유지했습니다. 2,000 update 중 source VAL 클래스별 BCE 평균이 최소인 <b>update100</b>을 선택했습니다.</p>
    {table(['Source split','실제 D → [D,E,U]','실제 E → [D,E,U]','D recall','E recall','known U'], [['TRAIN_LAB (65)','[56, 0, 0]','[0, 9, 0]','100.0%','100.0%','0%'],['VAL_HALL (92)','[78, 3, 0]','[0, 11, 0]','96.3%','100.0%','0%']])}
    <p class="small">사전 source 기준 D recall≥85%, E recall≥75%, known U≤5%를 모두 통과했습니다. 한 물리 bin의 LAB/HALL 분리이며 flat pallet 일반화 증거는 제한적입니다. crop에 기록된 cue가 FULL 포함된다는 사실이 native expected-cue 대응을 증명하지는 않습니다.</p>
    <h2>현재 데이터: 같은 query·RGB에서 D 대 외부 E</h2>
    <p>목재 N=645 (D616/E29), 정사각형 N=592 (D531/E61)를 모든 모델에 동일하게 적용했습니다. 혼동행렬은 <b>행=실제 D,E / 열=예측 D,hidden,U</b>입니다. P/R은 각각 D와 E를 positive로 계산하며 recall 분모에 보류도 포함합니다. C1/V91의 paired 집합 입력 U=0과 전체 입력의 45U는 서로 다른 분모입니다. 기존 모델의 보류는 아래 표에 별도로 포함했습니다.</p>
    {table(['대상','모델','CM: D행 / E행','D precision','D recall','E precision','E recall','보류율'],raw_body)}
    <p>C1은 V91보다 정사각형 D를 <b>472→492</b>, 외부 E를 <b>30→32</b> 더 맞혔습니다. 목재는 D <b>586→583</b>으로 줄었고 외부 E는 <b>11→12</b>로 늘었습니다. 높은 v61 E recall은 D를 대량으로 hidden으로 답하는 결과와 함께 봐야 합니다.</p>
    <h2>초기 좌표 오차≤3px인데 보이는 점을 없애는 문제</h2>
    <p>기존 manual-click D 중 valid query이며 초기 오차≤3px인 <b>목재153 / 정사각형223</b>을 그대로 비교했습니다. projected reference와 혼합하지 않았습니다.</p>
    {table(['모델','목재 D 유지','hidden','U','정사각형 D 유지','hidden','U'],near_body)}
    <p>C1의 잔여 오류는 목재4개·정사각형7개입니다. 이 11개 중 v61은 9개, V91은 5개를 D로 답했으므로 개별 회귀도 남습니다. 개선은 관찰됐지만 V91과 구조·source q/L 분포·학습 readout/operator·supervision이 함께 바뀌었으므로 특정 변경 하나의 효과로 분리할 수 없습니다.</p>
    {figure(*figure_specs[2])}{figure(*figure_specs[3])}
    <h2>자기 가림은 별도의 binary 반응으로만 해석</h2>
    <p>인간이 S로 기록한 목재 중 <b>paired 같은-query 분모21</b>과 <b>pipeline 분모136</b>을 분리했습니다. 아래는 그 중 hidden으로 답한 수이며, 학습된 self recall이나 E/S 원인 구분 정확도가 아닙니다. 정사각형에는 S gold가 없습니다.</p>
    {table(['모델/route','paired S→hidden','paired U','pipeline S→hidden','pipeline U'],self_body)}
    <p class="small">기존 geometry 적용은 secondary route입니다. D/E paired 수치는 raw와 같지만 pipeline S에서는 별도의 기하 판정이 크게 작용합니다. invalid/U는 geometry보다 먼저 적용됩니다.</p>
    <details><summary>공개 source 실제 crop과 남은 source 오류</summary>{figure(*figure_specs[0])}{figure(*figure_specs[1])}</details>
    <h2>해석과 다음 단계의 경계</h2>
    <p>현재223은 이전 개발·진단에서 반복 노출된 집합이므로 새 blind test가 아닙니다. 이번 C1의 학습·문턱·체크포인트 선택에는 쓰지 않았습니다. 자연 외부 가림의 recall이 충분하지 않고, expected target/cue 대응은 UNKNOWN이며 재질 shortcut도 배제되지 않았습니다. 따라서 <b>전체 목표 완료는 false</b>입니다.</p>
    <p>controlledV3는 준비 단계이며 이번 결과에 섞지 않았습니다. 새 학습의 효과를 주장하려면 source의 target/cue 대응, 원 q/L에서 실제 관측 가능한 단서, natural foreign occlusion 전이를 계속 구분해야 합니다. 의미적 불확실성을 학습한 보류나 confidence calibration도 아직 없습니다.</p>
    <p class="small">원표: <a href="current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT/METRICS.csv">현재 D/E·S 전체 metrics</a> · <a href="current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT/MANUAL_NEAR_D.csv">manual-near D</a> · <a href="runs/C1_NATURAL_KNOWNSET_SOURCE_V1/SOURCE_RESULTS.json">source 결과</a> · <a href="ROOT_RESULTS_KO.json">이 보고서 JSON</a> · <a href="current_nearD_failure_crop_diagnosis/ALL11_CASES_AND_PROVENANCE.json">고정 11사례 provenance</a></p>
    <p class="small">이 보고서는 저장 JSON/CSV와 기존 그림만 사용했습니다. 보고서 작성 중 새 모델 추론·학습·원RGB decode·좌표 변경·PnP는 수행하지 않았습니다. 최종 root 시각 QA와 봉인은 별도입니다.</p></main></html>'''
    for name, content in [('ROOT_RESULTS_KO.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'), ('ROOT_READOUT_KO.html', text)]:
        with (V / name).open('x') as f:
            f.write(content)
    print(json.dumps({n: bind(V / n) for n in ['ROOT_RESULTS_KO.json', 'ROOT_READOUT_KO.html']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
