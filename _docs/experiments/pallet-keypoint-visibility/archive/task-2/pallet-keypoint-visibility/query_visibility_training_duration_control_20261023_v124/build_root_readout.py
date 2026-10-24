"""Saved-result report only. No model, fit, RGB crop extraction or selection."""
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).resolve().parent
P = V.parent
R = V / 'runs/C16_EXACT_C15_DURATION_CONTROL_8000'
E = V / 'private_evaluation/C16_FROZEN_CURRENT223/evaluation'

def read(p):
    return json.loads(Path(p).read_text())

def bind(p):
    p = Path(p)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def write(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')

def rel(p):
    return html.escape(os.path.relpath(p, V), quote=True)

def img(p, caption):
    return f'<figure><a href="{rel(p)}"><img src="{rel(p)}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'

primary = read(E / 'CURRENT_PRIMARY_RESULTS.json')
source = read(R / 'SOURCE_RESULTS.json')
new = read(R / 'NEW_FLAT_SOURCE_RESULTS.json')
source_review = read(V / 'saved_source_review/REVIEW.json')
current_review = read(V / 'saved_current_review/REVIEW.json')
coverage = read(V / 'ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_AUDIT.json')
gallery_path = V / 'visual_followup/all_external90/MAPPING_AND_COUNTS.json'
gallery = read(gallery_path)
assert gallery['counts']['external_E_correct_E'] == 35
assert gallery['counts']['external_E_missed_D'] == 55
assert len(gallery['external90_original_order']) == 90
assert len(gallery['visible_D_false_E10_original_order']) == 10
assert source_review['PASS'] and current_review['PASS'] and coverage['PASS']
assert source['selected_update'] == 4500
models = ['v61', 'legacy', 'V111_C11_RAW', 'V119_C14_RAW', 'V123_C15_RAW', 'V124_C16_RAW']
rows = [r for r in primary['tables']['D_vs_external_E_2x3']
        if r['cohort'] == 'paired_same_query_RGB' and r['model'] in models]
assert len(rows) == 12
c16 = [r for r in rows if r['model'] == 'V124_C16_RAW']
assert sum(r['D_pred_D'] for r in c16) == 1137
assert sum(r['E_pred_hidden'] for r in c16) == 35
now = datetime.now(timezone.utc).isoformat()
observations = [
    {'status': 'observed', 'case': 200, 'KO': 'ノ출된 테두리·모서리 구조가 보이고 C16은 D로 기존 사람 D와 일치한다.'.replace('ノ', '노')},
    {'status': 'observed', 'case': 1416, 'KO': '콘이 질의 부근을 차지하며 C16은 E로 기존 사람 E와 일치한다.'},
    {'status': 'observed', 'case': 325, 'KO': '콘과 남은 목재가 함께 보이는 경계에서 사람 E인데 C16은 D다.'},
    {'status': 'hypothesis_not_causal_proof', 'case': 325, 'KO': '주변 팔레트 재질의 존재를 기대한 구조 단서의 가시성과 혼동할 수 있다.'},
    {'status': 'observed', 'case': 1021, 'KO': '과거 C12에서 놓쳤지만 현재 C15/C16은 콘 중심 crop을 E로 맞힌다. 현재 실패로 제시하지 않는다.'},
    {'status': 'observed_all90_gallery', 'cases': [317, 748], 'KO': '가림 누락 목록에는 콘 몸통이 중심을 차지하는 crop도 있다. 모든55개를 경계 문제로 설명할 수 없다. 라벨 정의 차이 외에 실제 전이 실패가 남는다는 근거다.'},
    {'status': 'observed_all90_gallery', 'cases': [325, 996, 1692], 'KO': '가림 물체와 남은 목재·초록 팔레트가 섞인 경계가 여러 누락 crop에 반복된다. 이 유형의 전체 원인 비율은 수치화하지 않았다.'},
    {'status': 'observed_all90_and_D_false_E_gallery', 'cases': [632, 635, 602, 699], 'KO': '어둡고 구조가 잘 드러나지 않는 crop은 성공과 실패 양쪽에 있다. 조도 하나를 단일 원인으로 확정할 수 없고, 정보 부족과 모델 오판을 추가 분리해야 한다.'},
]
limits = [
    '현재223은 평가 전용으로 사용했지만 과거 개발에 노출된 자료이므로 독립 blind test가 아니다.',
    '사람 E의 숨은 reference는 추정/미지정 출처이며 정확한 좌표 gold가 아니다. D/E 판단 성공이 초기 좌표 검증을 뜻하지 않는다.',
    '최근 추가4896행은9장 원본·32개 약한 구조 단서에서 파생됐고, 추가 TRAIN known846행은3장 LOCO 원본의6개 단서다. 전체 이전 학습 자료의 원본 수를3장이라고 말하지 않는다.',
    '공개 실사 배경과 물체 cutout을 사용한 COVER/SHAM은 조건부 AUX 정답이다. 자연 가림의 native 키포인트 gold와 같지 않다.',
    '공개 source의 COVER2070행 q가 전부 alpha 내부라는 것은 이 source의 구성 빈틈이다. 전체 이전 데이터나 실제 실패55개의 단일 원인 증명이 아니다.',
    '자기 가림은 기존 초기 기하 경로이며 학습 D/E와 분리했다. 현재 실사로 문턱·checkpoint·모델 arm을 선택하지 않았다.',
    '고정4사례는 정성 진단 표본이고, 전체 E90 갤러리는 이 평가의 오류 목록이다. 어느 쪽도 자연 환경 전체의 원인 빈도 증명은 아니다.',
    '전체1784개에서 입력 U45개와 주 비교 D/E1237개 안의 U0개는 서로 다른 분모다.',
]
next_steps = [
    '같은 데이터의 추가 반복을 기본 처방으로 삼지 않는다. 총8000회 대조에서 공개 검증은8000회 시점에4500회 최저값보다 악화했고 실사 E정답 총수는 늘지 않았다.',
    '공개 source에서 확인된 구성 빈틈을 보완한다: 정확한 기준점을 갖는 합성 구조 또는 근거가 명확한 공개 구조 단서에서 의미상 가림과 q-pixel foreground 여부가 항상 일치하지 않는 경계·좌표오차 조합을 만들고 별도 검증한다. 약한 단서의 불확실 범위를 임의로 줄여 확정 라벨을 만들지 않는다.',
    '추가 확보한 LOCO 공식 TRAIN 촬영 묶음에서 원본·시점·배경 다양성을 늘린다. TOMIE scenario1은 별도 archive지만 다른 시설/날짜/실물 독립성은 미확인이다. 기존 VAL을 TRAIN으로 옮기지 않는다.',
    '주변에 가림 물체가 있어도 기준 단서가 보이는 D와, 주변 팔레트 재질은 남아도 기준 단서가 가려진 E를 대조한다. 공개 검증에서 보임 보존·E precision/recall·판단 보류를 함께 확인하고 설정을 동결한다.',
]
results = {
    'completed': True, 'created_UTC': now, 'goal_status': 'active', 'fully_solved': False,
    'phase': 'C16 duration-only control + saved-current visual diagnosis',
    'outcome_KO': '학습 기간 연장 후 보임1130→1137/1147, 외부가림35→35/90. 반복 학습만으로 잔여55개 외부가림을 해결하지 못했다.',
    'actual_work': {'optimizer_updates_total': 8000, 'exact_C15_replay': 1000,
                    'continuation_same_optimizer_RNG': 7000, 'source_selected_update': 4500,
                    'source_criterion': source['criterion_fourteen_VAL_cohorts_BCE'],
                    'source_regression_gates_PASS': 15, 'current_head_calls': 2,
                    'current_inference_reruns': 0, 'original_slots': 1784,
                    'valid_inputs': 1739, 'factual_U': 45, 'extra_operator_U': 0,
                    'current_used_for_fit_normalizer_threshold_checkpoint_arm_selection': False,
                    'PnP': 0, 'q_L_correction': 0, 'new_public_intake_used_by_C16': False},
    'current_primary_comparison': rows,
    'manual_near_D': [r for r in primary['tables']['manual_near_D'] if r['model'] == 'V124_C16_RAW'],
    'human_self_binary_response': [r for r in primary['tables']['human_self_binary_response'] if r['model'].startswith('V124_C16')],
    'new_public_source_metrics': new['metrics'],
    'public_source_boundary_coverage': {'known_COVER_E': 2070, 'q_inside_foreground_alpha': 2070,
                                      'known_SHAM_D': 2070, 'q_outside_foreground_alpha': 2070,
                                      'semantic_E_q_outside_example_count': 0,
                                      'correlated_rows_not_independent_scenes': True},
    'visual_observations': observations,
    'goal_assessment_KO': '初기 기하로 self를 분리한 뒤 보임/외부가림을 판단하는 목표가 과도하다는 증거는 없다. 다만 모든 작은 crop에서 숨은 부품 식별·좌표 정확성·가림을100% 동시에 확정한다는 목표는 현재 증거와 입력으로 보장할 수 없다.'.replace('初', '초'),
    'limits_KO': limits, 'next_action_KO': next_steps,
    'evidence': {name: bind(path) for name, path in {
        'source_freeze': R/'MODEL_FREEZE.json', 'fit': R/'FIT_COMPLETE.json',
        'source_review': V/'saved_source_review/REVIEW.json',
        'current_review': V/'saved_current_review/REVIEW.json',
        'current_metrics': E/'CURRENT_PRIMARY_RESULTS.json', 'evaluation': E/'EVALUATION_SEAL.json',
        'inference': E.parent/'INFERENCE_SEAL.json',
        'boundary_audit': V/'ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_AUDIT.json',
        'fixed4': V/'visual_followup/C16_FIXED4_MAPPING_AND_METRICS.json',
        'all_E90': gallery_path, 'public_intake': V/'public_intake/FINDINGS.json',
    }.items()},
}
write(V/'ROOT_RESULTS_KO.json', results)
label = {'v61':'v61', 'legacy':'기존 실사 모델', 'V111_C11_RAW':'C11', 'V119_C14_RAW':'C14', 'V123_C15_RAW':'C15', 'V124_C16_RAW':'C16 현재'}
tr = []
for r in rows:
    g = '목재' if r['group'].startswith('WOOD') else '정사각형'
    matrix = f"{r['D_pred_D']}, {r['D_pred_hidden']}, {r['D_pred_U']} / {r['E_pred_D']}, {r['E_pred_hidden']}, {r['E_pred_U']}"
    tr.append(f'<tr><td>{g}</td><td>{label[r["model"]]}</td><td>{matrix}</td><td>{100*r["D_recall"]:.1f}%</td><td>{100*r["external_E_response_precision"]:.1f}%</td><td>{100*r["external_E_response_recall"]:.1f}%</td><td>{100*r["unresolved_rate"]:.1f}%</td></tr>')
parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>C16 실제 crop 성공·실패와 남은 문제</title>',
'''<style>body{margin:0;background:#edf2f5;color:#183044;font:17px/1.65 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}section{background:white;padding:24px;border-radius:12px;margin:22px 0}h1{line-height:1.3}h2{margin-top:0}img{display:block;width:100%;height:auto}figure{margin:20px 0}figcaption{color:#476279;font-size:14px}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:15px}th,td{padding:10px;border-bottom:1px solid #d7e1e8;text-align:left}.table{overflow-x:auto}.lead{font-size:22px;font-weight:650}.note{background:#fff3d9;padding:16px;border-left:5px solid #d19a23}a{color:#126293}li{margin:8px 0}summary{cursor:pointer;font-weight:bold}</style><main>''',
'<h1>실제 crop으로 본 성공·실패와 남은 문제</h1>',
'<p class="lead">현재 C16: 보임 1,137/1,147개 유지 · 외부 가림 35/90개 검출 · 외부 가림 55개 누락</p>',
'<p>자기 가림은 초기 기하로 처리하고, 학습 head는 초기 q와 문맥에서 해당 구조 단서의 보임/외부 가림을 판단한다. 좌표를 새로 찾거나 교정하지 않았다.</p>',
'<section><h2>같은 네 사례, 현재 모델의 판정</h2>',
img(V/'visual_followup/INLINE_SUCCESS_FAILURE_C16_KO.png','원 q/L을 유지한 실제 저장 RGB. 왼쪽 아래는 남은 외부 가림 실패, 오른쪽 아래는 과거 실패에서 현재 성공으로 바뀐 사례.'),
'<details><summary>전체 원본·q 표시와 C15/C16 점수를 함께 보기</summary>'+img(V/'visual_followup/C16_FIXED4_SUCCESS_FAILURE.png','같은 고정4사례의 전체 원본, raw crop, 표시 복사본과 저장 점수.')+'</details>',
'<p>가림 물체인 콘을 전혀 못 알아보는 상태는 아니다. 그러나 남은 팔레트 재질의 존재와 기대한 구조 단서의 가시성을 혼동할 수 있다는 가설이 남는다. 그림만으로 단일 원인이나 숨은 reference 좌표의 정확성을 확정하지 않는다.</p></section>',
'<section><h2>같은 평가 대상의 수치</h2><p>행은 사람 D/E, 열은 예측 D/E/U. D=보임, E=외부 가림, U=보류. 같은 초기 query의 비교이며 v61/기존 모델과 C11 이후의 입력 특징은 서로 다르다.</p><div class="table"><table><thead><tr><th>유형</th><th>모델</th><th>혼동행렬 D행 / E행</th><th>보임 recall</th><th>가림 precision</th><th>가림 recall</th><th>보류율</th></tr></thead><tbody>',
''.join(tr), '</tbody></table></div><p>C15→C16: 목재 E14→12/29, 정사각형 E21→23/61, 합계35/90 유지. 보임은7개 추가 보존했다. 최신 모델은 현재 평가로 선정한 최상 모델을 뜻하지 않는다.</p></section>',
'<section><h2>학습을 더 오래 하면 해결되는가</h2>',
img(V/'source_visual_followup/SOURCE_DURATION_CURVES.png','데이터·head·학습률·문턱을 고정한 기간 대조. 공개 검증으로4500회 모델 선택. 총8000회까지 반복해도 잔여 가림의 개선은 제한적이다.'),
'<p>공개 평균 BCE는 1,000회 0.04267 → 4,500회 0.03433 → 8,000회 0.03908이었다. 공개 목재 블록 가림의 중앙 성공은16→18/32로 소폭 개선했지만, 현재 실사 가림 검출 합계는 늘지 않았다. 같은 자료의 반복 횟수 부족만으로 설명하기 어렵다.</p></section>',
'<section><h2>공개 학습 자료에서 직접 확인한 빈틈</h2>',
'<p>최근 추가 source의 known COVER E 2,070/2,070개는 q pixel도 붙인 가림 물체 안에 있었다. SHAM D 2,070/2,070개는 q pixel이 그 물체 밖이었다. 즉 이 묶음에는 <b>기준 단서는 가려졌으나 초기 q만 경계를 넘어 주변 원본 쪽을 보는 E</b>가 없다. 합성 정답이 틀렸다는 뜻이 아니라, 목표에 필요한 조합의 범위가 부족하다는 뜻이다.</p>',
img(P/'query_visibility_flat_pallet_cue_source_20261023_v122/fixed_corpus_visual_review/CUE_07_CENTRAL_RANK_01.png','실제 사용한 공개 배경 기반 BEFORE / COVER / SHAM 예시. COVER는 기준 단서와 q를 덮고 SHAM은 단서를 보존한다.'),
'<p>최근4,896개 파생 crop의 원본은9장·구조 단서32개이며, 새 학습에 들어간846known행은3장·6개 단서다. 공개 실사를 사용했어도 자연 외부 가림의 종류와 촬영 환경이 충분해졌다고 할 수 없다.</p></section>',
'<section><h2>목표의 적절성과 다음 해결 방향</h2>',
'<p class="note">보임과 외부 가림을 판별하는 목표 자체가 과도하다는 증거는 없다. 지금은 일부 전이와 성공이 확인됐지만 신뢰할 수준으로 해결되지는 않았다. 모든 작은 crop에서 좌표 정확성까지100% 증명하는 문제로 확대하면 안 된다.</p>',
'<ol>'+''.join('<li>'+html.escape(s)+'</li>' for s in next_steps)+'</ol>',
'<p>새 공개 원본 후보 조사와 소량 샘플 확보는 마쳤지만 C16에는 사용하지 않았다. 다음 데이터 구성·학습 실험은 아직 완료했다고 주장하지 않는다.</p></section>',
'<section><h2>전체 외부 가림90개와 보임 오판10개</h2><p>잘 나온 사례만 골라 보이지 않도록 원행 순서로 전체 목록을 펼친다. 이 시각화는 진단용이며 현재 평가로 학습·문턱·모델을 선택하지 않는다.</p>',
'<p><b>전체 확인 결과:</b> 콘과 남은 팔레트 재질이 섞이는 경계가 여러 실패에 반복된다. 그러나 행317·748처럼 콘 몸통이 중심을 차지하는데도 놓치는 사례가 있어, 경계나 목표 정의 차이만으로 설명할 수 없다. 실제 전이 실패도 남아 있다. 어두운 crop은 성공·실패 양쪽에 있어 조도 하나로도 설명할 수 없다. 원인별 전체 비율을 새 정답처럼 확정하지 않았다.</p>',
]
for page in sorted((V/'visual_followup/all_external90').glob('*.png')):
    parts.append('<details><summary>'+html.escape(page.name)+'</summary>'+img(page,page.name)+'</details>')
parts += ['</section><section><h2>해석의 한계와 저장 근거</h2><ul>',
          ''.join('<li>'+html.escape(s)+'</li>' for s in limits), '</ul><p>',
          f'<a href="{rel(V/"ROOT_RESULTS_KO.json")}">결과 JSON</a> · <a href="{rel(gallery_path)}">전체 crop 대응표</a> · <a href="{rel(V/"ROOT_PUBLIC_QUERY_BOUNDARY_COVERAGE_AUDIT.json")}">공개 source 경계 감사</a>',
          '</p></section></main></html>']
(V/'ROOT_READOUT_KO.html').write_text('\n'.join(parts))
print(json.dumps({'report': str(V/'ROOT_READOUT_KO.html'), 'results':str(V/'ROOT_RESULTS_KO.json'), 'fully_solved':False},ensure_ascii=False))
