"""Render the closed, saved V183 source experiment; no model or pixel cache."""
from pathlib import Path
import argparse, datetime, hashlib, html, json

V = Path(__file__).resolve().parent
ARMS = ['C50_RAW96_DUAL_SUPPORT_OWNER_AUX',
        'C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX',
        'C52_RAW96_JITTER_PAIRED_SOURCE',
        'C53_RAW96_SAM_JITTER_PAIRED_SOURCE']
LABEL = dict(zip(ARMS, ['C50 기존 RGB', 'C51 기존 RGB+SAM',
                      'C52 증강 RGB', 'C53 증강 RGB+SAM']))


def bind(path):
    p = Path(path).resolve()
    assert p.is_relative_to(V.parent) and '_v112' not in str(p).lower()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(b):
    assert bind(b['path']) == b
    return Path(b['path'])


def read(p):
    return json.loads(Path(p).read_text())


def pct(v):
    return '—' if v is None else f'{v*100:.2f}%'


def table(headers, rows):
    esc = lambda s: html.escape(str(s))
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def run(entry):
    e = read(entry)
    assert e['approved_saved_root_readout'] is True and e['code'] == bind(__file__)
    fit, ev, vis = [read(checked(e[k])) for k in ['fit', 'evaluation', 'visual']]
    assert all(x['completed'] and x['PASS'] for x in [fit, ev, vis])
    assert fit['total_updates'] == 16000 and fit['all2_frozen_before_outputs_or_evaluation_access']
    assert ev['fit_complete'] == vis['fit_complete'] == e['fit']
    assert vis['evaluation_complete'] == e['evaluation']
    for b in e['independent_reviews'].values():
        r = read(checked(b)); assert r['completed'] and r['PASS']
    m = read(checked(ev['metrics']))
    assert m['COMMON_source_TRAIN_derived_not_heldout'] and m['VAL_historically_reused_not_blind']
    assert m['selected_model'] is None and not m['fully_solved']
    fixed = []
    for a in ARMS:
        pair = m['COMMON_matched_pairs'][a]['BEFORE_to_COVER']['joint_visible_only_to_foreign_only']
        g = m['original_fixed401_geometry_sensitivity'][a]
        z = g['fixed_old401_current_eligible']
        fixed.append([LABEL[a], f"{pair['both_supports_correct_at_both_images']}/{pair['pairs']}",
                      z['rows'], z['nominal_correct'], z['common_correct'],
                      z['nominal_correct_became_wrong'], z['nominal_wrong_became_correct'],
                      g['fixed_old401_now_unknown_truth'], g['fixed_old401_now_changed_truth'],
                      g['fixed_old401_now_inputU']])
    performance = []
    for domain in ['COMMON4096', 'SYNTH_VAL2048']:
        for a in ARMS:
            overall = m['results'][a][domain]['ALL']
            for task, name in [('visible_end_support', '부위 보임'), ('foreign_front_support', '외부 물체가 앞에 있음')]:
                z = overall[task]['primary_known_available']
                assert sum(map(sum, z['CM'])) == z['truth_known']
                performance.append([domain, LABEL[a], name, str(z['class_support']),
                                    pct(z['precision'][1]), pct(z['recall'][1]),
                                    str(z['CM']), z['truth_unknown'],
                                    overall['original_inputU'], z['prediction_U_among_known']])
    source_count = []
    for domain in ['SYNTH_AUG65536', 'PUBLIC_TRAIN1480', 'CF3816']:
        z = m['results'][ARMS[2]][domain]['ALL']
        source_count.append([domain, z['rows_all'], z['original_input_available'], z['original_inputU'],
                             z['visible_end_support']['primary_known_available']['truth_known'],
                             z['foreign_front_support']['primary_known_available']['truth_known'],
                             z['owner_hard_weak_only']['truth_known']])
    results = dict(completed=True, PASS=True, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   scope=e['scope'], fit=e['fit'], evaluation=e['evaluation'], visual=e['visual'],
                   independent_reviews=e['independent_reviews'],
                   table_fixed_pairs_and_old401=fixed, table_known_available_support=performance,
                   table_training_rows=source_count,
                   metric_rows_order='Binary truth NO/YES; prediction NO/YES/U',
                   fixed_decoder='Two independent support logits strictly >0; original inputU first',
                   historical_v61_VAL_task_mismatch_diagnostic=m['historical_v61_VAL_task_mismatch_diagnostic'],
                   source_training_apparent=True, COMMON_source_TRAIN_derived_not_heldout=True,
                   VAL_historically_reused_not_blind=True,
                   private_publicheld_new_pixel_prediction_annotation_body_access_this_phase=0,
                   new_training_updates=16000, no_PnP_or_original_q_L_native_correction=True,
                   selected_model=None, adopted=False, fully_solved=False, goal_status='active_unsolved',
                   full_goal_remaining='Verify frozen models on untouched current/publicheld inputs and references; reliable natural-real visibility/external occlusion remains unproven.')
    out = V/'ROOT_RESULTS_KO.json'
    with out.open('x') as f: json.dump(results, f, ensure_ascii=False, indent=2); f.write('\n')
    css = 'body{font:17px/1.65 sans-serif;margin:32px auto;padding:0 24px;max-width:1500px;color:#e6edf8;background:#101827}h1,h2{line-height:1.3}a{color:#8dc5ff}td,th{padding:9px 12px;border:1px solid #42536c;text-align:left}th{background:#20324d}table{border-collapse:collapse;font-size:14px}.scroll{overflow:auto;margin:20px 0}img{max-width:100%;height:auto}.note{padding:16px;background:#203047;border-left:5px solid #d89852}code{background:#23334b;padding:2px 5px}'
    page = '<!doctype html><html lang="ko"><meta charset="utf-8"><title>V183 고정 좌표 대비·미세 crop 증강 학습</title><style>'+css+'</style><h1>V183: 미세 crop 증강과 같은 좌표의 가림 대비 학습</h1>'
    page += '<p class="note">두 새 모델의 8,000회 학습과 저장 출력 비교를 완료했습니다. 아래 COMMON과 401개 진단은 합성 TRAIN에서 파생됐으며, VAL도 이전에 사용된 자료입니다. 실사 일반화·기존 실사 학습 모델 추월·전체 목표 해결을 증명하지 않습니다. 모델을 채택하거나 결과로 문턱을 바꾸지 않았습니다.</p>'
    page += '<h2>무엇을 바꿨나</h2><p>기존 합성 512장면의 초기 추론 q/L을 출발점으로, 4,096개 query에 사전 고정한 15개 이동·크기 상태를 적용했습니다. 여기에 BEFORE 초기 좌표를 네 가림 변형 이미지에 그대로 사용하는 4,096개 항목을 더해 총 65,536개 crop을 만들었습니다. 새로운 장면 65,536개가 아닙니다. 원래 좌표·저장 crop·PnP는 수정하지 않았습니다.</p><p>각 crop에서 같은 부위 보임/외부 앞가림 정의로 감독을 다시 계산했습니다. 자기 가림·모호함·누락은 UNKNOWN을 유지했습니다. 원 nominal 4,096개 RGB/valid는 원 저장 crop과 바이트 일치합니다. 기존 공개 실사와 CF의 약한 owner 감독은 그대로 보조 사용했고, 공개 mask를 키포인트 가시성 정답으로 바꾸지 않았습니다.</p>'
    page += table(['학습 영역','전체','입력 유효','입력 U','보임 정답 known','외부 정답 known','약한 hard owner known'],source_count)
    page += '<p>C52는 RGB/valid/고정 XY, C53은 같은 입력에 동결 SAM mask/availability를 추가합니다. 같은 작은 head·초기화·배치 추출·8,000회 업데이트·고정 decoder를 사용했습니다. 두 최종 가중치를 모두 동결한 뒤 평가 입력과 저장 출력을 열었습니다.</p>'
    page += '<h2>고정 좌표 대비와 기존 401개 민감도</h2><p>쌍 성공은 BEFORE 보임/외부없음과 COVER 안보임/외부있음의 네 판정을 모두 맞힌 경우입니다. 401개는 이전 모델의 같은 이미지·known 동일 정답으로 정의한 고정 집합입니다. 새 감독이 UNKNOWN/변경이면 제외 수를 별도 보존하며 정답을 강제로 일치시키지 않습니다.</p>'
    page += table(['모델','대비 쌍 모두 정답','401 중 현재 평가 가능','원 q/L 정답','공통 q/L 정답','정답→오답','오답→정답','새 정답 UNKNOWN','새 정답 변경','입력 U'],fixed)
    page += '<h2>같은 평가 입력의 support precision/recall·혼동행렬</h2><p>양성은 각 열의 support YES입니다. 두 support는 독립이므로 동시에 YES일 수 있습니다. 주 표는 원 입력 유효이고 해당 정답이 known인 항목입니다. CM 행은 NO/YES, 열은 NO/YES/U입니다. 정답 UNKNOWN 예측 census, 입력 U를 포함한 보조 분모, 재질·형상·native·self·좌표 오차별 전체 표는 저장 metric JSON에 보존했습니다.</p>'
    page += table(['자료','모델','과제','정답 NO/YES','양성 precision','양성 recall','CM NO/YES→NO/YES/U','유효 입력의 정답 U','전체 입력 U','known의 예측 U'],performance)
    page += '<p>v61은 같은 원 VAL q/L/native의 역사적 저장 출력을 별도로 비교했습니다. 한 픽셀 존재와 부위 support는 목표가 다르고 v61은 두 support 동시 YES를 표현하지 못하므로 직접 동등한 모델 성능 주장에 쓰지 않습니다. 원 정답·좌표·키포인트 순서와 보류는 바꾸지 않았습니다.</p>'
    page += '<h2>기존 0.38px 오류 사례와 전체 수치 그림</h2><p>BASE59/native1은 새 결과를 보기 전에 이전 C50 오류에서 고른 예시입니다. 대표 성공률이 아닙니다. 같은 BEFORE 이미지의 원 crop과 공통 crop, COVER crop을 그대로 복사하고 네 모델 수치만 밖에 표시했습니다. 중심과 크기가 함께 달라져 좌표 하나만의 인과 효과로 주장하지 않습니다.</p><a href="fixed_source_case_and_plot_v1/CARD.png"><img src="fixed_source_case_and_plot_v1/CARD.png" alt="기존 고정 오류 사례의 네 모델 비교"></a><a href="fixed_source_case_and_plot_v1/SCIENTIFIC_SOURCE_COMPARISON.png"><img src="fixed_source_case_and_plot_v1/SCIENTIFIC_SOURCE_COMPARISON.png" alt="고정 쌍·민감도·합성 VAL 비교"></a>'
    page += '<h2>남은 확인</h2><p>자기 가림은 upstream 기하 처리와 구분하며 이 모델을 키포인트 위치 보정기로 해석하지 않습니다. 고정된 두 모델을 모두 현재 실사·공개 held 자료에서 비교하는 V184 조건은 이번 새 결과 전에 작성했습니다. 현재 223장과 공개 held 관측은 학습·문턱·모델 선택에 사용하지 않았습니다. 자연 실사에서 부위 가시성과 외부 가림을 안정적으로 판단하는 전체 목표는 아직 미해결입니다.</p>'
    metrics_href = html.escape(str(Path(ev['metrics']['path']).relative_to(V)))
    page += '<p><a href="ROOT_RESULTS_KO.json">표와 출처 JSON</a> · <a href="'+metrics_href+'">전체 confusion/strata/paired/보류 수치</a> · <a href="fixed_source_case_and_plot_v1/index.html">고정 실제 crop</a></p></html>'
    pagepath=V/'ROOT_READOUT_KO.html'
    with pagepath.open('x') as f:f.write(page)
    print(json.dumps(dict(completed=True,PASS=True,results=bind(out),report=bind(pagepath))))


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path,required=True)
    run(ap.parse_args().entry)
