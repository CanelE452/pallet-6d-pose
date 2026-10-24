"""Render only completed source phases; never reads current evaluation inputs."""
import hashlib
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
V85 = PARENT / 'query_visibility_source_cue_corpus_20261023_v85'
V86 = PARENT / 'query_visibility_native_cue_evidence_20261023_v86'
V87 = PARENT / 'query_visibility_public_external_ray_20261023_v87'


def read(path):
    return json.loads(path.read_text())


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def link(path, label):
    return '<a href="' + html.escape(os.path.relpath(path, ROOT), quote=True) + '">' + html.escape(label) + '</a>'


def image(path, label):
    return '<figure><img src="' + html.escape(os.path.relpath(path, ROOT), quote=True) + '" alt="' + html.escape(label, quote=True) + '"><figcaption>' + html.escape(label) + '</figcaption></figure>'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + html.escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(c)) + '</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def pct(value):
    return '—' if value is None else f'{100 * value:.1f}%'


def main():
    run = ROOT / 'runs/YCB_EXTERNAL_RAY_V1'
    completion = read(run / 'FIT_COMPLETE.json')
    assert completion['completed'] and completion['optimizer_updates'] == 2000
    fit = read(run / 'SOURCE_RESULTS.json'); freeze = read(run / 'MODEL_FREEZE.json')
    corpus = read(V85 / 'all49_completed_corpus/CORPUS_SUMMARY_KO.json')
    native = read(V86 / 'ROOT_NATIVE24_DEFINITION_DIAGNOSIS_KO.json')
    ray = read(V87 / 'corpus_build_v1/CORPUS_SUMMARY_KO.json')
    gate = fit['predeclared_source_gate']; passed = sum(gate.values())
    files = [run / 'FIT_COMPLETE.json', run / 'SOURCE_RESULTS.json', run / 'MODEL_FREEZE.json', ROOT / 'independent_postfit_audit/INDEPENDENT_POSTFIT_AUDIT.json', V85 / 'all49_completed_corpus/CORPUS_SUMMARY_KO.json', V85 / 'all49_completed_corpus/SEAL.json', V85 / 'final_join_independent_review/INDEPENDENT_ALL49_PRIMARY294_AUDIT_KO.json', V86 / 'ROOT_NATIVE24_DEFINITION_DIAGNOSIS_KO.json', V87 / 'independent_output_audit/INDEPENDENT_ALL33600_OUTPUT_AUDIT_KO.json', V87 / 'ROOT_ACTUAL_RGB_REVIEW_RECEIPT.json']
    for f in files:
        assert f.is_file(), str(f)
    result = {'phase_complete': True, 'goal_status': 'active', 'fully_solved': False, 'current_capture_confirmation': '직접 촬영했고 KU Leuven 자료와 무관합니다', 'V85': corpus, 'V86': native, 'V87_corpus_summary': ray, 'V88': fit, 'V88_freeze': freeze, 'new_current223_reads_or_inference_in_V85_through_V88': 0, 'new_PnP': 0, 'next_direction_KO': '외부ray component의 공개검증 결과와 물리단서 감독을 구분해 유지한다. 공개 physicalcue 보조감독용 입력은 GT를 보지 않는 고정 격자/공간입력으로 설계하고 native8과의 대응을 새로 발명하지 않는다. 현재223으로 문턱/모델을 선택하지 않는다.', 'bindings': [binding(f) for f in files]}
    result_path = ROOT / 'ROOT_PHASE_RESULTS_AND_NEXT_KO.json'
    assert not result_path.exists()
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    status = '공개 ray 기준 통과 · 물리 모서리 평가는 별도 필요' if passed == 4 else '공개 ray 기준 미달 · 현재 평가셋으로 진행하지 않음'
    chunks = ['<p class="eyebrow">공개 실사 · 합성 / source-only 진단과 외부 가림 head</p><h1>가리는 물체와 보이는 모서리를 분리했습니다</h1>', '<p class="lead">' + status + '</p>', '<p>현재 223장은 직접 촬영했고 KU Leuven 데이터와 원본을 공유하지 않는다는 답변을 기록했습니다. 이 단계의 학습·문턱·체크포인트 선택에는 사용하지 않았고 PnP도 변경하지 않았습니다.</p>', '<section><h2>이번에 실제로 완료한 결과</h2>', table(['항목', '완료 범위', '해석'], [['공개 팔레트 물리 단서', '49쌍 × 6슬롯 = 294개 + 별도 seed 1쌍', '한 물리 bin의 환경 holdout / AI 관찰, native8 gold 아님'], ['합성 원래 native8 검토', '3개 TRAIN asset · 원래 24점', '좌표·키포인트 순서 유지, 새 물리점으로 이동하지 않음'], ['공개 YCB-V 외부 ray corpus', '4,200 instance · 33,600점 · 전행 독립 재현', 'clear 11,680 / foreign 3,787 / typed U 18,133'], ['작은 외부 가림 head', f'41,237 parameters · 2,000 updates · best {freeze["best_update"]}', f'고정 문턱 0.5 · 사전 고정 기준 {passed}/4']]]), '</section><section><h2>새 head의 공개 검증 결과</h2><p>합성 G38/P0/TEX로 사전학습한 공유 특징을 재사용하고, V87의 단일 ray 외부 물체 라벨로 YCB-V만 추가 학습했습니다. 합성의 기존 다중 ray 가림률을 같은 정답처럼 섞지 않았습니다. forward에는 RGB·P3·초기 query context의 6개 입력만 들어갑니다.</p>']
    rows = []
    for split, label in [('train', 'TRAIN'), ('val', 'VAL')]:
        metrics = fit['source_metrics'][split]['all_known_fixed_gate_metrics']
        rows.append([label, metrics['N'], metrics['confusion_rows_GT_CLEAR_EXTERNAL_cols_pred_CLEAR_EXTERNAL_U'], pct(metrics['clear_recall']), pct(metrics['external_precision']), pct(metrics['external_recall']), pct(metrics['factual_U_rate'])])
    chunks += [table(['분할', '알려진 ray 수', 'CM [[clear→C,E,U],[E→C,E,U]]', 'clear recall', 'E precision', 'E recall', '입력 무효 U'], rows), '<p>표는 teacher-known 전체를 포함하며, 정답 기준점이 초기 crop 밖인 실패도 분모에 남겼습니다. 학습에 들어간 조건부 결과·기준점 이탈·자기 가림 proxy별 결과와 teacher-U 전체 수는 ' + link(run / 'SOURCE_RESULTS.json', '상세 수치') + '에 있습니다. Teacher-U는 모델의 올바른 보류로 계산하지 않습니다.</p>', table(['사전에 고정한 공개 기준', '결과'], [[key, '통과' if value else '미달'] for key, value in gate.items()]), '<p>YCB sourceVAL은 반복 개발에 사용된 자료이며 독립 최종 일반화 평가가 아닙니다. 공개 query도 합성 TRAIN 오차 분포로 만든 고정 입력이어서 실제 초기 검출기의 출력과는 다릅니다. 이 출력은 단일 ray 외부 가림 proxy이고, clear를 보이는 물리 모서리 D로 바꿔 부르지 않습니다. 현재223 평가 진입은 허용하지 않았습니다.</p></section><section><h2>좌표가 맞아도 정의 때문에 판정이 달라집니다</h2><p>합성 원래 24점에서 관찰된 구조 단서 10개 중 8개는 정확한 r 픽셀의 M0=M4가 0이었습니다. 반면 뒤쪽 기준점이 자기 팔레트의 보이는 윗면 위에 투영되기도 했습니다. 주변 target support, 점의 pixel membership, 해당 모서리의 보임은 각각 다른 목표입니다. 이 사례들은 현재 평가셋 v61 FN을 새로 골라 본 결과나 인간 gold가 아닙니다.</p>', image(V86 / 'independent_native_audit/G38__G__f16972_ALL8_RAW96_MARKED.png', '합성 실제 원본 crop. 각 슬롯은 raw / 표시본 순서. 빨강 r, 청록 초기 q. k3·k4 주변 단서와 픽셀 소속 차이, k6의 보이는 윗면과 뒤 기준점 구분.'), '<p>' + link(V86 / 'ROOT_NATIVE24_DEFINITION_DIAGNOSIS_KO.json', '정의 차이 진단과 한계') + '</p></section><section><h2>공개 실사에서 외부 가림과 clear의 차이</h2>', image(V87 / 'visual_audit/04_EXTERNAL_OTHER_FIRST_HIT_PROXY/FULL_RAW_AND_TARGET_OUTLINE.png', '대상은 녹색 윤곽의 목재 block. 고정 r는 앞의 별도 캔 금속면에 놓임: foreign first hit 근거.'), image(V87 / 'visual_audit/00_CLEAR_TO_VIRTUAL_ENDPOINT_PROXY/FULL_RAW_AND_TARGET_OUTLINE.png', '고정 r는 목재 옆 배경에 놓임. ray가 비어 있다는 라벨은 물리 모서리가 보인다는 라벨과 다름.'), '<p>깊이 1,000장과 모든 visible instance mask 4,200장을 별도 코드로 다시 읽어 33,600개 라벨·사유·소유 대상·무효 처리의 정확한 일치를 확인했습니다. 자기 물체 먼저 닿음, band, depth 부재, 소유권 없음·겹침, 화면 밖은 모두 별도 U로 보존했습니다.</p><p>' + link(V87 / 'independent_output_audit/INDEPENDENT_ALL33600_OUTPUT_AUDIT_KO.json', '전행 독립 감사') + ' · ' + link(V87 / 'visual_audit/ALL8_FIXED_REASON_RAW_MARKED_NATIVE96_CONTACT.png', '고정 8사유 실제 crop') + '</p></section><section><h2>공개 팔레트 50쌍의 물리 단서 기록</h2><p>새 49쌍은 before RGB에서 6개 view-relative 슬롯의 좌표와 불확실 영역을 고정한 뒤 after의 같은 영역을 확인했습니다. 원래 단서를 정할 수 없던 점은 after에서 보이더라도 U로 남겼습니다. 기존 seed 1쌍의 8개 원제안·6개 연결된 물리 영역은 따로 보존했습니다.</p>', table(['after 관찰', '원래 primary', '독립 재검토 반영 제안'], [[key, corpus['primary_after_counts'].get(key, 0), corpus['proposed_aux_after_counts'].get(key, 0)] for key in sorted(set(corpus['primary_after_counts']) | set(corpus['proposed_aux_after_counts']))]), '<p>독립 secondary 검토는 사전에 고른 5쌍·30슬롯입니다. 단서 정체가 불명확했던 VAL의 2개는 기존 제안대로 U로 남겼습니다. 모든 49쌍을 두 사람이 검토한 것으로 표현하지 않습니다. 한 물리 bin이며 native virtual8 대응이나 인간 정답을 추가한 자료도 아닙니다.</p><p>' + link(V85 / 'all49_completed_corpus/CORPUS_SUMMARY_KO.json', '분할별 수치·라벨 대응·계보') + ' · ' + link(V85 / 'cumulative_20pair_review/CUMULATIVE20_RECONCILIATION_AND_COVERAGE_KO.json', '기존20쌍의 source crop 포함 여부') + '</p></section><section><h2>남은 작업</h2><p>목표 차이는 실제 source crop에서 확인됐지만 실사 전이 실패가 모두 해소됐다는 증거는 아닙니다. 외부 ray component의 성적을 보존하고, 물리 단서 보조 감독에 필요한 GT 비의존 공간 입력과 native 기준점 대응을 구분해 진행해야 합니다. 작은 실제 crop에 단서가 들어오지 않는 문제를 GT로 재중심화하거나 모델 보류 성공으로 숨기지 않습니다.</p><p>목표는 계속 진행 중이며 아직 해결 완료로 표시하지 않았습니다. 마지막 실제 현재223 평가는 V76이고 이 단계에서는 새 현재 평가를 실행하지 않았습니다.</p><p>' + link(result_path, '결과·다음 결정 JSON') + ' · ' + link(ROOT / 'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json', '실행 전 고정 프로토콜') + ' · ' + link(ROOT / 'independent_postfit_audit/INDEPENDENT_POSTFIT_AUDIT.json', '학습 결과 독립 감사') + '</p></section>']
    content = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>공개 실사 가림 진단 · 외부 ray head</title><style>body{font:16px/1.7 "Nanum Gothic",sans-serif;color:#182c3b;background:#f5f7f9;margin:0}main{max-width:1100px;margin:0 auto;padding:38px 28px 70px}h1{font-size:32px;line-height:1.3;margin:8px 0 18px}h2{font-size:23px;line-height:1.4;margin:0 0 18px}.eyebrow{color:#46667c;font-size:14px}.lead{font-size:21px;font-weight:700;color:#174c66}section{background:white;padding:26px;margin-top:25px;border:1px solid #d9e2e8;border-radius:12px}p{margin:12px 0}a{color:#12658a}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dbe4ea;padding:11px 9px}th{background:#edf3f7}td{overflow-wrap:anywhere}.scroll{overflow-x:auto}figure{margin:24px 0}img{width:100%;height:auto;display:block;border:1px solid #d6dee4}figcaption{font-size:14px;color:#476171;margin-top:8px}@media(max-width:650px){main{padding:20px 12px}section{padding:16px}h1{font-size:27px}}</style><main>' + ''.join(chunks) + '</main></html>'
    output = ROOT / 'ROOT_READOUT_KO.html'
    assert not output.exists()
    output.write_text(content)
    print(json.dumps({'report': binding(output), 'results': binding(result_path), 'source_gate_passed': passed, 'goal_status': 'active', 'fully_solved': False}))


if __name__ == '__main__':
    main()
