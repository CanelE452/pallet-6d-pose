from pathlib import Path
from datetime import datetime, timezone
import hashlib
import html
import json

V = Path(__file__).resolve().parent
E = V / 'saved_five_model_evaluation_v1'
P = V.parent


def bind(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(path.read_text())


def link(path, label):
    return f'<a href="{html.escape(str(path))}">{html.escape(label)}</a>'


def table(headers, rows):
    head = ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def pct(x):
    return '정의 불가' if x is None else f'{x*100:.1f}%'


data = read(E / 'CURRENT_FROZEN_BENCHMARK_RESULTS.json')
numbers = read(V / 'scientific_readout_v1/PLOT_NUMBERS.json')['models']
audit = V / 'independent_actual_saved_evaluation_review_a/REVIEW.json'
assert read(audit)['PASS'] and data['PASS'] and data['selected_model'] is None
visual = V / 'ROOT_ACTUAL_PLOT_AND_FIXED19_VISUAL_REVIEW.json'
gallery_qa = V / 'fixed19_current_crop_review_v1/ACTUAL_VISUAL_QA.json'
assert read(visual)['layout_readable']
plot = V / 'scientific_readout_v1/CURRENT223_DIRECT_OWNER_DIAGNOSTIC_V2.png'
gallery = V / 'fixed19_current_crop_review_v1/index.html'
public = P / 'query_visibility_public_held_blind_center_owner_review_20261024_v177/ROOT_READOUT_KO.html'
material = [r for r in data['tables']['D_vs_external_E_2x3'] if r['cohort'] == 'paired_same_query_RGB']
result = dict(
    schema='V178_ROOT_FROZEN_DIRECT_OWNER_CURRENT_READOUT_v1',
    UTC=datetime.now(timezone.utc).isoformat(), completed=True, phase_complete=True,
    conclusion_KO='C47/C48/C49의 동결 사용자 평가를 완료했다. C48/C49는 기존 보임 주석 유지율이15.8%/16.8%로 낮아 v61/legacy 대비 실사용 개선을 입증하지 못했다. 목표 차이와 남은 인식 실패가 함께 관찰되며 모델 채택과 전체 해결은 하지 않는다.',
    frames=223, original_slots=1784, original_available=1739, original_inputU=45,
    primary_cohort='paired_same_query_RGB_D_vs_E', primary_N=1237, D_N=1147, E_N=90,
    primary_aggregate=numbers, primary_material=material,
    manual_near_D=data['tables']['manual_near_D'],
    self_response=data['tables']['human_self_binary_response'],
    owner_native_reference_cross=data['tables']['owner_vs_native_visibility_and_reference_error'],
    all1784_model_inventory=data['tables']['all1784_model_inventory'],
    numerical_audit=bind(audit), prediction_seal=bind(V / 'frozen_all3_direct_owner_predictions_v1/PREDICTION_SEAL.json'),
    evaluation_seal=bind(E / 'EVALUATION_SEAL.json'),
    root_visual_review=bind(visual), fixed19_visual_QA=bind(gallery_qa),
    plot=bind(plot), fixed19_gallery=bind(gallery),
    source_public_report=bind(public),
    current_data_used_for_training_or_calibration_or_selection=False,
    new_training_updates=0, new_encoder_or_initializer_or_PnP_calls=0,
    existing_reference_le3_not_physical_gold=True,
    literal_owner_not_native_external_self_gold=True,
    historically_exposed_private_and_legacy_CAL_overlap=True,
    selected_model=None, adopted=False, fully_solved=False, goal_status='active_unsolved',
)
out = V / 'ROOT_RESULTS_KO.json'
with out.open('x') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write('\n')

summary = table(['모델','D 유지/1147','D precision','E 응답/90','E 응답 precision','보류율'], [
    [r['model'], f"{r['CM_rows_D_E_columns_D_E_U'][0][0]}/1147 ({pct(r['D_recall'])})",
     pct(r['D_precision']), f"{r['CM_rows_D_E_columns_D_E_U'][1][1]}/90 ({pct(r['E_response_recall'])})",
     pct(r['E_response_precision']), pct(r['U_rate'])] for r in numbers])
matrices = ''.join(f"<h3>{r['model']}</h3>"+table(['기존 주석','D 응답','E 응답','U'],
    [['D', *r['CM_rows_D_E_columns_D_E_U'][0]],['외부 E',*r['CM_rows_D_E_columns_D_E_U'][1]]]) for r in numbers)
materials = table(['유형','모델','D 유지','외부 E 응답','D precision','E 응답 precision','보류율'], [
    [r['group'], r['model'], f"{r['D_pred_D']}/{r['human_D_N']} ({pct(r['D_recall'])})",
     f"{r['E_pred_hidden']}/{r['human_external_E_N']} ({pct(r['external_E_response_recall'])})",
     pct(r['D_precision']),pct(r['external_E_response_precision']),pct(r['unresolved_rate'])] for r in material])
near = table(['모델','기존 기준점과의 차이≤3px인 보임376개 유지'],[
    [r['model'],f"{r['near376_visible']}/376 ({pct(r['near376_recall'])})"] for r in numbers])
cases = [(256,'중심은 빨간 콘이며 새3모델 모두 F. 기존 외부가림 주석과 같은 방향의 응답입니다.'),
         (840,'중심은 짙은 녹색 팔레트 면처럼 보이고 콘은 위쪽입니다. 새3모델 모두 F여서 목표 차이만으로 모든 오류를 설명할 수 없습니다.'),
         (401,'저조도 목재 모서리로 중심 소속을 시각적으로 확정하기 어렵습니다. C47 BG, C48/C49 F이며 명확한 외부 물체는 보이지 않습니다.'),
         (300,'중심은 회색 배경과 목재 경계 부근이고 목재는 아래쪽에 보입니다. C48/C49 T. 기존 projected 기준점 차이103.57px여서 native 대응도 미확정입니다.')]
case_html=''.join(f'<h3>고정 사례 g{i}</h3><p>{html.escape(text)}</p><img loading="lazy" src="{V / "fixed19_current_crop_review_v1" / f"CASE_{i:04d}_CARD.png"}">' for i,text in cases)
refs = ' · '.join(link(path,label) for path,label in [
    (out,'기계 판독 결과'),(E/'CURRENT_FROZEN_BENCHMARK_RESULTS.json','전체 지표·self·좌표오차'),
    (E/'ALL1784_NATIVE_LEDGER.csv','전1784행 ledger'),(audit,'독립 수치 감사'),
    (gallery,'같은 고정19 crop'),(gallery_qa,'19개 실제 시각 QA'),(public,'별도의 공개 실사 중심소속 평가')])
doc=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V178 동결 사용자 평가</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;max-width:1280px;margin:36px auto;padding:0 24px;color:#182638}}h1{{font-size:30px}}table{{border-collapse:collapse;margin:18px 0;width:100%;font-size:15px}}th,td{{border:1px solid #ccd3da;padding:9px;text-align:left}}th{{background:#eef3f6}}img{{max-width:100%;height:auto}}.note{{background:#fff2df;padding:18px}}a{{color:#155c91}}</style>
<h1>초기 query 소속 모델: 사용자 평가에서는 아직 실패</h1>
<p>동결한 C47·C48·C49 전부를 기존 v61·legacy와 같은 평가 대상에서 비교했습니다. 현재 데이터로 학습·문턱·normalizer·모델/arm/data/architecture 선택을 하지 않았습니다. q/L/native 순서와 기존 입력 보류도 유지했습니다.</p>
<p class="note">주 비교1237개: 보임 D1147 · 외부가림 E90. C48은 D181개(15.8%), C49는193개(16.8%)만 유지했습니다. v61은260개(22.7%), legacy는655개(57.1%)입니다. 새 모델은 채택하지 않습니다.</p>
<p>출력 T를 D 응답, F/BG를 E 응답으로 묶은 비교 진단입니다. 타물체/배경 소속은 실제 native 외부가림 원인을 인증하지 않습니다. 외부 E 응답 recall이 높아도 보이는 점을 대부분 E로 보내면 유용한 가림 판별기가 아닙니다.</p>
{summary}<img src="{plot}"><h2>같은 분모의 혼동행렬</h2>{matrices}
<h2>목재와 정사각형</h2>{materials}<h2>초기 좌표 차이가 작은 기존 보임 점</h2>{near}
<p>위376개는 기존 수동 클릭 기준과의 차이가≤3px인 유효 보임 점입니다. 물리 native 좌표 정답의 완전 인증이 아닙니다. 전체 주 비교를 이 부분집합으로 제한하지 않았습니다.</p>
<h2>무엇이 섞여 있었나</h2><p>중심 픽셀 소속, crop 안의 팔레트 존재, 원 native 키포인트의 보임은 서로 다른 질문입니다. 공개 평가에서는 중심 BG84개 중78개의 crop에 팔레트가 보였습니다. 이 차이와 함께, g840처럼 팔레트 면에 F를 예측하는 남은 인식 실패도 관찰했습니다. 고정19 예시만으로 각 원인의 전체 비율이나 조명·특징 해상도·색상에 대한 인과 원인을 확정하지 않습니다.</p>
<p>전체223장/1784슬롯 중 입력 가능1739/원 보류45를 보존했습니다. native 자기 가림은 기존 별도 집계로 남겼고 upstream 기하 필터를 바꾸지 않았습니다. native D/E/S/UNKNOWN×owner T/F/BG/U 및 기존 좌표오차 구간 교차표는 전체 결과에 있습니다.</p>
<h2>원 픽셀을 유지한 고정 사례</h2><p>과거 평가에서 선택한 같은19개를 재사용했습니다. B는19개, root는아래4개와 과학 그림을 실제 확인했습니다. 대표 표본·새 human/native/pixel gold·학습 편입이 아닙니다.</p>{case_html}
<h2>기록과 한계</h2><p>{refs}</p><p>legacy는 과거 실사 CAL 학습과 반복 평가 노출 이력이 있습니다. 현재223도 새 블라인드 시험이 아닙니다. 기존 baseline 응답1784개와 지표는 그대로 재현됐으며, 독립 감사는 새5352예측과 전체549표행을 검산했습니다. 공개 실사의 약한 중심소속 개선을 이 native 보임 지표와 같은 성능으로 해석하지 않습니다.</p>
<p>새 학습0, encoder/초기 추론 재실행0, PnP0, q/L 수정0. HTML은 소스·파일 링크 검사를 수행하며 브라우저 렌더 검증은 하지 않았습니다. 전체 목표는 미해결이고 selected_model=null, adopted=false입니다.</p></html>'''
with (V/'ROOT_READOUT_KO.html').open('x') as f:
    f.write(doc)
print(json.dumps({'results':bind(out),'readout':bind(V/'ROOT_READOUT_KO.html')},ensure_ascii=False))
