"""Saved C17 source result and diagnosis; no new model/current work."""
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
V=Path(__file__).resolve().parent
P=V.parent
R=V/'runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rel(p):return html.escape(os.path.relpath(p,V),quote=True)
def image(p,caption):return '<figure><a href="'+rel(p)+'"><img src="'+rel(p)+'" alt="'+html.escape(caption)+'"></a><figcaption>'+html.escape(caption)+'</figcaption></figure>'
fit=read(R/'FIT_COMPLETE.json');wide=read(R/'WIDE_SOURCE_RESULTS.json');gates=read(R/'SOURCE_GATE_RESULTS.json')
comparison=read(V/'saved_boundary_diagnosis/C16_C17_BOUNDARY_COMPARISON.json')
assert fit['completed'] and fit['optimizer_updates']==1000 and fit['selected_update']==900
assert gates['all_gates_pass'] and not gates['all_wide_checks_pass']
baseline_review=read(V/'saved_baseline_review/REVIEW.json')
fit_review=read(V/'saved_fit_review/REVIEW.json')
assert baseline_review['PASS'] and fit_review['PASS']
limits=[
 '이번 수치는 공개 실사 배경에 물체 cutout을 배치한 조건부 weak cue 실험이다. 현재223 실사·정확한 native 키포인트 gold 성능이 아니다.',
 '4896행은9장 원본·32개 cue의 상관 파생 자료다. 새 원본 다양성은 늘지 않았으며 추가 학습용은3장·6cue의795known available행이다.',
 '원 cue uncertainty·합성물·donor·원본 fold는 유지했다. q offset은각축±0.25L0의공개emulation이며 실제 초기 추론 좌표를 교정한 것이 아니다.',
 '교사 U468과 입력 U51(D27/E24)은 별개다. 입력 U는 전체 계획 평가 분모에 보존하고 학습 손실에서 제외했다.',
 'V95에도 q alpha와 semantic cue label 분리 사례가 있었다. 이번 기여는 같은 장면의 고정 q 격자로 의존성과 학습 효과를 대조한 것이다.',
 '새 공개 원본6장은후보검토만했다. take5에목재구조가있고전체적으로흐림·작은대상·대상불명확성이있어플라스틱다양성확보로해석하지않는다.',
 '초기 기하 self 경로는 별도이며 S/U를 D/E학습에 넣지 않았다. 현재223/V112/PnP/좌표교정 호출은0이다.',
]
next_actions=[
 '현재 실사 재평가로 새 모델을 고르지 않는다. C17은 공개 경계 검증에서도 보임·가림을 함께 만족하지 못해 후속 원인 분리가 필요하다.',
 '기존 학습 목표의 출처를 확인한 뒤, 동일 cached-feature/구조에서 새 semantic 경계 loss만 학습하는 공개 자료 진단으로 기존 loss의 간섭과 특징·head 한계를 분리한다. 이를 배포 모델이나 목표 해결로 대체하지 않는다.',
 '그 결과를 바탕으로 기대 구조와 주변 재질의 공간 관계를 유지하는 입력/학습 또는 실제 공개 원본 cue 다양성을 보강한다. 현재 평가 라벨로 설정을 선택하지 않는다.',
]
evidence={k:bind(p)for k,p in {
 'corpus':V/'actual_corpus/CORPUS_COMPLETE.json','geometry':V/'source_geometry_feasibility/NORMALIZED_QUARTER_L0_FEASIBILITY.json',
 'label_contract':V/'label_contract_review/REVIEW.json','corpus_visual_review':V/'independent_corpus_visual_review/REVIEW.json',
 'training_contract':V/'training_contract_review/REVIEW.json','baseline':V/'frozen_C16_source_baseline/PREDICTION_SEAL.json',
 'baseline_review':V/'saved_baseline_review/REVIEW.json','fit':R/'FIT_COMPLETE.json','freeze':R/'MODEL_FREEZE.json',
 'fit_review':V/'saved_fit_review/REVIEW.json','comparison':V/'saved_boundary_diagnosis/C16_C17_BOUNDARY_COMPARISON.json',
 'public_input_intake':V/'public_loco_take_intake/ACTUAL_VISUAL_OBSERVATIONS.json',
}.items()}
result=dict(completed=True,created_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active',fully_solved=False,
 phase='C17 controlled boundary-query AUX fit complete; public validation tradeoff remains; no new current evaluation',
 outcome_KO='새 경계 조건은 일부 개선했지만 공개 검증 전체 보임·가림은 함께 개선하지 못했다. 현재 실사는 재평가하지 않았으며 최신 실사 수치는C16그대로다.',
 actual_work=dict(public_occurrences=4896,unique_feature_units=4428,centre_exact_inputs_and_features=544,encoder_primary_calls=554,encoder_repeat_calls=1,frozen_head_calls=2,optimizer_updates=1000,selected_update=900,threshold_logit=0,old_regression_gates_pass=15,wide_diagnostics_pass=sum(x['PASS']for x in wide['prospective_checks'].values()),wide_diagnostics_total=10,current223_calls=0,V112_calls=0,PnP=0),
 public_validation_comparison=comparison['models'],wide_checks=wide['prospective_checks'],
 interpretation_KO=[
 '같은 semantic cue/합성물에서 q를 움직이면 C16의 판정은 q alpha 상태에 강하게 의존하는 패턴을 보였다. 모델이 오직 한 픽셀만 사용한다는 내부 기제 증명은 아니다.',
 '이 반례를 추가한 C17에서 E-q밖과SHAM D-q안 정답은 늘었다. 그러나 다른 보임과 가림 조건의 손실이 커서 전체 성능은 개선하지 못했다.',
 '기존15회귀 PASS만으로 해결됐다고 볼 수 없다. 새로 드러낸10개 진단 중8개는 여전히 FAIL이다.',
 '목표가 과도하다는 결론은 아니다. 데이터 추가만으로 목표 구조의 가시성을 안정적으로 읽게 만들 수 있는지는 미해결이며, 기존 loss와 입력 표현의 영향을 나눠야 한다.'
 ],limits_KO=limits,next_action_KO=next_actions,evidence=evidence)
save(V/'ROOT_RESULTS_KO.json',result)
rows=[]
names={'E_qOUT':'기준 단서는 가림 · q는 물체 밖','SHAM_D_qIN':'기준 단서는 보임 · q는 물체 안','ALL_D':'전체 보임','ALL_E':'전체 가림'}
for key,label in names.items():
 a=comparison['models']['C16']['VALIDATION_COMBINED'][key];b=comparison['models']['C17']['VALIDATION_COMBINED'][key]
 rows.append(f'<tr><td>{label}</td><td>{a["correct"]}/{a["available"]} ({100*a["recall_available"]:.1f}%)</td><td>{b["correct"]}/{b["available"]} ({100*b["recall_available"]:.1f}%)</td></tr>')
doc=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C17 공개 경계 실험</title>',
 '<style>body{font:17px/1.65 system-ui,sans-serif;background:#edf3f6;color:#183044;margin:0}main{max-width:1240px;margin:auto;padding:28px}section{padding:24px;background:white;border-radius:10px;margin:22px 0}h1{line-height:1.3}h2{margin-top:0}img{width:100%;height:auto;display:block}figure{margin:20px 0}figcaption{font-size:14px;color:#526b7d}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #d2dee6;padding:10px;text-align:left}a{color:#155e8e}li{margin:10px 0}.lead{font-size:22px;font-weight:650}.note{background:#fff3da;padding:15px;border-left:5px solid #d28c35}summary{cursor:pointer;font-weight:bold}</style><main>',
 '<h1>같은 가림 장면에서 초기 점만 옮겨본 결과</h1><p class="lead">경계 조건은 일부 개선했지만, 보임과 가림을 함께 안정적으로 맞히지는 못했다.</p>',
 '<p>공개 source에서4,896행을 만들고, frozen C16 기준선과 경계 반례를 추가 학습한 C17을 비교했다. 현재223 실사는 이번 실험에 사용하지 않았다.</p>',
 '<section><h2>공개 검증 결과 · TRAIN 제외</h2>',
 image(V/'saved_boundary_diagnosis/C16_C17_PUBLIC_VALIDATION_COMPARISON.png','LOCO VAL + 한 TOMIE 개발 episode의 상관 파생 사례. 현재 실사 성능이 아니다.'),
 '<table><thead><tr><th>조건</th><th>C16</th><th>C17</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>',
 '<p>전체 가림 정답도1,204→1,167/1,674로 줄었다. 특정 경계 조건의 향상을 전체 개선으로 해석하지 않는다. 문턱은raw logit0으로 유지했다.</p></section>',
 '<section><h2>어떤 반례를 추가했는가</h2><p>같은 기준 cue를 덮는 COVER와 cue를 보존하는 SHAM은 그대로 두고 q를각축±0.25L0 움직였다. 원 uncertainty를 줄이지 않았으며 전체가 crop안에 남는 것을 검산했다. 노란+는q, 하늘원은기존weakreference다. 표시는 설명용 복사본이며 모델 입력은 무표시다.</p>',
 image(V/'independent_corpus_visual_review/FIXED6_MARKED_6x3.png','고정6개 조건·5개서로다른triplet. 같은TOMIEtriplet이두조건에해당하므로중복을유지했다.'),
 '<p>q가 가림 물체 밖이어도기준cue는가려질수있고, q가물체안이어도기준cue는보일수있다. alpha밖이 곧target pixel이라는뜻은아니다.</p></section>',
 '<section><h2>실제 실행과 확인</h2><p>기존 중앙544행의crop·valid·q/L/context·1928차원특징·C16점수가정확히일치했다. C17은기존작은head221,825개파라미터만1,000회학습했고공개18항목BCE로900회checkpoint를동결했다. 기존15회귀는PASS이나새10진단중8개는FAIL이다.</p><p class="note">현재 실사 최신 결과는C16의보임1,137/1,147·외부가림35/90그대로다. 새C17의실사성능을측정하거나기존모델보다좋아졌다고주장하지않는다.</p>',
 '<p><a href="'+rel(P/'query_visibility_training_duration_control_20261023_v124/ROOT_READOUT_KO.html')+'">C16 실제 성공·실패 crop과전체90개평가 보기</a></p></section>',
 '<section><h2>별도로 확인한 공개 원본 후보</h2>',
 image(V/'public_loco_take_intake/CONTACT_SIX_TAKES.png','LOCO 공식TRAIN의6take에서모델점수없이메타데이터로고른6장. 이번C17학습에는사용하지않았다.'),
 '<p>일부 목재 구조는보이지만흐림·작은대상·랙/용기구조와의혼동이있다. 플라스틱팔레트의원본다양성을확보했다고할수없다.</p></section>',
 '<section><h2>다음 원인 분리</h2><ol>'+''.join('<li>'+html.escape(s)+'</li>'for s in next_actions)+'</ol><p>새경계자료로어떤오류가바뀌는지는확인했다. 기존목표와loss의간섭, 작은crop/특징에서기대부품을식별하는한계는더검증해야한다. 아직목표해결상태가아니다.</p></section>',
 '<section><h2>범위와 저장 근거</h2><ul>'+''.join('<li>'+html.escape(s)+'</li>'for s in limits)+'</ul><p><a href="ROOT_RESULTS_KO.json">결과JSON</a> · <a href="'+rel(V/'saved_fit_review/REVIEW.json')+'">독립저장결과검산</a></p></section></main></html>']
(V/'ROOT_READOUT_KO.html').write_text('\n'.join(doc))
print(json.dumps(dict(report=str(V/'ROOT_READOUT_KO.html'),results=str(V/'ROOT_RESULTS_KO.json'),fully_solved=False),ensure_ascii=False))
