"""Assemble immutable V148 source-only root readout from completed saved results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,html
H=Path(__file__).resolve().parent;V=H.parent
RUN=V/'runs/C33_C34_PUBLIC_NATIVE_FIXED2000'
ARMS=['C33_PUBLIC_NATIVE_WEIGHT0_FIXED2000','C34_PUBLIC_NATIVE_WEIGHT1_FIXED2000']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
for n in('ROOT_RESULTS_KO.json','ROOT_READOUT_KO.html','ROOT_READOUT_QA.json'):assert not(V/n).exists()
controlp=RUN/'CONTROL_COMPLETE.json';control=read(controlp);assert control['completed']and control['PASS']and control['selected_model']is None
auditp=V/'independent_saved_review/REVIEW.json';assert sha(auditp)=='3d9a0ede32c5009b904e786127a3d25f840196c1e8010fe444b8fdd30effd8ad'
audit=read(auditp);assert audit['PASS']and audit['control_complete']==bind(controlp)
diagnosis=read(H/'SAVED_DIAGNOSIS_COMPLETE.json');qa=read(H/'QA_AND_DIAGNOSIS_COMPLETE.json');assert diagnosis['PASS']and qa['PASS']
source_results={};freezes={}
for arm in ARMS:
 fit=read(check(control['arms'][arm]['fit']));assert fit['completed']and fit['optimizer_updates']==2000
 source_results[arm]=dict(binding=fit['source_results'],data=read(check(fit['source_results'])))
 freezes[arm]=dict(binding=fit['freeze'],data=read(check(fit['freeze'])))
 assert freezes[arm]['data']['current_allowed']is False and freezes[arm]['data']['selected_model']is None
policy=read(V/'ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json')
first=read(V/'ROOT_FIRST_PREFLIGHT_FAILURE_DIAGNOSIS.json');assert first['terminal_exit_code']==1 and first['actual_fit_updates']==0 and first['classifier_model_calls']==0
m=diagnosis['metrics'];pairs=read(check(diagnosis['all4_pair_results']))
result=dict(completed=True,phase='V148',created_UTC=datetime.now(timezone.utc).isoformat(),goal_status='active',fully_solved=False,source_only=True,
 title_KO='공개 실사 하부 역할 학습: 보이는 점을 보존하는 효과는 확인, 전체 목표는 미달',
 conclusion_KO='공개 하부 역할의 약한 지도신호를 더한 C34는 같은 자료군의 HALL에서 보이는 D를 가림으로 잘못 판단하던4개를 회복했습니다. 외부 가림 E6개는 초기부터 모두 정답이어서 이번 새 이득은 D 보존입니다. 다양한 팔레트·가림으로의 전이나 현재 평가셋 성공은 이 source 단계에서 확인하지 않았습니다.',
 successful_execution=dict(session=25096,terminal_exit_code=0,updates_each=2000,arms=ARMS,control=bind(controlp)),
 first_attempt_preserved=dict(session=22384,terminal_exit_code=1,fit_updates=0,classifier_calls=0,reason='학습 전 global_index의 int32/int64 byte 비교 guard 중단. 값·q/L은 동일했으며 명시적 guard 수리 후 별도 실행.',diagnosis=bind(V/'ROOT_FIRST_PREFLIGHT_FAILURE_DIAGNOSIS.json')),
 comparison=dict(same_C32_start=True,model_parameters=224513,geometry_allzero=True,threshold_logit0=True,same_fixed_native_prefix_draws=True,
  public_coefficient=dict(C33=0,C34=1),only_objective_intervention='공개 TRAIN43(D35/E8) loss 계수0 vs1. 같은 원 q/L/crop과 native loss·2000 update·optimizer·seed.',
  HALL_gradient_rows=0,old_AUX_loss_body_reads=0,context='C16 mixed-AUX 이력이 있는 C32에서 시작한 matched fine-tune이며 from-scratch native-only 학습이 아닙니다.'),
 public_inventory=dict(frames=64,slots=512,D=70,E=14,U=428,known_DE=84,TRAIN_known43=dict(D=35,E=8),HALL_known41=dict(D=35,E=6),
  HALL_input_valid39=dict(D=33,E=6),HALL_teacherD_original_inputU=[402,434],original_input_valid=463,original_input_OOB=49),
 public_metrics=m,HALL_available_D_correct=dict(C32_INITIAL=29,C33_CONTROL=28,C34_PUBLIC=33),HALL_E_correct_all_models=6,
 public_ALL_E_correct_all_models=14,initial_D_to_E_errors=6,C34_recovered_initial_errors=6,
 same_provided_T_K_four_pairs_both_correct=pairs['both_correct_pair_counts'],same_T_K_is_not_same_actual_query_or_isolated_image_only_intervention=True,
 native35=dict(C33_passed=22,C34_passed=22,total=35,all_pass=False,AUX29_not_retested=True,original64_qualification_not_claimed=True),
 source_model_admission=dict(selected_model=None,current_allowed=False,adopted=False,source_qualification_failed=True),
 limitations_KO=['공개 자료의 동일 목재 통합 조립체·하부2/3·종이상자 가림에 한정한 assistant weak 라벨입니다. 사람 gold나 전체8점 정답이 아닙니다.',
 'UP=-OBJY는 조건부 정합이며 scanAABB의34.5mm X outlier 한계를 유지했습니다. 가상 역할 참조 offset은 정밀 physical GT 오차가 아닙니다.',
 'LAB/HALL 원 분할은 유지했지만 같은 개발 노출 자료군입니다. 이 작은 HALL 결과만으로 다양한 실사 전이를 주장할 수 없습니다.',
 '전체84에는 학습 LAB43이 포함됩니다. HALL 전체41과 입력 유효39는 별도 분모입니다. teacherD2개의 입력 보류는 남아 있습니다.',
 '원512슬롯·q/L·첫 선택·보류 flags를 보존했고 좌표 오차로 학습행을 골라내거나 라벨을 바꾸지 않았습니다.'],
 evidence=dict(policy=bind(V/'ROOT_PUBLIC_NATIVE_SUPERVISION_POLICY_BEFORE_FIT.json'),independent_saved_audit=bind(auditp),
  source_results={a:source_results[a]['binding']for a in ARMS},source_freezes={a:freezes[a]['binding']for a in ARMS},
  saved_diagnosis=bind(H/'SAVED_DIAGNOSIS_COMPLETE.json'),visual_QA=bind(H/'QA_AND_DIAGNOSIS_COMPLETE.json'),
  root_visual_QA=bind(V/'ROOT_SAVED_SCOREBOARD_VISUAL_QA.json'),scoreboards=qa['actually_viewed_board_PNGs']),
 new_model_calls_in_readout=0,new_private_current223_V112_or_weight_feature_reads_in_readout=0,parent_pointers_or_historical_phase_files_changed=False)
out=V/'ROOT_RESULTS_KO.json';write(out,result)
names=[('C32_INITIAL','C32 시작'),('C33_CONTROL','C33 대조 · 공개 계수0'),('C34_PUBLIC','C34 공개 · 공개 계수1')]
def cmrow(name,label):
 q=m['ALL'][name];cm=q['CM_D_E_rows_D_E_U_columns'];return f'<tr><th>{html.escape(label)}</th><td>{cm[0][0]} / {cm[0][1]} / {cm[0][2]}</td><td>{cm[1][0]} / {cm[1][1]} / {cm[1][2]}</td><td>{q["correct"]} / 84</td></tr>'
def hallrow(name,label):
 q=m['VAL_HALL'][name];cm=q['CM_D_E_rows_D_E_U_columns'];return f'<tr><th>{html.escape(label)}</th><td>{cm[0][0]} / 35</td><td>{cm[0][0]} / 33</td><td>{cm[1][1]} / 6</td><td>{q["U"]}</td></tr>'
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V148 공개 source 하부 가림 학습 결과</title>
<style>body{margin:0;background:#f4f6fa;color:#233044;font:17px/1.7 "Noto Sans CJK KR","NanumGothic",sans-serif}main{max-width:1280px;margin:auto;padding:30px 24px 60px}h1{font-size:34px;line-height:1.4;margin:8px 0 18px}h2{font-size:25px;margin-top:32px}p{margin:12px 0}.tag{color:#53677f;font-size:15px}.lead{background:#e8f0f8;border-left:5px solid #436a90;padding:20px 24px;font-size:21px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}.card{padding:18px 22px;background:white;border:1px solid #dbe2eb;border-radius:12px}.big{font-size:32px;font-weight:700}.muted{color:#617082;font-size:15px}table{width:100%;border-collapse:collapse;background:white;margin:16px 0}th,td{padding:13px;border:1px solid #dbe2eb;text-align:center}th:first-child{text-align:left}.note{background:#fff3dc;border-left:4px solid #c49a49;padding:14px 20px}img{width:100%;height:auto;background:white;border:1px solid #dbe2eb;border-radius:10px}a{color:#285c91}details{background:white;padding:14px 20px;border:1px solid #dbe2eb;margin-top:24px}summary{cursor:pointer;font-weight:700}code{font-size:14px}footer{margin-top:30px;color:#617082}.scroll{overflow:auto}@media(max-width:700px){.cards{grid-template-columns:1fr}h1{font-size:27px}main{padding:18px 12px}.lead{font-size:18px}th,td{padding:8px;font-size:14px}}</style><main>
<div class="tag">V148 · 완료된 저장 source 결과 · final2000 / raw logit0 고정</div>
<h1>공개 실사에서 보이는 하부 보존은 개선됐습니다.<br>전체 목표는 아직 달성하지 못했습니다.</h1>
<p class="lead">공개 역할 라벨을 추가한 C34는 HALL의 입력 가능한 보임 D를 <b>33/33</b> 맞혔습니다. 시작 모델은29/33, 같은 업데이트를 더한 대조 모델은28/33입니다. 외부 가림 E6개는 세 모델 모두 처음부터 정답이어서, 이번 새 이득은 <b>보이는 D 보존</b>입니다.</p>
<div class="cards"><div class="card">HALL 보임 D · 입력 유효<br><span class="big">29 → 33 / 33</span><br><span class="muted">C32 시작 → C34 공개</span></div><div class="card">HALL 외부 가림 E<br><span class="big">6 → 6 / 6</span><br><span class="muted">초기부터 모두 정답</span></div><div class="card">기존 native 조건 통과<br><span class="big">22 / 35</span><br><span class="muted">C33·C34 모두 전체 기준 미달</span></div></div>
<h2>같은 조건에서 공개 지도신호만 더했습니다</h2>
<p>C33과 C34는 같은 C32 가중치에서 시작해 같은 native 표본 순서·optimizer·seed로 각2000번 갱신했습니다. 차이는 공개 TRAIN43개(D35/E8)의 loss 계수0과1입니다. HALL은 학습 gradient에 사용하지 않았고, 원 q/L·첫 선택·crop·보류 flags를 유지했습니다.</p>
<h2>HALL: 전체41개와 입력 유효39개를 구분합니다</h2>
<p>HALL의 약한 정답은 D35/E6입니다. D2개는 원 q가 영상 밖이라 모델 입력 보류 U로 남습니다. 입력 유효39개는 D33/E6입니다.</p>
<div class="scroll"><table><thead><tr><th>모델</th><th>D 정답 / 전체35</th><th>D 정답 / 입력 유효33</th><th>E 정답 /6</th><th>원 입력 보류</th></tr></thead><tbody>'''+''.join(hallrow(n,l)for n,l in names)+'''</tbody></table></div>
<h2>공개 known84개 전체의 저장 출력</h2>
<p>전체64장·512입력 중 약한 D70/E14만 아래 정확도 분모에 들어갑니다. 의미 미확정 U428은 정오 집계에서 제외하며 원행은 유지했습니다. 이84개에는 <b>학습에 사용한 LAB43개</b>도 포함됩니다.</p>
<div class="scroll"><table><thead><tr><th>모델</th><th>실제 D → 예측 D / E / U</th><th>실제 E → 예측 D / E / U</th><th>정답 /84</th></tr></thead><tbody>'''+''.join(cmrow(n,l)for n,l in names)+'''</tbody></table></div>
<p>초기 C32가 D를 E로 잘못 판단한6개는 C34에서 모두 회복했습니다. 같은 sequence에서 제공 T/K 값이 bitwise 동일한 D↔E 전환4쌍의 양끝 정답은3쌍에서4쌍으로 늘었습니다. 이 비교에서도 원래 q는 서로 다르며, image-only 고정-query 실험을 뜻하지 않습니다.</p>
<h2>같은 포즈 계열의 보임·외부 가림 대조</h2>
<p>원 저장96 crop과 초기 C32·C33·C34 raw score를 함께 표시했습니다. 원 q/L로 얻은 픽셀은 수정하지 않았습니다.</p>
<a href="source_visual_diagnosis/BOARD_01_SAVED_SCORE_VISIBLE_EXTERNAL_PAIRS.png"><img src="source_visual_diagnosis/BOARD_01_SAVED_SCORE_VISIBLE_EXTERNAL_PAIRS.png" alt="LAB12와15, HALL37과33의 보임 외부가림 대조와 세 모델 저장 점수"></a>
<h2>개체 대응·입력 보류·역할 보임은 따로 봅니다</h2>
<p>다른 물체를 고른02와 대응이 모호한57은 의미 U라서 모델의 D/E 응답을 정답으로 세지 않습니다. 50은 teacher D이지만 원 입력 OOB라 최종 U입니다. 08은 원 crop이 바닥을 보더라도 같은 조립체의 기대 하부 역할은 보이는 사례입니다.</p>
<a href="source_visual_diagnosis/BOARD_02_SAVED_SCORE_ENTITY_INPUT_ROLE.png"><img src="source_visual_diagnosis/BOARD_02_SAVED_SCORE_ENTITY_INPUT_ROLE.png" alt="다른 개체 U, 입력 OOB D, 큰 좌표 어긋남 D, 대응 모호 U와 세 모델 저장 점수"></a>
<h2>확인한 효과의 범위</h2>
<p class="note">같은 목재 통합 조립체·종이상자·하부2/3의 구조적 약한 라벨에 한정된 결과입니다. 사람 gold, 전체8점 정답, 다양한 실사 전이 성공을 뜻하지 않습니다. native35는 두 arm22/35로 미달이며 모델 채택은 없습니다.</p>
<p>조건부 축 정합과 scanAABB의34.5mm X outlier 한계는 유지했습니다. 가상 역할 참조까지의 거리는 정밀 physical GT 오차가 아니며 학습 필터나 재라벨링에 사용하지 않았습니다. LAB/HALL은 원 분할을 유지한 개발 노출 공개 자료군입니다. 이 단계는 현재 평가223장에 대한 성공을 주장하지 않습니다.</p>
<details><summary>실행·독립 검산·원본 결과</summary><p>성공 실행25096은 EXIT0, 두 arm 각2000 update입니다. 첫 실행22384는 학습 전 integer dtype byte 비교 guard에서 EXIT1이었고 update0/model call0인 실패 기록을 보존했습니다. 명시적으로 guard를 수리한 별도 실행이며 과거 기록을 덮어쓰지 않았습니다.</p><p>독립 감사는 원 ID/qL/보류·초기 parity·동일2000 draw·공개 계수0 gradient0·loss/exposure·최종 parameter/optimizer·source 지표를 검산했습니다. <code>selected_model=null</code>, <code>current_allowed=false</code>를 유지합니다.</p>
<p><a href="ROOT_RESULTS_KO.json">결과 JSON</a> · <a href="independent_saved_review/REVIEW.json">독립 저장 결과 감사</a> · <a href="source_visual_diagnosis/SAVED_DIAGNOSIS_COMPLETE.json">84개 paired 진단</a> · <a href="source_visual_diagnosis/INITIAL_C32_VISIBLE_D_PREDICTED_E_ALL6.json">초기 오류6개 전수</a> · <a href="source_visual_diagnosis/ALL4_SAME_PROVIDED_T_K_DE_PAIR_RESULTS.json">동일 제공 T/K4쌍</a> · <a href="ROOT_FIRST_PREFLIGHT_FAILURE_DIAGNOSIS.json">첫 guard 실패 기록</a></p></details>
<footer>원64장·고유512입력 보존 · source-only 저장 결과 · 이 readout 작성 중 모델 호출/비공개 자료 접근0</footer></main></html>'''
with(V/'ROOT_READOUT_KO.html').open('x')as f:f.write(body)
print(json.dumps(dict(results=bind(out),readout=bind(V/'ROOT_READOUT_KO.html'))))
