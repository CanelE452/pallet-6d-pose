from pathlib import Path
from datetime import datetime, timezone
import json, csv, hashlib, html

V=Path(__file__).resolve().parent
P=V.parent
RUN=V/'runs/C15_PUBLIC_FLAT_PALLET_AUGMENTED_SPATIAL_QUERY'
EV=V/'private_evaluation/C15_FROZEN_CURRENT223/evaluation'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,x):
    with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def csvrows(p):
    with Path(p).open()as f:return list(csv.DictReader(f))

fit=read(RUN/'FIT_COMPLETE.json')
source=read(RUN/'NEW_FLAT_SOURCE_RESULTS.json')
results=read(EV/'CURRENT_PRIMARY_RESULTS.json')
assert fit['completed'] and results['completed'] and fit['selected_update']==1000
allcm=csvrows(EV/'D_VS_EXTERNAL_E_2X3.csv')
models=['v61','legacy','V111_C11_RAW','V119_C14_RAW','V123_C15_RAW']
names=dict(zip(models,['v61','기존 실사 모델','C11 (이전 실험)','C14','C15 (이번 source 선택)']))
primary=[r for r in allcm if r['cohort']=='paired_same_query_RGB' and r['model']in models]
assert len(primary)==10
near=[r for r in csvrows(EV/'MANUAL_NEAR_D.csv')if r['model']=='V123_C15_RAW']
selfrows=[r for r in csvrows(EV/'HUMAN_SELF_BINARY_RESPONSE.csv')if r['model'].startswith('V123_C15')]
fixed=read(V/'visual_followup/C15_FIXED4_MAPPING_AND_METRICS.json')
out={'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'goal_status':'active','fully_solved':False,
 'phase':'V123_C15_SOURCE_ONLY_HEAD_TRAINED_AND_FROZEN_CURRENT_EVALUATION_COMPLETE',
 'outcome_KO':'C15는 공개 source 추가 학습 후 실사 외부가림15/90→35/90, 보임1130/1147 유지. 큰 콘 중심 고정사례1021은 가림으로 수정됐지만 경계325는 여전히 보임 오판. 외부가림55/90 미검출로 목표 미해결.',
 'actual_work':{'optimizer_updates':1000,'source_selected_update':1000,'head_parameters':221825,'source_feature_units':19432,'source_occurrences':51004,'new_source_known_TRAIN':846,'new_TRAIN_cues':6,'new_TOMIE_TRAIN':0,'encoder_calls':0,'current_head_calls':2,'current_slots':1784,'current_factual_U':45,'current_extra_U':0,'current_fit_normalizer_threshold_checkpoint_arm_selection':0,'PnP':0},
 'warmstart_parity':read(RUN/'UPDATE0_WARMSTART_PARITY.json'),
 'source_14_BCE':{'initial':0.10109466223706463,'selected':fit['criterion_fourteen_VAL_cohorts_BCE']},
 'new_public_source_metrics':source['metrics'],'new_public_source_pairs':source['paired_by_rank'],
 'old15_source_regression_gates':read(RUN/'SOURCE_GATE_RESULTS.json'),
 'new_source_descriptive_targets':read(V/'NEW_SOURCE_DESCRIPTIVE_TARGETS.json'),
 'current_primary_comparison':primary,'manual_near_D':near,'human_self_binary_response':selfrows,
 'fixed4_saved_comparison':fixed['cases'],
 'interpretation_KO':['공개 source 학습 변경이 실제 외부가림20개 추가 검출로 일부 전이됐다. 단순히 합성만 더 잘 맞힌 결과는 아니다.','같은 보임 정답 수가 유지되어 이번 향상이 전체 보임을 더 많이 버린 결과로 나타나지는 않았다. 개별 전환은 저장표에 남긴다.','C15외부가림 합계35/90은 과거C11합계35/90과 같다. 목재/정사각형 분포는 다르며 이번 모델을 전체 실험 최상으로 선택하지 않았다.','기존 실사 모델은 외부가림83/90이나 보임655/1147·보류177/1237의 다른 절충을 보인다. reliable external visibility 해결이라고 할 수 없다.','공개 TOMIE 중앙 가림 누락16개는 목재 블록 조건에 남았다. 데이터 형태/촬영조건과 기대 단서의 구조를 일반화하는 문제가 여전히 있다.'],
 'limits_KO':['현재223은 평가에만 사용했지만 과거 개발에 노출된 자료이며 fresh blind test가 아니다.','새4896행은32개의 약한 구조 단서·9장 원본에서 만든 상관된 파생 사례다. 새 학습용은6cue/3LOCO원본의846known행이다.','LOCO는 기존source와같은fold원본중복, TOMIE4장은기존학습원본과겹침없으나하나의episode다.','공개 합성 D/E는 조건부 AUX이며 정확한 native 키포인트·사람·pixel-ownership gold가 아니다.','9개 공개 offset은각축±3px이고대각4.24px이며실제초기q를교정한결과가아니다.','S/U는학습손실에서제외하며self는기존기하경로. visibility와좌표정답인증은별개다.','학습1000회끝공개검증BCE가계속감소했지만기간연장·구조변경효과는아직검증하지않았다.'],
 'saved_evaluation_schema_repair':{'failure':bind(V/'ROOT_FAILED_SAVED_EVALUATION_SCHEMA_V1.json'),'repair_entry':bind(V/'ROOT_REPAIRED_C15_SAVED_CURRENT_EVALUATION_ENTRY.json'),'model_or_inference_reruns':0,'note_KO':'C14 최소 CSV의 geometry 연결 오류만 저장값 전용 adapter로 고쳤다. 원 C15 추론·code·출력은 보존했다.'},
 'evidence':{'design':bind(V/'ROOT_C15_DESIGN_BEFORE_SOURCE_FIT.json'),'fit':bind(RUN/'FIT_COMPLETE.json'),'model_freeze':bind(RUN/'MODEL_FREEZE.json'),'source_saved_review':bind(V/'source_audit/SAVED_FIT_REVIEW.json'),'current_inference':bind(EV.parent/'INFERENCE_SEAL.json'),'current_evaluation':bind(EV/'EVALUATION_SEAL.json'),'current_saved_review':bind(V/'source_audit/SAVED_CURRENT_REVIEW.json')},
 'next_action_KO':'공개14cohort 검증곡선과 TOMIE 목재블록 잔여 실패를 근거로 다음 source 실험을 결정한다. 현재 실사로 문턱·checkpoint·모델을 고르지 않으며, 공개 검증상 endpoint 미수렴과 부족한 독립 TRAIN 다양성을 구분해 검증한다. PnP·초기q/L은 유지한다.'}
write(V/'ROOT_RESULTS_KO.json',out)

tables=[]
for group in ['WOOD_PRIMARY105','SQUARE118']:
    cells=[]
    for model in models:
        r=next(q for q in primary if q['group']==group and q['model']==model)
        cm=f"[{r['D_pred_D']}, {r['D_pred_hidden']}, {r['D_pred_U']}] / [{r['E_pred_D']}, {r['E_pred_hidden']}, {r['E_pred_U']}]"
        percent=lambda k:f"{100*float(r[k]):.1f}%"
        cells.append(f"<tr><td>{names[model]}</td><td>{r['D_pred_D']}/{r['human_D_N']}</td><td>{r['E_pred_hidden']}/{r['human_external_E_N']}</td><td>{percent('D_precision')} / {percent('D_recall')}</td><td>{percent('external_E_response_precision')} / {percent('external_E_response_recall')}</td><td>{percent('unresolved_rate')}</td><td class='cm'>{cm}</td></tr>")
    tables.append(f"<h3>{'목재105장'if group.startswith('WOOD')else'정사각형118장'}</h3><div class='scroll'><table><tr><th>모델</th><th>보임 정답</th><th>외부 가림 정답</th><th>보임 P/R</th><th>외부 가림 P/R</th><th>보류율</th><th>혼동행렬 D행 / E행</th></tr>{''.join(cells)}</table></div>")

page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C15 공개 학습 후 실사 평가</title>
<style>body{margin:0;background:#edf2f5;color:#182d3b;font:17px/1.7 system-ui,"Noto Sans CJK KR",sans-serif}main{max-width:1300px;margin:auto;padding:28px 22px 70px}h1{font-size:32px;line-height:1.4}h2{font-size:25px}h3{font-size:20px}section{background:white;padding:23px;border:1px solid #dce4e9;border-radius:14px;margin:22px 0}img{display:block;width:100%;height:auto;margin:18px 0}table{border-collapse:collapse;font-size:15px;width:100%}th,td{border-bottom:1px solid #dae3e8;padding:10px;text-align:left;white-space:nowrap}th{background:#edf2f5}.cm{font:14px ui-monospace,monospace}.scroll{overflow-x:auto}a{color:#096b97}.note{font-size:15px;color:#506775}.lead{font-size:21px}.status{display:inline-block;background:#fff0d7;color:#785219;padding:5px 12px;border-radius:8px}li{margin:8px 0}p{margin:10px 0}</style>
<main><span class="status">C15 실제 학습·평가 완료 · 목표는 계속 진행 중</span>
<h1>공개 source 추가 학습이 실사 외부 가림에도 일부 전이됐습니다</h1>
<p class="lead">외부 가림 정답 <b>15/90 → 35/90</b>, 보임 정답 <b>1,130/1,147 유지</b>. 아직55개 외부 가림을 놓치므로 해결이라고 부르지 않습니다.</p>
<p>모델 선택은 공개 검증14개 cohort의 BCE로 고정했습니다. 현재223장은 학습·정규화·문턱·checkpoint·모델 선택에 사용하지 않았습니다. 초기 q/L과 기하 self 경로를 유지했고 PnP는0입니다.</p>
<section><h2>실사 전체 비교: 같은 초기 query와 같은 사람 라벨</h2>TABLES
<p class="note">행은 사람 D/E, 열은 모델 보임/가림/보류입니다. 보류는 recall 분모에서 빼지 않습니다. Self는 이 D/E표에서 제외합니다. v61은 한 pixel의 target 존재, 기존 실사 모델은 평가 촬영 도메인에서 학습한 이전 기준선으로 목표·학습 자료가 서로 다릅니다. C11은 과거 실험이고 C15도 외부 가림 합계35/90으로 같습니다. 이 표로 checkpoint나 모델을 다시 선택하지 않았습니다.</p>
<p>기존 실사 모델은 외부 가림83/90을 잡지만 보임655/1147·보류177/1237입니다. C15는 보임을 더 많이 유지하지만 외부 가림 recall은 아직 낮습니다. <a href="private_evaluation/C15_FROZEN_CURRENT223/evaluation/D_VS_EXTERNAL_E_2X3.csv">전체 혼동행렬·P/R·보류 CSV</a></p></section>
<section><h2>실제로 바뀐 crop과 남은 실패</h2><img src="visual_followup/C15_FIXED4_SUCCESS_FAILURE.png" alt="고정 실제 촬영 네 사례 C14와 C15 비교">
<p>같은4사례를 유지했습니다. <b>원행1021</b>은 콘이 crop 중심을 차지하는데도 보임이던 판정이 C15에서 가림으로 바뀌었습니다. <b>원행325</b>의 콘·목재 경계는 아직 보임으로 놓칩니다. 노출된 모서리200과 기존 가림 성공1416은 유지됩니다.</p>
<p class="note">진단용 고정4개이며 전체 빈도 표본이 아닙니다. 가려진 E reference는 추정값일 수 있어 좌표 정답으로 보지 않습니다. 그림의 표시점은 별도 복사본이며 모델 입력에는 넣지 않았습니다.</p></section>
<section><h2>무엇을 학습했고 공개 검증에서는 무엇이 바뀌었나</h2><img src="source_visual_followup/SOURCE_C14_TO_C15.png" alt="공개 source C14에서 C15로 물체별 개선">
<p>이전 source46,108행을 유지하고, 공개 팔레트32cue의4,896행을 추가했습니다. 새 학습 손실에 들어간 것은 원래 LOCO TRAIN6cue/3원본의<b>보임450·외부 가림396</b>입니다. TOMIE22cue는 학습에 넣지 않았습니다. DINO의1928개 특징과64차원 hidden을 쓰는 head는 유지하고 C14 가중치에서1,000회 갱신했습니다.</p>
<p>기존128개와 새64개를 함께 학습했습니다. 새64개는 덮기E32·원본보임D16·같은물체를옆으로옮긴D16입니다. 기존 source의15개 회귀 검사를 모두 통과했습니다. 공개 검증14cohort 평균BCE는0.10109→0.04267로 줄었고 source에서step1000을 선택했습니다.</p>
<p>중앙 query에서 공개 원본+sham 보임은261/262→262/262, 외부 가림은184/230→214/230입니다. 남은 중앙 가림16개는 모두 TOMIE의 목재 블록 조건입니다. <a href="runs/C15_PUBLIC_FLAT_PALLET_AUGMENTED_SPATIAL_QUERY/NEW_FLAT_SOURCE_RESULTS.json">source 원본 수치</a></p>
<p class="note">32cue/9원본의 상관된 파생 자료이며 독립 실사4,896장이 아닙니다. LOCO5원본은 기존 source와 같은fold에서 겹칩니다. TOMIE4원본은 이전 학습 원본과 해시 겹침이 없지만 같은episode라 폭넓은 독립 검증을 뜻하지 않습니다. 공개 구조 cue는 약한 조건부 AUX이며 native 키포인트/사람 gold가 아닙니다.</p></section>
<section><h2>좌표 오류·자기 가림·보류를 구분했습니다</h2><p>사람이 클릭한 보임 reference와 초기 query의 오차가3px 이하인 사례는 목재<b>153/153</b>, 정사각형<b>223/223</b> 모두 보임으로 유지했습니다. 이것은 가시성 판정 결과이며 좌표가 맞는지 검증하는 별도 head는 아닙니다.</p>
<p>목재의 전체 human-self136개에서 C15 raw는15개를 binary 가림으로 응답하고, 기존 기하 경로를 합치면115개를 걸러냅니다. 이 수치를 learned self 분류라고 해석하지 않습니다. 동일 query 비교에 남는 self21개는 모두 보임으로 남으며, 정사각형에는 human-self gold가 없습니다.</p>
<p>전체1,784슬롯을 유지했고 원래 입력 미가용45개, 추가 보류0개입니다. 위 paired D/E에서는 보류0입니다. <a href="private_evaluation/C15_FROZEN_CURRENT223/evaluation/HUMAN_SELF_BINARY_RESPONSE.csv">Self 진단</a> · <a href="private_evaluation/C15_FROZEN_CURRENT223/evaluation/MANUAL_NEAR_D.csv">3px 이하 보임</a></p></section>
<section><h2>현재 결론과 다음 검증</h2><p>공개 source에서 확인한 실패를 학습으로 일부 개선했고 실제 촬영 외부 가림도20개 추가 검출했습니다. 목표 자체를 낮춰야 한다는 증거는 없지만, 일반화가 충분하다는 증거도 아직 없습니다. 실제 가림55개 누락과 공개 TOMIE 목재 블록 실패가 남습니다.</p>
<p>다음 source 실험은 공개 검증곡선의 endpoint 미수렴과 부족한 독립 TRAIN 다양성을 구분해 진행합니다. 공개 검증BCE는1,000회 끝까지 계속 감소했습니다. 기간 연장과 다른 구조/추가 자료 중 무엇이 필요한지는 source에서 검증한 뒤 고정하고 현재는 평가만 합니다.</p>
<p class="note">현재223은 과거 개발에 노출되어 fresh blind holdout으로 부르지 않습니다. C14 비교 CSV의 최소 schema 연결 오류가1회 있었으며, 저장값 전용 adapter로 geometry를 복원했습니다. 원 C15 추론·코드·출력을 보존했고 새 추론이나 학습은 하지 않았습니다. 기존 지표80행 재현과 독립 검산을 통과했습니다. HTML 브라우저 자동 열기는 이전 file:// 정책 제한을 우회하지 않고 PNG 직접검수·정적 경로 검사를 사용합니다.</p></section>
<p class="note"><a href="ROOT_RESULTS_KO.json">결과·근거 JSON</a> · <a href="runs/C15_PUBLIC_FLAT_PALLET_AUGMENTED_SPATIAL_QUERY/MODEL_FREEZE.json">선택 모델 정보</a> · <a href="runs/C15_PUBLIC_FLAT_PALLET_AUGMENTED_SPATIAL_QUERY/BEST_SPATIAL_TARGET_QUERY_HEAD.pt">C15 체크포인트</a> · <a href="source_audit/SAVED_CURRENT_REVIEW.json">독립 저장 평가 검산</a> · <a href="../query_visibility_flat_pallet_cue_source_20261023_v122/ROOT_READOUT_KO.html">앞 단계 실제 crop 진단</a></p></main></html>'''.replace('TABLES',''.join(tables))
with (V/'ROOT_READOUT_KO.html').open('x')as f:f.write(page)
print(json.dumps({'completed':True,'report':bind(V/'ROOT_READOUT_KO.html'),'results':bind(V/'ROOT_RESULTS_KO.json')},ensure_ascii=False))
