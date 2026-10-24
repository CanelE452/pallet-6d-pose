"""Korean artifact readout from frozen source/current outputs; no inference."""
from pathlib import Path
import json, hashlib, html
from datetime import datetime, timezone
R=Path(__file__).resolve().parent
P=R.parent
RUN=R/'runs/PHYSICAL_CUE_AUXILIARY_V1'
E=R/'contract/FROZEN_CURRENT_COMPARISON'

def load(p):return json.loads(Path(p).read_text())
def b(p):p=Path(p);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def write(p,d):
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def esc(x):return html.escape(str(x))
def pct(x):return '—'if x in (None,'')else f'{100*float(x):.1f}%'
def table(headers,rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</tbody></table></div>'
def link(label,path):return '<a href="'+esc(str(Path(path).relative_to(R)))+'">'+esc(label)+'</a>'
def figure(path,caption):
    rel=Path(path).relative_to(R)
    return '<figure><a href="'+esc(rel)+'"><img loading="lazy" src="'+esc(rel)+'" alt="'+esc(caption)+'"></a><figcaption>'+esc(caption)+'</figcaption></figure>'

source=load(RUN/'SOURCE_RESULTS.json');current=load(E/'CURRENT_PRIMARY_RESULTS.json')
source_audit=load(R/'independent_postfit_audit/REVIEW.json')
current_review=R/'independent_current_saved_audit/REVIEW.json'
assert source_audit['PASS'] and load(current_review)['PASS']
assert source['all_nine_pass'] and current['historical_baseline_replay']['PASS']
source_fit=load(RUN/'FIT_COMPLETE.json')
inp=load(R/'private_evaluation/NATIVE_CENTER_ONE_SHOT/INPUT_PREPARATION_AUDIT.json')
inf=load(R/'private_evaluation/NATIVE_CENTER_ONE_SHOT/INFERENCE_SEAL.json')
quant=R/'saved_quantitative_diagnosis/SAVED_COORDINATE_SCALE_DIAGNOSIS.json'
rootobs=R/'ROOT_CURRENT_ACTUAL_CROP_DIAGNOSIS_KO.json'
native=source['source_metrics']['VAL_HALL']
methods=['v61','legacy','V76_SUPPORT','V91_RAW']
names={'v61':'v61','legacy':'기존 가림 모델','V76_SUPPORT':'V76','V91_RAW':'V91 (새 head)'}
groups={'WOOD_PRIMARY105':'목재','SQUARE118':'정사각형'}
paired=[x for x in current['tables']['CLASSIFICATION_METRICS']if x['cohort']=='paired_same_query_RGB'and x['task']=='D_vs_E'and x['model']in methods]
paired.sort(key=lambda x:(list(groups).index(x['group']),methods.index(x['model'])))
manual=[x for x in current['manual_click_near_D_LE3']if x['model']in methods]
selfrows=[x for x in current['tables']['CLASSIFICATION_METRICS']if x['cohort']=='same_slot_pipeline'and x['task']=='D_vs_S']
cases=load(R/'current_actual_crop_diagnosis_v1/ACTUAL_CROPS_AND_FIGURES.json')['cases']
source_cases=load(R/'source_visual_diagnosis_v2/FIXED_CASE_SELECTION_AND_FIGURES.json')['cases']
readout=load(R/'current_actual_crop_diagnosis_v1/FIXED8_SAVED_REFERENCE_READOUT_DIAGNOSTIC.json')

summary=dict(schema='V91_PUBLIC_PHYSICAL_CUE_HEAD_AND_FROZEN_CURRENT_TRANSFER_COMPLETE',completed=True,created_UTC=datetime.now(timezone.utc).isoformat(),
    source_fit=b(RUN/'FIT_COMPLETE.json'),source_results=b(RUN/'SOURCE_RESULTS.json'),source_audit=b(R/'independent_postfit_audit/REVIEW.json'),
    current_inference=b(R/'private_evaluation/NATIVE_CENTER_ONE_SHOT/INFERENCE_SEAL.json'),current_evaluation=b(E/'EVALUATION_SEAL.json'),current_independent_audit=b(current_review),
    source_VAL_region=native['region'],source_VAL_reference=native['reference'],source_VAL_pairs=native['paired_views'],
    current_paired_D_E=paired,manual_near_D=manual,current_same_slot_D_S=selfrows,
    current_global_input_U=dict(total=1784,valid=1739,U=45,rate=45/1784,additional_U=0,semantic_U_detector=False),
    observed_source_to_current_failure='Source9/9pass did not transfer sufficient external recognition: currentErecallWOOD11/29,SQUARE30/61. Nearvisible errors also remain8/153 and26/223. Definition gap and residual errors under changed target/operator/data/scale conditions remain; query-reference displacement can affect saved readouts. No implementation phase fault has been proven.',
    saved_quantitative_diagnosis=b(quant),actual_crop_diagnosis=b(rootobs),actual8_crop_manifest=b(R/'current_actual_crop_diagnosis_v1/ACTUAL_CROPS_AND_FIGURES.json'),
    fixed8_saved_map_reference_diagnostic=b(R/'current_actual_crop_diagnosis_v1/FIXED8_SAVED_REFERENCE_READOUT_DIAGNOSTIC.json'),
    source_single_tile_bridge=b(R/'source_operator_bridge/SOURCE_SINGLE_TILE_REGION_BRIDGE.json'),
    source_scope='Singlephysicalwoodbin/environmentholdout conditionalAIcue labels. No human/native8gold. Sourceallgridmosaic vscurrentoneoriginalcentre, L128/256 vsnative16..233. Not calibrated or independentfinalgeneralization.',
    current_fit_or_threshold_or_checkpoint_selection=False,new_current_inference_runs=1,new_source_fit_runs=1,new_source_training_updates=2000,
    PnP=0,current_or_source_query_correction=0,fully_solved=False,goal_status='active',
    next_required='Source-only flatpallet intake/nativeinitializer-query feasibility and matched localcue/operator/scale contract. ExistingpublicLOCO assets; no current-driven threshold/secondcheckpoint selection.')
write(R/'ROOT_PHASE_RESULTS_KO.json',summary)

css='''body{margin:0;background:#f4f6fa;color:#172435;font:16px/1.65 "Noto Sans CJK KR","NanumGothic",sans-serif}main{max-width:1160px;margin:auto;padding:36px 26px}h1{font-size:30px;line-height:1.35}h2{margin-top:38px;font-size:23px}h3{font-size:18px}p{max-width:1050px}.lead{font-size:20px}.badge{display:inline-block;background:#fee2db;color:#9b2c18;padding:5px 14px;border-radius:20px;font-weight:700}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{background:white;border:1px solid #dbe1eb;border-radius:10px;padding:18px}.card strong{display:block;font-size:23px;color:#164484}.note{background:#e9eff9;border-left:4px solid #4d70a4;padding:14px 18px}.tablewrap{overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 10px;border:1px solid #dbe1eb;white-space:nowrap}th{background:#eaf0f8}a{color:#164484}figure{background:white;border:1px solid #dbe1eb;padding:12px;margin:20px 0}img{display:block;width:100%;height:auto}figcaption{padding:10px 5px;font-size:15px}details{background:white;border:1px solid #dbe1eb;padding:14px;margin:16px 0}summary{cursor:pointer;font-weight:bold}.small{font-size:14px;color:#506077}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:800px){.cards,.grid2{grid-template-columns:1fr}main{padding:18px 12px}h1{font-size:25px}}'''
s='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V91 공개 실사 학습과 현재 팔레트 고정 비교</title><style>'+css+'</style></head><body><main>'
s+='<div class="badge">현재 문제 미해결 · checkpoint/문턱 고정 비교 완료</div><h1>공개 물리 단서로 학습하면 보임 누락은 줄지만,<br>현재 팔레트의 외부 가림은 아직 충분히 구별하지 못합니다.</h1>'
s+='<p class="lead">V91은 같은 초기 좌표에서 목재 D 95.1%, 정사각형 D 88.9%를 맞혔습니다. 외부 가림 E는 37.9%, 49.2%입니다. 좌표 오차가 작은 실제 실패와 위치에 따라 판정이 바뀌는 사례를 함께 확인했습니다.</p>'
s+='<div class="cards"><div class="card">공개 검증 · 물리 cue 영역<strong>D 146/148 · E 22/22</strong>사전 조건 9/9 통과. 단일 목재 상자의 환경 holdout, 조건부 AI 라벨.</div><div class="card">현재 수동 좌표 ≤3px D<strong>목재 145/153 · 정사 197/223</strong>v61은 각각 42/153, 43/223. 실패는 8개와 26개가 남습니다.</div><div class="card">추론 입력 및 실행<strong>1,739 유효 · 45 보류</strong>223장/1,784점, 추가 보류 0. 새 head 한 번 추론. 현재 학습·문턱 조정·PnP 0.</div></div>'
s+='<h2>같은 paired D/E 행에서 비교</h2><p>D는 보이는 점, E는 외부 가림 정답입니다. 혼동행렬의 행은 D/E, 열은 보임/비보임/U입니다. U를 제외해 recall 분모를 줄이지 않았습니다.</p>'
tab=[]
for x in paired:
    iv=lambda k:int(x[k]);up=iv('U_visible')+iv('out_of_task_S_visible')+iv('other_unresolved_visible');un=iv('U_occluded')+iv('out_of_task_S_occluded')+iv('other_unresolved_occluded')
    tab.append([groups[x['group']],names[x['model']],f"[{x['TP']},{x['FN_decided']},{up}] / [{x['FP']},{x['TN']},{un}]",pct(x['visible_precision']),pct(x['visible_recall_all_GT']),pct(x['occluded_precision']),pct(x['occluded_recall_all_GT']),pct(x['unresolved_rate'])])
s+=table(['대상','모델','CM D행 / E행','D precision','D recall','E precision','E recall','보류/미해결'],tab)
s+='<p class="small">목재 N=645(D616/E29), 정사각형 N=592(D531/E61). 이 비교 가능 subset에서 새 head U는 0%; 전체 native1,784점에서는 입력 무효 45점(2.52%)을 U로 유지합니다. 새 head에는 의미적 불확실도 보류 기능이 없습니다. v61/기존 가림 모델/V76의 저장 결과와 기존 분모를 그대로 재현했습니다. 이 표의 비보임 출력만으로 인과 E/S를 확정하지 않습니다.</p>'
s+='<h2>수동 클릭 기준 오차 ≤3px인데도 놓치는 점</h2>'
s+=table(['대상','모델','보임 정답 수','보임으로 판정','비보임','보류','D recall'],[[groups[x['group']],names[x['model']],x['N'],x['pred_visible'],x['pred_absent_nonvisible'],x['pred_legacy_U']+x['pred_other_unresolved'],pct(x['semantic_D_recall'])]for x in manual])
s+='<p>아래 두 실패에는 노출된 목재 발/녹색 외곽 모서리가 실제 입력 crop 안에 보입니다. q 오차는 1.00px와 1.75px이며, 같은 저장9×9맵에서 기존 r를 읽어도 E 판정이 유지됩니다. 단순 좌표 누락만으로 설명할 수 없는 국소 판정·전이 실패가 남습니다.</p>'
for i,caption in [(0,'목재: 수동 D, q–r 1.002px. q/r의 저장 zE=+19.012/+19.522로 모두 비보임.'),(4,'정사각형: 수동 D, q–r 1.745px. q/r의 저장 zE=+2.055/+5.685로 모두 비보임.')]:s+=figure(cases[i]['figure']['path'],caption)
s+='<h2>위치 오류가 기여하는 실패와, 위치만으로 설명되지 않는 실패</h2><p>아래 정사각형 E 사례는 q–r가 23.0px이고, q에서 D였던 저장 공간 응답이 r에서는 E로 바뀝니다. 반면 목재 E 사례는 15.8px 차이가 있어도 q/r 모두 D입니다. 두 r 모두 원래 crop 안에 있습니다. 평가용 r 조회는 원래 판정을 바꾸지 않는 사후 진단이며, 좌표 교정이나 새 추론 성능으로 집계하지 않았습니다.</p>'
for i,caption in [(6,'정사각형 E 오류: q zE=-25.432(D), r 조회 +6.630(E). cone 경계/표면의 위치 차이가 응답을 바꿉니다.'),(2,'목재 E 오류: q zE=-28.644(D), r 조회 -5.970(D). cone가 가리는 실제 crop에서도 외부 가림을 놓칩니다.')]:s+=figure(cases[i]['figure']['path'],caption)
s+='<p class="note">외부 가림의 거짓 보임 중 r가 crop 안인 사례는 목재 14/18, 정사각형 30/31입니다. 수동 ≤3px D의 실패 8/26개도 모두 r가 crop 안입니다. crop 밖으로 벗어난 경우만 고쳐서는 현재 오류를 모두 없앨 수 없습니다.</p>'
s+='<details><summary>나머지 고정 실제 사례 4개: 맞힌 보임/가림과 위치 조회 반례</summary>'
for i in [1,3,5,7]:s+=figure(cases[i]['figure']['path'],cases[i]['stratum']+' · '+cases[i]['id'])
s+='</details><h2>자기 가림 S는 별도로 평가</h2><p>새 head는 S를 학습하지 않았습니다. 기존 초기 cuboid 기하 flag를 적용한 route를 분리했습니다. 목재 같은-slot D626/S136에서 raw S recall은 21/136(15.4%), 기존 기하를 적용하면 117/136(86.0%)입니다. D는 592→586개로 줄어듭니다. 정사각형은 S 정답이 없어 S 성능을 평가할 수 없습니다. PnP나 자세 최적화는 실행하지 않았습니다.</p>'
s+=table(['모델','D행 보임/비보임/U','S행 보임/비보임/U','D recall','S recall','S precision','보류/미해결'],[[x['model'],f"{x['TP']}/{x['FN_decided']}/{int(x['U_visible'])+int(x['other_unresolved_visible'])}",f"{x['FP']}/{x['TN']}/{int(x['U_occluded'])+int(x['other_unresolved_occluded'])}",pct(x['visible_recall_all_GT']),pct(x['occluded_recall_all_GT']),pct(x['occluded_precision']),pct(x['unresolved_rate'])]for x in selfrows])
s+='<h2>목표 정의 차이와 실제 전이 실패를 구분</h2>'
s+=table(['구분','저장 증거','해석'],[
    ['목표 정의 차이','V86 고정 native24: 실제 구조 cue로 관찰한10개 중8개는 원r의 material mask가0','가상 cuboid 기준점의 보임을 단일 목재 픽셀 membership으로 대체할 수 없습니다. 이 원본 정의 차이는 확인됐습니다.'],
    ['실제 전이 실패','V91은 공개 물리 cue D/E로 학습 후에도 현재 수동≤3px D에서8/26개 누락','정의만 바꾸면 모든 오류가 사라진다는 주장은 성립하지 않습니다. 실제 노출 모서리 실패를 crop으로 확인했습니다.'],
    ['위치/운영 조건 차이','원래q만 사용; source는129tile mosaic, 현재는 한 crop centre','사후r조회가 판정을 바꾸는 사례도 있지만 반대 방향/그대로인 사례도 있습니다.'],
    ['데이터/대상 차이','단일 tallwoodbin/box vsflatwood·plasticpallet/trafficcone; targetID 입력없음','외형·part·가림 물체·대상 결부 차이와 양립합니다. 어느 하나가 유일한 원인인지는 ablation하지 않았습니다.']])
s+='<p>공개 실사 source는 L128/256이지만 현재 원L은 16.37–233.06이고, 1,632/1,784점(91.5%)은 L128보다 작습니다. 입력 dtype도 source ROI F16과 현재 원본 보존 F32라는 차이가 있습니다. 이를 기록했으며 인과 효과를 측정한 ablation으로 주장하지 않습니다.</p>'
s+='<h2>이번에 실제로 학습한 것</h2><p>frozen YOLO P3와 RGB96/valid96 세 입력으로 9×9 물리 cue evidence를 읽는 25,381 parameter head입니다. 합성→YCB에서 학습한 공유 encoder만 이전하고 새 출력부를 학습했습니다. 공개 Pallox 50쌍/100장, 원cue302개/604view(D261/E37/U306)를 유지했습니다. 학습은 LAB22쌍, 검증은 HALL28쌍이며 unknown·crop MISS·지원 없음은 정답 음성으로 만들지 않았습니다. 영역의 고정9위치 logit 평균에 한 번 BCE를 주었습니다.</p>'
s+=f'<p>한 번의2,000-update 학습에서 source VAL region BCE로 update{source_fit["best_update"]}를 선택했습니다. threshold0.5는 사전 고정했습니다. 공개 VAL region CM=[[146,2,0],[0,22,0]], r-only CM=[[147,1,0],[0,22,0]], D→E22쌍 모두 맞혔습니다. 미정의166view는 이170known 성능에 섞지 않았습니다. source/native8 대응은 UNKNOWN이며 조건부 AI 관찰을 human gold로 부르지 않습니다.</p>'
s+='<details><summary>실제 source crop 3사례와 원본 before/after</summary>'
for x in source_cases:s+=figure(x['figure']['path'],x['stratum']+' · '+x['cue_id'])
s+='</details><h2>source의 검증 연산도 현재 운영과 달랐습니다</h2><p>저장된 VAL1,264개 FULL tile/cue 단위를 전부 사용해 각 tile의 원 uncertainty 영역만 읽어도, cue/view와 scale을 균등화한 가중 D recall은 95.50%, E recall은 97.73%였습니다. source 내 단일 tile의 원 uncertainty 영역에서도 높은 점수가 남으므로, source 성공이 전적으로 mosaic 합성만의 산물이라고 볼 근거는 없습니다. 현재는 q-centre·라벨·대상·스케일이 달라 각 요인의 기여량을 분리할 수 없습니다. 두 표의 집계 단위는 다르므로 원시 count를 합치지 않았습니다.</p>'
s+='<p class="note">이1,264단위 중 q 중심이 원 uncertainty box 안에 있는 경우는 0개였습니다. 따라서 source의 높은 영역 점수는 현재가 읽는 q-centre 판정의 직접 검증이 아닙니다. 이것이 centre cell의 학습 gradient가 0이었다는 뜻은 아닙니다. 보간·공유 convolution을 통해 학습 신호는 전달될 수 있습니다.</p>'
s+='<h2>다음 source 작업</h2><p>현재 점수를 보고 문턱이나 checkpoint를 다시 고르지 않았습니다. 기존 공개 LOCO flat-pallet 자료의 실제 frozen initializer query 접근성을 조사하고 있습니다. 다음 단계에서는 정답을 보고 query를 고르는 대신, 먼저 고정된 실제 초기 q/crop을 만들고 그 조건에서 물리 cue의 보임·가림·위치 누락을 구분해야 합니다. 작은 crop과 centre 판정을 source 검증에서도 직접 맞추는 계약이 필요합니다. source mask가 비었다는 사실을 가림 라벨로 대체하지 않고, 자기 가림은 별도로 유지합니다.</p>'
s+='<h2>원본 결과와 검산 파일</h2><ul>'
for label,path in [('전체 혼동행렬·precision/recall·보류율·모든 cohort 표',E/'CURRENT_TABLES.html'),('현재 평가 JSON',E/'CURRENT_PRIMARY_RESULTS.json'),('현재 독립 saved-only 감사',current_review),('오차×crop×L 진단',quant),('실제8crop 관찰과 근거',rootobs),('공개 source 독립 감사',R/'independent_postfit_audit/REVIEW.json'),('단일 source tile과 mosaic 연산 bridge',R/'source_operator_bridge/SOURCE_SINGLE_TILE_REGION_BRIDGE.json'),('사전 학습 계약',R/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'),('현재 평가 사전 계약',R/'CURRENT_PROTOCOL_BEFORE_CURRENT_ARTIFACTS.json'),('source checkpoint',RUN/'BEST_PHYSICAL_CUE_HEAD.pt'),('집계 호출 경로 오류 보존/수정 기록',R/'contract/ROOT_RELATIVE_OUTPUT_INVOCATION_REPAIR.json')]:s+='<li>'+link(label,path)+'</li>'
s+='</ul><p class="small">현재223장은 직접 촬영했고 KU Leuven과 무관하다는 사용자 확인을 기록했습니다. 현재 자료는 평가에만 사용했으며 학습/threshold/checkpoint 선택에 사용하지 않았습니다. 역사적으로 반복 평가된 자료이므로 새 독립 holdout이라고 주장하지 않습니다. source/current 모델·좌표·라벨·저장 출력은 고정됐습니다. 독립 감사 PASS는 저장값·좌표·집계의 일관성 검증이며 물리 정답이나 일반화 성능의 인증이 아닙니다. 전체 문제는 아직 해결되지 않았고 goal은 active입니다.</p></main></body></html>'
with (R/'ROOT_READOUT_KO.html').open('x')as f:f.write(s+'\n')
print(json.dumps(dict(completed=True,result=b(R/'ROOT_PHASE_RESULTS_KO.json'),report=b(R/'ROOT_READOUT_KO.html'))))
