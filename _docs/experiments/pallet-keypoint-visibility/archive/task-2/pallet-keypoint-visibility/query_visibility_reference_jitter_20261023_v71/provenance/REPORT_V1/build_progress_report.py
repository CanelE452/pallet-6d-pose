from pathlib import Path
import base64,datetime,hashlib,html,json

V=Path(__file__).resolve().parent
P=V.parent
B=V/'data/synth_reference_jitter/BUILD_V2'
Y=V/'data/ycb_cue_review/FROZEN_CUE_OWNERSHIP_SANITY'
G=V/'data/ycb_proposal_grid'
W=V/'data/r2b_cue_review'

def read(p):return json.loads(p.read_text())
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,d):
    assert not p.exists(),p
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def link(p,label):
    import os
    return f'<a href="{html.escape(os.path.relpath(p,V))}">{html.escape(label)}</a>'
def image(p,caption):
    return '<figure><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+str(x)+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'

pilot=read(B/'PILOT_RESULT.json')
audit=read(V/'source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json')
ycb=read(Y/'SANITY_REPORT.json')
grid=read(G/'GRID_RESULT.json')
wood=read(W/'R2B_WOOD_CUE_PROPOSAL_LEDGER.json')
assert audit['status']=='PASS' and audit['GT_input_leakage_arrays']==0
assert pilot['rows']==260 and not pilot['contradictory_supervised_D_E_target_collisions']
report={
 'schema':'V71_ACTUAL_REFERENCE_QUERY_DATA_PROGRESS_KO',
 'completed_data_progress':True,'model_training_completed_this_version':False,'full_goal_solved':False,'goal_status':'active',
 'purpose_ko':'query-pixel support와reference visibility를분리하는실제source RGB/crop/opaque-intervention 데이터를구성했다. 사용자native8 의미전체해결또는새모델성능개선의증거는아직아니다.',
 'actual_synth_pilot':bind(B/'PILOT_RESULT.json'),'actual_synth_independent_audit':bind(V/'source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json'),
 'actual_ycb_cue_sanity':bind(Y/'SANITY_REPORT.json'),'actual_ycb_source_proposal_grid':bind(G/'GRID_RESULT.json'),
 'source_grid_counts':grid['counts'],'source_grid_by_split':grid['by_split'],
 'public_wood_cues':bind(W/'R2B_WOOD_CUE_PROPOSAL_LEDGER.json'),
 'dataset_training_labels_S_not_fabricated':True,'native8_semantic_mapping':'UNKNOWN for new generic cues; deployment association still unresolved',
 'current223_access':0,'current223_fit_or_threshold_choice':False,'new_model_forward':0,'optimizer_updates':0,'PnP':0,
 'next_execution_ko':'고정된YCB sourcegrid와RGB조건부D cue를유지한채uncertainty전체를덮는새opaque E와untouched-D 대조쌍을구성한다. source-family를유지하고native8↔cue배포대응및자기가림교사한계를분리한다. 새학습은해당semantic/availability감독과평가protocol을먼저고정해진행한다.',
 'created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
write(V/'PROGRESS_KO.json',report)
summary_rows=[]
for row in pilot['counts']:
    if row['jitter_role']=='PRIMARY':summary_rows.append([row['variant'],row['reference_state'],row['availability'],row['training_target'],row['N']])
css='body{font:16px/1.7 system-ui,sans-serif;color:#15283b;background:#edf2f7;margin:0}main{max-width:1150px;margin:auto;padding:40px 48px;background:#fff}h1{font-size:31px;line-height:1.35}h2{border-top:2px solid #dce5ed;margin-top:38px;padding-top:18px}.box{border-left:5px solid #b87812;background:#fff4df;padding:16px 20px}.small,figcaption{font-size:14px;color:#53677b}img{max-width:100%;height:auto}figure{margin:26px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border:1px solid #d8e2ed;text-align:left}th{background:#edf3fa}.scroll{overflow:auto}a{color:#1666a6}code{font-size:13px;white-space:pre-wrap;overflow-wrap:anywhere}'
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v71 실제 reference/query 학습 데이터 준비</title><style>{css}</style><main>
<p class="small">2026-10-23 · 실제 source 데이터 구성·검산 · 새 모델 학습 0 · 현재 223장 접근 0 · PnP 0</p>
<h1>v71: query가 빗나가도 같은 reference의<br>가시성 라벨을 유지하는 데이터</h1>
<p>이전 v70은 공개 source의 conditional V/E를 일부 개선했지만 배경 false-visible과 실제 목재 외부 가림 실패가 남았다. 이번에는 모델을 다시 학습하기 전에, 원하는 가시성과 query 한 픽셀의 존재 여부를 구분하는 <b>실제 RGB 개입·crop 데이터</b>를 만들었다.</p>
<div class="box"><b>데이터 준비의 진전이며, 새 가림 모델 성능 향상 결과는 아니다.</b><br>합성 pilot 260행은 실제 생성·독립 검산을 완료했다. 공개 YCB cue와 source proposal 입력도 추가했다. 새로운 cue는 AI 조건부 teacher이며 human gold나 사용자의 canonical native8 정답으로 승격하지 않았다. 전체 목표는 진행 중이다.</div>
<h2>1. 실제 만든 합성 reference/query pilot</h2>
<p>기존 고정 cue 제안 8개를 그대로 보존하고, 실제 RGB·소유권 sanity가 양립한 4곳만 source D teacher로 허용했다. 2개 null과 2개 불확실한 association은 U로 남겼다. cue 좌표·불확실성 영역은 움직이지 않았다. 기존 초기 q와 L·rough box를 보존한 채 q만 고정 grid로 이동했다.</p>
<p>각 허용 cue에 대해 원본, 불확실성 영역을 전부 덮는 다른 물체의 opaque layer, 일부만 덮는 layer, cue를 건드리지 않는 다른 물체 대조군을 생성했다. source YCB TRAIN cutout을 실제 RGB 위에 불투명 전경 layer로 합성했다. 완전 가림만 E teacher, 부분 가림은 U다. source가 원래 보였다는 근거는 AI RGB 관찰 조건부이며 pose·mask 자체를 human semantic 정답으로 부르지 않는다.</p>
{table(['원래 reference 상태와 입력 상태','결과'],[['실제 raster / query crop','20개 / 260개'],['학습 교사 D / E / U','54 / 27 / 179'],['D지만 query 한 픽셀은 배경','42 crop행'],['E지만 query 한 픽셀은 대상','2 crop행'],['같은 관측 crop의 숨은 D/E 상태 충돌','21묶음 — 모두 학습 U'],['관측 가능한 supervised D/E 라벨 모순','0'],['독립 검산','260행 RGB·valid·q/L/box·GT 일치, 채널 차이 0']])}
<p class="small">행 수는 독립 reference 수가 아니다. 4 cue·3 RGB 중 실제 primary D/E 학습에 관측 가능한 것은 <b>3 cue·2 RGB</b>뿐이다. q±0.125L의 9개는 primary, ±0.75L 축 4개는 위치 stress이며 학습 D/E에서 U로 제외했다. 이 합성 pilot 자체에는 독립 VAL이 없다. U는 데이터의 감독 불확실성이며 모델의 예측 보류율이 아니다.</p>
{image(B/'ACTUAL_FIXED_REFERENCE_COUNTEREXAMPLES.png','동일 q/L/box에서 original / full cover / partial cover / untouched-cue control을 비교했다. 표시된 cue와 영역은 GT-only 진단용이며 모델 입력 PNG에는 들어가지 않는다.')}
<p>첫째는 cue가 보여도 q가 배경이다. 둘째는 cue가 가려져도 q에는 대상 표면이 보인다. 셋째는 cue를 덮거나 안 덮어도 crop이 그 위치를 놓쳐 입력이 같다. 따라서 세번째의 reference D/E를 억지로 binary 학습시키면 입력에서 알 수 없는 정답을 요구하게 된다. crop MISS는 위치·관측 한계이며 외부 가림이나 물리적 부재로 바꾸지 않았다.</p>
<p>cue 밖에 다른 물체를 붙여도 D를 유지하는 27개 primary 대조군 중 21개는 실제 입력 crop의 RGB가 달라지고, 6개는 입력 crop 밖의 개입이라 원본과 같다. ‘paste가 있으면 E’라는 지름길을 대조할 수 있는 실제 D 관측을 포함했다.</p>
<details><summary>합성 primary 감독 분모 상세</summary>{table(['variant','reference','crop 관측','학습 target','행수'],summary_rows)}</details>
<h2>2. 공개 YCB의 실제 cue를 기존 query와 대조</h2>
<p>source ID SHA 순으로 TRAIN 8 target·VAL 4 target을 고정하고, unmarked RGB에서 실제 can rim·blade tip·box corner·wood corner·pitcher lip 등을 관찰해 cue 좌표와 불확실성 영역 12개를 먼저 동결했다. 이후 mask 소유권과 crop을 대조했다. 이전 mask 노출이 있었으므로 blind annotation 또는 human gold가 아니다.</p>
<p>source RGB cue와 소유권이 양립하는 조건부 후보는 7개, 충돌·경계 모호 후보 5개는 U다. 원 q와 L을 유지하면 FULL 4·PARTIAL 1·MISS 7이며, FULL과 D 후보가 동시에 성립하는 것은 <b>VAL의 2개뿐</b>이다. 기존 controlled paste는 12개 cue의 불확실성 영역을 하나도 덮지 않았다. 따라서 기존 query-pixel 가림 label을 새 cue의 E label로 재사용할 수 없다. 기존 q0 조건의 TRAIN 학습 D/E는 0개이며 이 결과를 그대로 보존한다.</p>
<p>{link(Y/'FROZEN_CUE_NATIVE64_SANITY_1.png','YCB 실제 cue 1–4')} · {link(Y/'FROZEN_CUE_NATIVE64_SANITY_2.png','cue 5–8')} · {link(Y/'FROZEN_CUE_NATIVE64_SANITY_3.png','cue 9–12')} · {link(Y/'SANITY_REPORT.json','고정 좌표·소유권·crop 대조 JSON')}</p>
<h3>GT 쪽으로 crop을 옮기는 대신 별도 source proposal 분포를 구성</h3>
<p>기존 q0/q8 결과는 유지했다. 추가 source 입력은 old rough box의 각 축을 9등분한 cell center 81곳이며, <b>box와 old L만으로 좌표·순서·hash를 먼저 고정</b>했다. cue/reference/state/mask로 좌표를 고르지 않았고, L도 확대하지 않았다. 이후 원본과 기존 controlled RGB에서 모든 crop을 생성하고 GT-only cue 관측 가능성을 대조했다.</p>
<p>이는 <b>공개 source용 emulated proposal 분포</b>다. source rough box의 기존 teacher-derived approximate proposal 계보를 유지했으며 실제 현재 native8 초기 추론 분포나 그 성능으로 부르지 않는다. 12 target×81 query×2 view의 전체 1,944행을 남기고, 관측 가능한 좋은 q만 숨겨 선별하지 않았다. 기존 paste는 cue를 건드리지 않은 neutral-D 역할이며 새 semantic E 학습 사례는 아니다.</p>
{table(['source grid 분모','전체행','FULL','PARTIAL','MISS','조건부 학습 D','E','U'],[['TRAIN',1296,50,80,1166,22,0,1274],['VAL',648,24,46,578,20,0,628],['전체',1944,74,126,1744,42,0,1902]])}
<p>조건부 D를 유지하며 cue 전체를 볼 수 있는 TRAIN crop 22행을 새로 확보했다. 원본 11행과 같은 cue가 보존된 controlled 11행이며 독립 reference가 22개 생긴 것은 아니다. E는 여전히 0이라 이 공개 실사 데이터만으로 binary 가림 head를 학습할 준비가 됐다고 하지 않는다. 전체 U 1,902행도 모델 예측 보류율이 아니다.</p>
<p>{link(G/'GRID_RESULT.json','실제 source grid 집계·입력/GT 파일')}에 전체 TRAIN/VAL·FULL/PARTIAL/MISS·D/U 분모와 hash를 기록했다. 이 단계도 native8 대응과 E source coverage를 해결한 학습 완료 선언은 아니다.</p>
<h2>3. 실제 낮은 목재 팔레트 cue도 추가</h2>
<p>이미 확보한 R2B RGB 48장을 검토해 기존 3개를 재검토하고, 4개의 목재 cue 후보와 U 1개를 추가했다. 총 <b>AI visible-cue 후보 7개·U 1개</b>다. 상자·보라 crate를 팔레트 cue로 취급하지 않았으며, 불명확한 seam에는 좌표를 만들지 않았다. 기존 좌표는 유지했고 blur·layered edge 불확실성 범위는 새 ledger에 따로 기록했다.</p>
{image(W/'EIGHT_RECORD_WOOD_CUE_RAW_MARKED_CONTACT_SHEET.png','공개 낮은 목재의 raw/marked crop. 같은 clip·sensor·물리적 cue 반복일 수 있으므로 7개의 독립 reference family로 세지 않는다.')}
<p>두 clip은 유사 촬영 계열 위험이 있어 전체를 한 unresolved coarse family로 묶었다. 독립 train/VAL 분리, native8 대응, human gold, E/S teacher는 없다. source 후보를 학습 정답으로 자동 승격하거나 현재 평가용 자료로 섞지 않았다.</p>
<h2>4. 남은 일과 범위</h2>
<p>다음은 고정된 공개 source proposal과 cue를 유지하면서 <b>cue 불확실성 영역 전체를 덮는 새로운 opaque E</b>와 untouched-D 대조쌍을 구성하는 일이다. 그 뒤 semantic cue visibility와 crop availability의 loss·평가·문턱을 source에서 먼저 고정하고 새 component를 학습한다. 사용자의 native8과 새 generic cue 사이의 배포 가능한 reference association도 별도로 검증해야 한다.</p>
<p>자기 가림의 semantic 교사는 이번에 새로 만들지 않았다. 기존 convex/ray 또는 기하 S proxy를 human semantic S로 바꾸지 않고 별도로 유지한다. 어떤 mask가 0이거나 crop이 MISS라는 이유로 E/S를 만들지 않는다. 일반 crop 내 target support만 맞추는 것으로 전체 native-keypoint 가시성 목표를 대체하지 않는다.</p>
<p>현재 223장의 RGB·annotation·좌표·점수는 이번 데이터 구성에 접근하지 않았고 fit·문턱 선택·checkpoint 선택도 없다. 새 model forward·학습·PnP는 모두 0이다. 이전 source 분류 결과는 {link(P/'query_visibility_target_ownership_20261023_v70/RESULTS_KO.html','v70 비교 보고서')}에 보존했다.</p>
<h2>5. 검산과 실패 기록</h2>
<p>V1 검증이 underlying reference D/E와 감독 target을 구분하지 않아 MISS의 동일 입력 충돌에서 종료했다. 실패 코드·entry·260 crop을 보존했고, V2는 숨은 상태 충돌 21묶음과 supervised target 충돌을 따로 검사했다. <b>20개 raster PNG가 V1과 V2에서 byte 동일</b>하며 q·cue·label·protocol을 바꾸지 않았다. 독립 구현으로 모든 crop과 실제 alpha·RGB 개입을 재계산했다.</p>
<p>{link(V/'source_audit/ACTUAL_PILOT_INDEPENDENT_AUDIT.json','실제 데이터 독립 감사 PASS')} · {link(V/'source_audit/REFERENCE_LABEL_EXECUTION_REVIEW.json','라벨 실행 검토')} · {link(B/'INPUT_COLLISION_AUDIT.json','동일 입력의 상태·감독 모순 감사')} · {link(B/'V1_RASTER_REPLAY_PARITY.json','V1/V2 raster 동일성')} · {link(B/'MODEL_INPUTS_ONLY.npz','모델 입력만 있는 NPZ')} · {link(B/'GT_ONLY.npz','GT-only NPZ')} · {link(V/'PROGRESS_KO.json','진전 JSON')}</p>
<p class="small">작성 {html.escape(report['created_at_UTC'])}. 전체 목표 진행 중. 데이터 행 증가·감사 PASS는 실제 가림 모델이 해결됐다는 증거가 아니다.</p></main></html>'''
assert not (V/'PROGRESS_KO.html').exists()
(V/'PROGRESS_KO.html').write_text(body)
print(json.dumps({'report':bind(V/'PROGRESS_KO.html'),'JSON':bind(V/'PROGRESS_KO.json')}))
