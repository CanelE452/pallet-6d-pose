from pathlib import Path
import hashlib,json,html
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent
def load(f):return json.loads(Path(f).read_text())
def bind(f):return dict(path=str(Path(f).resolve()),sha256=hashlib.sha256(Path(f).read_bytes()).hexdigest())
def write(f,j):
 with open(f,'x')as h:h.write(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(head,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</tbody></table>'
def pct(x):return '—'if x is None else f'{100*x:.1f}%'
find=load(R/'coverage/results/COVERAGE_FINDINGS_KO.json')
obj=load(R/'object_split_diagnosis/OBJECT_SPLIT_SAVED_RESULTS.json')
gal=load(R/'natural_failure_gallery/GALLERY_METADATA.json')
coll=load(R/'input_collisions/INPUT_COLLISION_AUDIT.json')
assert find['PASS'] and find['strict535_reference_vs_query']['N']==535
cm={}
for model in('V76','V77','V78'):
 rows=[x['metrics']['CM_rows_CV_cols_CV']for x in obj['VAL_seen_unseen']if x['model']==model]
 cm[model]=np.sum(rows,axis=0)
metrics=[]
for name,c in cm.items():
 metrics.append([name,str(c.tolist()),pct(c[0,0]/c[:,0].sum()),pct(c[0,0]/c[0].sum()),pct(c[1,1]/c[:,1].sum()),pct(c[1,1]/c[1].sum())])
decisions=P/'query_visibility_dense_support_20261023_v80/EXPERIMENT_DECISIONS_BEFORE_GT_BUILD.json'
root=dict(completed=True,goal_turn='PROGRESS',fully_solved=False,diagnostic_scope='Saved source outputs plus raw source masks/RGB; no new model forward, fit or current223 access.',facts={
 'full_source_rows':10002,'queries':80016,'strict535_reference_and_query_state_match':535,'strict_reference_C_q_native3x3_all_C':76,'V78_strict_C_correct':17,'V78_strict_V_correct':357,
 'strict_C_crop_visible_fraction_median':find['strict_raw3x3']['0']['RGB96_V_fraction_median'],'ROI_float32_mask_class_changes':0,'exact_inference_input_collisions':0,
 'VAL_target_instances':800,'VAL_unique_RGB':200,'natural_primary_VAL_C':287,'natural_primary_VAL_V':1731,'VAL_only_target_objects':[6,11]},
 interpretation_KO=['고정 근접535에서 reference-vs-query 중심 pixel 목표 차이와 큰 좌표 오차만으로 실패를 설명할 수 없다.','가려진 strict76 모두 q의 raw3x3도 C이며, crop V 중앙값2.15%다. 전역 crop의 가시 우세만으로 모든 오류를 설명할 수 없다.','실제 주전자/드릴과 나무블록 예시는 가림/가시 source mask와 육안이 부합한다. 네 예시는 결과 cell별 선고정 선택으로, 전체 label오류율이나 전이실패의 단일 원인을 추정하지 않는다.','V78은 TRAIN에서 본 종류에서도 자연C recall38/217로 낮다. 미학습 종류만의 문제라는 설명은 배제된다.','crop내 target128은 대상과 occluder가 섞일 수 있다. 대상 소유권·공간 감독·readout 문제는 가능한 가설이며 인과적으로 확정되지 않았다.','완전히 같은 입력의 상충 GT는 없었다. 연속 입력값의 유일성은 대상 식별 가능성을 보장하지 않는다.'],
 limits_KO=['원래 public q는 GT 기반 모사이며 실제 초기추론 좌표가 아니다.','M0/M4는 렌더·깊이 정책에 따른 support proxy다. 사람 D/E/S와 동일하지 않고 기존 mask 생성모드/commit/delta도 미확인이다.','전체 primary에서는 q중심 outsideM0가 많다. outside/무효를 semantic가림으로 합치지 않는다.','반복 sourceVAL은 개발 지표다. current223 역시 앞선 실험들의 반복 평가셋이며 독립 새검증으로 부르지 않는다.','RGBbilinear/P3 RF와 nominalmaskNN의 물리적 동일성을 주장하지 않는다.'],
 next_action='V80 same-input spatial auxiliary source control; labels, primary admission, sampler, initialization, threshold and selector frozen before execution.',next_decisions=bind(decisions),current223_reads=0,new_model_forward=0,new_fit=0,PnP=0)
write(R/'ROOT_DIAGNOSIS_AND_NEXT_DECISION.json',root)
body='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>V79 자연 가림 오류 진단</title><style>body{font-family:Arial,"NanumGothic",sans-serif;margin:35px auto;max-width:1280px;color:#17222b;line-height:1.6;padding:0 24px}h1{font-size:30px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #bbc8d0;padding:9px;text-align:left}th{background:#e8f0f5}.note{padding:17px;background:#edf4f8;border-left:5px solid #527a90}img{width:100%;height:auto}a{color:#175575}code{background:#eef1f3}small{color:#4c5c66}</style>
<h1>좌표가 맞아도 자연 가림을 놓치는 이유: V79 저장 자료 진단</h1>
<p class="note"><b>좌표·중심 라벨 차이만으로 설명되지 않는 오류가 남았습니다.</b> 고정 근접 535점(q-r≤3px·STRONG teacher·L∞/L≤1/18)은 reference와 query 중심의 가림 상태가 모두 일치했습니다. DINO 후보 V78은 이 중 가려진 76점에서 17점만 맞혔습니다. 대상 소유권과 공간 감독을 다음 통제실험에서 확인합니다. 이 단계의 새 학습·모델 추론·현재 평가 223장 접근·PnP는 모두 0입니다.</p>
<h2>같은 공개 실사에서 확인한 성능</h2><p>YCB 자연 원본 VAL 800개 target instance / RGB 200장, 고정 primary 2,018점(C287·V1731). C는 M0에 있지만 M4에는 없는 대상 support proxy, V는 가시 support입니다. 사람의 외부/자기 가림 라벨을 의미하지 않습니다. 행·열 순서 [C,V], 문턱 0.5 및 기존 체크포인트를 유지했습니다.</p>'''
body+=table(['모델','혼동행렬 GT행/예측열 [C,V]','C precision','C recall','V precision','V recall'],metrics)
body+='<h2>전체 crop 감사로 배제한 설명</h2>'+table(['점검','결과','해석 범위'],[
 ['모든 source','10,002 instance rows / 80,016 queries','원래5,203 + controlled after4,799 모두 유지'],
 ['근접535점','C76→qC76 / V459→qV459','중심 q와 r의 pixel-state 차이0'],
 ['가려진76점의 q 주변','raw3×3 모두 C: 76/76','q주변 가시/비가시 혼합만으로 설명 불가'],
 ['가려진76점의 crop','대상 visible 비율 중앙값2.15%','전체 crop의 V우세가 일률적 이유는 아님'],
 ['좌표 float32 phase','최대4.70e−5px / mask class 변경0','이번 ROI 수치 오차의 설명력 없음'],
 ['완전 동일 입력 충돌','V77·V78·rawRGB 각각80,016/80,016 unique','유일성은 target식별 가능성 증명이 아님'],
 ['TRAIN에서 본 종류','V78 C recall 38/217 =17.5%','VAL-only 종류만의 문제 아님'],
 ['TRAIN에서 없는 종류6·11','V78 C recall23/70 =32.9%','GT objectID는 저장 결과 분류에만 사용']])
body+='<p>전체 primary에서는 q 중심이 M0 밖에 있는 경우도 있습니다. 자연 YCB VAL의 reference C287 중 qC170·qV9·qOutside108입니다. 이 108점을 semantic C로 새로 라벨링하지 않았습니다. 기존 primary와 분모는 모두 유지했습니다.</p>'
body+='<h2>고정 네 사례의 실제 RGB와 원본 mask</h2><p>V78의 네 GT/예측 cell에서 각각 SHA 최소 사례를 RGB 확인 전에 고정했습니다. 59·17·357·102개 cell의 예시이며 오류 비율을 대표하는 임의 표본은 아닙니다. 아래 마스크는 입력이 아니라 GT 진단용입니다.</p>'
texts=['파란 주전자의 해당 위치는 빨간 드릴에 가려져 있습니다. q-r 2.19px인데 V78 pV=.540으로 visible을 출력했습니다. target128 안에 주전자와 드릴이 함께 있습니다.','흰 세정제를 겨자병이 가린 위치입니다. q-r 1.17px, 세 후보 모두 covered로 맞혔습니다.','참치캔의 해당 위치가 보입니다. q-r 1.53px, V78은 visible로 맞혔지만 V76/V77은 covered였습니다.','나무블록 윗면의 해당 위치가 보입니다. q-r 2.28px, V78 pV=.336으로 covered를 출력했습니다. V77은 visible을 맞혔습니다.']
for i,(c,t)in enumerate(zip(gal['cases'],texts),1):
 f=next((R/'natural_failure_gallery').glob(f'{i}_*_PANEL_READABLE_V2.png'));body+=f'<h3>{i}. {html.escape(t)}</h3><img src="{f.relative_to(R)}" alt="source RGB/crop/GT mask case{i}">'
body+='<h2>다음 대조: 공간 감독 추가</h2><p>같은 V77 primary head·입력·초기 가중치·96개 sampler·2,000회 학습·체크포인트 선택·0.5 문턱을 유지합니다. 기존 spatial24×9×9에서 50개 파라미터의 보조 출력만 추가해 예상 support M0와 M0 내부의 가시 여부를 감독합니다. 밖/무효를 가림 음성으로 만들지 않습니다. 보조 loss 계수0.25는 실행 전에 고정하며 탐색하지 않습니다.</p><p>이는 같은 정보에서 공간 감독이 도움이 되는지 확인하는 실험입니다. 명시적인 대상 식별 정보가 새로 생기지는 않습니다. 마스크는 loss에만 쓰고 추론 입력에 넣지 않습니다. 자연 가림 C precision≥50%, C recall≥80%, V recall≥80%, controlled pair 양쪽 정답≥70%를 모두 통과해야 현재 평가 재실행을 검토합니다.</p><p>GroundingDINO/SAM2는 대상 category라는 입력 정보를 추가하는 별도 후보로 보존했습니다. pallet 상수 category 또는 모든 YCB category를 고정 적용하는 설계가 필요합니다. 검출 실패를 곧바로 가림으로 만들지 않으며, 평가 GT category/mask/template로 대상을 고르지 않습니다.</p>'
body+='<h2>자료·감사</h2><p>mask/cache 유효 영역을 별도로 비교했고 NN 밖 ROI9 3,294개는 제외했습니다. RGB96의 음수 반 픽셀 경계199,455개도 별도 집계했습니다. 2,000 batch/192,000 query draw의 RGB96 pixel 노출은 V28.78%·C15.09%·Outside56.13%입니다. 픽셀 수는 독립 관측 수가 아닙니다.</p><p>원본 public mask의 생성 모드·depth tolerance는 미확인입니다. 이번 네 사례만으로 모든 mask가 정확하다거나 단일 전이 실패 원인이 확정됐다고 주장하지 않습니다. 사용자 확인: 현재223장은 직접 촬영했고 KU Leuven 자료와 무관합니다.</p><ul>'
links=['ROOT_DIAGNOSIS_AND_NEXT_DECISION.json','coverage/results/COVERAGE_FINDINGS_KO.json','coverage/results/REFERENCE_TEACHER_VS_Q_CENTER.json','input_collisions/INPUT_COLLISION_AUDIT.json','object_split_diagnosis/OBJECT_SPLIT_SAVED_RESULTS.json','independent_audits/COVERAGE_AGGREGATE_AND_SCALAR_AUDIT.json','independent_audits/OBJECT_SPLIT_SAVED_RESULTS_INDEPENDENT_AUDIT.json','target_identity_review/TARGET_GROUNDING_FEASIBILITY_PRIMARY_SOURCES.json']
body+=''.join(f'<li><a href="{x}">{x}</a></li>'for x in links)+'</ul><p><b>상태: 진단 완료, 전체 문제는 미해결. V80 source 통제실험으로 계속 진행.</b></p></html>'
(R/'RESULTS_KO.html').write_text(body)
write(R/'STATUS.json',dict(stage='V79_FULL_NATURAL_OCCLUSION_AUDIT_COMPLETE',fully_solved=False,goal_turn='PROGRESS',next=bind(decisions),current223_reads=0,new_forward=0,new_fit=0,PnP=0))
print(bind(R/'RESULTS_KO.html'))
