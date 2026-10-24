"""Korean V96 source readout; saved JSON/scalar-score NPZ and image bindings only.
Never loads image pixels, model parameters, optimizer or private/current artifacts.
"""
from pathlib import Path
import argparse,hashlib,html,json,math
import numpy as np
V=Path(__file__).resolve().parent.parent
C1=V/'source_baselines/C1_FROZEN_ON_V96';C3=V/'runs/C3_QUERY_ANCHORED_SOURCE_V1'
COHORTS=('LOCO_BEFORE_D','PALLOX_BEFORE_D','SHAM_D','BEFORE_E','COVER_E')
CN={'LOCO_BEFORE_D':'LOCO 원본 D','PALLOX_BEFORE_D':'Pallox 원본 D','SHAM_D':'단서 보존 대조 D','BEFORE_E':'자연 원본 E','COVER_E':'제어 가림 E'}
GN={'LOCO_before_D_recall':'LOCO 원본 D recall','PALLOX_before_D_recall':'Pallox 원본 D recall','natural_before_E_recall':'자연 원본 E recall','sham_D_recall':'단서 보존 대조 D recall','cover_E_recall':'제어 가림 E recall','accepted_VAL_pair_all3_correct_rate':'VAL 세 장 모두 정답','known_VAL_U_rate':'VAL known 보류율'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
 p=Path(b['path']);assert p.resolve().is_relative_to(V.resolve()),'Only V96 source report inputs permitted'
 assert sha(p)==b['sha256'];return p

def read(p):return json.loads(Path(p).read_text())
def bound_json(b):return read(checked(b))
def esc(x):return html.escape(str(x))
def pct(x):return '—'if x is None else f'{x*100:.1f}%'
def num(x):return '—'if x is None else f'{x:.4f}'
def row(cells):return '<tr>'+''.join('<td>'+esc(x)+'</td>'for x in cells)+'</tr>'
def table(headers,rs):return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr></thead><tbody>'+''.join(row(x)for x in rs)+'</tbody></table></div>'
def href(p):return Path(p).resolve().relative_to(V.resolve()).as_posix()
def link(p,label):return '<a href="'+html.escape(href(p),quote=True)+'">'+esc(label)+'</a>'
def completed(path,expected):
 assert expected and sha(path)==expected,'Completion SHA must be provided by root before saved-result bodies are read'
 x=read(path);assert x['completed']and x['PASS_execution'];return x

def scores(b):
 with np.load(checked(b),allow_pickle=False)as z:
  keys=('query_logit','query_available','scalar_target_D0_E1_Uminus1','source','split','variant','occurrence_index','occurrence_id','accepted_pair')
  x={k:np.array(z[k],copy=True)for k in keys}
 assert x['query_logit'].shape==x['query_available'].shape==(31680,)and x['accepted_pair'].shape==(960,16)
 assert np.array_equal(x['occurrence_index'],np.arange(31680))and np.isfinite(x['query_logit']).all()
 return x

def cohort_BCE(x,split):
 y=x['scalar_target_D0_E1_Uminus1'];v=x['variant'];s=x['source'];m=x['split']==split
 masks=dict(LOCO_BEFORE_D=m&(s=='LOCO')&(v=='BEFORE')&(y==0),PALLOX_BEFORE_D=m&(s=='PALLOX')&(v=='BEFORE')&(y==0),SHAM_D=m&(v=='CUE_PRESERVING_SHAM')&(y==0),BEFORE_E=m&(v=='BEFORE')&(y==1),COVER_E=m&(v=='OPAQUE_COVER')&(y==1))
 z=np.where(x['query_available'],x['query_logit'],0.).astype(np.float64);loss=np.logaddexp(0.,z)-y*z
 return {k:dict(N=int(a.sum()),mean_BCE=float(loss[a].mean())if a.any()else None)for k,a in masks.items()}

def main(c1_sha,c3_sha,final_path=None,final_sha=None):
 # Hash completion files first, then read only their bound source artifacts.
 done1=completed(C1/'BASELINE_COMPLETE.json',c1_sha);done3=completed(C3/'FIT_COMPLETE.json',c3_sha)
 one=bound_json(done1['results']);three=bound_json(done3['source_results']);gate=bound_json(done3['source_gates'])
 x1=scores(done1['scores']);x3=scores(done3['scores'])
 for k in('scalar_target_D0_E1_Uminus1','source','split','variant','occurrence_index','occurrence_id','accepted_pair'):assert np.array_equal(x1[k],x3[k]),k
 inv=read(V/'QUERY_ANCHORED_INVENTORY_SUMMARY.json');intake=read(V/'actual_query_anchored_image_intake/IMAGE_ONLY_SUMMARY.json');pack=read(V/'actual_source_pair_input_pack/PACK_MANIFEST.json')
 assert inv['labels']=={'0':61,'1':4,'-1':895}and intake['accepted_pairs']==194 and pack['occurrences']==31680
 assert intake['by_source_raw_split']['PALLOX|TRAIN_LAB']['accepted']==75 and intake['by_source_raw_split']['PALLOX|VAL_HALL']['accepted']==119
 assert intake['by_source_raw_split']['LOCO|TRAIN']['accepted']==intake['by_source_raw_split']['LOCO|VALIDATION']['accepted']==0
 assert three['selected_update']==done3['selected_update']==0 and len(gate['gates'])==7
 assert sum(x['PASS']for x in gate['gates'].values())==3 and gate['PASS']is False
 history=[json.loads(s)for s in checked(done3['history']).read_text().splitlines()if s]
 assert [r['update']for r in history]==list(range(0,2001,100))
 assert all(r['criterion_five_VAL_cohorts_BCE']>math.log(2)for r in history[1:])
 nll={name:{s:cohort_BCE(x,s)for s in('TRAIN','VALIDATION')}for name,x in [('C1',x1),('C3',x3)]}
 assert abs(np.mean([r['mean_BCE']for r in nll['C3']['VALIDATION'].values()])-three['selector_five_VAL_cohorts_BCE'])<1e-14
 distinct={s:int(np.any(x3['accepted_pair'][x3['split'][::33]==s],axis=1).sum())for s in('TRAIN','VALIDATION')}
 preview=read(V/'actual_query_anchored_image_intake/fixed_order_preview/PREVIEW_MANIFEST.json')
 for c in preview['cases']:checked(c['figure'])
 prototype=V/'synthetic_appearance_feasibility/ACTUAL_ONE_PROTOTYPE_V3';proto=read(prototype/'PROTOTYPE_RESULT.json')
 bindings={k:bind(V/p)for k,p in dict(inventory='QUERY_ANCHORED_INVENTORY_SUMMARY.json',inventory_seal='FINAL_METADATA_INVENTORY_SEAL.json',image_intake_summary='actual_query_anchored_image_intake/IMAGE_ONLY_SUMMARY.json',image_intake_seal='actual_query_anchored_image_intake/IMAGE_ONLY_INTAKE_SEAL.json',input_pack='actual_source_pair_input_pack/PACK_MANIFEST.json',input_pack_seal='actual_source_pair_input_pack/INPUT_PACK_SEAL.json',source_design='C3_DESIGN_BEFORE_SOURCE_FIT.json',preview='actual_query_anchored_image_intake/fixed_order_preview/PREVIEW_MANIFEST.json',prototype_result='synthetic_appearance_feasibility/ACTUAL_ONE_PROTOTYPE_V3/PROTOTYPE_RESULT.json',prototype_seal='synthetic_appearance_feasibility/FEASIBILITY_FINAL_SEAL.json').items()}
 result=dict(schema='V96_KOREAN_SOURCE_READOUT',completed=True,goal_status='active',fully_solved=False,current223_new_access_or_inference=0,latest_actual_current_evaluation='V94_C1',PnP=0,original_q_L_slot_candidate_order_preserved=True,native_expected_part_state='UNKNOWN',human_gold=False,source_target='Unchanged query central4 native bilinear footprint contained in one matched certified physical-cue support/uncertainty group; conditional AUX only.',before_D_E_U=[61,4,895],accepted_rank_pairs=dict(TRAIN=75,VALIDATION=119,total=194,all_source='PALLOX',distinct_original_queries=distinct,LOCO_controlled_pairs=0,LOCO_controlled_generalization='UNTESTED'),all_occurrences=dict(total=31680,D=255,E=198,U=31227,known=453),feature_pack=dict(unique_units=pack['unique_input_units'],unchanged_ref_occurrences=pack['counts']['base_reused'],changed_fullRGB_matching_units=pack['counts']['changed'],source_predict_calls=pack['source_predict_calls'],repeat_calls=pack['first_repeat_calls'],blank_warmup_calls=pack['blank_warmup_calls']),bindings=bindings,C1=dict(completion=bind(C1/'BASELINE_COMPLETE.json'),results=done1['results'],scores=done1['scores'],selected_update=100,diagnostic_not_selector=True,metrics=one['metrics']),C3=dict(completion=bind(C3/'FIT_COMPLETE.json'),results=done3['source_results'],scores=done3['scores'],selected_update=0,updates=2000,seconds=done3['seconds'],selector_five_VAL_BCE=three['selector_five_VAL_cohorts_BCE'],learned_checkpoint_N=20,learned_all_worse_than_ln2=True,metrics=three['metrics']),cohort_BCE=nll,source_gates=gate,current_inference_authorized=False,synthetic_prototype_in_C3=False,conclusion_KO='query와 단서를 연결한 제어 자료는 늘었지만 C3 학습 상태는 채택되지 않았습니다. 공개 소스 일반화와 실제 팔레트 목표 해결은 입증하지 못했습니다.',limits_KO=['194개는28개 원 query에서 여러 donor rank로 만든 쌍이며 독립적인194장면이 아닙니다.','성공한 제어쌍은 모두 Pallox입니다. LOCO controlled E는 학습·검증 모두0이라 미검증입니다.','원본 자연E는 TRAIN2/VAL2로 매우 작고, 대상은 조건부 AI 보조 단서입니다. Native8와 사람 D/E/S gold가 아닙니다.','자료·라벨 admission·sampler·selector를 함께 바꿔 단일 요인의 효과로 해석할 수 없습니다.','Pallox는 동일 물리 bin의LAB/HALL 계열이며, 다른 팔레트 구조나 자연 가림 전이를 보장하지 않습니다.','절차적 목재 렌더는 별도 실행가능성 결과이며 이번 C3에 포함되지 않았습니다.'])
 auditfiles={'selected_source_saved_audit':V/'independent_actual_source_fit_audit/REVIEW.json','fixed_final_saved_audit':V/'independent_actual_source_fit_audit/FINAL2000_DIAGNOSTIC_AUDIT/REVIEW.json'}
 for path in auditfiles.values():assert read(path)['PASS']is True
 result['independent_saved_audits']={k:bind(path)for k,path in auditfiles.items()}
 result['audit_scope_KO']='입력·저장 점수·분모·선택 절차의 재현 검산이며 의미 라벨의 gold 정당성이나 과학적 일반화 검증을 뜻하지 않습니다.'
 final_html='';failure_html=''
 if final_path is not None or final_sha is not None:
  assert final_path is not None and final_sha is not None
  final_path=Path(final_path);assert final_path.resolve().is_relative_to(V.resolve());fd=completed(final_path,final_sha)
  rb=fd.get('results',fd.get('source_results'));assert rb is not None
  fr=bound_json(rb);result['fixed_final2000_diagnostic']=dict(completion=bind(final_path),results=rb,not_selector=True,authoritative_selected_update=0,criterion=fr['five_VAL_cohorts_BCE_diagnostic'],history_criterion_difference=fr['history_criterion_difference'],metrics=fr['metrics'])
  rr=[]
  for split in('TRAIN','VALIDATION'):
   m=fr['metrics'][split];p=m['paired'];rr.append([split,m['all_known']['matrix'],pct(m['five_cohorts']['LOCO_BEFORE_D']['D_recall']),pct(m['five_cohorts']['PALLOX_BEFORE_D']['D_recall']),pct(m['five_cohorts']['BEFORE_E']['E_recall']),pct(m['five_cohorts']['SHAM_D']['D_recall']),pct(m['five_cohorts']['COVER_E']['E_recall']),f"{p['all3_correct']}/{p['accepted_pairs']}"])
  final_html='<h2 id="final2000">선택과 별개인 고정 step 2000 진단</h2>'+'<p>마지막 상태 하나를 동일 공개 소스에서만 읽은 진단입니다. 선택 step 0을 바꾸거나 새 모델을 고르지 않습니다.</p>'+table(['분할','혼동행렬','LOCO D','Pallox D','자연 E','대조 D','가림 E','세 장 정답'],rr)
  fm=fr['metrics']['VALIDATION'];ft=fr['metrics']['TRAIN'];final_html+=f'<p>학습 쌍은{ft["paired"]["all3_correct"]}/{ft["paired"]["accepted_pairs"]}, 검증은{fm["paired"]["all3_correct"]}/{fm["paired"]["accepted_pairs"]}입니다. 검증 가림 E는111/119인데 단서 보존 대조 D는49/119입니다. 자연 E는2/2뿐입니다. 다섯 계층 BCE {fr["five_VAL_cohorts_BCE_diagnostic"]:.6f}는 원래 step2000 history와 정확히 일치합니다. 이 차이는 자료별 과적합이나 편법 단서 사용 가능성을 제기하지만, 원인을 증명하지는 않습니다.</p>'
  failure=V/'source_diagnosis/ACTUAL_CROP_FAILURE_REVIEW';receipt=read(failure/'ROOT_ACTUAL_CROP_VIEW.json')
  assert sha(failure/'ROOT_ACTUAL_CROP_VIEW.json')=='ec83be2289a5b3d4aa5257f855c87783d21d9155325b358764cbeccff5fd0fdd'
  selection=read(failure/'SELECTION_BEFORE_CROP_READS.json');assert selection['diagnostic_only']and len(selection['selected'])==4
  for f in receipt['root_actually_viewed']:checked(f)
  result['fixed_final_failure_crop_review']=dict(root_receipt=bind(failure/'ROOT_ACTUAL_CROP_VIEW.json'),selection=bind(failure/'SELECTION_BEFORE_CROP_READS.json'),comparison=receipt['comparison'],figures=receipt['root_actually_viewed'],posthoc_diagnostic_only=True,model_threshold_training_row_selection_used=False)
  failure_html='<h2 id="failure_crops">마지막 상태의 실제 저장 crop 실패 4개</h2><p>고정 마지막 step2000에서 원본 D·가림 E·대조 E가 나온 첫 4개 distinct VAL base와 그 첫 rank를 저장 점수로 고른 사후 진단입니다. 원래 q/L 및 실제 모델 RGB96을 그대로 표시했고 새 좌표·정답 crop을 찾지 않았습니다. 기대 대조 라벨은 조건부 AUX D입니다.</p><p>실제 접촉표 검토기록에서는 SHAM에서도 query 인근 구조 모서리 단서가 남아 있는데 가림 E가 출력되었습니다. 일부 query 중앙은 인접 바닥이므로 중심 픽셀 재질을 정답으로 삼지는 않습니다. 주변 물체 존재나 합성 흔적에 대한 지름길 사용 및 과적합과 양립하는 관찰이지만, 단독 원인 증명은 아닙니다. 새 모델·문턱·학습 행 선택에 사용하지 않았습니다.</p>'
  for f in receipt['root_actually_viewed']:
   path=checked(f);failure_html+='<figure><a href="'+href(path)+'"><img loading="eager" src="'+href(path)+'" alt="고정 마지막2000의 실제 모델 입력 실패 비교"></a></figure>'
  failure_html+='<p class="small">'+link(failure/'ROOT_ACTUAL_CROP_VIEW.json','실제 시각 검토기록')+' · '+link(failure/'SELECTION_BEFORE_CROP_READS.json','사후 고정 선택규칙')+'</p>'
 else:
  final_html='<h2 id="final2000">고정 step 2000 진단</h2>'<p>선택 결과와 분리한 저장 진단은 완료 SHA를 받은 뒤에만 표에 추가합니다. 이 보고서는 그 결과를 추정하거나 마지막 상태를 재선택하지 않습니다.</p>'
 cohort_html=''
 for split,title in [('TRAIN','학습 자료'),('VALIDATION','고정 검증 자료')]:
  rr=[]
  for c in COHORTS:
   data=[]
   for model,m in [('C1',one),('C3',three)]:
    q=m['metrics'][split]['five_cohorts'][c];cm=q['matrix'];k=1 if c in('BEFORE_E','COVER_E') else 0
    data += [f"{cm[k][k]}/{sum(cm[k])}",pct(q['E_recall']if k else q['D_recall']),int(cm[0][2]+cm[1][2]),num(nll[model][split][c]['mean_BCE'])]
   rr.append([CN[c]]+data)
  cohort_html+='<h3>'+title+'</h3>'+table(['계층','C1 정답/N','C1 recall','C1 U','C1 BCE','C3 정답/N','C3 recall','C3 U','C3 BCE'],rr)
 allrows=[]
 for split in('TRAIN','VALIDATION'):
  for name,m in [('C1',one),('C3 선택0',three)]:
   q=m['metrics'][split]['all_known'];allrows.append([split,name,q['known_N'],q['matrix'],pct(q['D_precision']),pct(q['D_recall']),pct(q['E_precision']),pct(q['E_recall']),pct(q['known_U_rate'])])
 pairs=[]
 for split in('TRAIN','VALIDATION'):
  for name,m in [('C1',one),('C3 선택0',three)]:
   q=m['metrics'][split]['paired'];pairs.append([split,name,f"{q['all3_correct']}/{q['accepted_pairs']}",pct(q['all3_correct_rate']),'Pallox만 / LOCO 미검증'])
 grows=[[GN[k],pct(q['value']),('≥'+pct(q['minimum']))if'minimum'in q else('≤'+pct(q['maximum'])),'통과'if q['PASS']else'미통과']for k,q in gate['gates'].items()]
 sourcecounts=[[name,d['attempts'],d['accepted']]for name,d in intake['by_source_raw_split'].items()]
 previewlinks=' · '.join(link(checked(c['figure']),f"고정 사례{i+1} · "+('성립'if c['pair_ready']else'실패 U'))for i,c in enumerate(preview['cases']))
 imgs=''
 for index in(3,5):
  c=preview['cases'][index];p=checked(c['figure']);imgs+='<figure><a href="'+href(p)+'"><img loading="eager" src="'+href(p)+'" alt="원본 가림 대조 고정사례'+str(index+1)+'"></a><figcaption>'+('TRAIN'if index==3 else'VAL')+f" 고정 사례 · base {c['base_row_index']}, donor rank {c['donor_rank']} · 원래 q/L 유지. 표시는 보기 전용입니다.</figcaption></figure>"
 summary_table=table(['원본 query','D / E / U','전체 occurrence','known D / E / U','원 q/L','현재223 신규'],[['960','61 / 4 / 895','31,680','255 / 198 / 31,227','그대로','0']])
 body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V96 query 단서 연결 소스 실험</title><style>
body{{font-family:"Nanum Gothic","Noto Sans CJK KR",sans-serif;max-width:1220px;margin:34px auto;padding:0 24px;color:#172a3a;line-height:1.65}}h1{{font-size:30px}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:18px;margin-top:25px}}.lead{{background:#fff1df;border-left:5px solid #ae641d;padding:18px}}.small,figcaption{{font-size:13px;color:#485c6b}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd6df;padding:8px;vertical-align:top;text-align:left;overflow-wrap:anywhere}}th{{background:#edf2f7}}.tablewrap{{width:100%}}img{{display:block;max-width:100%;height:auto}}figure{{margin:22px 0}}a{{color:#135c9d}}.prototype{{max-width:640px}}code{{overflow-wrap:anywhere}}</style>
<h1>V96 · 같은 query가 지칭하는 단서로 다시 구성한 공개 소스 실험</h1>
<p class="lead"><b>소스 기준 미통과: C3는 학습 전 step 0이 선택됐습니다.</b><br>7개 기준 중3개만 통과했습니다. 학습 후20개 저장 후보는 모두 초기 BCE ln(2)보다 나빴습니다. 목표는 진행 중이며, 현재223장에 C3를 실행하지 않았습니다.</p>
{summary_table}
<h2 id="construction">실제로 만든 자료와 입력</h2>
<p>기존960개 query의 좌표·L·논리 슬롯·선택 candidate·분할을 유지했습니다. query 중앙 네 bilinear 샘플의 native footprint 전체가 한 보임/가림 단서의 인증 범위에 들어가는 경우만 조건부 보조 라벨을 부여했습니다. 원본은 D61·E4·U895입니다. 원 native8 정답이나 사람 D/E/S gold는 아닙니다.</p>
<p>원본 D61개마다 미리 배정한16개 donor로976번 시도해194개 rank 쌍이 성립했습니다(학습75, 검증119). 원본·가림·단서 보존 대조의31,680행을 모두 보존했고 실패 after는 U로 남겼습니다. 성공쌍은 모두 Pallox이며 LOCO의96번 시도는0쌍입니다. LOCO controlled E 일반화는 <b>미검증</b>입니다.</p>
{table(['원 source / split','시도 rank 수','성립 rank 쌍'],sourcecounts)}
<p class="small">194개 rank 쌍은 서로 독립인194장면이 아닙니다. 성공한 원 query는 학습{distinct['TRAIN']}개·검증{distinct['VALIDATION']}개이고, 그 위에 여러 donor를 붙였습니다. donor는 학습 odd object/scene0–7, 검증 even object/scene8–9로 분리했지만 대상 Pallox 물리 bin은 공유합니다.</p>
<p>388개 변경 full RGB에는 원래 q/L에서 P3ROI와 RGB96을 새로 추출했습니다. 같은 full RGB인31,292행은 기존 세 입력의 정확한 참조를 재사용했습니다. 저장 입력 단위는1,348개이며, 이 중960개가 원본입니다. 원본 RGB와 edited RGB의 특징을 섞지 않았고 GT·단서 좌표·마스크·donor·분할 정보는 모델 forward에 전달하지 않았습니다.</p>
<h2 id="source_metrics">같은 소스 입력의 C1과 C3</h2>
<p>C1은 V94에서 고정된 step100을 그대로 읽은 진단 기준선입니다. C1이 학습한 이전 known-set 라벨과 이번 query-anchored 라벨은 다르므로, 같은 새 소스 위의 비교가 단일 요인 ablation을 뜻하지는 않습니다. C3는 동일26,573파라미터 모델을 seed194로 새로 초기화해2,000회 학습했습니다({done3['seconds']:.2f}초). 배치64는 LOCO 원본D8·Pallox 원본D8·대조D16·자연E8·제어가림E24입니다. 소스 검증 다섯 계층 BCE의 같은 가중 평균으로만 체크포인트를 골랐습니다.</p>
<p class="lead">선택 step0은 마지막 출력층이0이어서 유효 입력의 raw logit이 모두0입니다. 고정 규칙 D iff z&lt;0, E iff z≥0에 따라 모두 E가 됩니다. 아래 E recall100%는 학습한 가림 인식 성공이 아닙니다. 좌표 연결로 구성 자료는 늘었지만 C3의 소스 일반화는 확보하지 못했습니다.</p>
{cohort_html}
<p class="small">D/E는 query에 연결된 조건부 AUX의 보임/가림입니다. U는 입력/연산 지원이 없어 보류한 예측이며 recall 분모에 포함합니다. GT가 U인31,227행은 가림 정답으로 쓰지 않습니다. BCE는 각 계층 평균이며, unavailable은 selector에서 safe0 BCE=ln2로 계산합니다. 자연E는 TRAIN2·VAL2뿐입니다.</p>
<h3>known 전체의 P/R와 혼동행렬</h3><p class="small">행 [D,E], 열 [D,E,U]. P/R는 이 소스의 known 라벨과 반복 donor occurrence에 조건부입니다.</p>{table(['분할','모델','known N','CM','D P','D R','E P','E R','U율'],allrows)}
<h2 id="pairs">원본·가림·대조 세 장을 모두 맞힌 비율</h2>{table(['분할','모델','모두 정답 / rank 쌍','비율','대상 범위'],pairs)}
<h2 id="gates">사전에 고정한 소스 기준 7개</h2>{table(['기준','실제','요구','판정'],grows)}
{final_html}
{failure_html}
<h2 id="crops">실제 고정 crop 미리보기</h2><p>{previewlinks}</p><p>성립한 TRAIN/VAL의 고정 첫 사례를 아래에 보입니다. 앞선 미성립 사례도 위 링크에 그대로 남겼습니다. pasted 물체는 제어 증강이며 자연 가림 gold가 아닙니다. 다른 목재가 보여도 지정한 단서의 보임과는 다를 수 있습니다.</p>{imgs}
<h2 id="appearance">별도 절차적 목재 렌더: 이번 학습에서 제외</h2>
<p>기존 USD의308개 mesh·110,478개 triangle을 기존 Linux NVIDIA EGL로 렌더했습니다. 숫자형 material factor의 정규화 오류로 어두웠던 V1/V2는 보존하고, float factor를 명시한 V3에서 목재색과 grain이 보이는 실제 RGB를 만들었습니다. part ID는 RGB에 넣지 않고 별도 기하 ownership에 저장했습니다.</p>
<a href="synthetic_appearance_feasibility/ACTUAL_ONE_PROTOTYPE_V3/APPEARANCE_RGB.png"><img class="prototype" loading="eager" src="synthetic_appearance_feasibility/ACTUAL_ONE_PROTOTYPE_V3/APPEARANCE_RGB.png" alt="소스 학습에서 제외한 실제 procedural wood mesh 렌더"></a>
<p class="small">하나의 규칙적 texture·camera·light 예시입니다. authored MDL 복원이나 재질 identity, photorealism, sim2real 효과, native association을 검증한 것이 아닙니다. part152 alone{proto['part152_alone_pixels']}px / visible{proto['part152_visible_pixels']}px는 별도 보조 기하 출력입니다. GPU ID 색상 오류{proto['GPU_color_guard']['invalid_N']}px는 U로 남겼습니다.</p>
<h2 id="limits">해석과 다음 단계의 경계</h2>
<ul>{''.join('<li>'+esc(x)+'</li>'for x in result['limits_KO'])}</ul>
<p>독립 저장 감사 두 건은 모두 PASS입니다. {link(auditfiles['selected_source_saved_audit'],'선택 결과 검산')} · {link(auditfiles['fixed_final_saved_audit'],'고정 마지막 상태 검산')}입니다. 이 PASS는 입력·점수·분모·선택 절차의 재현 검산이며, 의미 라벨의 gold 정당성이나 과학적 일반화 검증을 뜻하지 않습니다.</p>
<p>현재 평가의 최신 실행은 <a href="../query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html">V94 C1 보고서</a>입니다. 이번 V96은 private 현재 영상·GT·점수를 읽거나 checkpoint/threshold 선택에 사용하지 않았고, 새 current 추론·PnP도0입니다. 자기 가림 원인을 학습하거나 D/E/S 전체 목표를 해결했다고 주장하지 않습니다.</p>
<p class="small">{link(V/'ROOT_RESULTS_KO.json','정량 JSON')} · {link(C3/'SOURCE_RESULTS.json','C3 선택 결과')} · {link(C1/'SOURCE_RESULTS.json','고정 C1 기준선')} · {link(V/'actual_query_anchored_image_intake/IMAGE_ONLY_INTAKE_SEAL.json','실제 이미지 구성 봉인')} · {link(V/'actual_source_pair_input_pack/INPUT_PACK_SEAL.json','특징 입력 봉인')} · {link(prototype/'PROTOTYPE_RESULT.json','별도 렌더 근거')}</p></html>'''
 (V/'ROOT_RESULTS_KO.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 (V/'ROOT_READOUT_KO.html').write_text(body)
 print(json.dumps(dict(report=bind(V/'ROOT_READOUT_KO.html'),result=bind(V/'ROOT_RESULTS_KO.json'),selected_update=0,source_gate_PASS=False,current_new=0,final2000_included=final_path is not None)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--c1-complete-sha',required=True);p.add_argument('--c3-complete-sha',required=True);p.add_argument('--final-complete',type=Path);p.add_argument('--final-complete-sha');a=p.parse_args();main(a.c1_complete_sha,a.c3_complete_sha,a.final_complete,a.final_complete_sha)
