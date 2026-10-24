"""Compact Korean source-only readout; saved JSON/images only, no model/private IO."""
from pathlib import Path
import hashlib,html,json
V=Path(__file__).resolve().parent.parent;P=V.parent
V96=P/'query_visibility_query_anchored_cue_20261023_v96';V97=P/'query_visibility_expected_cue_triplets_20261023_v97'
RUN=V/'runs/C4_C1_WARMSTART_SOURCE_V1';RGB=V97/'render_preparation/actual_matched_rgb_triplet';QA=V97/'render_preparation/saved_actual_visual_QA'
FIT_SHA='7ac024621ff48f78d8d8f0fab0d173336af645c486cd27c2f316b2795175048d';RENDER_SHA='fac353b66ea3435e9c118ae56951d84ecc1bc3857be748920e3cd90e07001a84'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def bound(b):return read(check(b))
def esc(x):return html.escape(str(x))
def pct(x):return '—'if x is None else f'{100*x:.1f}%'
def href(p):return __import__('os').path.relpath(Path(p).resolve(),V.resolve())
def link(p,label):return '<a href="'+html.escape(href(p),quote=True)+'">'+esc(label)+'</a>'
def correct(m,k):
 q=m['five_cohorts'][k];target=1 if k in('BEFORE_E','COVER_E')else 0;cm=q['matrix'];return f'{cm[target][target]}/{sum(cm[target])}',q['E_recall']if target else q['D_recall']
def table(headers,rows):return '<table><tr>'+''.join('<th>'+esc(x)+'</th>'for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</table>'
def main():
 assert sha(RUN/'FIT_COMPLETE.json')==FIT_SHA and sha(RGB/'RENDER_SEAL.json')==RENDER_SHA
 fit=read(RUN/'FIT_COMPLETE.json');assert fit['completed']and fit['PASS_execution']and fit['selected_update']==2000
 source=bound(fit['source_results']);gate=bound(fit['source_gates']);parity=bound(fit['update0_parity']);root_review=read(V/'ROOT_SAVED_EXECUTION_REVIEW.json')
 assert gate['PASS']is False and sum(x['PASS']for x in gate['gates'].values())==3
 assert parity['PASS']and parity['all31680_logit_bitexact']and parity['all31680_availability_bitexact']and root_review['PASS']
 base=read(V96/'source_baselines/C1_FROZEN_ON_V96/SOURCE_RESULTS.json');render=read(RGB/'RESULT.json');qa=read(QA/'ACTUAL_VISUAL_OBSERVATIONS_AND_DRIFT.json')
 assert render['target_aux7_states']=={'BEFORE':'D','COVER':'E','SHAM':'D'}and render['native8_mapping']=='UNKNOWN'and render['training_admission']is False
 cm=source['metrics']['VALIDATION']['all_known']['matrix'];assert cm==[[37,115,0],[16,105,0]]
 val=source['metrics']['VALIDATION'];bv=base['metrics']['VALIDATION'];rows=[]
 items=[('LOCO 원본 D','LOCO_BEFORE_D','LOCO_before_D_recall'),('Pallox 원본 D','PALLOX_BEFORE_D','PALLOX_before_D_recall'),('단서 보존 SHAM D','SHAM_D','sham_D_recall'),('자연 원본 E','BEFORE_E','natural_before_E_recall'),('제어 COVER E','COVER_E','cover_E_recall')]
 for title,cohort,g in items:
  old,rold=correct(bv,cohort);new,rnew=correct(val,cohort);goal=gate['gates'][g]
  rows.append([title,old+' · '+pct(rold),new+' · '+pct(rnew),'≥'+pct(goal['minimum']),'통과'if goal['PASS']else'미통과'])
 for title,k in [('세 장 모두 정답','accepted_VAL_pair_all3_correct_rate'),('known 예측 보류율','known_VAL_U_rate')]:
  g=gate['gates'][k]
  if title=='세 장 모두 정답':old=f"{bv['paired']['all3_correct']}/{bv['paired']['accepted_pairs']}";new=f"{val['paired']['all3_correct']}/{val['paired']['accepted_pairs']}"
  else:old=pct(bv['all_known']['known_U_rate']);new=pct(val['all_known']['known_U_rate'])
  rows.append([title,old,new,('≥'+pct(g['minimum']))if'minimum'in g else('≤'+pct(g['maximum'])),'통과'if g['PASS']else'미통과'])
 figures=[RGB/'MATCHED_FULL_RGB_CONTACT.png',QA/'AUX7_SAVED_UNMARKED_RGB96_LARGE_CONTACT.png']
 for p in figures:assert p.exists()
 result=dict(completed=True,goal_status='active',fully_solved=False,status='HOLD',candidate='C4_C1_WARMSTART_LR1e_5',source_only=True,current_new_access_or_inference=0,current_latest_actual='V94_C1',PnP=0,C4=dict(fit_complete=bind(RUN/'FIT_COMPLETE.json'),source_results=fit['source_results'],source_gates=fit['source_gates'],initial_C1_parity=fit['update0_parity'],root_saved_review=bind(V/'ROOT_SAVED_EXECUTION_REVIEW.json'),initial_C1_all31680_bitexact=True,updates=2000,lr=1e-5,selected_update=2000,seconds=fit['seconds'],source_gates_pass=3,source_gates_fail=4,VAL_CM=cm,VAL_five_cohorts=val['five_cohorts'],VAL_paired=val['paired'],source_current_admission=False,conditional_AUX_nativeUNKNOWN_humanfalse=True),V97=dict(render_seal=bind(RGB/'RENDER_SEAL.json'),geometry_result=bind(RGB/'EXPECTED_LOCAL_CUE_GEOMETRY_GT_ONLY.json'),visual_and_drift_receipt=bind(QA/'ACTUAL_VISUAL_OBSERVATIONS_AND_DRIFT.json'),AUX7_states=render['target_aux7_states'],AUX7_S_state_count=0,other_AUX_S_U_retained=True,q_origin='V96 source-emulated virtual AUX8; not actual initializer/native8 or corrected current queries',all8original_q_L_preserved=True,before_newly_rendered=True,V96_old_RGB_cache_reused=False,before_vs_V96_drift=qa['before_vs_priorV96_RGB'],in_C4_training=False,training_admission=False,native8_mapping='UNKNOWN',human_gold=False,figures=[bind(p)for p in figures]),conclusion_KO='C1에서 작은 학습률로 시작해도 C4는 원본/SHAM의 보임을 보존하지 못해 소스 기준을 통과하지 못했습니다. 별도 V97 렌더는 단서와 가림을 연결한 자료 구성의 실행가능성만 보여주며 학습 성능이나 전이 성공이 아닙니다.',limits_KO=['194rank쌍은 학습10/검증18개 원 query에 중첩돼 있고 성공 controls는 모두Pallox입니다. LOCO controlled E는0이라 미검증입니다.','자연 E는TRAIN2/VAL2이며, 조건부 AI AUX 라벨은 사람 D/E/S 또는 native8 gold가 아닙니다.','초기화와 학습률을 함께 바꾼 소스 실험이며 이 원인만의 효과를 분리하지 않습니다. 반복 노출된 sourceVAL입니다.','V97은 onecamera/procedural appearance/GT기하로 배치한 floating foreign object 한쌍입니다. 그림자와 crop내 물체면적도 완전히 같지 않습니다.','저장 검산 PASS는 수치·입력·선택 재현 확인이며 과학적 일반화 검증이 아닙니다.'])
 images=''.join('<figure><a href="'+href(p)+'"><img loading="eager" src="'+href(p)+'" alt="'+esc('V97 실제 full triplet'if i==0 else'V97 고정 AUX7 unmarked crop triplet')+'"></a></figure>'for i,p in enumerate(figures))
 page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V98 C4 소스 실패와 별도 단서 렌더</title><style>body{{font-family:"Nanum Gothic","Noto Sans CJK KR",sans-serif;max-width:1080px;margin:30px auto;padding:0 24px;line-height:1.65;color:#192d3d}}h1{{font-size:29px}}h2{{font-size:22px;margin-top:30px}}.lead{{padding:16px;background:#fff0dd;border-left:5px solid #b66d22}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #cbd6e0;padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#edf2f7}}img{{display:block;max-width:100%;height:auto}}figure{{margin:18px 0}}a{{color:#145d9d}}.small{{font-size:13px;color:#4b5f6d}}</style>
<h1>V98 · C4 소스 기준 미통과와 별도 단서 렌더</h1>
<p class="lead"><b>HOLD · C4는7개 기준 중4개를 통과하지 못했습니다.</b><br>C1의 시작 출력을 정확히 재현했지만, 낮은 학습률로도 원본과 단서 보존 대조의 보임을 유지하지 못했습니다. 새 current223 추론은0이며 최신 실제 현재 평가는V94 C1입니다.</p>
<h2 id="source">C1에서 시작한 C4의 실제 결과</h2>
<p>동일26,573파라미터·같은31,680개 소스 항목을 사용했습니다. optimizer 전 모든 logit·availability가 C1과 bitexact임을 확인한 뒤 AdamW 학습률1e−5로2,000회 학습했습니다({fit['seconds']:.2f}초). 다섯 VAL 계층 BCE로 step2000을 선택했지만 보임 보존과 세 장 구별 기준은 실패했습니다. 초기화와 학습률을 함께 바꿨으므로 한 요인의 인과 효과를 입증하는 실험은 아닙니다.</p>
{table(['고정 VAL 항목','C1 동일 입력 진단','C4 선택2000','사전 C4 기준','판정'],rows)}
<p class="small">D/E는 query에 연결된 조건부 AUX의 보임/가림이며 native/human gold가 아닙니다. unknown GT는 E로 바꾸지 않습니다. known VAL273개 혼동행렬은 <b>[[37,115,0],[16,105,0]]</b>(행D/E, 열D/E/U)입니다. 알려진 원본은D33/E2이고,194rank쌍은 학습10·검증18개 원 query에 중첩됩니다. 성공한 controls는 전부Pallox이며 LOCO controlled E는 미검증입니다.</p>
<p>{link(V/'ROOT_SAVED_EXECUTION_REVIEW.json','Root 저장 검산 PASS')}는 입력·2000회 추출배치·21회 selector 및 부호/U 재현을 확인한 기록입니다. source gate 통과나 의미 라벨·일반화의 검증을 뜻하지 않습니다.</p>
<h2 id="triplet">별도 V97: 같은 q에서 실제 RGB 단서 가림·보존 구성</h2>
<p>기존 mesh·camera·절차적 목재 appearance를 고정하고, 같은 cardboard cuboid를 고정한 국소 구조 모서리 앞에 놓거나 옆으로 옮겼습니다. 얼굴 patch·두 edge arm·endpoint의 GT-only 기하 witness로 관심 AUX7이 <b>D→E→D</b>임을 확인했습니다. AUX7의 세 관측에서 S 상태는0이며, 다른 AUX의 S/U는 그대로 보존했습니다. whole-part mask로 이 단서의 상태를 대신 판정하지 않았습니다.</p>
<p><b>이 triplet은 C4 학습에 포함되지 않았습니다.</b> 라벨·appearance 구성의 실행가능성 한 예이며 모델 정확도가 아닙니다. 모든8개 q/L은 V96의 source-emulated virtual AUX 좌표 그대로이고, 배포 initializer 출력이나 native8 정답·사람의 D/E/S가 아닙니다.</p>
{images}
<p class="small">새 BEFORE는 새로 렌더한 baseline입니다. V96과2,771채널(994픽셀, 최대차36)이 달라 과거 RGB 캐시를 그대로 재사용했다고 간주하지 않습니다. 원인은 확정하지 않았고 교체·재시도도 하지 않았습니다. 같은 object여도 SHAM 위치에서 그림자와 crop내 투영 면적은 달라집니다.</p>
<p>{link(RGB/'ALL8_UNCHANGED_QUERY_RGB96_CONTACT.png','모든8개 원 q/L crop')} · {link(RGB/'EXPECTED_LOCAL_CUE_GEOMETRY_GT_ONLY.json','국소 단서 기하 결과')} · {link(QA/'ACTUAL_VISUAL_OBSERVATIONS_AND_DRIFT.json','실제 관찰과 RGB drift')}</p>
<h2 id="limits">이번 단계의 결론</h2>
<p>C4는 가림 recall을 높이는 동안 보임 보존과 SHAM 구별을 잃었습니다. 주변 물체·합성 흔적 등에 대한 반응이나 자료별 과적합과 양립하지만 단독 원인을 증명하지는 않습니다. 별도 V97은 목표 단서와 가림을 연결하는 작은 자료 구성 경로를 실제로 보인 단계이며, 학습·전이 성공으로 승격하지 않습니다.</p>
<p>현재 평가·학습 선택·threshold 변경·PnP는 추가 실행하지 않았습니다. 목표는 계속 진행 중입니다. {link(V96/'ROOT_READOUT_KO.html','이전 V96 소스 보고서')} · {link(P/'query_visibility_matched_query_scalar_20261023_v94/ROOT_READOUT_KO.html','최신 실제 현재 평가 V94 C1')} · {link(V/'ROOT_RESULTS_KO.json','정량 JSON')}</p></html>'''
 (V/'ROOT_RESULTS_KO.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');(V/'ROOT_READOUT_KO.html').write_text(page)
 print(json.dumps(dict(report=bind(V/'ROOT_READOUT_KO.html'),results=bind(V/'ROOT_RESULTS_KO.json'),source_gate_PASS=False,current_new=0)))
if __name__=='__main__':main()
