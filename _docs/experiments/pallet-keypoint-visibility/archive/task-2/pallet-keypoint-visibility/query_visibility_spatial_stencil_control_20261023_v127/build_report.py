"""Saved public results only; no model, encoder, optimizer or private inputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,html,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parent;P=V.parent
RUN=V/'runs/C19_MATCHED_STENCIL_FIXED1000'
ARMS=('POOLED9_REPEAT8','ORDERED_OUTER8')
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s)for s in p.read_text().splitlines()if s]
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def esc(x):return html.escape(str(x),quote=True)
def link(p,label):return f'<a href="{esc(p)}">{esc(label)}</a>'
root=read(V/'ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json')
rs=rows(Path(root['bindings']['input_occurrences']['path']))
gt=rows(Path(root['bindings']['GT_occurrences']['path']))
y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gt])
known=np.isin(y,[0,1]);split=np.array([r['split']for r in rs]);source=np.array([r['source']for r in rs])
variant=np.array([r['variant']for r in rs]);alpha=np.array([-1 if g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'] is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt])
Eout=(y==1)&(variant=='OPAQUE_COVER')&(alpha==0)
Din=(y==0)&(variant=='CUE_PRESERVING_SHAM')&(alpha==1)
cohorts={'TRAIN':split=='TRAIN','LOCO_VAL':(split=='VALIDATION')&(source=='LOCO'),'TOMIE_DEV':(split=='VALIDATION')&(source=='TOMIE'),'VAL_COMBINED':split=='VALIDATION'}
def metric(z,a,m):
 pred=np.where(a,(z>=0).astype(int),-1)
 cm=np.array([[int((m&(y==j)&(pred==k)).sum())for k in (0,1,-1)]for j in (0,1)])
 div=lambda k,n:float(k/n)if n else None
 result={'CM_rows_D_E_cols_D_E_U':cm.tolist(),'known_planned':int((m&known).sum()),'known_available':int((m&known&a).sum()),'teacher_U':int((m&~known).sum()),'input_U_known':int((m&known&~a).sum()),'D_recall_all_planned':div(cm[0,0],cm[0].sum()),'E_recall_all_planned':div(cm[1,1],cm[1].sum()),'D_precision':div(cm[0,0],cm[:,0].sum()),'E_precision':div(cm[1,1],cm[:,1].sum()),'U_rate_known':div(cm[:,2].sum(),cm.sum())}
 for label,mask,target in [('D_available',y==0,0),('E_available',y==1,1),('E_qOUT_available',Eout,1),('SHAM_D_qIN_available',Din,0)]:
  h=m&mask&a;result[label]={'correct':int((h&(pred==target)).sum()),'N':int(h.sum()),'recall':div(int((h&(pred==target)).sum()),int(h.sum()))}
 q=m&known&a;safe=z[q].astype(np.float64);result['available_mean_BCE']=float((np.logaddexp(0,safe)-y[q]*safe).mean())
 return result,pred

data={};preds={}
for arm in ARMS:
 with np.load(RUN/arm/'FINAL_ALL4896_SOURCE_SCORES.npz',allow_pickle=False)as f:
  z=f['query_logit'];a=f['query_available'];order=f['global_occurrence_index'];p=f['prediction']
 assert np.array_equal(order,np.arange(4896)) and z.dtype==np.float32 and np.isfinite(z).all()
 groups={name:metric(z,a,m)[0]for name,m in cohorts.items()}
 pred=metric(z,a,np.ones(4896,bool))[1];assert np.array_equal(p,pred);preds[arm]=pred
 data[arm]={'groups':groups,'scores':bind(RUN/arm/'FINAL_ALL4896_SOURCE_SCORES.npz'),'fit':bind(RUN/arm/'FIT_COMPLETE.json')}
switches={}
for name,m in cohorts.items():
 switches[name]={}
 for label,h,target in [('E_qOUT',Eout,1),('SHAM_D_qIN',Din,0)]:
  s=m&h&a;pc=preds[ARMS[0]]==target;oc=preds[ARMS[1]]==target
  switches[name][label]={'N':int(s.sum()),'both_correct':int((s&pc&oc).sum()),'ordered_improved':int((s&~pc&oc).sum()),'ordered_regressed':int((s&pc&~oc).sum()),'both_wrong':int((s&~pc&~oc).sum())}
assert data[ARMS[0]]['groups']['VAL_COMBINED']['E_qOUT_available']['correct']==229
assert data[ARMS[1]]['groups']['VAL_COMBINED']['E_qOUT_available']['correct']==279
assert data[ARMS[0]]['groups']['VAL_COMBINED']['SHAM_D_qIN_available']['correct']==148
assert data[ARMS[1]]['groups']['VAL_COMBINED']['SHAM_D_qIN_available']['correct']==97
for q in ['saved_feature_review/REVIEW.json','saved_fit_review/REVIEW.json']:
 assert read(V/q)['PASS']
current=read(P/'LATEST_STATUS_KO.json')['latest_current_evaluation'];assert current['version']=='V124_C16'
results={'completed':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'phase':'V127_C19_MATCHED_LOCAL_STENCIL_SOURCE_CONTROL','goal_status':'active','fully_solved':False,'outcome_KO':'위치별 특징은 가림 정답을 늘렸지만 보임 정답을 줄였다. 두 목표를 함께 개선하는 해법으로 채택하지 않는다.','arms':data,'paired_switches':switches,'current_unchanged':current,
 'controlled_scope':'Both418433params, sameC16zeroextra initialization, same1000wide64draws, samefixedbudget/loss/optimizer/raw0; noVALselection. Information+aggregation comparison, notpureorder/equalinformationrank. C18onlycontextualreference.',
 'actual_work':{'feature_attempt1':{'session_id':24238,'exit_code':1,'encoder_calls_before_API_failure':2,'fit_updates':0},'feature_attempt2':{'session_id':12296,'exit_code':0,'encoder_calls':555,'strict_cached_F_P_bitexact_units':4428},'two_arm_fit':{'session_id':43609,'exit_code':0,'updates_each':1000,'encoder_calls':0},'public_RGB_intake':{'session_id':21744,'exit_code':0,'new_originals':24},'current_V112_PnP_or_q_correction':0},
 'public_diversity_screen':read(V/'ROOT_PUBLIC24_ACTUAL_VISUAL_SCREEN.json')['counts'],
 'next_actions_KO':['등록 검증을 마친 새 메시 두 종류의 BEFORE/COVER/SHAM RGB·depth·owner를 렌더하고 코너별 구조 지지 영역을 실제 이미지에서 검수한다. 기존 파일의 in-frame visibility를 D/E/S로 재사용하지 않는다.','공개 원본24장의 구조 cue를 불확실성 범위와 함께 검토해 합성 가림의 배경·형상 다양성을 늘린다. 공식TRAIN을 유지하며 기존VAL을TRAIN으로 바꾸지 않는다.','현재 평가셋은 학습·정규화·문턱·checkpoint·arm 선택에 쓰지 않는다. PnP나 초기 좌표 보정 없이 외부 가림 성능을 계속 검증한다.'],
 'limits_KO':['TRAIN795는6cue/3원본의 상관 변형이다. VAL도26cue/6원본이며 TOMIE4장은 한 episode이고 반복 개발에 노출됐다.','공개 자료의 라벨은 특정 기대 구조 영역의 조건부AUX이며 native8 인간 정답 또는 q픽셀 재질 정답이 아니다.','nominal418433개 파라미터는 맞췄지만 pooled 반복 입력의 정보 rank는 더 작다. 순서 하나만의 인과 효과로 해석하지 않는다.','기존18 source cohort용 새 특징은 없으므로 old15 회귀 기준 통과를 주장하지 않는다. 이번 두 모델을 현재 실사에서 재평가하거나 배포 모델로 채택하지 않았다.'],
 'evidence':{name:bind(V/rel)for name,rel in {'policy':'ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json','precision':'ROOT_PREEXECUTION_PRECISION_ADDENDUM.json','API_repair':'ROOT_FEATURE_API_CORRECTION_BEFORE_V2.json','features':'stencil_features_v2/FEATURE_COMPLETE.json','feature_review':'saved_feature_review/REVIEW.json','control':'runs/C19_MATCHED_STENCIL_FIXED1000/CONTROL_COMPLETE.json','fit_review':'saved_fit_review/REVIEW.json','public_intake':'public_diversity_intake/FINAL_INTAKE_RECEIPT.json','root_public24_view':'ROOT_PUBLIC24_ACTUAL_VISUAL_SCREEN.json','native_feasibility':'native_target_feasibility/RESULT.json'}.items()}}
results['actual_crop_observations']=read(V/'ROOT_C19_ACTUAL_CROP_OBSERVATIONS_KO.json')
results['new_registered_geometry']=read(V/'ROOT_NEW_REGISTERED_GEOMETRY_REVIEW.json')
results['source_nominal_offset_diagnostic']=read(V/'ROOT_SOURCE_NOMINAL_OFFSET_DIAGNOSIS.json')
results['evidence'].update({name:bind(V/rel)for name,rel in {'case_gallery':'saved_case_gallery/GALLERY_COMPLETE.json','root_actual_crop_view':'ROOT_C19_ACTUAL_CROP_OBSERVATIONS_KO.json','new_registered_geometry':'registered_native_teacher_feasibility/RESULT.json','root_geometry_review':'ROOT_NEW_REGISTERED_GEOMETRY_REVIEW.json'}.items()})
results['evidence']['source_nominal_offset_diagnostic']=bind(V/'ROOT_SOURCE_NOMINAL_OFFSET_DIAGNOSIS.json')
out=V/'visual_followup';out.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(2,2,figsize=(13,8.4));fig.suptitle('공개 자료의 위치별 특징 비교: 가림 개선과 보임 손실',fontsize=20)
names=['TRAIN','LOCO_VAL','TOMIE_DEV'];xs=np.arange(3)
for ax,key,title in zip(axes.flat,['D_available','E_available','E_qOUT_available','SHAM_D_qIN_available'],['전체 보임 recall','전체 외부 가림 recall','어려운 가림: q는 가림체 밖','어려운 보임: q는 가림체 안']):
 for j,arm in enumerate(ARMS):
  vals=[data[arm]['groups'][n][key]for n in names];bars=ax.bar(xs+(j-.5)*.34,[v['recall']*100 for v in vals],.32,label=['주변 평균','위치별 유지'][j],color=['#698197','#d07d39'][j])
  for bar,value in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f"{value['correct']}\n/{value['N']}",ha='center',fontsize=9)
 ax.set_title(title,fontsize=13);ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);ax.set_xticks(xs,['학습 자료','LOCO 검증','TOMIE 개발']);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0,0].legend(loc='upper right',fontsize=10)
fig.text(.5,.025,'두 모델 모두 고정 1,000회 · 학습 분모는 입력 가용 자료 · 검증 모두 입력 가용 · 공개 조건부 구조 cue, 현재 실사 성능 아님',ha='center',fontsize=10)
fig.tight_layout(rect=[.01,.06,.99,.94],w_pad=2,h_pad=2)
figure=out/'C19_MATCHED_SPATIAL_COMPARISON_KO.png';fig.savefig(figure,dpi=150);plt.close(fig);results['figure']=bind(figure)
write(V/'ROOT_RESULTS_KO.json',results)
def percent(v):return '—'if v is None else f'{v*100:.1f}%'
def frac(v):return f"{v['correct']}/{v['N']} ({percent(v['recall'])})"
trs=''
for name,label in [('TRAIN','학습 · 입력 가용'),('LOCO_VAL','LOCO 검증'),('TOMIE_DEV','TOMIE 개발'),('VAL_COMBINED','공개 검증 합계')]:
 for arm in ARMS:
  g=data[arm]['groups'][name];cells=[label,'주변 평균'if arm==ARMS[0]else'위치별 유지',frac(g['D_available']),frac(g['E_available']),frac(g['E_qOUT_available']),frac(g['SHAM_D_qIN_available']),percent(g['D_precision']),percent(g['E_precision'])]
  trs+='<tr>'+''.join(f'<td>{esc(c)}</td>'for c in cells)+'</tr>'
valcm=''.join(f'<p><b>{esc(arm)}</b>: {esc(data[arm]["groups"]["VAL_COMBINED"]["CM_rows_D_E_cols_D_E_U"])}</p>'for arm in ARMS)
case_gallery=V/'saved_case_gallery/GALLERY_KO.html'
case_extra=f'''<p>{link(case_gallery,"판단이 바뀐 실제 공개 crop: BEFORE / COVER / SHAM")}</p><p>24사례의72개 raw crop을 직접 봤습니다. 아래 사례는 캔을 옮겨 지정된 목재 지지부는 드러내고 query q는 캔 위에 남긴 경우입니다. 주변 평균 모델은 보임으로 유지했지만 위치별 모델은 가림으로 버립니다. 노란 표시는 q, 청록은 진단용 weak reference와 불확실성입니다.</p><img src="saved_case_gallery/figures/CASE_17_MARKED.png" alt="기대 목재 지지부는 보이지만 query는 캔에 덮인 보임 오판 사례"><p>반대로 지정된 구조가 가려져도 주변에 목재가 남으면 두 모델이 보임으로 놓치는 사례도 있습니다. 가림체 존재와 특정 구조의 가림을 구별하지 못하는 동작이 반복됩니다. 이 관찰만으로 attention의 원인이나 한 가지 실패 원인을 증명하지는 않습니다.</p><p class="note">사례는 검증만 대상으로 고정 규칙으로 골랐으며5cue/3source group에 집중돼 있습니다. 전체 검증 분포는 위 표와 JSON을 봐야 합니다. 청록 기준은 모델에 주지 않았고 실제 native8 인간 정답도 아닙니다.</p>'''if case_gallery.exists()else''
proto=V/'registered_native_teacher_feasibility/RESULT.json'
proto_extra=f'''<p>기하 prototype도 실제로 만들었습니다. 새 board/block 형태와 rib/leg 형태 두 메시에서 native8 참조가 지정한 deck/runner 꼭짓점에 정확히 놓이며, 코너별37개 지지부 표본을 등록했습니다. 알려진 합성 카메라와 같은 불투명 물체의 이동으로 고정 코너의 <b>보임→외부 가림→보임</b>을 ray 검사에서 확인했습니다. 별도의 자기 가림과 부분 불명확 U도 보존했습니다.</p><p class="note">새로 구성한 단순 기하 자산입니다. 기존 primary CAD나 실물 팔레트와 동일하다고 주장하지 않습니다. RGB 렌더·현실적인 외형·이 자료로 학습한 성능은 아직 확인하지 않았습니다. 다음은 렌더 결과와 코너별 구조 지지 영역을 실제 이미지에서 검수하는 것입니다.</p><p>{link(proto,"새 native 참조 등록 기하 prototype")} · {link(V/'ROOT_NEW_REGISTERED_GEOMETRY_REVIEW.json','root 메시 대응·ray 재검산')}</p>'''if proto.exists()else''
evidence='<li>'+('</li><li>'.join(link(b['path'],name)for name,b in results['evidence'].items()))+'</li>'
body=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>V127 공간 특징 통제 비교</title><style>body{{font-family:system-ui,"Noto Sans CJK KR",sans-serif;color:#25323c;background:#f5f7fa;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce3e9;border-radius:12px;margin:20px 0;padding:22px}}h1{{font-size:29px}}h2{{font-size:21px}}p,li{{line-height:1.8}}.lead{{font-size:19px}}.note{{color:#566575}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce3e9}}.scroll{{overflow:auto}}a{{color:#176ba4}}</style><main>
<h1>위치별 특징을 유지해도 보임·가림의 균형은 해결되지 않았습니다</h1><p class="lead">공개 검증의 어려운 가림은 <b>229→279/483</b>으로 늘었지만 어려운 보임은 <b>148→97/394</b>로 줄었습니다. 가림을 더 찾는 만큼 보이는 구조도 더 버리는 문제가 남아, 이번 모델을 현재 실사에 채택하지 않았습니다.</p>
<section><img src="visual_followup/C19_MATCHED_SPATIAL_COMPARISON_KO.png" alt="같은 학습 조건의 주변 평균 특징과 위치별 특징 비교"><p class="note">학습 자료는6cue/3원본, 검증은26cue/6원본입니다. TOMIE4장은 한 episode입니다. 수천 개 독립 자연 장면의 검증이나 현재 실사의 새 평가로 해석하지 않습니다.</p></section>
<section><h2>무엇을 실제로 비교했나</h2><p>같은 frozen DINO local96에서 3×3 간격의 patch 특징을 뽑았습니다. 한 모델은 9개 raw 특징을 평균한 값을 반복해서 받고, 다른 모델은 중앙을 제외한 8개 위치의 특징을 각각 받습니다. 기존1928 특징은 그대로 유지했습니다. 두 head는418,433개 파라미터·같은 초기 상태·같은1,000개 배치·같은 손실과 optimizer를 사용했습니다.</p><p>추출4,428개에서 기존 전체crop F와 중앙P 특징이 bit 단위로 일치했습니다. 두 모델의 최초4,896개 점수도 기존 C16과 정확히 같았습니다. 최종1,000회끼리 비교했고 검증으로 checkpoint나 문턱을 고르지 않았습니다.</p><p class="note">평균 입력은 반복해도 정보 rank가 늘지 않습니다. 따라서 동일 파라미터 수를 맞춘 ‘개별 위치 정보+집계’ 비교이며 순서 하나만 바꾼 실험은 아닙니다. 첫 추출은 Torch2.1 마스크 API 오류로 중단됐고, 실패 기록을 남긴 뒤 boolean flatten-all로 수정한 별도V2에서 성공했습니다. 정답·입력 조건·허용 오차는 바꾸지 않았습니다.</p></section>
<section><h2>학습과 별도 공개 검증의 결과</h2><div class="scroll"><table><tr><th>집합</th><th>특징</th><th>보임 recall</th><th>가림 recall</th><th>가림·q 밖</th><th>보임·q 안</th><th>보임 precision</th><th>가림 precision</th></tr>{trs}</table></div><p>공개 검증 혼동행렬은 아래와 같습니다. 행은 실제 D/E, 열은 예측 D/E/U입니다. 양쪽 모두 검증3,582개에서 보류0입니다.</p>{valcm}<p class="note">학습 예정 known846개 중 입력을 쓸 수 없는51개는 U로 보존했고795개만 학습했습니다. 별도의 teacherU468개는 학습 손실에서 제외했습니다. 위 학습 recall은 입력 가용 분모이고, 전체 예정 분모와 U는 JSON에 함께 기록했습니다.</p>{case_extra}</section>
<section><h2>추가 공개 원본24장 확인</h2><p>공식 TRAIN subset2에서 촬영 take별로4장씩 선정한 뒤 원본을 추출했습니다. 모두 CRC/JPEG/decoded SHA가 기존 공식 원본 기록과 일치했습니다. 실제 fullframe과 bbox crop을 모두 봤고, 9장은 구조 cue를 더 검토할 만했고7장은 거친 구조만 확인했으며8장은 불명확하게 남겼습니다. 라벨이나 학습 데이터로 자동 편입하지 않았습니다.</p><p>가까운 밝은 팔레트, 어두운 낮은 팔레트, 목재 지지부 등 외형 다양성이 확인됐습니다. 재질 정답이나 native8 대응은 확정하지 않았습니다.6take가 서로 독립인 환경이라는 뜻도 아닙니다.</p><p>{link(V/'public_diversity_intake/GALLERY_KO.html','공개24장 원본·native crop 갤러리')} · {link(V/'ROOT_PUBLIC24_ACTUAL_VISUAL_SCREEN.json','root 실제 검토와 불확실성 기록')}</p></section>
<section><h2>현재 실사와 다음 단계</h2><p>현재 실사의 최신 평가는 여전히 C16: <b>보임1137/1147, 외부가림35/90</b>입니다. 외부 가림55개를 놓칩니다. 이번 두 모델은 현재 실사에 새로 적용하지 않았습니다. 기존18개 source cohort의 새 stencil도 없으므로 old15 회귀 통과를 주장하지 않습니다.</p><p>다음은 공개 원본의 구조 다양성을 실제 학습용 cue로 보강하고, native8 참조와 실제 구조 부위가 명시적으로 연결된 합성 teacher를 구성하는 것입니다. native8 순서 자체는 이미 알려져 있지만, 기존 primary CAD의 부위 endpoint 대응까지 확인된 것은 아닙니다. 기존 합성 생성기의 화면 내 visibility를 외부/자기 가림 정답으로 재사용하지 않습니다.</p>{proto_extra}<p>{link(current['report'],'현재 실사 C16 평가와 실제 실패 crop')}</p><p class="note">현재223은 학습·정규화·문턱·checkpoint·arm 선택에 사용하지 않았습니다. 자기 가림은 초기 기하에서 처리하는 방침을 유지하고, PnP/초기 좌표 보정은 수행하지 않았습니다. 목표는 아직 해결되지 않았습니다.</p></section>
<section><h2>재현·검토 기록</h2><ul>{evidence}</ul><p>{link(V/'ROOT_RESULTS_KO.json','전체 결과 JSON')}</p><p class="note">PNG는 실제 시각 검수하고 HTML 링크와 수치는 정적으로 검사합니다. 브라우저 렌더링 검수를 주장하지 않습니다.</p></section></main></html>'''
(V/'ROOT_READOUT_KO.html').write_text(body)
print(json.dumps({'report':str(V/'ROOT_READOUT_KO.html'),'results':str(V/'ROOT_RESULTS_KO.json'),'figure':str(figure),'VAL_counts':{arm:data[arm]['groups']['VAL_COMBINED']for arm in ARMS}},ensure_ascii=False))
