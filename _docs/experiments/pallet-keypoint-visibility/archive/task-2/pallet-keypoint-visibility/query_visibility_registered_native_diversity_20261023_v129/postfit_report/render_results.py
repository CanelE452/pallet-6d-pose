"""Render completed source-only JSON results; no arrays, pixels or model loads."""
from pathlib import Path
import hashlib,html,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;R=V/'runs/C20_C21_FIXED1000_SOURCE_CONTROL'
COHORTS=('OLD_PUBLIC','PUBLIC_AUX','NATIVE');MODELS=('C16','C20','C21')
ARMS={'C20':'C20_PUBLIC_REPLAY_CONTROL','C21':'C21_PUBLIC_PLUS_NATIVE'}
LABELS={'OLD_PUBLIC':'기존 공개 AUX','PUBLIC_AUX':'새 공개 AUX','NATIVE':'새 등록 합성'}
COLORS={'C16':'#6b7785','C20':'#2783b7','C21':'#d77925'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert bind(b['path'])==b;return read(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ratio(a,b):return float(a/b)if b else None
def summary(x):
 cm=np.array(x['CM_D_E_rows_D_E_U_columns'],np.int64);assert cm.shape==(2,3)and(cm>=0).all()
 vals={'D_precision':ratio(cm[0,0],cm[:,0].sum()),'D_recall':ratio(cm[0,0],cm[0].sum()),'E_precision':ratio(cm[1,1],cm[:,1].sum()),'E_recall':ratio(cm[1,1],cm[1].sum())}
 assert cm[0].sum()==x['raw_states']['D']and cm[1].sum()==x['raw_states']['E']and sum(x['raw_states'].values())==x['all_rows']
 for k,value in vals.items():assert value==x[k],(k,value,x[k])
 assert int(cm[:,2].sum())==x['known_input_U']
 return {'CM':cm.tolist(),'CM_rows':['D_visible','E_external'],'CM_columns':['prediction_D','prediction_E','prediction_U'],'planned_rows':x['all_rows'],'teacher_states':x['raw_states'],'known_DE':int(cm.sum()),**vals,'prediction_U_known':int(cm[:,2].sum()),'prediction_U_known_rate':ratio(cm[:,2].sum(),cm.sum()),'S_hidden_response':x['teacher_S_hidden_response'],'S_available':x['teacher_S_available'],'teacher_U_available':x['teacher_U_available']}
def figures(table):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,3,figsize=(14,5.4),sharey=True)
 for ax,cohort in zip(axes,COHORTS):
  x=np.arange(2)
  for j,m in enumerate(MODELS):
   values=[100*table[cohort][m][s+'_recall']for s in ('D','E')];bars=ax.bar(x+(j-1)*.24,values,.23,label=m,color=COLORS[m])
   ax.bar_label(bars,labels=[f'{v:.1f}%'for v in values],padding=4,fontsize=10)
  ax.set_xticks(x,['D 보임 recall','E 외부 가림 recall']);ax.set_ylim(0,110);ax.set_yticks(np.arange(0,101,20));ax.set_title(LABELS[cohort]+' · VALIDATION',pad=14);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  ax.text(.5,-.16,'D '+str(table[cohort]['C16']['teacher_states']['D'])+' / E '+str(table[cohort]['C16']['teacher_states']['E']),ha='center',transform=ax.transAxes,color='#42576b')
 axes[0].set_ylabel('Recall (%) · 예측 U를 분모에 유지');handles,names=axes[-1].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
 fig.suptitle('합성 추가 후 E recall 상승과 D 보존의 tradeoff — 두 source control 모두 기준 미달',fontproperties=fp,fontsize=15,y=.99)
 fig.tight_layout(rect=(0,0,1,.85));p=H/'VAL_RECALL_COMPARISON.png';fig.savefig(p,dpi=160,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(3,3,figsize=(12.5,9.5));fig.subplots_adjust(hspace=.6,wspace=.25)
 for i,c in enumerate(COHORTS):
  for j,m in enumerate(MODELS):
   ax=axes[i,j];cm=np.array(table[c][m]['CM']);norm=cm/np.maximum(1,cm.sum(1,keepdims=True));ax.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='auto')
   ax.set_xticks(range(3),['예측 D','예측 E','예측 U']);ax.set_yticks(range(2),['정답 D','정답 E']);ax.set_title(LABELS[c]+' / '+m,pad=8)
   for a,b in np.ndindex(cm.shape):ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=14,color='white'if norm[a,b]>.55 else'#193348')
   ax.tick_params(length=0)
 fig.suptitle('원래 validation 전체 분모 · 혼동행렬 2×3 (S / teacher U는 별도 보존)',fontproperties=fp,fontsize=16,y=.98)
 p2=H/'VAL_CONFUSION_MATRICES.png';fig.savefig(p2,dpi=160,bbox_inches='tight');plt.close(fig)
 return [bind(p),bind(p2)]
def render_html(result):
 def pc(v):return '—'if v is None else f'{100*v:.1f}%'
 rows=''
 for c in COHORTS:
  for m in MODELS:
   x=result['validation'][c][m];rows+=f'<tr><td>{LABELS[c]}</td><td>{m}</td><td><code>{x["CM"]}</code></td><td>{pc(x["D_precision"])}</td><td>{pc(x["D_recall"])}</td><td>{pc(x["E_precision"])}</td><td>{pc(x["E_recall"])}</td><td>{x["prediction_U_known"]}/{x["known_DE"]}</td></tr>'
 gates=''.join(f'<tr><td>{m}</td><td>{result["source_gates"][m]["passed_gates"]}/8</td><td>{", ".join(k for k,v in result["source_gates"][m]["gates"].items()if not v["PASS"])}</td></tr>'for m in ('C20','C21'))
 links=''.join(f'<li><a href="{html.escape(Path(b["path"]).as_uri())}">{html.escape(k)}</a></li>'for k,b in result['bindings'].items())
 counts=''.join(f'<tr><td>{LABELS[c]}</td><td>{result["validation"][c]["C16"]["planned_rows"]}</td><td>{result["validation"][c]["C16"]["teacher_states"]["D"]}</td><td>{result["validation"][c]["C16"]["teacher_states"]["E"]}</td><td>{result["validation"][c]["C16"]["teacher_states"]["S"]}</td><td>{result["validation"][c]["C16"]["teacher_states"]["U"]}</td></tr>'for c in COHORTS)
 text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V129 source control 결과 초안</title>
 <style>body{{font:16px/1.65 Arial,"Noto Sans CJK KR",sans-serif;color:#193348;max-width:1380px;margin:26px auto;padding:0 24px}}h1{{font-size:29px;line-height:1.35}}h2{{margin-top:30px;font-size:22px}}.lead{{background:#fff0dc;padding:17px;border-left:5px solid #c47b23}}.setup{{background:#edf4f8;padding:13px}}table{{width:100%;border-collapse:collapse;font-size:14px;margin:14px 0}}td,th{{border:1px solid #d4dfe6;text-align:left;padding:8px}}th{{background:#edf4f8}}img{{width:100%;height:auto}}figure{{margin:22px 0}}code{{overflow-wrap:anywhere}}a{{color:#075b91}}.small{{font-size:14px;color:#485e70}}</style>
 <h1>V129: 등록 합성 source 추가는 외부 가림을 더 잡았지만,<br>보이는 cue 보존과 새 공개 실사 조건을 함께 통과하지 못했습니다</h1>
 <p class="lead">C21은 새 등록 합성 validation에서 E recall을 <b>14.1% → 78.9%</b>로 높였지만 D recall은 <b>96.1% → 62.2%</b>로 떨어졌습니다. 새 공개 AUX validation도 <b>D 45.1%, E 74.3%</b>로 기준 미달입니다. C20·C21 모두 source 부적격이며 새 current 평가는 허용되지 않았습니다. 목표가 해결됐거나 실제 실사 성능이 개선됐다는 결과가 아닙니다.</p>
 <p class="setup"><b>비교 정의</b> C16은 기존 source-selected step4500의 고정 기준 모델입니다. C20은 기존 공개 source 795개 + 새 공개 TRAIN 612개를 재학습했고, C21은 <b>동일한 public 추출 순서</b>에 등록 합성의 D/E 균형 batch를 추가했습니다. 두 arm 모두 같은 C16에서 시작해 동일 lr의 1,000회 고정 update를 실행했습니다. backbone·1928 feature·head 구조·raw threshold 0은 유지했고 validation checkpoint 선택은 없었습니다. 두 학습이 끝난 뒤 validation 수치를 산출했습니다.</p>
 <figure><a href="VAL_RECALL_COMPARISON.png"><img src="VAL_RECALL_COMPARISON.png" alt="세 source validation cohort의 C16 C20 C21 D E recall"></a></figure>
 <h2>같은 validation 행의 혼동행렬·precision·recall</h2><p>CM의 행은 실제 D/E, 열은 예측 D/E/U입니다. recall은 known D/E 전체를 분모로 하며 예측 U를 포함합니다. 이 세 validation의 known input U는 모두 0입니다. teacher U는 예측 U와 다른 개념입니다.</p>
 <table><tr><th>cohort</th><th>모델</th><th>CM [[D→D,D→E,D→U],[E→D,E→E,E→U]]</th><th>D precision</th><th>D recall</th><th>E precision</th><th>E recall</th><th>예측 U / known</th></tr>{rows}</table>
 <figure><a href="VAL_CONFUSION_MATRICES.png"><img src="VAL_CONFUSION_MATRICES.png" alt="세 cohort 세 모델의 2x3 confusion matrices"></a></figure>
 <h2>판정과 해석</h2><table><tr><th>arm</th><th>통과</th><th>실패한 prospective source 조건</th></tr>{gates}</table>
 <p>C20 public 재학습만으로는 합성 E recall이 11.1%에 머물렀습니다. C21의 합성 추가는 그 값을 크게 높였으나 합성 D 누락도 287개에서 2,816개로 늘었습니다. 기존 공개 validation에서도 C21은 E recall 71.9%→83.9%와 함께 D recall 77.8%→66.7%의 tradeoff를 보였습니다. 따라서 이번 제어 실험은 source 데이터가 판정에 영향을 준다는 증거이지만, D/E를 안정적으로 구분하는 균형 해법이나 실사 전이 성공의 증거는 아닙니다.</p>
 <h2>목표와 분모를 구분합니다</h2><table><tr><th>validation cohort</th><th>전체 행</th><th>D</th><th>E</th><th>S</th><th>teacher U</th></tr>{counts}</table>
 <p>등록 합성 teacher는 지정 모서리 인접 영역의 연속 ray witness에 대한 보임/외부 가림입니다. <b>q 픽셀에 target 물질이 있는가</b>와 같지 않습니다. 그 차이와 q-pixel owner 진단, C16/v61의 각자 전체·공통 availability 분모는 <a href="../saved_source_baseline_evaluation_v1/READOUT_KO.html">완료된 baseline 비교</a>에 따로 남겼습니다. 공개 AUX는 weak cue region과 보호 영역을 가리는 합성 조건이며 human/native exact gold로 승격하지 않습니다. S는 기하 경로로 별도 처리하며 외부 E 학습 대상이 아닙니다. S/U 행은 버리지 않고 JSON의 typed counts에 보존했습니다.</p>
 <p>Native corpus는 192개 base 설정·768개 장면에서 9개 상관된 q8 조건과 8개 slot을 만든 자료입니다. PUBLIC_AUX는 6개 cue 중 TRAIN 4개·validation 2개이며 원래 take group은 총 5개(TRAIN 3개·validation 2개)입니다. 같은 take의 두 TRAIN cue와 각 cue의 query·donor 반복은 독립 새 실사 표본이 아닙니다. 기존 공개 source도 반복 개발에 노출됐습니다. 합성 wood/polymer 재질과 rectangular/square 형상은 독립 교차 축이며 실제 WOOD/SQUARE 평가군과 다릅니다.</p>
 <p class="small">이번 보고서는 source 저장 JSON만 읽었습니다. 새 encoder/head forward·fit·현재223/V112 접근·좌표 보정·PnP가 없습니다. V124의 실제 current 결과는 갱신되지 않았고 전체 목표는 미해결입니다. 아래 원본 저장 결과와 보고서 수치는 선택·임계값 재조정에 사용하지 않았습니다.</p>
 <p><a href="RESULTS.json">컴팩트 결과 JSON</a> · <a href="ARTIFACTS.json">생성물·입력 SHA</a></p><details><summary>원본 saved 결과</summary><ul>{links}</ul></details></html>'''
 (H/'REPORT_DRAFT_KO.html').write_text(text)
def run():
 assert not(H/'RESULTS.json').exists(),'Preserve completed report; no overwrite.'
 control=read(R/'CONTROL_COMPLETE.json');assert control['completed']and control['PASS']and control['both_fixed_fits_before_VAL_metrics']and control['current_evaluation_authorized']is False
 baseline=checked(control['baseline_results']);assert baseline['completed']and baseline['all61110_retained']
 evidence={'CONTROL_COMPLETE':bind(R/'CONTROL_COMPLETE.json'),'C16_source_results':control['baseline_results'],'baseline_evaluation':bind(V/'saved_source_baseline_evaluation_v1/EVALUATION_SEAL.json')};source={'C16':baseline};gates={}
 for m,b in zip(('C20','C21'),control['outputs']):
  fit=checked(b);assert fit['completed']and fit['PASS']and fit['optimizer_updates']==1000 and fit['fixed_final_not_VAL_selection']and fit['current_evaluation_authorized']is False
  source[m]=checked(fit['source_results']);gates[m]=checked(fit['source_gates']);evidence[m+'_fit_complete']=b;evidence[m+'_source_results']=fit['source_results'];evidence[m+'_source_gates']=fit['source_gates']
  assert gates[m]['completed']and gates[m]['all_gates_pass']is False and gates[m]['current_evaluation_authorized']is False
 table={c:{m:summary(source[m]['groups'][c+'::VALIDATION']['all_planned'])for m in MODELS}for c in COHORTS}
 for c in COHORTS:
  assert all(table[c][m]['teacher_states']==table[c]['C16']['teacher_states']for m in MODELS)
  assert all(table[c][m]['prediction_U_known']==0 for m in MODELS)
 evalresult=read(V/'saved_source_baseline_evaluation_v1/SOURCE_RESULTS.json');assert evalresult['completed']
 for c in ('PUBLIC_AUX','NATIVE'):assert evalresult['semantic_reports'][c+'/fold/VALIDATION']['C16']['own_full_denominator']['CM']==table[c]['C16']['CM']
 cuefile=V/'public_source_preparation/CUES6_INPUT_ONLY.json';cues=read(cuefile)['rows'];groupcounts={fold:len({c['source_frame_group']for c in cues if c['split']==fold})for fold in ('TRAIN','VALIDATION')};assert len(cues)==6 and groupcounts=={'TRAIN':3,'VALIDATION':2};evidence['public_cues_input_metadata']=bind(cuefile)
 result={'completed':True,'source_only':True,'validation':table,'source_gates':gates,'eligible_source_arm':control['eligible_source_arm'],'current_evaluation_authorized':False,'old15_regression_PASS_claimed':False,'CM_P_R_U_recomputed_from_saved_CM':True,'both_fixed1000_fits_before_VAL_metrics':True,'no_new_model_encoder_fit_current223_V112_PnP':True,'public_cues':6,'public_take_groups':groupcounts,'real_current_status':'V124 unchanged; no current payload read by this report; broad goal unresolved.','source_limits':['Registered synthetic continuous structural region teacher is distinct from q-pixel material presence.','Public labels are conditional weak cue-region supervision, not human/native exact corner gold.','Repeated queries/variants share base assets/cues; not independent images.','Source validation has development exposure; no real private improvement is claimed.'],'bindings':evidence}
 result['figures']=figures(table);write(H/'RESULTS.json',result);render_html(result)
 artifacts=[bind(H/f)for f in ('render_results.py','VAL_RECALL_COMPARISON.png','VAL_CONFUSION_MATRICES.png','RESULTS.json','REPORT_DRAFT_KO.html')];write(H/'ARTIFACTS.json',{'completed':True,'artifacts':artifacts,'input_bindings':evidence,'saved_JSON_only':True,'no_new_forward_fit':True,'human_visual_QA_pending':True})
 print(json.dumps({'completed':True,'artifacts':bind(H/'ARTIFACTS.json'),'report':bind(H/'REPORT_DRAFT_KO.html'),'figures':result['figures'],'eligible_source_arm':control['eligible_source_arm']},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
