"""Compact saved-JSON duration report. No model, feature, score-array or RGB I/O."""
from pathlib import Path
import hashlib,html,importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
PREV=P/'query_visibility_registered_native_diversity_20261023_v129'
R1=PREV/'runs/C20_C21_FIXED1000_SOURCE_CONTROL';R8=V/'runs/C22_C23_FIXED8000_SOURCE_CONTROL'
MODELS=('C16','C20','C21','C22','C23');COHORTS=('OLD_PUBLIC','PUBLIC_AUX','NATIVE')
LABELS={'OLD_PUBLIC':'기존 공개 AUX','PUBLIC_AUX':'새 공개 AUX','NATIVE':'새 등록 합성'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def checked(b):assert bind(b['path'])==b;return read(b['path'])
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def figures(table):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 plt.rcParams.update({'font.family':fp.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 matrix=np.array([[100*table[c]['VALIDATION'][m][s+'_recall']for m in MODELS]for c in COHORTS for s in ('D','E')])
 fig,ax=plt.subplots(figsize=(10.8,6.2));im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(5),['C16\n고정 기준','C20\npublic · 1000','C21\n+native · 1000','C22\npublic · 8000','C23\n+native · 8000']);ax.set_yticks(range(6),[LABELS[c]+' · '+s for c in COHORTS for s in ('D','E')]);ax.tick_params(length=0)
 for i,j in np.ndindex(matrix.shape):ax.text(j,i,f'{matrix[i,j]:.1f}%',ha='center',va='center',fontsize=14,color='white'if matrix[i,j]>63 else'#18334a')
 for y in (1.5,3.5):ax.axhline(y,color='white',lw=4)
 ax.set_title('동일 source VALIDATION · D/E recall (%)\n전체 known D/E 분모 유지, 각 VAL cohort 예측 U=0',pad=16);cb=fig.colorbar(im,ax=ax,shrink=.9);cb.set_label('Recall (%)');fig.tight_layout();p1=H/'VAL_RECALL_HISTORY.png';fig.savefig(p1,dpi=160,bbox_inches='tight');plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(13,5.5),sharey=True);colors=['#69aac8','#245f88','#e8b66a','#b66c20'];labels=['C21 1000 TRAIN','C21 1000 VAL','C23 8000 TRAIN','C23 8000 VAL']
 for ax,c in zip(axes,('PUBLIC_AUX','NATIVE')):
  x=np.arange(2)
  for k,(m,fold)in enumerate((('C21','TRAIN'),('C21','VALIDATION'),('C23','TRAIN'),('C23','VALIDATION'))):
   values=[100*table[c][fold][m][s+'_recall']for s in ('D','E')];bars=ax.bar(x+(k-1.5)*.19,values,.18,label=labels[k],color=colors[k]);ax.bar_label(bars,labels=[f'{v:.1f}%'for v in values],padding=4,fontsize=10)
  ax.set_xticks(x,['D 보임','E 외부 가림']);ax.set_ylim(0,113);ax.set_yticks(range(0,101,20));ax.set_title(LABELS[c],pad=12);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  tr=table[c]['TRAIN']['C23']['teacher_states'];va=table[c]['VALIDATION']['C23']['teacher_states'];ax.text(.5,-.15,f'TRAIN D{tr["D"]}/E{tr["E"]} · VAL D{va["D"]}/E{va["E"]}',ha='center',transform=ax.transAxes,color='#42576b')
 handles,names=axes[-1].get_legend_handles_labels();fig.legend(handles,names,loc='upper center',bbox_to_anchor=(.5,.94),ncol=4,frameon=False,fontsize=10);axes[0].set_ylabel('Recall (%)');fig.suptitle('학습 적합도는 높아졌지만 TRAIN–VAL 격차가 남음 · 같은 public+native arm의 기간 비교',fontproperties=fp,fontsize=15,y=.99);fig.tight_layout(rect=(0,0,1,.84));p2=H/'TRAIN_VS_VAL_RECALL.png';fig.savefig(p2,dpi=160,bbox_inches='tight');plt.close(fig)
 return[bind(p1),bind(p2)]
def render_html(result):
 def pc(x):return '—'if x is None else f'{100*x:.1f}%'
 rows=''
 for c in COHORTS:
  for m in MODELS:
   x=result['metrics'][c]['VALIDATION'][m];rows+=f'<tr><td>{LABELS[c]}</td><td>{m}</td><td><code>{x["CM"]}</code></td><td>{pc(x["D_precision"])}</td><td>{pc(x["D_recall"])}</td><td>{pc(x["E_precision"])}</td><td>{pc(x["E_recall"])}</td><td>{x["prediction_U_known"]}/{x["known_DE"]}</td></tr>'
 gate=''.join(f'<tr><td>{m}</td><td>{result["gates"][m]["passed_gates"]}/8</td><td>{", ".join(k for k,v in result["gates"][m]["gates"].items()if not v["PASS"])}</td></tr>'for m in ('C22','C23'))
 links=''.join(f'<li><a href="{html.escape(Path(b["path"]).as_uri())}">{html.escape(k)}</a></li>'for k,b in result['bindings'].items())
 previous=PREV/'postfit_report/REPORT_DRAFT_KO.html';baseline=PREV/'saved_source_baseline_evaluation_v1/READOUT_KO.html'
 text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V130 학습 기간 source 제어</title><style>body{{font:16px/1.65 Arial,"Noto Sans CJK KR",sans-serif;color:#193348;max-width:1370px;margin:26px auto;padding:0 24px}}h1{{font-size:29px;line-height:1.35}}h2{{margin-top:28px;font-size:22px}}.lead{{background:#fff0dc;border-left:5px solid #c47b23;padding:17px}}.setup{{background:#edf4f8;padding:13px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d4dfe6;padding:7px;text-align:left}}th{{background:#edf4f8}}code{{overflow-wrap:anywhere}}a{{color:#075b91}}figure{{margin:22px 0}}.small{{font-size:14px;color:#485e70}}</style>
 <h1>V130: 8,000회 학습으로 TRAIN 적합도는 높아졌지만,<br>source 일반화와 D/E 균형 문제는 남았습니다</h1>
 <p class="lead">C23의 새 공개 TRAIN은 <b>D 318/324 (98.1%), E 284/288 (98.6%)</b>까지 올랐지만 validation은 <b>D 99/162 (61.1%), E 97/144 (67.4%)</b>입니다. 등록 합성도 TRAIN <b>D 94.0%, E 95.7%</b>와 validation <b>D 78.1%, E 88.3%</b> 사이 격차가 남았습니다. C22·C23 모두 기존 source 기준 미달이며 새 current 평가 권한은 없습니다. 학습 횟수 부족만으로 앞선 실패를 설명하거나 더 오래 학습하면 해결된다고 결론 내릴 수 없습니다.</p>
 <p class="setup">새 데이터·feature·head·loss·lr·seed·threshold 없이 <b>고정 1,000회 → 고정 8,000회</b>만 비교했습니다. C22는 public replay, C23은 동일 public batch + native D/E 균형 batch입니다. 두 arm 모두 C16 step4500에서 시작했습니다. 첫 1,000회의 sampler·매-step loss/gradient·모델 state SHA가 각각 V129와 <b>비트 단위 일치</b>했고 같은 optimizer로 7,000회를 연속 수행했습니다. 두 arm의 8,000회 최종 상태를 고정한 뒤 VAL을 읽었으며 VAL checkpoint 선택은 없습니다.</p>
 <figure><a href="TRAIN_VS_VAL_RECALL.png"><img src="TRAIN_VS_VAL_RECALL.png" alt="C21 C23 기간 비교의 train validation D E recall"></a></figure>
 <h2>다섯 고정 모델의 같은 source validation 비교</h2><figure><a href="VAL_RECALL_HISTORY.png"><img src="VAL_RECALL_HISTORY.png" alt="C16 C20 C21 C22 C23 세 validation cohort recall"></a></figure>
 <p>아래 CM은 행 D/E, 열 예측 D/E/U입니다. 예측 U를 recall 분모에서 제외하지 않았습니다. 이 세 validation의 known 입력 U는 모두 0이며, teacher U와 기하 S는 D/E 정답에 섞지 않고 JSON에서 별도로 보존합니다.</p><table><tr><th>cohort</th><th>모델</th><th>CM [[D→D,D→E,D→U],[E→D,E→E,E→U]]</th><th>D P</th><th>D R</th><th>E P</th><th>E R</th><th>예측 U/known</th></tr>{rows}</table>
 <h2>기간 control에서 알게 된 것</h2><p>같은 입력 표현과 head가 새로운 TRAIN 자료에는 훨씬 잘 맞출 수 있었습니다. 따라서 1,000회 결과만으로 feature가 필요한 정보를 전혀 담지 못한다고 주장할 수 없습니다. 반면 public-only C22는 새 공개 TRAIN D/E가 모두 100%이지만 VAL D 73.5%, E 56.9%입니다. C23도 TRAIN 개선에 비해 새 공개 VAL은 제한적이고, 합성에서는 E 개선과 D 누락이 동시에 남습니다. 단순 미관측 coverage만으로도 설명되지 않았으며, 이 결과는 과적합·자료/목표 차이 등의 가능성과 부합하지만 원인을 하나로 증명하지 않습니다. 다음 단계는 아직 새 학습으로 확정하지 않았습니다.</p>
 <p><b>임계값만의 문제인지에 대한 추가 진단:</b> root의 <a href="{Path(result['bindings']['posthoc_score_ranking']['path']).as_uri()}">저장 점수 순위 진단</a>에서 C23 새 공개 VAL의 E AUC는 0.684였습니다. 가능한 단일 cut을 모두 사후 계산해도 min(D recall, E recall)의 최대는 <b>62.96%</b>(D 102/162, E 92/144)로, 이 고정 source 점수에서 두 recall을 모두 80%로 만들 수 없었습니다. 합성 VAL은 AUC 0.918, 사후 최대 min recall 83.95%로 점수 분리가 개선됐습니다. 이는 합성과 공개 실사의 일반화 격차를 구분하는 진단이며, cut 값은 export하지 않았고 실제 threshold 0·모델·source 부적격 판정은 그대로입니다.</p>
 <table><tr><th>8,000회 arm</th><th>통과</th><th>실패한 동일 prospective 기준</th></tr>{gate}</table>
 <h2>해석 범위</h2><p>Native는 192개 base·768개 장면의 반복 q8/slot 조건이며 validation D7452/E540/S2880/U7560을 보존했습니다. 새 공개 AUX는 weak cue 6개, 원래 5개 take group(TRAIN 3·VAL 2)의 조건 반복입니다. 표의 수천 행은 독립 실사 수천 장이 아닙니다. source VAL은 이미 개발에 노출됐으므로 새 blind 일반화 증거가 아닙니다.</p><p>구조 cue region의 D/E와 q 픽셀 target 존재 여부는 다릅니다. S는 기하 경로로 별도 처리하며 외부 E loss에 넣지 않았습니다. 공개 weak region을 human/native exact gold로 승격하지 않습니다. 합성 재질 wood/polymer와 형상 rectangular/square는 독립 축이며 실제 WOOD/SQUARE 평가군과 다릅니다. <a href="{baseline.as_uri()}">C16/v61 baseline과 q-pixel 별도 진단</a> · <a href="{previous.as_uri()}">V129 1,000회 source 보고서</a></p>
 <p class="small">이 보고서는 저장 source JSON만 읽었습니다. 새 모델/encoder/fit·현재223/V112·좌표 보정·PnP 접근이 없습니다. 실제 V124 current 결과는 그대로이며 목표가 해결되거나 실사 성능이 개선됐다는 주장은 하지 않습니다. broader old15 regression gate는 계산하지 않아 PASS라고 주장하지 않습니다.</p><p><a href="RESULTS.json">전체 TRAIN/VAL CM·P/R/U 및 원본 근거</a> · <a href="ARTIFACTS.json">생성물 SHA</a></p><details><summary>저장 결과·prefix 증명</summary><ul>{links}</ul></details></html>'''
 (H/'REPORT_DRAFT_KO.html').write_text(text)
def run():
 assert not(H/'RESULTS.json').exists(),'Preserve prior report.'
 parent=PREV/'postfit_report/render_results.py';spec=importlib.util.spec_from_file_location('v129_report_reuse',parent);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 c1=read(R1/'CONTROL_COMPLETE.json');c8=read(R8/'CONTROL_COMPLETE.json');assert c1['completed']and c1['PASS']and c8['completed']and c8['PASS']and c8['current_evaluation_authorized']is False and c8['source_validation_development_exposed']and c8['total_updates_each']==8000 and c8['V129_prefix_updates_each']==1000 and c8['continued_updates_each']==7000
 assert c1['baseline_results']['sha256']==c8['baseline_results']['sha256'];source={'C16':checked(c8['baseline_results'])};gates={};parity={};evidence={'V129_control':bind(R1/'CONTROL_COMPLETE.json'),'V130_control':bind(R8/'CONTROL_COMPLETE.json'),'C16_baseline':c8['baseline_results'],'parent_report_code':bind(parent)}
 prefix=checked(c8['sampler_prefix_parity']);assert prefix['PASS']and prefix['public_first1000_exact']and prefix['native_first1000_exact'];evidence['sampler_prefix']=c8['sampler_prefix_parity']
 for control,names in ((c1,('C20','C21')),(c8,('C22','C23'))):
  for name,b in zip(names,control['outputs']):
   fit=checked(b);assert fit['completed']and fit['PASS']and fit['current_evaluation_authorized']is False;source[name]=checked(fit['source_results']);gates[name]=checked(fit['source_gates']);evidence[name+'_fit']=b;evidence[name+'_results']=fit['source_results'];evidence[name+'_gates']=fit['source_gates']
   if name in ('C22','C23'):
    p=checked(fit['update1000_parity']);assert p['completed']and p['PASS']and p['replayed_updates']==1000 and p['all1000_sampler_loss_gradient_trace_records_exact']and p['optimizer_continuation_without_reset']and p['model_state_sha256']==p['expected_model_state_sha256'];parity[name]=p;evidence[name+'_update1000_parity']=fit['update1000_parity']
 table={c:{fold:{name:m.summary(source[name]['groups'][c+'::'+fold]['all_planned'])for name in MODELS}for fold in ('TRAIN','VALIDATION')}for c in COHORTS}
 for c in COHORTS:
  assert all(table[c]['VALIDATION'][name]['prediction_U_known']==0 for name in MODELS)
  for fold in ('TRAIN','VALIDATION'):assert all(table[c][fold][name]['teacher_states']==table[c][fold]['C16']['teacher_states']for name in MODELS)
 diagfile=V/'saved_score_diagnosis/SOURCE_THRESHOLD_LIMIT_DIAGNOSTIC.json';diagnosis=read(diagfile);assert diagnosis['completed']and diagnosis['source_only']and diagnosis['current223_V112_model_fit_PnP']==0;rank=diagnosis['results']['C23_PUBLIC_PLUS_NATIVE_FIXED8000'];assert not rank['PUBLIC_AUX']['any_single_cut_can_reach_both_recalls_80pct']and all(not rank[c]['threshold_value_exported']for c in COHORTS);evidence['posthoc_score_ranking']=bind(diagfile)
 result={'completed':True,'source_only':True,'metrics':table,'gates':gates,'prefix_parity':parity,'eligible_source_arm':c8['eligible_source_arm'],'source_validation_development_exposed':True,'same_final_fixed8000_no_VAL_selection':True,'current_evaluation_authorized':False,'old15_regression_PASS_claimed':False,'CM_P_R_U_recomputed_from_saved_CMs':True,'no_new_model_encoder_fit_current223_V112_PnP':True,'duration_not_assumed_solution':True,'coverage_alone_does_not_explain_failures':True,'posthoc_C23_source_ranking_diagnostic':rank,'raw_threshold_still0_no_operating_cut_selected':True,'bindings':evidence}
 result['figures']=figures(table);write(H/'RESULTS.json',result);render_html(result)
 artifacts=[bind(H/f)for f in ('render_results.py','VAL_RECALL_HISTORY.png','TRAIN_VS_VAL_RECALL.png','RESULTS.json','REPORT_DRAFT_KO.html')];write(H/'ARTIFACTS.json',{'completed':True,'artifacts':artifacts,'input_bindings':evidence,'saved_JSON_only':True,'new_forward_fit':False,'human_visual_QA_pending':True})
 print(json.dumps({'completed':True,'report':bind(H/'REPORT_DRAFT_KO.html'),'artifacts':bind(H/'ARTIFACTS.json'),'figures':result['figures']}),flush=True)
if __name__=='__main__':run()
