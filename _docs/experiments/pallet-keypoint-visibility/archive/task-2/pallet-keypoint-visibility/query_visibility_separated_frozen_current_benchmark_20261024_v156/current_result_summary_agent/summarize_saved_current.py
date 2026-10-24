"""V156 completed saved evaluation only; no models, features, weights or RGB."""
from pathlib import Path
import csv,json,hashlib,datetime
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
EVAL=V/'saved_seven_current_evaluation_v1'
SEAL_SHA='52fb0fdaaeb1839afaea1cac4425379d0d488b75b81880a8f28aa469ad87654b'
RESULT_SHA='ea36f9ed5736ae3aa20350468b5b8321bed279dad7076e592508ca71cf68016d'
MODELS=('v61','legacy','C16','C35','C36','C37','C38');NEW=MODELS[3:]
GROUPS=('ALL','WOOD_PRIMARY105','SQUARE118')
FONT='/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),
 'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white','savefig.facecolor':'white'})
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_text())
def write(name,a):
 with (OUT/name).open('x')as f:json.dump(a,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ratio(n,d):return float(n/d)if d else None
def cm_metric(cm):
 a=np.array(cm,int);assert a.shape==(2,3)
 return {'CM_rows_D_E_columns_D_E_U':a.tolist(),'N':int(a.sum()),'D_N':int(a[0].sum()),'E_N':int(a[1].sum()),
  'D_precision':ratio(a[0,0],a[:,0].sum()),'D_recall':ratio(a[0,0],a[0].sum()),
  'E_precision':ratio(a[1,1],a[:,1].sum()),'E_recall':ratio(a[1,1],a[1].sum()),
  'U':int(a[:,2].sum()),'U_rate':ratio(a[:,2].sum(),a.sum()),'decided_coverage':ratio(a[:,:2].sum(),a.sum())}
def from_metrics(rows):
 def total(k):return sum(r[k]for r in rows)
 cm=[[total('TP'),total('FN_decided'),sum(total(k)for k in('U_visible','out_of_task_S_visible','other_unresolved_visible'))],
     [total('FP'),total('TN'),sum(total(k)for k in('U_occluded','out_of_task_S_occluded','other_unresolved_occluded'))]]
 result=cm_metric(cm)
 result['U_breakdown']={k:total(k)for k in('U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded')}
 return result

assert bind(EVAL/'EVALUATION_SEAL.json')['sha256']==SEAL_SHA
seal=read(EVAL/'EVALUATION_SEAL.json');assert seal['completed']and seal['PASS']and seal['selected_model']is None
rp=EVAL/'CURRENT_FROZEN_BENCHMARK_RESULTS.json';assert bind(rp)['sha256']==RESULT_SHA
r=read(rp);assert tuple(r['models'])==MODELS and r['original_slots']==1784 and r['original_input_U']==45
sealed={Path(b['path']).name:b for b in seal['outputs']}
lp=EVAL/'ALL1784_NATIVE_LEDGER.csv';assert bind(lp)==sealed[lp.name]
with lp.open(newline='')as f:ledger=list(csv.DictReader(f))
assert len(ledger)==1784 and all(int(x['global_index'])==i for i,x in enumerate(ledger))
tables=r['tables'];visibility={};near={};full={};aux={}
for cohort in('paired_same_query_RGB','same_slot_pipeline'):
 visibility[cohort]={}
 for group in GROUPS:
  visibility[cohort][group]={}
  for model in MODELS:
   rr=[x for x in tables['metrics']if x['task']=='D_vs_E'and x['cohort']==cohort and x['model']==model and(group=='ALL'or x['group']==group)]
   visibility[cohort][group][model]=from_metrics(rr)
for group in GROUPS:
 near[group]={};full[group]={};aux[group]={}
 rr=ledger if group=='ALL'else[x for x in ledger if x['group']==group]
 for model in MODELS:
  nr=[x for x in tables['manual_near_D']if x['model']==model and(group=='ALL'or x['group']==group)]
  a={k:sum(x[k]for x in nr)for k in('N','pred_visible','pred_binary_hidden','unresolved')}
  a['D_recall']=ratio(a['pred_visible'],a['N']);a['false_E_rate']=ratio(a['pred_binary_hidden'],a['N']);near[group][model]=a
  cm=[[sum(x['human_state']==state and x[model+'_D_E_U']==pr for x in rr)for pr in('D','E','U')]for state in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
  full[group][model]=cm_metric(cm)
 for model in NEW:
  paired=[x for x in rr if x['paired_same_query_RGB_D_vs_E']=='True'and x['human_state']in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
  bystate={}
  for state in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
   ss=[x for x in paired if x['human_state']==state]
   n=sum(float(x[model+'_raw_entity_logit'])<0 for x in ss)
   q=sum(float(x[model+'_raw_quality_logit'])<0 for x in ss)
   bystate[state]={'N':len(ss),'entity_negative':n,'entity_negative_rate':ratio(n,len(ss)),
     'quality_far_response':q,'quality_far_response_rate':ratio(q,len(ss))}
  gated=[[sum(x['human_state']==state and(('U'if float(x[model+'_raw_entity_logit'])<0 else x[model+'_D_E_U'])==pr)for x in paired)for pr in('D','E','U')]for state in('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')]
  av=[x for x in rr if x['query_available']=='True']
  aux[group][model]={'scope':'Prediction-response diagnostic, no independent current entity truth',
    'original_available_rows':len(av),'entity_negative_all_available':sum(float(x[model+'_raw_entity_logit'])<0 for x in av),
    'by_human_role_paired1237':bystate,'secondary_entity_gated_paired_visibility':cm_metric(gated),
    'untrained_neutral_auxiliary':model=='C35','entity_accuracy_claimed':False}
quality={}
for scope in('ALL_ORIGINAL','INPUT_VALID'):
 quality[scope]={}
 for group in GROUPS:
  quality[scope][group]={}
  for x in tables['secondary_fixed_reference_quality_diagnostic']:
   if x['scope']==scope and x['group']==group:
    z=dict(x);z['CM_rows_far_near_columns_far_near_U']=json.loads(z['CM_rows_far_near_columns_far_near_U'])
    quality[scope][group][x['model']]=z
teacher_inventory={group:dict(Counter(x['human_state']for x in ledger if group=='ALL'or x['group']==group))for group in GROUPS}
assert teacher_inventory['ALL']=={'DIRECT_VISIBLE':1168,'SELF_OCCLUDED':136,'UNANNOTATED':356,'OUT_OF_FRAME':24,'EXTERNAL_OCCLUDED':92,'UNKNOWN_MAPPING':8}
assert all(visibility['paired_same_query_RGB']['ALL'][m]['N']==1237 for m in MODELS)
assert all(visibility['same_slot_pipeline']['ALL'][m]['N']==1248 for m in MODELS)
assert all(near['ALL'][m]['N']==376 for m in MODELS)
assert r['quality_authority']['new_primary_quality_gold_known']==0
assert (r['quality_authority']['secondary_near'],r['quality_authority']['secondary_far'])==(376,39)
counts=['both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved']
paired=[]
for cohort in('paired_same_query_RGB','same_slot_pipeline'):
 for old,new in [(a,b)for b in NEW for a in('v61','legacy','C16')]+[('C35','C36'),('C37','C38'),('C35','C37'),('C36','C38')]:
  for cls in('ALL_DE','DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
   rr=[x for x in tables['paired_recovery_and_damage']if x['cohort']==cohort and x['baseline']==old and x['new_model']==new and x['truth_class']==cls]
   x=dict(cohort=cohort,group='ALL',truth_class=cls,baseline=old,new_model=new,N=sum(z['N']for z in rr),**{k:sum(z[k]for z in rr)for k in counts})
   paired.append(x)
qualification={model:{'source_checks_all_pass':False,'source_current_allowed':False,'adopted':False}for model in NEW}
summary={'schema':'V156_COMPLETED_SAVED_SEVEN_CURRENT_SUMMARY','completed':True,'PASS':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'models':list(MODELS),'inputs':[bind(EVAL/'EVALUATION_SEAL.json'),bind(rp),bind(lp)],
 'primary_cohort':{'name':'paired_same_query_RGB','N':1237,'D':1147,'E':90,'WOOD_N':645,'SQUARE_N':592},
 'visibility':visibility,'full_registered_labelled_DE':full,
 'manual_near_D':near,'manual_near_D_scope':'Original frozen manual/D/reference-error<=3 diagnostic376 unchanged; not independently certified physical-native/pixel gold',
 'secondary_fixed_reference_quality':{'scope':r['quality_scope'],'new_primary_gold_known':0,'new_primary_gold_unknown':1784,
   'near':376,'far':39,'known':415,'unknown':1369,'original_input_U_among_known':3,'authority':{k:z for k,z in r['quality_authority'].items()if k!='original_annotation_bindings'},'metrics':quality},
 'auxiliary_response_diagnostic':aux,'teacher_inventory':teacher_inventory,
 'full_registered_model_inventory':tables['all1784_model_inventory'],
 'human_self_binary_response':tables['human_self_binary_response'],
 'paired_recovery_damage_ALL':paired,'paired_recovery_damage_by_material':tables['paired_recovery_and_damage'],
 'source_qualification_unchanged':qualification,'selected_model':None,'adopted_model':None,'fully_solved':False,
 'original_input_valid':1739,'original_input_U':45,'all_registered_slots':1784,
 'historical_three_all_table_fields_exact':r['historical_three_all_table_fields_exact'],
 'limits_KO':['현재223은 과거 반복 평가로 개발에 노출된 자료이며 blind test가 아니다. 고정7모델의 전이 진단이지 모델/문턱 선택이 아니다.',
   'legacy는 과거 실사 CAL 학습 모델이며 synthetic-only 비교자가 아니다. 과거 학습/개발 노출을 숨기거나 독립 평가라고 주장하지 않는다.',
   'C35/C36은2000회/seed1540 shared-hidden, C37/C38은8000회/seed1550 독립보조 경로다. 이전 모델과의 차이는 branch 분리만의 인과효과가 아니다.',
   '모든 새 모델은 source 기준 미달 그대로이며 채택하지 않는다. 보임 보존과 외부가림 검출 손익을 함께 보고한다.',
   'S136/화면밖24/미주석356/매핑미확정8은 외부가림 정답으로 바꾸지 않았다. 모든1784와 원입력U45를 보존했다.',
   '새 실사 좌표 품질 gold는0개다. secondary415행은 원래 고정 reference까지의 거리 진단이며 직접 physical-native/3px 정확도를 증명하지 않는다.',
   'entity 음수와 보조 gate 수치는 예측 반응이다. 현재223에는 독립 entity 정답이 없으므로 entity accuracy나 오거부율이라고 주장하지 않는다.'],
 'access':{'saved_evaluation_results_and_ledger_only':True,'models_features_weights_source_RGB_PnP_or_training_calls':0,'thresholds_labels_crops_qL_or_future_design_changed':False}}
write('CURRENT_RESULTS_SUMMARY_KO.json',summary)

# Figure1: overall full2x3 matrices and comparable metrics.
fig,axes=plt.subplots(4,2,figsize=(16,14));axes=axes.ravel()
labels={'legacy':'legacy · 과거 실사 CAL','C35':'C35 · 2000회','C36':'C36 · 2000회','C37':'C37 warm · 8000회','C38':'C38 fresh · 8000회'}
for i,model in enumerate(MODELS):
 ax=axes[i];m=visibility['paired_same_query_RGB']['ALL'][model];cm=np.array(m['CM_rows_D_E_columns_D_E_U']);ax.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='Blues',aspect='auto')
 for a in range(2):
  for b in range(3):ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=19,color='white'if cm[a,b]/cm[a].sum()>.65 else'#233247')
 ax.set(xticks=[0,1,2],xticklabels=['예측 D','예측 E','보류 U'],yticks=[0,1],yticklabels=['실제 보임1147','실제 외부가림90'],title=labels.get(model,model))
ax=axes[7];ax.axis('off')
tab=[[m]+[f'{visibility["paired_same_query_RGB"]["ALL"][m][k]:.1%}'for k in('D_recall','E_recall','E_precision','U_rate')]for m in MODELS]
t=ax.table(cellText=tab,colLabels=['모델','D recall','E recall','E precision','U'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.5);ax.set_title('동일1237개 기준 · U도 recall 분모에 유지')
fig.suptitle('현재 실사: 외부가림 검출 증가와 보이는 점 손상을 함께 비교',fontsize=21,y=.985)
fig.text(.06,.025,'공통 paired1237 = 목재645 + 정사각형592 · full1784 입력 중 U45는 별도 전체표에 보존\n고정 raw0/원 baseline 규칙 · legacy 실사 학습 이력 · 개발 노출 평가 · 새 모델 source 미달/미채택',fontsize=11)
fig.subplots_adjust(top=.93,bottom=.09,left=.12,right=.97,hspace=.60,wspace=.42);fig.savefig(OUT/'CURRENT_ALL1237_CONFUSIONS_KO.png',dpi=150);plt.close(fig)

# Figure2: per-material precision/recall/U and fixed near376, no GT-based filtering.
fig,axes=plt.subplots(2,2,figsize=(17,11));fig.subplots_adjust(top=.89,bottom=.16,hspace=.48,wspace=.28)
for col,(group,title)in enumerate((('WOOD_PRIMARY105','목재 paired645 (D616/E29)'),('SQUARE118','정사각형 paired592 (D531/E61)'))):
 ax=axes[0,col];ax.axis('off');data=[]
 for m in MODELS:
  z=visibility['paired_same_query_RGB'][group][m];data.append([m]+[f'{z[k]:.1%}'for k in('D_precision','D_recall','E_precision','E_recall','U_rate')])
 tab=ax.table(cellText=data,colLabels=['모델','D prec.','D recall','E prec.','E recall','U'],loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.65);ax.set_title(title,fontsize=15)
 ax=axes[1,col];yy=np.arange(7);left=np.zeros(7)
 for k,label,color in(('pred_visible','보임 정답','#3875ad'),('pred_binary_hidden','외부가림으로 오판','#c87842'),('unresolved','보류','#aaaeb5')):
  vals=np.array([near[group][m][k]for m in MODELS]);ax.barh(yy,vals,left=left,color=color,label=label)
  for y,n,l in zip(yy,vals,left):
   if n:ax.text(l+n/2,y,str(n),ha='center',va='center',fontsize=10,color='white'if color!='#aaaeb5'else'black')
  left+=vals
 ax.set(yticks=yy,yticklabels=MODELS,xlim=(0,near[group]['C16']['N']),xlabel='원래 고정 near D 진단 개수',title=f'기존 manual/D ≤3 진단 · N={near[group]["C16"]["N"]}');ax.invert_yaxis()
handles,leg=axes[1,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
fig.suptitle('목재·정사각형을 나누면 보이는 점 손상의 차이가 드러남',fontsize=21,y=.995)
fig.text(.06,.035,'manual near376 = 목재153 + 정사각형223 · 기존 평가 보조 지표를 그대로 유지, primary1237 필터에 쓰지 않음\n표의 U는 RGB 보류·기하 S·입력 불가를 원래 규칙으로 유지한 값. 사람 S를 외부가림 E로 바꾸지 않음.\n참조 좌표의 독립적인 physical-native/pixel gold 권위는 미확정이며 새 좌표 품질 gold로 승격하지 않음.',fontsize=11)
fig.savefig(OUT/'CURRENT_MATERIAL_AND_NEAR_DIAGNOSTICS_KO.png',dpi=150);plt.close(fig)

# Figure3: explicitly conditional coordinate deviation and auxiliary responses.
fig,axes=plt.subplots(2,2,figsize=(17,11));fig.subplots_adjust(top=.88,bottom=.16,hspace=.65,wspace=.35)
ax=axes[0,0];xx=np.arange(4)
for c,(name,label,color)in enumerate((('near_recall','near≤3','#3875ad'),('far_recall','far>20','#c59026'))):
 vals=[quality['ALL_ORIGINAL']['ALL'][m][name]for m in NEW];bars=ax.bar(xx+(c-.5)*.30,[z*100 for z in vals],.28,color=color,label=label)
 for bar,m in zip(bars,NEW):
  cm=quality['ALL_ORIGINAL']['ALL'][m]['CM_rows_far_near_columns_far_near_U'];t=1 if c==0 else 0;n=cm[t][t];ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(n),ha='center',fontsize=10)
ax.set(xticks=xx,xticklabels=['C35*','C36','C37','C38'],ylim=(0,118),ylabel='조건부 진단 recall (%)',title='원 고정 reference까지 거리 · near376/far39');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),ncol=2,frameon=False)
ax=axes[0,1];ax.axis('off');data=[]
for m in NEW:
 q=quality['ALL_ORIGINAL']['ALL'][m];data.append([m+'*'if m=='C35'else m,'—'if q['far_precision']is None else f'{q["far_precision"]:.1%}',str(q['U_known'])])
t=ax.table(cellText=data,colLabels=['모델','far precision','known415 중 U'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(13);t.scale(1,1.8);ax.set_title('새 품질 gold 0개 / 전1784 UNKNOWN',fontsize=16)
ax.text(0,-.07,'secondary: known415 / 미확정1369 / known의 원입력U3\n*C35 보조 손실0: 중립0 출력이며 학습된 품질 모델이 아님',transform=ax.transAxes,fontsize=11)
ax=axes[1,0]
for c,(state,label,color)in enumerate((('DIRECT_VISIBLE','사람 D에서 entity 음수','#3875ad'),('EXTERNAL_OCCLUDED','사람 E에서 entity 음수','#c87842'))):
 vals=[aux['ALL'][m]['by_human_role_paired1237'][state]for m in NEW];bars=ax.bar(xx+(c-.5)*.30,[z['entity_negative_rate']*100 for z in vals],.28,label=label,color=color)
 for bar,z in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(z['entity_negative']),ha='center',fontsize=10)
ax.set(xticks=xx,xticklabels=['C35*','C36','C37','C38'],ylim=(0,118),ylabel='음수 출력 비율 (%)',title='보조 대상 반응: paired D1147/E90 · entity 정확도 아님');ax.legend(loc='upper left',frameon=False)
ax=axes[1,1];ax.axis('off');inv=teacher_inventory['ALL'];names=[('DIRECT_VISIBLE','D 보임'),('EXTERNAL_OCCLUDED','E 외부가림'),('SELF_OCCLUDED','S 자기 가림'),('OUT_OF_FRAME','화면 밖'),('UNANNOTATED','미주석'),('UNKNOWN_MAPPING','매핑 미확정')]
t=ax.table(cellText=[[name,str(inv[k])]for k,name in names],colLabels=['전체1784 사람 역할','개수'],loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(12);t.scale(1,1.6);ax.set_title('전체 역할·미확정 유지',fontsize=15)
fig.suptitle('보조 출력 진단은 가시성 평가·좌표 gold와 구분',fontsize=21,y=.985)
fig.text(.06,.035,'원 reference 거리 진단만 제공: 숨은 점·amodal·미확정 mapping을 실제 위치 정답으로 만들지 않음\nentity 독립 gold 없음 · primary visibility는 보조 점수/좌표 오차로 보류·필터하지 않음 · 고정 문턱/모델 미선택',fontsize=11)
fig.savefig(OUT/'CURRENT_AUXILIARY_AUTHORITY_DIAGNOSTICS_KO.png',dpi=150);plt.close(fig)
qa={'completed':True,'PASS':True,'summary':bind(OUT/'CURRENT_RESULTS_SUMMARY_KO.json'),'code':bind(Path(__file__)),
 'inputs':summary['inputs'],'figures':[bind(OUT/x)for x in('CURRENT_ALL1237_CONFUSIONS_KO.png','CURRENT_MATERIAL_AND_NEAR_DIAGNOSTICS_KO.png','CURRENT_AUXILIARY_AUTHORITY_DIAGNOSTICS_KO.png')],
 'numeric_checks':{'saved_primary_count_aggregation':True,'original1237_1248_1784_376_preserved':True,'rawpred_ledger_aux_diagnostic_only_noentitygold':True,'new_primary_quality_gold_zero':True,'sourcequalification_unchanged':True},
 'actual_view_image_QA':'PENDING separate final receipt','access':summary['access']}
write('NUMERIC_AND_RENDER_QA.json',qa)
print(json.dumps({'completed':True,'summary':bind(OUT/'CURRENT_RESULTS_SUMMARY_KO.json'),'primary_all':visibility['paired_same_query_RGB']['ALL']},ensure_ascii=False))
