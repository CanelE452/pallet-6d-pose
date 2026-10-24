"""Saved-only independent row arithmetic. No production metric/helper imports."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,time
V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_frozen_native_current_benchmark_20261024_v143')
OUT=V/'saved_current_benchmark_evaluation_v1';H=V/'independent_saved_evaluation_review'
MODELS=('v61','legacy','C16','C29','C32');GROUPS=('WOOD_PRIMARY105','SQUARE118')
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'
TASKS={'D_vs_E':(D,E),'D_vs_S':(D,S),'D_vs_ES':(D,E,S)}
VRESP,ERESP,URESP,SRESP,ORESP='VISIBLE','ABSENT_NONVISIBLE','ABSTAIN','OUT_OF_TASK_GEOMETRY_S','OTHER_UNRESOLVED'
def bind(p):return dict(path=str(Path(p)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert '_v112'not in b['path'].lower();assert bind(b['path'])==b,b['path'];return Path(b['path'])
def rc(path):
 with Path(path).open(newline='')as f:return list(csv.DictReader(f))
def key(r):return r['id'],str(r['native_corner'])
def yes(r,k):return r[k]=='True'
def ratio(a,b):return a/b if b else None
def numeq(a,b):
 if a is None:return b is None or b==''
 return b is not None and b!='' and abs(float(a)-float(b))<1e-14
start=time.monotonic();seal=read(OUT/'EVALUATION_SEAL.json')
assert bind(OUT/'EVALUATION_SEAL.json')['sha256']=='eadf271669e3b32a3522e1ed3cc366de8b7c7aed4b2535d9c8b6931f8b6a065a'
assert seal['completed']and seal['PASS']and seal['selected_model']is None and seal['source_admission_changed']is False
p=read(check(seal['protocol']));pred=read(check(seal['prediction_seal']));assert pred['completed']and pred['PASS']and pred['GT_read']is False and pred['selected_model']is None
assert seal['new_forward_fit_RGB_PnP']==0
# Every label/annotation/prior body read begins only below the completed seal.
truth=rc(check(p['truth']));manual=rc(check(p['manual']));prior=[json.loads(s)for s in check(p['prior']).read_text().splitlines()if s]
assert len(truth)==len(prior)==1784 and len({r['id']for r in truth})==223
truthmap={key(r):r for r in truth};priormap={key(r):r for r in prior};assert len(truthmap)==len(priormap)==1784 and set(truthmap)==set(priormap)
assert Counter(r['group']for r in truth)==Counter(WOOD_PRIMARY105=840,SQUARE118=944)
score={}
for name in ('C16','C29','C32'):
 binding=p['C16_scores']if name=='C16'else pred['models'][name]['scores_csv'];rr=rc(check(binding));assert len(rr)==1784
 score[name]={key(r):r for r in rr};assert len(score[name])==1784 and set(score[name])==set(truthmap)
 assert list(score[name])==list(priormap)
 for r in rr:
  old=priormap[key(r)];assert r['valid']==str(old['valid'])
  for x in ('query_x','query_y','roi_side'):
   a,b=r[x],old[x]
   missing=lambda v:v is None or str(v).lower()in('','none','nan')
   assert missing(a)and missing(b)or not missing(a)and not missing(b)and float(a)==float(b)
  assert r['authoritative_decision']==(('D'if float(r['raw_query_aux_hidden_logit'])<0 else'E')if r['valid']=='True'else'UNKNOWN')
def response(r,model,task='D_vs_E',inventory=False):
 if model=='v61':
  if inventory and (r['query_valid']!='True'or r.get('present_at_fixed_0_5')not in('True','False')):return ORESP
  return VRESP if yes(r,'present_at_fixed_0_5')else ERESP
 if model=='legacy':
  state=r['legacy_stored_state']
  if state in('B','I'):return VRESP
  if state=='O':return ERESP
  if state=='S':return SRESP if task=='D_vs_E'else ERESP
  return URESP if state=='U'else ORESP
 rscore=score[model][key(r)]
 return ORESP if rscore['valid']!='True'else VRESP if float(rscore['raw_query_aux_hidden_logit'])<0 else ERESP

def metric(rows,model,task):
 cm=Counter((r['human_state']==D,response(r,model,task))for r in rows)
 pos=sum(r['human_state']==D for r in rows);neg=len(rows)-pos
 tp,fn,fp,tn=cm[True,VRESP],cm[True,ERESP],cm[False,VRESP],cm[False,ERESP]
 up,un,sp,sn,op,on=(cm[True,URESP],cm[False,URESP],cm[True,SRESP],cm[False,SRESP],cm[True,ORESP],cm[False,ORESP])
 assert tp+fn+up+sp+op==pos and fp+tn+un+sn+on==neg
 return dict(N=len(rows),frames=len({r['id']for r in rows}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),abstention_rate=ratio(up+un,len(rows)),out_of_task_S_rate=ratio(sp+sn,len(rows)),unresolved_rate=ratio(up+un+sp+sn+op+on,len(rows)),decision_coverage=ratio(tp+fn+fp+tn,len(rows)),correct_fraction_all_GT=ratio(tp+tn,len(rows)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))
metrics=[];confusion=[];selfrows=[];transitions=[];primary=[]
for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
 for group in GROUPS:
  for task,states in TASKS.items():
   flag=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
   rr=[r for r in truth if r['group']==group and yes(r,flag)and r['human_state']in states]
   if not any(r['human_state']!=D for r in rr):continue
   for model in MODELS:
    m=dict(cohort=cohort,group=group,task=task,model=model,**metric(rr,model,task));metrics.append(m)
    if task=='D_vs_E':
     primary.append(dict(cohort=cohort,group=group,model=model,CM_D_E_rows_D_E_U_columns=[[m['TP'],m['FN_decided'],m['U_visible']+m['out_of_task_S_visible']+m['other_unresolved_visible']],[m['FP'],m['TN'],m['U_occluded']+m['out_of_task_S_occluded']+m['other_unresolved_occluded']]],D_precision=m['visible_precision'],D_recall=m['visible_recall_all_GT'],E_precision=m['occluded_precision'],E_recall=m['occluded_recall_all_GT'],U_rate_full_GT=m['unresolved_rate'],N=m['N']))
    for state in states:
     zz=[r for r in rr if r['human_state']==state];c=Counter(response(r,model,task)for r in zz)
     confusion.append(dict(cohort=cohort,group=group,task=task,model=model,human_state=state,N=len(zz),pred_visible=c[VRESP],pred_binary_hidden=c[ERESP],legacy_U=c[URESP],out_of_task_geometry_S=c[SRESP],input_or_operator_U=c[ORESP]))
  rr=[r for r in truth if r['group']==group and yes(r,cohort+'_D_vs_ES')and r['human_state']==S]
  for model in MODELS:
   c=Counter(response(r,model,'D_vs_S')for r in rr)
   selfrows.append(dict(cohort=cohort,group=group,model=model,human_S_support=len(rr),binary_hidden_response=c[ERESP],visible_response=c[VRESP],unresolved=len(rr)-c[ERESP]-c[VRESP],human_S_binary_hidden_response_rate=ratio(c[ERESP],len(rr))))
  rr=[r for r in truth if r['group']==group and yes(r,cohort+'_D_vs_E')and r['human_state']in(D,E)]
  for new in ('C29','C32'):
   for old in ('C16','v61','legacy'):
    c=Counter()
    for r in rr:
     a,b=response(r,old),response(r,new);gt=VRESP if r['human_state']==D else ERESP
     if b not in(VRESP,ERESP):c['new_unresolved']+=1
     elif a not in(VRESP,ERESP):c['baseline_unresolved_new_correct'if b==gt else'baseline_unresolved_new_wrong']+=1
     else:c['both_correct'if a==gt and b==gt else'baseline_correct_new_wrong'if a==gt else'baseline_wrong_new_correct'if b==gt else'both_wrong']+=1
    names=('both_correct','baseline_correct_new_wrong','baseline_wrong_new_correct','both_wrong','baseline_unresolved_new_correct','baseline_unresolved_new_wrong','new_unresolved')
    assert sum(c.values())==len(rr)
    transitions.append(dict(cohort=cohort,group=group,task='D_vs_E',baseline=old,new_model=new,N=len(rr),**{k:c[k]for k in names}))
selected=[r for r in manual if r['annotation_coordinate_source']=='manual_click'and r['human_state']==D and float(r['initial_reference_error_px'])<=3 and yes(r,'query_valid')]
assert len(selected)==376 and Counter(r['group']for r in selected)==Counter(WOOD_PRIMARY105=153,SQUARE118=223)
for r in selected:
 assert all(r[k]==truthmap[key(r)][k]for k in('image_sha256','query_x','query_y'))
manualresults=[]
for group in GROUPS:
 rr=[r for r in selected if r['group']==group]
 for model in MODELS:
  c=Counter(response(r,model,'D_vs_ES')for r in rr)
  manualresults.append(dict(group=group,model=model,N=len(rr),pred_visible=c[VRESP],pred_binary_hidden=c[ERESP],unresolved=len(rr)-c[VRESP]-c[ERESP],semantic_D_recall=ratio(c[VRESP],len(rr))))
ledger=[];inventory=[]
for r in truth:
 k=key(r);item=dict(id=k[0],native_corner=int(k[1]),group=r['group'],human_state=r['human_state'],mapping_status=r['mapping_status'],paired_same_query_RGB_D_vs_E=r['paired_same_query_RGB_D_vs_E'],same_slot_pipeline_D_vs_E=r['same_slot_pipeline_D_vs_E'],initial_reference_error_px=r['initial_reference_error_px'],original_query_valid=r['query_valid'],legacy_original_state=r['legacy_stored_state'])
 for model in MODELS:
  s=response(r,model,inventory=True);item[model+'_original_response']=s;item[model+'_D_E_U']='D'if s==VRESP else'E'if s==ERESP else'U'
 ledger.append(item)
for group in ('ALL',*GROUPS):
 rr=ledger if group=='ALL'else[r for r in ledger if r['group']==group]
 for model in MODELS:
  c=Counter(r[model+'_D_E_U']for r in rr);s=Counter(r[model+'_original_response']for r in rr)
  inventory.append(dict(group=group,model=model,all_registered_slots=len(rr),D=c['D'],E_binary_response=c['E'],U=c['U'],U_rate=c['U']/len(rr),legacy_abstain=s[URESP],out_of_task_geometry_S=s[SRESP],other_unresolved=s[ORESP],human_state_inventory=json.dumps(dict(Counter(r['human_state']for r in rr)),sort_keys=True)))
# Compare only after all independent row counts and summaries above are complete.
for b in seal['outputs']:check(b)
producer=read(OUT/'CURRENT_FROZEN_BENCHMARK_RESULTS.json');tables=producer['tables']
assert producer['selected_model']is None and producer['adopted_model']is None and producer['source_eligible']=={'C29':False,'C32':False}
def compare_rows(ours,theirs,keys):
 a={tuple(r[k]for k in keys):r for r in ours};b={tuple(r[k]for k in keys):r for r in theirs};assert len(a)==len(ours)and set(a)==set(b)
 for k,row in a.items():
  for field,value in row.items():
   other=b[k][field]
   if isinstance(value,(float,int))or value is None:assert numeq(value,other),(k,field,value,other)
   else:assert value==other,(k,field,value,other)
 return len(a)
verified={}
verified['all_primary_and_self_hidden_binary_metrics']=compare_rows(metrics,tables['metrics'],('cohort','group','task','model'))
verified['all_human_state_confusions']=compare_rows(confusion,tables['confusion'],('cohort','group','task','model','human_state'))
verified['human_S_separate_responses']=compare_rows(selfrows,tables['human_self_binary_response'],('cohort','group','model'))
verified['manual376']=compare_rows(manualresults,tables['manual_near_D'],('group','model'))
verified['paired_recovery_damage']=compare_rows(transitions,tables['paired_recovery_and_damage'],('cohort','group','baseline','new_model'))
verified['full_inventory']=compare_rows(inventory,tables['all1784_model_inventory'],('group','model'))
ledgerstored=rc(OUT/'ALL1784_NATIVE_LEDGER.csv');assert len(ledgerstored)==1784
for a,b in zip(ledger,ledgerstored):
 for k,value in a.items():assert str(value)==b[k],(key(b),k,value,b[k])
# Replay every historical row for tasks with nonempty negative support.
# SQUARE has no human S, so its empty D_vs_S task is not fabricated.
countfields=('N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded')
historical={}
for field,names,rename in [('historical_original',('v61','legacy'),{}),('historical_C16_metrics',('C16',),{'C16':'V124_C16_RAW'})]:
 rr=rc(check(p[field]));lookup={(r['cohort'],r['group'],r['task'],r['model']):r for r in rr};seen=0
 for r in metrics:
  if r['model']not in names:continue
  k=(r['cohort'],r['group'],r['task'],rename.get(r['model'],r['model']));old=lookup[k]
  assert all(int(r[x])==int(old[x])for x in countfields),(field,k);seen+=1
 historical[field]=seen
assert historical['historical_original']==sum(r['model']in('v61','legacy')for r in metrics)
assert historical['historical_C16_metrics']==sum(r['model']=='C16'for r in metrics)
assert historical['historical_original']==producer['historical_v61_legacy_rows_exact']
assert historical['historical_C16_metrics']==producer['historical_C16_rows_exact']
cohortn={c:sum(1 for r in truth if yes(r,c+'_D_vs_E')and r['human_state']in(D,E))for c in ('paired_same_query_RGB','same_slot_pipeline')};assert list(cohortn.values())==[1237,1248]
stateinv={group:dict(Counter(r['human_state']for r in truth if group=='ALL'or r['group']==group))for group in ('ALL',*GROUPS)}
mappinginv=dict(Counter(r['mapping_status']for r in truth))
result=dict(completed=True,PASS=True,source='Independent row loops; zero imports/calls of production metric/evaluator/model helpers',primary_D_E_by_group=primary,all_metrics=metrics,manual_D376=manualresults,paired_recovery_damage=transitions,all1784_inventory=inventory,human_state_inventory=stateinv,mapping_status_inventory=mappinginv,cohort_denominators=cohortn,verified_producer_rows=verified,historical_counts_replayed=historical,selected_model=None,source_ineligible_unchanged=True)
f=H/'INDEPENDENT_RECALCULATED_RESULTS.json'
with f.open('x')as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
review=dict(completed=True,PASS=True,code=bind(__file__),evaluation_seal=bind(OUT/'EVALUATION_SEAL.json'),prediction_seal=seal['prediction_seal'],recalculated=bind(f),labels_first_read_only_after_completed_PASS_seal=True,independent_producer_metric_helper_calls=0,all1784_and_unknown_S_mapping_inventory_preserved=True,cohort_denominators=cohortn,manual_D376=376,verified_producer_rows=verified,historical_counts_replayed=historical,source_selectednull_ineligible_unchanged=True,selected_model=None,model_checkpoint_RGB_feature_encoder_initializer_forward_fit_threshold_or_PnP_reads_or_calls=0,V112_access=0,seconds=time.monotonic()-start)
f=H/'REVIEW.json'
with f.open('x')as stream:json.dump(review,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(dict(completed=True,PASS=True,review=bind(f),primary=primary,manual=manualresults,states=stateinv,historical=historical)),flush=True)
