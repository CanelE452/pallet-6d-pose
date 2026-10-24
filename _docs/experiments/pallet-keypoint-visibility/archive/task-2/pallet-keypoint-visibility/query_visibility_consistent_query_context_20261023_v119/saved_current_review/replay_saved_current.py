"""Independent C14 saved-current replay, all1784 rows. No inference, weights or images."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math
import numpy as np
V=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;R=V/'private_evaluation/C14_FROZEN_CURRENT223';P=V.parent
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED';GROUPS=['WOOD_PRIMARY105','SQUARE118'];TASKS={'D_vs_E':[D,E],'D_vs_S':[D,S],'D_vs_ES':[D,E,S]}
MODELS=['v61','legacy','V91_RAW','V91_PLUS_OLD_GEOMETRY','V111_C11_RAW','V111_C11_PLUS_OLD_GEOMETRY','V119_C14_RAW','V119_C14_PLUS_OLD_GEOMETRY']
def read(p):return json.loads(Path(p).read_text())
def csvrows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def b(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def chk(d):assert b(Path(d['path']))['sha256']==d['sha256']
def key(r):return str(r['id']),str(r['native_corner'])
def flag(x):
 if x is True or str(x).lower()=='true':return True
 if x is False or str(x).lower()=='false':return False
 if x is None or str(x).lower()in('','none','null'):return None
 raise AssertionError('Malformed flag')
def same_num(x,y):
 def num(t):
  try:return float(t)
  except(TypeError,ValueError):return float('nan')
 a,c=num(x),num(y);return(math.isnan(a)and math.isnan(c))or a==c
checks=0
def eq(want,have):
 global checks
 for k,v in want.items():
  assert k in have,k;w=have[k]
  if isinstance(v,(float,np.floating))or v is None:assert(v is None and w is None)or math.isclose(float(v),float(w),rel_tol=0,abs_tol=1e-14),(k,v,w)
  else:assert v==w,(k,v,w)
  checks+=1
inf=read(R/'INFERENCE_SEAL.json');ev=read(R/'evaluation/EVALUATION_SEAL.json');assert b(R/'INFERENCE_SEAL.json')['sha256']=='68c9dad674e1004ed3dd8347551ebef271168f48ead83a6f5f7ab752c82f5a45';assert b(R/'evaluation/EVALUATION_SEAL.json')['sha256']=='a9a30f52d37f64bfaba26ed9910c33ed155bae92f8998b0841634c0d039e3954'
for z in inf['outputs']+ev['files']+[inf['entry'],ev['entry'],inf['source_model_freeze'],inf['C11_inference_seal']]:chk(z)
assert inf['completed']and inf['PASS']and inf['model_unchanged']and inf['head_state_before']==inf['head_state_after']and inf['head_repeat_exact']and inf['head_calls']==2 and inf['encoder_RGB_initializer_calls']==inf['optimizer_updates']==inf['PnP']==0
assert not inf['current_GT_or_baseline_scores_read']and not inf['current_used_for_selection']
freeze=read(inf['source_model_freeze']['path']);assert freeze['selected_update']==1200 and freeze['threshold_logit']==0 and freeze['checkpoint']==inf['checkpoint']
e=read(ev['entry']['path']);bindings=e['bindings']
for x in bindings.values():chk(x)
assert ev['source_freeze']==inf['source_model_freeze']and ev['inference']==b(R/'INFERENCE_SEAL.json')
truth=csvrows(bindings['evaluation_manifest']['path']);prior=rows(bindings['prior_input_rows']['path']);manual=csvrows(bindings['manual_provenance']['path']);minimal=csvrows(R/'HEAD_ONLY_SCORES.csv');new=csvrows(R/'evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv');v91=csvrows(bindings['baseline_v91_scores']['path']);c11=csvrows(bindings['baseline_C11_scores']['path']);out=read(R/'evaluation/CURRENT_PRIMARY_RESULTS.json');tab=out['tables']
with np.load(R/'RAW_HEAD_OUTPUTS.npz',allow_pickle=False)as f:a={k:f[k]for k in f.files}
# Read only original frozen feature values and13 metadata members; no old-logit member.
chk(inf['C11_current_packet'])
with np.load(inf['C11_current_packet']['path'],allow_pickle=False)as f:
 features=f['features1928'];assert hashlib.sha256(features.tobytes()).hexdigest()==inf['feature_values_SHA256']and features.shape==(1784,1928)and features.dtype==np.float32
 for k in inf['input_fields_read']:
  if k!='features1928':assert np.array_equal(a[k],f[k],equal_nan=True)if f[k].dtype.kind in 'fc'else np.array_equal(a[k],f[k])
assert len(prior)==len(truth)==len(new)==len(c11)==len(v91)==len(minimal)==1784
assert a['query_logit'].shape==(1784,)and a['query_logit'].dtype==np.float32 and np.isfinite(a['query_logit']).all()and np.isfinite(a['raw_head_logit_before_U']).all()
assert np.array_equal(a['query_logit'],np.where(a['valid'],a['raw_head_logit_before_U'],np.float32(0)))and np.array_equal(a['valid'],a['query_available'])and np.array_equal(a['valid'],a['prior_valid']&a['context_available']&a['local_central4_available'])
assert (~a['valid']).sum()==(~a['prior_valid']).sum()==45 and not(a['prior_valid']&~a['valid']).any()
bytruth={key(r):r for r in truth};byprior={key(r):r for r in prior};assert len(bytruth)==len(byprior)==1784
# The immutable C12 saved evaluation is compared descriptively on exactly the same bindings.
C12R=P/'query_visibility_material_augmented_query_head_20261023_v114/private_evaluation/C12_FROZEN_CURRENT223'
c12ev=read(C12R/'evaluation/EVALUATION_SEAL.json');assert b(C12R/'evaluation/EVALUATION_SEAL.json')['sha256']=='adefecde194c0e0e76c16f3661b195f4a8ba2968b584b3a223bdf4d705949399'
for z in c12ev['files']+[c12ev['entry'],c12ev['inference']]:chk(z)
assert c12ev['completed']and c12ev['PASS']and read(c12ev['entry']['path'])['bindings']==bindings
c12rows=csvrows(C12R/'evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv');c12tab=read(C12R/'evaluation/CURRENT_PRIMARY_RESULTS.json')['tables']
C13R=P/'query_visibility_same_material_query_head_20261023_v118/private_evaluation/C13_FROZEN_CURRENT223'
c13ev=read(C13R/'evaluation/EVALUATION_SEAL.json');assert b(C13R/'evaluation/EVALUATION_SEAL.json')['sha256']=='713a5de69159f73a2b9414475588f4c9f79ea4e3dcd59356aaacc2c670c52966'
for item in c13ev['files']+[c13ev['entry'],c13ev['inference']]:chk(item)
assert c13ev['completed']and c13ev['PASS']and read(c13ev['entry']['path'])['bindings']==bindings
c13rows=csvrows(C13R/'evaluation/SCORES_WITH_ORIGINAL_GEOMETRY_JOIN.csv');c13tab=read(C13R/'evaluation/CURRENT_PRIMARY_RESULTS.json')['tables']
records={}
for name,data,zfield in [('V91',v91,'raw_physical_external_logit'),('V111_C11',c11,'raw_query_aux_hidden_logit'),('V119_C14',new,'raw_query_aux_hidden_logit'),('V114_C12',c12rows,'raw_query_aux_hidden_logit'),('V118_C13',c13rows,'raw_query_aux_hidden_logit')]:
 assert [key(x)for x in data]==[key(x)for x in prior];index={}
 for i,r in enumerate(data):
  p=prior[i];t=bytruth[key(p)];valid=flag(r['valid']);z=float(r[zfield]);assert math.isfinite(z)
  geo=flag(r['geometry_self_hidden']);guard=flag(r['geometry_self_hidden_if_valid']);assert guard==(geo if valid else None)
  assert r['authoritative_decision']==(('D'if z<0 else'E')if valid else'UNKNOWN')and flag(r['decision_visible'])==((z<0)if valid else None)
  assert r['image_sha256']==p['image_sha256']==t['image_sha256']
  for f in('query_x','query_y','roi_side','pixel_x','pixel_y'):assert same_num(r[f],p[f])
  assert geo==p['geometry_self_hidden']and flag(r['saved_roi_pixel_valid'])==p['saved_roi_pixel_valid']and(not valid or p['valid'])
  for f in('query_x','query_y'):
   if t[f]and r[f]:assert abs(float(t[f])-float(r[f]))<=1e-5
  index[key(r)]=dict(valid=valid,z=z,geo=guard)
 records[name]=index
for i,(r,p,s)in enumerate(zip(new,prior,minimal)):
 assert key(r)==key(s)==(str(a['id'][i]),str(int(a['native_corner'][i])))and flag(r['valid'])==bool(a['valid'][i])
 assert float(r['raw_query_aux_hidden_logit'])==float(s['raw_query_aux_hidden_logit'])==float(a['query_logit'][i])and flag(s['valid'])==bool(a['valid'][i])
 for j,f in enumerate(('query_x','query_y')):assert same_num(p[f],a['original_query_xy'][i,j])and same_num(s[f],p[f])
 assert same_num(p['roi_side'],a['original_side_L'][i])and same_num(s['roi_side'],p['roi_side'])
 assert s['authoritative_decision']==r['authoritative_decision']
 if a['valid'][i]:assert float(r['p_visible'])==float(a['p_visible_FP32'][i])and 0<=float(r['p_visible'])<=1
 else:assert r['p_visible']==''and r['geometry_self_hidden_if_valid']==''
# Integer response:0visible,1hidden,2legacyU,3legacygeometryS(out-of-task),4input/operatorU.
def response(r,model,task):
 if model=='v61':return 0 if flag(r['present_at_fixed_0_5'])else 1
 if model=='legacy':
  s=r['legacy_stored_state']
  if s in('B','I'):return 0
  if s=='O':return 1
  if s=='S':return 3 if task=='D_vs_E'else 1
  return 2 if s=='U'else 4
 prefix='V91'if model.startswith('V91')else'V111_C11'if model.startswith('V111')else'V114_C12'if model.startswith('V114')else'V118_C13'if model.startswith('V118')else'V119_C14';rec=records[prefix][key(r)]
 if not rec['valid']:return 4
 if model.endswith('PLUS_OLD_GEOMETRY')and rec['geo']is True:return 1
 return 0 if rec['z']<0 else 1
rat=lambda x,n:float(x/n)if n else None
def calc(rr,model,task):
 cm=np.zeros((2,5),np.int64)
 for r in rr:cm[0 if r['human_state']==D else 1,response(r,model,task)]+=1
 pos,neg=map(int,cm.sum(1));tp,fn,up,sp,op=map(int,cm[0]);fp,tn,un,sn,on=map(int,cm[1]);n=pos+neg;dec=tp+fn+fp+tn
 return dict(N=n,frames=len({r['id']for r in rr}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=rat(tp,tp+fp),visible_recall_all_GT=rat(tp,pos),occluded_precision=rat(tn,tn+fn),occluded_recall_all_GT=rat(tn,neg),abstention_rate=rat(up+un,n),out_of_task_S_rate=rat(sp+sn,n),unresolved_rate=rat(up+un+sp+sn+op+on,n),decision_coverage=rat(dec,n),correct_fraction_all_GT=rat(tp+tn,n),selective_accuracy=rat(tp+tn,dec))
metrics={};conf={};selfs={};nears={};errors={};mats={}
for cohort in('paired_same_query_RGB','same_slot_pipeline'):
 for group in GROUPS:
  for task,states in TASKS.items():
   flagfield=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task);rr=[r for r in truth if r['group']==group and flag(r[flagfield])and r['human_state']in states]
   if not any(r['human_state']!=D for r in rr):continue
   for model in MODELS:
    k=(cohort,group,task,model);metrics[k]=calc(rr,model,task)
    for st in states:
     sub=[r for r in rr if r['human_state']==st];counts=Counter(response(r,model,task)for r in sub)
     conf[k+(st,)]=dict(N=len(sub),pred_visible=counts[0],pred_binary_hidden=counts[1],legacy_U=counts[2],out_of_task_geometry_S=counts[3],input_or_operator_U=counts[4])
  rr=[r for r in truth if r['group']==group and flag(r[cohort+'_D_vs_ES'])and r['human_state']==S]
  for model in MODELS:
   c=Counter(response(r,model,'D_vs_S')for r in rr);selfs[(cohort,group,model)]=dict(human_S_support=len(rr),binary_hidden_response=c[1],visible_response=c[0],unresolved=len(rr)-c[0]-c[1],human_S_binary_hidden_response_rate=rat(c[1],len(rr)),learned_self_prediction=False,external_vs_self_identified=False)
for r in tab['metrics']:eq(metrics[(r['cohort'],r['group'],r['task'],r['model'])],r)
for r in tab['confusion']:eq(conf[(r['cohort'],r['group'],r['task'],r['model'],r['human_state'])],r)
for r in tab['human_self_binary_response']:eq(selfs[(r['cohort'],r['group'],r['model'])],r)
assert len(metrics)==len(tab['metrics'])==80 and len(conf)==len(tab['confusion'])and len(selfs)==len(tab['human_self_binary_response'])
selected=[r for r in manual if r['annotation_coordinate_source']=='manual_click'and r['human_state']==D and float(r['initial_reference_error_px'])<=3 and flag(r['query_valid'])]
for r in selected:
 t=bytruth[key(r)];assert all(r[f]==t[f]for f in('image_sha256','query_x','query_y'))
for g in GROUPS:
 rr=[r for r in selected if r['group']==g];assert len(rr)==(153 if g==GROUPS[0]else 223)
 for model in MODELS:
  c=Counter(response(r,model,'D_vs_ES')for r in rr);nears[(g,model)]=dict(N=len(rr),pred_visible=c[0],pred_binary_hidden=c[1],unresolved=len(rr)-c[0]-c[1],semantic_D_recall=rat(c[0],len(rr)))
for r in tab['manual_near_D']:eq(nears[(r['group'],r['model'])],r)
manualmap={key(r):r for r in manual}
for cohort in('paired_same_query_RGB','same_slot_pipeline'):
 for group in GROUPS:
  for bucket in('LE3','GT3','MISSING_REFERENCE_ERROR'):
   rr=[]
   for r in truth:
    if r['group']!=group or not flag(r[cohort+'_D_vs_E'])or r['human_state']not in(D,E):continue
    try:err=float(r['initial_reference_error_px'])
    except(ValueError,TypeError):err=float('nan')
    k='MISSING_REFERENCE_ERROR'if not math.isfinite(err)else'LE3'if err<=3 else'GT3'
    if k==bucket:rr.append(r)
   nmanual=sum(manualmap.get(key(r),{}).get('annotation_coordinate_source')=='manual_click'for r in rr)
   for model in MODELS:errors[(cohort,group,bucket,model)]=dict(manual_click_reference_N=nmanual,other_or_unspecified_reference_N=len(rr)-nmanual,**calc(rr,model,'D_vs_E'))
for r in tab['coordinate_error_bins']:eq(errors[(r['cohort'],r['group'],r['error_bin'],r['model'])],r)
for r in tab['D_vs_external_E_2x3']:
 q=metrics[(r['cohort'],r['group'],'D_vs_E',r['model'])];want=dict(human_D_N=q['visible_support'],human_external_E_N=q['occluded_support'],D_pred_D=q['TP'],D_pred_hidden=q['FN_decided'],D_pred_U=q['U_visible']+q['out_of_task_S_visible']+q['other_unresolved_visible'],E_pred_D=q['FP'],E_pred_hidden=q['TN'],E_pred_U=q['U_occluded']+q['out_of_task_S_occluded']+q['other_unresolved_occluded'],D_precision=q['visible_precision'],D_recall=q['visible_recall_all_GT'],external_E_response_precision=q['occluded_precision'],external_E_response_recall=q['occluded_recall_all_GT'],unresolved_rate=q['unresolved_rate']);eq(want,r);mats[(r['cohort'],r['group'],r['model'])]=[[want['D_pred_D'],want['D_pred_hidden'],want['D_pred_U']],[want['E_pred_D'],want['E_pred_hidden'],want['E_pred_U']]]
eq(dict(rows=1784,prior_input_U=45,new_total_U=45,extra_operator_U=0,V91_total_U=sum(not flag(r['valid'])for r in v91),confidence_abstention_trained=False),tab['availability'])
for r in tab['availability_by_group']:
 keys=[key(t)for t in truth if t['group']==r['group']];pu=sum(not byprior[k]['valid']for k in keys);nu=sum(not records['V119_C14'][k]['valid']for k in keys);vu=sum(not records['V91'][k]['valid']for k in keys)
 eq(dict(all_native_slots=len(keys),prior_input_U=pu,extra_operator_U=nu-pu,new_total_U=nu,V91_total_U=vu),r)
# Historical60 rows: independently calculated original count fields, no denominator pruning.
historical=0;countfields=['N','visible_support','occluded_support','TP','FN_decided','FP','TN','U_visible','U_occluded','out_of_task_S_visible','out_of_task_S_occluded','other_unresolved_visible','other_unresolved_occluded']
for source,models in [('original_baseline_metrics',{'v61','legacy'}),('historical_v91_metrics',{'V91_RAW','V91_PLUS_OLD_GEOMETRY'}),('historical_C11_metrics',{'V111_C11_RAW','V111_C11_PLUS_OLD_GEOMETRY'})]:
 for row in csvrows(bindings[source]['path']):
  if row['model']not in models:continue
  own=metrics[(row['cohort'],row['group'],row['task'],row['model'])]
  for f in countfields:assert own[f]==int(row[f])
  historical+=1
assert historical==60 and out['historical_v61_legacy_V91_rows_exact']==40 and out['historical_C11_rows_exact']==20
primary=[dict(group=g,model='V119_C14_RAW',CM=mats[('paired_same_query_RGB',g,'V119_C14_RAW')],metrics=metrics[('paired_same_query_RGB',g,'D_vs_E','V119_C14_RAW')])for g in GROUPS]
assert primary[0]['CM']==[[599,17,0],[20,9,0]]and primary[1]['CM']==[[531,0,0],[55,6,0]]
# Recompute all20 C12 metric rows from its saved predictions against identical frozen truth.
c12metrics={};c12_near={};# Independently replay all20 C13 rows and full frozen-cohort D/E/S transitions.
c13metrics={};c13comparison=[];c13metricrows=0
for row in c13tab['metrics']:
 if row['model']not in('V118_C13_RAW','V118_C13_PLUS_OLD_GEOMETRY'):continue
 cohort,group,task,model=(row[k]for k in('cohort','group','task','model'))
 flagfield=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
 rr=[r for r in truth if r['group']==group and flag(r[flagfield])and r['human_state']in TASKS[task]]
 own=calc(rr,model,task);eq(own,row);c13metrics[(cohort,group,task,model)]=own;c13metricrows+=1
 current=metrics[(cohort,group,task,model.replace('V118_C13','V119_C14'))]
 for field in('N','frames','visible_support','occluded_support'):assert own[field]==current[field]
 if task=='D_vs_E':
  cm=[[own['TP'],own['FN_decided'],own['U_visible']+own['out_of_task_S_visible']+own['other_unresolved_visible']],[own['FP'],own['TN'],own['U_occluded']+own['out_of_task_S_occluded']+own['other_unresolved_occluded']]]
  c13comparison.append(dict(cohort=cohort,group=group,model=model,CM=cm,metrics=own))
assert c13metricrows==20
transitions=[]
for cohort in('paired_same_query_RGB','same_slot_pipeline'):
 for group in GROUPS:
  for state in(D,E,S):
   rr=[r for r in truth if r['group']==group and flag(r[cohort+'_D_vs_ES'])and r['human_state']==state]
   for suffix in('RAW','PLUS_OLD_GEOMETRY'):
    oldmodel='V118_C13_'+suffix;newmodel='V119_C14_'+suffix;cm=np.zeros((3,3),np.int64)
    for row in rr:
     aa=response(row,oldmodel,'D_vs_ES');bb=response(row,newmodel,'D_vs_ES');cm[min(aa,2),min(bb,2)]+=1
    assert int(cm.sum())==len(rr)
    transitions.append(dict(cohort=cohort,group=group,human_state=state,N=len(rr),source_model=oldmodel,new_model=newmodel,rows_old=['D','binary_hidden','U'],columns_new=['D','binary_hidden','U'],transition_matrix=cm.tolist(),self_is_binary_response_not_learned_S=state==S))
with(O/'C13_C14_ALL_FROZEN_GROUP_STATE_TRANSITIONS.json').open('x')as ff:json.dump(dict(completed=True,transitions=transitions,C13_eval=b(C13R/'evaluation/EVALUATION_SEAL.json'),C14_eval=b(R/'evaluation/EVALUATION_SEAL.json'),selection_or_training=False),ff,indent=2)
c12_comparison=[];c12metricrows=0
for row in c12tab['metrics']:
 if row['model']not in('V114_C12_RAW','V114_C12_PLUS_OLD_GEOMETRY'):continue
 cohort,group,task,model=(row[k]for k in('cohort','group','task','model'))
 flagfield=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
 rr=[r for r in truth if r['group']==group and flag(r[flagfield])and r['human_state']in TASKS[task]]
 own=calc(rr,model,task);eq(own,row);c12metrics[(cohort,group,task,model)]=own;c12metricrows+=1
 same=metrics[(cohort,group,task,model.replace('V114_C12','V119_C14'))]
 for field in('N','frames','visible_support','occluded_support'):assert own[field]==same[field]
 if task=='D_vs_E':
  cm=[[own['TP'],own['FN_decided'],own['U_visible']+own['out_of_task_S_visible']+own['other_unresolved_visible']],[own['FP'],own['TN'],own['U_occluded']+own['out_of_task_S_occluded']+own['other_unresolved_occluded']]]
  c12_comparison.append(dict(cohort=cohort,group=group,model=model,CM=cm,metrics=own))
assert c12metricrows==20
for row in c12tab['manual_near_D']:
 if row['model']not in('V114_C12_RAW','V114_C12_PLUS_OLD_GEOMETRY'):continue
 rr=[r for r in selected if r['group']==row['group']];cnt=Counter(response(r,row['model'],'D_vs_ES')for r in rr)
 own=dict(N=len(rr),pred_visible=cnt[0],pred_binary_hidden=cnt[1],unresolved=len(rr)-cnt[0]-cnt[1],semantic_D_recall=rat(cnt[0],len(rr)))
 eq(own,row);c12_near[(row['group'],row['model'])]=own
c12_raw_comparison=[]
for group in GROUPS:
 c12m=c12metrics[('paired_same_query_RGB',group,'D_vs_E','V114_C12_RAW')];c13m=metrics[('paired_same_query_RGB',group,'D_vs_E','V119_C14_RAW')]
 c12_raw_comparison.append(dict(group=group,C12_D_correct=c12m['TP'],C14_D_correct=c13m['TP'],D_support=c13m['visible_support'],C12_E_correct=c12m['TN'],C14_E_correct=c13m['TN'],E_support=c13m['occluded_support'],C12_near_D=c12_near[(group,'V114_C12_RAW')]['pred_visible'],C14_near_D=nears[(group,'V119_C14_RAW')]['pred_visible']))
comparison=[dict(cohort=c,group=g,model=m,CM=mats[(c,g,m)],metrics=metrics[(c,g,'D_vs_E',m)])for c in('paired_same_query_RGB','same_slot_pipeline')for g in GROUPS for m in MODELS]+c12_comparison+c13comparison
review=dict(completed=True,PASS=True,blockers=[],scope='Independent saved-only metadata/cache feature identity/raw-sign/GT/metric replay; no raw images/checkpoint parameters/new model/fit/current tuning/V112/PnP.',inference_seal=b(R/'INFERENCE_SEAL.json'),evaluation_seal=b(R/'evaluation/EVALUATION_SEAL.json'),source_freeze=inf['source_model_freeze'],source_selected_update=1200,original_all1784_q_L_ID_order_and45U_exact=True,extra_operator_U=0,feature_identity_certificate_SHA256=inf['feature_values_SHA256'],feature_cache_values_hash_independently_replayed=True,head_only_repeat_and_state_unchanged_certificate=True,head_output_not_independently_reforwarded=True,rawsign_D_lt0_else_hidden_and_U_first_exact=True,metrics_rows_replayed=len(metrics),historical_rows_exact=historical,numeric_count_rate_checks=checks,all_CM_precision_recall_U_errorbins_nearD_self_and_availability_tables_exact=True,primary_paired_D_vs_external_E=primary,descriptive_all_models_comparison=comparison,manual_near_D=[dict(group=g,**nears[(g,'V119_C14_RAW')])for g in GROUPS],same_slot_WOOD_S_raw=selfs[('same_slot_pipeline',GROUPS[0],'V119_C14_RAW')],same_slot_WOOD_S_geometry=selfs[('same_slot_pipeline',GROUPS[0],'V119_C14_PLUS_OLD_GEOMETRY')],human_state_counts=dict(Counter(r['human_state']for r in truth)),C12_evaluation_seal=b(C12R/'evaluation/EVALUATION_SEAL.json'),C12_metric_rows_independently_replayed=c12metricrows,C12_C13_C14_identical_manifest_bindings=True,C13_evaluation_seal=b(C13R/'evaluation/EVALUATION_SEAL.json'),C13_metric_rows_independently_replayed=c13metricrows,C13_C14_group_state_transitions=b(O/'C13_C14_ALL_FROZEN_GROUP_STATE_TRANSITIONS.json'),C12_C14_paired_raw_comparison=c12_raw_comparison,limits=['External recall is insufficient: WOOD20/29 andSQUARE55/61 false-visible externalE. Not solved or a universal replacement.','Current223 is repeatedly development-exposed. Comparisons are descriptive and do not select C11/C12/C13/C14 using private metrics.','Human S response after existing geometry gate is not learned self or external-cause identification.','General occluded-reference q error is not independently verified coordinate correctness. Source AUX/nativeUNKNOWN/humanfalse definitions remain distinct.','Feature identity/state/repeat certificates and saved arithmetic are verified; no model/checkpoint reforward was permitted or performed.'],new_model_forward_fit_raw_images_private_tuning_V112_PnP=0,own_code=b(Path(__file__)))
with(O/'REVIEW.json').open('x')as f:json.dump(review,f,indent=2)
with(O/'SEAL.json').open('x')as f:json.dump(dict(completed=True,PASS=True,review=b(O/'REVIEW.json'),files=[b(x)for x in sorted(O.iterdir())if x.is_file()and x.name!='SEAL.json']),f,indent=2)
print(json.dumps(dict(review=b(O/'REVIEW.json'),seal=b(O/'SEAL.json'),primary=primary,numeric_checks=checks,historical_rows=historical),indent=2))
