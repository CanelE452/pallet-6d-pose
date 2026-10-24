"""Independent NumPy/CSV-only current audit. Never imports a model/evaluator.

No original RGB decoding, inference, optimizer, PnP, or current model selection.
Human semantic labels evaluate transfer; exact real mask-support GT is absent.
"""
from pathlib import Path
from collections import Counter
import csv,gzip,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent;V76=HERE.parent;P=V76.parent
CURRENT=V76/'private_evaluation/MIXED_SUPPORT_V1';OLD=P/'query_visibility_cue_occlusion_20261023_v72/private_evaluation/CUE_ATTENTION_V3'
INFERENCE_SHA='ddece1626e16631c960d724a2723841e0b3588c095098e2f0546d7e9ac91d2ac'
D,E,S='DIRECT_VISIBLE','EXTERNAL_OCCLUDED','SELF_OCCLUDED'
GROUPS=['WOOD_PRIMARY105','SQUARE118'];MODELS=['v61','legacy','V76_SUPPORT','V76_SUPPORT+GEOMETRY']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def rows(p):
 with open(p,newline='')as f:return list(csv.DictReader(f))
def yes(r,k):return str(r[k]).lower()=='true'
def key(r):return (r['id'],int(r['native_corner']))
def table_key(r):return tuple(r[k]for k in ('cohort','group','task','model'))
def ratio(a,b):return a/b if b else None
def write(p,d):
 assert not p.exists(),p;p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rr):
 assert not p.exists(),p
 with p.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def float_equal(a,b):
 a=float(a);b=float(b);return a==b or(math.isnan(a)and math.isnan(b))

# Metadata authorization and hash provenance precedes opening current arrays,
# score rows or human GT. Checkpoint bytes are hashed, never deserialized.
gate=load(V76/'contract/POST_INFERENCE_EVALUATION_GATE.json')
for k in ('approved','source_frozen','current_outputs_complete','evaluation_authorized_after_source_freeze'):assert gate[k]is True
for k in ('source_model_freeze','protocol','checkpoint','inference_seal','scores','evaluator_code','base_evaluator'):check(gate[k])
assert gate['inference_seal']['sha256']==INFERENCE_SHA
frozen=load(gate['source_model_freeze']['path']);assert frozen['completed']and frozen['optimizer_updates']==2000 and frozen['best_update']==200
assert frozen['visible_probability_threshold']==.5 and not frozen['current_used_for_fit_selection']and frozen['PnP']==0
inf=load(CURRENT/'INFERENCE_SEAL.json')
for k,v in dict(completed=True,frames=223,slots=1784,valid=1739,threshold=.5,threshold_comparison='>=',repeat_logit_max_difference=0.,model_unchanged=True,current_used_for_selection=False,current_GT_read=False,PnP=0,new_backbone=0,optimizer_updates=0).items():assert inf[k]==v,k
for b in [inf['preparation'],*inf['outputs']]:check(b)
prep=load(inf['preparation']['path'])
for k,v in dict(completed=True,all223_same_q_L_P3_RGB96=True,current_GT_or_annotations_or_scores_read=False,new_backbone=0,fit=0,PnP=0).items():assert prep[k]==v
check(prep['input']);check(prep['prior_preparation']);check(prep['code']);check(prep['gate'])
oldprep=load(prep['prior_preparation']['path'])
for b in oldprep['outputs']:check(b)
oldgate=load(OLD.parent/'CURRENT_INPUT_ADAPTER_GATE.json')
for b in oldgate['inputs'].values():check(b)

with np.load(prep['input']['path'],allow_pickle=False)as z:new_input={k:z[k]for k in z.files}
with np.load(OLD/'CURRENT_MODEL_INPUT.npz',allow_pickle=False)as z:old_input={k:z[k]for k in z.files}
assert set(new_input)=={'p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42'}
input_parity={}
for a,b in [('p3_roi','p3_roi'),('local_rgb96','rgb96'),('local_valid96','crop_valid96')]:
 assert new_input[a].dtype==old_input[b].dtype and np.array_equal(new_input[a],old_input[b])
 input_parity[a]=dict(exact=True,shape=list(new_input[a].shape),dtype=str(new_input[a].dtype),different_values=0)
with np.load(CURRENT/'RAW_NETWORK_OUTPUTS.npz',allow_pickle=False)as z:raw={k:z[k]for k in z.files}
assert raw['support_logit'].shape==(223,8)and np.isfinite(raw['support_logit']).all()
assert np.array_equal(raw['original_query8'].reshape(-1,2).astype(np.float32),old_input['query_xy'],equal_nan=True)
assert np.array_equal(np.repeat(raw['original_L'].astype(np.float32),8),old_input['side_L'])
prepared_path=Path(oldgate['inputs']['prepared_inputs']['path'])
with gzip.open(prepared_path,'rt')as h:frames=json.load(h)
metadata=load(oldgate['inputs']['feature_metadata']['path']);assert len(frames)==len(metadata)==223
for i,(fr,me)in enumerate(zip(frames,metadata)):
 assert fr['id']==me['id']and np.array_equal(raw['original_query8'][i],np.asarray(fr['initial_points'],float)[:8],equal_nan=True)and raw['original_L'][i]==me['side']

truth_path=P/'evaluation_preparation_20261022/PREPARED_QUERY_MANIFEST.csv'
near_path=P/'near_error_diagnosis_20261022/real/NEAR_DIRECT_ROWS_WITH_PROVENANCE.csv'
truth=rows(truth_path);scores=rows(CURRENT/'REAL_QUERY_SCORES.csv');prior=rows(OLD/'REAL_QUERY_SCORES.csv')
priorrows=[json.loads(x)for x in (OLD/'ROWS_INPUT_ONLY.jsonl').read_text().splitlines()]
by={key(r):r for r in truth};scoreby={key(r):r for r in scores};oldby={key(r):r for r in prior}
assert len(truth)==len(scores)==len(prior)==len(priorrows)==len(by)==len(scoreby)==1784 and set(by)==set(scoreby)==set(oldby)
assert len({r['id']for r in scores})==223
identity=[];invalid=0;invalid_old_geometry_true=0;geometry_flag_changes=0;prob_diff=0.;valid_count=0
for i,r in enumerate(scores):
 k=key(r);t=by[k];old=oldby[k];p0=priorrows[i]
 assert k==key(priorrows[i])==key(prior[i])and r['image_sha256']==t['image_sha256']==old['image_sha256']
 assert all(float_equal(r[c],t[c])and float_equal(r[c],old[c])for c in ('query_x','query_y'))
 assert float(r['roi_side'])==float(t['v61_roi_side_raw_px'])==raw['original_L'][i//8]
 valid=yes(r,'valid');assert valid==(bool(p0['valid'])and bool(raw['input_available'].reshape(-1)[i]))==bool(raw['valid'].reshape(-1)[i])
 assert r['geometry_self_hidden']==old['geometry_self_hidden']and r['geometry_self_source']==old['geometry_self_source']
 assert r['geometry_self_hidden']==str(p0['geometry_self_hidden'])or r['geometry_self_hidden']==''and p0['geometry_self_hidden']is None
 assert not yes(r,'semantic_U_detector')and not yes(r,'self_occlusion_learned')and not yes(r,'geometry_overrides_learned_probability')
 logit=float(raw['support_logit'].reshape(-1)[i]);assert float(r['raw_support_logit'])==logit
 probability=float(np.exp(-np.logaddexp(0.,-logit)));prob_diff=max(prob_diff,abs(probability-float(r['raw_p_local_support_visible'])))
 if valid:
  valid_count+=1;assert r['status']=='OK'and float(r['logit'])==logit and float(r['p_visible'])==float(r['raw_p_local_support_visible'])
  assert yes(r,'present_at_frozen_threshold')==(logit>=0)==(float(r['p_visible'])>=.5)
  assert r['predicted_D_or_E']==('D'if logit>=0 else'E')and r['geometry_self_hidden_if_valid']==r['geometry_self_hidden']
 else:
  invalid+=1;assert r['status']=='UNKNOWN'and r['predicted_D_or_E']=='UNKNOWN'
  assert all(r[c]==''for c in ('p_visible','logit','present_at_frozen_threshold'))and r['geometry_self_hidden_if_valid']in ('','None')
  invalid_old_geometry_true+=r['geometry_self_hidden']=='True'
 if yes(t,'query_valid'):
  assert math.isfinite(float(t['presence_probability']))and yes(t,'present_at_fixed_0_5')==(float(t['presence_probability'])>=.5)
 identity.append(dict(id=r['id'],native_corner=int(r['native_corner']),human_state=t['human_state'],group=t['group'],valid=valid,
  raw_call='UNRESOLVED'if not valid else'VISIBLE'if float(r['p_visible'])>=.5 else'ABSENT_NONVISIBLE',
  original_geometry=r['geometry_self_hidden'],guarded_geometry=r['geometry_self_hidden_if_valid'],GT_and_score_identity_exact=True))
assert valid_count==1739 and invalid==45 and prob_diff<1e-15

def prediction(t,model,task):
 if model=='v61':return 'VISIBLE'if yes(t,'present_at_fixed_0_5')else'ABSENT_NONVISIBLE'
 if model=='legacy':
  state=t['legacy_stored_state']
  if state in ('B','I'):return 'VISIBLE'
  if state=='O':return 'ABSENT_NONVISIBLE'
  if state=='S':return 'OUT_OF_TASK_GEOMETRY_S'if task=='D_vs_E'else'ABSENT_NONVISIBLE'
  return 'ABSTAIN'if state=='U'else'OTHER_UNRESOLVED'
 s=scoreby[key(t)]
 if not yes(s,'valid'):return 'OTHER_UNRESOLVED'
 if model.endswith('+GEOMETRY')and s['geometry_self_hidden_if_valid']=='True':return 'ABSENT_NONVISIBLE'
 return 'VISIBLE'if float(s['p_visible'])>=.5 else'ABSENT_NONVISIBLE'

def calculate(rr,model,task):
 c=Counter((r['human_state']==D,prediction(r,model,task))for r in rr);pos=sum(r['human_state']==D for r in rr);neg=len(rr)-pos
 tp,fn,fp,tn=(c[True,'VISIBLE'],c[True,'ABSENT_NONVISIBLE'],c[False,'VISIBLE'],c[False,'ABSENT_NONVISIBLE'])
 u0,u1=c[True,'ABSTAIN'],c[False,'ABSTAIN'];s0,s1=c[True,'OUT_OF_TASK_GEOMETRY_S'],c[False,'OUT_OF_TASK_GEOMETRY_S'];o0,o1=c[True,'OTHER_UNRESOLVED'],c[False,'OTHER_UNRESOLVED']
 assert tp+fn+u0+s0+o0==pos and fp+tn+u1+s1+o1==neg
 return dict(N=len(rr),frames=len({r['id']for r in rr}),visible_support=pos,occluded_support=neg,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=u0,U_occluded=u1,
  out_of_task_S_visible=s0,out_of_task_S_occluded=s1,other_unresolved_visible=o0,other_unresolved_occluded=o1,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,pos),
  occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,neg),abstention_rate=ratio(u0+u1,len(rr)),out_of_task_S_rate=ratio(s0+s1,len(rr)),unresolved_rate=ratio(u0+u1+s0+s1+o0+o1,len(rr)),
  decision_coverage=ratio(tp+fn+fp+tn,len(rr)),correct_fraction_all_GT=ratio(tp+tn,len(rr)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))

reported=load(V76/'contract/V76_CURRENT_PRIMARY_RESULTS.json');reportby={table_key(r):r for r in reported['tables']['CLASSIFICATION_METRICS']}
baseline=rows(P/'saved_output_evaluation_20261022/CLASSIFICATION_METRICS.csv');baseby={table_key(r):r for r in baseline}
recomputed=[];comparisons=0
for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
 for group in GROUPS:
  for task,states in [('D_vs_E',(D,E)),('D_vs_S',(D,S)),('D_vs_ES',(D,E,S))]:
   flag=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
   rr=[r for r in truth if r['group']==group and yes(r,flag)and r['human_state']in states]
   if not any(r['human_state']!=D for r in rr):continue
   for model in MODELS:
    result=dict(cohort=cohort,group=group,task=task,model=model,**calculate(rr,model,task));before=reportby[table_key(result)]
    for k,v in result.items():
     if k in ('cohort','group','task','model'):continue
     assert v is None and before[k]==''or v is not None and abs(float(before[k])-v)<1e-12,(table_key(result),k,v,before[k]);comparisons+=1
     if model in ('v61','legacy'):
      old=baseby[table_key(result)];assert v is None and old[k]==''or v is not None and abs(float(old[k])-v)<1e-12
    recomputed.append(result)

near=rows(near_path);manual=[]
for r in near:
 if r['annotation_coordinate_source']=='manual_click'and r['human_state']==D and float(r['initial_reference_error_px'])<=3 and yes(r,'query_valid'):
  t=by[key(r)];assert r['image_sha256']==t['image_sha256']and all(float_equal(r[c],t[c])for c in ('query_x','query_y','human_reference_x','human_reference_y'))
  err=math.hypot(float(r['query_x'])-float(r['human_reference_x']),float(r['query_y'])-float(r['human_reference_y']))
  assert abs(err-float(r['initial_reference_error_px']))<1e-10
  manual.append(r)
assert len({key(r)for r in manual})==len(manual)
manual_rows=[]
for group,n in zip(GROUPS,(153,223)):
 rr=[r for r in manual if r['group']==group];assert len(rr)==n
 for model in MODELS:
  c=Counter(prediction(r,model,'D_vs_ES')for r in rr);result=dict(group=group,model=model,N=n,pred_visible=c['VISIBLE'],pred_absent_nonvisible=c['ABSENT_NONVISIBLE'],pred_U=c['ABSTAIN'],pred_other_unresolved=c['OTHER_UNRESOLVED'],semantic_D_recall=c['VISIBLE']/n)
  old=next(x for x in reported['manual_near_D_LE3']if x['group']==group and x['model']==model)
  for k,v in result.items():assert old[k]==v,(k,v,old[k])
  manual_rows.append(result)
selfrows=[]
for state,n in ((D,626),(S,136)):
 rr=[r for r in truth if r['group']==GROUPS[0]and yes(r,'same_slot_pipeline_D_vs_ES')and r['human_state']==state];assert len(rr)==n
 for model in ('V76_SUPPORT','V76_SUPPORT+GEOMETRY'):
  c=Counter(prediction(r,model,'D_vs_S')for r in rr)
  selfrows.append(dict(human_state=state,model=model,N=n,pred_visible=c['VISIBLE'],pred_absent_nonvisible=c['ABSENT_NONVISIBLE'],unresolved=c['OTHER_UNRESOLVED'],geometry_true=sum(scoreby[key(r)]['geometry_self_hidden_if_valid']=='True'for r in rr)))
squareS=sum(r['group']==GROUPS[1]and r['human_state']==S for r in truth);assert squareS==0
csvwrite(HERE/'ALL1784_JOIN_AND_VALIDITY.csv',identity);csvwrite(HERE/'METRICS_RECOMPUTED.csv',recomputed);csvwrite(HERE/'MANUAL_NEAR_D_RECOMPUTED.csv',manual_rows)
primary=[r for r in recomputed if r['cohort']=='paired_same_query_RGB'and r['task']=='D_vs_E'and r['model']in ('v61','legacy','V76_SUPPORT')]
out=HERE/'INDEPENDENT_CURRENT_SAVED_AUDIT.json'
write(out,dict(completed=True,PASS=True,code=bind(Path(__file__)),inference_seal=bind(CURRENT/'INFERENCE_SEAL.json'),source_freeze=gate['source_model_freeze'],
 original_223_q8_L_exact=True,original_q8_L_float32_parity_with_V72=True,unchanged_saved_input_arrays=input_parity,raw_RGB_files_decoded=0,
 recorded_repeat_logit_max_difference=inf['repeat_logit_max_difference'],recorded_model_unchanged=inf['model_unchanged'],repeat_scope='Sealedassertions/code/checkpointhasheschecked;modelnotloadedorreforwardedbythisaudit.',
 rows=1784,frames=223,valid=1739,unresolved_invalid=45,invalid_original_geometry_true=invalid_old_geometry_true,all_invalid_main_scores_blank=True,invalid_guarded_geometry_blank=True,
 geometry_original_flags_unchanged_all1784=True,no_confidence_U=True,sigmoid_max_abs_difference=prob_diff,GT_score_ID_RGB_query_joins_exact=True,
 metric_rows_recomputed=len(recomputed),reported_metric_fields_compared=comparisons,v61_legacy_original_metrics_reproduced=True,paired_D_E=primary,
 manual_click_LE3_D=manual_rows,self_wood_D626_S136=selfrows,square_S_gold_count=0,
 conclusions=['HighsemanticDrecalldoesnotestablishfullsolution: pairedsemanticEfalsevisible10/29wood,41/61square. NegativeisABSENT_NONVISIBLE,notcausalErecognition.',
  'Selfrouteissavedconditionalgeometryproxy,separatefromrawlocal-supportnetwork;notlearnedS. Invalid45neverforcednegative.',
  'Manual153/223subsetusesmanual_clickprovenanceandfixed<=3pxonly;errorisdescriptive,notverifiedcoordinatecause. Realexactpixel/footprintsupportGTunavailable.',
  'No model/threshold/geometry/radius selectionfromcurrent;currenthistoricallyexposeddevelopmentset,nofreshindependenttestclaim.'],
 bindings=[bind(p)for p in (V76/'contract/POST_INFERENCE_EVALUATION_GATE.json',V76/'contract/V76_CURRENT_PRIMARY_RESULTS.json',CURRENT/'INPUT_PREPARATION_SEAL.json',CURRENT/'CURRENT_SIX_INPUTS.npz',CURRENT/'RAW_NETWORK_OUTPUTS.npz',CURRENT/'REAL_QUERY_SCORES.csv',OLD/'CURRENT_MODEL_INPUT.npz',OLD/'REAL_QUERY_SCORES.csv',truth_path,near_path)],
 model_or_backbone_forward=0,optimizer_updates=0,PnP=0,current_used_for_selection=False))
write(HERE/'AUDIT_SEAL.json',dict(completed=True,PASS=True,audit=bind(out),code=bind(Path(__file__)),outputs=[bind(HERE/p)for p in ('ALL1784_JOIN_AND_VALIDITY.csv','METRICS_RECOMPUTED.csv','MANUAL_NEAR_D_RECOMPUTED.csv')],frozen_inputs_models_scores_evalfiles_modified=False,model_forward_fit_PnP=0))
print(json.dumps(dict(audit=bind(out),seal=bind(HERE/'AUDIT_SEAL.json'))))
