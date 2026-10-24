"""Independent saved current audit; no evaluator/model import or neural call."""
from pathlib import Path
import collections
import csv
import datetime
import hashlib
import json
import math
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
V=HERE.parent
OUT=V/'current_evaluation/C1_NATURAL_KNOWNSET_ONE_SHOT'
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
GROUPS=['WOOD_PRIMARY105','SQUARE118']
MODELS=['v61','legacy','V91_RAW','V91_PLUS_OLD_GEOMETRY','V94_C1_RAW','V94_C1_PLUS_OLD_GEOMETRY']
TASKS={'D_vs_E':[D,E],'D_vs_S':[D,S],'D_vs_ES':[D,E,S]}

def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def check(d):assert bind(d['path'])['sha256']==d['sha256'],d['path'];return Path(d['path'])
def read(p):return json.loads(Path(p).read_text())
def csvrows(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def jsonrows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,d):
 assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def key(r):return r['id'],str(r['native_corner'])
def flag(v):
 if v is True or str(v).lower()=='true':return True
 if v is False or str(v).lower()=='false':return False
 if v is None or str(v).lower()in ['', 'none','null']:return None
 raise ValueError('Nonboolean flag '+str(v))
def numeq(a,b):
 missing=lambda x:x is None or str(x).lower()in ['', 'none','null','nan']
 return missing(a)and missing(b)or not missing(a)and not missing(b)and float(a)==float(b)
def ratio(a,b):return a/b if b else None
def compare(got,pub,fields):
 for f in fields:
  a=got[f];b=pub[f]
  if a is None:assert b is None or b=='',(f,a,b)
  elif isinstance(a,(int,float))and not isinstance(a,bool):assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(f,a,b)
  else:assert a==b,(f,a,b)

def main():
 policy=read(HERE/'POLICY_BEFORE_CURRENT_OUTPUT_READS.json')
 seal=read(OUT/'EVALUATION_SEAL.json');assert seal['completed']is True and seal['PASS']is True
 for d in seal['files']:check(d)
 entry=read(check(seal['entry']));assert entry['approved']is True and entry['stage']=='EVALUATE_SAVED'and not entry['current_used_for_selection']and not entry['training_allowed']and not entry['PnP_allowed']
 packet=read(check(entry['artifact_bindings']));assert packet['completed']and packet['authorized_after_source_audit'];b=packet['bindings']
 for k in ['evaluation_manifest','manual_provenance','prior_input_rows','baseline_v91_scores','baseline_v91_inference_seal','historical_v91_metrics','original_baseline_metrics']:check(b[k])
 for k in ['inference_seal','scores','raw_outputs','preparation','protocol','code','common_code','generated','independent_review']:check(entry[k])
 inference=read(entry['inference_seal']['path']);assert inference['completed']and inference['frames']==223 and inference['slots']==1784
 assert inference['source_model_freeze']['sha256']==seal['source_freeze']['sha256']==b['source_model_freeze']['sha256']
 freeze=read(check(b['source_model_freeze']));assert freeze['selected_update']==100 and freeze['checkpoint']['sha256']==inference['checkpoint']['sha256']
 assert inference['D_rule']=='z<0'and inference['raw_E_threshold_logit']==0 and inference['model_unchanged']and inference['repeat_scalar_available_attention_bitexact']
 assert inference['optimizer_updates']==inference['PnP']==inference['new_backbone']==inference['new_initializer']==0
 assert inference['current_used_for_selection']is False and inference['current_GT_read']is False
 truth=csvrows(b['evaluation_manifest']['path']);manual=csvrows(b['manual_provenance']['path']);prior=jsonrows(b['prior_input_rows']['path'])
 new=csvrows(entry['scores']['path']);old=csvrows(b['baseline_v91_scores']['path'])
 assert len(truth)==len(prior)==len(new)==len(old)==1784
 maps=[{key(r):r for r in rr}for rr in [truth,prior,new,old]];tm,pm,nm,om=maps
 assert all(len(m)==1784 for m in maps)and set(tm)==set(pm)==set(nm)==set(om)
 assert list(pm)==list(nm)==list(om)
 rawfields=['query_logit','query_available','prior_valid','valid','p_visible_FP32','query_xy','side_L','original_query_xy','original_side_L','id','native_corner']
 with np.load(entry['raw_outputs']['path'],allow_pickle=False)as d:raw={k:d[k].copy()for k in rawfields}
 assert raw['query_logit'].shape==(1784,)and raw['query_logit'].dtype==np.float32 and np.isfinite(raw['query_logit']).all()
 assert raw['prior_valid'].dtype==raw['valid'].dtype==raw['query_available'].dtype==np.bool_
 assert np.array_equal(raw['valid'],raw['prior_valid']&raw['query_available'])
 assert raw['original_query_xy'].shape==(1784,2)and raw['original_side_L'].shape==(1784,)
 assert np.array_equal(raw['query_xy'],raw['original_query_xy'].astype(np.float32),equal_nan=True)
 assert np.array_equal(raw['side_L'],raw['original_side_L'].astype(np.float32),equal_nan=True)
 originalU=extraU=0;statescope=collections.Counter()
 for i,r in enumerate(new):
  k=key(r);p=pm[k];o=om[k];t=tm[k];valid=flag(r['valid']);z=float(r['raw_query_aux_hidden_logit'])
  assert k==(str(raw['id'][i]),str(int(raw['native_corner'][i])))and int(r['native_corner'])==i%8
  assert valid==bool(raw['valid'][i])and flag(p['valid'])==bool(raw['prior_valid'][i])and flag(r['operator_query_available'])==bool(raw['query_available'][i])
  assert flag(r['prior_valid'])==flag(p['valid'])and flag(r['additional_operator_U'])==(flag(p['valid'])and not flag(r['operator_query_available']))
  assert z==float(raw['query_logit'][i])and flag(r['saved_roi_pixel_valid'])==p['saved_roi_pixel_valid']==flag(o['saved_roi_pixel_valid'])
  assert r['image_sha256']==p['image_sha256']==o['image_sha256']==t['image_sha256']
  for f in ['query_x','query_y','roi_side','pixel_x','pixel_y']:assert numeq(r[f],p[f])and numeq(r[f],o[f]),(i,f)
  for j,f in enumerate(['query_x','query_y']):
   assert numeq(r[f],raw['original_query_xy'][i,j])
   if r[f]and t[f]:assert abs(float(r[f])-float(t[f]))<=1e-5
  assert numeq(r['roi_side'],raw['original_side_L'][i])
  geo=flag(r['geometry_self_hidden']);assert geo==p['geometry_self_hidden']==flag(o['geometry_self_hidden'])
  assert flag(r['geometry_self_hidden_if_valid'])==(geo if valid else None)
  assert flag(r['decision_visible'])==((z<0)if valid else None)
  assert r['authoritative_decision']==(('D'if z<0 else'E')if valid else'UNKNOWN')
  assert flag(r['old_geometry_route_visible'])==((False if geo else z<0)if valid else None)
  assert flag(r['present_at_frozen_threshold'])==((z<0)if valid else None)
  assert all(flag(r[f])is False for f in ['self_occlusion_learned','external_vs_self_identified','semantic_U_detector','native_association_known'])
  assert float(r['raw_p_visible_FP32'])==float(raw['p_visible_FP32'][i])
  if valid:assert float(r['p_visible'])==float(raw['p_visible_FP32'][i])and float(r['logit'])==-z
  else:assert not r['p_visible']and not r['logit']
  assert valid or not flag(r['decision_visible']);assert not valid or flag(p['valid'])
  originalU+=not flag(p['valid']);extraU+=bool(flag(p['valid'])and not valid);statescope[(t['group'],t['human_state'])]+=1
 assert originalU==45 and int((~raw['valid']).sum())==originalU+extraU
 assert inference['prior_U']==45 and inference['U']==originalU+extraU and inference['additional_U']==extraU
 assert len({r['id']for r in new})==223

 def decide(r,model,task):
  k=key(r)
  if model=='v61':return 'VISIBLE'if flag(r['present_at_fixed_0_5'])else'HIDDEN'
  if model=='legacy':
   s=r['legacy_stored_state']
   if s in ['B','I']:return 'VISIBLE'
   if s=='O':return 'HIDDEN'
   if s=='S':return 'OUT_S'if task=='D_vs_E'else'HIDDEN'
   return 'LEGACY_U'if s=='U'else'OTHER_U'
  rr=om[k]if model.startswith('V91')else nm[k]
  if not flag(rr['valid']):return 'OTHER_U'
  if 'GEOMETRY'in model and flag(rr['geometry_self_hidden_if_valid'])is True:return 'HIDDEN'
  field='raw_physical_external_logit'if model.startswith('V91')else'raw_query_aux_hidden_logit'
  return 'VISIBLE'if float(rr[field])<0 else'HIDDEN'

 def count(rr,model,task):
  counts=collections.Counter((r['human_state']==D,decide(r,model,task))for r in rr)
  p=sum(r['human_state']==D for r in rr);n=len(rr)-p
  tp,fn,fp,tn=[counts[pos,state]for pos,state in [(True,'VISIBLE'),(True,'HIDDEN'),(False,'VISIBLE'),(False,'HIDDEN')]]
  up,un,sp,sn,op,on=[counts[pos,state]for pos,state in [(True,'LEGACY_U'),(False,'LEGACY_U'),(True,'OUT_S'),(False,'OUT_S'),(True,'OTHER_U'),(False,'OTHER_U')]]
  assert tp+fn+up+sp+op==p and fp+tn+un+sn+on==n
  return dict(N=len(rr),frames=len({r['id']for r in rr}),visible_support=p,occluded_support=n,TP=tp,FN_decided=fn,FP=fp,TN=tn,U_visible=up,U_occluded=un,out_of_task_S_visible=sp,out_of_task_S_occluded=sn,other_unresolved_visible=op,other_unresolved_occluded=on,visible_precision=ratio(tp,tp+fp),visible_recall_all_GT=ratio(tp,p),occluded_precision=ratio(tn,tn+fn),occluded_recall_all_GT=ratio(tn,n),abstention_rate=ratio(up+un,len(rr)),out_of_task_S_rate=ratio(sp+sn,len(rr)),unresolved_rate=ratio(up+un+sp+sn+op+on,len(rr)),decision_coverage=ratio(tp+fn+fp+tn,len(rr)),correct_fraction_all_GT=ratio(tp+tn,len(rr)),selective_accuracy=ratio(tp+tn,tp+fn+fp+tn))
 allmetrics=[];confusion=[];selfrows=[];excluded=[]
 for cohort in ['paired_same_query_RGB','same_slot_pipeline']:
  for group in GROUPS:
   for task,allowed in TASKS.items():
    field=cohort+'_'+('D_vs_ES'if task=='D_vs_S'else task)
    rr=[r for r in truth if r['group']==group and flag(r[field])and r['human_state']in allowed]
    excluded.append(dict(cohort=cohort,group=group,task=task,eligible=len(rr),all_group=sum(r['group']==group for r in truth),eligible_states=dict(collections.Counter(r['human_state']for r in rr))))
    if not any(r['human_state']!=D for r in rr):continue
    for model in MODELS:
     allmetrics.append(dict(cohort=cohort,group=group,task=task,model=model,**count(rr,model,task)))
     for state in allowed:
      ss=[r for r in rr if r['human_state']==state];cc=collections.Counter(decide(r,model,task)for r in ss)
      confusion.append(dict(cohort=cohort,group=group,task=task,model=model,human_state=state,N=len(ss),pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['HIDDEN'],legacy_U=cc['LEGACY_U'],out_of_task_geometry_S=cc['OUT_S'],input_or_operator_U=cc['OTHER_U']))
   ss=[r for r in truth if r['group']==group and flag(r[cohort+'_D_vs_ES'])and r['human_state']==S]
   for model in MODELS:
    cc=collections.Counter(decide(r,model,'D_vs_S')for r in ss)
    selfrows.append(dict(cohort=cohort,group=group,model=model,human_S_support=len(ss),binary_hidden_response=cc['HIDDEN'],visible_response=cc['VISIBLE'],unresolved=len(ss)-cc['HIDDEN']-cc['VISIBLE'],human_S_binary_hidden_response_rate=ratio(cc['HIDDEN'],len(ss)),learned_self_prediction=False,external_vs_self_identified=False))
 published=read(OUT/'CURRENT_PRIMARY_RESULTS.json');assert published['completed']
 pmets={(r['cohort'],r['group'],r['task'],r['model']):r for r in published['tables']['metrics']}
 pconf={(r['cohort'],r['group'],r['task'],r['model'],r['human_state']):r for r in published['tables']['confusion']}
 assert len(allmetrics)==len(pmets)and len(confusion)==len(pconf)
 for r in allmetrics:compare(r,pmets[r['cohort'],r['group'],r['task'],r['model']],r.keys())
 for r in confusion:compare(r,pconf[r['cohort'],r['group'],r['task'],r['model'],r['human_state']],r.keys())
 pself={(r['cohort'],r['group'],r['model']):r for r in published['tables']['human_self_binary_response']}
 for r in selfrows:compare(r,pself[r['cohort'],r['group'],r['model']],r.keys())
 # Literal saved CSV readouts must agree with JSON too, not only the generator.
 metriccsv={(r['cohort'],r['group'],r['task'],r['model']):r for r in csvrows(OUT/'METRICS.csv')}
 for r in allmetrics:compare(r,metriccsv[r['cohort'],r['group'],r['task'],r['model']],r.keys())
 confcsv={(r['cohort'],r['group'],r['task'],r['model'],r['human_state']):r for r in csvrows(OUT/'CONFUSION.csv')}
 for r in confusion:compare(r,confcsv[r['cohort'],r['group'],r['task'],r['model'],r['human_state']],r.keys())
 selfcsv={(r['cohort'],r['group'],r['model']):r for r in csvrows(OUT/'HUMAN_SELF_BINARY_RESPONSE.csv')}
 for r in selfrows:
  z=selfcsv[r['cohort'],r['group'],r['model']]
  for f in ['learned_self_prediction','external_vs_self_identified']:z[f]=flag(z[f])
  compare(r,z,r.keys())
 near=[];selected=[r for r in manual if r['annotation_coordinate_source']=='manual_click'and r['human_state']==D and flag(r['query_valid'])and float(r['initial_reference_error_px'])<=3]
 assert len({key(r)for r in selected})==len(selected)
 for r in selected:
  t=tm[key(r)];assert r['image_sha256']==t['image_sha256']and numeq(r['query_x'],t['query_x'])and numeq(r['query_y'],t['query_y'])
 for group,N in [('WOOD_PRIMARY105',153),('SQUARE118',223)]:
  rr=[r for r in selected if r['group']==group];assert len(rr)==N
  for model in MODELS:
   cc=collections.Counter(decide(r,model,'D_vs_ES')for r in rr)
   near.append(dict(group=group,model=model,scope='MANUAL_CLICK_DIRECT_VALID_QUERY_ERROR_LE3',N=N,pred_visible=cc['VISIBLE'],pred_binary_hidden=cc['HIDDEN'],unresolved=N-cc['VISIBLE']-cc['HIDDEN'],semantic_D_recall=ratio(cc['VISIBLE'],N)))
 pnear={(r['group'],r['model']):r for r in published['tables']['manual_near_D']}
 nearcsv={(r['group'],r['model']):r for r in csvrows(OUT/'MANUAL_NEAR_D.csv')}
 for r in near:compare(r,pnear[r['group'],r['model']],r.keys());compare(r,nearcsv[r['group'],r['model']],r.keys())
 # Existing baseline counts/ratios are independently reconstructed from manifest.
 historic=[]
 for name,models in [('original_baseline_metrics',{'v61','legacy'}),('historical_v91_metrics',{'V91_RAW','V91_PLUS_OLD_GEOMETRY'})]:
  hm={(r['cohort'],r['group'],r['task'],r['model']):r for r in csvrows(b[name]['path'])if r['model']in models}
  matched=0
  for r in allmetrics:
   if r['model']in models:compare(r,hm[r['cohort'],r['group'],r['task'],r['model']],r.keys());matched+=1
  assert matched==len(hm);historic.append(dict(binding=b[name],independent_rows_exact=matched))
 availability=dict(rows=1784,prior_input_U=45,new_total_U=int((~raw['valid']).sum()),extra_operator_U=extraU,V91_total_U=sum(not flag(r['valid'])for r in old),confidence_abstention_trained=False)
 assert availability==published['tables']['availability']
 groupav=[]
 for g in GROUPS:
  keys=[key(r)for r in truth if r['group']==g];pu=sum(not flag(pm[k]['valid'])for k in keys);nu=sum(not flag(nm[k]['valid'])for k in keys)
  groupav.append(dict(group=g,all_native_slots=len(keys),prior_input_U=pu,extra_operator_U=nu-pu,new_total_U=nu,V91_total_U=sum(not flag(om[k]['valid'])for k in keys)))
 assert groupav==published['tables']['availability_by_group']
 groupcsv={r['group']:r for r in csvrows(OUT/'AVAILABILITY_BY_GROUP.csv')}
 for r in groupav:compare(r,groupcsv[r['group']],r.keys())
 general=[]
 for g in GROUPS:
  rr=[r for r in truth if r['group']==g and r['human_state']==D and flag(r['query_valid'])and float(r['initial_reference_error_px'])<=3]
  general.append(dict(group=g,general_valid_D_error_le3=len(rr),manual_click_near_D=sum(r['group']==g for r in selected),scope='General GT-coordinate cohort may include projected/locked references; not interchangeable with manual_click near denominator.'))
 write(HERE/'INDEPENDENT_CURRENT_TABLES.json',dict(metrics=allmetrics,confusion=confusion,human_S_binary_response=selfrows,manual_near_D=near,cohort_eligibility=excluded,availability=availability,availability_by_group=groupav,general_coordinate_counts_separate=general))
 summary=[r for r in allmetrics if r['model']=='V94_C1_RAW'and r['cohort']=='paired_same_query_RGB'and r['task']=='D_vs_E']
 report=dict(completed=True,PASS=True,blockers=[],policy=bind(HERE/'POLICY_BEFORE_CURRENT_OUTPUT_READS.json'),evaluation_seal=bind(OUT/'EVALUATION_SEAL.json'),evaluation_entry=seal['entry'],inference_seal=entry['inference_seal'],source_freeze=b['source_model_freeze'],source_selected_update=100,raw_output=entry['raw_outputs'],new_scores=entry['scores'],prior_rows=b['prior_input_rows'],human_evaluation_manifest=b['evaluation_manifest'],manual_provenance=b['manual_provenance'],V91_scores=b['baseline_v91_scores'],code=bind(Path(__file__)),
  all1784_identity=dict(original_key_native_order_exact=True,frames=223,all_image_SHA_exact=True,all_original_q_L_pixel_savedvalid_geometry_exact=True,original45U_preserved=True,extra_U=extraU,rawNPZ_CSV_rawsign_valid_and_probability_export_exact=True,feature_q_L_FP32_metadata_equals_original_cast=True,no_raw_pixel_or_input_feature_read=True),
  tables=bind(HERE/'INDEPENDENT_CURRENT_TABLES.json'),metric_rows_exact=len(allmetrics),confusion_rows_exact=len(confusion),human_S_response_rows_exact=len(selfrows),manual_near_rows_exact=len(near),baseline_historical_replay=historic,primary_paired_D_E=summary,manual_near_primary=[r for r in near if r['model']=='V94_C1_RAW'],availability=availability,general_coordinate_counts_separate=general,
  limits=['HumanS subset measures visible/binary-hidden/U response only. The source head did not learn or identify semantic self versus external cause. Old saved geometry is a separate valid-only override and cannot resolve factual U.','Current human semantic labels are evaluation truth; auxiliary source known-set visibility is a different training proxy. Binary hidden is not an independent causal external classifier or native expected-part correspondence certificate.','All1784 human-state/cohort exclusions and original45U remain. Manual_click<=3 uses exactly153WOOD/223SQUARE; general-coordinate error cohorts are separately reported and may include projected/locked references.','This checks stored numeric outputs/metadata only. Model-state repeat and no-update claims are bound producer certificates; no new model/checkpoint/forward/RGB/feature read, fit/threshold/checkpoint selection, pose/PnP, annotation correction or case substitution. Historically development-exposed223 is not a new independent holdout.'],
  new_model_forward_fit_threshold_PnP=0,created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(HERE/'REVIEW.json',report);write(HERE/'SEAL.json',dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),policy=bind(HERE/'POLICY_BEFORE_CURRENT_OUTPUT_READS.json'),code=bind(Path(__file__)),tables=bind(HERE/'INDEPENDENT_CURRENT_TABLES.json'),evaluation_seal=bind(OUT/'EVALUATION_SEAL.json')))
 print(json.dumps(dict(review=bind(HERE/'REVIEW.json'),seal=bind(HERE/'SEAL.json'),primary=summary,near=report['manual_near_primary'],availability=availability)))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  f=HERE/'FAILED_AUDIT.json'
  if not f.exists():write(f,dict(completed=False,PASS=False,error=repr(e),traceback=traceback.format_exc(),original_outputs_untouched=True,no_model_or_fit_restart=True))
  raise
