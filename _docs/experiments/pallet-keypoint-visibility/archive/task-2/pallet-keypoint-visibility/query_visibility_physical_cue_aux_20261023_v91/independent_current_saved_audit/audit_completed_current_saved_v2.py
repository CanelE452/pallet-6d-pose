"""Independent metadata/GT/savedscalar current audit. No evaluator/model imports."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib,gzip,math,datetime
import numpy as np
import current_metric_math as M
O=Path(__file__).resolve().parent;R=O.parent
ISEAL='300acd24206764ad5eb11ae331c1e5581ddc684c6fb0808a7fe93c526a8ab7ca'
ESEAL='dc6a05cf010a809feabf00a2d6b5a2ea03c9d42b511793cc689ea3341dc8c842'
MODELS=('v61','legacy','V76_SUPPORT','V76_SUPPORT+GEOMETRY','V91_RAW','V91_PLUS_OLD_GEOMETRY')

def b(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def read(p):
 p=Path(p)
 with(gzip.open if p.suffix=='.gz'else open)(p,'rt')as f:return json.load(f)
def jl(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def csvread(p):
 with Path(p).open(newline='')as f:return list(csv.DictReader(f))
def check(x):assert b(x['path'])['sha256']==x['sha256'],x['path']
def key(r):return r['id'],str(r['native_corner'])
def B(x,nullable=False):
 if x is True or x=='True'or x=='true':return True
 if x is False or x=='False'or x=='false':return False
 if nullable and(x is None or x in('','None','none','null')):return None
 raise ValueError('Explicitbool '+str(x))
def fnum(x):return None if x is None or x in('','None')else float(x)
def save(p,x):
 with Path(p).open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvsave(p,rows):
 with Path(p).open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compare(expected,actual):
 for k,v in expected.items():
  assert k in actual,('missingmetric',k)
  a=actual[k]
  if v is None:assert a is None or a=='', (k,v,a)
  elif isinstance(v,(int,float,np.number))and not isinstance(v,bool):assert a not in('',None)and abs(float(a)-float(v))<=1e-12,(k,v,a)
  else:assert a==v,(k,v,a)

def main():
 scope=read(O/'SCOPE_BEFORE_CURRENT_RESULTS.json');prep=read(O/'PREPARATION_SEAL.json');schema=read(O/'FINAL_CODE_SCHEMA_BINDINGS_V2_BEFORE_ACTUAL.json')
 for x in scope['bindings'].values():check(x)
 for k in('scope','math_code','generated_code','generated'):check(prep[k])
 for k in('adapter_code_V2','adapter_generated_V2','evaluator_code','evaluator_generated'):check(schema[k])
 assert read(prep['generated']['path'])['PASS']is True
 ip=R/'private_evaluation/NATIVE_CENTER_ONE_SHOT';ep=R/'contract/FROZEN_CURRENT_COMPARISON';assert b(ip/'INFERENCE_SEAL.json')['sha256']==ISEAL and b(ep/'EVALUATION_SEAL.json')['sha256']==ESEAL
 ins=read(ip/'INFERENCE_SEAL.json');es=read(ep/'EVALUATION_SEAL.json');assert ins['completed']and es['completed']and es['PASS']
 for k in('entry','protocol','artifact_bindings','source_postfit_audit','source_model_freeze','code','preparation'):check(ins[k])
 for x in ins['outputs']:check(x)
 for k in('entry','evaluator','inference','source_model_freeze','report'):check(es[k])
 for x in es['files']:check(x)
 e=read(es['entry']['path']);ie=read(ins['entry']['path']);assert e['approved']and e['current_inference_complete']and e['current_evaluation_authorized']and ie['approved']
 for k,x in e.items():
  if isinstance(x,dict)and'path'in x and not str(x['path']).endswith('.pt'):check(x)
 assert e['inference_seal']['sha256']==ISEAL and ie['code']['sha256']==schema['adapter_code_V2']['sha256']and e['evaluator_code']['sha256']==schema['evaluator_code']['sha256']
 freeze=read(ins['source_model_freeze']['path']);protocol=read(ins['protocol']['path']);assert protocol['source_gate_all9_pass']is True
 assert freeze['checkpoint']==ins['checkpoint']==e['checkpoint']and freeze['current_used_for_fit_selection']is False and freeze['external_threshold']==.5
 assert ins['source_model_freeze']==protocol['source_model_freeze']and read(ins['source_postfit_audit']['path'])['PASS']is True
 assert ins['model_unchanged']and ins['repeat_all_maps_and_valid_bitexact']and ins['repeat_max_logit_difference']==0 and ins['batch_size']==64
 assert ins['D_rule']=='zE<0'and ins['raw_E_threshold_logit']==0 and ins['probability_is_display_only']is True
 for k in('current_used_for_selection','current_GT_read','new_backbone','new_RGB_decodes','optimizer_updates','PnP'):assert ins[k]is False or ins[k]==0
 preparation=read(ins['preparation']['path']);assert preparation['completed']and preparation['frames']==223 and preparation['slots']==1784 and preparation['prior_U']==45
 check(preparation['audit']);art=read(ins['artifact_bindings']['path']);ab=art['bindings'];assert art['authorized_after_source_audit']and art['current_tensor_RGB_GT_score_deserialization']==0
 for k in('prior_input_rows','prepared_inputs','feature_metadata','prior_preparation_seal','prior_inference_seal','prior_input_gate'):check(ab[k])
 prior=jl(ab['prior_input_rows']['path']);frames=read(ab['prepared_inputs']['path']);fm=read(ab['feature_metadata']['path']);assert len(prior)==1784 and len(frames)==len(fm)==223
 newrows=csvread(e['scores']['path']);oldrows=csvread(e['baseline_v76_scores']['path']);truth=csvread(e['evaluation_manifest']['path']);provenance=csvread(e['manual_provenance']['path'])
 assert len(newrows)==len(oldrows)==len(truth)==1784
 pk=[key(x)for x in prior];nk=[key(x)for x in newrows];ok=[key(x)for x in oldrows];assert len(set(pk))==1784 and nk==pk==ok
 tmap={key(r):r for r in truth};assert set(tmap)==set(pk)
 with np.load(ip/'RAW_NATIVE_CENTRE_OUTPUTS.npz',allow_pickle=False)as z:raw={k:z[k]for k in z.files}
 assert set(raw)==set(schema['schema_from_code_only']['NPZ_raw_keys'])
 with np.load(preparation['audit']['path'],allow_pickle=False)as z:meta={k:z[k]for k in z.files}
 assert set(meta)==set(schema['schema_from_code_only']['metadata_audit_keys'])
 n=1784;z=raw['raw_physical_external_logit'];maps=raw['physical_external_logit_map'];mv=raw['map_valid'];valid=raw['valid'];pv=raw['p_visible_FP32'];pri=raw['prior_valid'];cv=raw['center_valid']
 assert maps.shape==mv.shape==(n,9,9)and maps.dtype==np.float32 and mv.dtype==bool and np.isfinite(maps).all()and np.array_equal(z,maps[:,4,4])and z.dtype==np.float32
 assert all(raw[k].shape==(n,)for k in('raw_physical_external_logit','center_valid','prior_valid','valid','p_visible_FP32','side_L','original_side_L','id','native_corner'))
 assert cv.dtype==pri.dtype==valid.dtype==bool and np.array_equal(cv,mv[:,4,4])and np.array_equal(valid,pri&cv)
 assert int(pri.sum())==1739 and int((~pri).sum())==45 and not(valid&~pri).any()and np.isfinite(pv).all()and((pv>=0)&(pv<=1)).all()
 for k in meta:assert np.array_equal(meta[k],raw[k],equal_nan=True)if np.issubdtype(meta[k].dtype,np.number)else np.array_equal(meta[k],raw[k])
 oq=[];ol=[]
 for fi,(f,m)in enumerate(zip(frames,fm)):
  assert f['id']==m['id'];q=np.asarray(f['initial_points'],np.float64)[:8];L=float(m['side']);assert q.shape==(8,2)and np.isfinite(L)and L>0
  for k in range(8):
   i=fi*8+k;r=prior[i];nr=newrows[i];tr=tmap[key(r)]
   assert r['id']==f['id']and int(r['native_corner'])==k and r['image_sha256']==f['image_sha256']
   assert nr['image_sha256']==r['image_sha256']==tr['image_sha256']and str(raw['id'][i])==r['id']and int(raw['native_corner'][i])==k
   for name,value in [('query_x',q[k,0]),('query_y',q[k,1])]:
    priorval=fnum(r[name]);newval=fnum(nr[name]);assert(priorval is None and newval is None)if not np.isfinite(value)else priorval==newval==float(value)
    if tr[name]and newval is not None:assert abs(float(tr[name])-newval)<=1e-5
   assert float(nr['roi_side'])==float(r['roi_side'])==L
   pixel=np.floor(np.where(np.isfinite(q[k]),q[k],-1e8)+.5).astype(int)
   assert int(nr['pixel_x'])==r['pixel_x']==pixel[0]and int(nr['pixel_y'])==r['pixel_y']==pixel[1]
   assert B(nr['saved_roi_pixel_valid'],True)==r['saved_roi_pixel_valid']and B(nr['geometry_self_hidden'],True)==r['geometry_self_hidden']
   assert B(nr['prior_valid'])==r['valid']==bool(pri[i])and B(nr['center_valid'])==bool(cv[i])and B(nr['valid'])==bool(valid[i])
   assert float(nr['raw_physical_external_logit'])==float(z[i]);expected=M.authoritative(float(z[i]),bool(valid[i]));dec='D'if expected=='VISIBLE'else'E'if expected=='ABSENT_NONVISIBLE'else'UNKNOWN'
   assert nr['authoritative_decision']==dec and B(nr['decision_visible'],True)==(bool(z[i]<0)if valid[i]else None)
   assert B(nr['geometry_self_hidden_if_valid'],True)==(r['geometry_self_hidden']if valid[i]else None)
   if valid[i]:assert float(nr['p_visible'])==float(pv[i])and float(nr['logit'])==-float(z[i])
   else:assert nr['p_visible']==nr['logit']==''
   oq.append(q[k]);ol.append(L)
 oq=np.array(oq,np.float64);ol=np.array(ol,np.float64)
 assert np.array_equal(raw['original_query_xy'],oq,equal_nan=True)and np.array_equal(raw['original_side_L'],ol)and np.array_equal(raw['query_xy'],oq.astype(np.float32),equal_nan=True)and np.array_equal(raw['side_L'],ol.astype(np.float32))
 assert int(valid.sum())==ins['valid']and int((~valid).sum())==ins['U']and int((pri&~valid).sum())==ins['additional_U']
 assert ins['rounded_probability_equals_half_N']==int(np.count_nonzero(valid&(pv==.5)))and ins['rounded_probability_ge_half_vs_authoritative_D_mismatch_N']==int(np.count_nonzero(valid&((pv>=.5)!=(z<0))))
 new={key(r):r for r in newrows};old={key(r):r for r in oldrows}
 def call(r,method,task):
  k=key(r)
  if method in('v61','legacy'):return M.baseline_call(r,method,task)
  a=new[k]if method.startswith('V91')else old[k];v=B(a['valid']);geo=B(a['geometry_self_hidden_if_valid'],True)
  assert geo is None if not v else geo==B(a['geometry_self_hidden'],True)
  if method.startswith('V91'):return M.authoritative(float(a['raw_physical_external_logit']),v,geo if method=='V91_PLUS_OLD_GEOMETRY'else None)
  return M.baseline_call(r,'V76',task,p=fnum(a['p_visible']),valid=v,geometry=geo if method.endswith('+GEOMETRY')else None)
 result=read(es['report']['path']);tables=result['tables'];computed=[];metric_lookup={}
 for cohort in('paired_same_query_RGB','same_slot_pipeline'):
  for group in('WOOD_PRIMARY105','SQUARE118'):
   for task in M.TASKS:
    rr=M.selected(truth,cohort,group,task)
    if not any(r['human_state']!=M.D for r in rr):continue
    for model in MODELS:
     metric=M.calculate(rr,[call(r,model,task)for r in rr]);row=dict(cohort=cohort,group=group,task=task,model=model,**metric);computed.append(row);metric_lookup[cohort,group,task,model]=row
 published=tables['CLASSIFICATION_METRICS'];assert len(published)==len(computed)and len({(r['cohort'],r['group'],r['task'],r['model'])for r in published})==len(published)
 for row in published:compare(metric_lookup[row['cohort'],row['group'],row['task'],row['model']],row)
 # Historical v61/legacy and V76 counts/ratios independently replayed fromsavedrows.
 historical_checks=0
 for name in('historical_baseline_metrics','original_baseline_metrics'):
  for row in csvread(e[name]['path']):
   k=tuple(row[x]for x in('cohort','group','task','model'))
   if k in metric_lookup:compare({kk:vv for kk,vv in metric_lookup[k].items()if kk in row},row);historical_checks+=1
 # All component matrices, per-recording metrics, generaldescriptive strata.
 checks=Counter()
 for row in tables['CONFUSION_BY_HUMAN_STATE']:
  rr=[r for r in M.selected(truth,row['cohort'],row['group'],row['task'])if r['human_state']==row['human_state']];counts=Counter(call(r,row['model'],row['task'])for r in rr)
  compare(dict(N=len(rr),pred_visible=counts['VISIBLE'],pred_absent_nonvisible=counts['ABSENT_NONVISIBLE'],pred_U=counts['ABSTAIN'],pred_out_of_task_geometry_S=counts['OUT_OF_TASK_GEOMETRY_S'],pred_other_unresolved=counts['OTHER_UNRESOLVED']),row);checks['humanstate_rows']+=1
 for row in tables['BINARY_2X3_CONFUSION']:
  m=metric_lookup[row['cohort'],row['group'],row['task'],row['model']];compare({'N':m['N'],'GT_D_pred_VISIBLE':m['TP'],'GT_D_pred_ABSENT':m['FN_decided'],'GT_D_pred_UNRESOLVED':m['U_visible']+m['out_of_task_S_visible']+m['other_unresolved_visible'],'GT_NEG_pred_VISIBLE':m['FP'],'GT_NEG_pred_ABSENT':m['TN'],'GT_NEG_pred_UNRESOLVED':m['U_occluded']+m['out_of_task_S_occluded']+m['other_unresolved_occluded']},row);checks['2x3_rows']+=1
 for row in tables['PER_RECORDING_METRICS']:
  rr=[r for r in M.selected(truth,row['cohort'],row['group'],row['task'])if r['recording']==row['recording']];compare(M.calculate(rr,[call(r,row['model'],row['task'])for r in rr]),row);checks['recording_rows']+=1
 for row in tables['ERROR_STRATA']:
  base=[r for r in truth if r['group']==row['group']and M.truth_bool(r[row['cohort']+'_D_vs_ES'])and r['human_state']==row['human_state']]
  label=row['error_bin'];rr=[]
  for r in base:
   yes=M.truth_bool(r['human_reference_inside_nominal_crop'])if label.startswith('reference_')else False
   if label=='reference_inside_crop':keep=yes
   elif label=='reference_outside_crop':keep=not yes
   else:keep=M.error_bin(r['initial_reference_error_px'])==label
   if keep:rr.append(r)
  c=Counter(call(r,row['model'],'D_vs_ES')for r in rr);correct='VISIBLE'if row['human_state']==M.D else'ABSENT_NONVISIBLE'
  compare(dict(N=len(rr),pred_visible=c['VISIBLE'],pred_absent_nonvisible=c['ABSENT_NONVISIBLE'],pred_U=c['ABSTAIN'],pred_other_unresolved=c['OTHER_UNRESOLVED'],class_recall_all_GT=c[correct]/len(rr)if rr else None),row);checks['error_strata_rows']+=1
 for row in tables['GEOMETRY_TRADEOFF']:
  rr=[r for r in M.selected(truth,row['cohort'],row['group'],row['task'])if r['human_state']==row['human_state']];rawcalls=Counter(call(r,row['raw_model'],row['task'])for r in rr);hybrid=Counter(call(r,row['hybrid_model'],row['task'])for r in rr);src=new if row['raw_model'].startswith('V91')else old;flags=[B(src[key(r)]['geometry_self_hidden_if_valid'],True)for r in rr];correct='VISIBLE'if row['human_state']==M.D else'ABSENT_NONVISIBLE'
  compare(dict(N=len(rr),geometry_true=sum(f is True for f in flags),geometry_unknown=sum(f is None for f in flags),raw_pred_visible=rawcalls['VISIBLE'],hybrid_pred_visible=hybrid['VISIBLE'],suppressed_raw_visible=rawcalls['VISIBLE']-hybrid['VISIBLE'],raw_class_recall=rawcalls[correct]/len(rr)if rr else None,hybrid_class_recall=hybrid[correct]/len(rr)if rr else None),row);checks['geometry_tradeoff_rows']+=1
 for table in('PAIRED_CHANGE','V91_V76_PAIRED_CHANGE'):
  for row in tables[table]:
   rr=M.selected(truth,row['cohort'],row['group'],row['task']);compare({'N':len(rr),**M.paired_change(rr,[call(r,row['baseline'],row['task'])for r in rr],[call(r,row['new_model'],row['task'])for r in rr])},row);checks[table+'_rows']+=1
 manual=M.manual_near(provenance);expectedN={'WOOD_PRIMARY105':153,'SQUARE118':223};manualcomputed=[]
 for group in expectedN:
  rr=[r for r in manual if r['group']==group];assert len(rr)==expectedN[group]
  for r in rr:
   tr=tmap[key(r)];assert r['image_sha256']==tr['image_sha256']and r['query_x']==tr['query_x']and r['query_y']==tr['query_y']
  for model in MODELS:
   c=Counter(call(r,model,'D_vs_ES')for r in rr);manualcomputed.append(dict(group=group,model=model,N=len(rr),pred_visible=c['VISIBLE'],pred_absent_nonvisible=c['ABSENT_NONVISIBLE'],pred_legacy_U=c['ABSTAIN'],pred_other_unresolved=c['OTHER_UNRESOLVED'],pred_out_of_task_S=c['OUT_OF_TASK_GEOMETRY_S'],semantic_D_recall=c['VISIBLE']/len(rr),exact_pixel_GT='UNAVAILABLE'))
 manual_lookup={(r['group'],r['model']):r for r in manualcomputed}
 for row in result['manual_click_near_D_LE3']:compare(manual_lookup[row['group'],row['model']],row)
 assert len(result['manual_click_near_D_LE3'])==len(manualcomputed)==12
 # Published all1784 rowdecisions are checked against originaltruth/nativeordering.
 for i,row in enumerate(csvread(ep/'CURRENT_ALL1784_SAVED_QUERY_REPLAY.csv')):
  tr=tmap[pk[i]];assert key(row)==pk[i];compare({'V91_raw_decision':call(tr,'V91_RAW','D_vs_ES'),'V91_geometry_decision':call(tr,'V91_PLUS_OLD_GEOMETRY','D_vs_ES'),'v61_saved_present':tr['present_at_fixed_0_5'],'legacy_stored_state':tr['legacy_stored_state']},row)
 av={'rows':1784,'valid':int(valid.sum()),'invalid':int((~valid).sum()),'historical_V76_valid':sum(B(r['valid'])for r in oldrows),'historical_V76_invalid':sum(not B(r['valid'])for r in oldrows),'original_prior_U':45,'additional_V91_factual_U':int((pri&~valid).sum())};compare(av,result['availability'])
 # Reconcile everycanonical evaluationCSV with its own JSONtable as savedserialization.
 names={'CLASSIFICATION_METRICS':'CURRENT_CLASSIFICATION_METRICS.csv','CONFUSION_BY_HUMAN_STATE':'CURRENT_CONFUSION_BY_HUMAN_STATE.csv','ERROR_STRATA':'CURRENT_ERROR_STRATA.csv','GEOMETRY_TRADEOFF':'CURRENT_GEOMETRY_TRADEOFF.csv','PER_RECORDING_METRICS':'CURRENT_PER_RECORDING_METRICS.csv','PAIRED_CHANGE':'CURRENT_PAIRED_CHANGE.csv','BINARY_2X3_CONFUSION':'CURRENT_BINARY_2X3_CONFUSION.csv','V91_V76_PAIRED_CHANGE':'CURRENT_V91_V76_PAIRED_CHANGE.csv'}
 for name,fn in names.items():
  csvrows=csvread(ep/fn);assert len(csvrows)==len(tables[name])
  for expected,actual in zip(tables[name],csvrows):compare(expected,actual)
 oldprep=read(ab['prior_preparation_seal']['path']);oldins=read(ab['prior_inference_seal']['path']);assert oldprep['completed']and oldins['completed']and oldprep['valid']==oldins['valid']==1739
 repair=R/'contract/ROOT_RELATIVE_OUTPUT_INVOCATION_REPAIR.json';assert repair.is_file();repairbinding=b(repair)
 out=O/'completed_audit_v2';out.mkdir(exist_ok=False);csvsave(out/'INDEPENDENT_CLASSIFICATION_METRICS.csv',computed);csvsave(out/'INDEPENDENT_MANUAL_NEAR_D.csv',manualcomputed)
 primary=[m for m in computed if m['cohort']=='paired_same_query_RGB'and m['task']=='D_vs_E'and m['model']in('V91_RAW','V91_PLUS_OLD_GEOMETRY')]
 coord=Counter((r.get('group','UNKNOWN'),r.get('annotation_coordinate_source','UNKNOWN_NOT_IN_FILE'))for r in provenance)
 report={'schema':'V91_INDEPENDENT_CURRENT_SAVED_AUDIT','completed':True,'PASS':True,'blockers':[],'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repair_addendum':b(O/'JSON_CSV_TYPES_V2_REPAIR_ADDENDUM.json'),'scope':b(O/'SCOPE_BEFORE_CURRENT_RESULTS.json'),'prepared_math':b(O/'current_metric_math.py'),'generated':b(O/'GENERATED_INDEPENDENT_CURRENT_MATH_TESTS.json'),'code':b(__file__),'code_schema_prelock':b(O/'FINAL_CODE_SCHEMA_BINDINGS_V2_BEFORE_ACTUAL.json'),'inference_seal':b(ip/'INFERENCE_SEAL.json'),'evaluation_seal':b(ep/'EVALUATION_SEAL.json'),'source_freeze':ins['source_model_freeze'],'checkpoint_descriptor_chain_metadata_exact':ins['checkpoint'],'no_checkpoint_file_or_parameter_read_in_current_audit':True,'exact_checks':{'all1784_ID_nativeorder_imageSHA_originalqL_pixel_geometry_parity':True,'all1784_rawmapcentre_F32_valid_bool_CSV_NPZ_exact':True,'prior45U_retained_and_additionalcentreU_counted':True,'authoritative_rawzE_and_invalidfirst_geometry':True,'allcommoncohort_D_E_D_S_D_ES_CM_P_R_U':True,'allbaseline_historical_counts_and_metrics':True,'allmanual_click_D153_223_provenance_exact':True,'allgeneral_errorbins_and_recording_geometry_pairedchanges':True,'allpublished_json_csvserialization_hashes_exact':True},'availability':av,'primary_paired_D_vs_E':primary,'manual_near_D_replay':manualcomputed,'metric_rows_replayed':len(computed),'historical_metric_rows_replayed':historical_checks,'other_replay_counts':dict(checks),'coordinate_provenance_file_counts':[{'group':k[0],'coordinate_source':k[1],'rows':v}for k,v in sorted(coord.items())],'coordinate_scope':'manualclickqueryvalidDIRECTD<=3 isseparate153/223 subset. Generalerrorstrata include sourcecoordinatesofmixedprovenance; providedprovenancefilepopulationcounts aredescriptive, notnew pixelGT. No mislabeledmanual projections.','inference_certificates':{'model_unchanged':ins['model_unchanged'],'state_SHA':ins['model_state_SHA256'],'repeat_all_maps_and_valid_bitexact':ins['repeat_all_maps_and_valid_bitexact'],'repeat_max_logit_difference':ins['repeat_max_logit_difference'],'no_current_selection':not ins['current_used_for_selection'],'new_backbone':ins['new_backbone'],'new_RGB_decodes':ins['new_RGB_decodes']},'execution_repair_preserved':repairbinding,'limits_KO':['이번감사는완료된저장출력/주석/메타데이터만검산했다. 모델state/repeat는동결코드와저장assertion certificate검증이며독립모델재추론이아니다.','원래registeredcohort와같은query/RGB를유지하고방법별U를분모에서버리지않았다. Geometry는oldselfproxy의별도route이며binaryabsence를인과E/S정답으로부르지않는다.','current는반복사용된평가군이며새독립holdout이아니다. exactquerypixelGT/native8mapping/targetassociation/fullgoal해결을인증하지않는다.','private성능으로checkpoint/문턱/모델선택·GT좌표이동·PnP를수행하지않았다.'], 'current_RGB_P3_tensor_weightparameter_model_forward_fit_PnP_reads_or_calls':0,'new_model_or_threshold_selection':False,'fully_solved':False}
 save(out/'REVIEW.json',report);save(O/'REVIEW.json',report);save(out/'SEAL.json',{'completed':True,'PASS':True,'review':b(O/'REVIEW.json'),'code':b(__file__),'scope':b(O/'SCOPE_BEFORE_CURRENT_RESULTS.json'),'input_inference_seal':b(ip/'INFERENCE_SEAL.json'),'evaluation_seal':b(ep/'EVALUATION_SEAL.json'),'replayed_classification':b(out/'INDEPENDENT_CLASSIFICATION_METRICS.csv'),'replayed_manual':b(out/'INDEPENDENT_MANUAL_NEAR_D.csv'),'new_forward_fit_currentRGB_weights_PnP':0});print(json.dumps({'review':b(O/'REVIEW.json'),'seal':b(out/'SEAL.json'),'primary':primary,'manual_new':[m for m in manualcomputed if m['model']=='V91_RAW']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
