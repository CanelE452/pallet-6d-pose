"""Authored full-shape packet/identity and source-barrier fixtures. No source bodies."""
from pathlib import Path
import copy,importlib.util,json,hashlib
import numpy as np
H=Path(__file__).parent
sp=importlib.util.spec_from_file_location('a',H/'audit_saved_fourteen_model_results.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
checks=[]
def reject(fn):
 try:fn()
 except(AssertionError,KeyError):return
 raise AssertionError('Expected rejection')
def basep():
 return dict(predictor_code={'path':'TOY_CODE'},predictor_protocol={'path':'TOY_PROTO'},root_scope={'path':'TOY_SCOPE'},decoder_spec={},input_complete={},original_raw_complete={},source_fit_complete={'path':'TOY_FIT'},source_fit_review={'path':'TOY_REVIEW'},source_fit_protocol={'path':'TOY_FIT_PROTO'},source_both_final_freeze={'path':'TOY_BARRIER'},spatial_head={},source_context_adapter={},input_domains={})
p=basep();seal=dict(completed=True,PASS=True,code=p['predictor_code'],protocol=p['predictor_protocol'],scope=p['root_scope'],decoder_spec={},input_complete={},original_raw_complete={},source_fit_complete=p['source_fit_complete'],source_fit_integrity_review=p['source_fit_review'],spatial_head={},source_context_adapter={},all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,selected_model=None,adopted=False,arms={})
for i,arm in enumerate(a.NEW_ARMS):seal['arms'][arm]=dict(updates=4000,use_sam=False,spatial_q8=True,weak_bridge=bool(i),weak_loss_weight=.25 if i else 0,checkpoint={'path':'TOY_CP_'+str(i)},source_final_state_sha='TOY_STATE_'+str(i),outputs={})
fit=dict(completed=True,PASS=True,all2_frozen_before_any_SOURCE_outputs=True,parameters_each=118359,steps_each=4000,total_updates=8000,protected_real_COMMON_VAL_access=0,protocol=p['source_fit_protocol'],spatial_head={},context_adapter={},all2_freeze=p['source_both_final_freeze'],arms={arm:dict(r,final_state_sha=r['source_final_state_sha'])for arm,r in seal['arms'].items()})
barrier=dict(completed=True,source_predictions_emitted=False,checkpoints={arm:r['checkpoint']for arm,r in fit['arms'].items()});review=dict(completed=True,PASS=True,fit_complete=p['source_fit_complete'])
orig_checked=a.checked
lookup={'TOY_FIT':fit,'TOY_REVIEW':review,'TOY_BARRIER':barrier}
a.checked=lambda b:json.dumps(lookup[b['path']]).encode()
a.source_gate(seal,p)
barrier['source_predictions_emitted']=True;reject(lambda:a.source_gate(seal,p));barrier['source_predictions_emitted']=False
fit['arms'][a.NEW_ARMS[1]]['weak_loss_weight']=0;reject(lambda:a.source_gate(seal,p));fit['arms'][a.NEW_ARMS[1]]['weak_loss_weight']=.25
review['fit_complete']={};reject(lambda:a.source_gate(seal,p));review['fit_complete']=p['source_fit_complete'];a.checked=orig_checked
checks.append('Actual source_gate accepts both4000 and rejects early emission/wrong weak role/exact-fit review drift')
packets={};metas={};ledgers={}
for domain,n,start,U in [('CURRENT223_1784',1784,0,45),('PUBLIC_HELD45_360',360,584,108)]:
 av=np.ones(n,bool);av[:U]=False;z=dict(row_id=np.array([f'TOY_{domain}_{i}'for i in range(n)]),original_global_index=np.arange(start,start+n,dtype=np.int64),native_slot=(np.arange(n)%8).astype(np.int8),query_xy=np.zeros((n,2),np.float64),L=np.full(n,96.,np.float64),original_query_available=av,raw_support_logits=np.zeros((n,2),np.float32),raw_owner_logits=np.zeros((n,3),np.float32),raw_direct_end_external_logit=np.full(n,-1.,np.float32),support_prediction=np.zeros((n,2),np.int8),owner_prediction=np.zeros(n,np.int8),direct_end_prediction_D0_E1_Uminus1=np.zeros(n,np.int8),regional_support_proxy_D0_E1_Uminus1=np.full(n,-1,np.int8),support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.array(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']))
 for k in('raw_support_logits','raw_owner_logits','raw_direct_end_external_logit'):z[k][~av]=np.nan
 for k in('support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1'):z[k][~av]=-1
 metas[domain]=[dict(row_id=str(z['row_id'][i]),original_global_index=int(z['original_global_index'][i]),native_slot=int(z['native_slot'][i]),query_xy=z['query_xy'][i].tolist(),L=float(z['L'][i]),original_query_available=bool(av[i]))for i in range(n)]
 rr=[]
 for i,m in enumerate(metas[domain]):
  r=dict(global_index=start+i,native_corner=int(z['native_slot'][i]),query_x=0.,query_y=0.,roi_side=96.,query_available=bool(av[i]))if domain=='CURRENT223_1784'else dict(original_global_index=start+i,query_id=m['row_id'])
  for tag in('C62','C63'):
   r.update({tag+'_support_prediction':z['support_prediction'][i].tolist(),tag+'_owner_prediction':int(z['owner_prediction'][i]),tag+'_primary_prediction':int(z['direct_end_prediction_D0_E1_Uminus1'][i]),tag+'_regional_support_proxy_D0_E1_Uminus1':-1,tag+'_raw_support_logits':[0.,0.]if av[i]else[None,None],tag+'_raw_owner_logits':[0.,0.,0.]if av[i]else[None]*3,tag+'_raw_direct_end_external_logit':-1. if av[i]else None})
  rr.append(r)
 ledgers[domain]=rr;p['input_domains'][domain]=dict(metadata={'path':'TOY_META_'+domain},original_array_fingerprints={k:a.fingerprint(z[k])for k in a.META})
 for arm in a.NEW_ARMS:
  b={'path':'TOY_PACKET_'+domain+'_'+arm};seal['arms'][arm]['outputs'][domain]=b;packets[b['path']]=z
orig_packet=a.packet;a.packet=lambda b,allow,keys=None:packets[b['path']];a.checked=lambda b:'\n'.join(json.dumps(r)for r in metas[b['path'].replace('TOY_META_','')]).encode()
r=a.prediction_audit(seal,p,ledgers['CURRENT223_1784'],ledgers['PUBLIC_HELD45_360']);a.need(r['new_packet_rows']==4288,'All authored packet rows')
ledgers['CURRENT223_1784'][0]['native_corner']=7;reject(lambda:a.prediction_audit(seal,p,ledgers['CURRENT223_1784'],ledgers['PUBLIC_HELD45_360']));ledgers['CURRENT223_1784'][0]['native_corner']=0
metas['PUBLIC_HELD45_360'][0]['row_id']='WRONG_JOIN';reject(lambda:a.prediction_audit(seal,p,ledgers['CURRENT223_1784'],ledgers['PUBLIC_HELD45_360']));a.packet=orig_packet;a.checked=orig_checked
checks.append('Actual prediction_audit full4288 authored rows passes; changed ledger native and original row identity reject')
reject(lambda:a.packet({'path':'NOT_ALLOWED_BEFORE_HEADER'},{}));checks.append('Packet whitelist rejects before any header/file access')
# Exact self cohort source operator: ES true and DE false on S is valid.
truth={('TOY_SELF',0):dict(query_valid='True',present_at_fixed_0_5='False',legacy_stored_state='S')};rows=[dict(id='TOY_SELF',native_corner=0,human_state=a.S,C62_original_response='ABSENT_NONVISIBLE')]
r=a.calculated(rows,'C62','D_vs_S',truth);a.need(r['occluded_support']==1 and r['TN']==1,'S binary response kept separate');checks.append('SELF response operator does not relabel S as external E')
result=dict(completed=True,PASS=True,code=a.bind(H/'audit_saved_fourteen_model_results.py'),checks=checks,actual_private_GT_score_prediction_input_array_or_weight_reads=0,operational_model_calls=0)
with(H/'SUPPLEMENTAL_IDENTITY_FIXTURES.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
