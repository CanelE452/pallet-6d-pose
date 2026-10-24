"""Saved-only independent V184 raw10/SAM96 and decoder audit. No torch/model imports."""
from pathlib import Path
import argparse,hashlib,json,math,os,stat,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
RAW=V/'original_real_raw8_inputs_v1/COMPLETE.json';RAW_SHA='2d90a7863fe80cfce95e3d54ee1ce529b2b800bcc85373cd7b1f0551c5e68a37'
INPUT_REVIEW=V/'independent_original_input_review_a/REVIEW.json';INPUT_REVIEW_SHA='394564a9b7c2e4bd1a212e3cd4b3127ca5bad1d71ac28dc200f49d9b94b4908b'
CP=V/'frozen_prediction_preparation_c';CODE=CP/'compose_SAM96_and_infer_both_frozen_heads.py';CODE_SHA='ead5654961db0bca0dedb457a9212adc22a348a8b98be9b0abd83c7a15fbeaa0'
PROTO=CP/'PROTOCOL_BEFORE_REAL_RAW_SAM_OR_HEAD_WEIGHT_BODIES.json';PROTO_SHA='26f0aa63062b976d50c01a5797777960fc57ff33a8c204e92d22f5bbb819bafe'
META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
RAW8=set(META)|{'rgb96','valid96'};RAW10=RAW8|{'sam_mask96','sam_available96'}
SAM12={'image_logits','lowres_logits','predicted_quality','mask_logits_gt0','box_xyxy','original_query_xy','original_L','native_slot','original_global_index','original_occurrence_id','original_query_available','prediction_available'}
OUT13=set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}
ARMS={'C52_RAW96_JITTER_PAIRED_SOURCE':False,'C53_RAW96_SAM_JITTER_PAIRED_SOURCE':True}
DOMAINS={'CURRENT223_1784':(1784,45),'PUBLIC_HELD45_360':(360,108)}
BINDINGS={}
def need(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def safe(p):
 s=os.path.abspath(str(p));need('_v112' not in s.lower(),'Forbidden path before access');pending=s.split('/')[1:];cur='/';hops=0
 while pending:
  n=os.path.normpath(os.path.join(cur,pending.pop(0)));need('_v112' not in n.lower(),'Forbidden component');a=os.lstat(n)
  if stat.S_ISLNK(a.st_mode):
   t=os.readlink(n);need('_v112' not in t.lower(),'Forbidden symlink');t=os.path.normpath(t if os.path.isabs(t) else os.path.join(os.path.dirname(n),t));need('_v112' not in t.lower(),'Forbidden expansion');hops+=1;need(hops<40,'Symlink loop');pending=t.split('/')[1:]+pending;cur='/'
  else:cur=n
 return Path(cur)
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def check(b):need(sha(b['path'])==b['sha256'],'Binding '+b['path']);BINDINGS[b['path']]=b['sha256'];return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def lines(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def numeric(a,b):return np.array_equal(np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64),equal_nan=True)
def npz(b,keys):
 with np.load(check(b),allow_pickle=False) as z:
  need(set(z.files)==keys,'Exact saved NPZ key set');return {k:z[k] for k in keys}
def fingerprints(x):return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=ash(v)) for k,v in x.items()}
def sample(logits,q,L,original_available,runtime_available,hw):
 """Independent scalar axes + four 2D tap arrays, fixed FP64 sum order."""
 mask=np.zeros((96,96),bool);domain=np.zeros_like(mask)
 if not original_available or not runtime_available:return mask,domain
 q=np.asarray(q,np.float64)
 if not np.isfinite(q).all() or (q==[-1.,-1.]).all() or not math.isfinite(L) or L<=0:return mask,domain
 h,w=hw;need(logits.dtype==np.float32 and logits.shape==(h,w) and np.isfinite(logits).all(),'Full SAM logits dtype/HW')
 x=np.asarray([float(q[0])+((j+0.5)/96.0-0.5)*float(L) for j in range(96)],np.float64)
 y=np.asarray([float(q[1])+((j+0.5)/96.0-0.5)*float(L) for j in range(96)],np.float64)
 ix=np.asarray([math.floor(t) for t in x],np.int64);iy=np.asarray([math.floor(t) for t in y],np.int64);dx=x-ix;dy=y-iy
 total=np.zeros((96,96),np.float64)
 for ordinal in range(4):
  ox=ordinal%2;oy=ordinal//2;xx=ix+ox;yy=iy+oy
  wx=dx if ox else 1.0-dx;wy=dy if oy else 1.0-dy
  inside=np.logical_and.outer((yy>=0)&(yy<h),(xx>=0)&(xx<w))
  pixels=logits[np.clip(yy,0,h-1)[:,None],np.clip(xx,0,w-1)[None,:]].astype(np.float64)
  contribution=pixels*np.multiply.outer(wy,wx);total[inside]+=contribution[inside]
 domain=np.logical_and.outer((y>=-.5)&(y<h-.5),(x>=-.5)&(x<w-.5))
 return (total>0)&domain,domain
def scalar_decode(s,o,av):
 n=len(av);sp=np.full((n,2),-1,np.int8);op=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
 for i in range(n):
  if not av[i]:need(all(math.isnan(float(v)) for v in s[i]) and all(math.isnan(float(v)) for v in o[i]),'OriginalU all raw NaN');continue
  need(all(math.isfinite(float(v)) for v in s[i]) and all(math.isfinite(float(v)) for v in o[i]),'Available raw finite')
  sp[i,0]=int(float(s[i,0])>0.0);sp[i,1]=int(float(s[i,1])>0.0)
  winner=0
  for k in range(1,3):
   if float(o[i,k])>float(o[i,winner]):winner=k
  op[i]=winner;proxy[i]=1 if sp[i,1]==1 else 0 if sp[i,0]==1 else -1
 return sp,op,proxy
def invented():
 tests={};logits=(np.arange(12,dtype=np.float32).reshape(3,4)-5);q=[1.25,.75];L=2.
 a,d=sample(logits,q,L,True,True,[3,4]);oracle=np.zeros((96,96),bool);valid=np.zeros_like(oracle)
 for row in range(96):
  for col in range(96):
   x=q[0]+((col+.5)/96-.5)*L;y=q[1]+((row+.5)/96-.5)*L;inside=-.5<=x<3.5 and -.5<=y<2.5;v=0.
   x0=math.floor(x);y0=math.floor(y)
   for yy in [y0,y0+1]:
    for xx in [x0,x0+1]:
     if 0<=xx<4 and 0<=yy<3:v+=float(logits[yy,xx])*(1-abs(x-xx))*(1-abs(y-yy))
   valid[row,col]=inside;oracle[row,col]=inside and v>0
 tests['actual_sampler_matches_independent_scalar_fourtap']=exact(a,oracle) and exact(d,valid)
 tests['zero_strict_sign_and_padded_domain']=not sample(np.zeros((3,4),np.float32),[-.5,-.5],4,True,True,[3,4])[0].any()
 tests['factualU_and_runtime_missing_do_not_gain_SAM']=not sample(logits,q,L,False,True,[3,4])[1].any() and not sample(logits,q,L,True,False,[3,4])[1].any()
 s=np.array([[0.,-0.],[np.nextafter(np.float32(0),np.float32(1)),1.],[-1,1],[np.nan,np.nan]],np.float32);o=np.array([[0,0,0],[0,1,1],[3,2,1],[np.nan]*3],np.float32);sp,op,pr=scalar_decode(s,o,np.array([1,1,1,0],bool))
 tests['strict_support_both_firsttie_foreign_priority_Ufirst']=sp.tolist()==[[0,0],[1,1],[0,1],[-1,-1]] and op.tolist()==[0,1,0,-1] and pr.tolist()==[-1,1,1,-1]
 try:scalar_decode(s,o,np.ones(4,bool));rejected=False
 except AssertionError:rejected=True
 tests['available_NaN_rejected']=rejected
 p=H/'INVENTED_NUMERIC_SERIALIZATION.json';write(p,dict(rows=int(np.int64(2144)),PASS=bool(np.bool_(True))));tests['actual_receipt_writer_native_scalar_types']=read(p)==dict(rows=2144,PASS=True)
 need(all(tests.values()),'Invented regression failure');return tests
def prepare():
 need(sha(CODE)==CODE_SHA and sha(PROTO)==PROTO_SHA and sha(RAW)==RAW_SHA and sha(INPUT_REVIEW)==INPUT_REVIEW_SHA,'Frozen reviewed dependencies')
 tests=invented();write(H/'GENERATED_INVENTED_CHECKS.json',dict(completed=True,PASS=True,checks=tests,count=len(tests),real_saved_input_SAM_or_prediction_body_reads=0))
 p=dict(completed=True,prepared_only=True,code=bind(__file__),static_review=bind(H/'STATIC_REVIEW.json'),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),composer_code=bind(CODE),composer_protocol=bind(PROTO),original_raw_complete=bind(RAW),independent_original_input_review=bind(INPUT_REVIEW),coverage='All2144 SAM96 masks/availability reconstructed from saved fullimage FP32 SAM logits with independent FP64 fourtap; all8 original arrays byteexact; all4288 output metadata, strictsign/firsttie/proxy/U checked. Head state/finalcheckpoint/eachbatch-repeat validated through exact source-fit and frozen-inference receipts, no model/weight reads or new forward.',actual_allowed_only_after='Root trigger binding actual PREDICTION_SEAL after successful terminal.',forbidden='No annotation/reference/oldscore/sourceGT/fullRGB/feature/checkpoint bodies; no torch/model/SAM/classifier calls or fits.',automatic_retry=False)
 write(H/'PROTOCOL.json',p);write(H/'ROOT_AUDIT_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_prediction_audit=False,code=p['code'],protocol=bind(H/'PROTOCOL.json'),prediction_seal=None));write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(H/'PROTOCOL.json'),checks=p['checks'],real_saved_input_SAM_prediction_bodies_read=0,model_forward_or_fit=0));print(json.dumps(dict(PASS=True,preparation=bind(H/'PREPARATION_COMPLETE.json'),code=p['code'],protocol=bind(H/'PROTOCOL.json'))),flush=True)
def audit(trigger):
 need(not (H/'ACTUAL_STARTED.json').exists() and not (H/'REVIEW.json').exists(),'One actual audit/preserve failure')
 e=read(trigger);need(e['approved_saved_prediction_audit'] is True and e['code']==bind(__file__) and e['protocol']==bind(H/'PROTOCOL.json'),'Root trigger exact prepared code/protocol')
 write(H/'ACTUAL_STARTED.json',dict(trigger=bind(trigger),pid=os.getpid()));start=time.monotonic()
 try:
  p=read(H/'PROTOCOL.json');need(p['composer_code']==bind(CODE) and p['composer_protocol']==bind(PROTO) and p['original_raw_complete']==bind(RAW) and p['independent_original_input_review']==bind(INPUT_REVIEW),'Prepared exactdependencies')
  seal=read(check(e['prediction_seal']));need(seal['completed'] and seal['PASS'] and seal['code']==p['composer_code'] and seal['protocol']==p['composer_protocol'] and seal['original_raw_complete']==p['original_raw_complete'],'Closed prediction provenance')
  need(seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'] and seal['all_batch_repeat2_state_and_input_bytes_exact'] and seal['original_rows2144']==2144 and seal['model_domain_rows4288']==4288,'All2 seal/count/repeat barrier')
  cp=read(PROTO);fit=read(check(seal['source_fit_complete']));fitreview=read(check(seal['source_fit_integrity_review']));need(fitreview['PASS'] and fitreview['actual_complete']==seal['source_fit_complete'] and fit['PASS'] and fit['updates_each']==8000 and fit['total_updates']==16000 and fit['all2_frozen_before_outputs_or_evaluation_access'],'Frozen original8000 heads/integrity')
  freeze=read(check(fit['all2_freeze']));need(freeze['completed'] and not freeze['saved_outputs_emitted'],'Both original heads frozen before outputs')
  raw=read(check(seal['original_raw_complete']));cache=read(check(seal['raw10_cache_complete']));sam=read(check(seal['SAM_complete']));need(cache['PASS'] and sam['PASS'] and cache['original_raw_complete']==seal['original_raw_complete'] and cache['SAM_complete']==seal['SAM_complete'] and sam['model_state_byte_unchanged'] and sam['first_prompt_group_repeat']['PASS'],'Closed raw10/SAM receipts')
  gg=lines(check(raw['groups']));smap={(g['domain'],g['group_id']):g for g in sam['groups']};need(len(gg)==len(smap)==268,'All268 groups')
  for g in gg:
   s=smap[(g['domain'],g['group_id'])];need(all(k in s and s[k]==v for k,v in g.items()),'Original group metadata exact');need(s['RGB_verified'] and s['observed_decoded_RGB_sha256']==g['decoded_RGB_sha256'],'Decoded RGB proof metadata')
  proof=lines(check(cache['proofs']));need(len(proof)==2144,'All2144 sample proofs');stats={};allx={};cursor=0;packet=None;packetkey=None
  for domain,(n,U) in DOMAINS.items():
   old=npz(raw['domains'][domain]['inputs'],RAW8);rec=cache['domains'][domain];x=npz(rec['inputs'],RAW10);rows=lines(check(rec['metadata']));need(rec['metadata']==raw['domains'][domain]['metadata'] and rec['original_raw8']==raw['domains'][domain]['inputs'] and len(rows)==n,'Original row/source bindings')
   need(all(exact(x[k],old[k]) for k in RAW8),'Original raw8 every array byteexact');need(rec['original_array_fingerprints']==fingerprints(old),'Raw8 receipt fingerprints')
   need(x['sam_mask96'].dtype==x['sam_available96'].dtype==bool and x['sam_mask96'].shape==x['sam_available96'].shape==(n,96,96),'SAM96 bool shape')
   checkedrows=0;sam_none=0
   for i,r in enumerate(rows):
    key=(domain,r['group_id']);g=smap[key];av=bool(x['original_query_available'][i]);runtime=bool(g['runtime_available'])
    if runtime and key!=packetkey:
     packet=npz(g['outputs'],SAM12);packetkey=key;hw=tuple(g['raw_hw']);need(packet['image_logits'].dtype==np.float32 and packet['image_logits'].shape==hw and np.isfinite(packet['image_logits']).all(),'SAM original arbitraryHW logit')
     need(packet['mask_logits_gt0'].dtype==bool and np.array_equal(packet['mask_logits_gt0'],packet['image_logits']>0),'Full SAM strictsign')
     need(numeric(packet['original_query_xy'],g['original_query_xy8']) and numeric(packet['original_L'],g['original_L8']) and numeric(packet['box_xyxy'],g['box_xyxy']),'SAM original F64 qL/box')
     need(packet['native_slot'].tolist()==list(range(8)) and packet['original_occurrence_id'].tolist()==[t['row_id'] for t in g['original_input_metadata8']] and packet['original_global_index'].tolist()==[t['original_global_index'] for t in g['original_input_metadata8']] and packet['original_query_available'].tolist()==g['original_query_available8'] and exact(packet['prediction_available'],packet['original_query_available']),'SAM native/IDs/input availability')
    expected,available=sample(None if not runtime else packet['image_logits'],x['query_xy'][i],float(x['L'][i]),av,runtime,g['raw_hw']);need(exact(expected,x['sam_mask96'][i]) and exact(available,x['sam_available96'][i]),'Independent all2144 FP64 fourtap row '+str((domain,i)))
    if av and runtime:need(exact(available,x['valid96'][i]),'SAM grid equals original validdomain')
    else:need(not expected.any() and not available.any(),'Missing SAM stayszero without changing originalavailability')
    pr=proof[cursor];cursor+=1;need(pr['row_id']==r['row_id'] and pr['domain']==domain and pr['domain_index']==i and pr['original_global_index']==r['original_global_index'] and pr['native_slot']==r['native_slot'] and pr['original_query_available']==av and pr['runtime_available']==runtime and pr['prompt_available']==g['prompt_available'] and pr['SAM_output']==g['outputs'],'Per-row input proof lineage')
    need(pr['sam_mask96_sha256']==ash(expected) and pr['sam_available96_sha256']==ash(available) and pr['original_valid96_sha256']==ash(x['valid96'][i]),'Saved row facthash')
    checkedrows+=1;sam_none+=not available.any()
   need(int((~x['original_query_available']).sum())==U and rec['SAM_allpixel_unavailable_rows']==sam_none,'FactualU and separateSAMmissing')
   stats[domain]=dict(rows=n,original_inputU=U,all_original8_byteexact=True,SAM96_independent_fourtap_rows=checkedrows,SAM_allpixel_unavailable_rows=sam_none);allx[domain]=x
  need(cursor==2144 and set(seal['arms'])==set(ARMS),'All rows/arms');outstats={}
  for arm,use_sam in ARMS.items():
   a=seal['arms'][arm];fixed=fit['arms'][arm];need(a['checkpoint']==fixed['checkpoint']==freeze['checkpoints'][arm] and a['updates']==8000 and a['use_sam']==use_sam and a['source_final_state_sha']==fixed['final_state_sha']==cp['models'][arm]['final_state_sha'],'Exact finalcheckpoint/state receipt binding only')
   outstats[arm]={}
   for domain,(n,U) in DOMAINS.items():
    x=allx[domain];z=npz(a['outputs'][domain],OUT13);av=x['original_query_available'];need(all(exact(z[k],x[k]) for k in META),'All output META6 exactbytes')
    need(z['raw_support_logits'].dtype==z['raw_owner_logits'].dtype==np.float32 and z['raw_support_logits'].shape==(n,2) and z['raw_owner_logits'].shape==(n,3),'Raw head FP32 shape')
    sp,op,proxy=scalar_decode(z['raw_support_logits'],z['raw_owner_logits'],av)
    need(exact(sp,z['support_prediction']) and exact(op,z['owner_prediction']) and exact(proxy,z['regional_support_proxy_D0_E1_Uminus1']),'All scalar strictsign/firsttie/proxy decodes')
    need(z['support_order'].tolist()==['visible_end_support','foreign_front_support'] and z['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'Output separate semantic orders')
    receipt=a['inference'][domain];batches=(n-U+191)//192;need(receipt['batches']==batches and receipt['head_calls']==2*batches and receipt['all_batch_repeat2_exact'] and receipt['available_rows_forwarded']==n-U and receipt['original_U_not_forwarded']==U,'Fixed available-only batch192 repeat2 counts')
    need(receipt['state_before']==receipt['state_after']==fixed['final_state_sha'] and receipt['input_fingerprints']==fingerprints(x),'Finalstate and input receipt fingerprints')
    outstats[arm][domain]=dict(rows=n,original_U=U,available=n-U,META6_byteexact=True,scalar_decoded_rows=n,batches=batches,repeat2_head_calls=2*batches,source_final_state_receipt_equal=True)
  for path,d in BINDINGS.items():need(sha(path)==d,'Saved audit inputs unchanged')
  review=dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(H/'PROTOCOL.json'),checks=bind(H/'GENERATED_INVENTED_CHECKS.json'),trigger=bind(trigger),prediction_seal=e['prediction_seal'],independent_original_input_review=p['independent_original_input_review'],domains=stats,outputs=outstats,all2144_SAM96_FP64_fourtap_exact=True,all4288_output_META6_scalar_decode_exact=True,all_original_domain_ID_qL_inputU_preserved=True,checkpoint_state_validation='Exact finalcheckpoint binding/stateSHA receipts match frozen fit, original all2freeze and inference before/after; no checkpoint body loaded/hashed and no model constructed/forwarded.',limitations=['Batch-repeat and final model state verified through frozen source code and exact bound runtime receipts; no independent new forward.','Original8 compared byte-for-byte to independently audited original raw8 adapter; original fullRGB not reopened.','No annotation/reference/oldscore/sourceGT or metric interpretation; outputs checked only input preservation and decoding.'],verified_input_bindings=[dict(path=k,sha256=v) for k,v in BINDINGS.items()],saved_audit_input_hashes_unchanged=True,annotation_reference_oldscore_GT_fullRGB_feature_checkpoint_body_reads=0,model_SAM_classifier_forward_fit_calls=0,selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-start)
  write(H/'REVIEW.json',review);print(json.dumps(dict(PASS=True,review=bind(H/'REVIEW.json'),domains=stats,outputs=outstats,seconds=review['seconds'])),flush=True)
 except BaseException:
  write(H/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--prepare',action='store_true');m.add_argument('--trigger',type=Path);a=ap.parse_args();prepare() if a.prepare else audit(a.trigger)
