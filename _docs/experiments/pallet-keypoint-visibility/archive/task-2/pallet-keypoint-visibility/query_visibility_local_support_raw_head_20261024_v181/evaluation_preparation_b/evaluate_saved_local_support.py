"""Saved-output-only source evaluation; prepare never opens model/cache outputs.

The two supports are nonexclusive. Literal owner, fixed end-support and the
historical v61 pixel-presence diagnostic remain distinct tasks/denominators.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_RAW_OR_SAM_CACHE_OR_FIT.json'
SCOPE_SHA='8cd21d5eb4caf9663927859ead0cbcbd823e701fd4a2164a66aa765e046bfcdc'
EVAL_SCOPE=V/'ROOT_SAVED_EVALUATION_AND_FIXED_VISUAL_SCOPE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_NEW_SOURCE_OUTPUT_OR_VAL_TARGET_BODIES.json'
TESTS=H/'GENERATED_INVENTED_CHECKS.json'
OUT=V/'saved_local_support_source_evaluation_v1'
TRAIN_DOMAINS={'SYNTH_TRAIN4096':4096,'PUBLIC_TRAIN1480':1480,'CF3816':3816}
DOMAINS=dict(TRAIN_DOMAINS,SYNTH_VAL2048=2048)
ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
V132=P/'query_visibility_actual_initializer_source_20261024_v132'
V61_SEAL=V132/'frozen_v61_actual_initializer_v1/PREDICTION_SEAL.json'
V61_PRED={'path':str(V132/'frozen_v61_actual_initializer_v1/V61_PREDICTIONS_INPUT_ONLY.npz'),'sha256':'9361eb530b42e3150287a79127f7809181f8252408abb5d37ff79b8454500df9'}
V61_CODE=V132/'frozen_visibility_preparation/run_frozen_visibility.py'
SOURCE_DIAGNOSTIC={'path':str(P/'query_visibility_full_source_query_causal_census_20261024_v173/census_preparation/ALL6144_ORIGINAL_QUERY_METADATA_BEFORE_MASKS.jsonl'),'sha256':'09c4d69b0c6a6b8b10c025ad3dacb808a4fa208045753ba7cb99e85c5026a609'}
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']

def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(p):
 p=Path(p).resolve();require(p.is_relative_to(P) and not any(a in str(p).lower() for a in ('_v112','_v169','_v177','_v178','current223','/private_evaluation/')),'Forbidden source path')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1<<20),b''):h.update(z)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):require(bind(b['path'])==b,'SHA '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def equal(a,b):
 require(a.dtype==b.dtype and a.shape==b.shape,'Array dtype/shape')
 require(np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b),'Original array equality')
def ratio(a,b):return None if b==0 else a/b

def cm_metric(truth,pred,classes):
 cm=[[0]*(classes+1) for _ in range(classes)];unknown=[0]*(classes+1)
 for y,p in zip(truth,pred):
  require(-1<=int(y)<classes and -1<=int(p)<classes,'Categorical schema')
  col=classes if p<0 else int(p)
  if y<0:unknown[col]+=1
  else:cm[int(y)][col]+=1
 precision=[];recall=[];support=[]
 for c in range(classes):
  n=sum(cm[c]);support.append(n);precision.append(ratio(cm[c][c],sum(row[c] for row in cm)));recall.append(ratio(cm[c][c],n))
 result=dict(CM=cm,truth_classes=list(range(classes)),prediction_columns=list(range(classes))+['U'],rows_all=len(truth),truth_known=sum(support),truth_unknown=sum(unknown),class_support=support,precision=precision,recall=recall,prediction_U_among_known=sum(row[-1] for row in cm),prediction_counts_among_unknown_truth=unknown)
 if classes==2:result.update(TN=cm[0][0],FP=cm[0][1],FN_without_positive_U=cm[1][0],TP=cm[1][1],U_per_truth=[cm[0][2],cm[1][2]])
 return result

def soft_owner_metric(soft,amb,invalid,pred):
 buckets=[[[] for _ in range(4)] for _ in range(3)];missing=0
 for i,p in enumerate(pred):
  require(-1<=int(p)<3,'Owner prediction schema')
  if not np.isfinite(soft[i]).all():
   require(np.isnan(soft[i]).all(),'No partially missing owner mass');missing+=1;continue
  require(((soft[i]>=0)&(soft[i]<=1)).all() and soft[i].sum()<=1+1e-12,'Raw partial soft owner mass')
  col=3 if p<0 else int(p)
  for c in range(3):buckets[c][col].append(float(soft[i,c]))
 cm=[[math.fsum(v) for v in row] for row in buckets]
 finite=np.isfinite(soft).all(1);known=[math.fsum(row) for row in cm]
 return dict(CM_T_F_BG_to_T_F_BG_U=cm,rows_all=len(pred),rows_owner_truth_unobserved_or_missing=missing,raw_known_class_mass=known,
  ambiguous_mass=math.fsum(float(v) for v in amb[finite]),invalid_mass=math.fsum(float(v) for v in invalid[finite]),
  precision=[ratio(cm[c][c],math.fsum(row[c] for row in cm)) for c in range(3)],recall=[ratio(cm[c][c],known[c]) for c in range(3)],
  prediction_U_known_mass=math.fsum(row[-1] for row in cm),no_cutoff_or_row_renormalization=True,not_native_visibility=True)

def decode(raw_support,raw_owner,available):
 n=len(available);require(raw_support.dtype==raw_owner.dtype==np.float32 and raw_support.shape==(n,2) and raw_owner.shape==(n,3),'Saved raw schema')
 require(np.isfinite(raw_support[available]).all() and np.isfinite(raw_owner[available]).all() and np.isnan(raw_support[~available]).all() and np.isnan(raw_owner[~available]).all(),'Original U rawNaN first')
 support=np.full((n,2),-1,np.int8);owner=np.full(n,-1,np.int8);proxy=np.full(n,-1,np.int8)
 support[available]=(raw_support[available]>0).astype(np.int8);owner[available]=raw_owner[available].argmax(1).astype(np.int8)
 proxy[available]=np.where(support[available,1]==1,1,np.where(support[available,0]==1,0,-1)).astype(np.int8)
 return support,owner,proxy

def joint(values):
 z=np.full(len(values),-1,np.int8);known=(values>=0).all(1)
 # States NONE00=0,VISIBLE_ONLY10=1,FOREIGN_ONLY01=2,BOTH11=3.
 z[known]=values[known,0]+2*values[known,1];return z
def truth_proxy(target):
 # Fixed conditional proxy is compared only where BOTH support truths are known.
 both=(target>=0).all(1);out=np.full(len(target),-1,np.int8)
 out[both&(target[:,1]==1)]=1;out[both&(target[:,1]==0)&(target[:,0]==1)]=0;return out

def source_native_association(rows,pred):
 roles=['D','E','S','U'];cm=[[0]*3 for _ in roles];not_available=0
 for r,p in zip(rows,pred):
  role=r.get('original_semantic_native_role')
  if role not in roles:not_available+=1;continue
  cm[roles.index(role)][2 if p<0 else int(p)]+=1
 return dict(declared_semantic_rows=roles,regional_proxy_columns=['D','E','U'],counts=cm,rows_no_original_native_truth=not_available,
  authority='Original CF/V4 declared semantic association only; physical initializer native identity is uncertified. This is separate from fixed support truth and is not a relabel or success gate.')

def strata(rows):
 groups={'ALL':list(range(len(rows)))}
 for slot in range(8):groups['native:'+str(slot)]=[i for i,r in enumerate(rows) if r['native_slot']==slot]
 for field in ['material','aspect','geometry_family','foreign_family','variant']:
  values=[]
  for r in rows:
   original=r['original'];values.append(original.get('variant','UNDECLARED') if field=='variant' else original.get('factors',{}).get(field,'UNDECLARED'))
  for value in sorted(set(values),key=str):groups[field+':'+str(value)]=[i for i,v in enumerate(values) if v==value]
 for field in ['original_semantic_native_role','reference_error_stratum','position_band']:
  values=[str(r[field]) if r.get(field) is not None else ('missing' if field!='original_semantic_native_role' else 'NOT_AVAILABLE') for r in rows]
  for value in sorted(set(values)):groups[field+':'+value]=[i for i,v in enumerate(values) if v==value]
 return {k:np.asarray(v,np.int64) for k,v in groups.items()}

def summary(rows,available,target,soft,amb,invalid,owner_target,support,owner,proxy):
 results={}
 for name,ids in strata(rows).items():
  st=target[ids];sp=support[ids];pred=proxy[ids]
  av=available[ids]
  results[name]=dict(rows_all=len(ids),original_input_available=int(av.sum()),original_inputU=int((~av).sum()),
    visible_end_support=dict(primary_known_available=cm_metric(st[av,0],sp[av,0],2),secondary_known_all=cm_metric(st[:,0],sp[:,0],2)),
    foreign_front_support=dict(primary_known_available=cm_metric(st[av,1],sp[av,1],2),secondary_known_all=cm_metric(st[:,1],sp[:,1],2)),
    joint4=cm_metric(joint(st),joint(sp),4),owner_soft=soft_owner_metric(soft[ids],amb[ids],invalid[ids],owner[ids]),
    owner_hard_weak_only=cm_metric(owner_target[ids],owner[ids],3),regional_proxy_D0_E1=cm_metric(truth_proxy(st),pred,2),
    original_native_semantic_proxy_association=source_native_association([rows[int(i)] for i in ids],pred),
    proxy_truth_NEITHER_or_UNKNOWN_rows=int((truth_proxy(st)<0).sum()),source_support_not_physical_native_gold=True)
 return results

def load_output(binding,cache_record,n,diagnostic):
 with np.load(check(binding),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 # Deliberately never load pixel INPUTS: identity is verified against ROWS metadata.
 rows=lines(check(cache_record['metadata']));require(len(rows)==n,'All source metadata rows')
 for key,dtype,shape in [('original_query_available',np.bool_,(n,)),('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,))]:
  require(a[key].dtype==dtype and a[key].shape==shape,'Saved original metadata schema '+key)
 for k in META:
  if k=='query_xy':expected=np.asarray([[np.nan,np.nan] if r[k] is None else r[k] for r in rows],np.float64)
  elif k=='L':expected=np.asarray([np.nan if r[k] is None else r[k] for r in rows],np.float64)
  else:expected=np.asarray([r[k] for r in rows],dtype=a[k].dtype)
  equal(a[k],expected)
 for r in rows:
  if r['split'] in ['TRAIN','VALIDATION']:
   d=diagnostic[r['original_global_index']]
   require(d['original_occurrence_id']==r['row_id'] and d['native_slot']==r['native_slot'] and d['split']==r['split'],'Original source diagnostic ID/slot/split join')
   q=np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);dq=np.asarray([np.nan,np.nan] if d['query_xy'] is None else d['query_xy'],np.float64)
   require(np.array_equal(q,dq,equal_nan=True) and d['L']==r['L'],'Original source diagnostic unchanged qL')
   r['original_semantic_native_role']=d['original_semantic_state'];error=d['q_reference_error_px']
   r['reference_error_stratum']='missing' if error is None or not math.isfinite(error) else '<=3px' if error<=3 else '>3px'
   r['position_band']=d['position_band']
  else:r.update(original_semantic_native_role='NOT_AVAILABLE',reference_error_stratum='missing',position_band='missing')
 require(a['support_order'].tolist()==['visible_end_support','foreign_front_support'] and a['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'Output task order')
 with np.load(check(cache_record['targets']),allow_pickle=False) as z:y={k:z[k].copy() for k in z.files}
 support,owner,proxy=decode(a['raw_support_logits'],a['raw_owner_logits'],a['original_query_available'])
 for k,v in [('support_prediction',support),('owner_prediction',owner),('regional_support_proxy_D0_E1_Uminus1',proxy)]:equal(a[k],v)
 require(y['support_target'].shape==(n,2) and y['owner_soft'].shape==(n,3),'Separate target rows')
 return a,y,rows,support,owner,proxy

def v61_diagnostic(binding,packets):
 with np.load(check(binding),allow_pickle=False) as z:
  keys=['global_index','occurrence_id','native_slot','query_xy','L','query_available','prediction_D0_E1_Uminus1','raw_visible_logit']
  a={k:z[k].copy() for k in keys}
 require(len(a['global_index'])==6144 and a['raw_visible_logit'].dtype==np.float32,'Historic original6144 v61 schema')
 results={}
 for domain in ['SYNTH_TRAIN4096','SYNTH_VAL2048']:
  x,y,rows,*unused=packets[domain];ids=x['original_global_index'].astype(np.int64)
  for k,dst in [('global_index','original_global_index'),('occurrence_id','row_id'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L')]:
   v=a[k][ids];expected=x[dst];require(v.shape==expected.shape and (np.array_equal(v,expected,equal_nan=True) if v.dtype.kind in 'fc' else np.array_equal(v,expected)),'v61 exact same original q/ID/L/native join')
  raw=a['raw_visible_logit'][ids];available=a['query_available'][ids];pred=a['prediction_D0_E1_Uminus1'][ids]
  expected=np.full(len(ids),-1,np.int8);expected[available]=(raw[available]<0).astype(np.int8);equal(pred,expected)
  support=np.full((len(ids),2),-1,np.int8);support[available,0]=(pred[available]==0).astype(np.int8);support[available,1]=(pred[available]==1).astype(np.int8)
  unknown_owner=np.full(len(ids),-1,np.int8)
  results[domain]=summary(rows,available,y['support_target'],y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass'],y['owner_target'],support,unknown_owner,pred)
  for name,ii in strata(rows).items():
   m=results[domain][name]
   m['historical_v61_available']=m.pop('original_input_available');m['historical_v61_U']=m.pop('original_inputU')
   m['original_V181_query_available']=int(x['original_query_available'][ii].sum());m['original_V181_inputU']=int((~x['original_query_available'][ii]).sum())
   del m['owner_soft'];del m['owner_hard_weak_only'];m['owner3_output_not_defined']=True
 return dict(meaning='Historical pixel-presence binary sign shown against experimental regional-end targets only as a task-mismatch diagnostic. v61 cannot predict BOTH, has its own original availability, and has no owner3 output. No equivalence or native certification claimed.',baseline=results)

def invented():
 c=[]
 def passed(name,ok):require(ok,name);c.append(dict(name=name,PASS=True))
 y=np.array([0,0,1,1,-1,1],np.int8);p=np.array([0,1,1,-1,0,0],np.int8);m=cm_metric(y,p,2)
 passed('Binary CM keeps U and unknowntruth separate',m['CM']==[[1,1,0],[1,1,1]] and m['truth_known']==5 and m['truth_unknown']==1 and m['prediction_counts_among_unknown_truth']==[1,0,0])
 passed('Precision recall use knowntruth denominators including U',m['precision']==[.5,.5] and m['recall']==[.5,1/3])
 z=np.array([[0,0],[1,0],[0,1],[1,1],[-1,1]],np.int8)
 passed('Four joint nonexclusive states and UNKNOWN',joint(z).tolist()==[0,1,2,3,-1])
 passed('Regional truth requires both support known; NEITHER and unknown retained',truth_proxy(z).tolist()==[-1,0,1,1,-1])
 s=np.array([[.2,.1,0],[0,0,.7],[np.nan]*3],np.float64);amb=np.array([.3,.1,np.nan]);iv=np.array([.4,.2,np.nan]);o=np.array([0,-1,1],np.int8);m=soft_owner_metric(s,amb,iv,o)
 passed('Owner raw partialmass and predictionU preserved without hardcutoff',m['CM_T_F_BG_to_T_F_BG_U']==[[.2,0,0,0],[.1,0,0,0],[0,0,0,.7]] and m['rows_owner_truth_unobserved_or_missing']==1 and m['prediction_U_known_mass']==.7)
 sub=np.nextafter(np.float32(0),np.float32(1));raw=np.array([[0.,-0.],[sub,-sub],[np.nan,np.nan]],np.float32);owner=np.array([[0,0,0],[1,2,0],[np.nan]*3],np.float32);sp,op,proxy=decode(raw,owner,np.array([1,1,0],bool))
 passed('Strict greater0 and float32 subnormal boundary; argmax and Ufirst',sp.tolist()==[[0,0],[1,0],[-1,-1]] and op.tolist()==[0,1,-1] and proxy.tolist()==[-1,0,-1])
 r=[dict(native_slot=0,original=dict(factors=dict(material='WOOD',aspect='SQUARE',foreign_family='TEST'),variant='BEFORE'))]
 st=strata(r);passed('All8native groups retained including empty groups',all('native:'+str(k) in st for k in range(8)) and len(st['native:7'])==0)
 passed('Empty CM denominator explicitly null',cm_metric(np.array([],np.int8),np.array([],np.int8),2)['recall']==[None,None])
 native=source_native_association([{'original_semantic_native_role':x} for x in ['D','E','S','U','NOT_AVAILABLE']],np.array([0,1,-1,0,1],np.int8))
 passed('Original declared nativeS/U retained as separate association, no visibility relabel',native['counts']==[[1,0,0],[0,1,0],[0,0,1],[1,0,0]] and native['rows_no_original_native_truth']==1)
 rows=[dict(native_slot=0,original={},position_band=None,reference_error_stratum='missing')]
 passed('Missing association metadata creates explicit missingstratum',len(strata(rows)['position_band:missing'])==1)
 return c

def prepare():
 require(sha(SCOPE)==SCOPE_SHA,'Frozen scope')
 tests=invented();write(TESTS,dict(completed=True,PASS=True,code=bind(__file__),tests=tests,actual_source_output_target_feature_or_weight_reads=0))
 v61=read(V61_SEAL);require(v61['completed'] and v61['PASS'] and v61['predictions']==V61_PRED,'Historic v61 metadata-only binding')
 p=dict(completed=True,prepared_only=True,scope=bind(SCOPE),evaluation_scope=bind(EVAL_SCOPE),code=bind(__file__),tests=bind(TESTS),output=str(OUT),domains=DOMAINS,arms=ARMS,
   evaluation='All saved outputs from both final8000 frozen heads, same fixed threshold0 and owner argmax. No model/threshold/epoch/arm selection.',
   binary='Each support separately: primary knowntruth&originalavailable and secondary allknowntruth, 2x3 CM NO/YES x NO/YES/U, TP/FP/FN/TN/U plus P/R and unknowntruth prediction census. Supports nonexclusive.',
   joint='4x5 NONE/VISIBLE_ONLY/FOREIGN_ONLY/BOTH x same4+U; truthbothknown only. Keep unknowntruth counts.',
   owner='Raw partial T/F/BG classmass -> T/F/BG/U softCM, explicit AMBIG/INVALID/missingweaktruth; no cutoff or row-normalization; hardweak CM separate.',
   regional_proxy='foreignYES→E else visibleYES→D else U. Truthcomparison only bothsupportknown: E when fixedforeign1, D only foreign0&visible1; NEITHER/UNKNOWN stays outsideD/Etruth. Regional diagnostic, not physical native labels.',
   strata='Each originaldomain/split separately; ALL + native0..7 including empty + provided material/aspect/geometry/foreignfamily/variant, declaredoldnativeD/E/S/U, referenceerror<=3/>3/missing andpositionband. Referenceassociation only, not pixelerror/nativeidentity gold; no inferred public strata.',
   sourceVAL='Historically exposed synthetic VALIDATION2048, evaluationonly, not a blind independent research holdout.',
   v61_metadata_only_readiness=dict(seal=bind(V61_SEAL),producer=bind(V61_CODE),inherited_output_binding_not_opened_or_rehashed=V61_PRED,rows6144=True,same_qL_native_id_fields_saved=True,own_decoder='D visible>=0,E visible<0; original own continuous/nearest/ROIany U retained',separate_task_diagnostic_only=True),
   source_diagnostic_metadata_inherited_binding_not_read_or_hashed=SOURCE_DIAGNOSTIC,
   exact_input_cache_complete_bindings_pending=True,no_pixel_feature_weight_model_forward_required=True,new_training=0,selected_model=None,fully_solved=False)
 write(PROTOCOL,p)
 write(H/'ROOT_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_source_evaluation=False,source_only=True,all_two_models_frozen=False,scope=bind(SCOPE),evaluation_scope=bind(EVAL_SCOPE),code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=None,training_cache_complete=None,evaluation_cache_complete=None,source_diagnostic_metadata=SOURCE_DIAGNOSTIC,include_historic_v61_fixed_diagnostic=True))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),tests=bind(TESTS),actual_output_or_new_target_bodies_read=0,new_model_forward_fit=0))

def run(entry_path):
 e=read(entry_path);require(e.get('approved_saved_source_evaluation') is True and e.get('source_only') is True and e.get('all_two_models_frozen') is True,'Root all2freeze saved-only trigger required')
 for k,p in [('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL)]:require(e[k]==bind(p),'Entry '+k)
 p=read(PROTOCOL);require(p['tests']==bind(TESTS) and p['scope']==bind(SCOPE) and sha(SCOPE)==SCOPE_SHA,'Frozen evaluation rules')
 require(e['evaluation_scope']==p['evaluation_scope']==bind(EVAL_SCOPE),'Saved-only evaluation scope')
 fit=read(check(e['fit_complete']));require(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_VAL_access'] and set(fit['arms'])==set(ARMS),'Both fixed fits complete')
 freeze=read(check(fit['all2_freeze']));require(freeze['completed'] and not freeze['saved_outputs_emitted'],'Explicit all2freeze barrier before saved output')
 train=read(check(e['training_cache_complete']));val=read(check(e['evaluation_cache_complete']));require(train['completed'] and train['PASS'] and val['completed'] and val['PASS'],'Source target caches ready')
 require(e['source_diagnostic_metadata']==p['source_diagnostic_metadata_inherited_binding_not_read_or_hashed'],'Exact source diagnostic metadata binding')
 dd=lines(check(e['source_diagnostic_metadata']));require(len(dd)==6144 and [r['original_global_index'] for r in dd]==list(range(6144)),'All6144 diagnostic originalrows');diagnostic={r['original_global_index']:r for r in dd}
 records=dict(train['domains'],**val['domains']);require(set(records)==set(DOMAINS),'All original11440 source rows')
 require(not OUT.exists(),'No overwrite or automatic retry');OUT.mkdir();start=time.monotonic()
 write(OUT/'STARTED.json',dict(entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),saved_only=True))
 try:
  result={};ledger_bindings={};all_used=[]
  for arm in ARMS:
   require(fit['arms'][arm]['updates']==8000,'Fixed final8000');packets={};result[arm]={};ledger=OUT/(arm+'_ALL11440_SOURCE_SAVED_ROWS.jsonl')
   with ledger.open('x') as f:
    for domain,n in DOMAINS.items():
     packets[domain]=load_output(fit['arms'][arm]['outputs'][domain],records[domain],n,diagnostic);a,y,rows,sp,op,proxy=packets[domain]
     expected_split='VALIDATION' if domain=='SYNTH_VAL2048' else 'TRAIN' if domain=='SYNTH_TRAIN4096' else 'SOURCE_DEVELOPMENT';require(all(r['split']==expected_split for r in rows),'Fixed TRAIN/VAL/source-only split')
     result[arm][domain]=summary(rows,a['original_query_available'],y['support_target'],y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass'],y['owner_target'],sp,op,proxy)
     for i,r in enumerate(rows):
      row=dict(domain=domain,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],split=r['split'],query_xy=r['query_xy'],L=r['L'],original_query_available=r['original_query_available'],support_target=y['support_target'][i].tolist(),support_prediction=sp[i].tolist(),owner_target=int(y['owner_target'][i]),owner_prediction=int(op[i]),regional_support_proxy=int(proxy[i]),new_native_visibility_or_quality_label=False)
      f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
     all_used.extend([fit['arms'][arm]['outputs'][domain],records[domain]['targets'],records[domain]['metadata']])
   ledger_bindings[arm]=bind(ledger)
   if e.get('include_historic_v61_fixed_diagnostic') and arm==ARMS[0]:result['historical_v61_separate_fixed_diagnostic']=v61_diagnostic(p['v61_metadata_only_readiness']['inherited_output_binding_not_opened_or_rehashed'],packets)
  rp=OUT/'FIXED_SUPPORT_OWNER_AND_STRATA.json';write(rp,dict(completed=True,source_only=True,results=result,training_domains_apparent_only=True,syntheticVAL_reused_research_split=True,task_and_label_authority_separated=True,selected_model=None,fully_solved=False))
  for b in all_used+[e['fit_complete'],e['training_cache_complete'],e['evaluation_cache_complete']]:check(b)
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=e['fit_complete'],all2_freeze=fit['all2_freeze'],training_cache_complete=e['training_cache_complete'],evaluation_cache_complete=e['evaluation_cache_complete'],summary=bind(rp),row_ledgers=ledger_bindings,original_source_rows_each_arm=11440,all2_frozen_before_readout=True,new_model_forward_fit_threshold_or_model_selection=0,GT_native_or_quality_relabel=0,private_or_publicheld_access=0,selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
 a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=a.parse_args();prepare() if args.prepare else run(args.entry)
