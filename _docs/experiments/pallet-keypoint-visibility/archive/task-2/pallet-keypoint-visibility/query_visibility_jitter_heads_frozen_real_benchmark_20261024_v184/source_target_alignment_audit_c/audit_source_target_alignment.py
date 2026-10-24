"""Source TRAIN saved-only owner/end-support descriptive alignment census.
No RGB, input features, weights, model imports/forward, new labels or selection.
"""
from pathlib import Path
import argparse,hashlib,json,math,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
F=P/'query_visibility_query_jitter_paired_source_training_20261024_v183'
FIT=F/'runs/C52_C53_JITTER_PAIRED_SOURCE_FIXED8000/COMPLETE.json';FIT_SHA='7e6c231a7daabb69c0b545e06bb04edd0a3da8f2eb5b3ed313e9faa85435ec8c'
OP=P/'query_visibility_frozen_sam_source_baseline_20261024_v180/source_end_supervision_preparation_a/source_end_support.py'
OLDPRODUCER=P/'query_visibility_local_support_raw_head_20261024_v181/raw_cache_preparation/build_source_raw96_cache.py'
AUGPRODUCER=F/'augmentation_preparation_a/build_source_augmentations.py'
TRAINER=F/'source_fit_preparation/train_jitter_paired_source.py'
RULES=H/'PROSPECTIVE_RULES_BEFORE_SOURCE_TARGET_OR_SAVED_OUTPUT_BODIES.json';CHECKS=H/'GENERATED_MEANINGFUL_CHECKS.json';PREP=H/'PREPARATION_COMPLETE.json'
DOMAINS={'PUBLIC_TRAIN1480':1480,'CF3816':3816,'SYNTH_AUG65536':65536};US={'PUBLIC_TRAIN1480':194,'CF3816':360,'SYNTH_AUG65536':1264}
ARMS=['C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE'];OWNER=['TARGET','FOREIGN','BACKGROUND','UNKNOWN'];JOINT=['NONE','V_ONLY','F_ONLY','BOTH','U'];META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
TARGETS={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'}
KEYS=set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}
def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def safe(p):
 s=str(p).lower();require(not any(t in s for t in ['_v112','/private_evaluation/','/saved_real_head_evaluation','/frozen_c52_c53_real_predictions','/public_held','synth_val','synthetic_val','_v177/','_v178/','_v169/']),'Outside source-only scope before read');return Path(p)
def sha(p):
 h=hashlib.sha256()
 with safe(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).absolute()),sha256=sha(p))
def checked(b):require(bind(b['path'])=={k:b[k] for k in ['path','sha256']},'Exact binding '+b['path']);return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def write(p,x):
 with Path(p).open('x') as f:json.dump(clean(x),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def clean(v):
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(w) for k,w in v.items()}
 if isinstance(v,(list,tuple)):return [clean(w) for w in v]
 return v
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def eq(a,b):return np.array_equal(a,b,equal_nan=True)
def load(b):
 with np.load(checked(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def joint(sp):
 require(sp.dtype==np.int8 and sp.ndim==2 and sp.shape[1]==2 and np.isin(sp,[-1,0,1]).all(),'Frozen support2 bits');missing=(sp<0).any(1);require(np.array_equal((sp[:,0]<0),(sp[:,1]<0)),'No halfmissing predictions');return np.where(missing,4,sp[:,0]+2*sp[:,1]).astype(np.int8)
def packet(x,n):
 require(set(x)==KEYS and x['support_order'].tolist()==['visible_end_support','foreign_front_support'] and x['owner_order'].tolist()==OWNER[:3],'Exact saved13key orders')
 av=x['original_query_available'];require(av.dtype==bool and av.shape==(n,),'Original availability')
 for k,dt,shape in [('raw_support_logits',np.float32,(n,2)),('raw_owner_logits',np.float32,(n,3)),('support_prediction',np.int8,(n,2)),('owner_prediction',np.int8,(n,)),('regional_support_proxy_D0_E1_Uminus1',np.int8,(n,)),('query_xy',np.float64,(n,2)),('L',np.float64,(n,)),('original_global_index',np.int64,(n,)),('native_slot',np.int8,(n,))]:require(x[k].dtype==dt and x[k].shape==shape,'Saved schema '+k)
 require(x['row_id'].dtype.kind=='U' and len(set(x['row_id'].tolist()))==n,'Unique occurrence rowID; original source index need not unique')
 s=x['raw_support_logits'];o=x['raw_owner_logits'];sp=x['support_prediction'];op=x['owner_prediction'];pr=x['regional_support_proxy_D0_E1_Uminus1']
 require(np.isfinite(s[av]).all() and np.isfinite(o[av]).all() and np.isnan(s[~av]).all() and np.isnan(o[~av]).all(),'Ufirst rawNaN')
 require(np.array_equal(sp[av],(s[av]>0).astype(np.int8)) and (sp[~av]==-1).all() and np.array_equal(op[av],o[av].argmax(1)) and (op[~av]==-1).all(),'Frozen strictsign/firsttieowner/U')
 expected=np.where(sp[:,1]==1,1,np.where(sp[:,0]==1,0,-1)).astype(np.int8);expected[~av]=-1;require(np.array_equal(expected,pr),'Saved proxy unchanged');return joint(sp)
def targets(y,n,av):
 require(set(y)==TARGETS,'Exact separate target5');require(y['support_target'].dtype==np.int8 and y['support_target'].shape==(n,2) and np.isin(y['support_target'],[-1,0,1]).all(),'Original support2 targets')
 require(y['owner_target'].dtype==np.int8 and y['owner_target'].shape==(n,) and np.isin(y['owner_target'],[-1,0,1,2]).all(),'Original owner hardtarget')
 for k,shape in [('owner_soft',(n,3)),('owner_ambiguous_mass',(n,)),('owner_invalid_mass',(n,))]:require(y[k].dtype==np.float64 and y[k].shape==shape,'Original rawmass '+k)
 allmass=np.column_stack((y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass']));finite=np.isfinite(allmass).all(1);missing=np.isnan(allmass).all(1);require((finite|missing).all(),'Missingowner is allNaN, never imputed')
 require((allmass[finite]>=0).all() and np.allclose(allmass[finite].sum(1),1,rtol=0,atol=2e-13),'Unnormalized raw5 mass conservation')
 require((y['support_target'][~av]==-1).all() and (y['owner_target'][~av]==-1).all(),'FactualU remains unknown targets')
 known=y['owner_target']>=0;require(np.array_equal(y['owner_soft'][known],np.eye(3)[y['owner_target'][known]]) and (allmass[known,3:]==0).all(),'Original hardweak onehot retained');return allmass,finite,missing
def owner_census(y,pred,j):
 truth=y['owner_target'];cm=np.zeros((4,4),np.int64);table=np.zeros((4,5),np.int64)
 for t,p,s in zip(truth,pred,j):cm[int(t) if t>=0 else 3,int(p) if p>=0 else 3]+=1;table[int(t) if t>=0 else 3,int(s)]+=1
 require(cm.sum()==table.sum()==len(truth),'Owner allrows/U census');return dict(truth_order=OWNER,prediction_order=OWNER[:3]+['U'],owner3_confusion=cm,support_joint_order=JOINT,known_owner_vs_support_joint=table,training_apparent_only=True)
def mass_census(y,allmass,finite,missing,av,j):
 result={}
 for label,selected in [('ALL_ROWS',np.ones(len(j),bool)),('INPUT_AVAILABLE',av),('ORIGINAL_INPUT_U',~av)]:
  matrix=np.zeros((5,5),np.float64);counts=np.zeros(5,np.int64);missingcounts=np.zeros(5,np.int64)
  for k in range(5):
   ix=selected&(j==k);matrix[k]=np.sum(allmass[ix&finite],axis=0,dtype=np.float64);counts[k]=ix.sum();missingcounts[k]=(ix&missing).sum()
  require(int(counts.sum())==int(selected.sum()) and int(missingcounts.sum())==int((selected&missing).sum()),'All missing/U softmass rows retained')
  require(np.allclose(matrix.sum(0),np.sum(allmass[selected&finite],axis=0),rtol=0,atol=2e-8),'Rawmass joint sum closure without renormalization')
  truthrows=[]
  for v in [-1,0,1]:
   for f in [-1,0,1]:
    ix=selected&(y['support_target'][:,0]==v)&(y['support_target'][:,1]==f);truthrows.append(dict(visible_target=v,foreign_target=f,N=int(ix.sum()),raw_owner_mass5=np.sum(allmass[ix&finite],axis=0,dtype=np.float64),predicted_joint_counts=np.bincount(j[ix],minlength=5)))
  require(sum(t['N'] for t in truthrows)==int(selected.sum()),'Nonexclusive truth includingUNKNOWN exhaustive')
  result[label]=dict(rows=int(selected.sum()),measured_mass_rows=int((selected&finite).sum()),missing_owner_rows=int((selected&missing).sum()),support_joint_order=JOINT,owner_mass_order=OWNER[:3]+['AMBIGUOUS','INVALID_SUPPORT'],raw_owner_mass_by_predicted_support_joint=matrix,predicted_joint_row_counts=counts,missing_owner_by_predicted_joint=missingcounts,raw_total_mass5=np.sum(allmass[selected&finite],axis=0,dtype=np.float64),support_truth_combinations=truthrows)
 return result
def metadata_index(row,domain):
 key='diagnostic_row_index' if domain=='SYNTH_AUG65536' else 'domain_index';require(type(row[key]) is int,'Original metadata index schema');return row[key]
def inventory():
 require(sha(FIT)==FIT_SHA,'Fixed completed sourcefit metadata');fit=read(FIT);require(fit['completed'] and fit['PASS'] and fit['updates_each']==8000 and fit['total_updates']==16000,'Both fixed sourceheads')
 aug=read(checked(fit['augmentation_cache_complete']));old=read(checked(fit['original_training_cache_complete']));require(aug['completed'] and aug['PASS'] and aug['rows']==65536 and old['completed'] and old['PASS'],'Completed sourceTRAIN target caches')
 domains={k:dict(rows=n,targets=(aug if k=='SYNTH_AUG65536' else old['domains'][k])['targets'],metadata=(aug if k=='SYNTH_AUG65536' else old['domains'][k])['metadata'],outputs={a:fit['arms'][a]['outputs'][k] for a in ARMS}) for k,n in DOMAINS.items()}
 return fit,domains
def invented():
 sp=np.array([[0,0],[1,0],[0,1],[1,1],[-1,-1]],np.int8);require(joint(sp).tolist()==[0,1,2,3,4],'Independent nonexclusive BOTH/NONE/U');y=dict(owner_target=np.array([0,1,2,-1,-1],np.int8));c=owner_census(y,np.array([0,1,2,-1,0],np.int8),joint(sp));require(np.array_equal(c['known_owner_vs_support_joint'],np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,1]])),'Knownowner andUNKNOWN retaindifferent jointconditions')
 mass=np.array([[.2,.3,.1,.4,0],[0,1,0,0,0],[np.nan]*5],np.float64);yy=dict(support_target=np.array([[1,1],[0,-1],[-1,-1]],np.int8));z=mass_census(yy,mass,np.array([1,1,0],bool),np.array([0,0,1],bool),np.array([1,1,0],bool),np.array([3,0,4],np.int8));require(np.allclose(z['ALL_ROWS']['raw_total_mass5'],[.2,1.3,.1,.4,0],rtol=0,atol=2e-15) and z['ALL_ROWS']['missing_owner_rows']==1 and z['ORIGINAL_INPUT_U']['rows']==1,'Partialmass no normalization/no UNKNOWN asBG')
 require(sum(z['ALL_ROWS']['support_truth_combinations'][i]['N'] for i in range(9))==3,'Unknown combinations retained');require(metadata_index({'diagnostic_row_index':7},'SYNTH_AUG65536')==metadata_index({'domain_index':7},'CF3816')==7,'Exact declared producer metadata index schemas');return ['Independent supportjoint NONE/V/F/BOTH/U exact','Ownerhard UNKNOWN not dropped or relabeled','Partial rawmass+ambiguity/missing conserved without normalizing','All9 nonexclusive supporttruth combinations exhaustive','Exact AUGdiagnostic_row_index vs originalcache domain_index adapter']
def prepare():
 fit,domains=inventory();checks=invented();write(CHECKS,dict(completed=True,PASS=True,checks=checks,code=bind(__file__),actual_target_saved_prediction_or_source_metadata_row_bodies_read=0))
 rules=dict(completed=True,prepared_only=True,code=bind(__file__),fit_complete=bind(FIT),augmentation_cache_complete=fit['augmentation_cache_complete'],original_training_cache_complete=fit['original_training_cache_complete'],domains=domains,checks=bind(CHECKS),definitions_sources=dict(end_operator=bind(OP),original_cache_producer=bind(OLDPRODUCER),augmentation_producer=bind(AUGPRODUCER),trainer=bind(TRAINER)),
  calculations='All70832 SOURCE TRAIN occurrences kept; IDs/qL/native/availability projected from boundmetadata and exact13key outputs. CForiginalsourceindex repeats preserved, row_id unique. Ownerhard4x4CM inclUNKNOWN truth/U prediction; ownerhard4×supportjoint5 census. Synthetic raw5mass byjoint and all9 supporttruth combinations, full/available/U separately; no hardening/renormalization/imputation.',
  originalU=US,public_hard_owner_counts=[53,52,148],CF_hard_owner_counts=[157,157,0],support_truth_public_CF='All2labels−1 on everyrow: ownerweakdata never source endvisibility/foreign support labels.',
  definitions=dict(end='Fixed selected-end geometric quads/camera projection, classmap registered against target-onlyBEFORE, projected through exact96fullcropbilinear samples. Visible means same targetpart/triangle/depth as BEFORE. Foreignfront means sourceforeign999 depth in front of an otherwise registered target-supported endpatch. Any positive sampled visible/foreignmass gives independent YES; BOTH permitted; self/ambiguity/registration/out-of-frame remainUNKNOWN. No sampled endmass means visibleNO/foreignUNKNOWN, never nativeE.',literal_owner='Separate central2x2 of fixed96 grid at [47:49]^2 with fourtap raster owner rawmass. Synthetic T aggregates anytargetpart, F999, BG, AMBIG/INVALID separate. Public253 assistantweak literal-center owner and CF314 selected-anchor opaque-cover/sham owner-only; not nativevisibility/qualitygold.',training='Five equal mean losses, supportBCE only synthetic; public253/CF314 ownerCE only. Shared trunk may couple representation, but ownerFOREIGN is not a direct foreign-support target. No new model or architecture conclusion.'),
  exclusions='No realbenchmark current/publicheld/VAL outputs, annotations, pixels, RGB, inputNPZ/features, SAM, modelweights/forward/fit. No label/decoder/threshold/architecture/data/model choice. This audit itself admits no new data.',training_admission=False,selected_model=None,adopted=False,fully_solved=False)
 write(RULES,rules);write(PREP,dict(completed=True,PASS=True,code=bind(__file__),rules=bind(RULES),checks=bind(CHECKS),actual_source_saved_output_target_body_reads=0));print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),rules=bind(RULES),preparation=bind(PREP))))
def run():
 prep=read(PREP);require(prep['code']==bind(__file__) and prep['rules']==bind(RULES) and prep['checks']==bind(CHECKS),'Frozen prospective audit rules');p=read(RULES);checked(p['fit_complete']);checked(p['augmentation_cache_complete']);checked(p['original_training_cache_complete']);[checked(b) for b in p['definitions_sources'].values()]
 require(not (H/'REVIEW.json').exists() and not (H/'STARTED_SOURCE_TRAIN_READ.json').exists(),'Single authorized sourceaudit attempt no automaticretry');write(H/'STARTED_SOURCE_TRAIN_READ.json',dict(completed=True,code=bind(__file__),rules=bind(RULES),root_task_authorizes_saved_source_TRAIN_audit_once=True,no_realbenchmark_or_newmodel_access=True))
 used=[];summary={};ledger=H/'ALL70832_SOURCE_TRAIN_TARGET_AND_PREDICTION_JOINTS.jsonl';total=0
 try:
  with ledger.open('x') as handle:
   for domain,record in p['domains'].items():
    n=record['rows'];y=load(record['targets']);pred={a:load(record['outputs'][a]) for a in ARMS};used.extend([record['targets'],record['metadata'],*record['outputs'].values()]);reference=pred[ARMS[0]];joints={a:packet(pred[a],n) for a in ARMS};av=reference['original_query_available'];require(int((~av).sum())==US[domain],'Original per-domain U')
    for a in ARMS:require(all(same(pred[a][k],reference[k]) for k in META),'Bothsourceheads same originalMETA6 byteexact')
    mass,finite,missing=targets(y,n,av)
    if domain!='SYNTH_AUG65536':
     expected=p['public_hard_owner_counts'] if domain=='PUBLIC_TRAIN1480' else p['CF_hard_owner_counts'];require(np.bincount(y['owner_target'][y['owner_target']>=0],minlength=3).tolist()==expected and (y['support_target']==-1).all(),'Public/CF exactweak knowncounts and allsupportUNKNOWN')
    else:require((y['owner_target']==-1).all(),'Syntheticraw owner stayssoft')
    count=0;sourceindices=[]
    with checked(record['metadata']).open() as f:
     for i,line in enumerate(f):
      r=json.loads(line);require(i<n and metadata_index(r,domain)==i and r['row_id']==str(reference['row_id'][i]) and r['original_global_index']==int(reference['original_global_index'][i]) and r['native_slot']==int(reference['native_slot'][i]) and r['original_query_available']==bool(av[i]),'Exact source metadata rowjoin with repeatedCFsourceindices allowed')
      q=np.array([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.nan if r['L'] is None else r['L'];require(eq(q,reference['query_xy'][i]) and eq(np.asarray(L),np.asarray(reference['L'][i])) and r['split'] in ['TRAIN','SOURCE_DEVELOPMENT'],'SourceTRAIN q/L/split unchanged')
      out={k:r[k] for k in ['row_id','original_global_index','native_slot','query_xy','L','original_query_available','split']};out.update(domain=domain,domain_index=i,support_target=y['support_target'][i],owner_target=y['owner_target'][i],owner_raw_mass5=mass[i],predictions={a:dict(support_joint=JOINT[int(joints[a][i])],support_prediction=pred[a]['support_prediction'][i],owner_prediction=int(pred[a]['owner_prediction'][i])) for a in ARMS},training_admission=False)
      handle.write(json.dumps(clean(out),ensure_ascii=False,allow_nan=False)+'\n');count+=1;sourceindices.append(r['original_global_index'])
    require(count==n,'All original rows retained');total+=count
    ds=dict(rows=n,original_inputU=int((~av).sum()),input_available=int(av.sum()),unique_occurrence_ids=n,unique_original_source_indices=len(set(sourceindices)),repeated_original_source_index_occurrences=n-len(set(sourceindices)),owner_hard_counts={str(k):int((y['owner_target']==k).sum()) for k in [-1,0,1,2]},support_target_counts=[{str(k):int((y['support_target'][:,c]==k).sum()) for k in [-1,0,1]} for c in [0,1]],missing_owner_rows=int(missing.sum()),models={})
    for a in ARMS:
     ds['models'][a]=dict(allrows_support_joint_counts=np.bincount(joints[a],minlength=5),owner_census=owner_census(y,pred[a]['owner_prediction'],joints[a]),raw_owner_mass_census=mass_census(y,mass,finite,missing,av,joints[a]))
    summary[domain]=ds
  require(total==70832,'All sourceTRAIN rows once');[checked(b) for b in used];checked(p['fit_complete'])
  result=dict(completed=True,PASS=True,code=bind(__file__),rules=bind(RULES),preparation=bind(PREP),checks=bind(CHECKS),fit_complete=p['fit_complete'],source_bindings_used=used,definitions=p['definitions'],source_line_evidence=dict(end_operator=['1:10 fixedsmallpatch/nonexclusive','76:105 supportedBEFORE plus triangle/depth/foreigndepth','108:164 full96bilinear +unknown/existence'],augmentation_producer=['85:91 literalownerclasses and central4 separate','153:161 targetauthority'],original_cache_producer=['74:80 owner6collapse no normalization','181:210 publicweak253/CF314 onlyowner','93:94 initializedsupportUNKNOWN'],trainer=['127:141 sourceonlysupport pools/publicCFsupportUNKNOWN','155:165 fiveequalmeanlosses']),rows=total,domains=summary,all_rows_qL_native_split_U_and_CF_repeats_preserved=True,ledger=bind(ledger),public_CF_support_truth_all_UNKNOWN=True,new_data_training_admission=False,selected_model=None,adopted=False,fully_solved=False,limitations=['All outputs are apparent SOURCE TRAIN diagnostics, not heldout performance.','OwnerFOREIGN literalcenter and registered-end foreignfront are different statements; do not relabel nativeD/E or infer physicalcornergold.','Publicsupport truth is entirelyUNKNOWN, so jointcounts measure response association only.','Softmass is rawpartialmass with AMBIG/INVALID/missing retained, never normalized or thresholded into ownergold.','Sharedrepresentation coupling is possible but these descriptive counts do not independently establish a mechanism or cause of realbenchmark failure.'],new_forward_fit_decoder_threshold_label_or_model_data_selection=0,realbenchmark_current_publicheld_VAL_RGB_features_weights_reads=0)
  write(H/'REVIEW.json',result);print(json.dumps(dict(completed=True,PASS=True,rows=total,review=bind(H/'REVIEW.json'))))
 except BaseException:
  write(H/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run-source-audit',action='store_true');a=ap.parse_args();prepare() if a.prepare else run()
