"""V183 independent saved numerical audit. Preparation reads code/metadata only.
No producer/evaluator helper, torch/model, image/cache inputs or checkpoint loads.
--complete requires root's all2-frozen + saved-evaluation terminal notification.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
NEW=('C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE');OLD=('C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX')
VAR=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM');SUP=('visible_end','foreign_front');META=('row_id','original_global_index','native_slot','query_xy','L','original_query_available')
AUTHORITY='Original CF/V4 declared semantic association only; physical initializer native identity is uncertified. This is separate from fixed support truth and is not a relabel or success gate.'
def sha(p):
 p=Path(p).resolve();assert not any(x in str(p).lower() for x in ('_v112','_v169','_v177','_v178','current223','private')) and p.suffix not in ('.pt','.pth','.ckpt')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert bind(b['path'])==b,b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def lines(b):return [json.loads(s) for s in check(b).read_text().splitlines() if s]
def load(b,keys=None):
 with np.load(check(b),allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def div(a,b):return None if b==0 else a/b
def categorical(t,p,n):
 cm=[[0]*(n+1) for _ in range(n)];un=[0]*(n+1)
 for a,b in zip(t,p):
  a,b=int(a),int(b);assert -1<=a<n and -1<=b<n;column=n if b<0 else b
  if a<0:un[column]+=1
  else:cm[a][column]+=1
 total=[sum(r) for r in cm]
 r=dict(CM=cm,truth_classes=list(range(n)),prediction_columns=list(range(n))+['U'],rows_all=len(t),truth_known=sum(total),truth_unknown=sum(un),class_support=total,precision=[div(cm[k][k],sum(row[k] for row in cm)) for k in range(n)],recall=[div(cm[k][k],total[k]) for k in range(n)],prediction_U_among_known=sum(row[-1] for row in cm),prediction_counts_among_unknown_truth=un)
 if n==2:r.update(TN=cm[0][0],FP=cm[0][1],FN_without_positive_U=cm[1][0],TP=cm[1][1],U_per_truth=[cm[0][2],cm[1][2]])
 return r

def soft_mass(y,amb,invalid,pred,ii):
 cells=[[[] for _ in range(4)] for _ in range(3)];aa=[];iv=[];missing=0
 for i in ii:
  row=y[i]
  if not all(math.isfinite(float(v)) for v in row):assert all(math.isnan(float(v)) for v in row);missing+=1;continue
  col=3 if pred[i]<0 else int(pred[i]);aa.append(float(amb[i]));iv.append(float(invalid[i]))
  for c in range(3):cells[c][col].append(float(row[c]))
 cm=[[math.fsum(v) for v in row] for row in cells];known=[math.fsum(row) for row in cm]
 return dict(CM_T_F_BG_to_T_F_BG_U=cm,rows_all=len(ii),rows_owner_truth_unobserved_or_missing=missing,raw_known_class_mass=known,ambiguous_mass=math.fsum(aa),invalid_mass=math.fsum(iv),precision=[div(cm[k][k],math.fsum(row[k] for row in cm)) for k in range(3)],recall=[div(cm[k][k],known[k]) for k in range(3)],prediction_U_known_mass=math.fsum(row[-1] for row in cm),no_cutoff_or_row_renormalization=True,not_native_visibility=True)
def joint(row):return -1 if any(int(v)<0 for v in row) else int(row[0])+2*int(row[1])
def proxy_truth(row):return -1 if any(int(v)<0 for v in row) else 1 if int(row[1])==1 else 0 if int(row[0])==1 else -1

def groups_of(rows):
 groups={'ALL':list(range(len(rows)))}
 for k in range(8):groups['native:'+str(k)]=[i for i,r in enumerate(rows) if r['native_slot']==k]
 for key in ('material','aspect','geometry_family','foreign_family','variant','original_semantic_native_role','reference_error_stratum','position_band'):
  buckets=defaultdict(list)
  for i,r in enumerate(rows):
   if key=='variant':value=r['variant']
   elif key in ('material','aspect','geometry_family','foreign_family'):value=r['factors'].get(key,'UNDECLARED')
   else:value=r[key] if r[key] is not None else 'missing'
   buckets[str(value)].append(i)
  for k,ii in buckets.items():groups[key+':'+k]=ii
 return groups

def summary_cell(rows,a,y,ii):
 av=a['original_query_available'];p=a['support_prediction'];op=a['owner_prediction'];proxy=a['regional_support_proxy_D0_E1_Uminus1'];t=y['support_target'];take=[i for i in ii if av[i]]
 r=dict(rows_all=len(ii),original_input_available=len(take),original_inputU=len(ii)-len(take))
 for j,name in enumerate(('visible_end_support','foreign_front_support')):r[name]=dict(primary_known_available=categorical([t[i,j] for i in take],[p[i,j] for i in take],2),secondary_known_all=categorical([t[i,j] for i in ii],[p[i,j] for i in ii],2))
 r['joint4']=categorical([joint(t[i]) for i in ii],[joint(p[i]) for i in ii],4)
 r['owner_soft']=soft_mass(y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass'],op,ii)
 r['owner_hard_weak_only']=categorical([y['owner_target'][i] for i in ii],[op[i] for i in ii],3)
 r['regional_proxy_D0_E1']=categorical([proxy_truth(t[i]) for i in ii],[proxy[i] for i in ii],2)
 roles=('D','E','S','U');cm=[[0]*3 for _ in roles];na=0
 for i in ii:
  role=rows[i]['original_semantic_native_role']
  if role not in roles:na+=1
  else:cm[roles.index(role)][2 if proxy[i]<0 else int(proxy[i])]+=1
 r['original_native_semantic_proxy_association']=dict(declared_semantic_rows=list(roles),regional_proxy_columns=['D','E','U'],counts=cm,rows_no_original_native_truth=na,authority=AUTHORITY)
 r['proxy_truth_NEITHER_or_UNKNOWN_rows']=sum(proxy_truth(t[i])<0 for i in ii);r['source_support_not_physical_native_gold']=True
 return r

def canonical(rawrows,domain,diagnostic,scenes):
 out=[]
 for r in rawrows:
  if domain=='SYNTH_AUG65536':origin=scenes[r['scene_index']]['original_input_metadata8'][r['native_slot']];factors=scenes[r['scene_index']]['factors'];variant=scenes[r['scene_index']]['variant'];sourceid=r['source_image_original_global_index'];assert origin['original_global_index']==sourceid
  elif domain=='COMMON4096':origin=r['source_image_original_metadata'];factors=origin['factors'];variant=r['variant'];sourceid=r['source_image_original_global_index']
  elif domain=='SYNTH_VAL2048':origin=r['original'];factors=origin.get('factors',{});variant=origin.get('variant','UNDECLARED');sourceid=r['original_global_index']
  else:origin=r.get('original',{});factors=origin.get('factors',{});variant=origin.get('variant','UNDECLARED');sourceid=None
  if sourceid is None:role='NOT_AVAILABLE';error='missing';band='missing'
  else:
   d=diagnostic[sourceid];assert d['original_global_index']==sourceid and d['native_slot']==r['native_slot'];role=d['original_semantic_state'];err=d['q_reference_error_px'];error='missing' if err is None or not math.isfinite(err) else '<=3px' if err<=3 else '>3px';band=d['position_band']
  out.append(dict(native_slot=r['native_slot'],factors=factors,variant=variant,original_semantic_native_role=role,reference_error_stratum=error,position_band=band))
 return out

def evaluate(rows,rawrows,a,y,domain):
 g=groups_of(rows)
 if domain=='SYNTH_AUG65536':
  for key in ('family','state_name'):
   for val in {r[key] for r in rawrows}:g[key+':'+val]=[i for i,r in enumerate(rawrows) if r[key]==val]
 return {name:summary_cell(rows,a,y,ii) for name,ii in g.items()}

def delta(values):
 v=np.asarray(values,np.float32);return dict(n=len(values),mean=None if not len(v) else float(v.mean()),median=None if not len(v) else float(np.median(v)),positive_count=sum(x>0 for x in values),zero_count=sum(x==0 for x in values),negative_count=sum(x<0 for x in values))
def pairs(t,a,pairlist):
 result={};p=a['support_prediction'];z=a['raw_support_logits'];av=a['original_query_available']
 for j,name in enumerate(SUP):
  cells={}
  for ta in (-1,0,1):
   for tb in (-1,0,1):
    count=available=correct=0;cm=[[0]*3 for _ in range(3)];dd=[]
    for left,right in pairlist:
     if int(t[left,j])!=ta or int(t[right,j])!=tb:continue
     count+=1;both=bool(av[left] and av[right]);available+=both;pa,pb=int(p[left,j]),int(p[right,j]);cm[2 if pa<0 else pa][2 if pb<0 else pb]+=1;correct+=bool(both and pa==ta and pb==tb)
     if math.isfinite(float(z[left,j])) and math.isfinite(float(z[right,j])):dd.append(float(np.float32(z[right,j]-z[left,j])))
    cells[f'{ta}_to_{tb}']=dict(pairs=count,both_input_available=available,prediction_transition_NO_YES_U_by_NO_YES_U=cm,both_predictions_correct=correct if ta>=0 and tb>=0 else None,right_minus_left_logit=delta(dd))
  result[name]=cells
 canonical=available=correct=0
 for l,r in pairlist:
  if tuple(t[l])==(1,0) and tuple(t[r])==(0,1):canonical+=1;both=bool(av[l] and av[r]);available+=both;correct+=bool(both and np.array_equal(t[l],p[l]) and np.array_equal(t[r],p[r]))
 result['joint_visible_only_to_foreign_only']=dict(pairs=canonical,both_available=available,both_supports_correct_at_both_images=correct,all_other_or_unknown_pairs=len(pairlist)-canonical,not_a_success_gate=True)
 return result

def change(t,old,new,ii):
 cm=[[0]*3 for _ in range(3)];known=unknown=oc=nc=recovered=damaged=both=wrong=0
 for i in ii:
  truth=int(t[i])
  if truth<0:unknown+=1;continue
  known+=1;left=int(old[i]);right=int(new[i]);a=left==truth;b=right==truth;oc+=a;nc+=b;recovered+=not a and b;damaged+=a and not b;both+=a and b;wrong+=not a and not b;cm[2 if left<0 else left][2 if right<0 else right]+=1
 return dict(rows_all=len(ii),truth_known=known,truth_unknown=unknown,old_correct=oc,new_correct=nc,recovered=recovered,damaged=damaged,both_correct=both,both_wrong_or_U=wrong,prediction_transition_NO_YES_U=cm)

def geometry(nt,ct,npred,cpred,nav,cav,fixed):
 eligible=[];fk=[];unknown=changed=U=0
 for i in range(len(nt)):
  bothknown=all(int(x)>=0 for x in nt[i]) and all(int(x)>=0 for x in ct[i]);same=bothknown and np.array_equal(nt[i],ct[i]);valid=bool(nav[i] and cav[i] and same)
  if valid:eligible.append(i)
  if fixed[i]:
   if valid:fk.append(i)
   unknown+=not bothknown;changed+=bothknown and not same;U+=not bool(nav[i] and cav[i])
 def cell(ii):
  oc=[np.array_equal(npred[i],nt[i]) for i in ii];nc=[np.array_equal(cpred[i],ct[i]) for i in ii]
  return dict(rows=len(ii),nominal_correct=sum(oc),common_correct=sum(nc),nominal_correct_became_wrong=sum(a and not b for a,b in zip(oc,nc)),nominal_wrong_became_correct=sum(not a and b for a,b in zip(oc,nc)))
 return dict(BEFORE_rows=len(nt),current_same_known_truth_both_available=cell(eligible),fixed_old401_rows=sum(bool(v) for v in fixed),fixed_old401_current_eligible=cell(fk),fixed_old401_now_unknown_truth=unknown,fixed_old401_now_changed_truth=changed,fixed_old401_now_inputU=U,unknown_or_changed_truth_not_forced_equal=True)

def compare(a,b,path='root'):
 if isinstance(a,dict):
  assert set(a)==set(b),(path,set(a)^set(b))
  for k,v in a.items():compare(v,b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),(path,'length')
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'/{i}')
 else:assert a==b,(path,a,b)

def output(b,rows,n):
 a=load(b);assert set(a)==set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}
 assert len(rows)==n and a['row_id'].shape==(n,) and a['row_id'].dtype.kind=='U';assert len(set(a['row_id'].tolist()))==n;assert a['original_global_index'].dtype==np.int64 and a['native_slot'].dtype==np.int8 and a['query_xy'].dtype==a['L'].dtype==np.float64 and a['original_query_available'].dtype==bool
 assert a['raw_support_logits'].shape==(n,2) and a['raw_owner_logits'].shape==(n,3) and a['raw_support_logits'].dtype==a['raw_owner_logits'].dtype==np.float32
 assert a['support_order'].tolist()==['visible_end_support','foreign_front_support'] and a['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND']
 for i,r in enumerate(rows):
  assert str(a['row_id'][i])==r['row_id'] and int(a['original_global_index'][i])==r['original_global_index'] and int(a['native_slot'][i])==r['native_slot'] and bool(a['original_query_available'][i])==r['original_query_available']
  q=np.array([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);assert np.array_equal(q,a['query_xy'][i],equal_nan=True) and (math.isnan(a['L'][i]) if r['L'] is None else a['L'][i]==r['L'])
  if r['original_query_available']:
   sr=a['raw_support_logits'][i];ow=a['raw_owner_logits'][i];assert np.isfinite(sr).all() and np.isfinite(ow).all();sp=[int(x>0) for x in sr];op=max(range(3),key=lambda c:float(ow[c]));proxy=1 if sp[1] else 0 if sp[0] else -1
  else:assert np.isnan(a['raw_support_logits'][i]).all() and np.isnan(a['raw_owner_logits'][i]).all();sp=[-1,-1];op=proxy=-1
  assert a['support_prediction'][i].tolist()==sp and int(a['owner_prediction'][i])==op and int(a['regional_support_proxy_D0_E1_Uminus1'][i])==proxy
 return a

def invented():
 c={};t=[0,0,1,1,1,-1];p=[0,1,1,0,-1,1];m=categorical(t,p,2);assert m['CM']==[[1,1,0],[1,1,1]] and m['recall'][1]==1/3 and m['truth_unknown']==1;c['CM_P_R_U_unknown']=True
 y=np.array([[.2,.1,0],[0,0,.7],[np.nan]*3]);r=soft_mass(y,np.array([.3,.1,np.nan]),np.array([.4,.2,np.nan]),np.array([0,-1,1]),[0,1,2]);assert r['prediction_U_known_mass']==.7 and r['rows_owner_truth_unobserved_or_missing']==1 and r['raw_known_class_mass']==[.2,.1,.7];c['partial_softmass_and_missing']=True
 assert [joint(x) for x in ([0,0],[1,0],[0,1],[1,1],[-1,1])]==[0,1,2,3,-1];assert [proxy_truth(x) for x in ([0,0],[1,0],[0,1],[1,1],[-1,1])]==[-1,0,1,1,-1];c['joint_nonexclusive_proxy_neither_UNKNOWN']=True
 t=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8);a=dict(support_prediction=t.copy(),raw_support_logits=np.array([[2,-2],[-3,3],[np.nan,np.nan],[np.nan,np.nan]],np.float32),original_query_available=np.array([1,1,0,0],bool));r=pairs(t,a,[(0,1),(2,3)]);assert r['foreign_front']['0_to_1']['right_minus_left_logit']['mean']==5 and r['foreign_front']['-1_to_-1']['both_predictions_correct'] is None;c['paired_signs_and_unknown']=True
 z=change([0,1,1,-1],[0,0,1,1],[-1,1,1,0],[0,1,2,3]);assert z['recovered']==z['damaged']==1 and z['truth_unknown']==1;c['recovery_damage_U']=True
 nt=np.array([[1,0],[1,0],[0,1],[-1,-1]],np.int8);ct=np.array([[1,0],[0,1],[0,1],[1,0]],np.int8);g=geometry(nt,ct,nt,ct,np.ones(4,bool),np.ones(4,bool),[1]*4);assert g['fixed_old401_now_changed_truth']==g['fixed_old401_now_unknown_truth']==1;c['geometry_fixedset_changed_unknown']=True
 assert max(range(3),key=lambda i:[1.,1.,0][i])==0 and int(np.float32(0)>0)==0;c['fixed_zero_argmax_firsttie']=True
 rows=[dict(native_slot=k,factors={'material':'WOOD','aspect':'SQUARE','foreign_family':'TEST'},variant='BEFORE',original_semantic_native_role=role,reference_error_stratum=err,position_band=band) for k,role,err,band in [(0,'D','<=3px','GT_LE3_PX'),(7,'U','missing',None)]]
 a=dict(original_query_available=np.array([1,0],bool),support_prediction=np.array([[1,0],[-1,-1]],np.int8),owner_prediction=np.array([0,-1],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([0,-1],np.int8))
 y=dict(support_target=np.array([[1,0],[-1,-1]],np.int8),owner_soft=np.array([[.4,.1,0],[np.nan]*3],np.float64),owner_ambiguous_mass=np.array([.2,np.nan]),owner_invalid_mass=np.array([.3,np.nan]),owner_target=np.array([0,-1],np.int8))
 z=summary_cell(rows,a,y,[0,1]);assert z['visible_end_support']['primary_known_available']['rows_all']==1 and z['visible_end_support']['secondary_known_all']['truth_unknown']==1 and z['owner_soft']['precision'][0]==.8 and z['original_native_semantic_proxy_association']['counts']==[[1,0,0],[0,0,0],[0,0,0],[0,0,1]];c['full_summary_schema_primary_secondary_partial_owner_native_association']=True
 g=groups_of(rows);assert g['native:1']==[] and 'position_band:missing' in g and 'geometry_family:UNDECLARED' in g;c['all8_native_empty_and_missing_strata_retained']=True
 import tempfile
 with tempfile.TemporaryDirectory(prefix='INVENTED_CF_LINEAGE_',dir=H) as td:
  td=Path(td);toy=dict(row_id=np.array(['a','b']),original_global_index=np.array([7,7],np.int64),native_slot=np.array([0,1],np.int8),query_xy=np.array([[1.,2.],[np.nan,np.nan]],np.float64),L=np.array([4.,np.nan],np.float64),original_query_available=np.array([1,0],bool),raw_support_logits=np.array([[0.,1.],[np.nan,np.nan]],np.float32),raw_owner_logits=np.array([[1.,1.,0.],[np.nan]*3],np.float32),support_prediction=np.array([[0,1],[-1,-1]],np.int8),owner_prediction=np.array([0,-1],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([1,-1],np.int8),support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']))
  rr=[dict(row_id='a',original_global_index=7,native_slot=0,query_xy=[1.,2.],L=4.,original_query_available=True),dict(row_id='b',original_global_index=7,native_slot=1,query_xy=None,L=None,original_query_available=False)]
  np.savez_compressed(td/'shared_origin.npz',**toy);output(bind(td/'shared_origin.npz'),rr,2)
  bad=dict(toy,row_id=np.array(['a','a']));np.savez_compressed(td/'duplicate_row.npz',**bad)
  try:output(bind(td/'duplicate_row.npz'),[rr[0],dict(rr[1],row_id='a')],2)
  except AssertionError:pass
  else:raise AssertionError('duplicate row identity must reject')
 c['actual_output_helper_accepts_shared_origin_rejects_duplicate_occurrence']=True

 return c

def audit(cp):
 cp=Path(cp).resolve();assert cp==V/'saved_jitter_source_evaluation_v2_IDENTITY_GUARD_REPAIR/COMPLETE.json';c=read(cp);assert c['completed'] and c['PASS'] and c['source_only'] and c['model_domain_rows']==166240 and c['selected_model'] is None and not c['adopted'] and not c['fully_solved']
 e=read(check(c['entry']));p=read(check(c['protocol']));assert e['approved_saved_jitter_source_evaluation'] and e['all_two_models_frozen'] and e['source_only'] and e['code']==c['code']==p['code'];check(c['code'])
 fit=read(check(c['fit_complete']));freeze=read(check(c['all2_freeze']));assert fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'] and set(fit['arms'])==set(NEW)
 assert fit['updates_each']==8000 and fit['total_updates']==16000 and freeze['completed'] and freeze['validation_cache_or_targets_read'] is False and freeze['common_diagnostic_inputs_or_targets_read'] is False and freeze['saved_outputs_emitted'] is False
 assert fit['same_draw_sha']==freeze['same_draw_sha'] and fit['same_initial_state_sha']==freeze['same_initial_state_sha']
 for a in NEW:assert fit['arms'][a]['updates']==8000 and fit['arms'][a]['checkpoint']==freeze['checkpoints'][a] and fit['arms'][a]['draw_sha']==fit['same_draw_sha']
 assert fit['protocol']==p['training_protocol'] and fit['code']==p['training_source']
 records=c['target_and_metadata_bindings'];sizes={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'COMMON4096':4096,'SYNTH_VAL2048':2048};assert set(records)==set(sizes)
 rawrows={d:lines(r['metadata']) for d,r in records.items()};diagnostic=lines(p['source_original_annotation_metadata_binding_not_read']);assert len(diagnostic)==6144
 scenes=lines(records['SYNTH_AUG65536']['scene_metadata']);assert len(scenes)==512
 for d,rr in rawrows.items():assert len(rr)==sizes[d] and all(r['split']==('VALIDATION' if d=='SYNTH_VAL2048' else 'SOURCE_DEVELOPMENT' if d in ('PUBLIC_TRAIN1480','CF3816') else 'TRAIN') for r in rr)
 desc=p['descriptors'];oldfit=read(check(desc['old_fit']));oldcommon=read(check(desc['old_common_prediction_complete']));oldcache=read(check(desc['old_training_cache_complete']))
 assert fit['original_training_cache_complete']==desc['old_training_cache_complete'] and fit['evaluation_cache_complete']==desc['original_VAL_cache_complete']
 bindings=c['model_output_bindings'];assert set(bindings)==set(NEW+OLD);packets={}
 for arm,domains in bindings.items():
  assert set(domains)==(set(sizes) if arm in NEW else {'COMMON4096','SYNTH_VAL2048'});packets[arm]={}
  for d,b in domains.items():
   expected=fit['arms'][arm]['outputs'][d] if arm in NEW else oldcommon['arms'][arm]['outputs'] if d=='COMMON4096' else oldfit['arms'][arm]['outputs'][d];assert b==expected
   packets[arm][d]=output(b,rawrows[d],sizes[d])
 for d in ('COMMON4096','SYNTH_VAL2048'):
  for arm in NEW[1:]+OLD:
   for k in META:
    aa,bb=packets[NEW[0]][d][k],packets[arm][d][k];assert aa.shape==bb.shape and (np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind=='f' else aa.tolist()==bb.tolist())
 assert [int((~packets[NEW[0]][d]['original_query_available']).sum()) for d in sizes]==[1264,194,360,96,8]
 # Both fixed heads and every saved response are verified before separate GT access.
 targets={d:load(r['targets']) for d,r in records.items()}
 for d,y in targets.items():
  n=sizes[d];assert set(y)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'} and y['support_target'].dtype==np.int8 and y['support_target'].shape==(n,2) and np.isin(y['support_target'],[-1,0,1]).all() and y['owner_soft'].dtype==np.float64 and y['owner_soft'].shape==(n,3)
  assert (y['support_target'][~packets[NEW[0]][d]['original_query_available']]==-1).all()
 for d,known in (('PUBLIC_TRAIN1480',253),('CF3816',314)):assert int((targets[d]['owner_target']>=0).sum())==known and (targets[d]['support_target']==-1).all()
 rows={d:canonical(rawrows[d],d,diagnostic,scenes) for d in sizes};reported=read(check(c['metrics']));assert reported['completed'] and reported['PASS'] and reported['COMMON_source_TRAIN_derived_not_heldout'] and reported['VAL_historically_reused_not_blind'] and reported['source_associations_not_physical_native_or_coordinate_gold'] and reported['selected_model'] is None
 results={};cells=0
 for arm,domains in packets.items():
  results[arm]={}
  for d,a in domains.items():
   results[arm][d]=evaluate(rows[d],rawrows[d],a,targets[d],d);compare(results[arm][d],reported['results'][arm][d],arm+'/'+d);cells+=len(results[arm][d]);print(json.dumps(dict(checked_arm=arm,domain=d,strata=len(results[arm][d]))),flush=True)
 common_index={(r['base_index'],r['variant'],r['native_slot']):i for i,r in enumerate(rawrows['COMMON4096'])};assert len(common_index)==4096
 commonpairs={v:[(common_index[b,'BEFORE',k],common_index[b,v,k]) for b in range(128) for k in range(8)] for v in VAR[1:]};paired={}
 for arm in NEW+OLD:
  paired[arm]={}
  for v,ii in commonpairs.items():paired[arm]['BEFORE_to_'+v]=pairs(targets['COMMON4096']['support_target'],packets[arm]['COMMON4096'],ii)
  assert paired[arm]['BEFORE_to_COVER']['joint_visible_only_to_foreign_only']['pairs']==95
 compare(paired,reported['COMMON_matched_pairs'])
 changes={}
 for d in ('COMMON4096','SYNTH_VAL2048'):
  changes[d]={}
  for old,new in zip(OLD,NEW):
   changes[d][old+' -> '+new]={}
   for name,ii in groups_of(rows[d]).items():changes[d][old+' -> '+new][name]={s:change(targets[d]['support_target'][:,j],packets[old][d]['support_prediction'][:,j],packets[new][d]['support_prediction'][:,j],ii) for j,s in enumerate(('visible_end_support','foreign_front_support'))}
 compare(changes,reported['corresponding_model_recovery_damage'])
 oldrec=oldcache['domains']['SYNTH_TRAIN4096'];oldrows=lines(oldrec['metadata']);oldy=load(oldrec['targets']);oldpack={a:output(oldfit['arms'][a]['outputs']['SYNTH_TRAIN4096'],oldrows,4096) for a in OLD}
 before_ids=[common_index[b,'BEFORE',k] for b in range(128) for k in range(8)];original_ids=[rawrows['COMMON4096'][i]['source_image_original_global_index'] for i in before_ids];ct=targets['COMMON4096']['support_target'][before_ids];ot=oldy['support_target'][original_ids];oldav=oldpack[OLD[0]]['original_query_available'][original_ids];cav=packets[NEW[0]]['COMMON4096']['original_query_available'][before_ids]
 fixed=[bool(oldav[i] and cav[i] and all(int(x)>=0 for x in ot[i]) and all(int(x)>=0 for x in ct[i]) and np.array_equal(ot[i],ct[i])) for i in range(1024)];assert sum(fixed)==401
 nominal_map={r['source_image_original_global_index']:i for i,r in enumerate(rawrows['SYNTH_AUG65536']) if r['family']=='SELF_SCENE15' and r['state_index']==0};assert len(nominal_map)==4096
 fragility={}
 for arm in NEW+OLD:
  if arm in NEW:
   ids=[nominal_map[k] for k in original_ids];a=packets[arm]['SYNTH_AUG65536'];nt=targets['SYNTH_AUG65536']['support_target'][ids];nav=a['original_query_available'][ids];npred=a['support_prediction'][ids]
  else:a=oldpack[arm];nt=ot;nav=a['original_query_available'][original_ids];npred=a['support_prediction'][original_ids]
  fragility[arm]=geometry(nt,ct,npred,packets[arm]['COMMON4096']['support_prediction'][before_ids],nav,cav,fixed)
 compare(fragility,reported['original_fixed401_geometry_sensitivity'])
 # Historical6144-container read is limited to the saved8 fields; evaluate only VAL2048.
 v61=load(desc['historical_v61_output_binding_not_opened'],('global_index','occurrence_id','native_slot','query_xy','L','query_available','prediction_D0_E1_Uminus1','raw_visible_logit'));ref=packets[NEW[0]]['SYNTH_VAL2048'];ids=ref['original_global_index'];assert ids.tolist()==list(range(4096,6144))
 for sk,dk in (('global_index','original_global_index'),('occurrence_id','row_id'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L')):
  aa,bb=v61[sk][ids],ref[dk];assert aa.shape==bb.shape and (np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind=='f' else aa.tolist()==bb.tolist())
 av=v61['query_available'][ids];pred=v61['prediction_D0_E1_Uminus1'][ids];raw=v61['raw_visible_logit'][ids];assert av.dtype==bool and raw.dtype==np.float32 and np.isfinite(raw[av]).all();sp=np.full((2048,2),-1,np.int8)
 for i in range(2048):
  expected=int(raw[i]<0) if av[i] else -1;assert int(pred[i])==expected
  if av[i]:sp[i]=[int(expected==0),int(expected==1)]
 hist_a=dict(original_query_available=av,support_prediction=sp,owner_prediction=np.full(2048,-1,np.int8),regional_support_proxy_D0_E1_Uminus1=pred);hist={}
 for name,ii in groups_of(rows['SYNTH_VAL2048']).items():
  z=summary_cell(rows['SYNTH_VAL2048'],hist_a,targets['SYNTH_VAL2048'],ii);z['historical_v61_available']=z.pop('original_input_available');z['historical_v61_U']=z.pop('original_inputU');z['original_source_inputU']=sum(not ref['original_query_available'][i] for i in ii);del z['owner_soft'];del z['owner_hard_weak_only'];z['owner3_output_not_defined']=True;hist[name]=z
 compare(hist,reported['historical_v61_VAL_task_mismatch_diagnostic']['strata'])
 # Stream every166240 saved row; no enormous ledger list/duplicate raw input cache.
 count=Counter();total=0
 with check(c['rows']).open() as f:
  for line in f:
   l=json.loads(line);arm,d,i=l['arm'],l['domain'],l['row_index'];assert i==count[arm,d];a=packets[arm][d];y=targets[d];r=rawrows[d][i]
   expected=dict(arm=arm,domain=d,row_index=i,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],original_query_available=r['original_query_available'],support_truth=y['support_target'][i].tolist(),support_prediction=a['support_prediction'][i].tolist(),owner_weak_hard_truth=int(y['owner_target'][i]),owner_prediction=int(a['owner_prediction'][i]),regional_proxy=int(a['regional_support_proxy_D0_E1_Uminus1'][i]));compare(expected,l);count[arm,d]+=1;total+=1
 assert total==166240 and all(count[a,d]==len(rawrows[d]) for a,domains in packets.items() for d in domains)
 for domains in bindings.values():
  for b in domains.values():check(b)
 for r in records.values():check(r['targets']);check(r['metadata'])
 return dict(completed=True,PASS=True,code=bind(__file__),evaluation_complete=bind(cp),root_entry=c['entry'],fit_complete=c['fit_complete'],all2_freeze=c['all2_freeze'],metrics=c['metrics'],ledger=c['rows'],all_model_domain_metrics_strata_exact=True,total_model_domain_strata_verified=cells,COMMON_all4heads_all9truthcells_and95canonical_pairs_exact=True,corresponding_recovery_damage_allstrata_exact=True,fixed_old401_and_new_changed_UNKNOWN_geometry_cells_exact=True,historical_v61_VAL_ownU_decoder_task_mismatch_exact=True,ledger166240_rows_exact=True,all_metadata_decoders_and_fixed_U1264_194_360_96_8=True,independent_method='Separate scalar categorical/softmass/joint/proxy/association counts and explicit ID pair joins, no producer/evaluator import. math.fsum preserves raw partial owner mass; FP32 pair differences/means remain fixed. Historical6144 saved field container opened, only originalVAL2048 contributes to v61 metrics.',actual_raw_pixel_feature_weight_model_body_or_new_forward_fit=0,limitations='TRAIN apparent, COMMON sourceTRAIN-derived and historically reused syntheticVAL are separate. Native association and original-q reference bands are not new native/physical coordinate truth. No independent natural-real performance/selection/adoption claim.',selected_model=None,adopted=False,fully_solved=False)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--complete',type=Path);a=ap.parse_args()
 if a.prepare:
  assert a.complete is None;tests=invented();out=H/'PREPARATION_COMPLETE.json';r=dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),invented_checks=tests,actual_new_output_VAL_COMMON_target_bodies_read=0,requires_root_all2freeze_and_saved_eval_terminal=True)
 else:assert a.complete is not None;out=H/'REVIEW.json';r=audit(a.complete)
 put(out,r);print(json.dumps(dict(PASS=True,receipt=bind(out))))
if __name__=='__main__':main()
