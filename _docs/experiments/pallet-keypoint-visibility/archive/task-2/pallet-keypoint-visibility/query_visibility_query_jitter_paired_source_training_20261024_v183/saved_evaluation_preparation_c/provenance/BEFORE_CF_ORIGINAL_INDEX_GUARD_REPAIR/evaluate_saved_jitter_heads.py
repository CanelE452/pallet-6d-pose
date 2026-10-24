"""V183 saved-only all-final source evaluation. Prepare uses invented data only.

No model, image, pixel cache, encoder or checkpoint is loaded. Synthetic VAL is
historically exposed; COMMON is source-TRAIN derived, not heldout performance.
"""
from pathlib import Path
import argparse,ast,hashlib,json,math,sys,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
OLD=P/'query_visibility_local_support_raw_head_20261024_v181'
COMMON=P/'query_visibility_source_same_query_causal_diagnosis_20261024_v182'
SCOPE=V/'ROOT_SCOPE_BEFORE_NEW_AUGMENTATION_OR_FIT.json'
SCOPE_SHA='d8394deb1c0a5844ecb987447391d7fd61ce5aa80d073cf33adf76ff385fcb95'
PURE_SOURCE=OLD/'evaluation_preparation_b/evaluate_saved_local_support.py'
PAIR_SOURCE=COMMON/'evaluate_saved_common_query.py'
TRAIN_SOURCE=V/'source_fit_preparation/train_jitter_paired_source.py'
TRAIN_PROTOCOL=V/'source_fit_preparation/PROTOCOL_BEFORE_AUGMENTED_CACHE_OR_FIT_BODIES.json'
OLD_FIT=OLD/'runs/C50_C51_LOCAL_SUPPORT_FIXED8000_v2/COMPLETE.json'
OLD_COMMON=COMMON/'frozen_common_query_heads_v1/COMPLETE.json'
OLD_CACHE=OLD/'source_raw96_cache_v1/COMPLETE.json'
VAL_CACHE=OLD/'synthetic_VAL2048_raw96_cache_v1/COMPLETE.json'
COMMON_CACHE=COMMON/'common_query_diagnostic_v1/COMPLETE.json'
V61_SEAL=P/'query_visibility_actual_initializer_source_20261024_v132/frozen_v61_actual_initializer_v1/PREDICTION_SEAL.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_OUTPUT_OR_EVALUATION_TARGET_BODIES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json';OUT=V/'saved_jitter_source_evaluation_v1'
NEW_ARMS=['C52_RAW96_JITTER_PAIRED_SOURCE','C53_RAW96_SAM_JITTER_PAIRED_SOURCE']
OLD_ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
OUTPUT_KEYS=set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'}
TARGET_KEYS={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'}
VARIANTS=('BEFORE','COVER','NEAR_SHAM','FAR_SHAM')


def require(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(path):
 path=Path(path).resolve();require(path.is_relative_to(P) and not any(s in str(path).lower() for s in ['_v112','_v169','_v177','_v178','current223','private_evaluation']),'Forbidden path')
 h=hashlib.sha256()
 with path.open('rb') as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def checked(b):require(bind(b['path'])==b,'Exact binding '+b['path']);return Path(b['path'])
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load(b,keys=None):
 with np.load(checked(b),allow_pickle=False) as z:return {k:z[k] for k in (z.files if keys is None else keys)}
def same(a,b):
 return a.dtype==b.dtype and a.shape==b.shape and (np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b))


def pure_helpers():
 # Compile only explicitly enumerated pure metric functions; never import old
 # evaluator modules, their run functions, datasets or runtime producers.
 names={'ratio','cm_metric','soft_owner_metric','decode','joint','truth_proxy','source_native_association','strata','summary','invented'}
 ns=dict(np=np,math=math,require=require)
 tree=ast.parse(PURE_SOURCE.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 require({n.name for n in selected}==names,'Exact immutable pure helpers')
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(PURE_SOURCE),'exec'),ns)
 names2={'fraction','metrics','pair_summary','self_test'};tree=ast.parse(PAIR_SOURCE.read_text())
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names2]
 require({n.name for n in selected}==names2,'Exact immutable pair helpers')
 ns.update(SUPPORT=('visible_end','foreign_front'))
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(PAIR_SOURCE),'exec'),ns)
 tree=ast.parse(TRAIN_SOURCE.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_targets']
 require(len(nodes)==1,'Exact separate-target guard')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(TRAIN_SOURCE),'exec'),ns)
 return ns


def validate_identity(a,rows,n):
 require(set(a)==OUTPUT_KEYS,'Exact saved output schema, no input pixels/features/targets')
 specs={'original_global_index':(np.int64,(n,)),'native_slot':(np.int8,(n,)),'query_xy':(np.float64,(n,2)),'L':(np.float64,(n,)),'original_query_available':(np.bool_,(n,))}
 require(len(rows)==n and a['row_id'].dtype.kind=='U' and a['row_id'].shape==(n,),'All rows and Unicode identities')
 require(len(set(a['row_id'].tolist()))==n and len(set(int(x) for x in a['original_global_index']))==n,'Unique original/model domain identities')
 for key,(dt,shape) in specs.items():require(a[key].dtype==dt and a[key].shape==shape,'Exact metadata schema '+key)
 for i,r in enumerate(rows):
  require(str(a['row_id'][i])==r['row_id'] and int(a['original_global_index'][i])==r['original_global_index'] and int(a['native_slot'][i])==r['native_slot'] and bool(a['original_query_available'][i])==r['original_query_available'],'Exact row identity/availability')
  q=np.array([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64);L=np.nan if r['L'] is None else r['L']
  require(np.array_equal(a['query_xy'][i],q,equal_nan=True) and (a['L'][i]==L or np.isnan(a['L'][i]) and np.isnan(L)),'Exact q/L metadata join')
 require(a['support_order'].tolist()==['visible_end_support','foreign_front_support'] and a['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'Independent support/owner task order')


def checked_output(binding,rows,n,helpers):
 a=load(binding);validate_identity(a,rows,n)
 sp,op,proxy=helpers['decode'](a['raw_support_logits'],a['raw_owner_logits'],a['original_query_available'])
 for key,value in [('support_prediction',sp),('owner_prediction',op),('regional_support_proxy_D0_E1_Uminus1',proxy)]:require(same(a[key],value),'Unchanged saved decoder '+key)
 return a


def canonical_rows(rows,domain,diagnostic,scenes=None):
 result=[]
 for r in rows:
  c=dict(r)
  if domain=='SYNTH_AUG65536':
   group=scenes[r['scene_index']];origin=group['original_input_metadata8'][r['native_slot']]
   require(origin['original_global_index']==r['source_image_original_global_index'],'Compact source image/native mapping')
   source_id=r['source_image_original_global_index'];factors=group['factors'];variant=group['variant']
  elif domain=='COMMON4096':
   source_id=r['source_image_original_global_index'];origin=r['source_image_original_metadata'];factors=origin['factors'];variant=r['variant']
  elif domain=='SYNTH_VAL2048':
   source_id=r['original_global_index'];origin=r['original'];factors=origin.get('factors',{});variant=origin.get('variant','UNDECLARED')
  else:source_id=None;origin=r.get('original',{});factors=origin.get('factors',{});variant=origin.get('variant','UNDECLARED')
  c['original']=dict(factors=factors,variant=variant)
  if source_id is not None:
   d=diagnostic[source_id];require(d['original_global_index']==source_id and d['native_slot']==r['native_slot'],'Original source association IDs/native')
   c['original_semantic_native_role']=d['original_semantic_state'];error=d['q_reference_error_px']
   c['reference_error_stratum']='missing' if error is None or not math.isfinite(error) else '<=3px' if error<=3 else '>3px'
   c['position_band']=d['position_band']
  else:c.update(original_semantic_native_role='NOT_AVAILABLE',reference_error_stratum='missing',position_band='missing')
  result.append(c)
 return result


def evaluate(rows,a,y,helpers):
 out=helpers['summary'](rows,a['original_query_available'],y['support_target'],y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass'],y['owner_target'],a['support_prediction'],a['owner_prediction'],a['regional_support_proxy_D0_E1_Uminus1'])
 if rows and 'family' in rows[0]:
  for field in ['family','state_name']:
   for value in sorted({r[field] for r in rows}):
    ids=np.array([i for i,r in enumerate(rows) if r[field]==value],np.int64)
    ar={k:(v[ids] if v.shape and len(v)==len(rows) else v) for k,v in a.items()}
    yr={k:v[ids] for k,v in y.items()}
    out[field+':'+value]=helpers['summary']([rows[int(i)] for i in ids],ar['original_query_available'],yr['support_target'],yr['owner_soft'],yr['owner_ambiguous_mass'],yr['owner_invalid_mass'],yr['owner_target'],ar['support_prediction'],ar['owner_prediction'],ar['regional_support_proxy_D0_E1_Uminus1'])['ALL']
 return out


def paired_change(truth,left,right):
 require(truth.shape==left.shape==right.shape,'Fixed same-row truth/prediction shapes')
 known=truth>=0;lc=left==truth;rc=right==truth
 trans=np.zeros((3,3),np.int64)
 for a,b in zip(left[known],right[known]):trans[2 if a<0 else a,2 if b<0 else b]+=1
 return dict(rows_all=len(truth),truth_known=int(known.sum()),truth_unknown=int((~known).sum()),old_correct=int((known&lc).sum()),new_correct=int((known&rc).sum()),recovered=int((known&~lc&rc).sum()),damaged=int((known&lc&~rc).sum()),both_correct=int((known&lc&rc).sum()),both_wrong_or_U=int((known&~lc&~rc).sum()),prediction_transition_NO_YES_U=trans.tolist())


def geometry_comparison(nominal_truth,common_truth,nominal_pred,common_pred,nominal_av,common_av,fixed_old_set):
 require(nominal_truth.shape==common_truth.shape==nominal_pred.shape==common_pred.shape,'Same-image support shape')
 both_known=(nominal_truth>=0).all(1)&(common_truth>=0).all(1)
 same_truth=both_known&(nominal_truth==common_truth).all(1)
 valid=nominal_av&common_av&same_truth
 oc=(nominal_pred==nominal_truth).all(1);cc=(common_pred==common_truth).all(1)
 def counts(mask):
  return dict(rows=int(mask.sum()),nominal_correct=int((mask&oc).sum()),common_correct=int((mask&cc).sum()),nominal_correct_became_wrong=int((mask&oc&~cc).sum()),nominal_wrong_became_correct=int((mask&~oc&cc).sum()))
 return dict(BEFORE_rows=len(valid),current_same_known_truth_both_available=counts(valid),fixed_old401_rows=int(fixed_old_set.sum()),fixed_old401_current_eligible=counts(valid&fixed_old_set),fixed_old401_now_unknown_truth=int((fixed_old_set&~both_known).sum()),fixed_old401_now_changed_truth=int((fixed_old_set&both_known&~same_truth).sum()),fixed_old401_now_inputU=int((fixed_old_set&~(nominal_av&common_av)).sum()),unknown_or_changed_truth_not_forced_equal=True)


def historical_v61_VAL(b,reference,y,rows,helpers):
 keys=['global_index','occurrence_id','native_slot','query_xy','L','query_available','prediction_D0_E1_Uminus1','raw_visible_logit']
 a=load(b,keys);require(len(a['global_index'])==6144 and a['raw_visible_logit'].dtype==np.float32,'Historical original6144 v61')
 ids=reference['original_global_index'];require(np.array_equal(ids,np.arange(4096,6144,dtype=np.int64)),'Original VAL2048 IDs only')
 for src,dst in [('global_index','original_global_index'),('occurrence_id','row_id'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L')]:
  value=a[src][ids];expected=reference[dst]
  require(value.shape==expected.shape and (np.array_equal(value,expected,equal_nan=True) if value.dtype.kind in 'fc' else np.array_equal(value,expected)),'v61 same original VAL q/L/native/ID')
 available=a['query_available'][ids];raw=a['raw_visible_logit'][ids];pred=a['prediction_D0_E1_Uminus1'][ids]
 require(available.dtype==bool and np.isfinite(raw[available]).all(),'Historical own available logits')
 expected=np.full(2048,-1,np.int8);expected[available]=(raw[available]<0).astype(np.int8);require(np.array_equal(pred,expected),'Original v61 >=0D/<0E/Ufirst')
 support=np.full((2048,2),-1,np.int8);support[available,0]=(pred[available]==0).astype(np.int8);support[available,1]=(pred[available]==1).astype(np.int8)
 out=helpers['summary'](rows,available,y['support_target'],y['owner_soft'],y['owner_ambiguous_mass'],y['owner_invalid_mass'],y['owner_target'],support,np.full(2048,-1,np.int8),pred)
 for name,ii in helpers['strata'](rows).items():
  z=out[name];z['historical_v61_available']=z.pop('original_input_available');z['historical_v61_U']=z.pop('original_inputU');z['original_source_inputU']=int((~reference['original_query_available'][ii]).sum())
  del z['owner_soft'];del z['owner_hard_weak_only'];z['owner3_output_not_defined']=True
 return dict(meaning='Historical original-q pixel-presence sign against regional fixed-support targets only. Own availability; complementary binary output cannot represent BOTH. No native/external/self/coordinate gold or owner3 claim.',strata=out)


def validate_fit_closure(fit,freeze):
 require(fit.get('completed') is True and fit.get('PASS') is True and fit.get('all2_frozen_before_outputs_or_evaluation_access') is True and set(fit['arms'])==set(NEW_ARMS),'Both final frozen new heads')
 require(fit.get('updates_each')==8000 and fit.get('total_updates')==16000,'Fixed final updates')
 require(freeze.get('completed') is True and freeze.get('validation_cache_or_targets_read') is False and freeze.get('common_diagnostic_inputs_or_targets_read') is False and freeze.get('saved_outputs_emitted') is False,'Barrier before any evaluation body/output')
 require(fit['same_draw_sha']==freeze['same_draw_sha'] and fit['same_initial_state_sha']==freeze['same_initial_state_sha'],'Same fixed initialization and draws')
 for arm in NEW_ARMS:require(fit['arms'][arm]['updates']==8000 and fit['arms'][arm]['checkpoint']==freeze['checkpoints'][arm] and fit['arms'][arm]['draw_sha']==fit['same_draw_sha'],'All final8000/checkpoint/draw bindings')


def invented(helpers):
 tests=helpers['invented']();helpers['self_test']()
 def passed(name,value):require(value,name);tests.append(dict(name=name,PASS=True))
 def rejects(fn):
  try:fn()
  except (AssertionError,ValueError,KeyError):return True
  return False
 t=np.array([0,1,1,-1],np.int8);old=np.array([0,0,1,1],np.int8);new=np.array([-1,1,1,0],np.int8);p=paired_change(t,old,new)
 passed('Same-row recovery damage includes predictionU as error; unknown truth unscored',p['recovered']==1 and p['damaged']==1 and p['truth_unknown']==1 and p['both_correct']==1)
 nt=np.array([[1,0],[1,0],[0,1],[-1,-1]],np.int8);ct=np.array([[1,0],[0,1],[0,1],[1,0]],np.int8);av=np.ones(4,bool)
 g=geometry_comparison(nt,ct,nt,ct,av,av,np.ones(4,bool))
 passed('Fixed historical set preserves changed and unknown targets without forced equality',g['current_same_known_truth_both_available']['rows']==2 and g['fixed_old401_now_changed_truth']==1 and g['fixed_old401_now_unknown_truth']==1)
 base=dict(row_id=np.array(['r0','r1']),original_global_index=np.array([0,1],np.int64),native_slot=np.array([0,1],np.int8),query_xy=np.array([[1.,2.],[np.nan,np.nan]],np.float64),L=np.array([4.,np.nan],np.float64),original_query_available=np.array([1,0],bool),raw_support_logits=np.array([[0.,1.],[np.nan,np.nan]],np.float32),raw_owner_logits=np.array([[1.,1.,0.],[np.nan]*3],np.float32),support_prediction=np.array([[0,1],[-1,-1]],np.int8),owner_prediction=np.array([0,-1],np.int8),regional_support_proxy_D0_E1_Uminus1=np.array([1,-1],np.int8),support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']))
 rows=[dict(row_id='r0',original_global_index=0,native_slot=0,query_xy=[1.,2.],L=4.,original_query_available=True),dict(row_id='r1',original_global_index=1,native_slot=1,query_xy=None,L=None,original_query_available=False)]
 validate_identity(base,rows,2)
 passed('Exact meta6 join accepts missing q/L and rejects swapped order',rejects(lambda:validate_identity(base,list(reversed(rows)),2)))
 wrong=dict(base);wrong['query_xy']=base['query_xy'].copy();wrong['query_xy'][0,0]+=.25
 passed('Exact q drift is rejected before scoring',rejects(lambda:validate_identity(wrong,rows,2)))
 # Explicit pair test includes all9 truth transition cells and UNKNOWN, not only
 # canonical successful cover cases.
 tt=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8);pp=np.array([[1,0],[0,1],[-1,-1],[-1,-1]],np.int8);zz=np.array([[1.,-1.],[-1.,1.],[np.nan,np.nan],[np.nan,np.nan]],np.float32)
 pair=helpers['pair_summary'](tt,pp,zz,np.array([0,2]),np.array([1,3]),np.array([1,1,0,0],bool))
 passed('Matched pair unknown cells preserved separately from known canonical support',sum(z['pairs'] for z in pair['foreign_front'].values())==2 and pair['foreign_front']['-1_to_-1']['pairs']==1)
 freeze=dict(completed=True,validation_cache_or_targets_read=False,common_diagnostic_inputs_or_targets_read=False,saved_outputs_emitted=False,same_draw_sha='invented-draw',same_initial_state_sha='invented-init',checkpoints={a:{'path':'invented/'+a,'sha256':'invented'} for a in NEW_ARMS})
 fit=dict(completed=True,PASS=True,all2_frozen_before_outputs_or_evaluation_access=True,updates_each=8000,total_updates=16000,same_draw_sha='invented-draw',same_initial_state_sha='invented-init',arms={a:dict(updates=8000,checkpoint=freeze['checkpoints'][a],draw_sha='invented-draw') for a in NEW_ARMS})
 validate_fit_closure(fit,freeze)
 passed('Actual closure guard accepts fixed8000 and rejects prefreeze evaluation or wrong updates',rejects(lambda:validate_fit_closure(fit,dict(freeze,common_diagnostic_inputs_or_targets_read=True))) and rejects(lambda:validate_fit_closure(dict(fit,updates_each=7999),freeze)))
 return tests


def prepare():
 require(sha(SCOPE)==SCOPE_SHA,'Frozen source scope')
 helpers=pure_helpers();tests=invented(helpers)
 # Small completion metadata only: never open referenced NPZ/target bodies here.
 oldfit=read(OLD_FIT);oldcommon=read(OLD_COMMON);oldcache=read(OLD_CACHE);common=read(COMMON_CACHE);v61=read(V61_SEAL)
 require(all(x['completed'] and x['PASS'] for x in [oldfit,oldcommon,oldcache,common,v61]),'Historical completion metadata')
 require(set(oldfit['arms'])==set(oldcommon['arms'])==set(OLD_ARMS),'Both historical fixed arms')
 descriptors=dict(old_fit=bind(OLD_FIT),old_common_prediction_complete=bind(OLD_COMMON),old_training_cache_complete=bind(OLD_CACHE),original_VAL_cache_complete=bind(VAL_CACHE),COMMON_raw_complete=bind(COMMON_CACHE),historical_v61_seal=bind(V61_SEAL),historical_v61_output_binding_not_opened=v61['predictions'])
 source=read(OLD/'evaluation_preparation_b/PROTOCOL_BEFORE_NEW_SOURCE_OUTPUT_OR_VAL_TARGET_BODIES.json')['source_diagnostic_metadata_inherited_binding_not_read_or_hashed']
 write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),tests=tests,actual_source_output_target_pixel_feature_or_weight_bodies_read=0))
 protocol=dict(completed=True,prepared_only=True,scope=bind(SCOPE),code=bind(__file__),checks=bind(CHECKS),pure_metric_source=bind(PURE_SOURCE),pure_pair_source=bind(PAIR_SOURCE),training_source=bind(TRAIN_SOURCE),training_protocol=bind(TRAIN_PROTOCOL),descriptors=descriptors,source_original_annotation_metadata_binding_not_read=source,
 new_arms=NEW_ARMS,old_arms=OLD_ARMS,output=str(OUT),
 domains='New2 apparent SYNTH_AUG65536/publicTRAIN1480/CF3816 separately; all4 identical COMMON4096 and original SYNTH_VAL2048.',
 denominators='Per support primary knowntruth & originalinputavailable; secondary allknowntruth including U. Unknowntruth prediction census, joint4 nonexclusive, owner raw partial mass + ambiguity/invalid/missing and weakhard owner separate.',
 pairs='Each COMMON base/native BEFORE vs COVER/NEAR_SHAM/FAR_SHAM, all1024pairs percomparison and all9 support truth transitions. Corresponding oldRAW->newRAW and oldSAM->newSAM same-row recovery/damage.',
 geometry='Recompute fixed old401 sameknownsupport/bothavailable BEFORE set from originalsource and COMMONtargets; newownnominal indexoriginalglobal*15. Changed/unknown newtarget remains separate, never forced identical. q and L, and SAMprompt can differ; not isolatedqcausality.',
 strata='ALL/material/aspect/geometry/foreignfamily/variant/native0..7; augmented family/state. Original source declarednative/self and original reference/position associations only, not changed-query pixelerror gold.',
 historical_v61='VAL portion of original6144 only; exactID/native/qL join and ownU/sign rule. Separate taskmismatch regional diagnostic; no owner3 or native physical gold.',
 evaluation_contract='Require newbothfinal8000 COMPLETE and freeze receipt before any savedprediction/targetbody. No model import/forward/checkpoint/image/inputpixelcache/GTbuffer. No threshold/model/arm/data selection; trainapparent and historicallyreusedVAL not independent natural-real performance.',
 selected_model=None,adopted=False,fully_solved=False,new_training_or_threshold_or_label_changes=0)
 write(PROTOCOL,protocol)
 write(H/'ROOT_ACTUAL_SAVED_EVALUATION_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_saved_jitter_source_evaluation=False,source_only=True,all_two_models_frozen=False,scope=bind(SCOPE),code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=None))
 write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),actual_output_or_target_bodies_read=0,model_forward_fit=0))
 print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))


def run(entry_path):
 e=read(entry_path);require(e.get('approved_saved_jitter_source_evaluation') is True and e.get('source_only') is True and e.get('all_two_models_frozen') is True,'Root saved-only all2freeze entry')
 for key,path in [('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL)]:require(e[key]==bind(path),'Frozen entry '+key)
 p=read(PROTOCOL);require(sha(SCOPE)==SCOPE_SHA and p['checks']==bind(CHECKS),'Frozen rules/checks')
 for key,path in [('pure_metric_source',PURE_SOURCE),('pure_pair_source',PAIR_SOURCE),('training_source',TRAIN_SOURCE),('training_protocol',TRAIN_PROTOCOL)]:require(p[key]==bind(path),'Frozen pure/prepared source '+key)
 fit=read(checked(e['fit_complete']));require(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_evaluation_access'] and set(fit['arms'])==set(NEW_ARMS),'Both final frozen8000 new heads')
 require(fit['protocol']==p['training_protocol'] and fit['code']==p['training_source'] and fit['updates_each']==8000 and fit['total_updates']==16000,'Exact frozen training protocol/final updates')
 freeze=read(checked(fit['all2_freeze']));require(freeze['completed'] and not freeze['validation_cache_or_targets_read'] and not freeze['common_diagnostic_inputs_or_targets_read'] and not freeze['saved_outputs_emitted'],'Freeze before evaluation input/target/output access')
 validate_fit_closure(fit,freeze)
 descriptors=p['descriptors'];oldfit=read(checked(descriptors['old_fit']));oldcommon=read(checked(descriptors['old_common_prediction_complete']));oldcache=read(checked(descriptors['old_training_cache_complete']));common=read(checked(descriptors['COMMON_raw_complete']));val=read(checked(descriptors['original_VAL_cache_complete']))
 require(fit['original_training_cache_complete']==descriptors['old_training_cache_complete'] and fit['evaluation_cache_complete']==descriptors['original_VAL_cache_complete'],'Unchanged public/CF and VAL authority')
 descriptor=read(checked(fit['common_evaluation_descriptor']));require(descriptor['raw_complete']==descriptors['COMMON_raw_complete'] and descriptor['targets5']==common['targets'] and descriptor['metadata']==common['metadata'],'Same COMMON authority')
 aug=read(checked(fit['augmentation_cache_complete']));require(aug['rows']==65536 and aug['original_aug_inputU']==1264,'All augmentation rows/U')
 helpers=pure_helpers();require(not OUT.exists(),'Preserve all attempts; no overwrite/retry');OUT.mkdir()
 try:
  diagnostic=lines(checked(p['source_original_annotation_metadata_binding_not_read']));require(len(diagnostic)==6144 and [r['original_global_index'] for r in diagnostic]==list(range(6144)),'Original source association metadata only')
  scenes=lines(checked(aug['scene_metadata']));require(len(scenes)==512,'Compact original TRAIN512 metadata')
  records={'SYNTH_AUG65536':aug,'PUBLIC_TRAIN1480':oldcache['domains']['PUBLIC_TRAIN1480'],'CF3816':oldcache['domains']['CF3816'],'COMMON4096':common,'SYNTH_VAL2048':val['domains']['SYNTH_VAL2048']}
  rawrows={d:lines(checked(r['metadata'])) for d,r in records.items()}
  for d,rr in rawrows.items():require(all(r['split']==('VALIDATION' if d=='SYNTH_VAL2048' else 'SOURCE_DEVELOPMENT' if d in ['PUBLIC_TRAIN1480','CF3816'] else 'TRAIN') for r in rr),'Fixed source split '+d)
  sizes={'SYNTH_AUG65536':65536,'PUBLIC_TRAIN1480':1480,'CF3816':3816,'COMMON4096':4096,'SYNTH_VAL2048':2048}
  packets={a:{} for a in NEW_ARMS+OLD_ARMS}
  for arm in NEW_ARMS:
   require(set(fit['arms'][arm]['outputs'])==set(sizes),'All unchanged domains saved')
   for domain,n in sizes.items():packets[arm][domain]=checked_output(fit['arms'][arm]['outputs'][domain],rawrows[domain],n,helpers)
  for arm in OLD_ARMS:
   packets[arm]['COMMON4096']=checked_output(oldcommon['arms'][arm]['outputs'],rawrows['COMMON4096'],4096,helpers)
   packets[arm]['SYNTH_VAL2048']=checked_output(oldfit['arms'][arm]['outputs']['SYNTH_VAL2048'],rawrows['SYNTH_VAL2048'],2048,helpers)
  for domain in ['COMMON4096','SYNTH_VAL2048']:
   reference=packets[NEW_ARMS[0]][domain]
   for arm in NEW_ARMS[1:]+OLD_ARMS:
    for key in META:require(same(reference[key],packets[arm][domain][key]),'All4 exact same evaluation input '+key)
  require([int((~packets[NEW_ARMS[0]][d]['original_query_available']).sum()) for d in sizes]==[1264,194,360,96,8],'Every fixed domain U denominator')
  # All prediction packets/decoders have been checked before separate GT access.
  targets={d:load(r['targets']) for d,r in records.items()}
  for domain,y in targets.items():helpers['validate_targets'](y,packets[NEW_ARMS[0]][domain]['original_query_available'])
  rows={d:canonical_rows(rawrows[d],d,diagnostic,scenes) for d in sizes}
  results={arm:{domain:evaluate(rows[domain],a,targets[domain],helpers) for domain,a in domains.items()} for arm,domains in packets.items()}
  cube=np.arange(4096).reshape(128,4,8);left=cube[:,0,:].reshape(-1)
  for i,r in enumerate(rawrows['COMMON4096']):require((r['original_global_index'],r['base_index'],r['variant'],r['native_slot'])==(i,i//32,VARIANTS[i%32//8],i%8),'All COMMON paired order/IDs')
  reference=packets[NEW_ARMS[0]]['COMMON4096']
  for base in range(128):
   for vi in range(4):
    for key in ['query_xy','L','native_slot','original_query_available']:require(same(reference[key][cube[base,vi]],reference[key][cube[base,1]]),'Same COVER q/L/native/availability across variants')
  paired={};changes={}
  for arm in NEW_ARMS+OLD_ARMS:
   a=packets[arm]['COMMON4096'];paired[arm]={'BEFORE_to_'+variant:helpers['pair_summary'](targets['COMMON4096']['support_target'],a['support_prediction'],a['raw_support_logits'],left,cube[:,vi,:].reshape(-1),a['original_query_available']) for vi,variant in enumerate(VARIANTS) if vi}
  for domain in ['COMMON4096','SYNTH_VAL2048']:
   changes[domain]={}
   for old,new in zip(OLD_ARMS,NEW_ARMS):
    key=old+' -> '+new;changes[domain][key]={}
    for stratum,ids in helpers['strata'](rows[domain]).items():changes[domain][key][stratum]={name:paired_change(targets[domain]['support_target'][ids,j],packets[old][domain]['support_prediction'][ids,j],packets[new][domain]['support_prediction'][ids,j]) for j,name in enumerate(['visible_end_support','foreign_front_support'])}
  # Same saved BEFORE images, original nominal q/L versus COMMON COVER q/L.
  oldrecord=oldcache['domains']['SYNTH_TRAIN4096'];oldrows=lines(checked(oldrecord['metadata']));oldtargets=load(oldrecord['targets']);oldoriginal={a:checked_output(oldfit['arms'][a]['outputs']['SYNTH_TRAIN4096'],oldrows,4096,helpers) for a in OLD_ARMS}
  for a in OLD_ARMS:require(np.array_equal(oldoriginal[a]['original_global_index'],np.arange(4096,dtype=np.int64)),'Original TRAIN4096 exact row ordering')
  helpers['validate_targets'](oldtargets,oldoriginal[OLD_ARMS[0]]['original_query_available'])
  original_ids=np.array([rawrows['COMMON4096'][int(i)]['source_image_original_global_index'] for i in left],np.int64)
  require(np.array_equal(original_ids,np.array([b*32+k for b in range(128) for k in range(8)],np.int64)),'Same source BEFORE image IDs')
  commontruth=targets['COMMON4096']['support_target'][left];oldtruth=oldtargets['support_target'][original_ids]
  oldav=oldoriginal[OLD_ARMS[0]]['original_query_available'][original_ids];commonav=packets[NEW_ARMS[0]]['COMMON4096']['original_query_available'][left]
  fixed=oldav&commonav&(oldtruth>=0).all(1)&(commontruth>=0).all(1)&(oldtruth==commontruth).all(1);require(int(fixed.sum())==401,'Original fixed401 knownsame BEFORE set')
  fragility={}
  for arm in NEW_ARMS+OLD_ARMS:
   original=packets[arm]['SYNTH_AUG65536'] if arm in NEW_ARMS else oldoriginal[arm]
   ids=original_ids*15 if arm in NEW_ARMS else original_ids
   if arm in NEW_ARMS:
    require(np.array_equal(original['original_global_index'],np.arange(65536,dtype=np.int64)),'New augmented output exact row ordering')
    for sourceid,idx in zip(original_ids,ids):
     row=rawrows['SYNTH_AUG65536'][int(idx)];require(row['family']=='SELF_SCENE15' and row['state_index']==0 and row['source_image_original_global_index']==int(sourceid),'New original-q ownnominal mapping without target equality assumption')
   nt=targets['SYNTH_AUG65536']['support_target'][ids] if arm in NEW_ARMS else oldtruth
   fragility[arm]=geometry_comparison(nt,commontruth,original['support_prediction'][ids],packets[arm]['COMMON4096']['support_prediction'][left],original['original_query_available'][ids],commonav,fixed)
  historic=historical_v61_VAL(descriptors['historical_v61_output_binding_not_opened'],packets[NEW_ARMS[0]]['SYNTH_VAL2048'],targets['SYNTH_VAL2048'],rows['SYNTH_VAL2048'],helpers)
  metrics=OUT/'ALL_FIXED_METRICS_STRATA_PAIRS_AND_FRAGILITY.json'
  write(metrics,dict(completed=True,PASS=True,results=results,COMMON_matched_pairs=paired,corresponding_model_recovery_damage=changes,original_fixed401_geometry_sensitivity=fragility,historical_v61_VAL_task_mismatch_diagnostic=historic,source_TRAIN_apparent_domains=['SYNTH_AUG65536','PUBLIC_TRAIN1480','CF3816'],COMMON_source_TRAIN_derived_not_heldout=True,VAL_historically_reused_not_blind=True,source_associations_not_physical_native_or_coordinate_gold=True,selected_model=None,adopted=False,fully_solved=False))
  ledger=OUT/'ALL166240_MODEL_DOMAIN_ROW_DIAGNOSTICS.jsonl';written=0
  with ledger.open('x') as f:
   for arm,domains in packets.items():
    for domain,a in domains.items():
     y=targets[domain]
     for i,r in enumerate(rawrows[domain]):
      row=dict(arm=arm,domain=domain,row_index=i,row_id=r['row_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],original_query_available=r['original_query_available'],support_truth=y['support_target'][i].tolist(),support_prediction=a['support_prediction'][i].tolist(),owner_weak_hard_truth=int(y['owner_target'][i]),owner_prediction=int(a['owner_prediction'][i]),regional_proxy=int(a['regional_support_proxy_D0_E1_Uminus1'][i]))
      f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');written+=1
  require(written==166240,'All model-domain rows including U/UNKNOWN')
  outputbindings={arm:{d:(fit['arms'][arm]['outputs'][d] if arm in NEW_ARMS else oldcommon['arms'][arm]['outputs'] if d=='COMMON4096' else oldfit['arms'][arm]['outputs'][d]) for d in domains} for arm,domains in packets.items()}
  for domains in outputbindings.values():
   for binding in domains.values():checked(binding)
  for record in records.values():
   for key in ['targets','metadata']:checked(record[key])
  write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,entry=bind(entry_path),code=bind(__file__),protocol=bind(PROTOCOL),fit_complete=e['fit_complete'],all2_freeze=fit['all2_freeze'],metrics=bind(metrics),rows=bind(ledger),model_domain_rows=written,all4_same_eval_metadata=True,all_U_UNKNOWN_partial_mass_preserved=True,fixed_old401_rows=401,model_output_bindings=outputbindings,target_and_metadata_bindings=records,no_raw_pixel_input_weight_model_or_forward=True,new_fit_label_or_threshold_changes=0,selected_model=None,adopted=False,fully_solved=False))
  print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
 except BaseException:
  write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,automatic_retry=False,traceback=traceback.format_exc()));raise


if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args();prepare() if args.prepare else run(args.entry)
