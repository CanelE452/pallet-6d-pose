"""Independent V160 saved-source review. No model imports/feature/checkpoint loads.

Preparation: --prepare-only reads this code and invented arrays only.
Execution: --all-four-completed-trigger ROOT_JSON requires all4_completed and
source_only both true before any run/metric/output body is read. Checkpoints
are compared as declared seal dictionaries; their bytes/tensors stay unopened.
"""
from pathlib import Path
import argparse, ast, datetime, hashlib, json, math, traceback
import numpy as np
O=Path(__file__).resolve().parent; V=O.parent; P=V.parent
H=V/'source_fit_preparation'
RUN=V/'runs/C39_C40_C41_C42_NATIVE_SLOT_OWNER_FIXED8000'
BASE=P/'query_visibility_independent_auxiliary_source_control_20261024_v155/runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000/C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000'
FAMILIES=('OLD6144','NEW20480','PALLOX512','MR1840')
ARMS=('C39_BASE_NATIVE0_OWNER0','C40_NATIVE_SLOT8_OWNER0','C41_NATIVE0_OWNER3_AUX','C42_NATIVE_SLOT8_OWNER3_AUX')
CONFIGS=((False,0),(True,0),(False,1),(True,1))
COUNTS=(6144,20480,512,1840)
MATERIALS=('PROCEDURAL_WOOD','DARK_POLYMER');ASPECTS=('RECTANGULAR','SQUARE')
GEOMETRIES=('BOARD_BLOCK_RUNNER','RIB_LEG_RUNNER')
FOREIGN=('SAME_TARGET_MATERIAL','CARDBOARD_TAN','BACKGROUND_LIKE','HIGH_CONTRAST')
BANDS=('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY')
OUTPUT_KEYS={'raw_three_logits','raw_owner_three_logits','owner_three_probabilities','raw_six_logits','owner_class_order','primary_visibility_prediction','secondary_entity_gated_visibility'}
BINDINGS=[];CHECKS=[]
def safe(p):
 p=Path(p).resolve();s=str(p).lower()
 assert p.is_relative_to(P) and not any(x in s for x in ('_v112','/private_evaluation/','current223'))
 return p
def bind(p):
 p=safe(p);assert p.suffix not in ('.pt','.pth','.pkl'), 'Checkpoint bytes intentionally unopened'
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 out={'path':str(p),'sha256':h.hexdigest()};BINDINGS.append(out);return out
def verify(b):
 assert bind(b['path'])==b,b['path'];return safe(b['path'])
def read(p):return json.loads(safe(p).read_text())
def lines(p):
 with safe(p).open() as f:
  for s in f:
   if s.strip():yield json.loads(s)
def write(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def check(name,b=True,detail=None):
 assert b,name;CHECKS.append({'name':name,'PASS':True,**({'detail':detail} if detail is not None else {})})
def same(a,b):
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if isinstance(a,(float,np.floating)) and isinstance(b,(float,int,np.floating,np.integer)):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return a==b
def ae(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def ratio(n,d):return float(n/d) if d else None
def metric(y,p,mask,states=None):
 cm=np.array([[np.count_nonzero(mask&(y==t)&(p==k)) for k in (0,1,-1)] for t in (0,1)],np.int64)
 if states is None:
  return dict(CM_rows0_1_columns0_1_U=cm.tolist(),N=int(cm.sum()),negative_recall=ratio(cm[0,0],cm[0].sum()),positive_recall=ratio(cm[1,1],cm[1].sum()),negative_precision=ratio(cm[0,0],cm[:,0].sum()),positive_precision=ratio(cm[1,1],cm[:,1].sum()),U_rate=ratio(cm[:,2].sum(),cm.sum()),all_mask_rows=int(mask.sum()),unknown_truth=int((mask&~np.isin(y,[0,1])).sum()))
 return dict(CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=ratio(cm[0,0],cm[0].sum()),E_recall=ratio(cm[1,1],cm[1].sum()),D_precision=ratio(cm[0,0],cm[:,0].sum()),E_precision=ratio(cm[1,1],cm[:,1].sum()),prediction_U_known=int(cm[:,2].sum()),prediction_U_rate_known=ratio(cm[:,2].sum(),cm.sum()),all_original_rows=int(mask.sum()),known_DE=int(cm.sum()),teacher_S_outside_external=int((mask&(states=='S')).sum()),teacher_U_outside_external=int((mask&(states=='U')).sum()),teacher_S_binary_responses={str(k):int((mask&(states=='S')&(p==k)).sum()) for k in (0,1,-1)},teacher_U_binary_responses={str(k):int((mask&(states=='U')&(p==k)).sum()) for k in (0,1,-1)})
def synthetic_meta(rows,n):
 assert len(rows)==2*n;rs=rows[::2]
 assert all(r['convention']=='CF_PRIMARY_DECLARED' for r in rs)
 state=np.array([r['teacher_state'] for r in rs]);m={'rows':rs,'state':state,'target':np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8)}
 for key,field in [('split','split'),('boundary','correspondence_boundary'),('bands','position_band'),('variant','variant')]:m[key]=np.array([r[field] for r in rs])
 for key in ('material','aspect','geometry_family','foreign_family'):m[key]=np.array([r['factors'][key] for r in rs])
 return m
def native_groups(m,new):
 out={};n=len(m['target'])
 for fold in ('ALL','TRAIN','VALIDATION'):
  f=np.ones(n,bool) if fold=='ALL' else m['split']==fold;out[fold]=f
  if new:
   out[fold+'::historical_area_agreement']=f&~m['boundary'];out[fold+'::historical_area_disagreement']=f&m['boundary']
  else:
   out[fold+'::NONBOUNDARY']=f&~m['boundary'];out[fold+'::BOUNDARY']=f&m['boundary'];out[fold+'::NONBOUNDARY::SAME_TARGET_MATERIAL']=f&~m['boundary']&(m['foreign_family']=='SAME_TARGET_MATERIAL')
  for band in BANDS:
   out[fold+'::'+band]=f&(m['bands']==band)
   if not new:out[fold+'::NONBOUNDARY::'+band]=f&~m['boundary']&(m['bands']==band)
  for key in ('variant','material','aspect','geometry_family','foreign_family'):
   values={'material':MATERIALS,'aspect':ASPECTS,'geometry_family':GEOMETRIES,'foreign_family':FOREIGN}.get(key,np.unique(m[key])) if new else np.unique(m[key][f])
   for value in values:out[fold+'::'+key+'::'+str(value)]=f&(m[key]==value)
  if new:
   for material in MATERIALS:
    for aspect in ASPECTS:out[fold+'::material_aspect::'+material+'::'+aspect]=f&(m['material']==material)&(m['aspect']==aspect)
 return out
def mr_report(raw,a,rr):
 y=np.array([r['role_target'] for r in rr]);entity=np.array([r['entity_target'] for r in rr]);state=np.array([r['role_state'] for r in rr]);occ=np.array([r['occluder_or_uncertainty'] for r in rr]);training=np.array([r['split']=='SOURCE_DEVELOPMENT' for r in rr]);av=a['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1);ep=np.where(av,(raw[:,1]>=0).astype(np.int8),-1);gated=np.where(av&(raw[:,1]>=0),pred,-1);out={}
 for name,mask in [('ALL',np.ones(1840,bool)),('SOURCE_DEVELOPMENT',training),('SOURCE_VALIDATION',~training)]:
  out[name]=dict(visibility=metric(y,pred,mask),visibility_inputvalid=metric(y,pred,mask&av),entity=metric(entity,ep,mask),entity_inputvalid=metric(entity,ep,mask&av),secondary_entity_gated_visibility=metric(y,gated,mask),additional_entity_rejection_U=int((mask&av&(raw[:,1]<0)).sum()),semantic_states={str(s):int((mask&(state==s)).sum()) for s in np.unique(state)})
  for k in range(8):
   m=mask&(a['native_slot']==k);out[name+f'::native{k}']=dict(visibility=metric(y,pred,m),entity=metric(entity,ep,m))
  for value in np.unique(occ):
   m=mask&(occ==value);out[name+'::occluder::'+str(value)]=dict(visibility=metric(y,pred,m),entity=metric(entity,ep,m))
 return out

def packet(p):
 p=safe(p);assert p.suffix=='.npz' and ('_OUTPUTS.npz' in p.name), 'Only saved output NPZ permitted'
 bind(p)
 with np.load(p,allow_pickle=False) as z:
  assert 'features' not in z.files
  return {k:z[k] for k in z.files}
def check_metadata(family,a,expected):
 n=len(expected);assert len(a['raw_three_logits'])==n
 identity='occurrence_id' if family in FAMILIES[:2] else 'query_id'
 assert a[identity].astype(str).tolist()==[r['query_id'] for r in expected]
 for field in ('global_index','native_slot','frame_index','query_available'):
  assert np.array_equal(a[field],np.array([r[field] for r in expected])),(family,field)
 q=np.array([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in expected],float);L=np.array([np.nan if r['L'] is None else r['L'] for r in expected])
 assert np.array_equal(a['query_xy'],q,equal_nan=True) and np.array_equal(a['L'],L,equal_nan=True)
 if family in FAMILIES[2:]:assert a['split'].astype(str).tolist()==[r['split'] for r in expected]
 constructed=a.get('feature_constructed',np.ones(n,bool));assert np.array_equal(constructed,[r['feature_constructed'] for r in expected])
 if family!='PALLOX512':assert np.array_equal(a['no_detection'],[r['no_detection'] for r in expected])
 raw=a['raw_six_logits'];assert raw.shape==(n,6) and raw.dtype==np.float32
 assert ae(a['raw_three_logits'],raw[:,:3].copy()) and ae(a['raw_owner_three_logits'],raw[:,3:].copy())
 assert np.isfinite(raw[constructed]).all() and np.isnan(raw[~constructed]).all()
 pred=np.where(a['query_available'],(raw[:,0]>=0).astype(np.int8),-1)
 assert np.array_equal(a['primary_visibility_prediction'],pred)
 assert np.array_equal(a['secondary_entity_gated_visibility'],np.where(a['query_available']&(raw[:,1]>=0),pred,-1))
 assert a['owner_class_order'].tolist()==['TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED']
 z=raw[constructed,3:];ex=np.exp(z-z.max(1,keepdims=True));want=ex/ex.sum(1,keepdims=True)
 assert np.allclose(a['owner_three_probabilities'][constructed],want,rtol=1e-6,atol=1e-7)
 assert np.isnan(a['owner_three_probabilities'][~constructed]).all()
 return pred

def draw_review(protocol,prev,metas,pub,rr,owner,reports):
 role=np.array([r['role_target'] for r in rr]);entity=np.array([r['entity_target'] for r in rr]);tr=np.array([r['split']=='SOURCE_DEVELOPMENT' for r in rr]);av=np.array([r['original_input_valid'] for r in rr]);vi=np.flatnonzero(av&tr&(role>=0));oi_known=np.flatnonzero(owner>=0)
 assert len(vi)==390 and np.isin(oi_known,vi).all() and len(oi_known)==253
 pstate=np.array([r['semantic_state'] for r in pub]);pav=np.array([r['original_input_valid'] for r in pub]);ptr=np.array([r['split']=='TRAIN_LAB' for r in pub]);pi=np.flatnonzero(ptr&pav&np.isin(pstate,['D','E']));assert len(pi)==43
 ep=[np.flatnonzero(av&tr&(entity==k)) for k in (0,1)]
 qualities=[list(lines(verify(b))) for b in protocol['quality_labels']]
 qp=[]
 for rows,m in zip(qualities,metas):
  q=np.array([-1 if r['point_quality_label'] is None else r['point_quality_label'] for r in rows]);qa=np.array([r['query_available'] for r in EXPECTED[m['family']]])
  qp.append([np.flatnonzero((m['split']=='TRAIN')&qa&(q==k)) for k in (0,1)])
 def draw(rng,pair,each):return np.r_[rng.choice(pair[0],each,replace=True),rng.choice(pair[1],each,replace=True)].astype(np.int64)
 rng=np.random.default_rng(1550);dh=hashlib.sha256();streams=[lines(RUN/a/'ALL8000_DRAWS_AND_LOSSES.jsonl') for a in ARMS];n=0
 for step,ref in enumerate(lines(verify(prev['C32_draws'])),1):
  n=step;logs=[next(s) for s in streams];oi=np.array(ref['native_occurrences'],np.int64);ni=np.array(ref['extra_native_occurrences'],np.int64);op=np.array(ref['native_pair_pool_positions'],np.int64);np_=np.array(ref['extra_pair_pool_positions'],np.int64);ei=draw(rng,ep,32);qi=[draw(rng,p,16) for p in qp]
  assert ref['update']==step and ref['extra_source']=='V136_NATIVE'
  for indices,m,family in zip((oi,ni),metas,FAMILIES[:2]):
   assert indices.shape==(64,) and (m['split'][indices]=='TRAIN').all()
   assert all(EXPECTED[family][int(i)]['query_available'] for i in indices)
   assert np.array_equal(m['target'][indices],np.r_[np.ones(16),np.zeros(32),np.ones(16)])
  for arm,r,(hint,weight) in zip(ARMS,logs,CONFIGS):
   assert r['update']==step and r['native_slot_hint']==hint and r['owner_auxiliary_weight']==weight and r['auxiliary_weight']==1
   for key,want in [('old_indices',oi),('new_indices',ni),('entity_indices',ei),('quality_old_indices',qi[0]),('quality_new_indices',qi[1]),('MR_visibility_indices',vi),('owner_known_indices',oi_known)]:assert np.array_equal(r[key],want),(arm,step,key)
   assert r['owner_class_counts']==[53,52,148]
   assert all(math.isfinite(r[k]) for k in ('old_native_BCE','new_native_BCE','Pallox_BCE','MR_BCE','entity_BCE','quality_BCE','owner_balanced_CE','total_loss','visibility_grad_norm','auxiliary_grad_norm','new_branch_grad_norm'))
   want=sum(r[k] for k in ('old_native_BCE','new_native_BCE','Pallox_BCE','MR_BCE','entity_BCE','quality_BCE'))+weight*r['owner_balanced_CE']
   assert math.isclose(r['total_loss'],want,rel_tol=2e-6,abs_tol=2e-6)
  for ix in (oi,ni,op,np_,pi,vi,ei,*qi):dh.update(np.asarray(ix,dtype='<i8').tobytes())
 assert n==8000
 for s in streams:assert next(s,None) is None
 for r in reports:assert r['draw_sha']==dh.hexdigest()
 for arm in ARMS:bind(RUN/arm/'ALL8000_DRAWS_AND_LOSSES.jsonl')
 return dh.hexdigest()

EXPECTED={}
def execute(trigger):
 # This is deliberately the first body read on the execution route.
 t=read(trigger);assert t['all4_completed'] is True and t['source_only'] is True
 trigger_binding=bind(trigger)
 p=read(H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json');assert p['code']['sha256']=='3dd58095938f63b37b92be2b42a628fb0862169087e9c41f7af873c8001c9260';verify(p['code']);bind(H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json')
 static_review=read(V/'independent_trainer_review/PRE_EXECUTION_TRAINER_REVIEW.json');assert static_review['PASS'] and static_review['bindings']['trainer']==p['code'] and static_review['bindings']['protocol']==bind(H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json');bind(V/'independent_trainer_review/PRE_EXECUTION_TRAINER_REVIEW.json')
 control=read(RUN/'CONTROL_COMPLETE.json');assert control['completed'] and control['PASS'] and control['source_only'] and control['updates_each']==8000 and list(control['arms'])==list(ARMS)
 assert control['code']==p['code'] and control['protocol']==bind(H/'PROTOCOL_BEFORE_SOURCE_FEATURE_OR_MODEL_BODIES.json')
 entry=read(verify(control['entry']));assert entry['approved_actual_source_fit'] is True and entry['source_only'] is True and entry['code']==p['code'] and entry['protocol']==control['protocol']
 assert not any(entry[k] for k in ('private_allowed','new_encoder_R0_allowed','PnP_allowed','q_L_change_allowed','automatic_retry'))
 assert control['selected_model'] is None and control['fully_solved'] is False
 for forbidden in ('FAILED_EXECUTION.json',):assert not (RUN/forbidden).exists()
 barrier=read(verify(control['all4_frozen_before_source_comparison']));assert barrier['completed'] and barrier['arms']==list(ARMS) and barrier['source_validation_reports_emitted'] is False
 expected=list(lines(verify(p['expected_source_metadata'])));assert len(expected)==28976
 for f,n in zip(FAMILIES,COUNTS):EXPECTED[f]=[r for r in expected if r['family']==f];assert len(EXPECTED[f])==n
 contract=read(verify(p['input_metadata_contract']));prev=read(verify(p['parent_protocol']))
 metas=[synthetic_meta(list(lines(verify(b['GT']))),n) for b,n in zip(prev['native'],COUNTS[:2])]
 for m,f in zip(metas,FAMILIES[:2]):m['family']=f
 pub=list(lines(verify(prev['public_labels'])));rr=list(lines(verify(p['source_labels'])));owner_rows=list(lines(verify(p['owner_labels'])));owner=np.array([r['owner_aux_target'] for r in owner_rows])
 assert len(pub)==512 and len(rr)==len(owner_rows)==1840
 assert np.bincount(owner[owner>=0],minlength=3).tolist()==[53,52,148] and int((owner<0).sum())==1587
 for i,(r,e) in enumerate(zip(owner_rows,EXPECTED['MR1840'])):
  assert r['query_id']==e['query_id'] and r['native_slot']==e['native_slot'] and r['split']==e['split']
  if r['owner_aux_target']>=0:assert r['split']=='SOURCE_DEVELOPMENT' and r['center4_valid'] is True
  if r['split']=='SOURCE_VALIDATION':assert r['owner_aux_target']==-1 and not r['prior_owner_observation_present']
 assert owner[1043]==-1 and not owner_rows[1043]['center4_valid']
 check('owner253_only_original_TRAIN390_and_held360_UNKNOWN')
 limits=read(verify(p['original_native35']))['native35_limits'];assert len(limits)==35
 reports=[];initials=[];finals=[];parities=[]
 for arm,(hint,weight) in zip(ARMS,CONFIGS):
  d=RUN/arm;fit=read(verify(control['arms'][arm]['fit']));r=read(verify(fit['source_results']));freeze=read(verify(fit['model_freeze']));pre=read(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json');ip=read(d/'INITIAL_PARITY.json')
  for x in (fit,freeze,pre,ip,r):assert x['completed']
  assert fit['optimizer_updates']==r['update']==freeze['update']==pre['update']==8000 and pre['source_comparison_done'] is False
  assert pre['checkpoint']==barrier['checkpoints'][arm]==freeze['checkpoint']==r['checkpoint']
  assert freeze['source_checks_all_pass']==r['prospective_source_checks_all_pass'] and freeze['diagnostic_only'] and not freeze['current_allowed'] and freeze['selected_model'] is None and freeze['threshold_logit']==0 and freeze['all_geometry_input_zero']
  assert control['arms'][arm]['native35_passed']==r['native']['native35_passed'] and control['arms'][arm]['MR_weak_sanity_PASS']==r['MR_weak_sanity_PASS']
  assert r['arm']==arm and r['native_slot_hint']==hint and r['owner_auxiliary_weight']==weight and r['visibility_initialization']=='FRESH'
  assert r['selected_model'] is None and r['fully_solved'] is False and r['adopted'] is False and r['current_allowed'] is False
  assert r['primary_visibility_never_filtered_by_entity_quality_or_GTerror'] and r['source_labels_weak_not_human_gold'] and r['point_quality_not_a_visibility_gate']
  aa=[];bb=[]
  for family in FAMILIES:
   a=packet(d/f'INITIAL_{family}_OUTPUTS.npz');b=packet(d/f'FINAL_{family}_OUTPUTS.npz');check_metadata(family,a,EXPECTED[family]);check_metadata(family,b,EXPECTED[family])
   assert a.keys()==b.keys()
   for key in a.keys()-OUTPUT_KEYS:assert ae(a[key],b[key]),(arm,family,key)
   assert not a['raw_six_logits'][a.get('feature_constructed',np.ones(len(a['query_available']),bool)),1:].any()
   aa.append(a);bb.append(b)
  assert sum(np.count_nonzero(x['primary_visibility_prediction']==-1) for x in bb)==535
  reports.append(r);initials.append(aa);finals.append(bb);parities.append(ip)
  bind(d/'INITIAL_PARITY.json');bind(d/'FROZEN_BEFORE_ANY_SOURCE_COMPARISON.json')
 check('all4_exact8000_frozen_barrier_no_adoption')
 check('original_optimizer_groups_clip5_bound_to_PASS_frozen_static_review')
 check('all28976_original_metadata_inputU535_and_saved_decisions')
 assert len({x['all4_original_initial_state_sha'] for x in parities})==len({x['all4_full_initial_state_sha'] for x in parities})==1
 assert parities[0]['all4_original_initial_state_sha']==barrier['common_original_initial_state']
 assert len({r['auxiliary_initial_state'] for r in reports})==len({r['auxiliary_final_state'] for r in reports})==1
 for j in range(1,4):
  for i in range(4):
   assert ae(initials[0][i]['raw_six_logits'],initials[j][i]['raw_six_logits'])
   assert ae(finals[0][i]['raw_three_logits'][:,1:].copy(),finals[j][i]['raw_three_logits'][:,1:].copy())
   for key in finals[0][i].keys()-OUTPUT_KEYS:assert ae(finals[0][i][key],finals[j][i][key]),(ARMS[j],FAMILIES[i],key)
 check('all4_same_original_initial_states_and_auxiliary_final_states_outputs')
 draw_sha=draw_review(p,prev,metas,pub,rr,owner,reports);assert draw_sha==barrier['common_draw_sha']==control['draw_sha']
 check('8000_original_draw_replay_and_full253_owner_loss_each_arm')
 for arm,r,outs in zip(ARMS,reports,finals):
  assert r['native']['original_AUX29_not_retested'] and r['native']['original64_qualification_not_claimed']
  for i,key in enumerate(('old_native','new_native')):
   m=metas[i];pred=outs[i]['primary_visibility_prediction'];want={name:metric(m['target'],pred,mask,m['state']) for name,mask in native_groups(m,i==1).items()};assert same(r['native'][key],want),(arm,key)
  gates={}
  for name,base in limits.items():
   new=name.startswith('NEW_NATIVE::');prefix='NEW_NATIVE::' if new else 'NATIVE::';group,state=name[len(prefix):].rsplit('::',1);rrr=r['native']['new_native' if new else 'old_native'][group];value=rrr[state+'_recall'];N=sum(rrr['CM_D_E_rows_D_E_U_columns'][int(state=='E')]);limit=base['limit'];gates[name]=dict(value=value,limit=limit,N=N,PASS=bool(N>0 and value is not None and limit is not None and value>=limit))
  assert same(r['native']['gates'],gates) and r['native']['native35_passed']==sum(x['PASS'] for x in gates.values()) and r['native']['native35_all_pass']==all(x['PASS'] for x in gates.values())
  ps=np.array([x['semantic_state'] for x in pub]);py=np.where(ps=='D',0,np.where(ps=='E',1,-1));split=np.array([x['split'] for x in pub]);pa=outs[2];groups={}
  for fold in ('ALL','TRAIN_LAB','VAL_HALL'):
   mask=np.ones(512,bool) if fold=='ALL' else split==fold;groups[fold]=metric(py,pa['primary_visibility_prediction'],mask,ps);groups[fold+'::INPUT_VALID']=metric(py,pa['primary_visibility_prediction'],mask&pa['query_available'],ps)
  assert same(r['Pallox']['groups'],groups)
  cm=groups['VAL_HALL::INPUT_VALID']['CM_D_E_rows_D_E_U_columns'];sanity=dict(D_correct=cm[0][0],D_N=33,D_minimum=30,E_correct=cm[1][1],E_N=6,E_minimum=5,PASS=bool(cm[0][0]>=30 and cm[1][1]>=5));assert same(r['Pallox']['sanity'],sanity)
  want=mr_report(outs[3]['raw_six_logits'],outs[3],rr);assert same(r['MR'],want)
  vis=want['SOURCE_VALIDATION']['visibility_inputvalid'];ent=want['SOURCE_VALIDATION']['entity_inputvalid'];ms=all(v is not None and v>=lim for v,lim in zip([vis['negative_recall'],vis['positive_recall'],ent['negative_recall'],ent['positive_recall']],[.90,.80,.80,.80]));assert r['MR_weak_sanity_PASS']==ms
  assert r['prospective_source_checks_all_pass']==bool(all(x['PASS'] for x in gates.values()) and sanity['PASS'] and ms)
  assert same(r['primary_visibility_component'],dict(native35_all_pass=all(x['PASS'] for x in gates.values()),native35_passed=sum(x['PASS'] for x in gates.values()),Pallox_original_sanity=sanity,MR_held_visibility=vis,does_not_replace_full_qualification=True))
  for i,family in enumerate(FAMILIES[:2]):
   qrows=list(lines(verify(p['quality_labels'][i])));y=np.array([-1 if x['point_quality_label'] is None else x['point_quality_label'] for x in qrows]);m=metas[i];pred=np.where(outs[i]['query_available'],(outs[i]['raw_three_logits'][:,2]>=0).astype(np.int8),-1);qr={}
   for fold in ('TRAIN','VALIDATION'):
    mask=m['split']==fold;qr[fold]=metric(y,pred,mask)
    for factor in ('material','aspect','foreign_family','geometry_family'):
     for value in np.unique(m[factor]):qr[fold+'::'+factor+'::'+str(value)]=metric(y,pred,mask&(m[factor]==value))
    for k in range(8):qr[fold+f'::native{k}']=metric(y,pred,mask&(outs[i]['native_slot']==k))
   assert same(r['synthetic_point_quality'][family],qr)
  known=(owner>=0)&outs[3]['query_available'];op=np.argmax(outs[3]['raw_owner_three_logits'],1);cm=[[int((known&(owner==k)&(op==j)).sum()) for j in range(3)] for k in range(3)];assert r['owner_diagnostic']['known253']==253 and r['owner_diagnostic']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG']==cm and r['owner_diagnostic']['held_owner_accuracy'] is None
 check('independent_confusion_arithmetic_all_reports_native35_and_full_gate')
 # Source-scale control parity; no checkpoint deserialization required.
 baseline_fit=read(BASE/'FIT_COMPLETE.json');baseline=read(verify(baseline_fit['source_results']));assert baseline_fit['optimizer_updates']==8000 and baseline['visibility_initialization']=='FRESH' and baseline['draw_sha']==draw_sha
 baseline_initial=read(BASE/'INITIAL_PARITY.json');assert baseline_initial['initial_state']==parities[0]['all4_original_initial_state_sha']
 for i,family in enumerate(FAMILIES):
  for prefix,newout in [('INITIAL',initials[0][i]),('FINAL',finals[0][i])]:
   old=packet(BASE/f'{prefix}_{family}_OUTPUTS.npz');assert ae(old['raw_three_logits'],newout['raw_three_logits']),(prefix,family,'raw3');assert ae(old['primary_visibility_prediction'],newout['primary_visibility_prediction']),(prefix,family,'pred')
 check('C39_vs_V155_C38_all28976_original3_logits_predictions_and_draw_SHA_exact')
 bind(RUN/'CONTROL_COMPLETE.json');bind(BASE/'FIT_COMPLETE.json');bind(BASE/'INITIAL_PARITY.json')
 out={'schema':'V160_INDEPENDENT_SAVED_SOURCE_EXECUTION_REVIEW','completed':True,'PASS':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'trigger':trigger_binding,'review_code':bind(__file__),'checks':CHECKS,'checks_passed':len(CHECKS),'all4_source_output_rows_each':28976,'inputU_each':535,'owner_known253_T_F_BG':[53,52,148],'common_draw_sha':draw_sha,'C39_C38_source_scale_control_exact':True,'source_qualification_values_not_reinterpreted':True,'selected_model':None,'adopted':False,'fully_solved':False,'current_allowed':False,'bindings':list({(x['path'],x['sha256']):x for x in BINDINGS}.values()),'source_feature_packets_or_checkpoint_bytes_read':0,'model_imports_or_forwards_or_training':0,'limitations':['Original13 final checkpoint and optimizer tensors are not deserialized/independently compared; exact original3 outputs, initial13-state hash and draw SHA provide source-scale control check. Optimizer partition/clip invariants bind the PASS frozen static review.','Weak source-role and center-owner semantics remain unchanged; no gold/admission/primary-success claim.']}
 write(O/'SAVED_SOURCE_EXECUTION_REVIEW.json',out);print(json.dumps({'PASS':True,'checks':len(CHECKS),'review':bind(O/'SAVED_SOURCE_EXECUTION_REVIEW.json')}))

def prepare_only():
 ast.parse(Path(__file__).read_text())
 y=np.array([0,0,1,1,-1]);pred=np.array([0,-1,1,0,1]);mask=np.ones(5,bool);m=metric(y,pred,mask)
 assert m['CM_rows0_1_columns0_1_U']==[[1,0,1],[1,1,0]] and m['N']==4 and m['unknown_truth']==1 and m['U_rate']==.25
 assert same({'x':[None,.5]},{'x':[None,1/2]}) and not same({'x':.5},{'x':.4})
 result={'schema':'V160_INDEPENDENT_SAVED_REVIEW_PREPARATION_COMPLETE','completed':True,'PASS':True,'prepared_only':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code':bind(__file__),'invented_confusion_and_comparison_checks_PASS':True,'V160_output_checkpoint_score_metric_bodies_read':0,'source_feature_or_checkpoint_bytes_read':0,'execution_requires_explicit_root_all4_completed_trigger':{'all4_completed':True,'source_only':True},'all4_saved_review_executed':False,'no_idle_output_polling':True}
 write(O/'PREPARATION_COMPLETE.json',result);print(json.dumps({'prepared_only':True,'PASS':True,'preparation':bind(O/'PREPARATION_COMPLETE.json')}))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--all-four-completed-trigger',type=Path);args=parser.parse_args()
 if args.prepare_only and not args.all_four_completed_trigger:prepare_only()
 elif args.all_four_completed_trigger and not args.prepare_only:
  try:execute(args.all_four_completed_trigger)
  except BaseException:
   write(O/'FAILED_SAVED_REVIEW.json',{'completed':False,'traceback':traceback.format_exc(),'automatic_retry':False});raise
 else:raise SystemExit('Choose --prepare-only or explicit --all-four-completed-trigger ROOT_JSON')
