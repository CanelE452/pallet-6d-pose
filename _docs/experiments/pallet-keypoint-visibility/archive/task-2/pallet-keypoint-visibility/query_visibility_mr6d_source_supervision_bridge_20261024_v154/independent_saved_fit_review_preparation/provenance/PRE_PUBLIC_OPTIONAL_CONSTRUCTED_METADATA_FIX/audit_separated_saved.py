"""Saved-only C35/C36 audit. No trainer imports, features, weights or forward.

--prepare uses invented NumPy arrays and protocol metadata only. Actual saved
prediction/GT reads require --run --expected-control-sha ROOT_NOTIFIED_SHA.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent
PROTOCOL=V/'head_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
RUN=V/'runs/C35_C36_SEPARATE_QUERY_FIXED2000';OUT=V/'independent_saved_fit_review'
ARMS=('C35_MATCHED_VISIBILITY_ONLY_CONTROL','C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY')
NAMES=('OLD6144','NEW20480','PALLOX512','MR1840');SIZES=(6144,20480,512,1840)
PREP=H/'PREPARATION_READY.json'
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(b):assert bind(b['path'])==b;return Path(b['path'])
def lines(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def packet(p):
 with np.load(p,allow_pickle=False)as f:
  assert 'features'not in f.files,'Feature bodies forbidden in saved evaluator'
  return{k:f[k].copy()for k in f.files}
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def equal(a,b,path=''):
 if isinstance(a,dict):
  assert set(a)==set(b),(path,set(a)^set(b))
  for k in a:equal(a[k],b[k],path+'/'+str(k))
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y)in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
 elif isinstance(a,float):assert b is not None and np.isclose(a,b,rtol=1e-13,atol=1e-15),(path,a,b)
 else:assert a==b,(path,a,b)
def metric(y,pred,mask):
 known=mask&np.isin(y,[0,1]);cm=[[int((known&(y==t)&(pred==p)).sum())for p in(0,1,-1)]for t in(0,1)]
 ratio=lambda n,d:n/d if d else None
 return dict(CM_rows0_1_columns0_1_U=cm,N=int(known.sum()),negative_recall=ratio(cm[0][0],sum(cm[0])),positive_recall=ratio(cm[1][1],sum(cm[1])),negative_precision=ratio(cm[0][0],cm[0][0]+cm[1][0]),positive_precision=ratio(cm[1][1],cm[0][1]+cm[1][1]),U_rate=ratio(cm[0][2]+cm[1][2],int(known.sum())),all_mask_rows=int(mask.sum()),unknown_truth=int((mask&~np.isin(y,[0,1])).sum()))
def external(y,pred,mask,states):
 m=metric(y,pred,mask);cm=m['CM_rows0_1_columns0_1_U'];ratio=lambda n,d:n/d if d else None
 return dict(CM_D_E_rows_D_E_U_columns=cm,D_recall=m['negative_recall'],E_recall=m['positive_recall'],D_precision=m['negative_precision'],E_precision=m['positive_precision'],prediction_U_known=cm[0][2]+cm[1][2],prediction_U_rate_known=m['U_rate'],all_original_rows=int(mask.sum()),known_DE=m['N'],teacher_S_outside_external=int((mask&(states=='S')).sum()),teacher_U_outside_external=int((mask&(states=='U')).sum()),teacher_S_binary_responses={str(p):int((mask&(states=='S')&(pred==p)).sum())for p in(0,1,-1)},teacher_U_binary_responses={str(p):int((mask&(states=='U')&(pred==p)).sum())for p in(0,1,-1)})
def target(states):return np.where(states=='D',0,np.where(states=='E',1,-1)).astype(np.int8)
def native_meta(rows,n):
 rows=[r for r in rows if r['convention']=='CF_PRIMARY_DECLARED'];assert len(rows)==n
 for i,r in enumerate(rows):assert r['scene_index']==i//8 and r['native_slot']==i%8
 m=dict(rows=rows,state=np.array([r['teacher_state']for r in rows]),split=np.array([r['split']for r in rows]),boundary=np.array([r['correspondence_boundary']for r in rows],bool),bands=np.array([r['position_band']for r in rows]),variant=np.array([r['variant']for r in rows]))
 for k in('material','aspect','foreign_family','geometry_family'):m[k]=np.array([r['factors'][k]for r in rows])
 m['target']=target(m['state']);return m
def group(meta,name):
 bits=name.split('::');mask=np.ones(len(meta['state']),bool)if bits[0]=='ALL'else meta['split']==bits[0];i=1
 while i<len(bits):
  k=bits[i]
  if k=='NONBOUNDARY':mask&=~meta['boundary'];i+=1
  elif k=='BOUNDARY':mask&=meta['boundary'];i+=1
  elif k=='SAME_TARGET_MATERIAL':mask&=meta['foreign_family']==k;i+=1
  elif k=='material_aspect':mask&=(meta['material']==bits[i+1])&(meta['aspect']==bits[i+2]);i+=3
  elif k in('variant','material','aspect','foreign_family','geometry_family'):mask&=meta[k]==bits[i+1];i+=2
  elif k=='historical_area_agreement':mask&=~meta['boundary'];i+=1
  elif k=='historical_area_disagreement':mask&=meta['boundary'];i+=1
  elif k in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):mask&=meta['bands']==k;i+=1
  else:raise AssertionError('Unknown immutable group '+name)
 return mask
def mr_metrics(raw,pack,rows):
 av=pack['query_available'];pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1);entity=np.where(av,(raw[:,1]>=0).astype(np.int8),-1);gated=np.where(av&(raw[:,1]>=0),pred,-1)
 y=np.array([r['role_target']for r in rows]);ey=np.array([r['entity_target']for r in rows]);states=np.array([r['role_state']for r in rows]);oc=np.array([r['occluder_or_uncertainty']for r in rows]);tr=np.array([r['split']=='SOURCE_DEVELOPMENT'for r in rows]);out={}
 for name,mask in [('ALL',np.ones(1840,bool)),('SOURCE_DEVELOPMENT',tr),('SOURCE_VALIDATION',~tr)]:
  out[name]=dict(visibility=metric(y,pred,mask),visibility_inputvalid=metric(y,pred,mask&av),entity=metric(ey,entity,mask),entity_inputvalid=metric(ey,entity,mask&av),secondary_entity_gated_visibility=metric(y,gated,mask),additional_entity_rejection_U=int((mask&av&(raw[:,1]<0)).sum()),semantic_states={str(s):int((mask&(states==s)).sum())for s in np.unique(states)})
  for k in range(8):m=mask&(pack['native_slot']==k);out[name+f'::native{k}']=dict(visibility=metric(y,pred,m),entity=metric(ey,entity,m))
  for o in np.unique(oc):m=mask&(oc==o);out[name+'::occluder::'+str(o)]=dict(visibility=metric(y,pred,m),entity=metric(ey,entity,m))
 return out

def prepare():
 p=read(PROTOCOL);assert tuple(p['arms'])==ARMS and p['updates_each']==2000 and p['seed_each']==1540
 y=np.array([0,0,1,1,-1]);pred=np.array([0,-1,1,0,1]);m=metric(y,pred,np.ones(5,bool));assert m['CM_rows0_1_columns0_1_U']==[[1,0,1],[1,1,0]]and m['N']==4 and m['unknown_truth']==1 and m['U_rate']==.25
 meta=dict(state=np.array(['D','E','U','D']),split=np.array(['VALIDATION']*4),boundary=np.array([0,1,0,0],bool),foreign_family=np.array(['SAME_TARGET_MATERIAL']*4),material=np.array(['PROCEDURAL_WOOD']*4),aspect=np.array(['SQUARE']*4))
 assert group(meta,'VALIDATION::NONBOUNDARY::SAME_TARGET_MATERIAL').tolist()==[True,False,True,True]
 assert group(meta,'VALIDATION::material_aspect::PROCEDURAL_WOOD::SQUARE').all()
 assert not exact(np.array([0.],np.float32),np.array([-0.],np.float32))
 write(PREP,dict(completed=True,prepared_only=True,code=bind(__file__),prospective_protocol=bind(PROTOCOL),fixed_design=p['design'],arms=list(ARMS),updates_each=2000,expected_run=str(RUN),requires_both_terminal_CONTROL_COMPLETE_and_explicit_expected_SHA_argument=True,checks=dict(invented_CM_U_precision_recall_denominators=True,immutable_gate_group_parser=True,strict_byte_parity_signed_zero=True),actual_saved_scores_GT_feature_weight_RGB_or_model_reads=0))
 print(json.dumps(dict(prepared=True,code=bind(__file__),preparation=bind(PREP))),flush=True)

def run(expected):
 pre=read(PREP);assert pre['code']==bind(__file__)and len(expected)==64 and not OUT.exists()
 controlpath=RUN/'CONTROL_COMPLETE.json';assert bind(controlpath)['sha256']==expected
 control=read(controlpath);assert control['completed']and control['PASS']and control['updates_each']==2000 and control['selected_model']is None and control['fully_solved']is False
 assert set(control['arms'])==set(ARMS)
 # Gate all prediction/GT reads on BOTH completed source fits.
 reports={};frozen={};fits={};metadata_bindings=[]
 for arm in ARMS:
  d=RUN/arm;fit=read(check(control['arms'][arm]['fit']));assert fit['completed']and fit['optimizer_updates']==2000
  report=read(check(fit['source_results']));freeze=read(check(fit['model_freeze']));assert report['completed']and report['update']==2000 and report['selected_model']is None and not report['fully_solved']and not report['current_allowed']and not report['adopted']
  assert freeze['completed']and freeze['update']==2000 and freeze['parameters']==224643 and freeze['selected_model']is None and freeze['current_allowed']is False and freeze['threshold_logit']==0 and freeze['all_geometry_input_zero']
  assert report['checkpoint']==freeze['checkpoint']and fit['draw_sha']==report['draw_sha']==control['draw_sha']
  reports[arm]=report;frozen[arm]=freeze;fits[arm]=fit
 p=read(check(control['protocol']));assert tuple(p['arms'])==ARMS and p['updates_each']==2000 and p['seed_each']==1540
 assert p['design']==pre['fixed_design'];check(p['code']);check(p['design']);parent=read(check(p['parent_protocol']))
 native=[native_meta(lines(check(b['GT'])),n)for b,n in zip(parent['native'],SIZES[:2])]
 qualityrows=[lines(check(b))for b in p['quality_labels']];quality=[np.array([-1 if r['point_quality_label']is None else r['point_quality_label']for r in rr],np.int8)for rr in qualityrows]
 mrrows=lines(check(p['source_labels']));publicrows=lines(check(parent['public_labels']));assert len(mrrows)==1840 and len(publicrows)==512
 nativebaselines=[packet(check(b['baseline']))for b in parent['native']]
 facts=[lines(check(p[k]))for k in('Pallox_inputfacts','MR_inputfacts')]
 allraw={};allpack={};output_bindings=[]
 excluded={'raw_three_logits','primary_visibility_prediction','secondary_entity_gated_visibility'}
 for arm in ARMS:
  allraw[arm]={};allpack[arm]={}
  for stage in('INITIAL','FINAL'):
   allraw[arm][stage]=[];allpack[arm][stage]=[]
   for name,n in zip(NAMES,SIZES):
    path=RUN/arm/f'{stage}_{name}_OUTPUTS.npz';a=packet(path);output_bindings.append(bind(path));raw=a['raw_three_logits'];meta={k:v for k,v in a.items()if k not in excluded}
    assert raw.shape==(n,3)and raw.dtype==np.float32 and np.array_equal(a['global_index'],np.arange(n))
    construct=a.get('feature_constructed',np.ones(n,bool));av=a['query_available'];assert construct.dtype==av.dtype==bool and not(av&~construct).any()
    assert np.isfinite(raw[construct]).all()and np.isnan(raw[~construct]).all()
    pred=np.where(av,(raw[:,0]>=0).astype(np.int8),-1);assert exact(pred,a['primary_visibility_prediction'])and exact(np.where(av&(raw[:,1]>=0),pred,-1),a['secondary_entity_gated_visibility'])
    allraw[arm][stage].append(raw);allpack[arm][stage].append(meta)
    if stage=='FINAL':
     init=allpack[arm]['INITIAL'][len(allpack[arm][stage])-1];assert set(init)==set(meta)
     for k in meta:assert exact(meta[k],init[k]),(arm,name,k)
  parity=read(RUN/arm/'INITIAL_PARITY.json');assert parity['completed']and parity['PASS']and parity['C32_visibility_old_new_raw_bitexact']and parity['all4_datasets_repeated_three_outputs_bitexact']and parity['auxiliary_logits_initial_zero']
  for i,base in enumerate(nativebaselines):assert exact(allraw[arm]['INITIAL'][i][:,0].copy(),base['raw_head_logit_before_U'])
  for i,a in enumerate(allpack[arm]['INITIAL']):
   raw=allraw[arm]['INITIAL'][i];construct=a.get('feature_constructed',np.ones(len(raw),bool));assert not raw[construct,1:].any()
   if arm==ARMS[1]:
    assert exact(raw,allraw[ARMS[0]]['INITIAL'][i])
    for k in a:assert exact(a[k],allpack[ARMS[0]]['INITIAL'][i][k])
  assert reports[arm]['primary_visibility_never_filtered_by_entity_quality_or_GTerror']and reports[arm]['point_quality_not_a_visibility_gate']
 # Original metadata fidelity through independent historical saved packets/facts.
 packs=allpack[ARMS[0]]['INITIAL']
 for i,(a,base,meta,qr)in enumerate(zip(packs,nativebaselines,native,qualityrows)):
  assert len(qr)==SIZES[i]
  for key in set(a)&set(base):
   if key=='global_index':assert np.array_equal(a[key],base[key])and a[key].dtype.kind in'iu'and base[key].dtype.kind in'iu'
   else:assert exact(a[key],base[key]),key
  assert exact(a['feature_constructed'],base['head_forwarded'])
  for j,(r,qrow)in enumerate(zip(meta['rows'],qr)):
   assert a['occurrence_id'][j]==r['query_id']==qrow['query_id']and qrow['global_index']==j and qrow['split']==r['split']
   q=np.asarray(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan],np.float64);assert np.array_equal(a['query_xy'][j],q,equal_nan=True)
   assert qrow['original_role_teacher_state_UNCHANGED']==r['teacher_state']and qrow['role_visibility_training_mask_UNCHANGED']and not qrow['coordinates_corrected']
   error=r['error_px'];finite=qrow['reference_xy_exact_synthetic']is not None and error is not None and np.isfinite(q).all()and np.isfinite(error)
   good=r['nearest_inbounds']and finite and not r['correspondence_boundary']and r['teacher_state']in('D','E')
   expected=1 if good and error<=3 else 0 if good and error>20 else -1;assert quality[i][j]==expected
 for a,ff,rows in zip(packs[2:],facts,(publicrows,mrrows)):
  for j,(f,r)in enumerate(zip(ff,rows)):
   assert a['query_id'][j]==f['query_id']and a['frame_id'][j]==f['frame_id']==r['id']and a['source_id'][j]==f['source_id']==r['source_id']and a['split'][j]==f['split']==r['split']
   assert a['native_slot'][j]==r['native_slot']and a['frame_index'][j]==r['frame_index']and f['global_index']==j
   assert np.array_equal(a['query_xy'][j],np.asarray(r['query_xy'],np.float64),equal_nan=True)
   side=np.nan if r['L']is None else r['L'];assert np.array_equal(a['L'][j],np.asarray(side),equal_nan=True)
   assert a['query_available'][j]==f['native_continuous_valid']==r['original_input_valid']and a['nearest_valid'][j]==f['nearest_valid']==r['nearest_valid']
   assert a['feature_constructed'][j]==f['feature_constructed']and a['unit_Q_inside'][j]==f['Q_inside']and a['local_central4_available'][j]==f['local_central4_available']and a['context_available'][j]==f['context_available']
 # Reconstruct fixed pools and all2000 ordered draws without source trainer code.
 rm={k:np.array([r[v]for r in mrrows])for k,v in [('entity','entity_target'),('role','role_target'),('split','split')]};mrav=packs[3]['query_available'];tr=rm['split']=='SOURCE_DEVELOPMENT'
 ep=[np.flatnonzero(tr&mrav&(rm['entity']==k))for k in(0,1)];vi=np.flatnonzero(tr&mrav&(rm['role']>=0))
 pubstate=np.array([r['semantic_state']for r in publicrows]);pubtarget=target(pubstate);pubsplit=np.array([r['split']for r in publicrows]);pi=np.flatnonzero((pubsplit=='TRAIN_LAB')&packs[2]['query_available']&(pubtarget>=0))
 qp=[[np.flatnonzero((meta['split']=='TRAIN')&a['query_available']&(y==k))for k in(0,1)]for meta,a,y in zip(native,packs,quality)]
 assert [len(x)for x in ep]==[365,638]and len(vi)==390 and len(pi)==43
 assert all(r['point_quality_label']==-1 for r in mrrows)
 prefix=[]
 with check(parent['C32_draws']).open()as f:
  for step in range(1,2001):
   r=json.loads(next(f));assert r['update']==step;prefix.append(r)
 drawproof={};first_indices=None
 for ai,arm in enumerate(ARMS):
  logs=lines(RUN/arm/'ALL2000_DRAWS_AND_LOSSES.jsonl');assert len(logs)==2000;rng=np.random.default_rng(1540);dh=hashlib.sha256();unknown_visibility_quality=0;exposure=[Counter(),Counter(),Counter()]
  all_indices=[]
  for step,(r,old)in enumerate(zip(logs,prefix),1):
   assert r['update']==step and r['auxiliary_weight']==ai
   oi=np.asarray(old['native_occurrences'],np.int64);ni=np.asarray(old['extra_native_occurrences'],np.int64);opos=np.asarray(old['native_pair_pool_positions'],np.int64);npos=np.asarray(old['extra_pair_pool_positions'],np.int64)
   ei=np.r_[rng.choice(ep[0],32,replace=True),rng.choice(ep[1],32,replace=True)].astype(np.int64)
   qi=[np.r_[rng.choice(pair[0],16,replace=True),rng.choice(pair[1],16,replace=True)].astype(np.int64)for pair in qp]
   for name,expected_ix in [('old_indices',oi),('new_indices',ni),('entity_indices',ei),('quality_old_indices',qi[0]),('quality_new_indices',qi[1]),('MR_visibility_indices',vi)]:assert r[name]==expected_ix.tolist(),(arm,step,name)
   for ix in(oi,ni,opos,npos,pi,vi,ei,*qi):dh.update(ix.astype('<i8').tobytes())
   for i,ix in enumerate((oi,ni)):
    assert np.all(native[i]['split'][ix]=='TRAIN')and packs[i]['query_available'][ix].all()and np.isin(native[i]['target'][ix],[0,1]).all();unknown_visibility_quality+=int((quality[i][ix]<0).sum());exposure[i].update(ix.tolist())
    assert np.all(native[i]['split'][qi[i]]=='TRAIN')and packs[i]['query_available'][qi[i]].all()and np.isin(quality[i][qi[i]],[0,1]).all()
   assert tr[ei].all()and mrav[ei].all()and tr[vi].all();exposure[2].update(ei.tolist())
   total=np.float32(np.float32(np.float32(np.float32(r['old_native_BCE'])+np.float32(r['new_native_BCE']))+np.float32(r['Pallox_BCE']))+np.float32(r['MR_BCE']))+np.float32(ai)*np.float32(np.float32(r['entity_BCE'])+np.float32(r['quality_BCE']))
   assert np.isfinite(r['grad_norm'])and r['grad_norm']>=0 and np.isclose(total,r['total_loss'],rtol=3e-6,atol=2e-6)
   all_indices.append((r['old_indices'],r['new_indices'],r['entity_indices'],r['quality_old_indices'],r['quality_new_indices'],r['MR_visibility_indices']))
  if first_indices is None:first_indices=all_indices
  else:assert first_indices==all_indices
  assert dh.hexdigest()==reports[arm]['draw_sha']==control['draw_sha']
  drawproof[arm]=dict(updates=2000,draw_SHA256=dh.hexdigest(),native_visibility_rows_with_unknown_quality_exposures=unknown_visibility_quality,unique_old_native_drawn=len(exposure[0]),unique_new_native_drawn=len(exposure[1]),unique_MRentity_drawn=len(exposure[2]),all_training_only=True,all_ordered_indices_replayed=True)
 # Recalculate metrics and immutable35gates directly from NumPy masks.
 limits=read(check(p['original_native35']))['native35_limits'];assert len(limits)==35;recomputed={}
 for arm in ARMS:
  report=reports[arm];raws=allraw[arm]['FINAL'];mr=mr_metrics(raws[3],packs[3],mrrows);equal(mr,report['MR'])
  gates={}
  for name,b in limits.items():
   new=name.startswith('NEW_NATIVE::');tail=name[len('NEW_NATIVE::'if new else'NATIVE::'):];g,state=tail.rsplit('::',1);meta=native[int(new)];a=packs[int(new)];pred=np.where(a['query_available'],(raws[int(new)][:,0]>=0).astype(np.int8),-1);mm=group(meta,g);tt=0 if state=='D'else 1;known=mm&(meta['target']==tt);N=int(known.sum());value=float((pred[known]==tt).mean())if N else None
   gate=dict(value=value,limit=b['limit'],N=N,PASS=bool(N>0 and value is not None and b['limit']is not None and value>=b['limit']));gates[name]=gate;equal(gate,report['native']['gates'][name])
  assert sum(v['PASS']for v in gates.values())==report['native']['native35_passed']and all(v['PASS']for v in gates.values())==report['native']['native35_all_pass']
  # Quality full unknown/U denominators, wood/polymer and rectangular/square.
  qualityreports={}
  for name,a,m,z,y in zip(NAMES,packs,native,raws,quality):
   pred=np.where(a['query_available'],(z[:,2]>=0).astype(np.int8),-1);groups={}
   for fold in('TRAIN','VALIDATION'):
    mask=m['split']==fold;groups[fold]=metric(y,pred,mask)
    for factor in('material','aspect','foreign_family','geometry_family'):
     for value in np.unique(m[factor]):groups[fold+'::'+factor+'::'+str(value)]=metric(y,pred,mask&(m[factor]==value))
    for slot in range(8):groups[fold+f'::native{slot}']=metric(y,pred,mask&(np.arange(len(y))%8==slot))
   qualityreports[name]=groups
  equal(qualityreports,report['synthetic_point_quality'])
  pubpred=np.where(packs[2]['query_available'],(raws[2][:,0]>=0).astype(np.int8),-1);pg={}
  for fold in('ALL','TRAIN_LAB','VAL_HALL'):
   mask=np.ones(512,bool)if fold=='ALL'else pubsplit==fold;pg[fold]=external(pubtarget,pubpred,mask,pubstate);pg[fold+'::INPUT_VALID']=external(pubtarget,pubpred,mask&packs[2]['query_available'],pubstate)
  equal(pg,report['Pallox']['groups']);pcm=pg['VAL_HALL::INPUT_VALID']['CM_D_E_rows_D_E_U_columns'];sanity=dict(D_correct=pcm[0][0],D_N=33,D_minimum=30,E_correct=pcm[1][1],E_N=6,E_minimum=5,PASS=bool(pcm[0][0]>=30 and pcm[1][1]>=5));equal(sanity,report['Pallox']['sanity'])
  vals=[mr['SOURCE_VALIDATION']['visibility_inputvalid'][k]for k in('negative_recall','positive_recall')]+[mr['SOURCE_VALIDATION']['entity_inputvalid'][k]for k in('negative_recall','positive_recall')];ms=all(v is not None and v>=threshold for v,threshold in zip(vals,[.9,.8,.8,.8]));assert ms==report['MR_weak_sanity_PASS']
  assert report['prospective_source_checks_all_pass']==bool(all(g['PASS']for g in gates.values())and sanity['PASS']and ms)
  recomputed[arm]=dict(MR=mr,native35_gates=gates,native35_passed=sum(g['PASS']for g in gates.values()),Pallox=pg,synthetic_point_quality=qualityreports)
 OUT.mkdir();recalc=OUT/'RECOMPUTED_SOURCE_METRICS_AND_DRAW_AUDIT.json';write(recalc,dict(completed=True,PASS=True,arms=recomputed,draw_audit=drawproof,all_metadata_arrays_stages_arms_exact=True,all_original_26624_512_1840_preserved=True,primary_visibility_quality_or_entity_filter=False,selected_model=None,fully_solved=False))
 review=OUT/'REVIEW.json';write(review,dict(completed=True,PASS=True,code=bind(__file__),control_complete=bind(controlpath),protocol=control['protocol'],recalculation=bind(recalc),outputs=output_bindings,all2000_draws_each_arm_replayed=True,initial_C32_visibility_raw_byteexact=True,all_metadata_q_L_flags_original_order_checked=True,fullMR_metrics_and_native35_and_quality_reports_independently_recomputed=True,all_held_validation_excluded_from_gradients_by_draw_ledger=True,selected_model=None,fully_solved=False,model_feature_weight_RGB_private_reads_or_learned_calls=0,limits=['Saved-output numerical audit, not independent semantic truth adjudication.','Final checkpoint/state/repeat claims checked through sealed metadata and initial/final arrays; weights were not deserialized or a model rerun.','Primary visibility retains its original availability and never uses entity/quality scores or GTerror as a filter.']))
 print(json.dumps(dict(review=bind(review),native35={a:recomputed[a]['native35_passed']for a in ARMS})),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--run',action='store_true');a.add_argument('--expected-control-sha');args=a.parse_args()
 if args.prepare and not args.run and args.expected_control_sha is None:prepare()
 elif args.run and not args.prepare and args.expected_control_sha:run(args.expected_control_sha)
 else:raise SystemExit('Use --prepare or root-notified --run --expected-control-sha SHA')
