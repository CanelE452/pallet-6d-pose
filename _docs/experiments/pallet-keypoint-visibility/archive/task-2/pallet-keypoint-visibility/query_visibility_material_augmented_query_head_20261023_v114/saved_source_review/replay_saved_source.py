"""Independent saved C12 joins, draws, selector arithmetic and metrics. No model."""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np
import torch
V=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent;P=V.parent;R=V/'runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111';V113=P/'query_visibility_ycb_wood_foam_source_20261023_v113'

def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def b(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(d):assert b(Path(d['path']))['sha256']==d['sha256']
def save(n,x):
 with(O/n).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
freeze=R/'MODEL_FREEZE.json';assert b(freeze)['sha256']=='d920fdfd15fcc352c4e53c83f0a14a281d5903166ea385ef5fc3bda1eda4c694'
save('SCOPE.json',dict(scope='Saved-only independent45184 identity/typed-label/availability/unit joins,2000draw replay,21history criterion arithmetic+selected raw BCE,15gates and source-baseline comparison. No checkpoint/weight load, head/encoder call, raw RGB, fit, current/private/V112. Old41984 source metadata is retrieved using the frozen pre-C12 C10 join helper; all new3200 joins and all metric/selector/draw math below are independent of C12 runtime helpers.',freeze=b(freeze)))
complete=read(R/'FIT_COMPLETE.json');result=read(R/'SOURCE_RESULTS.json');savedgates=read(R/'SOURCE_GATE_RESULTS.json');design=read(V/'ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json');start=read(R/'RUN_STARTED.json');check(start['entry']);e=read(start['entry']['path'])
assert complete['completed']and complete['optimizer_updates']==2000 and complete['selected_update']==1200 and complete['current_V112_or_PnP']==0
for d in [complete['scores'],complete['gates'],complete['source_results'],complete['model_freeze'],e['trainer'],e['design'],e['protocol'],e['feature_receipt']]:check(d)
assert e['approved_source_fit']and not e['current_allowed']and not e['V112_allowed']and not e['PnP_allowed']
c11=mod(C11/'train_spatial_target_query_head.py','saved_c12_old_c11');c10=c11.pinned_c10();_,d=c10.old_modules()
# Frozen old metadata join only: no inputs/features/checkpoints are loaded by this helper.
old=c10.load_metadata(e,read(design['source']['C10_feature_receipt']['path']),read(design['source']['C10_design']['path']),d)
receipt=read(e['feature_receipt']['path']);line=rows(receipt['new_query_lineage']['path']);inputs=rows(design['source']['material_input']['path']);ts=read(design['source']['material_teacher_seal']['path']);packet=next(z for z in ts['outputs']if Path(z['path']).name=='RAW_SURFACE_TEACHER3200_GT_ONLY.npz');check(packet)
with np.load(packet['path'],allow_pickle=False)as f:newstate=f['raw_surface_teacher_state'].reshape(3200);newtarget=f['external_target_D0_E1_else_minus1'].reshape(3200);newq=f['query_xy_input_copy'];newL=f['roi_side_input_copy']
assert len(inputs)==400 and len(line)==3200 and np.array_equal(newtarget,np.where(newstate=='D',0,np.where(newstate=='E',1,-1)))
for i,r in enumerate(inputs):
 assert r['instance_row_index']==i and np.array_equal(newq[i],r['initial_points'])and newL[i]==r['roi_side']
 for k in range(8):
  n=8*i+k;l=line[n];assert l['local_occurrence_index']==l['local_input_unit_index']==n and l['global_occurrence_index']==41984+n and l['global_input_unit_index']==11652+n
  assert l['input_id']==r['input_id']and l['query_id']==r['input_id']+'::q'+str(k)and l['instance_row_index']==i and l['logical_aux_slot']==k and l['query_xy']==r['initial_points'][k]and l['L']==r['roi_side']and l['query_box_xyxy']==r['query_box_xyxy']and l['split']==r['split']and l['original_split']==r['original_split']and l['image']==r['image']
state=np.r_[old['typed_state'],newstate];target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8);known=target>=0
split=np.r_[old['split'],np.repeat([x['split']for x in inputs],8)];source=np.r_[old['source'],[x['source']for x in line]];domain=np.r_[old['domain'],np.full(3200,'V113_MATERIAL_SURFACE')];variant=np.r_[old['variant'],np.full(3200,'NATURAL_SCENE')];lineage=old['lineage']+line
with np.load(complete['scores']['path'],allow_pickle=False)as f:a={k:f[k]for k in f.files}
z=a['query_logit'];av=a['query_available'];pred=np.where(av,np.where(z<0,0,1),-1).astype(np.int8)
assert z.shape==av.shape==(45184,)and z.dtype==np.float32 and av.dtype==bool and np.isfinite(z).all()and np.array_equal(pred,a['prediction'])and not z[~av].any()
for k,want in [('raw_teacher_state',state),('external_target',target),('split',split),('domain',domain),('source',source),('variant',variant)]:assert np.array_equal(a[k],want),k
assert np.array_equal(a['global_occurrence_index'],np.arange(45184))and np.array_equal(av,a['factual_query_available']&a['context_available'])
q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64);L=np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64)
assert np.array_equal(a['query_xy_raw'],q,equal_nan=True)and np.array_equal(a['L_raw'],L,equal_nan=True)
for i,r in enumerate(lineage):
 identifiers=json.loads(a['original_identifier_metadata_json'][i]);expected={k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')}
 assert identifiers==expected and a['original_split'][i]==(r.get('raw_split')or r.get('original_split')or split[i])
with np.load(design['source']['C11_source_features']['path'],allow_pickle=False)as f:ou=f['occurrence_unit'];oa=f['query_available'];of=f['factual_query_available'];oc=f['context_available']
with np.load(e['feature_pack']['path'],allow_pickle=False)as f:nu=f['global_input_unit_index'];na=f['query_available'];nf=f['factual_query_available'];nc=f['context_available']
assert np.array_equal(a['input_unit_index'],np.r_[ou,nu])and np.array_equal(av,np.r_[oa,na])and np.array_equal(a['factual_query_available'],np.r_[of,nf])and np.array_equal(a['context_available'],np.r_[oc,nc])
assert a['input_unit_index'].min()==0 and a['input_unit_index'].max()==14851
assert {s:int((state==s).sum())for s in'DESU'}==design['source']['raw_states_before_availability']
# Independently construct fixed old/source masks and both source pair lists.
real=domain=='V96_REAL';synth=domain=='V99_SYNTH';material=domain=='V113_MATERIAL_SURFACE';ycb=domain=='V109_YCB_SURFACE'
assert real.sum()==31680 and synth.sum()==2304 and ycb.sum()==8000 and material.sum()==3200
oldpairs=old['old']['old_pairs'];newpairs=old['old']['new_pairs'];assert np.array_equal(a['old_accepted_triplets'],oldpairs)and np.array_equal(a['new_selected_triplets'],newpairs)
def old_masks(s):
 m=real&(split==s)
 return dict(LOCO_BEFORE_D=m&(source=='LOCO')&(variant=='BEFORE')&(state=='D'),PALLOX_BEFORE_D=m&(source=='PALLOX')&(variant=='BEFORE')&(state=='D'),SHAM_D=m&(variant=='CUE_PRESERVING_SHAM')&(state=='D'),BEFORE_E=m&(variant=='BEFORE')&(state=='E'),COVER_E=m&(variant=='OPAQUE_COVER')&(state=='E'))
val={f'old_{k}':v for k,v in old_masks('VALIDATION').items()}
for name,st,stage in [('new_BEFORE_D','D','BEFORE'),('new_SHAM_D','D','SHAM'),('new_COVER_E','E','COVER')]:val[name]=synth&(split=='VALIDATION')&(variant==stage)&(state==st)
for st in'DE':val['material_VALIDATION_'+st]=material&(split=='VALIDATION')&(state==st)
assert len(val)==10 and all(m.any()for m in val.values())
safe=np.where(av,z,0).astype(np.float64);loss=np.logaddexp(0,safe)-np.where(known,target,0)*safe
parts={k:dict(count=int(m.sum()),mean_BCE=float(loss[m].mean()),unavailable=int((m&~av).sum()))for k,m in val.items()}
criterion=float(np.mean([r['mean_BCE']for r in parts.values()]));assert criterion==result['criterion_ten_VAL_cohorts_BCE']==complete['criterion_ten_VAL_cohorts_BCE']and parts==result['VAL_cohorts']
history=rows(R/'HISTORY.jsonl');assert [r['update']for r in history]==list(range(0,2001,100))
for h in history:
 assert set(h['VAL_cohorts'])==set(val)
 for k,m in val.items():assert h['VAL_cohorts'][k]['count']==int(m.sum())and h['VAL_cohorts'][k]['unavailable']==int((m&~av).sum())
 assert float(np.mean([x['mean_BCE']for x in h['VAL_cohorts'].values()]))==h['criterion_ten_VAL_cohorts_BCE']
selected=min(history,key=lambda r:r['criterion_ten_VAL_cohorts_BCE']);assert selected['update']==1200 and selected['VAL_cohorts']==parts
# Independent RNG replay: original ordered pools, separate208/214/215 generators.
def filtered(ix):
 ix=np.asarray(ix,np.int64);return ix[av[ix]]if ix.ndim==1 else ix[av[ix].all(1)]
tr=old_masks('TRAIN');pools={'old_triplets':filtered(oldpairs[split[oldpairs[:,0]]=='TRAIN']),'synth_triplets':filtered(newpairs[split[newpairs[:,0]]=='TRAIN']),'old_LOCO_D':filtered(np.flatnonzero(tr['LOCO_BEFORE_D'])),'old_PALLOX_D':filtered(np.flatnonzero(tr['PALLOX_BEFORE_D'])),'old_natural_E':filtered(np.flatnonzero(tr['BEFORE_E']))}
for prefix,dom in [('synth',synth),('YCB',ycb)]:
 for st in'DE':pools[prefix+'_'+st]=filtered(np.flatnonzero(dom&(split=='TRAIN')&(state==st)))
mp={st:np.flatnonzero(material&(split=='TRAIN')&(state==st)&av)for st in'DE'};assert all(len(x)>0 for x in list(pools.values())+list(mp.values()))
g=torch.Generator().manual_seed(208);mg=torch.Generator().manual_seed(214);sg=torch.Generator().manual_seed(215);oldhash=hashlib.sha256();fullhash=hashlib.sha256();trace=rows(R/'TRAIN_UPDATES.jsonl');assert len(trace)==2000
quotas=(('old_LOCO_D',2),('old_PALLOX_D',2),('synth_D',4),('YCB_D',8),('old_natural_E',4),('synth_E',8),('YCB_E',12));loss_combination_max_error=0.
for step,row in enumerate(trace,1):
 oi=torch.randint(len(pools['old_triplets']),(4,),generator=g).numpy();si=torch.randint(len(pools['synth_triplets']),(4,),generator=g).numpy();trip=np.r_[pools['old_triplets'][oi],pools['synth_triplets'][si]]
 ordered=np.concatenate([trip.ravel()]+[pools[k][torch.randint(len(pools[k]),(n,),generator=g).numpy()]for k,n in quotas]);perm=torch.randperm(64,generator=g).numpy();oldix=ordered[perm];oldtr=np.argsort(perm)[np.arange(24).reshape(8,3)]
 md=mp['D'][torch.randint(len(mp['D']),(16,),generator=mg).numpy()];me=mp['E'][torch.randint(len(mp['E']),(16,),generator=mg).numpy()];pp=torch.randperm(96,generator=sg).numpy();ix=np.r_[oldix,md,me][pp];tt=np.argsort(pp)[oldtr]
 assert row['update']==step and np.array_equal(ix,row['occurrence_indices'])and np.array_equal(tt,row['matched_triplet_batch_positions'])and np.array_equal(a['input_unit_index'][ix],row['input_unit_indices'])
 audit=row['sampling'];assert audit['base64_occurrence_indices']==oldix.tolist()and audit['base64_triplet_positions']==oldtr.tolist()and audit['material_D_occurrence_indices']==md.tolist()and audit['material_E_occurrence_indices']==me.tolist()and audit['final96_permutation']==pp.tolist()
 assert np.array_equal(ix[tt],trip)and np.array_equal(target[ix][tt],np.tile([0,1,0],(8,1)))and np.array_equal(np.bincount(target[ix],minlength=2),[48,48])and known[ix].all()and av[ix].all()and(split[ix]=='TRAIN').all()
 assert row['available_loss_rows']==96 and row['sampled_unavailable_rows']==0
 err=abs(row['loss']-(row['BCE']+.25*row['matched_hinge']));loss_combination_max_error=max(err,loss_combination_max_error);assert err<2e-6+5e-7*abs(row['loss'])
 for hh,ii,t0 in[(oldhash,oldix,oldtr),(fullhash,ix,tt)]:hh.update(ii.astype('<i8').tobytes());hh.update(t0.astype('<i8').tobytes())
assert oldhash.hexdigest()==result['base64_all2000_draws_sha256']==design['sampling']['base64_draw_SHA256_required']and fullhash.hexdigest()==result['all96_all2000_draws_sha256']
# Own confusion/precision/recall/U and selected matched-triplet calculations.
def cm(mask,pred0=pred,states=state):
 ss=states[:len(pred0)];mask=mask[:len(pred0)];mat=np.array([[int((mask&(ss==st)&(pred0==pr)).sum())for pr in(0,1,-1)]for st in'DE']);rat=lambda x,y:float(x/y)if y else None
 return dict(matrix=mat.tolist(),D_precision=rat(mat[0,0],mat[:,0].sum()),D_recall=rat(mat[0,0],mat[0].sum()),E_precision=rat(mat[1,1],mat[:,1].sum()),E_recall=rat(mat[1,1],mat[1].sum()),U_rate=rat(mat[:,2].sum(),mat.sum()),D_N=int(mat[0].sum()),E_N=int(mat[1].sum()))
def pairs(pa,s,p=pred):
 choose=split[pa[:,0]]==s;hit=(p[pa]==[0,1,0]).all(1);return dict(N=int(choose.sum()),all3_correct=int((choose&hit).sum()),rate=float(hit[choose].mean())if choose.any()else None)
metrics={}
for s in('TRAIN','VALIDATION'):
 for name,mask,expected in [('old_public',real&(split==s),result['metrics']['old_exact_V96'][s]['all_known']['matrix']),('synthetic',synth&(split==s),result['metrics']['new_external'][s]['all_rows']['matrix']),('material',material&(split==s),result['material_surface'][s]['matrix'])]:
  cc=cm(mask);assert cc['matrix']==expected;metrics[name+'_'+s]=cc
  if name!='material':metrics[name+'_'+s]['triplets']=pairs(oldpairs if name=='old_public'else newpairs,s)
assert cm(ycb)['matrix']==result['YCB_surface_TRAIN']['matrix']
values={}
for gate,mask in [('old_LOCO_before_D_recall_min',val['old_LOCO_BEFORE_D']),('old_PALLOX_before_D_recall_min',val['old_PALLOX_BEFORE_D']),('old_natural_before_E_recall_min',val['old_BEFORE_E']),('old_sham_D_recall_min',val['old_SHAM_D']),('old_cover_E_recall_min',val['old_COVER_E']),('new_BEFORE_D_recall_min',val['new_BEFORE_D']),('new_SHAM_D_recall_min',val['new_SHAM_D']),('new_COVER_E_recall_min',val['new_COVER_E']),('material_VAL_D_recall_min',val['material_VALIDATION_D']),('material_VAL_E_recall_min',val['material_VALIDATION_E'])]:values[gate]=float((pred[mask]==target[mask]).mean())
values['old_accepted_VAL_pair_all3_min']=pairs(oldpairs,'VALIDATION')['rate'];values['new_selected_VAL_pair_all3_min']=pairs(newpairs,'VALIDATION')['rate']
for k,mask in [('old_known_VAL_U_rate_max',real&(split=='VALIDATION')&known),('new_known_VAL_U_rate_max',synth&(split=='VALIDATION')&known),('material_VAL_known_U_rate_max',material&(split=='VALIDATION')&known)]:values[k]=float((~av[mask]).mean())
assert set(values)==set(savedgates['gates'])==set(design['source_gates'])
for k,v in values.items():assert v==savedgates['gates'][k]['value']and savedgates['gates'][k]['limit']==design['source_gates'][k]and savedgates['gates'][k]['PASS']==(v<=design['source_gates'][k]if k.endswith('_max')else v>=design['source_gates'][k])
assert all(v['PASS']for v in savedgates['gates'].values())and savedgates['all_gates_pass']and not savedgates['current_inference_authorized']and not savedgates['V112_inference_authorized']
# Descriptive saved SOURCE baselines on exact old33984 only, no material scoring invented.
c11scores=C11/'runs/C11_DINOV2_SPATIAL_TARGET_QUERY/SELECTED_ALL41984_SOURCE_SCORES.npz';v61scores=P/'query_visibility_local_residual_head_20261023_v101/v61_source_baseline/outputs/SOURCE_SCORES.npz'
with np.load(c11scores,allow_pickle=False)as f:c11p=np.where(f['query_available'],np.where(f['query_logit']<0,0,1),-1);assert np.array_equal(f['raw_teacher_state'],state[:41984])and np.array_equal(f['query_xy_raw'],q[:41984],equal_nan=True)and np.array_equal(f['L_raw'],L[:41984],equal_nan=True)
with np.load(v61scores,allow_pickle=False)as f:
 v61p=np.where(f['available'],np.where(f['raw_visible_logit']>=0,0,1),-1);assert np.array_equal(v61p,f['decoded_aux_prediction'])and np.array_equal(f['typed_state'],state[:33984])and np.array_equal(f['query_xy'],q[:33984],equal_nan=True)and np.array_equal(f['L'],L[:33984],equal_nan=True)
comparison={}
for name,p0 in [('v61_source',v61p),('C11_source1800',c11p),('C12_source1200',pred)]:
 comparison[name]={}
 for s in('TRAIN','VALIDATION'):
  for dom,dommask,pa in [('old_public',real,oldpairs),('synthetic',synth,newpairs)]:comparison[name][dom+'_'+s]=dict(**cm(dommask&(split==s),p0),triplets=pairs(pa,s,p0))
comparison['material_baselines']='C11 and v61 not scored on material3200; no material baseline invented.'
save('DESCRIPTIVE_OLD_SOURCE_BASELINES.json',dict(completed=True,same_old33984_rows_only=True,models=comparison,baselines=dict(C11=b(c11scores),v61=b(v61scores)),selection_use=False))
review=dict(completed=True,PASS=True,blockers=[],scope='Saved-only source labels/input metadata/mapping/score/history/sampler replay; no checkpoint parameter, model/encoder forward, raw RGB/current/private/V112, optimizer/fit or PnP.',fit_complete=b(R/'FIT_COMPLETE.json'),source_freeze=b(freeze),scores=complete['scores'],trainer=e['trainer'],design=e['design'],selected_update=1200,criterion_ten_VAL_cohorts_BCE=criterion,all45184_ID_q_L_order_rawfolds_typed_labels_exact=True,unique_input_units=14852,all_feature_mapping_and_factual_context_availability_exact=True,raw_sign_U_exact=True,raw_state_counts={k:int((state==k).sum())for k in'DESU'},availability=dict(all_rows=45184,available=int(av.sum()),U=int((~av).sum()),known_TRAIN_U=int((known&(split=='TRAIN')&~av).sum()),known_VAL_U=int((known&(split=='VALIDATION')&~av).sum())),all2000_TRAIN_only_known_available_48D48E_and8triplets_exact=True,base64_hash=oldhash.hexdigest(),full96_hash=fullhash.hexdigest(),sampler_generators=[208,214,215],all21_criterion_arithmetic_and_strict_earliest_exact=True,selected1200_all10_BCE_recomputed_from_raw_scores=True,history_limit='Other20 per-step raw logits are not archived; their10 saved cohort mean BCE values/counts/unavailable and criterion arithmetic/earliest selection are checked, not independently reconstructed by model forward.',training_loss_limit='All sampled indices and loss/BCE+.25hinge arithmetic are checked; raw logits/gradients for each update are not archived and not recomputed.',logged_loss_combination_max_abs_error=loss_combination_max_error,metrics=metrics,all15_gates_exact=True,passed_gates=15,gate_values=values,source_VAL_development_exposure=True,foamE_and_VAL_absent=True,descriptive_source_baselines=b(O/'DESCRIPTIVE_OLD_SOURCE_BASELINES.json'),limits=['Surface AUX/nativeUNKNOWN/humanfalse source labels do not certify native pallet junction visibility or coordinate correctness.','Material VAL is development exposed; foamE and foamVAL absent; YCB old8000 is TRAIN only.','This PASS is procedural arithmetic/consistency, not a fresh generalization proof or current admission.','Composition/batch/selector change together; no isolated causal attribution.'],new_model_forward_fit_current_V112_PnP=0,own_code=b(Path(__file__)))
save('REVIEW.json',review);save('SEAL.json',dict(completed=True,PASS=True,review=b(O/'REVIEW.json'),files=[b(x)for x in sorted(O.iterdir())if x.is_file()and x.name!='SEAL.json']))
print(json.dumps(dict(review=b(O/'REVIEW.json'),seal=b(O/'SEAL.json'),selected_update=1200,metrics=metrics),indent=2))
