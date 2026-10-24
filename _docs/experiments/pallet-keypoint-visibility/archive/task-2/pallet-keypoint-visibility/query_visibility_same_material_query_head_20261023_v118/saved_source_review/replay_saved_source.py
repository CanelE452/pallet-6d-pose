"""Independent saved C13 NumPy metrics/joins and CPU RNG replay; no model calls."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent;R=V/'runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY'
BASE=45184;N=46108
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(b):assert bind(b['path'])['sha256']==b['sha256'],b['path']
def write(n,v):
 with(H/n).open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

freeze_path=R/'MODEL_FREEZE.json';assert bind(freeze_path)['sha256']=='eb63579d72b004f5cc98a77039181306dd49742dc894eae8f25ac6da26013771'
write('SCOPE.json',dict(scope='Saved source metadata/GT and feature-mapping fields only; independently replay all46108 identities,2000 CPU sampler traces,21selector arithmetic/selected rawBCE,15gates and numeric source matrices. Checkpoint bytes hashed only, no parameters loaded. No raw images, encoder/head calls, optimizer/fit/private/current/V112/V117/PnP.',source_freeze=bind(freeze_path)))
fit=read(R/'FIT_COMPLETE.json');result=read(R/'SOURCE_RESULTS.json');gates=read(R/'SOURCE_GATE_RESULTS.json');freeze=read(freeze_path);design=read(V/'ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json');started=read(R/'RUN_STARTED.json');check(started['entry']);entry=read(started['entry']['path'])
assert fit['completed']and fit['optimizer_updates']==2000 and fit['selected_update']==1000 and fit['current_V112_or_PnP']==0
for b in[*[fit[k]for k in('source_results','scores','gates','model_freeze','additive_reports')],*[entry[k]for k in('trainer','protocol','design','generated','source_bindings','new_feature_pack','new_feature_receipt','new_source_admission','feature_integrity_review')],freeze['checkpoint'],freeze['head_class_code']]:check(b)
assert entry['approved_source_fit']and all(entry[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
assert freeze['completed']and freeze['update']==freeze['selected_update']==1000 and freeze['threshold_logit']==0 and freeze['trainable_parameters']==221825 and freeze['encoder_trainable_parameters']==0 and freeze['current_used_for_fit_or_selection']is False
checkpoint_before=bind(freeze['checkpoint']['path']);initial=bind(R/'INITIAL_CHECKPOINT.pt');final=bind(R/'FINAL_CHECKPOINT.pt');assert len({checkpoint_before['sha256'],initial['sha256'],final['sha256']})==3
c12=module(design['parent_C12_trainer']['path'],'saved_C13_exact_parent_metadata');c11,c10,_,d=c12.source_modules();parent=read(design['parent_C12_design']['path']);pp=read(c12.PROTOCOL);material_receipt=read(pp['new_feature_receipt']['path'])
# The frozen pre-C13 helper loads only historical source GT/metadata; no model or feature tensors.
old=c12.load_metadata(entry,material_receipt,parent,c10,d)
inputs=rows(design['new_source']['input_occurrences']['path']);newgt=rows(design['new_source']['GT_occurrences']['path']);assert len(inputs)==len(newgt)==924
newlabels=[];samewood=[]
for j,(r,g)in enumerate(zip(inputs,newgt)):
 b,off=divmod(j,33);rank=None if off==0 else(off-1)//2;variant='BEFORE'if off==0 else'OPAQUE_COVER'if off%2 else'CUE_PRESERVING_SHAM'
 assert (r['occurrence_index'],r['base_index'],r['donor_rank'],r['variant'])==(j,b,rank,variant)
 assert all(r[k]==g[k]for k in('occurrence_index','occurrence_id','base_index','original_base_row_index','donor_rank','variant'))
 assert r['split']=='TRAIN'and r['raw_split']==('TRAIN'if r['source']=='LOCO'else'TRAIN_LAB')and g['native_mapping']=='UNKNOWN'and g['human_gold']is False
 before=inputs[33*b];assert all(r[k]==before[k]for k in('query_xy','L','slot_index','native_slot_logical_only','source','raw_split','frozen_context_box_xyxy'))
 if off==0:assert g['candidate_aux_hidden_label_GT_only']==0 and not g['pair_ready']and not r['full_rgb_changed']
 elif g['pair_ready']:
  assert r['full_rgb_changed']and g['candidate_aux_hidden_label_GT_only']==(1 if off%2 else 0)
  if off%2:samewood.append([BASE+j,BASE+j+1])
 else:assert g['candidate_aux_hidden_label_GT_only']==-1 and not r['full_rgb_changed']
 newlabels.append(g['candidate_aux_hidden_label_GT_only'])
newlabels=np.asarray(newlabels,np.int8);samewood=np.array(samewood,np.int64);assert samewood.shape==(62,2)
state=np.r_[old['typed_state'],np.where(newlabels==0,'D',np.where(newlabels==1,'E','U'))];target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8);known=target>=0
split=np.r_[old['split'],np.full(924,'TRAIN')];domain=np.r_[old['domain'],np.full(924,'V116_SAME_WOOD_CONTROLLED')];source=np.r_[old['source'],[r['source']for r in inputs]];variant=np.r_[old['variant'],[r['variant']for r in inputs]];lineage=old['lineage']+inputs
with np.load(fit['scores']['path'],allow_pickle=False)as z:a={k:z[k]for k in z.files}
raw=a['query_logit'];av=a['query_available'];pred=np.where(av,np.where(raw<0,0,1),-1).astype(np.int8)
assert raw.shape==av.shape==(N,)and raw.dtype==np.float32 and av.dtype==bool and np.isfinite(raw).all()and not raw[~av].any()and np.array_equal(pred,a['prediction'])
for key,want in[('raw_teacher_state',state),('external_target',target),('split',split),('domain',domain),('source',source),('variant',variant)]:assert np.array_equal(a[key],want),key
assert np.array_equal(a['global_occurrence_index'],np.arange(N))and np.array_equal(av,a['factual_query_available']&a['context_available'])
q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64);L=np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64)
assert np.array_equal(q,a['query_xy_raw'],equal_nan=True)and np.array_equal(L,a['L_raw'],equal_nan=True)
for i,r in enumerate(lineage):
 expected={k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')}
 assert json.loads(a['original_identifier_metadata_json'][i])==expected and a['original_split'][i]==(r.get('raw_split')or r.get('original_split')or split[i])
with np.load(parent['source']['C11_source_features']['path'],allow_pickle=False)as z:f11={k:z[k]for k in('occurrence_unit','query_available','factual_query_available','context_available','unit_Q_inside')}
with np.load(pp['new_feature_pack']['path'],allow_pickle=False)as z:fm={k:z[k]for k in('global_input_unit_index','query_available','factual_query_available','context_available','unit_Q_inside')}
with np.load(entry['new_feature_pack']['path'],allow_pickle=False)as z:fn={k:z[k]for k in('occurrence_unit','query_available','factual_query_available','context_available','unit_Q_inside','global_occurrence_index')}
expected_units=[];nextunit=28
for r in inputs:
 expected_units.append(nextunit if r['full_rgb_changed']else r['base_index'])
 if r['full_rgb_changed']:nextunit+=1
assert nextunit==152 and np.array_equal(fn['occurrence_unit'],expected_units)and np.array_equal(fn['global_occurrence_index'],np.arange(924))
units=np.r_[f11['occurrence_unit'],fm['global_input_unit_index'],14852+fn['occurrence_unit']].astype(np.int32)
assert np.array_equal(a['input_unit_index'],units)and units.min()==0 and units.max()==15003
for name in('query_available','factual_query_available','context_available'):assert np.array_equal(a[name],np.r_[f11[name],fm[name],fn[name]])
unitinside=np.r_[f11['unit_Q_inside'],fm['unit_Q_inside'],fn['unit_Q_inside']];assert np.array_equal(a['context_Q_inside'],unitinside[units])
assert {s:int((state==s).sum())for s in'DESU'}==design['combined']['raw_states']and int((known&(split=='TRAIN')).sum())==3233 and int((known&(split=='VALIDATION')).sum())==636
real=domain=='V96_REAL';synth=domain=='V99_SYNTH';ycb=domain=='V109_YCB_SURFACE';material=domain=='V113_MATERIAL_SURFACE';newwood=domain=='V116_SAME_WOOD_CONTROLLED'
oldpairs=old['old']['old_pairs'];synthpairs=old['old']['new_pairs']
assert np.array_equal(oldpairs,a['old_accepted_triplets'])and np.array_equal(synthpairs,a['new_selected_triplets'])and np.array_equal(samewood,a['samewood_accepted_cover_sham_pairs'])
def masks(s):
 base=real&(split==s)
 return dict(LOCO_BEFORE_D=base&(source=='LOCO')&(variant=='BEFORE')&(state=='D'),PALLOX_BEFORE_D=base&(source=='PALLOX')&(variant=='BEFORE')&(state=='D'),BEFORE_E=base&(variant=='BEFORE')&(state=='E'),SHAM_D=base&(variant=='CUE_PRESERVING_SHAM')&(state=='D'),COVER_E=base&(variant=='OPAQUE_COVER')&(state=='E'))
val={f'old_{k}':v for k,v in masks('VALIDATION').items()}
for name,st,stage in[('new_BEFORE_D','D','BEFORE'),('new_SHAM_D','D','SHAM'),('new_COVER_E','E','COVER')]:val[name]=synth&(split=='VALIDATION')&(variant==stage)&(state==st)
for st in'DE':val['material_VALIDATION_'+st]=material&(split=='VALIDATION')&(state==st)
safe=np.where(av,raw,0).astype(np.float64);loss=np.logaddexp(0,safe)-np.where(known,target,0)*safe
parts={key:dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~av).sum()))for key,mask in val.items()};criterion=float(np.mean([p['mean_BCE']for p in parts.values()]))
assert len(parts)==10 and parts==result['VAL_cohorts']and criterion==result['criterion_ten_VAL_cohorts_BCE']==fit['criterion_ten_VAL_cohorts_BCE']
history=rows(R/'HISTORY.jsonl');assert [h['update']for h in history]==list(range(0,2001,100))
for h in history:
 assert set(h['VAL_cohorts'])==set(val)and h['criterion_ten_VAL_cohorts_BCE']==float(np.mean([p['mean_BCE']for p in h['VAL_cohorts'].values()]))
 for key,mask in val.items():assert h['VAL_cohorts'][key]['count']==int(mask.sum())and h['VAL_cohorts'][key]['unavailable']==int((mask&~av).sum())
best=min(history,key=lambda h:h['criterion_ten_VAL_cohorts_BCE']);assert best['update']==1000 and best['VAL_cohorts']==parts
# Independent ordered pools and CPU RNG streams208/214/215/216/217.
def filtered(ix):
 ix=np.asarray(ix,np.int64);return ix[av[ix]]if ix.ndim==1 else ix[av[ix].all(1)]
tm=masks('TRAIN');pools={'old_triplets':filtered(oldpairs[split[oldpairs[:,0]]=='TRAIN']),'synth_triplets':filtered(synthpairs[split[synthpairs[:,0]]=='TRAIN']),'old_LOCO_D':filtered(np.flatnonzero(tm['LOCO_BEFORE_D'])),'old_PALLOX_D':filtered(np.flatnonzero(tm['PALLOX_BEFORE_D'])),'old_natural_E':filtered(np.flatnonzero(tm['BEFORE_E']))}
for prefix,dom in[('synth',synth),('YCB',ycb)]:
 for st in'DE':pools[prefix+'_'+st]=filtered(np.flatnonzero(dom&(split=='TRAIN')&(state==st)))
mp={st:np.flatnonzero(material&(split=='TRAIN')&(state==st)&av)for st in'DE'};wp=samewood[av[samewood].all(1)];assert len(wp)==62
g,mg,sg,ng,fg=[torch.Generator().manual_seed(seed)for seed in(208,214,215,216,217)];hashes=[hashlib.sha256()for _ in range(3)];trace=rows(R/'TRAIN_UPDATES.jsonl');assert len(trace)==2000
quotas=(('old_LOCO_D',2),('old_PALLOX_D',2),('synth_D',4),('YCB_D',8),('old_natural_E',4),('synth_E',8),('YCB_E',12));max_loss_error=0.
for step,row in enumerate(trace,1):
 oi=torch.randint(len(pools['old_triplets']),(4,),generator=g).numpy();si=torch.randint(len(pools['synth_triplets']),(4,),generator=g).numpy();trip=np.r_[pools['old_triplets'][oi],pools['synth_triplets'][si]]
 ordered=np.concatenate([trip.ravel()]+[pools[key][torch.randint(len(pools[key]),(n,),generator=g).numpy()]for key,n in quotas]);perm64=torch.randperm(64,generator=g).numpy();ix64=ordered[perm64];tr64=np.argsort(perm64)[np.arange(24).reshape(8,3)]
 md=mp['D'][torch.randint(len(mp['D']),(16,),generator=mg).numpy()];me=mp['E'][torch.randint(len(mp['E']),(16,),generator=mg).numpy()];perm96=torch.randperm(96,generator=sg).numpy();ix96=np.r_[ix64,md,me][perm96];tr96=np.argsort(perm96)[tr64]
 choices=torch.randint(62,(16,),generator=ng).numpy();newpairs=wp[choices];perm128=torch.randperm(128,generator=fg).numpy();ix=np.r_[ix96,newpairs.ravel()][perm128];pairs=np.argsort(perm128)[np.r_[tr96[:,1:3],np.arange(96,128).reshape(16,2)]]
 assert row['update']==step and np.array_equal(ix,row['occurrence_indices'])and np.array_equal(pairs,row['matched_cover_sham_batch_positions'])and np.array_equal(units[ix],row['input_unit_indices'])
 audit=row['sampling'];audit96=audit['base96'];assert audit['base96_occurrence_indices']==ix96.tolist()and audit['base96_triplet_positions']==tr96.tolist()and audit['new_pair_pool_indices']==choices.tolist()and audit['new_cover_sham_occurrence_indices']==newpairs.tolist()and audit['final128_permutation']==perm128.tolist()
 assert audit96['base64_occurrence_indices']==ix64.tolist()and audit96['base64_triplet_positions']==tr64.tolist()and audit96['material_D_occurrence_indices']==md.tolist()and audit96['material_E_occurrence_indices']==me.tolist()and audit96['final96_permutation']==perm96.tolist()
 assert known[ix].all()and av[ix].all()and(split[ix]=='TRAIN').all()and np.array_equal(np.bincount(target[ix],minlength=2),[64,64])and np.array_equal(target[ix[pairs]],np.tile([1,0],(24,1)))and row['available_loss_rows']==128
 assert np.array_equal(ix[pairs[:8]],trip[:,1:3])and np.array_equal(ix[pairs[8:]],newpairs)
 err=abs(row['loss']-(row['BCE']+.25*row['matched_hinge']));max_loss_error=max(max_loss_error,err);assert err<2e-6+5e-7*abs(row['loss'])and row['BCE']>=0 and row['matched_hinge']>=0 and np.isfinite(row['gradient_norm_before_clip'])
 for h,ii,pp0 in zip(hashes,(ix64,ix96,ix),(tr64,tr96,pairs)):h.update(ii.astype('<i8').tobytes());h.update(pp0.astype('<i8').tobytes())
assert hashes[0].hexdigest()==result['base64_all2000_draws_sha256']==design['sampling']['base64_draw_SHA256_required']and hashes[1].hexdigest()==result['base96_all2000_draws_sha256']==design['sampling']['base96_draw_SHA256_required']and hashes[2].hexdigest()==result['all128_all2000_draws_sha256']
def cm(mask):
 mat=np.array([[int((mask&(state==st)&(pred==pr)).sum())for pr in(0,1,-1)]for st in'DE']);ratio=lambda x,y:float(x/y)if y else None
 return dict(matrix=mat.tolist(),D_precision=ratio(mat[0,0],mat[:,0].sum()),D_recall=ratio(mat[0,0],mat[0].sum()),E_precision=ratio(mat[1,1],mat[:,1].sum()),E_recall=ratio(mat[1,1],mat[1].sum()),U_rate=ratio(mat[:,2].sum(),mat.sum()))
def triple(pa,s):
 ix=split[pa[:,0]]==s;hit=(pred[pa]==[0,1,0]).all(1);return dict(N=int(ix.sum()),all3_correct=int(hit[ix].sum()),rate=float(hit[ix].mean()))
metrics={};add=read(fit['additive_reports']['path'])
for dom in sorted(set(domain)):
 for s in('TRAIN','VALIDATION'):
  mask=(domain==dom)&(split==s)
  if not mask.any():continue
  c=cm(mask);saved=add['source_by_domain_split'][dom+'_'+s]
  assert c['matrix']==saved['matrix']and all(c[k]==saved[k]for k in('D_precision','D_recall','E_precision','E_recall'))and c['U_rate']==saved['known_external_U_rate']
  assert saved['all_N']==int(mask.sum())and saved['known_external_N']==int((mask&known).sum())and saved['known_geometric_S_out_of_task_N']==int((mask&(state=='S')).sum())and saved['unresolved_teacher_U_N']==int((mask&(state=='U')).sum())
  for label,key in[('S','S_out_of_task_raw_decisions'),('U','teacher_U_raw_decisions')]:assert saved[key]=={str(pr):int((mask&(state==label)&(pred==pr)).sum())for pr in(0,1,-1)}
  metrics[dom+'_'+s]=c
values={}
for name,mask in[('old_LOCO_before_D_recall_min',val['old_LOCO_BEFORE_D']),('old_PALLOX_before_D_recall_min',val['old_PALLOX_BEFORE_D']),('old_natural_before_E_recall_min',val['old_BEFORE_E']),('old_sham_D_recall_min',val['old_SHAM_D']),('old_cover_E_recall_min',val['old_COVER_E']),('new_BEFORE_D_recall_min',val['new_BEFORE_D']),('new_SHAM_D_recall_min',val['new_SHAM_D']),('new_COVER_E_recall_min',val['new_COVER_E']),('material_VAL_D_recall_min',val['material_VALIDATION_D']),('material_VAL_E_recall_min',val['material_VALIDATION_E'])]:values[name]=float((pred[mask]==target[mask]).mean())
values['old_accepted_VAL_pair_all3_min']=triple(oldpairs,'VALIDATION')['rate'];values['new_selected_VAL_pair_all3_min']=triple(synthpairs,'VALIDATION')['rate']
for name,mask in[('old_known_VAL_U_rate_max',real&(split=='VALIDATION')&known),('new_known_VAL_U_rate_max',synth&(split=='VALIDATION')&known),('material_VAL_known_U_rate_max',material&(split=='VALIDATION')&known)]:values[name]=float((~av[mask]).mean())
assert set(values)==set(gates['gates'])==set(design['source_gates'])and len(values)==15
for name,value in values.items():
 gate=gates['gates'][name];assert value==gate['value']and gate['limit']==design['source_gates'][name]and gate['PASS']==(value<=gate['limit']if name.endswith('_max')else value>=gate['limit'])
assert gates['all_gates_pass']and all(g['PASS']for g in gates['gates'].values())and gates['current_inference_authorized']is False and gates['V112_inference_authorized']is False
pair_result=dict(N=62,cover_E_correct=int((pred[samewood[:,0]]==1).sum()),sham_D_correct=int((pred[samewood[:,1]]==0).sum()),both_correct=int((pred[samewood]==[1,0]).all(1).sum()),any_prediction_U=int((pred[samewood]<0).any(1).sum()),relative_margin_ge2=int(((raw[samewood[:,0]].astype(np.float64)-raw[samewood[:,1]])>=2).sum()),relative_margin_ge2_but_cover_D=int((((raw[samewood[:,0]].astype(np.float64)-raw[samewood[:,1]])>=2)&(pred[samewood[:,0]]==0)).sum()))
assert pair_result==add['new_samewood_TRAIN_pairs']and bind(freeze['checkpoint']['path'])==checkpoint_before and bind(freeze_path)==fit['model_freeze']
review=dict(completed=True,PASS=True,blockers=[],fit_complete=bind(R/'FIT_COMPLETE.json'),source_freeze=bind(freeze_path),model_freeze=bind(freeze_path),scores=fit['scores'],trainer=entry['trainer'],design=entry['design'],selected_update=1000,criterion_ten_VAL_cohorts_BCE=criterion,all46108_ID_qL_order_rawfold_labels_exact=True,feature_units=15004,all_feature_mapping_and_availability_exact=True,all2000_TRAIN_known_available_D64E64_and24_truepairs_exact=True,draw_hashes=[h.hexdigest()for h in hashes],all21_selector_arithmetic_and_earliest_strict_minimum_exact=True,selected1000_all10_BCE_recomputed_from_saved_raw_logits=True,all15_gates_exact=True,passed_gates=15,gate_values=values,raw_state_counts={s:int((state==s).sum())for s in'DESU'},availability=dict(N=N,available=int(av.sum()),U=int((~av).sum()),known_TRAIN_U=int((known&(split=='TRAIN')&~av).sum()),known_VAL_U=int((known&(split=='VALIDATION')&~av).sum())),source_metrics=metrics,old_public_VAL_triplets=triple(oldpairs,'VALIDATION'),synthetic_VAL_triplets=triple(synthpairs,'VALIDATION'),new_samewood_TRAIN_pairs=pair_result,logged_loss_arithmetic_max_error=max_loss_error,checkpoint=checkpoint_before,checkpoint_bytes_unchanged=True,checkpoint_parameters_deserialized=False,limits=['Other20 per-step logits and all2000 gradients are not archived; verify their saved means/counts/criterion arithmetic and loggedBCE+.25hinge plus complete draw eligibility, not a new model forward.','Checkpoint byte identity/freeze/selection/code are bound; no parameter load or independent head reforward was performed.','Source VAL is development exposed. New62 pairs are PALLOX TRAIN, not a new validation set; material foamE/foamVAL absent, source AUX/nativeUNKNOWN/humanfalse.','Arithmetic consistency and15gatesPASS are not a universal-transfer proof or automatic current admission. No private, V112 or V117 artifact was read.'],new_encoder_head_forward_fit_current_V112_V117_PnP=0,review_code=bind(__file__))
write('REVIEW.json',review);write('SEAL.json',dict(completed=True,PASS=True,review=bind(H/'REVIEW.json'),scope=bind(H/'SCOPE.json'),code=bind(__file__)))
print(json.dumps(dict(PASS=True,review=bind(H/'REVIEW.json'),seal=bind(H/'SEAL.json'),selected_update=1000,criterion=criterion,old_public_VAL_triplets=review['old_public_VAL_triplets'],new_samewood_TRAIN_pairs=pair_result)))
