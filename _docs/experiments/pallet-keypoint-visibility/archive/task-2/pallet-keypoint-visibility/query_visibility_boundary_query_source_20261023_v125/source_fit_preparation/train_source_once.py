"""C17: exact C15 old192 plus source-only wide64; inert until exact root entry."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DECISIONS=V/'ROOT_C17_DECISIONS_BEFORE_BASELINE_READ.json'
DECISIONS_SHA='623239107ad7ba27a25038af99ff25561824b89bdd2c7ff7287e0efedf4c217d'
DESIGN=V/'ROOT_C17_DESIGN_BEFORE_SOURCE_FIT.json'
DESIGN_SHA='4a415c196acb6748a1883c0115bfe188e3690cab79211caf5b3badfc9c65529a'
C15=P/'query_visibility_flat_pallet_augmented_head_20261023_v123/source_fit_preparation/train_source_once.py'
C15_SHA='1ae7f8daada610257772a64df30c52b4fe41e604e05a8497c878570742b9ba1f'
assert hashlib.sha256(C15.read_bytes()).hexdigest()==C15_SHA
sp=importlib.util.spec_from_file_location('C17_exact_C15',C15);c15=importlib.util.module_from_spec(sp);sp.loader.exec_module(c15)
old=c15.old
sha,bind,check,read,rows,write=c15.sha,c15.bind,c15.check,c15.read,c15.rows,c15.write
BASE,WIDE,N=51004,4896,55900
OLDUNITS,FLATUNITS,WIDEUNITS,UNITS=15004,4428,4428,23860
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FIT.json';RUN=V/'runs/C17_QUERY_BOUNDARY_AUGMENTED_SPATIAL_QUERY'
STRATA=(('E_alpha_outside',16),('E_alpha_inside',16),('D_SHAM_alpha_inside',16),('D_SHAM_alpha_outside',8),('D_BEFORE',8))

def join_wide(inputs,gt):
 w=c15.join_flat(inputs,gt);w['domain']=np.full(WIDE,'V125_PUBLIC_QUERY_BOUNDARY')
 w['alpha']=np.array([-1 if g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt],np.int8)
 w['pair_ready']=np.array([g['geometry_pair_ready']for g in gt],bool)
 w['nearest_q_in_image']=np.array([g['nearest_q_in_image_GT_diagnostic_only']for g in gt],bool)
 assert set(np.unique(w['alpha']))<=set((-1,0,1))
 for i,(r,g)in enumerate(zip(inputs,gt)):
  assert g['same_expected_weak_cue_identity_under_query_shift'] and g['uncertainty_support_unchanged']
  if w['alpha'][i]>=0:assert r['variant']!='BEFORE'and w['pair_ready'][i]
 return w

def wide_pools(w,a):
 eligible=(w['split']=='TRAIN')&np.isin(w['labels'],[0,1])&a
 masks={'E_alpha_outside':(w['labels']==1)&w['pair_ready']&(w['variant']=='OPAQUE_COVER')&(w['alpha']==0),
        'E_alpha_inside':(w['labels']==1)&w['pair_ready']&(w['variant']=='OPAQUE_COVER')&(w['alpha']==1),
        'D_SHAM_alpha_inside':(w['labels']==0)&w['pair_ready']&(w['variant']=='CUE_PRESERVING_SHAM')&(w['alpha']==1),
        'D_SHAM_alpha_outside':(w['labels']==0)&w['pair_ready']&(w['variant']=='CUE_PRESERVING_SHAM')&(w['alpha']==0),
        'D_BEFORE':(w['labels']==0)&(w['variant']=='BEFORE')}
 pools={k:BASE+np.flatnonzero(eligible&m)for k,m in masks.items()}
 assert all(len(pools[k])for k,_ in STRATA),'Empty prospective wide stratum; no fallback/replacement stratum.'
 return pools

def draw_wide64(pools,generator):
 parts=[pools[k][torch.randint(len(pools[k]),(n,),generator=generator).numpy()]for k,n in STRATA]
 return np.concatenate(parts).astype(np.int64),{k:p.tolist()for(k,_),p in zip(STRATA,parts)}

def loss256(z,target,available,pairs):
 assert z.shape==target.shape==available.shape==(256,)and pairs.shape==(24,2)and int(pairs.max())<128
 oldloss,obce,hinge,flatbce=c15.loss192(z[:192],target[:192],available[:192],pairs)
 eligible=available[192:]&((target[192:]==0)|(target[192:]==1))
 widebce=torch.nn.functional.binary_cross_entropy_with_logits(z[192:][eligible],target[192:][eligible])if eligible.any()else z[192:].sum()*0
 return oldloss+widebce,obce,hinge,flatbce,widebce

def selector18(c12,z,a,meta,flat,wide,d):
 _,parts=c15.selector14(c12,z[:BASE],a[:BASE],meta,flat,d);assert len(parts)==14
 safe=np.where(a[BASE:],z[BASE:],0).astype(np.float64);bce=np.logaddexp(0,safe)-np.where(wide['labels']>=0,wide['labels'],0)*safe
 for source,name in(('LOCO','WIDE_LOCO_VALIDATION'),('TOMIE','WIDE_TOMIE_DEVELOPMENT')):
  for state in('D','E'):
   m=(wide['source']==source)&(wide['split']=='VALIDATION')&(wide['typed_state']==state)
   assert m.any(),'Empty prospective source selector cohort'
   parts[name+'_'+state]=dict(count=int(m.sum()),mean_BCE=float(bce[m].mean()),unavailable=int((m&~a[BASE:]).sum()))
 assert len(parts)==18
 return float(np.mean([p['mean_BCE']for p in parts.values()])),parts

def earliest_history(history):
 best=float('inf');step=None
 for r in history:
  if r['criterion_eighteen_VAL_cohorts_BCE']<best:best=r['criterion_eighteen_VAL_cohorts_BCE'];step=r['update']
 return step,best

def wide_report(z,a,factual,w):
 y=w['labels'];pred=np.where(a,(z>=0).astype(np.int8),-1);known=np.isin(y,[0,1]);rs=w['lineage'];groups={};checks={}
 div=lambda x,n:float(x/n)if n else None
 for source,fold in [('ALL','ALL')]+sorted({(r['source'],r['raw_split'])for r in rs}):
  m=np.array([(source=='ALL'or r['source']==source)and(fold=='ALL'or r['raw_split']==fold)for r in rs]);name=source+'::'+fold
  centre=np.array([r['offset_xy']==[0,0]for r in rs])
  group=dict(all_planned=c15.flat_metric(y[m],pred[m]),factual_available=c15.flat_metric(y[m&factual],pred[m&factual]),model_input_available=c15.flat_metric(y[m&a],pred[m&a]),centre_offset0=c15.flat_metric(y[m&centre],pred[m&centre]),known_input_U=int((m&known&~a).sum()),teacher_U_available=int((m&~known&a).sum()),alpha_interactions={})
  for variant in('OPAQUE_COVER','CUE_PRESERVING_SHAM'):
   for alpha,label in((0,'q_alpha_outside'),(1,'q_alpha_inside'),(-1,'alpha_undefined')):
    b=m&(w['variant']==variant)&(w['alpha']==alpha)
    group['alpha_interactions'][variant+'::'+label]=dict(all_planned=c15.flat_metric(y[b],pred[b]),factual_available=c15.flat_metric(y[b&factual],pred[b&factual]),model_input_available=c15.flat_metric(y[b&a],pred[b&a]),nearest_q_in_image=int((b&w['nearest_q_in_image']).sum()))
  groups[name]=group
  if (source,fold)in(('LOCO','VALIDATION'),('TOMIE','TOMIE_DEVELOPMENT_STRESS')):
   D=m&(y==0);E=m&(y==1);EO=E&w['pair_ready']&(w['variant']=='OPAQUE_COVER')&(w['alpha']==0);SI=D&w['pair_ready']&(w['variant']=='CUE_PRESERVING_SHAM')&(w['alpha']==1)
   definitions=(('D_recall',D,pred==0,.85,'min'),('E_recall',E,pred==1,.75,'min'),('known_model_U_rate',m&known,pred<0,.05,'max'),('E_q_alpha_outside_recall',EO,pred==1,.75,'min'),('SHAM_D_q_alpha_inside_recall',SI,pred==0,.75,'min'))
   for key,mask,correct,threshold,direction in definitions:
    value=div(int((mask&correct).sum()),int(mask.sum()));passed=value is not None and(value>=threshold if direction=='min'else value<=threshold)
    checks[name+'::'+key]=dict(value=value,numerator=int((mask&correct).sum()),denominator=int(mask.sum()),threshold=threshold,direction=direction,PASS=bool(passed))
 return dict(completed=True,all_rows=WIDE,groups=groups,prospective_checks=checks,all_prospective_checks_pass=all(r['PASS']for r in checks.values()),native_mapping='UNKNOWN',human_gold=False,weak_region_component_visibility_not_q_pixel_gold=True,source_VAL_development_exposed=True)

def clone_cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:clone_cpu(v)for k,v in x.items()}
 if isinstance(x,list):return [clone_cpu(v)for v in x]
 if isinstance(x,tuple):return tuple(clone_cpu(v)for v in x)
 return x

def capture_rng(gens,flatgen,widegen):
 return dict(old_cpu_generators=[g.get_state().clone()for g in gens],flat_cpu_generator=flatgen.get_state().clone(),wide_cpu_generator=widegen.get_state().clone(),torch_cpu=torch.get_rng_state().clone(),torch_cuda_all=[s.clone()for s in torch.cuda.get_rng_state_all()])

def validate_entry(path):
 assert not RUN.exists(),'One fit only; preserve failure and never automatically restart.'
 e=read(path);assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
 for k in('trainer','protocol','design','generated','source_bindings'):check(e[k])
 assert e['trainer']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 assert sha(DECISIONS)==DECISIONS_SHA
 p=read(PROTOCOL);d=read(DESIGN);proof=read(e['generated']['path'])
 assert p['trainer']==e['trainer']and p['design']==e['design']and p['generated']==e['generated']and proof['PASS']and proof['code']==e['trainer']and p['decisions']==bind(DECISIONS)
 assert e['source_bindings']==p['source_bindings']==d['source_bindings']and d['decisions_before_baseline_read']==bind(DECISIONS)
 for v in d.values():
  if isinstance(v,dict)and set(v)>={'path','sha256'}:check(v)
 assert d['parent_C15_trainer']==bind(C15)
 freeze=read(d['warmstart_model_freeze']['path']);assert freeze['completed']and freeze['selected_update']==4500 and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']and freeze['checkpoint']==d['warmstart_checkpoint']and freeze['head_class_code']==d['head_class_code']
 for k,n,u in(('old',46108,OLDUNITS),('old_flat',4896,FLATUNITS),('wide',WIDE,WIDEUNITS)):
  receipt=read(d[k+'_feature_receipt']['path']);assert receipt['completed']and receipt['PASS']and receipt['output']==d[k+'_features']and receipt['features_shape']==[u,1928]and receipt['occurrences']==n and receipt['model_state_unchanged']and receipt['first_batch_repeat_exact']
 for k in('old_source_scores','wide_baseline_predictions','old192_trace'):check(d[k])
 return e,d,p

def run(entry_path):
 e,design,protocol=validate_entry(entry_path) # Source-only checks precede parameter/feature/GT body loads.
 d15=read(design['parent_C15_design']['path']);d14=read(d15['parent_C14_design']['path']);p14=read(d15['parent_C14_protocol']['path']);c12,c11,c10,c9,d=old.source_modules()
 meta,base,parent=c15.load_old_metadata(e,d14,p14,c12,c10,d)
 assert design['old_flat_input_occurrences']==d15['new_input_occurrences']and design['old_flat_GT_occurrences']==d15['new_GT_occurrences']
 flat=c15.join_flat(rows(design['old_flat_input_occurrences']['path']),rows(design['old_flat_GT_occurrences']['path']))
 wide=join_wide(rows(design['wide_input_occurrences']['path']),rows(design['wide_GT_occurrences']['path']))
 op=c15.load_feature_packet(design['old_features'],46108,OLDUNITS);fp=c15.load_feature_packet(design['old_flat_features'],4896,FLATUNITS);wp=c15.load_feature_packet(design['wide_features'],WIDE,WIDEUNITS)
 units=np.r_[op['occurrence_unit'],OLDUNITS+fp['occurrence_unit'],OLDUNITS+FLATUNITS+wp['occurrence_unit']].astype(np.int32)
 a=np.r_[op['query_available'],fp['query_available'],wp['query_available']];factual=np.r_[op['factual_query_available'],fp['factual_query_available'],wp['factual_query_available']];context=np.r_[op['context_available'],fp['context_available'],wp['context_available']]
 target=np.r_[meta['external_target'],flat['labels'],wide['labels']].astype(np.int8);known=np.isin(target,[0,1]);split=np.r_[meta['split'],flat['split'],wide['split']]
 assert len(target)==N and len(meta['lineage'])==46108 and sum(~wp['query_available'])==51 and sum(wide['labels']<0)==468
 assert {s:int(((wide['labels']==s)&~wp['query_available']).sum())for s in(0,1)}=={0:27,1:24}
 for i,r in enumerate(wide['lineage']):assert bool(wp['query_available'][i])==r['query_available']and bool(wp['factual_query_available'][i])==r['factual_query_available']and bool(wp['context_available'][i])==r['context_available']
 pools=c10.sampling_pools(base['old'],{k:base[k]for k in('known','split','external_target','typed_state','domain')},a[:old.BASE],d);mp=c12.material_pools(base,a[:old.BASE]);woodp=old.available_pairs(meta,a[:46108]);flatp=c15.flat_pools(flat,a[46108:BASE]);widep=wide_pools(wide,a[BASE:])
 selector18(c12,np.zeros(N,np.float32),a,meta,flat,wide,d)
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(219);torch.cuda.manual_seed_all(219)
 model=c11.SpatialTargetQueryHead().cuda().float();payload=torch.load(design['warmstart_checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==4500 and payload['head_class_code']==design['head_class_code'];model.load_state_dict(payload['state_dict'],strict=True)
 assert all(torch.equal(v.detach().cpu(),payload['state_dict'][k])for k,v in model.state_dict().items())and sum(p.numel()for p in model.parameters())==221825
 del payload;initial_state_sha=c15.state_sha(model)
 # Preserve each saved evaluation GEMM shape/alignment; never combine the three evaluation forwards.
 xo=torch.as_tensor(op['features'],device='cuda');xf=torch.as_tensor(fp['features'],device='cuda');xw=torch.as_tensor(wp['features'],device='cuda');xt=torch.cat((xo,xf,xw));ug=torch.as_tensor(units,device='cuda',dtype=torch.long);tg=torch.as_tensor(target,device='cuda',dtype=torch.float32);ag=torch.as_tensor(a,device='cuda')
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);assert not optimizer.state
 gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];flatgen=torch.Generator().manual_seed(218);widegen=torch.Generator().manual_seed(220)
 history=[];best=float('inf');selected=None;best_state=None;updates=0;oldhash=hashlib.sha256();widehash=hashlib.sha256();RUN.mkdir(parents=True);start=time.monotonic()
 write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),source_only=True,optimizer_updates=0,wide_pools={k:len(v)for k,v in widep.items()},wide_teacher_U=468,wide_input_U=51,all55900_retained=True))
 try:
  def checkpoint(step):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=clone_cpu(optimizer.state_dict()),RNG_states=capture_rng(gens,flatgen,widegen),update=step,warmstart_parent_update=4500,trainer=bind(__file__),design=bind(DESIGN),entry=bind(entry_path),head_class_code=design['head_class_code'],threshold_logit=0,true_resume_state_at_update=step)
  @torch.no_grad()
  def predict():
   model.eval();raw=torch.cat((model(xo),model(xf),model(xw)));return np.where(a,raw[ug].cpu().numpy(),0).astype(np.float32)
  @torch.no_grad()
  def evaluate(step):
   nonlocal best,selected,best_state
   z=predict();value,parts=selector18(c12,z,a,meta,flat,wide,d);row=dict(update=step,criterion_eighteen_VAL_cohorts_BCE=value,VAL_cohorts=parts);history.append(row)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   if value<best:best,selected=value,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt')
   return z
  initial=predict()
  with np.load(design['old_source_scores']['path'],allow_pickle=False)as f:oldref=f['query_logit'];oldav=f['query_available']
  with np.load(design['wide_baseline_predictions']['path'],allow_pickle=False)as f:wideref=f['query_logit'];wideav=f['query_available']
  parity=dict(old51004_scores_bitexact=np.array_equal(initial[:BASE],oldref),wide4896_scores_bitexact=np.array_equal(initial[BASE:],wideref),old_max_drift=float(np.max(np.abs(initial[:BASE]-oldref))),wide_max_drift=float(np.max(np.abs(initial[BASE:]-wideref))),old_available_exact=np.array_equal(a[:BASE],oldav),wide_available_exact=np.array_equal(a[BASE:],wideav),warmstart_state_exact=c15.state_sha(model)==initial_state_sha,separate_evaluation_units=[OLDUNITS,FLATUNITS,WIDEUNITS],optimizer_updates=0)
  write(RUN/'UPDATE0_WARMSTART_PARITY.json',parity);assert all(parity[k]for k in('old51004_scores_bitexact','wide4896_scores_bitexact','old_available_exact','wide_available_exact','warmstart_state_exact')),'Preserve update0 parity failure, no silent tolerance relaxation.'
  torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');evaluate(0)
  with Path(design['old192_trace']['path']).open()as trace:
   for step in range(1,1001):
    ix192,pairs,audit,_,_,_,_,_=c15.draw192(c12,c10,pools,mp,woodp,flatp,gens,flatgen);extra,waudit=draw_wide64(widep,widegen);ix=np.r_[ix192,extra]
    ref=json.loads(next(trace));assert ref['update']==step and np.array_equal(ix192,ref['occurrence_indices'])and np.array_equal(pairs,ref['matched_cover_sham_batch_positions'])
    oldhash.update(ix192.astype('<i8').tobytes());oldhash.update(pairs.astype('<i8').tobytes());widehash.update(extra.astype('<i8').tobytes())
    assert len(ix)==256 and known[ix].all()and a[ix].all()and(split[ix]=='TRAIN').all()and np.array_equal(np.bincount(target[ix],minlength=2),[128,128])
    assert np.array_equal(target[ix[pairs]],np.tile([1,0],(24,1)))and np.array_equal(ix[176:192],ix[128:144]+1)
    idx=torch.as_tensor(ix,device='cuda',dtype=torch.long);pp=torch.as_tensor(pairs,device='cuda',dtype=torch.long);model.train();optimizer.zero_grad(set_to_none=True);z=model(xt[ug[idx]]);loss,ob,hinge,fb,wb=loss256(z,tg[idx],ag[idx],pp);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
    row=dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_cover_sham_batch_positions=pairs.tolist(),sampling=dict(old192=audit,wide64=waudit),available_loss_rows=256,loss=float(loss.detach()),old128_BCE=float(ob.detach()),old24_hinge=float(hinge.detach()),oldflat64_BCE=float(fb.detach()),wide64_BCE=float(wb.detach()),gradient_norm_before_clip=float(norm))
    with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    if step%100==0:evaluate(step)
  assert len(history)==11 and earliest_history(history)==(selected,best)
  torch.save(checkpoint(1000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);z=predict();value,parts=selector18(c12,z,a,meta,flat,wide,d);assert value==best
  oldmetrics,gates=c9.report(d,z[:33984],a[:33984],meta['old'],dict(d14,source_gates={k:v for k,v in d14['source_gates'].items()if not k.startswith('material_')}));material,mgates=c12.material_report(d,z[:old.BASE],a[:old.BASE],base,parent);gates.update(mgates);assert len(gates)==15 and set(gates)==set(d14['source_gates'])
  flatreport=c15.flat_report(z[46108:BASE],a[46108:BASE],flat);wr=wide_report(z[BASE:],a[BASE:],factual[BASE:],wide);write(RUN/'OLD_FLAT_SOURCE_RESULTS.json',flatreport);write(RUN/'WIDE_SOURCE_RESULTS.json',wr)
  line=meta['lineage']+flat['lineage']+wide['lineage'];q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line]);Ls=np.array([r['L']if r['L']is not None else np.nan for r in line]);ids=np.array([json.dumps({k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')},sort_keys=True,separators=(',',':'),allow_nan=False)for r in line])
  np.savez_compressed(RUN/'SELECTED_ALL55900_SOURCE_SCORES.npz',query_logit=z,query_available=a,factual_query_available=factual,context_available=context,prediction=d.decode(z,a),input_unit_index=units,global_occurrence_index=np.arange(N),raw_teacher_state=np.r_[meta['typed_state'],flat['typed_state'],wide['typed_state']],external_target=target,split=split,source=np.r_[meta['source'],flat['source'],wide['source']],variant=np.r_[meta['variant'],flat['variant'],wide['variant']],domain=np.r_[meta['domain'],flat['domain'],wide['domain']],query_xy_raw=q,L_raw=Ls,original_identifier_metadata_json=ids,original_split=np.array([r.get('raw_split')or r.get('original_split')or split[i]for i,r in enumerate(line)]),context_Q_inside=np.r_[op['unit_Q_inside'],fp['unit_Q_inside'],wp['unit_Q_inside']][units],old_accepted_triplets=meta['old']['old_pairs'],new_selected_triplets=meta['old']['new_pairs'],samewood_accepted_cover_sham_pairs=meta['new_pairs2'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,gates=gates,all_gates_pass=all(v['PASS']for v in gates.values()),gate_scope='Exact old15 regression gates unchanged; prospective wide diagnostics reported separately.',wide_source_diagnostics=wr['prospective_checks'],all_wide_checks_pass=wr['all_prospective_checks_pass'],current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_eighteen_VAL_cohorts_BCE=best,VAL_cohorts=parts,old_metrics=oldmetrics,material_surface=material,old_additive_reports=old.additive_reports(z[:46108],a[:46108],meta,d),old_flat_source_report=bind(RUN/'OLD_FLAT_SOURCE_RESULTS.json'),wide_source_report=bind(RUN/'WIDE_SOURCE_RESULTS.json'),all55900_retained=True,old192_all1000_draws_sha256=oldhash.hexdigest(),wide64_all1000_draws_sha256=widehash.hexdigest(),source_VAL_development_exposed=True,TOMIE_TRAIN=0,known_unavailable={s:int((known&(split==s)&~a).sum())for s in('TRAIN','VALIDATION')},native_mapping='UNKNOWN',human_gold=False,learned_self_classification=False))
  assert sha(design['warmstart_checkpoint']['path'])==design['warmstart_checkpoint']['sha256']
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),head_class_code=design['head_class_code'],trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),warmstart_checkpoint=design['warmstart_checkpoint'],parent_C16_freeze=design['warmstart_model_freeze'],old_feature_pack=design['old_features'],old_flat_feature_pack=design['old_flat_features'],wide_feature_pack=design['wide_features'],wide_feature_receipt=design['wide_feature_receipt'],optimizer_and_RNG_resume_saved=True,total_optimizer_updates=1000,selected_weights_unchanged_parent=c15.state_sha(model)==initial_state_sha,current_used_for_fit_or_selection=False,current_inference_authorized=False,V112_inference_authorized=False,PnP=0))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=1000,selected_update=selected,criterion_eighteen_VAL_cohorts_BCE=best,all_source_gates_pass=all(v['PASS']for v in gates.values()),all_wide_checks_pass=wr['all_prospective_checks_pass'],seconds=time.monotonic()-start,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL55900_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),final_resume_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),update0_parity=bind(RUN/'UPDATE0_WARMSTART_PARITY.json'),wide_source_results=bind(RUN/'WIDE_SOURCE_RESULTS.json'),current_V112_or_PnP=0));print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise

def prepare():
 assert sha(DECISIONS)==DECISIONS_SHA and sha(DESIGN)==DESIGN_SHA
 proof=H/'GENERATED_TRAINER_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 parent=read(C15.parent/'FIT_PROTOCOL_BEFORE_SOURCE_FIT.json')
 p=dict(completed=True,trainer=bind(__file__),decisions=bind(DECISIONS),generated=bind(proof),parent_C15_trainer=bind(C15),source_bindings=parent['source_bindings'],design_path=str(DESIGN),design_binding_pending=not DESIGN.exists(),feature_units=[OLDUNITS,FLATUNITS,WIDEUNITS],feature_dimension=1928,occurrences=N,runtime=parent['runtime'],actual_fit_model_feature_or_parameter_loads=0,run_directory=str(RUN),source_only_entry_before_data_and_parameters=True)
 if DESIGN.exists():p['design']=bind(DESIGN)
 write(PROTOCOL,p);write(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',dict(approved_source_fit=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_after_update=False,trainer=bind(__file__),protocol=bind(PROTOCOL),design=p.get('design',dict(path=str(DESIGN),sha256='ROOT_FILL_BEFORE_FIT')),generated=bind(proof),source_bindings=p['source_bindings']))
 write(H/'PREPARATION.json',dict(completed=True,trainer=bind(__file__),protocol=bind(PROTOCOL),generated=bind(proof),template=bind(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),run_directory=str(RUN),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',actual_feature_parameter_model_fit_current_V112=0));print(json.dumps(read(H/'PREPARATION.json')),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_EXACT_ENTRY.json')
