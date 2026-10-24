"""C16 duration-only exact C15 replay and continuation; inert until root entry."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C16_DESIGN_BEFORE_SOURCE_FIT.json';DESIGN_SHA='29fc0054eb395214e8b69e39acc6f2a3409de3605f55a1dc8df86c95f24a9fb9'
C15=P/'query_visibility_flat_pallet_augmented_head_20261023_v123/source_fit_preparation/train_source_once.py';C15_SHA='1ae7f8daada610257772a64df30c52b4fe41e604e05a8497c878570742b9ba1f'
assert hashlib.sha256(C15.read_bytes()).hexdigest()==C15_SHA
sp=importlib.util.spec_from_file_location('C16_exact_C15',C15);c15=importlib.util.module_from_spec(sp);sp.loader.exec_module(c15)
old=c15.old
sha,bind,check,read,rows,write=c15.sha,c15.bind,c15.check,c15.read,c15.rows,c15.write
BASE,NEW,N,OLDUNITS,NEWUNITS,UNITS=c15.BASE,c15.NEW,c15.N,c15.OLDUNITS,c15.NEWUNITS,c15.UNITS
join_flat,flat_pools,draw192,loss192,selector14,earliest_history,flat_report,load_old_metadata,load_feature_packet,state_sha=(getattr(c15,k)for k in('join_flat','flat_pools','draw192','loss192','selector14','earliest_history','flat_report','load_old_metadata','load_feature_packet','state_sha'))
OLD_PREFIX_SHA=c15.OLD_PREFIX_SHA
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FIT.json';RUN=V/'runs/C16_EXACT_C15_DURATION_CONTROL_8000'

def clone_cpu(x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:clone_cpu(v)for k,v in x.items()}
 if isinstance(x,list):return [clone_cpu(v)for v in x]
 if isinstance(x,tuple):return tuple(clone_cpu(v)for v in x)
 return x

def capture_rng(gens,newgen):
 return dict(old_cpu_generators=[g.get_state().clone()for g in gens],new_cpu_generator=newgen.get_state().clone(),torch_cpu=torch.get_rng_state().clone(),torch_cuda_all=[x.clone()for x in torch.cuda.get_rng_state_all()])

def validate_entry(path):
 assert not RUN.exists(),'One duration control only; no automatic restart.'
 e=read(path);assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
 for k in('trainer','protocol','design','generated','source_bindings'):check(e[k])
 assert e['trainer']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 protocol=read(PROTOCOL);proof=read(e['generated']['path']);assert proof['PASS']and proof['code']==e['trainer']and protocol['trainer']==e['trainer']and protocol['generated']==e['generated']and protocol['design']==e['design']
 duration=read(DESIGN)
 for v in duration.values():
  if isinstance(v,dict)and set(v)>={'path','sha256'}:check(v)
 assert duration['parent_C15_trainer']==bind(C15)and duration['training_control']['total_updates']==8000 and duration['training_control']['exact_replay_updates']==1000 and duration['training_control']['additional_updates']==7000
 assert duration['selection']['updates']==list(range(0,8001,100))and not duration['current_training_normalizer_threshold_checkpoint_arm_selection']and not duration['V112_allowed']
 design=read(duration['parent_C15_design']['path']);assert sha(duration['parent_C15_design']['path'])==c15.DESIGN_SHA
 for v in design.values():
  if isinstance(v,dict)and set(v)>={'path','sha256'}:check(v)
 assert e['source_bindings']==protocol['source_bindings']
 for k in('old_source_scores','new_baseline_predictions','old_trace','parent_C14_fit_complete'):check(protocol[k])
 return e,design,protocol

def run(entry_path):
 e,design,protocol=validate_entry(entry_path);duration=read(DESIGN);c15_history={r['update']:r for r in rows(duration['parent_C15_history']['path'])} # All source-only entry checks precede data/parameter loads.
 d14=read(design['parent_C14_design']['path']);p14=read(design['parent_C14_protocol']['path']);c12,c11,c10,c9,d=old.source_modules()
 meta,base,parent=load_old_metadata(e,d14,p14,c12,c10,d)
 inputs=rows(design['new_input_occurrences']['path']);gt=rows(design['new_GT_occurrences']['path']);new=join_flat(inputs,gt)
 op=load_feature_packet(design['old_features'],BASE,OLDUNITS);npk=load_feature_packet(design['new_features'],NEW,NEWUNITS)
 units=np.r_[op['occurrence_unit'],OLDUNITS+npk['occurrence_unit']].astype(np.int32)
 available=np.r_[op['query_available'],npk['query_available']];factual=np.r_[op['factual_query_available'],npk['factual_query_available']];context=np.r_[op['context_available'],npk['context_available']]
 target=np.r_[meta['external_target'],new['labels']].astype(np.int8);known=np.isin(target,[0,1]);split=np.r_[meta['split'],new['split']]
 assert len(meta['lineage'])==BASE and {s:int((new['typed_state']==s).sum())for s in'DEU'}==dict(D=2358,E=2070,U=468)
 assert np.array_equal(np.bincount(new['labels'][(new['split']=='TRAIN')&(new['labels']>=0)],minlength=2),[450,396])
 pools=c10.sampling_pools(base['old'],{k:base[k]for k in('known','split','external_target','typed_state','domain')},available[:old.BASE],d)
 mp=c12.material_pools(base,available[:old.BASE]);wp=old.available_pairs(meta,available[:BASE]);npools=flat_pools(new,available[BASE:])
 assert len(wp)==62
 selector14(c12,np.zeros(N,np.float32),available,meta,new,d)
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(211);torch.cuda.manual_seed_all(211)
 model=c11.SpatialTargetQueryHead().cuda().float();payload=torch.load(design['warmstart_checkpoint']['path'],map_location='cpu',weights_only=True)
 assert payload['update']==1200 and payload['head_class_code']==design['head_class_code']
 model.load_state_dict(payload['state_dict'],strict=True)
 assert all(torch.equal(v.detach().cpu(),payload['state_dict'][k])for k,v in model.state_dict().items())and sum(p.numel()for p in model.parameters())==221825
 del payload;initial_state_sha=state_sha(model)
 # Two separately allocated matrices retain the exact C14/V122 evaluation GEMM shapes/alignment.
 xo=torch.as_tensor(op['features'],device='cuda');xn=torch.as_tensor(npk['features'],device='cuda');xt=torch.cat((xo,xn));ug=torch.as_tensor(units,dtype=torch.long,device='cuda');ag=torch.as_tensor(available,device='cuda');tg=torch.as_tensor(target,dtype=torch.float32,device='cuda')
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);assert not optimizer.state
 gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];newgen=torch.Generator().manual_seed(218)
 history=[];best=float('inf');selected=None;best_state=None;updates=0;tracehash=hashlib.sha256();newhash=hashlib.sha256()
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),optimizer_updates=0,source_only=True,known_new_TRAIN_D=450,known_new_TRAIN_E=396,new_TRAIN_cues=npools['cues'].tolist(),new_teacherU=468,all51004_retained=True));start=time.monotonic()
 try:
  def checkpoint(step):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=clone_cpu(optimizer.state_dict()),RNG_states=capture_rng(gens,newgen),update=step,warmstart_parent_update=1200,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path),head_class_code=design['head_class_code'],threshold_logit=0,true_resume_state_at_update=step)
  @torch.no_grad()
  def predict():
   model.eval();raw=torch.cat((model(xo),model(xn)));return np.where(available,raw[ug].cpu().numpy(),0).astype(np.float32)
  @torch.no_grad()
  def evaluate(step):
   nonlocal best,selected,best_state
   z=predict();value,parts=selector14(c12,z,available,meta,new,d);row=dict(update=step,criterion_fourteen_VAL_cohorts_BCE=value,VAL_cohorts=parts);history.append(row)
   if step<=1000:assert row==c15_history[step],'C15 source history differs before continuation'
   safe=np.where(available,z,0).astype(np.float64);bceloss=np.logaddexp(0,safe)-np.where(known,target,0)*safe
   with(RUN/'ALL_SOURCE_BCE_CURVE.jsonl').open('a')as cf:cf.write(json.dumps(dict(update=step,known_TRAIN_mean_BCE=float(bceloss[known&(split=='TRAIN')].mean()),known_VALIDATION_mean_BCE=float(bceloss[known&(split=='VALIDATION')].mean()),VAL_cohorts=parts),allow_nan=False)+'\n')
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   if value<best:best,selected=value,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt')
   return z
  initial=predict()
  with np.load(protocol['old_source_scores']['path'],allow_pickle=False)as f:oldref=np.array(f['query_logit'],copy=True);oldav=np.array(f['query_available'],copy=True)
  with np.load(protocol['new_baseline_predictions']['path'],allow_pickle=False)as f:newref=np.array(f['query_logit'],copy=True);newav=np.array(f['query_available'],copy=True)
  parity=dict(old_scores_bitexact=np.array_equal(initial[:BASE],oldref),new_scores_bitexact=np.array_equal(initial[BASE:],newref),old_max_drift=float(np.max(np.abs(initial[:BASE]-oldref))),new_max_drift=float(np.max(np.abs(initial[BASE:]-newref))),old_available_exact=np.array_equal(available[:BASE],oldav),new_available_exact=np.array_equal(available[BASE:],newav),warmstart_state_exact=state_sha(model)==initial_state_sha,old_eval_units=15004,new_eval_units=4428,optimizer_updates=0)
  write(RUN/'UPDATE0_WARMSTART_PARITY.json',parity)
  assert all(parity[k]for k in('old_scores_bitexact','new_scores_bitexact','old_available_exact','new_available_exact','warmstart_state_exact')),'Preserve exact parity failure; never loosen tolerance silently.'
  torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');evaluate(0)
  optimizer_identity=id(optimizer);generator_identities=[id(g)for g in gens]+[id(newgen)];prefix1000=None;newprefix1000=None
  with Path(protocol['old_trace']['path']).open()as oldtrace,Path(duration['parent_C15_trace']['path']).open()as replaytrace:
   for step in range(1,8001):
    ix,pairs,audit,ix128,_,_,_,_=draw192(c12,c10,pools,mp,wp,npools,gens,newgen)
    if step<=2000:
     ref=json.loads(next(oldtrace));assert ref['update']==step and np.array_equal(ix128,ref['occurrence_indices'])and np.array_equal(pairs,ref['matched_cover_sham_batch_positions'])
    old.hash_draw(tracehash,ix128,pairs);newhash.update(ix[128:].astype('<i8').tobytes())
    assert known[ix].all()and available[ix].all()and(split[ix]=='TRAIN').all()and np.array_equal(np.bincount(target[ix],minlength=2),[96,96])
    assert np.array_equal(target[ix[pairs]],np.tile([1,0],(24,1)))and np.array_equal(ix[176:192],ix[128:144]+1)
    idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');pp=torch.as_tensor(pairs,dtype=torch.long,device='cuda');model.train();optimizer.zero_grad(set_to_none=True)
    z=model(xt[ug[idx]]);loss,bce,hinge,nbce=loss192(z,tg[idx],ag[idx],pp);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
    row=dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_cover_sham_batch_positions=pairs.tolist(),new_correspondence_positions_not_in_hinge=[[128+i,176+i]for i in range(16)],sampling=audit,available_loss_rows=192,loss=float(loss.detach()),old_BCE=float(bce.detach()),old_hinge=float(hinge.detach()),new_BCE=float(nbce.detach()),gradient_norm_before_clip=float(norm))
    with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    if step<=1000:
     replay=json.loads(next(replaytrace))
     if row!=replay:
      write(RUN/'REPLAY_STEP_FAILURE.json',dict(update=step,different_fields=[k for k in row if row[k]!=replay[k]],actual=row,expected=replay))
      raise AssertionError('C15 exact batch/loss replay mismatch')
    if step%100==0:lastz=evaluate(step)
    if step==1000:
     prefix1000=tracehash.hexdigest();newprefix1000=newhash.hexdigest();assert prefix1000==OLD_PREFIX_SHA
     saved=torch.load(duration['parent_C15_checkpoint']['path'],map_location='cpu',weights_only=True);assert saved['update']==1000
     parameters_exact=all(torch.equal(v.detach().cpu(),saved['state_dict'][k])for k,v in model.state_dict().items());del saved;c15_state_sha_at1000=state_sha(model)
     with np.load(duration['parent_C15_scores']['path'],allow_pickle=False)as rf:score_exact=np.array_equal(lastz,rf['query_logit'])and np.array_equal(available,rf['query_available'])
     bce_exact=history[-1]==c15_history[1000]
     continuous_state=id(optimizer)==optimizer_identity and [id(g)for g in gens]+[id(newgen)]==generator_identities and all(float(v['step'])==1000 for v in optimizer.state.values())
     cert=dict(completed=True,PASS=bool(parameters_exact and score_exact and bce_exact and continuous_state),replayed_updates=1000,batches_exact=True,losses_exact=True,step1000_parameters_exact=parameters_exact,step1000_scores_exact=score_exact,step1000_fourteen_BCE_exact=bce_exact,history0_to1000_exact=True,optimizer_and_RNG_not_reset_at1000=continuous_state,old128_first1000_draws_sha256=prefix1000,new64_first1000_draws_sha256=newprefix1000,parent_C15_fit=duration['parent_C15_fit'],optimizer_state_certification='Recovered by deterministic full batch/loss/parameter replay; original C15 optimizer state was not persisted.')
     write(RUN/'C15_REPLAY1000_CERTIFICATE.json',cert);assert cert['PASS'],'Exact C15 boundary mismatch; no continuation allowed'
     torch.save(checkpoint(1000),RUN/'REPLAY1000_RESUME_CHECKPOINT.pt')
    if step==2000:assert tracehash.hexdigest()==old.FULL128_SHA
  assert prefix1000==OLD_PREFIX_SHA and len(history)==81 and earliest_history(history)==(selected,best)
  torch.save(checkpoint(8000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);z=predict();value,parts=selector14(c12,z,available,meta,new,d);assert value==best
  oldmetrics,gates=c9.report(d,z[:33984],available[:33984],meta['old'],dict(d14,source_gates={k:v for k,v in d14['source_gates'].items()if not k.startswith('material_')}))
  material,mgates=c12.material_report(d,z[:old.BASE],available[:old.BASE],base,parent);gates.update(mgates);assert len(gates)==15 and set(gates)==set(d14['source_gates'])
  newreport=flat_report(z[BASE:],available[BASE:],new);write(RUN/'NEW_FLAT_SOURCE_RESULTS.json',newreport)
  line=meta['lineage']+new['lineage'];q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line]);Ls=np.array([r['L']if r['L']is not None else np.nan for r in line])
  ids=np.array([json.dumps({k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')},sort_keys=True,separators=(',',':'),allow_nan=False)for r in line])
  states=np.r_[meta['typed_state'],new['typed_state']];sources=np.r_[meta['source'],new['source']];variants=np.r_[meta['variant'],new['variant']];domains=np.r_[meta['domain'],new['domain']]
  np.savez_compressed(RUN/'SELECTED_ALL51004_SOURCE_SCORES.npz',query_logit=z,query_available=available,factual_query_available=factual,context_available=context,prediction=d.decode(z,available),input_unit_index=units,global_occurrence_index=np.arange(N),raw_teacher_state=states,external_target=target,split=split,source=sources,variant=variants,domain=domains,query_xy_raw=q,L_raw=Ls,original_identifier_metadata_json=ids,original_split=np.array([r.get('raw_split')or r.get('original_split')or split[i]for i,r in enumerate(line)]),context_Q_inside=np.r_[op['unit_Q_inside'],npk['unit_Q_inside']][units],old_accepted_triplets=meta['old']['old_pairs'],new_selected_triplets=meta['old']['new_pairs'],samewood_accepted_cover_sham_pairs=meta['new_pairs2'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,gates=gates,all_gates_pass=all(v['PASS']for v in gates.values()),gate_scope='Exact existing15 C14 source regression definitions; failed source model never promoted.',current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_fourteen_VAL_cohorts_BCE=best,VAL_cohorts=parts,old_metrics=oldmetrics,material_surface=material,old_additive_reports=old.additive_reports(z[:BASE],available[:BASE],meta,d),new_flat_source_report=bind(RUN/'NEW_FLAT_SOURCE_RESULTS.json'),all51004_retained=True,old128_first1000_draws_sha256=prefix1000,new64_first1000_draws_sha256=newprefix1000,old128_all8000_draws_sha256=tracehash.hexdigest(),new64_all8000_draws_sha256=newhash.hexdigest(),source_VAL_development_exposed=True,TOMIE_TRAIN=0,known_unavailable={s:int((known&(split==s)&~available).sum())for s in('TRAIN','VALIDATION')},native_mapping='UNKNOWN',human_gold=False,learned_self_classification=False))
  assert sha(design['warmstart_checkpoint']['path'])==design['warmstart_checkpoint']['sha256']
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),head_class_code=design['head_class_code'],trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),warmstart_checkpoint=design['warmstart_checkpoint'],old_feature_pack=design['old_features'],new_feature_pack=design['new_features'],new_feature_receipt=design['new_feature_receipt'],optimizer_and_RNG_resume_saved=True,total_optimizer_updates=8000,replay_certificate=bind(RUN/'C15_REPLAY1000_CERTIFICATE.json'),selected_weights_exact_C15=state_sha(model)==c15_state_sha_at1000,parent_C15_freeze=duration['parent_C15_freeze'],current_used_for_fit_or_selection=False,current_inference_authorized=False,V112_inference_authorized=False,PnP=0))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=8000,total_updates=8000,replayed_C15_updates=1000,continued_updates=7000,replay_certificate=bind(RUN/'C15_REPLAY1000_CERTIFICATE.json'),selected_update=selected,criterion_fourteen_VAL_cohorts_BCE=best,all_source_gates_pass=all(v['PASS']for v in gates.values()),seconds=time.monotonic()-start,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL51004_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),final_resume_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),source_learning_curve=bind(RUN/'ALL_SOURCE_BCE_CURVE.jsonl'),update0_parity=bind(RUN/'UPDATE0_WARMSTART_PARITY.json'),new_flat_source_results=bind(RUN/'NEW_FLAT_SOURCE_RESULTS.json'),current_V112_or_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise


def prepare():
 assert sha(DESIGN)==DESIGN_SHA
 duration=read(DESIGN);parent=read(check(duration['parent_C15_protocol']));proof=H/'GENERATED_TRAINER_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 protocol=dict(parent,trainer=bind(__file__),design=bind(DESIGN),generated=bind(proof),parent_C15=bind(C15),parent_C15_design=duration['parent_C15_design'],duration_total8000_replay1000_continue7000=True,run_directory=str(RUN),actual_fit_or_head_call=0)
 write(PROTOCOL,protocol);template=dict(approved_source_fit=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_after_update=False,trainer=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),generated=bind(proof),source_bindings=protocol['source_bindings'])
 write(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',template)
 write(H/'PREPARATION.json',dict(completed=True,trainer=bind(__file__),protocol=bind(PROTOCOL),generated=bind(proof),template=bind(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),run_directory=str(RUN),command=protocol['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',actual_models_parameters_features_pixels_fit_current_V112=0))
 print(json.dumps(read(H/'PREPARATION.json')),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path);ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_EXACT_ENTRY.json')
