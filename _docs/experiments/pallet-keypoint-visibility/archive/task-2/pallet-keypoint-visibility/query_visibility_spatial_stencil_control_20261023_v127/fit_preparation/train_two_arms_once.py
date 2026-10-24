"""Root-only matched fixed1000 source fits. No Torch/model/data I/O at import."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=module(V/'stencil_common.py','v127_fit_common')
MODEL=H/'stencil_head.py';PROTOCOL=H/'FIT_PROTOCOL_BEFORE_FIT.json';OUT=V/'runs/C19_MATCHED_STENCIL_FIXED1000'
C18=P/'query_visibility_wide_only_loss_diagnostic_20261023_v126/source_fit_preparation/train_source_once.py';C18_SHA='6757420aca27346360392c2728632cd4d046c40340548fa7664d0f6153aae76f'
ARMS=('POOLED9_REPEAT8','ORDERED_OUTER8');STRATA=(('E_alpha_outside',16),('E_alpha_inside',16),('D_SHAM_alpha_inside',16),('D_SHAM_alpha_outside',8),('D_BEFORE',8))

def metric(y,p):
 cm=np.array([[int(((y==a)&(p==b)).sum())for b in(0,1,-1)]for a in(0,1)],np.int64);div=lambda x,n:float(x/n)if n else None
 return dict(all_rows=len(y),teacher_D=int((y==0).sum()),teacher_E=int((y==1).sum()),teacher_U=int((y<0).sum()),CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),known_model_U=int(cm[:,2].sum()),teacher_U_model_available=int(((y<0)&(p>=0)).sum()))

def report(z,a,factual,rs,gt):
 y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gt],np.int8);pred=np.where(a,(z>=0).astype(np.int8),-1);alpha=np.array([-1 if g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt]);groups={}
 for source,fold in [('ALL','ALL')]+sorted({(r['source'],r['raw_split'])for r in rs}):
  keep=np.array([(source=='ALL'or r['source']==source)and(fold=='ALL'or r['raw_split']==fold)for r in rs]);variants=np.array([r['variant']for r in rs]);centre=np.array([r['offset_xy']==[0,0]for r in rs]);g=dict(all_planned=metric(y[keep],pred[keep]),factual_available=metric(y[keep&factual],pred[keep&factual]),model_input_available=metric(y[keep&a],pred[keep&a]),centre_offset0=metric(y[keep&centre],pred[keep&centre]),by_variant={k:metric(y[keep&(variants==k)],pred[keep&(variants==k)])for k in('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')},alpha_interactions={})
  for variant in('OPAQUE_COVER','CUE_PRESERVING_SHAM'):
   for value,name in((0,'q_alpha_outside'),(1,'q_alpha_inside'),(-1,'alpha_undefined')):
    mask=keep&(variants==variant)&(alpha==value);g['alpha_interactions'][variant+'::'+name]=dict(all_planned=metric(y[mask],pred[mask]),factual_available=metric(y[mask&factual],pred[mask&factual]))
  groups[source+'::'+fold]=g
 return dict(completed=True,groups=groups,all4896_retained=True,threshold_logit=0,no_VAL_checkpoint_or_arm_choice=True,old_cohorts_not_scored=True,old15_regression_PASS_claimed=False,conditional_AUX_not_native_human_pixel_gold=True,source_VAL_repeated_development_exposure=True)

def clone_cpu(torch,x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return{k:clone_cpu(torch,v)for k,v in x.items()}
 if isinstance(x,list):return[clone_cpu(torch,v)for v in x]
 if isinstance(x,tuple):return tuple(clone_cpu(torch,v)for v in x)
 return x

def authorize(entry):
 e=c.read(entry);assert e['approved_source_two_arm_fit']is True and e['source_only']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','automatic_restart'))
 for k in('code','model_code','common_code','protocol','policy','precision_addendum','generated','feature_receipt','stencil_features'):c.check(e[k])
 root=c.policy();p=c.read(PROTOCOL);proof=c.read(e['generated']['path'])
 assert e['code']==p['code']==c.bind(__file__)and e['model_code']==p['model_code']==c.bind(MODEL)and e['common_code']==p['common_code']==c.bind(V/'stencil_common.py')and e['policy']==p['policy']==c.bind(c.POLICY)and e['precision_addendum']==p['precision_addendum']==c.bind(c.ADDENDUM)and e['generated']==p['generated']
 assert proof['PASS']and proof['fit_code']==e['code']and proof['model_code']==e['model_code']
 for k in('input_occurrences','GT_occurrences','base_features','base_feature_receipt','warmstart_checkpoint','warmstart_freeze','head_class_code','baseline_scores','C18_wide_sampler_trace'):assert p[k]==root['bindings'][k];c.check(p[k])
 c.check(p['C18_sampler_code']);assert c.sha(C18)==C18_SHA
 receipt=c.read(e['feature_receipt']['path']);assert receipt['completed']and receipt['PASS']and receipt['output']==e['stencil_features']and receipt['base_features']==p['base_features']and receipt['occurrences']==c.N and receipt['unique_inputs']==c.U and receipt['all_F_P_bitexact']and receipt['first_batch_repeat_exact']and receipt['model_state_unchanged']and receipt['all_source_mapping_and_availability_exact']
 freeze=c.read(p['warmstart_freeze']['path']);assert freeze['completed']and freeze['selected_update']==4500 and freeze['checkpoint']==p['warmstart_checkpoint']and freeze['head_class_code']==p['head_class_code']and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']
 return e,p

def run(entry):
 e,p=authorize(entry);assert not OUT.exists(),'Preserve previous attempt; no silent restart.'
 import torch
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 rs=c.rows(p['input_occurrences']['path']);gt=c.rows(p['GT_occurrences']['path']);assert len(rs)==len(gt)==c.N
 for i,(r,g)in enumerate(zip(rs,gt)):
  for k in('occurrence_index','occurrence_id','cue_index','jitter_index','variant','donor_rank'):assert r[k]==g[k]
  assert r['occurrence_index']==i and g['candidate_aux_hidden_label_GT_only']in(-1,0,1)and not g['human_gold']and g['native_mapping']=='UNKNOWN'
 with np.load(p['base_features']['path'],allow_pickle=False)as f:base={k:np.array(f[k],copy=True)for k in('features','occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index')}
 with np.load(e['stencil_features']['path'],allow_pickle=False)as f:st={k:np.array(f[k],copy=True)for k in('pooled9_repeat8','ordered_outer8','occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index')}
 assert base['features'].shape==(c.U,1928)and base['features'].dtype==np.float32 and np.isfinite(base['features']).all()
 for k in('occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index'):assert np.array_equal(base[k],st[k])
 for k in('pooled9_repeat8','ordered_outer8'):assert st[k].shape==(c.U,3072)and st[k].dtype==np.float32 and np.isfinite(st[k]).all()
 um=base['occurrence_unit'];a=base['query_available'];factual=base['factual_query_available'];context=base['context_available'];y=np.array([g['candidate_aux_hidden_label_GT_only']for g in gt],np.int8);split=np.array([r['split']for r in rs]);alpha=np.array([-1 if g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt],np.int8)
 assert np.array_equal(base['global_occurrence_index'],np.arange(c.N))and np.array_equal(a,factual&context)and int((~a).sum())==51 and int((y<0).sum())==468
 w=dict(labels=y,split=split,variant=np.array([r['variant']for r in rs]),pair_ready=np.array([g['geometry_pair_ready']for g in gt],bool),alpha=alpha)
 ns=c.pure_functions(C18,('wide_pools','draw_wide64'),dict(np=np,torch=torch,BASE=51004,STRATA=STRATA));pools=ns['wide_pools'](w,a)
 assert {k:len(v)for k,v in pools.items()}==dict(E_alpha_outside=156,E_alpha_inside=216,D_SHAM_alpha_inside=77,D_SHAM_alpha_outside=295,D_BEFORE=51)
 known=np.isin(y,[0,1]);train=(split=='TRAIN')&a&known;assert train.sum()==795 and np.array_equal(np.bincount(y[train],minlength=2),[423,372])
 with np.load(p['baseline_scores']['path'],allow_pickle=False)as f:baseline=f['query_logit'];baseline_a=f['query_available'];baseline_unit=f['unit_raw_logit']
 assert np.array_equal(a,baseline_a)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 x=torch.as_tensor(base['features'],device='cuda');extras={'POOLED9_REPEAT8':torch.as_tensor(st['pooled9_repeat8'],device='cuda'),'ORDERED_OUTER8':torch.as_tensor(st['ordered_outer8'],device='cuda')};ug=torch.as_tensor(um,device='cuda',dtype=torch.long);yg=torch.as_tensor(y,device='cuda',dtype=torch.float32)
 payload=torch.load(p['warmstart_checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==4500 and payload['head_class_code']==p['head_class_code'];parent_state=payload['state_dict'];head=module(MODEL,'v127_head_factory');OUT.mkdir(parents=True);start=time.monotonic();completed=[];initial_hashes=[];trace_hashes=[]
 c.write(OUT/'STARTED.json',dict(entry=c.bind(entry),fixed_updates=1000,arms=list(ARMS),validation_metrics_after_both_fits=True,source_only=True))
 try:
  for arm in ARMS:
   armout=OUT/arm;armout.mkdir();updates=0;torch.manual_seed(219);torch.cuda.manual_seed_all(219)
   model=head.make_head(torch,parent_state).cuda().float();assert sum(t.numel()for t in model.parameters())==418433 and all(t.requires_grad for t in model.parameters())
   for dest,source in((model.base_affine.weight,'head.0.weight'),(model.base_affine.bias,'head.0.bias'),(model.output.weight,'head.2.weight'),(model.output.bias,'head.2.bias')):assert torch.equal(dest.detach().cpu(),parent_state[source])
   initial_sha=c.state_sha(model);initial_hashes.append(initial_sha);extra=extras[arm]
   with torch.inference_mode():initial_unit=model.eval()(x,extra).cpu().numpy();initial=np.where(a,initial_unit[um],0).astype(np.float32)
   parity=dict(completed=True,all4896_logits_bitexact_C16=np.array_equal(initial,baseline),all4428_unit_logits_bitexact_C16=np.array_equal(initial_unit,baseline_unit),max_drift=float(np.max(np.abs(initial-baseline))),initial_state_sha256=initial_sha,extra_allzero=bool(not model.extra_affine.weight.count_nonzero()),old_parameters_exact=True,optimizer_updates=0)
   c.write(armout/'INITIAL_PARITY.json',parity);assert parity['all4896_logits_bitexact_C16']and parity['all4428_unit_logits_bitexact_C16'],'Preserve exact initialization failure; do not widen tolerance.'
   np.savez_compressed(armout/'INITIAL_LOGITS_INPUT_ONLY.npz',query_logit=initial,query_available=a,global_occurrence_index=np.arange(c.N))
   optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);assert not optimizer.state;gen=torch.Generator().manual_seed(220);tracehash=hashlib.sha256()
   with Path(p['C18_wide_sampler_trace']['path']).open()as reference:
    for step in range(1,1001):
     global_ix,audit=ns['draw_wide64'](pools,gen);ix=global_ix-51004;ref=json.loads(next(reference));assert ref['update']==step and np.array_equal(ix,np.asarray(ref['occurrence_indices'][-64:])-51004)
     assert len(ix)==64 and train[ix].all()and np.array_equal(np.bincount(y[ix],minlength=2),[32,32]);tracehash.update(ix.astype('<i8').tobytes());idx=torch.as_tensor(ix,device='cuda',dtype=torch.long)
     model.train();optimizer.zero_grad(set_to_none=True);z=model(x[ug[idx]],extra[ug[idx]]);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,yg[idx]);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
     row=dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=um[ix].tolist(),labels=y[ix].tolist(),sampling_reference_global=audit,known_available_TRAIN_D=32,known_available_TRAIN_E=32,loss=float(loss.detach()),gradient_norm_before_clip=float(norm))
     with(armout/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   with torch.inference_mode():raw=model.eval()(x,extra);again=model(x,extra);assert torch.equal(raw,again)
   raw=raw.cpu().numpy().astype(np.float32);z=np.where(a,raw[um],0).astype(np.float32);pred=np.where(a,(z>=0).astype(np.int8),-1).astype(np.int8);final_sha=c.state_sha(model)
   score=armout/'FINAL_ALL4896_SOURCE_SCORES.npz';np.savez_compressed(score,query_logit=z,raw_head_logit_before_U=raw[um],unit_raw_logit=raw,prediction=pred,query_available=a,factual_query_available=factual,context_available=context,occurrence_unit=um,global_occurrence_index=np.arange(c.N),occurrence_id=np.array([r['occurrence_id']for r in rs]),external_target=y)
   ck=armout/'FINAL_RESUME_CHECKPOINT.pt';torch.save(dict(state_dict=clone_cpu(torch,model.state_dict()),optimizer_state=clone_cpu(torch,optimizer.state_dict()),RNG_states=dict(wide_generator=gen.get_state().clone(),torch_cpu=torch.get_rng_state().clone(),torch_cuda_all=[t.clone()for t in torch.cuda.get_rng_state_all()]),update=1000,arm=arm,model_class_code=c.bind(MODEL),trainer=c.bind(__file__),policy=e['policy'],threshold_logit=0,checkpoint_selection=False,trainable_parameters=418433,parent_C16_checkpoint=p['warmstart_checkpoint']),ck)
   trace_hashes.append(tracehash.hexdigest());completed.append(dict(arm=arm,out=armout,z=z,scores=c.bind(score),checkpoint=c.bind(ck),trace=c.bind(armout/'TRAIN_UPDATES.jsonl'),trace_sha256=tracehash.hexdigest(),initial_parity=c.bind(armout/'INITIAL_PARITY.json'),initial_state_sha256=initial_sha,final_state_sha256=final_sha,updates=updates,final_repeat_exact=True))
   del model,optimizer
  assert len(set(initial_hashes))==len(set(trace_hashes))==1
  # Both runs have completed their fixed optimization before any VAL metric is produced.
  outputs=[]
  for item in completed:
   result=report(item['z'],a,factual,rs,gt);result.update(arm=item['arm'],fixed_update=1000,scores=item['scores'],initial_parity=item['initial_parity'],sampler_local1000_sha256=item['trace_sha256']);rp=item['out']/'SOURCE_RESULTS.json';c.write(rp,result)
   fp=item['out']/'FIT_COMPLETE.json';c.write(fp,dict(completed=True,PASS=True,arm=item['arm'],optimizer_updates=1000,fixed_final_not_selector=True,checkpoint_selection=False,scores=item['scores'],final_resume_checkpoint=item['checkpoint'],source_results=c.bind(rp),trace=item['trace'],sampler_local1000_sha256=item['trace_sha256'],initial_state_sha256=item['initial_state_sha256'],final_state_sha256=item['final_state_sha256'],initial_C16_logits_bitexact=True,final_repeat_exact=True,encoder_calls=0,current_V112_PnP=0));outputs.append(c.bind(fp))
  c.check(p['warmstart_checkpoint']);c.check(p['base_features']);c.check(e['stencil_features']);c.write(OUT/'CONTROL_COMPLETE.json',dict(completed=True,PASS=True,entry=c.bind(entry),policy=e['policy'],precision_addendum=e['precision_addendum'],feature_receipt=e['feature_receipt'],base_features=p['base_features'],stencil_features=e['stencil_features'],arms=list(ARMS),outputs=outputs,trainable_parameters=418433,initial_states_identical=True,initial_logits_both_bitexact_C16=True,all1000_wide_batches_identical=True,sampler_local1000_sha256=trace_hashes[0],fixed_updates_each=1000,no_VAL_checkpoint_or_arm_choice=True,no_current_promotion=True,old15_regression_claimed=False,seconds=time.monotonic()-start,encoder_current_V112_PnP=0));print(c.read(OUT/'CONTROL_COMPLETE.json'),flush=True)
 except BaseException:
  c.write(OUT/'FAILED_EXECUTION.json',dict(arm=arm if'arm'in locals()else None,optimizer_updates_completed=updates if'updates'in locals()else 0,traceback=traceback.format_exc(),automatic_retry=False));raise

def prepare():
 root=c.policy();g=H/'GENERATED_FIT_TESTS.json';proof=c.read(g);assert proof['PASS']and proof['fit_code']==c.bind(__file__)and proof['model_code']==c.bind(MODEL)and proof['common_code']==c.bind(V/'stencil_common.py');assert c.sha(C18)==C18_SHA
 parent=c.read(V/'feature_preparation/FEATURE_PROTOCOL_BEFORE_FORWARD.json');p=dict(completed=True,code=c.bind(__file__),model_code=c.bind(MODEL),common_code=c.bind(V/'stencil_common.py'),policy=c.bind(c.POLICY),precision_addendum=c.bind(c.ADDENDUM),generated=c.bind(g),C18_sampler_code=c.bind(C18),runtime=parent['runtime'],output=str(OUT),feature_packet_pending_root_entry=True,actual_source_feature_parameter_model_or_fit_calls=0)
 for k in('input_occurrences','GT_occurrences','base_features','base_feature_receipt','warmstart_checkpoint','warmstart_freeze','head_class_code','baseline_scores','C18_wide_sampler_trace'):p[k]=root['bindings'][k]
 c.write(PROTOCOL,p);t=dict(approved_source_two_arm_fit=False,source_only=True,current_allowed=False,V112_allowed=False,PnP_allowed=False,automatic_restart=False,code=p['code'],model_code=p['model_code'],common_code=p['common_code'],protocol=c.bind(PROTOCOL),policy=p['policy'],precision_addendum=p['precision_addendum'],generated=p['generated'],feature_receipt=dict(path='ROOT_BIND_COMPLETED_FEATURE_RECEIPT_PATH',sha256='ROOT_BIND_AFTER_FEATURE_COMPLETION'),stencil_features=dict(path='ROOT_BIND_COMPLETED_FEATURE_PACKET_PATH',sha256='ROOT_BIND_AFTER_FEATURE_COMPLETION'));c.write(H/'ROOT_FIT_ENTRY_TEMPLATE.json',t)
 print(dict(code=c.bind(__file__),model_code=c.bind(MODEL),protocol=c.bind(PROTOCOL),template=c.bind(H/'ROOT_FIT_ENTRY_TEMPLATE.json'),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',actual_fit=0),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_EXACT_ENTRY.json')
