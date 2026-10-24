"""Two root-entry-only fixed1000 source head fits. No encoder/private/model I/O at import."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback,collections
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C20_C21_TRAIN_DESIGN_BEFORE_ARRAYS.json';DESIGN_SHA='45f971b5d24026fec976387be321b1eb0f38ccecb22b8ce5d32263961e60be76'
OLD_POLICY=P/'query_visibility_spatial_stencil_control_20261023_v127/ROOT_C19_SOURCE_CONTROL_POLICY_BEFORE_FEATURES.json'
PARENT_PROTOCOL=V/'feature_preparation/FROZEN_FEATURE_PROTOCOL_BEFORE_FORWARD.json'
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_ARRAYS.json';OUT=V/'runs/C20_C21_FIXED1000_SOURCE_CONTROL'
ARMS=('C20_PUBLIC_REPLAY_CONTROL','C21_PUBLIC_PLUS_NATIVE');OLD=4896;NEW=56214;N=61110
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return[json.loads(s)for s in Path(p).read_text().splitlines()if s]
def write(p,v):
 with Path(p).open('x')as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def ident(r):return r['id']if'id'in r else r['occurrence_id']
def npz(path,keys):
 with np.load(path,allow_pickle=False)as f:return{k:np.array(f[k],copy=True)for k in keys}
def metric(y,p,state):
 cm=np.array([[int(((y==a)&(p==b)).sum())for b in(0,1,-1)]for a in(0,1)],np.int64);div=lambda a,b:float(a/b)if b else None
 return dict(all_rows=len(y),raw_states={s:int((state==s).sum())for s in'DESU'},CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),known_input_U=int(cm[:,2].sum()),teacher_S_hidden_response=int(((state=='S')&(p==1)).sum()),teacher_S_available=int(((state=='S')&(p>=0)).sum()),teacher_U_available=int(((state=='U')&(p>=0)).sum()))
def report(z,a,y,state,split,corpus,line,gt):
 pred=np.where(a,(z>=0).astype(np.int8),-1);out={}
 for name in('OLD_PUBLIC','PUBLIC_AUX','NATIVE'):
  for fold in('TRAIN','VALIDATION'):
   mask=(corpus==name)&(split==fold);group=dict(all_planned=metric(y[mask],pred[mask],state[mask]),input_available=metric(y[mask&a],pred[mask&a],state[mask&a]),by_variant={})
   for variant in sorted({line[i]['variant']for i in np.flatnonzero(mask)}):
    mm=mask&np.array([r['variant']==variant for r in line]);group['by_variant'][variant]=metric(y[mm],pred[mm],state[mm])
   if name=='PUBLIC_AUX':
    group['by_source_case']={str(case):metric(y[mm],pred[mm],state[mm])for case in sorted({gt[i]['selection_index']for i in np.flatnonzero(mask)})for mm in[mask&np.array([g.get('selection_index')==case for g in gt])]}
   if name=='NATIVE':
    group['by_factor']={}
    for factor in('geometry_family','aspect','material','asset_seed','camera_pair_deg','signed_quarter','foreign_family','target_role','distance_scale'):
     values=sorted({json.dumps(line[i]['factors'][factor],sort_keys=True)for i in np.flatnonzero(mask)})
     group['by_factor'][factor]={value:metric(y[mm],pred[mm],state[mm])for value in values for mm in[mask&np.array([json.dumps(r.get('factors',{}).get(factor),sort_keys=True)==value for r in line])]}
    group['by_condition']={str(c):metric(y[mm],pred[mm],state[mm])for c in range(9)for mm in[mask&np.array([r.get('condition_index')==c for r in line])]}
   out[name+'::'+fold]=group
 return dict(completed=True,groups=out,all61110_retained=True,threshold_logit=0,no_VAL_checkpoint_selection=True,geometric_S_not_learned_external_E=True,source_validation_not_blind_real_native_gold=True)
def gate_results(z,a,baseline,baseline_a,y,state,split,corpus,gt):
 pred=np.where(a,(z>=0).astype(np.int8),-1);base=np.where(baseline_a,(baseline>=0).astype(np.int8),-1);gates={};recalls=[]
 def recall(mask,pr,label):return float(((pr==label)&mask).sum()/mask.sum())if mask.any()else None
 old=(corpus=='OLD_PUBLIC')&(split=='VALIDATION')
 for label,tag in((0,'D'),(1,'E')):
  m=old&(y==label);value=recall(m,pred,label);ref=recall(m,base,label);passed=value is not None and ref is not None and value>=ref-.01;gates['old_public_'+tag]=dict(value=value,baseline=ref,threshold=ref-.01 if ref is not None else None,denominator=int(m.sum()),PASS=bool(passed));recalls.append(value)
 alpha=np.array([-1 if g.get('nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only')is None else int(g['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only'])for g in gt]);variant=np.array([g.get('variant','')for g in gt]);ready=np.array([g.get('geometry_pair_ready',False)for g in gt],bool)
 for name,label,m,denom in(('old_Eout_plus1',1,old&(y==1)&(alpha==0)&(variant=='OPAQUE_COVER')&ready,483),('old_SHAMin_plus1',0,old&(y==0)&(alpha==1)&(variant=='CUE_PRESERVING_SHAM')&ready,394)):
  assert int(m.sum())==denom;correct=int((m&(pred==label)).sum());ref=int((m&(base==label)).sum());gates[name]=dict(correct=correct,baseline_correct=ref,minimum_correct=ref+1,denominator=denom,PASS=bool(correct>=ref+1))
 for name in('PUBLIC_AUX','NATIVE'):
  for label,tag in((0,'D'),(1,'E')):
   m=(corpus==name)&(split=='VALIDATION')&(y==label);value=recall(m,pred,label);gates[name+'_VAL_'+tag]=dict(value=value,threshold=.80,denominator=int(m.sum()),PASS=bool(value is not None and value>=.80));recalls.append(value)
 return dict(completed=True,gates=gates,passed_gates=sum(g['PASS']for g in gates.values()),all_gates_pass=all(g['PASS']for g in gates.values()),worst_six_class_recall=min(recalls)if all(v is not None for v in recalls)else None,current_evaluation_authorized=False,old15_regression_gates=dict(computed=False,PASS_claimed=False,reason='Only bound V125 public source plus V129 features evaluated; broader old source featurepack not loaded.'))
def choose(gates):
 eligible=[a for a in ARMS if gates[a]['all_gates_pass']]
 if not eligible:return None
 return max(eligible,key=lambda a:(gates[a]['worst_six_class_recall'],-ARMS.index(a)))
def eligibility(y,a,split,corpus):
 train=(split=='TRAIN')&a&np.isin(y,[0,1]);return train&np.isin(corpus,['OLD_PUBLIC','PUBLIC_AUX']),train&(corpus=='NATIVE')&(y==0),train&(corpus=='NATIVE')&(y==1)
def ordered_new_rows(names,indices,by):
 line=[];gt=[]
 for tag,j in zip(names,indices):
  assert tag in by and 0<=int(j)<len(by[tag][0]);r,g=by[tag][0][int(j)],by[tag][1][int(j)];line.append(r);gt.append(g)
 for tag,(r,g)in by.items():assert sorted(indices[names==tag].tolist())==list(range(len(r)))
 return line,gt
def clone(torch,x):
 if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
 if isinstance(x,dict):return{k:clone(torch,v)for k,v in x.items()}
 if isinstance(x,list):return[clone(torch,v)for v in x]
 if isinstance(x,tuple):return tuple(clone(torch,v)for v in x)
 return x
def authorize(entry):
 e=read(entry);assert e['approved_source_C20_C21_fit']is True and all(e[k]is False for k in('current223','V112','PnP','encoder_forward','automatic_retry'))
 for k in('code','protocol','design','generated','new_feature_receipt','new_features','new_prediction_seal','new_baseline_scores','native_complete','native_inputs','native_GT','public_complete','public_inputs','public_GT'):check(e[k])
 assert sha(DESIGN)==DESIGN_SHA and e['code']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)
 p=read(PROTOCOL);assert p['code']==e['code']and p['design']==e['design']and p['generated']==e['generated']and read(e['generated']['path'])['PASS']
 for k in('old_inputs','old_GT','old_features','old_feature_receipt','old_baseline_scores','warmstart_checkpoint','warmstart_freeze','head_class_code','state_hash_code'):check(p[k])
 f=read(e['new_feature_receipt']['path']);s=read(e['new_prediction_seal']['path']);assert f['completed']and f['PASS']and f['output']==e['new_features']and f['rows']==NEW and f['features_shape']==[NEW,1928]and f['first_batch_repeat_exact']and f['encoder_state_unchanged']and not f['GT_read'];assert s['completed']and s['PASS']and s['features']==e['new_features']and s['feature_receipt']==e['new_feature_receipt']and s['predictions']==e['new_baseline_scores']and s['checkpoint']==p['warmstart_checkpoint']and s['head_state_unchanged']and not s['GT_read']
 for corpus in('native','public'):
  complete=read(e[corpus+'_complete']['path']);assert complete['completed']and complete['PASS']and complete['input_manifest']==e[corpus+'_inputs']and complete['GT_only']==e[corpus+'_GT']
 freeze=read(p['warmstart_freeze']['path']);assert freeze['selected_update']==4500 and freeze['checkpoint']==p['warmstart_checkpoint']and freeze['head_class_code']==p['head_class_code']and freeze['threshold_logit']==0 and not freeze['current_used_for_fit_or_selection']
 return e,p
def load_metadata(e,p,old,new):
 op=rows(p['old_inputs']['path']);og=rows(p['old_GT']['path']);assert len(op)==len(og)==OLD;by={}
 for i,(r,g)in enumerate(zip(op,og)):assert r['occurrence_index']==g['occurrence_index']==i and ident(r)==ident(g)
 for name,tag in(('native','NATIVE'),('public','PUBLIC_AUX')):
  ip=read(e[name+'_inputs']['path']);gp=read(e[name+'_GT']['path']);assert ip['completed']and gp['completed'];rr=ip['rows'];gg=gp['rows'];assert len(rr)==len(gg)==(55296 if tag=='NATIVE'else 918)
  for i,(r,g)in enumerate(zip(rr,gg)):assert r['occurrence_index']==g['occurrence_index']==i and ident(r)==ident(g)and r['split']==g['split']
  by[tag]=(rr,gg)
 nr,ng=ordered_new_rows(new['corpus'],new['source_occurrence_index'],by);line=list(op)+nr;gt=list(og)+ng
 for i,r in enumerate(nr):
  for key in('query_available','factual_query_available','context_available'):assert bool(new[key][i])==r[key]
 assert len(line)==N and np.array_equal(new['global_index'],np.arange(NEW));y=np.array([g['candidate_aux_hidden_label_GT_only']if i<OLD else g['external_target_D0_E1_else_minus1']for i,g in enumerate(gt)],np.int8);state=np.array([('D'if y[i]==0 else'E'if y[i]==1 else'U')if i<OLD else g['raw_state']for i,g in enumerate(gt)]);split=np.array([r['split']for r in line]);corpus=np.r_[np.full(OLD,'OLD_PUBLIC'),new['corpus']];a=np.r_[old['query_available'],new['query_available']];factual=np.r_[old['factual_query_available'],new['factual_query_available']];context=np.r_[old['context_available'],new['context_available']]
 assert np.isin(y,[-1,0,1]).all()and np.array_equal(a,factual&context)and np.array_equal(np.where(state=='D',0,np.where(state=='E',1,-1)),y)
 pub,nd,ne=eligibility(y,a,split,corpus);assert int((pub&(corpus=='OLD_PUBLIC')).sum())==795 and np.bincount(y[pub&(corpus=='OLD_PUBLIC')],minlength=2).tolist()==[423,372];assert int(((corpus=='OLD_PUBLIC')&(split=='VALIDATION')&(y>=0)).sum())==3582 and int(((corpus=='PUBLIC_AUX')&(split=='TRAIN')).sum())==612 and int(((corpus=='PUBLIC_AUX')&(split=='VALIDATION')).sum())==306 and int(((corpus=='NATIVE')&(split=='VALIDATION')).sum())==18432;assert pub.any()and nd.any()and ne.any()
 return line,gt,y,state,split,corpus,a,factual,context,pub,nd,ne
def run(entry):
 e,p=authorize(entry);assert not OUT.exists(),'Preserve attempt; no silent retry.'
 import torch
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()and Path(sys.executable).resolve()==Path(p['runtime']['executable']).resolve()
 old=npz(p['old_features']['path'],('features','occurrence_unit','query_available','factual_query_available','context_available'));new=npz(e['new_features']['path'],('features','query_available','factual_query_available','context_available','Q_inside','global_index','corpus','source_occurrence_index'));assert old['features'].shape==(4428,1928)and new['features'].shape==(NEW,1928)and old['features'].dtype==new['features'].dtype==np.float32 and np.isfinite(old['features']).all()and np.isfinite(new['features']).all()
 line,gt,y,state,split,corpus,a,factual,context,pub,nd,ne=load_metadata(e,p,old,new);ob=npz(p['old_baseline_scores']['path'],('query_logit','query_available','unit_raw_logit'));nb=npz(e['new_baseline_scores']['path'],('query_logit','query_available','raw_head_logit_before_U','global_index','corpus','source_occurrence_index'));assert np.array_equal(ob['query_available'],a[:OLD])and np.array_equal(nb['query_available'],a[OLD:])and all(np.array_equal(nb[k],new[k])for k in('global_index','corpus','source_occurrence_index'));baseline=np.r_[ob['query_logit'],nb['query_logit']]
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 pools=[np.flatnonzero(m)for m in(pub,nd,ne)];pg=torch.Generator().manual_seed(1301);ng=torch.Generator().manual_seed(1302);public_plan=np.stack([pools[0][torch.randint(len(pools[0]),(64,),generator=pg).numpy()]for _ in range(1000)]);native_plan=np.stack([np.r_[pools[1][torch.randint(len(pools[1]),(32,),generator=ng).numpy()],pools[2][torch.randint(len(pools[2]),(32,),generator=ng).numpy()]]for _ in range(1000)])
 OUT.mkdir(parents=True);started=time.monotonic();np.savez_compressed(OUT/'SAMPLER_PLAN_BEFORE_HEAD.npz',public_occurrence_indices=public_plan,native_occurrence_indices=native_plan);write(OUT/'DATA_COUNTS_BEFORE_HEAD.json',dict(completed=True,known_available_old_public_TRAIN=795,known_available_new_public_TRAIN=int((pub&(corpus=='PUBLIC_AUX')).sum()),planned_new_public_TRAIN=612,native_TRAIN_D=len(pools[1]),native_TRAIN_E=len(pools[2]),all_rows=N,all_S=int((state=='S').sum()),all_teacher_U=int((state=='U').sum()),known_input_U=int(((y>=0)&~a).sum()),no_VAL_or_S_U_unavailable_in_plans=bool(pub[public_plan].all()and(nd|ne)[native_plan].all())))
 xo=torch.as_tensor(old['features'],device='cuda');xn=torch.as_tensor(new['features'],device='cuda');ug=torch.as_tensor(old['occurrence_unit'],device='cuda',dtype=torch.long);yg=torch.as_tensor(y,device='cuda',dtype=torch.float32)
 def batch(ix):
  ix=np.asarray(ix);out=torch.empty((len(ix),1928),device='cuda');older=ix<OLD
  if older.any():out[torch.as_tensor(older,device='cuda')]=xo[ug[torch.as_tensor(ix[older],device='cuda',dtype=torch.long)]]
  if (~older).any():out[torch.as_tensor(~older,device='cuda')]=xn[torch.as_tensor(ix[~older]-OLD,device='cuda',dtype=torch.long)]
  return out
 def predict(model):
  with torch.inference_mode():
   raw_old=model.eval()(xo).cpu().numpy();raw_new=np.concatenate([model(xn[lo:lo+512]).cpu().numpy()for lo in range(0,NEW,512)])
  return raw_old,raw_new,np.r_[raw_old[old['occurrence_unit']],raw_new]
 hm=module(p['head_class_code']['path'],'v129_fixed_C11_head');statehash=module(p['state_hash_code']['path'],'v129_state_hash');payload=torch.load(p['warmstart_checkpoint']['path'],map_location='cpu',weights_only=True);assert payload['update']==4500 and payload['head_class_code']==p['head_class_code'];parent=payload['state_dict'];completed=[]
 try:
  for arm in ARMS:
   armout=OUT/arm;armout.mkdir();torch.manual_seed(1300);torch.cuda.manual_seed_all(1300);model=hm.SpatialTargetQueryHead().cuda().float();model.load_state_dict(parent,strict=True);assert sum(t.numel()for t in model.parameters())==221825 and all(t.requires_grad for t in model.parameters());initial_hash=statehash.state_sha(model);ro,rn,raw=predict(model);z=np.where(a,raw,0).astype(np.float32)
   parity=dict(completed=True,old4428_units_bitexact=np.array_equal(ro,ob['unit_raw_logit']),new56214_raw_rows_bitexact=np.array_equal(rn,nb['raw_head_logit_before_U']),all61110_safe_logits_bitexact=np.array_equal(z,baseline),initial_state_sha256=initial_hash);write(armout/'INITIAL_C16_PARITY.json',parity);assert all(parity[k]for k in('old4428_units_bitexact','new56214_raw_rows_bitexact','all61110_safe_logits_bitexact')),'Do not loosen C16 warmstart parity.'
   opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.001);assert not opt.state
   with(armout/'TRAIN_UPDATES.jsonl').open('x')as trace:
    for step in range(1000):
     pi=public_plan[step];ni=native_plan[step];assert pub[pi].all()and(nd|ne)[ni].all()and np.bincount(y[ni],minlength=2).tolist()==[32,32];model.train();opt.zero_grad(set_to_none=True);pl=torch.nn.functional.binary_cross_entropy_with_logits(model(batch(pi)),yg[torch.as_tensor(pi,device='cuda')]);nl=pl.detach()*0
     if arm=='C21_PUBLIC_PLUS_NATIVE':nl=torch.nn.functional.binary_cross_entropy_with_logits(model(batch(ni)),yg[torch.as_tensor(ni,device='cuda')])
     loss=pl+nl;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);opt.step();trace.write(json.dumps(dict(update=step+1,public_occurrence_indices=pi.tolist(),native_occurrence_indices=ni.tolist()if arm=='C21_PUBLIC_PLUS_NATIVE'else[],public_loss=float(pl.detach()),native_loss=float(nl.detach()),total_loss=float(loss.detach()),gradient_norm_before_clip=float(norm),old_proxy_loss=0,known_available_TRAIN_only=True),allow_nan=False)+'\n')
   ro,rn,raw=predict(model);_,_,again=predict(model);assert np.array_equal(raw,again);z=np.where(a,raw,0).astype(np.float32);pred=np.where(a,(z>=0).astype(np.int8),-1).astype(np.int8);score=armout/'FINAL_ALL61110_SOURCE_SCORES.npz';np.savez_compressed(score,query_logit=z,raw_head_logit_before_U=raw,prediction=pred,query_available=a,factual_query_available=factual,context_available=context,external_target=y,raw_state=state,split=split,corpus=corpus,source_occurrence_index=np.r_[np.arange(OLD),new['source_occurrence_index']],global_index=np.arange(N),occurrence_id=np.array([ident(r)for r in line]),query_xy_raw=np.array([r['query_xy']for r in line]),L_raw=np.array([r['L']for r in line]),feature_unit_index=np.r_[old['occurrence_unit'],4428+np.arange(NEW)])
   ck=armout/'FINAL_RESUME_CHECKPOINT.pt';torch.save(dict(state_dict=clone(torch,model.state_dict()),optimizer_state=clone(torch,opt.state_dict()),RNG_states=dict(torch_cpu=torch.get_rng_state().clone(),torch_cuda_all=[t.clone()for t in torch.cuda.get_rng_state_all()],public_generator_end=pg.get_state().clone(),native_generator_end=ng.get_state().clone()),update=1000,trainer=bind(__file__),head_class_code=p['head_class_code'],design=e['design'],arm=arm,sampler_plan=bind(OUT/'SAMPLER_PLAN_BEFORE_HEAD.npz'),no_VAL_checkpoint_selection=True),ck);freeze=armout/'MODEL_FREEZE.json';write(freeze,dict(completed=True,checkpoint=bind(ck),trainer=bind(__file__),head_class_code=p['head_class_code'],design=e['design'],selected_update=1000,fixed_final_not_VAL_selection=True,threshold_logit=0,arm=arm,model_state_sha256=statehash.state_sha(model),initial_state_sha256=initial_hash,current_used_for_fit_or_selection=False,PnP=False));completed.append(dict(arm=arm,out=armout,z=z,scores=bind(score),freeze=bind(freeze),initial_hash=initial_hash));del model,opt
  assert len(completed)==2 and completed[0]['initial_hash']==completed[1]['initial_hash']
  # First performance readout of VAL only after BOTH fixed1000 optimizations finish.
  gates={};baseline_report=OUT/'FROZEN_C16_SOURCE_RESULTS.json';write(baseline_report,report(baseline,a,y,state,split,corpus,line,gt));arm_receipts=[]
  for item in completed:
   arm=item['arm'];out=item['out'];rp=out/'SOURCE_RESULTS.json';gp=out/'SOURCE_GATE_RESULTS.json';write(rp,report(item['z'],a,y,state,split,corpus,line,gt));gates[arm]=gate_results(item['z'],a,baseline,a,y,state,split,corpus,gt);write(gp,gates[arm]);fp=out/'FIT_COMPLETE.json';write(fp,dict(completed=True,PASS=True,optimizer_updates=1000,arm=arm,fixed_final_not_VAL_selection=True,model_freeze=item['freeze'],scores=item['scores'],source_results=bind(rp),source_gates=bind(gp),sampler_plan=bind(OUT/'SAMPLER_PLAN_BEFORE_HEAD.npz'),public_sequence_identical_both_arms=True,model_repeat_exact=True,encoder_calls=0,current_evaluation_authorized=False));arm_receipts.append(bind(fp))
  final=OUT/'CONTROL_COMPLETE.json';write(final,dict(completed=True,PASS=True,entry=bind(entry),design=e['design'],outputs=arm_receipts,baseline_results=bind(baseline_report),eligible_source_arm=choose(gates),source_only_selection_rule='Passing gates then maximum worst6class recall; exact ties C20.',all_gates={arm:gates[arm]['all_gates_pass']for arm in ARMS},both_fixed_fits_before_VAL_metrics=True,current_evaluation_authorized=False,fully_solved=False,old15_regression_PASS_claimed=False,seconds=time.monotonic()-started,encoder_fit_private_V112_PnP=dict(encoder=0,source_fit_updates_each=1000,private=0,V112=0,PnP=0)));print(json.dumps(dict(completed=True,control_complete=bind(final),source_eligible_arm=choose(gates),seconds=time.monotonic()-started)),flush=True)
 except BaseException:write(OUT/'FAILED_NO_RETRY.json',dict(arm=arm if'arm'in locals()else None,traceback=traceback.format_exc(),automatic_retry=False));raise
def prepare():
 assert sha(DESIGN)==DESIGN_SHA;d=read(DESIGN);old=read(OLD_POLICY)['bindings'];parent=read(PARENT_PROTOCOL);g=H/'GENERATED_TRAINER_TESTS.json';proof=read(g);assert proof['PASS']and proof['code']==bind(__file__)
 p=dict(completed=True,code=bind(__file__),generated=bind(g),design=bind(DESIGN),old_source_binding_policy=bind(OLD_POLICY),parent_feature_protocol=bind(PARENT_PROTOCOL),old_inputs=old['input_occurrences'],old_GT=old['GT_occurrences'],old_features=old['base_features'],old_feature_receipt=old['base_feature_receipt'],old_baseline_scores=old['baseline_scores'],warmstart_checkpoint=old['warmstart_checkpoint'],warmstart_freeze=old['warmstart_freeze'],head_class_code=old['head_class_code'],state_hash_code=parent['C11_producer'],runtime=parent['runtime'],output=str(OUT),new_bindings_pending=True,actual_arrays_parameters_model_fit_calls=0);write(PROTOCOL,p)
 t=dict(approved_source_C20_C21_fit=False,current223=False,V112=False,PnP=False,encoder_forward=False,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),design=p['design'],generated=p['generated'])
 for key,path in dict(new_feature_receipt='frozen_C16_source_baseline_v1/FEATURE_COMPLETE.json',new_features='frozen_C16_source_baseline_v1/SOURCE_FEATURES_INPUT_ONLY.npz',new_prediction_seal='frozen_C16_source_baseline_v1/PREDICTION_SEAL.json',new_baseline_scores='frozen_C16_source_baseline_v1/FROZEN_C16_PREDICTIONS_INPUT_ONLY.npz',native_complete='source_crops_v1/CROP_COMPLETE.json',native_inputs='source_crops_v1/INPUT_ROWS.json',native_GT='source_crops_v1/GT_ROWS.json',public_complete='public_corpus_v1/CORPUS_COMPLETE.json',public_inputs='public_corpus_v1/INPUT_ROWS.json',public_GT='public_corpus_v1/GT_ROWS.json').items():t[key]=dict(path=str(V/path),sha256='ROOT_BIND_COMPLETED_SOURCE')
 write(H/'ROOT_FIT_ENTRY_TEMPLATE.json',t);print(json.dumps(dict(code=p['code'],protocol=bind(PROTOCOL),template=bind(H/'ROOT_FIT_ENTRY_TEMPLATE.json'),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_ACTUAL_C20_C21_ENTRY.json',actual_fit=0)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_ACTUAL_C20_C21_ENTRY.json')
