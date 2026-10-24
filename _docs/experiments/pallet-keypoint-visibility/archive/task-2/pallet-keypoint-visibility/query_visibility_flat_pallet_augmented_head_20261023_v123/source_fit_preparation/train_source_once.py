"""C15: C14 warmstart, exact old128 trace plus hierarchical public new64. Inert import."""
import os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, importlib.util, json, time, traceback
import numpy as np
import torch

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C15_DESIGN_BEFORE_SOURCE_FIT.json'
DESIGN_SHA='78b6734aba6822b2b8162f4b70583e2d7d0e848dea74749aee1e7c8d977a5efc'
C14=P/'query_visibility_consistent_query_context_20261023_v119/source_fit_preparation/train_source_once.py'
C14_SHA='5022f6048b366eef6b7e36d1bd6dc05279c9117a89c5ba062be9a8ab6b05cdba'
assert hashlib.sha256(C14.read_bytes()).hexdigest()==C14_SHA
s=importlib.util.spec_from_file_location('C15_pinned_C14',C14);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
sha,bind,check,read,rows,write=old.sha,old.bind,old.check,old.read,old.rows,old.write
BASE,NEW,N,OLDUNITS,NEWUNITS,UNITS=46108,4896,51004,15004,4428,19432
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FIT.json';RUN=V/'runs/C15_PUBLIC_FLAT_PALLET_AUGMENTED_SPATIAL_QUERY'
OLD_PREFIX_SHA='34157f08cea851ee613d8333e829c7a590d76bb51fe6a33b823f33b6790316d5'


def join_flat(inputs,gt):
 assert len(inputs)==len(gt)==NEW
 labels=[]
 for i,(r,g)in enumerate(zip(inputs,gt)):
  qi,role=divmod(i,17);rank=None if role==0 else(role-1)//2
  variant='BEFORE'if role==0 else('OPAQUE_COVER'if role%2 else'CUE_PRESERVING_SHAM')
  for key,value in(('occurrence_index',i),('cue_index',qi//9),('jitter_index',qi%9),('donor_rank',rank),('variant',variant)):
   assert r[key]==g[key]==value,(i,key)
  assert r['query_index']==qi and r['occurrence_id']==g['occurrence_id']
  assert r['source']in('LOCO','TOMIE')and r['split']in('TRAIN','VALIDATION')
  assert (r['source'],r['raw_split'],r['split'])in(('LOCO','TRAIN','TRAIN'),('LOCO','VALIDATION','VALIDATION'),('TOMIE','TOMIE_DEVELOPMENT_STRESS','VALIDATION'))
  assert not g['human_gold']and g['native_mapping']=='UNKNOWN'
  y=g['candidate_aux_hidden_label_GT_only'];assert y in(-1,0,1)
  before=inputs[17*qi]
  assert all(r[k]==before[k]for k in('query_xy','L','initial_points','context_box_xyxy','source','split','raw_split','source_frame_group'))
  if role==0:assert y==0 and not g['geometry_pair_ready']
  else:
   mate=gt[i+1]if role%2 else gt[i-1];assert g['geometry_pair_ready']==mate['geometry_pair_ready']
   assert y==((1 if role%2 else 0)if g['geometry_pair_ready']else-1)
  labels.append(y)
 labels=np.asarray(labels,np.int8)
 return dict(labels=labels,typed_state=np.where(labels==0,'D',np.where(labels==1,'E','U')),
             split=np.array([r['split']for r in inputs]),domain=np.full(NEW,'V122_PUBLIC_FLAT_PALLET'),
             source=np.array([r['source']for r in inputs]),variant=np.array([r['variant']for r in inputs]),lineage=inputs)


def flat_pools(new,a):
 inputs=new['lineage'];train=(new['split']=='TRAIN')&np.isin(new['labels'],[0,1])&a
 assert not((new['source']=='TOMIE')&(new['split']=='TRAIN')).any()
 cues=sorted({r['cue_index']for i,r in enumerate(inputs)if train[i]})
 assert len(cues)==6
 cover,before={},{}
 for ci in cues:
  cover[ci]=np.array([BASE+i for i,r in enumerate(inputs)if r['cue_index']==ci and train[i]and r['variant']=='OPAQUE_COVER'and new['labels'][i]==1],np.int64)
  before[ci]=np.array([BASE+i for i,r in enumerate(inputs)if r['cue_index']==ci and train[i]and r['variant']=='BEFORE'],np.int64)
  assert len(cover[ci])>0 and len(before[ci])==9
  for j in cover[ci]-BASE:
   assert inputs[j+1]['variant']=='CUE_PRESERVING_SHAM'and new['labels'][j+1]==0 and a[j+1]
   assert all(inputs[j][k]==inputs[j+1][k]for k in('cue_index','query_index','donor_rank','query_xy','L'))
 return dict(cues=np.array(cues,np.int64),cover=cover,before=before)


def draw_new64(pools,generator):
 def choose(table):
  ci=int(pools['cues'][int(torch.randint(len(pools['cues']),(1,),generator=generator))])
  pool=table[ci];ix=int(pool[int(torch.randint(len(pool),(1,),generator=generator))])
  return ci,ix
 selected_E=[choose(pools['cover'])for _ in range(32)]
 selected_B=[choose(pools['before'])for _ in range(16)]
 E=np.array([x[1]for x in selected_E],np.int64);B=np.array([x[1]for x in selected_B],np.int64);S=E[:16]+1
 return np.r_[E,B,S],dict(E_cue_choices=[x[0]for x in selected_E],E_indices=E.tolist(),BEFORE_cue_choices=[x[0]for x in selected_B],BEFORE_indices=B.tolist(),SHAM_indices=S.tolist())


def draw192(c12,c10,pools,mp,wp,newp,oldgens,newgen):
 ix,pairs,audit,i64,t64,i96,t96=old.draw128(c12,c10,pools,mp,wp,*oldgens)
 extra,extra_audit=draw_new64(newp,newgen)
 return np.r_[ix,extra],pairs,dict(old128=audit,new64=extra_audit),ix,i64,t64,i96,t96


def loss192(z,target,available,pairs):
 assert z.shape==target.shape==available.shape==(192,)and pairs.shape==(24,2)and int(pairs.max())<128
 oldloss,bce,hinge=old.supervised_loss(z[:128],target[:128],available[:128],pairs)
 eligible=available[128:]&((target[128:]==0)|(target[128:]==1))
 extra=torch.nn.functional.binary_cross_entropy_with_logits(z[128:][eligible],target[128:][eligible])if eligible.any()else z[128:].sum()*0
 return oldloss+extra,bce,hinge,extra


def selector14(c12,z,a,meta,new,d):
 _,parts=old.selector10(c12,z[:BASE],a[:BASE],meta,d)
 assert len(parts)==10
 zz=np.where(a[BASE:],z[BASE:],0).astype(np.float64);loss=np.logaddexp(0,zz)-np.where(new['labels']>=0,new['labels'],0)*zz
 for source,name in(('LOCO','LOCO_VALIDATION'),('TOMIE','TOMIE_DEVELOPMENT')):
  for state in('D','E'):
   mask=(new['source']==source)&(new['split']=='VALIDATION')&(new['typed_state']==state)
   assert mask.any(),'Empty required new source selector cohort'
   parts[name+'_'+state]=dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~a[BASE:]).sum()))
 assert len(parts)==14
 return float(np.mean([v['mean_BCE']for v in parts.values()])),parts


def earliest_history(history):
 best=float('inf');step=None
 for r in history:
  if r['criterion_fourteen_VAL_cohorts_BCE']<best:best=r['criterion_fourteen_VAL_cohorts_BCE'];step=r['update']
 return step,best


def flat_metric(y,pred):
 cm=np.zeros((2,3),np.int64)
 for yy,pp in zip(y,pred):
  if yy in(0,1):cm[int(yy),2 if pp<0 else int(pp)]+=1
 div=lambda x,y:float(x/y)if y else None
 return dict(all_rows=len(y),teacher_D=int((y==0).sum()),teacher_E=int((y==1).sum()),teacher_U=int((y<0).sum()),
             CM_D_E_rows_D_E_U_columns=cm.tolist(),D_recall=div(cm[0,0],cm[0].sum()),E_recall=div(cm[1,1],cm[1].sum()),
             D_precision=div(cm[0,0],cm[:,0].sum()),E_precision=div(cm[1,1],cm[:,1].sum()),model_U_on_known=int(cm[:,2].sum()))


def flat_report(z,a,new):
 pred=np.where(a,(z>=0).astype(np.int8),-1);y=new['labels'];rs=new['lineage'];centre=np.array([r['jitter_index']==4 for r in rs]);groups={};paired={}
 keys=[('ALL','ALL')]+sorted({(r['source'],r['raw_split'])for r in rs})
 for source,fold in keys:
  keep=np.array([(source=='ALL'or r['source']==source)and(fold=='ALL'or r['raw_split']==fold)for r in rs]);name=source+'::'+fold
  groups[name]=dict(all9_correlated=flat_metric(y[keep],pred[keep]),centre_offset0=flat_metric(y[keep&centre],pred[keep&centre]),
      by_variant={v:flat_metric(y[keep&(new['variant']==v)],pred[keep&(new['variant']==v)])for v in('BEFORE','OPAQUE_COVER','CUE_PRESERVING_SHAM')})
  for rank in range(8):
   triples=[];centr=[]
   for qi in range(288):
    ix=np.array([17*qi,17*qi+1+2*rank,17*qi+2+2*rank]);j=ix[0]
    if keep[j]and np.array_equal(y[ix],[0,1,0]):
     triples.append(bool(np.array_equal(pred[ix],[0,1,0])))
     if centre[j]:centr.append(triples[-1])
   paired[name+'::R'+str(rank)]=dict(known_triplets=len(triples),all3_correct=sum(triples),centre_known_triplets=len(centr),centre_all3_correct=sum(centr))
 return dict(metrics=groups,paired_by_rank=paired,all4896_retained=True,native_mapping='UNKNOWN',human_gold=False,source_VAL_development_exposed=True)


def load_old_metadata(e,d14,p14,c12,c10,d):
 parent=read(check(d14['parent_C12_design']));p12=read(check(p14['parent_C12_protocol']));material_receipt=read(check(p12['new_feature_receipt']))
 base=c12.load_metadata(e,material_receipt,parent,c10,d)
 addition=old.join_new(rows(check(d14['new_source']['input_occurrences'])),rows(check(d14['new_source']['GT_occurrences'])))
 return old.combine_metadata(base,addition),base,parent


def load_feature_packet(binding,n,u):
 with np.load(check(binding),allow_pickle=False)as f:pack={k:np.array(f[k],copy=True)for k in('features','occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index','unit_Q_inside')}
 assert pack['features'].shape==(u,1928)and pack['features'].dtype==np.float32 and np.isfinite(pack['features']).all()
 assert pack['occurrence_unit'].shape==(n,)and np.issubdtype(pack['occurrence_unit'].dtype,np.integer)and ((pack['occurrence_unit']>=0)&(pack['occurrence_unit']<u)).all()
 for k in('query_available','factual_query_available','context_available'):assert pack[k].shape==(n,)and pack[k].dtype==bool
 assert np.array_equal(pack['query_available'],pack['factual_query_available']&pack['context_available'])and np.array_equal(pack['global_occurrence_index'],np.arange(n))
 assert pack['unit_Q_inside'].shape==(u,)and pack['unit_Q_inside'].dtype==bool and np.array_equal(pack['features'][:,1927],pack['unit_Q_inside'].astype(np.float32))
 return pack


def validate_entry(path):
 assert not RUN.exists(),'Single fit; preserve failures, no automatic restart.'
 e=read(path)
 assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
 for k in('trainer','protocol','design','generated','source_bindings'):check(e[k])
 assert e['trainer']==bind(__file__)and e['protocol']==bind(PROTOCOL)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 p=read(PROTOCOL);proof=read(e['generated']['path']);assert p['trainer']==e['trainer']and p['generated']==e['generated']and proof['PASS']and proof['code']==e['trainer']
 design=read(DESIGN)
 for k,v in design.items():
  if isinstance(v,dict)and set(v)>={'path','sha256'}:check(v)
 assert design['parent_C14_trainer']==bind(C14)and design['head']['parameters']==221825 and design['sampling']['updates']==1000
 assert e['source_bindings']==p['source_bindings']
 for k in('old_source_scores','new_baseline_predictions','old_trace','parent_C14_fit_complete'):check(p[k])
 for key,shape,count in(('old',[15004,1928],46108),('new',[4428,1928],4896)):
  receipt=read(design[key+'_feature_receipt']['path']);assert receipt['completed']and receipt['PASS']and receipt['output']==design[key+'_features']and receipt['features_shape']==shape and receipt['occurrences']==count and receipt['model_state_unchanged']and receipt['first_batch_repeat_exact']
 return e,design,p


def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()


def run(entry_path):
 e,design,protocol=validate_entry(entry_path) # All source-only entry checks precede data/parameter loads.
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
  def checkpoint(step):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},update=step,warmstart_parent_update=1200,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path),head_class_code=design['head_class_code'],threshold_logit=0)
  @torch.no_grad()
  def predict():
   model.eval();raw=torch.cat((model(xo),model(xn)));return np.where(available,raw[ug].cpu().numpy(),0).astype(np.float32)
  @torch.no_grad()
  def evaluate(step):
   nonlocal best,selected,best_state
   z=predict();value,parts=selector14(c12,z,available,meta,new,d);row=dict(update=step,criterion_fourteen_VAL_cohorts_BCE=value,VAL_cohorts=parts);history.append(row)
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
  with Path(protocol['old_trace']['path']).open()as oldtrace:
   for step in range(1,1001):
    ix,pairs,audit,ix128,_,_,_,_=draw192(c12,c10,pools,mp,wp,npools,gens,newgen)
    ref=json.loads(next(oldtrace));assert ref['update']==step and np.array_equal(ix128,ref['occurrence_indices'])and np.array_equal(pairs,ref['matched_cover_sham_batch_positions'])
    old.hash_draw(tracehash,ix128,pairs);newhash.update(ix[128:].astype('<i8').tobytes())
    assert known[ix].all()and available[ix].all()and(split[ix]=='TRAIN').all()and np.array_equal(np.bincount(target[ix],minlength=2),[96,96])
    assert np.array_equal(target[ix[pairs]],np.tile([1,0],(24,1)))and np.array_equal(ix[176:192],ix[128:144]+1)
    idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');pp=torch.as_tensor(pairs,dtype=torch.long,device='cuda');model.train();optimizer.zero_grad(set_to_none=True)
    z=model(xt[ug[idx]]);loss,bce,hinge,nbce=loss192(z,tg[idx],ag[idx],pp);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
    with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_cover_sham_batch_positions=pairs.tolist(),new_correspondence_positions_not_in_hinge=[[128+i,176+i]for i in range(16)],sampling=audit,available_loss_rows=192,loss=float(loss.detach()),old_BCE=float(bce.detach()),old_hinge=float(hinge.detach()),new_BCE=float(nbce.detach()),gradient_norm_before_clip=float(norm)),allow_nan=False)+'\n')
    if step%100==0:evaluate(step)
  assert tracehash.hexdigest()==OLD_PREFIX_SHA and earliest_history(history)==(selected,best)
  torch.save(checkpoint(1000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);z=predict();value,parts=selector14(c12,z,available,meta,new,d);assert value==best
  oldmetrics,gates=c9.report(d,z[:33984],available[:33984],meta['old'],dict(d14,source_gates={k:v for k,v in d14['source_gates'].items()if not k.startswith('material_')}))
  material,mgates=c12.material_report(d,z[:old.BASE],available[:old.BASE],base,parent);gates.update(mgates);assert len(gates)==15 and set(gates)==set(d14['source_gates'])
  newreport=flat_report(z[BASE:],available[BASE:],new);write(RUN/'NEW_FLAT_SOURCE_RESULTS.json',newreport)
  line=meta['lineage']+new['lineage'];q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line]);Ls=np.array([r['L']if r['L']is not None else np.nan for r in line])
  ids=np.array([json.dumps({k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')},sort_keys=True,separators=(',',':'),allow_nan=False)for r in line])
  states=np.r_[meta['typed_state'],new['typed_state']];sources=np.r_[meta['source'],new['source']];variants=np.r_[meta['variant'],new['variant']];domains=np.r_[meta['domain'],new['domain']]
  np.savez_compressed(RUN/'SELECTED_ALL51004_SOURCE_SCORES.npz',query_logit=z,query_available=available,factual_query_available=factual,context_available=context,prediction=d.decode(z,available),input_unit_index=units,global_occurrence_index=np.arange(N),raw_teacher_state=states,external_target=target,split=split,source=sources,variant=variants,domain=domains,query_xy_raw=q,L_raw=Ls,original_identifier_metadata_json=ids,original_split=np.array([r.get('raw_split')or r.get('original_split')or split[i]for i,r in enumerate(line)]),context_Q_inside=np.r_[op['unit_Q_inside'],npk['unit_Q_inside']][units],old_accepted_triplets=meta['old']['old_pairs'],new_selected_triplets=meta['old']['new_pairs'],samewood_accepted_cover_sham_pairs=meta['new_pairs2'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,gates=gates,all_gates_pass=all(v['PASS']for v in gates.values()),gate_scope='Exact existing15 C14 source regression definitions; failed source model never promoted.',current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_fourteen_VAL_cohorts_BCE=best,VAL_cohorts=parts,old_metrics=oldmetrics,material_surface=material,old_additive_reports=old.additive_reports(z[:BASE],available[:BASE],meta,d),new_flat_source_report=bind(RUN/'NEW_FLAT_SOURCE_RESULTS.json'),all51004_retained=True,old128_first1000_draws_sha256=tracehash.hexdigest(),new64_all1000_draws_sha256=newhash.hexdigest(),source_VAL_development_exposed=True,TOMIE_TRAIN=0,known_unavailable={s:int((known&(split==s)&~available).sum())for s in('TRAIN','VALIDATION')},native_mapping='UNKNOWN',human_gold=False,learned_self_classification=False))
  assert sha(design['warmstart_checkpoint']['path'])==design['warmstart_checkpoint']['sha256']
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),head_class_code=design['head_class_code'],trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),warmstart_checkpoint=design['warmstart_checkpoint'],old_feature_pack=design['old_features'],new_feature_pack=design['new_features'],new_feature_receipt=design['new_feature_receipt'],current_used_for_fit_or_selection=False,current_inference_authorized=False,V112_inference_authorized=False,PnP=0))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=1000,selected_update=selected,criterion_fourteen_VAL_cohorts_BCE=best,all_source_gates_pass=all(v['PASS']for v in gates.values()),seconds=time.monotonic()-start,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL51004_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),update0_parity=bind(RUN/'UPDATE0_WARMSTART_PARITY.json'),new_flat_source_results=bind(RUN/'NEW_FLAT_SOURCE_RESULTS.json'),current_V112_or_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise


def prepare():
 assert sha(DESIGN)==DESIGN_SHA
 design=read(DESIGN);p14=read(check(design['parent_C14_protocol']));oldfit=Path(design['parent_model_freeze']['path']).parent/'FIT_COMPLETE.json';fit=read(oldfit)
 newpredseal=Path(design['new_feature_receipt']['path']).parent/'PREDICTION_SEAL.json';pred=read(newpredseal)
 proof=H/'GENERATED_TRAINER_TESTS.json';assert read(proof)['PASS']and read(proof)['code']==bind(__file__)
 p=dict(completed=True,trainer=bind(__file__),design=bind(DESIGN),generated=bind(proof),parent_C14=bind(C14),source_bindings=p14['source_bindings'],old_source_scores=fit['scores'],new_baseline_predictions=pred['predictions'],parent_C14_fit_complete=bind(oldfit),old_trace=bind(oldfit.parent/'TRAIN_UPDATES.jsonl'),old128_first1000_trace_sha256=OLD_PREFIX_SHA,feature_shape=[UNITS,1928],occurrences=N,runtime=dict(executable='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python',torch='2.1.1+cu118'),source_only_entry_before_data_and_parameters=True,exact_design_controls=True,run_directory=str(RUN),actual_fit_or_head_call=0)
 write(PROTOCOL,p);write(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',dict(approved_source_fit=False,current_allowed=False,V112_allowed=False,PnP_allowed=False,restart_after_update=False,trainer=bind(__file__),protocol=bind(PROTOCOL),design=bind(DESIGN),generated=bind(proof),source_bindings=p['source_bindings']))
 write(H/'PREPARATION.json',dict(completed=True,trainer=bind(__file__),protocol=bind(PROTOCOL),generated=bind(proof),template=bind(H/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json'),run_directory=str(RUN),command=p['runtime']['executable']+' '+str(Path(__file__).resolve())+' --entry ROOT_EXACT_ENTRY.json',fit_head_parameters_features_pixels_current_V112_reads_or_calls=0))
 print(json.dumps(read(H/'PREPARATION.json')),flush=True)


if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',type=Path);ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
 if args.prepare:prepare()
 elif args.entry:run(args.entry)
 else:raise SystemExit('Use --prepare or --entry ROOT_EXACT_ENTRY.json')
