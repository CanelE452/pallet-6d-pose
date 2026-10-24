"""One source-only C13 fit. Imports are inert; actual bodies require root entry."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C13_DESIGN_BEFORE_SOURCE_FORWARD.json'
DESIGN_SHA='18eb784ea40cdb726dd4303b65904446f2a1bd9655542bca36d534112948f089'
C12=P/'query_visibility_material_augmented_query_head_20261023_v114/source_fit_preparation/train_source_once.py'
C12_SHA='a330cb5ed3d77386b5dcd0f02d737cdaab29234394cf1ad69ad5ec011f2e4b2d'
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
RUN=V/'runs/C13_SAME_MATERIAL_FOREGROUND_SPATIAL_QUERY'
BASE,NEW,N,OLDUNITS,NEWUNITS,UNITS=45184,924,46108,14852,152,15004
DOMAIN='V116_SAME_WOOD_CONTROLLED'
OLD64_SHA='8d6ec2eecbdd0937e710173328cea8e0d4211d21b11d722faea5d5c6fb8cf680'
OLD96_SHA='77e99c76f7c0f797aaadd00efaddd73824dd564defc84535114c3e7d24b1484a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def source_modules():
 assert sha(C12)==C12_SHA
 s=importlib.util.spec_from_file_location('c13_immutable_c12',C12);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 return (m,)+m.source_modules()

def join_new(inputs,gt):
 """Actual GT joins downstream only. No GT field enters features/model.forward."""
 assert len(inputs)==len(gt)==NEW
 labels=np.empty(NEW,np.int8);pairs=[]
 for j,(r,g)in enumerate(zip(inputs,gt)):
  b,off=divmod(j,33);rank=None if off==0 else(off-1)//2
  variant='BEFORE'if off==0 else('OPAQUE_COVER'if off%2 else'CUE_PRESERVING_SHAM')
  for key,value in [('occurrence_index',j),('base_index',b),('donor_rank',rank),('variant',variant)]:assert r[key]==g[key]==value,(j,key)
  assert r['occurrence_id']==g['occurrence_id']and r['original_base_row_index']==g['original_base_row_index']
  assert r['split']=='TRAIN'and r['raw_split']in('TRAIN','TRAIN_LAB')and r['source']in('LOCO','PALLOX')
  assert r['raw_split']==('TRAIN'if r['source']=='LOCO'else'TRAIN_LAB')
  before=inputs[33*b]
  assert all(r[k]==before[k]for k in('query_xy','L','slot_index','native_slot_logical_only','source','split','raw_split','original_base_row_index','frozen_context_box_xyxy'))
  assert g['native_mapping']=='UNKNOWN'and g['human_gold']is False
  y=g['candidate_aux_hidden_label_GT_only'];assert y in(-1,0,1)
  if off==0:assert y==0 and g['pair_ready']is False and r['full_rgb_changed']is False
  else:
   mate=gt[j+1]if off%2 else gt[j-1]
   assert g['pair_ready']==mate['pair_ready']
   if g['pair_ready']:
    assert y==(1 if off%2 else 0)and r['full_rgb_changed']is True
    if off%2:pairs.append([BASE+j,BASE+j+1])
   else:assert y==-1 and r['full_rgb_changed']is False
  labels[j]=y
 state=np.where(labels==0,'D',np.where(labels==1,'E','U'))
 return dict(typed_state=state,labels=labels,split=np.full(NEW,'TRAIN'),domain=np.full(NEW,DOMAIN),source=np.array([r['source']for r in inputs]),variant=np.array([r['variant']for r in inputs]),pairs=np.array(pairs,np.int64).reshape(-1,2),lineage=inputs)

def combine_metadata(old,new):
 m={k:np.concatenate((old[k],new[k]))for k in('typed_state','labels','split','domain','source','variant')}
 m.update(known=np.isin(m['typed_state'],['D','E']),known_geometric_S=m['typed_state']=='S',unresolved_teacher_U=m['typed_state']=='U',lineage=old['lineage']+new['lineage'],old=old['old'],parent_meta=old,new_pairs2=new['pairs'])
 m['external_target']=np.where(m['known'],m['labels'],-1).astype(np.int8)
 return m

def expected_new_units(inputs):
 counter=28;out=[]
 for r in inputs:
  out.append(counter if r['full_rgb_changed']else r['base_index'])
  if r['full_rgb_changed']:counter+=1
 assert len(out)==924 and counter==152
 return np.array(out,np.int32)

def available_pairs(meta,a):
 p=meta['new_pairs2'];assert p.ndim==2 and p.shape[1]==2
 assert(meta['split'][p]=='TRAIN').all()and np.array_equal(meta['external_target'][p],np.tile([1,0],(len(p),1)))
 p=p[a[p].all(1)]
 assert len(p)>0,'No available accepted samewood TRAIN pairs: no fit/fallback.'
 return p

def draw128(c12,c10,pools,mp,wp,oldgen,mg,sg,ng,fg):
 ix96,tr96,audit96,old64,tr64=c12.draw96(c10,pools,mp,oldgen,mg,sg)
 choices=torch.randint(len(wp),(16,),generator=ng).numpy();new=wp[choices]
 before=np.r_[ix96,new.reshape(-1)];perm=torch.randperm(128,generator=fg).numpy();inv=np.argsort(perm)
 # Pair API is cover/sham only. Original old triples are never fabricated for new data.
 oldpairs=tr96[:,1:3];newpos=np.arange(96,128).reshape(16,2)
 pairs=inv[np.r_[oldpairs,newpos]]
 return before[perm],pairs,dict(base96=audit96,base96_occurrence_indices=ix96.tolist(),base96_triplet_positions=tr96.tolist(),new_pair_pool_indices=choices.tolist(),new_cover_sham_occurrence_indices=new.tolist(),final128_permutation=perm.tolist()),old64,tr64,ix96,tr96

def supervised_loss(z,target,a,pairs):
 assert z.ndim==target.ndim==a.ndim==1 and z.shape==target.shape==a.shape
 assert pairs.ndim==2 and pairs.shape[1]==2
 known=((target==0)|(target==1));eligible=known&a
 bce=torch.nn.functional.binary_cross_entropy_with_logits(z[eligible],target[eligible])if eligible.any()else z.sum()*0
 cover,sham=pairs[:,0],pairs[:,1];ok=eligible[cover]&eligible[sham]
 assert(target[cover[ok]]==1).all()and(target[sham[ok]]==0).all()
 hinge=torch.relu(2-(z[cover[ok]]-z[sham[ok]])).mean()if ok.any()else z.sum()*0
 return bce+.25*hinge,bce,hinge

def selector10(c12,z,a,meta,d):return c12.selector10(z[:BASE],a[:BASE],meta['parent_meta'],d)
def earliest_history(history):
 best=float('inf');step=None
 for r in history:
  if r['criterion_ten_VAL_cohorts_BCE']<best:best=r['criterion_ten_VAL_cohorts_BCE'];step=r['update']
 return step,best
def hash_draw(h,ix,pairs):h.update(ix.astype('<i8').tobytes());h.update(pairs.astype('<i8').tobytes())

def pair_summary(z,a,p):
 pred=np.where(a,(z>=0).astype(np.int8),-1);n=len(p)
 return dict(N=n,cover_E_correct=int((pred[p[:,0]]==1).sum()),sham_D_correct=int((pred[p[:,1]]==0).sum()),both_correct=int((pred[p]==[1,0]).all(1).sum()),any_prediction_U=int((pred[p]<0).any(1).sum()),relative_margin_ge2=int(((z[p[:,0]].astype(np.float64)-z[p[:,1]])>=2).sum()),relative_margin_ge2_but_cover_D=int((((z[p[:,0]].astype(np.float64)-z[p[:,1]])>=2)&(pred[p[:,0]]==0)).sum()))

def additive_reports(z,a,meta,d):
 groups={}
 for domain in sorted(set(meta['domain'])):
  for split in('TRAIN','VALIDATION'):
   mask=(meta['domain']==domain)&(meta['split']==split)
   if mask.any():groups[domain+'_'+split]=d.external_cm(z,a,meta,mask)
 return dict(new_samewood_TRAIN_pairs=pair_summary(z,a,meta['new_pairs2']),source_by_domain_split=groups,all_new924_retained=True,new_samewood_is_not_VAL=True,relative_margin_is_not_absolute_E=True,old194_donor_comparison_report_owned_by_root=True)

def validate_entry(entry_path):
 assert not RUN.exists(),'One fit only; preserve failure, no silent restart.'
 e=read(entry_path)
 assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
 for k in('trainer','protocol','design','generated','source_bindings','new_feature_pack','new_feature_receipt','new_source_admission','feature_integrity_review'):check(e[k])
 assert e['trainer']==bind(__file__)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 protocol=read(e['protocol']['path']);proof=read(e['generated']['path'])
 assert protocol['trainer']==e['trainer']and protocol['design']==e['design']and protocol['generated']==e['generated']and proof['PASS']and proof['code']==e['trainer']
 review=read(e['feature_integrity_review']['path']);assert review['completed']and review['PASS']and review['feature_receipt']==e['new_feature_receipt']and review['feature_pack']==e['new_feature_pack']
 design=read(DESIGN);assert e['source_bindings']==design['base_source']['old_source_binding']and e['new_source_admission']==design['new_source']['root_actual_execution_and_conditional_AUX_admission']
 admission=read(e['new_source_admission']['path']);assert admission['completed']and admission['current223_or_V112_used']is False and admission['PnP']==0 and admission['conditional_AUX_admission']['admitted']is True
 receipt=read(e['new_feature_receipt']['path'])
 assert receipt['completed']and receipt['PASS']and receipt['output']==e['new_feature_pack']and receipt['features_shape']==[NEWUNITS,1928]and receipt['occurrences']==NEW
 assert receipt['model_state_unchanged']and receipt['first_batch_repeat_exact']and receipt['old_C11_feature_units_exactly_reused']==28 and receipt['changed_fullRGB_units_freshly_encoded']==124
 assert receipt['input_occurrences']==design['new_source']['input_occurrences']
 return e,design,receipt

def run(entry_path):
 e,design,receipt=validate_entry(entry_path) # Entry/receipt/admission before feature and GT bodies.
 c12,c11,c10,c9,d=source_modules();parent=read(check(design['parent_C12_design']));assert design['parent_C12_trainer']==bind(C12)
 parent_protocol=read(check(read(PROTOCOL)['parent_C12_protocol']));material_receipt=read(check(parent_protocol['new_feature_receipt']))
 old=c12.load_metadata(e,material_receipt,parent,c10,d)
 inputs=rows(check(design['new_source']['input_occurrences']));gt=rows(check(design['new_source']['GT_occurrences']))
 new=join_new(inputs,gt);assert len(new['pairs'])==design['new_source']['accepted_pair_count'];meta=combine_metadata(old,new)
 assert {s:int((meta['typed_state']==s).sum())for s in'DESU'}==design['combined']['raw_states']
 assert {s:int((meta['known']&(meta['split']==s)).sum())for s in('TRAIN','VALIDATION')}=={'TRAIN':3233,'VALIDATION':636}
 # The exact C12 source features are reused, not C12 trained parameters.
 with np.load(check(parent['source']['C11_source_features']),allow_pickle=False)as f:
  x11=np.array(f['features'],copy=True);f11={k:np.array(f[k],copy=True)for k in('occurrence_unit','query_available','factual_query_available','context_available')}
 with np.load(check(parent_protocol['new_feature_pack']),allow_pickle=False)as f:
  xm=np.array(f['features'],copy=True);fm={k:np.array(f[k],copy=True)for k in('global_input_unit_index','query_available','factual_query_available','context_available')}
 with np.load(e['new_feature_pack']['path'],allow_pickle=False)as f:
  xn=np.array(f['features'],copy=True);fn={k:np.array(f[k],copy=True)for k in('occurrence_unit','query_available','factual_query_available','context_available','unit_Q_inside','global_occurrence_index')}
 assert x11.shape==(11652,1928)and xm.shape==(3200,1928)and xn.shape==(152,1928)and x11.dtype==xm.dtype==xn.dtype==np.float32
 assert np.array_equal(fm['global_input_unit_index'],np.arange(11652,14852))and np.array_equal(fn['global_occurrence_index'],np.arange(924))
 assert fn['occurrence_unit'].dtype==np.int32 and fn['occurrence_unit'].shape==(NEW,)and fn['unit_Q_inside'].shape==(152,)and fn['unit_Q_inside'].dtype==bool
 assert np.array_equal(fn['occurrence_unit'],expected_new_units(inputs))
 assert all(np.array_equal(xn[b],x11[inputs[33*b]['original_base_row_index']])for b in range(28))
 assert np.array_equal(xn[:,1927],fn['unit_Q_inside'].astype(np.float32))and not xn[~fn['unit_Q_inside'],384:768].any()
 for pack,count in((f11,41984),(fm,3200),(fn,924)):
  for k in('query_available','factual_query_available','context_available'):assert pack[k].shape==(count,)and pack[k].dtype==bool
  assert np.array_equal(pack['query_available'],pack['factual_query_available']&pack['context_available'])
 x=np.concatenate((x11,xm,xn));units=np.r_[f11['occurrence_unit'],fm['global_input_unit_index'],OLDUNITS+fn['occurrence_unit']].astype(np.int32)
 available=np.r_[f11['query_available'],fm['query_available'],fn['query_available']];factual=np.r_[f11['factual_query_available'],fm['factual_query_available'],fn['factual_query_available']];context=np.r_[f11['context_available'],fm['context_available'],fn['context_available']]
 assert x.shape==(UNITS,1928)and np.isfinite(x).all()and units.shape==available.shape==(N,)
 pools=c10.sampling_pools(old['old'],{k:old[k]for k in('known','split','external_target','typed_state','domain')},available[:BASE],d);mp=c12.material_pools(old,available[:BASE]);wp=available_pairs(meta,available)
 assert len(wp)==62,'All fixed62 pairs must have factual features; do not silently change intended pair pool.'
 assert(meta['source'][wp]=='PALLOX').all(),'V116 accepted source family must match frozen actual corpus.'
 selector10(c12,np.zeros(N,np.float32),available,meta,d)
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(211);torch.cuda.manual_seed_all(211);model=c11.SpatialTargetQueryHead().cuda();assert sum(p.numel()for p in model.parameters())==221825
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);xt=torch.as_tensor(x,device='cuda');ug=torch.as_tensor(units,dtype=torch.long,device='cuda');ag=torch.as_tensor(available,device='cuda');tg=torch.as_tensor(meta['external_target'],dtype=torch.float32,device='cuda')
 gens=[torch.Generator().manual_seed(s)for s in(208,214,215,216,217)];hashes=[hashlib.sha256()for _ in range(3)];history=[];updates=0;best=float('inf');selected=None;best_state=None
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),optimizer_updates=0,base_pool_counts={k:len(v)for k,v in pools.items()},material_pool_counts={k:len(v)for k,v in mp.items()},samewood_pair_pool_count=len(wp),known_counts={s:{t:int(((meta['split']==s)&(meta['typed_state']==t)).sum())for t in'DESU'}for s in('TRAIN','VALIDATION')},known_unavailable={s:int((meta['known']&(meta['split']==s)&~available).sum())for s in('TRAIN','VALIDATION')},all46108_retained=True));start=time.monotonic()
 try:
  def checkpoint(step):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},update=step,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path),head_class_code=bind(c12.C11),threshold_logit=0)
  @torch.no_grad()
  def evaluate(step):
   nonlocal best,selected,best_state
   model.eval();z=np.where(available,model(xt)[ug].cpu().numpy(),0).astype(np.float32);value,parts=selector10(c12,z,available,meta,d);row=dict(update=step,criterion_ten_VAL_cohorts_BCE=value,VAL_cohorts=parts);history.append(row)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   if value<best:best,selected=value,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt')
   return z
  torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');initial=evaluate(0);assert not initial.any();write(RUN/'INITIAL_SOURCE_REPORT.json',additive_reports(initial,available,meta,d))
  for step in range(1,2001):
   ix,pairs,audit,ix64,tr64,ix96,tr96=draw128(c12,c10,pools,mp,wp,*gens)
   assert meta['known'][ix].all()and available[ix].all()and(meta['split'][ix]=='TRAIN').all()and np.array_equal(np.bincount(meta['external_target'][ix],minlength=2),[64,64])
   assert pairs.shape==(24,2)and np.array_equal(meta['external_target'][ix[pairs]],np.tile([1,0],(24,1)))
   for h,ii,pp in zip(hashes,(ix64,ix96,ix),(tr64,tr96,pairs)):hash_draw(h,ii,pp)
   idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');pp=torch.as_tensor(pairs,dtype=torch.long,device='cuda');model.train();optimizer.zero_grad(set_to_none=True);z=model(xt[ug[idx]])
   loss,bce,hinge=supervised_loss(z,tg[idx],ag[idx],pp);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_cover_sham_batch_positions=pairs.tolist(),sampling=audit,available_loss_rows=128,loss=float(loss.detach()),BCE=float(bce.detach()),matched_hinge=float(hinge.detach()),gradient_norm_before_clip=float(norm)),allow_nan=False)+'\n')
   if step%100==0:evaluate(step)
  assert hashes[0].hexdigest()==OLD64_SHA and hashes[1].hexdigest()==OLD96_SHA and earliest_history(history)==(selected,best)
  torch.save(checkpoint(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);model.eval()
  with torch.no_grad():z=np.where(available,model(xt)[ug].cpu().numpy(),0).astype(np.float32)
  value,parts=selector10(c12,z,available,meta,d);assert value==best
  old_design=dict(design,source_gates={k:v for k,v in design['source_gates'].items()if not k.startswith('material_')});metrics,gates=c9.report(d,z[:33984],available[:33984],meta['old'],old_design)
  material,mgates=c12.material_report(d,z[:BASE],available[:BASE],old,parent);gates.update(mgates);assert len(gates)==15 and set(gates)==set(design['source_gates']);passed=all(v['PASS']for v in gates.values())
  line=meta['lineage'];ids=np.array([json.dumps({k:v for k,v in r.items()if k=='id'or k.endswith('_id')or k.endswith('_index')or k in('native_slot','logical_aux_slot')},sort_keys=True,separators=(',',':'),allow_nan=False)for r in line]);q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line]);L=np.array([r['L']if r['L']is not None else np.nan for r in line]);rawsplit=np.array([r.get('raw_split')or r.get('original_split')or meta['split'][i]for i,r in enumerate(line)])
  np.savez_compressed(RUN/'SELECTED_ALL46108_SOURCE_SCORES.npz',original_identifier_metadata_json=ids,original_split=rawsplit,query_logit=z,query_available=available,factual_query_available=factual,context_available=context,context_Q_inside=x[:,1927][units].astype(bool),prediction=d.decode(z,available),input_unit_index=units,raw_teacher_state=meta['typed_state'],external_target=meta['external_target'],split=meta['split'],domain=meta['domain'],source=meta['source'],variant=meta['variant'],query_xy_raw=q,L_raw=L,old_accepted_triplets=meta['old']['old_pairs'],new_selected_triplets=meta['old']['new_pairs'],samewood_accepted_cover_sham_pairs=meta['new_pairs2'],global_occurrence_index=np.arange(N))
  additive=additive_reports(z,available,meta,d);write(RUN/'ADDITIVE_SOURCE_REPORTS.json',additive)
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_ten_VAL_cohorts_BCE=best,VAL_cohorts=parts,metrics=metrics,material_surface=material,raw_state_counts={k:int((meta['typed_state']==k).sum())for k in'DESU'},known_by_split={s:int((meta['known']&(meta['split']==s)).sum())for s in('TRAIN','VALIDATION')},availability=dict(factual_unavailable=int((~factual).sum()),context_unavailable=int((~context).sum()),final_unavailable=int((~available).sum()),known_unavailable={s:int((meta['known']&(meta['split']==s)&~available).sum())for s in('TRAIN','VALIDATION')}),base64_all2000_draws_sha256=hashes[0].hexdigest(),base96_all2000_draws_sha256=hashes[1].hexdigest(),all128_all2000_draws_sha256=hashes[2].hexdigest(),all46108_retained=True,additive_reports=bind(RUN/'ADDITIVE_SOURCE_REPORTS.json'),learned_self_classification=False,native_mapping='UNKNOWN',human_gold=False,source_VAL_development_exposed=True,new_samewood_TRAIN_only=True))
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,gates=gates,current_inference_authorized=False,V112_inference_authorized=False))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),head_class_code=bind(c12.C11),trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),new_feature_pack=e['new_feature_pack'],new_feature_receipt=e['new_feature_receipt'],current_used_for_fit_or_selection=False,current_inference_authorized=False,V112_inference_authorized=False,PnP=0))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,selected_update=selected,criterion_ten_VAL_cohorts_BCE=best,all_source_gates_pass=passed,seconds=time.monotonic()-start,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL46108_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),additive_reports=bind(RUN/'ADDITIVE_SOURCE_REPORTS.json'),current_V112_or_PnP=0));print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',type=Path,required=True);run(p.parse_args().entry)
