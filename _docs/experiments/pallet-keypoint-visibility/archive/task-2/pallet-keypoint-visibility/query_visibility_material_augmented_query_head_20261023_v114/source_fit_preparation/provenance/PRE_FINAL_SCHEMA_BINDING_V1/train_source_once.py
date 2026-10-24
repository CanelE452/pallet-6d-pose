"""C12 preparation is inert. Only a root entry admits one source-only fit."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C12_DESIGN_BEFORE_SOURCE_FORWARD.json';DESIGN_SHA='59cb674e7e3a9e884df9f4653c27709508926acc27da72d3042f65b4acf3e32b'
C11=P/'query_visibility_spatial_target_query_features_20261023_v111/train_spatial_target_query_head.py';C11_SHA='f34ce3b44cc491593bbe6b63fa498c0417a5ba70f54318787094ac0802efa5de'
RUN=V/'runs/C12_MATERIAL_AUGMENTED_SPATIAL_QUERY'
BASE,NNEW,N,OLDUNITS,UNITS=41984,3200,45184,11652,14852
EXPECTED_OLD_DRAW_SHA='8d6ec2eecbdd0937e710173328cea8e0d4211d21b11d722faea5d5c6fb8cf680'
MATERIAL='V113_MATERIAL_SURFACE'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(d):assert sha(d['path'])==d['sha256'],d['path'];return Path(d['path'])
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def source_modules():
 assert sha(C11)==C11_SHA
 s=importlib.util.spec_from_file_location('c12_exact_c11',C11);c11=importlib.util.module_from_spec(s);s.loader.exec_module(c11)
 c10=c11.pinned_c10();c9,d=c10.old_modules();return c11,c10,c9,d

def join_material(inputs,lineage,state,target,query_copy,L_copy):
 assert len(inputs)==400 and len(lineage)==3200 and state.shape==target.shape==(400,8)
 assert query_copy.shape==(400,8,2)and L_copy.shape==(400,)and np.isin(state,list('DESU')).all()
 assert np.array_equal(target,np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8))
 for i,r in enumerate(inputs):
  assert r['instance_row_index']==i and r['source_emulation']is True and r['actual_initializer_query']is False and r['native_mapping']=='UNKNOWN'
  assert r['split']==('TRAIN'if r['original_split']=='train'else'VALIDATION')and r['original_split']in('train','val')
  assert np.array_equal(query_copy[i],r['initial_points'])and L_copy[i]==r['roi_side']
  for k in range(8):
   j=i*8+k;l=lineage[j]
   assert l['local_occurrence_index']==l['local_input_unit_index']==j and l['global_occurrence_index']==BASE+j and l['global_input_unit_index']==OLDUNITS+j
   assert l['instance_row_index']==i and l['plan_row_index']==r['plan_row_index']and l['logical_aux_slot']==k
   assert l['input_id']==r['input_id']and l['query_id']==r['input_id']+'::q'+str(k)
   assert l['query_xy']==r['initial_points'][k]and l['L']==r['roi_side']and l['image']==r['image']and l['raw_hw']==r['raw_hw']
   assert l['query_box_xyxy']==r['query_box_xyxy']and l['split']==r['split']and l['original_split']==r['original_split']and l['source']==r['source']
 return dict(typed_state=state.reshape(-1),labels=target.reshape(-1).astype(np.int8),split=np.repeat([r['split']for r in inputs],8),domain=np.full(3200,MATERIAL),source=np.array([r['source']for r in lineage]),variant=np.full(3200,'NATURAL_SCENE'))

def load_metadata(e,receipt,design,c10,d):
 c10_design=read(check(design['source']['C10_design']));c10_receipt=read(check(design['source']['C10_feature_receipt']))
 old=c10.load_metadata(e,c10_receipt,c10_design,d)
 teacher=read(check(design['source']['material_teacher_seal']));assert teacher['completed']and teacher['all3200_preserved']and teacher['inputs']==design['source']['material_input']
 packet=next(x for x in teacher['outputs']if Path(x['path']).name=='RAW_SURFACE_TEACHER3200_GT_ONLY.npz')
 inputs=rows(check(design['source']['material_input']));lineage=rows(check(receipt['new_query_lineage']))
 with np.load(check(packet),allow_pickle=False)as f:
  new=join_material(inputs,lineage,f['raw_surface_teacher_state'],f['external_target_D0_E1_else_minus1'],f['query_xy_input_copy'],f['roi_side_input_copy'])
 meta={k:np.concatenate((old[k],new[k]))for k in('typed_state','labels','split','domain','source','variant')}
 meta.update(known=np.isin(meta['typed_state'],['D','E']),known_geometric_S=meta['typed_state']=='S',unresolved_teacher_U=meta['typed_state']=='U',lineage=old['lineage']+lineage,old=old['old'],material_teacher_packet=packet)
 meta['external_target']=np.where(meta['known'],meta['labels'],-1).astype(np.int8)
 assert {s:int((meta['typed_state']==s).sum())for s in 'DESU'}==design['source']['raw_states_before_availability']
 assert {s:int((meta['known']&(meta['split']==s)).sum())for s in('TRAIN','VALIDATION')}==design['source']['known_before_availability']
 return meta

def material_pools(meta,available):
 pools={s:np.flatnonzero((meta['domain']==MATERIAL)&(meta['split']=='TRAIN')&(meta['typed_state']==s)&available)for s in('D','E')}
 assert all(len(x)>0 for x in pools.values()),'Empty required materialTRAIN D/E: no fit/fallback.'
 for s,ix in pools.items():assert(meta['typed_state'][ix]==s).all()and available[ix].all()and(meta['split'][ix]=='TRAIN').all()
 return pools

def draw96(c10,basepools,newpools,oldgen,materialgen,shufflegen):
 old,tr,old_audit=c10.draw(basepools,oldgen)
 md=newpools['D'][torch.randint(len(newpools['D']),(16,),generator=materialgen).numpy()]
 me=newpools['E'][torch.randint(len(newpools['E']),(16,),generator=materialgen).numpy()]
 before=np.concatenate((old,md,me));perm=torch.randperm(96,generator=shufflegen).numpy();inv=np.argsort(perm)
 return before[perm],inv[tr],dict(base64_occurrence_indices=old.tolist(),base64_triplet_positions=tr.tolist(),base64_sampling=old_audit,material_D_occurrence_indices=md.tolist(),material_E_occurrence_indices=me.tolist(),final96_permutation=perm.tolist()),old,tr

def selector10(z,a,meta,d):
 _,parts=d.selector(z[:33984],a[:33984],meta['old'])
 assert len(parts)==8
 safe=np.where(a,z,0).astype(np.float64);loss=np.logaddexp(0.,safe)-np.where(meta['known'],meta['external_target'],0)*safe
 for state in('D','E'):
  mask=(meta['domain']==MATERIAL)&(meta['split']=='VALIDATION')&(meta['typed_state']==state);assert mask.any(),'Empty required materialVAL cohort.'
  parts['material_VALIDATION_'+state]=dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~a).sum()))
 assert len(parts)==10
 return float(np.mean([x['mean_BCE']for x in parts.values()])),parts

def material_report(d,z,a,meta,design):
 result={s:d.external_cm(z,a,meta,(meta['domain']==MATERIAL)&(meta['split']==s))for s in('TRAIN','VALIDATION')};gates={};pred=d.decode(z,a)
 for state,target in [('D',0),('E',1)]:
  mask=(meta['domain']==MATERIAL)&(meta['split']=='VALIDATION')&(meta['typed_state']==state);assert mask.any()
  key='material_VAL_'+state+'_recall_min';gates[key]=dict(value=float((pred[mask]==target).mean()),limit=design['source_gates'][key],N=int(mask.sum()))
 mask=(meta['domain']==MATERIAL)&(meta['split']=='VALIDATION')&meta['known'];key='material_VAL_known_U_rate_max'
 gates[key]=dict(value=float((~a[mask]).mean()),limit=design['source_gates'][key],N=int(mask.sum()))
 for key,v in gates.items():v['PASS']=v['value']<=v['limit']if key.endswith('_max')else v['value']>=v['limit']
 return result,gates

def earliest_history(history):
 best=float('inf');step=None
 for r in history:
  if r['criterion_ten_VAL_cohorts_BCE']<best:best=r['criterion_ten_VAL_cohorts_BCE'];step=r['update']
 return step,best

def run(entry_path):
 assert not RUN.exists(),'One source fit only; no silent restart.'
 e=read(entry_path);assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))and e['new_source_visual_admitted']is True
 for k in('trainer','design','feature_pack','feature_receipt','source_bindings','new_source_visual_admission','generated'):check(e[k])
 assert e['trainer']==bind(__file__)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 proof=read(e['generated']['path']);assert proof['PASS']and proof['code']==e['trainer']
 design=read(DESIGN);admission=read(e['new_source_visual_admission']['path'])
 assert admission['completed']and admission['conditional_source_training_admitted']and admission['teacher_corpus']==design['source']['material_teacher_seal']and admission['current_evaluation_training_allowed']is False and admission['PnP_allowed']is False
 receipt=read(e['feature_receipt']['path']);assert receipt['completed']and receipt['PASS']and receipt['output']==e['feature_pack']and receipt['features_shape']==[3200,1928]and receipt['occurrences']==3200
 assert receipt['model_state_unchanged']and receipt['first_batch_repeat_exact']
 c11,c10,c9,d=source_modules();assert design['source']['C11_source_trainer']==bind(C11)and e['source_bindings']==design['source']['old_source_binding']
 for k in('C11_source_feature_receipt','C11_source_features'):check(design['source'][k])
 oldreceipt=read(design['source']['C11_source_feature_receipt']['path']);assert oldreceipt['completed']and oldreceipt['PASS']and oldreceipt['output']==design['source']['C11_source_features']
 meta=load_metadata(e,receipt,design,c10,d)
 with np.load(design['source']['C11_source_features']['path'],allow_pickle=False)as f:
  oldx=np.array(f['features'],copy=True);oldmeta={k:np.array(f[k],copy=True)for k in('occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index')}
 with np.load(e['feature_pack']['path'],allow_pickle=False)as f:
  newx=np.array(f['features'],copy=True);nm={k:np.array(f[k],copy=True)for k in('occurrence_unit','query_available','factual_query_available','context_available','global_occurrence_index','global_input_unit_index','unit_Q_inside')}
 assert oldx.shape==(11652,1928)and newx.shape==(3200,1928)and oldx.dtype==newx.dtype==np.float32
 assert oldmeta['occurrence_unit'].shape==(BASE,)and np.array_equal(oldmeta['global_occurrence_index'],np.arange(BASE))
 assert np.array_equal(nm['occurrence_unit'],np.arange(3200))and nm['occurrence_unit'].dtype==np.int32 and np.array_equal(nm['global_occurrence_index'],np.arange(BASE,N))and np.array_equal(nm['global_input_unit_index'],np.arange(OLDUNITS,UNITS))
 features=np.concatenate((oldx,newx));units=np.r_[oldmeta['occurrence_unit'],nm['global_input_unit_index']].astype(np.int32)
 available=np.r_[oldmeta['query_available'],nm['query_available']];factual=np.r_[oldmeta['factual_query_available'],nm['factual_query_available']];context=np.r_[oldmeta['context_available'],nm['context_available']]
 assert features.shape==(UNITS,1928)and np.isfinite(features).all()and units.shape==available.shape==factual.shape==context.shape==(N,)
 assert available.dtype==factual.dtype==context.dtype==bool and np.array_equal(available,factual&context)
 assert np.array_equal(features[:OLDUNITS],oldx)and np.array_equal(newx[:,1927],nm['unit_Q_inside'].astype(np.float32))and not newx[~nm['unit_Q_inside'],384:768].any()
 assert ((features[:,1924:]>=0)&(features[:,1924:]<=1)).all()and np.isin(features[:,1927],[0,1]).all()
 # Old64 RNG/pools retain exact C10 behavior and all old factual flags.
 pools=c10.sampling_pools(meta['old'],{k:meta[k][:BASE]for k in('known','split','external_target','typed_state','domain')},available[:BASE],d);mp=material_pools(meta,available)
 selector10(np.zeros(N,np.float32),available,meta,d)
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(211);torch.cuda.manual_seed_all(211);model=c11.SpatialTargetQueryHead().cuda();assert sum(p.numel()for p in model.parameters())==221825
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);x=torch.as_tensor(features,device='cuda');ug=torch.as_tensor(units,dtype=torch.long,device='cuda');ag=torch.as_tensor(available,device='cuda');tg=torch.as_tensor(meta['external_target'],dtype=torch.float32,device='cuda')
 oldgen=torch.Generator().manual_seed(208);mg=torch.Generator().manual_seed(214);sg=torch.Generator().manual_seed(215);oldhash=hashlib.sha256();allhash=hashlib.sha256();history=[]
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry_path),optimizer_updates=0,base_pool_counts={k:len(v)for k,v in pools.items()},material_pool_counts={k:len(v)for k,v in mp.items()},all45184_retained=True,oldC11_features_copied_bitexact=True,known_unavailable={s:int((meta['known']&(meta['split']==s)&~available).sum())for s in('TRAIN','VALIDATION')}))
 updates=0;best=float('inf');selected=None;best_state=None;start=time.monotonic()
 try:
  def checkpoint(step):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},update=step,design=bind(DESIGN),trainer=bind(__file__),entry=bind(entry_path),head_class_code=bind(C11))
  @torch.no_grad()
  def evaluate(step):
   nonlocal best,selected,best_state
   model.eval();z=np.where(available,model(x)[ug].cpu().numpy(),0).astype(np.float32);value,parts=selector10(z,available,meta,d);row=dict(update=step,criterion_ten_VAL_cohorts_BCE=value,VAL_cohorts=parts);history.append(row)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   if value<best:best,selected=value,step;best_state=checkpoint(step);torch.save(best_state,RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt')
   return z
  torch.save(checkpoint(0),RUN/'INITIAL_CHECKPOINT.pt');assert not np.count_nonzero(evaluate(0))
  for step in range(1,2001):
   ix,tr,audit,baseix,basetr=draw96(c10,pools,mp,oldgen,mg,sg)
   assert meta['known'][ix].all()and available[ix].all()and(meta['split'][ix]=='TRAIN').all()and np.array_equal(np.bincount(meta['external_target'][ix],minlength=2),[48,48])
   assert np.array_equal(meta['external_target'][ix][tr],np.tile([0,1,0],(8,1)))
   for h,xx,tt in((oldhash,baseix,basetr),(allhash,ix,tr)):h.update(xx.astype('<i8').tobytes());h.update(tt.astype('<i8').tobytes())
   idx=torch.as_tensor(ix,dtype=torch.long,device='cuda');triples=torch.as_tensor(tr,dtype=torch.long,device='cuda');model.train();optimizer.zero_grad(set_to_none=True);z=model(x[ug[idx]])
   loss,bce,hinge=c10.supervised_loss(z,tg[idx],ag[idx],triples);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5);assert torch.isfinite(norm);optimizer.step();updates=step
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,occurrence_indices=ix.tolist(),input_unit_indices=units[ix].tolist(),matched_triplet_batch_positions=tr.tolist(),sampling=audit,available_loss_rows=96,sampled_unavailable_rows=0,loss=float(loss.detach()),BCE=float(bce.detach()),matched_hinge=float(hinge.detach()),gradient_norm_before_clip=float(norm)),allow_nan=False)+'\n')
   if step%100==0:evaluate(step)
  assert oldhash.hexdigest()==EXPECTED_OLD_DRAW_SHA and earliest_history(history)==(selected,best)
  torch.save(checkpoint(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state['state_dict']);model.eval()
  with torch.no_grad():z=np.where(available,model(x)[ug].cpu().numpy(),0).astype(np.float32)
  value,parts=selector10(z,available,meta,d);assert value==best
  old_design=dict(design,source_gates={k:v for k,v in design['source_gates'].items()if not k.startswith('material_')});metrics,gates=c9.report(d,z[:33984],available[:33984],meta['old'],old_design)
  material,mgates=material_report(d,z,available,meta,design);gates.update(mgates);assert len(gates)==15 and set(gates)==set(design['source_gates']);passed=all(v['PASS']for v in gates.values())
  ycb=d.external_cm(z,available,meta,meta['domain']=='V109_YCB_SURFACE');line=meta['lineage'];q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line]);L=np.array([r['L']if r['L']is not None else np.nan for r in line])
  np.savez_compressed(RUN/'SELECTED_ALL45184_SOURCE_SCORES.npz',query_logit=z,query_available=available,factual_query_available=factual,context_available=context,context_Q_inside=features[:,1927][units].astype(bool),prediction=d.decode(z,available),input_unit_index=units,raw_teacher_state=meta['typed_state'],external_target=meta['external_target'],split=meta['split'],domain=meta['domain'],source=meta['source'],variant=meta['variant'],query_xy_raw=q,L_raw=L,old_accepted_triplets=meta['old']['old_pairs'],new_selected_triplets=meta['old']['new_pairs'],global_occurrence_index=np.arange(N))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,selected_update=selected,criterion_ten_VAL_cohorts_BCE=best,VAL_cohorts=parts,metrics=metrics,YCB_surface_TRAIN=ycb,material_surface=material,raw_state_counts={k:int((meta['typed_state']==k).sum())for k in'DESU'},known_by_split={s:int((meta['known']&(meta['split']==s)).sum())for s in('TRAIN','VALIDATION')},availability=dict(factual_unavailable=int((~factual).sum()),context_unavailable=int((~context).sum()),final_unavailable=int((~available).sum()),known_unavailable={s:int((meta['known']&(meta['split']==s)&~available).sum())for s in('TRAIN','VALIDATION')}),base64_all2000_draws_sha256=oldhash.hexdigest(),all96_all2000_draws_sha256=allhash.hexdigest(),same_C10_base64_sampler_trace=True,learned_self_classification=False,native_mapping='UNKNOWN',human_gold=False,source_VAL_development_exposed=True,foam21_E_and_VAL_absent=True))
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,gates=gates,current_inference_authorized=False,V112_inference_authorized=False))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,selected_update=selected,checkpoint=bind(RUN/'BEST_SPATIAL_TARGET_QUERY_HEAD.pt'),head_class_code=bind(C11),old_feature_pack=design['source']['C11_source_features'],feature_pack=e['feature_pack'],feature_receipt=e['feature_receipt'],trainable_parameters=221825,encoder_trainable_parameters=0,threshold_logit=0,design=bind(DESIGN),trainer=bind(__file__),current_used_for_fit_or_selection=False,current_inference_authorized=False,V112_inference_authorized=False,PnP=0))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,selected_update=selected,criterion_ten_VAL_cohorts_BCE=best,all_source_gates_pass=passed,seconds=time.monotonic()-start,source_results=bind(RUN/'SOURCE_RESULTS.json'),gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL45184_SOURCE_SCORES.npz'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),current_V112_or_PnP=0));print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(optimizer_updates_completed=updates,traceback=traceback.format_exc(),restart_automatically=False));raise

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--entry',type=Path,required=True);run(parser.parse_args().entry)
