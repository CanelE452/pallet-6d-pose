"""C14: exact C13 source trainer helpers, one full context-aligned feature loader."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DESIGN=V/'ROOT_C14_DESIGN_BEFORE_SOURCE_FORWARD.json';DESIGN_SHA='01e2bf2ae6f19d554dbd01cca2894aa2ab87219c47d81711d011a1c8f7b9c7f6'
C13=P/'query_visibility_same_material_query_head_20261023_v118/source_fit_preparation/train_source_once.py';C13_SHA='48418a0e1ca495a489ce19a95f1c08c2c37f7fa9c5ed46ec5f6ab36e053a54e4'
assert hashlib.sha256(C13.read_bytes()).hexdigest()==C13_SHA
spec=importlib.util.spec_from_file_location('c14_pinned_c13',C13);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sha,bind,check,read,rows,write=m.sha,m.bind,m.check,m.read,m.rows,m.write
source_modules=m.source_modules;join_new=m.join_new;combine_metadata=m.combine_metadata;available_pairs=m.available_pairs;draw128=m.draw128;supervised_loss=m.supervised_loss;selector10=m.selector10;earliest_history=m.earliest_history;hash_draw=m.hash_draw;additive_reports=m.additive_reports
BASE,NEW,N,OLDUNITS,NEWUNITS,UNITS=m.BASE,m.NEW,m.N,m.OLDUNITS,m.NEWUNITS,m.UNITS
OLD64_SHA=m.OLD64_SHA;OLD96_SHA=m.OLD96_SHA;FULL128_SHA='ece1ac68638dc2d93a75139f209fce266b560bb707d0f12b241fe80c42e76cc9'
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FORWARD.json';RUN=V/'runs/C14_CONSISTENT_Q8_CONTEXT_SPATIAL_QUERY'
FIELDS=('occurrence_unit','query_available','factual_query_available','context_available')

def validate_feature_control(new,original,unit_metadata=None):
 x=new['features'];old=original['features']
 assert x.shape==old.shape==(15004,1928)and x.dtype==old.dtype==np.float32 and np.isfinite(x).all()
 assert np.array_equal(x[:,768:1920],old[:,768:1920])and np.array_equal(x[:,1925:1927],old[:,1925:1927]),'F/C/P/localfraction changed'
 for k in FIELDS:
  assert new[k].shape==original[k].shape==(N,)and new[k].dtype==original[k].dtype
  assert np.array_equal(new[k],original[k]),'C13 mapping/availability changed: '+k
 assert new['occurrence_unit'].dtype==np.int32
 for k in FIELDS[1:]:assert new[k].dtype==bool
 assert np.array_equal(new['query_available'],new['factual_query_available']&new['context_available'])
 assert np.array_equal(new['global_occurrence_index'],np.arange(N))and np.issubdtype(new['global_occurrence_index'].dtype,np.integer)
 q=new['unit_Q_inside'];assert q.shape==(15004,)and q.dtype==bool and np.array_equal(x[:,1927],q.astype(np.float32))and not x[~q,384:768].any()
 assert ((x[:,1924:]>=0)&(x[:,1924:]<=1)).all()
 assert np.array_equal((x[:,1924]>0)[new['occurrence_unit']],new['context_available'])
 if unit_metadata is not None:
  assert len(unit_metadata)==15004
  aliases=0
  for i,r in enumerate(unit_metadata):
   assert r['source_feature_unit_index']==i
   if r['whole1928_exact_reuse_eligible_by_bound_metadata']:assert np.array_equal(x[i],old[i]),'Equal-box row was recomputed'
   if r['is_explicit_V116_BEFORE_alias']:
    canonical=r['canonical_context_feature_unit_index'];assert np.array_equal(x[i],x[canonical]),'Alias not updated consistently';aliases+=1
   if r['context_unavailable']:assert not x[i,:768].any()and x[i,1924]==x[i,1927]==0
  assert aliases==28
 return True

def validate_entry(path):
 assert not RUN.exists(),'One fit only; no silent restart'
 e=read(path);assert e['approved_source_fit']is True and all(e[k]is False for k in('current_allowed','V112_allowed','PnP_allowed','restart_after_update'))
 for k in('trainer','protocol','design','generated','source_bindings','new_feature_pack','new_feature_receipt','new_source_admission','feature_integrity_review'):check(e[k])
 assert e['trainer']==bind(__file__)and e['design']==bind(DESIGN)and sha(DESIGN)==DESIGN_SHA
 p=read(e['protocol']['path']);proof=read(e['generated']['path']);assert p['trainer']==e['trainer']and p['design']==e['design']and p['generated']==e['generated']and proof['PASS']and proof['code']==e['trainer']
 review=read(e['feature_integrity_review']['path']);assert review['completed']and review['PASS']and review['feature_pack']==e['new_feature_pack']and review['feature_receipt']==e['new_feature_receipt']
 design=read(DESIGN);assert design['parent_C13_trainer']==bind(C13)and e['source_bindings']==design['base_source']['old_source_binding']and e['new_source_admission']==design['new_source']['root_actual_execution_and_conditional_AUX_admission']
 ad=read(e['new_source_admission']['path']);assert ad['completed']and ad['conditional_AUX_admission']['admitted']and not ad['current223_or_V112_used']and ad['PnP']==0
 receipt=read(e['new_feature_receipt']['path']);assert receipt['completed']and receipt['PASS']and receipt['output']==e['new_feature_pack']and receipt['features_shape']==[15004,1928]and receipt['occurrences']==46108 and receipt['model_state_unchanged']and receipt['first_batch_repeat_exact']
 parent=read(check(design['parent_C13_design']));assert design['sampling']['full128_draw_SHA256_required']==FULL128_SHA
 assert all(design[k]==parent[k]for k in('head','optimizer','selector','source_gates','combined'))
 return e,design,receipt

def load_features(e,design):
 protocol=read(PROTOCOL);parent_protocol=read(check(protocol['parent_C12_protocol']));freeze=read(check(protocol['parent_C13_freeze']))
 sources=[design['base_source']['C11_source_features'],parent_protocol['new_feature_pack'],freeze['new_feature_pack']]
 packs=[]
 for i,b in enumerate(sources):
  with np.load(check(b),allow_pickle=False)as f:
   pack={k:np.array(f[k],copy=True)for k in ('features',)+FIELDS[1:]}
   pack['occurrence_unit']=np.array(f['global_input_unit_index']if i==1 else f['occurrence_unit'],copy=True)
  packs.append(pack)
 original={'features':np.concatenate([p['features']for p in packs]),'occurrence_unit':np.r_[packs[0]['occurrence_unit'],packs[1]['occurrence_unit'],OLDUNITS+packs[2]['occurrence_unit']].astype(np.int32)}
 original.update({k:np.concatenate([p[k]for p in packs])for k in FIELDS[1:]})
 with np.load(e['new_feature_pack']['path'],allow_pickle=False)as f:new={k:np.array(f[k],copy=True)for k in('features',)+FIELDS+('global_occurrence_index','unit_Q_inside')}
 validate_feature_control(new,original,rows(check(design['metadata_inputs']['units'])))
 return new

def run(entry_path):
 e,design,receipt=validate_entry(entry_path) # Before any actual source/feature/GT body.
 c12,c11,c10,c9,d=source_modules();parent=read(check(design['parent_C12_design']))
 parent_protocol=read(check(read(PROTOCOL)['parent_C12_protocol']));material_receipt=read(check(parent_protocol['new_feature_receipt']))
 old=c12.load_metadata(e,material_receipt,parent,c10,d)
 inputs=rows(check(design['new_source']['input_occurrences']));gt=rows(check(design['new_source']['GT_occurrences']));new=join_new(inputs,gt);assert len(new['pairs'])==62;meta=combine_metadata(old,new)
 assert {s:int((meta['typed_state']==s).sum())for s in'DESU'}==design['combined']['raw_states']
 assert {s:int((meta['known']&(meta['split']==s)).sum())for s in('TRAIN','VALIDATION')}=={'TRAIN':3233,'VALIDATION':636}
 data=load_features(e,design);x=data['features'];units=data['occurrence_unit'];available=data['query_available'];factual=data['factual_query_available'];context=data['context_available']
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
  assert hashes[0].hexdigest()==OLD64_SHA and hashes[1].hexdigest()==OLD96_SHA and hashes[2].hexdigest()==FULL128_SHA and earliest_history(history)==(selected,best)
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
