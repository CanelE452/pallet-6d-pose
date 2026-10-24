"""C4: exactC1 model state, fresh lowLR optimizer, unchanged source operator.
No source data/weights read by import. Actual fit requires root entry.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
DESIGN=V/'ROOT_C4_DESIGN_BEFORE_SOURCE_FIT.json';DESIGN_SHA='ecc3281f2aacd8e4a631eacf17c459cbb8dfaaf8b00d1caa1c9c60a901c5d6cb'
SHARED=P/'query_visibility_query_anchored_cue_20261023_v96/source_fit_preparation/train_source_once.py';SHARED_SHA='330f817e45fe8cbb3cda2c48bea5f2502b2f9053ce8e2921892fef44920ade03'
PROTOCOL=HERE/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json';RUN=V/'runs/C4_C1_WARMSTART_SOURCE_V1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for x in iter(lambda:f.read(1<<20),b''):h.update(x)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def verify(b):assert sha(b['path'])==b['sha256'],b['path']
def same(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def shared():
 assert sha(SHARED)==SHARED_SHA
 s=importlib.util.spec_from_file_location('c4_bound_c3_helpers',SHARED);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def settings(t):
 p=dict(t.SETTINGS);p['AdamW_lr']=1e-5;p['initialization']='Exact immutableC1selected100 state_dict; freshAdamW/all trainable; full31680 update0 bitexact baseline before optimizer'
 return p
def require_entry(e,t):
 assert e.get('approved_actual_source_fit')is True and e.get('source_only')is True
 assert e.get('current_allowed')is False and e.get('PnP_allowed')is False and e.get('allow_restart_after_optimizer_update')is False
 for k in list(t.BINDINGS)+['shared_trainer','reference_C1_freeze','reference_C1_checkpoint','reference_C1_samepack_scores']:verify(e[k])
 assert same(e['trainer'],bind(__file__))and same(e['protocol'],bind(PROTOCOL))and same(e['shared_trainer'],bind(SHARED))and e['shared_trainer']['sha256']==SHARED_SHA
 assert same(e['candidate_definition'],bind(DESIGN))and e['candidate_definition']['sha256']==DESIGN_SHA
 d=read(DESIGN);assert not d['current_allowed']and not d['PnP_allowed']and d['optimizer']['lr']==1e-5
 for key in ['reference_C1_freeze','reference_C1_checkpoint','reference_C1_samepack_scores']:assert same(e[key],d[key])
 assert same(e['input_pack_seal'],d['immutable_data'])and same(e['GT_ledger'],d['weak_GT'])and same(e['model_code'],bind(t.MODEL))
 p=read(PROTOCOL);assert p['settings']==settings(t)and p['source_only']and not p['current_allowed']
 for k,b in p['bindings'].items():assert k in e and same(e[k],b),k
 for k in ['generated_tests','independent_preparation_review','independent_input_audit']:assert read(e[k]['path'])['PASS']is True
 assert same(read(e['generated_tests']['path'])['trainer'],e['trainer'])
 # This is the pre-existing independent review of the unchanged shared helpers,
 # not a claim that the new C4 wrapper received a separate independent review.
 assert same(read(e['independent_preparation_review']['path'])['trainer'],e['shared_trainer'])
 assert Path(e['run_directory']).resolve()==RUN.resolve()and not RUN.exists()
 return p
def apply_C1_state(model,payload,model_code):
 """Pure for generated fixtures. Old optimizer state is deliberately ignored."""
 assert payload['update']==100 and payload['parameter_count']==26573 and same(payload['model_code'],model_code)
 model.load_state_dict(payload['state_dict'],strict=True)
 for p in model.parameters():p.requires_grad_(True)
 assert all(p.requires_grad for p in model.parameters())
def checkpoint(model,opt,update,e):
 return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,seed=194,parameter_count=26573,model_code=e['model_code'],protocol=e['protocol'],source_input_bindings=e,warmstart_C1_checkpoint=e['reference_C1_checkpoint'])
def run(entry):
 e=read(entry);t=shared();p=require_entry(e,t)
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),trainer=bind(__file__),optimizer_updates=0,restart_forbidden_after_any_update=True));updates=0
 try:
  manifest,lineage,meta=t.load_metadata(e,p);cohort_counts=t.feasibility(meta)
  if any(v==0 for c in cohort_counts.values()for v in c.values()):
   write(RUN/'NO_FIT_FEASIBILITY_ONLY.json',dict(completed=True,fit_executed=False,reason='Required TRAIN/VAL cohort empty',cohort_counts=cohort_counts,current_allowed=False));return
  pools=t.make_pools(meta);data,unit,factual=t.load_inputs(e,lineage)
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  random.seed(194);np.random.seed(194);torch.manual_seed(194);torch.cuda.manual_seed_all(194)
  op=t.operator();device='cuda:0';model=op.QueryCenterScalar().to(device);assert sum(x.numel()for x in model.parameters())==26573
  freeze=read(e['reference_C1_freeze']['path']);assert freeze['completed']and freeze['source_only']and freeze['selected_update']==100 and same(freeze['checkpoint'],e['reference_C1_checkpoint'])
  payload=torch.load(e['reference_C1_checkpoint']['path'],map_location=device,weights_only=True)
  assert same(payload['protocol'],freeze['protocol'])
  apply_C1_state(model,payload,e['model_code']);model.eval()
  # This full matching forward is required BEFORE optimizer construction/step.
  z0,a0=t.predict(model,data,unit,device)
  with np.load(e['reference_C1_samepack_scores']['path'],allow_pickle=False)as reference:
   assert np.array_equal(z0,reference['query_logit']),'C1 update0 logits mismatch: stop before optimizer'
   assert np.array_equal(a0,reference['query_available'])and np.array_equal(a0,factual)
   assert np.array_equal(reference['occurrence_id'],[r['occurrence_id']for r in lineage])and np.array_equal(reference['scalar_target_D0_E1_Uminus1'],meta['labels'])
  write(RUN/'UPDATE0_C1_PARITY.json',dict(completed=True,PASS=True,all31680_logit_bitexact=True,all31680_availability_bitexact=True,all_occurrence_ids_and_labels_exact=True,optimizer_updates=0,optimizer_constructed_before_parity=False,reference=e['reference_C1_samepack_scores'],warmstart=e['reference_C1_checkpoint'],all26573parameters_trainable=True))
  opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.0001);assert not opt.state
  g=torch.Generator(device='cpu').manual_seed(194)
  torch.save(checkpoint(model,opt,0,e),RUN/'INITIAL_CHECKPOINT.pt');best=float('inf');best_update=None;best_state=None;started=time.monotonic()
  def evaluate(update,prediction=None):
   nonlocal best,best_update,best_state
   z,a=t.predict(model,data,unit,device)if prediction is None else prediction;assert np.array_equal(a,factual)
   value,cohorts=t.selector(z,a,meta)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,criterion_five_VAL_cohorts_BCE=value,VAL_cohorts=cohorts),allow_nan=False)+'\n')
   if value<best:
    best=value;best_update=update;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};torch.save(checkpoint(model,opt,update,e),RUN/'BEST_QUERY_CENTER_SCALAR.pt')
  evaluate(0,(z0,a0))
  for update in range(1,2001):
   ix=t.draw(pools,g);model.train();opt.zero_grad(set_to_none=True);out=t.forward(model,data,unit[ix],device)
   loss=op.known_binary_loss(out,torch.as_tensor(meta['labels'][ix],device=device),torch.as_tensor(meta['known'][ix],device=device));assert torch.isfinite(loss['loss'])
   loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step();updates=update
   quota={name:int(mask[ix].sum())for name,mask in t.cohort_masks(meta,'TRAIN').items()};assert list(quota.values())==list(t.QUOTAS)and meta['known'][ix].all()
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,occurrence_indices=ix.tolist(),input_unit_indices=unit[ix].tolist(),cohort_quotas=quota,loss=float(loss['loss'].detach()),available_known_count=int(loss['available_known_count']),known_unavailable_count=int(loss['known_unavailable_count']),gradient_norm_before_clip=float(grad)),allow_nan=False)+'\n')
   if update%100==0:evaluate(update)
  torch.save(checkpoint(model,opt,2000,e),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state);z,a=t.predict(model,data,unit,device);assert np.array_equal(a,factual)
  value,cohorts=t.selector(z,a,meta);assert value==best;results,checks,passed=t.report(z,a,meta);label=meta['labels'];nll=(np.logaddexp(np.float32(0),z)-np.where(label>=0,label,0)*z).astype(np.float32)
  np.savez_compressed(RUN/'SELECTED_ALL31680_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=a,prediction=t.decode(z,a),NLL_FP32=nll,candidate_known=meta['known'],scalar_target_D0_E1_Uminus1=label,source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],occurrence_index=np.arange(t.N),occurrence_id=np.array([r['occurrence_id']for r in lineage]),base_row_index=np.arange(t.N)//33,donor_rank=np.array([-1 if r['donor_rank']is None else r['donor_rank']for r in lineage],np.int8),input_unit_index=unit,input_id=np.array([r['input_id']for r in lineage]),native_slot_logical_only=np.array([r['native_slot_logical_only']for r in lineage]),image_sha256=np.array([r['image_sha256']for r in lineage]),original_image_sha256=np.array([r['original_image_sha256']for r in lineage]),query_xy_raw=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage],np.float64),L_raw=np.array([r['L']if r['L']is not None else np.nan for r in lineage],np.float64),accepted_pair=meta['pair_ready'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,PASS=passed,all_gates_pass=passed,gates=checks,current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,selected_update=best_update,selector_five_VAL_cohorts_BCE=value,VAL_cohorts=cohorts,counts=meta['counts'],metrics=results,all31680_preserved=True,input_units=len(data['p3_roi']),native_expected_part_state='UNKNOWN',human_gold=False,NLL_unknown_diagnostic_only=True,update0_C1_parity=bind(RUN/'UPDATE0_C1_PARITY.json'),initialization_and_LR_changed_jointly=True))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,source_only=True,model_code=e['model_code'],checkpoint=bind(RUN/'BEST_QUERY_CENTER_SCALAR.pt'),initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),parameter_count=26573,selected_update=best_update,updates=2000,selector=settings(t)['selector'],threshold_rawlogit=0,protocol=e['protocol'],entry=bind(entry),source_input_bindings=e,warmstart=e['reference_C1_checkpoint'],update0_C1_parity=bind(RUN/'UPDATE0_C1_PARITY.json'),current_inference_authorized=False))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-started,MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL31680_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),update0_parity=bind(RUN/'UPDATE0_C1_PARITY.json'),new_current_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(completed=False,optimizer_updates_completed=updates,exception=traceback.format_exc(),restart_automatically=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--entry',required=True);run(ap.parse_args().entry)
