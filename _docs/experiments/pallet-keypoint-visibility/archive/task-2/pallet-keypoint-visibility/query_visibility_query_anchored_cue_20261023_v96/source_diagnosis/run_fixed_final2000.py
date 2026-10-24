"""Authorized one-pass FINAL2000 source diagnostic; selected step0 unchanged."""
from pathlib import Path
import hashlib,importlib.util,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
V=Path(__file__).resolve().parent.parent;H=V/'source_diagnosis';R=V/'runs/C3_QUERY_ANCHORED_SOURCE_V1'
T=V/'source_fit_preparation/train_source_once.py';TSHA='330f817e45fe8cbb3cda2c48bea5f2502b2f9053ce8e2921892fef44920ade03'
OUT=H/'FINAL2000_FIXED_NOT_SELECTOR';PLAN=H/'FINAL2000_BEFORE_ARRAY_AND_PARAMETER_READS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,o):
 with Path(p).open('x')as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def verify(d):assert sha(d['path'])==d['sha256']
def state_sha(model):
 h=hashlib.sha256()
 for k,v in sorted(model.state_dict().items()):
  a=v.detach().cpu().contiguous().numpy();h.update(k.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
 return h.hexdigest()
def prepare():
 f=read(R/'MODEL_FREEZE.json');complete=read(R/'FIT_COMPLETE.json');e=read(f['entry']['path']);assert f['completed']and complete['completed']and complete['optimizer_updates']==2000
 assert f['selected_update']==complete['selected_update']==0 and f['updates']==2000 and f['source_only']
 assert f['final_checkpoint']['path']==str(R/'FINAL_CHECKPOINT.pt')
 for d in [f['final_checkpoint'],f['checkpoint'],f['model_code'],f['entry'],f['protocol']]:verify(d)
 assert sha(T)==TSHA and not OUT.exists()
 paths={k:e[k]for k in ['model_code','input_pack_manifest','input_pack_seal','base_input_lineage','GT_ledger','protocol','trainer','independent_input_audit']}
 for d in paths.values():verify(d)
 write(PLAN,dict(completed=True,root_explicit_authorization='One fixed FINAL_CHECKPOINT update2000 public-source forward all31680, same inputs and three-tensor operator. Diagnostic only, no selection/replacement/optimizer/current/PnP.',fixed_update=2000,selected_update_remains=0,
   code=b(__file__),checkpoint=f['final_checkpoint'],selected_checkpoint=f['checkpoint'],selected_source_freeze=b(R/'MODEL_FREEZE.json'),fit_complete=b(R/'FIT_COMPLETE.json'),bindings=paths,entry=f['entry'],
   original_run_files_before={p.name:b(p)for p in sorted(R.iterdir())if p.is_file()},forward_passes=1,expected_rows=31680,batch_size=128,repeat_passes=0,
   runtime='pallet-yolo26 Python3.10 Torch2.1.1+cu118 CUDA0; deterministic/noTF32',diagnostic_only_not_selector=True,new_optimizer_current_backbone_initializer_PnP=0,
   checkpoint_parameter_or_source_array_reads_before_this_manifest=0))
def run():
 p=read(PLAN);assert b(__file__)==p['code']and p['fixed_update']==2000 and p['selected_update_remains']==0
 for d in p['bindings'].values():verify(d)
 verify(p['checkpoint']);verify(p['selected_source_freeze']);verify(p['entry'])
 import torch
 assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 s=importlib.util.spec_from_file_location('v96_fixed_final_shared',T);t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
 e=read(p['entry']['path']);protocol=read(e['protocol']['path'])
 OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',dict(plan=b(PLAN),fixed_checkpoint=p['checkpoint'],no_repeat_or_selector=True))
 manifest,lineage,meta=t.load_metadata(e,protocol);data,unit,factual=t.load_inputs(e,lineage)
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(194);torch.cuda.manual_seed_all(194)
 model=t.operator().QueryCenterScalar().to('cuda:0');payload=torch.load(p['checkpoint']['path'],map_location='cuda:0',weights_only=True)
 assert payload['update']==2000 and payload['parameter_count']==26573 and t.same_binding(payload['model_code'],e['model_code'])and t.same_binding(payload['protocol'],e['protocol'])
 model.load_state_dict(payload['state_dict'],strict=True);model.eval().requires_grad_(False);before=state_sha(model);calls=[0,0]
 def count(_,args):calls[0]+=1;calls[1]+=len(args[0])
 hook=model.register_forward_pre_hook(count);start=time.monotonic()
 with torch.inference_mode():z,available=t.predict(model,data,unit,'cuda:0')
 hook.remove();assert calls==[11,1348]and np.array_equal(available,factual)and np.isfinite(z).all()
 assert np.all(z[~available]==0)and state_sha(model)==before
 results,gates,passed=t.report(z,available,meta);criterion,strata=t.selector(z,available,meta)
 history=[json.loads(x)for x in(R/'HISTORY.jsonl').read_text().splitlines()];last=next(x for x in history if x['update']==2000)
 history_delta=criterion-last['criterion_five_VAL_cohorts_BCE']
 labels=meta['labels'];nll=(np.logaddexp(np.float32(0),z)-np.where(labels>=0,labels,0)*z).astype(np.float32)
 np.savez_compressed(OUT/'ALL31680_FINAL_SOURCE_SCORES.npz',query_logit=z.astype(np.float32),query_available=available,prediction=t.decode(z,available),NLL_FP32=nll,
  candidate_known=meta['known'],scalar_target_D0_E1_Uminus1=labels,source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],occurrence_index=np.arange(31680),base_row_index=np.arange(31680)//33,donor_rank=np.array([-1 if r['donor_rank']is None else r['donor_rank']for r in lineage],np.int8),input_unit_index=unit,
  occurrence_id=np.array([r['occurrence_id']for r in lineage]),input_id=np.array([r['input_id']for r in lineage]),native_slot_logical_only=np.array([r['native_slot_logical_only']for r in lineage]),
  image_sha256=np.array([r['image_sha256']for r in lineage]),original_image_sha256=np.array([r['original_image_sha256']for r in lineage]),query_xy_raw=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in lineage]),L_raw=np.array([r['L']if r['L']is not None else np.nan for r in lineage]),accepted_pair=meta['pair_ready'])
 write(OUT/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,fixed_diagnostic_update=2000,authoritative_selected_update=0,diagnostic_not_selector=True,checkpoint=p['checkpoint'],selected_source_freeze=p['selected_source_freeze'],
  counts=meta['counts'],metrics=results,diagnostic_existing_seven_gate_checks=gates,diagnostic_existing_seven_checks_allpass=passed,
  five_VAL_cohorts_BCE_diagnostic=criterion,VAL_strata=strata,original_final_history_criterion=last['criterion_five_VAL_cohorts_BCE'],history_criterion_difference=history_delta,
  all31680_preserved=True,one_pass=True,repeat_passes=0,forward_batches=calls[0],forward_unique_input_units=calls[1],logical_occurrences=31680,seconds=time.monotonic()-start,
  model_state_sha256=before,model_unchanged=True,no_checkpoint_reselection_or_replacement=True,current_authorized=False,new_optimizer_backbone_initializer_current_PnP=0,
  source_semantics='Conditional AI query-anchored cue/controlled augmentation; native mapping UNKNOWN and not human semantic gold. This posthoc final-checkpoint diagnostic cannot replace selected step0.'))
 for name,d in p['original_run_files_before'].items():assert b(R/name)==d
 write(OUT/'DIAGNOSIS_COMPLETE.json',dict(completed=True,PASS_execution=True,plan=b(PLAN),code=b(__file__),fixed_checkpoint=p['checkpoint'],authoritative_selected_source_freeze=p['selected_source_freeze'],
  results=b(OUT/'SOURCE_RESULTS.json'),scores=b(OUT/'ALL31680_FINAL_SOURCE_SCORES.npz'),all_original_run_files_unchanged=True,model_unchanged=True,one_source_pass_1348_units_expanded_to31680_occurrences=True,
  selected_update_remains=0,diagnostic_only=True,new_optimizer_backbone_initializer_private_current_PnP=0))
 print(json.dumps(dict(complete=b(OUT/'DIAGNOSIS_COMPLETE.json'),TRAIN=results['TRAIN'],VALIDATION=results['VALIDATION'],criterion=criterion,history_delta=history_delta)),flush=True)
if __name__=='__main__':prepare();run()
