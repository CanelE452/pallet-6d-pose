"""Own saved-only arithmetic. Never imports trainer/model or calls forward."""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V=HERE.parent
RUN=V/'runs/C1_NATURAL_KNOWNSET_SOURCE_V1'

def bind(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def check(d):
 p=Path(d['path']);assert bind(p)['sha256']==d['sha256'],str(p);return p
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def write(p,d):
 assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def samebinding(a,b):return Path(a['path']).resolve()==Path(b['path']).resolve()and a['sha256']==b['sha256']
def close(a,b):
 assert a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)

def metrics(z,available,y,known,part):
 pred=np.where(available,(z>=0).astype(np.int8),-1)
 cm=np.array([[int(np.sum(part&known&(y==gt)&(pred==pr)))for pr in [0,1,-1]]for gt in [0,1]],np.int64)
 totals=cm.sum(1);n=int(cm.sum());out=dict(matrix=cm.tolist(),known_N=n,known_U_count=int(cm[:,2].sum()),all_rows_N=int(part.sum()),unknown_GT_N=int(np.sum(part&~known)),unknown_GT_pred_distribution={str(pr):int(np.sum(part&~known&(pred==pr)))for pr in [0,1,-1]})
 out['D_recall']=float(cm[0,0]/totals[0]);out['E_recall']=float(cm[1,1]/totals[1]);out['known_U_rate']=float(cm[:,2].sum()/n)
 out['D_precision']=float(cm[0,0]/cm[:,0].sum())if cm[:,0].sum()else None
 out['E_precision']=float(cm[1,1]/cm[:,1].sum())if cm[:,1].sum()else None
 out['GT_unknown_is_negative']=False
 return out

def main():
 assert torch.__version__=='2.1.1+cu118'
 policy=read(HERE/'POLICY_BEFORE_SAVED_FIT_OUTPUTS.json')
 complete=read(RUN/'FIT_COMPLETE.json');assert complete['completed']is True and complete['PASS_execution']is True and complete['optimizer_updates']==2000
 for k in ['MODEL_FREEZE','source_results','source_gates','scores','history','train_updates']:check(complete[k])
 freeze=read(complete['MODEL_FREEZE']['path']);result=read(complete['source_results']['path']);gate=read(complete['source_gates']['path']);started=read(RUN/'RUN_STARTED.json')
 entry=read(check(freeze['entry']));assert entry['approved_actual_source_fit']and entry['source_only']and not entry['current_allowed']and not entry['PnP_allowed']and not entry['allow_restart_after_optimizer_update']
 protocol=read(check(entry['protocol']))
 for k,d in protocol['bindings'].items():check(d);assert samebinding(d,entry[k]),k
 assert bind(entry['trainer']['path'])['sha256']==policy['trainer']['sha256']
 assert bind(entry['protocol']['path'])['sha256']==policy['protocol']['sha256']
 assert samebinding(started['entry'],freeze['entry'])and samebinding(started['trainer'],entry['trainer'])
 assert started['initial_updates']==0 and started['restart_forbidden_after_any_update']is True
 assert freeze['updates']==2000 and freeze['parameter_count']==26573 and freeze['threshold_rawlogit']==0
 for k in ['checkpoint','initial_checkpoint','final_checkpoint','source_gates','source_results']:check(freeze[k])
 assert samebinding(freeze['model_code'],entry['model_code'])and samebinding(freeze['protocol'],entry['protocol'])
 assert freeze['source_input_bindings']==entry
 ledger=rows(entry['GT_ledger']['path']);assert len(ledger)==800
 y=np.array([r['scalar_target_D0_E1_Uminus1']for r in ledger],np.int8);known=np.array([r['candidate_known']for r in ledger],bool)
 assert np.array_equal(known,np.isin(y,[0,1]))
 # Read only identity/coordinate/factual arrays from the audited source pack.
 inputseal=read(entry['input_pack_seal']['path']);assert inputseal['completed']and inputseal['PASS']
 manifest=read(check(inputseal['manifest']));parts={}
 fields=['query_id','frame_index','native_slot','split','image_sha256','query_xy_raw','L_raw','operator_centre_support_v94_draft','original_query_continuous_valid']
 for c in manifest['chunks']:
  check(c['file'])
  with np.load(c['file']['path'],allow_pickle=False)as a:
   for k in fields:parts.setdefault(k,[]).append(a[k].copy())
 data={k:np.concatenate(x)for k,x in parts.items()};splits=data['split']
 with np.load(complete['scores']['path'],allow_pickle=False)as a:
  score={k:a[k].copy()for k in a.files}
 assert set(score)=={'query_logit','query_available','prediction','candidate_known','scalar_target_D0_E1_Uminus1','query_id','frame_index','native_slot','split','original_q_raw','original_L_raw'}
 assert score['query_logit'].shape==(800,)and score['query_logit'].dtype==np.float32 and np.isfinite(score['query_logit']).all()
 z=score['query_logit'];available=score['query_available'];assert available.dtype==bool
 assert np.array_equal(available,data['operator_centre_support_v94_draft'])
 assert not z[~available].any()
 assert np.array_equal(score['candidate_known'],known)and np.array_equal(score['scalar_target_D0_E1_Uminus1'],y)
 for k in ['query_id','frame_index','native_slot','split']:assert np.array_equal(score[k],data[k]),k
 assert np.array_equal(score['original_q_raw'],data['query_xy_raw'],equal_nan=True)and np.array_equal(score['original_L_raw'],data['L_raw'],equal_nan=True)
 for i,r in enumerate(ledger):
  assert r['row_index']==i and r['query_id']==data['query_id'][i]and r['image_sha256']==data['image_sha256'][i]
  assert r['split']==splits[i]and r['native_slot']==data['native_slot'][i]and r['frame_index']==data['frame_index'][i]
  assert r['native_expected_part_state']=='UNKNOWN'and r['human_gold']is False
  if r['original_q']is None:assert np.isnan(score['original_q_raw'][i]).all()and np.isnan(score['original_L_raw'][i])
  else:assert np.array_equal(r['original_q'],score['original_q_raw'][i])and r['original_L']==score['original_L_raw'][i]
 pred=np.where(available,(z>=0).astype(np.int8),-1);assert np.array_equal(pred,score['prediction'])
 recomputed={s:metrics(z,available,y,known,splits==s)for s in ['TRAIN_LAB','VAL_HALL']}
 for s,m in recomputed.items():
  published=result['metrics'][s]
  for k in ['matrix','known_N','known_U_count','all_rows_N','unknown_GT_N','unknown_GT_pred_distribution']:assert m[k]==published[k],(s,k)
  for k in ['D_recall','E_recall','known_U_rate']:close(m[k],published[k])
 losses=np.logaddexp(0.,z.astype(np.float64))-y*z.astype(np.float64)
 val=splits=='VAL_HALL';classmeans=[float(losses[val&known&(y==t)].mean())for t in [0,1]];macro=float(sum(classmeans)/2)
 close(macro,result['selector_macro_BCE_all_VAL_known']);close(classmeans[0],result['VAL_D_BCE']);close(classmeans[1],result['VAL_E_BCE'])
 history=rows(complete['history']['path']);assert len(history)==21 and [r['update']for r in history]==list(range(0,2001,100))
 best=float('inf');chosen=None;history_replay=[]
 for r in history:
  u=r['update'];assert r['VAL_D_N']==81 and r['VAL_E_N']==11
  own=(r['VAL_D_BCE_all_known']+r['VAL_E_BCE_all_known'])/2
  close(own,r['criterion_macro_BCE_all_VAL_candidateknown'])
  assert r['VAL_known_unavailable']==int(np.sum(val&known&~available))
  cm=np.array(r['VAL_metrics']['matrix']);assert np.array_equal(cm.sum(1),[81,11])and cm[:,2].sum()==r['VAL_known_unavailable']
  improved=own<best
  if improved:best=own;chosen=u
  history_replay.append(dict(update=u,class_BCE_average=own,saved_macro=r['criterion_macro_BCE_all_VAL_candidateknown'],difference=own-r['criterion_macro_BCE_all_VAL_candidateknown'],strict_improvement=improved))
 assert chosen==freeze['selected_update']==complete['selected_update']==result['selected_update']
 close(best,macro)
 selectedhistory=next(h for h in history if h['update']==chosen)
 assert selectedhistory['VAL_metrics']['matrix']==recomputed['VAL_HALL']['matrix']
 # Exact own sampler replay: three RNG calls per update, no trainer import.
 pools=[np.flatnonzero((splits=='TRAIN_LAB')&known&(y==t))for t in [0,1]];assert list(map(len,pools))==[56,9]
 generator=torch.Generator(device='cpu').manual_seed(194)
 updates=rows(complete['train_updates']['path']);assert len(updates)==2000
 for u,r in enumerate(updates,1):
  picked=[p[torch.randint(0,len(p),(32,),generator=generator).numpy()]for p in pools]
  expected=np.concatenate(picked)[torch.randperm(64,generator=generator).numpy()]
  actual=np.array(r['query_indices'],np.int64)
  assert r['update']==u and np.array_equal(actual,expected),u
  assert len(actual)==64 and np.all(known[actual])and np.all(splits[actual]=='TRAIN_LAB')
  assert int(np.sum(y[actual]==0))==r['batch_D']==32 and int(np.sum(y[actual]==1))==r['batch_E']==32
  assert r['candidate_known_count']==64 and r['loss_available_known_count']==int(available[actual].sum())and r['known_unavailable_count']==int((~available[actual]).sum())
  assert np.isfinite(r['loss'])and np.isfinite(r['gradient_norm_before_clip'])
 own_gates={'VAL_D_recall':dict(value=recomputed['VAL_HALL']['D_recall'],minimum=.85),'VAL_E_recall':dict(value=recomputed['VAL_HALL']['E_recall'],minimum=.75),'VAL_known_U_rate':dict(value=recomputed['VAL_HALL']['known_U_rate'],maximum=.05)}
 for k,d in own_gates.items():
  d['PASS']=d['value']>=d['minimum']if 'minimum'in d else d['value']<=d['maximum']
  assert d==gate['gates'][k],k
 passed=all(d['PASS']for d in own_gates.values());assert passed==gate['PASS']==gate['all_gates_pass']
 assert gate['current_inference_authorized']is False
 # CPU checkpoint schema/metadata only; no model construction or tensor forward.
 ckpt=torch.load(freeze['checkpoint']['path'],map_location='cpu')
 assert ckpt['update']==chosen and ckpt['seed']==194 and ckpt['parameter_count']==26573
 assert samebinding(ckpt['model_code'],entry['model_code'])and samebinding(ckpt['protocol'],entry['protocol'])and ckpt['source_input_bindings']==entry
 state=ckpt['state_dict'];assert 'native_offsets'in state and tuple(state['native_offsets'].shape)==(1,2,9,9)
 assert sum(t.numel()for k,t in state.items()if k!='native_offsets')==26573
 assert all(t.device.type=='cpu'and t.dtype==torch.float32 and torch.isfinite(t).all()for t in state.values())
 steps=[float(d['step'])for d in ckpt['optimizer_state']['state'].values()];assert steps and all(s==chosen for s in steps)
 write(HERE/'HISTORY_SELECTION_REPLAY.json',dict(completed=True,PASS=True,rows=history_replay,earliest_selected=chosen))
 report=dict(completed=True,PASS=True,blockers=[],policy=bind(HERE/'POLICY_BEFORE_SAVED_FIT_OUTPUTS.json'),FIT_COMPLETE=bind(RUN/'FIT_COMPLETE.json'),MODEL_FREEZE=complete['MODEL_FREEZE'],entry=freeze['entry'],protocol=entry['protocol'],trainer=entry['trainer'],model_code=entry['model_code'],GT_ledger=entry['GT_ledger'],input_pack_seal=entry['input_pack_seal'],saved_scores=complete['scores'],code=bind(Path(__file__)),
  all800=dict(ID_order_q_L_split_native_imageSHA_GT_exact=True,known=int(known.sum()),unknown_GT=int((~known).sum()),query_available=int(available.sum()),query_unavailable=int((~available).sum()),rawsign_predictions_and_U_exact=True,unavailable_safezero_exact=True,all800_retained=True),metrics=recomputed,source_gates=own_gates,all_source_gates_pass=passed,current_inference_authorized_by_audit=False,
  selection=dict(history_rows=21,updates=list(range(0,2001,100)),selected_update=chosen,selected_macro_BCE=macro,selected_class_BCE=classmeans,selected_macro_published_difference=macro-result['selector_macro_BCE_all_VAL_known'],strict_min_earliest_tie=True,history_replay=bind(HERE/'HISTORY_SELECTION_REPLAY.json')),
  sampler=dict(updates=2000,batch64=True,class32_32=True,TRAIN_pools56_9=True,seed194_full128000_indices_exact=True,unknown_or_VAL_draws=0,all_logged_available_counts_exact=True,finite_logged_losses_gradnorms=True),
  checkpoint=dict(selected=freeze['checkpoint'],CPU_metadata_schema_only=True,parameter_count=26573,state_keys=len(state),buffer_excluded='native_offsets162',optimizer_step_metadata_all=chosen,model_not_constructed_or_forwarded=True),
  restart_evidence=dict(RUN_STARTED_initial0=True,unique_contiguous2000_updates=True,initial_best_final_files_bound=True,entry_restart_false=True,limit='Saved records and prefit code show the declared single run; this is not an exhaustive proof that no external hidden run exists elsewhere.'),
  limits=['Only selected checkpoint predictions are independently recomputed from saved logits. Each of21 history criteria is reconstructed from saved per-class BCE and checked for strict selection/denominators; unavailable earlier per-checkpoint logits are not re-inferred.','Individual train loss/gradient values are checked finite and sampler/admission counts exact; no per-step neural forward/backward is replayed.','Conditional recorded FULL physical-cue known-set AI proxy only,31 known rows retain null limits; native expectedpart UNKNOWN/humanfalse. Tall single-bin lab/hall development split is not native/public-flat/private generalization or goal completion.','No raw RGB/masks/P3 feature VALUES were needed; pack identity/factual arrays only. Selected checkpoint tensors were loaded on CPU for schema/count/update inspection; no model object, forward, fit, PnP or current artifact access.'],
  comparison_tolerance=dict(class_predictions_counts_and_indices='EXACT',reported_float_absolute=1e-12,no_threshold_or_policy_change=True),created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
 write(HERE/'REVIEW.json',report)
 write(HERE/'SEAL.json',dict(completed=True,PASS=True,review=bind(HERE/'REVIEW.json'),policy=bind(HERE/'POLICY_BEFORE_SAVED_FIT_OUTPUTS.json'),code=bind(Path(__file__)),FIT_COMPLETE=bind(RUN/'FIT_COMPLETE.json')))
 print(json.dumps(dict(review=bind(HERE/'REVIEW.json'),seal=bind(HERE/'SEAL.json'),metrics=recomputed,source_gate=passed)))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  f=HERE/'FAILED_AUDIT.json'
  if not f.exists():write(f,dict(completed=False,PASS=False,error=repr(e),traceback=traceback.format_exc(),no_agent_fit_restart=True,original_outputs_untouched=True))
  raise
