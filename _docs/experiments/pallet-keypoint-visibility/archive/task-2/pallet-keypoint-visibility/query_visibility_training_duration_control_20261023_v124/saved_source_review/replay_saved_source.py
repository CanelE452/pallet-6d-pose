"""Independent saved NumPy/JSON duration audit; no Torch, model or feature imports."""
from pathlib import Path
import hashlib,json,math
import numpy as np
V=Path(__file__).resolve().parent.parent;R=V/'runs/C16_EXACT_C15_DURATION_CONTROL_8000';O=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):
 with Path(p).open()as f:
  for s in f:
   if s.strip():yield json.loads(s)
def check(b):assert sha(b['path'])==b['sha256'],b;return Path(b['path'])
def checktree(x):
 if isinstance(x,dict):
  if set(x)>={'path','sha256'}:check(x)
  for v in x.values():checktree(v)
 elif isinstance(x,list):
  for v in x:checktree(v)

def run():
 assert not(O/'REVIEW.json').exists(),'Preserve existing audit'
 fit=read(R/'FIT_COMPLETE.json');assert fit['completed']and(fit['optimizer_updates'],fit['total_updates'],fit['replayed_C15_updates'],fit['continued_updates'])==(8000,8000,1000,7000);checktree(fit)
 freeze=read(check(fit['model_freeze']));cert=read(check(fit['replay_certificate']));s=read(check(fit['source_results']));g=read(check(fit['gates']));design=read(V/'ROOT_C16_DESIGN_BEFORE_SOURCE_FIT.json');parent=read(check(design['parent_C15_design']));p14=read(check(parent['parent_C14_design']));checktree(freeze);checktree(cert)
 flags=('batches_exact','losses_exact','step1000_parameters_exact','step1000_scores_exact','step1000_fourteen_BCE_exact','optimizer_and_RNG_not_reset_at1000','history0_to1000_exact')
 assert cert['completed']and cert['PASS']and cert['replayed_updates']==1000 and all(cert[k]is True for k in flags)
 assert freeze['completed']and freeze['threshold_logit']==0 and freeze['total_optimizer_updates']==8000 and freeze['optimizer_and_RNG_resume_saved']and freeze['current_used_for_fit_or_selection']is False and freeze['PnP']==0 and freeze['design']==bind(V/'ROOT_C16_DESIGN_BEFORE_SOURCE_FIT.json')
 hist=list(lines(R/'HISTORY.jsonl'));phist=list(lines(check(design['parent_C15_history'])));curve=list(lines(check(fit['source_learning_curve'])))
 assert len(hist)==len(curve)==81 and [r['update']for r in hist]==[r['update']for r in curve]==list(range(0,8001,100))and hist[:11]==phist
 names=list(hist[0]['VAL_cohorts']);assert len(names)==14
 counts={k:v['count']for k,v in hist[0]['VAL_cohorts'].items()}
 for row,bc in zip(hist,curve):
  assert list(row['VAL_cohorts'])==names and len(row['VAL_cohorts'])==14
  assert all(v['count']==counts[k]and v['count']>0 and v['unavailable']>=0 and math.isfinite(v['mean_BCE'])for k,v in row['VAL_cohorts'].items())
  assert float(np.mean([v['mean_BCE']for v in row['VAL_cohorts'].values()]))==row['criterion_fourteen_VAL_cohorts_BCE']
  assert bc['VAL_cohorts']==row['VAL_cohorts']
 best=min(hist,key=lambda r:r['criterion_fourteen_VAL_cohorts_BCE']);step=best['update'];criterion=best['criterion_fourteen_VAL_cohorts_BCE']
 assert step==fit['selected_update']==freeze['selected_update']==freeze['update']==s['selected_update']and criterion==fit['criterion_fourteen_VAL_cohorts_BCE']==s['criterion_fourteen_VAL_cohorts_BCE']and best['VAL_cohorts']==s['VAL_cohorts']
 with np.load(check(fit['scores']),allow_pickle=False)as f:a={k:np.array(f[k],copy=True)for k in f.files}
 n=51004;z=a['query_logit'];y=a['external_target'];av=a['query_available'];pred=np.where(av,(z>=0).astype(np.int8),-1);known=np.isin(y,[0,1]);va=a['split']=='VALIDATION';tr=a['split']=='TRAIN';D=y==0;E=y==1;dom=a['domain'];src=a['source'];v=a['variant']
 assert z.shape==(n,)and z.dtype==np.float32 and np.isfinite(z).all()and np.array_equal(a['global_occurrence_index'],np.arange(n))and np.array_equal(pred,a['prediction'])and np.array_equal(av,a['factual_query_available']&a['context_available'])
 assert np.array_equal(y,np.where(a['raw_teacher_state']=='D',0,np.where(a['raw_teacher_state']=='E',1,-1)))
 with np.load(check(design['parent_C15_scores']),allow_pickle=False)as f:
  for k in a:
   if k in ('query_logit','prediction'):continue
   assert np.array_equal(a[k],f[k],equal_nan=True)if np.issubdtype(a[k].dtype,np.floating)else np.array_equal(a[k],f[k]),k
 old=dom=='V96_REAL';syn=dom=='V99_SYNTH';mat=dom=='V113_MATERIAL_SURFACE';flat=dom=='V122_PUBLIC_FLAT_PALLET'
 masks={'old_LOCO_BEFORE_D':old&va&D&(src=='LOCO')&(v=='BEFORE'),'old_PALLOX_BEFORE_D':old&va&D&(src=='PALLOX')&(v=='BEFORE'),'old_SHAM_D':old&va&D&(v=='CUE_PRESERVING_SHAM'),'old_BEFORE_E':old&va&E&(v=='BEFORE'),'old_COVER_E':old&va&E&(v=='OPAQUE_COVER'),'new_BEFORE_D':syn&va&D&(v=='BEFORE'),'new_SHAM_D':syn&va&D&(v=='SHAM'),'new_COVER_E':syn&va&E&(v=='COVER'),'material_VALIDATION_D':mat&va&D,'material_VALIDATION_E':mat&va&E,'LOCO_VALIDATION_D':flat&va&D&(src=='LOCO'),'LOCO_VALIDATION_E':flat&va&E&(src=='LOCO'),'TOMIE_DEVELOPMENT_D':flat&va&D&(src=='TOMIE'),'TOMIE_DEVELOPMENT_E':flat&va&E&(src=='TOMIE')}
 safe=np.where(av,z,0).astype(np.float64);loss=np.logaddexp(0,safe)-np.where(known,y,0)*safe;parts={}
 for k,mask in masks.items():
  parts[k]=dict(count=int(mask.sum()),mean_BCE=float(loss[mask].mean()),unavailable=int((mask&~av).sum()));assert parts[k]==best['VAL_cohorts'][k],k
 assert float(np.mean([parts[k]['mean_BCE']for k in names]))==criterion
 for split in('TRAIN','VALIDATION'):
  m=known&(a['split']==split);assert s['known_unavailable'][split]==int((m&~av).sum());assert curve[step//100]['known_'+split+'_mean_BCE']==float(loss[m].mean())
 # Independent all8000 saved draws, class balance, role structure and exact C15 first1000 trace.
 oldhash=hashlib.sha256();newhash=hashlib.sha256();prefix=None;replayed=iter(lines(check(design['parent_C15_trace'])));num=0
 for num,row in enumerate(lines(R/'TRAIN_UPDATES.jsonl'),1):
  assert row['update']==num;ix=np.asarray(row['occurrence_indices'],np.int64);pairs=np.asarray(row['matched_cover_sham_batch_positions'],np.int64)
  assert ix.shape==(192,)and pairs.shape==(24,2)and pairs.min()>=0 and pairs.max()<128
  assert(ix>=0).all()and(ix<n).all()and known[ix].all()and av[ix].all()and tr[ix].all()and np.array_equal(a['input_unit_index'][ix],row['input_unit_indices'])
  assert np.array_equal(np.bincount(y[ix],minlength=2),[96,96])and np.array_equal(np.bincount(y[ix[:128]],minlength=2),[64,64])and np.array_equal(np.bincount(y[ix[128:]],minlength=2),[32,32])
  assert np.array_equal(y[ix[pairs]],np.tile([1,0],(24,1)))and np.array_equal(ix[176:],ix[128:144]+1)
  assert(ix[128:]>=46108).all()and(src[ix[128:]]=='LOCO').all()and(v[ix[128:160]]=='OPAQUE_COVER').all()and(v[ix[160:176]]=='BEFORE').all()and(v[ix[176:]]=='CUE_PRESERVING_SHAM').all()
  assert row['available_loss_rows']==192 and np.array_equal(row['new_correspondence_positions_not_in_hinge'],[[128+i,176+i]for i in range(16)])
  assert all(math.isfinite(row[k])for k in ('loss','old_BCE','old_hinge','new_BCE','gradient_norm_before_clip'))
  oldhash.update(ix[:128].astype('<i8').tobytes());oldhash.update(pairs.astype('<i8').tobytes());newhash.update(ix[128:].astype('<i8').tobytes())
  if num<=1000:assert row==next(replayed),'C15 first1000 full loss/draw trace mismatch'
  if num==1000:prefix=(oldhash.hexdigest(),newhash.hexdigest())
 assert num==8000 and prefix==(cert['old128_first1000_draws_sha256'],cert['new64_first1000_draws_sha256'])and oldhash.hexdigest()==s['old128_all8000_draws_sha256']and newhash.hexdigest()==s['new64_all8000_draws_sha256']
 # Independent unchanged15 gate values, absolute sign decisions and all-row denominators.
 recall=lambda mask,state:float((pred[mask]==state).sum()/mask.sum())
 u=lambda mask:float((pred[mask]<0).sum()/mask.sum())
 pairval={}
 for key in ('old_accepted_triplets','new_selected_triplets'):
  q=a[key];q=q[va[q[:,0]]];assert np.array_equal(y[q],np.tile([0,1,0],(len(q),1)));pairval[key]=float((pred[q]==[0,1,0]).all(1).mean())
 vals={'old_LOCO_before_D_recall_min':recall(masks['old_LOCO_BEFORE_D'],0),'old_PALLOX_before_D_recall_min':recall(masks['old_PALLOX_BEFORE_D'],0),'old_natural_before_E_recall_min':recall(masks['old_BEFORE_E'],1),'old_sham_D_recall_min':recall(masks['old_SHAM_D'],0),'old_cover_E_recall_min':recall(masks['old_COVER_E'],1),'old_accepted_VAL_pair_all3_min':pairval['old_accepted_triplets'],'old_known_VAL_U_rate_max':u(old&va&known),'new_BEFORE_D_recall_min':recall(masks['new_BEFORE_D'],0),'new_SHAM_D_recall_min':recall(masks['new_SHAM_D'],0),'new_COVER_E_recall_min':recall(masks['new_COVER_E'],1),'new_selected_VAL_pair_all3_min':pairval['new_selected_triplets'],'new_known_VAL_U_rate_max':u(syn&va&known),'material_VAL_D_recall_min':recall(masks['material_VALIDATION_D'],0),'material_VAL_E_recall_min':recall(masks['material_VALIDATION_E'],1),'material_VAL_known_U_rate_max':u(mat&va&known)}
 assert set(vals)==set(g['gates'])==set(p14['source_gates'])and len(vals)==15
 for k,value in vals.items():
  saved=g['gates'][k];limit=p14['source_gates'][k];passed=value<=limit if k.endswith('_max')else value>=limit
  assert saved['value']==value and saved['limit']==limit and saved['PASS']==passed,k
 assert g['all_gates_pass']==fit['all_source_gates_pass']==all(v['PASS']for v in g['gates'].values())
 result=dict(completed=True,PASS=True,blockers=[],scope='Independent source-only saved arithmetic; no model/weight deserialization/forward/current/V112.',fit_complete=bind(R/'FIT_COMPLETE.json'),source_freeze=fit['model_freeze'],scores=fit['scores'],replay_certificate=fit['replay_certificate'],history=bind(R/'HISTORY.jsonl'),train_trace=bind(R/'TRAIN_UPDATES.jsonl'),script=bind(Path(__file__)),all51004_metadata_labels_qL_order_availability_exact_C15=True,raw_states={k:int((a['raw_teacher_state']==k).sum())for k in'DESU'},optimizer_updates=8000,C15_trace_rows_bitexact=1000,C15_history_rows_bitexact=11,all8000_saved_batches192known_available_TRAIN_D96_E96=True,no_S_U_VAL_loss_rows=True,histories81_fourteen_cohort_means_and_counts=True,selected_update=step,selected_criterion=criterion,selected14_BCE_from51004_logits_exact=True,old15_gate_values_limits_and_PASS_exact=True,source_gates_pass=sum(v['PASS']for v in g['gates'].values()),known_unavailable=s['known_unavailable'],weight_checkpoint_bindings_SHA_verified_not_deserialized=True,producer_certificate_only=['step1000 parameter/saved-score equality','optimizer/RNG identity and saved resume-state contents'],limitation='No intermediate checkpoint/head replay. All81 criteria replayed from saved14 means; selected14 BCE independently from saved scores. Producer-only parameter/RNG claims distinguished above.',actual_model_optimizer_encoder_current_V112_PnP_calls=0)
 text=json.dumps(result,indent=2)+'\n';(O/'REVIEW.json').write_text(text);(O/'SAVED_SOURCE_REVIEW.json').write_text(text);print(json.dumps(dict(completed=True,PASS=True,review=bind(O/'REVIEW.json'),selected_update=step,gates=result['source_gates_pass'])))
if __name__=='__main__':run()
