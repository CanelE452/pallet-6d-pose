"""C6/C7 external-only source fits. --prepare is metadata-only; root owns runs."""
import os,sys
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,time,traceback
import numpy as np
import torch
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
ADAPTER=H/'external_data_adapter.py';MODEL=V/'model/bounded_prior_residual.py';DESIGN=V/'ROOT_C6_C7_DESIGN_BEFORE_SOURCE_FORWARD.json'
DESIGN_SHA='7782c8e994a1c370d293ad483e446dbed4f315d3e883aba386052ad374bd51cc'
MODEL_SHA='d6be1dd0fd3bd545e3de4be0a746093350573146397616b9f2a7b201ee60e011'
PROTOCOL=H/'FIT_PROTOCOL_BEFORE_SOURCE_FORWARD.json'
ARMS=('C6_EXTERNAL_ONLY_RAW_PRIOR','C7_EXTERNAL_ONLY_CLIPPED_PRIOR2')
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ADAPTER,'v102_external_adapter');sha=d.sha;bind=d.bind;check=d.check;read=d.read;rows=d.rows;same=d.same

def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def default_bindings():
 b=read(d.C5/'source_fit_preparation/SOURCE_BINDINGS.json')
 complete=read(d.C5/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1/FIT_COMPLETE.json')
 b['C5_scores_reference_base_only']=complete['scores']
 b['C5_adapter']=bind(d.C5_ADAPTER)
 return b

def training_loss(out,target,pair_positions):
 z=out['query_logit'];a=out['query_available'].bool();known=target>=0;valid=known&a;zero=z.sum()*0
 bce=torch.nn.functional.binary_cross_entropy_with_logits(z[valid],target[valid].to(z.dtype))if valid.any()else zero
 # Preservation always references original C1, not arm-specific clipped anchor.
 correct=valid&((out['reference_base_logit']>=0)==(target==1))
 penalty=out['delta_logit'][correct].square().mean()if correct.any()else zero
 c=pair_positions[:,1];s=pair_positions[:,2];pm=a[c]&a[s]
 assert torch.all(target[c]==1)and torch.all(target[s]==0)
 hinge=torch.relu(2.-(z[c[pm]]-z[s[pm]])).mean()if pm.any()else zero
 return dict(loss=bce+.05*penalty+.25*hinge,BCE=bce,delta_L2=penalty,matched_hinge=hinge,available_external_known_N=int(valid.sum()),known_unavailable_N=int((known&~a).sum()),no_external_target_N=int((~known).sum()),correct_original_base_N=int(correct.sum()),available_margin_pairs=int(pm.sum()))

def metrics(z,a,reference,delta,meta,design):
 h=d.old_helper();oldresults,oldchecks,_=h.report(z[:d.NOLD],a[:d.NOLD],meta['old_meta']);out={'old_exact_V96':oldresults,'new_external':{},'preservation':{}}
 for split in('TRAIN','VALIDATION'):
  m=(meta['domain']=='V99_SYNTH')&(meta['split']==split)
  out['new_external'][split]=dict(all_rows=d.external_cm(z,a,meta,m),by_raw_teacher_state={t:d.external_cm(z,a,meta,m&(meta['typed_state']==t))for t in('D','E','S','U')},by_variant={v:d.external_cm(z,a,meta,m&(meta['variant']==v))for v in d.STAGES},point_metrics_by_axes={})
  pm=meta['split'][meta['new_pairs'][:,0]]==split;out['new_external'][split]['selected_pairs']=d.new_pair_metrics(z,a,meta,pm);out['new_external'][split]['pairs_by_axes']={}
  for axis in('appearance','aspect','azimuth','elevation'):
   out['new_external'][split]['point_metrics_by_axes'][axis]={str(v):d.external_cm(z,a,meta,m&(meta[axis]==v))for v in np.unique(meta[axis][m])}
   vals=meta[axis][meta['new_pairs'][:,0]];out['new_external'][split]['pairs_by_axes'][axis]={str(v):d.new_pair_metrics(z,a,meta,pm&(vals==v))for v in np.unique(vals[pm])}
 for domain in('V96_REAL','V99_SYNTH'):
  out['preservation'][domain]={}
  for split in('TRAIN','VALIDATION'):
   m=(meta['domain']==domain)&(meta['split']==split)&meta['known'];bp=d.decode(reference,a);p=d.decode(z,a);bc=m&(bp==meta['labels']);fc=m&(p==meta['labels'])
   out['preservation'][domain][split]=dict(known_external_N=int(m.sum()),original_base_correct=int(bc.sum()),base_correct_still_correct=int((bc&fc).sum()),base_correct_became_wrong_or_U=int((bc&~fc).sum()),base_wrong_becomes_correct=int((m&~bc&fc).sum()),mean_abs_delta=float(abs(delta[m]).mean())if m.any()else None,max_abs_delta=float(abs(delta[m]).max())if m.any()else None)
 gates={};mapping={'old_LOCO_before_D_recall_min':'LOCO_before_D_recall','old_PALLOX_before_D_recall_min':'PALLOX_before_D_recall','old_natural_before_E_recall_min':'natural_before_E_recall','old_sham_D_recall_min':'sham_D_recall','old_cover_E_recall_min':'cover_E_recall','old_accepted_VAL_pair_all3_min':'accepted_VAL_pair_all3_correct_rate','old_known_VAL_U_rate_max':'known_VAL_U_rate'}
 for key,ok in mapping.items():gates[key]=dict(value=oldchecks[ok]['value'],limit=design['source_gates'][key])
 vm=d.val_masks(meta);pred=d.decode(z,a)
 for key,cohort,target in [('new_BEFORE_D_recall_min','new_BEFORE_D',0),('new_SHAM_D_recall_min','new_SHAM_D',0),('new_COVER_E_recall_min','new_COVER_E',1)]:
  m=vm[cohort];gates[key]=dict(value=float((pred[m]==target).mean()),limit=design['source_gates'][key],N=int(m.sum()))
 gates['new_selected_VAL_pair_all3_min']=dict(value=out['new_external']['VALIDATION']['selected_pairs']['all3_correct_rate'],limit=design['source_gates']['new_selected_VAL_pair_all3_min'])
 m=(meta['domain']=='V99_SYNTH')&(meta['split']=='VALIDATION')&meta['known'];assert m.sum()==243
 gates['new_known_VAL_U_rate_max']=dict(value=float((~a[m]).mean()),limit=design['source_gates']['new_known_VAL_U_rate_max'],N=int(m.sum()))
 for k,g in gates.items():g['PASS']=g['value']is not None and(g['value']<=g['limit']if k.endswith('_max')else g['value']>=g['limit'])
 assert set(gates)==set(design['source_gates'])-{'empty_required_cohort'}and len(gates)==12
 return out,gates,all(x['PASS']for x in gates.values())

def forward(model,data,u,device):return model(*(torch.as_tensor(np.ascontiguousarray(data[k][u]),device=device)for k in d.FIELDS))
@torch.no_grad()
def predict(model,data,units,device):
 model.eval();banks=[]
 for lo,hi in((0,1348),(1348,len(data['p3_roi']))):
  v={k:[]for k in('query_logit','query_available','base_logit','reference_base_logit','delta_logit')}
  for s in range(lo,hi,128):
   out=forward(model,data,np.arange(s,min(s+128,hi)),device)
   for k in v:v[k].append(out[k].detach().cpu().numpy())
  banks.append({k:np.concatenate(a)for k,a in v.items()})
 return {k:np.concatenate([b[k]for b in banks])[units]for k in banks[0]}
def residual_state_sha(model):
 h=hashlib.sha256()
 for name,v in model.state_dict().items():
  if name.startswith('base.'):continue
  a=v.detach().cpu().contiguous().numpy();h.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0'+str(a.shape).encode()+b'\0'+a.tobytes())
 return h.hexdigest()
def draw_hash_update(h,ix,triples):h.update(ix.astype('<i8').tobytes());h.update(triples.astype('<i8').tobytes())

def prepare():
 assert sha(DESIGN)==DESIGN_SHA and sha(MODEL)==MODEL_SHA and torch.__version__=='2.1.1+cu118'
 b=default_bindings()
 for x in b.values():check(x)
 meta=d.load_metadata(b);pools=d.sampling_pools(meta);raw=meta['typed_state'];externalcounts=meta['external_counts'];assert {s:int(((meta['split']==s)&meta['known']).sum())for s in('TRAIN','VALIDATION')}=={'TRAIN':1002,'VALIDATION':516}
 val={k:int(v.sum())for k,v in d.val_masks(meta).items()};design=read(DESIGN);assert val=={k:v['N']for k,v in design['settings']['VAL_cohorts'].items()}
 generator=torch.Generator(device='cpu').manual_seed(205);digest=hashlib.sha256()
 for update in range(2000):
  ix,t,_=d.draw(pools,generator);assert meta['known'][ix].all()and np.all(meta['split'][ix]=='TRAIN')and np.sum(meta['labels'][ix]==0)==32 and np.sum(meta['labels'][ix]==1)==32;draw_hash_update(digest,ix,t)
 proof=bind(H/'GENERATED_TRAINER_TESTS.json');mp=bind(V/'model/GENERATED_BOUNDED_PRIOR_TESTS.json');assert read(proof['path'])['PASS']and read(mp['path'])['PASS']
 code=dict(trainer=bind(__file__),adapter=bind(ADAPTER),model=bind(MODEL),immutable_C5_adapter=bind(d.C5_ADAPTER),immutable_C3_helper=bind(d.c5.OLD_CODE))
 p=dict(completed=True,source_only=True,current_allowed=False,PnP_allowed=False,design=bind(DESIGN),code=code,generated=proof,model_generated=mp,bindings=b,raw_counts=meta['counts'],external_counts=externalcounts,external_D_E_total=1518,known_geometric_S_out_of_task_total=396,unresolved_teacher_U_total=32070,VAL_eight_cohort_counts=val,TRAIN_pool_sizes={k:len(v)for k,v in pools.items()},settings=design['settings'],source_gates=design['source_gates'],frozen_2000_draws_sha256=digest.hexdigest(),runtime='Torch2.1.1+cu118 CUDAFP32, CPUthreads1, deterministic cuBLAS4096:8, TF32off',all_occurrences=33984,unique_input_units=3652,arms=list(ARMS),between_arm_selection=design['between_arm_selection'],source_development_exposure=True,native8_mapping='UNKNOWN',human_gold=False)
 write(PROTOCOL,p);write(H/'SOURCE_BINDINGS.json',b)
 for arm in ARMS:write(H/(arm+'_ROOT_ENTRY_TEMPLATE.json'),dict(approved_actual_source_fit=False,source_only=True,current_allowed=False,PnP_allowed=False,allow_restart_after_optimizer_update=False,arm=arm,protocol=bind(PROTOCOL),design=bind(DESIGN),code=code,generated=proof,model_generated=mp,bindings=b,run_directory=str(V/'runs'/arm)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),code=code,external_counts=externalcounts,VAL=val,TRAIN_pool_sizes=p['TRAIN_pool_sizes'],draw_hash=p['frozen_2000_draws_sha256'])),flush=True)

def require_entry(e):
 assert e.get('approved_actual_source_fit')is True and e.get('source_only')is True and e.get('current_allowed')is False and e.get('PnP_allowed')is False and e.get('allow_restart_after_optimizer_update')is False
 assert e.get('arm')in ARMS;run=V/'runs'/e['arm'];assert Path(e['run_directory']).resolve()==run.resolve()and not run.exists()
 for k in('protocol','design','generated','model_generated'):check(e[k])
 assert same(e['design'],bind(DESIGN))and sha(DESIGN)==DESIGN_SHA and same(e['protocol'],bind(PROTOCOL))
 for b in e['code'].values():check(b)
 assert same(e['code']['trainer'],bind(__file__))and same(e['code']['adapter'],bind(ADAPTER))and same(e['code']['model'],bind(MODEL))and sha(MODEL)==MODEL_SHA
 p=read(PROTOCOL);assert p['settings']==read(DESIGN)['settings']and p['code']==e['code']and p['bindings']==e['bindings']
 for b in e['bindings'].values():check(b)
 for k in('generated','model_generated'):assert read(e[k]['path'])['PASS']
 assert same(read(e['generated']['path'])['trainer'],e['code']['trainer'])and same(read(e['generated']['path'])['adapter'],e['code']['adapter'])
 return p,run

def run(entry):
 e=read(entry);p,RUN=require_entry(e);assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available();arm=e['arm'];RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),arm=arm,optimizer_updates=0,restart_automatically=False));updates=0
 try:
  meta=d.load_metadata(e['bindings']);assert meta['counts']==p['raw_counts']and meta['external_counts']==p['external_counts'];pools=d.sampling_pools(meta);data,unit,factual=d.load_inputs(e['bindings'],meta)
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  random.seed(205);np.random.seed(205);torch.manual_seed(205);torch.cuda.manual_seed_all(205)
  op=module(MODEL,'v102_source_operator');model,receipt=op.load_frozen_c1(arm=arm);model=model.to('cuda:0');trainable=[v for v in model.parameters()if v.requires_grad]
  assert sum(v.numel()for v in trainable)==3221 and sum(v.numel()for v in model.parameters())==29794 and all(not v.requires_grad for v in model.base.parameters())and not model.base.training
  opt=torch.optim.AdamW(trainable,lr=.001,weight_decay=.0001);assert len(opt.state)==0;generator=torch.Generator(device='cpu').manual_seed(205);drawdigest=hashlib.sha256()
  initial=predict(model,data,unit,'cuda:0');assert np.array_equal(initial['query_available'],factual)
  reference=initial['reference_base_logit'];anchor=initial['base_logit']
  with np.load(check(e['bindings']['old_C1_scores']),allow_pickle=False)as f:assert np.array_equal(reference[:d.NOLD],f['query_logit'])and np.array_equal(initial['query_available'][:d.NOLD],f['query_available'])
  with np.load(check(e['bindings']['C5_scores_reference_base_only']),allow_pickle=False)as f:assert np.array_equal(reference[d.NOLD:],f['base_logit'][d.NOLD:])and np.array_equal(initial['query_available'][d.NOLD:],f['query_available'][d.NOLD:])
  expected=reference if arm==ARMS[0]else np.clip(reference,-2.,2.)
  assert np.array_equal(anchor.view(np.uint32),expected.view(np.uint32))and np.array_equal(initial['query_logit'],anchor)and np.count_nonzero(initial['delta_logit'])==0
  assert np.array_equal(d.decode(initial['query_logit'],factual),d.decode(reference,factual))
  for k,v in model.residual_encoder.state_dict().items():assert torch.equal(v,model.base.local_encoder.state_dict()[k])
  basestate=model.assert_base_unchanged();branchstate=residual_state_sha(model)
  peer=V/'runs'/ARMS[1-ARMS.index(arm)]/'UPDATE0_C1_PARITY.json';peerchecked=False
  if peer.exists():assert read(peer)['initial_residual_state_sha256']==branchstate;peerchecked=True
  write(RUN/'UPDATE0_C1_PARITY.json',dict(completed=True,PASS=True,arm=arm,all33984_originalC1_rawbase_and_availability_exact=True,old_reference=e['bindings']['old_C1_scores'],new_reference_base_only=e['bindings']['C5_scores_reference_base_only'],anchor_clamp_or_raw_exact=True,initial_raw0_decisions_match_originalC1=True,all_delta_zero=True,base_state_sha256=basestate,initial_residual_state_sha256=branchstate,peer_initial_receipt_compared=peerchecked,load_receipt=receipt,fresh_AdamW_state=True,optimizer_updates=0))
  def ckpt(update):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,arm=arm,seed=205,parameter_count=29794,trainable_parameter_count=3221,base_state_sha256=basestate,initial_residual_state_sha256=branchstate,model_code=e['code']['model'],protocol=e['protocol'],entry=bind(entry))
  torch.save(ckpt(0),RUN/'INITIAL_CHECKPOINT.pt');best=float('inf');best_update=None;best_state=None;started=time.monotonic()
  def evaluate(update,out=None):
   nonlocal best,best_update,best_state
   out=predict(model,data,unit,'cuda:0')if out is None else out;assert np.array_equal(out['query_available'],factual);model.assert_base_unchanged();value,parts=d.selector(out['query_logit'],out['query_available'],meta)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,criterion_eight_VAL_cohorts_BCE=value,VAL_cohorts=parts),allow_nan=False)+'\n')
   torch.save(ckpt(update),RUN/f'CHECKPOINT_UPDATE_{update:04d}.pt')
   if value<best:best=value;best_update=update;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};torch.save(ckpt(update),RUN/'BEST_EXTERNAL_SCALAR.pt')
  evaluate(0,initial)
  for update in range(1,2001):
   ix,triples,drawaudit=d.draw(pools,generator);assert meta['known'][ix].all()and np.all(meta['split'][ix]=='TRAIN')and np.sum(meta['labels'][ix]==0)==32 and np.sum(meta['labels'][ix]==1)==32;draw_hash_update(drawdigest,ix,triples)
   model.train();opt.zero_grad(set_to_none=True);out=forward(model,data,unit[ix],'cuda:0');loss=training_loss(out,torch.as_tensor(meta['external_target'][ix],device='cuda:0'),torch.as_tensor(triples,device='cuda:0'));assert torch.isfinite(loss['loss']);loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(trainable,5.);assert torch.isfinite(grad);opt.step();updates=update
   rec=dict(update=update,occurrence_indices=ix.tolist(),input_unit_indices=unit[ix].tolist(),matched_triplet_batch_positions=triples.tolist(),draw=drawaudit,gradient_norm_before_clip=float(grad))
   for k,v in loss.items():rec[k]=float(v.detach())if torch.is_tensor(v)else v
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
   if update%100==0:evaluate(update)
  assert drawdigest.hexdigest()==p['frozen_2000_draws_sha256'];torch.save(ckpt(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state);model.assert_base_unchanged();out=predict(model,data,unit,'cuda:0');z=out['query_logit'];a=out['query_available'];assert np.array_equal(a,factual);value,parts=d.selector(z,a,meta);assert value==best
  result,gates,passed=metrics(z,a,out['reference_base_logit'],out['delta_logit'],meta,read(DESIGN));line=meta['lineage'];qid=np.array([r.get('occurrence_id',r.get('image_id','')+'::AUX'+str(r.get('aux_corner_index',-1)))for r in line]);q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line],np.float64);L=np.array([r['L']if r['L']is not None else np.nan for r in line],np.float64)
  nll=np.full(d.N,np.nan,np.float64);safe=np.where(a,z,0).astype(np.float64);nll[meta['known']]=np.logaddexp(0,safe[meta['known']])-meta['labels'][meta['known']]*safe[meta['known']]
  np.savez_compressed(RUN/'SELECTED_ALL33984_SOURCE_SCORES.npz',**out,prediction=d.decode(z,a),external_loss_candidate=meta['known'],external_target_D0_E1_no_target_minus1=meta['external_target'],original_C5_target_D0_EorS1_Uminus1=meta['labels'],raw_teacher_state=meta['typed_state'],known_geometric_S_out_of_task=meta['known_geometric_S'],unresolved_teacher_U=meta['unresolved_teacher_U'],domain=meta['domain'],source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],global_occurrence_index=np.arange(d.N),domain_occurrence_index=np.r_[np.arange(d.NOLD),np.arange(d.NNEW)],occurrence_id=qid,input_unit_index=unit,query_xy_raw=q,L_raw=L,NLL_external_known_FP64=nll,old_accepted_triplets=meta['old_pairs'],new_selected_triplets=meta['new_pairs'],appearance=meta['appearance'],aspect=meta['aspect'],azimuth=meta['azimuth'],elevation=meta['elevation'],old_original_C1_V95_label=meta['old_original_C1_V95_label'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,PASS=passed,gates=gates,current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,arm=arm,source_only=True,selected_update=best_update,criterion_eight_VAL_cohorts_BCE=value,VAL_cohorts=parts,raw_counts=meta['counts'],external_counts=meta['external_counts'],metrics=result,all33984_preserved=True,known_geometricS_is_not_unknownU=True,learned_self_classification=False,source_development_exposure=True,all2000_draws_sha256=drawdigest.hexdigest()))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,arm=arm,source_only=True,selected_update=best_update,updates=2000,checkpoint=bind(RUN/'BEST_EXTERNAL_SCALAR.pt'),initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),model_code=e['code']['model'],base_checkpoint=receipt['checkpoint'],base_state_sha256=basestate,initial_residual_state_sha256=branchstate,base_frozen=True,parameter_count=29794,trainable_parameter_count=3221,protocol=e['protocol'],entry=bind(entry),threshold_raw_logit=0.,source_input_bindings=e['bindings'],current_used_for_fit_selection=False,current_inference_authorized=False))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,arm=arm,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-started,MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL33984_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),update0_parity=bind(RUN/'UPDATE0_C1_PARITY.json'),new_current_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(completed=False,optimizer_updates_completed=updates,exception=traceback.format_exc(),restart_automatically=False));raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare metadataonly or --entry exactrootapprovedarmfit')
