"""One root-authorized C5 source fit; import and --prepare are model-forward inert."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;V=HERE.parent;P=V.parent
ADAPTER=HERE/'data_adapter.py';MODEL=V/'model/local_rgb_residual.py';DESIGN=V/'ROOT_C5_DESIGN_BEFORE_SOURCE_HEAD_FORWARD.json'
DESIGN_SHA='d7625dcce1b2862695d9665e6fb5fbe329b47d137834fea816b4aa4111f4f81d'
MODEL_SHA='836f7309d8aa2e4db28fe3b7197c2befc3c8643dd986bc1571a07d17906701f4'
PROTOCOL=HERE/'FIT_PROTOCOL_BEFORE_SOURCE_HEAD_FORWARD.json';RUN=V/'runs/C5_FROZEN_C1_CENTRAL_RGB_SOURCE_V1'
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ADAPTER,'c5_data_adapter');sha=d.sha;bind=d.bind;check=d.check;read=d.read;rows=d.rows;same=d.same

def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def default_bindings():
 old=d.OLD;new=d.NEW;om=read(old/'actual_source_pair_input_pack/PACK_MANIFEST.json');nm=read(new/'matching_feature_pack/PACK_MANIFEST.json')
 g=read(new/'geometry_query_preparation/GEOMETRY96_SEAL.json')
 return dict(old_pack_seal=bind(old/'actual_source_pair_input_pack/INPUT_PACK_SEAL.json'),old_manifest=bind(old/'actual_source_pair_input_pack/PACK_MANIFEST.json'),old_base_lineage=om['base_input_lineage'],old_GT=bind(old/'actual_query_anchored_image_intake/OCCURRENCES31680_GT_ONLY.jsonl'),old_C1_scores=bind(old/'source_baselines/C1_FROZEN_ON_V96/SELECTED_ALL31680_SOURCE_SCORES.npz'),old_helper=bind(d.OLD_CODE),new_pack_seal=bind(new/'matching_feature_pack/INPUT_PACK_SEAL.json'),new_manifest=bind(new/'matching_feature_pack/PACK_MANIFEST.json'),new_GT=g['GT_only_targets'],new_queries=bind(new/'geometry_query_preparation/QUERY_INPUTS96_INPUT_ONLY.jsonl'),new_config=bind(new/'BASE96_SOURCE_CONFIG_INPUT_ONLY.jsonl'),new_fullRGB_corpus=bind(new/'render_preparation/FINAL_PREPARATION_AND_CORPUS_SEAL.json'))

def training_loss(out,target,pair_positions):
 """Unknown/invalid rows have zero loss. One scalar; components loss-only."""
 z=out['query_logit'];a=out['query_available'].bool();y=target.to(z.dtype);known=target>=0;valid=known&a
 zero=z.sum()*0
 bce=torch.nn.functional.binary_cross_entropy_with_logits(z[valid],y[valid])if valid.any()else zero
 base_correct=valid&((out['base_logit']>=0)==(target==1))
 delta=(out['delta_logit'][base_correct]**2).mean()if base_correct.any()else zero
 cover=pair_positions[:,1];sham=pair_positions[:,2];pm=a[cover]&a[sham]
 assert torch.all(target[cover]==1)and torch.all(target[sham]==0)
 hinge=torch.relu(2.-(z[cover[pm]]-z[sham[pm]])).mean()if pm.any()else zero
 return dict(loss=bce+.05*delta+.25*hinge,BCE=bce,delta_L2=delta,matched_hinge=hinge,available_known_count=int(valid.sum()),known_unavailable_count=int((known&~a).sum()),unknown_count=int((~known).sum()),correct_base_preservation_N=int(base_correct.sum()),available_margin_pairs=int(pm.sum()))

def decode(z,a):return np.where(a,(z>=0).astype(np.int8),-1)
def new_pair_metrics(z,a,meta,mask=None):
 pairs=meta['new_pairs'];pred=decode(z,a);selected=np.ones(len(pairs),bool)if mask is None else mask
 hit=(pred[pairs]==np.array([0,1,0])).all(1);n=int(selected.sum())
 return dict(selected_triplets=n,all3_correct=int((selected&hit).sum()),all3_correct_rate=float(hit[selected].mean())if n else None,before_D_correct=int((selected&(pred[pairs[:,0]]==0)).sum()),cover_E_correct=int((selected&(pred[pairs[:,1]]==1)).sum()),sham_D_correct=int((selected&(pred[pairs[:,2]]==0)).sum()),any_U=int((selected&(pred[pairs]<0).any(1)).sum()))
def metrics(z,a,base,delta,meta,design):
 h=d.old_helper();oldresults,oldchecks,_=h.report(z[:d.NOLD],a[:d.NOLD],meta['old_meta']);out={'old_exact_V96':oldresults,'new':{},'preservation':{}}
 def new_cm(mask):
  c=h.cm_metrics(z,a,meta,mask);c['rows']=['conditional_AUX_D','conditional_AUX_hidden_E_or_S'];c['columns']=['D','hidden','U']
  c['binary_hidden_recall']=c.pop('E_recall');c['binary_hidden_precision']=c.pop('E_precision');c['cause_classification']=False
  return c
 for split in('TRAIN','VALIDATION'):
  m=(meta['domain']=='V99_SYNTH')&(meta['split']==split)
  out['new'][split]={'all_known':new_cm(m),'by_state':{t:new_cm(m&(meta['typed_state']==t))for t in('D','E','S','U')},'by_variant':{v:new_cm(m&(meta['variant']==v))for v in d.STAGES},'point_metrics_by_axes':{}}
  for axis in('appearance','aspect','azimuth','elevation'):
   out['new'][split]['point_metrics_by_axes'][axis]={str(v):new_cm(m&(meta[axis]==v))for v in np.unique(meta[axis][m])}
  pm=meta['split'][meta['new_pairs'][:,0]]==split;out['new'][split]['selected_pairs']=new_pair_metrics(z,a,meta,pm)
  out['new'][split]['pairs_by_axes']={}
  for axis in('appearance','aspect','azimuth','elevation'):
   vals=meta[axis][meta['new_pairs'][:,0]];out['new'][split]['pairs_by_axes'][axis]={str(v):new_pair_metrics(z,a,meta,pm&(vals==v))for v in np.unique(vals[pm])}
 for domain in('V96_REAL','V99_SYNTH'):
  out['preservation'][domain]={}
  for split in('TRAIN','VALIDATION'):
   m=(meta['domain']==domain)&(meta['split']==split)&meta['known'];bp=decode(base,a);p=decode(z,a);bc=m&(bp==meta['labels']);fc=m&(p==meta['labels'])
   out['preservation'][domain][split]=dict(known_N=int(m.sum()),base_correct=int(bc.sum()),base_correct_still_correct=int((bc&fc).sum()),base_correct_became_wrong_or_U=int((bc&~fc).sum()),base_wrong_becomes_correct=int((m&~bc&fc).sum()),known_mean_abs_delta=float(abs(delta[m]).mean())if m.any()else None,max_abs_delta=float(abs(delta[m]).max())if m.any()else None)
 checks={}
 mapping={'old_LOCO_before_D_recall_min':'LOCO_before_D_recall','old_PALLOX_before_D_recall_min':'PALLOX_before_D_recall','old_natural_before_E_recall_min':'natural_before_E_recall','old_sham_D_recall_min':'sham_D_recall','old_cover_E_recall_min':'cover_E_recall','old_accepted_VAL_pair_all3_min':'accepted_VAL_pair_all3_correct_rate','old_known_VAL_U_rate_max':'known_VAL_U_rate'}
 for key,oldkey in mapping.items():checks[key]=dict(value=oldchecks[oldkey]['value'],limit=design['source_gates'][key])
 vm=d.val_masks(meta);pred=decode(z,a)
 for key,cohort,target in [('new_BEFORE_D_recall_min','new_BEFORE_D',0),('new_SHAM_D_recall_min','new_SHAM_D',0),('new_COVER_E_recall_min','new_COVER_E',1),('new_ALL_S_hidden_recall_min','new_ALL_S',1)]:
  m=vm[cohort];checks[key]=dict(value=float((pred[m]==target).mean()),limit=design['source_gates'][key],N=int(m.sum()))
 checks['new_selected_VAL_pair_all3_min']=dict(value=out['new']['VALIDATION']['selected_pairs']['all3_correct_rate'],limit=design['source_gates']['new_selected_VAL_pair_all3_min'])
 m=(meta['domain']=='V99_SYNTH')&(meta['split']=='VALIDATION')&meta['known'];checks['new_known_VAL_U_rate_max']=dict(value=float((~a[m]).mean()),limit=design['source_gates']['new_known_VAL_U_rate_max'],N=int(m.sum()))
 for key,c in checks.items():c['PASS']=c['value']is not None and(c['value']<=c['limit']if key.endswith('_max')else c['value']>=c['limit'])
 assert set(checks)==set(design['source_gates'])-{'empty_required_cohort'}
 return out,checks,all(c['PASS']for c in checks.values())

def forward(model,data,u,device):return model(*(torch.as_tensor(np.ascontiguousarray(data[k][u]),device=device)for k in d.FIELDS))
@torch.no_grad()
def predict(model,data,units,device):
 model.eval();banks=[]
 # Preserve exact old C1 batch partition (1348 units, finalbatch68), not a mixed old/new batch.
 for lo,hi in((0,1348),(1348,len(data['p3_roi']))):
  v={k:[]for k in('query_logit','query_available','base_logit','delta_logit')}
  for s in range(lo,hi,128):
   out=forward(model,data,np.arange(s,min(s+128,hi)),device)
   for k in v:v[k].append(out[k].detach().cpu().numpy())
  banks.append({k:np.concatenate(a)for k,a in v.items()})
 return {k:np.concatenate([b[k]for b in banks])[units]for k in banks[0]}

def prepare():
 assert sha(DESIGN)==DESIGN_SHA and sha(MODEL)==MODEL_SHA
 b=default_bindings()
 for x in b.values():check(x)
 modelproof=bind(V/'model/GENERATED_MODEL_TESTS.json');assert read(modelproof['path'])['PASS']
 m=d.load_metadata(b);pools=d.sampling_pools(m)
 expected=read(DESIGN)['sources']['new_counts'];actual=m['counts']['V99_SYNTH']
 assert all(actual[s]['typed_states']==expected[s]for s in expected)
 assert list(m['counts']['VAL_cohorts'].values())==[3,30,119,2,119,82,77,32,192]
 assert m['counts']['old_accepted_pairs']=={'TRAIN':75,'VALIDATION':119}and m['counts']['new_selected_pairs']=={'TRAIN':64,'VALIDATION':32}
 proof=bind(HERE/'GENERATED_TRAINER_TESTS.json');assert read(proof['path'])['PASS']
 code={'trainer':bind(__file__),'adapter':bind(ADAPTER),'model':bind(MODEL),'old_helper':bind(d.OLD_CODE)}
 protocol=dict(completed=True,source_only=True,current_allowed=False,PnP_allowed=False,design=bind(DESIGN),code=code,generated=proof,model_generated=modelproof,bindings=b,counts=m['counts'],TRAIN_pool_sizes={k:len(v)for k,v in pools.items()},settings=read(DESIGN)['settings'],run_directory=str(RUN),runtime='Torch2.1.1+cu118 CUDAFP32, CPUthreads1, deterministic cuBLAS:4096:8, TF32off',rows=d.N,unique_input_units=3652,update0_guard='Old31680 combined/base/availability exact immutable C1 samepack scores; new2304 combined==frozen base and zero delta, loaded base state hash exact C1. No second base forward inside C5.',training_admission='Root C5 design prospectively authorizes this conditional AUX supervision; producer training_admission:false is preserved as generator-history.',limitations=['V96 controls194 ranks cluster10TRAIN/18VALbasequeries and allPALLOX; oldnaturalEVAL only2.','V99 simplified composite and geometricS proxy; source-emulated q not actual initializer; native8UNKNOWN/humanfalse.','Repeated sourceVAL development, no freshblind generalization claim; binaryhidden output does not identifycause.'])
 write(PROTOCOL,protocol)
 write(HERE/'ROOT_SOURCE_FIT_ENTRY_TEMPLATE.json',dict(approved_actual_source_fit=False,source_only=True,current_allowed=False,PnP_allowed=False,allow_restart_after_optimizer_update=False,protocol=bind(PROTOCOL),design=bind(DESIGN),code=code,generated=proof,model_generated=modelproof,bindings=b,run_directory=str(RUN)))
 print(json.dumps(dict(protocol=bind(PROTOCOL),counts=m['counts'],TRAIN_pool_sizes=protocol['TRAIN_pool_sizes'])),flush=True)

def require_entry(e):
 assert e.get('approved_actual_source_fit')is True and e.get('source_only')is True and e.get('current_allowed')is False and e.get('PnP_allowed')is False
 assert e.get('allow_restart_after_optimizer_update')is False and not RUN.exists()and Path(e['run_directory']).resolve()==RUN.resolve()
 for k in('protocol','design','generated','model_generated'):check(e[k])
 assert same(e['design'],bind(DESIGN))and sha(DESIGN)==DESIGN_SHA and same(e['protocol'],bind(PROTOCOL))
 for b in e['code'].values():check(b)
 assert same(e['code']['trainer'],bind(__file__))and same(e['code']['adapter'],bind(ADAPTER))and same(e['code']['model'],bind(MODEL))and sha(MODEL)==MODEL_SHA
 p=read(PROTOCOL);assert p['settings']==read(DESIGN)['settings']and p['code']==e['code']and p['bindings']==e['bindings']
 for b in e['bindings'].values():check(b)
 for k in('generated','model_generated'):assert read(e[k]['path'])['PASS']
 return p

def run(entry):
 e=read(entry);p=require_entry(e);assert torch.__version__=='2.1.1+cu118'and torch.cuda.is_available()
 RUN.mkdir(parents=True);write(RUN/'RUN_STARTED.json',dict(entry=bind(entry),optimizer_updates=0,restart_automatically=False));updates=0
 try:
  meta=d.load_metadata(e['bindings']);assert meta['counts']==p['counts'];pools=d.sampling_pools(meta);data,unit,factual=d.load_inputs(e['bindings'],meta)
  torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  random.seed(205);np.random.seed(205);torch.manual_seed(205);torch.cuda.manual_seed_all(205)
  op=module(MODEL,'c5_source_operator');model,loadreceipt=op.load_frozen_c1();model=model.to('cuda:0');trainable=[v for v in model.parameters()if v.requires_grad]
  assert sum(v.numel()for v in trainable)==3221 and sum(v.numel()for v in model.parameters())==29794
  assert all(not v.requires_grad for v in model.base.parameters())and not model.base.training
  opt=torch.optim.AdamW(trainable,lr=.001,weight_decay=.0001);assert len(opt.state)==0;g=torch.Generator(device='cpu').manual_seed(205)
  initial=predict(model,data,unit,'cuda:0');assert np.array_equal(initial['query_available'],factual)
  with np.load(check(e['bindings']['old_C1_scores']),allow_pickle=False)as f:
   assert np.array_equal(initial['query_logit'][:d.NOLD],f['query_logit'])and np.array_equal(initial['base_logit'][:d.NOLD],f['query_logit'])and np.array_equal(initial['query_available'][:d.NOLD],f['query_available'])
  assert np.array_equal(initial['query_logit'],initial['base_logit'])and np.count_nonzero(initial['delta_logit'])==0
  for k,v in model.residual_encoder.state_dict().items():assert torch.equal(v,model.base.local_encoder.state_dict()[k])
  basestate=model.assert_base_unchanged();write(RUN/'UPDATE0_C1_PARITY.json',dict(completed=True,PASS=True,old31680_reference=e['bindings']['old_C1_scores'],old_combined_base_logit_and_availability_bitexact=True,new2304_combined_equals_exact_frozenbase=True,all_delta_zero=True,base_state_sha256=basestate,load_receipt=loadreceipt,fresh_optimizer_state=True,optimizer_updates=0))
  def ckpt(update):return dict(state_dict={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer_state=opt.state_dict(),update=update,seed=205,parameter_count=29794,trainable_parameter_count=3221,base_state_sha256=basestate,base_checkpoint=read(DESIGN)['base_checkpoint'],model_code=e['code']['model'],protocol=e['protocol'],entry=bind(entry))
  torch.save(ckpt(0),RUN/'INITIAL_CHECKPOINT.pt');best=float('inf');best_update=None;best_state=None;started=time.monotonic()
  def evaluate(update,out=None):
   nonlocal best,best_update,best_state
   out=predict(model,data,unit,'cuda:0')if out is None else out;assert np.array_equal(out['query_available'],factual);model.assert_base_unchanged()
   value,parts=d.selector(out['query_logit'],out['query_available'],meta)
   with(RUN/'HISTORY.jsonl').open('a')as f:f.write(json.dumps(dict(update=update,criterion_nine_VAL_cohorts_BCE=value,VAL_cohorts=parts),allow_nan=False)+'\n')
   cp=RUN/f'CHECKPOINT_UPDATE_{update:04d}.pt';torch.save(ckpt(update),cp)
   if value<best:best=value;best_update=update;best_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};torch.save(ckpt(update),RUN/'BEST_C5_SCALAR.pt')
  evaluate(0,initial)
  for update in range(1,2001):
   ix,triples,drawaudit=d.draw(pools,g);assert meta['known'][ix].all()and np.all(meta['split'][ix]=='TRAIN')
   model.train();opt.zero_grad(set_to_none=True);out=forward(model,data,unit[ix],'cuda:0');loss=training_loss(out,torch.as_tensor(meta['labels'][ix],device='cuda:0'),torch.as_tensor(triples,device='cuda:0'))
   assert torch.isfinite(loss['loss']);loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(trainable,5.);assert torch.isfinite(grad);opt.step();updates=update
   rec=dict(update=update,occurrence_indices=ix.tolist(),input_unit_indices=unit[ix].tolist(),matched_triplet_batch_positions=triples.tolist(),draw=drawaudit,gradient_norm_before_clip=float(grad))
   for k,v in loss.items():rec[k]=float(v.detach())if torch.is_tensor(v)else v
   with(RUN/'TRAIN_UPDATES.jsonl').open('a')as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
   if update%100==0:evaluate(update)
  torch.save(ckpt(2000),RUN/'FINAL_CHECKPOINT.pt');model.load_state_dict(best_state);model.assert_base_unchanged();out=predict(model,data,unit,'cuda:0');z=out['query_logit'];a=out['query_available'];assert np.array_equal(a,factual)
  value,parts=d.selector(z,a,meta);assert value==best;result,gates,passed=metrics(z,a,out['base_logit'],out['delta_logit'],meta,read(DESIGN))
  line=meta['lineage'];qid=np.array([r.get('occurrence_id',r.get('image_id','')+'::AUX'+str(r.get('aux_corner_index',-1)))for r in line]);q=np.array([r['query_xy']if r['query_xy']is not None else[np.nan,np.nan]for r in line],np.float64);L=np.array([r['L']if r['L']is not None else np.nan for r in line],np.float64)
  nll=np.full(d.N,np.nan,np.float64);safe=np.where(a,z,0).astype(np.float64);nll[meta['known']]=np.logaddexp(0,safe[meta['known']])-meta['labels'][meta['known']]*safe[meta['known']]
  np.savez_compressed(RUN/'SELECTED_ALL33984_SOURCE_SCORES.npz',**out,prediction=decode(z,a),candidate_known=meta['known'],scalar_target_D0_EorS1_Uminus1=meta['labels'],typed_state=meta['typed_state'],domain=meta['domain'],source=meta['source'],split=meta['split'],original_split=meta['original_split'],variant=meta['variant'],global_occurrence_index=np.arange(d.N),domain_occurrence_index=np.r_[np.arange(d.NOLD),np.arange(d.NNEW)],occurrence_id=qid,input_unit_index=unit,query_xy_raw=q,L_raw=L,NLL_known_FP64=nll,old_accepted_triplets=meta['old_pairs'],new_selected_triplets=meta['new_pairs'],appearance=meta['appearance'],aspect=meta['aspect'],azimuth=meta['azimuth'],elevation=meta['elevation'],old_original_C1_V95_label=meta['old_original_C1_V95_label'])
  write(RUN/'SOURCE_GATE_RESULTS.json',dict(completed=True,all_gates_pass=passed,PASS=passed,gates=gates,current_inference_authorized=False))
  write(RUN/'SOURCE_RESULTS.json',dict(completed=True,source_only=True,selected_update=best_update,criterion_nine_VAL_cohorts_BCE=value,VAL_cohorts=parts,counts=meta['counts'],metrics=result,occurrences=d.N,unique_input_units=len(data['p3_roi']),native8_mapping='UNKNOWN',human_gold=False,U_excluded_from_loss=True,E_S_shared_hidden_binary_not_cause=True,source_development_exposure=True))
  write(RUN/'MODEL_FREEZE.json',dict(completed=True,source_only=True,selected_update=best_update,updates=2000,checkpoint=bind(RUN/'BEST_C5_SCALAR.pt'),initial_checkpoint=bind(RUN/'INITIAL_CHECKPOINT.pt'),final_checkpoint=bind(RUN/'FINAL_CHECKPOINT.pt'),model_code=e['code']['model'],base_checkpoint=read(DESIGN)['base_checkpoint'],base_freeze=read(DESIGN)['base_freeze'],base_state_sha256=basestate,base_frozen=True,parameter_count=29794,trainable_parameter_count=3221,protocol=e['protocol'],entry=bind(entry),threshold_raw_logit=0.,source_input_bindings=e['bindings'],current_used_for_fit_selection=False,current_inference_authorized=False))
  write(RUN/'FIT_COMPLETE.json',dict(completed=True,PASS_execution=True,optimizer_updates=2000,selected_update=best_update,seconds=time.monotonic()-started,MODEL_FREEZE=bind(RUN/'MODEL_FREEZE.json'),source_results=bind(RUN/'SOURCE_RESULTS.json'),source_gates=bind(RUN/'SOURCE_GATE_RESULTS.json'),scores=bind(RUN/'SELECTED_ALL33984_SOURCE_SCORES.npz'),history=bind(RUN/'HISTORY.jsonl'),train_updates=bind(RUN/'TRAIN_UPDATES.jsonl'),update0_parity=bind(RUN/'UPDATE0_C1_PARITY.json'),new_current_PnP=0))
  print(json.dumps(read(RUN/'FIT_COMPLETE.json')),flush=True)
 except BaseException:
  write(RUN/'FAILED_EXECUTION.json',dict(completed=False,optimizer_updates_completed=updates,exception=traceback.format_exc(),restart_automatically=False));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
 if a.prepare:prepare()
 elif a.entry:run(a.entry)
 else:raise SystemExit('Use --prepare metadataonly or --entry exactrootapprovedfit')
