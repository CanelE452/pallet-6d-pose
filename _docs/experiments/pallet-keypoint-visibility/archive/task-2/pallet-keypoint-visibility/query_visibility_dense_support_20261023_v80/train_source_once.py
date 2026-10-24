"""V80 single frozen source-only auxiliary control; never reads current data.

generated/freeze have no source forward. run requires the completed entry lock
and is executed only after parent review. No restart after update1.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,random,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F

R=Path(__file__).resolve().parent;P=R.parent
V77=P/'query_visibility_controlled_pairs_20261023_v77'
DECISIONS=R/'EXPERIMENT_DECISIONS_BEFORE_GT_BUILD.json'
DECISION_SHA='81ac5890ce9a6d210bce5403425d589351c6306c6379755f153182ed4fafd35a'
RUN=R/'runs/SOURCE_DENSE_AUX_V1'
FIELDS=('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(x):assert sha(x['path'])==x['sha256'],x['path']
def load(p):return json.loads(Path(p).read_text())
def write(p,d):
 assert not p.exists(),str(p);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def imp(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def protocol():
 assert sha(DECISIONS)==DECISION_SHA
 p=load(DECISIONS);assert p['frozen']and p['model']['total_parameters']==41287 and p['loss']['lambda']==.25 and p['training']['updates']==2000 and p['training']['batch_size']==96
 return p
def modules():
 p=protocol();check(p['bindings']['V77_trainer'])
 return p,imp(p['bindings']['V77_trainer']['path'],'v80_unchanged_V77_helpers'),imp(R/'model/dense_support_head.py','v80_dense_model'),imp(R/'model/dense_auxiliary_loss.py','v80_dense_loss')
def pure_sample():
 n=next(n for n in ast.parse((V77/'train_paired_once.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='sample')
 env={'np':np};exec(compile(ast.Module(body=[n],type_ignores=[]),'unchangedV77sampleAST','exec'),env);return env['sample']
def generated():
 p=protocol();proof=load(R/'model/GENERATED_DENSE_SUPPORT_TESTS.json');assert proof['PASS']and proof['model_sha256']==sha(R/'model/dense_support_head.py')and proof['loss_sha256']==sha(R/'model/dense_auxiliary_loss.py')
 source=np.repeat([0,1],48);y=np.tile(np.repeat([0,1],24),2);meta={}
 meta['natural_pools']={(s,c):np.flatnonzero((source==s)&(y==c))[:8]for s in range(2)for c in range(2)}
 meta['nonanchor_pools']={(s,c):np.flatnonzero((source==s)&(y==c))[8:16]for s in range(2)for c in range(2)}
 meta['pair_train_pools']={0:np.arange(4),1:np.arange(4,8)};meta['pair_before_query']=np.r_[np.arange(40,44),np.arange(88,92)];meta['pair_after_query']=np.r_[np.arange(16,20),np.arange(64,68)]
 f=pure_sample();rng=np.random.default_rng(20261023);digest=hashlib.sha256()
 for _ in range(2000):
  ix=f(meta,rng);digest.update(ix.tobytes());assert len(ix)==96
  assert all(np.sum((source[ix]==s)&(y[ix]==c))==24 for s in range(2)for c in range(2))
  assert all(y[ix[k]]==1 and y[ix[k+1]]==0 and source[ix[k]]==source[ix[k+1]]for k in range(64,96,2))
 write(R/'GENERATED_TRAINER_TESTS.json',dict(PASS=True,code=b(Path(__file__)),decisions=b(DECISIONS),model_proof=b(R/'model/GENERATED_DENSE_SUPPORT_TESTS.json'),sampler_code=p['bindings']['V77_trainer'],generated_2000_batches_sourceclass24=True,generated_sample_sequence_SHA256=digest.hexdigest(),same_primary_admission_sampler_selector=True,aux_weight=.25,source_RGB_GT_cache_checkpoint_values_read=0,actual_source_forward_optimizer_current_PnP=0));print('GENERATED_PASS',flush=True)
def freeze(loss_review,loss_sha):
 p=protocol();proof=load(R/'GENERATED_TRAINER_TESTS.json');assert proof['PASS']and proof['code']==b(Path(__file__))
 for x in p['bindings'].values():check(x)
 gs=R/'dense_gt/SEAL.json';assert sha(gs)=='e62da313439d0bb441978711bddc3f4e5251b0a903c30900411307ad655353bb';gt=load(gs);assert gt['completed'];[check(gt[k])for k in('protocol','targets','results')]
 gi=R/'independent_gt_audit/INDEPENDENT_DENSE_GT9_AUDIT.json';assert sha(gi)=='c015af976d167b5d30cc825233bf8ce451b56f426ecd6b3a4a6e12072d1bb876';assert load(gi)['PASS']
 assert loss_review and loss_sha and sha(loss_review)==loss_sha;lr=load(loss_review);assert lr['PASS']
 artifacts=[b(R/'model/dense_support_head.py'),b(R/'model/dense_auxiliary_loss.py'),b(R/'model/GENERATED_DENSE_SUPPORT_TESTS.json'),b(gs),gt['targets'],b(gi),b(loss_review),p['bindings']['V77_inputs'],p['bindings']['V77_trainer'],p['bindings']['unchanged_model_code'],p['bindings']['V75_initialization']]
 write(R/'TRAINER_ENTRYLOCK.json',dict(schema='V80_SINGLE_DENSE_AUXILIARY_SOURCE_FIT',decisions=b(DECISIONS),trainer=b(Path(__file__)),generated=b(R/'GENERATED_TRAINER_TESTS.json'),artifacts=artifacts,dense_GT=gt['targets'],dense_GT_seal=b(gs),independent_GT=b(gi),loss_review=b(loss_review),run_directory=str(RUN),runtime=dict(python=str(Path(sys.executable).resolve()),torch=str(torch.__version__),device='cuda:0',threads=1,TF32=False),restart_after_first_update=False,current_read_or_PnP=False));print(json.dumps(b(R/'TRAINER_ENTRYLOCK.json')),flush=True)
def load_dense(entry,meta,device):
 check(entry['dense_GT'])
 with np.load(entry['dense_GT']['path'],allow_pickle=False)as z:
  assert z['valid9'].shape==(10002,8,9,9)
  for k,want in [('id',meta['all_frame_id']),('source',meta['all_frame_source']),('split',meta['all_frame_split']),('domain',meta['domain'].reshape(-1,8)[:,0])]:assert np.array_equal(z[k],want),k
  dense={k:torch.from_numpy(np.ascontiguousarray(z[k].reshape(-1,9,9))).to(device)for k in('valid9','expected_support_M0','visible_target_M4')}
 assert all(x.dtype==torch.bool for x in dense.values())
 assert not(dense['valid9']&dense['visible_target_M4']&~dense['expected_support_M0']).any()
 return dense
def predict_all(base,model,gpu,ids,labels):
 # Same FP32 operations/batch128 as frozen V76/V77 primary predict helper.
 model.eval();zs=[];ns=[];av=[];aux=[]
 with torch.inference_mode():
  for start in range(0,len(ids),128):
   ix=ids[start:start+128];out=base.forward(model,gpu,ix)
   zs.append(out['support_logit'].cpu().numpy());av.append(out['input_available'].cpu().numpy())
   ns.append(F.binary_cross_entropy_with_logits(out['support_logit'],labels[torch.as_tensor(ix,device=labels.device)].clamp(0,1).float(),reduction='none').cpu().numpy())
   aux.append(out['aux_mask_logits'].to(torch.float16).cpu().numpy())
 return dict(logits=np.concatenate(zs),NLL_F32=np.concatenate(ns),input_available=np.concatenate(av)),np.concatenate(aux)
def save_aux(path,a,meta):
 assert a.shape==(80016,2,9,9)and a.dtype==np.float16 and np.isfinite(a).all()
 np.savez_compressed(path,aux_mask_logits=a.reshape(10002,8,2,9,9),id=meta['all_frame_id'],source=meta['all_frame_source'],split=meta['all_frame_split'],domain=meta['domain'].reshape(-1,8)[:,0],channels=np.asarray(['expected_support_M0_logit','visible_given_M0_logit']))
def run():
 p,paired,M,L=modules();e=load(R/'TRAINER_ENTRYLOCK.json');assert e['trainer']==b(Path(__file__))and e['decisions']==b(DECISIONS);check(e['generated']);[check(x)for x in e['artifacts']]
 assert not RUN.exists(),'Preserve existing run; no automatic restart'
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/bin/python').resolve()and str(torch.__version__)=='2.5.1+cu121'and torch.cuda.is_available()
 RUN.mkdir(parents=True);(RUN/'code_snapshot').mkdir()
 for x in[e['trainer'],e['decisions'],e['generated'],*e['artifacts']]:
  # Preserve all code/binding artifacts under unique deterministic basenames.
  dest=RUN/'code_snapshot'/(x['sha256'][:12]+'_'+Path(x['path']).name)
  if not dest.exists():shutil.copy2(x['path'],dest)
 write(RUN/'RUN_STARTED.json',dict(entry=b(R/'TRAINER_ENTRYLOCK.json'),optimizer_updates=0,current_PnP=0))
 torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 base,gpu,labels,meta=paired.load_source(paired.decisions(),'cuda:0');assert set(gpu)==set(FIELDS)and meta['N']==80016
 # Fixed source payload/admission identity is checked without selecting by scores.
 with np.load(V77/'runs/PAIRED_SUPPORT_V1/SELECTED_SOURCE_SCORES.npz',allow_pickle=False)as z:
  for k,want in [('id',meta['all_frame_id']),('source',meta['all_frame_source']),('split',meta['all_frame_split']),('primary_loss_mask',meta['primary'].reshape(10002,8)),('input_available',meta['input_available'].reshape(10002,8)),('reference_in_local_square_GT_only',meta['square'].reshape(10002,8))]:assert np.array_equal(z[k],want),k
 dense=load_dense(e,meta,'cuda:0');model=M.DenseSupportHead().cuda();initial=torch.load(p['bindings']['V75_initialization']['path'],map_location='cuda',weights_only=True);M.initialize_from_proxy(model,initial)
 O=imp(p['bindings']['unchanged_model_code']['path'],'source_preflight_original_V77_model');old=O.NativeContextSupportHead().cuda();O.initialize_from_proxy(old,initial);del initial
 for k,v in old.state_dict().items():assert torch.equal(v,model.state_dict()[k]),k
 assert sum(x.numel()for x in model.parameters())==41287 and not model.aux_decoder.weight.any()
 ids=paired.sample(meta,np.random.default_rng(20261023));ix=torch.as_tensor(ids,device='cuda');out=base.forward(model,gpu,ids)
 with torch.no_grad():oldout=base.forward(old,gpu,ids)
 assert torch.equal(out['support_logit'],oldout['support_logit'])and torch.equal(out['input_available'],oldout['input_available']);del old,oldout
 loss=L.combined_support_loss(out['support_logit'],labels[ix],out['aux_mask_logits'],**{k:v[ix]for k,v in dense.items()},aux_weight=.25)
 assert torch.isfinite(loss['loss']);loss['loss'].backward();assert all(x.grad is None or torch.isfinite(x.grad).all()for x in model.parameters());model.zero_grad(set_to_none=True)
 assert all(np.sum((meta['source'][ids]==s)&(meta['y'][ids]==c))==24 for s in range(2)for c in range(2))and meta['primary'][ids].all()and meta['train'][ids].all()
 poolcounts=dict(natural={f'{paired.SOURCES[s]}|{c}':len(meta['natural_pools'][s,c])for s in range(2)for c in range(2)},after_nonanchor={f'{paired.SOURCES[s]}|{c}':len(meta['nonanchor_pools'][s,c])for s in range(2)for c in range(2)},pairs_train={paired.SOURCES[s]:len(meta['pair_train_pools'][s])for s in range(2)})
 write(RUN/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,total_loss=float(loss['loss'].detach()),primary_BCE=float(loss['primary_BCE'].detach()),M0_loss=float(loss['M0_loss'].detach()),VgivenM0_loss=float(loss['VgivenM0_loss'].detach()),same_V77_initial_shared_state_and_primary_FP32=True,same_primary_admission=True,pools=poolcounts,frames=10002,slots=80016,sourceclass24=True,optimizer_updates=0,GTforward=False,current_PnP=0,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated()))
 del out,loss
 torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'UNTRAINED_UPDATE0.pt');allids=np.arange(meta['N']);validids=np.sort(np.concatenate(list(meta['valpools'].values())))
 baseline,aux=predict_all(base,model,gpu,allids,labels);assert np.array_equal(baseline['input_available'],meta['input_available']);paired.save_scores(RUN/'BASELINE0_SOURCE_SCORES.npz',baseline,meta);save_aux(RUN/'BASELINE0_AUX_LOGITS.npz',aux,meta);write(RUN/'BASELINE0_RESULTS.json',dict(selection_eligible=False,results=paired.report(base,baseline,meta)));del baseline,aux
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,betas=(.9,.999));rng=np.random.default_rng(20261023);best=float('inf');beststep=None;history=[];samplehash=hashlib.sha256()
 for step in range(1,2001):
  ids=paired.sample(meta,rng);samplehash.update(ids.tobytes());ix=torch.as_tensor(ids,device='cuda');assert meta['primary'][ids].all()and meta['train'][ids].all();model.train();opt.zero_grad(set_to_none=True)
  out=base.forward(model,gpu,ids);loss=L.combined_support_loss(out['support_logit'],labels[ix],out['aux_mask_logits'],**{k:v[ix]for k,v in dense.items()},aux_weight=.25);assert torch.isfinite(loss['loss']);loss['loss'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step()
  if step==1:write(RUN/'FIRST_UPDATE.json',dict(update=1,loss=float(loss['loss'].detach()),restart_allowed=False,entry=b(R/'TRAINER_ENTRYLOCK.json'),current_PnP=0));print('FIRST_UPDATE_COMMITTED',flush=True)
  if step%100==0:
   val=base.predict(model,gpu,validids,labels);metric,by=paired.natural_val_metric(val,validids,meta);saved=metric<best
   if saved:best=metric;beststep=step;torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'BEST_SUPPORT_HEAD.pt')
   rec=dict(update=step,train_loss=float(loss['loss'].detach()),primary_BCE=float(loss['primary_BCE'].detach()),M0_loss=float(loss['M0_loss'].detach()),VgivenM0_loss=float(loss['VgivenM0_loss'].detach()),naturalVAL4strata_macroBCE=metric,stratumBCE=by,saved=saved,best_update=beststep,seconds=time.monotonic()-start);history.append(rec);(RUN/'HISTORY.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(rec),flush=True)
 model.load_state_dict(torch.load(RUN/'BEST_SUPPORT_HEAD.pt',map_location='cuda',weights_only=True));model.eval().requires_grad_(False)
 write(RUN/'MODEL_FREEZE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,best_naturalVAL4strata_macroBCE=best,checkpoint=b(RUN/'BEST_SUPPORT_HEAD.pt'),model_code=b(R/'model/dense_support_head.py'),loss_code=b(R/'model/dense_auxiliary_loss.py'),decisions=b(DECISIONS),trainer=b(Path(__file__)),entry=b(R/'TRAINER_ENTRYLOCK.json'),parameter_count=41287,added_auxiliary_parameters=50,threshold=.5,aux_weight=.25,auxiliary_used_by_primary_selector=False,training_sample_sequence_SHA256=samplehash.hexdigest(),current_used_for_fit_or_checkpoint_or_threshold_selection=False,PnP=0,fully_solved=False))
 selected,aux=predict_all(base,model,gpu,allids,labels);assert np.array_equal(selected['input_available'],meta['input_available']);paired.save_scores(RUN/'SELECTED_SOURCE_SCORES.npz',selected,meta);save_aux(RUN/'SELECTED_AUX_LOGITS.npz',aux,meta)
 write(RUN/'SOURCE_RESULTS.json',dict(completed=True,model_freeze=b(RUN/'MODEL_FREEZE.json'),best_update=beststep,selection_naturalVALmacroBCE=best,results=paired.report(base,selected,meta),pools=poolcounts,scope='Unchanged reference-local primary support; auxiliaryM0/VgivenM0 are source-proxy diagnostics, not humanD/E/S.',GTforward=False,current_PnP=0))
 write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,model_freeze=b(RUN/'MODEL_FREEZE.json'),history=b(RUN/'HISTORY.json'),scores=b(RUN/'SELECTED_SOURCE_SCORES.npz'),auxiliary_logits=b(RUN/'SELECTED_AUX_LOGITS.npz'),source_results=b(RUN/'SOURCE_RESULTS.json'),seconds=time.monotonic()-start,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),current_PnP=0));print(json.dumps(b(RUN/'FIT_COMPLETE.json')),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generated','freeze','run']);ap.add_argument('--loss-review');ap.add_argument('--loss-review-sha');args=ap.parse_args();{'generated':generated,'freeze':lambda:freeze(args.loss_review,args.loss_review_sha),'run':run}[args.stage]()
