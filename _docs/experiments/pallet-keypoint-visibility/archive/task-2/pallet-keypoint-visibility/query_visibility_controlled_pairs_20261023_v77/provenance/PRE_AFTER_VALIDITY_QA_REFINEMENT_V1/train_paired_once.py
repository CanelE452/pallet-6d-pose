"""One V77 source-only paired-visibility data intervention, unchanged V76 model.

No current/private data paths. Original natural VAL alone chooses checkpoint.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,hashlib,importlib.util,json,random,shutil,sys,time
import numpy as np
import torch
from torch.nn import functional as F
R=Path(__file__).resolve().parent;V=R.parent/'query_visibility_reference_local_support_20261023_v76'
DECISIONS=R/'EXPERIMENT_DECISIONS_BEFORE_PAIR_BUILD.json';DECISION_SHA='01b856d88c88677c178ce27d103d7b6877f962a3f0cb2a736e0a0bcf4a9d2f1c'
RUN=R/'runs/PAIRED_SUPPORT_V1';SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL');FIELDS=('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for a in iter(lambda:f.read(1<<20),b''):h.update(a)
 return h.hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(x):assert sha(x['path'])==x['sha256'],x['path']
def load(p):return json.loads(Path(p).read_text())
def write(p,x):
 assert not p.exists(),str(p);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def imp(binding,name):
 check(binding);spec=importlib.util.spec_from_file_location(name,binding['path']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def decisions():
 assert sha(DECISIONS)==DECISION_SHA
 p=load(DECISIONS)
 for x in p['bindings'].values():check(x)
 assert p['sampling']['natural_original_per_source_class']==8 and p['sampling']['after_known_nonanchor_per_source_class']==8 and p['sampling']['matched_pairs_per_source']==8 and p['optimizer']['updates']==2000
 return p

def base_module():return imp(b(V/'mixed_fit/train_mixed_once.py'),'unchanged_v76_source_helpers')

def sample(meta,rng):
 ids=[]
 for name in ('natural_pools','nonanchor_pools'):
  for s in range(2):
   for c in range(2):ids.extend(rng.choice(meta[name][s,c],8,replace=True).tolist())
 for s in range(2):
  af=rng.choice(meta['pair_train_pools'][s],8,replace=True);ids.extend(np.stack((meta['pair_before_query'][af],meta['pair_after_query'][af]),1).reshape(-1).tolist())
 return np.asarray(ids,np.int64)

def generated():
 p=decisions();torch.set_num_threads(1);torch.manual_seed(20261023)
 mm=imp(p['bindings']['unchanged_model_code'],'bound_model');m=mm.NativeContextSupportHead();assert sum(v.numel()for v in m.parameters())==41237
 mm.initialize_from_proxy(m,torch.load(p['bindings']['V75_initialization']['path'],map_location='cpu',weights_only=True));assert not m.classifier[2].weight.any()
 # Artificial disjoint rows mimic source,class and pair association without sourcevalues.
 source=np.repeat([0,1],48);y=np.tile(np.repeat([0,1],24),2);meta={}
 meta['natural_pools']={(s,c):np.flatnonzero((source==s)&(y==c))[:8]for s in range(2)for c in range(2)}
 meta['nonanchor_pools']={(s,c):np.flatnonzero((source==s)&(y==c))[8:16]for s in range(2)for c in range(2)}
 meta['pair_train_pools']={0:np.arange(4),1:np.arange(4,8)}
 meta['pair_before_query']=np.r_[np.arange(40,44),np.arange(88,92)];meta['pair_after_query']=np.r_[np.arange(16,20),np.arange(64,68)]
 rng=np.random.default_rng(20261023)
 for _ in range(64):
  ids=sample(meta,rng);assert len(ids)==96 and all(np.sum((source[ids]==s)&(y[ids]==c))==24 for s in range(2)for c in range(2));assert all(y[ids[k]]==1 and y[ids[k+1]]==0 and source[ids[k]]==source[ids[k+1]]for k in range(64,96,2))
 logits=torch.randn(96,requires_grad=True);target=torch.randint(0,2,(96,)).float();loss=F.binary_cross_entropy_with_logits(logits,target);parts=torch.stack([F.binary_cross_entropy_with_logits(logits[i:i+32],target[i:i+32])for i in range(0,96,32)]);assert torch.allclose(loss,parts.mean());loss.backward();assert torch.isfinite(logits.grad).all()
 write(R/'GENERATED_TRAINER_TESTS.json',dict(PASS=True,code=b(Path(__file__)),decisions=b(DECISIONS),model_code=p['bindings']['unchanged_model_code'],parameter_count=41237,balanced96_source_class24_checked_batches=64,three_equal32components=True,paired_beforeV_afterCadjacency=True,only_generated_inputs=True,source_RGB_masks_P3_GT_values_read=0,optimizer_updates=0,current_PnP=0));print('GENERATED_PASS',flush=True)

def freeze():
 p=decisions();proof=load(R/'GENERATED_TRAINER_TESTS.json');assert proof['PASS']and proof['code']==b(Path(__file__))
 inp=load(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json')
 for x in inp['bindings'].values():check(x)
 write(R/'TRAINER_ENTRYLOCK.json',dict(schema='V77_ONE_PAIRED_SOURCE_FIT_ENTRY',decisions=b(DECISIONS),trainer=b(Path(__file__)),generated=b(R/'GENERATED_TRAINER_TESTS.json'),source_inputs=b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json'),run_directory=str(RUN),runtime=dict(python=str(Path(sys.executable).resolve()),torch=str(torch.__version__),device='cuda:0',threads=1),restart_after_first_update=False,current_read_or_PnP=False));print('ENTRY_FROZEN',flush=True)

def load_source(p,device):
 base=base_module();oldp=load(V/'mixed_fit/MIXED_FIT_PROTOCOL.json');gpu,oldlabels,meta0=base.load_source(oldp,device)
 inp=load(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json');[check(x)for x in inp['bindings'].values()]
 man=load(inp['bindings']['after_input_manifest']['path']);assert man['completed'];A=man['frame_count'];assert A>0
 idx=[json.loads(x)for x in Path(inp['bindings']['after_input_lineage']['path']).read_text().splitlines()];assert len(idx)==A
 names=['p3_roi','rgb96','crop_valid96','target_rgb128','target_valid128','context42','query8_xy','query_center_valid','ROI_anyvalid','L'];parts={k:[]for k in names}
 for chunk in man['chunks']:
  check(chunk['file'])
  with np.load(chunk['file']['path'],allow_pickle=False)as z:
   for k in names:parts[k].append(z[k])
 data={k:np.ascontiguousarray(np.concatenate(v))for k,v in parts.items()};del parts
 with np.load(inp['bindings']['after_teacher_GT']['path'],allow_pickle=False)as z:gt={k:z[k]for k in z.files}
 with np.load(p['bindings']['teacher_GT']['path'],allow_pickle=False)as z:original={k:z[k]for k in ['id','source','split','native_GTref8','convex_self_proxy8_separate']}
 orig=gt['original_row_index'].astype(np.int64);assert len(orig)==A and len(np.unique(orig))==A and (orig>=0).all()and(orig<5203).all()and np.array_equal(gt['after_row_index'],np.arange(A))
 assert all(line['id']==gt['id'][i]and line['original_row_index']==orig[i]and {'validation':'val'}.get(line['split'],line['split'])==gt['split'][i]for i,line in enumerate(idx))
 assert np.array_equal(gt['source'],original['source'][orig])and np.array_equal(gt['split'],original['split'][orig])and np.array_equal(gt['native_GTref8'],original['native_GTref8'][orig],equal_nan=True)and np.array_equal(gt['convex_self_proxy8_separate'],original['convex_self_proxy8_separate'][orig])
 local=data['crop_valid96'].reshape(A,8,-1).any(-1);target=data['target_valid128'].reshape(A,-1).any(-1)
 primary,square,offset,available=base.admission(gt['label8'],data['query_center_valid'],data['ROI_anyvalid'],local,target,gt['native_GTref8'],data['query8_xy'],data['L'])
 mapped=dict(p3_roi=data['p3_roi'].reshape(-1,65,9,9),local_rgb96=data['rgb96'].reshape(-1,3,96,96),local_valid96=data['crop_valid96'].reshape(-1,1,96,96),target_rgb128=data['target_rgb128'],target_valid128=data['target_valid128'],context42=data['context42'].reshape(-1,42))
 for k in FIELDS:
  add=torch.from_numpy(np.ascontiguousarray(mapped[k])).to(device);gpu[k]=torch.cat((gpu[k],add),0)
 del data,mapped
 meta={};newmeta=dict(y=gt['label8'].reshape(-1).astype(np.int64),primary=primary.reshape(-1),square=square.reshape(-1),offset=offset.reshape(-1,2),qvalid=gt['query_center_valid8'].reshape(-1)if 'query_center_valid8'in gt else None,input_available=available.reshape(-1),source=np.repeat([SOURCES.index(s)for s in gt['source']],8),train=np.repeat(gt['split']=='train',8),val=np.repeat(gt['split']=='val',8),near_boundary=gt['near_boundary8'].reshape(-1),rule=gt['rule8'].reshape(-1),self_visible_proxy=gt['convex_self_proxy8_separate'].reshape(-1))
 # Factual qvalid is from input cache, never fromGT. Keep it before datarelease.
 # Original validity can be recovered exactly for same unchangedgeometry via originalmapping.
 newmeta['qvalid']=meta0['qvalid'].reshape(5203,8)[orig].reshape(-1)
 for key in ['y','primary','square','offset','qvalid','input_available','source','train','val','near_boundary','rule','self_visible_proxy']:meta[key]=np.concatenate((meta0[key],newmeta[key]))
 N=(5203+A)*8;assert len(meta['y'])==N
 domain=np.r_[np.zeros(41624,np.int8),np.ones(A*8,np.int8)];anchor=gt['anchor_k'].astype(np.int64);assert ((anchor>=0)&(anchor<8)).all()
 pairbefore=orig*8+anchor;pairafter=(5203+np.arange(A))*8+anchor
 pairgood=meta['primary'][pairbefore]&(meta['y'][pairbefore]==1)&meta['primary'][pairafter]&(meta['y'][pairafter]==0)
 natural={(s,c):np.flatnonzero((domain==0)&meta['primary']&meta['train']&(meta['source']==s)&(meta['y']==c))for s in range(2)for c in range(2)}
 is_anchor=np.zeros(N,bool);is_anchor[pairafter]=True
 nonanchor={(s,c):np.flatnonzero((domain==1)&~is_anchor&meta['primary']&meta['train']&(meta['source']==s)&(meta['y']==c))for s in range(2)for c in range(2)}
 pairtrain={s:np.flatnonzero(pairgood&(gt['split']=='train')&(gt['source']==SOURCES[s]))for s in range(2)}
 assert all(len(x)>0 for x in [*natural.values(),*nonanchor.values(),*pairtrain.values()]),'No silent empty-poolfallback'
 meta.update(domain=domain,natural_pools=natural,nonanchor_pools=nonanchor,pair_train_pools=pairtrain,pair_before_query=pairbefore,pair_after_query=pairafter,pair_good=pairgood,after_original_row=orig,anchor_k=anchor,after_source=gt['source'],after_split=gt['split'],all_frame_id=np.r_[original['id'],gt['id']],all_frame_source=np.r_[original['source'],gt['source']],all_frame_split=np.r_[original['split'],gt['split']],valpools=meta0['valpools'],N=N,A=A)
 assert all(np.array_equal(natural[s,c],meta0['pools'][s,c])for s in range(2)for c in range(2))
 labels=torch.as_tensor(meta['y'],device=device);del oldlabels
 return base,gpu,labels,meta

def natural_val_metric(pred,ids,meta):
 where=np.full(meta['N'],-1,np.int64);where[ids]=np.arange(len(ids));by={f'{SOURCES[s]}|{c}':float(pred['NLL_F32'][where[ix]].astype(np.float64).mean())for(s,c),ix in meta['valpools'].items()};return float(np.mean(list(by.values()))),by

def report(base,pred,meta):
 result={}
 for domain,name in [(0,'NATURAL_ORIGINAL'),(1,'CONTROLLED_AFTER')]:
  for part in ('train','val'):
   for s,source in enumerate(SOURCES):
    raw=(meta['domain']==domain)&meta[part]&(meta['source']==s);m=raw&meta['primary'];met=base.cm_metrics(pred,m,meta)
    result[f'{name}|{source}|{part}']=dict(metrics=met,all_slots=int(raw.sum()),known=int(np.sum(raw&np.isin(meta['y'],[0,1]))),admitted=int(m.sum()),U=int(np.sum(raw&(meta['y']==-1))),known_ref_outside_square=int(np.sum(raw&np.isin(meta['y'],[0,1])&~meta['square'])),subgroups={n:base.cm_metrics(pred,m&cond,meta)for n,cond in dict(strong=np.char.startswith(meta['rule'].astype(str),'STRONG'),fallback=~np.char.startswith(meta['rule'].astype(str),'STRONG'),convex_self_hidden=meta['self_visible_proxy']==0,convex_may_visible=meta['self_visible_proxy']==1).items()})
 pairs={}
 for part in ('train','val'):
  for s,source in enumerate(SOURCES):
   take=np.flatnonzero(meta['pair_good']&(meta['after_split']==part)&(meta['after_source']==source));bi=meta['pair_before_query'][take];ai=meta['pair_after_query'][take];bz=pred['logits'][bi];az=pred['logits'][ai];before=bz>=0;after=az<0
   pairs[f'{source}|{part}']=dict(N=len(take),before_VISIBLE_correct=int(before.sum()),after_COVERED_correct=int(after.sum()),both_correct=int((before&after).sum()),before_logit_gt_after=int((bz>az).sum()),both_correct_rate=float((before&after).mean())if len(take)else None,mean_before_logit=float(bz.mean())if len(take)else None,mean_after_logit=float(az.mean())if len(take)else None)
 return dict(natural_and_after=result,paired_anchors=pairs)

def save_scores(path,pred,meta):np.savez_compressed(path,support_logit=pred['logits'].reshape(-1,8),NLL_FP32=pred['NLL_F32'].reshape(-1,8),input_available=pred['input_available'].reshape(-1,8),primary_loss_mask=meta['primary'].reshape(-1,8),reference_in_local_square_GT_only=meta['square'].reshape(-1,8),id=meta['all_frame_id'],source=meta['all_frame_source'],split=meta['all_frame_split'],domain=meta['domain'].reshape(-1,8)[:,0],after_original_row_index=meta['after_original_row'],after_anchor_k=meta['anchor_k'],pair_good=meta['pair_good'])

def run():
 p=decisions();e=load(R/'TRAINER_ENTRYLOCK.json');assert e['trainer']==b(Path(__file__))and e['decisions']==b(DECISIONS)and e['source_inputs']==b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json');check(e['generated']);assert not RUN.exists(),'Preserveexistingrun;noautomaticrestart'
 assert Path(sys.executable).resolve()==Path('/home/minjae/anaconda3/bin/python').resolve()and str(torch.__version__)=='2.5.1+cu121'and torch.cuda.is_available()
 RUN.mkdir(parents=True);(RUN/'code_snapshot').mkdir()
 for x in [e['trainer'],e['decisions'],e['source_inputs'],p['bindings']['unchanged_model_code']]:shutil.copy2(x['path'],RUN/'code_snapshot'/Path(x['path']).name)
 write(RUN/'RUN_STARTED.json',dict(entry=b(R/'TRAINER_ENTRYLOCK.json'),optimizer_updates=0,current_PnP=0))
 torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 base,gpu,labels,meta=load_source(p,'cuda:0');M=imp(p['bindings']['unchanged_model_code'],'locked_pair_model');model=M.NativeContextSupportHead().cuda();M.initialize_from_proxy(model,torch.load(p['bindings']['V75_initialization']['path'],map_location='cuda',weights_only=True))
 ids=sample(meta,np.random.default_rng(20261023));ix=torch.as_tensor(ids,device='cuda');out=base.forward(model,gpu,ids);loss=F.binary_cross_entropy_with_logits(out['support_logit'],labels[ix].float());assert torch.isfinite(loss);loss.backward();model.zero_grad(set_to_none=True)
 assert all(np.sum((meta['source'][ids]==s)&(meta['y'][ids]==c))==24 for s in range(2)for c in range(2))and meta['primary'][ids].all()and meta['train'][ids].all()
 poolcounts=dict(natural={f'{SOURCES[s]}|{c}':len(meta['natural_pools'][s,c])for s in range(2)for c in range(2)},after_nonanchor={f'{SOURCES[s]}|{c}':len(meta['nonanchor_pools'][s,c])for s in range(2)for c in range(2)},pairs_train={SOURCES[s]:len(meta['pair_train_pools'][s])for s in range(2)})
 write(RUN/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,loss=float(loss.detach()),pools=poolcounts,frames=5203+meta['A'],slots=meta['N'],sourceclass24=True,optimizer_updates=0,GTforward=False,current_PnP=0,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated()))
 torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'UNTRAINED_UPDATE0.pt');allids=np.arange(meta['N']);validids=np.concatenate(list(meta['valpools'].values()));validids=np.sort(validids)
 baseline=base.predict(model,gpu,allids,labels);assert np.array_equal(baseline['input_available'],meta['input_available']);save_scores(RUN/'BASELINE0_SOURCE_SCORES.npz',baseline,meta);write(RUN/'BASELINE0_RESULTS.json',dict(selection_eligible=False,results=report(base,baseline,meta)));del baseline
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,betas=(.9,.999));rng=np.random.default_rng(20261023);best=float('inf');beststep=None;history=[]
 for step in range(1,2001):
  ids=sample(meta,rng);ix=torch.as_tensor(ids,device='cuda');assert meta['primary'][ids].all()and meta['train'][ids].all();model.train();opt.zero_grad(set_to_none=True);z=base.forward(model,gpu,ids)['support_logit'];loss=F.binary_cross_entropy_with_logits(z,labels[ix].float());assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(g);opt.step()
  if step==1:write(RUN/'FIRST_UPDATE.json',dict(update=1,loss=float(loss.detach()),restart_allowed=False,current_PnP=0,entry=b(R/'TRAINER_ENTRYLOCK.json')));print('FIRST_UPDATE_COMMITTED',flush=True)
  if step%100==0:
   val=base.predict(model,gpu,validids,labels);metric,by=natural_val_metric(val,validids,meta);saved=metric<best
   if saved:best=metric;beststep=step;torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'BEST_SUPPORT_HEAD.pt')
   rec=dict(update=step,train_loss=float(loss.detach()),naturalVAL4strata_macroBCE=metric,stratumBCE=by,saved=saved,best_update=beststep,seconds=time.monotonic()-start);history.append(rec);(RUN/'HISTORY.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(rec),flush=True)
 model.load_state_dict(torch.load(RUN/'BEST_SUPPORT_HEAD.pt',map_location='cuda',weights_only=True));model.eval().requires_grad_(False)
 write(RUN/'MODEL_FREEZE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,best_naturalVAL4strata_macroBCE=best,checkpoint=b(RUN/'BEST_SUPPORT_HEAD.pt'),model_code=p['bindings']['unchanged_model_code'],decisions=b(DECISIONS),trainer=b(Path(__file__)),entry=b(R/'TRAINER_ENTRYLOCK.json'),source_inputs=b(R/'COMPLETED_SOURCE_INPUT_BINDINGS.json'),parameter_count=41237,threshold=.5,current_used_for_fit_or_checkpoint_or_threshold_selection=False,PnP=0,fully_solved=False))
 selected=base.predict(model,gpu,allids,labels);assert np.array_equal(selected['input_available'],meta['input_available']);save_scores(RUN/'SELECTED_SOURCE_SCORES.npz',selected,meta);write(RUN/'SOURCE_RESULTS.json',dict(completed=True,model_freeze=b(RUN/'MODEL_FREEZE.json'),best_update=beststep,selection_naturalVALmacroBCE=best,results=report(base,selected,meta),pools=poolcounts,scope='Sourceconditionalreference-localsupport; controlledpairs nothumansemanticgold. NaturalVALselectoronly;no current.',GTforward=False,current_PnP=0))
 write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,model_freeze=b(RUN/'MODEL_FREEZE.json'),history=b(RUN/'HISTORY.json'),scores=b(RUN/'SELECTED_SOURCE_SCORES.npz'),source_results=b(RUN/'SOURCE_RESULTS.json'),seconds=time.monotonic()-start,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),current_PnP=0));print(json.dumps(dict(COMPLETE=b(RUN/'FIT_COMPLETE.json'),best_update=beststep,best_BCE=best)),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['generated','freeze','run']);a=ap.parse_args();{'generated':generated,'freeze':freeze,'run':run}[a.stage]()
