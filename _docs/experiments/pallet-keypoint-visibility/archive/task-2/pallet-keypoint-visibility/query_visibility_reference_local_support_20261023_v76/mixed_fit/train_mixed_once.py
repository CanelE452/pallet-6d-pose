"""One source-only V76 mixed weak-support fit. Current evaluation is separate.

GT joins supervise loss/admission/report only; forward has six Tensor inputs.
No restart or mutable protocol after optimizer update1.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,importlib.util,inspect,json,random,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
PROTOCOL=ROOT/'MIXED_FIT_PROTOCOL.json'
PROTOCOL_SHA='8e2d4e88e8638e2e2c6ef0a1291112d4d1ead40531eca175de2d8086d7ba80f6'
FIELDS=('p3_roi','local_rgb96','local_valid96','target_rgb128','target_valid128','context42')
SOURCES=('SYNTHETIC','YCB_PUBLIC_REAL')
RUN=ROOT/'runs/MIXED_SUPPORT_V1'

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def load(p):return json.loads(Path(p).read_text())
def write(p,v):
 assert not Path(p).exists(),str(p)
 Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def replace_history(v):
 (RUN/'HISTORY.json').write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def protocol():
 assert sha(PROTOCOL)==PROTOCOL_SHA
 p=load(PROTOCOL)
 for b in p['bindings'].values():check(b)
 assert p['parameter_count']==41237 and p['optimizer']['updates']==2000
 assert p['sampling']['batch_queries']==96 and p['sampling']['per_source_class']==24
 return p

def module(p):
 s=importlib.util.spec_from_file_location('bound_v76_support',p['bindings']['model_code']['path']);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def admission(y,qvalid,roi,local,target,r,q,L):
 offset=(r-q)/L[:,None,None]
 square=np.isfinite(offset).all(-1)&(offset>=-.5).all(-1)&(offset<.5).all(-1)
 available=roi&local&target[:,None]
 primary=np.isin(y,[0,1])&qvalid&available&square
 return primary,square,offset,available

def objective(logits,labels,primary):
 known=primary.bool()&((labels==0)|(labels==1))
 return F.binary_cross_entropy_with_logits(logits[known],labels[known].to(logits.dtype)) if known.any() else logits.sum()*0

def sample(pools,rng):return np.concatenate([rng.choice(pools[(s,c)],24,replace=True)for s in range(2)for c in range(2)])

def generated():
 p=protocol();torch.set_num_threads(1);torch.manual_seed(20261023);m=module(p);model=m.NativeContextSupportHead()
 assert tuple(inspect.signature(model.forward).parameters)==FIELDS
 assert sum(x.numel()for x in model.parameters())==41237 and not any('localizer' in k for k in model.state_dict())
 rejected=m.initialize_from_proxy(model,torch.load(p['bindings']['pretrain']['path'],map_location='cpu',weights_only=True))
 assert sorted(rejected)==['classifier.2.bias','classifier.2.weight','reference_localizer.bias','reference_localizer.weight']
 assert not model.classifier[2].weight.any() and not model.classifier[2].bias.any()
 vals=dict(p3_roi=torch.randn(1,8,65,9,9),local_rgb96=torch.randint(0,256,(1,8,3,96,96),dtype=torch.uint8),local_valid96=torch.ones(1,8,1,96,96,dtype=torch.bool),target_rgb128=torch.randint(0,256,(1,3,128,128),dtype=torch.uint8),target_valid128=torch.ones(1,1,128,128,dtype=torch.bool),context42=torch.randn(1,8,42));vals['p3_roi'][:,:,64]=1
 whole=model(**vals);flat={k:(v.repeat_interleave(8,0) if k in ('target_rgb128','target_valid128')else v.reshape(8,*v.shape[2:]))for k,v in vals.items()}
 result=model(**flat);assert torch.equal(whole['support_logit'].flatten(),result['support_logit']) and (result['support_logit']==0).all()
 try:model(**flat,reference_GT=torch.zeros(8,2))
 except TypeError:pass
 else:raise AssertionError('GTforwardmustreject')
 z=torch.randn(7,requires_grad=True);y=torch.tensor([1,0,-1,1,0,1,0]);pm=torch.tensor([1,1,1,0,0,1,0],dtype=torch.bool);objective(z,y,pm).backward();assert not z.grad[[2,3,4,6]].any() and z.grad[[0,1,5]].abs().min()>0
 yy=np.array([[1,0,-1,1,0,1,1,0]]);q=np.zeros((1,8,2));r=q.copy();r[0,3]=[.5,0];r[0,4]=[-.5,0];r[0,5]=[0,np.nan]
 qv=np.ones((1,8),bool);qv[0,6]=False;roi=np.ones((1,8),bool);roi[0,7]=False
 pm,sq,_,_=admission(yy,qv,roi,np.ones_like(roi),np.ones(1,bool),r,q,np.ones(1));assert pm.tolist()==[[True,True,False,False,True,False,False,False]] and sq[0,4] and not sq[0,3]
 src=np.repeat(np.arange(2),8);labels=np.tile(np.repeat(np.arange(2),4),2);pools={(s,c):np.flatnonzero((src==s)&(labels==c))for s in range(2)for c in range(2)};rng=np.random.default_rng(20261023)
 for _ in range(64):
  ix=sample(pools,rng);assert len(ix)==96 and all(np.sum((src[ix]==s)&(labels[ix]==c))==24 for s in range(2)for c in range(2))
 write(ROOT/'GENERATED_TRAINER_TESTS.json',dict(PASS=True,protocol=bind(PROTOCOL),code=bind(Path(__file__)),model_code=p['bindings']['model_code'],six_tensor_forward=True,GT_rejected=True,U_invalid_miss_loss_gradient_zero=True,reference_halfopen_square_edges_checked=True,whole_flat_exact_zero_initialized=True,old_DES_and_localizer_excluded=rejected,balanced_fourstrata_batches=64,source_arrays_RGB_read=0,optimizer_updates=0,current_PnP=0))
 print('GENERATED_PASS',flush=True)

def freeze():
 p=protocol();g=load(ROOT/'GENERATED_TRAINER_TESTS.json');assert g['PASS'] and g['code']==bind(Path(__file__))
 src=load(ROOT/'SOURCE_INPUT_BINDINGS.json')
 assert [x['source']for x in src['packs']]==list(SOURCES)
 for b in src['additional_bindings']:check(b)
 for pack in src['packs']:
  for key in ('manifest','seal','lineage'):check(pack[key])
 write(ROOT/'TRAINER_ENTRYLOCK.json',dict(schema='V76_ONE_MIXED_SUPPORT_FIT_ENTRYLOCK',protocol=bind(PROTOCOL),trainer=bind(Path(__file__)),generated=bind(ROOT/'GENERATED_TRAINER_TESTS.json'),source_inputs=bind(ROOT/'SOURCE_INPUT_BINDINGS.json'),run_directory=str(RUN),restart_allowed_after_first_update=False,forward_fields=list(FIELDS),current_read_or_PnP_allowed=False,python=str(Path(sys.executable).resolve()),torch=str(torch.__version__)))
 print('ENTRY_FROZEN',flush=True)

def load_source(p,device):
 bindings=load(ROOT/'SOURCE_INPUT_BINDINGS.json');parts={k:[]for k in ['p3_roi','rgb96','crop_valid96','target_rgb128','target_valid128','context42','query8_xy','query_center_valid','ROI_anyvalid','L']};indices=[]
 for pack in bindings['packs']:
  for key in ('manifest','seal','lineage'):check(pack[key])
  man=load(pack['manifest']['path']);assert man['completed']
  lines=[json.loads(x)for x in Path(pack['lineage']['path']).read_text().splitlines()];assert len(lines)==man['frame_count']
  indices.extend(lines)
  for chunk in man['chunks']:
   check(chunk['file'])
   with np.load(chunk['file']['path'],allow_pickle=False)as z:
    for key in parts:parts[key].append(z[key])
 data={k:np.ascontiguousarray(np.concatenate(v))for k,v in parts.items()};del parts
 with np.load(p['bindings']['teacher_GT']['path'],allow_pickle=False)as z:gt={k:z[k]for k in z.files}
 assert len(indices)==5203 and data['p3_roi'].shape==(5203,8,65,9,9)
 assert all(r['id']==gt['id'][i] and {'validation':'val'}.get(r['split'],r['split'])==gt['split'][i]for i,r in enumerate(indices))
 assert np.array_equal(gt['source'],np.array(['SYNTHETIC']*1003+['YCB_PUBLIC_REAL']*4200)) and np.all(gt['human_semantic_visibility8']==-1)
 local=data['crop_valid96'].reshape(5203,8,-1).any(-1);target=data['target_valid128'].reshape(5203,-1).any(-1)
 primary,square,offset,available=admission(gt['label8'],data['query_center_valid'],data['ROI_anyvalid'],local,target,gt['native_GTref8'],data['query8_xy'],data['L'])
 meta=dict(y=gt['label8'].reshape(-1).astype(np.int64),primary=primary.reshape(-1),square=square.reshape(-1),offset=offset.reshape(-1,2),qvalid=data['query_center_valid'].reshape(-1),input_available=available.reshape(-1),
  source=np.repeat(np.array([SOURCES.index(s)for s in gt['source']]),8),train=np.repeat(gt['split']=='train',8),val=np.repeat(gt['split']=='val',8),near_boundary=gt['near_boundary8'].reshape(-1),rule=gt['rule8'].reshape(-1),self_visible_proxy=gt['convex_self_proxy8_separate'].reshape(-1),ids=gt['id'],source_frames=gt['source'],split_frames=gt['split'])
 meta['pools']={(s,c):np.flatnonzero(meta['primary']&meta['train']&(meta['source']==s)&(meta['y']==c))for s in range(2)for c in range(2)}
 meta['valpools']={(s,c):np.flatnonzero(meta['primary']&meta['val']&(meta['source']==s)&(meta['y']==c))for s in range(2)for c in range(2)}
 assert all(len(ix)>0 for ix in list(meta['pools'].values())+list(meta['valpools'].values()))
 gpu=dict(p3_roi=torch.from_numpy(data['p3_roi'].reshape(-1,65,9,9)).to(device),local_rgb96=torch.from_numpy(data['rgb96'].reshape(-1,3,96,96)).to(device),local_valid96=torch.from_numpy(data['crop_valid96'].reshape(-1,1,96,96)).to(device),target_rgb128=torch.from_numpy(data['target_rgb128']).to(device),target_valid128=torch.from_numpy(data['target_valid128']).to(device),context42=torch.from_numpy(data['context42'].reshape(-1,42)).to(device))
 del data
 return gpu,torch.as_tensor(meta['y'],device=device),meta

def forward(model,gpu,ids):
 ix=torch.as_tensor(ids,dtype=torch.long,device=gpu['p3_roi'].device);frame=ix//8
 return model(**{k:v[frame if k in ('target_rgb128','target_valid128')else ix]for k,v in gpu.items()})

def predict(model,gpu,ids,labels):
 model.eval();zs=[];ns=[];av=[]
 with torch.inference_mode():
  for start in range(0,len(ids),128):
   ix=ids[start:start+128];out=forward(model,gpu,ix);zs.append(out['support_logit'].cpu().numpy());av.append(out['input_available'].cpu().numpy())
   ns.append(F.binary_cross_entropy_with_logits(out['support_logit'],labels[torch.as_tensor(ix,device=labels.device)].clamp(0,1).float(),reduction='none').cpu().numpy())
 return dict(logits=np.concatenate(zs),NLL_F32=np.concatenate(ns),input_available=np.concatenate(av))

def metric(pred,ids,meta):
 where=np.full(41624,-1,np.int64);where[ids]=np.arange(len(ids));by={f'{SOURCES[s]}|{c}':float(pred['NLL_F32'][where[ix]].astype(np.float64).mean())for(s,c),ix in meta['valpools'].items()}
 return float(np.mean(list(by.values()))),by

def cm_metrics(pred,mask,meta):
 ix=np.flatnonzero(mask);y=meta['y'][ix];yp=(pred['logits'][ix]>=0).astype(np.int64);cm=np.zeros((2,2),np.int64);np.add.at(cm,(y,yp),1)
 return dict(N=len(ix),confusion_rows_GT_cols_pred_COVERED_VISIBLE=cm.tolist(),visible_precision=float(cm[1,1]/cm[:,1].sum())if cm[:,1].sum()else None,visible_recall=float(cm[1,1]/cm[1].sum())if cm[1].sum()else None,covered_precision=float(cm[0,0]/cm[:,0].sum())if cm[:,0].sum()else None,covered_recall=float(cm[0,0]/cm[0].sum())if cm[0].sum()else None,mean_BCE_FP32=float(pred['NLL_F32'][ix].astype(np.float64).mean())if len(ix)else None)

def report(pred,meta):
 result={}
 for part in ('train','val'):
  for s,name in enumerate(SOURCES):
   base=meta[part]&(meta['source']==s);mask=base&meta['primary'];known=base&np.isin(meta['y'],[0,1])
   sub={}
   for key,cond in dict(near_boundary=meta['near_boundary'],not_near_boundary=~meta['near_boundary'],strong_center=np.char.startswith(meta['rule'].astype(str),'STRONG'),fallback=~np.char.startswith(meta['rule'].astype(str),'STRONG'),convex_self_hidden=meta['self_visible_proxy']==0,convex_self_may_visible=meta['self_visible_proxy']==1).items():sub[key]=cm_metrics(pred,mask&cond,meta)
   result[f'{name}|{part}']=dict(metrics=cm_metrics(pred,mask,meta),all_slots=int(base.sum()),teacher_known=int(known.sum()),teacher_U=int(np.sum(base&(meta['y']==-1))),query_valid=int(np.sum(base&meta['qvalid'])),input_available=int(np.sum(base&meta['input_available'])),known_reference_outside_query_square=int(np.sum(known&~meta['square'])),admitted=int(mask.sum()),admission_rate=float(mask.sum()/base.sum()),conditional_subgroups=sub)
 return result

def save_scores(path,pred,meta):np.savez_compressed(path,support_logit=pred['logits'].reshape(5203,8),NLL_FP32=pred['NLL_F32'].reshape(5203,8),input_available=pred['input_available'].reshape(5203,8),primary_loss_mask=meta['primary'].reshape(5203,8),reference_in_local_square_GT_only=meta['square'].reshape(5203,8),id=meta['ids'],source=meta['source_frames'],split=meta['split_frames'])

def run():
 p=protocol();entry=load(ROOT/'TRAINER_ENTRYLOCK.json');assert entry['trainer']==bind(Path(__file__)) and entry['protocol']==bind(PROTOCOL) and entry['source_inputs']==bind(ROOT/'SOURCE_INPUT_BINDINGS.json');check(entry['generated'])
 assert not RUN.exists(),'Existingfitmustbepreserved; no automatic restart'
 assert Path(sys.executable).resolve()==Path(p['runtime']['python']).resolve() and str(torch.__version__)==p['runtime']['torch'] and torch.cuda.is_available()
 RUN.mkdir(parents=True);(RUN/'code_snapshot').mkdir()
 for b in [entry['trainer'],entry['protocol'],p['bindings']['model_code'],entry['source_inputs']]:shutil.copy2(b['path'],RUN/'code_snapshot'/Path(b['path']).name)
 write(RUN/'RUN_STARTED.json',dict(protocol=bind(PROTOCOL),entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),optimizer_updates=0,current_PnP=0))
 torch.set_num_threads(1);torch.manual_seed(20261023);torch.cuda.manual_seed_all(20261023);np.random.seed(20261023);random.seed(20261023);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 gpu,labels,meta=load_source(p,'cuda:0');M=module(p);model=M.NativeContextSupportHead().cuda();M.initialize_from_proxy(model,torch.load(p['bindings']['pretrain']['path'],map_location='cuda',weights_only=True));assert sum(x.numel()for x in model.parameters())==41237
 poolcounts={f'{SOURCES[s]}|{c}':dict(train=len(meta['pools'][(s,c)]),val=len(meta['valpools'][(s,c)]))for s in range(2)for c in range(2)};validids=np.flatnonzero(meta['primary']&meta['val']);allids=np.arange(41624)
 ids=sample(meta['pools'],np.random.default_rng(20261023));out=forward(model,gpu,ids);loss=objective(out['support_logit'],labels[torch.as_tensor(ids,device='cuda')],torch.ones(96,dtype=torch.bool,device='cuda'));assert torch.isfinite(loss);loss.backward();model.zero_grad(set_to_none=True)
 write(RUN/'SOURCE_GPU_PREFLIGHT.json',dict(PASS=True,optimizer_updates=0,pool_counts=poolcounts,loss=float(loss.detach()),peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),forward_fields=list(FIELDS),GT_forward=False,current_PnP=0))
 torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'UNTRAINED_UPDATE0.pt')
 baseline=predict(model,gpu,allids,labels);assert np.array_equal(baseline['input_available'],meta['input_available']);save_scores(RUN/'BASELINE0_SOURCE_SCORES.npz',baseline,meta);bm,_=metric(dict(NLL_F32=baseline['NLL_F32'][validids]),validids,meta)
 write(RUN/'BASELINE0_RESULTS.json',dict(update=0,selection_eligible=False,heldout4strata_macroBCE=bm,metrics=report(baseline,meta)));del baseline
 opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001,betas=(.9,.999));rng=np.random.default_rng(20261023);best=float('inf');beststep=None;history=[]
 for step in range(1,2001):
  ids=sample(meta['pools'],rng);ix=torch.as_tensor(ids,device='cuda');model.train();opt.zero_grad(set_to_none=True);loss=objective(forward(model,gpu,ids)['support_logit'],labels[ix],torch.ones(96,dtype=torch.bool,device='cuda'));assert torch.isfinite(loss);loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad);opt.step()
  if step==1:write(RUN/'FIRST_UPDATE.json',dict(update=1,loss=float(loss.detach()),restart_allowed=False,entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),current_PnP=0));print('FIRST_UPDATE_COMMITTED',flush=True)
  if step%100==0:
   pred=predict(model,gpu,validids,labels);score,by=metric(pred,validids,meta);saved=score<best
   if saved:best=score;beststep=step;torch.save({k:v.detach().cpu()for k,v in model.state_dict().items()},RUN/'BEST_SUPPORT_HEAD.pt')
   rec=dict(update=step,train_loss=float(loss.detach()),heldout4strata_macroBCE=score,stratumBCE=by,saved=saved,best_update=beststep,seconds=time.monotonic()-start);history.append(rec);replace_history(history);print(json.dumps(rec),flush=True)
 model.load_state_dict(torch.load(RUN/'BEST_SUPPORT_HEAD.pt',map_location='cuda',weights_only=True));model.eval().requires_grad_(False)
 write(RUN/'MODEL_FREEZE.json',dict(completed=True,phase='MIXED_PUBLIC_REAL_SYNTH_REFERENCE_LOCAL_SUPPORT',optimizer_updates=2000,best_update=beststep,best_heldout4strata_macroBCE=best,checkpoint=bind(RUN/'BEST_SUPPORT_HEAD.pt'),model_code=p['bindings']['model_code'],protocol=bind(PROTOCOL),trainer=bind(Path(__file__)),entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),source_inputs=bind(ROOT/'SOURCE_INPUT_BINDINGS.json'),parameter_count=41237,visible_probability_threshold=.5,semantic_gold=False,current_used_for_fit_selection=False,PnP=0,fully_solved=False))
 selected=predict(model,gpu,allids,labels);assert np.array_equal(selected['input_available'],meta['input_available']);save_scores(RUN/'SELECTED_SOURCE_SCORES.npz',selected,meta)
 write(RUN/'SOURCE_RESULTS.json',dict(completed=True,model_freeze=bind(RUN/'MODEL_FREEZE.json'),best_update=beststep,selection_macroBCE=best,source_metrics=report(selected,meta),pool_counts=poolcounts,GT_forward=False,current_PnP=0,scope='Weakreference-local support; publicsourceGT-derivedqueryemulation, notactualinitializerperformance.'))
 write(RUN/'FIT_COMPLETE.json',dict(completed=True,optimizer_updates=2000,best_update=beststep,model_freeze=bind(RUN/'MODEL_FREEZE.json'),history=bind(RUN/'HISTORY.json'),scores=bind(RUN/'SELECTED_SOURCE_SCORES.npz'),source_results=bind(RUN/'SOURCE_RESULTS.json'),seconds=time.monotonic()-start,peak_GPU_memory_bytes=torch.cuda.max_memory_allocated(),current_PnP=0))
 print(json.dumps(dict(COMPLETE=bind(RUN/'FIT_COMPLETE.json'),best_update=beststep,best_BCE=best)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['generated','freeze','run']);args=p.parse_args();{'generated':generated,'freeze':freeze,'run':run}[args.stage]()
