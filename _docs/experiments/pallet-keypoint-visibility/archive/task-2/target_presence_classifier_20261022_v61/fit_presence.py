"""One actual synthetic fit to center-pixel M4 presence, never ray labels or pose."""
from pathlib import Path
import os,sys,time,random,json,csv
import numpy as np,torch
from torch.nn import functional as F
R=Path(__file__).resolve().parent;OLD=R.parent/'external_ray_proxy_head_20261022_v60';sys.path.insert(0,str(OLD))
from common import sha,read,utc
from head_and_roi import ProxyHead
from metrics import classification
def save(n,x):
 with(R/n).open('x')as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def csvsave(n,rr):
 with(R/n).open('x')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
assert not(R/'FIT_STARTED.json').exists();assert read(R/'LABEL_PREPARATION.json')['PASS'];assert sha(OLD/'SYNTH_ROI_CACHE.npz')==read(R/'LABEL_PREPARATION.json')['ROI_cache_sha256'];torch.set_num_threads(1);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018);random.seed(20261018);np.random.seed(20261018);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
save('FIT_STARTED.json',dict(UTC=utc(),fit_attempt=1,target='M4 predicted-query presence',protocol_sha256=sha(R/'PROTOCOL.json'),max_updates=480,old_ray_head_loaded=False));cache=np.load(OLD/'SYNTH_ROI_CACHE.npz');d=np.load(R/'PRESENCE_LABELS.npz');x=torch.from_numpy(cache['x']);y=d['y'];valid=d['valid'];tr=d['train'];va=d['validation'];model=ProxyHead().cuda();assert sum(p.numel()for p in model.parameters())==28033;opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');best_epoch=None;history=[];update=0;start=time.monotonic()
def forward(ix):
 out=[];model.eval()
 with torch.no_grad():
  for j in range(0,len(ix),32):out.append(model(x[ix[j:j+32]].reshape(-1,65,9,9).cuda().float()).reshape(-1,8).cpu().numpy())
 return np.concatenate(out)
for epoch in range(1,21):
 model.train();perm=tr[np.random.permutation(len(tr))];losses=[]
 for j in range(0,len(perm),32):
  ix=perm[j:j+32];xx=x[ix].reshape(-1,65,9,9).cuda().float();yy=torch.tensor(y[ix].reshape(-1),device='cuda');mask=torch.tensor(valid[ix].reshape(-1),device='cuda');opt.zero_grad(set_to_none=True);z=model(xx);loss=F.binary_cross_entropy_with_logits(z[mask],yy[mask]);assert torch.isfinite(loss);loss.backward();opt.step();update+=1;losses.append(float(loss.detach().cpu()))
  if update==1:save('FIRST_OPTIMIZER_UPDATE.json',dict(UTC=utc(),update=1,loss=losses[-1],target='visible target pixel presence',backbone_not_in_optimizer=True));print('FIRST_PRESENCE_OPTIMIZER_UPDATE',update,losses[-1],utc(),flush=True)
  if update==20:elapsed=time.monotonic()-start;print('PRESENCE_TRAIN_PROGRESS',20,480,'elapsed_s',round(elapsed,3),'estimated_step_remaining_s',round(elapsed*460/20,3),utc(),flush=True)
 z=forward(va);m=valid[va];v=float(np.mean(np.logaddexp(0,z[m])-y[va][m]*z[m]));saved=v<best
 if saved:best=v;best_epoch=epoch;torch.save({k:t.detach().cpu()for k,t in model.state_dict().items()},R/'BEST_PRESENCE_HEAD.pt')
 history.append(dict(epoch=epoch,updates=update,train_batch_BCE_mean=float(np.mean(losses)),VAL_BCE=v,saved_best=saved,elapsed_s=time.monotonic()-start));print('PRESENCE_EPOCH',epoch,'update',update,'VAL_BCE',round(v,6),'best',best_epoch,'elapsed_s',round(time.monotonic()-start,3),flush=True)
assert update==480;model.load_state_dict(torch.load(R/'BEST_PRESENCE_HEAD.pt',map_location='cuda'));zz=np.empty((len(x),8),np.float32);zz[tr]=forward(tr);zz[va]=forward(va);p=1/(1+np.exp(-zz));np.savez_compressed(R/'SYNTH_PRESENCE_OUTPUTS.npz',logits=zz,probability=p,valid=valid);prior=float(y[tr][valid[tr]].mean());rr=[]
for split,ix in [('TRAIN',tr),('VAL',va)]:
 yy=y[ix][valid[ix]]
 for arm,pp in [('PRESENCE_HEAD',p[ix][valid[ix]]),('TRAIN_PRIOR',np.full(len(yy),prior)),('ALWAYS_ABSENT',np.zeros(len(yy))),('ALWAYS_PRESENT',np.ones(len(yy)))]:
  s=classification(yy,pp);s['balanced_accuracy']=.5*(s['recall']+1-s['FPR'])if s['recall']is not None and s['FPR']is not None else None;rr.append(dict(split=split,arm=arm,**s))
csvsave('CLASSIFICATION_METRICS.csv',rr);csvsave('TRAINING_HISTORY.csv',history);head=next(r for r in rr if r['split']=='VAL'and r['arm']=='PRESENCE_HEAD');constant=next(r for r in rr if r['split']=='VAL'and r['arm']=='TRAIN_PRIOR');gate=dict(BCE_below_TRAIN_prior=head['BCE']<constant['BCE'],AP_above_VAL_prevalence=head['AUPRC']>head['prevalence'],positive_recall=head['recall']>0);save('MODEL_FREEZE.json',dict(UTC=utc(),checkpoint_sha256=sha(R/'BEST_PRESENCE_HEAD.pt'),protocol_sha256=sha(R/'PROTOCOL.json'),best_epoch=best_epoch,selection='lowest synthetic VAL BCE',threshold=.5,target='predicted query center-pixel M4 visible target presence',pose_gate=False,real_fitting_allowed=False));save('FIT_COMPLETE.json',dict(UTC=utc(),fit_attempts=1,optimizer_updates=update,parameters=28033,seconds=time.monotonic()-start,best_epoch=best_epoch,best_VAL_BCE=best,screen=gate,PASS=all(gate.values()),head_sha256=sha(R/'BEST_PRESENCE_HEAD.pt'),real_fit_updates=0,new_backbone_forward=0));print('PRESENCE_FIT_COMPLETE',update,'epoch',best_epoch,head,gate,flush=True)
