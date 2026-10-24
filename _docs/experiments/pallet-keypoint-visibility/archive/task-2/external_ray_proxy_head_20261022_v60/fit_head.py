from common import *
import os,time,signal,random
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np,torch
from torch.nn import functional as F
from head_and_roi import ProxyHead
from metrics import classification
torch.set_num_threads(1);torch.manual_seed(20261018);torch.cuda.manual_seed_all(20261018);np.random.seed(20261018);random.seed(20261018);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available();assert(ROOT/'SYNTH_FEATURES_COMPLETE.json').exists();assert not(ROOT/'FIT_STARTED.json').exists();signal.alarm(300)
save('FIT_STARTED.json',dict(UTC=utc(),fit_attempt=1,max_updates=480,protocol_sha256=sha(ROOT/'PROTOCOL.json')));cache=np.load(ROOT/'SYNTH_ROI_CACHE.npz');labels=np.load(ROOT/'SYNTH_LABELS.npz');x=torch.from_numpy(cache['x']);valid=labels['valid']&cache['pixel_valid'];y=labels['y'];tr=labels['train'];va=labels['validation'];net=ProxyHead().cuda();assert sum(p.numel()for p in net.parameters())==28033;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);start=time.monotonic();update=0;history=[];best=float('inf');best_epoch=None
def logits_for(indices):
 out=[];net.eval()
 with torch.no_grad():
  for j in range(0,len(indices),32):
   ix=indices[j:j+32];z=net(x[ix].reshape(-1,65,9,9).cuda().float()).reshape(len(ix),8);out.append(z.cpu().numpy())
 return np.concatenate(out)
for epoch in range(1,21):
 net.train();permutation=tr[np.random.permutation(len(tr))];losses=[]
 for j in range(0,len(tr),32):
  ix=permutation[j:j+32];xx=x[ix].reshape(-1,65,9,9).cuda().float();yy=torch.tensor(y[ix].reshape(-1),device='cuda');mask=torch.tensor(valid[ix].reshape(-1),device='cuda');assert mask.any();opt.zero_grad(set_to_none=True);logits=net(xx);loss=F.binary_cross_entropy_with_logits(logits[mask],yy[mask]);assert torch.isfinite(loss);loss.backward();opt.step();update+=1;losses.append(float(loss.detach().cpu()))
  if update==1:save('FIRST_OPTIMIZER_UPDATE.json',dict(UTC=utc(),update=1,loss=losses[-1],head_parameters_changed=True,backbone_not_in_optimizer=True));print('FIRST_OPTIMIZER_UPDATE',update,losses[-1],utc(),flush=True)
  if update==20:torch.cuda.synchronize();elapsed=time.monotonic()-start;print('UPDATE_PROGRESS',update,480,'train_elapsed_s',round(elapsed,3),'step_only_remaining_s',round((480-update)*elapsed/update,2),utc(),flush=True)
 v=logits_for(va);v_mask=valid[va];val_loss=float(np.mean(np.logaddexp(0,v[v_mask])-y[va][v_mask]*v[v_mask]));assert np.isfinite(val_loss);saved=False
 if val_loss<best:
  best=val_loss;best_epoch=epoch;torch.save({k:t.detach().cpu()for k,t in net.state_dict().items()},ROOT/'BEST_HEAD.pt');saved=True
 history.append(dict(epoch=epoch,update=update,train_batch_BCE_mean=float(np.mean(losses)),VAL_BCE=val_loss,strictly_lower_checkpoint=saved,elapsed_s=time.monotonic()-start));print('EPOCH',epoch,'updates',update,'VAL_BCE',round(val_loss,6),'best',best_epoch,'elapsed_s',round(time.monotonic()-start,3),'estimated_remaining_s',round((20-epoch)*(time.monotonic()-start)/epoch,2),utc(),flush=True)
assert update==480;net.load_state_dict(torch.load(ROOT/'BEST_HEAD.pt',map_location='cuda'));out=np.empty((len(x),8),np.float32);out[tr]=logits_for(tr);out[va]=logits_for(va);prob=1/(1+np.exp(-out));np.savez_compressed(ROOT/'SYNTH_HEAD_OUTPUTS.npz',logits=out,probability=prob,pixel_valid=cache['pixel_valid'],loss_valid=valid);inputs=read(ROOT/'PREPARED_INPUTS_SYNTH.json.gz');qr=[]
for i,u in enumerate(inputs):
 for k in range(8):qr.append(dict(id=u['id'],split=u['split'],native_corner=k,blocked_probability=float(prob[i,k]),target=int(y[i,k]),loss_valid=bool(valid[i,k]),pixel_valid=bool(cache['pixel_valid'][i,k]),coordinates_changed=False))
csvsave('SYNTH_QUERY_OUTPUTS.csv',qr);csvsave('TRAINING_HISTORY.csv',history);prior=float(y[tr][valid[tr]].mean());metrics=[]
for name,indices in [('TRAIN',tr),('VAL',va)]:
 mask=valid[indices];yy=y[indices][mask]
 for arm,pp in [('HEAD',prob[indices][mask]),('ALWAYS_NOT_BLOCKED',np.zeros(len(yy))),('TRAIN_PRIOR',np.full(len(yy),prior))]:metrics.append(dict(split=name,arm=arm,**classification(yy,pp)))
csvsave('CLASSIFICATION_METRICS.csv',metrics);save('FIT_COMPLETE.json',dict(UTC=utc(),fit_attempts=1,optimizer_updates=update,epochs=20,best_epoch=best_epoch,best_VAL_BCE=best,seconds=time.monotonic()-start,parameters=28033,head_sha256=sha(ROOT/'BEST_HEAD.pt'),no_real_rows=True,VALID_TRAIN_queries=int(valid[tr].sum()),VALID_VAL_queries=int(valid[va].sum()),TRAIN_positive=int(y[tr][valid[tr]].sum()),VAL_positive=int(y[va][valid[va]].sum()),full_source_pixel_abstain_queries=int((~cache['pixel_valid']).sum()),features_seconds=read(ROOT/'SYNTH_FEATURES_COMPLETE.json')['seconds']));save('MODEL_FREEZE.json',dict(UTC=utc(),checkpoint_sha256=sha(ROOT/'BEST_HEAD.pt'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),crop_and_threshold_frozen=True,threshold=.5,min_native_points=6,source_scope='EXTERNAL_RAY_BLOCKAGE_PROXY_ONLY not physical DIRECT',no_location_branch=True,real_selection_allowed=False));print('FIT_COMPLETE_FROZEN',update,'best_epoch',best_epoch,'seconds',round(time.monotonic()-start,3),flush=True)
