import numpy as np
import torch
from torch import nn

class Scorer(nn.Module):
    def __init__(self,d,linear=False):
        super().__init__();self.net=nn.Linear(d,1) if linear else nn.Sequential(nn.Linear(d,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def normalize(x,train):
    vals=x[train].reshape(-1,x.shape[-1]);mean=vals.mean(0);std=np.maximum(vals.std(0),1e-6)
    return (x-mean)/std,mean,std

def fit(x,y,train,val,linear=False,pairwise=True):
    torch.manual_seed(42);np.random.seed(42)
    model=Scorer(x.shape[-1],linear).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    xx=torch.tensor(x,device='cuda');yy=torch.tensor(y,dtype=torch.float32,device='cuda');rng=np.random.default_rng(42);best=-1;state=None;stale=0;log=[]
    def logit(v):
        score=model(v);return score[:,0]-score[:,1] if pairwise else score
    for epoch in range(30):
        model.train();losses=[]
        for idx in np.array_split(rng.permutation(np.flatnonzero(train)),max(1,int(np.ceil(train.sum()/256)))):
            opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(logit(xx[idx]),yy[idx]);loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():prob=torch.sigmoid(logit(xx[val]));acc=float(((prob>.5)==yy[val].bool()).float().mean())
        log.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_accuracy=acc))
        print('SMALL_FIT',('PAIR' if pairwise else 'ROUTER'),('LINEAR' if linear else 'MLP'),epoch+1,acc,flush=True)
        if acc>best:
            best=acc;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0;best_epoch=epoch+1
        else:stale+=1
        if stale>=5:break
    model.load_state_dict(state);return state,dict(best_val_accuracy=best,best_epoch=best_epoch,epochs=len(log),curve=log)

def pack_inputs(z,arm,variant):
    geo=z[arm+'_geo']
    if variant=='GEO_LINEAR':return geo
    context=np.broadcast_to(z[arm+'_ctx'][:,None,:],(*geo.shape[:2],z[arm+'_ctx'].shape[-1]))
    return np.concatenate([geo,context],-1)

def scores(ck,x):
    model=Scorer(ck['d'],ck.get('variant')=='GEO_LINEAR');model.load_state_dict(ck['state']);model.eval()
    norm=(x-ck['mean'])/ck['std']
    with torch.no_grad():return model(torch.tensor(norm,dtype=torch.float32)).numpy()

def selection(score,names):
    # Explicit name tie breaker, invariant to storage order.
    return np.array([min(range(len(row)),key=lambda j:(float(row[j]),names[j])) for row in score])
