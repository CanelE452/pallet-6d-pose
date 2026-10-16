from context44 import *
L=legacy()
from analyze37 import features as oldfeatures,load_seed
from observations40 import extract,observed_roles,encode,canonical
from experiment41 import choose
import pickle

def population44():
 m=read(RUN/'cache/manifest.json');tt=read(RUN/'TARGETS.json')['rows'];rows=[r for r in m['rows'] if r['detected']];g=dict(np.load(RUN/'geometry.npz'));bb=np.array([r['box'] for r in rows]);wh=np.maximum(bb[:,2:]-bb[:,:2],1)
 return dict(rows=rows,g=g,N=len(rows),bb=bb,diag=np.linalg.norm(wh,axis=-1),center=(bb[:,:2]+bb[:,2:])/2,aspect=np.log(wh[:,0]/wh[:,1]),train=np.array([r['split']=='train' for r in rows]),calibration=np.array([r['split']=='calibration' for r in rows]),check=np.array([r['split']=='check' for r in rows]),groups=np.array([r['source_group'] for r in rows]),condition=np.array([r['condition'] for r in rows]))

def construct(p,z,f,raw):
 inc=np.array(L.INC);N=p['N'];line=z['both'];pn,pr,score=[line[k] for k in ('normal','rho','score')];ni=pn[:,inc].reshape(N,8,24,2);ri=pr[:,inc].reshape(N,8,24);si=score[:,inc].reshape(N,8,24);visual=z['visual'][:,inc].reshape(N,8,24,32);ir=np.repeat(inc,8,axis=1);offsets=read(L.V35/'OFFSET_CALIBRATION.json')['models'];median=np.array([[offsets[f'{j}_{r}']['median'] for r in ir[j]] for j in range(8)]);anchor=z['v34'];amount=np.clip(ri+median[None]*p['diag'][:,None,None]-(ni*anchor[:,:,None]).sum(-1),-.15*p['diag'][:,None,None],.15*p['diag'][:,None,None]);X=oldfeatures(p,z,anchor,ni,amount,si,ir,visual)
 ff=f[:,inc].reshape(N,8,24,48);loc=ff[...,43:45];spread=np.maximum(ff[...,45],1e-4);mass=ff[...,47];tangent=np.stack((-ni[...,1],ni[...,0]),-1);anchorn=(anchor-p['center'][:,None])/p['diag'][:,None,None];lever=anchorn[:,:,None]-loc;along=(lever*tangent).sum(-1);across=(lever*ni).sum(-1);w=score/np.maximum(score.sum(-1,keepdims=True),1e-12);mean=(pn*w[...,None]).sum(-2);angvar=(w*np.sum((pn-mean[:,:,None])**2,-1)).sum(-1);rr=(pr[:,:,None,:]-np.einsum('nrkd,njd->nrjk',pn,anchor))/p['diag'][:,None,None,None];rm=(rr*w[:,:,None,:]).sum(-1);disp=np.sqrt((w[:,:,None,:]*(rr-rm[...,None])**2).sum(-1));dis=np.stack([disp[:,inc[j],j].repeat(8,axis=-1) for j in range(8)],axis=1);angle=np.repeat(angvar[:,inc],8,axis=-1);extra=np.concatenate((loc,spread[...,None],mass[...,None],along[...,None],abs(along)[...,None],across[...,None],np.log1p(abs(along)/spread)[...,None],np.maximum(abs(along)-2*spread,0)[...,None],dis[...,None],angle[...,None]),-1);geo=np.concatenate((X['visual_geometry'],extra),-1).astype(np.float32)
 seg,valid,index=observed_roles(p,z,dict(support_center=loc),raw);segfeat=encode(seg,valid,anchor,ni,p['diag']);_,normal,_=canonical(seg);rho=(normal*seg[:,:,0]).sum(-1);n=normal[:,inc];ok=valid[:,inc];med=median[:,::8];prior=np.where(ok,(rho[:,inc]-(n*anchor[:,:,None]).sum(-1))/p['diag'][:,None,None]+med[None],0);d=np.clip(prior,-.15,.15)*p['diag'][:,None,None];candidate=anchor[:,:,None]+n*d[...,None];g=geo.reshape(N,8,3,8,-1);e=segfeat[...,::8,:].copy();e[...,4]=ok.astype(float);feature=np.concatenate((g.mean(3),g.std(3),e,prior[...,None]),-1).astype(np.float32)
 point=np.concatenate([*[(z[k]-p['center'][:,None])/p['diag'][:,None,None] for k in ('base','strong','v34')],z['query'],p['aspect'][:,None,None].repeat(8,1),np.broadcast_to(np.eye(8),(N,8,8))],-1).astype(np.float32);assert point.shape[-1]==47
 lineinput=np.concatenate((point,feature.mean(2),feature.std(2),ok.astype(np.float32)),-1).astype(np.float32)
 a=dict(X=feature,anchor=anchor,normal=n,amount=d,valid=ok,candidate=candidate);return a,dict(point=point,line=lineinput),dict(segments=seg,valid=valid,index=index)

def old_features(seed):
 p=L.population();z=load_seed(seed,p);f=np.zeros((p['N'],12,8,48));lookup={r['id']:i for i,r in enumerate(p['rows'])}
 import torch
 for path in [L.V36/'cache'/f'training_seed{seed}.pt']+sorted((L.V36/'cache/render_check').glob(f'*/seed{seed}.pt')):
  c=torch.load(path,map_location='cpu',weights_only=True)
  for i,name in enumerate(c['ids']):
   if name in lookup:f[lookup[name]]=c['x']['features'][i].numpy().reshape(12,8,48)
 raw=dict(np.load(L.V40/'segments.npz'));return p,z,f,construct(p,z,f,raw)

def parity():
 rows=[]
 for seed in SEEDS:
  p,z,f,(a,X,m)=old_features(seed);old=L.inputs(p,seed,'pred_segment')['physical3'];diff=float(abs(a['X']-old['X']).max());assert np.allclose(a['X'],old['X'],atol=2e-6,rtol=1e-5),diff
  b=pickle.loads((L.V41/'models'/f'seed{seed}.pkl').read_bytes())['pred_segment_physical3_direct'];mu=b.predict(a['X'].reshape(-1,a['X'].shape[-1])).reshape(a['X'].shape[:-1]);pts,_=choose(a,mu,p,'physical3');saved=np.load(L.V41/'arrays'/f'seed{seed}.npz')['points_pred_segment_physical3_direct'];assert np.allclose(pts,saved,atol=1e-8);rows.append(dict(seed=seed,feature_max=float(diff),point_max=float(abs(pts-saved).max())));np.savez_compressed(RUN/'features'/f'reused_seed{seed}.npz',**X,anchor=a['anchor'],gt=p['g']['gt'],fit=L.partition(p)['fit'],groups=p['groups'])
 write(RUN/'FEATURE_PARITY.json',dict(status='PASS',rows=rows));print('FEATURE PARITY44',rows,flush=True)

if __name__=='__main__':
 (RUN/'features').mkdir(exist_ok=True);parity()
