from features44 import *
sys.path.insert(0,str(KIT.parent/'pallet_continuous_lines_v36'))
from model36 import Model
from context36 import setup36
from extract35 import physical
import torch

def main():
 setup36();manifest=read(RUN/'cache/manifest.json');rows=[r for r in manifest['rows'] if r['detected']];targets=[t for r,t in zip(manifest['rows'],read(RUN/'TARGETS.json')['rows']) if r['detected']];arr=[physical(r) for r in rows];n,rho,length,seg,support,off=[np.array(x) for x in zip(*arr)];np.savez_compressed(RUN/'geometry.npz',normal=n,rho=rho,length=length,segment=seg,support=support,offset3d=off,gt=np.array([t['points'] for t in targets]),outside=np.array([t['outside'] for t in targets]),hidden=np.array([t['hidden'] for t in targets]));p=population44();raw=extract(p);np.savez_compressed(RUN/'segments.npz',**raw)
 chunks=[torch.load(RUN/'cache'/s,weights_only=True,map_location='cpu') for s in manifest['chunks']];O={k:torch.cat([c[k] for c in chunks]) for k in ('features','content','edge','box','image_hw')};(RUN/'arrays').mkdir(exist_ok=True);(RUN/'features').mkdir(exist_ok=True)
 for seed in SEEDS:
  path=RUN/'features'/f'fresh_seed{seed}.npz'
  if path.exists():continue
  model=Model(seed).cuda().eval();ck=torch.load(L.V36/'models'/f'both_seed{seed}'/'final.pt',weights_only=True,map_location='cpu');model.load_state_dict(ck['state_dict']);S=torch.load(RUN/'parents'/f'seed{seed}.pt',weights_only=True,map_location='cpu');S={k:S[k] for k in ('base','strong')};xparts=[];outparts=[]
  with torch.no_grad():
   for i in range(0,p['N'],32):
    oo={k:v[i:i+32].cuda() for k,v in O.items()};ss={k:v[i:i+32].cuda() for k,v in S.items()};x=model.core.capture(oo,ss);out=model(x,'both');xparts.append({k:v.cpu() for k,v in x.items()});outparts.append({k:out[k].cpu().numpy() for k in ('normal','offset','score','points')})
  xx={k:torch.cat([c[k] for c in xparts]) for k in xparts[0]};out={k:np.concatenate([c[k] for c in outparts]) for k in outparts[0]};pn=out['normal'].reshape(-1,12,8,2).astype(float);pr=out['offset'].reshape(-1,12,8).astype(float)*p['diag'][:,None,None]+(pn*p['center'][:,None,None]).sum(-1);sign=np.where(pn[...,1]<0,-1.,1.);pn*=sign[...,None];pr*=sign;f=xx['features'].numpy().reshape(-1,12,8,48).astype(float);z={k:xx[k].numpy() for k in ('base','strong','v34')};z.update(query=xx['query'].numpy().reshape(-1,8,2,32).mean(2).astype(float),visual=f[...,:32],both=dict(normal=pn,rho=pr,score=out['score'].reshape(-1,12,8).astype(float),points=out['points']));a,X,matched=construct({k:v for k,v in p.items() if k!='g'},z,f,raw);bundle=pickle.loads((L.V41/'models'/f'seed{seed}.pkl').read_bytes());m=bundle['pred_segment_physical3_direct'];mu=m.predict(a['X'].reshape(-1,a['X'].shape[-1])).reshape(a['X'].shape[:-1]);pts,selected=choose(a,mu,p,'physical3')
  np.savez_compressed(path,**X,anchor=z['v34'],gt=p['g']['gt'],fit=p['train'],calibration=p['calibration'],check=p['check'],groups=p['groups']);np.savez_compressed(RUN/'arrays'/f'parents_seed{seed}.npz',base=z['base'],strong=z['strong'],anchor=z['v34'],frozen_v41=pts,selected_frozen=selected,normal=pn,rho=pr,score=z['both']['score'],physical_features=a['X'],candidate=a['candidate'],valid=a['valid'],**{'match_'+k:v for k,v in matched.items()});print('INFER44',seed,p['N'],flush=True);del model
 write(RUN/'INFERENCE_COMPLETE.json',dict(complete=True,GT_forward=False,GT_absent_feature_construction=True,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'features').glob('*.npz')),*sorted((RUN/'arrays').glob('*.npz')),RUN/'geometry.npz',RUN/'segments.npz',KIT/'infer44.py']}))
if __name__=='__main__':main()
