from context38 import *
import torch,pickle
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr


def frame_weights(groups,take):
    names,counts=np.unique(groups[take],return_counts=True);lookup=dict(zip(names,counts));w=np.array([1/lookup[g] for g in groups[take]]);return w/w.mean()

def wquantile(values,groups,take,q):
    v=values[take].reshape(-1);w=np.repeat(frame_weights(groups,take),int(np.prod(values.shape[1:])));order=np.argsort(v,kind='stable');v=v[order];w=w[order];return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum()),len(v)-1)])

def fit_model(X,y,take,groups,seed):
    xx=X[take].reshape(-1,X.shape[-1]);yy=y[take].reshape(-1);w=np.repeat(frame_weights(groups,take),y.shape[1]*y.shape[2]);m=HistGradientBoostingRegressor(loss='absolute_error',random_state=seed,**PARAMS);m.fit(xx,yy,sample_weight=w);return m

def pred(m,X):return m.predict(X.reshape(-1,X.shape[-1])).reshape(X.shape[:-1])

def observable(p,z,seed,arm):
    a=dict(np.load(V37/'arrays'/f'v34_{arm}_seed{seed}.npz'));N=p['N'];inc=np.array(INC);ir=np.repeat(inc,8,axis=1);visual=z['visual'][:,inc].reshape(N,8,24,32);X=features(p,z,a['anchor'],a['candidate_normal'],a['candidate_amount'],a['candidate_score'],ir,visual)
    lookup={r['id']:i for i,r in enumerate(p['rows'])};f=np.zeros((N,12,8,48));seen=np.zeros(N,int)
    for path in [V36/'cache'/f'training_seed{seed}.pt']+sorted((V36/'cache/render_check').glob(f'*/seed{seed}.pt')):
        c=torch.load(path,map_location='cpu',weights_only=True)
        for i,name in enumerate(c['ids']):
            if name in lookup:
                j=lookup[name];f[j]=c['x']['features'][i].numpy().reshape(12,8,48);seen[j]+=1
    assert np.all(seen==1)
    # Proxy support center/spread are from frozen bin sampling, not GTvisible endpoints.
    ff=f[:,inc].reshape(N,8,24,48);loc=ff[...,43:45];spread=np.maximum(ff[...,45],1e-4);mass=ff[...,47];normal=a['candidate_normal'];tangent=np.stack((-normal[...,1],normal[...,0]),-1);anchor=(a['anchor']-p['center'][:,None])/p['diag'][:,None,None];lever=(anchor[:,:,None]-loc);along=(lever*tangent).sum(-1);across=(lever*normal).sum(-1)
    pn=z[arm]['normal'];ps=z[arm]['score'];w=ps/np.maximum(ps.sum(-1,keepdims=True),1e-12);mean=(pn*w[...,None]).sum(-2);angvar=(w*np.sum((pn-mean[:,:,None])**2,-1)).sum(-1);rho=z[arm]['rho'];r=(rho[:,:,None,:]-np.einsum('nrkd,njd->nrjk',pn,a['anchor']))/p['diag'][:,None,None,None];rm=(r*w[:,:,None,:]).sum(-1);disp=np.sqrt((w[:,:,None,:]*(r-rm[...,None])**2).sum(-1));dis=np.stack([disp[:,inc[j],j].repeat(8,axis=-1) for j in range(8)],axis=1);angle=np.repeat(angvar[:,inc],8,axis=-1)
    extra=np.concatenate((loc,spread[...,None],mass[...,None],along[...,None],np.abs(along)[...,None],across[...,None],np.log1p(np.abs(along)/spread)[...,None],np.maximum(np.abs(along)-2*spread,0)[...,None],dis[...,None],angle[...,None]),-1)
    return a,dict(appearance=X['visual'].astype(np.float32),geometry=np.concatenate((X['visual_geometry'],extra),-1).astype(np.float32)),dict(support_center=loc,support_spread=spread,along=along,disagreement=dis)


def select(a,mu,width,p):
    am=a['candidate_amount']/p['diag'][:,None,None];norm2=(a['candidate_normal']**2).sum(-1);lower=2*am*mu-am**2*norm2-2*np.abs(am)*width;k=lower.argmax(-1);selected=np.where(lower.max(-1)>0,k,-1);points=np.where((selected>=0)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],k],a['anchor']);return points,selected,lower


def main():
    register();p=population();part=partition(p);(RUN/'models').mkdir(exist_ok=True);(RUN/'arrays').mkdir(exist_ok=True);fitcount=0;calibration={}
    write(RUN/'SOURCE_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in KIT.glob('*.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        name=f'{arm}_seed{seed}';a,X,support=observable(p,z,seed,arm);target=(a['candidate_normal']*(p['g']['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];oof=np.full_like(target,np.nan);fold_models=[]
        for k,groups in enumerate(part['fold_groups']):
            hold=np.isin(p['groups'],groups);take=part['fit']&~hold;m=fit_model(X['geometry'],target,take,p['groups'],38000+seed+k);oof[hold]=pred(m,X['geometry'][hold]);fold_models.append(m);fitcount+=1;print('FIT38',name,'mean fold',k,'total',fitcount,flush=True)
        assert np.isfinite(oof[part['fit']]).all();m=fit_model(X['geometry'],target,part['fit'],p['groups'],38000+seed);mu=pred(m,X['geometry']);fitcount+=1;residual=np.abs(target-oof);logtarget=np.log(residual+1e-4);lo=max(wquantile(residual,p['groups'],part['fit'],.01),1e-4);hi=max(wquantile(residual,p['groups'],part['fit'],.99),lo);scales={'constant':np.full_like(target,wquantile(residual,p['groups'],part['fit'],.9))};scale_models={}
        for mode in ('appearance','geometry'):
            model=fit_model(X[mode],logtarget,part['fit'],p['groups'],38100+seed);scale_models[mode]=model;scales[mode]=np.clip(np.exp(np.clip(pred(model,X[mode]),-20,5)),lo,hi);fitcount+=1;print('FIT38',name,'scale',mode,'total',fitcount,flush=True)
        outputs=dict(anchor=a['anchor'],candidate=a['candidate'],candidate_normal=a['candidate_normal'],candidate_amount=a['candidate_amount'],target=target,mu=mu,oof=oof,**support);cal={};points,sel,lower=select(a,mu,np.zeros_like(target),p);outputs.update(points_mean_only=points,selected_mean_only=sel,lower_mean_only=lower)
        for mode in SCALES:
            scale=scales[mode];cal[mode]={};outputs['scale_'+mode]=scale;ratio=np.abs(target-mu)/np.maximum(scale,1e-8)
            for family,qdata in [('marginal',ratio),('point',ratio.max(-1))]:
                q=wquantile(qdata,p['groups'],part['calibration'],.9);width=q*scale;method=mode+'_'+family;points,sel,lower=select(a,mu,width,p);outputs.update({'width_'+method:width,'points_'+method:points,'selected_'+method:sel,'lower_'+method:lower});cal[mode][family]=q
        bundle=dict(mean=m,fold_means=fold_models,scales=scale_models,lo=lo,hi=hi,calibration=cal,fit_groups=part['fit_groups'],calibration_groups=part['calibration_groups'],feature_shapes={k:v.shape[-1] for k,v in X.items()});(RUN/'models'/f'{name}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(RUN/'arrays'/f'{name}.npz',**outputs);calibration[name]=dict(multiplier=cal,clip=[lo,hi],features=bundle['feature_shapes']);write(RUN/'models'/f'{name}.json',dict(complete=True,hgb_fits=6,boosting_rounds=6*60,fit_groups=part['fit_groups'],calibration_groups=part['calibration_groups']));print('DONE38',name,flush=True)
    write(RUN/'CALIBRATION.json',dict(fit_groups=part['fit_groups'],calibration_groups=part['calibration_groups'],check_groups=part['check_groups'],models=calibration));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=fitcount,boosting_rounds=fitcount*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [RUN/'CALIBRATION.json',*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz'))]}));print('TRAIN38 COMPLETE',fitcount,flush=True)

if __name__=='__main__':main()
