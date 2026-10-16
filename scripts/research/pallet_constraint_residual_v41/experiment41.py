from context41 import *
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle

def inputs(p,seed,source):
    a=dict(np.load(V37/'arrays'/f'v34_both_seed{seed}.npz'));f=dict(np.load(V40/'features'/f'seed{seed}.npz'));m=dict(np.load(V40/'arrays'/f'matching_seed{seed}.npz'));v=dict(np.load(V40/'arrays'/f'seed{seed}.npz'));inc=np.array(INC);offsets=read(V35/'OFFSET_CALIBRATION.json')['models'];median=np.array([[offsets[f'{j}_{r}']['median'] for r in inc[j]] for j in range(8)])
    if source=='GT_line':segments=p['g']['support'];valid=p['g']['length']>0
    else:segments=m[source];valid=m['pred_valid' if source=='pred_segment' else 'GT_match_valid']
    _,nr,_=canonical(segments);rho=(nr*segments[:,:,0]).sum(-1);n=nr[:,inc];ok=valid[:,inc];amount=(rho[:,inc]-(n*a['anchor'][:,:,None]).sum(-1))/p['diag'][:,None,None]+median[None];amount=np.where(ok,amount,0);d=np.clip(amount,-.15,.15)*p['diag'][:,None,None];candidate=a['anchor'][:,:,None]+n*d[...,None];n24=np.repeat(n,8,axis=2);valid24=np.repeat(ok,8,axis=2);prior24=np.repeat(amount,8,axis=2)*(n24*a['candidate_normal']).sum(-1)
    feature24=np.concatenate((f['geometry'],f[source]),-1);geo=f['geometry'].reshape(p['N'],8,3,8,-1);extra=f[source][...,::8,:].copy();extra[...,4]=ok.astype(float);feature3=np.concatenate((geo.mean(3),geo.std(3),extra,amount[...,None]),-1).astype(np.float32)
    return dict(legacy24=dict(X=feature24,prior=prior24,valid=valid24,normal=a['candidate_normal'],candidate=a['candidate'],amount=a['candidate_amount'],anchor=a['anchor'],direct=v['mu_'+source+'_all']),physical3=dict(X=feature3,prior=amount,valid=ok,normal=n,candidate=candidate,amount=d,anchor=a['anchor']))

def fit(X,y,valid,take,groups,seed):
    mask=np.broadcast_to(take[:,None,None],y.shape)&valid;weights=np.broadcast_to(frame_weights(groups,take)[:,None,None],y[take].shape)[valid[take]];weights/=weights.mean();m=HistGradientBoostingRegressor(loss='absolute_error',random_state=seed,**PARAMS);m.fit(X[mask],y[mask],sample_weight=weights);return m

def choose(a,mu,p,family):
    am=a['amount']/p['diag'][:,None,None];gain=2*am*mu-am**2*(a['normal']**2).sum(-1)
    if family=='physical3':gain=np.where(a['valid'],gain,-np.inf)
    index=gain.argmax(-1);selected=np.where(gain.max(-1)>0,index,-1);points=np.where((selected>=0)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],index],a['anchor']);return points,selected

def main():
    register();p=population();part=partition(p)
    for name in ('models','arrays'): (RUN/name).mkdir(exist_ok=True)
    write(RUN/'SOURCE_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in (KIT/'context41.py',KIT/'experiment41.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')));count=0
    for seed in SEEDS:
        old=dict(np.load(V39/'arrays'/f'both_seed{seed}.npz'));out=dict(anchor=old['anchor'],hard=old['hard'],hard_prediction=old['hard_prediction_line']);bundle={}
        for source in SOURCES:
            data=inputs(p,seed,source)
            for family,a in data.items():
                prefix=source+'_'+family;y=(a['normal']*(p['g']['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];valid=a['valid'] if family=='physical3' else np.ones_like(a['valid']);out.update({prefix+'_target':y,prefix+'_prior':a['prior'],prefix+'_valid':a['valid'],prefix+'_candidate':a['candidate'],prefix+'_normal':a['normal'],prefix+'_amount':a['amount']})
                mus={}
                if family=='legacy24':mus['direct']=a['direct'];mus['analytic']=np.where(a['valid'],a['prior'],old['mu_standard'])
                else:
                    m=fit(a['X'],y,valid,part['fit'],p['groups'],41000+seed);bundle[prefix+'_direct']=m;mus['direct']=pred(m,a['X']);count+=1;mus['analytic']=a['prior']
                m=fit(a['X'],y-a['prior'],valid,part['fit'],p['groups'],41000+seed);bundle[prefix+'_residual']=m;mus['residual']=a['prior']+pred(m,a['X']);count+=1
                for mode,mu in mus.items():
                    key=prefix+'_'+mode;out['mu_'+key]=mu;pt,sel=choose(a,mu,p,family);out['points_'+key]=pt;out['selected_'+key]=sel;out['points_'+key+'_gate']=np.where(old['hard_prediction_line'][...,None],pt,a['anchor']);out['selected_'+key+'_gate']=np.where(old['hard_prediction_line'],sel,-1);safe=np.linalg.norm(pt-p['g']['gt'],axis=-1)<np.linalg.norm(a['anchor']-p['g']['gt'],axis=-1);out['points_'+key+'_GT_safe']=np.where(safe[...,None],pt,a['anchor'])
                ce=np.linalg.norm(a['candidate']-p['g']['gt'][:,:,None],axis=-1);ce=np.where(a['valid'] if family=='physical3' else True,ce,np.inf);best=ce.argmin(-1);base=np.linalg.norm(a['anchor']-p['g']['gt'],axis=-1);out['points_'+prefix+'_GT_candidate']=np.where((ce.min(-1)<base)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],best],a['anchor']);print('FIT41',seed,source,family,'total',count,flush=True)
        (RUN/'models'/f'seed{seed}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));np.savez_compressed(RUN/'arrays'/f'seed{seed}.npz',**out);write(RUN/'models'/f'seed{seed}.json',dict(complete=True,new_models=9,rounds=540))
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=count,boosting_rounds=count*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz'))]}));print('TRAIN41 COMPLETE',count,flush=True)
if __name__=='__main__':main()
