from context42 import *
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

def load_inputs(p,seed):
    a=inputs(p,seed,'pred_segment')['physical3'];v=dict(np.load(V41/'arrays'/f'seed{seed}.npz'));F=(a['X'].shape[-1]-15)//2;keep=np.ones(F,bool);keep[8:40]=False;keep[63:95]=False;mask=np.r_[keep,keep,np.ones(15,bool)];X=dict(visual=a['X'],geometry=a['X'][...,mask]);return a,v,X

def labels(p,a):
    inc=np.array(INC);g=p['g'];n=a['normal'];prior=a['prior'];median=np.array([[read(V35/'OFFSET_CALIBRATION.json')['models'][f'{j}_{r}']['median'] for r in inc[j]] for j in range(8)]);rho=(n*a['anchor'][:,:,None]).sum(-1)+(prior-median[None])*p['diag'][:,None,None];distance=np.abs((g['support'][:,inc]*n[...,None,:]).sum(-1)-rho[...,None]).mean(-1);angle=np.degrees(np.arccos(np.clip(np.abs((n*g['normal'][:,inc]).sum(-1)),0,1)));visible=g['length'][:,inc]>0;physical=visible&(distance<=3)&(angle<=10);residual=((n*g['gt'][:,:,None]).sum(-1)-rho)-median[None]*p['diag'][:,None,None];return dict(visible=visible,physical=physical,constraint=physical&(np.abs(residual)<=3))

def fit(X,y,valid,p,part,seed):
    take=part['fit'];mask=take[:,None,None]&valid;w=np.broadcast_to(frame_weights(p['groups'],take)[:,None,None],valid[take].shape)[valid[take]];w/=w.mean();m=HistGradientBoostingClassifier(random_state=42000+seed,**PARAMS);m.fit(X[mask],y[mask],sample_weight=w);return m

def probabilities(m,X):return m.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(X.shape[:-1])

def threshold(prob,y,valid,p,part):
    take=part['calibration'];mask=valid[take];w=np.broadcast_to(frame_weights(p['groups'],take)[:,None,None],mask.shape);rows=[]
    for th in np.arange(.1,.951,.05):
        sel=(prob[take]>=th)&mask;tp=float((w*sel*y[take]).sum());precision=tp/max(float((w*sel).sum()),1e-12);recall=tp/max(float((w*mask*y[take]).sum()),1e-12);groups=int(len(set(p['groups'][take][sel.any((1,2))])));rows.append(dict(threshold=float(round(th,2)),precision=precision,recall=recall,accepted=int(sel.sum()),groups=groups))
    ok=[r for r in rows if r['precision']>=.9 and r['accepted']>=10 and r['groups']>=2];th=max(ok,key=lambda r:(r['recall'],r['precision'],r['threshold']))['threshold'] if ok else 1.1;return th,rows

def route(p,a,v,allow,kind):
    fallback=v['points_pred_segment_physical3_direct'];prefix='pred_segment_physical3';am=a['amount']/p['diag'][:,None,None];gain=2*am*v['mu_'+prefix+'_residual']-am**2*(a['normal']**2).sum(-1)
    if kind.startswith('pre'):
        gain=np.where(allow&a['valid'],gain,-np.inf);k=gain.argmax(-1);sel=np.where(gain.max(-1)>0,k,-1)
        if kind=='pre_hard':sel=np.where(v['hard_prediction'],sel,-1)
    else:
        k=np.maximum(v['selected_'+prefix+'_residual_gate'],0);ok=allow[np.arange(p['N'])[:,None],np.arange(8)[None],k];sel=np.where(ok,v['selected_'+prefix+'_residual_gate'],-1)
    points=np.where((sel>=0)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],np.maximum(sel,0)],fallback);return points,sel

def main():
    register();p=population();part=partition(p)
    for name in ('models','arrays'): (RUN/name).mkdir(exist_ok=True)
    write(RUN/'SOURCE_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in (KIT/'context42.py',KIT/'experiment42.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')));count=0
    for seed in SEEDS:
        a,v,X=load_inputs(p,seed);Y=labels(p,a);out=dict(anchor=a['anchor'],fallback=v['points_pred_segment_physical3_direct'],hard=v['hard'],valid=a['valid']);models={};thresholds={}
        for target in TARGETS:
            out['label_'+target]=Y[target]
            for feature in FEATURES:
                key=target+'_'+feature;m=fit(X[feature],Y[target],a['valid'],p,part,seed);prob=probabilities(m,X[feature]);th,grid=threshold(prob,Y[target],a['valid'],p,part);models[key]=m;thresholds[key]=dict(threshold=th,grid=grid);out['prob_'+key]=prob;out['accepted_'+key]=(prob>=th)&a['valid'];count+=1
                for kind in ROUTES:
                    pts,sel=route(p,a,v,out['accepted_'+key],kind);out['points_'+key+'_'+kind]=pts;out['selected_'+key+'_'+kind]=sel
                print('FIT42',seed,key,'threshold',th,'models',count,flush=True)
            for kind in ROUTES:
                key='GT_'+target+'_'+kind;pts,sel=route(p,a,v,Y[target]&a['valid'],kind);out['points_'+key]=pts;out['selected_'+key]=sel
        # Keep fallback as an explicit member of the oracle candidate family.
        ce=np.linalg.norm(a['candidate']-p['g']['gt'][:,:,None],axis=-1);ce=np.where(a['valid'],ce,np.inf);k=ce.argmin(-1);be=np.linalg.norm(out['fallback']-p['g']['gt'],axis=-1);out['points_GT_candidate']=np.where((ce.min(-1)<be)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],k],out['fallback']);key='constraint_visual_pre_all';pts=out['points_'+key];safe=np.linalg.norm(pts-p['g']['gt'],axis=-1)<be;out['points_GT_safe_primary']=np.where(safe[...,None],pts,out['fallback'])
        (RUN/'models'/f'seed{seed}.pkl').write_bytes(pickle.dumps(models,protocol=4));write(RUN/'models'/f'seed{seed}.json',dict(complete=True,new_models=6,rounds=360,thresholds=thresholds));np.savez_compressed(RUN/'arrays'/f'seed{seed}.npz',**out)
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=count,boosting_rounds=count*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz'))]}));print('TRAIN42 COMPLETE',count,flush=True)
if __name__=='__main__':main()
