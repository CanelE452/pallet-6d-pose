from context43 import *
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

def observed(p,seed):
    a,v,X=load_inputs(p,seed);fallback=v['points_pred_segment_physical3_direct'];delta=(a['candidate']-fallback[:,:,None])/p['diag'][:,None,None,None];extra=np.concatenate([delta,np.linalg.norm(delta,axis=-1)[...,None]],-1)
    return a,v,{k:np.concatenate([x,extra],-1) for k,x in X.items()}

def truth(p,a,v):
    valid=labels(p,a)['constraint'];be=np.linalg.norm(v['points_pred_segment_physical3_direct']-p['g']['gt'],axis=-1);ce=np.linalg.norm(a['candidate']-p['g']['gt'][:,:,None],axis=-1);gain=be[:,:,None]-ce
    return dict(valid=valid,useful=valid&(gain>1),gain=gain,base_error=be,candidate_error=ce)

def fit43(X,Y,valid,p,part,seed,arm):
    target,weight,feature=arm.split('_');mask=part['fit'][:,None,None]&valid;w=np.broadcast_to(frame_weights(p['groups'],part['fit'])[:,None,None],valid[part['fit']].shape)[valid[part['fit']]]
    if weight=='weighted':w=w*np.where(Y['useful'][mask],4.,1.)
    w/=w.mean();m=HistGradientBoostingClassifier(random_state=43000+seed,**PARAMS);m.fit(X[feature][mask],Y[target][mask],sample_weight=w);return m

def route43(p,a,v,prob,th,kind):
    allow=(prob>=th)&a['valid'];fallback=v['points_pred_segment_physical3_direct']
    if kind=='residual':
        am=a['amount']/p['diag'][:,None,None];score=2*am*v['mu_pred_segment_physical3_residual']-am**2*(a['normal']**2).sum(-1);allow=allow&(score>0)
    else:score=prob
    scores=np.where(allow,score,-np.inf);k=scores.argmax(-1);sel=np.where(allow.any(-1),k,-1);pts=np.where((sel>=0)[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],k],fallback)
    return pts,sel,allow

def calibrate(p,part,a,v,Y,prob,kind):
    take=part['calibration'];w=frame_weights(p['groups'],take)[:,None];rows=[]
    for th in np.round(np.arange(.1,.951,.05),2):
        pts,sel,allow=route43(p,a,v,prob,th,kind);idx=np.maximum(sel,0);truthsel=Y['valid'][np.arange(p['N'])[:,None],np.arange(8)[None],idx];used=take[:,None]&(sel>=0);sw=np.linalg.norm(pts-v['points_pred_segment_physical3_direct'],axis=-1)>1e-8;err=np.linalg.norm(pts-p['g']['gt'],axis=-1);gain=float(((Y['base_error'][take]-err[take])*w).sum()/(8*w.sum()));precision=float(truthsel[used].mean()) if used.any() else 0.;groups=len(set(p['groups'][(used&sw).any(-1)]));preserve=all(err[take&(p['condition']==c)].mean()<=1.01*Y['base_error'][take&(p['condition']==c)].mean() for c in sorted(set(p['condition'][take])))
        changed=int((used&sw).sum());admissible=precision>=.9 and changed>=10 and groups>=2 and preserve and gain>0
        rows.append(dict(threshold=float(th),gain_px=gain,precision=precision,switches=changed,groups=groups,condition_preserved=bool(preserve),admissible=bool(admissible)))
    ok=[r for r in rows if r['admissible']];th=max(ok,key=lambda q:(q['gain_px'],q['threshold']))['threshold'] if ok else 1.1
    return th,rows

def main():
    register();p=population();part=partition(p)
    for name in ('models','arrays','logs'):(RUN/name).mkdir(exist_ok=True)
    write(RUN/'SOURCE_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in (KIT/'context43.py',KIT/'experiment43.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    count=0
    for seed in SEEDS:
        a,v,X=observed(p,seed);Y=truth(p,a,v);old=dict(np.load(V42/'arrays'/f'seed{seed}.npz'));out=dict(anchor=a['anchor'],fallback=v['points_pred_segment_physical3_direct'],hard=old['hard'],valid=a['valid'],label_valid=Y['valid'],label_useful=Y['useful'],gain=Y['gain'],points_V42=old['points_constraint_visual_pre_all'],accepted_V42=old['accepted_constraint_visual'],prob_V42=old['prob_constraint_visual']);models={};thresholds={}
        for arm in ARMS:
            m=fit43(X,Y,a['valid'],p,part,seed,arm);pr=probabilities(m,X[arm.rsplit('_',1)[1]]);models[arm]=m;out['prob_'+arm]=pr;count+=1
            for kind in ROUTES:
                name=arm+'_'+kind;th,grid=calibrate(p,part,a,v,Y,pr,kind);thresholds[name]=dict(threshold=th,grid=grid);pt,sel,allow=route43(p,a,v,pr,th,kind);out['points_'+name]=pt;out['selected_'+name]=sel;out['accepted_'+name]=allow
            print('FIT43',seed,arm,'thresholds',[thresholds[arm+'_'+k]['threshold'] for k in ROUTES],count,flush=True)
        err=np.where(Y['useful']&a['valid'],Y['candidate_error'],np.inf);idx=err.argmin(-1);out['points_GT_useful_best']=np.where((err.min(-1)<Y['base_error'])[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],idx],out['fallback']);out['points_GT_useful_residual']=route43(p,a,v,Y['useful'].astype(float),.5,'residual')[0]
        (RUN/'models'/f'seed{seed}.pkl').write_bytes(pickle.dumps(models,protocol=4));write(RUN/'models'/f'seed{seed}.json',dict(complete=True,thresholds=thresholds));np.savez_compressed(RUN/'arrays'/f'seed{seed}.npz',**out)
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=count,boosting_rounds=count*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz'))]}))
if __name__=='__main__':main()
