from context39 import *
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
import pickle

def fit_reg(X,y,hard,take,groups,seed,mode):
    mask=np.broadcast_to(take[:,None,None],y.shape).copy();gw=np.broadcast_to(frame_weights(groups,take)[:,None,None],y[take].shape).copy();hh=np.broadcast_to(hard[take,:,None],gw.shape)
    if mode=='balanced':
        frac=(gw*hh).sum()/gw.sum();assert 0<frac<1;gw*=np.where(hh,.5/frac,.5/(1-frac))
    elif mode.startswith('hard'):mask&=hard[:,:,None]
    xx=X[mask];yy=y[mask];weights=gw[np.broadcast_to(hard[take,:,None],gw.shape)] if mode.startswith('hard') else gw.reshape(-1);weights/=weights.mean();m=HistGradientBoostingRegressor(loss='absolute_error',random_state=seed,**PARAMS);m.fit(xx,yy,sample_weight=weights);return m

def gate_features(X):
    # V37 first49 columns are candidate-independent point/parent/query/aspect/pointID.
    point=X['appearance'][:,:,0,:49];assert np.all(X['appearance'][...,:49]==point[:,:,None]);extra=X['geometry'][...,49:];return dict(point=point,line=np.concatenate((point,extra.mean(2),extra.std(2)),axis=-1))

def threshold(scores,hard,p,part):
    take=part['calibration'];w=np.broadcast_to(frame_weights(p['groups'],take)[:,None],hard[take].shape);table=[]
    for th in np.arange(.1,.901,.05):
        sel=scores[take]>=th;tp=(w*sel*hard[take]).sum();fp=(w*sel*~hard[take]).sum();fn=(w*~sel*hard[take]).sum();f1=2*tp/max(2*tp+fp+fn,1e-12);table.append(dict(threshold=float(round(th,2)),f1=float(f1),precision=float(tp/max(tp+fp,1e-12)),recall=float(tp/max(tp+fn,1e-12))))
    return max(table,key=lambda x:(x['f1'],x['threshold']))['threshold'],table

def main():
    register();p=population();part=partition(p);(RUN/'models').mkdir(exist_ok=True);(RUN/'arrays').mkdir(exist_ok=True);fits=0
    write(RUN/'SOURCE_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in KIT.glob('*.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        name=f'{arm}_seed{seed}';a,X,_=observable(p,z,seed,arm);X['hard_appearance']=np.concatenate((X['appearance'],a['candidate_normal']),-1).astype(np.float32);target=(a['candidate_normal']*(p['g']['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];baseerr=np.linalg.norm(a['anchor']-p['g']['gt'],axis=-1);hard=baseerr>=HARD_PX;standard=np.load(V38/'arrays'/f'{name}.npz')['mu'];out=dict(anchor=a['anchor'],target=target,hard=hard,mu_standard=standard);reg={};fold=[]
        for mode in REGRESSORS:
            feature=X['hard_appearance'] if mode=='hard_appearance' else X['geometry'];m=fit_reg(feature,target,hard,part['fit'],p['groups'],39000+seed,mode);out['mu_'+mode]=pred(m,feature);reg[mode]=m;fits+=1;print('FIT39',name,mode,'total',fits,flush=True)
        oof=np.full_like(target,np.nan)
        for k,groups in enumerate(part['fold_groups']):
            hold=np.isin(p['groups'],groups);take=part['fit']&~hold;m=fit_reg(X['geometry'],target,hard,take,p['groups'],39000+seed+k,'hard_geometry');oof[hold]=pred(m,X['geometry'][hold]);fold.append(m);fits+=1
        out['oof_hard_geometry']=oof;gx=gate_features(X);classifiers={};thresholds={}
        for mode in GATES:
            feature=gx[mode];take=part['fit'];xx=feature[take].reshape(-1,feature.shape[-1]);yy=hard[take].reshape(-1);w=np.repeat(frame_weights(p['groups'],take),8);m=HistGradientBoostingClassifier(random_state=39100+seed,**PARAMS);m.fit(xx,yy,sample_weight=w);score=m.predict_proba(feature.reshape(-1,feature.shape[-1]))[:,1].reshape(p['N'],8);th,table=threshold(score,hard,p,part);classifiers[mode]=m;thresholds[mode]=dict(threshold=th,calibration_grid=table);out['hard_probability_'+mode]=score;out['hard_prediction_'+mode]=score>=th;fits+=1
        for mode in ('standard',*REGRESSORS):
            points,selected,gain=select(a,out['mu_'+mode],np.zeros_like(target),p);out.update({'points_'+mode:points,'selected_'+mode:selected,'gain_'+mode:gain})
        for mode in GATES:
            take=out['hard_prediction_'+mode];out['points_'+mode+'_gate']=np.where(take[...,None],out['points_hard_geometry'],a['anchor']);out['selected_'+mode+'_gate']=np.where(take,out['selected_hard_geometry'],-1)
        for mode in ('hard_geometry','hard_appearance'):
            out['points_GT_'+mode]=np.where(hard[...,None],out['points_'+mode],a['anchor']);out['selected_GT_'+mode]=np.where(hard,out['selected_'+mode],-1)
        e=np.linalg.norm(out['points_GT_hard_geometry']-p['g']['gt'],axis=-1);take=e<baseerr;out['points_GT_safe_hard_geometry']=np.where(take[...,None],out['points_GT_hard_geometry'],a['anchor']);out['selected_GT_safe_hard_geometry']=np.where(take,out['selected_GT_hard_geometry'],-1);ce=np.linalg.norm(a['candidate']-p['g']['gt'][:,:,None],axis=-1);k=ce.argmin(-1);take=ce.min(-1)<baseerr;out['points_candidate_oracle']=np.where(take[...,None],a['candidate'][np.arange(p['N'])[:,None],np.arange(8)[None],k],a['anchor']);out['selected_candidate_oracle']=np.where(take,k,-1)
        np.savez_compressed(RUN/'arrays'/f'{name}.npz',**out);bundle=dict(regressors=reg,fold_regressors=fold,classifiers=classifiers,thresholds=thresholds,fit_groups=part['fit_groups'],calibration_groups=part['calibration_groups']);(RUN/'models'/f'{name}.pkl').write_bytes(pickle.dumps(bundle,protocol=4));write(RUN/'models'/f'{name}.json',dict(complete=True,new_models=8,rounds=480,thresholds=thresholds,fit_hard_points=int(hard[part['fit']].sum()),calibration_hard_points=int(hard[part['calibration']].sum()),check_hard_points=int(hard[part['check']].sum())));print('DONE39',name,fits,flush=True)
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,hgb_fits=fits,boosting_rounds=fits*60,neural_updates=0,files={str(q.relative_to(ROOT)):sha(q) for q in [*sorted((RUN/'models').glob('*')),*sorted((RUN/'arrays').glob('*.npz'))]}));print('TRAIN39 COMPLETE',fits,flush=True)
if __name__=='__main__':main()
