from context38 import *
from experiment38 import observable,pred,fit_model,wquantile,select
from evaluate38 import METHODS,masks,uqstats
import pickle


def main():
    p=population();part=partition(p);s=read(RUN/'SUMMARY.json');g=p['g'];assert read(RUN/'TRAINING_COMPLETE.json')['hgb_fits']==36
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for path,h in read(RUN/file)['files'].items():assert sha(ROOT/path)==h,path
    assert len(part['fit_groups'])==12 and len(part['calibration_groups'])==4 and len(part['check_groups'])==8
    for a,b in [('fit_groups','calibration_groups'),('fit_groups','check_groups'),('calibration_groups','check_groups')]:assert not set(part[a])&set(part[b])
    assert len(set(sum(part['fold_groups'],[])))==12 and sorted(sum(part['fold_groups'],[]))==part['fit_groups'];count=0;replay=0.;identity=0.;safe=0;refit=0;foldcount=0
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        name=f'{arm}_seed{seed}';a,X,_=observable(p,z,seed,arm);v=dict(np.load(RUN/'arrays'/f'{name}.npz'));model=pickle.loads((RUN/'models'/f'{name}.pkl').read_bytes());mu=pred(model['mean'],X['geometry']);replay=max(replay,float(np.abs(mu-v['mu']).max()));assert model['mean'].n_iter_==60
        target=(a['candidate_normal']*(g['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];assert np.array_equal(target,v['target'])
        for k,groups in enumerate(part['fold_groups']):
            hold=np.isin(p['groups'],groups);pm=pred(model['fold_means'][k],X['geometry'][hold]);assert np.array_equal(pm,v['oof'][hold]);foldcount+=1
        assert np.isnan(v['oof'][~part['fit']]).all();error=np.abs(target-mu);residual=np.abs(target-v['oof']);constant=wquantile(residual,p['groups'],part['fit'],.9);assert np.all(v['scale_constant']==constant)
        for mode in ('appearance','geometry'):
            raw=pred(model['scales'][mode],X[mode]);scale=np.clip(np.exp(np.clip(raw,-20,5)),model['lo'],model['hi']);assert np.array_equal(scale,v['scale_'+mode]);assert model['scales'][mode].n_iter_==60
        for mode in SCALES:
            scale=v['scale_'+mode];ratio=error/np.maximum(scale,1e-8)
            for family,rr in [('marginal',ratio),('point',ratio.max(-1))]:
                q=wquantile(rr,p['groups'],part['calibration'],.9);assert q==model['calibration'][mode][family];changed=rr.copy();changed[part['check']]+=1e6;assert wquantile(changed,p['groups'],part['calibration'],.9)==q
                assert np.array_equal(q*scale,v['width_'+mode+'_'+family])
        # The exact normal-squared gain identity uses stored FP32 normal lengths.
        amount=a['candidate_amount']/p['diag'][:,None,None];norm2=(a['candidate_normal']**2).sum(-1);be=np.linalg.norm(a['anchor']-g['gt'],axis=-1);ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);gain=(be[...,None]**2-ce**2)/p['diag'][:,None,None]**2;identity=max(identity,float(np.abs(gain-(2*amount*target-amount**2*norm2)).max()))
        for method in METHODS:
            width=np.zeros_like(mu) if method=='mean_only' else v['width_'+method];point,selected,lower=select(a,mu,width,p);assert np.array_equal(selected,v['selected_'+method]);assert np.array_equal(point,v['points_'+method]);assert np.array_equal(lower,v['lower_'+method]);err=np.hypot(point[...,0]-g['gt'][...,0],point[...,1]-g['gt'][...,1]);assert np.isfinite(err).all();count+=1
            for key,mask in masks(p):assert abs(err[mask].mean()-s['results'][key]['methods'][arm][method]['per_seed_px'][seed-1])<1e-9
            ix=np.maximum(selected,0);take=(selected>=0);covered=take&(error[np.arange(p['N'])[:,None],np.arange(8)[None],ix]<=width[np.arange(p['N'])[:,None],np.arange(8)[None],ix]);assert np.all(err[covered]<=be[covered]+1e-8);safe+=int(covered.sum())
        # Repeat 3 fits with all calibration/check feature/target values corrupted.
        if seed==1 and arm=='both':
            xx=X['geometry'].copy();yy=target.copy();xx[~part['fit']]=12345;yy[~part['fit']]=9999;mm=fit_model(xx,yy,part['fit'],p['groups'],38000+seed);assert np.array_equal(pred(mm,X['geometry']),mu);refit+=1
            for mode in ('appearance','geometry'):
                xx=X[mode].copy();yy=np.log(residual+1e-4);xx[~part['fit']]=-54321;yy[~part['fit']]=1234;mm=fit_model(xx,yy,part['fit'],p['groups'],38100+seed);assert np.array_equal(pred(mm,X[mode]),pred(model['scales'][mode],X[mode]));refit+=1
    assert replay<1e-12 and identity<1e-10
    write(RUN/'AUDIT.json',dict(status='PASS',point_prediction_arrays=count,mean_prediction_replay_max=replay,OOF_fold_replays=foldcount,scale_model_replays=12,calibration_q_check_GT_perturbation_unchanged=True,new_fit_models_repeated_with_calibration_check_corruption=refit,analytic_squared_gain_identity_max=identity,selected_interval_covered_nonworsening_checks=safe,prior_files_unchanged=True,source_partitions_verified=True,limits='Only newmean/scalefits heldoutcalibration; originalfrozenupstreamandmedianoffsets sawold16train. 3fitrepetitions, notall36twice. Predictionrecaching fromnewimagesnotrequired; inheritedcachedpredictions used.'));print('AUDIT38 PASS',count,flush=True)

if __name__=='__main__':main()
