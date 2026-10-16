from context39 import *
from experiment39 import fit_reg,gate_features,threshold
from evaluate39 import METHODS,masks,metric
from sklearn.ensemble import HistGradientBoostingClassifier
import pickle

def main():
    p=population();part=partition(p);g=p['g'];s=read(RUN/'SUMMARY.json');count=0;replayed=0;refits=0;gatepoint={}
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for path,h in read(RUN/file)['files'].items():assert sha(ROOT/path)==h,path
    hardall=np.stack([np.load(RUN/'arrays'/f'both_seed{k}.npz')['hard'] for k in SEEDS])
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        name=f'{arm}_seed{seed}';a,X,_=observable(p,z,seed,arm);X['hard_appearance']=np.concatenate((X['appearance'],a['candidate_normal']),-1).astype(np.float32);v=dict(np.load(RUN/'arrays'/f'{name}.npz'));b=pickle.loads((RUN/'models'/f'{name}.pkl').read_bytes());be=np.hypot(a['anchor'][...,0]-g['gt'][...,0],a['anchor'][...,1]-g['gt'][...,1]);assert np.array_equal(v['hard'],be>=10);assert np.array_equal(v['hard'],hardall[seed-1]);target=(a['candidate_normal']*(g['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];assert np.array_equal(target,v['target']);assert np.array_equal(v['mu_standard'],np.load(V38/'arrays'/f'{name}.npz')['mu'])
        for mode in REGRESSORS:
            feature=X['hard_appearance'] if mode=='hard_appearance' else X['geometry'];m=b['regressors'][mode];assert m.n_iter_==60;assert np.array_equal(pred(m,feature),v['mu_'+mode]);replayed+=1
        for k,groups in enumerate(part['fold_groups']):
            take=np.isin(p['groups'],groups);assert np.array_equal(pred(b['fold_regressors'][k],X['geometry'][take]),v['oof_hard_geometry'][take]);replayed+=1
        assert np.isnan(v['oof_hard_geometry'][~part['fit']]).all();gx=gate_features(X)
        for mode in GATES:
            feature=gx[mode];prob=b['classifiers'][mode].predict_proba(feature.reshape(-1,feature.shape[-1]))[:,1].reshape(p['N'],8);assert np.array_equal(prob,v['hard_probability_'+mode]);th,grid=threshold(prob,v['hard'],p,part);assert b['thresholds'][mode]==dict(threshold=th,calibration_grid=grid);assert np.array_equal(prob>=th,v['hard_prediction_'+mode]);changed=prob.copy();hh=v['hard'].copy();changed[part['check']]=999;hh[part['check']]=~hh[part['check']];assert threshold(changed,hh,p,part)==(th,grid);replayed+=1
        gatepoint[(arm,seed)]=v['hard_probability_point']
        for mode in ('standard',*REGRESSORS):
            pts,sel,gain=select(a,v['mu_'+mode],np.zeros_like(target),p);assert np.array_equal(pts,v['points_'+mode]) and np.array_equal(sel,v['selected_'+mode]) and np.array_equal(gain,v['gain_'+mode])
        for mode in GATES:
            expected=np.where(v['hard_prediction_'+mode][...,None],v['points_hard_geometry'],a['anchor']);assert np.array_equal(expected,v['points_'+mode+'_gate'])
        for mode in ('hard_geometry','hard_appearance'):assert np.array_equal(np.where(v['hard'][...,None],v['points_'+mode],a['anchor']),v['points_GT_'+mode])
        for mode in METHODS[1:]:
            error=np.hypot(v['points_'+mode][...,0]-g['gt'][...,0],v['points_'+mode][...,1]-g['gt'][...,1]);assert np.isfinite(error).all();count+=1
            for key,mask in masks(p,hardall):assert abs(error[mask[seed-1]].mean()-s['results'][key]['methods'][arm][mode]['per_seed_px'][seed-1])<1e-9
        assert np.all(np.linalg.norm(v['points_GT_safe_hard_geometry']-g['gt'],axis=-1)<=be+1e-8);ce=np.linalg.norm(a['candidate']-g['gt'][:,:,None],axis=-1);assert np.allclose(np.linalg.norm(v['points_candidate_oracle']-g['gt'],axis=-1),np.minimum(be,ce.min(-1)),atol=1e-9)
        if seed==1 and arm=='both':
            yy=target.copy();yy[~part['fit']]=1e5;hh=v['hard'].copy();hh[~part['fit']]=~hh[~part['fit']];xx=X['geometry'].copy();xx[~part['fit']]=-999
            for mode in ('balanced','hard_geometry'):
                m=fit_reg(xx,yy,hh,part['fit'],p['groups'],39000+seed,mode);assert np.array_equal(pred(m,X['geometry']),v['mu_'+mode]);refits+=1
            feature=gx['line'].copy();feature[~part['fit']]=999;take=part['fit'];m=HistGradientBoostingClassifier(random_state=39100+seed,**PARAMS);m.fit(feature[take].reshape(-1,feature.shape[-1]),hh[take].reshape(-1),sample_weight=np.repeat(frame_weights(p['groups'],take),8));assert np.array_equal(m.predict_proba(gx['line'].reshape(-1,feature.shape[-1]))[:,1].reshape(p['N'],8),v['hard_probability_line']);refits+=1
    for seed in SEEDS:assert np.array_equal(gatepoint[('grid',seed)],gatepoint[('both',seed)])
    write(RUN/'AUDIT.json',dict(status='PASS',new_models_replayed=replayed,point_prediction_arrays_checked=count,new_fit_repetitions_with_nonfit_corruption=refits,check_GT_gate_threshold_invariant=True,point_classifier_identical_across_line_arms=True,prior_files_unchanged=True,scope='48newmodel replays,60pointoutputs;3auditrefits. Fixedupstreamold16group exposure remains. GToracles only diagnostics.'))
    print('AUDIT39 PASS',count,replayed,flush=True)
if __name__=='__main__':main()
