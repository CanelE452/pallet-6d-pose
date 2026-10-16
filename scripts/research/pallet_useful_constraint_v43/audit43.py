from context43 import *
from experiment43 import observed,truth,fit43,route43,calibrate
from evaluate43 import METHODS,diagnose
import pickle,copy

def main():
    p=population();part=partition(p);summary=read(RUN/'SUMMARY.json');count=0;refits=0;points=0
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for name,h in read(RUN/file)['files'].items():assert sha(ROOT/name)==h,name
    assert read(RUN/'SOURCE_BINDINGS.json')['protocol_sha256']==sha(RUN/'PROTOCOL.json')
    qs=[dict(np.load(RUN/'arrays'/f'seed{s}.npz')) for s in SEEDS];hard=np.stack([q['hard'] for q in qs])
    for seed,q in zip(SEEDS,qs):
        a,v,X=observed(p,seed);Y=truth(p,a,v);models=pickle.loads((RUN/'models'/f'seed{seed}.pkl').read_bytes());meta=read(RUN/'models'/f'seed{seed}.json');old=np.load(V42/'arrays'/f'seed{seed}.npz');assert np.array_equal(q['label_valid'],old['label_constraint']);assert np.array_equal(q['fallback'],old['fallback']);assert np.array_equal(q['points_V42'],old['points_constraint_visual_pre_all'])
        ce=np.hypot(a['candidate'][...,0]-p['g']['gt'][:,:,None,0],a['candidate'][...,1]-p['g']['gt'][:,:,None,1]);be=np.hypot(q['fallback'][...,0]-p['g']['gt'][...,0],q['fallback'][...,1]-p['g']['gt'][...,1]);useful=q['label_valid']&(be[:,:,None]-ce>1);assert np.array_equal(useful,q['label_useful'])
        for arm in ARMS:
            model=models[arm];assert model.n_iter_==60;pr=probabilities(model,X[arm.rsplit('_',1)[1]]);assert np.array_equal(pr,q['prob_'+arm]);count+=1
            for kind in ROUTES:
                name=arm+'_'+kind;th,grid=calibrate(p,part,a,v,Y,pr,kind);assert meta['thresholds'][name]==dict(threshold=th,grid=grid);pt,sel,allow=route43(p,a,v,pr,th,kind)
                for x,y in [(pt,q['points_'+name]),(sel,q['selected_'+name]),(allow,q['accepted_'+name])]:assert np.array_equal(x,y)
                # Full fixed-grid calibration remains identical after changing check GT, labels and scores.
                pp=copy.deepcopy(p);pp['g']['gt'][part['check']]=999;yy=copy.deepcopy(Y)
                for z in yy:yy[z][part['check']]=0
                prob=pr.copy();prob[part['check']]=.777;assert calibrate(pp,part,a,v,yy,prob,kind)==(th,grid)
                for split in ('fit','calibration','check'):assert diagnose(q,p,part,name,split)==summary['diagnostics'][str(seed)][name][split]
            if seed==1 and arm in ('valid_weighted_visual','useful_weighted_visual'):
                xx={k:z.copy() for k,z in X.items()};yy=copy.deepcopy(Y);valid=a['valid'].copy()
                for z in xx.values():z[~part['fit']]=999
                for z in yy.values():z[~part['fit']]=0
                valid[~part['fit']]=False;mm=fit43(xx,yy,valid,p,part,seed,arm);assert np.array_equal(probabilities(mm,X[arm.rsplit('_',1)[1]]),pr);refits+=1
        for method in METHODS:
            pts=q[method if method in ('anchor','fallback') else 'points_'+method];err=np.hypot(pts[...,0]-p['g']['gt'][...,0],pts[...,1]-p['g']['gt'][...,1]);assert np.isfinite(err).all();points+=1
            for key,M in masks(p,hard):
                if key in summary['results']:assert abs(err[M[seed-1]].mean()-summary['results'][key][method]['per_seed_px'][seed-1])<1e-9
            if method=='GT_useful_best':assert np.allclose(err,np.minimum(be,np.where(useful&a['valid'],ce,np.inf).min(-1)))
        if seed==1:
            pp=copy.deepcopy(p)
            for z in pp['g']:pp['g'][z]=np.full_like(pp['g'][z],123)
            aa,vv,xx=observed(pp,seed)
            for k in X:assert np.array_equal(X[k],xx[k])
            for kind in ROUTES:
                for x,y in zip(route43(p,a,v,q['prob_useful_weighted_visual'],.5,kind),route43(pp,aa,vv,q['prob_useful_weighted_visual'],.5,kind)):assert np.array_equal(x,y)
    write(RUN/'AUDIT.json',dict(status='PASS',models_replayed=count,point_arrays_recalculated=points,nonfit_corruption_refits=refits,check_cannot_change_calibration=True,inference_GT_independent=True,useful_label_independently_checked=True,prior_files_unchanged=True));print('AUDIT43 PASS',count,points,refits,flush=True)
if __name__=='__main__':main()
