from context42 import *
from experiment42 import load_inputs,labels,fit,probabilities,threshold,route
from evaluate42 import METHODS
import pickle,copy

def main():
    p=population();part=partition(p);s=read(RUN/'SUMMARY.json');hard=np.stack([np.load(RUN/'arrays'/f'seed{k}.npz')['hard'] for k in SEEDS]);models=0;points=0;refits=0
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for path,h in read(RUN/file)['files'].items():assert sha(ROOT/path)==h,path
    for seed in SEEDS:
        a,v,X=load_inputs(p,seed);Y=labels(p,a);q=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));b=pickle.loads((RUN/'models'/f'seed{seed}.pkl').read_bytes());meta=read(RUN/'models'/f'seed{seed}.json');assert np.array_equal(q['fallback'],v['points_pred_segment_physical3_direct']);inc=np.array(INC);quality=np.load(V41/'arrays'/f'quality_oracle_seed{seed}.npz')['quality'];assert np.array_equal(Y['physical'],quality[:,inc]);assert np.array_equal(Y['visible'],p['g']['length'][:,inc]>0);offset_error=(v['pred_segment_physical3_target']-v['pred_segment_physical3_prior'])*p['diag'][:,None,None];assert np.array_equal(Y['constraint'],Y['physical']&(np.abs(offset_error)<=3));assert np.all(~Y['constraint']|Y['physical']) and np.all(~Y['physical']|Y['visible'])
        for target in TARGETS:
            assert np.array_equal(q['label_'+target],Y[target])
            for feature in FEATURES:
                name=target+'_'+feature;m=b[name];assert m.n_iter_==60;pr=probabilities(m,X[feature]);assert np.array_equal(pr,q['prob_'+name]);th,grid=threshold(pr,Y[target],a['valid'],p,part);assert meta['thresholds'][name]==dict(threshold=th,grid=grid);models+=1;changed=pr.copy();changed[part['check']]=999;yy=Y[target].copy();yy[part['check']]=~yy[part['check']];assert threshold(changed,yy,a['valid'],p,part)==(th,grid)
                for kind in ROUTES:
                    pt,sel=route(p,a,v,(pr>=th)&a['valid'],kind);key=name+'_'+kind;assert np.array_equal(pt,q['points_'+key]) and np.array_equal(sel,q['selected_'+key]);mask=part['check'][:,None]&(sel>=0);idx=np.maximum(sel,0);truth=Y['constraint'][np.arange(p['N'])[:,None],np.arange(8)[None],idx];assert int(mask.sum())==s['selected'][str(seed)][key]['selected_points']
                    if mask.any():assert abs(truth[mask].mean()-s['selected'][str(seed)][key]['constraint_precision'])<1e-12
                if seed==1 and feature=='visual':
                    xx=X[feature].copy();xx[~part['fit']]=999;yy=Y[target].copy();yy[~part['fit']]=~yy[~part['fit']];valid=a['valid'].copy();valid[~part['fit']]=~valid[~part['fit']];refit=fit(xx,yy,valid,p,part,seed);assert np.array_equal(probabilities(refit,X[feature]),pr);refits+=1
            for kind in ROUTES:
                pt,sel=route(p,a,v,Y[target]&a['valid'],kind);assert np.array_equal(pt,q['points_GT_'+target+'_'+kind])
        for method in METHODS:
            pt=q[method if method in ('anchor','fallback') else 'points_'+method];error=np.hypot(pt[...,0]-p['g']['gt'][...,0],pt[...,1]-p['g']['gt'][...,1]);assert np.isfinite(error).all();points+=1
            for key,M in masks(p,hard):
                if key in s['results']:assert abs(error[M[seed-1]].mean()-s['results'][key][method]['per_seed_px'][seed-1])<1e-9
            if method in ('GT_candidate','GT_safe_primary'):assert np.all(error<=np.linalg.norm(q['fallback']-p['g']['gt'],axis=-1)+1e-8)
        if seed==1:
            pp=copy.deepcopy(p)
            for key in pp['g']:pp['g'][key]=np.full_like(pp['g'][key],123)
            aa,vv,xx=load_inputs(pp,seed)
            for key in X:assert np.array_equal(X[key],xx[key])
            for kind in ROUTES:
                before=route(p,a,v,q['accepted_constraint_visual'],kind);after=route(pp,aa,vv,q['accepted_constraint_visual'],kind)
                for x,y in zip(before,after):assert np.array_equal(x,y)
    write(RUN/'AUDIT.json',dict(status='PASS',classifiers_replayed=models,point_arrays_recalculated=points,nonfit_corruption_refits=refits,check_labels_cannot_change_calibration_threshold=True,actual_inputs_and_routing_GT_independent=True,nested_labels_independent_V41_geometry_check=True,prior_files_unchanged=True));print('AUDIT42 PASS',models,points,refits,flush=True)
if __name__=='__main__':main()
