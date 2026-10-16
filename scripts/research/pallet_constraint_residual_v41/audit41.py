from context41 import *
from experiment41 import inputs,fit,choose
from evaluate41 import METHODS
import pickle,copy

def main():
    p=population();part=partition(p);s=read(RUN/'SUMMARY.json');assert read(RUN/'GEOMETRY_CHECK.json')['status']=='PASS';models=0;refits=0;point_arrays=0
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for path,h in read(RUN/file)['files'].items():assert sha(ROOT/path)==h,path
    hard=np.stack([np.load(RUN/'arrays'/f'seed{k}.npz')['hard'] for k in SEEDS])
    for seed in SEEDS:
        v=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));b=pickle.loads((RUN/'models'/f'seed{seed}.pkl').read_bytes());old=dict(np.load(V39/'arrays'/f'both_seed{seed}.npz'));assert np.array_equal(old['hard_prediction_line'],v['hard_prediction'])
        for source in SOURCES:
            for family,a in inputs(p,seed,source).items():
                prefix=source+'_'+family;y=(a['normal']*(p['g']['gt']-a['anchor'])[:,:,None]).sum(-1)/p['diag'][:,None,None];assert np.array_equal(y,v[prefix+'_target']);assert np.array_equal(a['prior'],v[prefix+'_prior']);assert np.array_equal(a['valid'],v[prefix+'_valid']);valid=a['valid'] if family=='physical3' else np.ones_like(a['valid'])
                for mode in MODES:
                    key=prefix+'_'+mode
                    if mode=='analytic':mu=a['prior'] if family=='physical3' else np.where(a['valid'],a['prior'],old['mu_standard'])
                    elif family=='legacy24' and mode=='direct':mu=a['direct']
                    else:
                        assert b[key].n_iter_==60;mu=pred(b[key],a['X'])+(a['prior'] if mode=='residual' else 0);models+=1
                    assert np.array_equal(mu,v['mu_'+key]);pt,sel=choose(a,mu,p,family);assert np.array_equal(pt,v['points_'+key]) and np.array_equal(sel,v['selected_'+key]);assert np.array_equal(np.where(v['hard_prediction'][...,None],pt,a['anchor']),v['points_'+key+'_gate'])
                    if family=='legacy24' and mode=='direct':assert np.array_equal(pt,np.load(V40/'arrays'/f'seed{seed}.npz')['points_'+source+'_all'])
                    for split in ('fit','check'):
                        M=part[split][:,None,None]&v['hard'][...,None]&a['valid'];err=np.abs(mu-y)*p['diag'][:,None,None];assert abs(err[M].mean()-s['direction'][split+'/all/hard'][prefix][mode]['per_seed_px'][seed-1])<1e-9
                if family=='physical3':
                    norm=np.linalg.norm(a['normal'],axis=-1);assert np.allclose(norm[a['valid']],1,atol=1e-8);delta=a['candidate']-a['anchor'][:,:,None];t=np.stack((-a['normal'][...,1],a['normal'][...,0]),-1);assert np.allclose((delta*t).sum(-1),0,atol=1e-8);assert np.all(np.linalg.norm(delta,axis=-1)<=.15*p['diag'][:,None,None]+1e-8)
                    offsets=read(V35/'OFFSET_CALIBRATION.json')['models'];med=np.array([[offsets[f'{j}_{r}']['median'] for r in INC[j]] for j in range(8)]);rho=(a['normal']*a['anchor'][:,:,None]).sum(-1)+(a['prior']-med[None])*p['diag'][:,None,None];residual=((a['normal']*p['g']['gt'][:,:,None]).sum(-1)-rho)/p['diag'][:,None,None]-med[None];assert np.allclose((y-a['prior'])[a['valid']],residual[a['valid']],atol=1e-10)
                if seed==1 and source=='pred_segment':
                    for mode in (('residual',) if family=='legacy24' else ('direct','residual')):
                        xx=a['X'].copy();xx[~part['fit']]=999;yy=(y-a['prior'] if mode=='residual' else y).copy();yy[~part['fit']]=-999;vv=valid.copy();vv[~part['fit']]=~vv[~part['fit']];m=fit(xx,yy,vv,part['fit'],p['groups'],41000+seed);predicted=pred(m,a['X'])+(a['prior'] if mode=='residual' else 0);assert np.array_equal(predicted,v['mu_'+prefix+'_'+mode]);refits+=1
        for method in METHODS:
            if method=='V40_actual':pt=np.load(V40/'arrays'/f'seed{seed}.npz')['points_pred_segment_all']
            else:pt=v['anchor' if method=='anchor' else 'points_'+method]
            err=np.hypot(pt[...,0]-p['g']['gt'][...,0],pt[...,1]-p['g']['gt'][...,1]);assert np.isfinite(err).all();point_arrays+=1
            for key,M in masks(p,hard):
                if key in s['results']:assert abs(err[M[seed-1]].mean()-s['results'][key][method]['per_seed_px'][seed-1])<1e-9,(method,key)
            if method.endswith('_GT_safe') or method.endswith('_GT_candidate'):assert np.all(err<=np.linalg.norm(v['anchor']-p['g']['gt'],axis=-1)+1e-8)
        if seed==1:
            changed=copy.deepcopy(p)
            for key in changed['g']:changed['g'][key]=np.full_like(changed['g'][key],123)
            before=inputs(p,seed,'pred_segment');after=inputs(changed,seed,'pred_segment')
            for fam in FAMILIES:
                for key in before[fam]:assert np.array_equal(before[fam][key],after[fam][key]),key
    write(RUN/'AUDIT.json',dict(status='PASS',models_replayed=models,point_arrays_recalculated=point_arrays,nonfit_corruption_refits=refits,actual_inputs_GT_independent=True,legacy24_direct_reproducesV40=True,tangent_and_bound_checked=True,residual_offset_identity_checked=True,prior_files_unchanged=True));print('AUDIT41 PASS',models,point_arrays,refits,flush=True)
if __name__=='__main__':main()
