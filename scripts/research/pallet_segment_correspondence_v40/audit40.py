from context40 import *
from observations40 import extract,build,observed_roles,privileged_roles,encode,canonical
from evaluate40 import METHODS
import pickle,copy

def main():
    p=population();part=partition(p);s=read(RUN/'SUMMARY.json');raw=dict(np.load(RUN/'segments.npz'));models=0;points=0;refits=0
    for file in ('PRIOR_BINDINGS.json','SOURCE_BINDINGS.json','TRAINING_COMPLETE.json','EVALUATION_COMPLETE.json'):
        for name,h in read(RUN/file)['files'].items():assert sha(ROOT/name)==h,name
    repeated=extract(p)
    for key in raw:assert np.array_equal(raw[key],repeated[key]),key
    hardall=np.stack([np.load(RUN/'arrays'/f'seed{k}.npz')['hard'] for k in SEEDS])
    for seed in SEEDS:
        z=load_seed(seed,p);a,X,support=observable(p,z,seed,'both');extra,matched=build(p,z,a,support,raw);f=dict(np.load(RUN/'features'/f'seed{seed}.npz'));v=dict(np.load(RUN/'arrays'/f'seed{seed}.npz'));b=pickle.loads((RUN/'models'/f'seed{seed}.pkl').read_bytes());old=dict(np.load(V39/'arrays'/f'both_seed{seed}.npz'))
        assert np.array_equal(f['geometry'],X['geometry'])
        assert np.array_equal(v['hard_prediction'],old['hard_prediction_line'])
        assert np.array_equal(v['target'],np.einsum('njkd,njd->njk',a['candidate_normal'],p['g']['gt']-a['anchor'])/p['diag'][:,None,None]) or np.allclose(v['target'],np.einsum('njkd,njd->njk',a['candidate_normal'],p['g']['gt']-a['anchor'])/p['diag'][:,None,None],atol=1e-12)
        for source in SOURCES:
            assert np.array_equal(extra[source],f[source]);feature=np.concatenate((f['geometry'],f[source]),-1)
            for family in FAMILIES:
                key=source+'_'+family;mu=pred(b[key],feature);assert b[key].n_iter_==60;assert np.array_equal(mu,v['mu_'+key]);models+=1;pt,sel,gain=select(a,mu,np.zeros_like(mu),p)
                # Independent gain identity for hypothetical predicted-normal displacement.
                amount=a['candidate_amount']/p['diag'][:,None,None];assert np.allclose(gain,mu**2-(mu-amount)**2,atol=1e-7)
                if family=='all':assert np.array_equal(pt,v['points_'+key]) and np.array_equal(sel,v['selected_'+key])
                else:
                    for name,mask in [('hard_gate',v['hard_prediction']),('GT_hard',v['hard'])]:assert np.array_equal(np.where(mask[...,None],pt,a['anchor']),v['points_'+source+'_'+name])
        for method in METHODS:
            pt=v['anchor' if method=='anchor' else 'points_'+method];err=np.hypot(pt[...,0]-p['g']['gt'][...,0],pt[...,1]-p['g']['gt'][...,1]);points+=1
            assert np.isfinite(err).all()
            for key,M in masks(p,hardall):assert abs(err[M[seed-1]].mean()-s['results'][key][method]['per_seed_px'][seed-1])<1e-9,(key,method)
        for family in FAMILIES:
            for source in ('baseline',*SOURCES):
                mu=v['mu_'+source+'_'+family];err=np.abs(mu-v['target'])*p['diag'][:,None,None]
                for split in ('fit','check'):
                    mask=part[split][:,None]&v['hard'];assert abs(err[mask].mean()-s['direction'][split+'/all/hard'][family][source]['per_seed_px'][seed-1])<1e-9
        if seed==1:
            changed=copy.deepcopy(p)
            for key in changed['g']:changed['g'][key]=np.full_like(changed['g'][key],123)
            before=observed_roles(p,z,support,raw);after=observed_roles(changed,z,support,raw)
            for x,y in zip(before,after):assert np.array_equal(x,y)
            changed=copy.deepcopy(p);changed['g']['gt']+=999
            for x,y in zip(privileged_roles(p,raw),privileged_roles(changed,raw)):assert np.array_equal(x,y)
            for source,family in [('pred_segment','all'),('pred_segment','hard'),('GT_segment','all')]:
                key=source+'_'+family;feature=np.concatenate((f['geometry'],f[source]),-1);xx=feature.copy();xx[~part['fit']]=999;yy=v['target'].copy();yy[~part['fit']]=-999;hh=v['hard'].copy();hh[~part['fit']]=~hh[~part['fit']];m=fit_reg(xx,yy,hh,part['fit'],p['groups'],40000+seed,'hard_geometry' if family=='hard' else 'all');assert np.array_equal(pred(m,feature),v['mu_'+key]);refits+=1
            # Reversing endpoint order and translating/scaling image coordinates must preserve encoding.
            q=p['g']['support'];valid=p['g']['length']>0;enc=encode(q,valid,a['anchor'],a['candidate_normal'],p['diag']);reverse=encode(q[:,:,::-1],valid,a['anchor'],a['candidate_normal'],p['diag']);assert np.allclose(enc,reverse,atol=1e-6)
            shift=np.array([13.,-22.]);transformed=encode(q*2+shift,valid,a['anchor']*2+shift,a['candidate_normal'],p['diag']*2);assert np.allclose(enc,transformed,atol=1e-6)
            assert np.all(extra['GT_line'][...,5:]==0)
            linevalid=extra['GT_line'][...,0]>0;g=p['g'];ir=np.repeat(np.array(INC),8,axis=1);expected=(g['rho'][:,ir]-np.einsum('njkd,njd->njk',g['normal'][:,ir],a['anchor']))/p['diag'][:,None,None];assert np.allclose(extra['GT_line'][...,3][linevalid],expected[linevalid],atol=1e-6)
    write(RUN/'AUDIT.json',dict(status='PASS',models_replayed=models,point_arrays_recalculated=points,nonfit_corruption_refits=refits,raw_segment_extraction_replayed=True,actual_matching_GT_independent=True,GT_match_virtual_corner_independent=True,endpoint_reversal_translation_scale_invariant=True,GT_line_residual_independently_verified=True,prior_files_unchanged=True));print('AUDIT40 PASS',models,points,refits,flush=True)
if __name__=='__main__':main()
