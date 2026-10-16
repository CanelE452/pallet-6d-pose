from context37 import *
from analyze37 import load_seed,features,fit,predict,masks


def main():
    p=population();g=p['g'];s=read(RUN/'OBJECTIVES.json');r=read(RUN/'RESIDUAL.json');assert s['additional_ridge_fits']==48 and s['total_ridge_fits']==144 and r['identity_max_px']<1e-8
    for protocol,source in [('OBJECTIVE_PROTOCOL.json','objectives37.py'),('RESIDUAL_PROTOCOL.json','residual37.py')]:
        v=read(RUN/protocol);assert v['primary_sha256']==sha(RUN/'SUMMARY.json') and v['source_sha256']==sha(KIT/source)
    inc=np.array(INC);ir=np.repeat(inc,8,axis=1);count=0;maxerror=0.;repeated=0
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ('grid','both'):
        a=dict(np.load(RUN/'arrays'/f'v34_{arm}_seed{seed}.npz'));anchor=a['anchor'];n=a['candidate_normal'];amount=a['candidate_amount'];candidate=a['candidate'];baseerr=np.linalg.norm(anchor-g['gt'],axis=-1);ce=np.linalg.norm(candidate-g['gt'][:,:,None],axis=-1);vi=z['visual'][:,inc].reshape(p['N'],8,24,32);X=features(p,z,anchor,n,amount,a['candidate_score'],ir,vi)
        targets=dict(normal=(n*(g['gt']-anchor)[:,:,None]).sum(-1)/p['diag'][:,None,None],pixel=(baseerr[...,None]-ce)/10)
        # Independent analytic squared-gain identity at the true normal target.
        norm=amount/p['diag'][:,None,None];actual=(baseerr[...,None]**2-ce**2)/p['diag'][:,None,None]**2;assert np.allclose(norm*(2*targets['normal']-norm),actual,atol=1e-8,rtol=0)
        assert np.allclose(norm*(2*targets['normal']-norm*(n*n).sum(-1)),actual,atol=1e-12,rtol=0)
        for target,y in targets.items():
            saved=dict(np.load(RUN/'objectives'/f'{target}_{arm}_seed{seed}.npz'));coefs=dict(np.load(RUN/'objectives'/f'fit_{target}_{arm}_seed{seed}.npz'))
            for mode in MODES:
                model={k:coefs[mode+'_'+k] for k in ('center','scale','coef')};v=predict(model,X[mode]);assert np.allclose(v,saved['predicted_'+mode],atol=1e-12);utility=norm*(2*v-norm) if target=='normal' else v;k=utility.argmax(-1);selected=np.where(utility.max(-1)>0,k,-1);assert np.array_equal(selected,saved['selected_'+mode]);pred=np.where((selected>=0)[...,None],candidate[np.arange(p['N'])[:,None],np.arange(8)[None],k],anchor);maxerror=max(maxerror,float(np.abs(pred-saved['points_'+mode]).max()));err=np.hypot(pred[...,0]-g['gt'][...,0],pred[...,1]-g['gt'][...,1]);assert np.isfinite(err).all()
                for key,mask in masks(p):assert abs(err[mask].mean()-s['results'][key][target+'/'+arm+'/'+mode]['per_seed_px'][seed-1])<1e-9
                count+=1
                if seed==1 and arm=='both':
                    changed=y.copy();changed[~p['train']]+=999;XX=X[mode].copy();XX[~p['train']]*=-999;model2=fit(XX,changed,p['train'],p['groups']);assert all(np.array_equal(model[k],model2[k]) for k in model);repeated+=1
    assert maxerror<1e-9
    write(RUN/'EXTRA_AUDIT.json',dict(status='PASS',additional_selectors_replayed=count,additional_fits_repeated_with_check_GT_features_changed=repeated,selection_replay_max_px=maxerror,analytic_normal_gain_identity=True,residual_identity_max_px=r['identity_max_px'],primary_summary_unchanged=True,objectives_sha256=sha(RUN/'OBJECTIVES.json'),residual_sha256=sha(RUN/'RESIDUAL.json')));print('EXTRA AUDIT37 PASS',count,flush=True)

if __name__=='__main__':main()
