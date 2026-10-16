from context37 import *
from analyze37 import load_seed,features,fit,predict,masks


def main():
    p=population();g=p['g'];s=read(RUN/'SUMMARY.json');assert s['complete'] and s['ridge_fits']==96
    for record in ('PRIOR_BINDINGS.json','ANALYSIS_BINDINGS.json','ANALYSIS_COMPLETE.json'):
        for name,h in read(RUN/record)['files'].items():assert sha(ROOT/name)==h,name
    assert len(set(p['groups'][p['train']]))==16 and len(set(p['groups'][~p['train']]))==8
    assert not set(p['groups'][p['train']])&set(p['groups'][~p['train']])
    assert sum(~p['train'])==130 and p['supported'][~p['train']].sum()==820
    checked=0;identity=0.;sign_error=0.;fitreplay=0.;line_error=0.;selector_replay=0.;models=read(V35/'OFFSET_CALIBRATION.json')['models'];inc=np.array(INC);ir=np.repeat(inc,8,axis=1)
    # Check-only GT corruption cannot affect physical median or observable ridge fits.
    altered=g['gt'].copy();altered[~p['train']]+=10000
    fitted=fit_offsets(g['normal'],g['rho'],g['gt'],p['diag'],p['aspect'],g['length']>0,p['train']);other=fit_offsets(g['normal'],g['rho'],altered,p['diag'],p['aspect'],g['length']>0,p['train']);assert fitted==other
    for key in fitted:
        assert fitted[key]['median']==models[key]['median']
        for field in fitted[key]:assert np.allclose(fitted[key][field],models[key][field],atol=1e-12,rtol=0), (key,field)
    for seed in SEEDS:
      z=load_seed(seed,p)
      for arm in ARMS:
        for an in ANCHORS:
            a=dict(np.load(RUN/'arrays'/f'{an}_{arm}_seed{seed}.npz'));anchor=z[an];n=a['candidate_normal'];rho=a['candidate_rho'];med=np.array([[models[f'{j}_{r}']['median'] for r in ir[j]] for j in range(8)])*p['diag'][:,None,None];d=rho+med-(n*anchor[:,:,None]).sum(-1);full=anchor[:,:,None]+n*d[...,None];bounded=anchor[:,:,None]+n*np.clip(d,-.15*p['diag'][:,None,None],.15*p['diag'][:,None,None])[...,None]
            assert np.allclose(full,a['full_candidate'],atol=1e-9) and np.allclose(bounded,a['candidate'],atol=1e-9)
            # Exact coordinate identity and line-representation sign invariance.
            desired=(n*(g['gt']-anchor)[:,:,None]).sum(-1);physical_bias=(n*g['gt'][:,:,None]).sum(-1)-rho
            identity=max(identity,float(np.abs(desired-((rho-(n*anchor[:,:,None]).sum(-1))+physical_bias)).max()))
            flipped=anchor[:,:,None]+(-n)*((-rho-med)-((-n)*anchor[:,:,None]).sum(-1))[...,None];sign_error=max(sign_error,float(np.abs(flipped-full).max()))
            tangent=np.stack([-n[...,1],n[...,0]],-1);assert np.abs(((full-anchor[:,:,None])*tangent).sum(-1)).max()<1e-8
            baseerr=np.linalg.norm(anchor-g['gt'],axis=-1);ce=np.linalg.norm(bounded-g['gt'][:,:,None],axis=-1);expected=np.minimum(baseerr,ce.min(-1));assert np.allclose(expected,np.linalg.norm(a['free_point_oracle']-g['gt'],axis=-1),atol=1e-9)
            for k in ('free_point_oracle','free_full_oracle','matched_point_oracle','gt_normal_oracle','gt_joint_oracle'):assert np.all(np.linalg.norm(a[k]-g['gt'],axis=-1)<=baseerr+1e-8)
            for i,k in enumerate(a['names']):
                error=np.hypot(a[k][...,0]-g['gt'][...,0],a[k][...,1]-g['gt'][...,1]);assert np.isfinite(error).all();assert np.allclose(error,a['errors'][i],atol=1e-9);checked+=1
                for key,mask in masks(p):
                    r=s['results'][key]['methods'][an+'/'+arm][k];assert abs(error[mask].mean()-r['per_seed_px'][seed-1])<1e-9
            # Replay all 96 observable selectors from serialized fit coefficients.
            vi=z['visual'][:,inc].reshape(p['N'],8,24,32);X=features(p,z,anchor,n,a['candidate_amount'],a['candidate_score'],ir,vi);f=dict(np.load(RUN/'fits'/f'{an}_{arm}_seed{seed}.npz'));target=(baseerr[...,None]**2-ce**2)/p['diag'][:,None,None]**2
            for mode in MODES:
                m={key:f[mode+'_'+key] for key in ('center','scale','coef')};v=predict(m,X[mode]);k=v.argmax(-1);sel=np.where(v.max(-1)>0,k,-1);assert np.array_equal(sel,a['selected_selector_'+mode]);pred=np.where((sel>=0)[...,None],bounded[np.arange(p['N'])[:,None],np.arange(8)[None],k],anchor);selector_replay=max(selector_replay,float(np.abs(pred-a['selector_'+mode]).max()))
            if seed==1 and arm=='both' and an=='v34':
                for mode in MODES:
                    m=fit(X[mode],target,p['train'],p['groups']);changed=target.copy();changed[~p['train']]+=10000;xx=X[mode].copy();xx[~p['train']]*=-100;mm=fit(xx,changed,p['train'],p['groups']);assert all(np.array_equal(m[k],mm[k]) for k in m)
                    fitreplay=max(fitreplay,float(np.abs(m['coef']-f[mode+'_coef']).max()))
            if an=='base':
                mask=(~p['train'])[:,None]&p['supported'];distance=a['physical_distance'][mask].mean();ref=read(V36/'LINE_SUMMARY.json')['results']['check/all']['methods'][arm]['per_seed_px'][seed-1];line_error=max(line_error,abs(distance-ref))
    assert identity<1e-8 and sign_error<1e-8 and fitreplay<1e-10 and line_error<1e-8 and selector_replay<1e-9
    # Check paired bootstrap fields on primary controls against the stored arrays.
    mask=(~p['train'])[:,None]&np.ones((p['N'],8),bool)
    for method in ('free_point_oracle','selector_visual_geometry'):
        ee=np.stack([np.linalg.norm(np.load(RUN/'arrays'/f'v34_both_seed{k}.npz')[method]-g['gt'],axis=-1) for k in SEEDS]);rr=np.stack([np.linalg.norm(np.load(RUN/'arrays'/f'v34_grid_seed{k}.npz')[method]-g['gt'],axis=-1) for k in SEEDS]);assert ci(ee-rr,mask,p['groups'])==s['comparisons']['check/all/all_detected']['v34/both/'+method+'_vs_grid']['difference']
    write(RUN/'AUDIT.json',dict(status='PASS',ridge_fits=96,neural_updates=0,prediction_arrays_checked=checked,all_scalar_metrics_recomputed=True,selector_replay_max_px=selector_replay,physical_line_V36_reproduction_max_px=line_error,normal_identity_max_px=identity,sign_invariance_max_px=sign_error,fit_repeat_max=fitreplay,check_GT_and_features_perturbation_train_fit_unchanged=True,train_groups=16,check_groups=8,check_detected_frames=130,check_supported_points=820,check_all_detected_points=1040,prior_files_unchanged=True,limitations='Audit of cached geometry/fit/masks, not new physical GT annotation review. Four gain fits repeated, not all96 twice; selectors all replayed.'));print('AUDIT37 PASS',checked,flush=True)

if __name__=='__main__':main()
