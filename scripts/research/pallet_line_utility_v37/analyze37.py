from context37 import *
import torch


def load_seed(seed,p):
    N=p['N'];lookup={r['id']:i for i,r in enumerate(p['rows'])};out={};query=np.zeros((N,8,32));visual=np.zeros((N,12,8,32));seen=np.zeros(N,int)
    paths=[V36/'cache'/f'training_seed{seed}.pt']+sorted((V36/'cache/render_check').glob(f'*/seed{seed}.pt'))
    for path in paths:
        z=torch.load(path,map_location='cpu',weights_only=True)
        for k,name in enumerate(z['ids']):
            if name not in lookup:continue
            i=lookup[name];query[i]=z['x']['query'][k].numpy().reshape(8,2,32).mean(1);visual[i]=z['x']['features'][k,:,:32].numpy().reshape(12,8,32);seen[i]+=1
    assert np.all(seen==1)
    old=dict(np.load(V35/f'analysis_seed{seed}.npz'));out.update({k:old[k] for k in ('base','strong','v34')});out.update(query=query,visual=visual)
    for arm in ARMS:
        pn=np.zeros((N,12,8,2));pr=np.zeros((N,12,8));ps=np.zeros_like(pr);points=np.zeros((N,8,2));seen=np.zeros(N,int)
        for path in [V36/'training_predictions'/f'{arm}_seed{seed}.npz']+sorted((V36/'predictions/render_check').glob(f'*/{arm}_seed{seed}.npz')):
            z=np.load(path);take=z['detected'] if 'detected' in z else np.ones(len(z['ids']),bool);ix=np.array([lookup[i] for i in z['ids'][take]]);pn[ix]=z['normal'][take].reshape(-1,12,8,2);pr[ix]=z['offset'][take].reshape(-1,12,8)*p['diag'][ix,None,None]+(pn[ix]*p['center'][ix,None,None]).sum(-1);ps[ix]=z['score'][take].reshape(-1,12,8);points[ix]=z['points'][take];seen[ix]+=1
        assert np.all(seen==1) and np.allclose(np.linalg.norm(pn,axis=-1),1,atol=2e-6)
        # Keep n.rho sign consistent with train-median canonical physical offsets.
        sign=np.where(pn[...,1]<0,-1.,1.);pn*=sign[...,None];pr*=sign
        out[arm]=dict(normal=pn,rho=pr,score=ps,points=points)
    return out


def fit(X,y,train,groups):
    # X=N,8,24,F. Give each source group equal total weight; no check inputs used.
    xx=X[train].reshape(-1,X.shape[-1]);yy=y[train].reshape(-1);gg=np.repeat(groups[train],X.shape[1]*X.shape[2]);un,c=np.unique(gg,return_counts=True);w=np.array([1/c[np.where(un==g)[0][0]] for g in gg]);w/=w.sum()
    center=(xx*w[:,None]).sum(0);scale=np.maximum(np.sqrt(((xx-center)**2*w[:,None]).sum(0)),.05);A=np.c_[np.ones(len(xx)),np.clip((xx-center)/scale,-10,10)];pen=np.eye(A.shape[1])*.01;pen[0,0]=0;coef=np.linalg.solve(A.T@(A*w[:,None])+pen,A.T@(yy*w))
    return dict(center=center,scale=scale,coef=coef)


def predict(m,X):
    return m['coef'][0]+np.clip((X-m['center'])/m['scale'],-10,10)@m['coef'][1:]


def features(p,z,anchor,normal,amount,score,roles,visual):
    N=p['N'];diag=p['diag'][:,None,None];cent=p['center'][:,None];pp=(anchor-cent)/diag;pa=np.concatenate([pp,*[(z[k]-cent)/diag for k in ('base','strong','v34')],z['query'],np.broadcast_to(p['aspect'][:,None,None],(N,8,1)),np.broadcast_to(np.eye(8),(N,8,8))],-1)
    common=np.concatenate([np.broadcast_to(pa[:,:,None],(N,8,24,pa.shape[-1])),np.broadcast_to(np.eye(12)[roles],(N,8,24,12)),score[...,None],np.broadcast_to(np.tile(np.arange(8)/7,3)[None,None,:,None],(N,8,24,1))],-1)
    a=amount/p['diag'][:,None,None];d=normal*a[...,None];deltas=np.stack([(z[k]-anchor)/diag for k in ('base','strong','v34')],-2);dot=np.einsum('njkd,njld->njkl',normal,deltas)
    geo=np.concatenate([normal,a[...,None],np.abs(a)[...,None],a[...,None]**2,d,d**2,d[...,0:1]*d[...,1:2],dot,dot*a[...,None],score[...,None]*a[...,None]],-1)
    return dict(point=common,visual=np.concatenate([common,visual],-1),geometry=np.concatenate([common,geo],-1),visual_geometry=np.concatenate([common,visual,geo],-1))


def rowstats(errors,reference,mask,groups):
    per=errors[:,mask].mean(-1);ref=reference[:,mask].mean(-1)
    return dict(mean_px=float(per.mean()),per_seed_px=per.tolist(),gain_percent=((1-per/ref)*100).tolist(),difference=ci(errors-reference,mask,groups))


def masks(p):
    for split,t in [('train',p['train']),('check',~p['train'])]:
      for cond in ['all',*sorted(set(p['condition']))]:
        f=t&(True if cond=='all' else p['condition']==cond)
        for name,m in [('all_detected',np.ones((p['N'],8),bool)),('supported',p['supported']),('outside',p['g']['outside']),('hidden',p['g']['hidden'])]:
            mask=f[:,None]&m
            if mask.any():yield split+'/'+cond+'/'+name,mask


def main():
    register();p=population();g=p['g'];N=p['N'];models=read(V35/'OFFSET_CALIBRATION.json')['models'];roles=p['roles'];idx=np.arange(N)[:,None];inc=np.array(INC);assert inc.shape==(8,3)
    write(RUN/'ANALYSIS_BINDINGS.json',dict(files={str(q.relative_to(ROOT)):sha(q) for q in KIT.glob('*.py')},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    (RUN/'arrays').mkdir(exist_ok=True);(RUN/'fits').mkdir(exist_ok=True);errors={};direction={};fit_count=0
    for seed in SEEDS:
      z=load_seed(seed,p);np.savez_compressed(RUN/'arrays'/f'inputs_seed{seed}.npz',**{k:z[k] for k in ('base','strong','v34')})
      for arm in ARMS:
        line=z[arm];pn,pr,score=[line[k] for k in ('normal','rho','score')];distance=np.abs((pn[:,:,:,None]*g['support'][:,:,None]).sum(-1)-pr[:,:,:,None]).mean(-1);top=score.argmax(-1);best=distance.argmin(-1)
        ni=pn[:,inc].reshape(N,8,24,2);ri=pr[:,inc].reshape(N,8,24);si=score[:,inc].reshape(N,8,24);vi=z['visual'][:,inc].reshape(N,8,24,32);ir=np.repeat(inc,8,axis=1);median=np.array([[models[f'{j}_{r}']['median'] for r in ir[j]] for j in range(8)])*p['diag'][:,None,None]
        for an in ANCHORS:
            anchor=z[an];full_amount=ri+median-(ni*anchor[:,:,None]).sum(-1);amount=np.clip(full_amount,-.15*p['diag'][:,None,None],.15*p['diag'][:,None,None]);candidate=anchor[:,:,None]+ni*amount[...,None];full=anchor[:,:,None]+ni*full_amount[...,None];baseerr=np.linalg.norm(anchor-g['gt'],axis=-1);ce=np.linalg.norm(candidate-g['gt'][:,:,None],axis=-1);fe=np.linalg.norm(full-g['gt'][:,:,None],axis=-1);pp=dict(anchor=anchor.copy(),strong=z['strong'],v34=z['v34'],v36_point=line['points']);selected={}
            for which,kk in [('top',top),('physical_oracle',best)]:
                nr=pn[idx,np.arange(12)[None],kk][idx,roles];rr=pr[idx,np.arange(12)[None],kk][idx,roles];b=np.array([[models[f'{j}_{roles[i,j]}']['median'] for j in range(8)] for i in range(N)])*p['diag'][:,None];a=rr+b-(nr*anchor).sum(-1);pred=anchor+nr*a[...,None];pp['matched_'+which]=np.where(p['supported'][...,None],pred,anchor)
            # GT-role point oracle may abstain; full projections preserve V35 comparison.
            match=ir[None]==roles[...,None];e=np.where(match,fe,np.inf);k=e.argmin(-1);choose=full[np.arange(N)[:,None],np.arange(8)[None],k];accept=(e.min(-1)<baseerr)&p['supported'];pp['matched_point_oracle']=np.where(accept[...,None],choose,anchor)
            for name,pool,err in [('free_point_oracle',candidate,ce),('free_full_oracle',full,fe)]:
                k=err.argmin(-1);pred=pool[np.arange(N)[:,None],np.arange(8)[None],k];accept=err.min(-1)<baseerr;pp[name]=np.where(accept[...,None],pred,anchor);selected[name]=np.where(accept,k,-1)
            gn=g['normal'][idx,roles];exact=(gn*(g['gt']-anchor)).sum(-1);pp['gt_normal_oracle']=np.where(p['supported'][...,None],anchor+gn*exact[...,None],anchor);pp['gt_joint_median']=anchor.copy();pp['gt_joint_oracle']=anchor.copy();rank=np.zeros((N,8),int)
            for i in range(N):
              for j in range(8):
                rs=[r for r in INC[j] if g['length'][i,r]>0]
                if not rs:continue
                A=g['normal'][i,rs];u,s,v=np.linalg.svd(A,full_matrices=False);keep=s>s[0]*.1;rank[i,j]=keep.sum();b=np.array([models[f'{j}_{r}']['median'] for r in rs])*p['diag'][i];d=g['rho'][i,rs]+b-A@anchor[i,j];pp['gt_joint_median'][i,j]=anchor[i,j]+v[keep].T@((u[:,keep].T@d)/s[keep]);e=g['gt'][i,j]-anchor[i,j];pp['gt_joint_oracle'][i,j]=anchor[i,j]+v[keep].T@(v[keep]@e)
            X=features(p,z,anchor,ni,amount,si,ir,vi);target=(baseerr[...,None]**2-ce**2)/p['diag'][:,None,None]**2;fitdata={};fitmetrics={}
            for mode in MODES:
                model=fit(X[mode],target,p['train'],p['groups']);value=predict(model,X[mode]);k=value.argmax(-1);accept=value.max(-1)>0;pred=candidate[np.arange(N)[:,None],np.arange(8)[None],k];pp['selector_'+mode]=np.where(accept[...,None],pred,anchor);selected['selector_'+mode]=np.where(accept,k,-1)
                fitdata.update({mode+'_'+key:val for key,val in model.items()});fit_count+=1
                fitmetrics[mode]=dict(features=X[mode].shape[-1],train_gain_mse=float(np.mean((value[p['train']]-target[p['train']])**2)),check_gain_mse=float(np.mean((value[~p['train']]-target[~p['train']])**2)))
            # Detailed point-role residual decomposition for the matched score-selected line.
            cn=pn[idx,np.arange(12)[None],top][idx,roles];cr=pr[idx,np.arange(12)[None],top][idx,roles];mb=np.array([[models[f'{j}_{roles[i,j]}']['median'] for j in range(8)] for i in range(N)])*p['diag'][:,None];proxy=cr+mb-(cn*anchor).sum(-1);desired=(cn*(g['gt']-anchor)).sum(-1);normal_error=proxy-desired;direction[(an,arm,seed)]=dict(proxy=proxy,desired=desired,error=normal_error,physical_distance=distance[idx,np.arange(12)[None],top][idx,roles],point_gain=baseerr-np.linalg.norm(pp['matched_top']-g['gt'],axis=-1))
            names=list(pp);ee=np.stack([np.linalg.norm(pp[k]-g['gt'],axis=-1) for k in names]);errors[(an,arm,seed)]=dict(zip(names,ee));out=RUN/'arrays'/f'{an}_{arm}_seed{seed}.npz';np.savez_compressed(out,**pp,errors=ee,names=np.array(names),candidate=candidate,full_candidate=full,candidate_normal=ni,candidate_role=ir,candidate_rho=ri,candidate_score=si,candidate_amount=amount,rank=rank,**{'selected_'+k:v for k,v in selected.items()},**direction[(an,arm,seed)]);np.savez_compressed(RUN/'fits'/f'{an}_{arm}_seed{seed}.npz',**fitdata);write(RUN/'fits'/f'{an}_{arm}_seed{seed}.json',fitmetrics)
            print('V37',seed,arm,an,'fits',fit_count,flush=True)
    results={};differences={};diagnostics={}
    for key,mask in masks(p):
        res={};comp={};dg={}
        for an in ANCHORS:
          for arm in ARMS:
            pref=an+'/'+arm;reference=np.stack([errors[(an,arm,s)]['anchor'] for s in SEEDS]);res[pref]={}
            for name in errors[(an,arm,1)]:
                e=np.stack([errors[(an,arm,s)][name] for s in SEEDS]);res[pref][name]=rowstats(e,reference,mask,p['groups'])
                if name in ('free_point_oracle','matched_top','matched_physical_oracle','selector_visual_geometry'):
                    ref=np.stack([errors[(an,'grid',s)][name] for s in SEEDS]);comp[pref+'/'+name+'_vs_grid']=rowstats(e,ref,mask,p['groups'])
            e=np.stack([errors[(an,arm,s)]['selector_visual_geometry'] for s in SEEDS]);ref=np.stack([errors[(an,arm,s)]['selector_visual'] for s in SEEDS]);comp[pref+'/geometry_vs_visual']=rowstats(e,ref,mask,p['groups'])
            support=mask&p['supported'];v=[direction[(an,arm,s)] for s in SEEDS]
            if support.any():
                a=np.stack([x['proxy'] for x in v])[:,support];t=np.stack([x['desired'] for x in v])[:,support];active=np.abs(t)>=1.;corr=float(np.corrcoef(a.ravel(),t.ravel())[0,1]);pd=np.stack([x['physical_distance'] for x in v])[:,support];pg=np.stack([x['point_gain'] for x in v])[:,support]
                dg[pref]=dict(points=int(support.sum()),normal_mae_px=float(np.abs(a-t).mean()),no_correction_mae_px=float(np.abs(t).mean()),sign_agreement_when_target_ge1px=float((np.sign(a[active])==np.sign(t[active])).mean()) if active.any() else None,active_samples=int(active.sum()),normal_correlation=corr,physical_distance_px=float(pd.mean()),distance_vs_point_gain_correlation=float(np.corrcoef(pd.ravel(),pg.ravel())[0,1]),point_improved_fraction=float((pg>0).mean()),point_harmed_over1px_fraction=float((pg<-1).mean()))
        results[key]=dict(points=int(mask.sum()),frames=int(mask.any(-1).sum()),methods=res);differences[key]=comp;diagnostics[key]=dg
    rr=results['check/all/all_detected']['methods'];cc=differences['check/all/all_detected'];gate=lambda r:min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0
    potential=gate(cc['v34/both/free_point_oracle_vs_grid']);r=rr['v34/both']['selector_visual_geometry'];added=cc['v34/both/geometry_vs_visual']['difference']['ci95'][1]<0;preserve=all(max(row['methods']['v34/both']['selector_visual_geometry']['per_seed_px'][i]/row['methods']['v34/both']['anchor']['per_seed_px'][i] for i in range(3))<=1.01 for k,row in results.items() if k.startswith('check/'))
    write(RUN/'SUMMARY.json',dict(complete=True,neural_updates=0,ridge_fits=fit_count,results=results,comparisons=differences,diagnostics=diagnostics,gates=dict(added_candidate_potential=bool(potential),observable_improves_anchor=bool(gate(r)),geometry_adds_over_visual=bool(added),all_condition_preservation=bool(preserve),observable_screen_passed=bool(gate(r) and added and preserve)),limits=read(RUN/'PROTOCOL.json')['limitations']))
    write(RUN/'ANALYSIS_COMPLETE.json',dict(complete=True,files={str(q.relative_to(ROOT)):sha(q) for q in [RUN/'SUMMARY.json',*sorted((RUN/'arrays').glob('*.npz')),*sorted((RUN/'fits').glob('*'))]}));print('V37 COMPLETE',read(RUN/'SUMMARY.json')['gates'],flush=True)

if __name__=='__main__':main()
