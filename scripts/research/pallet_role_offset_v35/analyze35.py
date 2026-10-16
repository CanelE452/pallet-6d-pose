from context35 import *

INC=[[r for r,(a,b) in enumerate(EDGES) if j in (a,b)] for j in range(8)]
METHODS=('base','strong','v34','gt_zero','gt_median','gt_ridge','pred_zero','pred_median','pred_ridge','top4_oracle_ridge','virtual_identity','normal_oracle','joint_oracle','free_pred_ridge')

def features(n,aspect):
    x,y=n[...,0],n[...,1];a=np.broadcast_to(aspect,n.shape[:-1]);return np.stack((x,y,x*x,x*y,y*y,a,x*a,y*a),-1)

def fit_offsets(n,rho,gt,diag,aspect,present,train):
    models={}
    for j in range(8):
        for role in INC[j]:
            take=train&present[:,role];target=((n[:,role]*gt[:,j]).sum(-1)-rho[:,role])/diag;X=features(n[:,role],aspect)
            if take.any():
                center=X[take].mean(0);scale=X[take].std(0);scale[scale<1e-6]=1.;A=np.c_[np.ones(take.sum()),(X[take]-center)/scale];penalty=np.eye(A.shape[1])*.001;penalty[0,0]=0;coef=np.linalg.solve(A.T@A/len(A)+penalty,A.T@target[take]/len(A));median=float(np.median(target[take]))
            else:center=np.zeros(8);scale=np.ones(8);coef=np.zeros(9);median=0.
            models[f'{j}_{role}']=dict(count=int(take.sum()),median=median,center=center.tolist(),scale=scale.tolist(),coef=coef.tolist())
    return models

def bias(model,normal,aspect,kind):
    if kind=='median':return np.full(len(normal),model['median'])
    X=(features(normal,aspect)-model['center'])/model['scale'];return np.c_[np.ones(len(X)),X]@model['coef']

def ci(values,mask,groups):
    if values.ndim==3:values=values.mean(0)
    groupnames=np.unique(groups);ss=np.array([values[(groups==g)[:,None]&mask].sum() for g in groupnames]);cc=np.array([((groups==g)[:,None]&mask).sum() for g in groupnames]);keep=cc>0;ss=ss[keep];cc=cc[keep]
    if not len(cc):return dict(mean=None,ci95=None,groups=0)
    rng=np.random.default_rng(35001);ii=rng.integers(0,len(cc),(5000,len(cc)));z=ss[ii].sum(-1)/cc[ii].sum(-1)
    return dict(mean=float(ss.sum()/cc.sum()),ci95=np.quantile(z,[.025,.975]).tolist(),groups=len(cc))

def main():
    register();assert read(RUN/'EXTRACTION_COMPLETE.json')['complete'];write(RUN/'ANALYSIS_PROTOCOL.json',dict(files={str(p.relative_to(ROOT)):sha(p) for p in [KIT/'analyze35.py',KIT/'context35.py']},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    rows=read(RUN/'POPULATION.json')['rows'];N=len(rows);g=np.load(RUN/'geometry.npz');gt=g['gt'];n,rho,length=g['normal'],g['rho'],g['length'];present=length>0;bb=np.array([r['box'] for r in rows]);wh=np.maximum(bb[:,2:]-bb[:,:2],1);diag=np.linalg.norm(wh,axis=-1);center=(bb[:,:2]+bb[:,2:])/2;aspect=np.log(wh[:,0]/wh[:,1]);train=np.array([r['split']=='train' for r in rows]);check=~train;groups=np.array([r['source_group'] for r in rows]);condition=np.array([r['condition'] for r in rows]);models=fit_offsets(n,rho,gt,diag,aspect,present,train);write(RUN/'OFFSET_CALIBRATION.json',dict(models=models,train_groups=sorted(set(groups[train])),check_groups=sorted(set(groups[check])),ridge_lambda=.001,neural_updates=0,fit_pairs=len(models)))
    roles=np.zeros((N,8),int);supported=np.zeros((N,8),bool)
    for j in range(8):roles[:,j]=np.array(INC[j])[length[:,INC[j]].argmax(-1)];supported[:,j]=present[:,INC[j]].any(-1)
    ii=np.arange(N)[:,None];gn=n[ii,roles];gr=rho[ii,roles];delta=(gn*gt).sum(-1)-gr;allpred=[];normalerr=[];metrics=[];rank=np.zeros((N,8),int);nullfraction=[]
    for seed in SEEDS:
        z=np.load(RUN/f'pred_seed{seed}.npz');base=z['base'].astype(float);pn=z['line_normal'].reshape(N,12,4,2).astype(float);pr=z['line_offset'].reshape(N,12,4).astype(float)*diag[:,None,None]+(pn*center[:,None,None]).sum(-1);score=z['score'].reshape(N,12,4).astype(float)
        candidate=score.argmax(-1);rindex=np.arange(12)[None];pnormal=pn[np.arange(N)[:,None],rindex,candidate];prho=pr[np.arange(N)[:,None],rindex,candidate];sn=pnormal[ii,roles];sr=prho[ii,roles]
        # GT line oracle: candidates compared only to physical support endpoints, not point GT.
        physical_support=g['support'];dist=np.abs((pn[:,:,:,None,:]*physical_support[:,:,None]).sum(-1)-pr[:,:,:,None]).mean(-1);best=dist.argmin(-1);on=pn[np.arange(N)[:,None],rindex,best][ii,roles];orr=pr[np.arange(N)[:,None],rindex,best][ii,roles]
        pp={k:base.copy() for k in METHODS};pp.update(strong=z['strong'].astype(float),v34=z['points'].astype(float));errs={}
        for j in range(8):
            take=supported[:,j];rr=roles[:,j];med=np.zeros(N);reg=np.zeros(N);preg=np.zeros(N);omed=np.zeros(N)
            for r in INC[j]:
                selected=rr==r;m=models[f'{j}_{r}'];med[selected]=bias(m,gn[:,j],aspect,'median')[selected]*diag[selected];reg[selected]=bias(m,gn[:,j],aspect,'ridge')[selected]*diag[selected];preg[selected]=bias(m,sn[:,j],aspect,'ridge')[selected]*diag[selected];omed[selected]=bias(m,on[:,j],aspect,'ridge')[selected]*diag[selected]
            for name,normal,offset,b in [('gt_zero',gn[:,j],gr[:,j],0),('gt_median',gn[:,j],gr[:,j],med),('gt_ridge',gn[:,j],gr[:,j],reg),('pred_zero',sn[:,j],sr[:,j],0),('pred_median',sn[:,j],sr[:,j],med),('pred_ridge',sn[:,j],sr[:,j],preg),('top4_oracle_ridge',on[:,j],orr[:,j],omed),('normal_oracle',gn[:,j],gr[:,j],delta[:,j])]:
                amount=offset-(normal*base[:,j]).sum(-1)+b;pp[name][take,j]=base[take,j]+normal[take]*amount[take,None]
            # Virtual GT edge through target: algebra/control, never a candidate model.
            counterpart=np.array([EDGES[r][1] if EDGES[r][0]==j else EDGES[r][0] for r in rr]);v=gt[np.arange(N),counterpart]-gt[:,j];vn=np.stack((-v[:,1],v[:,0]),-1);vn/=np.maximum(np.linalg.norm(vn,axis=-1,keepdims=True),1e-12);pp['virtual_identity'][take,j]=base[take,j]+vn[take]*((gt[:,j]-base[:,j])*vn).sum(-1)[take,None]
            # Fully GT-free role choice: highest predicted incident-role confidence.
            free_role=np.array(INC[j])[score[:,INC[j]].max(-1).argmax(-1)];fn=pnormal[np.arange(N),free_role];fr=prho[np.arange(N),free_role];fb=np.zeros(N)
            for r in INC[j]:
                sel=free_role==r;fb[sel]=bias(models[f'{j}_{r}'],fn,aspect,'ridge')[sel]*diag[sel]
            pp['free_pred_ridge'][:,j]=base[:,j]+fn*(fr-(fn*base[:,j]).sum(-1)+fb)[:,None]
            for i in range(N):
                rr0=[r for r in INC[j] if present[i,r]]
                if not rr0:continue
                A=n[i,rr0];u,sv,vt=np.linalg.svd(A,full_matrices=False);keep=sv>sv[0]*.1;rank[i,j]=int(keep.sum());error=gt[i,j]-base[i,j];pp['joint_oracle'][i,j]=base[i,j]+vt[keep].T@(vt[keep]@error)
            errs[j]=dict(zero=np.abs(delta[:,j]),median=np.abs(delta[:,j]-med),ridge=np.abs(delta[:,j]-reg))
        errors=np.stack([np.linalg.norm(pp[k]-gt,axis=-1) for k in METHODS]);normalerr.append(np.stack([np.stack([errs[j][k] for j in range(8)],-1) for k in ['zero','median','ridge']]))
        # Exact normal oracle is an orthogonal projection; cannot increase point error.
        assert np.all(errors[METHODS.index('normal_oracle')][supported]<=errors[0][supported]+1e-8)
        assert np.all(errors[METHODS.index('joint_oracle')][supported]<=errors[0][supported]+1e-8)
        angle=np.degrees(np.arccos(np.clip(np.abs((sn*gn).sum(-1)),0,1)));dtop=dist[np.arange(N)[:,None],np.arange(12)[None],candidate][ii,roles];dbest=dist.min(-1)[ii,roles]
        metrics.append(dict(seed=seed,angle_deg=angle,support_distance_top1=dtop,support_distance_best4=dbest));np.savez_compressed(RUN/f'analysis_seed{seed}.npz',**pp,errors=errors,selected_role=roles,supported=supported,gt_normal=gn,gt_rho=gr,gt_bias_px=delta,pred_normal=sn,pred_rho=sr,rank=rank,normal_errors=normalerr[-1],angle_deg=angle,support_distance_top1=dtop,support_distance_best4=dbest)
        allpred.append(errors)
    allpred=np.array(allpred);normalerr=np.array(normalerr);results={};coverage={}
    for split,take0 in [('train',train),('check',check)]:
      for c in ['all',*sorted(set(condition))]:
        frame=take0&(True if c=='all' else condition==c)
        for stratum,extra in [('supported',supported),('all_detected',np.ones_like(supported)),('outside_supported',supported&g['outside']),('hidden_supported',supported&g['hidden'])]:
            mask=frame[:,None]&extra
            if not mask.any():continue
            rr={}
            for mi,k in enumerate(METHODS):
                per=allpred[:,mi][:,mask].mean(-1);deltaerr=allpred[:,mi]-allpred[:,0];rr[k]=dict(mean_px=float(per.mean()),per_seed_px=per.tolist(),gain_percent=((1-per/allpred[:,0][:,mask].mean(-1))*100).tolist(),**{'difference':ci(deltaerr,mask,groups)})
            results[f'{split}/{c}/{stratum}']=dict(points=int(mask.sum()),frames=int(mask.any(-1).sum()),methods=rr)
        mask=frame[:,None]&supported
        if mask.any():
            requested=[r for r in read(V32.RUN/'cache/manifest.json')['rows'] if r['split']==split and (c=='all' or r['condition']==c)];coverage[f'{split}/{c}']=dict(requested_frames=len(requested),detected_frames=int(frame.sum()),supported_points=int(mask.sum()),all_detected_points=int(frame.sum()*8),rank1=int((mask&(rank==1)).sum()),rank2=int((mask&(rank==2)).sum()),gt_abs_bias_median_px=float(np.median(np.abs(delta[mask]))),gt_abs_bias_p90_px=float(np.quantile(np.abs(delta[mask]),.9)),normal_mae_px={k:float(normalerr[:,i][:,mask].mean()) for i,k in enumerate(['zero','median','ridge'])},offset_ridge_vs_zero=ci(normalerr[:,2]-normalerr[:,0],mask,groups),prediction_angle_median_deg=float(np.median(np.stack([m['angle_deg'] for m in metrics])[:,mask])),prediction_distance_mean_px=float(np.stack([m['support_distance_top1'] for m in metrics])[:,mask].mean()),best4_oracle_distance_mean_px=float(np.stack([m['support_distance_best4'] for m in metrics])[:,mask].mean()))
    key=coverage['check/all'];offpass=key['normal_mae_px']['ridge']<=.9*key['normal_mae_px']['zero'] and key['offset_ridge_vs_zero']['ci95'][1]<0
    paired=results['check/all/supported']['methods']['pred_ridge'];free=results['check/all/all_detected']['methods']['free_pred_ridge'];transfer=lambda r:min(r['gain_percent'])>=1 and r['difference']['ci95'][1]<0
    write(RUN/'SUMMARY.json',dict(complete=True,neural_updates=0,offset_fits=len(models),results=results,coverage=coverage,gates=dict(gt_offset_explained=bool(offpass),matched_pred_transfer=bool(transfer(paired)),GT_free_transfer=bool(transfer(free))),limits=read(RUN/'PROTOCOL.json')['limits']))
    write(RUN/'ANALYSIS_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'SUMMARY.json',RUN/'OFFSET_CALIBRATION.json',*RUN.glob('analysis_seed*.npz')]}));print('ANALYSIS35',read(RUN/'SUMMARY.json')['gates'],flush=True)
if __name__=='__main__':main()
