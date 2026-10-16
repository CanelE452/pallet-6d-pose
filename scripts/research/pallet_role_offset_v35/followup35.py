"""Post-primary robustness audit; preserve the registered primary results."""
from context35 import *
from analyze35 import INC,features,bias,ci

NEW=('top4_oracle_median','gt_joint_median','pred_joint_median','best4_joint_median','pred_ridge_bounded','free_pred_ridge_bounded','free_pred_median')

def project(A,b):
    u,s,v=np.linalg.svd(A,full_matrices=False);keep=s>s[0]*.1
    return v[keep].T@((u[:,keep].T@b)/s[keep])

def main():
    assert read(RUN/'ANALYSIS_COMPLETE.json')['complete'];write(RUN/'FOLLOWUP_PROTOCOL.json',dict(exploratory=True,reason='Primary predicted ridge offsets exploded even on train. Isolate calibration extrapolation from line extraction using training-target range clipping and constant offsets, and compare multiple incident lines. Original protocol/summary unchanged.',methods=NEW,bounds='Per point/role training physical delta/diag min,max only; no check GT used. Ridge output bounded, not full point displacement. Joint constraints SVD relative0.1 as registered ceiling. Candidate best4 still GT-line oracle.',primary_summary_sha256=sha(RUN/'SUMMARY.json'),source_sha256=sha(KIT/'followup35.py')))
    rows=read(RUN/'POPULATION.json')['rows'];g=np.load(RUN/'geometry.npz');gt=g['gt'];n,rho,length=g['normal'],g['rho'],g['length'];N=len(rows);present=length>0;train=np.array([r['split']=='train' for r in rows]);groups=np.array([r['source_group'] for r in rows]);cond=np.array([r['condition'] for r in rows]);bb=np.array([r['box'] for r in rows]);wh=np.maximum(bb[:,2:]-bb[:,:2],1);diag=np.linalg.norm(wh,axis=-1);center=(bb[:,:2]+bb[:,2:])/2;aspect=np.log(wh[:,0]/wh[:,1]);models=read(RUN/'OFFSET_CALIBRATION.json')['models'];ranges={};xrange={};ext=[];arrays=[]
    for j in range(8):
      for r in INC[j]:
        mask=train&present[:,r];target=((n[:,r]*gt[:,j]).sum(-1)-rho[:,r])/diag;X=features(n[:,r],aspect);ranges[f'{j}_{r}']=np.quantile(target[mask],[0,1]).tolist() if mask.any() else [0.,0.];xrange[f'{j}_{r}']=(X[mask].min(0),X[mask].max(0)) if mask.any() else (np.zeros(8),np.zeros(8))
    for seed in SEEDS:
        z=np.load(RUN/f'pred_seed{seed}.npz');p=np.load(RUN/f'analysis_seed{seed}.npz');base=p['base'];roles=p['selected_role'];supported=p['supported'];pn=z['line_normal'].reshape(N,12,4,2).astype(float);pr=z['line_offset'].reshape(N,12,4).astype(float)*diag[:,None,None]+(pn*center[:,None,None]).sum(-1);score=z['score'].reshape(N,12,4);idx=np.arange(N)[:,None];rr=np.arange(12)[None];top=score.argmax(-1);sn=pn[idx,rr,top];sr=pr[idx,rr,top];d=np.abs((pn[:,:,:,None]*g['support'][:,:,None]).sum(-1)-pr[:,:,:,None]).mean(-1);best=d.argmin(-1);bn=pn[idx,rr,best];br=pr[idx,rr,best];out={k:base.copy() for k in NEW};rawdelta=np.zeros((N,8));bounded=np.zeros_like(rawdelta);isextra=np.zeros_like(supported)
        for j in range(8):
            free=np.array(INC[j])[score[:,INC[j]].max(-1).argmax(-1)]
            for choice,name in [(roles[:,j],'pred_ridge_bounded'),(free,'free_pred_ridge_bounded'),(free,'free_pred_median'),(roles[:,j],'top4_oracle_median')]:
                normal=(bn if name=='top4_oracle_median' else sn)[np.arange(N),choice];offset=(br if name=='top4_oracle_median' else sr)[np.arange(N),choice];predbias=np.zeros(N)
                for r in INC[j]:
                    mm=choice==r;key=f'{j}_{r}';kind='median' if 'median' in name else 'ridge';delta=bias(models[key],normal,aspect,kind);clipped=np.clip(delta,*ranges[key]) if kind=='ridge' else delta;predbias[mm]=clipped[mm]*diag[mm]
                    if name=='pred_ridge_bounded':
                        rawdelta[mm,j]=delta[mm]*diag[mm];bounded[mm,j]=clipped[mm]*diag[mm];X=features(normal,aspect);lo,hi=xrange[key];isextra[mm,j]=((X<lo-1e-6)|(X>hi+1e-6)).any(-1)[mm]
                selected=np.ones(N,bool) if name.startswith('free') else supported[:,j];amount=offset-(normal*base[:,j]).sum(-1)+predbias;out[name][selected,j]=base[selected,j]+normal[selected]*amount[selected,None]
            for i in range(N):
                rs=[r for r in INC[j] if present[i,r]]
                if not rs:continue
                b=np.array([models[f'{j}_{r}']['median'] for r in rs])*diag[i]
                for name,normal,offset in [('gt_joint_median',n,rho),('pred_joint_median',sn,sr),('best4_joint_median',bn,br)]:
                    A=normal[i,rs];target=offset[i,rs]+b-A@base[i,j];out[name][i,j]=base[i,j]+project(A,target)
        errors=np.stack([np.linalg.norm(out[k]-gt,axis=-1) for k in NEW]);arrays.append(errors);mask=(~train)[:,None]&supported;ext.append(dict(seed=seed,out_of_train_feature_range_fraction=float(isextra[mask].mean()),raw_offset_abs_p90_px=float(np.quantile(np.abs(rawdelta[mask]),.9)),raw_offset_abs_max_px=float(np.abs(rawdelta[mask]).max()),bounded_offset_abs_max_px=float(np.abs(bounded[mask]).max())));np.savez_compressed(RUN/f'followup_seed{seed}.npz',**out,errors=errors,raw_offset=rawdelta,bounded_offset=bounded,extrapolated=isextra)
    arrays=np.array(arrays);baseerr=np.stack([np.load(RUN/f'analysis_seed{s}.npz')['errors'][0] for s in SEEDS]);results={};extra={}
    for split,take in [('train',train),('check',~train)]:
      for c in ['all',*sorted(set(cond))]:
        frame=take&(True if c=='all' else cond==c)
        for typ,mask0 in [('supported',supported),('all_detected',np.ones_like(supported)),('outside_supported',supported&g['outside']),('hidden_supported',supported&g['hidden'])]:
            mask=frame[:,None]&mask0
            if not mask.any():continue
            rr={}
            for mi,k in enumerate(NEW):
                per=arrays[:,mi][:,mask].mean(-1);rr[k]=dict(mean_px=float(per.mean()),per_seed_px=per.tolist(),gain_percent=((1-per/baseerr[:,mask].mean(-1))*100).tolist(),difference=ci(arrays[:,mi]-baseerr,mask,groups))
            results[f'{split}/{c}/{typ}']=dict(points=int(mask.sum()),methods=rr)
        mask=frame[:,None]&supported
        if mask.any():
            p=np.load(RUN/'analysis_seed1.npz');ne=p['normal_errors'];extra[f'{split}/{c}']=dict(median_vs_zero=ci(ne[1]-ne[0],mask,groups),rank2_fraction=float((p['rank'][mask]==2).mean()),single_exact_normal_error_px=float(np.mean([np.linalg.norm(np.load(RUN/f'analysis_seed{s}.npz')['normal_oracle']-gt,axis=-1)[mask].mean() for s in SEEDS])))
    write(RUN/'FOLLOWUP.json',dict(complete=True,exploratory=True,results=results,offset_ranges=ranges,extrapolation=ext,extra=extra));write(RUN/'FOLLOWUP_COMPLETE.json',dict(complete=True,files={str(p.relative_to(ROOT)):sha(p) for p in [RUN/'FOLLOWUP.json',*RUN.glob('followup_seed*.npz')]}));print('FOLLOWUP35 COMPLETE',flush=True)
if __name__=='__main__':main()
