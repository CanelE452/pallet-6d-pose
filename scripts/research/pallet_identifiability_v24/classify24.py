"""Direction information probe; grouped AUROC bootstrap and unchanged geometric metrics."""
from context24 import *
from eval24 import population,prediction_path,SPLITS
from sklearn.metrics import roc_auc_score,average_precision_score
PROBES=ARMS+('unit_bce','unit_mass')
def metrics(y,p):
    y=y.astype(bool);n=len(y)
    if not n:return dict(points=0,auc=None)
    both=y.any() and (~y).any()
    return dict(points=n,positive_fraction=float(y.mean()),auc=float(roc_auc_score(y,p)) if both else None,average_precision=float(average_precision_score(y,p)) if y.any() else None,balanced_accuracy=float(((p[y]>=.5).mean()+(p[~y]<.5).mean())/2) if both else None,brier=float(np.mean((p-y)**2)),mean_positive_score=float(p[y].mean()) if y.any() else None,mean_negative_score=float(p[~y].mean()) if (~y).any() else None)
def clustered_auc(y,p,groups,levels,draws):
    y=y.astype(bool);pos=np.array([((groups==g)&y).sum() for g in levels]);neg=np.array([((groups==g)&~y).sum() for g in levels]);A=np.zeros((len(levels),len(levels)))
    for i,g in enumerate(levels):
        pp=p[(groups==g)&y]
        for j,h in enumerate(levels):
            nn=np.sort(p[(groups==h)&~y]);A[i,j]=(.5*(np.searchsorted(nn,pp,'left')+np.searchsorted(nn,pp,'right'))).sum()
    denominator=(draws@pos)*(draws@neg);numerator=np.einsum('bi,ij,bj->b',draws,A,draws);return np.divide(numerator,denominator,out=np.full(len(draws),np.nan),where=denominator>0)
def main():
    setup();assert read(RUN/'INFERENCE_COMPLETE.json')['complete'];results={};gates={};replays=[]
    # Independent weighted-rank validation of the cluster formula, including ties.
    y=np.array([1,0,1,0,1,0]);p=np.array([.5,.5,.9,.2,.4,.8]);g=np.array([0,0,1,1,2,2]);draw=np.array([[1,1,1],[2,0,1],[0,2,1]]);got=clustered_auc(y,p,g,np.arange(3),draw)
    for x,v in zip(draw,got):assert abs(v-roc_auc_score(y,p,sample_weight=x[g]))<1e-12
    for split in ('training',)+SPLITS:
        for c in (TRAIN_CONDITIONS if split in ('training','new_check','prospective') else CONDITIONS):
            records={};raw={};levels=None
            if split!='training':
                m,ts=population(split,c);gt=np.array([t['points'] for t in ts]);valid=np.array([t['valid'] for t in ts]);kind=C.P.target_type(ts).numpy();session=np.array([t['session'] for t in ts]);levels=np.unique(session)
            for arm in (ARMS if split=='training' else PROBES):
                for seed in SEEDS:
                    if split=='training':
                        z=torch.load(RUN/'models'/f'{arm}_seed{seed}'/'TRAIN_PREDICTIONS.pt',weights_only=True);sel=np.array([x.endswith('::'+c) for x in z['ids']]);score=z['score'].numpy()[sel];target=z['target'].numpy()[sel].astype(bool);mask=z['mask'].numpy()[sel];kind=z['kind'].numpy()[sel]
                    else:
                        z=np.load(prediction_path(split,c,f'{arm}_seed{seed}'));score=z['alpha'].astype(float);b=z['base'].astype(float);delta=z['proposal_delta'].astype(float);residual=gt-b;target=(residual*delta).sum(-1)>0;mask=valid&z['detected'][:,None]&(np.linalg.norm(residual,axis=-1)>.1)&(np.linalg.norm(delta,axis=-1)>.1)
                    rs={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        keep=mask if k is None else mask&(kind==k);y=target[keep];p=score[keep];rs[typ]=metrics(y,p)
                        if split=='real':raw[(arm,seed,typ)]=(y,p,np.broadcast_to(session[:,None],mask.shape)[keep])
                        if split!='training':
                            margin=np.abs((residual*delta).sum(-1))/np.maximum(np.linalg.norm(delta,axis=-1),1e-12)
                            for radius in (2,5):
                                robust=keep&(margin>radius);rs[typ+'_margin'+str(radius)]=metrics(target[robust],score[robust])
                    records[f'{arm}_seed{seed}']=rs
            out=dict(models=records)
            if split=='real':
                intervals={};pairs={}
                for typ in ('all','hidden','outside'):
                    rng=np.random.default_rng(int(digest('v24auc:'+c+typ)[:8],16));draws=rng.multinomial(len(levels),np.ones(len(levels))/len(levels),size=5000);boot={}
                    for arm in PROBES:
                        x=np.stack([clustered_auc(*raw[(arm,seed,typ)],levels,draws) for seed in SEEDS]);good=np.isfinite(x).all(0);boot[arm]=x.mean(0);aucs=[records[f'{arm}_seed{seed}'][typ]['auc'] for seed in SEEDS];intervals[arm+'/'+typ]=dict(per_seed_auc=aucs,mean_auc=float(np.mean(aucs)) if all(v is not None for v in aucs) else None,ci95=np.quantile(boot[arm][good],[.025,.975]).tolist() if good.any() else None,valid_bootstrap_draws=int(good.sum()))
                    for a,b in [('unit_sign','unit_sign_local'),('unit_sign','unit_sign_parent'),('unit_sign','unit_bce'),('unit_sign','unit_mass')]:
                        diff=boot[a]-boot[b];good=np.isfinite(diff);aa=intervals[a+'/'+typ]['per_seed_auc'];bb=intervals[b+'/'+typ]['per_seed_auc'];gains=[x-y for x,y in zip(aa,bb)] if all(x is not None for x in aa+bb) else None;ci=np.quantile(diff[good],[.025,.975]).tolist() if good.any() else None;pairs[a+'_vs_'+b+'/'+typ]=dict(per_seed_auc_gain=gains,ci95=ci,extra_signal_pass=gains is not None and min(gains)>=.02 and ci is not None and ci[0]>0)
                out.update(intervals=intervals,pairs=pairs)
                v=intervals['unit_sign/outside'];gates[c]=dict(signal_pass=all(x is not None and x>=.70 for x in v['per_seed_auc']) and v['ci95'] is not None and v['ci95'][0]>.5,extra_line_pass=pairs['unit_sign_vs_unit_sign_local/outside']['extra_signal_pass'])
            results[split+'/'+c]=out;print('CLASSIFIED',split,c,flush=True)
    write(RUN/'CLASSIFICATION.json',dict(complete=True,results=results,gates=gates,bootstrap_known_vector_pass=True,scope='Sign of infinitesimal squared-distance improvement; positive sign does not guarantee full-step improvement. Class-balanced scores are uncalibrated;0.5 is a fixed diagnostic threshold. All evaluation sets reused DEV. Absence of probe signal does not prove no information exists. Additional posthoc margin2/margin5 subsets have direction labels invariant to any GT perturbation within2/5px Euclidean ball; not an estimate of actual annotation error and never used to select inference points.'))
if __name__=='__main__':main()
