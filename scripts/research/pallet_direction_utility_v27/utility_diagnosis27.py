"""Subgroup-aligned utility and per-point selection checks, using saved predictions."""
from context27 import *
from eval27 import population
from sklearn.metrics import roc_auc_score

def main():
    rows=[]
    for split in ('prospective','synth_val','real'):
        c='crop20';m,t=population(split,c);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);outside=np.array([r['outside'] for r in t])
        for arm in ARMS:
            for seed in SEEDS:
                p=np.load(RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz');mask=valid&outside&p['detected'][:,None];before=np.linalg.norm(p['base']-gt,axis=-1);error=np.linalg.norm(p['base'][:,:,None]+p['alpha'][...,None]*p['candidates']-gt[:,:,None],axis=-1);gain=before[:,:,None]-error;cm=p['candidate_mask']==1;keep=mask[:,:,None]&cm;keep[:,:,0]=False;y=gain[keep]>.1;auc=float(roc_auc_score(y,p['utility'][keep])) if len(set(y))>1 else None
                best=np.where(cm,gain,-np.inf).max(-1);chosen=p['weights'].argmax(-1);selected=np.take_along_axis(gain,chosen[...,None],axis=-1)[...,0];nearbest=np.abs(selected-best)<=.1
                rows.append(dict(split=split,condition=c,arm=arm,seed=seed,subset='outside',points=int(mask.sum()),utility_help_auc=auc,chosen_near_best_fraction=float(nearbest[mask].mean()),chosen_gain_mean_px=float(selected[mask].mean()),GT_best_gain_at_predicted_amount_px=float(best[mask].mean()),chosen_harm_fraction=float((selected[mask]<-.1).mean()),positive_opportunity_points=int((mask&(best>.1)).sum()),scope='Candidate-wise AUC uses the same outside subgroup as training diagnosis. Candidate types and correlated points are pooled; this is not calibration or independent sample inference. Per-point selection uses argmax learned weights, while deployed output uses a soft mixture.'))
    write(RUN/'UTILITY_DIAGNOSIS.json',dict(complete=True,rows=rows,predictions_sha256=sha(RUN/'EVALUATION_BINDINGS.json')));print('UTILITY_DIAGNOSIS27',len(rows),flush=True)

if __name__=='__main__':main()
