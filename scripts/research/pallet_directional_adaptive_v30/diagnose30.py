"""Frozen-model gain/amount diagnosis on training and evaluation; no retuning."""
from context30 import *
from data30 import training_data,population,conditions
from model30 import DirectionalAdaptive
from sklearn.metrics import roc_auc_score

def stats(out,gt,valid,kind):
    pts=out['candidate_points'];error=np.linalg.norm(pts-gt[:,:,None],axis=-1);base=np.linalg.norm(out['base']-gt,axis=-1);rows={}
    gain=(base[:,:,None]-error)/out['diag'][:,None,None];mask=valid&np.isfinite(out['points']).all(-1)
    residual=gt[:,:,None]-out['anchors'];d=out['directions'];alpha=np.clip((residual*d).sum(-1)/np.maximum((d*d).sum(-1),1e-10),-1,1)
    for typ,k in (('all',None),('intact',0),('hidden',1),('outside',2)):
        use=mask if k is None else mask&(kind==k)
        if not use.any():continue
        mm=(out['candidate_mask']==1)&use[:,:,None];mm[:,:,0]=False;positive=gain>0;y=positive[mm];score=out['gain'][mm];moving=mm&(np.linalg.norm(d,axis=-1)>.1)
        rows[typ]=dict(points=int(use.sum()),candidate_help_auc=float(roc_auc_score(y,score)) if len(set(y))>1 else None,mean_amount_abs_error=float(np.abs(out['alpha']-alpha)[moving].mean()) if moving.any() else None,predicted_gain_mean=float(score.mean()),actual_gain_mean=float(gain[mm].mean()),line_selected_fraction=float((out['chosen'][use]>=8).mean()),mean_error=float(np.linalg.norm(out['points']-gt,axis=-1)[use].mean()),oracle_current_amount_error=float(np.where(out['candidate_mask']==1,error,np.inf).min(-1)[use].mean()),candidate_samples=int(mm.sum()))
    return rows

def main():
    setup();rows=[]
    for seed in SEEDS:
        O,S,gt,valid,ids,kind=training_data(seed);O={k:v.cuda() for k,v in O.items()};S={k:v.cuda() for k,v in S.items()}
        for arm in ARMS:
            model=DirectionalAdaptive(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',map_location='cpu',weights_only=True)['state_dict']);parts=[]
            with torch.no_grad():
                for i in range(0,len(ids),32):
                    sl=slice(i,i+32);out=model(subset(O,sl),subset(S,sl),arm);parts.append({k:v.cpu().numpy() for k,v in out.items()})
            out={k:np.concatenate([x[k] for x in parts]) for k in parts[0]};rows.append(dict(seed=seed,arm=arm,population='train',condition='all',stats=stats(out,gt.numpy(),valid.numpy(),kind.numpy())))
            del model
        del O,S
        for split in EVAL_SPLITS:
            for c in conditions(split):
                _,ts=population(split,c);g=np.array([t['points'] for t in ts]);v=np.array([t['valid'] for t in ts]);k=C.P.target_type(ts).numpy()
                for arm in ARMS:
                    p=np.load(RUN/'predictions'/split/c/f'{arm}_seed{seed}.npz');rows.append(dict(seed=seed,arm=arm,population=split,condition=c,stats=stats(p,g,v,k)))
    write(RUN/'GAIN_DIAGNOSIS.json',dict(complete=True,no_new_training=True,rows=rows,limitations='Candidate AUROC uses correlated alternatives at current predicted amounts, not independent samples or calibrated probabilities. Parent point residual training remains in-sample.'))
    print('DIAGNOSE30 COMPLETE',flush=True)

if __name__=='__main__':main()
