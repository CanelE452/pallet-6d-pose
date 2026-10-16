"""GT-privileged candidate ceilings and transfer diagnostics; never inference policy."""
from context14 import *

def main():
    result={}
    for split in ('insample','heldout','synth_val','real'):
        for condition in CONDITIONS:
            m=read((V11 if split in ('synth_val','real') else RUN)/'cache'/split/condition/'manifest.json')
            t=target_rows(split,condition);det=np.array([r['detected'] for r in m['rows']]);gt=np.array([r['points'] for r in t])[det];valid=np.array([r['valid'] for r in t])[det];kind=P.target_type(t).numpy()[det];rows=[]
            for seed in SEEDS:
                rep=repcache(split,condition,seed);cand=candidates(rep).numpy();errors=np.linalg.norm(cand-gt[:,:,None],axis=-1);gain=errors[...,:1]-errors
                length=np.linalg.norm(rep['line_delta'].numpy(),axis=-1);oracle_points=errors[...,:3].min(-1);oracle_fusion=errors.min(-1)
                prediction=np.load(RUN/'predictions'/split/condition/f'heldout_fusion_seed{seed}.npz') if split in ('real','synth_val') else None
                row={'seed':seed,'strata':{}}
                for typ,k in [('all',None),('hidden',1),('outside',2)]:
                    mask=valid if k is None else valid&(kind==k)
                    if not mask.any():continue
                    d=dict(points=int(mask.sum()),line_delta_mean_px=float(length[mask].mean()),line_delta_p90_px=float(np.quantile(length[mask],.9)),
                        base_error_mean_px=float(errors[...,0][mask].mean()),amodal_true_gain_mean_px=float(gain[...,2][mask].mean()),
                        amodal_beneficial_fraction=float((gain[...,2][mask]>0).mean()),
                        points_candidate_oracle_px=float(oracle_points[mask].mean()),fusion_candidate_oracle_px=float(oracle_fusion[mask].mean()),
                        line_candidate_oracle_additional_gain_px=float((oracle_points-oracle_fusion)[mask].mean()),
                        line_candidate_wins_over_points=int(((oracle_fusion<oracle_points)&mask).sum()))
                    if prediction is not None:
                        p=prediction;choice=p['choice'][det];e=np.linalg.norm(p['points'][det]-gt,axis=-1);apply=(choice>0)&mask;bad=apply&(e>errors[...,0]);score=p['gain'][det]
                        d.update(amodal_predicted_gain_mean_px=float(score[...,2][mask].mean()),applied_points=int(apply.sum()),harmful_applied_points=int(bad.sum()),
                            line_chosen_points=int(((choice>=3)&mask).sum()),selected_error_mean_px=float(e[mask].mean()))
                    row['strata'][typ]=d
                rows.append(row)
            result[split+'/'+condition]=rows
    write(RUN/'FAILURE_DIAGNOSIS.json',dict(posthoc=True,rows=result,scope='GT candidate oracle is privileged diagnostic only. No model selection, training, threshold changes or inference GT.'))
    for c,typ in [('clean','all'),('corner20','hidden'),('crop20','outside'),('crop40','outside')]:
        print(c,[r['strata'][typ] for r in result['real/'+c]],flush=True)

if __name__=='__main__':main()
