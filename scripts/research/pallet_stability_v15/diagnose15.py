"""Posthoc safety/usefulness tradeoff, line candidate ceilings and session errors."""
from context15 import *

def main():
    rows={};sessions={}
    for split in ('synth_val','real'):
        for condition in CONDITIONS:
            t=targets(split,condition);gt=np.array([r['points'] for r in t]);valid=np.array([r['valid'] for r in t]);kind=C.P.target_type(t).numpy();session=np.array([r['session'] for r in t]);key=split+'/'+condition;diag={};sd={}
            for seed in SEEDS:
                base=np.load(RUN/'predictions'/split/condition/f'base_seed{seed}.npz');full=np.load(RUN/'predictions'/split/condition/f'amodal_seed{seed}.npz')
                det=base['detected'];e0=np.linalg.norm(base['points']-gt,axis=-1);ef=np.linalg.norm(full['points']-gt,axis=-1)
                for arm in ARMS:
                    p=np.load(RUN/'predictions'/split/condition/f'{arm}_seed{seed}.npz');e=np.linalg.norm(p['points']-gt,axis=-1);ec=np.linalg.norm(p['candidates']-gt[:,:,None],axis=-1);truth=ec[...,:1]-ec;d={}
                    for typ,k in [('all',None),('hidden',1),('outside',2)]:
                        mask=valid&det[:,None]
                        if k is not None:mask=mask&(kind==k)
                        if not mask.any():continue
                        rich=mask&(p['choice']>=5);eligible=mask[:,:,None]&(p['eligible']>.5);eligible[...,0]=False
                        d[typ]=dict(points=int(mask.sum()),line_old_mean_delta_px=float(np.linalg.norm(p['candidates'][...,4,:]-p['candidates'][...,0,:],axis=-1)[mask].mean()),
                            line_rich_mean_delta_px=float(np.linalg.norm(p['candidates'][...,5:,:]-p['candidates'][...,:1,:],axis=-1)[mask].mean()),
                            oracle_old5_px=float(ec[...,:5].min(-1)[mask].mean()),oracle_rich8_px=float(ec.min(-1)[mask].mean()),
                            oracle_extra_gain_px=float((ec[...,:5].min(-1)-ec.min(-1))[mask].mean()),
                            lower_score_empirical_coverage=float((truth[eligible]>=p['gain'][eligible]).mean()),
                            new_line_applied_points=int(rich.sum()),new_line_harmful_points=int((rich&(e>e0)).sum()),
                            keep_points=int((mask&(p['choice']==0)).sum()),
                            unchosen_available_improvement_points=int((mask&(p['choice']==0)&(ec.min(-1)<e0-.25)).sum()))
                    name=f'{arm}_seed{seed}';diag[name]=d
                    if split=='real':
                        mask=valid&det[:,None]
                        typ=1 if condition=='corner20' else 2 if condition.startswith('crop') else None
                        if typ is not None:mask=mask&(kind==typ)
                        out=[]
                        for sess in np.unique(session):
                            sm=mask&(session==sess)[:,None]
                            if sm.any():out.append(dict(session=sess,points=int(sm.sum()),base_mean_px=float(e0[sm].mean()),amodal_mean_px=float(ef[sm].mean()),model_mean_px=float(e[sm].mean()),difference_vs_base_px=float((e-e0)[sm].mean())))
                        sd[name]=out
            rows[key]=diag
            if sd:sessions[key]=sd
    write(RUN/'STABILITY_DIAGNOSIS.json',dict(posthoc=True,rows=rows,session_target_point_errors=sessions,
        scope='GT diagnostic only. Oracle8 available to diagnostic even for old5 arms; inference masks unchanged. Lower score coverage is measured, not guaranteed; balancing changes target distribution.'))
    print('DIAGNOSIS COMPLETE',flush=True)

if __name__=='__main__':main()
