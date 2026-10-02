"""Session-paired pose bootstrap on frozen per-frame scores (no new inference)."""
import time
import numpy as np
from .compute import C,M,DOC,RAW,write,bind

FIELDS=('translation_cm','rotation_deg','yaw_deg')
def quantiles(rows,field,session_index,counts,q):
    valid=[r for r in rows if r['available']]
    order=np.argsort([r[field] for r in valid],kind='stable')
    values=np.array([r[field] for r in valid])[order]
    groups=np.array([session_index[r['session']] for r in valid])[order]
    if not len(values):return np.full(len(counts),np.nan)
    cumulative=np.cumsum(counts[:,groups],axis=1)
    total=cumulative[:,-1]; rank=(total-1)*q
    lo=np.floor(rank).astype(int);hi=np.ceil(rank).astype(int)
    li=np.argmax(cumulative>lo[:,None],axis=1);ui=np.argmax(cumulative>hi[:,None],axis=1)
    out=values[li]*(hi-rank)+values[ui]*(rank-lo)
    exact=lo==hi;out[exact]=values[li[exact]];out[total==0]=np.nan
    return out

def main():
    started=time.time(); y=C.read(RAW/'YOLO_SCORES.json')
    collections={'yolo':{k:v['pose_scores'] for k,v in y.items() if k=='R0' or k.startswith('N3_DIM_SYM')}}
    inputs=[bind(RAW/'YOLO_SCORES.json')]
    for backbone in ['dope','resnet18']:
        path=C.RAW/f'evaluation/{backbone}.json';d=C.read(path);inputs.append(bind(path))
        collections[backbone]={k:v['result']['pose_rows'] for k,v in d['methods'].items()}
    sessions=sorted({r['session'] for r in collections['yolo']['R0']});si={s:i for i,s in enumerate(sessions)}
    assert len(sessions)>1
    counts=np.random.default_rng(20260917).multinomial(len(sessions),np.ones(len(sessions))/len(sessions),size=10000)
    np.save(RAW/'BOOTSTRAP_SESSION_COUNTS.npy',counts)
    output={'contract':{'resamples':10000,'seed':20260917,'sessions':sessions,
        'same_draws_all_seeds_and_backbones':True,'estimand':'statistic(after successful frames) - statistic(before successful frames)',
        'CI':'percentile 2.5/97.5; seed-mean formed within each draw; no multiplicity correction',
        'failures':'Full319 failure/coverage/AUC retained. Common-success analysis is secondary.',
        'reference':'image/geometry reconstructed; reused DEV, posthoc, not independent physical 6D truth'},'inputs':inputs,'backbones':{}}
    pairrows=[]
    for backbone,methods in collections.items():
        base_name='R0' if backbone=='yolo' else 'base';base=methods[base_name]
        base_map={r['id']:r for r in base};assert len(base_map)==319
        before_ids={r['id'] for r in base if r['available']}
        seeds={};draws={}
        for seed in (1,2,3):
            name=f'N3_DIM_SYM_seed{seed}' if backbone=='yolo' else f'n3_seed{seed}'
            after=methods[name];am={r['id']:r for r in after};assert set(am)==set(base_map)
            assert all(am[i]['session']==base_map[i]['session'] for i in am)
            after_ids={r['id'] for r in after if r['available']};common=before_ids&after_ids
            transitions={'same_success_set':before_ids==after_ids,'before_success':len(before_ids),'after_success':len(after_ids),
                'common_success':len(common),'new_failure_ids':sorted(before_ids-after_ids),'recovered_ids':sorted(after_ids-before_ids),
                'both_failed_ids':sorted(set(am)-(before_ids|after_ids))}
            result={'success_sets':transitions,'before_full':M._pose_summary(base),'after_full':M._pose_summary(after),'metrics':{}}
            cb=[base_map[i] for i in sorted(common)];ca=[am[i] for i in sorted(common)]
            for f in FIELDS:
                delta=np.array([am[i][f]-base_map[i][f] for i in sorted(common)])
                desc={'median_after_minus_before_on_common_success':float(np.median(delta)) if len(delta) else None,
                    'improved':int((delta < -1e-9).sum()),'unchanged':int((abs(delta)<=1e-9).sum()),'worsened':int((delta>1e-9).sum())}
                for label,q in [('median',.5),('P90',.9)]:
                    k=f+'_'+label
                    b=np.array([r[f] for r in base if r['available']]);a=np.array([r[f] for r in after if r['available']])
                    d=quantiles(after,f,si,counts,q)-quantiles(base,f,si,counts,q)
                    draws.setdefault(k,[]).append(d)
                    valid=np.isfinite(d)
                    result['metrics'][k]={'delta_of_statistics':float(np.quantile(a,q)-np.quantile(b,q)),
                        'CI95':np.quantile(d,[.025,.975]).tolist() if valid.all() else None,
                        'undefined_draws':int((~valid).sum()),'secondary_common_success_delta_of_statistics':float(np.quantile([r[f] for r in ca],q)-np.quantile([r[f] for r in cb],q)) if common else None,
                        **desc}
            for i in sorted(am):
                pairrows.append({'backbone':backbone,'seed':seed,'id':i,'session':am[i]['session'],'before_success':i in before_ids,'after_success':i in after_ids,
                    'deltas_after_minus_before':{f:am[i][f]-base_map[i][f] for f in FIELDS} if i in common else None})
            seeds[str(seed)]=result
        aggregate={}
        for k,values in draws.items():
            mean_draw=np.mean(values,axis=0);points=[seeds[str(s)]['metrics'][k]['delta_of_statistics'] for s in (1,2,3)]
            aggregate[k]={'mean_seed_delta':float(np.mean(points)),'seed_sd_ddof1':float(np.std(points,ddof=1)),'seed_min':min(points),'seed_max':max(points),
                'CI95':np.quantile(mean_draw,[.025,.975]).tolist() if np.isfinite(mean_draw).all() else None,
                'undefined_draws':int((~np.isfinite(mean_draw)).sum())}
        output['backbones'][backbone]={'per_seed':seeds,'mean_of_seed_statistics':aggregate}
        print('Paired complete',backbone,flush=True)
    output['seconds']=time.time()-started
    write(DOC/'PAIRED_POSE_ANALYSIS.json',output);write(RAW/'PAIRED_POSE_FRAME_DELTAS.json',pairrows)
    # Sanity check vectorized integer replication against NumPy on actual draws.
    checks=[];base=collections['yolo']['R0']
    for j in range(20):
        expanded=[r['translation_cm'] for r in base if r['available'] for _ in range(int(counts[j,si[r['session']]]))]
        for q in (.5,.9): checks.append(abs(quantiles(base,'translation_cm',si,counts[j:j+1],q)[0]-np.quantile(expanded,q))<1e-10)
    assert all(checks)
    write(DOC/'BOOTSTRAP_VALIDATION.json',{'numpy_replication_checks':len(checks),'all_pass':all(checks),'session_counts':bind(RAW/'BOOTSTRAP_SESSION_COUNTS.npy')})

if __name__=='__main__':main()
