"""Same session draws for 2D contrasts and completed pose ablations."""
import time
import numpy as np
from .compute import C,M,DOC,RAW,write
from .paired import FIELDS,quantiles

def summarize(points,draws):
    d=np.mean(draws,axis=0);ok=np.isfinite(d)
    return {'mean_seed_delta':float(np.mean(points)),'per_seed_delta':points,'seed_sd_ddof1':float(np.std(points,ddof=1)),
        'seed_min':min(points),'seed_max':max(points),'CI95':np.quantile(d,[.025,.975]).tolist() if ok.all() else None,'undefined_draws':int((~ok).sum()),
        'per_seed_CI95':[np.quantile(x,[.025,.975]).tolist() if np.isfinite(x).all() else None for x in draws]}
def main():
    started=time.time();y=C.read(RAW/'YOLO_SCORES.json');paired=C.read(DOC/'PAIRED_POSE_ANALYSIS.json')
    sessions=paired['contract']['sessions'];si={s:i for i,s in enumerate(sessions)};counts=np.load(RAW/'BOOTSTRAP_SESSION_COUNTS.npy')
    comparisons=[('N2_minus_N0','N0_BASE_REPLAY','N2_DIM_ONLY'),('N1_minus_N0','N0_BASE_REPLAY','N1_SYM_ONLY'),('N3_minus_N2','N2_DIM_ONLY','N3_DIM_SYM'),('N3_minus_N1','N1_SYM_ONLY','N3_DIM_SYM')]
    posecache={};ablation={}
    for _,before,after in comparisons:
        for arm in [before,after]:
            for seed in (1,2,3):
                name=f'{arm}_seed{seed}'
                if name in posecache:continue
                rows=y[name]['pose_scores'];posecache[name]={}
                for f in FIELDS:
                    for label,q in [('median',.5),('P90',.9)]:posecache[name][f+'_'+label]=quantiles(rows,f,si,counts,q)
    for label,before,after in comparisons:
        ablation[label]={}
        for f in FIELDS:
            for stat,q in [('median',.5),('P90',.9)]:
                key=f+'_'+stat;points=[];draws=[]
                for seed in (1,2,3):
                    b=f'{before}_seed{seed}';a=f'{after}_seed{seed}'
                    bv=[r[f] for r in y[b]['pose_scores'] if r['available']];av=[r[f] for r in y[a]['pose_scores'] if r['available']]
                    points.append(float(np.quantile(av,q)-np.quantile(bv,q)));draws.append(posecache[a][key]-posecache[b][key])
                ablation[label][key]=summarize(points,draws)
    write(DOC/'ABLATION_POSE_UNCERTAINTY.json',{'contract':paired['contract'],'comparisons':ablation})
    collections={'yolo':y}
    for b in ['dope','resnet18']:
        d=C.read(C.RAW/f'evaluation/{b}.json');collections[b]={k:{'corner_scores':v['result']['corner_rows']} for k,v in d['methods'].items()}
    out={}
    for backbone,methods in collections.items():
        cache={};point={}
        for method,scores in methods.items():
            if method.startswith('OLD_P'):continue
            prep=M._bootstrap_arrays(scores['corner_scores'],si);stats=[M._prepared_bootstrap_stats(prep,c) for c in counts]
            cache[method]={k:np.array([r[k] if r[k] is not None else np.nan for r in stats]) for k in stats[0]}
            point[method]=M._bootstrap_stats(scores['corner_scores'])
        contrast=[('N3_minus_base','R0','N3_DIM_SYM')] if backbone=='yolo' else [('N3_minus_base','base','n3')]
        if backbone=='yolo':contrast+=comparisons
        out[backbone]={}
        for label,before,after in contrast:
            out[backbone][label]={}
            for key in point[next(iter(point))]:
                p=[];d=[]
                for seed in (1,2,3):
                    b=before if before in ('base','R0') else f'{before}_seed{seed}';a=f'{after}_seed{seed}'
                    p.append(point[a][key]-point[b][key]);d.append(cache[a][key]-cache[b][key])
                out[backbone][label][key]=summarize(p,d)
        print('2D uncertainty',backbone,flush=True)
    write(DOC/'PAIRED_2D_UNCERTAINTY.json',{'contract':paired['contract'],'comparisons':out,'seconds':time.time()-started})

if __name__=='__main__':main()
