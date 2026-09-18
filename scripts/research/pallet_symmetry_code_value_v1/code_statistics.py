"""Paired frame/cluster bootstrap; fixed seed and no population pooling."""
import numpy as np
import cv_env as E
from eval_math import summary,damage

def metric_path(pop,arm,seed,mode='PRIMARY'):
    return E.RAW/f'metrics/{pop}/{arm}_seed{seed}_{mode}.json'
def rows_for(pop,arm,mode='PRIMARY'):return [E.read(metric_path(pop,arm,s,mode)) for s in [1,2,3]]
def contrast(left,right,clusters=None):
    a=np.asarray(left,float);b=np.asarray(right,float);assert a.shape==b.shape and a.shape[0]==3
    d=(a-b).mean(0);per=(a-b).mean(1);rng=np.random.default_rng(20260918)
    _,g=np.unique(np.arange(len(d)) if clusters is None else clusters,return_inverse=True);n=len(np.unique(g))
    total=np.bincount(g,weights=d,minlength=n);count=np.bincount(g,minlength=n);samples=[]
    if n>=2:
        for _ in range(100):
            w=rng.multinomial(n,np.full(n,1/n),size=100);samples.extend(((w@total)/(w@count)).tolist())
    return dict(delta=float(d.mean()),CI95=np.quantile(samples,[.025,.975]).tolist() if samples else None,per_seed_delta=per.tolist(),improved_seeds=int((per<0).sum()),
      units=n,resamples=10000 if samples else 0,seed=20260918,level='frame' if clusters is None else 'cluster',
      LOSO=[dict(unit=str(unit),delta=float((total.sum()-total[i])/(count.sum()-count[i]))) for i,unit in enumerate(np.unique(clusters))] if clusters is not None and n>1 else [],
      multiplicity_adjusted=False,CI_status='available' if samples else 'N/A: fewer than two resampling units')
def paired(base,new,pop,indices=None):
    for a,b in zip(base,new):assert [r['id'] for r in a]==[r['id'] for r in b]
    ix=[i for i in (range(len(base[0])) if indices is None else indices) if all(x[i]['evaluable'] for x in base+new)]
    if not ix:return dict(n=0,status='N/A: empty evaluable group')
    a=[[r[i]['E_sym'] for i in ix] for r in new];b=[[r[i]['E_sym'] for i in ix] for r in base]
    clusters=None if pop=='SYNTH' else [base[0][i]['session'] for i in ix]
    return dict(n=len(ix),**contrast(a,b,clusters))
def grouped(base,new,pop,key):
    out={}
    for v in sorted({r[key] for r in base[0]}):
        ix=[i for i,r in enumerate(base[0]) if r[key]==v]
        out[str(v)]=dict(**paired(base,new,pop,ix),baseline_summary=mean_summary([[r[i] for i in ix] for r in base]),new_summary=mean_summary([[r[i] for i in ix] for r in new]))
    return out
def summaries(rows):return [summary(r) for r in rows]
def mean_summary(rows):
    s=summaries(rows);keys=['E_sym','E_fixed','matched_pooled_corner8_median_px','matched_pooled_corner8_P90_px','gross20','coverage']
    out={k:float(np.mean([r[k] for r in s])) if all(r[k] is not None for r in s) else None for k in keys}
    out['PCK10']=float(np.mean([r['PCK']['10'] for r in s]));out['frames']=len(rows[0]);out['per_seed']=s;return out
def movement(pop,basearm,newarm,basemode='PRIMARY',newmode='PRIMARY',indices=None):
    results=[]
    for seed in [1,2,3]:
        def f(kind,arm,mode,ext):return E.RAW/f'{kind}/{pop}/{arm}_seed{seed}_{mode}.{ext}'
        a=E.read(f('predictions',basearm,basemode,'json'))['records'];b=E.read(f('predictions',newarm,newmode,'json'))['records']
        assert [r['id'] for r in a]==[r['id'] for r in b];ix=list(range(len(a))) if indices is None else list(indices);changes=[]
        for i in ix:
            j=a[i]['selected_index'];assert j==b[i]['selected_index']
            if j is not None:
                p=np.asarray(a[i]['candidates'][j]['keypoints_xy'])[:8];q=np.asarray(b[i]['candidates'][j]['keypoints_xy'])[:8]
                good=np.isfinite(p).all(-1)&np.isfinite(q).all(-1)&~(p==-1).all(-1)&~(q==-1).all(-1);changes.extend(np.linalg.norm(p[good]-q[good],axis=-1).tolist())
        with np.load(f('logits',basearm,basemode,'npz')) as x,np.load(f('logits',newarm,newmode,'npz')) as y:
            assert np.array_equal(x['ids'],y['ids']);mask=x['support'][ix]&y['support'][ix];l=x['logits'][ix].astype(float);r=y['logits'][ix].astype(float)
            soft=lambda z:np.exp(z-z.max(-1,keepdims=True))/np.exp(z-z.max(-1,keepdims=True)).sum(-1,keepdims=True)
            p=soft(l);q=soft(r);m=(p+q)/2;js=.5*(np.sum(p*np.log(np.maximum(p,1e-300)/np.maximum(m,1e-300)),-1)+np.sum(q*np.log(np.maximum(q,1e-300)/np.maximum(m,1e-300)),-1))
            results.append(dict(mean_coordinate_change_px=float(np.mean(changes)) if changes else None,max_coordinate_change_px=max(changes) if changes else None,
              valid_corners=len(changes),supported_candidates=int(mask.sum()),JS=float(js[mask].mean()) if mask.any() else None,
              top1_change_rate=float((l.argmax(-1)!=r.argmax(-1))[mask].mean()) if mask.any() else None,
              first_layer_activation_mean_abs_change=float(np.abs(x['activation'][ix]-y['activation'][ix]).mean())))
    return dict(per_seed=results,**{k:float(np.mean([r[k] for r in results])) if all(r[k] is not None for r in results) else None for k in ['mean_coordinate_change_px','JS','top1_change_rate','first_layer_activation_mean_abs_change']},
      max_coordinate_change_px=max(r['max_coordinate_change_px'] for r in results if r['max_coordinate_change_px'] is not None))
def track_report(track):
    out={};blind,aware=E.ARMS[track]
    for pop in E.POPS[track]:
        b=rows_for(pop,blind);a=rows_for(pop,aware);d=[damage(x,y) for x,y in zip(b,a)]
        out[pop]=dict(blind=mean_summary(b),aware=mean_summary(a),contrast=paired(b,a,pop),groups=grouped(b,a,pop,'group'),objects=grouped(b,a,pop,'object'),
          sessions=grouped(b,a,pop,'session'),damage=d,movement=movement(pop,blind,aware),group_movement={g:movement(pop,blind,aware,indices=[i for i,r in enumerate(b[0]) if r['group']==g]) for g in sorted({r['group'] for r in b[0]})})
        if pop=='SYNTH':out[pop]['scenario_secondary']=contrast([[r['E_sym'] for r in s if r['evaluable']] for s in a],[[r['E_sym'] for r in s if r['evaluable']] for s in b],[r['session'] for r in b[0] if r['evaluable']])
        out[pop]['safety_pass']=sum(x['good5_to_bad10']-x['reverse_good5_to_bad10'] for x in d)<=0 and out[pop]['aware']['gross20']<=out[pop]['blind']['gross20']
    dst='A_C1C2_RESULTS.json' if track=='A' else 'B_C1C2C4_RESULTS.json'
    E.write(E.DOC/dst,dict(complete=True,populations=out,T=1.,estimand='aware minus blind; negative better',diagnostic_only=track=='B',reused_DEV=True,
      real_DEV_interpretation='C2 constant-token sensitivity; not routing across groups',domain_group_confounded=track=='B',groups_not_present='N/A, no fabricated group metrics'))
    print('TRACK_RESULTS',track,{p:r['contrast'] for p,r in out.items()},flush=True)
def perturbations():
    out={}
    for track in ['A','B']:
        arm=E.ARMS[track][1];out[arm]={}
        for pop in E.POPS[track]:
            base=rows_for(pop,arm);entry=dict(correct=mean_summary(base),modes={});donor=E.read(E.DOC/'CODE_DONOR_LOCK.json')['populations'][pop]
            for mode in ['NEUTRAL','ZERO','WRONG','SHUFFLED']:
                if mode=='SHUFFLED' and not donor['applicable']:
                    entry['modes'][mode]=dict(applicable=False,reason=donor['reason']);continue
                rows=rows_for(pop,arm,mode);entry['modes'][mode]=dict(applicable=True,summary=mean_summary(rows),contrast=paired(base,rows,pop),groups=grouped(base,rows,pop,'group'),
                  movement=movement(pop,arm,arm,newmode=mode),group_movement={g:movement(pop,arm,arm,newmode=mode,indices=[i for i,r in enumerate(base[0]) if r['group']==g]) for g in sorted({r['group'] for r in base[0]})})
                if mode=='SHUFFLED':entry['modes'][mode]['donor_counts']={k:donor[k] for k in ['same_code','different_code','same_frame']}
            out[arm][pop]=entry
    E.write(E.DOC/'CODE_PERTURBATION_RESULTS.json',dict(complete=True,optimizer_updates=0,models=out,estimand='perturbed minus correct; positive favors correct',dimensions_and_evaluator_correct=True,T=1.))
