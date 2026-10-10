"""Numerical summaries only; no image/model/PnP or hypothesis selection."""
from collections import defaultdict
import hashlib
import math
import statistics as stdlib
import numpy as np
from . import verdict as V

NUMERIC = {'T_cm':'translation_cm', 'R_deg':'rotation_deg', 'ADDsym_m':'ADDsym_m', 'IoU3D':'IoU3D'}
BINS = {
    'elevation_deg': (5,10,20,30),
    'reference_margin_px': (2,4,8,16),
    'distance_m': (2,3,4,6),
}
BIN_LABELS = {
    'elevation_deg': ('<5','5–10','10–20','20–30','>=30'),
    'reference_margin_px': ('<2','2–4','4–8','8–16','>=16'),
    'distance_m': ('<2','2–3','3–4','4–6','>=6'),
}


class Draws:
    def __init__(self, labels, level):
        _,self.group=np.unique(np.asarray(labels),return_inverse=True)
        self.n=int(self.group.max()+1)
        self.counts=np.bincount(self.group,minlength=self.n).astype(float)
        rng=np.random.default_rng(V.BOOTSTRAP_SEED)
        raw=np.concatenate([rng.multinomial(self.n,np.full(self.n,1/self.n),size=100)
                            for _ in range(V.BOOTSTRAP_DRAWS//100)])
        self.sha256=hashlib.sha256(raw.astype('<u2').tobytes()).hexdigest()
        self.weights=raw.astype(float);self.denominator=self.weights@self.counts
        self.level=level;self.master_frames=len(labels)

    def subset(self, indices):
        obj=object.__new__(Draws);obj.__dict__=self.__dict__.copy()
        obj.group=self.group[np.asarray(indices,int)]
        obj.counts=np.bincount(obj.group,minlength=self.n).astype(float)
        obj.denominator=obj.weights@obj.counts
        return obj

    def interval(self, values):
        values=np.asarray(values,float);valid=np.isfinite(values)
        if not valid.any():return None
        totals=np.bincount(self.group,weights=np.where(valid,values,0),minlength=self.n)
        counts=np.bincount(self.group,weights=valid.astype(float),minlength=self.n)
        denominator=self.weights@counts
        samples=np.divide(self.weights@totals,denominator,
            out=np.full(len(denominator),np.nan),where=denominator>0)
        samples=samples[np.isfinite(samples)]
        return np.quantile(samples,[.025,.975]).tolist() if len(samples) else None

    def metadata(self):
        return dict(level=self.level,units=self.n,frames=int(self.counts.sum()),
            master_frames=self.master_frames,resamples=V.BOOTSTRAP_DRAWS,seed=V.BOOTSTRAP_SEED,
            draws_sha256_uint16_le=self.sha256,interval='linear 2.5 and97.5 percentiles',
            scope_rule='reuse master population draws; absent units contribute zero',multiplicity_adjusted=False)


def describe(values, draws=None):
    values=np.asarray(values,float);x=values[np.isfinite(values)].tolist()
    if not x:return dict(n=0,mean=None,variance=None,std=None,median=None,P90=None,max=None,CI95=None)
    variance=stdlib.variance(x) if len(x)>1 else None
    return dict(n=len(x),mean=stdlib.fmean(x),variance=variance,std=math.sqrt(variance) if variance is not None else None,
        median=stdlib.median(x),P90=float(np.quantile(x,.9)),max=max(x),
        CI95=draws.interval(values) if draws else None,variance_ddof=1,CI95_target='mean')


def vectors(poses):
    indicators=[V.indicators(p) for p in poses]
    result={name:np.asarray([p[key] if p.get('available') and p.get(key) is not None else np.nan for p in poses],float)
            for name,key in NUMERIC.items()}
    result.update({name:np.asarray([p[name] for p in indicators],float) for name in ('confusion_rate','success_rate')})
    result['available']=np.asarray([p['available'] for p in indicators],bool)
    return result


def average_vectors(vectors_by_seed):
    out={}
    for metric in (*NUMERIC,'confusion_rate','success_rate'):
        values=np.stack([v[metric] for v in vectors_by_seed])
        valid=np.isfinite(values);counts=valid.sum(0)
        out[metric]=np.divide(np.where(valid,values,0).sum(0),counts,
            out=np.full(values.shape[1],np.nan),where=counts>0)
    out['available']=np.stack([v['available'] for v in vectors_by_seed]).all(0)
    out['available_seed_counts']=np.stack([v['available'] for v in vectors_by_seed]).sum(0)
    return out


def summary(v, draws):
    return dict(frames=len(v['available']),available_all_seeds=int(v['available'].sum()),
        numerical_available_frames={k:int(np.isfinite(v[k]).sum()) for k in NUMERIC},
        metrics={k:describe(v[k],draws) for k in NUMERIC},
        rates={k:dict(rate=float(v[k].mean()),CI95=draws.interval(v[k]),denominator=len(v[k]),
                      binary_before_seed_mean=True) for k in ('confusion_rate','success_rate')})


def auc(scores, labels):
    scores=np.asarray(scores,float);labels=np.asarray(labels,bool)
    valid=np.isfinite(scores);scores=scores[valid];labels=labels[valid]
    pos=int(labels.sum());neg=len(labels)-pos
    if not pos or not neg:return dict(AUC=None,positive=pos,negative=neg,n=len(labels))
    order=np.argsort(scores,kind='stable');rank_sum=0.;start=0
    while start<len(order):
        end=start+1
        while end<len(order) and scores[order[end]]==scores[order[start]]:end+=1
        rank_sum+=float(labels[order[start:end]].sum())*(start+1+end)/2
        start=end
    return dict(AUC=(rank_sum-pos*(pos+1)/2)/(pos*neg),positive=pos,negative=neg,n=len(labels))


def stage1(records, population):
    groups=defaultdict(list)
    for r in records:groups[r.get('backbone','YOLO')+'::'+r['method']].append(r)
    output=dict(population=population,phase='STAGE1_DIAGNOSTIC_ONLY',groups={},bin_edges=BINS,
        oracle='fixed GT W/D parity supplied; not minimum metric candidate or causal decomposition')
    for name,rows in sorted(groups.items()):
        ids=sorted({r['id'] for r in rows});seeds=sorted({r['seed'] for r in rows})
        index={(r['seed'],r['id']):r for r in rows};assert len(index)==len(rows)
        ordered=[[index[(s,i)] for i in ids] for s in seeds]
        reference=ordered[0];labels=[r['session'] for r in reference]
        level='cluster' if population!='SYNTH_HELDOUT' else 'frame'
        draws=Draws(labels if level=='cluster' else ids,level)
        if level=='cluster' and len(ids)==319:
            assert draws.sha256==V.definitions()['real_cluster_draw_sha256']
        arms={arm:[vectors([r['pose'][arm] for r in seedrows]) for seedrows in ordered] for arm in ('S0','ORACLE')}
        means={arm:average_vectors(values) for arm,values in arms.items()}
        group=dict(frames=len(ids),seeds=seeds,rows=len(rows),bootstrap=draws.metadata(),
            seed_mean={arm:summary(means[arm],draws) for arm in arms},
            per_seed={str(s):{arm:summary(arms[arm][j],draws) for arm in arms} for j,s in enumerate(seeds)},
            R_mean_difference_F_minus_oracle=describe(means['S0']['R_deg']-means['ORACLE']['R_deg'],draws),bins={})
        for field,edges in BINS.items():
            labels_for_ids=[]
            for r in reference:
                value=r.get(field)
                labels_for_ids.append('UNKNOWN' if value is None or not np.isfinite(value) else BIN_LABELS[field][int(np.digitize(value,edges))])
            group['bins'][field]={}
            for label in (*BIN_LABELS[field],'UNKNOWN'):
                selected=np.flatnonzero(np.asarray(labels_for_ids)==label)
                group['bins'][field][label]=dict(frames=len(selected),seed_mean={arm:summary({k:v[selected] for k,v in means[arm].items()},draws.subset(selected)) for arm in arms}) if len(selected) else dict(frames=0,seed_mean=None)
        for field in ('material','grade'):
            categories=sorted({str(r.get(field) or 'UNKNOWN') for r in reference})
            group['bins'][field]={}
            for label in categories:
                selected=np.flatnonzero([str(r.get(field) or 'UNKNOWN')==label for r in reference])
                group['bins'][field][label]=dict(frames=len(selected),seed_mean={arm:summary({k:v[selected] for k,v in means[arm].items()},draws.subset(selected)) for arm in arms})
        scores=[];conf=[];aucs=[];warnings={}
        for j,seedrows in enumerate(ordered):
            gap=np.asarray([r.get('formal_score_gap_px') if r.get('formal_score_gap_px') is not None else np.nan for r in seedrows],float)
            scores.extend((-gap).tolist());conf.extend(arms['S0'][j]['confusion_rate'].astype(bool).tolist())
            aucs.append(dict(seed=seeds[j],**auc(-gap,arms['S0'][j]['confusion_rate'])))
        group['score_gap_auc']=dict(pooled_descriptive=auc(scores,conf),per_seed=aucs,
            macro_seed_mean=stdlib.fmean(r['AUC'] for r in aucs if r['AUC'] is not None) if any(r['AUC'] is not None for r in aucs) else None,
            direction='smaller alternative-minus-selected gap predicts confusion; ROC score=-gap')
        gap=-np.asarray(scores);conf=np.asarray(conf,bool)
        for threshold in (1,2,3,5):
            warning=np.isfinite(gap)&(gap<threshold);caught=int((warning&conf).sum())
            warnings[str(threshold)]=dict(threshold_px=threshold,operator='<',warning_rate=float(warning.mean()),
                caught_confusion_rate=caught/int(conf.sum()) if conf.any() else None,
                warning_precision=caught/int(warning.sum()) if warning.any() else None,
                valid_score_predictions=int(np.isfinite(gap).sum()),predictions=len(gap),warnings=int(warning.sum()),caught=caught)
        group['warnings']=warnings;output['groups'][name]=group
    return output


def compare(baseline, changed, population, rule, feasibility=False):
    """Rows contain a single metric pose; three fixed seeds per method."""
    groups=defaultdict(list)
    for r in baseline:groups[r['method']].append(r)
    ci_index={(r['seed'],r['method'],r['id']):r for r in changed}
    metrics={};paired={};failures={}
    for method,base in sorted(groups.items()):
        ids=sorted({r['id'] for r in base});seeds=sorted({r['seed'] for r in base})
        bi={(r['seed'],r['id']):r for r in base};assert len(bi)==len(base)
        ordered_b=[[bi[s,i] for i in ids] for s in seeds]
        ordered_c=[[ci_index[s,method,i] for i in ids] for s in seeds]
        labels=[r['session'] for r in ordered_b[0]]
        level='frame' if population=='SYNTH_HELDOUT' or feasibility else 'cluster'
        draws=Draws(ids if level=='frame' else labels,level)
        if level=='cluster' and len(ids)==319:assert draws.sha256==V.definitions()['real_cluster_draw_sha256']
        secondary=Draws(labels,'scenario_cluster_secondary') if population=='SYNTH_HELDOUT' else None
        bv=[vectors([r['pose'] for r in rs]) for rs in ordered_b]
        cv=[vectors([r['pose'] for r in rs]) for rs in ordered_c]
        bm,cm=average_vectors(bv),average_vectors(cv)
        metrics[method]=dict(frames=len(ids),bootstrap=draws.metadata(),
            seed_mean={'S0':summary(bm,draws),rule:summary(cm,draws)},
            per_seed={str(s):{'S0':summary(bv[j],draws),rule:summary(cv[j],draws)} for j,s in enumerate(seeds)})
        def contrast(a,b):
            result={}
            for key in (*NUMERIC,'confusion_rate','success_rate'):
                delta=a[key]-b[key];valid=np.isfinite(delta)
                result[key]=dict(delta=float(delta[valid].mean()) if valid.any() else None,CI95=draws.interval(delta),
                    paired_frames=int(valid.sum()),binary_before_seed_mean=key.endswith('_rate'))
                if secondary:result[key]['scenario_cluster_secondary_CI95']=secondary.interval(delta)
            return result
        meancontrast=contrast(cm,bm)
        perseed={str(s):contrast(cv[j],bv[j]) for j,s in enumerate(seeds)}
        for key in (*NUMERIC,'confusion_rate','success_rate'):
            deltas=[perseed[str(s)][key]['delta'] for s in seeds]
            meancontrast[key].update(per_seed_delta=deltas,
                improved_seeds=sum(x is not None and (x>0 if key in ('success_rate','IoU3D') else x<0) for x in deltas))
        paired[method]=dict(seed_mean=meancontrast,per_seed=perseed,bootstrap=draws.metadata())
        failures[method]={}
        for j,s in enumerate(seeds):
            fields={name:[] for name in ('success_to_failure_ids','failure_to_success_ids','confusion_recovery_ids','confusion_damage_ids','hypothesis_change_ids','fallback_ids','unavailable_ids')}
            transitions=[]
            for b,c in zip(ordered_b[j],ordered_c[j]):
                i=b['id'];assert c['id']==i and c['qFinal']==b['qFinal'],'Coordinates must remain identical'
                ib,ic=V.indicators(b['pose']),V.indicators(c['pose'])
                if ib['success_rate'] and not ic['success_rate']:fields['success_to_failure_ids'].append(i)
                if not ib['success_rate'] and ic['success_rate']:fields['failure_to_success_ids'].append(i)
                if ib['confusion_rate'] and not ic['confusion_rate']:fields['confusion_recovery_ids'].append(i)
                if not ib['confusion_rate'] and ic['confusion_rate']:fields['confusion_damage_ids'].append(i)
                if b['hyp']!=c['hyp']:
                    fields['hypothesis_change_ids'].append(i);transitions.append(dict(id=i,before=b['hyp'],after=c['hyp']))
                if c.get('fallback'):fields['fallback_ids'].append(i)
                if not ic['available']:fields['unavailable_ids'].append(i)
            failures[method][str(s)]={**fields,**{key.removesuffix('_ids')+'_count':len(value) for key,value in fields.items()},'transitions':transitions}
    return dict(population=population,rule=rule,metrics=metrics,paired=paired,failures=failures,
        primary=paired[V.PRIMARY_METHOD]['seed_mean'],descriptive_only=feasibility)
