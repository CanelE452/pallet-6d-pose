"""Fixed paired diagnostics; no threshold or metric search (not stdlib statistics)."""
import numpy as np
from scipy.stats import rankdata
from contracts import SEED, BOOTSTRAPS


def errors(values):
    return np.array([np.inf if v is None else v for v in values],float)


def hard20(values):
    a=errors(values)
    # Inverse-CDF order statistic avoids inf-inf interpolation for missing targets.
    # Finite populations use the standard linear NumPy 80th percentile.
    threshold = float(np.quantile(a,.8)) if np.isfinite(a).all() else float(np.sort(a)[int(np.ceil(.8*len(a)))-1])
    return a>=threshold, threshold


def corr(score,error):
    a,b=rankdata(score),rankdata(error)
    if len(a)<2 or np.ptp(a)==0 or np.ptp(b)==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def auc(score, labels):
    y=np.asarray(labels,bool); n=int(y.sum());m=len(y)-n
    if not n or not m:
        return None
    return float((rankdata(score)[y].sum()-n*(n+1)/2)/(n*m))


def diagnostics(score,error,hard=None):
    score=np.asarray(score,float);error=errors(error)
    if hard is None:hard,_=hard20(error)
    ids=np.argsort(-score,kind='stable')[:min(30,len(score))]
    hits=int(np.asarray(hard)[ids].sum())
    return dict(spearman=corr(score,error),AUROC=auc(score,hard),hard_count=int(np.sum(hard)),
        precision_at30=hits/len(ids),recall_at30=hits/int(np.sum(hard)) if np.sum(hard) else None)


def interval(values,draws=BOOTSTRAPS):
    values=np.asarray([v for v in values if v is not None and np.isfinite(v)],float)
    return dict(valid_draws=len(values),undefined_draws=draws-len(values),
        mean=float(values.mean()) if len(values) else None,
        ci95=np.quantile(values,[.025,.975]).tolist() if len(values) else None)


def paired_bootstrap(task,base,error,hard,groups,cluster):
    rng=np.random.default_rng(SEED); groups=np.asarray(groups)
    unique=np.unique(groups);indices=[np.flatnonzero(groups==g) for g in unique]
    arrays=[np.asarray(a) for a in (task,base,error,hard)]
    out={k:[] for k in ('AUROC','spearman','precision_at30','recall_at30')}
    for _ in range(BOOTSTRAPS):
        ii=np.concatenate([indices[i] for i in rng.integers(len(unique),size=len(unique))]) if cluster else rng.integers(len(groups),size=len(groups))
        a,b,e,h=[a[ii] for a in arrays]
        da,db=diagnostics(a,e,h),diagnostics(b,e,h)
        for k in out:
            out[k].append(None if da[k] is None or db[k] is None else da[k]-db[k])
    return {k:interval(v) for k,v in out.items()}
