"""Unchanged k-center recurrence, with already-normalized 1+R_task weights."""
import numpy as np
from contracts import *


def select(features,risk,rows,budget=30):
    x=np.asarray(features,float);risk=np.asarray(risk,float)
    assert len(x)==len(risk)==len(rows) and np.isfinite(x).all()
    assert np.isfinite(risk).all() and ((risk>=0)&(risk<=1)).all()
    available=np.ones(len(x),bool);chosen=[];nearest=np.full(len(x),np.inf)
    initial=int(np.argmin(((x-x.mean(0))**2).sum(1)))
    while len(chosen)<budget and available.any():
        score=-((x-x[initial])**2).sum(1) if not chosen else nearest*(1+risk)
        score[~available]=-np.inf;idx=int(np.argmax(score));chosen.append(idx)
        nearest=np.minimum(nearest,((x-x[idx])**2).sum(1))
        for j,r in enumerate(rows):
            if r['capture_session']==rows[idx]['capture_session'] and abs(r['timestamp_ns']-rows[idx]['timestamp_ns'])<2000000000:
                available[j]=False
        available[idx]=False
    assert len(chosen)==budget
    return chosen


def old_parity():
    old=import_path('task_old_acquisition',OLD_CODE/'run.py')
    rows=read(DOC/'SPLIT_BINDING.json')['pool'];s=read(OLD_RAW/'pool/ACQUISITION_SIGNALS.json')
    x=np.load(OLD_RAW/'pool/FEATURES.npz')['features'];u=[v['selected_pose_instability'] for v in s]
    assert [r['image_sha256'] for r in rows]==[v['image_sha256'] for v in s]
    frozen=read(OLD_DOC/'SELECTION_LOCK.json')['selections']
    result={}
    for method,risk in [('diversity',np.zeros(len(rows))),('geometry_weighted_diversity',old.midranks(u))]:
        canonical=old.select(x,u,rows,30,method)
        actual=select(x,risk,rows)
        assert actual==canonical and [rows[i]['frame_id'] for i in actual]==frozen[method]
        result[method]=dict(count=30,exact_order=True)
    return result


def acquire():
    verify_lock()
    assert read(DOC/'TASK_RISK_VERDICT.json')['verdict']=='TASK_RISK_MECHANISM_PASS'
    rows=read(DOC/'SPLIT_BINDING.json')['pool']; risks=read(RAW/'TASK_RISK.json')
    assert [r['frame_id'] for r in rows]==[r['frame_id'] for r in risks]
    x=np.load(OLD_RAW/'pool/FEATURES.npz')['features']
    chosen=select(x,[r['R_task'] for r in risks],rows)
    write(DOC/'SELECTION_LOCK.json',dict(selections=dict(proposed=[rows[i]['frame_id'] for i in chosen],
        full174=[r['frame_id'] for r in rows]),task_risk_sha256=sha(RAW/'TASK_RISK.json'),
        GT_used_by_selection=False,budget=30,selection_seed=SEED,weights='1+R_task, not reranked'))


if __name__=='__main__':acquire()
