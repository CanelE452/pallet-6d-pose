"""S3 subset statistics with the frozen full-319, 13-session draw table.

Stage1 statistics.py is immutable. This companion retains its numerical and
binary definitions while deriving S3 uncertainty from a subset of the original
REAL population. Empty sessions retain their original bootstrap multiplicities.
"""
from collections import defaultdict
import numpy as np
from . import common as C
from . import statistics as M
from . import verdict as V


def compare_floor(baseline,changed,master):
    population='REAL_DEV';rule='S3'
    assert len(baseline)==len(changed)
    master_sessions={}
    for r in master:
        fid=r['id'];session=r['session']
        assert fid not in master_sessions or master_sessions[fid]==session
        master_sessions[fid]=session
    master_ids=sorted(master_sessions)
    assert len(master_ids)==319 and len(set(master_sessions.values()))==13
    master_draws=M.Draws([master_sessions[i] for i in master_ids],'cluster')
    assert master_draws.sha256==V.definitions()['real_cluster_draw_sha256']
    master_index={fid:index for index,fid in enumerate(master_ids)}
    groups=defaultdict(list)
    for r in baseline:groups[r['method']].append(r)
    ci_index={(r['seed'],r['method'],r['id']):r for r in changed}
    assert len(ci_index)==len(changed)
    assert set(ci_index)=={(r['seed'],r['method'],r['id']) for r in baseline}
    assert set(groups)==set(C.METHODS)
    metrics={};paired={};failures={}
    for method,base in sorted(groups.items()):
        ids=sorted({r['id'] for r in base});seeds=sorted({r['seed'] for r in base})
        assert tuple(seeds)==C.SEEDS and len({master_sessions[i] for i in ids})>3
        bi={(r['seed'],r['id']):r for r in base};assert len(bi)==len(base)
        ordered_b=[[bi[s,i] for i in ids] for s in seeds]
        ordered_c=[[ci_index[s,method,i] for i in ids] for s in seeds]
        for rows in ordered_b:
            assert all(r['session']==master_sessions[r['id']] for r in rows)
        draws=master_draws.subset([master_index[i] for i in ids])
        bootstrap=dict(draws.metadata(),participating_sessions=int((draws.counts>0).sum()),
            sessions_with_zero_scope_frames=int((draws.counts==0).sum()),
            nonempty_resamples=int((draws.denominator>0).sum()),
            zero_denominator_policy='exclude only resamples with no eligible frames',
            scope_frame_ids=ids)
        bv=[M.vectors([r['pose'] for r in rs]) for rs in ordered_b]
        cv=[M.vectors([r['pose'] for r in rs]) for rs in ordered_c]
        bm,cm=M.average_vectors(bv),M.average_vectors(cv)
        metrics[method]=dict(frames=len(ids),bootstrap=bootstrap,
            seed_mean={'S0':M.summary(bm,draws),rule:M.summary(cm,draws)},
            per_seed={str(s):{'S0':M.summary(bv[j],draws),rule:M.summary(cv[j],draws)} for j,s in enumerate(seeds)})
        def contrast(a,b):
            result={}
            for key in (*M.NUMERIC,'confusion_rate','success_rate'):
                delta=a[key]-b[key];valid=np.isfinite(delta)
                result[key]=dict(delta=float(delta[valid].mean()) if valid.any() else None,
                    CI95=draws.interval(delta),paired_frames=int(valid.sum()),
                    binary_before_seed_mean=key.endswith('_rate'))
            return result
        meancontrast=contrast(cm,bm)
        perseed={str(s):contrast(cv[j],bv[j]) for j,s in enumerate(seeds)}
        for key in (*M.NUMERIC,'confusion_rate','success_rate'):
            deltas=[perseed[str(s)][key]['delta'] for s in seeds]
            meancontrast[key].update(per_seed_delta=deltas,
                improved_seeds=sum(x is not None and (x>0 if key in ('success_rate','IoU3D') else x<0) for x in deltas))
        paired[method]=dict(seed_mean=meancontrast,per_seed=perseed,bootstrap=bootstrap)
        failures[method]={}
        for j,s in enumerate(seeds):
            fields={name:[] for name in ('success_to_failure_ids','failure_to_success_ids',
                'confusion_recovery_ids','confusion_damage_ids','hypothesis_change_ids','fallback_ids','unavailable_ids')}
            transitions=[]
            for b,c in zip(ordered_b[j],ordered_c[j]):
                fid=b['id'];assert c['id']==fid and c['qFinal']==b['qFinal'],'Coordinates must remain identical'
                ib,ic=V.indicators(b['pose']),V.indicators(c['pose'])
                if ib['success_rate'] and not ic['success_rate']:fields['success_to_failure_ids'].append(fid)
                if not ib['success_rate'] and ic['success_rate']:fields['failure_to_success_ids'].append(fid)
                if ib['confusion_rate'] and not ic['confusion_rate']:fields['confusion_recovery_ids'].append(fid)
                if not ib['confusion_rate'] and ic['confusion_rate']:fields['confusion_damage_ids'].append(fid)
                if b['hyp']!=c['hyp']:
                    fields['hypothesis_change_ids'].append(fid);transitions.append(dict(id=fid,before=b['hyp'],after=c['hyp']))
                if c.get('fallback'):fields['fallback_ids'].append(fid)
                if not ic['available']:fields['unavailable_ids'].append(fid)
            failures[method][str(s)]={**fields,**{key.removesuffix('_ids')+'_count':len(value) for key,value in fields.items()},'transitions':transitions}
    return dict(population=population,rule=rule,metrics=metrics,paired=paired,failures=failures,
        primary=paired[V.PRIMARY_METHOD]['seed_mean'],descriptive_only=True,
        uncertainty='frozen full319/13-session master draws, eligible frame subset; post-hoc feasibility only')
