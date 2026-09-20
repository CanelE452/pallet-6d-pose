"""Posthoc accounting only; never change thresholds or stored predictions."""
import numpy as np
from . import dino_evidence_gate as E

C=E.C


def stats(x):
    x=np.asarray(x);return dict(n=len(x),quantiles=np.quantile(x,[0,.5,.9,.99,1]).tolist() if len(x) else [])


def main():
    p=E.verify();receipt=C.read(E.DOC/'SOURCE_CACHE.json')
    inference={r['id']:r for r in C.read(E.RAW/'INFERENCE_RECEIPTS.json')}
    base={r['id']:r for r in C.read(E.R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC'}
    perms=next(r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects'] if r['object_type']==C.TYPES['PLASTIC'])
    tables={}
    for a in E.ARMS:
        C.verify(receipt['artifacts'][a])
        with np.load(C.ROOT/receipt['artifacts'][a]['path']) as z:
            train=np.isin(z['row'],p['source_train_rows']);positive=(z['after']<=10)&(z['after']+5<z['before'])
            dist=z['features'][:,8];source=dict(clean_positive=stats(dist[train&positive&(z['view']==0)]),
                artificial_positive=stats(dist[train&positive&(z['view']>0)]),
                all_positive_move_over_30pct=int((train&positive&(dist>.3)).sum()),all_positive=int((train&positive).sum()))
        rows=[]
        for r in C.read(E.V.RAW/f'SCREEN_{a}.json')['metrics']:
            key=r['id'];b=base[key]
            if not b['matched']:continue
            for j,(old,new) in enumerate(zip(b['canonical_errors'],r['canonical_errors'])):
                if old is None or old<=80 or new>10:continue
                native=perms[r['branch']].index(j);d=inference[key]['decisions'][a]
                rows.append(dict(id=key,GT_corner=j,candidate_native_corner=native,before_canonical_px=old,
                    unconditional_after_px=new,branch_changed=b['branch']!=r['branch'],
                    candidate_gate_score=d['scores'][native],accepted=bool(d['accepted'][native]),
                    move_over_boxdiag=d['features'][native][8],feature_vector=d['features'][native]))
        tables[a]=dict(source_training_positive_movement=source,large_recovered_before_gate=len(rows),
            same_native_candidate_accepted=sum(r['accepted'] for r in rows),
            involving_changed_global_C2_branch=sum(r['branch_changed'] for r in rows),
            candidate_movement=stats([r['move_over_boxdiag'] for r in rows]),cases=rows)
    C.freeze(E.DOC/'REJECTED_LARGE_DIAGNOSTIC.json',dict(status='POSTHOC_GT_ACCOUNTING_NOT_SELECTION',tables=tables,
        warning='Canonical recovery may change the global C2 branch; an accepted native candidate alone need not reproduce it. Quantiles do not establish a causal explanation.',
        threshold_changed=False,predictions_changed=False,evidence=[C.bound(__file__),C.bound(E.DOC/'PROTOCOL.json'),
            C.bound(E.DOC/'SOURCE_CACHE.json'),C.bound(E.RAW/'INFERENCE_RECEIPTS.json'),C.bound(E.V.DOC/'RESULTS.json')]))
    print({a:{k:v for k,v in t.items() if k!='cases'} for a,t in tables.items()})


if __name__=='__main__':main()
