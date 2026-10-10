"""Record every Stage-1 failure and oracle transition, without rerunning F."""
from collections import defaultdict
from . import common as C
from . import verdict as V


def run():
    path=C.DOC/'STAGE1_FAILURES.json'
    assert not path.exists()
    populations={}
    for population in ('REAL','SYNTH','AUX'):
        groups=defaultdict(list)
        for row in C.rows(C.DOC/f'STAGE1_ROWS_{population}.jsonl.gz'):
            groups[f"{row['backbone']}::{row['method']}::seed{row['seed']}"].append(row)
        result={}
        for name,rows in sorted(groups.items()):
            fields={f'{arm}_{label}_ids':[] for arm in ('S0','ORACLE')
                    for label in ('confusion','unsuccessful','unavailable')}
            fields.update({label+'_ids':[] for label in
                           ('confusion_recovery','confusion_damage','success_damage','success_recovery','hypothesis_change')})
            for r in sorted(rows,key=lambda r:r['id']):
                a,b=(V.indicators(r['pose'][arm]) for arm in ('S0','ORACLE'))
                for arm,v in [('S0',a),('ORACLE',b)]:
                    for label,hit in [('confusion',bool(v['confusion_rate'])),
                                      ('unsuccessful',not bool(v['success_rate'])),
                                      ('unavailable',not v['available'])]:
                        if hit:fields[f'{arm}_{label}_ids'].append(r['id'])
                for label,hit in [('confusion_recovery',a['confusion_rate'] and not b['confusion_rate']),
                                  ('confusion_damage',not a['confusion_rate'] and b['confusion_rate']),
                                  ('success_damage',a['success_rate'] and not b['success_rate']),
                                  ('success_recovery',not a['success_rate'] and b['success_rate']),
                                  ('hypothesis_change',r['hypS0']!=r['hypOracle'])]:
                    if hit:fields[label+'_ids'].append(r['id'])
            result[name]=dict(frames=len(rows),**fields,
                counts={name.removesuffix('_ids'):len(ids) for name,ids in fields.items()})
        populations[population]=result
    C.write(path,dict(phase='STAGE1_DIAGNOSTIC_ONLY',populations=populations,
        oracle='fixed GT parity supplied; not best-metric candidate search',
        missing_in_full_denominator=True,additional_F_calls=0))


if __name__=='__main__':run()
