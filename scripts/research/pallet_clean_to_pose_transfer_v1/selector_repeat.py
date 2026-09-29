"""Apply the identical frozen old-GEO to the matched OCC RAW/REF pair.

Reuses the already sealed reference42 compatibility result. New outputs have
distinct per-arm paths; neither primary D9 nor prior selector locks are edited.
Only the predeclared seeds42/43 and OCC pair are exposed. No fit is possible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from . import common as C
from . import selector_compat as S

ORIGINAL_REF_LOCK=C.DOC/'ZERO_FIT_SELECTOR_LOCK.json'
ORIGINAL_REF_RESULT=C.DOC/'ZERO_FIT_SELECTOR_RESULTS.json'
ORIGINAL_REF_PRIVATE=C.RAW/'selector_compat_S42'


def paths(seed):
    assert seed in (42,43)
    return dict(raw=C.RAW/f'selector_pair_S{seed}',
        lock=C.DOC/f'SELECTOR_PAIR_LOCK_S{seed}.json',
        result=C.DOC/f'SELECTOR_PAIR_RESULTS_S{seed}.json')


def configure(seed,target):
    """Binding adapter only; scoring/features/normalization stay byte-identical."""
    assert seed in (42,43) and target in ('RAW','REF')
    S.SEED=seed;S.ARM=f'CLEAN_{target}_OCC'
    if seed==42 and target=='REF':
        S.PRIVATE=ORIGINAL_REF_PRIVATE;S.LOCK=ORIGINAL_REF_LOCK;S.RESULT=ORIGINAL_REF_RESULT
    else:
        S.PRIVATE=paths(seed)['raw']/target
        S.LOCK=C.DOC/f'ZERO_FIT_SELECTOR_LOCK_{target}_S{seed}.json'
        S.RESULT=C.DOC/f'ZERO_FIT_SELECTOR_RESULTS_{target}_S{seed}.json'
    return dict(private=S.PRIVATE,lock=S.LOCK,result=S.RESULT,arm=S.ARM)


def verify_pair(seed):
    path=paths(seed)['lock'];lock=C.read(path)
    assert lock['seed']==seed and lock['same_frozen_old_GEO']
    for binding in lock['sources']:
        C.verify(binding)
    for target in ('RAW','REF'):
        configured=configure(seed,target);child=S.verify_lock()
        assert child['seed']==seed and child['arm']==configured['arm']
        assert child['old_scorer']==lock['scorer']
    return lock


def freeze(seed):
    p=paths(seed)
    if p['lock'].exists():
        verify_pair(seed);print('SELECTOR_PAIR_ALREADY_FROZEN',seed,flush=True);return
    start=time.monotonic();bindings=[];children={};scorer=None
    for target in ('RAW','REF'):
        configured=configure(seed,target)
        S.freeze()
        child=S.verify_lock()
        assert child['seed']==seed and child['arm']==configured['arm']
        if scorer is not None:
            assert child['old_scorer']==scorer
        scorer=child['old_scorer'];children[target]=dict(arm=configured['arm'],
            lock=C.bind(configured['lock']),counts=child['counts'])
        bindings.append(C.bind(configured['lock']))
    C.save(p['lock'],dict(created_at=C.now(),seed=seed,arms=children,scorer=scorer,
        same_frozen_old_GEO=True,unchanged_primary_D9=True,
        final_pose_policy='Both arms select only from their cached current final D9 candidates; same94features/scorer/normalization/tie/fallback.',
        reference42_preserved=True,pair_names_locked_before_pair_scoring=True,
        GT_read_during_new_selector_decisions=False,selector_fits=0,student_fits=0,GPU_seconds=0,
        sources=bindings+[C.bind(Path(__file__)),C.bind(Path(S.__file__))],seconds=time.monotonic()-start),True)
    print('SELECTOR_PAIR_FROZEN',seed,{k:v['counts'] for k,v in children.items()},flush=True)


def score(seed):
    p=paths(seed);lock=verify_pair(seed)
    if p['result'].exists():
        result=C.read(p['result'])
        for binding in result['sources']+result['private_artifacts']:
            C.verify(binding)
        print('SELECTOR_PAIR_ALREADY_SCORED',seed,flush=True);return
    from . import eval_student as E
    from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M
    start=time.monotonic();metrics={};sources=[C.bind(p['lock'])];children={}
    for target in ('RAW','REF'):
        configured=configure(seed,target);S.score()
        result=C.read(configured['result'])
        for binding in result['sources']+result['private_artifacts']:
            C.verify(binding)
        data=C.read(configured['private']/'SELECTED_METRICS_PRIVATE.json')
        metrics[target+'_D9']=data['D9'];metrics[target+'_GEO']=data['OLD_GEO']
        if 'R0' in metrics:
            assert metrics['R0']==data['R0'] and metrics['OLD_REF']==data['OLD_REF']
        else:
            metrics['R0']=data['R0'];metrics['OLD_REF']=data['OLD_REF']
        children[target]=dict(lock=C.bind(configured['lock']),result=C.bind(configured['result']))
        sources.extend([children[target]['lock'],children[target]['result'],*result['private_artifacts']])
    ep=E.paths(seed);rows=C.read(ep['metadata']);groups=E.group_ids(rows)
    summaries={group:{arm:M.summarize(values[fid] for fid in ids) for arm,values in metrics.items()}
        for group,ids in groups.items()}
    comparisons=(('RAW_D9','REF_D9'),('RAW_GEO','REF_GEO'),('RAW_D9','RAW_GEO'),
        ('REF_D9','REF_GEO'),('R0','REF_GEO'),('OLD_REF','REF_GEO'))
    paired={group:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],ids)
        for before,after in comparisons} for group,ids in groups.items()}
    classification={after+'-minus-'+before:M.classify_candidate(summaries['NATURAL99'][after],summaries['NATURAL99'][before])
        for before,after in comparisons}
    private=p['raw']/'MATCHED_SELECTED_METRICS_PRIVATE.json';C.save(private,M.clean(metrics),True)
    result=dict(created_at=C.now(),seed=seed,groups=summaries,paired=paired,
        classification_NATURAL99=classification,children=children,
        matched_common_selector=True,scorer=lock['scorer'],
        coordinate_causal_contrast='REF_GEO-minus-RAW_GEO uses the same frozen selector. REF_GEO-minus-R0 combines student and selector changes.',
        student_pair_contract='Existing OCC pair unchanged; no new fitting in this module.',
        reference42_preserved=True,unchanged_2D=True,
        sources=sources+[C.bind(ep['metadata']),C.bind(Path(__file__)),C.bind(Path(M.__file__))],
        private_artifacts=[C.bind(private)],new_selector_fits=0,new_student_fits=0,GPU_seconds=0,
        seconds=time.monotonic()-start)
    C.save(p['result'],M.clean(result),True)
    print('SELECTOR_MATCHED_PAIR_SCORED',seed,classification,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','score'))
    parser.add_argument('--seed',type=int,choices=(42,43),default=42)
    args=parser.parse_args();{'freeze':freeze,'score':score}[args.phase](args.seed)
