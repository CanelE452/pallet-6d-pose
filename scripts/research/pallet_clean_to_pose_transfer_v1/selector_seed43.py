"""Immutable runtime-path adapter for the already frozen seed42 GEO code.

This file, not selector_repeat/selector_compat, binds the follow-up seed43
evaluation layout. No source edits, new selector fits, or alternative scoring.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import common as C
from . import eval_student as E
from . import followup_pair as F
from . import selector_repeat as R
from . import selector_compat as S

STAGE='REPEAT_PRIMARY_S43'
CONTEXT=F.StageContext(STAGE)
ADAPTER_LOCK=CONTEXT.DOC/'SELECTOR_ADAPTER_INPUT_LOCK.json'
SCORE_LOCK=CONTEXT.DOC/'SELECTOR_ADAPTER_RESULT_LOCK.json'
METRIC_INDEX=CONTEXT.DOC/'CANDIDATE_METRICS_SELECTOR_INDEX.json'


def adapted_paths(seed):
    assert seed==43
    p=F.evaluation_paths(CONTEXT,seed)
    return dict(p,results=p['result'],oracle_result=METRIC_INDEX)


def activate():
    # Module-level path replacement is isolated to this CLI process. Existing
    # files and all current pose/feature/scoring code remain unchanged.
    E.paths=adapted_paths


def bind_adapter():
    p=adapted_paths(43)
    if ADAPTER_LOCK.exists():
        value=C.read(ADAPTER_LOCK)
        for binding in value['sources']:
            C.verify(binding)
        assert value['seed']==43 and value['stage']==STAGE
        return value
    assert C.read(CONTEXT.DOC/'PRIMARY_PROTOCOL.json')['seeds']==[43]
    source_paths=[Path(__file__),Path(F.__file__),Path(E.__file__),Path(R.__file__),Path(S.__file__),
        CONTEXT.DOC/'PRIMARY_PROTOCOL.json',p['candidate_lock'],p['lock'],p['metadata']]
    value=dict(created_at=C.now(),seed=43,stage=STAGE,
        evaluation_path_mapping={k:str(v.relative_to(C.ROOT)) for k,v in p.items()},
        runtime_only_path_adapter=True,unchanged_frozen_selector_code=True,
        pair_output_lock=str(R.paths(43)['lock'].relative_to(C.ROOT)),
        sources=[C.bind(path) for path in source_paths])
    C.save(ADAPTER_LOCK,value,True)
    return value


def freeze():
    bind_adapter();activate();R.freeze(43)
    print('SEED43_SELECTOR_PAIR_FROZEN_WITH_ADAPTER',flush=True)


def score():
    bind_adapter();activate()
    # Require prediction decisions to be sealed before opening candidate metrics.
    R.verify_pair(43)
    p=adapted_paths(43)
    source=p['raw']/'CANDIDATE_METRICS_LOCK.json'
    record=C.read(source);C.verify(record['file']);C.verify(record['candidate_lock'])
    assert record['candidate_lock']==C.bind(p['candidate_lock'])
    if METRIC_INDEX.exists():
        index=C.read(METRIC_INDEX)
        for binding in index['sources']+index['private_artifacts']:
            C.verify(binding)
    else:
        C.save(METRIC_INDEX,dict(created_at=C.now(),seed=43,
            private_artifacts=[record['file']],sources=[C.bind(source),C.bind(ADAPTER_LOCK)],
            purpose='Schema adapter to cached candidate metrics; no oracle selection or metric recomputation.',
            new_fits=0,GPU_seconds=0),True)
    R.score(43)
    if SCORE_LOCK.exists():
        for binding in C.read(SCORE_LOCK)['sources']:
            C.verify(binding)
    else:
        C.save(SCORE_LOCK,dict(created_at=C.now(),seed=43,stage=STAGE,
            unchanged_seed42_artifacts=True,new_selector_fits=0,
            sources=[C.bind(ADAPTER_LOCK),C.bind(METRIC_INDEX),C.bind(R.paths(43)['lock']),
                C.bind(R.paths(43)['result']),C.bind(Path(__file__))]),True)
    print('SEED43_SELECTOR_PAIR_SCORED_WITH_ADAPTER',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('freeze','score'))
    args=parser.parse_args();{'freeze':freeze,'score':score}[args.phase]()
