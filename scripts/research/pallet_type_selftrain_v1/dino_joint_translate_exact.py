"""Correction: full source score frontier instead of the truncated0..16 grid."""
import argparse
import copy
from contextlib import contextmanager
from unittest.mock import patch
from . import dino_joint_translate as T

C=T.C;D=T.D;PHASE='dino_joint_translate_exact';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE


@contextmanager
def scope():
    with patch.object(T,'PHASE',PHASE),patch.object(T,'DOC',DOC),patch.object(T,'RAW',RAW):yield


def feasible(s,combined):
    return s['PCK10']>=s['input_PCK10']-.01 and s['PCK20']>=s['input_PCK20'] and s['P90']<=1.1*s['input_P90'] and s['damaged']<=.01*s['good'] and combined['beneficial_precision']>=.95


def calibrate(rows):
    raw=[r for r in rows if r['view']==0]
    thresholds=sorted({0.,1e9,*[r['proposal']['gain'] for r in rows if T.M.changed(r['proposal'])]})
    grid=[];best=None
    for t in thresholds:
        s=T.stats(raw,t);combined=T.stats(rows,t);r=dict(threshold=t,raw=s,combined=combined,feasible=feasible(s,combined));grid.append(r)
        key=(s['recovered'],combined['recovered'],t)
        if r['feasible'] and (best is None or key>best[0]):best=(key,r)
    assert best is not None
    return best[1],grid


def prepare():
    original=T.verify();rows=C.read(T.RAW/'SOURCE_PROPOSALS.json')['records']
    selected,grid=calibrate([r for r in rows if r['row'] in original['calibration_rows']])
    p=copy.deepcopy(original)
    p.update(calibration='CORRECTION:all distinct observed source-calibration gains,plus0 and1e9null. Same strict gain>threshold comparator and quality constraints/tiebreak. Covers every realizable selected subset;no realGT choice.',
        correction='Original fixed0..16 grid omitted viable source thresholds above16. Preserve original null result;change only threshold candidate set. Same frozen heads,320CPU source proposals,source split,registration math and output preservation.',
        sources=original['sources']+[C.bound(f) for f in [__file__,C.HERE/'test_dino_joint_translate_exact.py',T.DOC/'PROTOCOL.json',T.DOC/'COMPLETION_AUDIT.json',
            T.DOC/'EXACT_SOURCE_FRONTIER_DIAGNOSTIC.json',T.DOC/'CPU_SOURCE_ADAPTER.json',T.RAW/'SOURCE_PROPOSALS.json']])
    C.freeze(DOC/'PROTOCOL.json',p);D.R.evaluation_protocol(PHASE,[T.ARM],p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    C.freeze(RAW/'SOURCE_PROPOSALS.json',C.read(T.RAW/'SOURCE_PROPOSALS.json'))
    test=[r for r in rows if r['row'] in p['test_rows']];threshold=selected['threshold']
    C.freeze(DOC/'FIT_REGISTER.json',dict(threshold=threshold,calibration=selected,grid=grid,
        test_raw=T.stats([r for r in test if r['view']==0],threshold),test_combined=T.stats(test,threshold),
        source_proposals=C.bound(RAW/'SOURCE_PROPOSALS.json'),GT_real_used=False,new_training_steps=0,
        corrected_only_truncated_calibration_grid=True))
    C.freeze(DOC/'DECISION_LOCK.json',dict(fit=C.bound(DOC/'FIT_REGISTER.json'),before_real_inference=True))
    print('REGISTER_EXACT_LOCKED',selected,'test',T.stats([r for r in test if r['view']==0],threshold),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        with scope():T.infer();T.report()
