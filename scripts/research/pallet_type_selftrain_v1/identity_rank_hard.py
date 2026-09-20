"""Natural source identity-error mining, with the original failed screen intact."""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import identity_rank as I
from . import train as T

C=I.C
PHASE='identity_rank_hard'
DOC=I.R.DOC/PHASE
RAW=I.R.RAW/PHASE


@contextmanager
def scope():
    with patch.object(I,'PHASE',PHASE),patch.object(I,'DOC',DOC),patch.object(I,'RAW',RAW):yield


def mine_arrays(a,side):
    q=np.asarray(a['points']);gt=np.asarray(a['gt_points'])
    valid=np.asarray(a['gt_valid'])[:,:8]&np.isfinite(gt[:,:8]).all(-1)
    box=np.asarray(a['boxes']);size=np.linalg.norm(box[:,2:]-box[:,:2],axis=-1)
    e=np.stack([np.linalg.norm(q[:,p,:][:,:8]-gt[:,:8],axis=-1) for p in I.F.PERMS],axis=1)
    mean=np.where(valid[:,None],e,0).sum(-1)/np.maximum(valid.sum(-1)[:,None],1)
    pair=np.minimum(mean[:,::2],mean[:,1::2])
    eligible=np.asarray(a['matched'])&(side['order']==2)&side['group_valid'][:,:2].all(-1)
    eligible&=(valid.sum(-1)>=6)&np.isfinite(pair).all(-1)&np.isfinite(size)&(size>1)
    positive=eligible&(pair[:,0]-pair[:,1]>.02*size)&(pair[:,0]>.10*size)&(pair[:,1]<.03*size)
    normal=eligible&(pair[:,0]<.03*size)&(pair[:,1]>.10*size)
    return eligible,positive,normal,pair,size


def prepare():
    parent=I.verify();data=I.SourceData().data;side=np.load(I.SIDECAR)
    eligible,positive,normal,pair,size=mine_arrays(data.arrays,side)
    train=data.partitions=='train';held=data.partitions=='heldout'
    positives=np.flatnonzero(positive&train);assert len(positives)>=100
    val=np.flatnonzero(eligible&held);assert (positive&held).sum()>=10
    rng=np.random.default_rng(20260924)
    negatives=rng.choice(np.flatnonzero(normal&train),len(positives),replace=False)
    chosen=np.sort(np.r_[positives,negatives]);records=[]
    for split,rows in [('train',chosen),('val',val)]:
        for row in rows:
            meta=data.source['records'][int(data.indices[row])]
            assert meta['source_kind']=='synthetic'
            np.testing.assert_array_equal(side['permutations'][row,:2],I.F.PERMS[:2])
            records.append(dict(row=int(row),id=meta['id'],scenario=meta['scenario_id'],split=split,
                image=C.bound(meta['image']),label=C.bound(meta['label']),
                natural_identity_error=bool(positive[row]),native_error_normalized=float(pair[row,0]/size[row]),
                quarter_error_normalized=float(pair[row,1]/size[row])))
    tr={r['scenario'] for r in records if r['split']=='train'};va={r['scenario'] for r in records if r['split']=='val'}
    assert not tr&va
    train_hash={r['image']['sha256'] for r in records if r['split']=='train'}
    val_hash={r['image']['sha256'] for r in records if r['split']=='val'}
    forbidden={r['image']['sha256'] for r in C.read(I.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert not train_hash&val_hash and not forbidden&(train_hash|val_hash)
    protocol=copy.deepcopy(parent);protocol['source_records']=records
    protocol.update(purpose='Same absolute identity ranker, now trained on actual frozen-R0 source identity failures rather than almost exclusively native-correct source.',
        mining='Source TRAIN only: all order2 matched>=6valid rows with native error>.10boxdiag and quarter error<.03boxdiag and improvement>.02boxdiag;equal number random native-good/quarter-wrong controls. Unchanged predicted points/features. No synthetic GT coordinates used as input.',
        validation='All eligible source HELDOUT rows, not only positives or an artificially balanced subset. Source train/heldout scenario and exact image hash disjoint. Calibration/selection partitions excluded. Still historical source development, not a global untouched holdout.',
        differences='Source composition and validation coverage change; architecture/loss/steps/optimizer/real217 unchanged. Both SYN and MIX retained. Previous identity_rank results/thresholds remain immutable.',
        sources=parent['sources']+[C.bound(__file__),C.bound(Path(__file__).with_name('test_identity_rank_hard.py')),
            C.bound(I.DOC/'RESULTS.json'),C.bound(I.RAW/'real.npz'),C.bound(I.DOC/'REAL_COMPLETE.json')],
        counts=dict(train_positive=len(positives),train_normal=len(negatives),validation=len(val),
            validation_strong_positive=int((positive&held).sum())))
    C.freeze(DOC/'PROTOCOL.json',protocol)
    I.R.evaluation_protocol(PHASE,I.ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    T.link(I.RAW/'real.npz',RAW/'real.npz')
    real=copy.deepcopy(C.read(I.DOC/'REAL_COMPLETE.json'));real['features']=C.bound(RAW/'real.npz')
    C.freeze(DOC/'REAL_COMPLETE.json',real)
    print('HARD_IDENTITY_PROTOCOL',protocol['counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','source','fit','infer','report']);p.add_argument('arm',nargs='?',choices=I.ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        with scope():
            if a.action=='source':I.source_features()
            elif a.action=='fit':I.fit(a.arm)
            elif a.action=='infer':I.infer()
            else:I.report()
