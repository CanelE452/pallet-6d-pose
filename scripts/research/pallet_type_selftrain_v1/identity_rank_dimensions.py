"""Known-dimension interactions for identity ranking, without new labels or tags."""
import argparse
import copy
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from . import identity_rank_linear as L

I=L.I
C=I.C
PHASE='identity_rank_dimensions'
DOC=I.R.DOC/PHASE
RAW=I.R.RAW/PHASE
REGISTRY=C.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'


def plastic_dimensions():
    rows=C.read(REGISTRY)['objects']
    d=next(r for r in rows if r['object_type']=='plastic_standard_110x130x11')['physical_dimensions_m']
    return np.array([d['x'],d['z'],d['y']],np.float32)


def conditioned_difference(x,dimensions):
    d=L.difference(x)
    dim=np.asarray(dimensions,np.float32)
    if dim.shape[-1:]!=(3,) or not np.isfinite(dim).all() or not (dim>0).all():
        raise ValueError('Positive finite fixed physical W,D,H are required')
    dim=np.broadcast_to(dim,d.shape[:-1]+(3,))
    # Fixed physical axes, never camera-facing or candidate-swapped dimensions.
    footprint=np.log(dim[...,0]/dim[...,1])[...,None]
    height=np.log(dim[...,2]/np.sqrt(dim[...,0]*dim[...,1]))[...,None]
    return np.concatenate([d,d*footprint,d*height],axis=-1).astype(np.float32)


@contextmanager
def scope():
    with patch.object(I,'PHASE',PHASE),patch.object(I,'DOC',DOC),patch.object(I,'RAW',RAW):yield


def verify():
    with scope():return I.verify()


def prepare():
    parent=L.verify()
    side=np.load(I.SIDECAR)
    rows=np.array([r['row'] for r in parent['source_records']])
    s=np.load(L.H.RAW/'source.npz')
    assert len(rows)==len(s['x'])==2062
    np.testing.assert_array_equal(s['train'],[r['split']=='train' for r in parent['source_records']])
    dims=np.asarray(side['dimensions'][rows],np.float32)
    assert dims.shape==(2062,3) and np.isfinite(dims).all() and (dims>0).all()
    I.save_npz(RAW/'dimensions.npz',source=dims,real=plastic_dimensions(),source_rows=rows)
    p=copy.deepcopy(parent)
    p.update(purpose='Test whether known fixed physical dimensions improve large correspondence-error recovery without changing data membership, targets, candidate points or calibration rule.',
        learner='Linear antisymmetric ranker on [d, d*log(W/D), d*log(H/sqrt(W*D))], d=previous C2-averaged descriptor difference. 13944 weights,no bias. Same fixed physical dimensions for both candidates; no pose,GT pixels or camera-facing dimension swaps.',
        dimension_order='W,D,H = fixed object X,Z,Y,metres;ratios are unit invariant',
        real_dimensions_WDH=plastic_dimensions().tolist(),
        dimension_input_provenance='Existing corrected synthetic DIMENSION_SIDECAR and existing PLASTIC deployment registry; no new dimensions measured,no new tags,no new annotations.',
        controls='Compare to immutable identity_rank_linear SYN/MIX on identical source594/source-heldout1468/real217;only dimension interactions and consequent train-only lambda selection change.',
        differences='One bounded known-dimension interaction experiment after failed linear ranking. Historical reused DEV,not independent confirmation. Identity reassignment only; physical point set unchanged. Not evidence of new pixel localization.',
        caveats='Source-heldout1468 includes native predictions and artificial quarter permutations; artificial recovery is not real localization. Source development and real32 pseudo consistency are not independent confirmation. Historical Replay teacher used9manual training images,3overlapfull194. No new manual data or automatic promotion.',
        sources=parent['sources']+[C.bound(p) for p in [__file__,Path(__file__).with_name('test_identity_rank_dimensions.py'),
            L.__file__,L.DOC/'PROTOCOL.json',L.DOC/'RESULTS.json',REGISTRY,I.SIDECAR,RAW/'dimensions.npz']])
    C.freeze(DOC/'PROTOCOL.json',p)
    I.R.evaluation_protocol(PHASE,I.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('DIMENSIONS_PROTOCOL_FROZEN',p['real_dimensions_WDH'],'source_train594 held1468 real217',flush=True)


def fit(arm):
    p=verify();I.N.setup();I.N.E.gpu();assert torch.cuda.is_available()
    path=DOC/f'FIT_{arm}.json'
    if path.exists():C.verify(C.read(path)['checkpoint']);print('ALREADY_FIT',arm);return
    s=np.load(L.H.RAW/'source.npz');r=np.load(L.H.RAW/'real.npz');dims=np.load(RAW/'dimensions.npz')
    sx=conditioned_difference(s['x'],dims['source']);rx=conditioned_difference(r['x'],dims['real'])
    train=s['train']&s['eligible'];real=r['train']&r['eligible']
    assert train.sum()==594 and real.sum()==217
    tx=sx[train];ty=s['y'][train];xx=rx[real];yy=r['y'][real]
    folds=np.asarray(p['source_train_folds']);assert len(folds)==len(tx)
    results=[];started=time.monotonic()
    for lam in p['regularization_grid']:
        oof=np.empty((len(tx),2));stats=[]
        for f in range(3):
            fitting=folds!=f;held=~fitting
            scale=np.maximum(np.sqrt(np.mean(tx[fitting]**2,axis=0)),.01)
            w,status=L.train_linear(L.normalize(tx[fitting],scale),ty[fitting],
                L.normalize(xx,scale) if arm=='MIX' else None,yy if arm=='MIX' else None,lam)
            oof[held]=L.probability(L.normalize(tx[held],scale),w);stats.append(status)
        results.append(dict(lam=lam,logloss=L.logloss(oof,ty),accuracy=float((oof.argmax(-1)==ty).mean()),fold_fits=stats))
        print('DIMENSIONS_TRAIN_CV',arm,lam,results[-1]['logloss'],results[-1]['accuracy'],flush=True)
    best=min(results,key=lambda d:(d['logloss'],-d['lam']))
    scale=np.maximum(np.sqrt(np.mean(tx**2,axis=0)),.01)
    w,status=L.train_linear(L.normalize(tx,scale),ty,L.normalize(xx,scale) if arm=='MIX' else None,yy if arm=='MIX' else None,best['lam'])
    val=~s['train'];vp=L.probability(L.normalize(sx[val],scale),w)
    calibration=I.calibrate(vp,s['errors'][val],s['y'][val],s['eligible'][val])
    held=~r['train'];rp=L.probability(L.normalize(rx[held],scale),w)
    checkpoint=RAW/f'{arm}.npz';I.save_npz(checkpoint,weight=w,scale=scale)
    C.freeze(path,dict(complete=True,checkpoint=C.bound(checkpoint),protocol=C.bound(DOC/'PROTOCOL.json'),
        train_cv=results,selected_lambda=best['lam'],fit=status,calibration=calibration,
        source_train_accuracy=float((L.probability(L.normalize(tx,scale),w).argmax(-1)==ty).mean()),
        pseudo32_consistency_NOT_accuracy=float((rp.argmax(-1)==r['y'][held]).mean()),
        seconds=time.monotonic()-started,real_evaluation_seen=False,auto_promoted=False))
    print('DIMENSIONS_FIT_COMPLETE',arm,'lambda',best['lam'],'source_accuracy',calibration['pair_accuracy'],
        'threshold',calibration['threshold'],I.N.E.gpu(),flush=True)


def infer():
    verify();I.N.setup();I.N.E.gpu();assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=True
    fits={a:C.read(DOC/f'FIT_{a}.json') for a in I.ARMS}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in I.ARMS},before_real_features=True))
    models={}
    for a,f in fits.items():C.verify(f['checkpoint']);models[a]=np.load(C.ROOT/f['checkpoint']['path'])
    old={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    predictions={a:[] for a in I.ARMS};decisions=[];dim=plastic_dimensions()
    extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0)
    try:
        for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            base=old[r['id']];x,_,_,_,_=I.capture_descriptor(extractor,im,base['prediction'])
            d=conditioned_difference(x,dim);choices={}
            for a,m in models.items():
                prob=L.probability(L.normalize(d,m['scale']),m['weight']);threshold=fits[a]['calibration']['threshold']
                choice=I.F.choose(prob,threshold);pred=copy.deepcopy(base['prediction'])
                if choice:I.top(pred)['keypoints_xy'][:8]=np.asarray(I.top(pred)['keypoints_xy'])[I.F.QUARTER][:8].tolist()
                I.assert_preserved(base['prediction'],pred)
                predictions[a].append(dict(id=r['id'],kind='PLASTIC',prediction=pred,raw_hw=base['raw_hw']))
                choices[a]=dict(probabilities=prob.tolist(),threshold=threshold,choice=choice)
            decisions.append(dict(id=r['id'],choices=choices))
            if (i+1)%50==0:print('DIMENSIONS_REAL_PREDICTIONS',i+1,'/194',flush=True)
    finally:extractor.close()
    for a,rows in predictions.items():C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,checkpoint=fits[a]['checkpoint'],records=rows,GT_free=True))
    C.freeze(RAW/'DECISIONS.json',decisions)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in I.ARMS]+[C.bound(RAW/'DECISIONS.json')],before_GT_scoring=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','infer','report']);p.add_argument('arm',nargs='?',choices=I.ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.arm)
    elif a.action=='infer':infer()
    else:
        with scope():I.report()
