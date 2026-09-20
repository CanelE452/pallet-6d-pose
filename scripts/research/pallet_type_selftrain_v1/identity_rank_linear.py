"""Regularized, antisymmetric linear identity ranking; source-only selection."""
import argparse
import copy
import hashlib
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.nn import functional as TF
from . import identity_rank as I
from . import identity_rank_hard as H

C=I.C
PHASE='identity_rank_linear'
DOC=I.R.DOC/PHASE
RAW=I.R.RAW/PHASE
LAMBDA=[.001,.01,.1,1.]


def difference(x):
    x=np.asarray(x,np.float32);assert x.shape[-3:]==(2,2,4648)
    # Candidate exchange negates this descriptor; no native-class intercept.
    q=x.mean(axis=-2)
    return q[...,1,:]-q[...,0,:]


def normalize(x,scale):
    return np.clip(np.asarray(x,np.float32)/scale,-8,8).astype(np.float32)


def train_linear(sx,sy,rx,ry,lam,device='cuda'):
    sx=torch.as_tensor(sx,device=device);sy=torch.as_tensor(sy,dtype=torch.float32,device=device)
    if rx is not None:
        rx=torch.as_tensor(rx,device=device);ry=torch.as_tensor(ry,dtype=torch.float32,device=device)
    w=torch.zeros(sx.shape[1],device=device,requires_grad=True)
    opt=torch.optim.LBFGS([w],lr=1.,max_iter=200,tolerance_grad=1e-6,tolerance_change=1e-9,
        history_size=20,line_search_fn='strong_wolfe')
    calls=[]
    def closure():
        opt.zero_grad(set_to_none=True)
        loss=TF.binary_cross_entropy_with_logits(sx@w,sy)
        if rx is not None:loss=.5*loss+.5*TF.binary_cross_entropy_with_logits(rx@w,ry)
        loss=loss+.5*lam*w.square().sum();assert torch.isfinite(loss)
        loss.backward();calls.append(float(loss.detach()));return loss
    opt.step(closure)
    with torch.no_grad():assert torch.isfinite(w).all()
    return w.detach().cpu().numpy(),dict(iterations=int(opt.state[w]['n_iter']),closure_calls=len(calls),
        initial_objective=calls[0],final_objective=calls[-1],weight_norm=float(w.detach().norm()))


def probability(x,w):
    z=np.asarray(x)@w
    p=torch.sigmoid(torch.as_tensor(z,dtype=torch.float64)).numpy()
    return np.stack([1-p,p],axis=-1)


def logloss(prob,y):
    return float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-15)).mean())


@contextmanager
def scope():
    with patch.object(I,'PHASE',PHASE),patch.object(I,'DOC',DOC),patch.object(I,'RAW',RAW):yield


def prepare():
    parent=C.read(H.DOC/'PROTOCOL.json')
    for b in parent['sources']:C.verify(b)
    s=np.load(H.RAW/'source.npz');r=np.load(H.RAW/'real.npz')
    assert s['x'].shape[0]==2062 and s['train'].sum()==594 and r['train'].sum()==217
    records=[p for p,t in zip(parent['source_records'],s['train']) if t]
    fold=np.array([int(hashlib.sha256(p['scenario'].encode()).hexdigest()[:8],16)%3 for p in records])
    y=s['y'][s['train']]
    assert all(len(np.unique(y[fold==i]))==2 for i in range(3))
    protocol=copy.deepcopy(parent)
    protocol.update(purpose='Reduce overfitting/overconfidence of the previous MLP without changing candidates, data, labels or source safety criteria.',
        learner='Linear scalar on difference of the two C2-averaged descriptors;4648weights,no bias. Source-fold RMS scale(min.01),symmetric clip8;sigmoid produces candidate probability. No source native-class flag.',
        regularization_grid=LAMBDA,
        model_selection='3fold scenario-hash CV on source TRAIN594 only;refit normalization in each fold;choose lambda by pooled source OOF logloss,tie stronger lambda. MIX may use fixed real217 in every source fold;no source-heldout or real-eval model selection.',
        source_train_folds=fold.tolist(),
        optimization='Full-batch convex BCE + lambda/2 ||w||^2,zero initialization,torch LBFGS strong_wolfe,max200iterations,tolerance_grad1e-6,tolerance_change1e-9. SYN mean(source);MIX .5mean(source)+.5mean(real). No source/real loss-weight sweep.',
        calibration='Reuse EXACT previous source-heldout1468 rule and threshold grid only after lambda frozen by trainCV. No real threshold/model sweep. None -> exact R0 fallback.',
        controls='Same SYN versus MIX domains and full input membership;optimizer/architecture changed together,not an isolated regularization-only ablation.',
        differences='Current attempt follows failed MLP,reported search history remains immutable. Reused source validation and real DEV,not independent confirmation.',
        sources=parent['sources']+[C.bound(p) for p in [__file__,Path(__file__).with_name('test_identity_rank_linear.py'),
            H.DOC/'PROTOCOL.json',H.RAW/'source.npz',H.RAW/'real.npz',H.DOC/'FIT_SYN.json',H.DOC/'FIT_MIX.json']])
    C.freeze(DOC/'PROTOCOL.json',protocol)
    I.R.evaluation_protocol(PHASE,I.ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('LINEAR_PROTOCOL_FROZEN','train594','calibration1468','pseudo217',flush=True)


def verify():
    with scope():return I.verify()


def fit(arm):
    p=verify();I.N.setup();I.N.E.gpu();assert torch.cuda.is_available()
    path=DOC/f'FIT_{arm}.json'
    if path.exists():C.verify(C.read(path)['checkpoint']);print('ALREADY_FIT',arm);return
    s=np.load(H.RAW/'source.npz');r=np.load(H.RAW/'real.npz')
    sx=difference(s['x']);rx=difference(r['x']);train=s['train']&s['eligible'];real=r['train']&r['eligible']
    assert train.sum()==594 and real.sum()==217
    tx=sx[train];ty=s['y'][train];xx=rx[real];yy=r['y'][real]
    folds=np.asarray(p['source_train_folds']);assert len(folds)==len(tx)
    results=[];started=time.monotonic()
    for lam in LAMBDA:
        oof=np.empty((len(tx),2));stats=[]
        for f in range(3):
            fitting=folds!=f;held=~fitting
            scale=np.maximum(np.sqrt(np.mean(tx[fitting]**2,axis=0)),.01)
            w,status=train_linear(normalize(tx[fitting],scale),ty[fitting],
                normalize(xx,scale) if arm=='MIX' else None,yy if arm=='MIX' else None,lam)
            oof[held]=probability(normalize(tx[held],scale),w);stats.append(status)
        results.append(dict(lam=lam,logloss=logloss(oof,ty),accuracy=float((oof.argmax(-1)==ty).mean()),fold_fits=stats))
        print('LINEAR_TRAIN_CV',arm,results[-1]['lam'],results[-1]['logloss'],results[-1]['accuracy'],flush=True)
    best=min(results,key=lambda d:(d['logloss'],-d['lam']))
    scale=np.maximum(np.sqrt(np.mean(tx**2,axis=0)),.01)
    w,status=train_linear(normalize(tx,scale),ty,normalize(xx,scale) if arm=='MIX' else None,yy if arm=='MIX' else None,best['lam'])
    val=~s['train'];vp=probability(normalize(sx[val],scale),w)
    calibration=I.calibrate(vp,s['errors'][val],s['y'][val],s['eligible'][val])
    held=~r['train'];rp=probability(normalize(rx[held],scale),w)
    checkpoint=RAW/f'{arm}.npz';I.save_npz(checkpoint,weight=w,scale=scale)
    C.freeze(path,dict(complete=True,checkpoint=C.bound(checkpoint),protocol=C.bound(DOC/'PROTOCOL.json'),
        train_cv=results,selected_lambda=best['lam'],fit=status,calibration=calibration,
        source_train_accuracy=float((probability(normalize(tx,scale),w).argmax(-1)==ty).mean()),
        pseudo32_consistency_NOT_accuracy=float((rp.argmax(-1)==r['y'][held]).mean()),
        seconds=time.monotonic()-started,real_evaluation_seen=False,auto_promoted=False))
    print('LINEAR_FIT_COMPLETE',arm,'lambda',best['lam'],'source_accuracy',calibration['pair_accuracy'],
        'threshold',calibration['threshold'],I.N.E.gpu(),flush=True)


def infer():
    verify();I.N.setup();I.N.E.gpu();assert torch.cuda.is_available();torch.backends.cudnn.allow_tf32=True
    fits={a:C.read(DOC/f'FIT_{a}.json') for a in I.ARMS}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in I.ARMS},before_real_features=True))
    models={}
    for a,f in fits.items():C.verify(f['checkpoint']);models[a]=np.load(C.ROOT/f['checkpoint']['path'])
    old={r['id']:r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    predictions={a:[] for a in I.ARMS};decisions=[]
    extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0)
    try:
        for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
            C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            base=old[r['id']];x,_,_,_,_=I.capture_descriptor(extractor,im,base['prediction']);d=difference(x);choices={}
            for a,m in models.items():
                prob=probability(normalize(d,m['scale']),m['weight']);threshold=fits[a]['calibration']['threshold']
                choice=I.F.choose(prob,threshold);pred=copy.deepcopy(base['prediction'])
                if choice:I.top(pred)['keypoints_xy'][:8]=np.asarray(I.top(pred)['keypoints_xy'])[I.F.QUARTER][:8].tolist()
                I.assert_preserved(base['prediction'],pred)
                predictions[a].append(dict(id=r['id'],kind='PLASTIC',prediction=pred,raw_hw=base['raw_hw']))
                choices[a]=dict(probabilities=prob.tolist(),threshold=threshold,choice=choice)
            decisions.append(dict(id=r['id'],choices=choices))
            if (i+1)%50==0:print('LINEAR_REAL_PREDICTIONS',i+1,'/194',flush=True)
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
