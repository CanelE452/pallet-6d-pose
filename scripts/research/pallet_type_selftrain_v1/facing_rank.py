"""A calibrated-camera, six-feature role selector. Not a localization refiner."""
import argparse
import copy
import hashlib
import io
import json
import time
import zipfile
from contextlib import ExitStack, contextmanager, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import identity_rank_linear as L
from . import facing_rank_features as F

I=L.I; C=I.C
PHASE='facing_rank'
DOC=I.R.DOC/PHASE; RAW=I.R.RAW/PHASE


@contextmanager
def scope():
    with patch.object(I,'PHASE',PHASE), patch.object(I,'DOC',DOC), patch.object(I,'RAW',RAW): yield


def verify():
    with scope(): return I.verify()


def prepare():
    parent=L.verify()
    protocol=dict(arms=I.ARMS, source_records=parent['source_records'], real_train=parent['real_train'],
        real_probe=parent['real_probe'], source_train_folds=parent['source_train_folds'],
        purpose='Use the empirically checked source front-facing convention as a calibrated parallel-edge feature, then learn absolute role preference without RGB domain features.',
        inputs='R0 nine2Dpoints and camera intrinsics only. Fit two horizontal vanishing directions with normalized ray-plane SVD. Six odd features from absolute viewing-cosine difference and SVD/orthogonality diagnostics. No GT pose, depth, new CAD, new manual labels/tags or dimensions.',
        feature_contract='C2 invariant, quarter reassignment negates features; six-weight linear scalar without intercept. Missing/degenerate features give zeros and retain R0. No frame/corner rejection.',
        labels='Exact previous natural-hard source594 and source-heldout1468; same existing real217 and pseudo32. Reuse previous target masks/errors/eligibility. No new pool filtering.',
        learner='Same source-train scenario3fold CV lambda [.001,.01,.1,1], RMS normalization(min.01),symmetric clip8,fullbatch LBFGS200 and SYN versus MIX loss as prior linear selector; six parameters, no RGB input.',
        calibration='Unchanged source-heldout rule and grid from identity_rank.calibrate; no real threshold selection. Failure means exact R0 fallback. Report natural and artificial recovery separately.',
        limits='Different inputs/feature representation, not an isolated architecture ablation. Same physical point set: cannot recover a target absent from all input corners. Repeated DEV, not untouched confirmation. Historical Replay teacher has3manual training overlaps in194.',
        real_gate=parent['real_gate'], regularization_grid=L.LAMBDA,
        new_annotations=0,new_tags=0,new_frame_filters=0,auto_promote=False,
        sources=parent['sources']+[C.bound(p) for p in [__file__,F.__file__,Path(__file__).with_name('test_facing_rank_features.py'),
            L.__file__,L.DOC/'PROTOCOL.json',L.DOC/'RESULTS.json',L.H.RAW/'source.npz',L.H.RAW/'real.npz',
            I.R.DOC/'renderer_front_visibility_audit/RESULTS.json']])
    C.freeze(DOC/'PROTOCOL.json',protocol)
    I.R.evaluation_protocol(PHASE,I.ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    data=I.SourceData().data; arr=data.arrays; source=[]; source_meta=[]
    affine=C.N.E.old('features').canvas_affine
    with ExitStack() as stack:
        archives={}
        for r in protocol['source_records']:
            row=r['row']; meta=data.source['records'][int(data.indices[row])];assert meta['id']==r['id']
            loc=meta['renderer_annotation_locator_provenance_only']
            if '::' in loc:
                archive,member=loc.split('::',1)
                if archive not in archives: archives[archive]=stack.enter_context(zipfile.ZipFile(archive))
                payload=archives[archive].read(member)
            else: payload=(C.ROOT/loc).read_bytes()
            k=F.camera_matrix(json.loads(payload)['camera_data']['intrinsics'])
            gain,offset=affine(meta['prepared_shape_hw'],arr['input_shape'][row])
            q=(np.asarray(arr['points'][row],float)-offset)/gain-meta['reflect_pad_px']
            x,d=F.describe(q,k); source.append(x)
            source_meta.append(dict(id=r['id'],row=row,locator=loc,raw_sha256=hashlib.sha256(payload).hexdigest(),
                K=k.tolist(),points_sha256=I.N.array_sha(arr['points'][row]),diagnostic=d))
    s=np.load(L.H.RAW/'source.npz')
    np.testing.assert_array_equal(s['train'],[r['split']=='train' for r in protocol['source_records']])
    I.save_npz(RAW/'source.npz',x=np.asarray(source,np.float32),**{k:s[k] for k in ['errors','y','eligible','train','diagonal']})
    C.freeze(RAW/'SOURCE_METADATA.json',source_meta)
    pool=[r for r in C.read(I.R.BASE_RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    ids=C.read(I.DOC/'REAL_COMPLETE.json')['ids'];assert [r['id'] for r in pool]==ids
    rx=[]; real_meta=[]
    for r in pool:
        x,d=F.describe(I.top(r['raw'])['keypoints_xy'],r['K']);rx.append(x)
        real_meta.append(dict(id=r['id'],K=r['K'],diagnostic=d))
    real=np.load(L.H.RAW/'real.npz')
    np.testing.assert_array_equal(real['train'],[r['id'] in protocol['real_train'] for r in pool])
    I.save_npz(RAW/'real.npz',x=np.asarray(rx,np.float32),**{k:real[k] for k in ['errors','y','eligible','train']})
    C.freeze(RAW/'REAL_METADATA.json',real_meta)
    # Extract calibration only. Annotation coordinates/pose are not passed to inference.
    cameras=[]
    for r in C.read(DOC/'EVAL_PROTOCOL.json')['records']:
        C.verify(r['annotation'])
        k=F.camera_matrix(C.read(C.ROOT/r['annotation']['path'])['camera_data']['intrinsics'])
        cameras.append(dict(id=r['id'],K=k.tolist(),source=r['annotation'],fields_used='camera_data.intrinsics only'))
    C.freeze(RAW/'EVAL_CAMERAS.json',cameras)
    C.freeze(DOC/'INPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/p) for p in
        ['source.npz','real.npz','SOURCE_METADATA.json','REAL_METADATA.json','EVAL_CAMERAS.json']],
        source594=int(s['train'].sum()),source_heldout1468=int((~s['train']).sum()),real217=int(real['train'].sum()),
        source_unavailable=sum(not r['diagnostic']['available'] for r in source_meta),
        real_unavailable=sum(not r['diagnostic']['available'] for r in real_meta),
        real_GT_coordinates_used=False))
    print('FACING_INPUTS_LOCKED',len(source),len(rx),flush=True)


def fit(arm):
    p=verify();I.N.setup();print('GPU',I.N.E.gpu(),flush=True);assert torch.cuda.is_available()
    for b in C.read(DOC/'INPUTS_LOCK.json')['artifacts']: C.verify(b)
    out=DOC/f'FIT_{arm}.json'
    if out.exists(): C.verify(C.read(out)['checkpoint']);return
    s=np.load(RAW/'source.npz');r=np.load(RAW/'real.npz')
    mask=s['train']&s['eligible'];rm=r['train']&r['eligible']
    assert mask.sum()==594 and rm.sum()==217
    x=s['x'][mask];y=s['y'][mask];rx=r['x'][rm];ry=r['y'][rm]
    fold=np.asarray(p['source_train_folds']);assert len(fold)==len(x)
    cv=[];started=time.monotonic()
    for lam in L.LAMBDA:
        oof=np.empty((len(x),2));stats=[]
        for f in range(3):
            keep=fold!=f;scale=np.maximum(np.sqrt(np.mean(x[keep]**2,axis=0)),.01)
            w,status=L.train_linear(L.normalize(x[keep],scale),y[keep],
                L.normalize(rx,scale) if arm=='MIX' else None,ry if arm=='MIX' else None,lam)
            oof[~keep]=L.probability(L.normalize(x[~keep],scale),w);stats.append(status)
        cv.append(dict(lam=lam,logloss=L.logloss(oof,y),accuracy=float((oof.argmax(-1)==y).mean()),fold_fits=stats))
        print('FACING_CV',arm,lam,cv[-1]['logloss'],cv[-1]['accuracy'],flush=True)
    best=min(cv,key=lambda r:(r['logloss'],-r['lam']))
    scale=np.maximum(np.sqrt(np.mean(x**2,axis=0)),.01)
    w,status=L.train_linear(L.normalize(x,scale),y,L.normalize(rx,scale) if arm=='MIX' else None,ry if arm=='MIX' else None,best['lam'])
    val=~s['train'];prob=L.probability(L.normalize(s['x'][val],scale),w)
    cal=I.calibrate(prob,s['errors'][val],s['y'][val],s['eligible'][val])
    checkpoint=RAW/f'{arm}.npz';I.save_npz(checkpoint,weight=w,scale=scale)
    rp=L.probability(L.normalize(r['x'][~r['train']],scale),w)
    C.freeze(out,dict(complete=True,checkpoint=C.bound(checkpoint),protocol=C.bound(DOC/'PROTOCOL.json'),
        inputs=C.bound(DOC/'INPUTS_LOCK.json'),train_cv=cv,selected_lambda=best['lam'],fit=status,
        source_train_accuracy=float((L.probability(L.normalize(x,scale),w).argmax(-1)==y).mean()),
        calibration=cal,pseudo32_consistency_NOT_accuracy=float((rp.argmax(-1)==r['y'][~r['train']]).mean()),
        seconds=time.monotonic()-started,real_evaluation_seen=False,auto_promoted=False))
    print('FACING_FIT',arm,'lambda',best['lam'],'source_accuracy',cal['pair_accuracy'],'threshold',cal['threshold'],flush=True)


def infer():
    verify()
    for b in C.read(DOC/'INPUTS_LOCK.json')['artifacts']: C.verify(b)
    fits={a:C.read(DOC/f'FIT_{a}.json') for a in I.ARMS}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in I.ARMS},before_real_features=True))
    models={}
    for a,f in fits.items(): C.verify(f['checkpoint']);models[a]=np.load(C.ROOT/f['checkpoint']['path'])
    cameras={r['id']:r['K'] for r in C.read(RAW/'EVAL_CAMERAS.json')}
    original=[r for r in C.read(I.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records'] if r['kind']=='PLASTIC']
    assert len(original)==len(cameras)==194
    outputs={a:[] for a in I.ARMS};decisions=[]
    for r in original:
        q=np.asarray(I.top(r['prediction'])['keypoints_xy']);x,d=F.describe(q,cameras[r['id']]);choices={}
        for a,m in models.items():
            prob=L.probability(L.normalize(x,m['scale']),m['weight']);threshold=fits[a]['calibration']['threshold']
            choice=I.F.choose(prob,threshold);pred=copy.deepcopy(r['prediction'])
            if choice: I.top(pred)['keypoints_xy'][:8]=q[I.F.QUARTER][:8].tolist()
            I.assert_preserved(r['prediction'],pred)
            outputs[a].append(dict(id=r['id'],kind='PLASTIC',prediction=pred,raw_hw=r['raw_hw']))
            choices[a]=dict(probabilities=prob.tolist(),threshold=threshold,choice=choice)
        decisions.append(dict(id=r['id'],features=x.tolist(),diagnostic=d,choices=choices))
    for a,rr in outputs.items():
        C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,checkpoint=fits[a]['checkpoint'],records=rr,GT_free=True))
    C.freeze(RAW/'DECISIONS.json',decisions)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in I.ARMS]+[C.bound(RAW/'DECISIONS.json')],before_GT_scoring=True))
    print('FACING_OUTPUTS_LOCKED', {a:sum(r['choices'][a]['choice'] for r in decisions) for a in I.ARMS},flush=True)


def report():
    # Reuse identical metrics, not the old module's RGB-specific report wording.
    with scope(), redirect_stdout(io.StringIO()), patch.object(C,'write_text',lambda *a:None): I.report()
    r=C.read(DOC/'RESULTS.json')
    print(json.dumps(dict(checks=r['checks'],changed_frames=r['changed_frames'],
        metrics={a:dict(pck=x['summary']['PCK']['20'],median=x['summary']['matched_pooled_corner8_median_px'],
            p90=x['summary']['matched_pooled_corner8_P90_px'],recovery=x['recovery'],tails=x['tails'])
            for a,x in r['results']['full194'].items()}),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','infer','report']);p.add_argument('arm',nargs='?',choices=I.ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.arm)
    elif a.action=='infer':infer()
    else:report()
