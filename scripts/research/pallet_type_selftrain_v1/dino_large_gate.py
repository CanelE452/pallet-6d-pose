"""Controlled larger synthetic input errors; targets and image heads unchanged."""
import argparse
import copy
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from . import dino_evidence_gate as E
from . import dino_evidence_gate_report as S

C=E.C;D=E.D;W=E.W;V=E.V;P=E.P;N=E.N;G=E.G;R=E.R
PHASE='dino_large_gate';DOC=R.DOC/PHASE;RAW=R.RAW/PHASE


@contextmanager
def scope():
    with patch.object(E,'PHASE',PHASE),patch.object(E,'DOC',DOC),patch.object(E,'RAW',RAW):yield


def verify():
    with scope():return E.verify()


def perturb(item,view):
    if view not in range(5):raise ValueError('View must be0..4')
    if view==0:return item
    rng=np.random.default_rng(np.random.SeedSequence([20261001,int(item['row']),view]))
    if view<=2:return P.perturb(item,rng,True)
    x=dict(item,points=item['points'].copy(),valid=item['valid'].copy())
    eligible=np.flatnonzero(item['target_valid'][:8])
    if not len(eligible):return x
    chosen=rng.choice(eligible,1,replace=False) if view==3 else eligible
    angle=rng.uniform(0,2*np.pi,len(chosen));radius=rng.uniform(.25,.60,len(chosen))*item['bbox_diagonal']
    delta=np.stack([np.cos(angle),np.sin(angle)],-1)*radius[:,None]
    x['points'][chosen]=item['target'][chosen]+delta@item['matrix'][:2,:2].T;x['valid'][chosen]=True
    return x


def prepare():
    parent=E.verify();p=copy.deepcopy(parent)
    p.update(objective='Test whether sparse large beneficial source motions cause rejection of large real corrections; still require accurate-corner preservation.',
        supervision='Same1412source images and untouched targets,masks. Five views:0raw;1/2byte-identical original forced .05-.15diag corruption;3one supported corner and4all supported corners placed at target+uniform angle,.25-.60diag. Same row/view seeds. No GT crop,clamping,extra trust,new manual annotations or tags.',
        intervention='Only source input corruption views3/4 change,including source calibration stress distribution. Same candidate heads,16features,MLP,500steps,batch order RNG,train/calibration/test32 split,threshold grid/selection rule. Target-derived benefit classes and feature normalization naturally change. Not a pure threshold ablation.',
        rationale='Parent rejected all16/14 >80to<=10 candidates; large movement>30%diag comprised only235/13230 and207/13034 training positives. Fixed broad25..60% test range,no real-GT tuned radius sweep. Reused DEV hypothesis,not independent confirmation.',
        evaluation=parent['evaluation']+' Use existing composite-checkpoint scorer adapter from the start; prediction coordinates remain locked.',
        sources=parent['sources']+[C.bound(x) for x in [__file__,C.HERE/'test_dino_large_gate.py',S.__file__,
            E.DOC/'PROTOCOL.json',E.DOC/'SOURCE_CACHE.json',E.DOC/'RESULTS.json',E.DOC/'COMPLETION_AUDIT.json',E.DOC/'REJECTED_LARGE_DIAGNOSTIC.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(E.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    R.evaluation_protocol(PHASE,E.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('LARGE_GATE_PROTOCOL_LOCKED',flush=True)


@torch.no_grad()
def cache_source():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    with W.scope():bank=D.load_bank()
    rows=[r for r in bank if r['domain']=='source'];models=E.heads();data={a:[] for a in E.ARMS}
    for offset in range(0,len(rows),4):
        batch=rows[offset:offset+4];b=D.tensor_batch(batch)
        logits={a:m(b['feature'],b['points'],b['valid']) for a,m in models.items()}
        for view in range(5):
            noisy=[perturb(r,view) for r in batch]
            q=torch.as_tensor(np.stack([r['points'] for r in noisy]),device='cuda')
            valid=torch.as_tensor(np.stack([r['valid'] for r in noisy]),device='cuda')
            diag=torch.as_tensor([r['bbox_diagonal']*r['matrix'][0,0] for r in batch],device='cuda',dtype=torch.float32)
            for a in E.ARMS:
                other=next(k for k in E.ARMS if k!=a);features,new=G.features(logits[a],logits[other],q,valid,diag)
                f=features.cpu().numpy();new=new.cpu().numpy()
                for i,(r,n) in enumerate(zip(batch,noisy)):
                    mask=r['target_valid']&n['valid'];mask[8]=False;gain=r['matrix'][0,0]
                    before=np.linalg.norm(n['points']-r['target'],axis=-1)/gain
                    after=np.linalg.norm(new[i]-r['target'],axis=-1)/gain
                    for j in np.flatnonzero(mask):data[a].append((int(r['row']),view,int(j),f[i,j],before[j],after[j]))
        if offset%400==0:print('LARGE_GATE_SOURCE',offset,'/',len(rows),N.E.gpu(),flush=True)
    receipt={}
    for a,rr in data.items():
        path=RAW/f'SOURCE_{a}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:np.savez(f,row=np.array([r[0] for r in rr]),view=np.array([r[1] for r in rr]),corner=np.array([r[2] for r in rr]),
            features=np.stack([r[3] for r in rr]),before=np.array([r[4] for r in rr]),after=np.array([r[5] for r in rr]))
        receipt[a]=C.bound(path)
    C.freeze(DOC/'SOURCE_CACHE.json',dict(artifacts=receipt,protocol=C.bound(DOC/'PROTOCOL.json'),source_images=len(rows),views=5))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        cache_source()
        with scope():E.train();E.infer();S.main()
