"""Isolate representation-space affine augmentation against completed TAIL."""
import argparse
import copy
from contextlib import contextmanager
from unittest.mock import patch
import torch
from . import dino_tail_sampling as T
from . import dino_tail_affine_model as A

C=T.C;D=T.D;U=T.U;N=T.N;M=T.M;P=T.P;W=T.W
PHASE='dino_tail_affine';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;ARMS=['AUG']


def update(model,optimizer,branches,step):
    assert len(branches)==1 and len(branches[0])==8
    optimizer.zero_grad(set_to_none=True);total=0.
    a=A.matrices(8,step,'cuda')
    for j in range(0,8,4):
        b=A.augment(U.tensor_batch(branches[0][j:j+4]),a[j:j+4])
        z=model(b['feature'],b['points'],b['valid'])
        loss=M.loss(z,b['target'],b['target_valid'])/2
        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
    optimizer.step();return total


@contextmanager
def scope():
    with patch.object(T,'PHASE',PHASE),patch.object(T,'DOC',DOC),patch.object(T,'RAW',RAW),\
         patch.object(T,'ARMS',ARMS),patch.object(T.X.L,'update',update):yield


def verify():
    with scope():return T.verify()


def prepare():
    p=copy.deepcopy(T.verify());audit=C.read(T.DOC/'COMPLETION_AUDIT.json')
    for b in audit['evidence']:C.verify(b)
    p.update(arms=ARMS,source_samples_by_arm={'AUG':p['source_samples_by_arm']['TAIL']},
        controls='Same8357pool,196hardstratum,50:50tail schedule,parentMID5000model+Adam,5000additionalupdates,micro4,lr.001/wd.0001,head,loss,decoder. Compare previously locked TAIL;only train-time representation affine augmentation differs.',
        augmentation='Perstepseed[20261011,globalzeroindexedstep],eachimage50percentidentity,else rotationU[-10,10]degrees,scaleU[.85,1.15],translationU[-.1,.1]cropwidth/height around pixelcenter. Same matrix on both frozen features and GT/input points. Bilinear zero padding,align_cornersFalse. Keep old validmask AND transformed crop support. No flips/reindexing.',
        augmentation_limitation='Feature-space warp is NOT equivalent to augmenting RGB and re-extracting nonlinear DINO tokens. Tests coordinate/feature-grid alignment only; no exact backbone equivariance claim. Identity samples bit-exact. No appearance or new scene data generated.',
        sampling='Exact completed TAIL schedule reused; firstmicro4hard196,secondmicro4others8161. No sample/threshold/checkpoint selection from real GT.',
        rationale='TAIL memorized460/460natural source-far training corners,held2/36;officialsymmetry unchanged,unorderedpointoracle8/36. Test reduced coordinate memorization withlabel-preserving representationaugmentation,not merely morehardrepetitions. This is an unproven hypothesis.',
        factor_selection='Single fixed augmentation recipe before fitting; no magnitude sweep. Compare full194,retained159,source64andfar13atfixedlast10000. All previous artifacts/finalmodel unchanged. RepeatedDEV,notindependentconfirmation.',
        sources=p['sources']+[C.bound(f) for f in [__file__,A.__file__,C.HERE/'test_dino_tail_affine_model.py',
            T.DOC/'PROTOCOL.json',T.DOC/'COMPLETION_AUDIT.json',T.DOC/'FIT_TAIL.json',T.DOC/'SOURCE_FAR_ROLE_DIAGNOSTIC.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(T.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    cache=copy.deepcopy(C.read(T.DOC/'CACHE_COMPLETE.json'));cache.update(protocol=C.bound(DOC/'PROTOCOL.json'),reused_from=C.bound(T.DOC/'CACHE_COMPLETE.json'))
    C.freeze(DOC/'CACHE_COMPLETE.json',cache);D.R.evaluation_protocol(PHASE,ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('TAIL_AFFINE_LOCKED',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    if args.action=='prepare':prepare();return
    with scope():
        N.setup();print('GPU',N.E.gpu(),flush=True);bank=T.load_bank();T.train('AUG',bank)
        T.X.read_feature.cache_clear();U.read_mid.cache_clear();T.infer()
        with T.scope():D.report()


if __name__=='__main__':main()
