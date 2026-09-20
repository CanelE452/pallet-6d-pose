"""Prespecified no-keypoint-hint control using the already verified wide cache."""
import argparse
import copy
from contextlib import contextmanager
from unittest.mock import patch
from . import dino_wide as W
from . import dino_wide_visual_model as V

D=W.D;C=W.C;P=W.P;N=W.N
PHASE='dino_wide_visual';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE


@contextmanager
def scope():
    with W.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),patch.object(D,'M',V):
        yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=W.verify();p=copy.deepcopy(parent)
    p.update(objective='Test whether keypoint conditioning/prior prevents recovery despite reachable image evidence. Learn positions without keypoint hints, using existing supervision only.',
        head='Same wide head parameters and seeded initialization; nine input hint channels fixed zero; omit output Gaussian prior. Image patch features and fixed XY grid only. Same192x144 heatmap and5x5 decoder. Zero-head control is uniform,NOT identity.',
        controls='Same wide cached features,targets,masks,1412source train+64held,217pseudo train+32probe,1000steps,optimizer,seed,sample order and scoring. Corruption generator called unchanged but its point hints are unused by the model. R0 predicted box still defines crop; invalid original points and center remain preserved at inference.',
        rationale='Wide model recovered3/4 of64 genuine far-spatial corners but damaged30/61 good corners. Parent point anchoring may still obscure image evidence; this is a diagnostic control,not a promised deployable improvement.',
        factor_selection='One fixed control removes both keypoint conditioning pathways. No radius,threshold,checkpoint sweep or GT-based choice. Evaluate both fixed final SYN/MIX heads even if source gates fail;never automatically promote.',
        targets=parent['targets'],new_annotations=0,new_tags=0,auto_promote=False,
        sources=parent['sources']+[C.bound(x) for x in [__file__,V.__file__,C.HERE/'test_dino_wide_visual_model.py',
            W.DOC/'PROTOCOL.json',W.DOC/'CACHE_COMPLETE.json',W.DOC/'RESULTS.json',W.DOC/'COMPLETION_AUDIT.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(W.DOC/'BACKBONE.json'))
    C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,D.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    receipt=copy.deepcopy(C.read(W.DOC/'CACHE_COMPLETE.json'))
    receipt.update(protocol=C.bound(DOC/'PROTOCOL.json'),reused_from=C.bound(W.DOC/'CACHE_COMPLETE.json'),
        note='NPZ protocol_sha256 intentionally binds original wide cache generation; all item arrays/features reused byte-exact.')
    C.freeze(DOC/'CACHE_COMPLETE.json',receipt)
    print('VISUAL_PROTOCOL_LOCKED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        with scope():
            for arm in D.ARMS:D.train(arm)
            D.infer();D.report()
