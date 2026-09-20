"""Posthoc clean source train/held generalization, not checkpoint selection."""
import torch
from . import dino_wide as W
from . import dino_wide_visual as V

C=W.C;D=W.D


@torch.no_grad()
def main():
    W.verify();V.verify();W.N.setup();print('GPU',W.N.E.gpu(),flush=True)
    with W.scope():bank=D.load_bank()
    groups={g:[r for r in bank if r['domain']=='source' and r['train']==train] for g,train in [('train1412',True),('held64',False)]}
    results={};bindings=[]
    for X,M in [(W,W.W),(V,V.V)]:
        results[X.PHASE]={}
        for arm in D.ARMS:
            fit=C.read(X.DOC/f'FIT_{arm}.json');C.verify(fit['checkpoint']);bindings.append(C.bound(X.DOC/f'FIT_{arm}.json'))
            ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
            model=M.Head().cuda();model.load_state_dict(ck['model']);model.eval()
            with X.scope():table={g:dict(images=len(rows),**D.probe(model,rows)) for g,rows in groups.items()}
            results[X.PHASE][arm]=table;print('SOURCE_FIT',X.PHASE,arm,table,flush=True)
    C.freeze(V.DOC/'SOURCE_FIT_DIAGNOSTIC.json',dict(status='POSTHOC_SOURCE_GT_ONLY_NOT_MODEL_SELECTION',
        final_checkpoints_only=True,results=results,evidence=[C.bound(__file__),C.bound(W.DOC/'CACHE_COMPLETE.json')]+bindings))


if __name__=='__main__':main()
