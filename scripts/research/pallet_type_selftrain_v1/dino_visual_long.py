"""Continue identical image-only heads and Adam state from1000 to5000 steps."""
import argparse
import copy
import time
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from . import dino_wide_visual as V

C=V.C;D=V.D;W=V.W;P=V.P;N=V.N
PHASE='dino_visual_long';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;STEPS=5000


@contextmanager
def scope():
    with V.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),patch.object(D,'STEPS',STEPS):yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=V.verify();p=copy.deepcopy(parent)
    p.update(steps=STEPS,source_samples=[parent['source_samples'][i%300] for i in range(STEPS)],
        real_samples=[parent['real_samples'][i%300] for i in range(STEPS)],
        objective='Improve generation of genuinely new corner locations; test whether1000steps left the unchanged image-only head undertrained.',
        controls='Resume exact parent1000 model AND Adam moments/step count; same1412source+64held,217pseudo+32probe,feature cache,targets,head,loss,lr,microbatch,seed,original300-step repeated sample schedule. No prior,gate,shape filter or new annotation/tag.',
        optimizer='Parent AdamW lr.001 weight_decay.0001,state restored exactly. Continue steps1001..5000 with original step-specific RNG;fixedlast5000,no LR sweep/early stopping/eval checkpoint selection.',
        training='At2000/3000/4000/5000 record clean source train1412 and held64 diagnostic curves. These are not model selection. Only final5000 evaluated on real194. Same source/real branch weights and exposures as parent.',
        factor_selection='One prespecified continuation to5000steps;only final5000 is scored on real194. No checkpoint selection from source curves or real evaluation.',
        rationale='Parent image-only SYN trainPCK10~81.44%,held~71.76%;joint selection cannot create missing coordinates. CurrentR0+SYN+MIX point pool contains no point within10px for56/64far-spatialGTcorners. This motivates generator training,not another real-GT threshold search.',
        sources=parent['sources']+[C.bound(x) for x in [__file__,C.HERE/'test_dino_visual_long.py',V.DOC/'PROTOCOL.json',V.DOC/'CACHE_COMPLETE.json',
            V.DOC/'FIT_SYN.json',V.DOC/'FIT_MIX.json',V.DOC/'SOURCE_FIT_DIAGNOSTIC.json',D.R.DOC/'dino_joint/SPATIAL_SUPPORT_DIAGNOSTIC.json']]+[C.bound(V.RAW/a/'resume_1000.pt') for a in D.ARMS])
    for key in ['source_samples','real_samples']:assert p[key][:1000]==parent[key]
    C.freeze(DOC/'BACKBONE.json',C.read(V.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    receipt=copy.deepcopy(C.read(V.DOC/'CACHE_COMPLETE.json'));receipt.update(protocol=C.bound(DOC/'PROTOCOL.json'),reused_from=C.bound(V.DOC/'CACHE_COMPLETE.json'))
    C.freeze(DOC/'CACHE_COMPLETE.json',receipt);D.R.evaluation_protocol(PHASE,D.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('LONG_PROTOCOL_LOCKED',flush=True)


def update(model,optimizer,branches,step):
    optimizer.zero_grad(set_to_none=True);values=[]
    for branch,rows in enumerate(branches):
        rng=np.random.default_rng(np.random.SeedSequence([20260929,step,branch]));noisy=[P.perturb(r,rng) for r in rows]
        for j in range(0,8,4):
            b=D.tensor_batch(rows[j:j+4],noisy[j:j+4]);z=model(b['feature'],b['points'],b['valid'])
            loss=V.V.loss(z,b['target'],b['target_valid'])/2/len(branches)
            assert torch.isfinite(loss);loss.backward();values.append(float(loss.detach()))
    optimizer.step();return sum(values)


def train(arm):
    p=verify();assert arm in D.ARMS;N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/f'FIT_{arm}.json').exists():C.verify(C.read(DOC/f'FIT_{arm}.json')['checkpoint']);return
    parentfit=C.read(V.DOC/f'FIT_{arm}.json');C.verify(parentfit['checkpoint'])
    init=C.bound(V.RAW/arm/'resume_1000.pt');C.verify(init)
    ck=torch.load(C.ROOT/init['path'],map_location='cpu',weights_only=False)
    assert ck['step']==1000 and ck['protocol_sha256']==C.sha(V.DOC/'PROTOCOL.json') and ck['arm']==arm
    final=torch.load(C.ROOT/parentfit['checkpoint']['path'],map_location='cpu',weights_only=False)
    for k,v in ck['model'].items():torch.testing.assert_close(v,final['model'][k],atol=0,rtol=0)
    with scope():bank=D.load_bank()
    sm={r['row']:r for r in bank if r['domain']=='source'};rm={r['id']:r for r in bank if r['domain']=='real'}
    held=[sm[i] for i in p['source_held_rows']];training=[r for r in sm.values() if r['train']];pseudo=[r for r in rm.values() if not r['train']]
    model=V.V.Head().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    resumes=sorted((RAW/arm).glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
    model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);startstep=ck['step'];history=copy.deepcopy(ck['history'])
    curves=copy.deepcopy(ck.get('curves',[]));prior=float(ck.get('additional_seconds',0));start=time.monotonic()
    for step in range(startstep,STEPS):
        src=[sm[i] for i in p['source_samples'][step]];real=[rm[i] for i in p['real_samples'][step]]
        assert all(r['train'] for r in src+real);branches=[src] if arm=='SYN' else [src,real]
        model.train();value=update(model,optimizer,branches,step);history.append(dict(step=step+1,loss=value))
        if (step+1)%1000==0:
            model.eval()
            with scope():curve=dict(step=step+1,source_train=D.probe(model,training),source_held=D.probe(model,held))
            curves.append(curve);print('LONG_SOURCE_CURVE',arm,curve,flush=True)
        if (step+1)%1000==0:
            path=RAW/arm/f'resume_{step+1:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm=arm,step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,curves=curves,
                additional_seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json'),parent=init),path)
        if step==startstep or (step+1)%500==0:print('LONG_TRAIN',arm,step+1,'/5000',value,N.E.gpu(),flush=True)
    model.eval()
    with scope():after=D.probe(model,held);stress=D.probe(model,held,True);pp=D.probe(model,pseudo);ps=D.probe(model,pseudo,True)
    path=RAW/arm/'last5000.pt';assert not path.exists()
    torch.save(dict(arm=arm,step=STEPS,model=model.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/f'FIT_{arm}.json',dict(complete=True,step=STEPS,new_updates=4000,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),
        cache=C.bound(DOC/'CACHE_COMPLETE.json'),initial_state=parentfit['initial_state'],continued_from=init,parent_fit=C.bound(V.DOC/f'FIT_{arm}.json'),
        source_after=after,source_stress=stress,source_curves=curves,pseudo32_consistency_NOT_accuracy=pp,pseudo32_stress_NOT_accuracy=ps,
        history=history,additional_seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),
        backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    print('LONG_FIT_COMPLETE',arm,after,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:
        for arm in D.ARMS:train(arm)
        with scope():D.infer();D.report()
