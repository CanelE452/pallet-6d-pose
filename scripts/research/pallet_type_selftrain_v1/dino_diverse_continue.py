"""Fixed SYN-only continuation on the larger existing source pool,5000->20000."""
import argparse
import copy
import time
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import torch
from . import dino_source_diversity as X

C=X.C;D=X.D;N=X.N;V=X.V;W=X.W;P=X.P;L=X.L
PHASE='dino_diverse_continue';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;STEPS=20000;ARM='SYN'


@contextmanager
def scope():
    with V.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),patch.object(D,'STEPS',STEPS),patch.object(D,'ARMS',[ARM]):yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=X.verify();p=copy.deepcopy(parent)
    train=sorted(r['row'] for r in parent['source_records'] if r['partition']=='train');schedule=X.sample_schedule(train,STEPS)
    assert schedule[:5000]==parent['source_samples']
    p.update(arms=[ARM],steps=STEPS,source_samples=schedule,real_training_images=0,
        real_samples=[parent['real_samples'][i%300] for i in range(STEPS)],
        objective='Improve genuine image-only corner generation on8192existing synthetic images;test remaining optimization headroom,not a more permissive filter.',
        controls='Continue parent SYN5000 model AND AdamW state exactly. Same8192source images+64held,features,targets,masks,head,loss,lr.001,wd.0001,micro4,step-specificcorruption RNG. No real/pseudo branch used. Real records retained as provenance only.',
        optimizer='Resume SYN5000 model and optimizer;continue5001..20000,fixed final20000,no LR sweep,early stopping or real-GT checkpoint choice.',
        sampling='Continue original seed20261004 shuffled complete source epochs without restarting the partial epoch. First5000sample steps bit-exact. Total160000exposures across8192images.',
        training='SYN only isolates exact existing supervision. Source common1412 and held64 curves at7500,10000,12500,15000,17500,20000 are diagnostics only. Only final20000 evaluated on all194real.',
        rationale='On8192source data, SYN common-train PCK10 rose71.61to81.73 andheld74.71to78.24 across1000..5000. This differs from the small1412pool overfit. Test additional optimization of the larger pool before another gate. Earlier realDEV has been seen;not independentconfirmation.',
        factor_selection='One prespecified fourfold total-step continuation. No new architecture/labels/tags/depth/CAD/filter/pool selection. SYN-only control chosen to avoid mixing optimization with pseudo-label noise;not a claim of blindness to prior real results.',
        sources=parent['sources']+[C.bound(f) for f in [__file__,C.HERE/'test_dino_diverse_continue.py',X.DOC/'PROTOCOL.json',X.DOC/'CACHE_COMPLETE.json',
            X.DOC/'FIT_SYN.json',X.DOC/'COMPLETION_AUDIT.json',X.RAW/'SYN/resume_5000.pt',D.R.DOC/'dino_joint_translate_exact/COMPLETION_AUDIT.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(X.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    receipt=copy.deepcopy(C.read(X.DOC/'CACHE_COMPLETE.json'));receipt.update(protocol=C.bound(DOC/'PROTOCOL.json'),reused_from=C.bound(X.DOC/'CACHE_COMPLETE.json'))
    C.freeze(DOC/'CACHE_COMPLETE.json',receipt);D.R.evaluation_protocol(PHASE,[ARM],p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('DIVERSE_CONTINUATION_LOCKED',STEPS,'SYN only',flush=True)


def train():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'FIT_SYN.json').exists():C.verify(C.read(DOC/'FIT_SYN.json')['checkpoint']);return
    init=C.bound(X.RAW/'SYN/resume_5000.pt');C.verify(init);parent=C.read(X.DOC/'FIT_SYN.json');C.verify(parent['checkpoint'])
    ck=torch.load(C.ROOT/init['path'],map_location='cpu',weights_only=False)
    assert ck['step']==5000 and ck['arm']=='SYN' and ck['protocol_sha256']==C.sha(X.DOC/'PROTOCOL.json')
    final=torch.load(C.ROOT/parent['checkpoint']['path'],map_location='cpu',weights_only=False)
    for k,t in ck['model'].items():torch.testing.assert_close(t,final['model'][k],atol=0,rtol=0)
    bank=X.load_bank();sm={r['row']:r for r in bank if r['domain']=='source'}
    held=[sm[i] for i in p['source_held_rows']];probe=[sm[i] for i in p['parent_source_rows']]
    m=V.V.Head().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    resumes=sorted((RAW/ARM).glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['arm']==ARM
    m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=copy.deepcopy(ck['history']);curves=copy.deepcopy(ck['curves'])
    startstep=ck['step'];prior=float(ck.get('additional_seconds',0));start=time.monotonic()
    for step in range(startstep,STEPS):
        src=[sm[i] for i in p['source_samples'][step]];assert all(r['train'] for r in src)
        m.train();loss=L.update(m,opt,[src],step);history.append(dict(step=step+1,loss=loss))
        if (step+1)%2500==0:
            m.eval()
            with scope():curves.append(dict(step=step+1,common_source_train1412=D.probe(m,probe),source_held=D.probe(m,held)))
            path=RAW/ARM/f'resume_{step+1:05d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm=ARM,step=step+1,model=m.state_dict(),optimizer=opt.state_dict(),history=history,curves=curves,
                additional_seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json'),parent=init),path)
            print('DIVERSE_CONTINUATION_CURVE',curves[-1],flush=True)
        if step==startstep or (step+1)%500==0:print('DIVERSE_CONTINUATION_STEP',step+1,'/20000',round(loss,4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    m.eval()
    with scope():after=D.probe(m,held);stress=D.probe(m,held,True)
    path=RAW/ARM/'last20000.pt';assert not path.exists()
    torch.save(dict(arm=ARM,step=STEPS,model=m.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/'FIT_SYN.json',dict(complete=True,step=STEPS,new_updates=15000,checkpoint=C.bound(path),continued_from=init,
        parent_fit=C.bound(X.DOC/'FIT_SYN.json'),protocol=C.bound(DOC/'PROTOCOL.json'),cache=C.bound(DOC/'CACHE_COMPLETE.json'),initial_state=parent['initial_state'],
        source_after=after,source_stress=stress,source_curves=curves,history=history,additional_seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),
        real_training_images=0,backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    X.read_feature.cache_clear();print('DIVERSE_CONTINUATION_FIT_COMPLETE',after,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        train()
        with scope():D.infer();D.report()
