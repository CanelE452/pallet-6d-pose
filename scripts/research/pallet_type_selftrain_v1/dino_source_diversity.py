"""Existing-label diversity control; no new annotation or real-pool changes.

Expand source1412 to8192 and train the same image-only head for fixed5000steps.
Lazy feature loading bounds application memory on the 32GB workstation.
"""
import argparse
import copy
import time
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import torch
from . import dino_visual_long as L

C=L.C;D=L.D;V=L.V;W=L.W;P=L.P;N=L.N
PHASE='dino_source_diversity';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE
STEPS=5000;SOURCE_COUNT=8192


def choose_rows(eligible,previous,count=SOURCE_COUNT):
    eligible=np.unique(eligible);previous=np.unique(previous)
    assert set(previous)<=set(eligible) and len(previous)<=count<=len(eligible)
    rng=np.random.default_rng(20261003)
    extra=rng.choice(np.setdiff1d(eligible,previous),count-len(previous),replace=False)
    return sorted(np.r_[previous,extra].astype(int).tolist())


def sample_schedule(rows,steps=STEPS):
    rng=np.random.default_rng(20261004);order=[]
    while len(order)<steps*8:order.extend(rng.permutation(rows).tolist())
    return np.asarray(order[:steps*8]).reshape(steps,8).tolist()


@contextmanager
def scope():
    with V.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),patch.object(D,'STEPS',STEPS):yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=V.verify();source=P.SourceData()
    old=sorted(set(np.asarray(parent['source_samples']).ravel().tolist()))
    train=choose_rows(source.train_rows,old);held=parent['source_held_rows']
    assert not set(train)&set(held)
    records=[]
    for partition,rows in [('train',train),('heldout',held)]:
        for i in rows:
            r=source.data.source['records'][int(source.data.indices[i])]
            assert r['partition']==source.partitions[i]==partition
            image=C.bound(r['image']);assert image['sha256']==r['image_sha256']
            records.append(dict(row=i,id=r['id'],partition=partition,image=image,scenario=r['scenario_id']))
    assert not {r['scenario'] for r in records if r['partition']=='train'}&{r['scenario'] for r in records if r['partition']=='heldout'}
    ev=C.read(D.R.BASE_DOC/'EVAL_PROTOCOL.json')['records'];forbid={r['image']['sha256'] for r in ev}
    assert not forbid&{r['image']['sha256'] for r in records+parent['real_records']}
    schedule=sample_schedule(train)
    assert set(np.asarray(schedule).ravel())==set(train)
    p=copy.deepcopy(parent)
    p.update(steps=STEPS,source_training_images=len(train),source_records=records,source_samples=schedule,
        real_samples=[parent['real_samples'][i%300] for i in range(STEPS)],
        source_cache_signature=N.source_cache_signature(source.data,np.r_[train,held]),
        objective='Recover missing corner locations by improving source generalization without adding manual labels.',
        controls='Same image-only architecture,frozen DINO,crop,loss,AdamW lr.001 wd.0001,micro4,seed20260929,fixed5000updates as long control. Same held64 and real217+32. Source pool/order are the changed factor;1412 original images retained plus6780 existing train-only images selected without errors/GT difficulty ranking.',
        sampling='8192 unique existing train rows,including all1412 parent rows;additional uniform seed20261003. Independently shuffled complete epochs seed20261004;40000 source exposures,no held/eval source rows.',
        optimizer='From scratch with identical parent seeded initialization;AdamW .001/.0001,5000updates,fixedlast5000,no LR/checkpoint sweep. Source8;MIX source8+real8 equal branch weights.',
        training='Source train/probe curves at1000/2000/3000/4000/5000 are diagnostics only. Train probe is fixed original1412 for a common denominator;held64 unchanged. Evaluate only final5000 on real194.',
        rationale='Fixed1412 source trainPCK10 rose81.44to96.46 but held71.76to68.04 when continuing1000to5000. There are55915 eligible existing source train rows. Test broader existing supervision,not longer repetition or new labels.',
        factor_selection='Exactly8192 total source images and5000steps chosen before fitting. No real-GT image selection. This changes sample distribution/order and unique exposures together;not a pure single-image-count causal estimate.',
        parent_source_rows=old,
        sources=parent['sources']+[C.bound(x) for x in [__file__,C.HERE/'test_dino_source_diversity.py',L.__file__,
            V.DOC/'PROTOCOL.json',V.DOC/'CACHE_COMPLETE.json',L.DOC/'RESULTS.json',L.DOC/'COMPLETION_AUDIT.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(V.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,D.ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('DIVERSITY_PROTOCOL',len(train),'source train',len(held),'held',flush=True)


def cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'CACHE_COMPLETE.json').exists():
        for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:C.verify(r['cache'])
        return
    old={r['key']:r for r in C.read(V.DOC/'CACHE_COMPLETE.json')['records']}
    source=P.SourceData();assert N.source_cache_signature(source.data,np.r_[sorted(r['row'] for r in p['source_records'] if r['partition']=='train'),p['source_held_rows']])==p['source_cache_signature']
    entries=[dict(id=r['id'],key='S'+str(r['row']),row=r['row'],domain='source',train=r['partition']=='train') for r in p['source_records']]
    entries += [dict(id=r['id'],key='R'+str(i),domain='real',train=r['train']) for i,r in enumerate(p['real_records'])]
    backbone,_=D.A.load();records=[];start=time.monotonic();new=0
    for i,r in enumerate(entries):
        if r['key'] in old:
            prior=old[r['key']];assert all(prior[k]==v for k,v in r.items());C.verify(prior['cache']);records.append(prior)
        else:
            assert r['domain']=='source' and r['train'];x=W.source_item(source,r['row']);path=RAW/'cache'/(r['key']+'.npz')
            if not path.exists():
                feature=W.extract(backbone,[x])[0];path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as f:np.savez(f,feature=feature,old_target_valid=x['old_target_valid'],protocol_sha256=np.array(C.sha(DOC/'PROTOCOL.json')),**D.item_arrays(x))
            with np.load(path) as z:
                assert str(z['protocol_sha256'])==C.sha(DOC/'PROTOCOL.json') and z['feature'].shape==(384,56,42)
                assert np.isfinite(z['feature']).all()
                for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
            records.append(dict(**r,cache=C.bound(path)));new+=1
        if (i+1)%250==0:print('DIVERSITY_CACHE',i+1,'/',len(entries),'new',new,round(time.monotonic()-start,1),N.E.gpu(),flush=True)
    assert len(records)==8505 and new==6780
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=records,new_items=new,reused_items=1725,seconds=time.monotonic()-start,
        protocol=C.bound(DOC/'PROTOCOL.json'),backbone=C.bound(DOC/'BACKBONE.json'),real_eval_features_used=False))
    print('DIVERSITY_CACHE_COMPLETE',len(records),flush=True)


@lru_cache(maxsize=128)
def read_feature(path):
    with np.load(path) as z:return np.array(z['feature'])


class LazyFeature:
    def __init__(self,path):self.path=str(path)
    def __array__(self,dtype=None):return np.asarray(read_feature(self.path),dtype=dtype)


def load_bank():
    items=[]
    for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:
        C.verify(r['cache']);path=C.ROOT/r['cache']['path']
        with np.load(path) as z:
            arrays={k:np.array(z[k]) for k in z.files if k not in ['protocol_sha256','feature']}
        items.append(dict(r,**arrays,feature=LazyFeature(path)))
    return items


def train(arm):
    p=verify();assert arm in D.ARMS;N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/f'FIT_{arm}.json').exists():C.verify(C.read(DOC/f'FIT_{arm}.json')['checkpoint']);return
    bank=load_bank();sm={r['row']:r for r in bank if r['domain']=='source'};rm={r['id']:r for r in bank if r['domain']=='real'}
    held=[sm[i] for i in p['source_held_rows']];probe=[sm[i] for i in p['parent_source_rows']];pseudo=[r for r in rm.values() if not r['train']]
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929);m=V.V.Head().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
    assert initial==C.read(V.DOC/f'FIT_{arm}.json')['initial_state']
    history=[];curves=[];startstep=0;prior=0.
    resumes=sorted((RAW/arm).glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['arm']==arm
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];curves=ck['curves'];startstep=ck['step'];prior=ck['seconds']
    start=time.monotonic()
    for step in range(startstep,STEPS):
        src=[sm[i] for i in p['source_samples'][step]];real=[rm[i] for i in p['real_samples'][step]]
        assert all(r['train'] for r in src+real);m.train();loss=L.update(m,opt,[src] if arm=='SYN' else [src,real],step);history.append(dict(step=step+1,loss=loss))
        if (step+1)%1000==0:
            m.eval()
            with scope():curves.append(dict(step=step+1,common_source_train1412=D.probe(m,probe),source_held=D.probe(m,held)))
            path=RAW/arm/f'resume_{step+1:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm=arm,step=step+1,model=m.state_dict(),optimizer=opt.state_dict(),history=history,curves=curves,seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
            print('DIVERSITY_CURVE',arm,curves[-1],flush=True)
        if step==startstep or (step+1)%250==0:print('DIVERSITY_TRAIN',arm,step+1,'/5000',round(loss,4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    m.eval()
    with scope():after=D.probe(m,held);stress=D.probe(m,held,True);pp=D.probe(m,pseudo)
    path=RAW/arm/'last5000.pt';assert not path.exists();torch.save(dict(arm=arm,step=STEPS,model=m.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/f'FIT_{arm}.json',dict(complete=True,step=STEPS,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),cache=C.bound(DOC/'CACHE_COMPLETE.json'),initial_state=initial,
        source_after=after,source_stress=stress,source_curves=curves,pseudo32_consistency_NOT_accuracy=pp,history=history,seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    read_feature.cache_clear();print('DIVERSITY_FIT_COMPLETE',arm,after,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:
        cache()
        for arm in D.ARMS:train(arm)
        with scope():D.infer();D.report()
