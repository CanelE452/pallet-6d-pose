"""Controlled natural source-tail oversampling, same model/optimizer/steps."""
import argparse
import copy
import time
from contextlib import contextmanager
from unittest.mock import patch
import cv2
import numpy as np
import torch
from . import dino_mid_feature as U
from . import dino_source_difficulty_full as Q

C=U.C;D=U.D;N=U.N;P=U.P;W=U.W;X=U.X;M=U.M
PHASE='dino_tail_sampling';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE
ARMS=['UNIFORM','TAIL'];START=5000;STEPS=10000;UPDATES=5000


def stream(rows,count,seed):
    rows=np.asarray(sorted(set(rows)),dtype=int)
    if not len(rows):raise ValueError('Nonempty pool required')
    rng=np.random.default_rng(seed);out=[]
    while len(out)<count:out.extend(rng.permutation(rows).tolist())
    return np.asarray(out[:count])


def schedule(pool,hard,steps=UPDATES):
    pool=set(pool);hard=set(hard);easy=pool-hard
    if not hard or not easy or not hard<=pool:raise ValueError('Both disjoint strata required')
    return dict(UNIFORM=stream(pool,steps*8,20261008).reshape(steps,8).tolist(),
        TAIL=np.concatenate([stream(hard,steps*4,20261009).reshape(steps,4),
            stream(easy,steps*4,20261010).reshape(steps,4)],1).tolist())


@contextmanager
def scope():
    with U.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),\
         patch.object(D,'STEPS',STEPS),patch.object(D,'ARMS',ARMS):yield


def verify():
    with scope():return D.verify()


def prepare():
    parent=U.verify();p=copy.deepcopy(parent);source=P.SourceData()
    inventory=C.read(U.DOC/'FULL_SOURCE_DIFFICULTY_AUDIT.json')
    for b in inventory['evidence']:C.verify(b)
    hard=sorted(r['row'] for r in inventory['records'] if r['partition']=='train' and r['far40']>0)
    farheld=sorted(r['row'] for r in inventory['records'] if r['partition']=='held' and r['far40']>0)
    oldtrain={r['row'] for r in parent['source_records'] if r['partition']=='train'}
    train=sorted(oldtrain|set(hard));held=sorted(set(parent['source_held_rows'])|set(farheld))
    assert len(hard)==196 and len(farheld)==13 and len(train)==8357 and len(held)==77
    assert set(train)<=set(source.train_rows) and set(held)<=set(source.held_rows) and not set(train)&set(held)
    records=[]
    for partition,rows in [('train',train),('heldout',held)]:
        for row in rows:
            r=source.data.source['records'][int(source.data.indices[row])];assert r['partition']==partition
            image=C.bound(r['image']);assert image['sha256']==r['image_sha256']
            records.append(dict(row=row,id=r['id'],partition=partition,image=image,scenario=r['scenario_id']))
    assert not {r['scenario'] for r in records if r['partition']=='train'}&{r['scenario'] for r in records if r['partition']=='heldout'}
    ev={r['image']['sha256'] for r in C.read(D.R.BASE_DOC/'EVAL_PROTOCOL.json')['records']}
    assert not ev&{r['image']['sha256'] for r in records}
    samples=schedule(train,hard)
    for arm in ARMS:assert set(np.asarray(samples[arm]).ravel())==set(train)
    p.pop('source_samples')
    p.update(arms=ARMS,steps=STEPS,additional_steps=UPDATES,start_step=START,source_training_images=len(train),real_training_images=0,
        source_records=records,source_held_rows=held,preservation_held_rows=parent['source_held_rows'],far_held_rows=farheld,
        hard_train_rows=hard,source_samples_by_arm=samples,
        source_cache_signature=N.source_cache_signature(source.data,np.r_[train,held]),
        initialization=C.bound(U.RAW/'SYN/resume_5000.pt'),
        objective='Recover real large spatial corner errors using existing natural source failures, not new annotations/tags.',
        controls='Identical mid+last frozen DINO head,exact parent model AND Adam5000 state,source8357pool,GT/masks,loss,lr.001/wd.0001,micro4,5000additional updates. Only source sampling differs between arms. No real/pseudo branch or new filters.',
        sampling='UNIFORM independently shuffled full8357 epochs seed20261008,8/update. TAIL firstmicro4 from196far-image stratum seed20261009,secondmicro4 from8161others seed20261010;complete shuffled stratumepochs. Both microbatch losses halfweight,unchanged update function. No GT relabeling.',
        far_definition='Existing R0 has no valid corner within40 prepared-image pixels of at least one supervised existing GT corner insidewidecrop; fixed before fitting from source-only inventory. No real-GT selection.',
        validation='Original64 source preservation probe retained unchanged;13 disjoint preexisting heldout natural-far images added as separate recovery probe. These13have36far corners and never enter training. Report separately,not replace ordinary validation. Real194/frozen159 unchanged.',
        optimizer='Both arms resume same exact parentmid5000model+Adam;globalsteps5001..10000,historynew5000,fixedlast10000,no checkpoint or threshold choice. Sourcecurves at6000..10000diagnostics only.',
        rationale='Old8192pool had31far images/77far corners;oldheld64had0far. Full existing source contains196train/13held far images. Add165train+13held to coverage,then isolate sampling withsamepool betweenarms. Not an isolated pool-size effect versus parent5k.',
        factor_selection='One fixed50:50 stratum choice,no sampling-ratio sweep,architecturechange,pseudo-targetchange orrealGT tuning. RepeatedDEV influenced motivation;notindependentconfirmation.',
        sources=parent['sources']+[C.bound(f) for f in [__file__,C.HERE/'test_dino_tail_sampling.py',Q.__file__,U.DOC/'COMPLETION_AUDIT.json',U.DOC/'FULL_SOURCE_DIFFICULTY_AUDIT.json',U.DOC/'CACHE_COMPLETE.json',U.DOC/'FIT_SYN.json',U.RAW/'SYN/resume_5000.pt']])
    C.freeze(DOC/'BACKBONE.json',C.read(U.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')]);print('TAIL_PROTOCOL_LOCKED',len(train),len(held),flush=True)


def cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'CACHE_COMPLETE.json').exists():
        for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:C.verify(r['cache']);C.verify(r['mid_cache'])
        return
    prior={r['row']:r for r in C.read(U.DOC/'CACHE_COMPLETE.json')['records']};source=P.SourceData();backbone,_=D.A.load()
    assert N.source_cache_signature(source.data,np.r_[sorted(r['row'] for r in p['source_records'] if r['partition']=='train'),p['source_held_rows']])==p['source_cache_signature']
    rows=[];new=0;start=time.monotonic()
    for rec in p['source_records']:
        row=rec['row'];r=dict(id=rec['id'],key='S'+str(row),row=row,domain='source',train=rec['partition']=='train')
        if row in prior:
            old=prior[row];assert all(old[k]==v for k,v in r.items());C.verify(old['cache']);C.verify(old['mid_cache']);rows.append(old);continue
        x=W.source_item(source,row);mid,late=U.extract(backbone,[x],True)
        path=RAW/'cache'/(r['key']+'.npz');mpath=RAW/'mid_cache'/(r['key']+'.npy');path.parent.mkdir(parents=True,exist_ok=True);mpath.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with path.open('xb') as f:np.savez(f,feature=late[0],old_target_valid=x['old_target_valid'],protocol_sha256=np.array(C.sha(DOC/'PROTOCOL.json')),**D.item_arrays(x))
        if not mpath.exists():
            with mpath.open('xb') as f:np.save(f,mid[0],allow_pickle=False)
        with np.load(path) as z:
            assert str(z['protocol_sha256'])==C.sha(DOC/'PROTOCOL.json');np.testing.assert_array_equal(z['feature'],late[0])
            for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
        np.testing.assert_array_equal(np.load(mpath),mid[0]);rows.append(dict(r,cache=C.bound(path),mid_cache=C.bound(mpath)));new+=1
        if new%32==0:print('TAIL_NEW_CACHE',new,'/178',N.E.gpu(),flush=True)
    assert len(rows)==8434 and new==178
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=rows,new_items=178,reused_items=8256,seconds=time.monotonic()-start,
        protocol=C.bound(DOC/'PROTOCOL.json'),parent=C.bound(U.DOC/'CACHE_COMPLETE.json')))


def load_bank():
    with patch.object(U,'DOC',DOC):return U.load_bank()


def spatial_summary(items,predicted):
    errors=[];before=[];far=[];records=[]
    for r,q in zip(items,predicted):
        gt=r['target'][:8];mask=r['target_valid'][:8];gain=float(r['matrix'][0,0]);old=r['points'][:8];valid=r['valid'][:8]&np.isfinite(old).all(-1)
        nearest=np.linalg.norm(gt[:,None]-old[valid][None],axis=-1).min(-1)/gain
        a=np.linalg.norm(old-gt,axis=-1)/gain;b=np.linalg.norm(q[:8]-gt,axis=-1)/gain
        before.extend(a[mask]);errors.extend(b[mask]);far.extend(nearest[mask]>40)
        records.append(dict(row=int(r['row']),far=int(((nearest>40)&mask).sum()),recovered=int(((nearest>40)&mask&(b<=10)).sum())))
    a=np.asarray(before);b=np.asarray(errors);f=np.asarray(far,bool)
    return dict(n=len(b),PCK10=float((b<=10).mean()),PCK20=float((b<=20).mean()),median_px=float(np.median(b)),P90_px=float(np.quantile(b,.9)),
        far=int(f.sum()),far_recovered=int((f&(b<=10)).sum()),good=int((a<5).sum()),damaged=int(((a<5)&(b>10)).sum()),records=records)


@torch.no_grad()
def spatial_probe(model,items):
    out=[]
    for i in range(0,len(items),4):
        b=U.tensor_batch(items[i:i+4]);out.extend(M.decode(model(b['feature'],b['points'],b['valid'])).cpu().numpy())
    return spatial_summary(items,out)


def train(arm,bank):
    p=verify();assert arm in ARMS;N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/f'FIT_{arm}.json').exists():C.verify(C.read(DOC/f'FIT_{arm}.json')['checkpoint']);return
    sm={r['row']:r for r in bank};held=[sm[i] for i in p['preservation_held_rows']];fh=[sm[i] for i in p['far_held_rows']];ht=[sm[i] for i in p['hard_train_rows']]
    C.verify(p['initialization']);ck=torch.load(C.ROOT/p['initialization']['path'],map_location='cpu',weights_only=False)
    assert ck['step']==START and ck['protocol_sha256']==C.sha(U.DOC/'PROTOCOL.json')
    m=M.Head().cuda();m.load_state_dict(ck['model']);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);opt.load_state_dict(ck['optimizer'])
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()};history=[];curves=[];step0=START;prior=0.
    resumes=sorted((RAW/arm).glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False);assert ck['arm']==arm and ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];curves=ck['curves'];step0=ck['step'];prior=ck['seconds']
    start=time.monotonic()
    for step in range(step0,STEPS):
        rows=[sm[i] for i in p['source_samples_by_arm'][arm][step-START]];assert all(r['train'] for r in rows);m.train()
        with scope():value=X.L.update(m,opt,[rows],step)
        history.append(dict(step=step+1,loss=value))
        if (step+1)%1000==0:
            m.eval()
            with scope():normal=D.probe(m,held)
            curves.append(dict(step=step+1,preservation64=normal,far_held13=spatial_probe(m,fh),hard_train196=spatial_probe(m,ht)))
            path=RAW/arm/f'resume_{step+1:05d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm=arm,step=step+1,model=m.state_dict(),optimizer=opt.state_dict(),history=history,curves=curves,
                seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
            shown={k:({kk:vv for kk,vv in v.items() if kk!='records'} if isinstance(v,dict) else v) for k,v in curves[-1].items()}
            print('TAIL_CURVE',arm,shown,flush=True)
        if step==step0 or (step+1)%500==0:print('TAIL_TRAIN',arm,step+1,'/10000',round(value,4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    m.eval()
    with scope():after=D.probe(m,held)
    far=spatial_probe(m,fh);assert far['far']==36
    path=RAW/arm/'last10000.pt';assert not path.exists();torch.save(dict(arm=arm,step=STEPS,model=m.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/f'FIT_{arm}.json',dict(complete=True,step=STEPS,new_updates=UPDATES,checkpoint=C.bound(path),initialization=p['initialization'],
        initial_state=initial,protocol=C.bound(DOC/'PROTOCOL.json'),source_after=after,far_held_after=far,source_curves=curves,history=history,seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),real_training_images=0,backbone_frozen=True,auto_promoted=False))
    print('TAIL_FIT_COMPLETE',arm,dict(far=far['far'],recovered=far['far_recovered'],normal=after),flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True);fits={a:C.read(DOC/f'FIT_{a}.json') for a in ARMS};models={}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in ARMS},before_real_features=True))
    for a,f in fits.items():
        C.verify(f['checkpoint']);ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['step']==STEPS and ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json');m=M.Head().cuda();m.load_state_dict(ck['model']);models[a]=m.eval()
    backbone,_=D.A.load();previous={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    prior={r['id']:r for r in C.read(U.RAW/'INFERENCE_RECEIPTS.json')};out={a:[] for a in ARMS};receipts=[]
    for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=previous[r['id']];x=W.prepare_input(im,old['prediction']);mid,late=U.extract(backbone,[x],True);mid=mid[0];late=late[0]
        assert N.array_sha(late)==prior[r['id']]['feature_sha'] and N.array_sha(mid)==prior[r['id']]['mid_sha']
        t=tuple(torch.as_tensor(f,device='cuda').float()[None] for f in [late,mid]);q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        for a,m in models.items():
            points=M.decode(m(t,q,v))[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']));new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
            pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=new.tolist();P.assert_preserved(old['prediction'],pred)
            out[a].append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipts.append(dict(id=r['id'],feature_sha=N.array_sha(late),mid_sha=N.array_sha(mid),matrix=x['matrix'].tolist(),image=r['image']))
        if (i+1)%50==0:print('TAIL_INFER',i+1,'/194',N.E.gpu(),flush=True)
    for a,rr in out.items():C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,checkpoint=fits[a]['checkpoint'],records=rr,GT_free=True))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts);C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+[C.bound(RAW/'INFERENCE_RECEIPTS.json')],all_predictions_before_GT_scoring=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        cache();bank=load_bank()
        for arm in ARMS:train(arm,bank)
        X.read_feature.cache_clear();U.read_mid.cache_clear();infer()
        with scope():D.report()
