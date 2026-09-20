"""Bounded DINOv2 frozen-feature large-displacement localization experiment."""
import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as TF

from . import recovery_pseudo_denoise as P
from . import dino_localization_assets as A
from . import dino_localization_model as M

C=P.C; R=P.R; N=P.N
PHASE='dino_localization'; DOC=R.DOC/PHASE; RAW=R.RAW/PHASE
ARMS=['SYN','MIX']; STEPS=1000


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    backbone=C.read(DOC/'BACKBONE.json')
    for b in backbone['code']+[backbone['checkpoint'],backbone['loader'],backbone['license']]:C.verify(b)
    return p


def prepare():
    parent=P.verify();s=P.SourceData();train=sorted(set(np.asarray(parent['source_samples']).ravel().tolist()))
    held=parent['source_held_rows'];assert not set(train)&set(held)
    rows=[]
    for partition,indices in [('train',train),('heldout',held)]:
        for i in indices:
            r=s.data.source['records'][int(s.data.indices[i])]
            assert s.partitions[i]==partition
            rows.append(dict(row=i,id=r['id'],partition=partition,image=C.bound(r['image']),scenario=r['scenario_id']))
    assert not {r['scenario'] for r in rows if r['partition']=='train'}&{r['scenario'] for r in rows if r['partition']=='heldout'}
    ev=C.read(R.BASE_DOC/'EVAL_PROTOCOL.json')['records']
    forbid={r['image']['sha256'] for r in ev}
    assert not forbid&{r['image']['sha256'] for r in rows+parent['real_records']}
    paths=[__file__,M.__file__,A.__file__,C.HERE/'test_dino_localization_model.py',
        C.HERE/'heatmap_mode_decoder.py',P.__file__,P.O.__file__,P.N.C.__file__,
        R.__file__,P.SourceData.__module__.replace('.','/')+'.py',DOC/'BACKBONE.json',
        P.DOC/'PROTOCOL.json',R.BASE_RAW/'PSEUDO_ACCEPTED.json',R.BASE_RAW/'EVAL_PREDICTIONS_R0.json',
        s.data.directory/'CACHE_MANIFEST.json',s.data.directory/'CACHE_COMPLETE.json',s.data.run_dir/'SOURCE_MANIFEST.json']
    p=dict(arms=ARMS,steps=STEPS,source_records=rows,real_records=parent['real_records'],
        source_samples=[parent['source_samples'][i%300] for i in range(STEPS)],
        real_samples=[parent['real_samples'][i%300] for i in range(STEPS)],
        source_held_rows=held,source_cache_signature=parent['source_cache_signature'],
        objective='Recover actual new corner pixel locations with pretrained image features, not only reassign existing points.',
        backbone='Official DINOv2 ViT-S/14 frozen; existing R0 predicted-box crop384x288 resized392x294,ImageNet normalization,last patch tokens384x28x21,FP16 cache. No GT crop. No new data annotation,tag,depth,CAD or camera input.',
        head='384->64 pointwise GroupNorm/GELU;9prior maps+2position channels;3x3 75->64;bilinear56x42 and3x3 64->32;zero-init1x1 32->9;bilinear96x72;add fixed R0 Gaussian log prior sigma12crop_px floor-6. Whole crop search,no displacement cap.',
        supervision='Previous syntheticGT and exact original REF pseudo targets/common trusted masks; no new filtering. Center excluded. Bilinear heatmap CE averaged over supported corners; no unknown/PnP labels. SYN source8;MIX same source8 plus real8,each branch half weight.',
        corruption='Both source and real use identical fixed 50percent unchanged/25percent single/25percent all trusted target+uniform .05-.15 boxdiag noise from previous perturb function; per-step RNG independent of arm. Targets never changed.',
        optimizer='Seed20260929,AdamW lr.001 weight_decay.0001,1000steps,fixed last checkpoint,micro4. Identical initialization and source exposures. New head from scratch; not an isolated backbone ablation versus old PoseFix.',
        decoder='Fixed5x5 local expectation around global heatmap maximum; prior zero-head control recorded before fitting. No real/synthetic radius sweep or GT candidate selection.',
        evaluation='Full194PLASTIC and unchanged159 subset;all predictions locked before GT scoring;boxes/confidence/center/nonselected predictions preserved;invalid original corners unchanged. No frame filtering. Report genuine spatial versus role recovery.',
        gate='>=5 >20to<=10 recoveries across>=3frames,<=1percent <5to>10 damage,PCK20>=R0;source64cleanPCK10 drop<=1pp andP90<=1.1R0. Neither pass nor train completion equals overall goal completion.',
        limitations='Repeated DEV. MIX pseudo teacher historical3 manual training overlaps with194;SYN has no real supervision. Existing pseudo targets may be wrong. Frozen image backbone adds inference cost. No automatic promotion.',
        new_annotations=0,new_tags=0,auto_promote=False,source_training_images=len(train),real_training_images=217,
        sources=[C.bound(p if Path(p).is_absolute() else C.ROOT/p) for p in paths])
    C.freeze(DOC/'PROTOCOL.json',p)
    R.evaluation_protocol(PHASE,ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('DINO_PROTOCOL',len(train),'source train',len(held),'source held','217 pseudo train',flush=True)


def item_arrays(x):
    return {k:np.asarray(x[k]) for k in ['points','valid','target','target_valid','matrix','original_points','bbox_diagonal']}


@torch.no_grad()
def extract(model,items):
    rgb=np.stack([x['rgb'] for x in items])+N.C.MEAN[None,:,None,None]
    x=torch.as_tensor(rgb,device='cuda')/255
    x=TF.interpolate(x,size=(392,294),mode='bilinear',align_corners=False)
    mean=torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
    z=model.forward_features((x-mean)/std)['x_norm_patchtokens']
    z=z.reshape(len(items),28,21,384).permute(0,3,1,2).contiguous()
    assert torch.isfinite(z).all()
    return z.cpu().numpy().astype(np.float16)


def cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    backbone,_=A.load();source=P.SourceData();parent=P.verify();real,_,_=P.load_inputs(parent)
    assert N.source_cache_signature(source.data,np.unique(np.r_[np.asarray(parent['source_samples']).ravel(),p['source_held_rows']]))==p['source_cache_signature']
    entries=[dict(id=r['id'],key='S'+str(r['row']),row=r['row'],domain='source',train=r['partition']=='train') for r in p['source_records']]
    entries += [dict(id=r['id'],key='R'+str(i),domain='real',train=r['train']) for i,r in enumerate(p['real_records'])]
    start=time.monotonic();records=[]
    for offset in range(0,len(entries),4):
        es=entries[offset:offset+4];items=[source.item(r['row']) if r['domain']=='source' else real[r['id']] for r in es]
        paths=[RAW/'cache'/(r['key']+'.npz') for r in es]
        missing=[i for i,path in enumerate(paths) if not path.exists()]
        feature=extract(backbone,[items[i] for i in missing]) if missing else []
        for j,i in enumerate(missing):
            path=paths[i];path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:
                np.savez(f,feature=feature[j],protocol_sha256=np.array(C.sha(DOC/'PROTOCOL.json')),**item_arrays(items[i]))
        for r,x,path in zip(es,items,paths):
            with np.load(path) as z:
                assert str(z['protocol_sha256'])==C.sha(DOC/'PROTOCOL.json') and z['feature'].shape==(384,28,21)
                assert np.isfinite(z['feature']).all()
                for k,v in item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
            records.append(dict(**r,cache=C.bound(path)))
        if offset%200==0:print('DINO_CACHE',min(offset+4,len(entries)),'/',len(entries),round(time.monotonic()-start,1),N.E.gpu(),flush=True)
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=records,seconds=time.monotonic()-start,
        protocol=C.bound(DOC/'PROTOCOL.json'),backbone=C.bound(DOC/'BACKBONE.json'),real_eval_features_used=False))
    print('DINO_CACHE_COMPLETE',len(records),flush=True)


def load_bank():
    receipt=C.read(DOC/'CACHE_COMPLETE.json');items=[]
    for r in receipt['records']:
        C.verify(r['cache'])
        with np.load(C.ROOT/r['cache']['path']) as z:
            items.append(dict(r,**{k:np.array(z[k]) for k in z.files if k!='protocol_sha256'}))
    return items


def tensor_batch(items,perturbed=None):
    q=items if perturbed is None else perturbed
    arrays={k:np.stack([x[k] for x in (items if k=='feature' else q)]) for k in ['feature','points','valid','target','target_valid']}
    return {k:torch.as_tensor(v,device='cuda').float() if v.dtype!=bool else torch.as_tensor(v,device='cuda') for k,v in arrays.items()}


@torch.no_grad()
def probe(model,items,stress=False):
    rng=np.random.default_rng(20260930);before=[];after=[]
    for offset in range(0,len(items),4):
        rows=items[offset:offset+4];noisy=[P.perturb(r,rng,True) for r in rows] if stress else rows
        b=tensor_batch(rows,noisy);out=M.decode(model(b['feature'],b['points'],b['valid'])).cpu().numpy()
        for r,n,q in zip(rows,noisy,out):
            mask=r['target_valid'];gain=r['matrix'][0,0]
            before.extend((np.linalg.norm(n['points'][mask]-r['target'][mask],axis=-1)/gain).tolist())
            after.extend((np.linalg.norm(q[mask]-r['target'][mask],axis=-1)/gain).tolist())
    a=np.asarray(before);b=np.asarray(after)
    return dict(n=len(b),PCK10=float((b<=10).mean()),PCK20=float((b<=20).mean()),median_px=float(np.median(b)),
        P90_px=float(np.quantile(b,.9)),mean_px=float(b.mean()),input_PCK10=float((a<=10).mean()),input_P90_px=float(np.quantile(a,.9)),
        hard=int((a>20).sum()),recovered=int(((a>20)&(b<=10)).sum()),good=int((a<5).sum()),damaged=int(((a<5)&(b>10)).sum()))


def train(arm):
    assert arm in ARMS;p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/f'FIT_{arm}.json').exists():C.verify(C.read(DOC/f'FIT_{arm}.json')['checkpoint']);return
    items=load_bank();sm={r['row']:r for r in items if r['domain']=='source'};rm={r['id']:r for r in items if r['domain']=='real'}
    held=[sm[i] for i in p['source_held_rows']];pseudo_held=[r for r in rm.values() if not r['train']]
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929)
    model=M.Head().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
    control=probe(model,held);history=[];step0=0;prior=0.
    resumes=sorted((RAW/arm).glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['arm']==arm
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);step0=ck['step'];history=ck['history'];prior=ck['seconds']
    start=time.monotonic();model.train()
    for step in range(step0,STEPS):
        source=[sm[i] for i in p['source_samples'][step]];real=[rm[i] for i in p['real_samples'][step]]
        assert all(r['train'] for r in source+real)
        branches=[source] if arm=='SYN' else [source,real]
        optimizer.zero_grad(set_to_none=True);values=[]
        for branch,rows in enumerate(branches):
            rng=np.random.default_rng(np.random.SeedSequence([20260929,step,branch]))
            noisy=[P.perturb(r,rng) for r in rows]
            for j in range(0,8,4):
                b=tensor_batch(rows[j:j+4],noisy[j:j+4]);z=model(b['feature'],b['points'],b['valid'])
                loss=M.loss(z,b['target'],b['target_valid'])/2/len(branches)
                assert torch.isfinite(loss);loss.backward();values.append(float(loss.detach()))
        optimizer.step();history.append(dict(step=step+1,loss=sum(values)))
        if (step+1)%250==0:
            path=RAW/arm/f'resume_{step+1:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm=arm,step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
                seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
        if step==step0 or (step+1)%100==0:print('DINO_TRAIN',arm,step+1,'/1000',round(history[-1]['loss'],4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    model.eval();after=probe(model,held);stress=probe(model,held,True)
    pp=probe(model,pseudo_held);ps=probe(model,pseudo_held,True)
    path=RAW/arm/'last1000.pt';assert not path.exists()
    torch.save(dict(arm=arm,step=STEPS,model=model.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/f'FIT_{arm}.json',dict(complete=True,step=STEPS,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),
        cache=C.bound(DOC/'CACHE_COMPLETE.json'),initial_state=initial,source_zero_head_control=control,source_after=after,source_stress=stress,
        pseudo32_consistency_NOT_accuracy=pp,pseudo32_stress_NOT_accuracy=ps,history=history,seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),
        backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    print('DINO_FIT_COMPLETE',arm,after,flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    fits={a:C.read(DOC/f'FIT_{a}.json') for a in ARMS}
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in ARMS},before_real_features=True))
    models={}
    for a,f in fits.items():
        C.verify(f['checkpoint']);ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json') and ck['step']==STEPS
        m=M.Head().cuda();m.load_state_dict(ck['model']);models[a]=m.eval()
    backbone,_=A.load();ev=C.read(DOC/'EVAL_PROTOCOL.json')['records']
    previous={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    out={a:[] for a in ARMS};receipts=[]
    for i,r in enumerate(ev):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
        old=previous[r['id']];x=N.C.prepare_input(im,old['prediction']);assert x is not None
        feature=extract(backbone,[x])[0];t=torch.as_tensor(feature,device='cuda').float()[None]
        q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        for a,m in models.items():
            logits=m(t,q,v);points=M.decode(logits)[0].cpu().numpy()
            restored=N.C.transform_points(points,np.linalg.inv(x['matrix']))
            restored[~x['valid']]=x['original_points'][~x['valid']];restored[8]=x['original_points'][8]
            pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=restored.tolist();P.assert_preserved(old['prediction'],pred)
            out[a].append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipt=dict(id=r['id'],feature_sha=N.array_sha(feature),matrix=x['matrix'].tolist(),image=r['image'])
        receipts.append(receipt)
        if (i+1)%50==0:print('DINO_INFER',i+1,'/194',N.E.gpu(),flush=True)
    for a,rows in out.items():C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,checkpoint=fits[a]['checkpoint'],records=rows,GT_free=True))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+[C.bound(RAW/'INFERENCE_RECEIPTS.json')],all_predictions_before_GT_scoring=True))
    print('DINO_OUTPUTS_LOCKED',flush=True)


def report():
    verify()
    for b in C.read(DOC/'OUTPUTS_LOCK.json')['artifacts']:C.verify(b)
    scores={a:R.score(PHASE,a,False)['metrics'] for a in ARMS}
    scores['R0']=[r for r in C.read(R.BASE_RAW/'EVAL_METRICS.json')['R0'] if r['kind']=='PLASTIC']
    ids=[r['id'] for r in scores['R0']];scores={a:[{r['id']:r for r in rr}[i] for i in ids] for a,rr in scores.items()}
    subset=C.read(R.BASE_DOC/'large_corner_recovery_v1/extreme_low_subset_v1/SUBSET_MANIFEST.json')
    keep={r['id'] for r in subset['records'] if r['kind']=='PLASTIC'};results={};checks={}
    for name,keys in [('full194',set(ids)),('retained159',keep)]:
        base=[r for r in scores['R0'] if r['id'] in keys];table={}
        for a,rr in scores.items():
            rows=[r for r in rr if r['id'] in keys];tails={}
            for t in [40,80,100]:
                tails[str(t)]=dict(hard=sum(e>t for b in base if b['matched'] for e in b['canonical_errors'] if e is not None),
                    recovered=sum(e is not None and e>t and n<=10 for b,x in zip(base,rows) if b['matched'] for e,n in zip(b['canonical_errors'],x['canonical_errors'])))
            table[a]=dict(summary=P.summary(rows),recovery=P.recovery_damage(base,rows,True),tails=tails,
                recovered_frames=sum(b['matched'] and any(e is not None and e>20 and n<=10 for e,n in zip(b['canonical_errors'],x['canonical_errors'])) for b,x in zip(base,rows)))
        results[name]=table
    for a in ARMS:
        x=results['full194'][a];f=C.read(DOC/f'FIT_{a}.json')
        checks[a]=dict(recovery5=x['recovery']['recovered']>=5,frames3=x['recovered_frames']>=3,
            damage1percent=x['recovery']['damage_rate']<=.01,pck20_preserved=x['summary']['PCK']['20']>=results['full194']['R0']['summary']['PCK']['20'],
            source_PCK10=f['source_gate']['PCK10'],source_P90=f['source_gate']['P90'])
    result=dict(results=results,checks=checks,passed={a:all(v.values()) for a,v in checks.items()},
        example={a:next(r for r in rows if r['id']=='eval_pallet07:1778652166837872128') for a,rows in scores.items()},
        goal_complete=False,new_annotations=0,new_tags=0,auto_promoted=False,independent_confirmation=False)
    C.freeze(DOC/'RESULTS.json',result)
    for a,x in results['full194'].items():print('DINO_RESULT',a,x,flush=True)
    print('DINO_GATES',checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','cache','train','infer','report']);p.add_argument('arm',nargs='?',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='cache':cache()
    elif a.action=='train':train(a.arm)
    elif a.action=='infer':infer()
    else:report()
