"""Fixed mid-layer+last-layer source-only corner localization experiment."""
import argparse
import copy
import shutil
import time
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import dino_source_diversity as X
from . import dino_rgb_detail as T
from . import dino_mid_feature_model as M

C=X.C;D=X.D;N=X.N;P=X.P;W=X.W;V=X.V
PHASE='dino_mid_feature';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;STEPS=5000
BASE_BATCH=D.tensor_batch


def tensor_batch(rows,perturbed=None):
    b=BASE_BATCH(rows,perturbed)
    b['feature']=(b['feature'],torch.as_tensor(np.stack([np.asarray(r['mid_feature']) for r in rows]),device='cuda').float())
    return b


@contextmanager
def scope():
    with V.scope(),patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),\
         patch.object(D,'STEPS',STEPS),patch.object(D,'ARMS',['SYN']),patch.object(D,'M',M),patch.object(D,'tensor_batch',tensor_batch):yield


def verify():
    with scope():return D.verify()


def prepare():
    p=copy.deepcopy(X.verify())
    p.update(arms=['SYN'],steps=STEPS,real_training_images=0,
        objective='Generate genuinely missing corner locations while preserving correct corners, without new labels/tags.',
        head='Parent image-only last-layer head plus normalized block6 DINO path:384->64 conv1 GN8 GELU->64 conv1 zero-init. Add to parent projected64 features before unchanged mix/up/out. Same192x144 decoder, no point hint or prior.',
        mid_feature='Frozen same official DINOv2ViTS14,zero-based block5 of12,official final LayerNorm and remove class/register tokens. Same crop768x576,resize784x588,normalization andFP16cache384x56x42. No feature-layer sweep.',
        controls='Same8192source+64held,source5000schedule,targets,masks,crop,last-layerfeaturecache,loss,decoder,seed,AdamW.001/.0001,micro4,source8. No real loss. Parent parameter initialization exact,extra path initial output zero. New feature information+capacity change together;not pure layer attribution.',
        optimizer='From scratch seed20260929,fixedlast5000,sourcecurves diagnostic only,no early stopping/GT-selected checkpoint.',
        training='5000steps SYN-only;common1412train andheld64 probes at1000..5000. No pseudo branch. Onlyfinal5000 scored onall194real.',
        rationale='Last-layerhead failed source andreal preservation despite20koptimization;smallRGBdetail path recovered4/64far butdamaged64/511. Test mid-transformer information with broader context,not another gate. Midlayer benefit is an unproven hypothesis.',
        factor_selection='Choose exactly midpoint block6 alongside existingblock12 before fitting. No layer/architecture/threshold sweep; repeatedDEV informed design,notindependentconfirmation.',
        sources=p['sources']+[C.bound(f) for f in [__file__,M.__file__,C.HERE/'test_dino_mid_feature_model.py',X.DOC/'CACHE_COMPLETE.json',X.DOC/'FIT_SYN.json',X.DOC/'COMPLETION_AUDIT.json',T.DOC/'RESULTS.json',T.DOC/'COMPLETION_AUDIT.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(X.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,['SYN'],p['sources']+[C.bound(DOC/'PROTOCOL.json')]);print('MID_FEATURE_PROTOCOL_LOCKED',flush=True)


def image_tensor(items):
    rgb=np.stack([x['rgb'] for x in items])+N.C.MEAN[None,:,None,None]
    x=torch.as_tensor(rgb,device='cuda')/255
    x=F.interpolate(x,size=(784,588),mode='bilinear',align_corners=False)
    mean=torch.tensor([.485,.456,.406],device=x.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=x.device)[None,:,None,None]
    return (x-mean)/std


def array(tokens):
    z=tokens.reshape(len(tokens),56,42,384).permute(0,3,1,2).contiguous()
    assert torch.isfinite(z).all()
    return z.cpu().numpy().astype(np.float16)


@torch.no_grad()
def extract(backbone,items,both=False):return [array(z) for z in M.block_features(backbone,image_tensor(items),both)]


def cache():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'CACHE_COMPLETE.json').exists():
        for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:C.verify(r['cache']);C.verify(r['mid_cache'])
        return
    assert shutil.disk_usage(C.ROOT).free>40*1024**3,'Keep disk safety reserve'
    source=P.SourceData();old=[r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source']
    images={r['row']:r['image'] for r in p['source_records']};backbone,_=D.A.load();records=[];chunks=[];start=time.monotonic()
    assert len(old)==8256
    for offset in range(0,len(old),256):
        group=old[offset:offset+256];receipt=DOC/'cache_chunks'/f'{offset:05d}.json'
        if receipt.exists():
            part=C.read(receipt);C.verify(part['protocol']);assert part['offset']==offset and len(part['records'])==len(group)
            for r,o in zip(part['records'],group):
                assert {k:v for k,v in r.items() if k!='mid_cache'}==o
                C.verify(r['cache']);C.verify(r['mid_cache']);C.verify(images[r['row']])
            records.extend(part['records']);chunks.append(C.bound(receipt));continue
        batch=[]
        for r in group:
            C.verify(r['cache']);C.verify(images[r['row']]);x=W.source_item(source,r['row'])
            with np.load(C.ROOT/r['cache']['path']) as z:
                for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
            mid=extract(backbone,[x])[0][0];path=RAW/'mid_cache'/(r['key']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with path.open('xb') as f:np.save(f,mid,allow_pickle=False)
            np.testing.assert_array_equal(np.load(path,allow_pickle=False),mid)
            batch.append(dict(r,mid_cache=C.bound(path)))
        C.freeze(receipt,dict(offset=offset,protocol=C.bound(DOC/'PROTOCOL.json'),records=batch,source_only=True))
        records.extend(batch);chunks.append(C.bound(receipt))
        print('MID_FEATURE_CACHE',len(records),'/8256',round(time.monotonic()-start,1),N.E.gpu(),flush=True)
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=records,chunks=chunks,source_images=8256,real_images=0,
        protocol=C.bound(DOC/'PROTOCOL.json'),parent=C.bound(X.DOC/'CACHE_COMPLETE.json'),current_run_seconds=time.monotonic()-start))
    print('MID_FEATURE_CACHE_COMPLETE',flush=True)


@lru_cache(maxsize=128)
def read_mid(path):return np.load(path,allow_pickle=False)


class LazyMid:
    def __init__(self,path):self.path=str(path)
    def __array__(self,dtype=None):return np.asarray(read_mid(self.path),dtype=dtype)


def load_bank():
    bank=[]
    for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:
        C.verify(r['cache']);C.verify(r['mid_cache']);path=C.ROOT/r['cache']['path']
        with np.load(path) as z:a={k:np.array(z[k]) for k in z.files if k not in ['feature','protocol_sha256']}
        bank.append(dict(r,**a,feature=X.LazyFeature(path),mid_feature=LazyMid(C.ROOT/r['mid_cache']['path'])))
    return bank


def train():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'FIT_SYN.json').exists():C.verify(C.read(DOC/'FIT_SYN.json')['checkpoint']);return
    sm={r['row']:r for r in load_bank()};held=[sm[i] for i in p['source_held_rows']];probe=[sm[i] for i in p['parent_source_rows']]
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929);m=M.Head().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
    for k,v in C.read(X.DOC/'FIT_SYN.json')['initial_state'].items():assert initial[k]==v
    history=[];curves=[];step0=0;prior=0.;resumes=sorted((RAW/'SYN').glob('resume_*.pt'))
    if resumes:
        ck=torch.load(resumes[-1],map_location='cpu',weights_only=False);assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
        m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);history=ck['history'];curves=ck['curves'];step0=ck['step'];prior=ck['seconds']
    start=time.monotonic()
    for step in range(step0,STEPS):
        src=[sm[i] for i in p['source_samples'][step]];assert all(r['train'] for r in src);m.train()
        with scope():value=X.L.update(m,opt,[src],step)
        history.append(dict(step=step+1,loss=value))
        if (step+1)%1000==0:
            m.eval()
            with scope():curves.append(dict(step=step+1,common_source_train1412=D.probe(m,probe),source_held=D.probe(m,held)))
            path=RAW/'SYN'/f'resume_{step+1:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
            torch.save(dict(arm='SYN',step=step+1,model=m.state_dict(),optimizer=opt.state_dict(),history=history,curves=curves,
                seconds=prior+time.monotonic()-start,protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
            print('MID_FEATURE_CURVE',curves[-1],flush=True)
        if step==step0 or (step+1)%250==0:print('MID_FEATURE_TRAIN',step+1,'/5000',round(value,4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    m.eval()
    with scope():after=D.probe(m,held);stress=D.probe(m,held,True)
    path=RAW/'SYN/last5000.pt';assert not path.exists()
    torch.save(dict(arm='SYN',step=STEPS,model=m.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/'FIT_SYN.json',dict(complete=True,step=STEPS,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),cache=C.bound(DOC/'CACHE_COMPLETE.json'),
        initial_state=initial,source_after=after,source_stress=stress,source_curves=curves,history=history,seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),
        source_training_images=8192,real_training_images=0,backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    X.read_feature.cache_clear();read_mid.cache_clear();print('MID_FEATURE_FIT_COMPLETE',after,flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True);fit=C.read(DOC/'FIT_SYN.json');C.verify(fit['checkpoint'])
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={'SYN':C.bound(DOC/'FIT_SYN.json')},before_real_features=True))
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['step']==STEPS and ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
    m=M.Head().cuda();m.load_state_dict(ck['model']);m.eval();backbone,_=D.A.load()
    ev=C.read(DOC/'EVAL_PROTOCOL.json')['records'];previous={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    prior={r['id']:r for r in C.read(X.RAW/'INFERENCE_RECEIPTS.json')};out=[];receipts=[]
    for i,r in enumerate(ev):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=previous[r['id']];x=W.prepare_input(im,old['prediction'])
        mid,late=extract(backbone,[x],True);mid=mid[0];late=late[0];assert N.array_sha(late)==prior[r['id']]['feature_sha']
        t=tuple(torch.as_tensor(f,device='cuda').float()[None] for f in [late,mid]);q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        points=M.decode(m(t,q,v))[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']))
        new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
        pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=new.tolist();P.assert_preserved(old['prediction'],pred)
        out.append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipts.append(dict(id=r['id'],feature_sha=N.array_sha(late),mid_sha=N.array_sha(mid),matrix=x['matrix'].tolist(),image=r['image']))
        if (i+1)%50==0:print('MID_FEATURE_INFER',i+1,'/194',N.E.gpu(),flush=True)
    C.freeze(RAW/'EVAL_PREDICTIONS_SYN.json',dict(complete=True,arm='SYN',checkpoint=fit['checkpoint'],records=out,GT_free=True))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f) for f in ['EVAL_PREDICTIONS_SYN.json','INFERENCE_RECEIPTS.json']],all_predictions_before_GT_scoring=True))
    print('MID_FEATURE_OUTPUTS_LOCKED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','cache','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='cache':cache()
    else:
        cache();train();infer()
        with scope():D.report()
