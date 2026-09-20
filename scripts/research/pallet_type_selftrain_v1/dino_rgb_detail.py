"""Fixed 5k SYN-only RGB-detail test on the unchanged 8192-image source pool."""
import argparse
import copy
import time
from contextlib import contextmanager
from functools import lru_cache
from unittest.mock import patch
import cv2
import numpy as np
import torch
from . import dino_source_diversity as X
from . import dino_diverse_continue as L
from . import dino_rgb_detail_model as M

C=X.C;D=X.D;N=X.N;P=X.P;W=X.W;V=X.V
PHASE='dino_rgb_detail';DOC=D.R.DOC/PHASE;RAW=D.R.RAW/PHASE;STEPS=5000
BASE_BATCH=D.tensor_batch


def tensor_batch(rows,perturbed=None):
    b=BASE_BATCH(rows,perturbed)
    b['feature']=(b['feature'],torch.as_tensor(np.stack([np.asarray(r['rgb_detail']) for r in rows]),device='cuda').float())
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
        objective='Recover genuinely missing large-error corner positions from image evidence, without new annotation or tags.',
        head='Parent image-only semantic head plus stride4 RGB detail branch:3->16 conv3 GN4 GELU->32 conv3 GN8 GELU->32 conv1 zero-init. Shared parent out.weight projects detail to9 heatmaps; no duplicate bias. Sum with unchanged parent output. No point hint/prior or movement cap.',
        detail_input='Same R0-box wide crop768x576, Gaussian5 sigma1 reflect101, sample pixel[4y,4x], RGB/255-.5,FP16cache3x192x144. No GT crop, edge annotation, tag or candidate choice.',
        controls='Same8192source images+64held,source5000sample schedule,targets,masks,frozen DINO cache,loss,decoder,seed,AdamW.001/.0001,micro4. Source8 only. No pseudo loss. Parent parameters seeded identically; extra branch initial output zero. Extra capacity and RGB information change together,not isolated attribution to RGB.',
        optimizer='From scratch seed20260929;fixedlast5000. No checkpoint or threshold choice from source/real GT.',
        training='Train common1412 and held64 curves at1000,2000,3000,4000,5000 diagnostics only. One fixed final real194 score. No real training branch.',
        rationale='8192 source optimization5k->20k improved common trainPCK10 81.73to91.16 but held78.24to77.84 and realPCK20 71.32to69.86. Test a direct higher-resolution image path instead of more steps or permissive gates;frozen14px patch features alone may lack useful detail. This is a hypothesis,not a proven cause.',
        factor_selection='One fixed architecture on existing source-only supervision. Repeated DEV informed design;not independent confirmation. No changed pool,mask,officialsymmetry,filters or finalmodel.',
        sources=p['sources']+[C.bound(f) for f in [__file__,M.__file__,C.HERE/'test_dino_rgb_detail_model.py',X.DOC/'COMPLETION_AUDIT.json',X.DOC/'FIT_SYN.json',X.DOC/'CACHE_COMPLETE.json',L.DOC/'COMPLETION_AUDIT.json',L.DOC/'RESULTS.json']])
    C.freeze(DOC/'BACKBONE.json',C.read(X.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    D.R.evaluation_protocol(PHASE,['SYN'],p['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('RGB_DETAIL_PROTOCOL_LOCKED',flush=True)


def cache():
    p=verify();N.setup()
    if (DOC/'CACHE_COMPLETE.json').exists():
        for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:C.verify(r['rgb_cache']);C.verify(r['cache'])
        return
    source=P.SourceData();rows={r['row']:r for r in p['source_records']}
    old=[r for r in C.read(X.DOC/'CACHE_COMPLETE.json')['records'] if r['domain']=='source'];records=[];start=time.monotonic()
    assert len(old)==8256
    for i,r in enumerate(old):
        C.verify(r['cache']);C.verify(rows[r['row']]['image']);x=W.source_item(source,r['row'])
        with np.load(C.ROOT/r['cache']['path']) as z:
            for k,v in D.item_arrays(x).items():np.testing.assert_array_equal(z[k],v)
        detail=M.detail_rgb(x['rgb'],N.C.MEAN);path=RAW/'rgb_cache'/(r['key']+'.npy');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            with path.open('xb') as f:np.save(f,detail,allow_pickle=False)
        np.testing.assert_array_equal(np.load(path,allow_pickle=False),detail)
        records.append(dict(r,rgb_cache=C.bound(path)))
        if (i+1)%500==0:print('RGB_DETAIL_CACHE',i+1,'/8256',round(time.monotonic()-start,1),flush=True)
    C.freeze(DOC/'CACHE_COMPLETE.json',dict(records=records,source_images=8256,real_images=0,protocol=C.bound(DOC/'PROTOCOL.json'),
        parent=C.bound(X.DOC/'CACHE_COMPLETE.json'),cv2_version=cv2.__version__,seconds=time.monotonic()-start))


@lru_cache(maxsize=128)
def read_rgb(path):return np.load(path,allow_pickle=False)


class LazyRGB:
    def __init__(self,path):self.path=str(path)
    def __array__(self,dtype=None):return np.asarray(read_rgb(self.path),dtype=dtype)


def load_bank():
    bank=[]
    for r in C.read(DOC/'CACHE_COMPLETE.json')['records']:
        C.verify(r['cache']);C.verify(r['rgb_cache']);path=C.ROOT/r['cache']['path']
        with np.load(path) as z:a={k:np.array(z[k]) for k in z.files if k not in ['feature','protocol_sha256']}
        bank.append(dict(r,**a,feature=X.LazyFeature(path),rgb_detail=LazyRGB(C.ROOT/r['rgb_cache']['path'])))
    return bank


def train():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    if (DOC/'FIT_SYN.json').exists():C.verify(C.read(DOC/'FIT_SYN.json')['checkpoint']);return
    sm={r['row']:r for r in load_bank()};held=[sm[i] for i in p['source_held_rows']];probe=[sm[i] for i in p['parent_source_rows']]
    torch.manual_seed(20260929);torch.cuda.manual_seed_all(20260929);m=M.Head().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
    assert {k:initial[k] for k in C.read(X.DOC/'FIT_SYN.json')['initial_state']}==C.read(X.DOC/'FIT_SYN.json')['initial_state']
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
            print('RGB_DETAIL_CURVE',curves[-1],flush=True)
        if step==step0 or (step+1)%250==0:print('RGB_DETAIL_TRAIN',step+1,'/5000',round(value,4),round(prior+time.monotonic()-start,1),N.E.gpu(),flush=True)
    m.eval()
    with scope():after=D.probe(m,held);stress=D.probe(m,held,True)
    path=RAW/'SYN/last5000.pt';assert not path.exists()
    torch.save(dict(arm='SYN',step=STEPS,model=m.state_dict(),protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    C.freeze(DOC/'FIT_SYN.json',dict(complete=True,step=STEPS,checkpoint=C.bound(path),protocol=C.bound(DOC/'PROTOCOL.json'),cache=C.bound(DOC/'CACHE_COMPLETE.json'),
        initial_state=initial,source_after=after,source_stress=stress,source_curves=curves,history=history,seconds=prior+time.monotonic()-start,
        source_gate=dict(PCK10=after['PCK10']>=after['input_PCK10']-.01,P90=after['P90_px']<=after['input_P90_px']*1.1),
        source_training_images=8192,real_training_images=0,backbone_frozen=True,real_eval_used=False,auto_promoted=False))
    X.read_feature.cache_clear();read_rgb.cache_clear();print('RGB_DETAIL_FIT_COMPLETE',after,flush=True)


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True);fit=C.read(DOC/'FIT_SYN.json');C.verify(fit['checkpoint'])
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={'SYN':C.bound(DOC/'FIT_SYN.json')},before_real_features=True))
    ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['step']==STEPS and ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
    m=M.Head().cuda();m.load_state_dict(ck['model']);m.eval();backbone,_=D.A.load()
    ev=C.read(DOC/'EVAL_PROTOCOL.json')['records'];previous={r['id']:r for r in C.read(D.R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    out=[];receipts=[]
    for i,r in enumerate(ev):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=previous[r['id']];x=W.prepare_input(im,old['prediction'])
        f=W.extract(backbone,[x])[0];rgb=M.detail_rgb(x['rgb'],N.C.MEAN)
        t=(torch.as_tensor(f,device='cuda').float()[None],torch.as_tensor(rgb,device='cuda').float()[None])
        q=torch.as_tensor(x['points'],device='cuda')[None];v=torch.as_tensor(x['valid'],device='cuda')[None]
        points=M.decode(m(t,q,v))[0].cpu().numpy();new=N.C.transform_points(points,np.linalg.inv(x['matrix']))
        new[~x['valid']]=x['original_points'][~x['valid']];new[8]=x['original_points'][8]
        pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=new.tolist();P.assert_preserved(old['prediction'],pred)
        out.append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipts.append(dict(id=r['id'],feature_sha=N.array_sha(f),rgb_sha=N.array_sha(rgb),matrix=x['matrix'].tolist(),image=r['image']))
        if (i+1)%50==0:print('RGB_DETAIL_INFER',i+1,'/194',N.E.gpu(),flush=True)
    C.freeze(RAW/'EVAL_PREDICTIONS_SYN.json',dict(complete=True,arm='SYN',checkpoint=fit['checkpoint'],records=out,GT_free=True))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f) for f in ['EVAL_PREDICTIONS_SYN.json','INFERENCE_RECEIPTS.json']],all_predictions_before_GT_scoring=True))
    print('RGB_DETAIL_OUTPUTS_LOCKED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':prepare()
    else:
        cache();train();infer()
        with scope():D.report()
