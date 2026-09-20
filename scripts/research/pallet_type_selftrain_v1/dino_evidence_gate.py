"""Source-supervised selective correction of fixed image-only DINO candidates."""
import argparse
import copy
import hashlib
from contextlib import contextmanager
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import dino_wide_visual as V
from . import dino_evidence_gate_model as G

W=V.W;D=V.D;C=V.C;P=V.P;N=V.N;R=D.R
PHASE='dino_evidence_gate';DOC=R.DOC/PHASE;RAW=R.RAW/PHASE;ARMS=D.ARMS


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    V.verify();return p


def prepare():
    p=V.verify();held=[r for r in p['source_records'] if r['partition']=='heldout']
    groups=sorted({r['scenario'] for r in held},key=lambda s:hashlib.sha256(('evidence-gate-split:'+s).encode()).hexdigest())
    assert len(groups)==64
    calibration=[r['row'] for r in held if r['scenario'] in groups[:32]]
    test=[r['row'] for r in held if r['scenario'] in groups[32:]]
    paths=[__file__,G.__file__,C.HERE/'test_dino_evidence_gate_model.py',W.DOC/'CACHE_COMPLETE.json',V.DOC/'PROTOCOL.json',
           V.DOC/'FIT_SYN.json',V.DOC/'FIT_MIX.json',V.DOC/'RESULTS.json',V.DOC/'COMPLETION_AUDIT.json',V.DOC/'BACKBONE.json',R.__file__]
    protocol=dict(arms=ARMS,source_train_rows=[r['row'] for r in p['source_records'] if r['partition']=='train'],
        calibration_rows=calibration,test_rows=test,feature_names=G.NAMES,steps=500,
        objective='Learn when a newly localized corner has stronger image evidence while preserving accurate R0 corners. No new manual labels or tags.',
        candidates='Frozen dino_wide_visual SYN and MIX last1000. Each gate proposes its own decoded point; the other correlated head supplies evidence,not independent corroboration. No corner reindexing,GT-selected candidates or displacement cap.',
        features='16 image heatmap evidence features: pooled5x5 old/new masses,ratios,normalized entropies,own/other peak masses,normalized displacement/model disagreement,bounds and input validity. No GT input. R0 box sets wide crop.',
        supervision='Existing synthetic1412 only. Per image raw input plus4 seeded forced perturb views,using unchanged existing source GT. Positive iff candidate error<=10px AND candidate+5px<old error. Targets are source-only evaluation of candidate benefit,not new manual annotations. Invalid/unsupported/center excluded.',
        fit='Separate SYN/MIX gates,16->32GELU->16GELU->1;train-only normalization std floor1e-4;clip standardized features[-10,10];seed20261001;AdamW .001/.0001;500 steps,batch512,unweighted BCE,fixed last. No new image head fitting.',
        calibration='64 previously held source scenarios split by SHA256 into32calibration+32audit before any gate fit. Threshold grid0..1 step.01 plus1.01 null. Require calibration raw PCK10 drop<=1pp,P90<=1.1R0,normal damage<=1%,and>=95%beneficial precision on raw+4corrupt. Maximize raw hard recovery,then combined recovery,then higher threshold. Never use real GT.',
        evaluation='All194PLASTIC retained. If gate declines a corner,original float64 R0 coordinates remain bit-exact;center/invalid/box/confidence/otherdetections preserved. All outputs locked before scoring. Test32source not used for fitting/threshold. Repeated DEV,not independent confirmation.',
        limitations='Image heads already trained on gate-fit images;source calibration controls but cannot eliminate confidence overfit/domain shift. MIX pseudo teacher historical3manual train/eval overlap remains. No frame filtering,final model change or automatic promotion.',
        new_annotations=0,new_tags=0,auto_promote=False,sources=[C.bound(x) for x in paths])
    C.freeze(DOC/'BACKBONE.json',C.read(V.DOC/'BACKBONE.json'))
    C.freeze(DOC/'PROTOCOL.json',protocol);R.evaluation_protocol(PHASE,ARMS,protocol['sources']+[C.bound(DOC/'PROTOCOL.json')])
    print('EVIDENCE_PROTOCOL_LOCKED',len(protocol['source_train_rows']),len(calibration),len(test),flush=True)


def heads():
    out={}
    for a in ARMS:
        fit=C.read(V.DOC/f'FIT_{a}.json');C.verify(fit['checkpoint'])
        ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
        m=V.V.Head().cuda();m.load_state_dict(ck['model']);out[a]=m.eval().requires_grad_(False)
    return out


def normalized(x,mean,std):return ((x-mean)/std).clamp(-10,10)


@torch.no_grad()
def cache_source():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    with W.scope():bank=D.load_bank()
    rows=[r for r in bank if r['domain']=='source'];models=heads();data={a:[] for a in ARMS}
    for offset in range(0,len(rows),4):
        batch=rows[offset:offset+4];b=D.tensor_batch(batch)
        logits={a:m(b['feature'],b['points'],b['valid']) for a,m in models.items()}
        for view in range(5):
            noisy=batch if view==0 else [P.perturb(r,np.random.default_rng(np.random.SeedSequence([20261001,int(r['row']),view])),True) for r in batch]
            q=torch.as_tensor(np.stack([r['points'] for r in noisy]),device='cuda')
            valid=torch.as_tensor(np.stack([r['valid'] for r in noisy]),device='cuda')
            diag=torch.as_tensor([r['bbox_diagonal']*r['matrix'][0,0] for r in batch],device='cuda',dtype=torch.float32)
            for a in ARMS:
                other=next(k for k in ARMS if k!=a);features,new=G.features(logits[a],logits[other],q,valid,diag)
                f=features.cpu().numpy();new=new.cpu().numpy()
                for i,(r,n) in enumerate(zip(batch,noisy)):
                    mask=r['target_valid']&n['valid'];mask[8]=False;gain=r['matrix'][0,0]
                    before=np.linalg.norm(n['points']-r['target'],axis=-1)/gain
                    after=np.linalg.norm(new[i]-r['target'],axis=-1)/gain
                    for j in np.flatnonzero(mask):data[a].append((int(r['row']),view,int(j),f[i,j],before[j],after[j]))
        if offset%400==0:print('GATE_SOURCE',offset,'/',len(rows),N.E.gpu(),flush=True)
    receipt={}
    for a,rr in data.items():
        path=RAW/f'SOURCE_{a}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:np.savez(f,row=np.array([r[0] for r in rr]),view=np.array([r[1] for r in rr]),corner=np.array([r[2] for r in rr]),
            features=np.stack([r[3] for r in rr]),before=np.array([r[4] for r in rr]),after=np.array([r[5] for r in rr]))
        receipt[a]=C.bound(path)
    C.freeze(DOC/'SOURCE_CACHE.json',dict(artifacts=receipt,protocol=C.bound(DOC/'PROTOCOL.json'),source_images=len(rows),views=5))


def train():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True);receipt=C.read(DOC/'SOURCE_CACHE.json')
    for arm in ARMS:
        C.verify(receipt['artifacts'][arm])
        with np.load(C.ROOT/receipt['artifacts'][arm]['path']) as z:data={k:np.array(z[k]) for k in z.files}
        tr=np.isin(data['row'],p['source_train_rows']);cal=np.isin(data['row'],p['calibration_rows']);test=np.isin(data['row'],p['test_rows'])
        assert (tr.astype(int)+cal+test==1).all()
        x=torch.as_tensor(data['features'],device='cuda');y=torch.as_tensor((data['after']<=10)&(data['after']+5<data['before']),device='cuda').float()
        tx=x[tr];ty=y[tr];mean=tx.mean(0);std=tx.std(0,unbiased=False).clamp_min(1e-4);tx=normalized(tx,mean,std)
        torch.manual_seed(20261001);torch.cuda.manual_seed_all(20261001);gate=G.Gate().cuda()
        opt=torch.optim.AdamW(gate.parameters(),lr=.001,weight_decay=.0001);history=[]
        for step in range(p['steps']):
            ix=torch.randint(len(tx),(512,),device='cuda');loss=F.binary_cross_entropy_with_logits(gate(tx[ix]),ty[ix])
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();history.append(float(loss.detach()))
        gate.eval()
        with torch.no_grad():score=gate(normalized(x,mean,std)).sigmoid().cpu().numpy()
        best,grid=G.calibrate(score[cal],data['before'][cal],data['after'][cal],data['view'][cal]==0)
        checks={}
        for name,mask in [('calibration32',cal),('test32',test)]:
            mask=mask&(data['view']==0);after=np.where(score[mask]>=best['threshold'],data['after'][mask],data['before'][mask])
            checks[name]=G.stats(data['before'][mask],after)
        path=RAW/f'GATE_{arm}.pt';assert not path.exists()
        torch.save(dict(model=gate.state_dict(),mean=mean.cpu(),std=std.cpu(),threshold=best['threshold'],protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
        C.freeze(DOC/f'FIT_{arm}.json',dict(checkpoint=C.bound(path),source_cache=receipt['artifacts'][arm],steps=p['steps'],history=history,
            train_samples=int(tr.sum()),train_positive=int(y[tr].sum()),calibration=best,calibration_grid=grid,source_checks=checks,
            source_gate=dict(PCK10=checks['test32']['PCK10']>=checks['test32']['input_PCK10']-.01,P90=checks['test32']['P90']<=checks['test32']['input_P90']*1.1),
            real_eval_used=False,auto_promoted=False))
        print('GATE_FIT',arm,'threshold',best['threshold'],checks,flush=True)
    C.freeze(DOC/'DECISION_LOCK.json',dict(fits={a:C.bound(DOC/f'FIT_{a}.json') for a in ARMS},before_real_inference=True))


def load_gates():
    models={}
    for a in ARMS:
        f=C.read(DOC/f'FIT_{a}.json');C.verify(f['checkpoint']);ck=torch.load(C.ROOT/f['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
        m=G.Gate().cuda();m.load_state_dict(ck['model']);models[a]=(m.eval(),ck['mean'].cuda(),ck['std'].cuda(),ck['threshold'])
    return models


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    for b in C.read(DOC/'DECISION_LOCK.json')['fits'].values():C.verify(b)
    models=heads();gates=load_gates();backbone,_=D.A.load()
    original={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']}
    out={a:[] for a in ARMS};receipts=[]
    for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=original[r['id']];x=W.prepare_input(im,old['prediction'])
        feature=W.extract(backbone,[x])[0];f=torch.as_tensor(feature,device='cuda').float()[None]
        q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        diag=torch.tensor([x['bbox_diagonal']*x['matrix'][0,0]],device='cuda',dtype=torch.float32)
        logits={a:m(f,q,valid) for a,m in models.items()};decisions={}
        for a in ARMS:
            other=next(k for k in ARMS if k!=a);feat,new=G.features(logits[a],logits[other],q,valid,diag)
            gate,mean,std,threshold=gates[a];s=gate(normalized(feat,mean,std)).sigmoid()[0].cpu().numpy()
            candidate=N.C.transform_points(new[0].cpu().numpy(),np.linalg.inv(x['matrix']))
            restored,mask=G.choose(x['original_points'],candidate,s,threshold,x['valid'])
            pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=restored.tolist();P.assert_preserved(old['prediction'],pred)
            out[a].append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
            decisions[a]=dict(scores=s.tolist(),accepted=mask.tolist(),features=feat[0].cpu().tolist(),candidate=candidate.tolist())
        receipts.append(dict(id=r['id'],feature_sha=N.array_sha(feature),decisions=decisions))
        if (i+1)%50==0:print('GATE_INFER',i+1,N.E.gpu(),flush=True)
    for a in ARMS:C.freeze(RAW/f'EVAL_PREDICTIONS_{a}.json',dict(complete=True,arm=a,records=out[a],GT_free=True))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/f'EVAL_PREDICTIONS_{a}.json') for a in ARMS]+[C.bound(RAW/'INFERENCE_RECEIPTS.json')],all_predictions_before_GT_scoring=True))


def report():
    verify()
    with patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW):D.report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:cache_source();train();infer();report()
