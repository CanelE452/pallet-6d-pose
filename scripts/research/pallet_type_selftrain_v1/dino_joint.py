"""Whole-hypothesis image-conditioned selection; no per-corner splicing."""
import argparse
import copy
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import dino_large_gate as L
from . import dino_joint_model as J

E=L.E;C=L.C;D=L.D;V=L.V;W=L.W;P=L.P;N=L.N;R=L.R
PHASE='dino_joint';DOC=R.DOC/PHASE;RAW=R.RAW/PHASE;ARMS=['JOINT']
SIDECAR=C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz'


def verify():
    p=C.read(DOC/'PROTOCOL.json')
    for b in p['sources']:C.verify(b)
    V.verify();return p


def prepare():
    parent=L.verify();p=dict(source_train_rows=parent['source_train_rows'],calibration_rows=parent['calibration_rows'],test_rows=parent['test_rows'],
        arms=ARMS,candidates=J.NAMES,steps=1000,features=J.DIM,
        objective='Recover large role and spatial errors by selecting one complete8corner hypothesis; stop mixing incompatible role assignments corner-by-corner.',
        hypotheses='12 GT-free full sets: R0, frozen image-only DINO SYN, frozen image-only DINO MIX;each identity/half/quarter/three-quarter channel assignment. Quarter is an actual edit,NOT extra evaluation symmetry. R0+permutation uses original float64 coordinates. Other candidates contain learned new locations. No pointwise splicing;center stays R0.',
        inputs='98 prediction/image features per candidate: two image heatmap local masses and differences toR0,normalized displacement/coordinates,bounds,distance to image candidates,entropy,nearest other predicted corner distances. No GT/dimensions/camera/depth input. Geometry is a learned descriptor,not an explicit rejection filter.',
        supervision='Same1412synthetic only. Source official per-row symmetry from existing sidecar used ONLY to score training/calibration targets,never choose inference. Six views:previous large-gate five+raw input quarter-permuted. Existing GT/target masks unchanged. Label=mean8error improvement>5px AND no R0<5 to candidate>10 damage. Training labels computed after GT-free proposals/features.',
        fitting='Shared98->64GELU->32GELU->1 binary ranker;candidate0 excluded from fit and reserved exact fallback. Train-only normalization std>=1e-4,clip10;seed20261002;AdamW lr.001 weight_decay.0001;1000steps,batch512,fixed last. Image heads frozen.',
        calibration='Same32source calibration+32test. Choose highest predicted beneficial probability among11nonbaseline candidates,orR0. Fixed threshold grid0..1 by.01 plus1.01 null. Calibration clean PCK10 drop<=1pp,PCK20>=R0,P90<=1.1R0,good damage<=1%;selected clean+artificial cases precision>=95%. Max raw corner recoveries,thencombined,thenhigherthreshold. No realGT tuning.',
        inference='All194PLASTIC,GT-free candidates/ranking. Original incomplete point sets fallback toR0,not removed. Chosen entire set,boxes/confidence/center/nonselected detections preserved. Full outputs locked before GT scoring.',
        limits='Choosing a whole hypothesis prevents hybridizing point sets but does not guarantee a correct cuboid. Source and repeated DEV already exposed;MIX teacher historical3manual train/eval overlap. A reindex-only gain is not new spatial localization. No newmanual labels/tags or automaticpromotion.',
        new_annotations=0,new_tags=0,auto_promote=False,
        sources=[C.bound(x) for x in [__file__,J.__file__,C.HERE/'test_dino_joint_model.py',L.__file__,SIDECAR,
            L.DOC/'PROTOCOL.json',L.DOC/'COMPLETION_AUDIT.json',L.DOC/'INTERVENTION_AUDIT.json',
            W.DOC/'CACHE_COMPLETE.json',V.DOC/'FIT_SYN.json',V.DOC/'FIT_MIX.json',V.DOC/'PROTOCOL.json',R.__file__]])
    C.freeze(DOC/'BACKBONE.json',C.read(V.DOC/'BACKBONE.json'));C.freeze(DOC/'PROTOCOL.json',p)
    R.evaluation_protocol(PHASE,ARMS,p['sources']+[C.bound(DOC/'PROTOCOL.json')]);print('JOINT_PROTOCOL_LOCKED',flush=True)


def view(r,j):
    if j<5:return L.perturb(r,j)
    assert j==5
    return dict(r,points=r['points'][J.Q].copy(),valid=r['valid'][J.Q].copy())


@torch.no_grad()
def cache():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True)
    with W.scope():bank=D.load_bank()
    bank=[r for r in bank if r['domain']=='source'];models=E.heads();side=np.load(SIDECAR)
    xx=[];ee=[];rows=[];views=[]
    for offset in range(0,len(bank),4):
        batch=bank[offset:offset+4];b=D.tensor_batch(batch);logits={a:m(b['feature'],b['points'],b['valid']) for a,m in models.items()}
        diag=torch.tensor([r['bbox_diagonal']*r['matrix'][0,0] for r in batch],device='cuda',dtype=torch.float32)
        for j in range(6):
            noisy=[view(r,j) for r in batch];q=torch.as_tensor(np.stack([r['points'] for r in noisy]),device='cuda')
            feature,candidates=J.describe(logits['SYN'],logits['MIX'],q,diag)
            feature=feature.cpu().numpy();candidates=candidates.cpu().numpy()
            for i,r in enumerate(batch):
                perms=side['permutations'][r['row'],:side['order'][r['row']]]
                err=J.canonical_errors(candidates[i],r['target'],r['target_valid'],perms,r['matrix'][0,0])
                xx.append(feature[i]);ee.append(err);rows.append(r['row']);views.append(j)
        if offset%400==0:print('JOINT_CACHE',offset,'/',len(bank),N.E.gpu(),flush=True)
    path=RAW/'SOURCE.npz';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez(f,features=np.asarray(xx),errors=np.asarray(ee),row=np.asarray(rows),view=np.asarray(views))
    C.freeze(DOC/'SOURCE_CACHE.json',dict(artifact=C.bound(path),images=len(bank),views=6,cases=len(rows),hypotheses=12,GT_only_for_targets=True))


def train():
    p=verify();N.setup();print('GPU',N.E.gpu(),flush=True);binding=C.read(DOC/'SOURCE_CACHE.json')['artifact'];C.verify(binding)
    with np.load(C.ROOT/binding['path']) as z:data={k:np.array(z[k]) for k in z.files}
    tr=np.isin(data['row'],p['source_train_rows']);cal=np.isin(data['row'],p['calibration_rows']);test=np.isin(data['row'],p['test_rows'])
    assert (tr.astype(int)+cal+test==1).all()
    x=torch.as_tensor(data['features'],device='cuda');labels=J.beneficial(data['errors'])
    tx=x[tr,1:].reshape(-1,J.DIM);ty=torch.as_tensor(labels[tr,1:].reshape(-1),device='cuda').float()
    mean=tx.mean(0);std=tx.std(0,unbiased=False).clamp_min(1e-4);tx=E.normalized(tx,mean,std)
    torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002);model=J.Ranker().cuda()
    initial={k:N.array_sha(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);history=[]
    for step in range(p['steps']):
        ix=torch.randint(len(tx),(512,),device='cuda');loss=F.binary_cross_entropy_with_logits(model(tx[ix]),ty[ix])
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step();history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():prob=model(E.normalized(x,mean,std)).sigmoid().cpu().numpy()
    best,grid=J.calibrate(prob[cal],data['errors'][cal],data['view'][cal]==0)
    checks={}
    for name,mask in [('calibration32',cal),('test32',test)]:
        mask=mask&(data['view']==0);choice=J.choose(prob[mask],best['threshold']);checks[name]=J.stats(data['errors'][mask],choice)
    path=RAW/'JOINT.pt';assert not path.exists()
    torch.save(dict(model=model.state_dict(),mean=mean.cpu(),std=std.cpu(),threshold=best['threshold'],protocol_sha256=C.sha(DOC/'PROTOCOL.json')),path)
    t=checks['test32']
    C.freeze(DOC/'FIT_JOINT.json',dict(checkpoint=C.bound(path),source_cache=binding,steps=p['steps'],history=history,
        initial_state=initial,train_images=len(p['source_train_rows']),train_candidates=len(tx),positive=int(ty.sum()),
        calibration=best,calibration_grid=grid,source_checks=checks,source_gate=dict(PCK10=t['PCK10']>=t['input_PCK10']-.01,P90=t['P90']<=t['input_P90']*1.1),
        real_eval_used=False,auto_promoted=False))
    C.freeze(DOC/'DECISION_LOCK.json',dict(fit=C.bound(DOC/'FIT_JOINT.json'),before_real_inference=True))
    print('JOINT_FIT',best,checks,flush=True)


def load():
    fit=C.read(DOC/'FIT_JOINT.json');C.verify(fit['checkpoint']);ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
    m=J.Ranker().cuda();m.load_state_dict(ck['model']);return m.eval(),ck['mean'].cuda(),ck['std'].cuda(),ck['threshold']


def restore(old,candidate_crop,index,matrix):
    # Exact original-pixel representation for pure role edits; no affine round trip.
    result=np.asarray(old).copy() if index==0 else (np.asarray(old)[J.PERMS[index]].copy() if index<4 else N.C.transform_points(candidate_crop,np.linalg.inv(matrix)))
    result[8]=np.asarray(old)[8];return result


@torch.no_grad()
def infer():
    verify();N.setup();print('GPU',N.E.gpu(),flush=True);C.verify(C.read(DOC/'DECISION_LOCK.json')['fit'])
    heads=E.heads();backbone,_=D.A.load();model,mean,std,threshold=load()
    original={r['id']:r for r in C.read(R.BASE_RAW/'EVAL_PREDICTIONS_R0.json')['records']};outputs=[];receipts=[]
    for i,r in enumerate(C.read(DOC/'EVAL_PROTOCOL.json')['records']):
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));old=original[r['id']];x=W.prepare_input(im,old['prediction'])
        feature=W.extract(backbone,[x])[0];f=torch.as_tensor(feature,device='cuda').float()[None]
        q=torch.as_tensor(x['points'],device='cuda')[None];valid=torch.as_tensor(x['valid'],device='cuda')[None]
        logits={a:m(f,q,valid) for a,m in heads.items()};diag=torch.tensor([x['bbox_diagonal']*x['matrix'][0,0]],device='cuda',dtype=torch.float32)
        xx,candidates=J.describe(logits['SYN'],logits['MIX'],q,diag);prob=model(E.normalized(xx,mean,std)).sigmoid()[0].cpu().numpy()
        choice=int(J.choose(prob,threshold)) if x['valid'][:8].all() else 0
        new=restore(x['original_points'],candidates[0,choice].cpu().numpy(),choice,x['matrix'])
        pred=copy.deepcopy(old['prediction']);P.top(pred)['keypoints_xy']=new.tolist();P.assert_preserved(old['prediction'],pred)
        outputs.append(dict(id=r['id'],kind='PLASTIC',raw_hw=old['raw_hw'],prediction=pred))
        receipts.append(dict(id=r['id'],choice=choice,candidate=J.NAMES[choice],probabilities=prob.tolist(),features=xx[0].cpu().tolist(),feature_sha=N.array_sha(feature)))
        if (i+1)%50==0:print('JOINT_INFER',i+1,N.E.gpu(),flush=True)
    C.freeze(RAW/'EVAL_PREDICTIONS_JOINT.json',dict(complete=True,arm='JOINT',records=outputs,GT_free=True,checkpoint=C.read(DOC/'FIT_JOINT.json')['checkpoint'],image_fits={a:C.bound(V.DOC/f'FIT_{a}.json') for a in E.ARMS}))
    C.freeze(RAW/'INFERENCE_RECEIPTS.json',receipts)
    C.freeze(DOC/'OUTPUTS_LOCK.json',dict(artifacts=[C.bound(RAW/'EVAL_PREDICTIONS_JOINT.json'),C.bound(RAW/'INFERENCE_RECEIPTS.json')],all_predictions_before_GT_scoring=True))


def report():
    with patch.object(D,'PHASE',PHASE),patch.object(D,'DOC',DOC),patch.object(D,'RAW',RAW),patch.object(D,'ARMS',ARMS):D.report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:cache();train();infer();report()
