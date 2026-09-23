"""Before-training metadata, zero-lambda, gradient and calibration gates."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import E2ELoss,PoseLoss26
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26
from . import common as E
from .geo_loss import quadratic,criterion,geo_total

def metadata():
    assert E.read(E.DOC/'SYNTH_GEOMETRY_BINDING.json')['bound']==512
    assert E.read(E.DOC/'SYNTH_PROJECTION_PARITY.json')['max_error_px']<=.05
    native=dict(np.load(E.RAW/'H_NATIVE.npz'));occ={(r['material'],r['epoch'],r['slot']):r for r in E.read(E.RAW/'SYNTH_OCCURRENCES.json')}
    plans=E.plans();counts={};coordinate_error=0.
    for mat in E.C.MATERIALS:
        Hs=[];enabled=[];invalid=0
        for r in [r for r in plans if r['material']==mat]:
            H=np.zeros((16,16));ok=False
            if not r['real']:
                m=occ[mat,r['epoch'],r['slot']];J=np.array(m['M'])[:2,:2];inv=np.kron(np.eye(8),np.linalg.inv(J));hn=native[m['stem']];H=inv.T@hn@inv
                # Independently verify quadratic invariance before masking/trace normalization.
                residual=np.arange(16)/16;coordinate_error=max(coordinate_error,abs(float(residual@H@residual-(inv@residual)@hn@(inv@residual))))
                with np.load(E.ROOT/r['cache']['path']) as z:mask=np.repeat(z['keypoints'][0,:8,2]==2,2)
                H=H*mask[:,None]*mask[None,:];trace=np.trace(H);n=int(mask.sum())
                ok=bool(np.isfinite(H).all() and trace>1e-12 and n>0)
                if ok:H*=n/trace
                else:H*=0;invalid+=1
                assert np.all(H[~mask]==0) and np.all(H[:,~mask]==0)
                assert np.linalg.eigvalsh(H).min()>=-1e-8
            Hs.append(H);enabled.append(ok)
        dest=E.RAW/f'H_MODEL_{mat}.npz';assert not dest.exists();np.savez(dest,H=np.array(Hs,dtype=np.float32),enabled=np.array(enabled))
        counts[mat]=dict(enabled=sum(enabled),disabled_synthetic=invalid,real_disabled=2560,cache=E.bind(dest))
    E.save(E.DOC/'H_MODEL_AUDIT.json',dict(counts=counts,quadratic_coordinate_invariance_max=coordinate_error,real_geo_H_zero=True,normalization='trace to valid scalar count; relative weighting, not absolute physical covariance'))

def calibrate():
    E.C.setup();E.C.guard();assert not (E.DOC/'STOP.json').exists()
    from .trainer import GeoDataset,collate
    torch.manual_seed(42);np.random.seed(42)
    ds=GeoDataset('PLASTIC');idx=[i for i,r in enumerate(ds.records[:1024]) if not r['real']][:16]
    batch=collate([ds[i] for i in idx])
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):batch[k]=v.cuda()
    batch['img']=batch['img'].float()/255
    model=YOLO(str(E.C.H.C.N.E.R0),task='pose').model.float().cuda().train()
    # Published inference checkpoint has requires_grad=False; match Trainer's unfreeze rule.
    for name,param in model.named_parameters():param.requires_grad_('.dfl' not in name)
    args=E.read(E.C.DOC/'PREFLIGHT.json')['args'];model.args=get_cfg(overrides=dict(args,task='pose'))
    model.lambda_geo=0.;model.collect_geo=False
    params=[p for p in model.parameters() if p.requires_grad]
    preds=model(batch['img']);old=E2ELoss(model,TrueIgnorePoseLoss26);new=criterion(model)
    a,ai=old(preds,batch);b,bi=new(preds,batch)
    torch.testing.assert_close(a,b,atol=0,rtol=0);torch.testing.assert_close(ai,bi,atol=0,rtol=0)
    ga=torch.autograd.grad(a.sum(),params,retain_graph=True,allow_unused=True);gb=torch.autograd.grad(b.sum(),params,retain_graph=True,allow_unused=True)
    maxg=0.
    for x,y in zip(ga,gb):
        if x is None:assert y is None
        else:torch.testing.assert_close(x,y,atol=0,rtol=0);maxg=max(maxg,float((x-y).abs().max()))
    stock=E2ELoss(model,PoseLoss26);ss,si=stock(preds,batch);torch.testing.assert_close(ss,a,atol=0,rtol=0)
    # Same forward, no model selection, no optimizer instantiated.
    for part in [new.one2many,new.one2one]:part.collect_geo=True
    base,_=new(preds,batch);geot=geo_total(new,len(idx))
    keyparams=[(n,p) for n,p in model.named_parameters() if p.requires_grad and any(t in n for t in ['cv4','kpts_sigma'])]
    assert keyparams
    ggbase=torch.autograd.grad(base[[1,2,5]].sum(),[p for _,p in keyparams],retain_graph=True,allow_unused=True)
    gggeo=torch.autograd.grad(geot,[p for _,p in keyparams],retain_graph=False,allow_unused=True)
    norm=lambda xx:float(torch.sqrt(sum((x.detach().double()**2).sum() for x in xx if x is not None)))
    gbase=norm(ggbase);ggeo=norm(gggeo);lam=.25*gbase/ggeo
    assert np.isfinite([gbase,ggeo,lam]).all() and min(gbase,ggeo,lam)>0
    # Loss-only boundary tests with intentional supervised/ignored/real perturbations.
    pred=torch.zeros((2,8,2),device='cuda',requires_grad=True);target=torch.zeros_like(pred);H=torch.eye(16,device='cuda').repeat(2,1,1)
    mask=torch.ones((2,8),dtype=torch.bool,device='cuda');mask[0,7]=False;on=torch.tensor([True,False],device='cuda')
    zero=quadratic(pred,target,mask,H,on);assert zero==0
    offset=torch.zeros_like(pred);offset[0,0,0]=10;pos=quadratic(pred+offset,target,mask,H,on);assert pos>0
    off2=offset.clone();off2[0,7]=100;off2[1]=100
    same=quadratic(pred+off2,target,mask,H,on);torch.testing.assert_close(same,pos,atol=0,rtol=0)
    grad=torch.autograd.grad(same,pred)[0];assert grad[0,0,0]>0 and torch.equal(grad[1],torch.zeros_like(grad[1])) and (grad[0,7]==0).all()
    assert quadratic(pred+offset,target,mask,H*0,on)==0
    E.save(E.DOC/'LOSS_CONTRACT_TEST.json',dict(passed=True,M0_total_components_exact=True,M0_gradient_max_diff=maxg,stock_synthetic_loss_exact=True,
        pred_equals_gt_zero=True,supervised_shift_positive=True,ignored_shift_same=True,real_geo_exact_zero=True,synthetic_geo_gradient_nonzero=True,real_geo_gradient_zero=True,ignored_geo_gradient_zero=True,H_zero_zero=True,
        old_true_ignore_contract=E.bind(E.C.DOC/'LOSS_TEST.json'),criterion_source_unchanged=E.bind(E.ROOT/'scripts/self_training_yolo/v3/true_ignore_pose_loss.py')))
    E.save(E.DOC/'LAMBDA_CALIBRATION.json',dict(lambda_geo=lam,target_ratio=.25,g_base=gbase,g_geo=ggeo,microbatch=dict(material='PLASTIC',epoch=0,slots=idx),parameter_names=[n for n,_ in keyparams],
        base_definition='weighted E2E existing location/visibility/RLE, after pose/kobj/rle gains and batch multiplier',geo_definition='weighted E2E mean synthetic positive quadratic, after batch multiplier',lambda_position='kpt raw += lambda_geo/hyp.pose * geo; effective lambda AFTER pose gain; same E2E and batch factors',optimizer_created=False,optimizer_steps=0,DEV_used=False,reset='model discarded; each fit new process original R0, trainer seed42, new optimizer'))
    E.save(E.DOC/'TRAINING_PROTOCOL.json',dict(fits=4,steps_each=320,total_steps=1280,seed=42,arms=['M0','M1'],init=E.bind(E.C.H.C.N.E.R0),args=args,
        lambda_calibration=E.bind(E.DOC/'LAMBDA_CALIBRATION.json'),frozen_S1=E.bind(E.C.DOC/'AUGMENTATION_PLAN.jsonl'),real_geo_zero=True,eval_during_train=False,checkpoint='final last only',no_rescue=True))
    print('CALIBRATION_READY',lam,gbase,ggeo,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','calibrate']);a=p.parse_args();globals()[a.stage]()
