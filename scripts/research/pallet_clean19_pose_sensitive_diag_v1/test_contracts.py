"""Independent fresh-model numerical gate; tolerance is fixed before execution."""
import gc
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.loss import E2ELoss
from scripts.self_training_yolo.v3.true_ignore_pose_loss import TrueIgnorePoseLoss26
from scripts.research.pallet_clean19_structured_easyhard_v1.train import FrozenDataset
from . import common as E
from .geo_diag_loss import criterion,diagonal

def batch():
    ds=FrozenDataset('PLASTIC','S1');weights=np.load(E.RAW/'WEIGHTS_PLASTIC.npz')
    idx=[i for i,r in enumerate(ds.records[:1024]) if not r['real']][:16]
    samples=[]
    for i in idx:
        s=ds[i];s['diag_weight']=torch.from_numpy(weights['w'][i].copy());s['diag_enabled']=torch.tensor(bool(weights['enabled'][i]));samples.append(s)
    b=YOLODataset.collate_fn(samples)
    for k in ('diag_weight','diag_enabled'):b[k]=torch.stack(b[k])
    hashes={k:E.C.digest(v.numpy()) for k,v in b.items() if isinstance(v,torch.Tensor)}
    for k,v in b.items():
        if isinstance(v,torch.Tensor):b[k]=v.cuda()
    b['img']=b['img'].float()/255
    return b,idx,hashes

def fresh():
    E.deterministic()
    m=YOLO(str(E.C.H.C.N.E.R0),task='pose').model.float().cuda().train()
    for n,p in m.named_parameters():p.requires_grad_('.dfl' not in n)
    m.args=get_cfg(overrides=dict(E.read(E.C.DOC/'PREFLIGHT.json')['args'],task='pose'))
    m.lambda_diag=0.;m.collect_diag=False
    return m

def run(label):
    m=fresh();b,idx,hashes=batch()
    initial={k:E.C.digest(v.detach().cpu().numpy()) for k,v in m.state_dict().items()}
    inventory=[dict(name=n,shape=list(p.shape),trainable=p.requires_grad) for n,p in m.named_parameters()]
    loss=criterion(m) if label=='NEW' else E2ELoss(m,TrueIgnorePoseLoss26)
    total,items=loss(m(b['img']),b);total.sum().backward()
    grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in m.named_parameters()}
    result=dict(label=label,initial=initial,batch=hashes,slots=idx,inventory=inventory,total=total.detach().cpu().tolist(),items=items.cpu().tolist(),settings=E.settings())
    del m,b,loss,total,items;gc.collect();torch.cuda.empty_cache()
    return result,grads

def compare(a,b):
    rows=[];passed=True
    for n,x in a.items():
        y=b[n]
        if x is None or y is None:
            ok=x is None and y is None;rows.append(dict(name=n,passed=ok,unused=True));passed &=ok;continue
        d=(x-y).double();xd=x.double();yd=y.double()
        try:torch.testing.assert_close(x,y,atol=E.ATOL,rtol=E.RTOL);ok=True
        except AssertionError:ok=False
        rows.append(dict(name=n,passed=ok,max_abs=float(d.abs().max()),max_rel=float((d.abs()/torch.maximum(xd.abs(),yd.abs()).clamp_min(1e-12)).max()),l2=float(d.norm()),cosine=float(torch.nn.functional.cosine_similarity(xd.flatten(),yd.flatten(),dim=0)),tolerance_ratio=float((d.abs()/(E.ATOL+E.RTOL*yd.abs())).max())))
        passed &=ok
    return dict(passed=passed,parameters=rows,worst=max((r for r in rows if 'max_abs' in r),key=lambda r:r['tolerance_ratio']))

def math():
    p=torch.zeros(2,8,2,device='cuda',requires_grad=True);gt=torch.zeros_like(p);w=torch.ones(2,16,device='cuda');w[1]=0
    mask=torch.ones(2,8,dtype=torch.bool,device='cuda');mask[0,7]=False;w[0,14:]=0;enabled=torch.tensor([True,False],device='cuda')
    assert diagonal(p,gt,mask,w,enabled)==0
    offset=torch.zeros_like(p);offset[0,0]=torch.tensor([1.,-1.],device='cuda')
    a=diagonal(p+offset,gt,mask,w,enabled);offset[0,7]=100;offset[1]=100
    b=diagonal(p+offset,gt,mask,w,enabled);assert torch.equal(a,b)
    g=torch.autograd.grad(b,p)[0];assert g[0,0,0]>0 and g[0,0,1]<0 and (g[0,7]==0).all() and (g[1]==0).all()
    r=torch.tensor([1.,-1.],device='cuda');H=torch.ones(2,2,device='cuda')
    assert r@H@r==0 and (H.diag()*r.square()).sum()==2
    return dict(passed=True,cancellation_full=0,cancellation_diag=2,ignored_gradient_zero=True,real_gradient_zero=True,gradient_sign_correct=True,center_excluded=True)

def archive_contracts():
    """Add explicit CPU boundary tests and named summaries without rerunning GPU gate."""
    g=E.read(E.DOC/'GPU_GRADIENT_PARITY.json');assert g['passed']
    p=torch.zeros(2,8,2,requires_grad=True);target=torch.zeros_like(p);weight=torch.ones(2,16);weight[1]=0;mask=torch.ones(2,8,dtype=torch.bool);mask[0,7]=False;weight[0,14:]=0;on=torch.tensor([True,False])
    offset=torch.zeros_like(p);offset[0,0,0]=10
    positive=diagonal(p+offset,target,mask,weight,on);assert positive>0
    assert diagonal(p+offset,target,mask,weight*0,on)==0
    assert diagonal(p+100,target,mask,weight,torch.zeros_like(on))==0
    gradient=torch.autograd.grad(positive,p)[0];assert gradient[0,0,0]>0 and (gradient[1]==0).all() and (gradient[0,7]==0).all()
    E.save(E.DOC/'LOSS_MATH_TEST.json',dict(**E.read(E.DOC/'DIAGONAL_MATH_TEST.json'),supervised_10px_positive=True,zero_weight_zero=True,disabled_real_only_zero=True,no_optimizer=True))
    E.save(E.DOC/'GRADIENT_REPRODUCIBILITY.json',dict(atol=E.ATOL,rtol=E.RTOL,old_old=g['old_old'],settings=g['runs'][0]['settings'],three_independent_fresh_models=True,optimizer_steps=0))
    E.save(E.DOC/'M0_PARITY.json',dict(passed=g['old_new']['passed'],old_new=g['old_new'],losses_exact=all(g['runs'][0][k]==r[k] for r in g['runs'][1:] for k in ('total','items')),lambda_zero_branch_adds_no_loss=True,inventory=g['runs'][0]['inventory'],detailed_runs=E.bind(E.DOC/'GPU_GRADIENT_PARITY.json')))

def main():
    E.immutable();E.deterministic();E.C.guard()
    E.save(E.DOC/'DIAGONAL_MATH_TEST.json',math())
    records=[];gradients=[]
    for label in ('OLD_A','OLD_B','NEW'):
        record,gradient=run(label);records.append(record);gradients.append(gradient);print('GATE_FORWARD_BACKWARD',label,flush=True)
    assert all(r['initial']==records[0]['initial'] and r['batch']==records[0]['batch'] and r['inventory']==records[0]['inventory'] for r in records)
    aa=compare(gradients[0],gradients[1]);ab=compare(gradients[0],gradients[2])
    loss_pass=True
    for r in records[1:]:
        for key in ('total','items'):
            try:torch.testing.assert_close(torch.tensor(records[0][key]),torch.tensor(r[key]),atol=E.ATOL,rtol=E.RTOL)
            except AssertionError:loss_pass=False
    passed=aa['passed'] and ab['passed'] and loss_pass
    E.save(E.DOC/'GPU_GRADIENT_PARITY.json',dict(passed=passed,atol=E.ATOL,rtol=E.RTOL,old_old=aa,old_new=ab,loss_pass=loss_pass,runs=records,optimizer_steps=0))
    if not passed:E.save(E.DOC/'STOP.json',dict(reason='PREDECLARED_GPU_PARITY_FAILED',fits_started=0,old_old=aa['passed'],old_new=ab['passed']))
    print('GATE_RESULT',passed,'OLD_OLD',aa['worst'],'OLD_NEW',ab['worst'],flush=True)

if __name__=='__main__':
    import sys
    archive_contracts() if len(sys.argv)>1 and sys.argv[1]=='archive' else main()
