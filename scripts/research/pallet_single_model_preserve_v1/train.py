"""Adapter-only fitting process; reference files denied by runtime guard."""
from . import common as C
READS=C.train_guard()
import time
import numpy as np
import torch
from torch.nn import functional as F
from .adapter import Model
from scripts.research.pallet_selector_recovery_v1.build_router_dataset import policy

def batch_data(rows):
    a,_=policy();images=[];targets=[];mask=[];tasks=[]
    for r in rows:
        with np.load(C.ROOT/r['cache']['path']) as z:
            im=z['img'].copy();kp=z['keypoints'][0].copy()
        assert C.digest(im)==r['base_RGB_sha256'];assert C.digest(kp[None])==r['target_sha256']
        if r['task']=='SYNTH_OCCLUDED_PRESERVE':im=a['apply'](im,r['plan'],'S1')
        images.append(im);targets.append(kp[:,:2]*640);mask.append(kp[:,2]==2);tasks.append(r['task']=='SYNTH_OCCLUDED_PRESERVE')
    return torch.tensor(np.array(images),device='cuda',dtype=torch.float32)/255,torch.tensor(np.array(targets),device='cuda'),torch.tensor(np.array(mask),device='cuda'),torch.tensor(tasks,device='cuda')

def losses(m,x,target,mask,is_occ):
    with torch.no_grad():base,bk=m.tensor(x,False)
    new,nk=m.tensor(x,True);assert torch.equal(new[:,:6],base[:,:6]) and torch.equal(nk[:,:,2],bk[:,:,2])
    q=nk[:,:,:2];clean_mask=mask[:,:,None].expand_as(q)&(~is_occ[:,None,None])&torch.isfinite(target)&torch.isfinite(q)
    preserve_mask=is_occ[:,None,None].expand_as(q)&torch.isfinite(q)&torch.isfinite(bk[:,:,:2])
    lc=F.smooth_l1_loss(q[clean_mask],target[clean_mask],reduction='sum')/clean_mask.sum().clamp_min(1)
    lp=F.smooth_l1_loss(q[preserve_mask],bk[:,:,:2].detach()[preserve_mask],reduction='sum')/preserve_mask.sum().clamp_min(1)
    return lc,lp,(q-bk[:,:,:2]).detach(),dict(clean_scalars=int(clean_mask.sum()),occ_scalars=int(preserve_mask.sum()))

def main():
    C.setup();gpu=C.gpu();lock=C.read(C.DOC/'OCCURRENCES_LOCK.json');C.verify(lock['occurrences']);rows=C.read(C.ROOT/lock['occurrences']['path']);parity=C.read(C.DOC/'ZERO_INIT_PARITY.json');assert parity['passed'];assert not (C.RAW/'last_adapter.pt').exists()
    m=Model();initial=C.state_hash(m.adapter);opt=torch.optim.AdamW(m.adapter.parameters(),lr=.001,weight_decay=.0001);trace=[];epochs=[];start=time.perf_counter()
    for epoch in range(5):
        el=[]
        for b in range(64):
            rr=rows[(epoch*64+b)*16:(epoch*64+b+1)*16];x,t,mask,is_occ=batch_data(rr);opt.zero_grad(set_to_none=True);lc,lp,res,counts=losses(m,x,t,mask,is_occ);loss=lc+lp
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None for p in m.net.model.parameters());grad=sum(float(p.grad.abs().sum()) for p in m.adapter.parameters() if p.grad is not None);assert grad>0
            opt.step();row=dict(step=len(trace)+1,epoch=epoch+1,L_clean=float(lc.detach()),L_occ_preserve=float(lp.detach()),residual_xy_mean_px=float(res.norm(dim=-1).mean()),adapter_grad_L1=grad,**counts);trace.append(row);el.append(row)
        m.integrity();epochs.append(dict(epoch=epoch+1,L_clean=float(np.mean([r['L_clean'] for r in el])),L_occ_preserve=float(np.mean([r['L_occ_preserve'] for r in el])),residual_xy_mean_px=float(np.mean([r['residual_xy_mean_px'] for r in el])),real_clean=256,synth_clean=256,synth_occ_preserve=512,gpu=C.gpu()))
        print('PRES1_EPOCH',epochs[-1],flush=True)
    m.integrity();assert len(trace)==320;path=C.RAW/'last_adapter.pt';torch.save(dict(adapter=m.adapter.state_dict(),base=m.bindings['checkpoint'],selector=m.bindings['scorer'],steps=320),path)
    C.freeze(C.RAW/'TRACE.json',trace);C.freeze(C.DOC/'FIT.json',dict(created_at=C.now(),complete=True,checkpoint=C.bind(path),trace=C.bind(C.RAW/'TRACE.json'),steps=320,epochs=epochs,seconds=time.perf_counter()-start,
        seed=42,last_only=True,lambda_occ=1.,base_state_before=m.base_hash,base_state_after=C.state_hash(m.net.model),adapter_initial=initial,adapter_final=C.state_hash(m.adapter),
        adapter_params=sum(p.numel() for p in m.adapter.parameters()),base_params=sum(p.numel() for p in m.net.model.parameters()),trainable='adapter_only',base_grad_zero=True,box_class_conf_unchanged_every_step=True,real_DEV_training=False,reference_guard=True,read_paths=sorted(set(READS)),gpu_start=gpu))
    print('PRES1_FIT_COMPLETE',flush=True)
if __name__=='__main__':main()
