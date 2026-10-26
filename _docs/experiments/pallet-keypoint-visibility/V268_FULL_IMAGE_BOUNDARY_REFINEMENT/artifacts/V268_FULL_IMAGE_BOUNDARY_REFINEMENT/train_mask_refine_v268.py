"""Paired fixed1000 SYN-only full-frame refiners; run only after Root seal."""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import time
import types
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).absolute().parent


def safe(path):
    p = Path(path); assert p.is_absolute() and not any(s in str(p).lower() for s in ('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def write_json(path, obj):
    with safe(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    started=time.monotonic(); pp=safe(HERE/'PROTOCOL_BEFORE_FIT.json'); p=json.loads(pp.read_bytes())
    bound=p['model_code']; path=safe(bound['path']); raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==bound['sha256'] and len(raw)==bound['size_bytes']
    m=types.ModuleType('v268_sealed_refiner');m.__file__=str(path);sys.modules[m.__name__]=m;exec(compile(raw,str(path),'exec'),m.__dict__)
    assert m.binding(Path(__file__).absolute())==p['training_code']
    assert tuple(p['train_arms'])==m.ARMS and p['seed']==17 and p['optimizer_updates_each']==1000 and p['batch_size']==8
    assert p['optimizer']==dict(name='AdamW',lr=.001,weight_decay=.0001)
    assert p['boundary_radius']==2 and p['boundary_loss_weight']==.5 and p['delta_bound']==6. and p['threshold']==.5
    b=p['source_bindings']
    for source in b.values():m.checked(source)
    rgb=np.load(m.checked(b['RGB256']),mmap_mode='r',allow_pickle=False)
    tokens=np.load(m.checked(b['tokensCS']),mmap_mode='r',allow_pickle=False)
    coarse=np.load(m.checked(b['coarse_logits']),allow_pickle=False)
    with np.load(io.BytesIO(m.checked(b['targets']).read_bytes()),allow_pickle=False) as z:
        target=np.array(z['mask'],copy=True);band=np.array(z['boundary_band'],copy=True);fit=np.array(z['fit'],copy=True).astype(bool);ids=np.array(z['frame_ids'],copy=True)
    draws=np.load(m.checked(b['draws']),allow_pickle=False)
    assert rgb.shape==(1003,3,256,256) and rgb.dtype==np.float16 and tokens.shape==(1003,1024,512)
    assert coarse.shape==(1003,1,32,32) and target.shape==band.shape==(1003,1,256,256) and band.dtype==bool
    assert fit.sum()==768 and (~fit).sum()==235 and np.array_equal(ids,np.arange(1003))
    assert np.isfinite(target).all() and target.min()>=0 and target.max()<=1
    expected=np.random.Generator(np.random.PCG64(17)).choice(np.flatnonzero(fit),size=(1000,8),replace=True)
    assert np.array_equal(draws,expected) and fit[draws].all()
    frequency=float(target[fit].astype(np.float64).mean());wpos=.5/frequency;wneg=.5/(1-frequency)
    assert frequency==p['FIT_foreground_frequency'] and abs(wpos-p['positive_weight'])<1e-10 and abs(wneg-p['negative_weight'])<1e-10
    out=m.safe(HERE/'syn_training_v1');assert not out.exists();out.mkdir()
    torch.set_num_threads(1);torch.manual_seed(17);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available();device='cuda:0';proofs={};trained={}
    draw_sha=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
    for arm in m.ARMS:
        torch.manual_seed(17);model=m.FullFrameRefiner().to(device);initial=m.state_sha(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        boundary_weight=.5 if arm=='BOUNDARY_REFINE' else 0.
        logpath=m.safe(out/(arm+'_TRAIN1000.jsonl'));model.train()
        with logpath.open('x') as log:
            for step,ix in enumerate(draws,1):
                arrays=[torch.as_tensor(np.array(a[ix],copy=True),dtype=torch.float32,device=device) for a in (rgb,tokens,coarse)]
                y=torch.as_tensor(np.array(target[ix],copy=True),dtype=torch.float32,device=device)
                bb=torch.as_tensor(np.array(band[ix],copy=True),dtype=torch.bool,device=device)
                optimizer.zero_grad(set_to_none=True);result=model(*arrays)
                loss,parts=m.refine_loss(result['mask_logits'],y,bb,wpos,wneg,boundary_weight)
                assert bool(torch.isfinite(loss));loss.backward();assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in model.parameters())
                optimizer.step();row=dict(step=step,loss=float(loss.detach()),boundary_weight=boundary_weight,**{k:float(v.detach()) for k,v in parts.items()})
                log.write(json.dumps(row,sort_keys=True)+'\n')
                if step%100==0:log.flush();print(json.dumps(dict(arm=arm,**row)),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()};cp=m.safe(out/(arm+'_FINAL1000.pt'))
        torch.save(dict(state_dict=state,arm=arm,steps=1000,model_code_sha256=bound['sha256'],threshold=.5,delta_bound=6.,target='Full original SYN M4 visible target, not native E'),cp)
        proofs[arm]=dict(initial_state_sha256=initial,final_state_sha256=m.state_sha(state),parameters=sum(v.numel() for v in model.parameters()),steps=1000,draws_sha256=draw_sha,checkpoint=m.binding(cp),log=m.binding(logpath),boundary_weight=boundary_weight)
        trained[arm]=model.eval()
    assert len({r['initial_state_sha256'] for r in proofs.values()})==len({r['parameters'] for r in proofs.values()})==1
    write_json(out/'BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json',dict(completed=True,arms=proofs,SYN_DEV_refiner_forwards=0,current_or_real_reads=0,encoder_or_coarse_head_updates=0))
    dev=np.flatnonzero(~fit);reports={}
    for arm in m.ARMS:
        pred=m.infer_refiner(trained[arm],rgb,tokens,coarse,dev,device=device,batch_size=8)
        np.savez_compressed(m.safe(out/(arm+'_SYN_DEV_MASKS.npz')),frame_ids=ids[dev],**pred)
        reports[arm]=m.mask_summary(pred['mask_logits'],target[dev])
    write_json(out/'SYN_DEV_RESULTS.json',dict(completed=True,fixed_final1000=True,no_checkpoint_threshold_or_recipe_selection=True,results=reports))
    names=['BOTH_CHECKPOINTS_FROZEN_BEFORE_SYN_DEV.json','SYN_DEV_RESULTS.json']+[a+s for a in m.ARMS for s in ('_FINAL1000.pt','_TRAIN1000.jsonl','_SYN_DEV_MASKS.npz')]
    write_json(out/'COMPLETE.json',dict(completed=True,PASS=True,protocol=m.binding(pp),code=m.binding(Path(__file__).absolute()),outputs={n:m.binding(out/n) for n in names},arms=proofs,new_optimizer_updates=2000,current_or_real_fit_reads=0,encoder_or_coarse_head_updates=0,FIT_foreground_frequency=frequency,positive_weight=wpos,negative_weight=wneg,SYN_DEV_results=reports,elapsed_seconds=time.monotonic()-started,runtime=dict(torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(0))))
    write_json(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json',dict(completed=True,PASS=True,protocol=m.binding(pp),model_code=bound,training_complete=m.binding(out/'COMPLETE.json'),checkpoints={a:proofs[a]['checkpoint'] for a in m.ARMS},threshold=.5,delta_bound=6.,full_frame_scope=True,current_or_real_fit_or_selection=0,no_extra_automatic_fits=True,target='M4 target/non-target, no external-cause class'))
    print(json.dumps(dict(completed=True,PASS=True,results=reports,freeze=m.binding(HERE/'NEW_CONFIG_FROZEN_BEFORE_CURRENT.json'))),flush=True)


if __name__=='__main__':main()
