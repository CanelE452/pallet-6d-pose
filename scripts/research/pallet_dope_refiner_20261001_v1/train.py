"""Six fixed-budget paired fits; frozen VGG online, no real data or best selection."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from common import *
from data import Dataset,orders
from refiner import GenericPointRefiner,DirectResidualControl,loss,direct_loss

CLASSES={'P':GenericPointRefiner,'D':DirectResidualControl}
LOSSES={'P':loss,'D':direct_loss}
KEYS=('p3','p4','points','boxes','point_valid','input_shape')

def forward(head,batch):return head(*(batch[k] for k in KEYS),lam=0)

def train(adapter):
    protocol=verify_lock();assert read(DOC/'GPU_SMOKE.json')['PASS']
    data=Dataset();orders(data);o=protocol['optimizer'];done=[]
    for seed in SEEDS:
        directory=RAW/'runs'/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
        completion=DOC/f'TRAIN_SEED{seed}.json';path=directory/'paired_last.pt'
        if completion.exists():
            c=read(completion);assert c['complete'] and sha(path)==c['checkpoint']['sha256'];done.append(c);continue
        models={};optimizers={}
        for arm in ('P','D'):
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            models[arm]=CLASSES[arm](**CONFIG).to(adapter.device).train()
            optimizers[arm]=torch.optim.AdamW(models[arm].parameters(),lr=o['lr'],betas=tuple(o['betas']),weight_decay=o['weight_decay'])
        order=np.load(RAW/f'order_seed{seed}.npy');order_sha=sha(RAW/f'order_seed{seed}.npy')
        start=0;metrics=[];previous_elapsed=0.;resumes=[]
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['protocol_sha256']==sha(DOC/'PROTOCOL.json') and state['order_sha256']==order_sha
            for arm in models:
                models[arm].load_state_dict(state['models'][arm]);optimizers[arm].load_state_dict(state['optimizers'][arm])
            torch.set_rng_state(state['torch_rng_state']);torch.cuda.set_rng_state_all(state['cuda_rng_state'])
            start=state['step'];metrics=state['metrics'];previous_elapsed=state['elapsed_seconds']
            resumes=state['resumes']+[dict(time=now(),from_step=start)]
        begin=time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            future=prefetch.submit(data.prepare,order[start],adapter) if start<6000 else None
            for step in range(start+1,6001):
                prepared=future.result()
                if step<6000:future=prefetch.submit(data.prepare,order[step],adapter)
                batch=data.batch_features(order[step-1],adapter,prepared);lr=learning_rate(step,protocol)
                update=dict(step=step,lr=lr,arms={})
                for arm in ('P','D'):
                    model=models[arm];optimizer=optimizers[arm];optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:group['lr']=lr
                    out=forward(model,batch);value=LOSSES[arm](out,batch['gt_points'],batch['gt_valid'])
                    assert torch.isfinite(value),(seed,arm,step)
                    assert out['point_support'].any(),('All unsupported batch',seed,step)
                    value.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
                    if step==1:
                        gradients={k:float(p.grad.norm()) for k,p in model.named_parameters()}
                        assert gradients['adapt3.0.weight']>0 and gradients['adapt4.0.weight']>0
                        write(DOC/f'GRADIENT_{arm}{seed}.json',dict(gradients=gradients,frozen_backbone_grads_none=all(p.grad is None for p in adapter.network.parameters())))
                    before=torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
                    optimizer.step();after=torch.nn.utils.parameters_to_vector(model.parameters()).detach()
                    change=float((after-before).norm());assert torch.isfinite(after).all() and change>0
                    update['arms'][arm]=dict(loss=float(value.detach()),preclip_grad_norm=float(norm),parameter_update_norm=change,
                      supported_frames=int((out['point_support']&batch['gt_valid'][:,:8]).any(-1).sum()),
                      supported_corners=int((out['point_support']&batch['gt_valid'][:,:8]).sum()))
                    del out,value,before,after
                assert all(p.grad is None and not p.requires_grad for p in adapter.network.parameters())
                metrics.append(update)
                elapsed=previous_elapsed+time.perf_counter()-begin
                state=dict(complete=step==6000,step=step,seed=seed,config=CONFIG,
                    models={a:m.state_dict() for a,m in models.items()},optimizers={a:x.state_dict() for a,x in optimizers.items()},
                    torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                    protocol_sha256=sha(DOC/'PROTOCOL.json'),order_sha256=order_sha,baseline_sha256=WEIGHT_SHA,
                    metrics=metrics,elapsed_seconds=elapsed,resumes=resumes)
                pending=path.with_name('paired_last.pending.pt');torch.save(state,pending);pending.replace(path)
                if step==1 or step%100==0:
                    row=dict(seed=seed,**update,elapsed_seconds=elapsed,gpu=gpu(),checkpoint_step=step)
                    write(RAW/'TRAIN_PROGRESS.json',row,freeze=False)
                    print('DOPE_HEAD_TRAIN',seed,step,'P',round(update['arms']['P']['loss'],5),'D',round(update['arms']['D']['loss'],5),'sec',round(elapsed,1),flush=True)
                del batch,prepared
        c=dict(complete=True,seed=seed,updates_per_arm=6000,exposures_per_arm=96000,arms=['P','D'],
            usable_training_rows=len(data.train_rows),checkpoint=bound(path),order=bound(RAW/f'order_seed{seed}.npy'),
            parameters={a:sum(p.numel() for p in m.parameters()) for a,m in models.items()},
            elapsed_seconds=state['elapsed_seconds'],resumes=resumes,real_training=0,backbone_retrained=False,
            final_checkpoint_only=True,finite_all_updates=True,paired_shared_frozen_features=True)
        write(directory/'STEP_METRICS.json',metrics);write(completion,c);done.append(c)
        del models,optimizers,state;torch.cuda.empty_cache()
    data.close();assert sha(WEIGHTS)==WEIGHT_SHA
    write(DOC/'TRAINING_COMPLETE.json',dict(complete=True,fits=6,steps_per_fit=6000,total_head_updates=36000,
        total_exposures=576000,seeds=done,baseline_sha256=WEIGHT_SHA))
