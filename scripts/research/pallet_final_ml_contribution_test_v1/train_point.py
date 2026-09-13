"""Exactly three frozen-budget fits; no architecture or optimization sweep."""
import time
import numpy as np
import torch
from common import *
from generic_point_refiner import GenericPointRefiner,loss
from preflight import dataset

def forward(model,batch):return model(*(batch[k] for k in ('p3','p4','points','boxes','point_valid','input_shape')),lam=0)

def run():
    assert read(B/'REGRESSION_TESTS.json')['PASS']
    lock=read(B/'B_IMPLEMENTATION_LOCK.json')
    for p,h in lock['paths'].items():assert sha(ROOT/p)==h
    write(B/'GPU_TRAIN_START.json',gpu())
    assert torch.cuda.is_available()
    config=read(B/'PARAMETER_BUDGET_LOCK.json')['config'];protocol=read(LINE/'TRAIN_PROTOCOL.json')
    data=dataset();legacy=old('train');completed=[]
    for seed in (1,2,3):
        destination=BRAW/f'runs/seed{seed}';destination.mkdir(parents=True,exist_ok=True)
        completion=B/f'P_SEED{seed}_COMPLETION.json'
        if completion.exists():
            c=read(completion);assert c['complete'] and sha(destination/'last.pt')==c['checkpoint_sha256'];completed.append(c);continue
        assert not (destination/'last.pt').exists(),'No silent retry/resume: inspect previous exact state first'
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=GenericPointRefiner(**config).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['optimizer']['lr'],
            betas=tuple(protocol['optimizer']['betas']),weight_decay=protocol['optimizer']['weight_decay'])
        order=np.load(BRAW/f'order_seed{seed}.npy');trace=hashlib.sha256();start=time.perf_counter();history=[]
        assert order.shape==(6000,16)
        for step,rows in enumerate(order,1):
            trace.update(rows.astype('<i8').tobytes());batch=data.batch(rows)
            optimizer.zero_grad(set_to_none=True);lr=legacy.learning_rate(step,protocol)
            for group in optimizer.param_groups:group['lr']=lr
            output=forward(model,batch);value=loss(output,batch['gt_points'],batch['gt_valid'])
            assert torch.isfinite(value),('Nonfinite loss',seed,step)
            value.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['optimizer']['gradient_clip_norm'],error_if_nonfinite=True)
            if step==1:
                gradients={k:float(p.grad.norm()) for k,p in model.named_parameters()}
                assert gradients['adapt3.0.weight']>0 and gradients['adapt4.0.weight']>0
                write(B/f'P_SEED{seed}_GRADIENT.json',dict(step=1,gradients=gradients,R0_not_in_graph=True,optimizer_P_only=True))
            optimizer.step()
            if step==1 or step%100==0:
                torch.cuda.synchronize();snapshot=gpu()
                row=dict(step=step,loss=float(value.detach()),lr=lr,gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-start,gpu=snapshot)
                history.append(row)
                with (destination/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print('P_TRAIN',seed,step,'loss',round(row['loss'],5),'seconds',round(row['elapsed_seconds'],1),'GPU',snapshot['gpu'],flush=True)
            if step%500==0:
                payload=dict(complete=step==6000,step=step,seed=seed,config=config,model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),
                    batch_trace_sha256=trace.hexdigest(),baseline_checkpoint_sha256=R0_SHA,implementation_lock_sha256=sha(B/'B_IMPLEMENTATION_LOCK.json'))
                temp=destination/'last.pending.pt';torch.save(payload,temp);temp.replace(destination/'last.pt')
        expected=read(B/'INIT_ORDER_PARITY.json')['seeds'][str(seed)]['batch_order_sha256']
        assert trace.hexdigest()==expected and sha(R0)==R0_SHA
        assert all(torch.isfinite(p).all() for p in model.parameters())
        c=dict(complete=True,seed=seed,step=6000,updates=6000,exposures=96000,unique_training_rows=len(data.train_rows),
            checkpoint_sha256=sha(destination/'last.pt'),batch_order_sha256=trace.hexdigest(),old_L_order_exact=True,
            elapsed_seconds=time.perf_counter()-start,AMP=False,finite=True,config=config,history=history,
            baseline_checkpoint_sha256=sha(R0),validation_best=False,real_train_frames=0)
        write(completion,c);completed.append(c)
        del model,optimizer,output,value,batch;torch.cuda.empty_cache()
    write(B/'P_TRAINING_COMPLETE.json',dict(complete=True,fits=3,total_optimizer_updates=18000,seeds=completed))
    print('THREE_P_FITS_COMPLETE',flush=True)

if __name__=='__main__':run()
