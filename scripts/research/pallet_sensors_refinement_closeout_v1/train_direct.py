"""Three fresh direct heads, exact P order/protocol, durable per-update state."""
import time
import numpy as np
import torch
from env import *
from direct_residual_control import DirectResidualControl,direct_loss
def lock(data):
    p=read(LINE/'TRAIN_PROTOCOL.json');config=read(B/'PARAMETER_BUDGET_LOCK.json')['config']
    rules=[dict(lam=l,max_move_image_diagonal_fraction=c) for l in p['selection']['lambda_grid'] for c in p['selection']['max_move_image_diagonal_fractions']]
    value=dict(seeds=[1,2,3],steps=6000,batch=16,optimizer=p['optimizer'],config=config,loss='normalized residual L1 with per-frame supported point averaging',
        real_training=0,P_retrained=False,R0_retrained=False,L_retrained=False,smoke_max=32,checkpoint='last6000 only',source_protocol_sha256=sha(LINE/'TRAIN_PROTOCOL.json'),
        order_files={str(s):bound(BRAW/f'order_seed{s}.npy') for s in (1,2,3)},train_rows_sha256=hashlib.sha256(data.train_rows.astype('<i8').tobytes()).hexdigest(),
        train_probe_rows=data.train_rows[:32].tolist(),cal_probe_rows=np.flatnonzero(data.partitions=='calibration')[:32].tolist(),probe_steps=[0,1]+list(range(500,6001,500)),
        code={p.name:sha(p) for p in [HERE/'direct_residual_control.py',HERE/'train_direct.py']},selection_rules=rules,temperature='none',selection_objective=p['selection'],
        output_difference='unbounded raw regression versus P candidate convex hull; same final cap grid',resume='durable optimizer+CPU/CUDA RNG+step+order prefix every update',nondeterminism='CUDA grid_sample backward not guaranteed bit deterministic')
    freeze(DOC/'D_PROTOCOL_LOCK.json',value);return value,p
@torch.no_grad()
def probe(model,data,lock):
    model.eval();r={}
    for name in ('train','cal'):
        bs=data.batch(lock[f'{name}_probe_rows']);out=fwd(model,bs)
        r[name]=float(direct_loss(out,bs['gt_points'],bs['gt_valid']))
    model.train();return r
def run():
    verify();assert read(DOC/'REGRESSION_TESTS.json')['PASS'];assert not gpu()['foreign_compute']
    data=dataset();lk,protocol=lock(data);completions=[]
    for seed in (1,2,3):
        dst=RAW/f'runs/D{seed}';dst.mkdir(parents=True,exist_ok=True);done=DOC/f'D{seed}_COMPLETION.json'
        if done.exists():
            c=read(done);assert c['complete'] and sha(dst/'last.pt')==c['checkpoint_sha256'];completions.append(c);continue
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=DirectResidualControl(**lk['config']).cuda();o=lk['optimizer']
        optimizer=torch.optim.AdamW(model.parameters(),lr=o['lr'],betas=tuple(o['betas']),weight_decay=o['weight_decay'])
        order=np.load(BRAW/f'order_seed{seed}.npy');assert order.shape==(6000,16) and np.isin(order,data.train_rows).all()
        start=0;history=[];probes=[];resumes=0;elapsed=0.;clipped=0;updates=[]
        if (dst/'last.pt').exists():
            ck=torch.load(dst/'last.pt',map_location='cpu',weights_only=False)
            assert ck['protocol_sha256']==sha(DOC/'D_PROTOCOL_LOCK.json')
            model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict'])
            torch.set_rng_state(ck['torch_rng_state']);torch.cuda.set_rng_state_all(ck['cuda_rng_state'])
            start=ck['step'];history=ck['history'];probes=ck['probes'];resumes=ck['resumes']+1;elapsed=ck['elapsed_seconds'];clipped=ck['clipped'];updates=ck['step_metrics']
        else:probes.append(dict(step=0,**probe(model,data,lk)))
        begin=time.perf_counter()
        for step in range(start+1,6001):
            batch=data.batch(order[step-1]);optimizer.zero_grad(set_to_none=True)
            lr=old('train').learning_rate(step,protocol)
            for group in optimizer.param_groups:group['lr']=lr
            value=direct_loss(fwd(model,batch),batch['gt_points'],batch['gt_valid']);assert torch.isfinite(value)
            value.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),o['gradient_clip_norm'],error_if_nonfinite=True)
            clipped+=int(norm>o['gradient_clip_norm'])
            before=torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone();optimizer.step()
            after=torch.nn.utils.parameters_to_vector(model.parameters()).detach();change=float((after-before).norm());assert torch.isfinite(after).all() and change>0
            updates.append(dict(step=step,loss=float(value.detach()),lr=lr,preclip_norm=float(norm),post_step_parameter_norm=change))
            if step in lk['probe_steps']:probes.append(dict(step=step,**probe(model,data,lk)))
            if step==1 or step%100==0:
                status=gpu();assert not status['foreign_compute'],status
                row=dict(**updates[-1],elapsed_seconds=elapsed+time.perf_counter()-begin,gpu=status);history.append(row)
                write(dst/'PROGRESS.json',row);print('D_TRAIN',seed,step,round(row['loss'],6),round(row['elapsed_seconds'],1),flush=True)
            state=dict(complete=step==6000,step=step,seed=seed,config=lk['config'],model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),protocol_sha256=sha(DOC/'D_PROTOCOL_LOCK.json'),
                baseline_checkpoint_sha256=C.R0_SHA,history=history,probes=probes,resumes=resumes,elapsed_seconds=elapsed+time.perf_counter()-begin,clipped=clipped,step_metrics=updates)
            pending=dst/'last.pending.pt';torch.save(state,pending);pending.replace(dst/'last.pt')
        c=dict(complete=True,updates=6000,seed=seed,exposures=96000,checkpoint_sha256=sha(dst/'last.pt'),batch_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            resumes=resumes,elapsed_seconds=state['elapsed_seconds'],clipped_steps=clipped,probes=probes,history=history,real_training=0,last_only=True,parameters=19450,
            post_step_parameter_norm_min=min(r['post_step_parameter_norm'] for r in updates),finite_all=True)
        write(dst/'STEP_METRICS.json',updates);write(done,c);completions.append(c)
        del model,optimizer,batch,value,state;torch.cuda.empty_cache()
    write(DOC/'D_TRAINING_AUDIT.json',dict(complete=True,fits=3,actual_main_updates=18000,smoke_updates=read(DOC/'REGRESSION_TESTS.json')['smoke_updates'],runs=completions,
        same_P_order=True,loss_scale_not_compared_with_CE=True,no_outcome_based_restarts=True))
    print('D_THREE_FITS_COMPLETE',flush=True)
if __name__=='__main__':run()
