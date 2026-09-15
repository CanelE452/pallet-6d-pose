"""Matched exposure training; full optimizer/RNG/order resume, never best-on-DEV."""
import random,time,gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from env import *
from prior_model import PoseFixPallet9,TFAdam
from prior_data import SourceRGB
KEYS=('rgb','points','valid','target','target_valid')
def batch(items,device='cuda'):
    return {k:torch.as_tensor(np.stack([r[k] for r in items]),device=device) for k in KEYS}
@torch.no_grad()
def probe(model,src,rows,micro):
    model.eval();values=[]
    for i in range(0,len(rows),micro):
        b=src.batch(rows[i:i+micro]);loss,parts=model.losses(model(b['rgb'],b['points'],b['valid']),b['target'],b['target_valid'])
        values.append({k:float(v) for k,v in parts.items()})
    model.train();return {k:float(np.mean([v[k] for v in values])) for k in values[0]}
def run():
    assert complete('IMPLEMENTATION_COMPLETE'),'Actual network/operator/resource gates incomplete'
    from gpu_parity import run as check_gpu
    check_gpu()
    if complete('TRAIN_COMPLETE'):verify();print('PRIOR_TRAIN_REUSED');return
    start=now();verify();assert not gpu()['foreign_compute'];assert torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    src=SourceRGB();micro=read(DOC/'RESOURCE_AMENDMENT.json')['microbatch'];dlock=read(OLD_DOC/'D_PROTOCOL_LOCK.json')
    lock=dict(protocol=bound(DOC/'PRIOR_PROTOCOL_LOCK.json'),resource=bound(DOC/'RESOURCE_AMENDMENT.json'),order_files={str(s):bound(BRAW/f'order_seed{s}.npy') for s in (1,2,3)},code={p.name:sha(p) for p in [HERE/'prior_model.py',HERE/'prior_data.py',HERE/'train_prior.py']},initialization={str(s):bound(RAW/f'tf_initial_seed{s}.npz') for s in (1,2,3)},probes={k:dlock[k+'_probe_rows'] for k in ('train','cal')},checkpoint_every_updates=50)
    freeze(DOC/'TRAIN_CODE_LOCK.json',lock);lockhash=sha(DOC/'TRAIN_CODE_LOCK.json');runs=[]
    for seed in (1,2,3):
        path=RAW/f'runs/PRIOR{seed}';path.mkdir(parents=True,exist_ok=True);done=DOC/f'PRIOR{seed}_COMPLETE.json';ckpath=path/'last.pt'
        if done.exists():
            r=read(done);assert r['updates']==6000 and sha(ckpath)==r['checkpoint_sha256'];runs.append(r);continue
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=PoseFixPallet9();model.load_tf(RAW/f'tf_initial_seed{seed}.npz');model.cuda().train();optimizer=TFAdam(model.parameters())
        order=np.load(BRAW/f'order_seed{seed}.npy');assert order.shape==(6000,16) and np.isin(order,src.data.train_rows).all()
        updates=[];probes=[];step0=0;resumes=0;elapsed=0.
        if ckpath.exists():
            ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['lock_sha256']==lockhash
            model.load_state_dict(ck['model_state_dict']);optimizer.load_state_dict(ck['optimizer_state_dict']);step0=ck['step'];updates=ck['updates'];probes=ck['probes'];resumes=ck['resumes']+1;elapsed=ck['elapsed_seconds']
            assert ck['order_prefix_sha256']==hashlib.sha256(order[:step0].astype('<i8').tobytes()).hexdigest()
            random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);del ck
        else:probes.append(dict(step=0,**{k:probe(model,src,v,micro) for k,v in lock['probes'].items()}))
        torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending=[pool.submit(src.item,int(r)) for r in order[step0]] if step0<6000 else []
            for step in range(step0+1,6001):
                items=[f.result() for f in pending]
                pending=[pool.submit(src.item,int(r)) for r in order[step]] if step<6000 else []
                assert all(x['source_partition']=='train' for x in items)
                lr=.0005*(.1**sum(step>=i for i in (3858,5143)))
                for g in optimizer.param_groups:g['lr']=lr
                optimizer.zero_grad(set_to_none=True);components=[]
                for j in range(0,16,micro):
                    b=batch(items[j:j+micro]);out=model(b['rgb'],b['points'],b['valid']);loss,parts=model.losses(out,b['target'],b['target_valid'])
                    assert torch.isfinite(loss);(loss*(micro/16)).backward();components.append({k:float(v.detach()) for k,v in parts.items()});del b,out,loss,parts
                grad=float(torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt());assert np.isfinite(grad)
                change=float(optimizer.step());assert np.isfinite(change) and change>0
                rec=dict(step=step,lr=lr,gradient_norm=grad,actual_update_norm=change,components={k:float(np.mean([v[k] for v in components])) for k in components[0]},supervised_crop_corners=sum(int(x['target_valid'].sum()) for x in items),source_corners=sum(int(x['original_gt_valid'].sum()) for x in items))
                updates.append(rec)
                if step%500==0:probes.append(dict(step=step,**{k:probe(model,src,v,micro) for k,v in lock['probes'].items()}))
                if step==1 or step%50==0:
                    state=dict(complete=step==6000,seed=seed,step=step,lock_sha256=lockhash,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),data_loader_rng='No augmentation/random worker state; exact deterministic row tasks; pending images can be reloaded',order_prefix_sha256=hashlib.sha256(order[:step].astype('<i8').tobytes()).hexdigest(),updates=updates,probes=probes,resumes=resumes,elapsed_seconds=elapsed+time.perf_counter()-begin)
                    temp=path/'last.pending.pt';torch.save(state,temp);temp.replace(ckpath)
                    status=gpu();write(path/'PROGRESS.json',dict(seed=seed,**rec,elapsed_seconds=state['elapsed_seconds'],checkpoint_step=step,gpu=status));print('PRIOR_TRAIN',seed,step,rec['components'],round(state['elapsed_seconds'],1),flush=True)
                    assert not status['foreign_compute'],status
        r=dict(complete=True,seed=seed,updates=6000,exposures=96000,checkpoint_sha256=sha(ckpath),initial_sha256=sha(RAW/f'tf_initial_seed{seed}.npz'),order_file=bound(BRAW/f'order_seed{seed}.npy'),resumes=resumes,elapsed_seconds=state['elapsed_seconds'],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,parameters=sum(p.numel() for p in model.parameters()),probes=probes,clip='NOT_USED',microbatch=micro,real_training=0,last_only=True,minimum_actual_update_norm=min(x['actual_update_norm'] for x in updates))
        write(path/'STEP_METRICS.json',updates);write(done,r);runs.append(r)
        del model,optimizer,state;gc.collect();torch.cuda.empty_cache()
    write(DOC/'TRAINING_AUDIT.json',dict(complete=True,runs=runs,updates=18000,smoke_updates=read(DOC/'SMOKE_COMPLETE.json')['disposable_optimizer_updates'],R0_P_D_L_retraining=0,convergence='fixed exposure budget; not140epochs and not converged-method superiority'))
    receipt('TRAIN_COMPLETE',[DOC/'TRAIN_CODE_LOCK.json'],[DOC/'TRAINING_AUDIT.json']+[RAW/f'runs/PRIOR{s}/last.pt' for s in (1,2,3)],start)
