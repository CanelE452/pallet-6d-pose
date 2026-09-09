"""Matched source-only local refinement training; point-coordinate loss only."""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib,json,math,time
from datetime import datetime,timezone
import numpy as np
import torch
import torch.nn.functional as F
from scripts.research.pallet_dht_structured_v2.cache import read,write,sha
from scripts.research.pallet_dht_structured_v2.train import tensor_sha,save_torch
from .cache import LocalData
from .model import LocalHoughPointRefiner


def verify(run):
    run=Path(run).resolve();protocol=read(run/'PROTOCOL.json');freeze=read(run/'SOURCE_FREEZE.json')
    if not freeze['complete'] or freeze['protocol_sha256']!=sha(run/'PROTOCOL.json'):raise ValueError('Frozen local protocol required')
    for key,name in [('implementation_spec_sha256','IMPLEMENTATION_SPEC.json'),('cache_completion_sha256','CACHE_COMPLETION.json')]:
        if freeze[key]!=sha(run/name):raise ValueError(f'Frozen local binding changed: {name}')
    for path,expected in freeze['source_sha256'].items():
        if sha(path)!=expected:raise ValueError(f'Frozen local source changed: {path}')
    return protocol,dict(protocol_sha256=sha(run/'PROTOCOL.json'),source_freeze_sha256=sha(run/'SOURCE_FREEZE.json'),
        source_sha256=freeze['source_sha256'],implementation_spec_sha256=sha(run/'IMPLEMENTATION_SPEC.json'),cache_completion_sha256=sha(run/'CACHE_COMPLETION.json'))


def plans(indices,cfg,seed):
    n=cfg['steps']*cfg['batch'];rng=np.random.default_rng(seed);stream=[]
    while len(stream)<n:stream.extend(rng.permutation(indices).tolist())
    order=np.array(stream[:n],np.int64).reshape(cfg['steps'],cfg['batch'])
    rng=np.random.default_rng(seed+1731)
    noisy=rng.random(n)>=cfg['clean_probability']
    jitter=np.clip(rng.normal(0,cfg['jitter_input_sigma_px'],(n,8,2)),-cfg['jitter_clip_input_px'],cfg['jitter_clip_input_px'])
    jitter*=noisy[:,None,None]
    return order,jitter.astype(np.float32).reshape(cfg['steps'],cfg['batch'],8,2)


def augment(inputs,jitter):
    batch=dict(inputs);points=inputs['baseline_points'].clone();gain=inputs['raw_to_input_affine'][:,0,0]
    if not torch.all(gain>0):raise ValueError('Positive original affine scale required')
    delta=torch.as_tensor(jitter,device=points.device,dtype=points.dtype)/gain[:,None,None]
    points[:,:8]+=delta*inputs['point_valid'][:,:8,None]
    batch['baseline_points']=points
    return batch


def point_loss(predicted,targets,batch):
    # The same isotropic affine is applied to predicted and target coordinates;
    # translation cancels, leaving actual input-pixel residuals.
    residual=(predicted[:,:8]-targets['points'][:,:8])*batch['raw_to_input_affine'][:,None,0,0,None]
    mask=targets['loss_valid'][:,:8].bool()
    if not bool(mask.any()):raise ValueError('Empty supervised optimizer batch')
    per=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=1.,reduction='none').mean(-1)
    return per[mask].mean(),int(mask.sum())


def train_arm(run,arm,seed=1,device='cuda:0'):
    run=Path(run).resolve();protocol,bindings=verify(run);cfg=protocol['training']
    if arm not in protocol['model']['arms'] or seed not in cfg['seeds']:raise ValueError('Unregistered local arm/seed')
    cell=run/'runs'/f'{arm}_seed{seed}';marker=cell/'COMPLETION.json'
    if marker.exists():
        old=read(marker)
        if not old['complete'] or not old['PASS'] or old['bindings']!=bindings or sha(old['checkpoint'])!=old['checkpoint_sha256']:
            raise ValueError('Existing local completed cell differs')
        return old
    if cell.exists() and any(cell.iterdir()):raise ValueError('Preserve interrupted cell; no silent overwrite')
    cell.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    data=LocalData(run);indices=data.populations['train'];order,jitter=plans(indices,cfg,seed)
    if set(order.ravel())&set(sum([data.populations[k] for k in ['calibration','synth_val','real_dev']],[])):raise ValueError('Training population leak')
    model=LocalHoughPointRefiner(arm=arm).to(device);model.train()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initialsha=tensor_sha(initial)
    save_torch(cell/'INITIAL_STATE.pth',initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    setup=dict(complete=True,arm=arm,seed=seed,bindings=bindings,parameters=sum(p.numel() for p in model.parameters()),
        model_config=model.model_config,inactive_parameter_prefixes=list(model.inactive_parameter_prefixes),
        initial_state_tensor_sha256=initialsha,order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),jitter_sha256=hashlib.sha256(jitter.tobytes()).hexdigest(),
        train_frames=len(indices),real_GT_used=False,new_backbone_forwards=0,expected_optimizer_steps=cfg['steps'])
    write(cell/'SETUP.json',setup)
    history=[];probes=[];window=[];start=time.perf_counter();totalcorners=0
    for step,ix in enumerate(order,1):
        inputs,targets=data.batch(ix,device);batch=augment(inputs,jitter[step-1])
        optimizer.zero_grad(set_to_none=True);predicted,diag=model(batch)
        if not torch.isfinite(predicted).all():raise ValueError('Nonfinite local prediction')
        if step==1 and not torch.equal(predicted,batch['baseline_points']):raise ValueError('Local model is not exact identity at initialization')
        if not torch.equal(predicted[:,8],batch['baseline_points'][:,8]):raise ValueError('Centroid changed')
        loss,count=point_loss(predicted,targets,batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'],error_if_nonfinite=True)
        lr=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*(step-1)/max(1,cfg['steps']-1))))
        for group in optimizer.param_groups:group['lr']=lr
        if step in [1,2,4,16,100,500,1000]:
            probes.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),
                gradient_l2={name:float(p.grad.detach().norm()) if p.grad is not None else None for name,p in model.named_parameters()},
                max_shift_input_px=float(((predicted[:,:8]-batch['baseline_points'][:,:8])*batch['raw_to_input_affine'][:,None,0,0,None]).norm(dim=-1).max())))
        optimizer.step();totalcorners+=count;window.append(float(loss.detach()))
        with (cell/'TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,indices=ix.tolist(),jitter_input_px=jitter[step-1].tolist(),supervised_corners=count),separators=(',',':'))+'\n')
        if step%cfg['log_every']==0 or step==cfg['steps']:
            row=dict(step=step,mean_loss=float(np.mean(window)),elapsed_seconds=time.perf_counter()-start,lr=lr);history.append(row);window=[]
            write(cell/'history.json',history);print(f'Local {arm} seed{seed} {step}/{cfg["steps"]} loss={row["mean_loss"]:.6f}',flush=True)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    if verify(run)[1]!=bindings:raise ValueError('Local source changed during training')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite final model state')
    saved=dict(schema='pallet_dht_local_v3_checkpoint',complete=True,arm=arm,seed=seed,optimizer_steps=cfg['steps'],
        model_config=model.model_config,state_dict=state,optimizer=optimizer.state_dict(),bindings=bindings,
        initial_state_tensor_sha256=initialsha,final_state_tensor_sha256=tensor_sha(state))
    checkpoint=cell/'checkpoint_final.pth';save_torch(checkpoint,saved)
    write(cell/'GRADIENT_PROBES.json',dict(complete=True,records=probes,point_loss_only=True))
    result=dict(complete=True,PASS=True,arm=arm,seed=seed,bindings=bindings,optimizer_steps=cfg['steps'],
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parameters=setup['parameters'],
        initial_state_tensor_sha256=initialsha,final_state_tensor_sha256=tensor_sha(state),trace_sha256=sha(cell/'TRACE.jsonl'),
        setup_sha256=sha(cell/'SETUP.json'),history_sha256=sha(cell/'history.json'),gradient_probes_sha256=sha(cell/'GRADIENT_PROBES.json'),
        supervised_corner_exposures=totalcorners,real_GT_used=False,new_backbone_forwards=0,elapsed_seconds=time.perf_counter()-start,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    write(marker,result);return result


def load_trained(run,arm,seed=1,device='cpu'):
    run=Path(run).resolve();protocol,bindings=verify(run);cell=run/'runs'/f'{arm}_seed{seed}';done=read(cell/'COMPLETION.json')
    if not (done['complete'] and done['PASS'] and done['bindings']==bindings and done['optimizer_steps']==protocol['training']['steps']):raise ValueError('Completed local checkpoint required')
    if sha(done['checkpoint'])!=done['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    saved=torch.load(done['checkpoint'],map_location='cpu')
    if not (saved['complete'] and saved['schema']=='pallet_dht_local_v3_checkpoint' and saved['arm']==arm and saved['seed']==seed and saved['bindings']==bindings and saved['optimizer_steps']==protocol['training']['steps']):raise ValueError('Checkpoint contract differs')
    for name,key in [('TRACE.jsonl','trace_sha256'),('SETUP.json','setup_sha256'),('history.json','history_sha256'),('GRADIENT_PROBES.json','gradient_probes_sha256')]:
        if sha(cell/name)!=done[key]:raise ValueError('Training evidence changed')
    if saved['final_state_tensor_sha256']!=tensor_sha(saved['state_dict']):raise ValueError('Weight tensor hash changed')
    model=LocalHoughPointRefiner(**saved['model_config']);model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    train_arm(a.run_dir,a.arm,a.seed,a.device)
