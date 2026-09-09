"""Two matched1000-update learned decoders on an immutable synthetic cache.

No frozen CNN is loaded and no real-array rows enter training or validation.
All checkpoints are last-step checkpoints; validation never chooses a model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from .cache import read,write,sha,verify_hashes,source_hashes,load_model_arrays,prepare_smoke
from .model import CandidateRefiner,to_device


def tensor_sha(state):
    h=hashlib.sha256()
    for name,value in state.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def torch_save(path,value):
    path=Path(path);temporary=path.with_suffix('.pending.pth');torch.save(value,temporary);temporary.replace(path)


def loss_value(predicted,arrays,indices,device,beta):
    gt=torch.tensor(np.asarray(arrays['gt_points'][indices]),device=device)
    mask=torch.tensor(np.asarray(arrays['loss_valid'][indices]),device=device,dtype=torch.bool)
    diagonal=torch.tensor(np.asarray(arrays['diagonal'][indices]),device=device)[:,None,None]
    per=F.smooth_l1_loss((predicted[:,:8]-gt[:,:8])/diagonal,torch.zeros_like(predicted[:,:8]),beta=beta,reduction='none').mean(-1)
    return per[mask].sum(),int(mask.sum())


def validate(model,arrays,indices,device,beta,batch_size):
    model.eval();total=0.;denominator=0
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size]
            out=model(to_device(arrays,ix,device))
            value,count=loss_value(out,arrays,ix,device,beta)
            total+=float(value);denominator+=count
    model.train()
    if not denominator:raise ValueError('No supervised validation corners')
    return dict(loss=total/denominator,supervised_corners=denominator,frames=len(indices))


def sample_order(indices,steps,batch,seed):
    rng=np.random.default_rng(seed);stream=[]
    while len(stream)<steps*batch:stream.extend(rng.permutation(indices).tolist())
    return np.asarray(stream[:steps*batch],np.int64).reshape(steps,batch)


def train_arm(run,arm,device):
    config=read(run/'TRAIN_PROTOCOL.json');cfg=config['training'];manifest=read(run/'MANIFEST.json')
    done=read(run/'CACHE_COMPLETION.json')
    if not done['complete'] or not done['PASS']:raise ValueError('Incomplete cache')
    verify_hashes(done['array_sha256']);verify_hashes(done['bindings']['source_sha256'])
    sources=source_hashes();sources[str(Path(__file__).resolve())]=sha(__file__)
    binding=dict(protocol_sha256=sha(run/'TRAIN_PROTOCOL.json'),manifest_sha256=sha(run/'MANIFEST.json'),
        cache_completion_sha256=sha(run/'CACHE_COMPLETION.json'),source_sha256=sources,
        backbone_sha256=config['backbone']['sha256'])
    cell=run/'runs'/arm;cell.mkdir(parents=True,exist_ok=True)
    completion=cell/'COMPLETION.json'
    if completion.exists():
        existing=read(completion)
        if not existing['complete'] or existing['bindings']!=binding:raise ValueError('Completed cell differs')
        if sha(existing['checkpoint'])!=existing['checkpoint_sha256']:raise ValueError('Changed completed checkpoint')
        return existing
    if (cell/'BATCH_TRACE.jsonl').exists():
        raise ValueError('Interrupted training is not silently resumed; preserve and explicitly restart this pilot cell')
    arrays=load_model_arrays(run,arm);training=np.array(manifest['populations']['synth_train'],np.int64)
    validation=np.array(manifest['populations']['synth_val'],np.int64)
    real=set(manifest['populations']['real_dev'])
    if set(training)&set(validation) or set(training)&real or set(validation)&real:raise ValueError('Population overlap')
    order=sample_order(training,cfg['steps'],cfg['batch'],cfg['seed'])
    if not set(order.ravel()).issubset(set(training)):raise ValueError('Nontrain row in optimizer sampler')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    model_config={k:config['model'][k] for k in ['residual_diagonal_fraction','geometry_features']}
    model=CandidateRefiner(**model_config).to(device)
    params=sum(p.numel() for p in model.parameters())
    if params!=config['model']['parameter_count']:raise ValueError('Unexpected parameter count')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_sha=tensor_sha(initial);torch_save(cell/'INITIAL_STATE.pth',initial)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    identity_batch=to_device(arrays,order[0],device)
    with torch.no_grad():identity=model(identity_batch)
    if not torch.equal(identity,identity_batch['points']):raise ValueError('Initial output is not exact identity')
    write(cell/'SETUP.json',dict(complete=True,stage=config['stage'],arm=arm,seed=cfg['seed'],parameters=params,
        initial_state_tensor_sha256=initial_sha,initial_state_file_sha256=sha(cell/'INITIAL_STATE.pth'),
        identity_exact=True,train_frames=len(training),val_frames=len(validation),
        expected_optimizer_steps=cfg['steps'],batch=cfg['batch'],bindings=binding,
        all_sample_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
    model.train();history=[];probes=[];loss_window=[];supervised_total=0
    started=time.perf_counter()
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    for step,indices in enumerate(order,1):
        batch=to_device(arrays,indices,device);optimizer.zero_grad(set_to_none=True)
        prediction,diagnostics=model(batch,return_diagnostics=True)
        loss_sum,count=loss_value(prediction,arrays,indices,device,cfg['smooth_l1_beta'])
        if not count:raise ValueError(f'No supervised corners in batch at step{step}; update not counted')
        loss=loss_sum/count
        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at step{step}')
        loss.backward()
        gradients={name:p.grad for name,p in model.named_parameters() if p.grad is not None}
        if any(not bool(torch.isfinite(g).all()) for g in gradients.values()):raise ValueError('Nonfinite gradient')
        if not torch.equal(prediction[:,8],batch['points'][:,8]):raise ValueError('Centroid changed')
        if step in [1,2,4,100,500,1000]:
            probes.append(dict(step=step,loss=float(loss),supervised_corners=count,
                gradient_l2={prefix:float(sum(g.detach().double().square().sum() for name,g in gradients.items() if name.startswith(prefix)).sqrt())
                    for prefix in ['candidate_encoder','score','gate','residual','global_encoder']},
                signed_gate_min=float(diagnostics['signed_gate'].min()),signed_gate_max=float(diagnostics['signed_gate'].max()),
                signed_gate_abs_mean=float(diagnostics['signed_gate'].abs().mean())))
        optimizer.step();supervised_total+=count;loss_window.append(float(loss))
        with (cell/'BATCH_TRACE.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step,indices=indices.tolist(),supervised_corners=count),separators=(',',':'))+'\n')
        if step%cfg['val_every']==0 or step==cfg['steps']:
            value=validate(model,arrays,validation,device,cfg['smooth_l1_beta'],cfg['batch'])
            row=dict(step=step,mean_training_loss_since_previous_log=float(np.mean(loss_window)),
                validation=value,elapsed_seconds=time.perf_counter()-started)
            history.append(row);loss_window=[];write(cell/'history.json',history)
            print(f'{arm} step{step}/{cfg["steps"]} train={row["mean_training_loss_since_previous_log"]:.6f} val={value["loss"]:.6f}',flush=True)
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    changed=sum(int((state[k]!=initial[k]).sum()) for k in state)
    if changed==0:raise ValueError('No decoder weights changed')
    checkpoint=cell/'checkpoint_final.pth'
    saved=dict(schema='pallet_dht_decoder_checkpoint_v1',complete=True,stage=config['stage'],arm=arm,seed=cfg['seed'],
        optimizer_steps=cfg['steps'],expected_optimizer_steps=cfg['steps'],parameters=params,
        state_dict=state,optimizer=optimizer.state_dict(),model_config=model_config,bindings=binding,
        initial_state_tensor_sha256=initial_sha,final_state_tensor_sha256=tensor_sha(state),
        trace_sha256=sha(cell/'BATCH_TRACE.jsonl'),history_sha256=sha(cell/'history.json'))
    torch_save(checkpoint,saved)
    write(cell/'GRADIENT_PROBES.json',dict(complete=True,records=probes,before_optimizer_step=True,backbone_forward_calls=0))
    result=dict(schema='decoder_probe_training_cell_v1',complete=True,PASS=True,stage=config['stage'],arm=arm,
        seed=cfg['seed'],parameters=params,optimizer_steps=cfg['steps'],expected_optimizer_steps=cfg['steps'],
        checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha(checkpoint),bindings=binding,
        initial_state_tensor_sha256=initial_sha,final_state_tensor_sha256=tensor_sha(state),changed_parameter_values=changed,
        trace_sha256=sha(cell/'BATCH_TRACE.jsonl'),history_sha256=sha(cell/'history.json'),
        gradient_probe_sha256=sha(cell/'GRADIENT_PROBES.json'),train_frames=len(training),val_frames=len(validation),
        total_batch_exposures=int(order.size),supervised_corner_exposures=supervised_total,
        all_zero_loss_batches=0,real_samples_used_for_optimization_or_validation=0,
        elapsed_seconds=elapsed,peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0)
    verify_hashes(sources);write(completion,result)
    del model,optimizer
    if torch.cuda.is_initialized():torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run=prepare_smoke(args.run_dir) if args.smoke else args.run_dir.resolve()
    if not (run/'PURPOSE.md').is_file():raise ValueError('PURPOSE.md required')
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    config=read(run/'TRAIN_PROTOCOL.json')
    cells=[train_arm(run,arm,args.device) for arm in config['arms']]
    if len({c['initial_state_tensor_sha256'] for c in cells})!=1:raise ValueError('Different initial states across arms')
    if len({c['trace_sha256'] for c in cells})!=1:raise ValueError('Different batch/mask trace across arms')
    result=dict(schema='decoder_probe_training_completion_v1',complete=True,PASS=True,stage=config['stage'],
        protocol_sha256=sha(run/'TRAIN_PROTOCOL.json'),cache_completion_sha256=sha(run/'CACHE_COMPLETION.json'),
        expected_runs=2,runs=cells,identical_initial_state=True,identical_full_batch_trace=True,
        actual_optimizer_steps=sum(c['optimizer_steps'] for c in cells),no_checkpoint_selection=True)
    write(run/'TRAINING_COMPLETION.json',result)


if __name__=='__main__':main()
