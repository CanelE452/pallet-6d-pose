"""Bounded prepared-feature training and GT-free scoring. All commands are real.

Training this cache runner is FROZEN-BACKBONE only. For online fine-tuning use
adapters.py inside the actual detector trainer; don't relabel cached learning.
"""
from __future__ import annotations
import argparse,hashlib,json,time,platform,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from .io import *
from .model import PointLinePoseModel,ModelConfig,initialize_matched
from .objective import compute_loss,LossConfig


def save_state(path,value):
    p=Path(path);tmp=p.with_suffix('.pending.pt');torch.save(value,tmp);tmp.replace(p)


def train(args):
    root=Path(args.output).resolve()
    if root.exists():raise ValueError('Nonempty or existing output; no overwrite or silent resume')
    root.mkdir(parents=True,exist_ok=False)
    ds=InstanceDataset(args.manifest)
    if len(args.seeds)!=len(set(args.seeds)) or any(s<0 for s in args.seeds):raise ValueError('Unique nonnegative seeds required')
    if not np.isfinite(args.lr) or args.lr<=0:raise ValueError('Positive finite learning rate required')
    if ds.meta['split'] not in ('train','generated'):raise ValueError('Training split required')
    cfg=read_json(args.config) if args.config else {}
    base=ModelConfig(**cfg)
    if args.steps<1 or args.steps>20000 or args.batch<1:raise ValueError('Budget outside bounded pilot range')
    torch.set_num_threads(2)
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable; no silent CPU fallback')
    protocol={'schema':'plpose_v5_cache_training_1','scope':'FROZEN_FEATURE_PILOT','manifest_sha256':sha256(ds.path),
              'model':asdict(base),'loss':asdict(LossConfig()),'steps':args.steps,'batch':args.batch,'seeds':args.seeds,
              'arms':['point','direct','hough'],'lr':args.lr,'source_sha256':source_hashes(),
              'precision':'FP32','checkpoint_selection':'last_step_only','no_real_training':True,
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':args.device}}
    write_json(root/'PROTOCOL.json',protocol);protocol_sha=sha256(root/'PROTOCOL.json')
    try:
        for seed in args.seeds:
            torch.manual_seed(seed)
            models={arm:PointLinePoseModel(ModelConfig(**{**asdict(base),'arm':arm})) for arm in protocol['arms']}
            common=initialize_matched(models,seed)
            rng=np.random.default_rng(seed);needed=args.steps*args.batch
            plan=np.concatenate([rng.permutation(len(ds)) for _ in range((needed+len(ds)-1)//len(ds))])[:needed].reshape(args.steps,args.batch)
            plan_sha=hashlib.sha256(plan.astype('<i8').tobytes()).hexdigest()
            for arm,model in models.items():
                cell=root/f'{arm}_seed{seed}';cell.mkdir()
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                save_state(cell/'INITIAL_STATE.pt',initial)
                common_sha=tensor_hash({k:initial[k] for k in common})
                model.to(args.device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
                history=[];start=time.perf_counter();steps_done=0
                with (cell/'TRACE.jsonl').open('w',encoding='utf-8') as trace:
                    for step,indices in enumerate(plan,1):
                        obs,gt=ds.batch(indices,args.device)
                        optimizer.zero_grad(set_to_none=True)
                        out=model(obs);loss,diag=compute_loss(out,obs,gt,model.lattice.lines)
                        if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss at {arm}/{seed}/{step}')
                        loss.backward()
                        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                        optimizer.step();steps_done+=1
                        row={'step':step,'indices':indices.tolist(),'ids':[ds.records[int(i)]['id'] for i in indices],
                             'loss':float(loss.detach()),'gradient_norm':float(grad),'symmetry_index':diag['symmetry_index']}
                        trace.write(json.dumps(row,separators=(',',':'))+'\n');trace.flush()
                        if step==1 or step%25==0 or step==args.steps:
                            row.update(diag);history.append(row)
                            print(f'{arm} seed{seed} {step}/{args.steps} loss={float(loss.detach()):.6f}',flush=True)
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                final={k:v.detach().cpu() for k,v in model.state_dict().items()}
                changed=sum(int(torch.count_nonzero(final[k]!=v)) for k,v in initial.items())
                if not all(torch.isfinite(v).all() for v in final.values()):raise ValueError('Nonfinite final state')
                write_json(cell/'HISTORY.json',history)
                payload={'schema':'plpose_v5_checkpoint_1','model_config':asdict(model.config),'state_dict':final,
                         'optimizer':optimizer.state_dict(),'steps':steps_done,'seed':seed,'arm':arm,
                         'protocol_sha256':protocol_sha,'source_sha256':protocol['source_sha256'],'training_split':ds.meta['split'],'complete':True}
                save_state(cell/'checkpoint_final.pt',payload)
                receipt={'complete':True,'scope':'FROZEN_FEATURE_TRAINING_NOT_ONLINE','steps':steps_done,'expected_steps':args.steps,
                         'parameters':sum(p.numel() for p in model.parameters()),'shared_initial_sha256':common_sha,'plan_sha256':plan_sha,
                         'changed_tensor_values':changed,'initial_state_sha256':tensor_hash(initial),'final_state_sha256':tensor_hash(final),
                         'checkpoint_sha256':sha256(cell/'checkpoint_final.pt'),'trace_sha256':sha256(cell/'TRACE.jsonl'),
                         'history_sha256':sha256(cell/'HISTORY.json'),'elapsed_seconds':time.perf_counter()-start,'seed':seed,'arm':arm,
                         'protocol_sha256':protocol_sha,
                         'active_gradient_elements_last_step':sum(int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None)}
                write_json(cell/'COMPLETION.json',receipt)
        write_json(root/'COMPLETION.json',{'complete':True,'cells':len(args.seeds)*3,'steps':args.steps*3*len(args.seeds),
                    'protocol_sha256':protocol_sha,'scientific_success':None,'next':'Calibration and held-out tests; not authorized automatically by completion.'})
    except BaseException as ex:
        write_json(root/'FAILURE.json',{'complete':False,'exception':repr(ex),'scientific_success':None});raise


def load_model(path,device='cpu',verify_source=True):
    path=Path(path)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('schema')!='plpose_v5_checkpoint_1' or not saved.get('complete'):raise ValueError('Complete kit checkpoint required')
    if verify_source and saved['source_sha256']!=source_hashes():raise ValueError('Source changed since training')
    receipt=read_json(path.parent/'COMPLETION.json')
    if receipt['checkpoint_sha256']!=sha256(path) or receipt['steps']!=saved['steps']:raise ValueError('Checkpoint receipt mismatch')
    model=PointLinePoseModel(ModelConfig(**saved['model_config']));model.load_state_dict(saved['state_dict'],strict=True)
    return model.to(device).eval(),saved


def score(args):
    ds=InstanceDataset(args.manifest);outpath=Path(args.output)
    if outpath.exists():raise ValueError('Prediction output already exists')
    model,saved=load_model(args.checkpoint,args.device)
    if saved.get('training_split')=='generated' and ds.meta['split']!='generated':
        raise ValueError('Generated-fixture checkpoint cannot score a real research population')
    rows=[]
    with torch.no_grad():
        for i,r in enumerate(ds.records):
            obs=ds.batch([i],args.device,with_target=False)
            out=model(obs)
            rows.append({'id':r['id'],'session':r['session'],'R':out['R'][0].cpu().tolist(),'t':out['t'][0].cpu().tolist(),
                         'points':out['points'][0].cpu().tolist(),'pose_valid':bool(out['pose_valid'][0]),
                         'observed_points':out['measurements'].points[0].cpu().tolist(),
                         'energy':out['energy'][0].cpu().tolist(),'selected':int(out['selected'][0]),
                         'rank':out['information_rank'][0].cpu().tolist(),
                         'source_observation_sha256':r['observation_sha256']})
    receipt={'schema':'plpose_v5_predictions_1','manifest_sha256':sha256(ds.path),'checkpoint_sha256':sha256(args.checkpoint),
             'arm':saved['arm'],'seed':saved['seed'],'scope':ds.meta['split'],'records':rows,
             'loader_observation_reads':ds.observation_reads,'loader_target_reads':ds.target_reads,
             'counter_scope':'Actual InstanceDataset loaders, not a universal OS syscall audit',
             'GT_based_inference_selection':False,'runtime_benchmarked':False}
    if ds.target_reads!=0:raise AssertionError('Scoring opened supervision')
    write_json(outpath,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--manifest',required=True);t.add_argument('--output',required=True)
    t.add_argument('--config');t.add_argument('--steps',type=int,default=2000);t.add_argument('--batch',type=int,default=4)
    t.add_argument('--seeds',type=int,nargs='+',default=[1,2,3]);t.add_argument('--lr',type=float,default=1e-4);t.add_argument('--device',default='cpu')
    s=sub.add_parser('score');s.add_argument('--manifest',required=True);s.add_argument('--checkpoint',required=True)
    s.add_argument('--output',required=True);s.add_argument('--device',default='cpu')
    a=sub.add_parser('audit');a.add_argument('--manifests',nargs='+',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='train':train(args)
    elif args.command=='score':score(args)
    else:audit_manifests(args.manifests,args.output)

if __name__=='__main__':main()
