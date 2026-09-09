"""Working cached-head trainer. A repository adapter must first export the data.

No downloaded models, optimizer-budget search, real-label training, auto-resume,
background jobs, Git writes, or notification calls occur here.
"""
from __future__ import annotations
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from .model import EvidenceVerifier,ARMS
from .objective import candidate_targets,quality_loss
from .cache_io import ExportDataset,sha256,tensor_state_sha,write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True);ap.add_argument('--protocol',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--arm',choices=ARMS,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--device',default='cpu')
    ap.add_argument('--smoke',action='store_true',help='Exactly 4 steps; never a main checkpoint')
    ap.add_argument('--allow-generated-fixture',action='store_true',help='Explicit generated-data software test; requires --smoke')
    args=ap.parse_args()
    protocol=json.loads(Path(args.protocol).read_text(encoding='utf8'))
    if protocol.get('locked') is not True:raise ValueError('CLI must resolve and freeze protocol before training')
    if protocol.get('train_manifest_sha256')!=sha256(args.manifest):raise ValueError('Manifest not bound to protocol')
    source_hashes={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if protocol.get('core_source_sha256')!=source_hashes:raise ValueError('Core source differs from frozen protocol')
    if args.seed not in protocol['seeds']:raise ValueError('Unregistered seed')
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output exists; preserve it, do not overwrite or silently resume')
    data=ExportDataset(args.manifest,verify_targets=True)
    if args.allow_generated_fixture and not args.smoke:
        raise ValueError('Generated fixture may never run as a main experiment')
    expected_origin='generated_fixture' if args.allow_generated_fixture else 'source_synthetic'
    if data.document.get('role')!='train' or any(r.get('origin')!=expected_origin for r in data.records):
        raise ValueError('Training requires source_synthetic split, or explicit generated fixture smoke')
    if data.channels!=tuple(protocol['channels']):raise ValueError('Feature channels differ from protocol')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cfg=protocol['training'];steps=4 if args.smoke else cfg['steps'];batch=cfg['batch']
    model=EvidenceVerifier(arm=args.arm,channels=data.channels,**protocol['model']).to(args.device)
    initial=tensor_state_sha(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(args.seed);sequence=[]
    while len(sequence)<cfg['steps']*batch:sequence.extend(rng.permutation(len(data)).tolist())
    plan=np.asarray(sequence[:cfg['steps']*batch],dtype=np.int64).reshape(cfg['steps'],batch)
    plan_sha=hashlib.sha256(plan.tobytes()).hexdigest()
    write_json(out/'START.json',dict(arm=args.arm,seed=args.seed,stage='smoke' if args.smoke else 'main',initial_state_sha256=initial,
        plan_sha256=plan_sha,manifest_sha256=sha256(args.manifest),protocol_sha256=sha256(args.protocol),core_source_sha256=source_hashes,
        parameters=model.parameter_receipt(),torch_version=torch.__version__,cuda=torch.cuda.is_available(),expected_steps=steps))
    started=time.monotonic();successful=0;history=[]
    try:
        model.train()
        for step in range(steps):
            obs,target=data.batch(plan[step],targets=True);obs=obs.to(args.device)
            target={k:v.to(args.device) for k,v in target.items()}
            mask=target['supervised'] & target['matched'][:,None] & obs.point_valid
            gt_cost=candidate_targets(obs.layouts,target['points'],mask,obs.diagonal)
            optimizer.zero_grad(set_to_none=True)
            lr=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*step/max(1,cfg['steps']-1))))
            for group in optimizer.param_groups:group['lr']=lr
            cost,corner=model(obs)
            loss,parts=quality_loss(cost,corner,gt_cost,obs.candidate_valid,**protocol['loss'])
            loss.backward()
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True))
            receipt=model.parameter_receipt() if step in (0,1,15,steps-1) else None
            optimizer.step();successful+=1
            trace={'step':successful,'indices':plan[step].tolist(),'loss':float(loss.detach()),'lr':lr,'gradient_norm':grad,**parts}
            if receipt is not None:trace['parameter_receipt']=receipt
            with (out/'TRACE.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(trace,allow_nan=False)+'\n')
            if successful%100==0 or successful==steps:
                history.append(trace);write_json(out/'HISTORY.json',history)
                print(f'{args.arm} seed={args.seed} step={successful}/{steps} loss={float(loss.detach()):.6f}',flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        checkpoint=dict(schema='pointline_v4_head_1',seed=args.seed,stage='smoke' if args.smoke else 'main',config=model.config,state_dict=state,
                        optimizer=optimizer.state_dict(),optimizer_steps=successful,protocol_sha256=sha256(args.protocol),
                        initial_state_sha256=initial,final_state_sha256=tensor_state_sha(state),plan_sha256=plan_sha)
        torch.save(checkpoint,out/'checkpoint_final.pt')
        write_json(out/'COMPLETION.json',dict(training_completed=True,accuracy_improved=None,stage=checkpoint['stage'],
            optimizer_steps=successful,checkpoint_sha256=sha256(out/'checkpoint_final.pt'),elapsed_seconds=time.monotonic()-started,
            initial_state_sha256=initial,final_state_sha256=checkpoint['final_state_sha256'],plan_sha256=plan_sha))
    except BaseException as exc:
        write_json(out/'FAILURE.json',dict(completed=False,successful_optimizer_steps=successful,error=repr(exc)))
        raise

if __name__=='__main__':main()
