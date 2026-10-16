"""Fixed-budget synthetic-only comparison; no validation selection."""
import time
import numpy as np
import torch
from common import *
from optional_model import OptionalRefiner,predict_lines
from cli_train import state_sha


def training_data():
    _,clean=observations('train','cpu')
    keys=('features','content','box','K','dims','image_hw','points')
    clean={k:clean[k] for k in keys};parts=[clean]
    target=read(V10/'targets_train.json')['rows']
    gt=[torch.tensor([r['points'] for r in target],dtype=torch.float32)]
    valid=[torch.tensor([r['valid'] for r in target])]
    bindings=[]
    for condition in CONDITIONS[1:]:
        manifest,O=load_observations('train',condition);parts.append(O)
        rows=read(RUN/'targets'/f'train_{condition}.json')['rows']
        rows=[r for r,m in zip(rows,manifest['rows']) if m['detected']]
        gt.append(torch.tensor([r['points'] for r in rows],dtype=torch.float32))
        valid.append(torch.tensor([r['valid'] for r in rows]))
        bindings.append(dict(condition=condition,frames=len(rows),manifest_sha256=sha(RUN/'cache'/'train'/condition/'manifest.json'),
            target_sha256=sha(RUN/'targets'/f'train_{condition}.json')))
    return {k:torch.cat([p[k] for p in parts]).cuda() for k in keys},torch.cat(gt).cuda(),torch.cat(valid).cuda(),bindings


def cached_training_lines(seed):
    z=np.load(V10/f'lines_train_seed{seed}.npz')
    ls=[torch.tensor(z['lines'])];ms=[torch.tensor(z['mass'])]
    for condition in CONDITIONS[1:]:
        p=frozen_lines('train',condition,seed);ls.append(p['lines']);ms.append(p['mass'])
    return torch.cat(ls).cuda(),torch.cat(ms).cuda()


def main():
    setup();assert read(RUN/'PREPARATION_COMPLETE.json')['complete']
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Training already registered; refusing overwrite')
    O,gt,valid,bindings=training_data()
    protocol=dict(steps=2000,batch=16,seeds=SEEDS,arms=ARMS,training_frames=len(gt),bindings=bindings,
        v10_train_target_sha256=sha(V10/'targets_train.json'),v10_protocol_sha256=sha(V10/'PROTOCOL.json'),
        architecture='Preserved frozen V10 local output + sigmoid(point_token gate)*extra residual. Gate init sigmoid(-2); extra final head zero. No force-to-line correction.',
        joint='YOLO and base local remain frozen. DHT trunk, line head and point-map gate update through piecewise differentiable mode decode from point loss only. No auxiliary line GT.',
        loss='V10 SmoothL1 + 0.25 nonworsening penalty relative to preserved local output, on synthetic valid points including artificially hidden/outside corners.',
        lr_refiner=3e-4,lr_dht=1e-5,seed_plan='Same shuffled training pool/sample/update sequence per seed for all arms.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')})
    write(RUN/'TRAIN_PROTOCOL.json',protocol)
    start=time.monotonic();complete=[];checks=[]
    for seed in SEEDS:
        ll,mm=cached_training_lines(seed);assert len(ll)==len(gt)
        base=local(seed);base.requires_grad_(False);bp=[]
        with torch.inference_mode():
            for first in range(0,len(gt),32):
                obs=subset(O,slice(first,first+32));bp.append(base(obs,ll[first:first+32],mm[first:first+32],'local')['points'])
        basepoints=torch.cat(bp).detach().clone();del base,bp
        rng=np.random.default_rng(seed+11000);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000],dtype=np.int64).reshape(2000,16)
        initial=None
        for arm in ARMS:
            cell=RUN/'models'/f'{arm}_seed{seed}';cell.mkdir(parents=True,exist_ok=False)
            model=OptionalRefiner(seed).cuda();model.train()
            ss=state_sha(model.state_dict())
            if initial is None:initial=ss
            assert ss==initial
            line=dht(seed) if arm=='lines_joint' else None
            params=[dict(params=model.parameters(),lr=3e-4)]
            if line is not None:
                line.train();line.requires_grad_(True);params.append(dict(params=line.parameters(),lr=1e-5))
            opt=torch.optim.AdamW(params,weight_decay=1e-4)
            write(cell/'START.json',dict(arm=arm,seed=seed,initial_state_sha256=ss,plan_sha256=digest(plan.tobytes().hex()),
                refiner_parameters=sum(p.numel() for p in model.parameters()),
                extra_dht_parameters=sum(p.numel() for p in line.parameters()) if line is not None else 0))
            obs=subset(O,slice(0,4))
            with torch.no_grad():
                pred=model(obs,basepoints[:4],ll[:4],mm[:4],arm)
                assert torch.equal(pred['points'],basepoints[:4])
            tick=time.monotonic();nonzero_dht=False
            with (cell/'TRACE.jsonl').open('w') as f:
                for step in range(2000):
                    ii=torch.tensor(plan[step],device='cuda:0');obs=subset(O,ii)
                    l,m=predict_lines(line,obs) if line is not None else (ll[ii],mm[ii])
                    out=model(obs,basepoints[ii],l,m,arm)
                    obj=loss(out,basepoints[ii],gt[ii],valid[ii]);opt.zero_grad(set_to_none=True);obj.backward()
                    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
                    dgn=float(torch.nn.utils.clip_grad_norm_(line.parameters(),5.,error_if_nonfinite=True)) if line is not None else 0.
                    nonzero_dht=nonzero_dht or dgn>0
                    opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(obj.detach()),grad_norm=gn,dht_grad_norm=dgn))+'\n')
                    if (step+1)%250==0:
                        print('TRAIN',arm,seed,step+1,'loss',round(float(obj.detach()),4),'elapsed',round(time.monotonic()-tick),flush=True);f.flush()
            with torch.no_grad():
                obs=subset(O,slice(0,4));off=model(obs,basepoints[:4],ll[:4],mm[:4],arm,off=True)
                assert torch.equal(off['points'],basepoints[:4])
            if line is not None:assert nonzero_dht
            state=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000)
            if line is not None:state['dht_state_dict']={k:v.detach().cpu() for k,v in line.state_dict().items()}
            torch.save(state,cell/'final.pt')
            receipt=dict(complete=True,arm=arm,seed=seed,steps=2000,elapsed_seconds=time.monotonic()-tick,
                checkpoint_sha256=sha(cell/'final.pt'),exact_initial_base_parity=True,exact_off_base_parity=True,
                dht_nonzero_point_loss_gradient=nonzero_dht if line is not None else None)
            write(cell/'COMPLETE.json',receipt);complete.append(receipt)
            del model,opt,line
        del ll,mm,basepoints
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,runs=complete,updates=18000,elapsed_seconds=time.monotonic()-start))
    print('TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
