"""Matched-budget learned-neighborhood experiment; finite, reproducible 9-cell run."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch
from prepare import RUN,ROOT,EXPORT,OLD,Split,sha,write
from model import GuidedDHT,point_maps,targets,objective
from cli_train import state_sha

V6=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
ARMS=('none','points','donor')
STEPS=2000
BATCH=8


def donor_indices(records,split):
    order=sorted(range(len(records)),key=lambda i:hashlib.sha256(('v9-donor:'+split+':'+records[i]['id']).encode()).digest())
    indices=[0]*len(order)
    for j,i in enumerate(order):indices[i]=order[(j+1)%len(order)]
    return indices


def pack(split,device,supervision):
    data=Split(EXPORT/f'{split}.json');rows=json.loads((RUN/f'points_{split}.json').read_text())['rows']
    assert [r['id'] for r in rows]==[r['id'] for r in data.records]
    oo=[];ss=[];order=[]
    for first in range(0,len(rows),32):
        ii=list(range(first,min(first+32,len(rows))));O,S,g=data.batch(ii,'cpu',targets=supervision)
        oo.append(O);order.append(g)
        if supervision:ss.append(S)
    O={k:torch.cat([o[k] for o in oo]).to(device) for k in oo[0]}
    points=torch.tensor([r['points'] for r in rows],dtype=torch.float32,device=device)
    bb=O['box'];wh=bb[:,2:]-bb[:,:2]
    pp=(points-bb[:,None,:2])/wh[:,None]
    donors=donor_indices(data.records,split)
    dp=bb[:,None,:2]+pp[donors]*wh[:,None]
    actual=point_maps(points,bb);donor=point_maps(dp,bb)
    maps={'none':torch.zeros_like(actual),'points':actual,'donor':donor}
    gt=None
    if supervision:
        S={k:torch.cat([s[k] for s in ss]).to(device) for k in ss[0]};gt=targets(S,O)
    return data,O,maps,torch.cat(order).to(device),gt,points,dp


def preflight():
    data,O,maps,orders,target,points,_=pack('calibration','cpu',True)
    ii=list(range(8));obs={k:v[ii] for k,v in O.items()};m=GuidedDHT(V6/'cache_heads/hough_seed1/checkpoint_final.pt',1)
    with torch.no_grad():
        a=m(obs,maps['none'][ii])['logits'];b=m(obs,maps['points'][ii])['logits']
        assert torch.equal(a,b)
        x=m.trunk(obs['features'],obs['content'],obs['K'],obs['dims'],obs['box'],obs['image_hw'])
        assert torch.equal(a,m.line(x))
        assert torch.equal(point_maps(points[ii],obs['box']),point_maps(points[ii][:,[7,3,0,4,1,5,2,6]],obs['box']))
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
    before=state_sha(m.state_dict());grad=[]
    for step in range(8):
        out=m(obs,maps['points'][ii]);loss=objective(out['logits'],target[0][ii],target[1][ii],orders[ii])
        opt.zero_grad();loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True)
        grad.append(float(gn));opt.step()
    with torch.no_grad():
        a=m(obs,maps['none'][ii])['logits'];b=m(obs,maps['points'][ii])['logits'];effect=float((a-b).abs().max())
        assert effect>1e-7
        g=m(obs,maps['points'][ii])['gate'];assert float(g.min())>=.5 and float(g.max())<=1.5
    checks=dict(initial_identity_and_arm_parity=True,point_permutation_invariance=True,smoke_updates=8,
        finite_gradient_norms=grad,trained_point_input_logit_max_effect=effect,
        model_changed=state_sha(m.state_dict())!=before,trainable=sum(p.numel() for p in m.parameters()),
        scope='Disposable 8-update smoke on reused calibration; weights discarded, not used for main training or selection.')
    write(RUN/'PREFLIGHT.json',checks);print('PREFLIGHT PASS',checks,flush=True)


def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    RUN.mkdir(parents=True,exist_ok=True)
    sources={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('prepare.py')]}
    protocol=dict(arms=ARMS,seeds=[1,2,3],steps=STEPS,batch=BATCH,lr=1e-4,precision='float32; TF32 disabled; no AMP',
        source_sha256=sources,manifests={s:sha(EXPORT/f'{s}.json') for s in ('train','calibration','synth_val')},
        points_sha256={s:sha(RUN/f'points_{s}.json') for s in ('train','calibration','synth_val')},
        initial_checkpoints={str(s):sha(V6/f'cache_heads/hough_seed{s}/checkpoint_final.pt') for s in (1,2,3)},
        dependency_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((OLD/'plpose_local').glob('*.py'))},
        architecture='Frozen YOLO point observations and ROI features. Train inherited DHT trunk/line plus '
                     'identical 51->16->48 spatial gates, input=image features+3 point-neighborhood maps. '
                     'Gate 1+.5*tanh, identity initialized. No point pairs or coordinate updates.',
        objective='Only supervised structural-line CE, minimum over one allowed object-wide C1/C2 element; '
                  'same loss for all arms. No weak scratch point/pose head or self-training.',
        baseline='Same gate architecture with zero point maps; donor arm receives wrong normalized point geometry in training.',
        matching='Per seed all arms have identical state, parameter count, sample order, steps and optimizer settings.',
        gate='On synth_val every seed: points arm improves role-aware and unordered mean line error by >=5% '
             'over none, role-aware P90 nonworse, role mean better than trained donor. Correct input must '
             'also beat donor-input intervention in each points model. No test-based checkpoint selection.',
        evaluation='Full matched synth_val505/512, calibration253/256 for causal input ablations only. '
                   'Both are reused sets, not fresh holdouts; GT read only after inference artifacts saved.',
        budget_optimizer_steps=STEPS*len(ARMS)*3,goal_scope='Learned point guidance improves line detection, '
            'not proof of final point/pose or real-image gains. Original YOLO outputs are fixed.')
    lock=RUN/'PROTOCOL.json'
    if lock.exists():assert json.loads(lock.read_text())==json.loads(json.dumps(protocol))
    else:write(lock,protocol)
    if not (RUN/'PREFLIGHT.json').exists():preflight()
    data,O,maps,orders,target,_,_=pack('train','cuda:0',True)
    started=time.monotonic();bindings={}
    for seed in (1,2,3):
        rng=np.random.default_rng(seed);order=[]
        while len(order)<STEPS*BATCH:order.extend(rng.permutation(len(data.records)).tolist())
        plan=np.array(order[:STEPS*BATCH],dtype=np.int64).reshape(STEPS,BATCH)
        plan_sha=hashlib.sha256(plan.tobytes()).hexdigest();shared=None
        for arm in ARMS:
            cell=RUN/'models'/f'{arm}_seed{seed}'
            if (cell/'COMPLETE.json').exists():
                j=json.loads((cell/'START.json').read_text());bindings[f'{arm}_seed{seed}']=j;continue
            if cell.exists():raise SystemExit('incomplete cell exists, preserve for investigation: '+str(cell))
            cell.mkdir(parents=True)
            model=GuidedDHT(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed).to('cuda:0')
            initial=state_sha(model.state_dict());param=sum(p.numel() for p in model.parameters())
            if shared is None:shared=initial
            assert shared==initial
            info=dict(arm=arm,seed=seed,initial_state_sha256=initial,plan_sha256=plan_sha,trainable=param,
                      protocol_sha256=sha(lock),steps=STEPS,batch=BATCH)
            bindings[f'{arm}_seed{seed}']=info;write(cell/'START.json',info)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);model.train();tick=time.monotonic()
            with (cell/'TRACE.jsonl').open('w') as f:
                for step in range(STEPS):
                    ii=torch.as_tensor(plan[step],device='cuda:0');obs={k:v[ii] for k,v in O.items()}
                    out=model(obs,maps[arm][ii]);loss=objective(out['logits'],target[0][ii],target[1][ii],orders[ii])
                    opt.zero_grad(set_to_none=True);loss.backward()
                    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),grad_norm=gn))+'\n')
                    if (step+1)%250==0:print(arm,'seed',seed,step+1,'/',STEPS,'loss',round(float(loss),4),flush=True);f.flush()
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=state,seed=seed,arm=arm,steps=STEPS,protocol_sha256=sha(lock)),cell/'final.pt')
            write(cell/'COMPLETE.json',dict(complete=True,steps=STEPS,elapsed_seconds=time.monotonic()-tick,
                checkpoint_sha256=sha(cell/'final.pt'),final_state_sha256=state_sha(state)))
            del opt,model
    for seed in (1,2,3):
        rows=[bindings[f'{a}_seed{seed}'] for a in ARMS]
        for k in ('initial_state_sha256','plan_sha256','trainable'):assert len({r[k] for r in rows})==1
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,optimizer_steps=STEPS*len(ARMS)*3,
        elapsed_seconds=time.monotonic()-started,bindings=bindings,matched_comparison_checks=True))
    print('ALL TRAINING COMPLETE',flush=True)


if __name__=='__main__':main()
