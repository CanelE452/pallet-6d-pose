"""Fixed synthetic-only training; every arm receives the same sample plan."""
import time
from context12 import *
from router12 import Router,objective
from cli_train import state_sha

def training_data():
    _,O=observations('train','cpu');parts=[{k:O[k] for k in KEYS}]
    targets=read(V10/'targets_train.json')['rows'];alltargets=list(targets);bindings=[]
    for condition in CONDITIONS[1:]:
        if condition=='corner20':
            m,o=load_corner_train();path=RUN/'targets_train_corner20.json'
        else:m,o=C.load_observations('train',condition);path=V11/'targets'/f'train_{condition}.json'
        parts.append(o);t=read(path)['rows'];assert [r['id'] for r in t]==[r['id'] for r in m['rows']]
        alltargets.extend([r for r,c in zip(t,m['rows']) if c['detected']]);bindings.append(dict(condition=condition,targets_sha256=sha(path),detected=len(o['points'])))
    obs={k:torch.cat([p[k] for p in parts]).cuda() for k in KEYS}
    gt=torch.tensor([r['points'] for r in alltargets],dtype=torch.float32,device='cuda:0')
    valid=torch.tensor([r['valid'] for r in alltargets],device='cuda:0');kind=target_type(alltargets).cuda()
    return obs,gt,valid,kind,bindings

def preflight(O,base):
    o=subset(O,slice(0,4));b=base[:4];m=Router(1).cuda();model_count=sum(p.numel() for p in m.parameters())
    for arm in ARMS:
        with torch.no_grad():
            p=m(o,b,arm);expected=b if arm=='always_amodal' else o['points']+.9*(b-o['points'])
            assert torch.allclose(p['points'],expected,atol=1e-4,rtol=0)
            assert torch.allclose(p['points'],o['points']+p['keep']*(b-o['points'])+p['recovery'],atol=1e-4,rtol=0)
    with torch.no_grad():m.recovery.bias.fill_(.2)
    p=m(o,b,'gated_amodal');assert p['recovery'].abs().max()>0
    gt=o['points']+5.;v=torch.ones(4,8,dtype=torch.bool,device='cuda:0');kind=torch.ones(4,8,dtype=torch.long,device='cuda:0')
    obj,_,_=objective(p,o,gt,v,kind,'gated_amodal',torch.ones(3,device='cuda:0'));obj.backward()
    gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True));assert gn>0
    write(RUN/'PREFLIGHT.json',dict(status='PASS',parameters=model_count,initial_gate=.9,initial_recovery_zero=True,
        mixture_replay=True,finite_gradient_norm=gn,scope='Temporary synthetic smoke, discarded. Observability q is not calibrated natural visibility.'))

def main():
    setup();assert read(RUN/'PREPARATION_COMPLETE.json')['complete']
    lock=RUN/'TRAIN_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite training')
    O,gt,valid,kind,bindings=training_data();counts=torch.bincount(kind[valid],minlength=3).float();cw=counts.rsqrt();cw=cw/cw.mean()
    write(lock,dict(steps=2000,batch=16,arms=ARMS,seeds=SEEDS,frames=len(gt),class_counts=counts.cpu().tolist(),
        class_weights=cw.cpu().tolist(),bindings=bindings,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},
        parent_protocol_sha256=sha(RUN/'PROTOCOL.json'),base_checkpoint_sha256={str(s):sha(V10/'models'/f'local_seed{s}'/'final.pt') for s in SEEDS},
        optimizer='AdamW lr3e-4 weight_decay1e-4, float32, clipnorm5. Fixed final checkpoint. No real training or validation selection.'))
    start=time.monotonic();complete=[]
    for seed in SEEDS:
        base=local(seed);base.requires_grad_(False);bp=[]
        with torch.inference_mode():
            for first in range(0,len(gt),32):bp.append(base_points(base,subset(O,slice(first,first+32))))
        bp=torch.cat(bp).detach().clone();del base
        if seed==1:preflight(O,bp)
        rng=np.random.default_rng(12000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000],dtype=np.int64).reshape(2000,16);initial=None
        for arm in ARMS:
            directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False)
            model=Router(seed).cuda();initialsha=state_sha(model.state_dict())
            if initial is None:initial=initialsha
            assert initial==initialsha
            write(directory/'START.json',dict(arm=arm,seed=seed,initial_state_sha256=initialsha,plan_sha256=digest(plan.tobytes().hex()),
                parameters=sum(p.numel() for p in model.parameters()),active_capacity='scalar_gate only one scalar active; gate-only recovery head inactive; two amodal arms same full network.'))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step in range(2000):
                    ii=torch.tensor(plan[step],device='cuda:0');obs=subset(O,ii);out=model(obs,bp[ii],arm)
                    obj,point,vis=objective(out,obs,gt[ii],valid[ii],kind[ii],arm,cw)
                    opt.zero_grad(set_to_none=True);obj.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(obj.detach()),point_loss=float(point.detach()),routing_loss=float(vis.detach()),gradient_norm=gn))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,'loss',round(float(obj.detach()),3),'sec',round(time.monotonic()-tick),flush=True);f.flush()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=2000),directory/'final.pt')
            done=dict(complete=True,seed=seed,arm=arm,steps=2000,elapsed_seconds=time.monotonic()-tick,checkpoint_sha256=sha(directory/'final.pt'))
            write(directory/'COMPLETE.json',done);complete.append(done);del model,opt
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,runs=complete,updates=24000,elapsed_seconds=time.monotonic()-start))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
