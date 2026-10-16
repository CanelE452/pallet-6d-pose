from context36 import *
from model36 import Model,loss36
from cli_train import state_sha
import subprocess

def precheck():
    flags=setup36();d=torch.load(RUN/'cache/training_seed1.pt',weights_only=True,map_location='cpu');ii=torch.where(d['hasline'].any(-1))[0][:16];x={k:v[ii].cuda() for k,v in d['x'].items()};args=[d[k][ii].cuda() for k in ('gt','valid','support','hasline')];m=Model(1).cuda();initial=state_sha(m.state_dict());out=m(x,'both');loss,terms=loss36(out,x,*args,'both');assert terms['reachable']>0
    refpars=list(m.refiner.parameters());grads=torch.autograd.grad(terms['point'],refpars,allow_unused=True,retain_graph=True);assert all(g is None or not g.any() for g in grads)
    grads=torch.autograd.grad(terms['geometry']+terms['rank'],refpars,allow_unused=True,retain_graph=True);assert any(g is not None and g.abs().sum()>0 for g in grads);loss.backward();grad1={k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None};m.zero_grad(set_to_none=True);out2=m(x,'both');loss2,_=loss36(out2,x,*args,'both');loss2.backward();assert torch.equal(out['points'],out2['points']);assert all(torch.equal(v,dict(m.named_parameters())[k].grad) for k,v in grad1.items());assert all(torch.isfinite(v).all() for v in grad1.values())
    assert torch.allclose(out['normal'],x['features'][...,40:42],atol=1e-6);assert torch.allclose(out['offset'],x['features'][...,42],atol=1e-6)
    with torch.no_grad():
        raw=m.core.head(x['context']);mix=raw[...,0].tanh();delta=raw[...,1:].tanh();delta/=delta.norm(dim=-1,keepdim=True).clamp_min(1);p=x['base']+mix[...,None]*(x['strong']-x['base'])+.15*out['diag'][:,None,None]*delta;cache_diff=float((p-x['v34']).abs().max());assert cache_diff<1e-4
        m.refiner[-1].bias[:2]=torch.tensor([math.atanh(.25),math.atanh(-.25)],device='cuda');shift=m(x,'refine');assert torch.allclose(shift['theta']-x['theta'],torch.full_like(x['theta'],math.pi/96),atol=1e-6);assert torch.allclose(shift['rho']-x['rho'],torch.full_like(x['rho'],-.25*2*math.sqrt(2)/47),atol=1e-6)
    write(RUN/'PREFLIGHT.json',dict(status='PASS',flags=flags,initial_sha256=initial,finite_gradients=True,point_gradient_to_refiner_blocked=True,physical_aux_gradient_nonzero=True,repeated_forward_gradient_exact=True,subbin_coordinate_shift_exact=True,cache_v34_max_px=cache_diff,reachable_candidates=int(terms['reachable'])))

def worker(arm,seed):
    flags=setup36();folder=RUN/'models'/f'{arm}_seed{seed}'
    if (folder/'COMPLETE.json').exists():return
    assert not (folder/'START.json').exists();d=torch.load(RUN/'cache'/f'training_seed{seed}.pt',weights_only=True,map_location='cpu');x={k:v.cuda() for k,v in d['x'].items()};g,v,support,has=[d[k].cuda() for k in ('gt','valid','support','hasline')];m=Model(seed).cuda();folder.mkdir(parents=True,exist_ok=True);write(folder/'START.json',dict(initial_sha256=state_sha(m.state_dict()),cache_sha256=sha(RUN/'cache'/f'training_seed{seed}.pt'),plan_sha256=digest(json.dumps(d['plan'])),arm=arm,seed=seed,steps=STEPS,flags=flags,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad)));opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
    with (folder/'TRACE.jsonl').open('w') as f:
        for step,ix in enumerate(d['plan']):
            ii=torch.tensor(ix,device='cuda');xx=subset(x,ii);out=m(xx,arm);loss,terms=loss36(out,xx,g[ii],v[ii],support[ii],has[ii],arm);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=norm,**{k:float(t.detach()) for k,t in terms.items()}))+'\n')
            if (step+1)%500==0:f.flush();print('TRAIN36',arm,seed,step+1,round(time.monotonic()-tick),flush=True)
    state={k:t.detach().cpu() for k,t in m.state_dict().items()};torch.save(dict(state_dict=state,arm=arm,seed=seed,steps=STEPS),folder/'final.pt');write(folder/'COMPLETE.json',dict(complete=True,steps=STEPS,state_sha256=state_sha(state),checkpoint_sha256=sha(folder/'final.pt'),trace_sha256=sha(folder/'TRACE.jsonl')))

def main():
    register();assert read(RUN/'CACHE_COMPLETE.json')['complete'];precheck();write(RUN/'TRAIN_PROTOCOL.json',dict(source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [KIT/'model36.py',KIT/'train36.py',KIT/'context36.py']},protocol_sha256=sha(RUN/'PROTOCOL.json')))
    for seed in SEEDS:
        for arm in ARMS:subprocess.run([sys.executable,str(KIT/'train36.py'),'--arm',arm,'--seed',str(seed)],cwd=ROOT,check=True,env=dict(os.environ,PYTHONHASHSEED='0',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    for seed in SEEDS:assert len({read(RUN/'models'/f'{a}_seed{seed}'/'START.json')['initial_sha256'] for a in ARMS})==1
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,models=12,updates=24000));print('TRAINING36 COMPLETE',flush=True)

def load_model(arm,seed):
    model=Model(seed).cuda().eval();model.load_state_dict(torch.load(RUN/'models'/f'{arm}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);return model

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.arm:worker(a.arm,a.seed)
    else:main()
