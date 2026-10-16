"""Train all twelve matched arms on real cached pretrained features."""
import time,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from core import *

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    protocol=read(OUT/'PROTOCOL.json');assert read(OUT/'PREFLIGHT.json')['PASS']
    assert read(OUT/'CACHE_AUDIT.json')['PASS']
    device='cuda:0';m=base_model().to(device);flow=m.model[-1].flow_model.eval()
    z={k:v.to(device) for k,v in load_cache('train').items()}
    N=len(z['features']);steps=protocol['steps'];bs=protocol['batch'];allcells={}
    lock=OUT/'TRAIN_SOURCE_LOCK.json'
    sources={name:sha(KIT/name) for name in ['core.py','prepare.py','verify.py','train.py','evaluate.py']}
    binding=dict(sources=sources,protocol_sha256=sha(OUT/'PROTOCOL.json'),
                 cache_sha256={s:sha(OUT/f'{s}.pt') for s in ['train','val']})
    if lock.exists():assert read(lock)==binding
    else:write(lock,binding)
    start=time.time()
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);indices=[]
        while len(indices)<steps*bs:indices.extend(rng.permutation(N).tolist())
        plan=np.array(indices[:steps*bs]).reshape(steps,bs)
        phash=hashlib.sha256(plan.tobytes()).hexdigest()
        for arm,(dims,sym) in ARMS.items():
            cell=OUT/'models'/f'{arm}_seed{seed}';done=cell/'COMPLETE.json'
            if done.exists():allcells[cell.name]=read(done);continue
            if cell.exists():raise RuntimeError(f'Incomplete cell exists: {cell}')
            cell.mkdir(parents=True)
            model=OutputAdapter(m.model[-1],seed).to(device)
            init=state_sha(model.state_dict())
            opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'],weight_decay=protocol['weight_decay'])
            history=[];flips=0;loss0=None
            for step,idx in enumerate(plan):
                batch=subset(z,idx)
                opt.zero_grad();pred=model(batch,dims)
                loss,terms,branch=objective(pred,batch,flow,sym);value=loss.mean()
                if loss0 is None:loss0=float(value.detach())
                assert torch.isfinite(value)
                value.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True)
                opt.step();flips+=int(branch.sum())
                if (step+1)%100==0:
                    history.append(dict(step=step+1,loss=float(value.detach()),terms=terms.detach().mean(0).cpu().tolist(),
                                        gradient_norm=float(gn),alternate_targets_so_far=flips))
                    print(f'{arm} seed{seed} {step+1}/{steps} loss={float(value.detach()):.5f} alternate={flips}',flush=True)
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(dict(state=state,arm=arm,seed=seed),cell/'final.pt')
            result=dict(completed=True,steps=steps,plan_sha256=phash,initial_state_sha256=init,
                        final_state_sha256=state_sha(state),checkpoint_sha256=sha(cell/'final.pt'),
                        trainable_parameters=sum(p.numel() for p in model.parameters()),
                        alternate_target_count=flips,initial_batch_loss=loss0,history=history)
            write(done,result);allcells[cell.name]=result
    for seed in protocol['seeds']:
        cells=[allcells[f'{a}_seed{seed}'] for a in ARMS]
        assert len({c['plan_sha256'] for c in cells})==1
        assert len({c['initial_state_sha256'] for c in cells})==1
    write(OUT/'TRAINING_COMPLETE.json',dict(PASS=True,cells=allcells,optimizer_steps=steps*len(allcells),
        same_initial_state_and_plan_per_seed=True,elapsed_seconds=time.time()-start))

if __name__=='__main__':main()
