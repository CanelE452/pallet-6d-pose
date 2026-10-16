import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common import *
from model import *
import numpy as np,time

def main():
    cfg=register();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    z={k:v.cuda() for k,v in torch.load(PREVIOUS/'train.pt',weights_only=True).items()}
    rows=read(PREVIOUS/'train_records.json');assert {r['asset'] for r in rows}==set(ASSETS[:2])
    gt=torch.tensor(np.array([r['gt'] for r in rows]),dtype=torch.float32,device='cuda')
    sources={n:sha(KIT/n) for n in ['common.py','model.py','train.py']}
    write(OUT/'TRAIN_LOCK.json',dict(sources=sources,protocol_sha256=sha(OUT/'PROTOCOL.json'),
        train_cache_sha256=sha(PREVIOUS/'train.pt'),train_records_sha256=sha(PREVIOUS/'train_records.json')))
    cells={};start=time.time()
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);plan=[]
        while len(plan)<cfg['steps']*cfg['batch']:plan.extend(rng.permutation(len(rows)).tolist())
        plan=np.array(plan[:cfg['steps']*cfg['batch']]).reshape(cfg['steps'],cfg['batch'])
        for arm in ARMS:
            name=f'{arm}_s{seed}';folder=OUT/'models'/name;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.json').exists():cells[name]=read(folder/'COMPLETE.json');continue
            m=SelectorResidual(arm,seed).cuda();small={k:v[:4] for k,v in z.items()}
            assert torch.equal(m(small)['points'],small['stock'])
            initial=hashlib.sha256(b''.join(v.cpu().numpy().tobytes() for v in m.state_dict().values())).hexdigest()
            opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[]
            for step,idx in enumerate(plan):
                opt.zero_grad(set_to_none=True);o=m({k:v[idx] for k,v in z.items()});loss,sel,res=objective(o,gt[idx],arm)
                assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
                if (step+1)%300==0:
                    history.append(dict(step=step+1,loss=float(loss.detach()),selector=float(sel.detach()),residual=float(res.detach()),gradient_norm=float(gn)))
                    print(name,step+1,history[-1],flush=True)
            torch.save({k:v.cpu() for k,v in m.state_dict().items()},folder/'final.pt')
            cells[name]=dict(completed=True,steps=cfg['steps'],initial_sha256=initial,plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
                checkpoint_sha256=sha(folder/'final.pt'),parameters=sum(p.numel() for p in m.parameters()),history=history)
            write(folder/'COMPLETE.json',cells[name])
    for seed in cfg['seeds']:
        for key in ['initial_sha256','plan_sha256']:assert len({cells[f'{a}_s{seed}'][key] for a in ARMS})==1
    write(OUT/'TRAIN_COMPLETE.json',dict(cells=cells,elapsed_seconds=time.time()-start,optimizer_steps=cfg['steps']*len(cells)))
if __name__=='__main__':main()
