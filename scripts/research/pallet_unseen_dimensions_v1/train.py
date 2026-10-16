import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common import *
from model import Residual
import torch,numpy as np,time
import torch.nn.functional as F

def main():
    cfg=register();assert read(OUT/'CACHE_COMPLETE.json')['GT_model_inputs'] is False
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    z={k:v.cuda() for k,v in torch.load(OUT/'train.pt',weights_only=True).items()}
    rows=read(OUT/'train_records.json');assert {r['asset'] for r in rows}==set(ASSETS[:2])
    target=torch.tensor(np.array([r['gt'] for r in rows]),dtype=torch.float32,device='cuda')
    lock=dict(protocol_sha256=sha(OUT/'PROTOCOL.json'),train_sha256=sha(OUT/'train.pt'),
              sources={n:sha(KIT/n) for n in ['common.py','render.py','cache.py','model.py','train.py']})
    write(OUT/'TRAIN_LOCK.json',lock)
    # Identity and useful-gradient preflight at nonzero geometric relations.
    small={k:v[:8] for k,v in z.items()};probe=Residual('C_geometry',1).cuda()
    assert torch.equal(probe(small),small['stock'])
    torch.nn.init.normal_(probe.output.weight,std=.001)
    F.smooth_l1_loss(probe(small)[:,:8],target[:8],beta=4).backward()
    grad=sum(float(p.grad.abs().sum()) for p in probe.relation.parameters() if p.grad is not None)
    assert grad>1e-8,grad
    write(OUT/'PREFLIGHT.json',dict(identity_exact=True,relation_gradient_sum=grad,GT_keys_absent=True))
    del probe
    start=time.time();cells={}
    for seed in cfg['seeds']:
        rng=np.random.default_rng(seed);plan=[]
        while len(plan)<cfg['steps']*cfg['batch']:plan.extend(rng.permutation(len(rows)).tolist())
        plan=np.array(plan[:cfg['steps']*cfg['batch']]).reshape(cfg['steps'],cfg['batch'])
        for arm in ARMS:
            name=f'{arm}_s{seed}';path=OUT/'models'/name;path.mkdir(parents=True,exist_ok=True)
            if (path/'COMPLETE.json').exists():cells[name]=read(path/'COMPLETE.json');continue
            m=Residual(arm,seed).cuda();opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            hist=[]
            for step,idx in enumerate(plan):
                batch={k:v[idx] for k,v in z.items()};opt.zero_grad(set_to_none=True)
                loss=F.smooth_l1_loss(m(batch)[:,:8],target[idx],beta=4)
                assert torch.isfinite(loss);loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(m.parameters(),10,error_if_nonfinite=True);opt.step()
                if (step+1)%200==0:
                    hist.append(dict(step=step+1,loss=float(loss.detach()),grad=float(gn)))
                    print(name,step+1,'loss',round(float(loss.detach()),4),flush=True)
            torch.save({k:v.cpu() for k,v in m.state_dict().items()},path/'final.pt')
            cells[name]=dict(steps=cfg['steps'],plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),
                parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(path/'final.pt'),history=hist)
            write(path/'COMPLETE.json',cells[name])
    write(OUT/'TRAIN_COMPLETE.json',dict(cells=cells,elapsed_seconds=time.time()-start,optimizer_steps=cfg['steps']*len(cells)))
if __name__=='__main__':main()
