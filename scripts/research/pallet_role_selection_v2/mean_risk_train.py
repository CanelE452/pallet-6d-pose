"""One predeclared objective correction: conditional mean risk vs robust regression."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from common import *
from model import *
import numpy as np,time

def register_followup():
    p=OUT/'mean_risk_followup'/'PROTOCOL.json'
    if p.exists():return read(p)
    o=dict(reason='Huber is not a proper conditional-mean estimator; minority large ID errors can be downweighted. Train-only audit found193 large-fixable samples, only7/14/14 fixed by original selector.',
        frozen_before_followup_evaluation=True,prior_summary_sha256=sha(OUT/'SUMMARY.json'),
        exact_change='Replace selector smooth L1 beta.05 with squared error of the same normalized candidate-error target. Same selector initialization, data order,1200steps,batch32,AdamWlr.0003wd.0001. No other loss or threshold sweep.',
        residual='Reuse corresponding trained C residual exactly, freeze all residual parameters. This isolates selector correction from coordinate correction.',
        selector_gradient_clip=10,previous_gradient_clip='Original clipped combined norm; selector/residual weights were nevertheless exactly independent of residual inclusion in observed training. Followup freezes residual.',
        seeds=[1,2,3],fresh_scene_seed_base=851000,observations=576,
        previous_fresh_is_now='diagnostic only; do not select followup on it',
        primary='same all-frame normalized error, unseen meshes; compare squared-risk C to stock and original robust-risk C',
        acceptance='all seeds >=2pct improvement vs stock and paired scene CI upper<0; no new stock<=10 to>20px frames across all meshes; every seen/unseen condition <=1.02stock',
        no_architecture_or_optimizer_sweep=True)
    write(p,o);return o

def main():
    cfg=register_followup();folder=OUT/'mean_risk_followup';torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    z={k:v.cuda() for k,v in torch.load(PREVIOUS/'train.pt',weights_only=True).items()}
    gt=torch.tensor(np.array([r['gt'] for r in read(PREVIOUS/'train_records.json')]),dtype=torch.float32,device='cuda')
    cells={};start=time.time()
    for seed in cfg['seeds']:
        m=SelectorResidual('C_select_residual',seed).cuda()
        old=torch.load(OUT/'models'/f'C_select_residual_s{seed}'/'final.pt',weights_only=True)
        state=m.state_dict()
        for name in state:
            if not name.startswith('risk'):state[name]=old[name].to('cuda')
        m.load_state_dict(state)
        for name,p in m.named_parameters():p.requires_grad_(name.startswith('risk'))
        parameters=[p for p in m.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=.0003,weight_decay=.0001)
        rng=np.random.default_rng(seed);plan=[]
        while len(plan)<1200*32:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:1200*32]).reshape(1200,32);history=[]
        for step,idx in enumerate(plan):
            opt.zero_grad(set_to_none=True);o=m({k:v[idx] for k,v in z.items()})
            target=(o['candidates'][:,:,:8]-gt[idx,None]).norm(dim=-1).mean(-1)/o['scale'][:,None]
            loss=F.mse_loss(o['risk'],target);assert torch.isfinite(loss);loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(parameters,10,error_if_nonfinite=True);opt.step()
            if (step+1)%300==0:
                history.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(gn)));print('MEAN_RISK',seed,history[-1],flush=True)
        state={k:v.cpu() for k,v in m.state_dict().items()}
        assert all(torch.equal(state[k],old[k]) for k in state if not k.startswith('risk'))
        path=folder/'models'/f'seed{seed}.pt';path.parent.mkdir(exist_ok=True);torch.save(state,path)
        cells[str(seed)]=dict(checkpoint_sha256=sha(path),residual_exactly_frozen=True,history=history,
            plan_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),source_sha256=sha(KIT/'mean_risk_train.py'))
    write(folder/'TRAIN_COMPLETE.json',dict(cells=cells,optimizer_steps=3600,elapsed_seconds=time.time()-start))
if __name__=='__main__':main()
