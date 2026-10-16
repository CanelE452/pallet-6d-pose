"""Training-probe-driven amendment registered before any V23 evaluation."""
from context23 import *
from train23 import training
from model20 import CauseGate,oracle_alpha
from cli_train import state_sha
import torch.nn.functional as F
MASS_ARMS=('unit_mass','unit_mass_local')
def mass_weights(r,gt,valid,ids,kind):
    before=(r['base']-gt).norm(dim=-1);bins=torch.bucketize(before.contiguous(),torch.tensor([5.,15.],device=gt.device),right=False);move=r['proposal_delta'].norm(dim=-1);mask=valid&(kind==2)&torch.tensor([x.endswith('::crop20') for x in ids],device=gt.device)[:,None]
    mass=torch.stack([move[mask&(bins==i)].sum() for i in range(3)]);f=(mass.sum()/(3*mass)).clamp(.125,8);w=torch.ones_like(before);w[mask]=f[bins[mask]];norm=(w[mask]*move[mask]).sum()/move[mask].sum();w[mask]/=norm
    assert (w>0).all() and abs(float((w[mask]*move[mask]).sum()/move[mask].sum())-1)<1e-5
    return w,dict(counts=[int((mask&(bins==i)).sum()) for i in range(3)],original_mass=mass.tolist(),factors=(f/norm).tolist(),weighted_mass=[float((move*w)[mask&(bins==i)].sum()) for i in range(3)])
def register():
    assert not (RUN/'EVAL_PROTOCOL.json').exists();assert not (RUN/'MASS_PROTOCOL.json').exists()
    write(RUN/'MASS_PROTOCOL.json',dict(arms=MASS_ARMS,seeds=SEEDS,steps=2000,updates=12000,registered_before_V23_evaluation=True,training_probe_sha256=sha(RUN/'TRAIN_QUALITY.json'),reason='Accurate points already28% of mild outside train points; count balancing only mildly changes weights, while movement-weighted BCE mass remains dominated by large corrections. Add synthetic-only bin coefficient-mass balance to distinguish counting from effective loss emphasis.',weights='Same fixed base-error bins and same target crop20outside. Per bin factor=sum movement/(3*bin movement), clamp[.125,8], renormalize so total weighted movement coefficient equals original. No negative/zero weights; other targets exactly1. Coefficient mass, not actual loss/gradient magnitude, is balanced.',control='Same intact old+new observations, parent predictions, initialization and2000group batches as unit_expand. Extra-line/local paired. Real and all evaluation GT unused for registration/weight fitting.',source_sha256=sha(Path(__file__))))
def main():
    setup();assert read(RUN/'TRAINING_COMPLETE.json')['complete'];assert sha(Path(__file__))==read(RUN/'MASS_PROTOCOL.json')['source_sha256'];runs=[];probes=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=training(seed);w,probe=mass_weights(r,gt,valid,ids,kind);probes.append(dict(seed=seed,**probe));g=read(RUN/f'GROUPS_seed{seed}.json');assert g['ids']==ids;print('MASS',seed,probe,flush=True)
        for arm in MASS_ARMS:
            model=CauseGate(seed).cuda();directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=state_sha(model.state_dict()),plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids))))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(g['plan']):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);target=oracle_alpha(out['base'],out['proposal_delta'],gt[ii]).detach();terms=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none')*out['proposal_delta'].norm(dim=-1).detach();loss=(terms*w[ii])[valid[ii]].mean();error=(out['points']-gt[ii]).norm(dim=-1)[valid[ii]].mean();opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,flush=True);f.flush()
            p=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),p);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(p),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);del model,opt
        del O,r,gt,valid,w
    write(RUN/'MASS_WEIGHTS.json',dict(rows=probes));write(RUN/'MASS_TRAINING_COMPLETE.json',dict(complete=True,updates=12000,runs=runs));print('MASS TRAINING COMPLETE',flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');a=ap.parse_args();register() if a.register else main()
