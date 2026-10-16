"""Exploratory synthetic base-quality intervention; real GT never enters training."""
from context20 import *
from model20 import CauseGate,objective
from model18 import VectorRefiner
from cli_train import state_sha
QUALITY_ARMS=('unit_weighted_quality','unit_weighted_quality_local')

def main():
    setup()
    if (RUN/'QUALITY_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'QUALITY_PROTOCOL.json',dict(stage='Exploratory followup after point-ID matched residual distribution diagnosis; not confirmatory holdout.',hypothesis='Synthetic baseline errors are much larger than real mild-crop errors. Include already correct and slightly wrong synthetic base points so the retention model learns no-op cases at appropriate query positions.',
      augmentation='Four views of each synthetic observation: original base, GT base, GT+.01*boxdiag*unit_random_direction, GT+.03*boxdiag*unit_random_direction. Fractions fixed to span correct/near-correct points, not fitted to real optimal scales. Directions fixed per seed/observation/point. Only synthetic valid points are changed. Frozen V18 proposal is recomputed for each changed base. Original features/global parent context/amodal output/line associations remain frozen; this is counterfactual base augmentation, not a claim to reproduce the entire parent distribution.',
      training='2000updates, each original grouped batch expanded4times; same original-ID plan/seed/optimizer as previous. Movement-weighted BCE from original synthetic GT, all original valid points kept. Same observation-forward count as CF and8000step original-only long control; optimizer step counts differ versus long. local zeros extra line token.',
      arms=QUALITY_ARMS,seeds=SEEDS,steps=2000,source_sha256=sha(Path(__file__)),model_sha256=sha(KIT/'model20.py'),diagnosis_sha256=sha(RUN/'POPULATION_DIAGNOSIS.json'),acceptance=read(RUN/'PROTOCOL.json')['acceptance'],evaluation=read(RUN/'PROTOCOL.json')['evaluation'],inference='Only original frozen parent/V18 outputs, no GT, synthetic perturbation or fallback rule at inference. All models evaluated, no real checkpoint selection.'));runs=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);g=read(RUN/f'TRAIN_GROUPS_seed{seed}.json');groups=g['groups'];names=sorted(groups);rng=np.random.default_rng(19000+seed);seq=[]
        while len(seq)<6000:seq.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in seq[s:s+3]],[]) for s in range(0,6000,3)];assert digest(json.dumps(plan))==g['plan_sha256'];jitter_rng=np.random.default_rng(20100+seed);angles=torch.tensor(jitter_rng.uniform(-np.pi,np.pi,size=(3,len(gt),8)),dtype=gt.dtype,device=gt.device);direction=torch.stack((angles.cos(),angles.sin()),-1);diag=(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1);bases=[rep['base']];deltas=[rep['proposal_delta']]
        proposer=VectorRefiner(seed).cuda().eval().requires_grad_(False);proposer.load_state_dict(torch.load(Q.RUN/'models'/f'line_vector_seed{seed}'/'final.pt',weights_only=True)['state_dict'])
        with torch.no_grad():
            for j,radius in enumerate((0.,.01,.03)):
                base=torch.where(valid[...,None],gt+radius*diag[:,None,None]*direction[j],rep['base']);parts=[]
                for first in range(0,len(gt),32):
                    sl=slice(first,first+32);r=subset(rep,sl);r['base']=base[sl];parts.append(proposer(subset(O,sl),r,'line_vector')['points']-base[sl])
                bases.append(base);deltas.append(torch.cat(parts))
        del proposer;torch.save(dict(bases=torch.stack(bases).cpu(),deltas=torch.stack(deltas).cpu(),ids=ids,scope='Synthetic counterfactual training only, same frozen image/parent context.'),RUN/f'QUALITY_VIEWS_seed{seed}.pt')
        for arm in QUALITY_ARMS:
            model=CauseGate(seed).cuda();ih=state_sha(model.state_dict());assert ih==read(RUN/'models'/f'unit_weighted_seed{seed}'/'START.json')['initial_sha256'];folder=RUN/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False);write(folder/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic();count=0
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');o={k:torch.cat([v[ii]]*4) for k,v in O.items()};r={k:torch.cat([v[ii]]*4) for k,v in rep.items()};r['base']=torch.cat([v[ii] for v in bases]);r['proposal_delta']=torch.cat([v[ii] for v in deltas]);y=gt[ii].repeat(4,1,1);v=valid[ii].repeat(4,1);out=model(o,r,arm);loss,error,_=objective(out,y,v,'unit_weighted');opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();count+=len(y);f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(y)))+'\n')
                    if (step+1)%1000==0:print('QUALITY TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            ck=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick,observation_forwards=count);write(folder/'COMPLETE.json',done);runs.append(done);del model,opt
    for r in runs:assert r['observation_forwards']==next(x['observation_forwards'] for x in read(RUN/'CF_TRAINING_COMPLETE.json')['runs'] if x['seed']==r['seed'])
    write(RUN/'QUALITY_TRAINING_COMPLETE.json',dict(complete=True,updates=12000,runs=runs,view_hashes={str(p.relative_to(ROOT)):sha(p) for p in RUN.glob('QUALITY_VIEWS_seed*.pt')}));print('QUALITY TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
