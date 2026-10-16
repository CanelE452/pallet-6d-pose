"""Exploratory proposal-conditioning intervention and matched observation-budget long control."""
from context20 import *
from model20 import CauseGate,objective
from cli_train import state_sha
CF_ARMS=('unit_weighted_long','unit_weighted_cf','unit_weighted_cf_local')

def main():
    setup()
    if (RUN/'CF_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'CF_PROTOCOL.json',dict(stage='Exploratory followup after initial2x2 results; not original preregistration.',hypothesis='One frozen proposal per point lets image-conditioned mean retention substitute for direction comparison. Present same image/base with original, reverse and two perpendicular directions, recompute alpha_star from existing synthetic GT; compare more original-only optimization.',arms=CF_ARMS,seeds=SEEDS,
      training='CF2000updates with four copies of each original grouped batch: d,-d,rot90(d),rot-90(d). All GT and point IDs retained. Same movement lengths; weighted BCE averaged across copies. Long control8000updates using the same grouped batches cyclically four times. Exactly same source-observation forward count per arm, but update counts differ. Same init/optimizer/lr/fixed final checkpoint. CF local zeros new line token.',
      inference='Always original frozen V18 proposal; no augmented candidates, target or selection at inference. Learned scalar still only0..1. No GT filtering or selftraining.',
      acceptance=read(RUN/'PROTOCOL.json')['acceptance'],evaluation=read(RUN/'PROTOCOL.json')['evaluation'],source_sha256=sha(Path(__file__)),model_sha256=sha(KIT/'model20.py'),initial_protocol_sha256=sha(RUN/'PROTOCOL.json')));runs=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);g=read(RUN/f'TRAIN_GROUPS_seed{seed}.json');groups=g['groups'];names=sorted(groups);rng=np.random.default_rng(19000+seed);seq=[]
        while len(seq)<6000:seq.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in seq[s:s+3]],[]) for s in range(0,6000,3)];assert digest(json.dumps(plan))==g['plan_sha256']
        for arm in CF_ARMS:
            mode='unit_weighted' if arm!='unit_weighted_cf_local' else 'unit_equal_local';steps=8000 if arm.endswith('_long') else 2000;model=CauseGate(seed).cuda();ih=state_sha(model.state_dict());assert ih==read(RUN/'models'/f'unit_weighted_seed{seed}'/'START.json')['initial_sha256'];folder=RUN/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False);write(folder/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters()),steps=steps));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic();count=0
            with (folder/'TRACE.jsonl').open('w') as f:
                for step in range(steps):
                    ii=torch.tensor(plan[step%2000],device='cuda:0');o=subset(O,ii);r=subset(rep,ii);y=gt[ii];v=valid[ii]
                    if not arm.endswith('_long'):
                        o={k:torch.cat([z]*4) for k,z in o.items()};r={k:torch.cat([z]*4) for k,z in r.items()};delta=rep['proposal_delta'][ii];turn=torch.stack((-delta[...,1],delta[...,0]),-1);r['proposal_delta']=torch.cat((delta,-delta,turn,-turn));y=y.repeat(4,1,1);v=v.repeat(4,1)
                        # Rotations preserve amplitude and the target projection is always recomputed.
                        assert torch.allclose(r['proposal_delta'].norm(dim=-1),delta.norm(dim=-1).repeat(4,1),atol=1e-5)
                    out=model(o,r,mode);loss,error,_=objective(out,y,v,'unit_weighted');opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();count+=len(y);f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(y)))+'\n')
                    if (step+1)%(2000 if steps==8000 else 1000)==0:print('CF TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            ck=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=steps),ck);done=dict(arm=arm,seed=seed,steps=steps,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick,observation_forwards=count);write(folder/'COMPLETE.json',done);runs.append(done);del model,opt
    for seed in SEEDS:assert len({r['observation_forwards'] for r in runs if r['seed']==seed})==1
    write(RUN/'CF_TRAINING_COMPLETE.json',dict(complete=True,updates=sum(r['steps'] for r in runs),runs=runs));print('CF TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
