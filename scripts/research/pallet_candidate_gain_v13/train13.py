"""Fixed-budget candidate gain training on frozen synthetic proposals."""
from context13 import *
from scorer13 import GainScorer,objective
from cli_train import state_sha

def true_gains(cand,gt):
    error=torch.linalg.vector_norm(cand-gt[:,:,None],dim=-1)
    return error[...,:1]-error

def main():
    setup();assert read(RUN/'PREPARATION_COMPLETE.json')['complete'];lock=RUN/'TRAIN_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite training')
    ids,O=training_observations('cuda:0');gt,valid=training_targets('cuda:0');assert ids==read(RUN/'TRAIN_IDS.json')['ids']
    write(lock,dict(steps=2000,batch=16,arms=ARMS,seeds=SEEDS,training_frames=len(gt),
        sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},
        protocol_sha256=sha(RUN/'PROTOCOL.json'),train_ids_sha256=sha(RUN/'TRAIN_IDS.json'),
        parent_cache_sha256={str(s):sha(RUN/'parents'/'train'/'mixed'/f'seed{s}.pt') for s in SEEDS},
        training_target_sha256={str(p.relative_to(ROOT)):sha(p) for p in [P.V10/'targets_train.json',V12/'targets_train_corner20.json']+[V11/'targets'/f'train_{c}.json' for c in CONDITIONS[1:-1]]},
        loss='SmoothL1 beta1px of candidate improvement + .25 soft candidate preference CE. No gradients enter parent networks.',
        optimizer='AdamW lr3e-4 weight_decay1e-4, float32, finite gradient clip5. Fixed final checkpoints, no evaluation selection.'))
    smoke_c=torch.tensor([[[[0.,0.],[5.,0.],[10.,0.]]]])
    for target,choice in [([0.,0.],0),([5.,0.],1),([10.,0.],2)]:
        gains=true_gains(smoke_c,torch.tensor([[target]]));assert int(gains.argmax(-1))==choice
    completed=[];started=time.monotonic()
    for seed in SEEDS:
        rep=parent_cache('train','mixed',seed,'cuda:0');cand=candidates(rep);truth=true_gains(cand,gt).detach()
        assert torch.equal(truth[...,0],torch.zeros_like(truth[...,0])) and torch.equal(cand[:,:,0],rep['base'])
        rng=np.random.default_rng(13000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000],dtype=np.int64).reshape(2000,16);initial=None
        for arm in ARMS:
            d=RUN/'models'/f'{arm}_seed{seed}';d.mkdir(parents=True,exist_ok=False)
            model=GainScorer(seed).cuda();state=state_sha(model.state_dict())
            if initial is None:initial=state
            assert state==initial
            with torch.no_grad():
                out=model(subset(O,slice(0,8)),subset(rep,slice(0,8)),cand[:8],arm)
                assert torch.equal(out['points'],rep['base'][:8]) and (out['choice']==0).all()
            write(d/'START.json',dict(seed=seed,arm=arm,initial_state_sha256=state,plan_sha256=digest(plan.tobytes().hex()),
                parameters=sum(p.numel() for p in model.parameters()),initial_exact_base=True,
                active_capacity='constant only3 scores active; geometry leaves image projection inactive and zeroes image/context inputs; shared scorer architecture otherwise identical.'))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (d/'TRACE.jsonl').open('w') as f:
                for step in range(2000):
                    ii=torch.tensor(plan[step],device='cuda:0');out=model(subset(O,ii),subset(rep,ii),cand[ii],arm)
                    loss,reg,rank=objective(out['gain'],truth[ii],valid[ii]);opt.zero_grad(set_to_none=True);loss.backward()
                    gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),regression=float(reg.detach()),ranking=float(rank.detach()),gradient_norm=gn))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,'loss',round(float(loss.detach()),3),'sec',round(time.monotonic()-tick),flush=True);f.flush()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),d/'final.pt')
            receipt=dict(complete=True,seed=seed,arm=arm,steps=2000,elapsed_seconds=time.monotonic()-tick,checkpoint_sha256=sha(d/'final.pt'))
            write(d/'COMPLETE.json',receipt);completed.append(receipt);del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',keep_half_full_target_sign_and_argmax=True,all_initial_outputs_exact_base=True,all_finite_training_gradients=True,
        candidate0_true_and_predicted_gain_exact_zero=True))
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,runs=completed,updates=18000,elapsed_seconds=time.monotonic()-started))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
