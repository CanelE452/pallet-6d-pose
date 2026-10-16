"""Matched evaluator budgets. The former calibration set is now training only."""
from context14 import *
from model14 import FusionEvaluator,objective
from cli_train import state_sha

def main():
    setup();assert read(RUN/'PREPARATION_COMPLETE.json')['complete'];lock=RUN/'TRAIN_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite registered training')
    protocol=read(RUN/'PROTOCOL.json');assert ARMS==tuple(protocol['arms'])
    write(lock,dict(steps=2000,batch=16,arms=ARMS,seeds=SEEDS,protocol_sha256=sha(RUN/'PROTOCOL.json'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},
        input_files={str(p.relative_to(ROOT)):sha(p) for p in (RUN/'targets').glob('*.json')},
        preparation_sha256=sha(RUN/'PREPARATION_COMPLETE.json'),optimizer='AdamW3e-4 wd1e-4 float32 clip5; fixed final checkpoints, no evaluation selection.'))
    started=time.monotonic();completed=[];checks=[]
    for seed in SEEDS:
        initial=None;heldout_plan=None
        for arm in ARMS:
            split='insample' if arm=='insample_fusion' else 'heldout';O,rep,gt,valid,ids=training(split,seed)
            rng=np.random.default_rng(14000+seed);plan=[]
            while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
            plan=np.array(plan[:32000],dtype=np.int64).reshape(2000,16);plan_sha=digest(plan.tobytes().hex())
            if split=='heldout':
                if heldout_plan is None:heldout_plan=plan_sha
                assert heldout_plan==plan_sha
            model=FusionEvaluator(seed).cuda();initialsha=state_sha(model.state_dict())
            if initial is None:initial=initialsha
            assert initial==initialsha
            with torch.no_grad():
                o=subset(O,slice(0,4));r=subset(rep,slice(0,4));p=model(o,r,arm)
                assert torch.equal(p['points'],r['base']) and (p['choice']==0).all()
                assert torch.equal(p['gain'][...,0],torch.zeros_like(p['gain'][...,0]))
                off=model(o,r,arm,line_off=True);assert (off['choice']<3).all() and not off['eligible'][...,3:].any()
                # Label algebra with every possible winner, not tied to real targets.
                toy=torch.arange(5,dtype=torch.float32).reshape(1,1,5,1).expand(-1,-1,-1,2)*5
                for k in range(5):
                    e=(toy-toy[:,:,k:k+1]).norm(dim=-1);assert int((e[...,:1]-e).argmax(-1))==k
            d=RUN/'models'/f'{arm}_seed{seed}';d.mkdir(parents=True,exist_ok=False)
            write(d/'START.json',dict(arm=arm,seed=seed,split=split,frames=len(gt),ids_sha256=digest(json.dumps(ids)),initial_state_sha256=initialsha,
                plan_sha256=plan_sha,parameters=sum(p.numel() for p in model.parameters()),exact_keep_initialization=True,
                active_capacity='Same head tensor shapes; point control masks candidates3/4 and zeros6 line features; effective active inputs differ.'))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (d/'TRACE.jsonl').open('w') as f:
                for step in range(2000):
                    ii=torch.tensor(plan[step],device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm)
                    loss,reg,rank=objective(out,gt[ii],valid[ii]);assert torch.isfinite(loss)
                    opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),regression=float(reg.detach()),ranking=float(rank.detach()),gradient_norm=gn))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,'loss',round(float(loss.detach()),3),'sec',round(time.monotonic()-tick),flush=True);f.flush()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),d/'final.pt')
            done=dict(complete=True,arm=arm,seed=seed,steps=2000,elapsed_seconds=time.monotonic()-tick,checkpoint_sha256=sha(d/'final.pt'))
            write(d/'COMPLETE.json',done);completed.append(done);checks.append(dict(seed=seed,arm=arm,initial_exact_keep=True,finite_gradients=True));del model,opt,O,rep,gt,valid
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks,toy_five_candidate_gain_sign=True,line_off_candidate_mask=True))
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,runs=completed,updates=18000,elapsed_seconds=time.monotonic()-started))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
