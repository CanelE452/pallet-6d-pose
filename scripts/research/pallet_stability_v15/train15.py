"""Loss-only control chain plus one richer-candidate arm, synthetic-only."""
from context15 import *
from model15 import StableEvaluator,objective,weights
from cli_train import state_sha

def main():
    setup();assert (RUN/'PROTOCOL.json').exists();lock=RUN/'TRAIN_PROTOCOL.json'
    if lock.exists():raise SystemExit('Refusing to overwrite training')
    write(lock,dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),arms=ARMS,seeds=SEEDS,steps=2000,batch=16,
        sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},
        targets={str(p.relative_to(ROOT)):sha(p) for p in (V14/'targets').glob('heldout_*.json')},
        parents={str(p.relative_to(ROOT)):sha(p) for p in (V14/'parents/heldout').glob('*/seed*_actual.pt')},
        optimizer='AdamW3e-4 weight_decay1e-4 float32 clip5, final checkpoint only',no_real_training=True))
    completed=[];preflight=[];started=time.monotonic()
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);cand=candidates(O,rep);error=(cand-gt[:,:,None]).norm(dim=-1);truth=(error[...,:1]-error).detach()
        w,counts=weights(truth,valid,kind);assert torch.isfinite(cand).all() and torch.isfinite(w).all()
        flipped=dict(rep,lines=-rep['lines']);assert torch.allclose(candidates(O,flipped),cand,atol=1e-5,rtol=0)
        radius=.05*(O['box'][:,2:]-O['box'][:,:2]).norm(dim=-1).clamp_min(1)
        assert ((cand[...,5:,:]-rep['base'][:,:,None]).norm(dim=-1)<=radius[:,None,None]+1e-4).all()
        write(RUN/f'TRAIN_DIAGNOSIS_seed{seed}.json',dict(frames=len(ids),counts=counts,oracle_old_px=float(error[...,:5].min(-1).values[valid].mean()),
            oracle_rich_px=float(error.min(-1).values[valid].mean()),scope='Training-only feasibility diagnostic, no arm selection or evaluation targets.'))
        rng=np.random.default_rng(15000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000],dtype=np.int64).reshape(2000,16);initial=None
        for arm in ARMS:
            model=StableEvaluator(seed).cuda();ih=state_sha(model.state_dict())
            if initial is None:initial=ih
            assert ih==initial
            with torch.no_grad():
                p=model(subset(O,slice(0,4)),subset(rep,slice(0,4)),arm)
                assert torch.equal(p['points'],rep['base'][:4]) and (p['choice']==0).all()
                if arm!='balanced_rich':assert not p['eligible'][...,5:].any()
                p=model(subset(O,slice(0,4)),subset(rep,slice(0,4)),arm,line_off=True);assert not p['eligible'][...,3:].any()
            d=RUN/'models'/f'{arm}_seed{seed}';d.mkdir(parents=True,exist_ok=False)
            write(d/'START.json',dict(arm=arm,seed=seed,frames=len(ids),ids_sha256=digest(json.dumps(ids)),initial_sha256=ih,
                plan_sha256=digest(plan.tobytes().hex()),parameters=sum(p.numel() for p in model.parameters()),tau=.5 if arm=='median_old' else .2,
                balanced=arm.startswith('balanced'),candidate_slots=8,eligible_candidates=8 if arm=='balanced_rich' else 5))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (d/'TRACE.jsonl').open('w') as f:
                for step in range(2000):
                    ii=torch.tensor(plan[step],device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss=objective(out,truth[ii],valid[ii],w[ii],arm)
                    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=gn))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,'loss',round(float(loss.detach()),3),'seconds',round(time.monotonic()-tick),flush=True);f.flush()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=2000),d/'final.pt')
            done=dict(complete=True,arm=arm,seed=seed,steps=2000,elapsed_seconds=time.monotonic()-tick,checkpoint_sha256=sha(d/'final.pt'));write(d/'COMPLETE.json',done);completed.append(done)
            preflight.append(dict(seed=seed,arm=arm,initial_keep=True,finite_gradients=True));del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',line_sign_invariance=True,geometric_step_bound=True,line_off_masks=True,rows=preflight))
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=24000,runs=completed,elapsed_seconds=time.monotonic()-started))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
