"""Prespecified causal repair comparison; real results never used for fitting."""
from context20 import *
from model20 import CauseGate,objective,encode_direction
from cli_train import state_sha

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'PROTOCOL.json',dict(stage='New exploratory DEV experiment chosen using training-only V20 probe and earlier V19 findings; no independent final holdout.',hypotheses=['Movement weighting suppresses stop-target contribution.','Tiny proposal coordinates are not encoded strongly enough relative to appearance/context.'],probe_sha256=sha(RUN/'PROBE.json'),arms=ARMS,seeds=SEEDS,steps=2000,
      factors='Raw versus unit-direction/radius-fraction encoding x movement-weighted versus equal-point BCE. All4 use same line context. unit_equal_area and unit_equal_local test extra line context against same sample-budget area and no extra context.',
      encoding='Original 228descriptor unchanged except amodal_delta/boxdiag scaled by1/.25, proposal d/boxdiag replaced with d/|d| (epsilon1e-8 normalized length), proposal length/boxdiag scaled by1/.15. Fixed .25/.15 are existing amodal/vector radius definitions, not GT-fitted scales. For nonzero d information is preserved; no new GT information.',
      architecture='Frozen V18 line_vector, same RetentionGate weights/tensors as V19. points=base+sigmoid(logit)*frozen_delta. New trainable head learns retained amount, not an external filter. No self-training or real GT forward.',
      training='253synthetic/1462detected variants/6conditions. Same original-ID grouped batches (3originals, up to18observations), initialization seed19000+seed and2000updates as V19 BCE. AdamW3e-4 wd1e-4 float32 clip5; last checkpoint. Every original valid point retained.',
      loss='BCE-with-logits against original synthetic GT projected alpha_star. Weight either proposal movement or1 per valid point. Equal weights alter relative emphasis; no benefit/harm GT selection or sampling.',
      acceptance='Everyseed clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% V18 line_vector benefit. Pairwise effect requires everyseed>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.',
      evaluation='Immutable real297/9sessions/1251manual corners + synth505,8conditions. Original IDs/detections/point GT; artificial stress. crop10/30 evaluation-only. All arms evaluated/reported, no real-driven model selection.',
      interventions='unit_equal actual versus zero line token and donor line-source features. Point patch/base/proposal/line geometry stay actual. OOD diagnostic, not proof of useful semantics.',
      reproduction='raw_weighted must reproduce V19 line_bce final state with same seed and batches before conclusions.'))
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},targets=read(T.RUN/'TRAIN_PROTOCOL.json')['targets'],parents=read(T.RUN/'TRAIN_PROTOCOL.json')['parents'],model19_sha256=sha(T.KIT/'model19.py'),context19_sha256=sha(T.KIT/'context19.py'),steps=2000))
    runs=[];checks=[];reproduce=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);g=read(T.RUN/f'TRAIN_GROUPS_seed{seed}.json');assert ids==g['ids'];groups=g['groups'];names=sorted(groups);rng=np.random.default_rng(19000+seed);sequence=[]
        while len(sequence)<6000:sequence.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in sequence[s:s+3]],[]) for s in range(0,6000,3)];ph=digest(json.dumps(plan));assert ph==g['plan_sha256'];write(RUN/f'TRAIN_GROUPS_seed{seed}.json',g)
        for arm in ARMS:
            model=CauseGate(seed).cuda();ih=state_sha(model.state_dict());assert ih==read(T.RUN/'models'/f'line_bce_seed{seed}'/'START.json')['initial_sha256'];sl=slice(0,16)
            with torch.no_grad():
                out=model(subset(O,sl),subset(rep,sl),arm);assert torch.all(out['alpha']==.5);assert torch.equal(out['points'],rep['base'][sl]+.5*rep['proposal_delta'][sl]);flip=model(subset(O,sl),dict(subset(rep,sl),lines=-rep['lines'][sl]),arm);assert torch.allclose(out['points'],flip['points'],atol=1e-5,rtol=0)
                x=torch.randn(20,228,device='cuda:0');x[:,227]=x[:,225:227].norm(dim=-1);y=encode_direction(x);assert torch.allclose(y[:,225:227]*y[:,227:228]*.15,x[:,225:227],atol=1e-6);assert torch.allclose(y[:,223:225]*.25,x[:,223:225],atol=1e-6)
            directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=ph,ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss,error,_=objective(out,gt[ii],valid[ii],arm);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);checks.append(dict(arm=arm,seed=seed,initial_half=True,invertible_encoding=True,sign_invariance=True,finite_updates=2000))
            if arm=='raw_weighted':
                previous=torch.load(T.RUN/'models'/f'line_bce_seed{seed}'/'final.pt',weights_only=True)['state_dict'];mx=max(float((v.cpu()-previous[k]).abs().max()) for k,v in model.state_dict().items());assert mx<1e-6;reproduce.append(dict(seed=seed,max_parameter_difference=mx,state_sha256=state_sha(model.state_dict()),previous_state_sha256=state_sha(previous)))
            del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks));write(RUN/'REPRODUCTION.json',dict(status='PASS',rows=reproduce));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=36000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
