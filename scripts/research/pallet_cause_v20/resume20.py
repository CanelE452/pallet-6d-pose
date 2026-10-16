"""Prespecified causal repair comparison; real results never used for fitting."""
from context20 import *
from model20 import CauseGate,objective,encode_direction
from cli_train import state_sha

def reproduce(model,seed,O,rep):
    previous=torch.load(T.RUN/'models'/f'line_bce_seed{seed}'/'final.pt',weights_only=True)['state_dict'];mx=max(float((v.cpu()-previous[k]).abs().max()) for k,v in model.state_dict().items());reference=CauseGate(seed).cuda().eval();reference.load_state_dict(previous);point_max=0.;alpha_max=0.
    with torch.no_grad():
        for first in range(0,len(O['points']),32):
            sl=slice(first,first+32);a=model(subset(O,sl),subset(rep,sl),'raw_weighted');b=reference(subset(O,sl),subset(rep,sl),'raw_weighted');point_max=max(point_max,float((a['points']-b['points']).abs().max()));alpha_max=max(alpha_max,float((a['alpha']-b['alpha']).abs().max()))
    assert mx<1e-4 and point_max<.001 and alpha_max<1e-4,(mx,point_max,alpha_max)
    return dict(seed=seed,max_parameter_difference=mx,training_point_difference_px=point_max,training_alpha_difference=alpha_max,bitwise_equal=state_sha(model.state_dict())==state_sha(previous),scope='Numerical reproduction on all training observations; not bitwise equality.')

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    assert sha(KIT/'train20.py')==read(RUN/'TRAIN_PROTOCOL.json')['sources'][str((KIT/'train20.py').relative_to(ROOT))]
    write(RUN/'RESUME_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),reason='Parameter exactness check stopped at seed2: max1.1474e-6 versus1e-6 bound; training loss differences begin at float32 roundoff size ~1.19e-7. Completed checkpoints kept. Require output agreement on all synthetic training observations: <.001px and alpha<1e-4; check parameter difference<1e-4 as supporting evidence. No GT/performance threshold adjustment.'))
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
            directory=RUN/'models'/f'{arm}_seed{seed}'
            if (directory/'COMPLETE.json').exists():
                done=read(directory/'COMPLETE.json');assert sha(directory/'final.pt')==done['sha256'];model.load_state_dict(torch.load(directory/'final.pt',weights_only=True)['state_dict']);runs.append(done);checks.append(dict(arm=arm,seed=seed,initial_half=True,invertible_encoding=True,sign_invariance=True,finite_updates=2000,reused_completed_checkpoint=True))
                if arm=='raw_weighted':reproduce.append(globals()['reproduce'](model,seed,O,rep))
                del model
                continue
            directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=ph,ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss,error,_=objective(out,gt[ii],valid[ii],arm);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);checks.append(dict(arm=arm,seed=seed,initial_half=True,invertible_encoding=True,sign_invariance=True,finite_updates=2000))
            if arm=='raw_weighted':
                reproduce.append(globals()['reproduce'](model,seed,O,rep))
            del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks));write(RUN/'REPRODUCTION.json',dict(status='PASS',rows=reproduce));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=36000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
