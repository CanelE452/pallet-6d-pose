"""Exploratory follow-up: warm-start exact same BCE state, continue BCE versus pixel."""
from context22 import *
from model22 import ObjectiveGate,objective22
from cli_train import state_sha
WARM_ARMS=('unit_warm_bce','unit_warm_pixel')

def main():
    setup();assert read(RUN/'DIAGNOSIS.json')['status']=='PASS';assert not (RUN/'WARM_PROTOCOL.json').exists()
    write(RUN/'WARM_PROTOCOL.json',dict(source_sha256=sha(Path(__file__)),model_sha256=sha(KIT/'model22.py'),arms=WARM_ARMS,seeds=SEEDS,steps=2000,lr=3e-4,weight_decay=1e-4,clip=5,
      hypothesis='Early full-retention saturation may prevent direct-distance learning of useful context. Start from already-trained BCE features/alpha, then compare another2000 BCE versus pixel updates.',
      initialization='Same V22 unit_bce final state per seed, fresh AdamW for both continuations, exactly original2000 grouped updates repeated. Same retained64D context and line samples. No real GT training or checkpoint selection.',
      scope='Exploratory after viewing initial V22 results; six new models12000updates. Not preregistered before initial results or independent final holdout. Reset optimizer in both controls.'))
    runs=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=proposal_training(seed);g=read(RUN/f'GROUPS_seed{seed}.json');assert ids==g['ids'];ck=RUN/'models'/f'unit_bce_seed{seed}'/'final.pt';initial=torch.load(ck,weights_only=True)['state_dict']
        for arm in WARM_ARMS:
            model=ObjectiveGate(seed).cuda();model.load_state_dict(initial);d=RUN/'models'/f'{arm}_seed{seed}';d.mkdir(parents=True,exist_ok=False);write(d/'START.json',dict(seed=seed,arm=arm,initial_sha256=state_sha(model.state_dict()),parent_sha256=sha(ck),plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids))));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (d/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(g['plan']):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);loss,error,_=objective22(out,gt[ii],valid[ii],arm);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            path=d/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000,parent_steps=2000),path);done=dict(arm=arm,seed=seed,steps=2000,parent_steps=2000,sha256=sha(path),elapsed_seconds=time.monotonic()-tick);write(d/'COMPLETE.json',done);runs.append(done);del model,opt
    write(RUN/'WARM_TRAINING_COMPLETE.json',dict(complete=True,updates=12000,runs=runs));print('WARM COMPLETE',flush=True)

if __name__=='__main__':main()
