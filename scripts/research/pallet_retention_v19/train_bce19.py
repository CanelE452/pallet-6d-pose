"""Exploratory loss control after saturation diagnosis; original runs stay immutable."""
from context19 import *
from model19 import RetentionGate,oracle_alpha
from cli_train import state_sha
import torch.nn.functional as F
BCE_ARMS=('local_bce','area_bce','line_bce')

def main():
    setup()
    if (RUN/'BCE_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'BCE_PROTOCOL.json',dict(stage='Exploratory followup after original train/DEV results; not part of initial preregistration.',hypothesis='Movement-weighted L1 on sigmoid alpha has logit gradient proportional to alpha*(1-alpha), weak at wrong saturation. Soft-target BCE-with-logits gradient is alpha-alpha_star. Compare same architecture/initial/batches/budget to distinguish this optimizer/objective limitation from absent context information. Conditional median versus mean objective also changes; not an isolated proof of saturation causality.',arms=BCE_ARMS,seeds=SEEDS,steps=2000,loss='mean(detached proposal movement * binary_cross_entropy_with_logits(logit, synthetic GT alpha_star)) over original valid points; no clamp on logits, no filtering.',selection='No real GT fitting, seed/checkpoint selection or further parameter search. Same final checkpoint, acceptance and evaluation populations. All outcomes reported, initial runs kept.',original_protocol_sha256=sha(RUN/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),model_sha256=sha(KIT/'model19.py'),context_sha256=sha(KIT/'context19.py')));runs=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);groups=read(RUN/f'TRAIN_GROUPS_seed{seed}.json')['groups'];names=sorted(groups);rng=np.random.default_rng(19000+seed);sequence=[]
        while len(sequence)<6000:sequence.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in sequence[s:s+3]],[]) for s in range(0,6000,3)];ph=digest(json.dumps(plan));assert ph==read(RUN/'models'/f'local_gate_seed{seed}'/'START.json')['plan_sha256']
        for arm in BCE_ARMS:
            mode=arm.replace('_bce','_gate');model=RetentionGate(seed).cuda();ih=state_sha(model.state_dict());assert ih==read(RUN/'models'/f'{mode}_seed{seed}'/'START.json')['initial_sha256'];captured={};hook=model.gate.register_forward_hook(lambda module,inputs,output:captured.update(logits=output[...,0]));directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=ph,ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),mode);target=oracle_alpha(out['base'],out['proposal_delta'],gt[ii]).detach();movement=out['proposal_delta'].norm(dim=-1).detach();loss=(F.binary_cross_entropy_with_logits(captured['logits'],target,reduction='none')*movement)[valid[ii]].mean();error=(out['points']-gt[ii]).norm(dim=-1)[valid[ii]].mean();assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('BCE TRAIN',arm,seed,step+1,round(float(loss.detach()),3),flush=True);f.flush()
            hook.remove();ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);del model,opt,captured
    write(RUN/'BCE_TRAINING_COMPLETE.json',dict(complete=True,updates=18000,runs=runs));print('BCE TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
