"""Unchanged summed objective versus stop/move PCGrad, on old and unseen residuals."""
from context21 import *
from model20 import CauseGate,objective,oracle_alpha
from cli_train import state_sha
import torch.nn.functional as F

def project_pair(g0,g1):
    dot=torch.dot(g0,g1);n0=torch.dot(g0,g0);n1=torch.dot(g1,g1)
    if dot<0 and n0>0 and n1>0:
        merged=(g0-dot/n1*g1)+(g1-dot/n0*g0)
    else:merged=g0+g1
    return merged,float(dot/(n0*n1).sqrt().clamp_min(1e-30)),bool(dot<0 and n0>0 and n1>0)

def flat_grad(loss,params,retain=True):
    gs=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gs)])

def run():
    setup();assert read(RUN/'SPLIT_AUDIT.json')['status']=='PASS';assert read(RUN/'TARGET_AUDIT.json')['status']=='PASS'
    assert not (RUN/'TRAIN_PROTOCOL.json').exists()
    # Independent known-vector and ordinary-sum cases.
    g0=torch.tensor([1.,0.],dtype=torch.float64);g1=torch.tensor([-1.,1.],dtype=torch.float64);out,co,yes=project_pair(g0,g1);assert yes and torch.allclose(out,torch.tensor([.5,1.5],dtype=torch.float64));assert torch.equal(project_pair(g0,g0)[0],2*g0);assert torch.equal(project_pair(g0,g0*0)[0],g0)
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),model_sha256=sha(OLD.KIT/'model20.py'),parent_bindings=read(RUN/'PARENTS_COMPLETE.json')['files'],target_bindings={str(p.relative_to(ROOT)):sha(p) for p in (RUN/'targets').glob('*.json')},split_audit_sha256=sha(RUN/'SPLIT_AUDIT.json'),preflight_known_vector=True))
    runs=[];reproduce=[];plans=[]
    for seed in SEEDS:
        for population in ('old','new'):
            O,rep,gt,valid,ids,kind=proposal_training(seed) if population=='old' else new_training(seed)
            groups={}
            for i,fid in enumerate(ids):groups.setdefault(fid.split('::')[0],[]).append(i)
            names=sorted(groups);rng=np.random.default_rng(19000+seed);sequence=[]
            while len(sequence)<6000:sequence.extend(rng.permutation(len(names)).tolist())
            plan=[sum([groups[names[k]] for k in sequence[s:s+3]],[]) for s in range(0,6000,3)];ph=digest(json.dumps(plan));write(RUN/f'GROUPS_{population}_seed{seed}.json',dict(ids=ids,groups=groups,plan=plan,plan_sha256=ph))
            if population=='old':assert ph==read(OLD.RUN/f'TRAIN_GROUPS_seed{seed}.json')['plan_sha256']
            arms=[a for a in ARMS if a.startswith('unit_'+population+'_')]
            for arm in arms:
                model=CauseGate(seed).cuda();params=list(model.parameters());directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);initial=state_sha(model.state_dict());write(directory/'START.json',dict(arm=arm,seed=seed,initial_sha256=initial,plan_sha256=ph,ids_sha256=digest(json.dumps(ids)),originals=len(groups),observations=len(ids),valid_points=int(valid.sum())))
                opt=torch.optim.AdamW(params,lr=3e-4,weight_decay=1e-4);pc=arm.endswith('pcgrad');tick=time.monotonic();conflicts=0;tested=0;decomp_max=0.;cosines=[]
                with (directory/'TRACE.jsonl').open('w') as f:
                    for step,indices in enumerate(plan):
                        ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss,error,target=objective(out,gt[ii],valid[ii],'unit_weighted');mask=valid[ii];stop=mask&(target<=.05);terms=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none')*out['proposal_delta'].norm(dim=-1).detach();loss0=terms[stop].sum()/mask.sum();loss1=terms[mask&~stop].sum()/mask.sum();dis=float(abs((loss0+loss1-loss).detach()));decomp_max=max(decomp_max,dis);assert dis<1e-4
                        opt.zero_grad(set_to_none=True);cos=None;conflict=None
                        if pc or step%20==0:
                            g0=flat_grad(loss0,params);g1=flat_grad(loss1,params,retain=not pc);merged,cos,conflict=project_pair(g0,g1)
                            if step%20==0:tested+=1;conflicts+=int(conflict);cosines.append(cos)
                            if pc:
                                offset=0
                                for p in params:p.grad=merged[offset:offset+p.numel()].reshape_as(p).clone();offset+=p.numel()
                        if not pc:loss.backward()
                        norm=float(torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,stop_points=int(stop.sum()),valid_points=int(mask.sum()),observations=len(indices),gradient_cosine=cos,conflict=conflict))+'\n')
                        if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
                ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick,sampled_conflict_fraction=conflicts/tested,sampled_mean_cosine=float(np.mean(cosines)),decomposition_max_difference=decomp_max,observation_forwards=sum(map(len,plan)));write(directory/'COMPLETE.json',done);runs.append(done)
                if arm=='unit_old_sum':
                    previous=CauseGate(seed).cuda().eval();previous.load_state_dict(torch.load(OLD.RUN/'models'/f'unit_weighted_seed{seed}'/'final.pt',weights_only=True)['state_dict']);maximum=0.
                    with torch.no_grad():
                        for first in range(0,len(ids),32):
                            sl=slice(first,first+32);args=(subset(O,sl),subset(rep,sl));a=model(*args,arm);b=previous(*args,'unit_weighted');maximum=max(maximum,float((a['points']-b['points']).abs().max()))
                    assert maximum<.001;reproduce.append(dict(seed=seed,max_training_point_error_px=maximum));del previous
                del model,opt
    write(RUN/'REPRODUCTION.json',dict(status='PASS',rows=reproduce));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=30000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':run()
