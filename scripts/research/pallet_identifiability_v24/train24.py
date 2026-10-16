from context24 import *
from model24 import DirectionGate,direction_target,objective,oracle_alpha
from cli_train import state_sha
import torch.nn.functional as F

def stats(out,gt,valid,kind):
    b=out['base'];d=out['proposal_delta'];a=oracle_alpha(b,d,gt);before=(b-gt).norm(dim=-1);after=(out['points']-gt).norm(dim=-1);oracle=(b+a[...,None]*d-gt).norm(dim=-1);masks=dict(all=valid,accurate=valid&(before<=5),outside=valid&(kind==2),accurate_outside=valid&(kind==2)&(before<=5));result={}
    for name,m in masks.items():
        if not m.any():result[name]=dict(points=0);continue
        result[name]=dict(points=int(m.sum()),base_px=float(before[m].mean()),after_px=float(after[m].mean()),oracle_px=float(oracle[m].mean()),regret_px=float((after-oracle)[m].mean()),alpha_mae=float((out['alpha']-a).abs()[m].mean()),mean_alpha=float(out['alpha'][m].mean()))
    return result

def predict(model,O,r,arm):
    parts=[]
    with torch.inference_mode():
        for first in range(0,len(O['points']),32):
            sl=slice(first,first+32);out=model(subset(O,sl),subset(r,sl),arm);parts.append({k:out[k] for k in ('points','base','proposal_delta','alpha','coverage')})
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def tiny():
    setup();protocol=read(RUN/'PROTOCOL.json');assert not (RUN/'TINY_TRAIN_PROTOCOL.json').exists();write(RUN/'TINY_TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),model_sha256=sha(KIT/'model24.py')));rows=[];controls=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=new_training(seed);pop={}
        for split,key in [('train','train_originals'),('check','check_originals')]:
            ii=torch.tensor([i for i,x in enumerate(ids) if x.split('::')[0] in protocol['tiny'][key]],device='cuda:0');pop[split]=(subset(O,ii),subset(r,ii),gt[ii],valid[ii],kind[ii],[ids[i] for i in ii.tolist()]);assert len({x.split('::')[0] for x in pop[split][-1]})==len(protocol['tiny'][key])
        del O,r,gt,valid,kind
        O,r,gt,valid,kind,ids=pop['train'];groups={}
        for i,x in enumerate(ids):groups.setdefault(x.split('::')[0],[]).append(i)
        names=sorted(groups);rng=np.random.default_rng(24000+seed);seq=[]
        while len(seq)<18000:seq.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in seq[s:s+3]],[]) for s in range(0,18000,3)];ph=digest(json.dumps(plan));write(RUN/f'TINY_GROUPS_seed{seed}.json',dict(ids=ids,plan=plan,plan_sha256=ph,check_ids=pop['check'][-1]))
        target=oracle_alpha(r['base'],r['proposal_delta'],gt).detach();free=torch.nn.Parameter(torch.zeros_like(target));opt=torch.optim.Adam([free],lr=.05)
        for step in range(2000):
            loss=(F.binary_cross_entropy_with_logits(free,target,reduction='none')*r['proposal_delta'].norm(dim=-1))[valid].mean();opt.zero_grad();loss.backward();assert torch.isfinite(free.grad).all();opt.step()
        alpha=free.detach().sigmoid();out=dict(base=r['base'],proposal_delta=r['proposal_delta'],alpha=alpha,points=r['base']+alpha[...,None]*r['proposal_delta']);controls.append(dict(seed=seed,steps=2000,statistics=stats(out,gt,valid,kind)));torch.save(dict(logits=free.detach().cpu(),ids=ids,privileged=True),RUN/f'LOOKUP_seed{seed}.pt');del opt,free
        for arm in TINY_ARMS:
            model=DirectionGate(seed).cuda();folder=RUN/'tiny_models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False);write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),plan_sha256=ph));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);curve=[]
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);loss,error,_=objective(out,gt[ii],valid[ii],'unit_weighted');opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm))+'\n')
                    if (step+1)%1000==0:
                        rec=dict(step=step+1)
                        for split,(ox,rx,gx,vx,kx,_) in pop.items():rec[split]=stats(predict(model,ox,rx,arm),gx,vx,kx)
                        curve.append(rec);write(folder/'CURVE.json',dict(rows=curve));print('TINY',arm,seed,step+1,rec['train']['all']['regret_px'],rec['train']['accurate']['after_px'],flush=True);f.flush()
                    if step+1 in (2000,6000):torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=step+1),folder/('early2000.pt' if step+1==2000 else 'final.pt'))
            final=curve[-1]['train'];passed=final['all']['regret_px']<=.25 and final['accurate']['after_px']<=final['accurate']['base_px']+.1;done=dict(seed=seed,arm=arm,steps=6000,curve=curve,memorization_pass=passed,sha256=sha(folder/'final.pt'));write(folder/'COMPLETE.json',done);rows.append(done);del model,opt
    write(RUN/'TINY_COMPLETE.json',dict(complete=True,neural_updates=36000,privileged_lookup_updates=6000,rows=rows,lookup=controls,all_memorization_pass=all(x['memorization_pass'] for x in rows)))

def sign():
    setup();assert not (RUN/'TRAIN_PROTOCOL.json').exists();previous=read(PREV.RUN/'TRAIN_PROTOCOL.json');write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'train24.py',KIT/'model24.py',KIT/'context24.py')},frozen_model_sha256=previous['frozen_model_sha256'],parents=previous['parents'],targets=previous['targets'],new_parent_bindings=previous['new_parent_bindings'],new_target_bindings=previous['new_target_bindings']));rows=[];train_stats=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=training(seed);target,mask=direction_target(r,gt,valid);counts=[int((mask&(target==i)).sum()) for i in range(2)];factors=target.new_tensor([mask.sum()/(2*n) for n in counts]);weight=factors[target.long()];g=read(PREV.RUN/f'GROUPS_seed{seed}.json');assert g['ids']==ids;write(RUN/f'GROUPS_seed{seed}.json',g);train_stats.append(dict(seed=seed,counts=counts,factors=factors.tolist(),valid_points=int(valid.sum()),classification_points=int(mask.sum())))
        for arm in ARMS:
            model=DirectionGate(seed).cuda();folder=RUN/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=False);write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids))));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(g['plan']):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);terms=F.binary_cross_entropy_with_logits(out['logits'],target[ii],reduction='none')*weight[ii];loss=terms[mask[ii]].mean();error=(out['points']-gt[ii]).norm(dim=-1)[valid[ii]].mean();opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm))+'\n')
                    if (step+1)%1000==0:print('SIGN',arm,seed,step+1,float(loss.detach()),flush=True);f.flush()
            p=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,arm=arm,steps=2000),p);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(p));write(folder/'COMPLETE.json',done);rows.append(done)
            out=predict(model,O,r,arm);torch.save(dict(score=out['alpha'].cpu(),target=target.cpu(),mask=mask.cpu(),kind=kind.cpu(),ids=ids),folder/'TRAIN_PREDICTIONS.pt');del model,opt
        del O,r,gt,valid
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=18000,rows=rows,training_labels=train_stats));print('SIGN TRAINING COMPLETE',flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['tiny','sign'],required=True);a=ap.parse_args();tiny() if a.phase=='tiny' else sign()
