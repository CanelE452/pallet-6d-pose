"""Intact-observation quality balance; no train-time coordinate or feature edits."""
from context23 import *
from model20 import CauseGate,oracle_alpha
from cli_train import state_sha
import torch.nn.functional as F

def training(seed,device='cuda:0'):
    a=proposal_training(seed,'cpu');b=new_training(seed,'cpu')
    assert not {x.split('::')[0] for x in a[4]}&{x.split('::')[0] for x in b[4]}
    return ({k:torch.cat([a[0][k],b[0][k]]).to(device) for k in a[0]}, {k:torch.cat([a[1][k],b[1][k]]).to(device) for k in a[1]},torch.cat([a[2],b[2]]).to(device),torch.cat([a[3],b[3]]).to(device),a[4]+b[4],torch.cat([a[5],b[5]]).to(device))

def weights(rep,gt,valid,ids,kind):
    before=(rep['base']-gt).norm(dim=-1);target=oracle_alpha(rep['base'],rep['proposal_delta'],gt)
    crop=torch.tensor([x.endswith('::crop20') for x in ids],device=gt.device)[:,None];mask=valid&crop&(kind==2)
    bins=torch.bucketize(before.contiguous(),torch.tensor([5.,15.],device=gt.device),right=False)
    counts=[int((mask&(bins==i)).sum()) for i in range(3)];assert min(counts)>0
    factors=torch.tensor([max(.125,min(8.,sum(counts)/(3*n))) for n in counts],device=gt.device)
    w=torch.ones_like(before);w[mask]=factors[bins[mask]];norm=w[mask].mean();w[mask]/=norm
    assert abs(float(w[mask].mean())-1)<1e-5 and (w>0).all() and torch.equal(w[~mask],torch.ones_like(w[~mask]))
    rows=[];move=rep['proposal_delta'].norm(dim=-1);full=(rep['base']+rep['proposal_delta']-gt).norm(dim=-1)
    for population in ('old','new','combined'):
        orig=torch.tensor([x.split('::')[0] in NEW_IDS for x in ids],device=gt.device)[:,None]
        pop=orig if population=='new' else ~orig if population=='old' else torch.ones_like(orig)
        for i in range(3):
            m=mask&(bins==i)&pop;n=int(m.sum())
            rows.append(dict(population=population,bin=i,points=n,originals=len({ids[j].split('::')[0] for j in torch.where(m.any(-1))[0].tolist()}),base_px=float(before[m].mean()) if n else None,full_px=float(full[m].mean()) if n else None,stop_fraction=float((target[m]<=.05).float().mean()) if n else None,weight=float((factors[i]/norm)),weighted_movement_mass=float((w[m]*move[m]).sum()),original_movement_mass=float(move[m].sum())))
    return w,dict(counts=counts,actual_weights=(factors/norm).tolist(),rows=rows,valid_points=int(valid.sum()),mild_outside_points=int(mask.sum()))

NEW_IDS=set()
def main():
    global NEW_IDS
    setup();assert read(RUN/'SPLIT_AUDIT.json')['status']=='PASS';assert read(RUN/'TARGET_AUDIT.json')['status']=='PASS'
    NEW_IDS={r['id'] for r in read(RUN/'POPULATIONS.json')['new_train']}
    assert not (RUN/'TRAIN_PROTOCOL.json').exists()
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'train23.py',KIT/'context23.py',KIT/'prepare23.py')},frozen_model_sha256=sha(OLD.KIT/'model20.py'),targets=read(OLD.RUN/'TRAIN_PROTOCOL.json')['targets'],parents=read(OLD.RUN/'TRAIN_PROTOCOL.json')['parents'],new_parent_bindings=read(RUN/'PARENTS_COMPLETE.json')['files'],new_target_bindings={str(p.relative_to(ROOT)):sha(p) for p in (RUN/'targets').glob('*.json')}))
    runs=[];probes=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=training(seed);w,probe=weights(r,gt,valid,ids,kind);probe['seed']=seed;probes.append(probe);write(RUN/'TRAIN_QUALITY.json',dict(training_only=True,seeds=probes));print('QUALITY',seed,probe['counts'],probe['actual_weights'],flush=True)
        groups={}
        for i,fid in enumerate(ids):groups.setdefault(fid.split('::')[0],[]).append(i)
        names=sorted(groups);rng=np.random.default_rng(19000+seed);sequence=[]
        while len(sequence)<6000:sequence.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in sequence[s:s+3]],[]) for s in range(0,6000,3)];ph=digest(json.dumps(plan));write(RUN/f'GROUPS_seed{seed}.json',dict(ids=ids,groups=groups,plan=plan,plan_sha256=ph))
        for arm in ARMS:
            model=CauseGate(seed).cuda();directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False)
            write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=state_sha(model.state_dict()),plan_sha256=ph,ids_sha256=digest(json.dumps(ids)),originals=len(groups),observations=len(ids),valid_points=int(valid.sum())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);target=oracle_alpha(out['base'],out['proposal_delta'],gt[ii]).detach();terms=F.binary_cross_entropy_with_logits(out['logits'],target,reduction='none')*out['proposal_delta'].norm(dim=-1).detach();weighted=terms*w[ii] if 'quality' in arm else terms;loss=weighted[valid[ii]].mean();error=(out['points']-gt[ii]).norm(dim=-1)[valid[ii]].mean()
                    opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            p=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),p);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(p),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);del model,opt
        del O,r,gt,valid,w
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=24000,runs=runs));print('TRAINING COMPLETE',flush=True)
if __name__=='__main__':main()
