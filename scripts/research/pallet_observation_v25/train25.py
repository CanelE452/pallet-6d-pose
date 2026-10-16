"""Matched original-group exposure, paired image intervention, fixed final checkpoints."""
from context25 import *
from prepare25 import ARMS,new_load25
from features25 import edges_for_manifest
from model25 import Correction25,loss25
from train23 import training
from cli_train import state_sha

def dataset(seed):
    O,r,gt,valid,ids,kind=training(seed,'cpu');edges=[]
    for pop in ('heldout','new_train'):
        for c in TRAIN_CONDITIONS:
            m=read((C.RUN if pop=='heldout' else PREV.RUN)/'cache'/pop/c/'manifest.json');edges.append(edges_for_manifest(m,pop+'_'+c))
    O['edge']=torch.cat(edges);assert len(O['edge'])==len(ids)
    m,a=new_load25('new_train');a['edge']=edges_for_manifest(m,'new_train_opposite20');ar=torch.load(RUN/'parents/new_train'/f'seed{seed}.pt',weights_only=True,map_location='cpu');ts=read(RUN/'targets/new_train_opposite20.json')['rows'];ts=[t for t,row in zip(ts,m['rows']) if row['detected']]
    n=len(ids);altids=[row['id']+'::opposite20' for row in m['rows'] if row['detected']];O={k:torch.cat((O[k],a[k])) for k in O};r={k:torch.cat((r[k],ar[k])) for k in r};gt=torch.cat((gt,torch.tensor([t['points'] for t in ts],dtype=torch.float32)));valid=torch.cat((valid,torch.tensor([t['valid'] for t in ts])));kind=torch.cat((kind,C.P.target_type(ts)));ids=ids+altids
    return O,r,gt,valid,ids,kind,n

def main():
    setup();assert read(RUN/'PARENTS_COMPLETE.json')['complete']
    protocol=RUN/'TRAIN_PROTOCOL.json'
    if not protocol.exists():write(protocol,dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'model25.py',KIT/'features25.py',KIT/'train25.py')},frozen_sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts/research').glob('pallet_*_v*/model*.py') if p.parent!=KIT},new_parents=read(RUN/'PARENTS_COMPLETE.json')['files']))
    else:
        for p,h in read(protocol)['sources'].items():assert sha(ROOT/p)==h,('source changed after training registration',p)
    runs=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind,n=dataset(seed);groups={};original_mild={};alt={}
        for i,fid in enumerate(ids):
            source,c=fid.split('::');groups.setdefault(source,[])
            if i<n:groups[source].append(i)
            if c=='crop20':original_mild[source]=i
            if c=='opposite20':alt[source]=i
        names=sorted(groups);rng=np.random.default_rng(25000+seed);seq=[]
        while len(seq)<6000:seq.extend(rng.permutation(len(names)).tolist())
        plans={a:[] for a in ('original','paired')};visits={}
        for start in range(0,6000,3):
            batch={a:[] for a in plans}
            for index in seq[start:start+3]:
                source=names[index];count=visits.get(source,0);visits[source]=count+1
                for a in plans:
                    batch[a].extend(groups[source]);extra=alt.get(source) if a=='paired' and count%2==0 and source in alt else original_mild.get(source)
                    if extra is not None:batch[a].append(extra)
            for a in plans:plans[a].append(batch[a])
        for a,plan in plans.items():assert set(range(n)).issubset({i for batch in plan for i in batch})
        assert set(range(n,len(ids))).issubset({i for batch in plans['paired'] for i in batch})
        write(RUN/f'GROUPS_seed{seed}.json',dict(ids=ids,original_count=n,plans=plans,plan_sha256={k:digest(json.dumps(v)) for k,v in plans.items()},exposures={k:sum(map(len,v)) for k,v in plans.items()},all_original_detected_observations_retained=True,all_paired_detected_observations_exposed=True))
        O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()};gt=gt.cuda();valid=valid.cuda();kind=kind.cuda()
        for arm in ARMS:
            folder=RUN/'models'/f'{arm}_seed{seed}'
            if (folder/'COMPLETE.json').exists():runs.append(read(folder/'COMPLETE.json'));continue
            folder.mkdir(parents=True,exist_ok=True);model=Correction25(seed).cuda();initial=state_sha(model.state_dict());plan=plans['original' if arm=='gate_original' else 'paired'];sl=slice(0,12)
            out=model(subset(O,sl),subset(r,sl),arm)
            if arm.startswith('vector'):assert torch.equal(out['points'],r['base'][sl])
            loss,error=loss25(out,gt[sl],valid[sl],arm);loss.backward();assert sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
            model.zero_grad(set_to_none=True);write(folder/'START.json',dict(initial_sha256=initial,plan_sha256=digest(json.dumps(plan)),observations=len(ids),valid_points=int(valid.sum()),zero_initial_residual=arm.startswith('vector'),gradient_probe_pass=True))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);loss,error=loss25(out,gt[ii],valid[ii],arm);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%500==0:f.flush();print('TRAIN25',seed,arm,step+1,round(float(error.detach()),3),round(time.monotonic()-tick),flush=True)
            p=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),p);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(p),elapsed_seconds=time.monotonic()-tick);write(folder/'COMPLETE.json',done);runs.append(done);del model,opt
        del O,r,gt,valid,kind
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=sum(r['steps'] for r in runs),rows=runs));record('2·3단계 학습 완료:7대조×3시드×2000, 총42,000업데이트. 최종 체크포인트를 고정하고 평가로 진행. 성공 여부는 아직 판단하지 않음.')

if __name__=='__main__':main()
