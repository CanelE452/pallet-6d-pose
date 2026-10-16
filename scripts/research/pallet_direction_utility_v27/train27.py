"""Matched synthetic-only training of four new utility structures."""
from context27 import *
from train25 import dataset
from model25 import Correction25
from model27 import DirectionUtility27,objective
from cli_train import state_sha

def point_proposal(O,r,seed):
    model=Correction25(seed).cuda().eval();p=PREV25.RUN/'models'/f'vector_local_edge_seed{seed}'/'final.pt';model.load_state_dict(torch.load(p,weights_only=True,map_location='cpu')['state_dict']);parts=[]
    with torch.inference_mode():
        for i in range(0,len(r['base']),32):
            sl=slice(i,i+32);parts.append(model(subset(O,sl),subset(r,sl),'vector_local_edge')['points']-r['base'][sl])
    del model;return torch.cat(parts)

def main():
    setup();assert read(RUN/'PRECHECK.json')['status']=='PASS';protocol=read(RUN/'PROTOCOL.json')
    sources=dict(protocol['sources']);amend=RUN/'PRETRAIN_SOURCE_AMENDMENT.json'
    if amend.exists():sources.update(read(amend)['sources'])
    for p,h in sources.items():assert sha(ROOT/p)==h,p
    files={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'train27.py',KIT/'model27.py',KIT/'context27.py')}
    if (RUN/'TRAIN_PROTOCOL.json').exists():assert read(RUN/'TRAIN_PROTOCOL.json')['sources']==files
    else:write(RUN/'TRAIN_PROTOCOL.json',dict(sources=files,protocol_sha256=sha(RUN/'PROTOCOL.json'),plan_bindings={str(p.relative_to(ROOT)):sha(p) for p in PREV25.RUN.glob('GROUPS_seed*.json')}))
    runs=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind,n=dataset(seed);old=read(PREV25.RUN/f'GROUPS_seed{seed}.json');assert ids==old['ids'];plan=old['plans']['paired'];assert len(plan)==2000;write(RUN/f'GROUPS_seed{seed}.json',dict(ids=ids,plan=plan,plan_sha256=digest(json.dumps(plan)),observations=len(ids),valid_points=int(valid.sum()),same_V25_paired=True))
        O={k:v.cuda() for k,v in O.items()};r={k:v.cuda() for k,v in r.items()};gt=gt.cuda();valid=valid.cuda();proposal=point_proposal(O,r,seed);(RUN/'train_cache').mkdir(exist_ok=True);torch.save(proposal.cpu(),RUN/'train_cache'/f'point_delta_seed{seed}.pt')
        for arm in ARMS:
            folder=RUN/'models'/f'{arm}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.json').exists():runs.append(read(folder/'COMPLETE.json'));continue
            model=DirectionUtility27(seed).cuda();write(folder/'START.json',dict(initial_sha256=state_sha(model.state_dict()),parameters=sum(p.numel() for p in model.parameters()),plan_sha256=digest(json.dumps(plan)),no_real_training=True));opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (folder/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),proposal[ii],arm);loss,terms=objective(out,gt[ii],valid[ii],arm);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},gradient_norm=norm))+'\n')
                    if (step+1)%500==0:f.flush();print('TRAIN27',seed,arm,step+1,round(float(terms['point'].detach()),3),round(time.monotonic()-tick),flush=True)
            path=folder/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),path);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(path),elapsed_seconds=time.monotonic()-tick);write(folder/'COMPLETE.json',done);runs.append(done);del model,opt
        del O,r,gt,valid,proposal
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=sum(r['steps'] for r in runs),runs=runs));record('V27 새 구조4대조×3시드×2000 실제 학습 완료. 최종 고정 체크포인트만 평가하며 결과에 따른 재학습 없음.')

if __name__=='__main__':main()
