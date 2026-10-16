"""Train DHT/point experts separately, then decoders, then held-original routers."""
from context29 import *
from train25 import dataset
from model29 import *
from cli_train import state_sha

def split_plan(ids,seed):
    names=sorted({x.split('::')[0] for x in ids});ordered=sorted(names,key=lambda s:digest('v29-split:'+s))
    cut=int(.8*len(ordered));experts=set(ordered[:cut]);routing=set(ordered[cut:]);assert not experts&routing
    groups={s:[] for s in names}
    for i,fid in enumerate(ids):groups[fid.split('::')[0]].append(i)
    def plan(sources,steps):
        rng=np.random.default_rng(29000+seed);ss=sorted(sources);out=[]
        for _ in range(steps):
            chosen=rng.choice(len(ss),BATCH,replace=False)
            out.append([int(rng.choice(groups[ss[i]])) for i in chosen])
        return out
    return dict(expert_sources=sorted(experts),router_sources=sorted(routing),expert_plan=plan(experts,STEPS),router_plan=plan(routing,ROUTER_STEPS),ids=ids)

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    (RUN/'PURPOSE.md').write_text('V29: 사용자 요청의 점/점 가이드 선/점 독립 선 분기와 점별 적응 선택을 실제 학습·검증한다.\n')
    write(RUN/'PROTOCOL.json',dict(seeds=SEEDS,steps_expert=STEPS,steps_decoder=STEPS,steps_router=ROUTER_STEPS,batch=BATCH,total_updates=3*(5*STEPS+3*ROUTER_STEPS),
        experts='New pure point correction from stock YOLO; guided/independent DHT each fine-tuned from identical V6 starts with structural line CE. Equal DHT architectures: independent always zero point maps. Frozen YOLO, no V12/V25 correction input.',
        decoder='27 intersections from 3 incident roles x3 modes with abs(det)>=.1, abs(ROI coords)<=4. Minimum-norm ridge1e-4 fallback only when all pairs invalid. Learned soft selection, no point-coordinate input. Cuboid roles, no new physical edge/endpoint GT.',
        training='Original-hash80/20 split of V25 synthetic training originals. Expert/decoder on80%; frozen-expert residuals for router on20%. Initial V6/YOLO pretrained sources may overlap; the new expert fine-tuning and router originals are disjoint. No real fitting.',
        router='Per-point keep, point, guided-line, independent-line candidates. Predict normalized expected errors and choose argmin. Three arms: state+quality, no-state, state point-only control. Synthetic hidden/outside masks supervise proxy state; no GT state at inference. Point control masks line destinations/qualities.',
        optimizer='AdamW decay1e-4 clip5 FP32 no TF32; DHT lr1e-4 other modules3e-4; fixed final step; no outcome-based tuning.',
        interventions='GT best expert (privileged diagnostic), GT state fixed mapping 0point/1guided/2independent (privileged), predicted state fixed mapping; guided line input zero/donor; independent point perturbation invariance; point control line replacement invariance.',
        evaluation='Continuation of real297/9session reusedDEV stress lineage, not canonical140/paper/final holdout. All8 real/synth conditions and prospective6; same detected population, retain misses. No newly captured natural occlusion/visibility GT.',
        acceptance='Each seed: clean and crop10/20/30 <=1.01*V12; corner20/crop40 >=90% V18 gain retained. Adaptive vs matched point_router crop20 >=1% each seed and paired session bootstrap95% upper<0, stability simultaneously. Report independent vs guided all crop and guided vs point all occlusion, no condition averaging or post-hoc winner deployment.',
        sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'context29.py',KIT/'model29.py',KIT/'train29.py')},
        initial_checkpoints={str(s):sha(V6/f'cache_heads/hough_seed{s}/checkpoint_final.pt') for s in SEEDS}))
    record('V29 사전 등록: 점/점가이드DHT/점독립DHT 단독 학습, 선 교차 전용 디코더, 합성 원본80/20 분리 선택기3대조, 3시드25,200업데이트. 기존 실사DEV 계보 유지, 자연 가시 경계GT 미확보 명시.')

def preflight():
    torch.set_num_threads(2);O,r,gt,valid,ids,kind,_=dataset(1);plan=split_plan(ids,1)
    obs=subset(O,slice(0,4));checks={}
    line=LineExpert(V6/'cache_heads/hough_seed1/checkpoint_final.pt',1,'independent')
    with torch.no_grad():
        a=line(obs,True);z=line(dict(obs,points=obs['points']+1000),True)
        assert torch.equal(a['logits'],z['logits']);assert torch.equal(a['lines'],z['lines'])
    loss=line_loss(a,obs,gt[:4],valid[:4]);assert torch.isfinite(loss)
    # Exact arbitrary subcell rectangle + depth: incident correct lines recover corners.
    g=torch.tensor([[.123,.234],[.623,.234],[.623,.734],[.123,.734],[-.177,-.166],[.323,-.166],[.323,.334],[-.177,.334]])[None]
    u,v=torch.tensor(EDGES).T;d=g[:,v]-g[:,u];n=torch.stack((-d[...,1],d[...,0]),-1);n=n/n.norm(dim=-1,keepdim=True);ll=torch.cat((n,-(n*g[:,u]).sum(-1,keepdim=True)),-1)[:,:,None].repeat(1,1,3,1);mass=torch.ones(1,12,3)/3
    c=intersection_candidates(ll,mass);err=(c['candidates']-g[:,:,None]).norm(dim=-1).masked_fill(~c['mask'],1e6).min(-1).values
    assert float(err.max())<1e-5
    decoder=LineDecoder(1);do=decoder(a['lines'],a['mass'],obs['box']);dl=decoder_loss(do,obs,gt[:4],valid[:4]);dl.backward();assert torch.isfinite(dl)
    assert all(torch.isfinite(p.grad).all() for p in decoder.parameters() if p.grad is not None)
    point=PointExpert(1);cand=torch.stack((obs['points'],point(obs),do['points'].detach(),do['points'].detach()+1),2);quality=torch.zeros(4,8,4)
    for arm in ROUTERS:
        router=Router(1,arm);out=router(obs,cand.detach(),quality);loss=router_loss(out,cand.detach(),obs,gt[:4],valid[:4],kind[:4],arm);loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in router.parameters() if p.grad is not None)
        if arm=='point_router':
            alt=cand.detach().clone();alt[:,:,2:]+=100;other=router(obs,alt,quality+100)
            assert torch.equal(out['points'],other['points']) and torch.equal(out['risk'],other['risk'])
    from eval27 import population,conditions
    sources=set(plan['expert_sources'])|set(plan['router_sources']);overlap={}
    for split in EVAL_SPLITS:
        m,_=population(split,conditions(split)[0]);ss={x['id'] for x in m['rows']};assert not sources&ss;overlap[split]=0
    write(RUN/'PRECHECK.json',dict(status='PASS',independent_point_coordinate_invariant=True,subcell_true_line_intersection_max_px=float(err.max()),decoder_router_finite_gradient=True,point_router_line_swap_invariant=True,expert_sources=len(plan['expert_sources']),router_sources=len(plan['router_sources']),expert_router_overlap=0,evaluation_overlap=overlap,gt_valid_points=int(valid.sum()),observations=len(ids),proxy_state_counts=kind[valid].bincount().tolist()))
    print('PRECHECK29 PASS',flush=True)

def fit(model,name,seed,plan,loss_fn,lr):
    folder=RUN/'models'/f'{name}_seed{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/'final.pt'
    if (folder/'COMPLETE.json').exists():
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state_dict']);return model.eval()
    assert not (folder/'START.json').exists(),f'incomplete run needs explicit investigation: {folder}'
    write(folder/'START.json',dict(initial_state_sha256=state_sha(model.state_dict()),parameters=sum(p.numel() for p in model.parameters()),plan_sha256=digest(json.dumps(plan)),steps=len(plan)))
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);model.train();tick=time.monotonic()
    with (folder/'TRACE.jsonl').open('w') as f:
        for step,indices in enumerate(plan):
            ii=torch.tensor(indices,device='cuda:0');loss=loss_fn(model,ii);assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step()
            f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gradient_norm=gn))+'\n')
            if (step+1)%200==0:
                f.flush();print('TRAIN29',seed,name,step+1,round(float(loss.detach()),4),round(time.monotonic()-tick),flush=True)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},seed=seed,name=name,steps=len(plan)),path)
    write(folder/'COMPLETE.json',dict(complete=True,name=name,seed=seed,steps=len(plan),checkpoint_sha256=sha(path),seconds=time.monotonic()-tick))
    return model.eval()

def cache_lines(model,O):
    parts=[]
    with torch.no_grad():
        for i in range(0,len(O['box']),32):
            out=model(subset(O,slice(i,i+32)),True);parts.append({k:out[k] for k in ('lines','mass')})
    return {k:torch.cat([a[k] for a in parts]) for k in parts[0]}

def cached_experts(point,decoders,O,lines):
    parts=[]
    with torch.no_grad():
        for i in range(0,len(O['box']),32):
            sl=slice(i,i+32);obs=subset(O,sl);out=[decoders[a](lines[a]['lines'][sl],lines[a]['mass'][sl],obs['box']) for a in ('guided','independent')]
            c=torch.stack((obs['points'],point(obs),out[0]['points'],out[1]['points']),2)
            q=torch.stack((out[0]['entropy'],out[0]['support'],out[1]['entropy'],out[1]['support']),-1)
            parts.append(dict(candidates=c,quality=q))
    return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def train():
    setup();assert read(RUN/'PRECHECK.json')['status']=='PASS'
    for p,h in read(RUN/'PROTOCOL.json')['sources'].items():assert sha(ROOT/p)==h,p
    for seed in SEEDS:
        O,r,gt,valid,ids,kind,_=dataset(seed);del r;O.pop('edge');plan=split_plan(ids,seed);write(RUN/f'PLAN_seed{seed}.json',plan)
        O={k:v.cuda() for k,v in O.items()};gt=gt.cuda();valid=valid.cuda();kind=kind.cuda();ep=plan['expert_plan'];rp=plan['router_plan']
        point=PointExpert(seed).cuda()
        def ploss(m,ii):return ((m(subset(O,ii))-gt[ii])/diagonal(subset(O,ii))[:,None,None]).norm(dim=-1)[valid[ii]].mean()
        point=fit(point,'point',seed,ep,ploss,3e-4);lines={};decoders={}
        for arm in ('guided','independent'):
            model=LineExpert(V6/f'cache_heads/hough_seed{seed}/checkpoint_final.pt',seed,arm).cuda()
            def lloss(m,ii):
                obs=subset(O,ii);return line_loss(m(obs),obs,gt[ii],valid[ii])
            model=fit(model,arm+'_dht',seed,ep,lloss,1e-4);lines[arm]=cache_lines(model,O);del model
            decoder=LineDecoder(seed).cuda()
            def dloss(m,ii):
                obs=subset(O,ii);out=m(lines[arm]['lines'][ii],lines[arm]['mass'][ii],obs['box']);return decoder_loss(out,obs,gt[ii],valid[ii])
            decoders[arm]=fit(decoder,arm+'_decoder',seed,ep,dloss,3e-4)
        cache=cached_experts(point,decoders,O,lines)
        folder=RUN/'training_cache';folder.mkdir(exist_ok=True)
        torch.save({k:v.cpu() for k,v in cache.items()},folder/f'seed{seed}.pt')
        for arm in ROUTERS:
            router=Router(seed,arm).cuda()
            def rloss(m,ii):
                obs=subset(O,ii);cc=cache['candidates'][ii];return router_loss(m(obs,cc,cache['quality'][ii]),cc,obs,gt[ii],valid[ii],kind[ii],arm)
            fit(router,arm,seed,rp,rloss,3e-4);del router
        del point,decoders,O,gt,valid,kind,cache,lines
    runs=[read(p) for p in sorted((RUN/'models').glob('*/COMPLETE.json'))];assert len(runs)==24
    write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=sum(r['steps'] for r in runs),runs=runs));record('V29 3시드24모델 실제 학습 완료. 최종 고정 체크포인트로 분기/선택/가림·잘림 평가 진행.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('prepare','train'),required=True);a=p.parse_args()
    if a.phase=='prepare':register();preflight()
    else:train()
