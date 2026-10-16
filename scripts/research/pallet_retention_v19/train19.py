"""Matched grouped-original batches; supervise retention with original synthetic GT."""
from context19 import *
from model19 import RetentionGate,objective,observed_samples,oracle_alpha
from cli_train import state_sha

def preflight(O,rep,gt,valid):
    o=subset(O,slice(0,16));r=subset(rep,slice(0,16));checks=[]
    for arm in ARMS:
        m=RetentionGate(1).to(o['points'].device);out=m(o,r,arm);assert torch.isfinite(out['points']).all();assert torch.all(out['alpha']==.5);assert torch.equal(out['points'],r['base']+.5*r['proposal_delta']);flip=m(o,dict(r,lines=-r['lines']),arm);assert torch.allclose(out['sample_pos'],flip['sample_pos'],atol=1e-4,rtol=0);loss=objective(out,gt[:16],valid[:16])[0];loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);checks.append(dict(arm=arm,finite_gradient=True,initial_half=True,sign_invariance=True))
    # Axis-aligned, intersecting diagonal and non-intersecting lines in a known rectangle.
    device=o['points'].device;box=torch.tensor([[0.,0.,10.,10.]],device=device);fixture=dict(box=box,image_hw=torch.tensor([[10.,10.]],device=device));base=torch.tensor([[[5.,5.]]],device=device);lines=torch.tensor([[[1.,0.,-5.],[0.,1.,-5.],[1.,-1.,0.],[1.,0.,-20.]]],device=device)
    for indices in ((0,1,2),(0,1,3)):
        weight=torch.zeros(1,1,5,device=device)
        for j,k in enumerate(indices):weight[0,0,k]=3-j
        r0=dict(base=base,lines=lines,line_weights=weight);samples=observed_samples(fixture,r0,'line');pos=samples['pos'].reshape(3,12,2);v=samples['valid'].reshape(3,12)
        for j,k in enumerate(indices):
            if k==3:assert not v[j].any()
            else:assert v[j].all() and ((pos[j]*lines[0,k,:2]).sum(-1)+lines[0,k,2]).abs().max()<1e-5
    # Zero movement target is defined; all points remain eligible, no division by zero.
    assert torch.equal(oracle_alpha(torch.zeros(1,1,2),torch.zeros(1,1,2),torch.ones(1,1,2)),torch.zeros(1,1))
    return checks

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    protocol=dict(question='Can frozen successful V18 corrections retain hard-case gains while suppressing mild-crop direction errors using observed image/line context?',arms=ARMS,seeds=SEEDS,steps=2000,batch='Three original image IDs per step; include all available detected variants among6 training conditions (up to18observations). Same original-ID permutation for all arms; no condition token or GT at inference.',
      proposal='Frozen V18 line_vector seed-matched; points=base+sigmoid(logit)*immutable_delta. No reversed direction or extra vector, no test GT or heuristic gate.',
      architecture='Shared local3x3 patch+parent64context+pointID+observability+coordinates+amodal_delta+proposal_delta ->64query. local gate adds zero32token. area gate reads6x6 samples from box intersect raw image. line gate reads12samples from each top3 associated DHT line clipped to that same rectangle. Both context arms use36samples, identical24channel projection/32token MLP/query attention and same final scalar head. Canonical line signs; empty segments masked. Pixel support is not physical edge correctness or visibility through occluders.',
      supervision='Original synthetic GT only. alpha_star=clip(dot(GT-base,delta)/norm(delta)^2,0,1). Loss=mean(norm(delta)*abs(alpha-alpha_star)) over all original valid points, equal to coordinate distance to projected attainable GT. Zero vectors have zero consequential loss; no self-training/filter/discard. Fixed final checkpoint; no real fitting.',
      controls='Same parameter tensors/initialization (alpha=.5), same grouped batches/2000steps/AdamW3e-4 wd1e-4 float32 clip5. area versus line isolates line-guided context at same sample budget; local versus context tests added nonlocal evidence. All use the same frozen line-based parent lineage, not pure YOLO controls.',
      acceptance='All3seeds clean andcrop10/20/30 outside <=1.01 base; corner20hidden andcrop40outside retain>=90% seed-matched V18 line_vector gain over base. Context incremental effect requires all3seeds>=1% gain and real9session paired95%CI upper<0. Multiplicity uncorrected.',
      population='Training253synthetic/1462detected observations6conditions already seen by frozen V18, fixed parents. Eval reused real297/9sessions/1251manual corners and synth505/8conditions; crop10/30 evaluation-only. Original IDs/GT/misses. No independent new final holdout or physical manual-click reannotation.',
      interventions='Same checkpoint: zero additional context token, or replace only context feature source by next detected-image features retaining point patch/geometry/proposal. OOD diagnostic. Fixed alpha.5 control. No performance-based choice of arm/seed.')
    write(RUN/'PROTOCOL.json',protocol);parents={str(p.relative_to(ROOT)):sha(p) for p in (Q.RUN/'models').glob('line_vector_seed*/final.pt')};parents.update(read(Q.RUN/'TRAIN_PROTOCOL.json')['parents'])
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},targets=read(Q.RUN/'TRAIN_PROTOCOL.json')['targets'],parents=parents,steps=2000,no_real_training=True))
    runs=[];checks=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);checks.extend(preflight(O,rep,gt,valid));groups={}
        for i,ident in enumerate(ids):groups.setdefault(ident.rsplit('::',1)[0],[]).append(i)
        names=sorted(groups);rng=np.random.default_rng(19000+seed);sequence=[]
        while len(sequence)<6000:sequence.extend(rng.permutation(len(names)).tolist())
        plan=[sum([groups[names[k]] for k in sequence[s:s+3]],[]) for s in range(0,6000,3)];assert set(sum(plan,[]))==set(range(len(ids)));planhash=digest(json.dumps(plan));initial=None
        write(RUN/f'TRAIN_GROUPS_seed{seed}.json',dict(originals=len(names),observations=len(ids),ids=ids,groups=groups,plan_sha256=planhash))
        for arm in ARMS:
            model=RetentionGate(seed).cuda();ih=state_sha(model.state_dict());initial=ih if initial is None else initial;assert ih==initial;directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=planhash,ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss,error,_=objective(out,gt[ii],valid[ii]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),3),'seconds',round(time.monotonic()-tick),flush=True);f.flush()
            ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck);done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks,axis_aligned_diagonal_and_missed_line_clipping=True));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=18000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
