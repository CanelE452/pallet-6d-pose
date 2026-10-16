"""Preregistered synthetic-only, matched seed/batch training; final checkpoint."""
from context17 import *
from model17 import DetailEvaluator,revised_targets,objective,weights,geometry
from cli_train import state_sha

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Refusing to overwrite training')
    assert read(RUN/'PREPARATION_COMPLETE.json')['complete']
    write(RUN/'PROTOCOL.json',dict(question='Factorial native RGB detail x paired visible-silhouette auxiliary targets',arms=ARMS,seeds=SEEDS,steps=2000,batch=16,
        controls='low/high RGB96 branch: low area24 then bilinear96, high original96. Same two3x3conv, same YOLO24 features and candidates, same initial weights/batches. none/proxy/mask aux factor. All gain losses tau.2; aux CE .25; common class weights derived from proxy labels.',
        labels='Only paired G38 renderer-mask frames with complete8corners for BOTH proxy and mask aux. Missing masks/invalid corners/outsideROI ignored in both auxiliary losses; all valid keypoints kept for gain loss. Mask revises positive cuboid-edge samples to background if >2.5px from original visible object silhouette (3x3 morphological contour). Original crop/occlusion transformations applied; artificial boundaries not positives. This narrows the semantic target to mask-supported outline; cannot supervise visible internal face creases.',
        architecture='Frozen YOLO/point/amodal/DHT parents + learned RGB96 3->16->8 3x3conv branch. Sample RGB and YOLO features at the same candidate-conditioned7x3x3line coordinates and sum8channel evidence. All head parameters trained, parents frozen. Not native YOLO high-resolution re-training.',
        train='Original253 synthetic images,1462 detected observations6conditions. No real labels or masks at inference, no real tuning.',
        evaluation='Reused real297/9sessions and synth505, same8 conditions. Same original manual IDs and reflect100 detection parent. No independent final holdout.',
        acceptance='All3seeds clean/crop10/20/30 <=1.01 base; corner20hidden and crop40outside retain>=80% amodal gains. Resolution and mask incremental claims require same other factor, all3seeds>=1% improvement and paired real9session95%CI upper<0. No multiple-comparison correction.',
        optimizer='AdamW3e-4 wd1e-4 float32 clip5;2000updates final checkpoint only',parent_protocol_sha256=sha(P.RUN/'PROTOCOL.json'),preparation_sha256=sha(RUN/'PREPARATION_COMPLETE.json')))
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},
        targets={str(p.relative_to(ROOT)):sha(p) for p in (C.RUN/'targets').glob('heldout_*.json')},
        parents={str(p.relative_to(ROOT)):sha(p) for p in (C.RUN/'parents/heldout').glob('*/seed*_actual.pt')},
        no_real_training=True,steps=2000,arms=ARMS,seeds=SEEDS))
    masks=[];bound=[]
    for c in TRAIN_CONDITIONS:
        m=read(C.RUN/'cache/heldout'/c/'manifest.json');t=C.target_rows('heldout',c)
        assert [r['id'] for r in m['rows']]==[r['id'] for r in t]
        for r in m['rows']:
            if not r['detected']:continue
            a=np.array(r['mask'] if r['mask'] is not None else [-1,-1,-1,-1],float)
            if r['mask'] is not None:a-=np.tile(r['origin'],2)
            masks.append(a);bound.append(r['id']+'::'+c)
    masks=torch.tensor(np.array(masks),dtype=torch.float32,device='cuda:0');runs=[];checks=[]
    sup=[torch.load(RUN/'supervision/heldout'/f'{c}.pt',weights_only=True) for c in TRAIN_CONDITIONS]
    boundary=torch.cat([x['boundary'] for x in sup]).cuda();has_mask=torch.cat([x['has_mask'] for x in sup]).cuda()
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);assert ids==bound
        cand=candidates(O,rep);err=(cand-gt[:,:,None]).norm(dim=-1);truth=(err[...,:1]-err).detach();w,counts=weights(truth,valid,kind)
        pair=[revised_targets(subset(O,slice(i,i+32)),subset(rep,slice(i,i+32)),gt[i:i+32],valid[i:i+32],masks[i:i+32],boundary[i:i+32],has_mask[i:i+32]) for i in range(0,len(gt),32)]
        proxy=torch.cat([x[0] for x in pair]);masked=torch.cat([x[1] for x in pair]);assert torch.equal(proxy==-100,masked==-100)
        count=torch.bincount(proxy[proxy>=0],minlength=4).float();rw=(count.sum()/count.clamp_min(1)).sqrt();rw=(rw/rw.mean()).clamp_max(5)
        write(RUN/f'TARGET_DIAGNOSIS_seed{seed}.json',dict(frames=len(ids),auxiliary_complete_mask_frames=int((valid.all(-1)&has_mask).sum()),proxy_class_counts=count.tolist(),mask_class_counts=torch.bincount(masked[masked>=0],minlength=4).tolist(),changed_positive_samples=int((proxy!=masked).sum()),common_role_weights=rw.tolist(),ids_sha256=digest(json.dumps(ids))))
        torch.save(dict(proxy=proxy.cpu(),masked=masked.cpu(),mask=masks.cpu(),ids=ids),RUN/f'training_roles_seed{seed}.pt')
        rng=np.random.default_rng(17000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000]).reshape(2000,16);initial=None
        for arm in ARMS:
            labels=masked if arm.endswith("_mask") else proxy
            model=DetailEvaluator(seed).cuda();ih=state_sha(model.state_dict())
            if initial is None:initial=ih
            assert ih==initial
            o=subset(O,slice(0,4));r=subset(rep,slice(0,4))
            with torch.no_grad():
                a=model(o,r,arm);assert torch.equal(a['points'],r['base']) and not a['choice'].any()
                g=geometry(o,r,a['candidates']);flipped=geometry(o,dict(r,lines=-r['lines']),a['candidates'])
                assert torch.allclose(g['sample'],flipped['sample'],atol=1e-4,rtol=0)
            directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False)
            write(directory/'START.json',dict(arm=arm,seed=seed,frames=len(ids),initial_sha256=ih,plan_sha256=digest(plan.tobytes().hex()),ids_sha256=digest(json.dumps(ids)),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,ii in enumerate(plan):
                    ii=torch.tensor(ii,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm)
                    loss,gl,aux=objective(out,truth[ii],valid[ii],w[ii],arm,labels[ii],rw)
                    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step()
                    f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),gain_loss=float(gl.detach()),aux_loss=float(aux.detach()),gradient_norm=gn))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),'seconds',round(time.monotonic()-tick),flush=True);f.flush()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),directory/'final.pt')
            done=dict(arm=arm,seed=seed,elapsed_seconds=time.monotonic()-tick,sha256=sha(directory/'final.pt'));write(directory/'COMPLETE.json',done);runs.append(done)
            checks.append(dict(seed=seed,arm=arm,initial_keep=True,line_sign_invariance=True,finite_updates=2000));del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=36000,runs=runs))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
