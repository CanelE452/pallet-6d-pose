"""Preregistered synthetic-only, matched seed/batch training; final checkpoint."""
from context16 import *
from model16 import EvidenceEvaluator,role_targets,objective,weights,geometry
from cli_train import state_sha

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Refusing to overwrite training')
    write(RUN/'PROTOCOL.json',dict(question='Can candidate-specific finite line strip evidence and incident boundary-role supervision resolve correction selection?',
        arms=ARMS,seeds=SEEDS,steps=2000,batch=16,optimizer='AdamW lr3e-4 wd1e-4 float32 clip5; final only',
        controls='All eight candidates, same model initialization and batch order. control zero strip embedding; strip adds evidence; role adds auxiliary CE .25; balanced_role adds V15 harm/neutral/benefit weights; median_role tau .5 vs .2. Same-model role evidence off and wrong-image strip features preserve all candidates.',
        architecture='Frozen YOLO + frozen point/amodal/DHT parents; candidate-specific projection to top3 associated DHT lines, 7 along x3 across samples on finite local strips. Raw-image outside samples masked. 192->8 feature projection; role tokens pooled per candidate; learned relative gain argmax with keep tie.',
        labels='Synthetic camera-facing 0123 original corner annotations. 12 finite cuboid edges; incident long, incident height, other edge, background. Center within2.5px segment; artificial mask/outside unsupported. Incomplete eight-corner labels excluded ONLY from auxiliary CE, not point loss. No true surface/self-occlusion line labels. No GT at inference.',
        train='253 former calibration synthetic images, 1462 detected observations across6 conditions; excluded from refinement parent training per V14 audit. No original YOLO scene-independence claim.',
        evaluation='Reused real297/9sessions/manual1251 corners and synthetic505, 8 conditions. No real training or real-driven selection. Natural occlusion unknown; crop10/30 are evaluation-only ratios.',
        acceptance='All3seeds clean and crop10/20/30 outside <=1.01 base; corner20hidden and crop40outside retain>=80% amodal improvement. Incremental evidence claim >=1% improvement all3seeds and real9session paired95%CI upper<0 vs matched control. No deployment based on reused DEV.',
        scope='Frozen-feature quick architecture test; not end-to-end line-detector training, not impossibility proof.',parent_protocol_sha256=sha(V.RUN/'PROTOCOL.json')))
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
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);assert ids==bound
        cand=candidates(O,rep);err=(cand-gt[:,:,None]).norm(dim=-1);truth=(err[...,:1]-err).detach();w,counts=weights(truth,valid,kind)
        labels=torch.cat([role_targets(subset(O,slice(i,i+32)),subset(rep,slice(i,i+32)),gt[i:i+32],valid[i:i+32],masks[i:i+32]) for i in range(0,len(gt),32)])
        count=torch.bincount(labels[labels>=0],minlength=4).float();rw=(count.sum()/count.clamp_min(1)).sqrt();rw=(rw/rw.mean()).clamp_max(5)
        write(RUN/f'TARGET_DIAGNOSIS_seed{seed}.json',dict(frames=len(ids),auxiliary_complete_frames=int(valid.all(-1).sum()),class_counts=count.tolist(),role_weights=rw.tolist(),ids_sha256=digest(json.dumps(ids)),balancing=counts))
        torch.save(dict(labels=labels.cpu(),mask=masks.cpu(),ids=ids),RUN/f'training_roles_seed{seed}.pt')
        rng=np.random.default_rng(16000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000]).reshape(2000,16);initial=None
        for arm in ARMS:
            model=EvidenceEvaluator(seed).cuda();ih=state_sha(model.state_dict())
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
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=30000,runs=runs))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
