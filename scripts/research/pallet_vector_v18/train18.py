"""Point GT only: matched local/line x symmetric/harm-aware vector learning."""
from context18 import *
from model18 import VectorRefiner,objective
from cli_train import state_sha

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'TRAIN_PROTOCOL.json').exists():raise SystemExit('Refusing overwrite')
    write(RUN/'PROTOCOL.json',dict(question='Does direct direction/magnitude supervision bypass candidate-selection failure?',arms=ARMS,seeds=SEEDS,steps=2000,batch=16,radius_fraction=RADIUS,
        architecture='Frozen V12 base/amodal context and V11 DHT. 3x3 point patches+context ->64 token; top3 associated finite7x3 line strips ->96 token; shared MLP outputs2D raw vector. delta=.15*box_diagonal*raw/sqrt(1+norm(raw)^2), zero initialization. No pre-enumerated candidate selection, line projection, GT filter, mask or line role supervision.',
        controls='Point arms zero optional96line token. Same parameter tensors, initial values, batch plan and2000updates. Local and line arms share parent representations; point control means no NEW line branch, not a line-free lineage. vector loss=mean Euclidean point error; risk loss adds2*mean positive error increase vs immutable base. No predicted gate/threshold or stop-grad trick.',
        training='Original253 synthetic images,1462 detected observations across6conditions. All original valid corner IDs; no real training. Refinement parents frozen. Fixed15% radius preregistered; feasibility diagnostics only, no radius search.',
        evaluation='Same reused synth505 and real297/9sessions/1251manual corners,8conditions. Original IDs/detections/reflect100 parent wrapper. crop10/30 evaluation-only. Artificial stress; no natural occlusion labels or independent final holdout.',
        acceptance='All3seeds clean/crop10/20/30outside <=1.01 base, corner20hidden and crop40outside retain>=80% original amodal improvement. Incremental line effect compared with matched point arm requires all3seeds>=1% and real9session paired95%CI upper<0. Multiplicity uncorrected.',
        optimizer='AdamW3e-4 wd1e-4 float32 clip5; fixed last checkpoint, no real selection',parent_protocol_sha256=sha(P.RUN/'PROTOCOL.json')))
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')},targets={str(p.relative_to(ROOT)):sha(p) for p in (C.RUN/'targets').glob('heldout_*.json')},parents={str(p.relative_to(ROOT)):sha(p) for p in (C.RUN/'parents/heldout').glob('*/seed*_actual.pt')},steps=2000,no_real_training=True))
    runs=[];checks=[];feas={}
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=training(seed);old=candidates(O,rep);old_error=(old-gt[:,:,None]).norm(dim=-1);before=(rep['base']-gt).norm(dim=-1);radius=RADIUS*(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1)
        oracle=(before-radius[:,None]).clamp_min(0);f={}
        for label,code in [('all',None),('hidden',1),('outside',2)]:
            mask=valid if code is None else valid&(kind==code)
            f[label]=dict(points=int(mask.sum()),base_error=float(before[mask].mean()),old8_oracle=float(old_error.min(-1).values[mask].mean()),bounded_vector_infimum=float(oracle[mask].mean()),within_vector_radius_fraction=float((before<=radius[:,None])[mask].float().mean()))
        feas[str(seed)]=f;write(RUN/'TRAIN_FEASIBILITY.json',dict(rows=feas,scope='GT-privileged training-only attainable-set diagnostic, not predicted performance. Smooth radial squash approaches radius asymptotically. Not used to select radius/checkpoint.'))
        rng=np.random.default_rng(18000+seed);plan=[]
        while len(plan)<32000:plan.extend(rng.permutation(len(gt)).tolist())
        plan=np.array(plan[:32000]).reshape(2000,16);initial=None
        for arm in ARMS:
            model=VectorRefiner(seed).cuda();ih=state_sha(model.state_dict())
            if initial is None:initial=ih
            assert ih==initial;o=subset(O,slice(0,4));r=subset(rep,slice(0,4))
            with torch.no_grad():
                out=model(o,r,arm);assert torch.equal(out['points'],r['base']);assert (out['delta']==0).all()
                flip=model(o,dict(r,lines=-r['lines']),arm);assert torch.equal(flip['sample_center'],out['sample_center'])
                po=model(o,r,'point_vector');off=model(o,r,'line_vector',line_off=True);assert torch.equal(po['points'],off['points'])
            directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False)
            write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=ih,plan_sha256=digest(plan.tobytes().hex()),ids_sha256=digest(json.dumps(ids)),frames=len(ids),parameters=sum(p.numel() for p in model.parameters())))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as file:
                for step,ii in enumerate(plan):
                    ii=torch.tensor(ii,device='cuda:0');out=model(subset(O,ii),subset(rep,ii),arm);loss,primary,harm=objective(out,gt[ii],valid[ii],arm)
                    assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step()
                    file.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(primary.detach()),harm=float(harm.detach()),gradient_norm=norm))+'\n')
                    if (step+1)%500==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),3),'seconds',round(time.monotonic()-tick),flush=True);file.flush()
            ck=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),ck)
            done=dict(arm=arm,seed=seed,steps=2000,sha256=sha(ck),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done);checks.append(dict(arm=arm,seed=seed,initial_keep=True,line_sign_invariance=True,finite_updates=2000));del model,opt
    write(RUN/'PREFLIGHT.json',dict(status='PASS',rows=checks));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=24000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
