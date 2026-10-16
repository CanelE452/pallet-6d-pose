"""Fixed 2x2 objective/context intervention with extra-line ablations."""
from context22 import *
from model22 import ObjectiveGate,objective22
from model20 import CauseGate
from cli_train import state_sha

def main():
    setup();assert read(RUN/'PROBE.json')['status']=='PASS';assert not (RUN/'TRAIN_PROTOCOL.json').exists()
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,lr=3e-4,weight_decay=1e-4,gradient_clip=5,float32=True,updates=36000,
      hypotheses=['Direct final-point Euclidean loss better weights attainable error reduction than movement-weighted alpha BCE.','Removing frozen64D parent context discourages a synthetic contextual shortcut; current image/geometry still available.'],
      factors='BCE versus Euclidean final-point error x retain64D parent context versus zero64D. Two pixel-loss local controls remove only extra line evidence. Identical tensors, initialization, batches, learning rate, final checkpoint.',
      objective='BCE is original V20 unit_weighted. Pixel loss is mean raw-pixel Euclidean error over all original valid points. No per-type weighting, preservation penalty, GT filter, pseudo-labels, or self-training. Pixel gradient traverses sigmoid and may saturate; explicitly diagnose.',
      context='Only rep.context64D is zeroed. Frozen observability probabilities, amodal delta, point ID, original features, geometry and proposed direction/length are retained. This is not a complete removal of parent-derived information.',
      architecture='Frozen V18 vector d, ObjectiveGate derives from CauseGate, p=base+sigmoid(logit)*d. No new direction or end-to-end YOLO/DHT training.',
      training='Original253synthetic/1462detections/6conditions,2000updates per model, same original-image grouped plan as V20. Use original pool to isolate objective/context without new source composition change.',
      evaluation='Reused real297/9sessions/manual1251 and synth505DEV x8conditions, plus V21 new_check253 x6 now also reused. No untouched final holdout or new real-GT fitting.',
      acceptance='All3seeds clean andcrop10/20/30outside<=1.01base; cornerhidden/crop40outside retain>=90% V18 full-vector gain. Pairwise everyseed>=1% and real9session95%CI upper<0; no multiple comparison correction.',
      interventions='no_context and no_extra_lines probe are OOD sensitivity only; trained 2x2 required for causal comparison.',probe_sha256=sha(RUN/'PROBE.json')))
    write(RUN/'TRAIN_PROTOCOL.json',dict(protocol_sha256=sha(RUN/'PROTOCOL.json'),sources={str(p.relative_to(ROOT)):sha(p) for p in (KIT/'train22.py',KIT/'model22.py',KIT/'context22.py')},frozen_model_sha256=sha(OLD.KIT/'model20.py'),targets=read(OLD.RUN/'TRAIN_PROTOCOL.json')['targets'],parents=read(OLD.RUN/'TRAIN_PROTOCOL.json')['parents']))
    runs=[];repro=[]
    for seed in SEEDS:
        O,r,gt,valid,ids,kind=proposal_training(seed);g=read(PREV.RUN/f'GROUPS_old_seed{seed}.json');assert ids==g['ids'];plan=g['plan'];write(RUN/f'GROUPS_seed{seed}.json',g)
        for arm in ARMS:
            model=ObjectiveGate(seed).cuda();initial=state_sha(model.state_dict());directory=RUN/'models'/f'{arm}_seed{seed}';directory.mkdir(parents=True,exist_ok=False);write(directory/'START.json',dict(seed=seed,arm=arm,initial_sha256=initial,plan_sha256=g['plan_sha256'],ids_sha256=digest(json.dumps(ids))))
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);tick=time.monotonic()
            with (directory/'TRACE.jsonl').open('w') as f:
                for step,indices in enumerate(plan):
                    ii=torch.tensor(indices,device='cuda:0');out=model(subset(O,ii),subset(r,ii),arm);loss,error,target=objective22(out,gt[ii],valid[ii],arm);opt.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True));opt.step();f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),point_error=float(error.detach()),gradient_norm=norm,observations=len(indices)))+'\n')
                    if (step+1)%1000==0:print('TRAIN',arm,seed,step+1,round(float(loss.detach()),4),flush=True);f.flush()
            p=directory/'final.pt';torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},arm=arm,seed=seed,steps=2000),p);done=dict(seed=seed,arm=arm,steps=2000,sha256=sha(p),elapsed_seconds=time.monotonic()-tick);write(directory/'COMPLETE.json',done);runs.append(done)
            if arm=='unit_bce':
                control=CauseGate(seed).cuda().eval();control.load_state_dict(torch.load(OLD.RUN/'models'/f'unit_weighted_seed{seed}'/'final.pt',weights_only=True)['state_dict']);maximum=0.
                with torch.no_grad():
                    for first in range(0,len(ids),32):
                        sl=slice(first,first+32);args=(subset(O,sl),subset(r,sl));maximum=max(maximum,float((model(*args,arm)['points']-control(*args,'unit_weighted')['points']).abs().max()))
                assert maximum<.001;repro.append(dict(seed=seed,max_point_error_px=maximum));del control
            del model,opt
    write(RUN/'REPRODUCTION.json',dict(status='PASS',rows=repro));write(RUN/'TRAINING_COMPLETE.json',dict(complete=True,updates=36000,runs=runs));print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
