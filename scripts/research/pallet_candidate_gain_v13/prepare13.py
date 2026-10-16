"""Freeze parent proposals and context without opening point targets."""
from context13 import *

def main():
    setup();RUN.mkdir(parents=True,exist_ok=True);lock=RUN/'PROTOCOL.json'
    if lock.exists():raise SystemExit('V13 already registered')
    sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')}
    parent_sha={f'{a}_{s}':sha(V12/'models'/f'{a}_seed{s}'/'final.pt') for a in ('learned_gate','gated_amodal') for s in SEEDS}
    write(lock,dict(arms=ARMS,seeds=SEEDS,steps=2000,batch=16,lr=3e-4,alphas=ALPHAS,
        architecture='Frozen YOLO + frozen V10 local + frozen V12 learned_gate baseline and gated_amodal proposal + learned candidate gain scorer.',
        candidates='base=V12 learned_gate; full=V12 gated_amodal. Candidate=base+alpha*(full-base), alpha0/.5/1. Retention is always available. No new directions proposed in this bounded pilot.',
        target='Original-ID Euclidean point error(base)-error(candidate) from synthetic GT; candidate0 gain exactly0. Frozen proposals detached. No real GT or masks at inference.',
        loss='SmoothL1 gain regression beta1px over nonzero candidates + .25 pairwise soft preference CE, preference temperature1px.',
        readout='argmax predicted gain over keep/half/full; ties choose keep. Same discrete readout during reporting of training and inference. Head trained by gain/ranking supervision, no GT filtering or selftraining.',
        arms_description='constant: input-independent3 candidate scores, baseline0. geometry: candidate/box/image coordinates, displacements and parent routing weights. image: same plus candidate image patches and frozen parent point context. Geometry still uses image-derived upstream predictions.',
        evaluation='Same reused synthetic505 and real297/9sessions, all six V11 conditions, same detections; real manual clicks only. No selection or tuning on real.',
        acceptance='Every real seed: clean within1% of learned_gate; crop20 outside within1% of learned_gate AND lower than full; retain>=80% of full-proposal improvement vs base on corner20 hidden and crop40 outside. Image scorer must beat geometry on crop20 outside and not worsen the two hard strata; paired session CI supports crop20 improvement over full. Catastrophe counts reported.',
        joint_followup='Only if the complete gate passes, run matched-budget joint refinement. Otherwise finish diagnosis without more training.',
        sources=sources,parent_checkpoint_sha256=parent_sha,parent_summary_sha256=sha(V12/'SUMMARY.json'),
        limits='Stage1 changes whole predictor architecture by appending a trained module; YOLO topology unchanged. Parent proposals were fitted on synthetic training data, so score training uses in-sample parent errors. Reused DEV artificial stress, not fresh natural-occlusion validation. No Hough benefit claim.'))
    ids,O=training_observations('cuda:0');write(RUN/'TRAIN_IDS.json',dict(ids=ids))
    trainO=O;audits=[]
    for split,condition in [('train','mixed')]+[(s,c) for s in ('synth_val','real') for c in CONDITIONS]:
        if split=='train':O=trainO;manifest=None
        else:manifest,O=P.C.load_observations(split,condition,'cuda:0')
        for seed in SEEDS:
            localnet=local(seed);localnet.requires_grad_(False)
            safe=parent_model(seed,'learned_gate');full=parent_model(seed,'gated_amodal');captured={}
            def hook(_m,args,out):captured['context']=(args[0]+out).detach()
            handle=full.feed.register_forward_hook(hook);parts=[]
            with torch.inference_mode():
                for first in range(0,len(O['points']),32):
                    sl=slice(first,first+32);obs=subset(O,sl);bp=base_points(localnet,obs)
                    b=safe(obs,bp,'learned_gate');f=full(obs,bp,'gated_amodal')
                    parts.append(dict(base=b['points'].cpu(),full=f['points'].cpu(),context=captured['context'].cpu(),probabilities=f['probabilities'].cpu()))
            handle.remove();rep={k:torch.cat([p[k] for p in parts]) for k in parts[0]}
            d=RUN/'parents'/split/condition;d.mkdir(parents=True,exist_ok=True);torch.save(rep,d/f'seed{seed}.pt')
            parity={}
            if split!='train':
                for arm,key in [('learned_gate','base'),('gated_amodal','full')]:
                    ref=np.load(V12/'predictions'/split/condition/f'{arm}_seed{seed}.npz');assert ref['ids'].tolist()==[r['id'] for r in manifest['rows']]
                    error=float(np.max(np.abs(rep[key].numpy()-ref['points'][ref['detected']])));assert error<.001
                    parity[key]=error
            audits.append(dict(split=split,condition=condition,seed=seed,frames=len(rep['base']),cache_sha256=sha(d/f'seed{seed}.pt'),parent_parity_px=parity))
            del localnet,safe,full
        print('PARENT CACHE',split,condition,len(O['points']),flush=True)
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,targets_opened_for_forward=False,rows=audits))

if __name__=='__main__':main()
