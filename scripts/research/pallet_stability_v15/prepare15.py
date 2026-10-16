"""Preregister hypotheses and generate unseen crop severities without GT placement."""
import argparse
import cv2
from context15 import *

def register():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'PROTOCOL.json').exists():return
    write(RUN/'PROTOCOL.json',dict(arms=ARMS,seeds=SEEDS,steps=2000,batch=16,lr=3e-4,
        hypotheses=['Lower quantile gain discourages optimistic harmful corrections.','Within exposure/candidate, sign-balanced synthetic loss reduces misleading positive correction prior.','Explicit larger line-normal proposals overcome tiny learned V14 line movements.'],
        control_chain='median_old tau.5 -> risk_old tau.2 -> balanced_old tau.2 plus sign weights -> balanced_rich same plus3 geometric line candidates. All share8-slot head initialization and sample plan. Existing V14 is an additional frozen reference, not identical loss/initialization.',
        candidate='Old5 V14 candidates plus projection of base toward top3 predicted attention lines, signed normal displacement clipped to5% observed box diagonal. Optional candidates only; not forced final snapping. DHT and all parents frozen.',
        loss='Pinball gain regression only; no preference CE. Median tau.5; conservative tau.2, not a calibrated20% real-world guarantee. Balance harm<-0.25px / neutral / benefit>0.25px within synthetic exposure class and candidate; inverse frequency weight capped10. All valid synthetic observations retained.',
        training='V14 heldout synthetic253 images,1462 detected observations across six conditions. Now training, not evaluation. Fixed2000updates x4arms x3seeds. No real training, pseudo-labels or GT inference filtering.',
        evaluation='Same reused real297/9sessions/manual1251 and synth_val505 in6conditions plus preregistered new crop10/crop30 image-level stress. New ratios are not new independent scenes. No post-evaluation tuning or checkpoint selection.',
        acceptance='For each arm all3real seeds: clean<=1.01*V12gate, crop20 outside<=1.01*V12gate; preserve>=80% V12amodal-vs-gate gains on corner20 hidden and crop40 outside. New crop10/crop30 outside<=1.01*V12gate. Report per-session results, harmful-applied rates, catastrophes, P90 and PCK; keep-all fails hard improvement retention. Claim stability only if core and new-ratio gates pass.',
        incremental_lines='balanced_rich vs balanced_old: every seed >=1% gain plus paired session95%CI upper<0 in relevant target stratum. Same-model line-off/donor diagnostics. Candidate oracle only posthoc GT diagnostic.',
        parent_v14_protocol_sha256=sha(V14/'PROTOCOL.json'),v14_training_sha256=sha(V14/'TRAIN_PROTOCOL.json'),
        parent_checkpoints=read(V14/'PROTOCOL.json')['parent_checkpoints'],
        sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')}))

def extract():
    setup();cv2.setNumThreads(1);register();extractor=PREP.Extractor();pop=read(V11/'POPULATIONS.json')
    for split in ('synth_val','real'):
        clean=manifest(split,'clean');lookup={r['id']:r for r in clean['rows']}
        for condition in NEW_CONDITIONS:
            d=RUN/'cache'/split/condition;d.mkdir(parents=True,exist_ok=True)
            if (d/'manifest.json').exists():continue
            rows=[];chunks=[];pending=[];n=0
            for i,rec in enumerate(pop[split]):
                im=PREP.raw_image(rec);h,w=im.shape[:2];box=lookup[rec['id']]['box']
                if box is None:box=[.2*w,.2*h,.8*w,.8*h]
                canvas,meta=PREP.stress(im,box,condition,rec['id']);K=np.array(rec['K']);K[:2,2]-=meta['origin'];o,info=extractor(canvas,K.tolist(),rec['dims'])
                rows.append(dict(**rec,condition=condition,**meta,**info,image_hw=list(canvas.shape[:2]),image_sha256=sha(ROOT/rec['image']),observation_index=n if o is not None else None))
                if o is not None:pending.append(o);n+=1
                if len(pending)>=64 or (i+1==len(pop[split]) and pending):
                    name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([x[k] for x in pending]) for k in KEYS},d/name);chunks.append(name);pending=[]
            write(d/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(d/p) for p in chunks},policy='Same highest-confidence YOLO and reflect100, no GT placement. New unseen crop severity only.'))
            print('EXTRACTED',split,condition,n,len(rows),flush=True)
    extractor.cap.handle.remove()
    # Targets are separated until image-level extraction has completed.
    for split in ('synth_val','real'):
        source={r['id']:r for r in targets(split,'clean')}
        for condition in NEW_CONDITIONS:
            rows=[]
            for rec in manifest(split,condition)['rows']:
                t=source[rec['id']];g=np.array(t['points']);shift=g-np.array(rec['origin']);h,w=rec['image_hw']
                outside=(shift[:,0]<0)|(shift[:,0]>=w)|(shift[:,1]<0)|(shift[:,1]>=h)
                rows.append(dict(id=rec['id'],points=shift.tolist(),valid=t['valid'],original_points=g.tolist(),hidden=[False]*8,outside=outside.tolist(),session=rec['session']))
            write(RUN/'targets'/f'{split}_{condition}.json',dict(rows=rows))
    write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,new_ratios_only=True))

def parents():
    setup();assert read(RUN/'EXTRACTION_COMPLETE.json')['complete'];receipts=[]
    for seed in SEEDS:
        net=FinalPointLineNetwork(seed).eval().requires_grad_(False);capture={}
        hook=net.evaluator.register_forward_pre_hook(lambda m,args:capture.update(rep=args[1]))
        for split in ('synth_val','real'):
            for condition in NEW_CONDITIONS:
                d=RUN/'parents'/split/condition;d.mkdir(parents=True,exist_ok=True);m,O=load(split,condition,'cuda:0');parts=[]
                with torch.inference_mode():
                    for first in range(0,len(O['points']),32):
                        o=subset(O,slice(first,first+32));out=net(o);r=dict(capture['rep'],lines=out['lines'],line_weights=out['line_weights']);parts.append({k:v.cpu() for k,v in r.items()})
                r={k:torch.cat([p[k] for p in parts]) for k in REP_KEYS};torch.save(r,d/f'seed{seed}_actual.pt')
                receipts.append(dict(seed=seed,split=split,condition=condition,frames=len(r['base']),cache_sha256=sha(d/f'seed{seed}_actual.pt')))
                print('PARENTS',seed,split,condition,len(r['base']),flush=True)
        hook.remove();del net
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,rows=receipts,no_target_parent_input=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['extract','parents'],default='extract');a=ap.parse_args()
    extract() if a.phase=='extract' else parents()
