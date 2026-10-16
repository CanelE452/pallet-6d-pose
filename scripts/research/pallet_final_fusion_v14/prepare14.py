"""New training observations; original real/synthetic evaluation remains immutable."""
import argparse
import cv2
from context14 import *

def register():
    RUN.mkdir(parents=True,exist_ok=True);path=RUN/'PROTOCOL.json'
    if path.exists():return
    train=read(P.EXPORT/'train.json')['records'];cal=read(P.EXPORT/'calibration.json')['records'];val=read(P.EXPORT/'synth_val.json')['records']
    allowed={r['id'] for r in read(V11/'POPULATIONS.json')['train']}
    selected=sorted([r for r in train if r['id'] in allowed],key=lambda r:digest('v14-insample:'+r['id']))[:len(cal)]
    assert len(cal)==len(selected)==253
    sets=[{r['id'] for r in group} for group in (train,cal,val)]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    hashes=[{sha(ROOT/r['original_image']) for r in group} for group in (train,cal,val)]
    assert not (hashes[0]&hashes[1] or hashes[0]&hashes[2] or hashes[1]&hashes[2])
    pop={}
    for split,records in [('insample',selected),('heldout',cal)]:
        rows=[]
        for r in records:
            o=torch.load(P.EXPORT/r['observation'],map_location='cpu',weights_only=True)
            rows.append(dict(id=r['id'],image=r['original_image'],padded=True,K=o['K'][0].tolist(),dims=o['dims'][0].tolist(),session=r['session']))
        pop[split]=rows
    write(RUN/'POPULATIONS.json',pop)
    checkpoints={}
    for seed in SEEDS:
        for version,arm in [(V10,'local'),(V12,'learned_gate'),(V12,'gated_amodal'),(V11,'lines_joint')]:
            p=version/'models'/f'{arm}_seed{seed}'/'final.pt';checkpoints[str(p.relative_to(ROOT))]=sha(p)
    write(path,dict(arms=ARMS,seeds=SEEDS,steps=2000,batch=16,lr=3e-4,conditions=CONDITIONS,
        final_candidate='Frozen YOLO + local correction suppression + amodal recovery + DHT line-conditioned residual + learned five-candidate evaluator. Keep/half/full amodal/half/full line candidates. No forced point-to-line projection or selftraining.',
        proposal_detail='Base V12 learned_gate; amodal V12 gated_amodal; line proposal OptionalRefiner V11 lines_joint evaluated around the base points, using its own joint-trained DHT. Line delta is added to base. All parents frozen.',
        scoring='Candidate and point identity, candidate patches and differences from base patches, parent global point context, geometry and line descriptor. Predict error reduction relative to keep; keep exact0; argmax, ties keep. Same SmoothL1 + .25 soft preference loss across arms.',
        training='Insample control:253 SHA-selected original training images; heldout arms:253 former synthetic calibration images excluded from refinement-parent training. Both freshly extracted in6conditions. Same update budget; observation counts may differ because of detections. heldout_points/fusion share exact batches and initialization.',
        split_audit='Original IDs and image SHA sets train/calibration/synth_val disjoint. Calibration previously used for diagnostics, now training: no longer an evaluation set. This does not certify exclusion from original YOLO pretraining or fresh scene independence.',
        evaluation='Fixed reused real297/9sessions/manual1251points and synthetic505, six conditions. No real training, real checkpoint selection, posthoc thresholds or GT input. Original IDs and failed detections preserved.',
        comparisons='YOLO, V10 local/directional, V11 local_extra/lines_joint, V12 learned_gate/gated_amodal, V13 image evaluator, plus matched V14 controls. Existing parents had different prior training budgets; only V14 control budgets matched.',
        acceptance='All3real seeds: clean <=1% worse than V12 learned_gate; crop20 outside <=1% worse than learned_gate; >=80% retention of amodal gains on corner20 hidden and crop40 outside. Additional Hough claim requires fusion lower than heldout_points in all3 seeds and session95%CI upper<0 on relevant stratum, >=1% relative gain each seed; same-model donor/off results reported separately. Do not choose deployment checkpoint on real evaluation.',
        primary='Conditional frame mean raw px; hidden/outside point means, PCK5/10, P90, full-population normalized failure1. Paired9session bootstrap real; image bootstrap synthetic. Per-condition verdict, no all-purpose success from pooled stress means.',
        donor='Same frozen evaluator and point/amodal candidates, wrong-image DHT lines affine mapped into target ROI; recompute line proposal and descriptor. Diagnostic only, no tuning.',
        parent_checkpoints=checkpoints,sources={str(p.relative_to(ROOT)):sha(p) for p in KIT.glob('*.py')}))
    write(RUN/'SPLIT_AUDIT.json',dict(status='PASS',train_images=len(train),heldout_images=len(cal),evaluation_synthetic_images=len(val),insample_selected=len(selected),id_disjoint=True,image_sha_disjoint=True))

def extract():
    setup();cv2.setNumThreads(1);register();pop=read(RUN/'POPULATIONS.json');extractor=PREP.Extractor()
    for split,records in pop.items():
        for condition in CONDITIONS:
            d=RUN/'cache'/split/condition
            if (d/'manifest.json').exists():continue
            d.mkdir(parents=True,exist_ok=True);rows=[];chunks=[];pending=[];n=0
            if condition!='clean':
                cm,co=load(split,'clean');lookup={r['id']:r for r in cm['rows']}
            for i,rec in enumerate(records):
                im=PREP.raw_image(rec);h,w=im.shape[:2]
                clean=lookup[rec['id']] if condition!='clean' else {};box=clean.get('box')
                if box is None:box=[.2*w,.2*h,.8*w,.8*h]
                if condition!='corner20':canvas,meta=PREP.stress(im,box,condition,rec['id'])
                else:
                    box=np.array(box);corner=[0,1,4,5][int(digest('v11-corner:'+rec['id'])[:8],16)%4]
                    center=co['points'][clean['observation_index'],corner].numpy() if clean['detected'] else (box[:2]+box[2:])/2
                    side=.2*np.linalg.norm(box[2:]-box[:2]);a=max(0,min(w-1,int(round(center[0]-side/2))));b=max(a+1,min(w,int(round(center[0]+side/2))))
                    u=max(0,min(h-1,int(round(center[1]-side/2))));v=max(u+1,min(h,int(round(center[1]+side/2))))
                    color=np.random.default_rng(int(digest('v11-corner-color:'+rec['id'])[:8],16)).integers(40,216,3).astype(np.uint8)
                    canvas=im.copy();canvas[u:v,a:b]=color;meta=dict(origin=[0,0],mask=[a,u,b,v],original_hw=[h,w],fill_bgr=color.tolist(),targeted_predicted_corner=corner)
                K=np.array(rec['K']);K[:2,2]-=meta['origin'];o,info=extractor(canvas,K.tolist(),rec['dims'])
                rows.append(dict(**rec,condition=condition,**meta,**info,image_hw=list(canvas.shape[:2]),image_sha256=sha(ROOT/rec['image']),observation_index=n if o is not None else None))
                if o is not None:pending.append(o);n+=1
                if len(pending)>=64 or (i+1==len(records) and pending):
                    name=f'obs_{len(chunks):03d}.pt';torch.save({k:torch.cat([x[k] for x in pending]) for k in KEYS},d/name);chunks.append(name);pending=[]
            write(d/'manifest.json',dict(rows=rows,chunks=chunks,chunk_sha256={p:sha(d/p) for p in chunks},policy='Same V11 extractor, no GT for stress or detection selection.'))
            print('EXTRACTED',split,condition,n,len(rows),flush=True)
    extractor.cap.handle.remove()
    for split in pop:
        original='train' if split=='insample' else 'calibration';source={r['id']:r for r in read(V10/f'targets_{original}.json')['rows']}
        for condition in CONDITIONS:
            m,_=load(split,condition);rows=[]
            for rec in m['rows']:
                t=source[rec['id']];g=np.array(t['points']);shift=g-np.array(rec['origin']);h,w=rec['image_hw'];outside=(shift[:,0]<0)|(shift[:,0]>=w)|(shift[:,1]<0)|(shift[:,1]>=h);hidden=np.zeros(8,bool)
                if rec['mask'] is not None:
                    a,u,b,v=rec['mask'];hidden=(g[:,0]>=a)&(g[:,0]<b)&(g[:,1]>=u)&(g[:,1]<v)
                rows.append(dict(id=rec['id'],points=shift.tolist(),valid=t['valid'],hidden=hidden.tolist(),outside=outside.tolist(),session=rec['session'],original_points=g.tolist()))
            write(RUN/'targets'/f'{split}_{condition}.json',dict(rows=rows))
    write(RUN/'EXTRACTION_COMPLETE.json',dict(complete=True,no_gt_image_placement=True))

def parents():
    setup();assert read(RUN/'EXTRACTION_COMPLETE.json')['complete'];receipts=[]
    for seed in SEEDS:
        loc=local(seed).requires_grad_(False);base=C13.parent_model(seed,'learned_gate');amodal=C13.parent_model(seed,'gated_amodal')
        ck=V11/'models'/f'lines_joint_seed{seed}'/'final.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
        lm=P.C.dht(seed);lm.load_state_dict(state['dht_state_dict']);lm.eval().requires_grad_(False)
        lr=OptionalRefiner(seed).cuda().eval().requires_grad_(False);lr.load_state_dict(state['state_dict']);capture={}
        handle=amodal.feed.register_forward_hook(lambda mod,args,out:capture.update(context=args[0]+out))
        for split in ('insample','heldout','synth_val','real'):
            for condition in CONDITIONS:
                d=RUN/'parents'/split/condition;d.mkdir(parents=True,exist_ok=True)
                variants=('actual','donor') if split in ('real','synth_val') else ('actual',)
                if all((d/f'seed{seed}_{v}.pt').exists() for v in variants):continue
                m,O=load(split,condition,'cuda:0');n=len(O['points']);parts=[];ls=[];ms=[]
                with torch.inference_mode():
                    for first in range(0,n,32):
                        o=subset(O,slice(first,first+32));bp=base_points(loc,o);b=base(o,bp,'learned_gate');a=amodal(o,bp,'gated_amodal');ll,mm=predict_lines(lm,o)
                        ls.append(ll);ms.append(mm);parts.append(dict(base=b['points'],amodal=a['points'],context=capture['context'],probabilities=a['probabilities']))
                    common={k:torch.cat([p[k] for p in parts]) for k in parts[0]};ll=torch.cat(ls);mm=torch.cat(ms)
                    ids=[r['id'] for r in m['rows'] if r['detected']];order=sorted(range(n),key=lambda i:digest('v14-donor:'+ids[i]));indices=np.empty(n,dtype=np.int64)
                    for j,i in enumerate(order):indices[i]=order[(j+1)%n]
                    dl,dm=donor_lines(ll,mm,O['box'],torch.tensor(indices,device='cuda:0'))
                    for variant in variants:
                        l,mass=(ll,mm) if variant=='actual' else (dl,dm);pieces=[]
                        for first in range(0,n,32):
                            sl=slice(first,first+32);o=subset(O,sl);b=common['base'][sl];out=lr(o,b,l[sl],mass[sl],'lines_joint')
                            normals=l[sl,...,:2]/l[sl,...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-8)
                            c=l[sl,...,2]/l[sl,...,:2].norm(dim=-1).clamp_min(1e-8);w=out['weights'][...,:-1]
                            diag=(o['box'][:,2:]-o['box'][:,:2]).norm(dim=-1).clamp_min(1)
                            r=((b[:,:,None]*normals[:,None]).sum(-1)+c[:,None])/diag[:,None,None]
                            vec=(w[...,None]*r[...,None]*normals[:,None]).sum(2);A=out['A']
                            descriptor=torch.cat((A[...,0,0,None],A[...,1,1,None],A[...,0,1,None],vec,w.sum(-1,keepdim=True)),-1)
                            pieces.append(dict(line_delta=out['delta'],line_descriptor=descriptor,line_weights=out['weights']))
                        rep=dict(common,**{k:torch.cat([p[k] for p in pieces]) for k in pieces[0]},lines=l,mass=mass)
                        torch.save({k:v.cpu() for k,v in rep.items()},d/f'seed{seed}_{variant}.pt')
                if split in ('real','synth_val'):
                    old=C13.parent_cache(split,condition,seed)
                    assert torch.allclose(common['base'].cpu(),old['base'],atol=.001,rtol=0)
                    assert torch.allclose(common['amodal'].cpu(),old['full'],atol=.001,rtol=0)
                receipts.append(dict(seed=seed,split=split,condition=condition,frames=n))
                print('PARENTS',seed,split,condition,n,flush=True)
        handle.remove();del loc,base,amodal,lm,lr
    files={str(p.relative_to(RUN)):sha(p) for p in (RUN/'parents').rglob('*.pt')}
    write(RUN/'PREPARATION_COMPLETE.json',dict(complete=True,files=files,rows=receipts,targets_used_for_parent_inference=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['extract','parents'],default='extract');args=ap.parse_args()
    extract() if args.phase=='extract' else parents()
