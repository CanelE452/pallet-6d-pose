"""One new RGB-only head; fixed source calibration and C2 selection only."""
import argparse, copy, hashlib, math, time
from pathlib import Path
import cv2, numpy as np, torch
import rgb_model as R
import train_rgb as T
import selection as prior_selection
import geometry as G

def batch_without_features(data, rows, device='cuda'):
    return {k:torch.from_numpy(np.array(data.arrays[k][rows],copy=True)).to(device)
            for k in ('points','boxes','point_valid','input_shape','gt_points','gt_valid')}

@torch.no_grad()
def run(args):
    doc,private,old=map(Path,(args.public_dir,args.private_dir,args.prior_public_dir))
    lock=T.read(doc/'PLAN_AND_CONTRACT.json')
    assert lock['status']=='LOCKED_BEFORE_FITS'
    assert T.read(doc/'RGB_ONLY_TRAIN_RECEIPTS.json')['actual_updates']==6000
    assert not (doc/'ROW_METRICS.jsonl.gz').exists()
    source,arrays,side,exact=G.inputs(args.source_root)
    selection_rows=np.flatnonzero(np.array([x['partition']=='selection' for x in source['records']])&(side['order']==2))
    assert len(selection_rows)==782
    assert hashlib.sha256(selection_rows.astype('<i8').tobytes()).hexdigest()==lock['selection']['row_sha256']
    old_rows=T.read(Path(args.prior_private_dir)/'SYNTHETIC_SELECTION_ROWS.json')
    reused=[r for r in old_rows if r['arm'] in ('M1','M3')]
    assert len(reused)==3128
    temperatures={};predictions=[];device_start=time.monotonic();T.configure()
    for seed in (1,2):
        env,refiner,data,head,order,path,targets,decode,loss=T.setup(args,seed)
        checkpoint=private/f'checkpoints/RGB_ONLY_seed{seed}.pt'
        receipt=T.read(doc/f'RGB_ONLY_TRAIN_RECEIPT_seed{seed}.json')
        assert receipt['status']=='DONE' and T.sha(checkpoint)==receipt['private_checkpoint']['sha256']
        ck=torch.load(checkpoint,map_location='cpu')
        assert ck['step']==3000 and ck['arm']=='RGB_ONLY'
        head.load_state_dict(ck['state_dict']);head.eval().requires_grad_(False)
        cal=np.flatnonzero(data.partitions=='calibration');assert len(cal)==1004
        totals=np.zeros(4);frames=0
        for begin in range(0,len(cal),16):
            rows=cal[begin:begin+16];batch=batch_without_features(data,rows)
            tiles=prior_selection.evaluation_tiles(data,rows,batch,head.base.displacements)
            output=head(batch,True,tiles);target=targets(output,batch['gt_points'],batch['gt_valid'])
            mask=target['support'];count=mask.sum(-1);used=count>0;frames+=int(used.sum())
            for j,temp in enumerate((.5,1.,2.,4.)):
                ce=-(target['distribution'].double()*(output['logits'].double()/temp).log_softmax(-1)).sum(-1)
                totals[j]+=float(((ce*mask).sum(-1)/count.clamp_min(1))[used].sum())
        candidates=[dict(temperature=t,score=float(v/frames)) for t,v in zip((.5,1.,2.,4.),totals)]
        best=min(candidates,key=lambda c:(c['score'],abs(math.log(c['temperature'])),c['temperature']))
        temperatures[f'RGB_ONLY_seed{seed}']=dict(**best,candidates=candidates,eligible_frames=frames)
        if not hasattr(run,'methods'):
            run.methods=G.load_module(env.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py','rgb_selection_subpix')
        for begin in range(0,len(selection_rows),16):
            rows=selection_rows[begin:begin+16];batch=batch_without_features(data,rows)
            tiles=prior_selection.evaluation_tiles(data,rows,batch,head.base.displacements)
            q=decode(head(batch,True,tiles),best['temperature'],1.,None).cpu().numpy()
            gain,offset,pad,hw=G.metadata(source,arrays,rows)
            for j,row in enumerate(rows):
                native=(q[j].astype(float)-offset[j])/gain[j]-pad[j]
                base=(np.asarray(arrays['points'][row],float)-offset[j])/gain[j]-pad[j]
                record=source['records'][int(arrays['record_index'][row])]
                image=cv2.imread(record['image'],cv2.IMREAD_COLOR);p=record['reflect_pad_px'];h,w=record['raw_shape_hw']
                image=image[p:p+h,p:p+w];support=np.array(arrays['point_valid'][row],bool)
                final,_=run.methods.correct(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),native,support,'SUBPIX')
                final=run.methods.cap_points(base,final,w,h,support)
                predictions.append(dict(arm='RGB_ONLY',seed=seed,row=int(row),qFinal=final))
        del data,head,ck;torch.cuda.empty_cache()
        print('RGB_ONLY_SYNTHETIC_PREDICTIONS',seed,best,flush=True)
    torch.cuda.synchronize();device_seconds=time.monotonic()-device_start
    T.write(doc/'CALIBRATION.json',dict(status='DONE_SYNTHETIC_ONLY',frames=1004,seeds=[1,2],temperatures=temperatures,
        objective='original fixed-index Gaussian candidate CE',tie='score,abs(log(T)),T',lambda_fixed=1.,
        actual_device_seconds=device_seconds,actual_p3_p4_reads=0,source_calibration_sha256=T.sha(old/'CALIBRATION.json')))
    cpu_start=time.monotonic();import pose
    counts={};original={}
    for name in ('solvePnP','solvePnPGeneric','solvePnPRefineLM'):
        if hasattr(cv2,name):
            original[name]=getattr(cv2,name);counts[name]=0
            def counted(*a,_name=name,_fn=original[name],**kw):counts[_name]+=1;return _fn(*a,**kw)
            setattr(cv2,name,counted)
    results=[]
    try:
        for p in predictions:
            row=p['row'];K,Rgt,t,X,xyz=G.native_reference(source,arrays,exact,row)
            actual=pose.infer(p['qFinal'],K,xyz,True)
            truth=dict(R=Rgt,t=t,xyz=xyz,body_R=Rgt,body_xyz=xyz,order=2)
            results.append(dict(arm=p['arm'],seed=p['seed'],row=row,pose=pose.metric((source['records'][row]['id'],actual,truth))))
    finally:
        for name,fn in original.items():setattr(cv2,name,fn)
    allrows=reused+results;T.write(private/'SYNTHETIC_SELECTION_ROWS.json',allrows)
    active=lock['selection']['active_parameter_elements'];scores={}
    for arm in ('M1','M3','RGB_ONLY'):
        rr=[r for r in allrows if r['arm']==arm];good=[r for r in rr if r['pose']['available']]
        success=sum(r['pose']['available'] and r['pose']['translation_cm']<5 and r['pose']['rotation_deg']<5 for r in rr)
        assert len(rr)==1564
        scores[arm]=dict(frames=782,seeds=[1,2],rows=len(rr),success_count=success,strict_success_rate=success/len(rr),
            T_median_cm=float(np.median([r['pose']['translation_cm'] for r in good])),pose_available=len(good),
            active_parameter_elements=active[arm],source='NEW_RGB_ONLY' if arm=='RGB_ONLY' else 'REUSED_EXACT_PRIOR_ROWS')
    selected=min(scores,key=lambda a:(-scores[a]['strict_success_rate'],scores[a]['T_median_cm'],scores[a]['active_parameter_elements'],a))
    T.write(doc/'SYNTHETIC_SELECTION.json',dict(status='SELECTED_BEFORE_REAL_ON_FIXED_C2_RENDERER_VAL',selected_arm=selected,arms=scores,
        selection_truth='exact renderer C2; unchanged signed-front C1 exclusion',previous_screen_selected_arm='M4',
        previous_screen_selection_sha256=T.sha(old/'SYNTHETIC_SELECTION.json'),real_selection=False,
        new_pose_F_calls=len(predictions),reused_pose_rows=len(reused),OpenCV_calls=counts,
        private_rows_sha256=T.sha(private/'SYNTHETIC_SELECTION_ROWS.json'),CPU_scoring_seconds=time.monotonic()-cpu_start,
        selection_lock_sha256=T.sha(doc/'PLAN_AND_CONTRACT.json'),calibration_sha256=T.sha(doc/'CALIBRATION.json')))
    print('NARROW_SYNTHETIC_SELECTED',selected,scores,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('source-root','private-dir','public-dir','prior-private-dir','prior-public-dir'):p.add_argument('--'+name,required=True)
    run(p.parse_args())
