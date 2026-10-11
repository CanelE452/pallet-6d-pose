"""Fixed source calibration and prelocked exact-renderer C2 VAL selection."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch,cv2
import train as T
import precision_model as M
import geometry as G

def evaluation_tiles(data,rows,batch,displacements):
    """No-detection rows retain original model input/support and failed output.

    Source Generic already uses [0,0,1,1] for invalid boxes and makes every
    corner unsupported. Use that same box only to allocate unused RGB tiles;
    never replace the model's box, points, masks, or evaluation denominator.
    """
    boxes=batch['boxes']
    valid=torch.isfinite(boxes).all(-1)&(boxes[:,2:]>boxes[:,:2]).all(-1)
    tile_batch=dict(batch)
    tile_batch['boxes']=torch.where(valid[:,None],boxes,boxes.new_tensor([0,0,1,1]))
    return M.native_tiles(data,rows,tile_batch,displacements)

def preflight(args):
    source,arrays,side,exact=G.inputs(args.source_root)
    side={k:side[k] for k in side.files} if hasattr(side,'files') else side
    env,refiner,Data,Base,targets,decode,loss=M.source_modules(args.source_root)
    import pose
    rows=np.array([i for i,r in enumerate(source['records']) if r['partition']=='selection' and int(side['order'][i])==2])
    assert len(rows)==782
    checks=[]
    for row in rows[:16]:
        K,R,t,X,xyz=G.native_reference(source,arrays,exact,int(row))
        gain,offset,pad,hw=G.metadata(source,arrays,[row])
        q=np.array(arrays['gt_points'][row],float);q=(q-offset[0])/gain[0]-pad[0]
        actual=pose.infer(q,K,xyz,True)
        truth=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=2)
        metrics=pose.metric((str(row),actual,truth))
        checks.append(dict(row=int(row),available=metrics['available'],T_cm=metrics.get('translation_cm'),R_deg=metrics.get('rotation_deg')))
    passed=all(c['available'] and c['T_cm']<.02 and c['R_deg']<.02 for c in checks)
    T.write(Path(args.public_dir)/'SYNTHETIC_SELECTION_LOCK.json',dict(status='PASS' if passed else 'BLOCKED_PHYSICAL_POSE_CONTRACT',selection_partition_rows=1031,physical_rotation_eligible_C2_rows=len(rows),excluded_C1_rows=249,excluded_C4_rows=0,criterion='pooled seed1/2 finalSubPix strict T<5cm/R<5deg maximum; tie translationmedian thenlowercost',selection_source='existing selection partition exact renderer truth; never REAL',C1_exclusion='legacy signed-front180physicalmapping failed; C1 still participates in TRAIN/coordinate calibration and descriptive2D',row_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest(),exact_GT_production_F_roundtrip=checks,exact_GT_roundtrip_PASS=passed,solver='unchanged productionF/SQPnP+RefineLM, source_basis=True',new_hyperparameter_selection=False))
    assert passed,'C2 VALIDATION POSE ROUNDTRIP BLOCKED'
    print('SYNTHETIC_SELECTION_PREFLIGHT_PASS',len(rows),flush=True)

@torch.no_grad()
def run(args):
    doc=Path(args.public_dir);private=Path(args.private_dir)
    lock=T.read(doc/'SYNTHETIC_SELECTION_LOCK.json');assert lock['status']=='PASS'
    train=T.read(doc/'TRAIN_RECEIPTS.json');arms=train['executed_arms']
    source,arrays,side,exact=G.inputs(args.source_root);side={k:side[k] for k in side.files} if hasattr(side,'files') else side
    allcalibration=[];predictions=[];start=time.monotonic();T.configure()
    temperatures={};GPU=0.
    for seed in (1,2):
        env,refiner,data,template,order,path,targets,decode,loss=T.setup(args,seed)
        heads={}
        for arm in arms:
            h=__import__('copy').deepcopy(template);ck=torch.load(private/f'checkpoints/{arm}_seed{seed}.pt',map_location='cpu');h.load_state_dict(ck['state_dict']);heads[arm]=h.eval().requires_grad_(False)
        del template
        rows=np.flatnonzero(data.partitions=='calibration');assert len(rows)==1004
        totals={a:np.zeros(4) for a in arms};frames=0
        for begin in range(0,len(rows),16):
            rr=rows[begin:begin+16];batch=data.batch(rr,'N3_DIM_SYM');tiles=evaluation_tiles(data,rr,batch,heads[arms[0]].base.displacements)
            for arm,h in heads.items():
                output=h(batch,arm in ('M3','M4'),tiles)
                # Exact original P temperature objective, fixed-index GT only.
                target=targets(output,batch['gt_points'],batch['gt_valid']);mask=target['support'];count=mask.sum(-1);used=count>0
                for j,temp in enumerate((.5,1.,2.,4.)):
                    ce=-(target['distribution'].double()*(output['logits'].double()/temp).log_softmax(-1)).sum(-1)
                    totals[arm][j]+=float(((ce*mask).sum(-1)/count.clamp_min(1))[used].sum())
                if arm==arms[0]:frames+=int(used.sum())
        for arm in arms:
            candidates=[dict(temperature=temp,score=float(v/frames)) for temp,v in zip((.5,1.,2.,4.),totals[arm])]
            best=min(candidates,key=lambda c:(c['score'],abs(math.log(c['temperature'])),c['temperature']))
            temperatures[f'{arm}_seed{seed}']=dict(**best,candidates=candidates)
        rows=np.flatnonzero((data.partitions=='selection')&(side['order']==2));assert len(rows)==782
        for begin in range(0,len(rows),16):
            rr=rows[begin:begin+16];batch=data.batch(rr,'N3_DIM_SYM');tiles=evaluation_tiles(data,rr,batch,heads[arms[0]].base.displacements)
            gain,offset,pad,hw=G.metadata(source,arrays,rr)
            for arm,h in heads.items():
                output=h(batch,arm in ('M3','M4'),tiles);q=decode(output,temperatures[f'{arm}_seed{seed}']['temperature'],1.,None).cpu().numpy()
                for j,row in enumerate(rr):
                    native=(q[j].astype(float)-offset[j])/gain[j]-pad[j]
                    base=(np.asarray(arrays['points'][row],float)-offset[j])/gain[j]-pad[j]
                    record=source['records'][int(arrays['record_index'][row])]
                    image=cv2.imread(record['image'],cv2.IMREAD_COLOR);p=record['reflect_pad_px'];height,width=record['raw_shape_hw'];image=image[p:p+height,p:p+width]
                    # Same fixed serial refinement/cap, then common F on native pixels.
                    if not hasattr(run,'methods'):
                        import importlib.util
                        s=importlib.util.spec_from_file_location('nonsevere_selection_sp',env.ROOT/'scripts/research/pallet_training_free_compare_20261007_v1/methods.py');run.methods=importlib.util.module_from_spec(s);s.loader.exec_module(run.methods)
                    support=np.array(arrays['point_valid'][row],bool)
                    final,_=run.methods.correct(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),native,support,'SUBPIX');final=run.methods.cap_points(base,final,width,height,support)
                    predictions.append(dict(arm=arm,seed=seed,row=int(row),qFinal=final))
        del heads,data;torch.cuda.empty_cache();print('SYNTHETIC_CALIBRATION_AND_VAL_PREDICTIONS',seed,flush=True)
    torch.cuda.synchronize();GPU=time.monotonic()-start
    T.write(doc/'CALIBRATION.json',dict(status='DONE_SYNTHETIC_ONLY',frames=1004,seeds=[1,2],temperatures=temperatures,objective='original fixed-index Gaussian candidate CE; noGTphasecalibration',lambda_fixed=1.,cap_image_diagonal_fraction=.01,actual_device_seconds=GPU))
    # Predictions complete before exact renderer scoring; no model selection updates.
    import pose
    results=[];cv_counts={};original={}
    for n in ('solvePnP','solvePnPGeneric','solvePnPRefineLM'):
        if hasattr(cv2,n):
            original[n]=getattr(cv2,n);cv_counts[n]=0
            def counted(*a,_n=n,_fn=original[n],**k):cv_counts[_n]+=1;return _fn(*a,**k)
            setattr(cv2,n,counted)
    try:
        for p in predictions:
            row=p['row'];K,R,t,X,xyz=G.native_reference(source,arrays,exact,row)
            actual=pose.infer(p['qFinal'],K,xyz,True);truth=dict(R=R,t=t,xyz=xyz,body_R=R,body_xyz=xyz,order=2)
            metric=pose.metric((source['records'][row]['id'],actual,truth));results.append(dict(arm=p['arm'],seed=p['seed'],row=row,pose=metric))
    finally:
        for n,fn in original.items():setattr(cv2,n,fn)
    scores={}
    for arm in arms:
        records=[r for r in results if r['arm']==arm];good=[r for r in records if r['pose']['available']]
        success=sum(r['pose']['available'] and r['pose']['translation_cm']<5 and r['pose']['rotation_deg']<5 for r in records)
        extra=2768 if arm in ('M3','M4') else 0
        scores[arm]=dict(frames=782,seeds=[1,2],rows=len(records),success_count=success,strict_success_rate=success/len(records),T_median_cm=float(np.median([r['pose']['translation_cm'] for r in good])),additional_active_RGB_params=extra,pose_available=len(good))
    selected=min(arms,key=lambda a:(-scores[a]['strict_success_rate'],scores[a]['T_median_cm'],scores[a]['additional_active_RGB_params'],a))
    T.write(private/'SYNTHETIC_SELECTION_ROWS.json',results)
    T.write(doc/'SYNTHETIC_SELECTION.json',dict(status='SELECTED_BEFORE_REAL_ON_FIXED_C2_RENDERER_VAL',selected_arm=selected,arms=scores,selection_lock_sha256=T.sha(doc/'SYNTHETIC_SELECTION_LOCK.json'),calibration_sha256=T.sha(doc/'CALIBRATION.json'),selection_truth='exact renderer; C1 excluded from physical-rotation selection byprelock, all2D/trainingretained',real_selection=False,new_fits=0,pose_F_calls=len(predictions),OpenCV_calls=cv_counts,private_rows_sha256=T.sha(private/'SYNTHETIC_SELECTION_ROWS.json'),CPU_scoring_seconds=time.monotonic()-start-GPU))
    print('FIXED_SYNTHETIC_VAL_SELECTED',selected,scores,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','run']);p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True)
    args=p.parse_args();(preflight if args.stage=='preflight' else run)(args)
