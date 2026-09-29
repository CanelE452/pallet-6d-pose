"""K8 TRAIN-only augmented640 target following, not physical GT accuracy.

Replays the exact worker/transform prefix sealed before fitting. Highest-score
detection only; no target-nearest instance, extra padding, fit, or loss change.
Four frozen primary students are evaluated on their respective CLEAR/OCC input.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from . import common as C

ARMS=('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR','CLEAN_RAW_OCC','CLEAN_REF_OCC')
OUT=C.RAW/'train_augmented_following'
LOCK=OUT/'PREDICTIONS_LOCK.json'
REPORT=C.DOC/'TRAIN_AUGMENTED_FOLLOWING.json'


def prefix():
    import cv2
    import torch
    from . import preflight as P,trainer as T
    from .augmentation import load_paired_labels
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protocol=C.read(C.DOC/'PRIMARY_PROTOCOL.json')
    assert protocol['train_unique_images']==78 and protocol['locked_before_fit']
    preflight=C.read(C.DOC/'PREFLIGHT.json');assert preflight['passed'] and preflight['K_batches']==8
    paired=load_paired_labels(*[C.ROOT/protocol['datasets'][t]['train_list']['path'] for t in ('RAW','REF')])
    recordings={r['train_id']+'.png':r['recording'] for r in C.read(C.RAW/'CLEAN_LOCKED_PRIVATE.json')['rows']}
    cache={};bindings=[]
    for target in ('RAW','REF'):
        for condition in ('CLEAR','OCC'):
            path=C.RAW/f'PREFLIGHT_TRACE_S42_{target}_{condition}_PRIVATE.json'
            binding=next(b for b in preflight['bindings'] if b['path']==str(path.relative_to(C.ROOT)))
            C.verify(binding);expected=C.read(path);bindings.append(binding)
            P.seed(42)
            dataset=T.make_dataset(C.ROOT/protocol['datasets'][target]['train_list']['path'],protocol['args'],paired,target,condition,recordings)
            loader=T.make_loader(dataset,batch=16,workers=2,training_seed=42,pin_memory=False)
            values=[]
            try:
                for index,batch in enumerate(loader):
                    info=batch.pop('transfer_info')
                    actual=T.batch_trace(batch,info,epoch=0,index=index)
                    assert actual==expected[index], (target,condition,index,'Pre-fit K8 trace mismatch')
                    values.append(dict(images=batch['img'].clone(),keypoints=batch['keypoints'].clone(),
                        batch_idx=batch['batch_idx'].clone(),names=actual['names'],info=info))
                    if index==7:break
            finally:
                loader.iterator._shutdown_workers()
            assert len(values)==8
            cache[target,condition]=values
    return protocol,cache,bindings


def targets_for(cache):
    rows=[]
    for index in range(8):
        reference=cache['REF','CLEAR'][index]
        for position,info in enumerate(reference['info']):
            if info['role']!='REAL':continue
            raw=cache['RAW','CLEAR'][index];ref=reference
            ri=raw['batch_idx'].long()==position;fi=ref['batch_idx'].long()==position
            assert int(ri.sum())==int(fi.sum())==1
            rawq=raw['keypoints'][ri][0].numpy();refq=ref['keypoints'][fi][0].numpy()
            assert np.array_equal(rawq[:,2],refq[:,2])
            support=refq[:8,2]==2
            plan=info['plan']
            covered=set(plan['covered']) if plan and plan['applied'] else set()
            assert covered<=set(np.flatnonzero(support))
            rows.append(dict(id=f'b{index:02d}_i{position:02d}',batch=index,position=position,
                name=info['name'],recording=info['recording'],support=support.tolist(),
                raw=(rawq[:8,:2]*640).tolist(),ref=(refq[:8,:2]*640).tolist(),
                canonical_REF_planned_covered=sorted(covered),plan=plan,
                CLEAR_actual_covered=0,OCC_canonical_actual_covered=len(covered)))
    assert rows and any(r['canonical_REF_planned_covered'] for r in rows)
    return rows


def infer():
    if LOCK.exists():
        for binding in C.read(LOCK)['files']+C.read(LOCK)['sources']:
            C.verify(binding)
        print('TRAIN_AUGMENTED_K8_ALREADY_FROZEN',flush=True);return
    from . import eval_student as E
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    from ultralytics import YOLO
    import torch
    from scripts.research.pallet_selector_recovery_v1.common import state_hash
    assert torch.cuda.is_available(), 'Host CUDA required; no silent CPU neural inference'
    start=time.monotonic();protocol,cache,trace_bindings=prefix()
    _,fits=E.fits_for(42)
    rows=targets_for(cache)
    target_path=OUT/'TARGETS_PRIVATE.json';C.save(target_path,rows,True)
    source_bindings=trace_bindings+[C.bind(C.DOC/'PRIMARY_PROTOCOL.json'),C.bind(C.DOC/'PREFLIGHT.json'),
        C.bind(C.DOC/'PAIR_INTEGRITY_S42.json'),C.bind(Path(__file__)),
        C.bind(Path(__file__).with_name('preflight.py')),C.bind(Path(__file__).with_name('trainer.py')),
        C.bind(Path(__file__).with_name('augmentation.py'))]
    files=[C.bind(target_path)];kwargs=dict(conf=.001,imgsz=640,rect=True,augment=False,half=False,
        device='cuda',verbose=False,save=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    allpred={}
    for arm in ARMS:
        target,condition=arm.split('_')[1:]
        path=OUT/(arm+'.json');seal=OUT/(arm+'_LOCK.json')
        stamp=dict(checkpoint=fits[arm]['checkpoint'],implementation=C.bind(Path(__file__)),
            target=C.bind(target_path),trace=C.bind(C.RAW/f'PREFLIGHT_TRACE_S42_{target}_{condition}_PRIVATE.json'))
        if path.exists():
            C.verify(C.read(seal)['file']);stored=C.read(path)
            assert stored['stamp']==stamp
        else:
            thermal_guard();C.verify(fits[arm]['checkpoint'])
            model=YOLO(str(C.ROOT/fits[arm]['checkpoint']['path']),task='pose')
            model.predict(np.zeros((640,640,3),np.uint8),**kwargs)
            model.model.eval()
            for parameter in model.model.parameters():parameter.requires_grad_(False)
            before=state_hash(model.model);predictions={}
            for index,batch in enumerate(cache[target,condition]):
                rr=[r for r in rows if r['batch']==index]
                # Dataset tensor is RGB; numpy predictor expects BGR. No
                # resize/reflection beyond the existing exact640 input.
                images=[batch['images'][r['position']].permute(1,2,0).numpy()[:,:,::-1].copy() for r in rr]
                thermal_guard()
                with torch.inference_mode():results=model.predict(images,**kwargs)
                assert len(results)==len(rr)
                for row,result in zip(rr,results):
                    assert result.orig_shape==(640,640)
                    cc=[]
                    if result.boxes is not None:
                        for j in range(len(result.boxes)):
                            cc.append(dict(candidate_index=j,score=float(result.boxes.conf[j]),
                                box_xyxy=result.boxes.xyxy[j].cpu().tolist(),
                                keypoints_xy=result.keypoints.xy[j].cpu().tolist(),
                                keypoints_conf=result.keypoints.conf[j].cpu().tolist()))
                    predictions[row['id']]=dict(candidates=cc,selected_index=int(np.argmax([c['score'] for c in cc])) if cc else None)
            assert state_hash(model.model)==before
            assert all(p.grad is None and not p.requires_grad for p in model.model.parameters())
            stored=dict(stamp=stamp,predictions=predictions,state_unchanged=True,gradients_absent=True,
                optimizer_constructed=False,coordinate_space='augmented640x640pixels',
                inputs='Own CLEAR/OCC fixed pre-fit K8 prefix; REAL only; source not inferred')
            C.save(path,stored,True);C.save(seal,dict(file=C.bind(path)),True)
            del model;torch.cuda.empty_cache()
        allpred[arm]=stored['predictions'];files.extend([C.bind(path),C.bind(seal)])
        source_bindings.append(C.bind(C.DOC/f'FIT_{arm}_S42.json'))
        print('TRAIN_AUGMENTED_K8_ARM',arm,len(rows),flush=True)
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    for condition in ('CLEAR','OCC'):
        for row in rows:
            assert_detector_parity(allpred[f'CLEAN_RAW_{condition}'][row['id']],allpred[f'CLEAN_REF_{condition}'][row['id']])
    C.save(LOCK,dict(created_at=C.now(),files=files,sources=source_bindings,
        K_batches=8,real_occurrences=len(rows),real_unique=len({r['name'] for r in rows}),
        prefit_trace_reconstruction_exact=True,all_four_frozen=True,
        source_inference=False,teacher_or_evaluation_GT_read=False,
        same_condition_detector_parity=True,optimizer_updates=0,new_fits=0,
        seconds=time.monotonic()-start,inference=kwargs),True)


def summary(points):
    valid=[p for p in points if not p['missing']]
    result=dict(point_occurrences=len(points),observed=len(valid),missing=len(points)-len(valid),
        coverage=len(valid)/len(points) if points else None)
    for key in ('l1_mean_xy_px','l2_px'):
        values=[p[key] for p in valid]
        result[key]=dict(mean=float(np.mean(values)) if values else None,
            median=float(np.median(values)) if values else None,P90=float(np.quantile(values,.9)) if values else None)
    penalty=[640*np.sqrt(2) if p['missing'] else p['l2_px'] for p in points]
    result['L2_missing_diagonal_penalty_mean_px']=float(np.mean(penalty)) if penalty else None
    return result


def score():
    lock=C.read(LOCK)
    for binding in lock['files']+lock['sources']:C.verify(binding)
    if REPORT.exists():
        for binding in C.read(REPORT)['sources']+C.read(REPORT)['private_artifacts']:C.verify(binding)
        return
    rows=C.read(OUT/'TARGETS_PRIVATE.json');points=[]
    for arm in ARMS:
        stored=C.read(OUT/(arm+'.json'));predictions=stored['predictions']
        condition=arm.split('_')[2]
        for row in rows:
            pred=predictions[row['id']];index=pred['selected_index']
            q=np.asarray(pred['candidates'][index]['keypoints_xy']) if index is not None else np.full((9,2),np.nan)
            for target in ('raw','ref'):
                for corner,keep in enumerate(row['support']):
                    if not keep:continue
                    delta=q[corner]-np.asarray(row[target][corner]);missing=not np.isfinite(delta).all()
                    points.append(dict(arm=arm,target=target,id=row['id'],name=row['name'],recording=row['recording'],corner=corner,
                        split='CANONICAL_REF_PLANNED_COVERED' if corner in row['canonical_REF_planned_covered'] else 'CANONICAL_REF_UNMASKED',
                        condition=condition,missing=missing,l1_mean_xy_px=None if missing else float(np.abs(delta).mean()),
                        l2_px=None if missing else float(np.linalg.norm(delta))))
    groups={'ALL':None,**{r: r for r in sorted({row['recording'] for row in rows})}}
    output={}
    for group,recording in groups.items():
        output[group]={}
        for arm in ARMS:
            output[group][arm]={}
            for target in ('raw','ref'):
                pp=[p for p in points if p['arm']==arm and p['target']==target and (recording is None or p['recording']==recording)]
                output[group][arm][target]={'ALL_SUPERVISED':summary(pp),**{split:summary([p for p in pp if p['split']==split])
                    for split in ('CANONICAL_REF_PLANNED_COVERED','CANONICAL_REF_UNMASKED')}}
    private=OUT/'RESIDUALS_PRIVATE.json';C.save(private,points,True)
    C.save(REPORT,dict(created_at=C.now(),groups=output,real_occurrences=lock['real_occurrences'],real_unique=lock['real_unique'],
        K_batches=8,coordinate_space='augmented640x640pixels, not nativepx',
        target_semantics='TRAIN pseudo-coordinate following; not physicalGT accuracy or real-pose performance',
        grouping='CanonicalREF planned covered IDs are common across RAW/REF and CLEAR/OCC. CLEAR is planned-covered counterfactual, not actually hidden. RAW coordinate may fall outside the same rectangle.',
        points='Only common v2 corner0..7; true-ignore excluded; center and source excluded',
        source_loss_decomposition_available=False,selection='Highestconfidence only, no GT/target-nearest candidate',
        coverage_policy='Conditional observed residuals plus explicit missing count and L2 diagonal640sqrt2 penalty mean',
        prefit_trace_reconstruction_exact=True,new_fits=0,optimizer_updates=0,
        private_artifacts=[C.bind(private)],sources=[C.bind(LOCK),C.bind(Path(__file__))]),True)
    print('TRAIN_AUGMENTED_K8_SCORED',lock['real_occurrences'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('infer','score'))
    args=parser.parse_args();{'infer':infer,'score':score}[args.phase]()
