"""E5-B: fixed evenly spaced TRAIN29, new mask family, native OO only."""
import time
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from . import run as C
from .inference import mask_plan,masked,detector,load_refiner,count_save
from scripts.research.pallet_posefix_large_error_v1 import core as CORE
from scripts.research.pallet_posefix_target_data_diagnosis_v1 import common as T

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    start,cpu=time.monotonic(),time.process_time();count=0
    records=sorted(enumerate(T.real_records()),key=lambda x:x[1]['id'])
    selected=[records[j] for j in np.rint(np.linspace(0,len(records)-1,29)).astype(int)]
    assert len({r['id'] for _,r in selected})==29
    rng=np.random.default_rng(42);inputs={};plans={}
    for ix,r in selected:
        C.verify(r['image']);x=T.pair(ix);p=x['metadata']['raw_prediction'];inp=x['pair']['CLEAN'];im=cv2.imread(str(C.ROOT/r['image']['path']))
        q=np.array(CORE.selected(p)['keypoints_xy']);box=np.array(CORE.selected(p)['box_xyxy'])
        plans[r['id']]=mask_plan(q,box,im.shape[:2],int(rng.integers(0,2**31-1)))
        inputs[r['id']]=dict(record=r,original_index=ix,clean_prediction=p,stored_pseudo_target=inp['original_gt'],
            target_valid=inp['original_gt_valid'],training_supervision_mask=inp['target_valid'])
    C.save(C.RAW/'E5B_INPUTS.json',inputs);C.save(C.RAW/'E5B_MASK_PLANS.json',plans)
    C.save(C.DOC/'E5B_PROTOCOL.json',dict(locked_before_inference=True,reason='E3 fixed-coordinate CO/OO difference leaves unseen-scene transfer versus same-scene RGB-mask limitation unresolved',
        selection='29 rounded evenly spaced ranks of253 sorted TRAIN IDs; no metric-based selection',seed=42,nativeOO_only=True,max_image_forwards=174,
        inputs=C.bind(C.RAW/'E5B_INPUTS.json'),masks=C.bind(C.RAW/'E5B_MASK_PLANS.json'),code=C.bind(__file__),
        targets='Actual frozen training pseudo target, not physical GT; clean R0 restoration reported separately. No new teacher forward.',
        limitations='TRAIN one recording; E3 clean29 three recordings; mask absolute area/error/crop differ. A/B/C/D gaps are not pure causal effects.'))
    b=C.read(C.ROOT/'_docs/experiments/pallet_existing_data_transfer_v1/SPLIT_LOCK.json')['R0_provenance']['baseline'];C.verify(b)
    model=YOLO(str(C.ROOT/b['path']),task='pose');preds={'identity':{},'PRIOR1':{},'FULL125':{}}
    images={}
    for i,x in inputs.items():
        im=cv2.imread(str(C.ROOT/x['record']['image']['path']));images[i]={}
        if plans[i]['status']!='READY':continue
        for kind in ('cover','avoid'):
            occ=masked(im,plans[i],kind);images[i][kind]=occ
            preds['identity'].setdefault(i,{})[kind]=detector(model,occ);count+=1
    del model
    for arm in ('PRIOR1','FULL125'):
        model=load_refiner(arm);before=C.L.state_hash(model.state_dict())
        for j,(i,variants) in enumerate(images.items()):
            preds[arm][i]={}
            for kind,im in variants.items():
                p=preds['identity'][i][kind];preds[arm][i][kind]=CORE.predict(model,im,p)
                count+=int(CORE.prepare_input(im,p) is not None)
            if (j+1)%7==0:print('E5B',arm,j+1,'total forwards',count,flush=True)
        assert C.L.state_hash(model.state_dict())==before
        del model
    C.save(C.RAW/'E5B_PREDICTIONS.json',preds)
    from scripts.research.pallet_posefix_large_error_v1.evaluate import iou
    errors={};support={}
    for i,x in inputs.items():
        target=np.array(x['stored_pseudo_target']);qc=np.array(CORE.selected(x['clean_prediction'])['keypoints_xy'])
        v=np.array(x['training_supervision_mask'],bool);v[8]=False
        errors[i]={};support[i]={}
        for kind,p in preds['identity'].get(i,{}).items():
            c=CORE.selected(p);same=c is not None and iou(c['box_xyxy'],CORE.selected(x['clean_prediction'])['box_xyxy'])>=.5
            support[i][kind]=dict(same_object_pair=same,supervised_corners=int(v.sum()),detected=c is not None)
            errors[i][kind]={}
            for arm in preds:
                z=CORE.selected(preds[arm][i][kind]);q=np.array(z['keypoints_xy']) if z else np.full((9,2),np.nan)
                good=np.isfinite(q).all(1)&~(q==-1).all(1)&v&same
                errors[i][kind][arm]=dict(pseudo_target_error_px=np.where(good,np.linalg.norm(q-target,axis=1),np.inf)[v],
                    clean_R0_restoration_px=np.where(good,np.linalg.norm(q-qc,axis=1),np.inf)[v],same_object_pair=same,
                    supervised_corners=int(v.sum()),valid_corners=int(good.sum()))
    # Serialize unavailable errors as explicit null with valid counts; summarize
    # extended real arrays before conversion to avoid silently dropping failure.
    summary={}
    for kind in ('cover','avoid'):
        summary[kind]={}
        for arm in preds:
            vals=[errors[i][kind][arm] for i in errors if kind in errors[i]]
            s={k:C.M.distribution([v for row in vals for v in row[k]]) for k in ('pseudo_target_error_px','clean_R0_restoration_px')}
            raw=np.array([v for i in errors if kind in errors[i] for v in errors[i][kind]['identity']['pseudo_target_error_px']])
            after=np.array([v for row in vals for v in row['pseudo_target_error_px']]);hard=np.isfinite(raw)&(raw>20)
            s.update(frames=len(vals),total_frames=29,matched=sum(r['same_object_pair'] for r in vals),corners=len(after),
                hard20=int(hard.sum()),recovered20to10=int((hard&(after<=10)).sum()),PCK10=float(np.mean(after<=10)))
            summary[kind][arm]=s
    C.save(C.RAW/'E5B_ERRORS.json',errors);C.save(C.RAW/'E5B_SUPPORT.json',support);C.save(C.DOC/'E5B_SUMMARY.json',summary)
    assert count<=174
    count_save('E5B',start,cpu,count,failed_placements=sum(p['status']!='READY' for p in plans.values()))

if __name__=='__main__':main()
