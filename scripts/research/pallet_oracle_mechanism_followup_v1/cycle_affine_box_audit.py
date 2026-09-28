"""TRAIN-only check of the single pre-existing augmented residual outlier."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from . import cycle_affine as A
from . import common as C


def iou(a,b):
    a,b=np.asarray(a),np.asarray(b)
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(intersection/(np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-intersection+1e-12))


def main():
    destination=A.DOC/'TRAIN_OUTLIER_BOX_AUDIT.json'
    if destination.exists():print('C2_BOX_AUDIT_ALREADY_COMPLETE');return
    from ultralytics import YOLO
    signal=C.read(C.DOC/'SIGNAL_DIAGNOSTIC_PLASTIC.json')
    records=signal['results']['REF_LR5']['EXISTING_AFFINE']['target_following']
    selected=max(records,key=lambda r:float(np.mean(r['normalized_errors'])))
    A.save(A.RAW/'TRAIN_OUTLIER_BOX_AUDIT_SPEC.json',dict(created_at=C.now(),
        signal=C.bind(C.DOC/'SIGNAL_DIAGNOSTIC_PLASTIC.json'),TRAIN_only=True,selected_id=selected['image_id'],
        selection='Single highest mean bbox-normalized TRAIN residual from already completed signal diagnostic; no DEV information',
        intervention='Compare original augmentation and real-only affine-off, same existing REF checkpoint and RNG; no update'),True)
    start=time.monotonic();torch.set_num_threads(4);assert torch.cuda.is_available()
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    dataset=A.make_dataset('PLASTIC','REF');original=copy.deepcopy(dataset.transforms)
    indexes=[i for i,p in enumerate(dataset.im_files) if Path(p).stem==selected['image_id']]
    real=[i for i,p in enumerate(dataset.im_files) if not Path(p).name.startswith('syn__')]
    sampled=[real[int((j+.5)*len(real)/32)] for j in range(32)]
    index=next(i for i in indexes if i in sampled)
    checkpoint=C.checkpoint('PLASTIC','REF_LR5');C.verify(checkpoint)
    model=YOLO(str(C.ROOT/checkpoint['path']),task='pose');output={}
    for mode in ('EXISTING_AFFINE','REAL_AFFINE_OFF'):
        dataset.transforms=copy.deepcopy(original)
        if mode=='REAL_AFFINE_OFF':dataset.transforms,_=A.wrap_affine(dataset.transforms)
        A.seed(20260928+index);sample=dataset[index]
        image=sample['img'].permute(1,2,0).numpy()[:,:,::-1].copy();h,w=image.shape[:2]
        box=sample['bboxes'][0].numpy()*[w,h,w,h];target=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]
        result=model.predict(image,conf=.001,imgsz=640,rect=True,augment=False,half=False,device=0,verbose=False)[0]
        assert result.boxes is not None and len(result.boxes)
        idx=int(result.boxes.conf.argmax());boxes=result.boxes.xyxy.cpu().numpy()
        overlaps=[iou(b,target) for b in boxes]
        keypoints=sample['keypoints'][0].numpy();mask=keypoints[:8,2]==2
        pred=result.keypoints.xy[idx,:8].cpu().numpy();err=np.linalg.norm(pred[mask]-keypoints[:8,:2][mask]*[w,h],axis=1)
        output[mode]=dict(candidates=len(boxes),selected_index=idx,selected_box_IoU_with_TRAIN_pseudo_box=overlaps[idx],
                          max_candidate_box_IoU_with_TRAIN_pseudo_box=max(overlaps),selected_confidence=float(result.boxes.conf[idx]),
                          mean_normalized_target_residual=float(np.mean(err)/np.linalg.norm(box[2:])),
                          median_normalized_target_residual=float(np.median(err)/np.linalg.norm(box[2:])),
                          P90_normalized_target_residual=float(np.quantile(err,.9)/np.linalg.norm(box[2:])),supervised=int(mask.sum()))
    A.save(destination,dict(created_at=C.now(),results=output,TRAIN_only=True,optimizer_updates=0,fits=0,
                            checkpoint=checkpoint,selection_spec=C.bind(A.RAW/'TRAIN_OUTLIER_BOX_AUDIT_SPEC.json'),
                            interpretation='Pseudo box matching is a TRAIN diagnostic, not verified GT detection accuracy; no target-based candidate selection deployed',
                            seconds=time.monotonic()-start,GPU_seconds=time.monotonic()-start),True)
    print('TRAIN_OUTLIER_BOX_AUDIT',output,flush=True)


if __name__=='__main__':main()
