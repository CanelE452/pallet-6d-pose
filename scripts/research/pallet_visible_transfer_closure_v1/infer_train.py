"""Reuse the historical native standalone inference recipe; highest-score only."""
import time
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scripts.self_training_yolo.v3 import true_ignore_trainer
from . import common as C

@torch.no_grad()
def predict(model,image):
    canvas=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101)
    torch.backends.cudnn.allow_tf32=True
    p=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
    out=[]
    if p.boxes is not None and len(p.boxes):
        for j in range(len(p.boxes)):
            out.append(dict(candidate_index=j,score=float(p.boxes.conf[j]),box_xyxy=(p.boxes.xyxy[j].cpu().numpy()-100).tolist(),
                keypoints_xy=(p.keypoints.xy[j].cpu().numpy()-100).tolist(),keypoints_conf=p.keypoints.conf[j].cpu().numpy().tolist()))
    return dict(candidates=out,selected_index=int(np.argmax([r['score'] for r in out])) if out else None)

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available(),'Host CUDA required; no silent fallback'
    C.save(C.DOC/'GPU_PREFLIGHT.json',dict(utc=C.now(),device=torch.cuda.get_device_name(0),status=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader'],text=True)))
    rows=C.read(C.RAW/'TRAIN_TARGETS_PRIVATE.json')
    for a in C.ARMS:
        dst=C.RAW/f'TRAIN_PREDICTIONS_{a}.json';ck=C.checkpoint(a);C.verify(ck)
        if dst.exists():assert C.read(dst)['checkpoint']==ck;continue
        progress=C.RAW/f'TRAIN_PROGRESS_{a}.json'
        done=C.read(progress)['predictions'] if progress.exists() else {}
        model=YOLO(str(C.ROOT/ck['path']),task='pose')
        for n,r in enumerate(rows):
            if r['id'] in done:continue
            if n%32==0:
                temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]);assert temp<80
                print('TRAIN_FROZEN',a,n,217,temp,flush=True)
                C.save(progress,dict(checkpoint=ck,predictions=done))
            im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            done[r['id']]=predict(model,im)
        C.save(dst,dict(checkpoint=ck,predictions=done,native_unaugmented=True,highest_confidence_only=True,GT_or_target_matching=False),True)
        del model;torch.cuda.empty_cache()
    # Preserve metadata on stripped old optimizer: never pretend last.pt is resumable.
    optimizer={}
    for a in C.ARMS[1:]:
        ck=torch.load(C.ROOT/C.checkpoint(a)['path'],map_location='cpu',weights_only=False)
        optimizer[a]=dict(epoch=ck.get('epoch'),optimizer_present=ck.get('optimizer') is not None,ema_present=ck.get('ema') is not None,updates=ck.get('updates'))
    C.save(C.DOC/'OLD_OPTIMIZER_STATE_AUDIT.json',optimizer,True)
    C.save(C.DOC/'TRAIN_PREDICTIONS_LOCK.json',dict(files=[C.bind(C.RAW/f'TRAIN_PREDICTIONS_{a}.json') for a in C.ARMS],scope='Native unaugmented TRAIN target-following, NOT physical accuracy; no evaluated reference loaded',utc=C.now()),True)
    print('TRAIN_INFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
