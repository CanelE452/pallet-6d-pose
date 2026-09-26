"""Frozen H_MANUAL inference; reuse hash-identical S1 synthetic outputs. No GT reads."""
import time
from . import common as C
READS=C.guard('inference')
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scripts.research.pallet_min_hard_ab_v1.train import HardModel  # checkpoint type only
from scripts.research.pallet_selector_recovery_v1 import features as F

def main(batch=16):
    C.U.setup();start=time.perf_counter();gpu=[C.U.gpu()]
    if (C.stage(2)/'SYNTH_FEATURES_LOCK.json').exists():
        for f in C.read(C.stage(2)/'SYNTH_FEATURES_LOCK.json')['files']:C.verify(f)
        print('SYNTH_ALREADY_FROZEN');return
    bindings=C.read(C.DOC/'INPUT_BINDINGS.json');b=bindings['checkpoints']['H_MANUAL'];C.verify(b)
    old=C.read(C.OLD/'stage2_synth_scorer/SYNTH_PREDICTION_LOCK.json')
    for key in ('features','predictions','feature_contract'):C.verify(old[key])
    assert old['feature_contract']==bindings['contract']
    z=dict(np.load(C.ROOT/old['features']['path']));rows=C.read(C.RAW/'SYNTH_INFERENCE_INPUTS.json')
    assert np.array_equal(z['ids'],[r['id'] for r in rows]) and z['S1_geo'].shape==(6144,2,94)
    model=YOLO(str(C.ROOT/b['path']),task='pose')
    kwargs=dict(conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False)
    model.predict(np.zeros((680,840,3),np.uint8),**kwargs);model.model.eval()
    for p in model.model.parameters():p.requires_grad_(False)
    initial=C.U.state_hash(model.model);pred={};geo=[];valid=[];current=[];retries=0
    torch.backends.cudnn.allow_tf32=True;torch.cuda.reset_peak_memory_stats()
    off=0
    while off<len(rows):
        if off%256==0:gpu.append(C.U.gpu());print('FROZEN_HMAN_SYNTH',off,len(rows),'batch',batch,flush=True)
        rr=rows[off:off+batch]
        for r in rr:C.verify(r['image'])
        ims=[cv2.imread(str(C.ROOT/r['image']['path'])) for r in rr]
        assert all(list(im.shape[:2])==r['hw'] for im,r in zip(ims,rr))
        try:
            with torch.inference_mode():results=model.predict(ims,**kwargs)
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
            batch=max(1,batch//2);retries+=1;torch.cuda.empty_cache();continue
        for r,result in zip(rr,results):
            cc=[]
            if result.boxes is not None:
                for j in range(len(result.boxes)):
                    cc.append(dict(candidate_index=j,score=float(result.boxes.conf[j]),box_xyxy=result.boxes.xyxy[j].cpu().tolist(),keypoints_xy=result.keypoints.xy[j].cpu().tolist(),keypoints_conf=result.keypoints.conf[j].cpu().tolist()))
            p=dict(candidates=cc,selected_index=int(np.argmax([c['score'] for c in cc])) if cc else None)
            f=F.extract(p,r['K'],r['dims'],r['hw']);pred[r['id']]=dict(prediction=p,geometry=f,hw=r['hw'])
            geo.append(f['features'] if f['valid'] else np.zeros((2,94)));valid.append(f['valid']);current.append(C.HYP.index(f['selection']) if f['selection'] in C.HYP else -1)
        off+=len(rr)
    assert C.U.state_hash(model.model)==initial and all(p.grad is None and not p.requires_grad for p in model.model.parameters());C.verify(b)
    path=C.RAW/'SYNTH_FEATURES.npz';path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists()
    np.savez_compressed(path,ids=z['ids'],S1_geo=z['S1_geo'],S1_valid=z['S1_valid'],S1_current=z['S1_current'],
                        H_MANUAL_geo=np.array(geo,np.float32),H_MANUAL_valid=np.array(valid),H_MANUAL_current=np.array(current))
    C.save(C.RAW/'HMAN_SYNTH_PREDICTIONS.json',pred)
    gpu.append(C.U.gpu())
    C.save(C.stage(2)/'SYNTH_PREDICTIONS_LOCK.json',dict(created_at=C.now(),GT_input=False,S1_reused=old['predictions'],H_MANUAL=C.bind(C.RAW/'HMAN_SYNTH_PREDICTIONS.json'),
        checkpoints=bindings['checkpoints'],frames_per_model=6144,keypoint_optimizer_steps={'S1':0,'H_MANUAL':0},HMAN_state_before=initial,HMAN_state_after=C.U.state_hash(model.model),
        weights_unchanged=True,gradients_absent=True,inference=kwargs,initial_batch=16,final_batch=batch,OOM_retries=retries,
        seconds=time.perf_counter()-start,gpu_samples=gpu,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,read_paths=sorted(set(READS))))
    C.save(C.stage(2)/'SYNTH_FEATURES_LOCK.json',dict(created_at=C.now(),files=[C.bind(path)],prediction_lock=C.bind(C.stage(2)/'SYNTH_PREDICTIONS_LOCK.json'),
        feature_contract=bindings['contract'],GT_input=False,shape=[6144,2,94],S1_exact_reuse=True,features_code=C.bind(C.ROOT/'scripts/research/pallet_selector_recovery_v1/features.py')))
    print('SYNTH_FROZEN',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
