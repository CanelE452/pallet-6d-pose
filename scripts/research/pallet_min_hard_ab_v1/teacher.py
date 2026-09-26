"""Run the historical frozen TYPE_REPLAY_PIPELINE only after human label lock."""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as C
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S
from scripts.research import pallet_replay_clean19_v1 as R
from scripts.research import pallet_visible_refine_hidden_pnp_v1 as V


@torch.no_grad()
def main():
    if (C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json').exists():
        C.verify(C.read(C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json')['predictions']);return
    lock=C.read(C.DOC/'HARD_LABEL_LOCK.json');C.verify(lock['labels']);S.setup();gpu=S.guard()
    frames=C.read(C.ROOT/lock['labels']['path'])['frames']
    fit=C.read(C.ROOT/'_docs/experiments/pallet_replay_by_type_v1/plastic/FIT.json');C.verify(fit['checkpoint'])
    r0=YOLO(str(S.H.C.N.E.R0),task='pose');ref=R.C.load_model()
    ref.load_state_dict(torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)['model_state_dict']);ref.eval()
    pred={};n=0;finite=0
    for fid,f in frames.items():
        im=cv2.imread(str(C.ROOT/f['image']['path']));raw=S.predict(r0,im,100)
        corr=R.C.predict(ref,im,raw,None);sel=R.C.selected(raw)
        pts=None if sel is None else np.array(sel['keypoints_xy'],float)
        if pts is not None:pts[(pts==-1).all(1)]=np.nan
        d=f['physical_dimensions_m'];dims=np.array([d['x'],d['y'],d['z']]);K=np.array(f['camera_K'])
        initial=V.Pose.infer(pts,K,dims,False)
        final,_,info=V.pipeline(raw,corr,initial,K,im.shape[:2]);c=R.C.selected(final)
        support=[i for i,p in enumerate(f['corners']) if p['status']=='DIRECT_VISIBLE']
        ok=[]
        for i in support:
            valid=c is not None and np.isfinite(c['keypoints_xy'][i]).all() and c['keypoints_xy'][i]!=[-1,-1]
            if valid:ok.append(i)
        n+=len(support);finite+=len(ok)
        pred[fid]=dict(raw=raw,refined=corr,prediction=final,decision=info,finite_support=ok)
        print('TEACHER',fid,len(ok),'/',len(support),flush=True)
    path=C.RAW/'TEACHER_HARD_PREDICTIONS_PRIVATE.json';C.save(path,pred,immutable=True)
    coverage=finite/n
    C.save(C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json',dict(created_at=C.now(),label_lock=C.bind(C.DOC/'HARD_LABEL_LOCK.json'),
           predictions=C.bind(path),R0=C.bind(S.H.C.N.E.R0),refiner=fit['checkpoint'],pipeline=C.bind(V.__file__),
           support_points=n,finite_points=finite,coverage=coverage,status='READY' if coverage>=.9 else 'H_PSEUDO_ARM_INCOMPLETE',
           label_selection_changed=False,gpu=gpu),immutable=True)
    print('TEACHER_COVERAGE',finite,n,flush=True)


if __name__=='__main__':main()
