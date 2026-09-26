"""Freeze all raw predictions and GEO_LINEAR pose decisions before reference scoring."""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as C
from .train import HardModel
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S
from scripts.research.pallet_selector_recovery_v1 import common as F
from scripts.research.pallet_single_model_preserve_v1.infer import pose

ARMS=('BASE','H_PSEUDO','H_MANUAL')


def main():
    S.setup();S.guard()
    old=C.ROOT/'data/pallet/results/pallet_single_model_preserve_v1'
    metadata=C.read(old/'INFERENCE_INPUTS.json');bindings=C.read(old/'INFERENCE_BINDINGS.json')
    C.verify(bindings['scorer']);scorer=torch.load(C.ROOT/bindings['scorer']['path'],map_location='cpu',weights_only=False)
    lock=C.read(C.DOC/'HARD_LABEL_LOCK.json');C.verify(lock['labels'])
    # Read dimensions / K only here; no training coordinates are fed into prediction.
    frames=C.read(C.ROOT/lock['labels']['path'])['frames']
    metadata['train']=[dict(id=fid,image=f['image'],K=f['camera_K'],dims=[f['physical_dimensions_m'][k] for k in ('x','y','z')]) for fid,f in frames.items()]
    C.save(C.RAW/'INFERENCE_INPUTS.json',metadata,immutable=True)
    checkpoints={'BASE':bindings['checkpoint'],**{a:C.read(C.DOC/f'FIT_{a}.json')['checkpoint'] for a in ARMS[1:]}}
    rawpath=C.RAW/'RAW_PREDICTIONS.json'
    if not (C.DOC/'RAW_PREDICTIONS_LOCK.json').exists():
        oldpred=C.read(F.PREV_RAW/'PREDICTIONS.json')['S1'];pred={pop:{a:{} for a in ARMS} for pop in metadata}
        for arm,b in checkpoints.items():
            C.verify(b);model=YOLO(str(C.ROOT/b['path']),task='pose')
            for pop,rows in metadata.items():
                for i,r in enumerate(rows):
                    C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']))
                    p=S.predict(model,im,0 if pop=='source' else 100);pred[pop][arm][r['id']]=p
                    if arm=='BASE' and pop=='real':
                        a,b=F.selected(p),F.selected(oldpred[r['id']]);assert (a is None)==(b is None)
                        if a:np.testing.assert_allclose(a['keypoints_xy'],b['keypoints_xy'],atol=1e-4,rtol=1e-6)
                    if i%64==0:print('HARD_INFER',arm,pop,i,len(rows),flush=True)
            del model;torch.cuda.empty_cache()
        C.save(rawpath,pred,immutable=True)
        C.save(C.DOC/'RAW_PREDICTIONS_LOCK.json',dict(created_at=C.now(),predictions=C.bind(rawpath),checkpoints=checkpoints,
               metadata=C.bind(C.RAW/'INFERENCE_INPUTS.json'),GT_input=False,base_reproduction=True),immutable=True)
    else:
        C.verify(C.read(C.DOC/'RAW_PREDICTIONS_LOCK.json')['predictions']);pred=C.read(rawpath)
    if not (C.DOC/'POSE_DECISIONS_LOCK.json').exists():
        poses={pop:{a:{} for a in ARMS} for pop in metadata}
        for pop,rr in metadata.items():
            for arm in ARMS:
                for i,r in enumerate(rr):
                    im=cv2.imread(str(C.ROOT/r['image']['path']))
                    poses[pop][arm][r['id']]=pose(pred[pop][arm][r['id']],dict(r,hw=im.shape[:2]),scorer)
                    if pop=='source':
                        result=poses[pop][arm][r['id']]
                        for p in [result['current'],result['D9']]+[h['pose'] for h in result['hypotheses']]:
                            # Avoid modifying aliased current/hypothesis objects twice.
                            if p.get('available') and not p.get('source_axis_converted'):
                                p['R_physical']=(np.array(p['R_physical'])@np.diag([1.,-1.,-1.])).tolist()
                                p['source_axis_converted']=True
                    if i%64==0:print('HARD_POSE',arm,pop,i,len(rr),flush=True)
        C.save(C.RAW/'POSE_DECISIONS.json',F.clean(poses),immutable=True)
        C.save(C.DOC/'POSE_DECISIONS_LOCK.json',dict(created_at=C.now(),poses=C.bind(C.RAW/'POSE_DECISIONS.json'),
               raw_lock=C.bind(C.DOC/'RAW_PREDICTIONS_LOCK.json'),scorer=bindings['scorer'],GT_input=False,oracle_used=False),immutable=True)
    C.set_state('PREDICTIONS_FROZEN_SCORING_PENDING',training='COMPLETE')


if __name__=='__main__':main()
