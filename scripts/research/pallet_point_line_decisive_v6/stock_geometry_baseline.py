"""Strong stock baseline backstop: unordered corner-set error (diagnostic).

Camera-dynamic YOLO IDs and fixed object-frame pose IDs cannot be directly
subtracted. An optimal one-to-one spatial assignment is used ONLY in scoring,
for BOTH methods. It is not a deployable role correction or a main metric.
"""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

KIT=Path(__file__).resolve().parent
ROOT=KIT.parents[2]
sys.path.insert(0,str(KIT))
from cli_train import Split,sha

R0=ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
EXPECTED='970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'


def set_error(points,gt):
    cost=np.linalg.norm(points[:,None]-gt[None],axis=-1)
    u,v=linear_sum_assignment(cost)
    return cost[u,v],v


def main():
    import cv2
    from ultralytics import YOLO
    run=ROOT/'data/pallet/results/pallet_point_line_decisive_v6'
    out=run/'STOCK_GEOMETRY_BASELINE.json'
    if out.exists():raise SystemExit('baseline already exists')
    assert sha(R0)==EXPECTED
    torch.set_num_threads(2);cv2.setNumThreads(1)
    model=YOLO(str(R0),task='pose');model.model.float().eval()
    data=Split(ROOT/'data/pallet/results/pallet_point_line_v5_local/export/synth_val.json')
    observations=[]
    # Prediction phase never loads a supervision tensor.
    for i,rec in enumerate(data.records):
        O,_,_=data.batch([i],'cpu',targets=False)
        image=cv2.imread(str(ROOT/rec['original_image']))
        result=model.predict(image,imgsz=640,conf=.001,iou=.7,max_det=300,verbose=False,half=False)[0]
        boxes=result.boxes.xyxy.cpu().numpy()-100.
        target=O['box'][0].numpy()
        if not len(boxes):raise RuntimeError('cached detection did not reproduce')
        distances=np.max(np.abs(boxes-target),axis=-1);best=int(distances.argmin())
        if distances[best]>.1:raise RuntimeError(f'cached detection changed {rec["id"]}: {distances[best]}')
        points=result.keypoints.xy[best].cpu().numpy()[:8]-100.
        observations.append(dict(id=rec['id'],points=points.tolist(),box_max_abs_delta=float(distances[best])))
        if (i+1)%100==0:print('stock baseline',i+1,flush=True)
    for i,row in enumerate(observations):
        _,S,_=data.batch([i],'cpu',targets=True)
        O,_,_=data.batch([i],'cpu',targets=False)
        err,assignment=set_error(np.asarray(row['points']),S['points'][0,:8].numpy())
        row.update(raw_mean_px=float(err.mean()),normalized=float(np.minimum(err/float(torch.linalg.vector_norm(O['image_hw'])),1.).mean()),
                   scoring_assignment=assignment.tolist())
    px=np.array([r['raw_mean_px'] for r in observations])
    out.write_text(json.dumps(dict(complete=True,checkpoint_sha256=EXPECTED,frames=len(observations),rows=observations,
        raw_frame_mean=float(px.mean()),raw_frame_median=float(np.median(px)),raw_frame_p90=float(np.quantile(px,.9)),
        conditional_normalized_mean=float(np.mean([r['normalized'] for r in observations])),
        scope='Unordered 8-corner assignment diagnostic on 505 cached matched instances. '
              'GT-free point inference; GT is used only for a scoring assignment. This is not the '
              'same-ID pose metric and cannot be substituted for the main architecture comparison.',
        inference_calls=len(observations),supervision_opened_during_inference=0),indent=2)+'\n')
    print(json.dumps({k:v for k,v in json.loads(out.read_text()).items() if k!='rows'},indent=2))


if __name__=='__main__':main()
