"""Separate GT-blind process: exactly eight real-image views per pool frame."""
import os
import sys
import time
from pathlib import Path
from contracts import *
from perturb import views
from pose_adapter import pose
from risk import calculate


def install_gt_guard():
    """Deny all annotation, mixed-GT and prior error-result reads at Python I/O."""
    split=read(DOC/'SPLIT_BINDING.json')
    denied={str((ROOT/r['label_path']).resolve()) for r in split['pool']+split['evaluation']}
    denied.update(str((POSE/n).resolve()) for n in ('AXIS_REVIEW_MANIFEST.json','GEOMETRY_RESOLVED_POSE_GT.json'))
    denied.update(str(p.resolve()) for p in [RAW/'POOL_GT_ERRORS.json',RAW/'POOL_GT_REFERENCE.json',
        DOC/'OLD_SELECTION_HARDNESS_AUDIT.json',DOC/'OLD_SELECTION_UNIQUE_SET_ANALYSIS.json'])
    events=[]
    def guard(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        path=str(Path(os.fsdecode(args[0])).resolve())
        if path in denied or '/annotations/' in path and path.endswith('.json'):
            events.append(path)
            raise PermissionError('GT_BLIND_PROCESS_READ_DENIED: '+path)
    sys.addaudithook(guard)
    # Test denial without consuming any byte of target data.
    for p in sorted(denied)[:1]:
        try:
            open(p,'rb')
        except PermissionError:
            pass
        else:
            raise AssertionError('GT guard did not deny read')
    tested=len(events)
    return events,tested


def main():
    verify_lock()
    assert (RAW/'POOL_GT_ERRORS.json').exists(), 'Phase1 must be completed first'
    assert not (RAW/'TASK_VIEWS.json').exists(), 'Do not repeat inference'
    resources=[gpu()]
    denied,tested=install_gt_guard()
    import torch
    import cv2
    from ultralytics import YOLO
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'Use known working host CUDA userspace; no CPU fallback'
    inputs=read(RAW/'TASK_INPUTS.json');baseline=read(BASELINE)['frames']
    signals=read(OLD_RAW/'pool/ACQUISITION_SIGNALS.json')
    model=YOLO(str(R0),task='pose'); captured={};counters=dict(real_image_forwards=0,head_calls=0)
    def on_start(predictor):
        if captured.get('attached'):return
        head=predictor.model.model.model[-1]
        assert head.end2end and head.nc==1 and head.nk==27
        def post(m,args,out):
            dense=m._inference(out[1]['one2one'])
            assert torch.equal(m.postprocess(dense.permute(0,2,1)),out[0])
            captured['index']=int(dense[0,4].argmax())
            captured['anchors']=m.anchors.detach().cpu().numpy().copy()
            captured['strides']=m.strides.detach().cpu().numpy().copy()
            counters['head_calls']+=1
        head.register_forward_hook(post)
        captured['attached']=True
    model.add_callback('on_predict_start',on_start)
    records=[];max_xy=max_score=0.;start=time.monotonic()
    with torch.inference_mode():
        for i,item in enumerate(inputs):
            image=cv2.imread(str(ROOT/item['image_path']));assert image is not None
            per_view=[];anchor=None
            for j,transformed in enumerate(views(image)):
                padded=cv2.copyMakeBorder(transformed,100,100,100,100,cv2.BORDER_REFLECT_101)
                before=counters['head_calls']
                result=model.predict(padded,conf=.001,imgsz=640,device='0',verbose=False,
                                     augment=False,half=False)[0]
                counters['real_image_forwards']+=1
                assert counters['head_calls']==before+1, 'Exactly one real-image head call per view'
                if anchor is None:anchor=(captured['anchors'].copy(),captured['strides'].copy())
                assert np.array_equal(anchor[0],captured['anchors']) and np.array_equal(anchor[1],captured['strides'])
                detected=result.boxes is not None and len(result.boxes)>0
                record=dict(view=j,detection=bool(detected),candidate_index=captured['index'],
                            box_xyxy=None,score=None,keypoints_xy=None)
                if detected:
                    index=int(result.boxes.conf.argmax())
                    record.update(score=float(result.boxes.conf[index]),
                        box_xyxy=(result.boxes.xyxy[index].cpu().numpy()-100).tolist(),
                        keypoints_xy=(result.keypoints.xy[index].cpu().numpy()-100).tolist())
                record.update(pose(record['keypoints_xy'],item['camera']['K'],item['dimensions']))
                if j==0:
                    old=baseline[str((ROOT/item['image_path']).resolve().relative_to(ROOT))]
                    assert bool(old)==detected and captured['index']==signals[i]['top_index']
                    if old:
                        old=max(old,key=lambda p:p['score'])
                        xy=max(float(np.max(np.abs(np.array(record[k])-np.array(old[k])))) for k in ('box_xyxy','keypoints_xy'))
                        score=abs(record['score']-old['score'])
                        max_xy=max(max_xy,xy);max_score=max(max_score,score)
                        assert xy<=1e-4 and score<=1e-6, ('P0 parity',item['frame_id'],xy,score)
                per_view.append(record)
            records.append(dict(frame_id=item['frame_id'],views=per_view))
            if (i+1)%20==0 or i+1==len(inputs):
                resources.append(gpu())
                print(f'GT-free inference {i+1}/174, forwards={counters["real_image_forwards"]}, {resources[-1]["gpu"]}',flush=True)
    assert counters['real_image_forwards']==1392 and len(denied)==tested
    write(RAW/'TASK_VIEWS.json',records)
    risk=calculate(records);write(RAW/'TASK_RISK.json',risk)
    write(DOC/'RISK_FREEZE.json',dict(status='FROZEN_BEFORE_RISK_ERROR_JOIN',
        views_sha256=sha(RAW/'TASK_VIEWS.json'),risk_sha256=sha(RAW/'TASK_RISK.json'),
        formula_sha256=sha(DOC/'TASK_RISK_DEFINITION.json'),GT_accesses=0))
    write(DOC/'EXTRACTION_AUDIT.json',dict(status='PASS',frames=174,views=8,**counters,
        extra_real_image_warmup=0,framework_dummy_warmup='Ultralytics pre-on_predict_start; hooks exclude synthetic tensor warmup',
        P0_max_coordinate_delta_px=max_xy,P0_max_score_delta=max_score,P0_grid_index_exact=True,
        optimizer_updates=0,GT_read_guard_selftests=tested,GT_coordinate_reads=0,
        GT_guard_scope='Python file-open auditing plus source-reviewed native cv2.imread calls using locked image paths only',
        resources=resources,seconds=time.monotonic()-start,R0_sha_unchanged=sha(R0)==R0_SHA))
    print('RISK_FROZEN',flush=True)


if __name__=='__main__':main()
