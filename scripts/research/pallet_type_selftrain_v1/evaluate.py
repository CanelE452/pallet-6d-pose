"""R0 versus independently self-trained students on complete fixed type populations."""
import argparse
import json
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as C
from .pseudo import top
from scripts.research.pallet_posefix_large_error_v1 import evaluate as O
from scripts.self_training_yolo.v3 import true_ignore_trainer  # checkpoint class import


def prepare():
    pe,pop=O.population_metadata()
    green=C.read(C.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json')
    rows=[]
    for item,meta in pop:
        rows.append(dict(id=item.frame_id,image=C.bound(C.ROOT/item.image),annotation=C.bound(C.ROOT/item.label),
                         kind=next(k for k,v in C.TYPES.items() if v==meta['object_type']),object_type=meta['object_type'],session=meta['session_id']))
    for r in green['records']:rows.append(dict(id=r['id'],image=r['image'],annotation=r['annotation'],kind='GREEN',object_type=C.TYPES['GREEN'],session=r['session']))
    assert len(rows)==469 and len({r['id'] for r in rows})==469
    protocol=dict(records=rows,arms=['R0','SYN_ONLY','PLASTIC','GREEN'],
        primary='Per kind, compare fixed R0 vs independently trained same-kind student on ALL evaluation images; no test-image confidence/LOO/stability filter',
        inference='R0 or student standalone detection/keypoints; no Replay or N2 inference module. Same reflect100,640,conf.001 top1 recipe.',
        labels='DEV319 existing reference; GREEN150 manually clicked known corners only, all-known GT box for IoU matching',
        metrics='Whole-object symmetry PCK5/10/20, full-denominator misses penalized at image diagonal; matched-only median/P90, detection/match coverage, good5->bad10 damage',
        tuning=False,GT_inference_input=False,independent_confirmation=False,
        wood='No wood-specific student: calibrated disjoint train pool unavailable. Cross-type results do not count as wood self-training.',
        sources=[C.bound(pe.POS),C.bound(C.ROOT/'_docs/paper/final_dimension_v1/green150_saved_labels_v1/DATASET_SNAPSHOT.json'),
                 C.bound(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json'),C.bound(O.__file__),C.bound(__file__)])
    C.freeze(C.DOC/'EVAL_PROTOCOL.json',protocol)
    print('EVAL_LOCKED',len(rows),flush=True)
    return protocol


@torch.no_grad()
def infer(arm):
    protocol=C.read(C.DOC/'EVAL_PROTOCOL.json')
    for b in protocol['sources']:C.verify(b)
    C.N.setup();torch.set_num_interop_threads(1);C.N.E.gpu();assert torch.cuda.is_available()
    checkpoint=C.bound(C.N.E.R0) if arm=='R0' else C.read(C.DOC/f'FIT_{arm}.json')['checkpoint'];C.verify(checkpoint)
    destination=C.RAW/f'EVAL_PREDICTIONS_{arm}.json'
    if destination.exists():
        assert C.read(destination)['checkpoint']==checkpoint;return
    model=YOLO(str(C.ROOT/checkpoint['path']),task='pose');rows=[];start=time.monotonic()
    for i,r in enumerate(protocol['records']):
        C.verify(r['image']);image=cv2.imread(str(C.ROOT/r['image']['path']));assert image is not None
        canvas=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_REFLECT_101)
        torch.backends.cudnn.allow_tf32=True
        pred=model.predict(canvas,conf=.001,imgsz=640,rect=True,augment=False,half=False,device='cuda',verbose=False,save=False,stream=False)[0]
        candidates=[]
        if pred.boxes is not None and len(pred.boxes):
            boxes=pred.boxes.xyxy.cpu().numpy();scores=pred.boxes.conf.cpu().numpy();points=pred.keypoints.xy.cpu().numpy();conf=pred.keypoints.conf.cpu().numpy()
            for j in range(len(scores)):
                candidates.append(dict(candidate_index=j,score=float(scores[j]),box_xyxy=(boxes[j]-100).astype(float).tolist(),
                    keypoints_xy=(points[j]-100).astype(float).tolist(),keypoints_conf=conf[j].astype(float).tolist()))
        output=dict(candidates=candidates,selected_index=int(np.argmax([p['score'] for p in candidates])) if candidates else None)
        rows.append(dict(id=r['id'],kind=r['kind'],prediction=output,raw_hw=list(image.shape[:2])))
        if (i+1)%100==0:print('TYPE_EVAL',arm,i+1,'/469',C.N.E.gpu(),flush=True)
    C.freeze(destination,dict(complete=True,arm=arm,checkpoint=checkpoint,protocol=C.bound(C.DOC/'EVAL_PROTOCOL.json'),records=rows,seconds=time.monotonic()-start))
    print('TYPE_EVAL_INFERRED',arm,flush=True)


def score():
    protocol=C.read(C.DOC/'EVAL_PROTOCOL.json')
    files={arm:C.bound(C.RAW/f'EVAL_PREDICTIONS_{arm}.json') for arm in protocol['arms']}
    C.freeze(C.DOC/'EVAL_OUTPUTS_LOCK.json',dict(artifacts=files,all_predictions_before_GT_scoring=True,protocol=C.bound(C.DOC/'EVAL_PROTOCOL.json')))
    pe,pop=O.population_metadata();targets={}
    for item,meta in pop:
        t=pe.E._legacy_forbidden_target(item)
        targets[item.frame_id]=(np.asarray(t.keypoints_xy),np.asarray(t.box_xyxy),np.asarray(t.keypoint_supervision_mask))
    from scripts.evaluation.green_saved_labels_v1 import annotation_arrays
    from scripts.research.pallet_dim_conditioned_p_v1 import eval_math
    groups={r['object_type']:r['permutations'] for r in C.read(C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    for r in protocol['records']:
        C.verify(r['annotation'])
        if r['kind']=='GREEN':
            doc=C.read(C.ROOT/r['annotation']['path']);gt,known=annotation_arrays(doc);_,valid=annotation_arrays(doc,True)
            h=doc['camera_data']['height'];w=doc['camera_data']['width']
            inside=known&(gt[:,0]>=0)&(gt[:,0]<w)&(gt[:,1]>=0)&(gt[:,1]<h)
            box=np.r_[gt[inside].min(0),gt[inside].max(0)];targets[r['id']]=(gt,box,valid)
    metadata={r['id']:r for r in protocol['records']};allmetrics={};summaries={}
    for arm,b in files.items():
        C.verify(b);rows=[]
        for r in C.read(C.ROOT/b['path'])['records']:
            meta=metadata[r['id']];gt,box,valid=targets[r['id']];candidate=top(r['prediction'])
            matched=candidate is not None and O.iou(candidate['box_xyxy'],box)>=.5
            q=np.full((9,2),np.nan) if candidate is None else candidate['keypoints_xy']
            result=eval_math.measure(q,gt,valid,groups[meta['object_type']],r['raw_hw'],matched,candidate is not None)
            rows.append(dict(id=r['id'],kind=r['kind'],session=meta['session'],**result))
        allmetrics[arm]=rows
        summaries[arm]={kind:eval_math.summary([r for r in rows if r['kind']==kind]) for kind in C.TYPES}
    # Reproduce original R0 scores on the overlapping archived222 examples.
    archived=C.read(C.N.RAW/'PER_FRAME_METRICS.json')
    ours={r['id']:r for r in allmetrics['R0']};parity=0
    for mode in ['DEV72','GREEN150_MANUAL']:
        for row in archived[mode]['R0']:
            assert row['matched']==ours[row['id']]['matched']
            np.testing.assert_allclose(row['errors'],ours[row['id']]['errors'],atol=.001,rtol=0)
            parity+=1
    contrasts={}
    for kind in ['PLASTIC','GREEN']:
        before=[r for r in allmetrics['R0'] if r['kind']==kind];after=[r for r in allmetrics[kind] if r['kind']==kind]
        contrasts[kind]=dict(damage=eval_math.damage(before,after),PCK20_delta_pp=100*(summaries[kind][kind]['PCK']['20']-summaries['R0'][kind]['PCK']['20']),
            PCK20_vs_SYN_ONLY_pp=100*(summaries[kind][kind]['PCK']['20']-summaries['SYN_ONLY'][kind]['PCK']['20']))
    C.freeze(C.RAW/'EVAL_METRICS.json',allmetrics)
    C.freeze(C.DOC/'RESULTS.json',dict(completed_kinds=['PLASTIC','GREEN'],wood=C.read(C.DOC/'POOL.json')['wood'],
        summaries=summaries,contrasts=contrasts,baseline_parity_images=parity,
        predictions_lock=C.bound(C.DOC/'EVAL_OUTPUTS_LOCK.json'),metrics=C.bound(C.RAW/'EVAL_METRICS.json'),
        fits={k:C.bound(C.DOC/f'FIT_{k}.json') for k in ['SYN_ONLY','PLASTIC','GREEN']},
        independent_confirmation=False,auto_promoted=False,one_seed_bounded_screen=True))
    print(json.dumps(dict(summaries=summaries,contrasts=contrasts),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','score','R0','SYN_ONLY','PLASTIC','GREEN']);a=p.parse_args()
    prepare() if a.stage=='prepare' else score() if a.stage=='score' else infer(a.stage)
