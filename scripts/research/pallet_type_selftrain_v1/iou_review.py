"""Read-only threshold audit of unchanged training pseudo-label convex hulls."""
import json
import cv2
import numpy as np
from . import common as C
from .pseudo import top
from . import stability_review as S

RAW=C.RAW/'iou_review'
DOC=C.DOC/'iou_review'
OUT=C.OUT/'iou_review'
THRESHOLDS=[.9,.8,.7]


def hull(points):
    if points is None:return None
    q=np.asarray(points,np.float32)[:8]
    if q.shape!=(8,2) or not np.isfinite(q).all():return None
    h=cv2.convexHull(q)
    return h if len(h)>=3 and cv2.contourArea(h)>1e-6 else None


def iou(a,b):
    a,b=hull(a),hull(b)
    if a is None or b is None:return 0.
    area1,area2=cv2.contourArea(a),cv2.contourArea(b)
    intersection,_=cv2.intersectConvexConvex(a,b,handleNested=True)
    intersection=max(0.,min(float(intersection),area1,area2))
    return min(1.,max(0.,intersection/(area1+area2-intersection)))


def rejected(row,mode,threshold):
    return row[mode]<threshold


def run():
    sources=[C.bound(C.RAW/'PSEUDO_ACCEPTED.json'),C.bound(S.RAW/'INFERENCE.json'),
             C.bound(S.DOC/'VERIFICATION.json'),C.bound(__file__),C.bound(C.HERE/'test_iou_review.py')]
    for b in sources:C.verify(b)
    for b in C.read(S.DOC/'VERIFICATION.json')['artifacts']:C.verify(b)
    C.freeze(DOC/'PROTOCOL.json',dict(population='249 ordinary-plastic training pseudo-label candidates',sources=sources,
        geometry='Convex hull of first8 predicted corners, continuous original-image coordinates, no image clipping',
        correction='raw R0 hull vs saved corrected Replay hull',augmentation='saved corrected Replay hull vs each inverse-mapped transformed Replay hull; minimum of4 IoUs',
        either='minimum of correction and augmentation IoU; reject hypothetically if strictly below threshold',
        thresholds=THRESHOLDS,invalid_hull_iou=0,GT_used=False,new_inference=False,training=False,
        automatic_label_change=False,warning='High hull IoU cannot rule out bad interior corners or index swaps. Low IoU can represent useful correction, not necessarily an error.'))
    stability={r['id']:r for r in C.read(S.RAW/'INFERENCE.json')};records=[]
    parents=S.parents(); unchanged_boxes=0
    for p in parents:
        r=stability[p['id']];raw=top(p['raw']);ref=top(p['refined']);q=ref['keypoints_xy']
        assert r['image']==p['image']
        unchanged_boxes+=raw['box_xyxy']==ref['box_xyxy']
        variants=[dict(name=o['variant'],iou=iou(q,o['refined'])) for o in r['outputs'][1:]]
        worst=min(variants,key=lambda o:o['iou']);before=iou(raw['keypoints_xy'],q)
        moves=[float(np.linalg.norm(np.asarray(o['refined'])[:8]-np.asarray(q)[:8],axis=1).max()) for o in r['outputs'][1:] if o['refined'] is not None]
        records.append(dict(id=p['id'],image=p['image'],session=p['session'].split('/')[-1],
            correction=before,augmentation=worst['iou'],either=min(before,worst['iou']),
            worst_variant=worst['name'],variants=variants,
            correction_max_px=max(r['correction_px']),augmentation_max_px=max(moves) if moves else None))
    summaries={}
    for threshold in THRESHOLDS:
        modes={}
        for mode in ['correction','augmentation','either']:
            ids=[r['id'] for r in records if rejected(r,mode,threshold)]
            modes[mode]=dict(rejected=len(ids),kept=len(records)-len(ids),retention=(len(records)-len(ids))/len(records),rejected_ids=ids)
        modes['overlap_rejected']=len(set(modes['correction']['rejected_ids'])&set(modes['augmentation']['rejected_ids']))
        summaries[str(threshold)]=modes
    result=dict(images=len(records),unchanged_detector_boxes=unchanged_boxes,thresholds=summaries,
        no_GT_accuracy_claim=True,coordinates_changed=False,automatic_rejections_applied=0,records=records,
        high_iou_large_move_examples={mode:[r['id'] for r in records if r[mode]>=.9 and r[mode+'_max_px'] is not None and r[mode+'_max_px']>20] for mode in ['correction','augmentation']})
    C.freeze(RAW/'RESULTS.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','thresholds']},ensure_ascii=False))
    for t,m in summaries.items():print(t,{k:{a:b for a,b in v.items() if a!='rejected_ids'} if isinstance(v,dict) else v for k,v in m.items()})
    gallery(records,parents,stability)


def hull_panel(image,a,b,title):
    canvas=image.copy();overlay=canvas.copy()
    for q,color in [(a,(255,185,45)),(b,(30,225,255))]:
        h=hull(q)
        if h is not None:cv2.fillConvexPoly(overlay,np.rint(h).astype(np.int32),color)
    canvas=cv2.addWeighted(overlay,.20,canvas,.80,0)
    for q,color in [(a,(255,185,45)),(b,(30,225,255))]:
        h=hull(q)
        if h is not None:cv2.polylines(canvas,[np.rint(h).astype(np.int32)],True,color,2,cv2.LINE_AA)
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],28),(25,25,25),-1)
    cv2.putText(canvas,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
    return canvas


def gallery(records,parents,stability):
    parents={p['id']:p for p in parents};OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    cards=[]
    for rank,r in enumerate(sorted(records,key=lambda r:(r['either'],r['id'])),1):
        p=parents[r['id']];C.verify(p['image']);image=cv2.imread(str(C.ROOT/p['image']['path']));assert image is not None
        a=top(p['raw'])['keypoints_xy'];b=top(p['refined'])['keypoints_xy']
        v=next(o['refined'] for o in stability[r['id']]['outputs'] if o['variant']==r['worst_variant'])
        panels=[S.panel(image,a,(255,185,45),'R0 raw / NO GT'),S.panel(image,b,(30,225,255),'Replay corrected / unchanged label'),
                hull_panel(image,a,b,f'Raw(blue) vs Replay(yellow): IoU {r["correction"]:.3f}'),
                hull_panel(image,b,v,f'Label(blue) vs {r["worst_variant"]}(yellow): {r["augmentation"]:.3f}')]
        name=f'images/{rank:03d}.jpg';dest=OUT/name
        if not dest.exists():assert cv2.imwrite(str(dest),np.concatenate(panels,axis=1),[cv2.IMWRITE_JPEG_QUALITY,88])
        cards.append(dict(**r,rank=rank,asset=name))
    template=(C.HERE/'iou_review.html').read_text()
    C.write_text(OUT/'index.html',template.replace('__DATA__',json.dumps(cards,ensure_ascii=False).replace('</','<\\/')))
    print('GALLERY',OUT/'index.html')


if __name__=='__main__':run()
