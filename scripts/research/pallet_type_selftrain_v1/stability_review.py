"""GT-free review of the 249 plastic training pseudo-labels, not evaluation GT.

No label coordinates, training data, student weights or masks are changed.
"""
import argparse
import copy
import json
import time
import cv2
import numpy as np
import torch
from . import common as C
from .pseudo import top
from scripts.research.pallet_posefix_utility_selector_v1 import augmentation_stability as A

DOC=C.DOC/'stability_review'
RAW=C.RAW/'stability_review'
OUT=C.OUT/'stability_review'
VARIANTS=['identity','brightness085','brightness115','resize090','resize110']


def parents():
    rows=[r for r in C.read(C.RAW/'PSEUDO_ACCEPTED.json') if r['kind']=='PLASTIC']
    assert len(rows)==249 and len({r['id'] for r in rows})==249
    return rows


def prepare():
    fit=C.read(C.N.DOC/'FIT.json')
    protocol=dict(population='249 ordinary-plastic training pseudo-label survivors, not evaluation images',
        variants=VARIANTS,scope='Frozen R0 -> frozen Replay on each variant; inverse-map coordinates; native corner IDs',
        score='max over 8 corners of RMS displacement from original corrected label across 4 variants / original box diagonal',
        inspection='Sort highest instability first, missing first, ID tie-break. Top50=floor80percent complement for review priority ONLY, not rejection or quality certification.',
        selection='Human keep/reject/unsure and per-corner exclusion proposals exported separately; never edits frozen coordinates or original labels',
        warning='Stable errors can pass. Symmetry/index changes can appear unstable. Not a correctness probability.',
        threshold_tuning=False,GT_used=False,training=False,automatic_rejection=False,
        precision='cuDNN TF32 True for R0 and False for Replay; matmul False',parity_tolerance_px=.001,
        sources=[C.bound(C.RAW/'PSEUDO_ACCEPTED.json'),C.bound(C.DOC/'PSEUDO_PROTOCOL.json'),
                 C.bound(C.DOC/'DATA_VERIFICATION.json'),C.bound(C.N.E.R0),fit['checkpoint'],
                 C.bound(A.__file__),C.bound(C.N.__file__),C.bound(C.N.C.__file__),C.bound(__file__)])
    for b in protocol['sources']:C.verify(b)
    C.freeze(DOC/'PROTOCOL.json',protocol)
    return protocol


def restored(pred,affine):
    p=top(pred)
    if p is None:return None
    q=np.asarray(p['keypoints_xy'],float)
    if not np.isfinite(q[:8]).all() or (q[:8]==-1).all(1).any():return None
    return A.map_points(q,np.linalg.inv(affine)).tolist()


def order(rows):
    return sorted(rows,key=lambda r:(r['score'] is not None,-r['score'] if r['score'] is not None else 0,r['id']))


@torch.no_grad()
def run():
    prepare();C.N.setup();torch.set_num_interop_threads(1)
    assert torch.cuda.is_available();print('GPU',C.N.E.gpu(),flush=True)
    extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0);model=C.N.load_model()
    start=time.monotonic();rows=[]
    try:
        for i,parent in enumerate(parents()):
            path=RAW/'frames'/f'{parent["id"]}.json'
            if path.exists():
                row=C.read(path);assert row['protocol_sha256']==C.sha(DOC/'PROTOCOL.json')
            else:
                C.verify(parent['image']);im=cv2.imread(str(C.ROOT/parent['image']['path']));assert im is not None
                outputs=[];parity={}
                for variant in VARIANTS:
                    aug,affine=A.transform(im,variant)
                    torch.backends.cudnn.allow_tf32=True
                    capture=extractor.predict(aug)
                    raw=dict(candidates=A.serial(capture['candidates']),selected_index=capture['selected_index'])
                    torch.backends.cudnn.allow_tf32=False
                    refined=C.N.C.predict(model,aug,raw)
                    if variant=='identity':
                        for name,pred in [('raw',raw),('refined',refined)]:
                            p,old=top(pred),top(parent[name]);assert p is not None and old is not None
                            delta=float(np.abs(np.asarray(p['keypoints_xy'])-old['keypoints_xy']).max())
                            assert delta<=.001,(parent['id'],name,delta)
                            assert abs(p['score']-old['score'])<=1e-6
                            parity[name]=delta
                    outputs.append(dict(variant=variant,affine=affine.tolist(),raw=restored(raw,affine),refined=restored(refined,affine)))
                b=np.asarray(top(parent['raw'])['box_xyxy']);diag=float(np.linalg.norm(b[2:]-b[:2]));assert diag>0
                values=[o['refined'] for o in outputs[1:]]
                measure=A.instability(top(parent['refined'])['keypoints_xy'],values,diag) if all(v is not None for v in values) else dict(score=None,max_corner_rms_px=None,per_corner_rms_px=None)
                raw_values=[o['raw'] for o in outputs[1:]]
                raw_measure=A.instability(top(parent['raw'])['keypoints_xy'],raw_values,diag) if all(v is not None for v in raw_values) else None
                move=np.linalg.norm(np.asarray(top(parent['refined'])['keypoints_xy'])[:8]-np.asarray(top(parent['raw'])['keypoints_xy'])[:8],axis=1)
                row=dict(id=parent['id'],image=parent['image'],session=parent['session'],protocol_sha256=C.sha(DOC/'PROTOCOL.json'),
                         **measure,raw_instability=raw_measure,correction_px=move.tolist(),outputs=outputs,identity_parity_px=parity)
                C.freeze(path,row)
            rows.append(row)
            if (i+1)%25==0 or i+1==249:
                print('REVIEW',i+1,'/249',round(time.monotonic()-start,1),'sec',C.N.E.gpu(),flush=True)
        for b in C.read(DOC/'PROTOCOL.json')['sources']:C.verify(b)
        C.freeze(RAW/'INFERENCE.json',rows)
        ranked=order(rows)
        C.freeze(RAW/'REVIEW_QUEUE.json',[dict(rank=i+1,id=r['id'],score=r['score'],priority=i<50,decision='UNREVIEWED') for i,r in enumerate(ranked)])
        # Exact unchanged parent payload, for later explicit review import only.
        original=parents();C.freeze(RAW/'UNCHANGED_CANDIDATES.json',copy.deepcopy(original))
        assert C.read(RAW/'UNCHANGED_CANDIDATES.json')==original
        C.freeze(DOC/'VERIFICATION.json',dict(complete=True,images=249,variants_per_image=5,
            missing_variants=sum(any(o['refined'] is None for o in r['outputs']) for r in rows),
            max_identity_parity_px={k:max(r['identity_parity_px'][k] for r in rows) for k in ['raw','refined']},
            corrected_coordinates_unchanged=True,automatic_rejections=0,training_started=False,
            artifacts=[C.bound(RAW/p) for p in ['INFERENCE.json','REVIEW_QUEUE.json','UNCHANGED_CANDIDATES.json']]))
    finally:extractor.close()


EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def panel(im,points,color,title,extra=None):
    canvas=im.copy()
    def draw(q,c,width):
        q=np.asarray(q,float)
        for a,b in EDGES:
            cv2.line(canvas,tuple(np.rint(q[a]).astype(int)),tuple(np.rint(q[b]).astype(int)),c,width,cv2.LINE_AA)
        for j,p in enumerate(q[:8]):
            xy=tuple(np.rint(p).astype(int));cv2.circle(canvas,xy,4,c,2,cv2.LINE_AA)
            cv2.putText(canvas,f'P{j}',(xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(canvas,f'P{j}',(xy[0]+5,xy[1]-5),cv2.FONT_HERSHEY_SIMPLEX,.45,c,1,cv2.LINE_AA)
    if extra is not None:draw(extra,(230,65,235),1)
    if points is not None:draw(points,color,2)
    cv2.rectangle(canvas,(0,0),(canvas.shape[1],28),(25,25,25),-1)
    cv2.putText(canvas,title,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
    return canvas


def gallery():
    data=order(C.read(RAW/'INFERENCE.json'));parent={r['id']:r for r in parents()}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    cards=[]
    for i,row in enumerate(data):
        p=parent[row['id']];C.verify(p['image']);im=cv2.imread(str(C.ROOT/p['image']['path']))
        refined=top(p['refined'])['keypoints_xy'];raw=top(p['raw'])['keypoints_xy']
        variants=[o for o in row['outputs'][1:] if o['refined'] is not None]
        worst=max(variants,key=lambda o:np.linalg.norm(np.asarray(o['refined'])[:8]-np.asarray(refined)[:8],axis=1).max()) if variants else None
        panels=[panel(im,None,None,'Original image / NO GT'),panel(im,raw,(255,185,45),'R0 raw prediction'),
                panel(im,refined,(30,225,255),'Replay label: unchanged'),
                panel(im,refined,(30,225,255),'Yellow: label / Magenta: '+(worst['variant'] if worst else 'missing'),worst['refined'] if worst else None)]
        strip=np.concatenate(panels,axis=1)
        filename=f'images/{i+1:03d}.jpg';dest=OUT/filename
        if not dest.exists():assert cv2.imwrite(str(dest),strip,[cv2.IMWRITE_JPEG_QUALITY,88])
        cards.append(dict(rank=i+1,id=row['id'],image=filename,source=row['image']['path'],session=row['session'].split('/')[-1],
                          score=row['score'],rms=row['max_corner_rms_px'],corners=row['per_corner_rms_px'],move=row['correction_px']))
    template=(C.HERE/'stability_review.html').read_text()
    html=template.replace('__DATA__',json.dumps(cards,ensure_ascii=False).replace('</','<\\/')).replace('__SOURCE_HASH__',C.sha(RAW/'UNCHANGED_CANDIDATES.json'))
    C.write_text(OUT/'index.html',html)
    print('GALLERY',OUT/'index.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['run','gallery']);args=parser.parse_args()
    run() if args.phase=='run' else gallery()
