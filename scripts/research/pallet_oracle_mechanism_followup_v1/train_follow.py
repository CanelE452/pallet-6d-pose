"""TRAIN-only native target-following; pseudo targets are not reference truth."""
import argparse,time,csv
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C
from scripts.research.pallet_visible_transfer_closure_v1 import common as V
from scripts.research.pallet_visible_transfer_closure_v1.diagnose import aggregate_train

def prepare_wood():
    path=C.RAW/'WOOD_TRAIN_TARGETS_PRIVATE.json'
    if path.exists():return C.read(path)
    protocol=C.read(C.M.DOC/'WOOD_TRAIN_PROTOCOL.json')
    lists={a:[Path(p) for p in (C.ROOT/protocol['datasets'][a]['train_list']['path']).read_text().splitlines()] for a in ('RAW','REF')}
    assert [p.name for p in lists['RAW']]==[p.name for p in lists['REF']]
    counts=Counter(p.stem for p in lists['RAW'] if not p.name.startswith('syn__'))
    assert len(counts)==361 and sum(counts.values())==512
    accepted={r['id']:r for r in C.read(C.M.RAW/'WOOD_ACCEPTED_SHARED.json')}
    lookup={a:{p.stem:p for p in pp} for a,pp in lists.items()};rows=[]
    for fid,n in sorted(counts.items()):
        r=accepted[fid];h,w=r['raw_hw'];targets={};masks={};bindings={}
        for a in ('RAW','REF'):
            p=lookup[a][fid];label=p.parent.parent/'labels'/f'{fid}.txt'
            q=np.asarray(label.read_text().split(),float)[5:].reshape(9,3)
            masks[a]=q[:,2]==2;targets[a]=q[:,:2]*[w+200,h+200]-100;bindings[a]=C.bind(label)
            stored=np.array(C.P.selected(r['raw' if a=='RAW' else 'refined'])['keypoints_xy'])
            assert np.allclose(targets[a][masks[a]],stored[masks[a]],rtol=0,atol=1e-5)
        assert np.array_equal(masks['RAW'],masks['REF'])
        raw=C.P.selected(r['raw']);ref=C.P.selected(r['refined'])
        assert raw['box_xyxy']==ref['box_xyxy'] and raw['keypoints_conf']==ref['keypoints_conf']
        rows.append(dict(id=fid,image=r['image'],hw=[h,w],recording=r.get('recording_id',r.get('recording')),
            occurrences_per_epoch=n,common_support=masks['RAW'],raw_target=targets['RAW'],ref_target=targets['REF'],
            padded_label_bindings=bindings,predicted_box=raw['box_xyxy'],keypoint_confidence=raw['keypoints_conf']))
    C.save(path,rows,True);return C.read(path)

def infer():
    import cv2,torch
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    rows=prepare_wood();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    for arm in C.ARMS:
        dst=C.RAW/f'WOOD_TRAIN_PREDICTIONS_{arm}.json';ck=C.checkpoint('WOOD',arm);C.verify(ck)
        if dst.exists():assert C.read(dst)['checkpoint']==ck;continue
        begin=time.monotonic();model=YOLO(str(C.ROOT/ck['path']),task='pose');done={}
        progress=C.RAW/f'WOOD_TRAIN_PROGRESS_{arm}.json'
        if progress.exists():
            old=C.read(progress);assert old['checkpoint']==ck;done=old['predictions']
        try:
            for n,row in enumerate(rows):
                if row['id'] in done:continue
                if n%32==0:
                    thermal_guard();C.save(progress,dict(checkpoint=ck,predictions=done))
                    print('TRAIN_FROZEN',arm,n,len(rows),flush=True)
                C.verify(row['image']);im=cv2.imread(str(C.ROOT/row['image']['path']));assert im is not None
                done[row['id']]=predict(model,im)
            C.save(dst,dict(checkpoint=ck,predictions=done,native_unaugmented=True,highest_confidence_only=True,
                           GT_or_target_matching=False),True)
        finally:
            C.resource('frozen_Wood_TRAIN_'+arm,time.monotonic()-begin,gpu=True)
            del model;torch.cuda.empty_cache()
    C.save(C.DOC/'TRAIN_PREDICTIONS_LOCK.json',dict(
        WOOD=[C.bind(C.RAW/f'WOOD_TRAIN_PREDICTIONS_{a}.json') for a in C.ARMS],
        PLASTIC_REUSED=[C.bind(V.RAW/f'TRAIN_PREDICTIONS_{a}.json') for a in C.ARMS],
        note='Highest score native/unaugmented target-following, no reference access, no target-based candidate selection'),True)

def analyze():
    output={}
    for material,rows,ps in [('WOOD',prepare_wood(),{a:C.read(C.RAW/f'WOOD_TRAIN_PREDICTIONS_{a}.json')['predictions'] for a in C.ARMS}),
       ('PLASTIC',C.read(V.RAW/'TRAIN_TARGETS_PRIVATE.json'),{a:C.read(V.RAW/f'TRAIN_PREDICTIONS_{a}.json')['predictions'] for a in C.ARMS})]:
        points=[];missing=Counter();areas=[max(0,r['predicted_box'][2]-r['predicted_box'][0])*max(0,r['predicted_box'][3]-r['predicted_box'][1]) for r in rows]
        boundaries=np.quantile(areas,[1/3,2/3])
        for r,area in zip(rows,areas):
            pred={a:C.P.selected(ps[a][r['id']]) for a in C.ARMS}
            for a in C.ARMS:missing[a]+=pred[a] is None
            assert all(pred.values()),'Missing TRAIN detection: implement explicit coverage before aggregation'
            for ci,valid in enumerate(r['common_support']):
                if not valid:continue
                raw,ref=np.array(r['raw_target'][ci]),np.array(r['ref_target'][ci]);p={a:np.array(pred[a]['keypoints_xy'][ci]) for a in C.ARMS}
                u,v=ref-raw,p['REF_LR5']-p['RAW_LR5'];un,vn=np.linalg.norm(u),np.linalg.norm(v);ok=un>=1 and vn>1e-8
                points.append(dict(id=r['id'],corner=ci,recording=r['recording'],occurrences=r['occurrences_per_epoch'],
                    support_count=sum(r['common_support']),bbox_area=area,bbox_tertile=int(np.searchsorted(boundaries,area)),
                    correction=float(un),student_change=float(vn),cosine=float(u@v/un/vn) if ok else None,
                    projection=float(u@v/un**2) if ok else None,
                    residuals={a:{t:float(np.linalg.norm(p[a]-g)) for t,g in [('raw',raw),('ref',ref)]} for a in C.ARMS}))
        result=dict(all=aggregate_train(points),unique_images=len(rows),real_slots=512,missing=dict(missing),grouped={})
        for key in ('corner','recording','bbox_tertile','occurrences','support_count'):
            result['grouped'][key]={str(g):aggregate_train([r for r in points if r[key]==g]) for g in sorted({r[key] for r in points})}
        result['curves']={}
        for a in C.ARMS[1:]:
            fit=C.read((C.M.DOC/f'FIT_WOOD_{a}.json') if material=='WOOD' else C.P.REC/'pose_only'/f'FIT_{a}.json')
            path=C.ROOT/fit['results_csv']['path'];result['curves'][a]=[{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader(path.open())]
        result['meaning']='Pseudo-target imitation, NOT GT accuracy; native highest-score no target matching; augmented-input study separate'
        output[material]=result;C.save(C.RAW/f'{material}_TRAIN_POINT_RESIDUALS.json',points,True)
        print(material,len(rows),result['all']['corners']['residuals']['REF_LR5']['ref']['mean_px'],flush=True)
    C.save(C.DOC/'TRAIN_TARGET_TRANSFER.json',output,True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['infer','analyze']);a=p.parse_args()
    infer() if a.phase=='infer' else analyze()
