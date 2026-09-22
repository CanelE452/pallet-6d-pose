import argparse
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from . import common as C
from . import augmentation as A


def lock():
    C.setup();C.immutable()
    plans=[json.loads(x) for x in (C.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()]
    targets=C.read(C.DOC/'TARGETS.json');chosen=[]
    for r in targets:
        rows=[p for p in plans if p['real'] and p['target_index']==r['index']]
        chosen.extend(rows[:8])
    C.save(C.DOC/'TRAIN_PROBE_PLAN.json',dict(records=chosen,rule='First8 fixed occurrences per original TRAIN image, before model scoring; no difficulty mining',frames=len(chosen)))
    from scripts.research.pallet_posefix_replay_v1.source import SourceData
    source=SourceData();held=np.load(C.H.P.RAW/'ORDERS.npz')['held_rows'];records=[]
    synpaths={str(Path(r['image']).resolve()) for r in plans if not r['real']}
    for row in held:
        record=source.data.source['records'][int(source.data.indices[int(row)])]
        assert record['partition']=='heldout' and str(Path(record['image']).resolve()) not in synpaths
        records.append(dict(id=record['id'],image=C.bind(Path(record['image'])),label=C.bind(Path(record['label'])),targets=record['targets'],hw=record['prepared_shape_hw']))
    assert len(records)==256
    C.save(C.DOC/'SOURCE_PROBE_PLAN.json',dict(records=records,source=C.bind(C.H.P.RAW/'ORDERS.npz'),existing_heldout=True,
        stress='Existing refiner stress perturbs keypoint input, absent in RGB-only student; not relabeled as student robustness. No new source occlusion recipe.',bookkeeping_overlap_possible=True))
    # Use actual masked TRAIN image for installed criterion verification; no eval inputs.
    p=next(p for p in chosen if p['plan']['applied'])
    z=np.load(C.ROOT/p['cache']['path']);image=A.apply(z['img'],p['plan'],'S2')
    from scripts.self_training_yolo.v3 import verify_true_ignore_contract as V
    V._SAMPLE=(torch.from_numpy(cv2.resize(image.transpose(1,2,0),(320,320)).transpose(2,0,1).copy())[None].float()/255,
        torch.from_numpy(z['keypoints'][0,:,:2].copy()),torch.from_numpy(z['bboxes'].copy()))
    vis=z['keypoints'][0,:,2].tolist();ignored=vis.index(1.);supervised=p['plan']['S2_masked'][0]
    torch.manual_seed(42);base=V.measure(vis)
    torch.manual_seed(42);shift_ignore=V.measure(vis,shift=(ignored,60.))
    torch.manual_seed(42);shift_super=V.measure(vis,shift=(supervised,60.))
    torch.manual_seed(42);all_ignore=V.measure([1.]*9)
    torch.manual_seed(42);stock=V.measure([2.]*4+[0.]*5,False)
    torch.manual_seed(42);custom=V.measure([2.]*4+[0.]*5,True)
    tests=dict(ignored_shift_same=V.close(base['total_loss'],shift_ignore['total_loss']),
        masked_supervised_shift_changes=not V.close(base['total_loss'],shift_super['total_loss']),
        excluded_kobj_location_rle_zero=all(all_ignore['items'][k]==0 for k in ('kpt_location','kpt_visibility','rle')),
        excluded_keypoint_gradient_zero=all_ignore['keypoint_branch_grad']==0,
        source_stock_loss_parity=V.close(stock['total_loss'],custom['total_loss']),
        source_stock_gradient_parity=V.close(stock['keypoint_branch_grad'],custom['keypoint_branch_grad']))
    C.save(C.DOC/'LOSS_TEST.json',dict(passed=all(tests.values()),tests=tests,actual_masked_TRAIN_occurrence=p['cache'],runs=dict(base=base,ignored=shift_ignore,supervised=shift_super,all_ignore=all_ignore,stock=stock,custom=custom)))
    assert all(tests.values()),tests
    audit=[]
    for p in plans:
        if not p['real']:continue
        with np.load(C.ROOT/p['cache']['path']) as z:
            t=targets[p['target_index']];kp=z['keypoints'];mask=np.array(t['mask'])
            assert (kp[0,~mask,2]==1).all()
            assert C.digest(kp)==p['target_sha256']
            if p['plan']['applied']:
                assert p['plan']['S1'][2:]==p['plan']['S2'][2:]
                assert p['plan']['bbox_overlap_difference']<=p['plan']['bbox_overlap_tolerance']
                assert p['plan']['S2_edge_count']>=1
            audit.append(p)
    C.save(C.DOC/'PAIR_TEST.json',dict(passed=True,real_occurrences=len(audit),size_fill_frequency=True,excluded_mask_preserved=True,coordinate_roundtrip=C.read(C.DOC/'AUGMENTATION_AUDIT.json')['max_roundtrip_px']))
    bindings=[C.bind(p) for p in sorted(C.HERE.glob('*.py'))]+[C.bind(C.DOC/f) for f in ('TARGETS.json','AUGMENTATION_PLAN.jsonl','TRAIN_PROBE_PLAN.json','SOURCE_PROBE_PLAN.json','LOSS_TEST.json','PAIR_TEST.json')]
    C.save(C.DOC/'PROTOCOL.json',dict(bindings=bindings,train19=True,evaluation300_fixed=True,epochs=5,steps_per_fit=320,fits=6,seed=42,
        augmentation_seed=20260922,area=[.1,.2,.3],aspect=[.5,1.,2.],planned_probability=.5,max_positions=64,
        base_augmentation='installed YOLO transforms cached once, including geometric clipping; nonmanual real sentinel restored1; same tensors replayed',
        source='unchanged synthetic512+old validation bookkeeping, locked held256',
        student_RGB_only=True,no_tuning=True,no_push=True,graph_source=C.bind(Path(A.__file__)),
        preflight_correction='Initial roundtrip test compared clipped coordinates to unclipped affine. Fixed check to model clipping; no training had started. Existing partial cache verified bit-exact, not overwritten.',
        albumentations='Installed optional Albumentations constructor rejects quality_range and disables itself; same behavior as previous student and all three new arms. No package edits.',
        references='User-supplied CVF papers; direct page retrieval403, not claimed full method reproduction. Low-frequency rectangles are not human-body Cut-Occlude.'))
    print('LOSS_AND_PAIR_TEST_PASS',tests,flush=True)


def probe_model(model,material,original=True):
    targets={r['index']:r for r in C.read(C.DOC/'TARGETS.json')};output=[]
    if original:
        for r in targets.values():
            if r['material']!=material:continue
            im=cv2.imread(str(C.ROOT/r['image']['path']))
            for mode,pad in [('native',0),('reflect100',100)]:
                pred=C.predict(model,im,pad)
                output.append(dict(id=r['id'],mode=mode,prediction=pred,
                    metrics={name:C.native(pred,r['manual'] if name=='manual' else r['target'],mask,r['hw']) for name,mask in [('pseudo8',r['old_mask'][:8]+[False]),('manual',r['manual_mask']),('new_mask',r['mask']),('center',[False]*8+[True])]}))
    for p in C.read(C.DOC/'TRAIN_PROBE_PLAN.json')['records']:
        if p['material']!=material:continue
        r=targets[p['target_index']];z=np.load(C.ROOT/p['cache']['path']);target=z['keypoints'][0,:,:2]*640;mask=z['keypoints'][0,:,2]==2
        raw=np.c_[r['manual'],np.ones(9)]@np.array(p['native_to_model']).T;manual=raw[:,:2]
        for mode in C.ARMS:
            rgb=A.apply(z['img'],p['plan'],mode);pred=C.predict(model,rgb.transpose(1,2,0)[:,:,::-1].copy(),0)
            covered=A.cover(target,p['plan'][mode]) if mode!='S0' and p['plan']['applied'] else np.zeros(9,bool)
            output.append(dict(id=r['id'],mode=mode,occ=[p['epoch'],p['slot']],applied=bool(covered.any()),prediction=pred,
                metrics={key:C.native(pred,manual if key=='manual' else target,m,(640,640)) for key,m in [('all',mask),('masked',mask&covered),('remaining',mask&~covered),('manual',np.array(r['manual_mask'])&mask)]}))
    return output


def source_model(model):
    rows=[]
    for r in C.read(C.DOC/'SOURCE_PROBE_PLAN.json')['records']:
        C.verify(r['image']);im=cv2.imread(str(C.ROOT/r['image']['path']));h,w=im.shape[:2]
        pred=C.predict(model,im,0);c=C.H.P.C.selected(pred)
        # Original locked target; single-object source records only.
        assert len(r['targets'])==1
        t=r['targets'][0];k=np.array(t['keypoints_normalized']);gt=k[:,:2]*[w,h];valid=k[:,2]>0;valid[8]=False
        b=np.array(t['box_xywh_normalized'])*[w,h,w,h];box=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]
        matched=c is not None and C.H.E.O.iou(c['box_xyxy'],box)>=.5
        metric=C.native(pred,gt,valid,(h,w))
        if not matched:metric['errors']=[float(np.hypot(h,w))]*int(valid.sum())
        rows.append(dict(id=r['id'],prediction=pred,matched=matched,**metric))
    return rows


def before():
    C.setup();C.guard();C.immutable();out={}
    for mat in C.MATERIALS:
        out[mat]={}
        for name,path in [('R0',C.H.C.N.E.R0),('OLD_STUDENT',C.ROOT/C.read(C.H.DOC/f'FIT_{mat}.json')['checkpoint']['path'])]:
            model=YOLO(str(path),task='pose')
            out[mat][name]=dict(train=probe_model(model,mat,False),source=source_model(model))
            del model;torch.cuda.empty_cache();print('PROBE',mat,name,flush=True)
    C.save(C.RAW/'PROBE_BEFORE_PREDICTIONS.json',out)
    stats={m:{a:{mode:{k:C.summary([r['metrics'][k] for r in v['train'] if r['mode']==mode]) for k in ('all','masked','remaining','manual')} for mode in C.ARMS} for a,v in arms.items()} for m,arms in out.items()}
    C.save(C.DOC/'PROBE_BEFORE.json',dict(complete=True,summary=stats,predictions=C.bind(C.RAW/'PROBE_BEFORE_PREDICTIONS.json'),no_model_output_used_for_plan=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['lock','before']);a=p.parse_args();globals()[a.stage]()
