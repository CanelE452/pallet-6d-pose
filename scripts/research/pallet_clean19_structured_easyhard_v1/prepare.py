import copy
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.augment import RandomPerspective
from . import common as C
from . import augmentation as A


def inputs():
    assert not C.DOC.exists() and not C.RAW.exists()
    C.DOC.mkdir(parents=True);C.RAW.mkdir(parents=True)
    split=C.read(C.H.P.DOC/'SPLIT.json');rows=C.read(C.H.RAW/'PSEUDO_LABELS.json')
    assert len(rows)==19 and len(split['evaluation'])==300
    assert not {r['image']['sha256'] for r in rows}&{r['image']['sha256'] for r in split['evaluation']}
    protected=[C.bind(C.H.RAW/'PSEUDO_LABELS.json'),C.bind(C.H.P.DOC/'SPLIT.json'),C.bind(C.H.DOC/'TRAIN_PROTOCOL.json'),C.bind(C.H.P.DOC/'TRAIN_SUPPORT.json'),C.bind(C.H.C.N.E.R0)]
    for mat in C.MATERIALS:
        protected.append(C.read(C.H.DOC/f'FIT_{mat}.json')['checkpoint'])
        protected.append(C.read(C.ROOT/f'_docs/experiments/pallet_replay_by_type_v1/{mat.lower()}/FIT.json')['checkpoint'])
    sourcefiles=['scripts/self_training_yolo/v3/true_ignore_trainer.py','scripts/self_training_yolo/v3/true_ignore_pose_loss.py','scripts/research/pallet_large_error_refiner_v1/run.py','scripts/research/pallet_clean19_student_v1.py','scripts/research/pallet_type_selftrain_v1/train.py','scripts/research/pallet_visible_refine_hidden_pnp_v1.py','scripts/research/pallet_clean19_student_pose_v1.py','scripts/paper/pose_metric_closure_v1/review_physical_axis.py']
    protected.extend(C.bind(C.ROOT/p) for p in sourcefiles)
    targets=[];counts=Counter()
    for i,r in enumerate(rows):
        ann=C.read(C.ROOT/r['annotation']['path']);manual,mm=C.H.P.C.R.manual_target(ann)
        target=np.array(C.H.P.C.selected(r['refined'])['keypoints_xy'])
        oldfile=C.H.RAW/'dataset/labels'/f'real_{i:02d}.txt'
        parsed=np.array(oldfile.read_text().split()[5:],float).reshape(9,3);old=parsed[:,2]==2
        replaced=np.zeros(9,bool);replaced[r['decision'].get('replaced',[])]=True
        unknown=np.zeros(9,bool)
        for k,entry in enumerate(ann['objects'][0]['keypoint_annotations']):
            unknown[k]=any('identity' in field.lower() and str(value).lower() in ('unknown','unverified','unresolved') for field,value in entry.items())
        mask=mm&old&~replaced&~unknown;mask[8]=False
        assert not (mask&replaced).any()
        counts.update(old_pseudo=int(old.sum()),old_pseudo_corners=int(old[:8].sum()),manual=int(mm.sum()),new_supervised=int(mask.sum()),
            manual_replaced=int((mm&replaced).sum()),PnP_replaced=int(replaced.sum()),not_manual=int((old&~mm).sum()),identity_unknown=int(unknown.sum()),center_removed=int(old[8]))
        targets.append(dict(id=r['id'],material=r['object_type'].upper(),index=i,image=r['image'],annotation=r['annotation'],hw=r['raw_hw'],
            target=target.tolist(),manual=np.nan_to_num(manual,nan=-1).tolist(),manual_mask=mm.tolist(),old_mask=old.tolist(),mask=mask.tolist(),replaced=replaced.tolist(),unknown_identity=unknown.tolist(),
            bbox=C.H.P.C.selected(r['raw'])['box_xyxy']))
        protected.extend([r['image'],r['annotation'],C.bind(oldfile)])
    assert counts['old_pseudo']==171 and counts['manual']==87
    C.save(C.DOC/'TARGETS.json',targets)
    C.save(C.DOC/'TRAIN_TARGET_AUDIT.json',dict(counts=dict(counts),records=targets,coordinates='frozen teacher, never manual replacement',visibility='manual provenance is NOT physical visibility',mask_threshold_tuning=False))
    parent=C.read(C.H.DOC/'TRAIN_PROTOCOL.json');datasets={}
    for mat in C.MATERIALS:
        folder=C.RAW/'dataset'/mat
        oldlist=C.ROOT/parent['datasets'][mat]['train_list']['path'];C.verify(parent['datasets'][mat]['train_list'])
        slots=[]
        for old in oldlist.read_text().splitlines():
            src=Path(old);label=src.parent.parent/'labels'/src.with_suffix('.txt').name
            dst=folder/'images'/src.name;C.H.T.link(src,dst);newlabel=folder/'labels'/label.name
            if not newlabel.exists():
                newlabel.parent.mkdir(parents=True,exist_ok=True)
                if src.name.startswith('real_'):
                    idx=int(src.stem.split('_')[1]);tokens=label.read_text().split();k=np.array(tokens[5:],float).reshape(9,3)
                    k[:,2]=np.where(targets[idx]['mask'],2,1)
                    C.save(newlabel,' '.join(tokens[:5]+[f'{v:.9f}' for v in k.ravel()])+'\n')
                else:shutil.copy2(label,newlabel)
            slots.append(str(dst))
            protected.extend([C.bind(src),C.bind(label)])
        assert len(slots)==1024
        C.save(folder/'train.txt','\n'.join(slots)+'\n')
        # bookkeeping val remains fixed and is never used for model selection.
        val=(C.H.RAW/'dataset/val.txt').read_text();C.save(folder/'val.txt',val)
        C.save(folder/'data.yaml',f'path: {folder}\ntrain: {folder/"train.txt"}\nval: {folder/"val.txt"}\nnc: 1\nnames: [pallet]\nkpt_shape: [9, 3]\nflip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 8]\n')
        datasets[mat]=str(folder/'data.yaml')
    uniq={b['path']:b for b in protected}
    for b in uniq.values():C.verify(b)
    C.save(C.DOC/'INPUT_BINDINGS.json',dict(files=list(uniq.values())))
    C.save(C.DOC/'PREFLIGHT.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        status=subprocess.check_output(['git','status','--short','--branch'],text=True),disk_free=shutil.disk_usage(C.ROOT).free,
        same_session=True,train_eval_image_overlap=0,near_duplicate=split['near_duplicate_audit'],datasets=datasets,args=parent['args'],
        graph=dict(edges=A.EDGES,source=C.bind(Path(A.__file__))),no_commit_push=True))
    print('TARGET_COUNTS',dict(counts),flush=True)


def train_check():
    C.setup();C.guard();C.immutable();targets=C.read(C.DOC/'TARGETS.json');outputs={}
    for mat in C.MATERIALS:
        for arm,checkpoint in [('R0',C.bind(C.H.C.N.E.R0)),('OLD_STUDENT',C.read(C.H.DOC/f'FIT_{mat}.json')['checkpoint'])]:
            model=YOLO(str(C.ROOT/checkpoint['path']),task='pose')
            outputs[mat+'_'+arm]={}
            for r in targets:
                if r['material']!=mat:continue
                im=cv2.imread(str(C.ROOT/r['image']['path']))
                outputs[mat+'_'+arm][r['id']]={mode:C.predict(model,im,pad) for mode,pad in [('native',0),('reflect100',100)]}
            del model;torch.cuda.empty_cache()
    C.save(C.RAW/'OLD_TRAIN_PREDICTIONS.json',outputs)
    summaries={};detailed=[]
    perms={r['object_type']:r['permutations'] for r in C.read(C.H.C.N.E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    for mat in C.MATERIALS:
        summaries[mat]={}
        for arm in ('R0','OLD_STUDENT','TEACHER'):
            summaries[mat][arm]={}
            for mode in ('native','reflect100'):
                group=[]
                for r in targets:
                    if r['material']!=mat:continue
                    pred=next(x['refined'] for x in C.read(C.H.RAW/'PSEUDO_LABELS.json') if x['id']==r['id']) if arm=='TEACHER' else outputs[mat+'_'+arm][r['id']][mode]
                    masks={'pseudo8':r['old_mask'][:8]+[False],'center':[False]*8+[r['old_mask'][8]],'manual':r['manual_mask'],'new_mask':r['mask'],'PnP_replaced':r['replaced']}
                    c=C.H.P.C.selected(pred);matched=c is not None and C.H.E.O.iou(c['box_xyxy'],r['bbox'])>=.5
                    row=dict(id=r['id'],material=mat,arm=arm,mode=mode,matched_teacher_box=matched,
                        metrics={key:C.native(pred,r['manual'] if key=='manual' else r['target'],mask,r['hw']) for key,mask in masks.items()})
                    q=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
                    row['whole_object_symmetry_pseudo8']=C.H.P.M.measure(q,r['target'],masks['pseudo8'],perms[C.H.C.TYPES[mat]],r['hw'],True,c is not None)
                    detailed.append(row);group.append(row)
                summaries[mat][arm][mode]={key:C.summary([r['metrics'][key] for r in group]) for key in masks}
                summaries[mat][arm][mode]['matched_teacher_box']=sum(r['matched_teacher_box'] for r in group)
    # Required native channel parity of the R0 padded path, independently re-inferred.
    cached=C.read(C.H.P.RAW/'BASELINE_PREDICTIONS.json')['R0'];deltas=[]
    for r in targets:
        a=C.H.P.C.selected(cached[r['id']]);b=C.H.P.C.selected(outputs[r['material']+'_R0'][r['id']]['reflect100'])
        deltas.append(float(np.abs(np.array(a['keypoints_xy'])-np.array(b['keypoints_xy'])).max()))
    assert max(deltas)<1e-3,('R0_coordinate_contract',max(deltas))
    C.save(C.DOC/'OLD_TRAIN_FIT.json',dict(summary=summaries,records=detailed,R0_pad_parity_max_px=max(deltas),no_training=True))
    print('OLD_TRAIN_FIT_COMPLETE',flush=True)


def cache():
    C.setup();C.immutable();assert (C.DOC/'OLD_TRAIN_FIT.json').exists()
    targets={r['index']:r for r in C.read(C.DOC/'TARGETS.json')}
    pre=C.read(C.DOC/'PREFLIGHT.json');cfg=get_cfg(overrides={**pre['args'],'task':'pose'})
    from ultralytics.data.utils import check_det_dataset
    audit=Counter();plans=[];transforms=[];max_roundtrip=0.
    assert shutil.disk_usage(C.ROOT).free>20*1024**3
    for mi,mat in enumerate(C.MATERIALS):
        data=check_det_dataset(pre['datasets'][mat]);ds=build_yolo_dataset(copy.deepcopy(cfg),data['train'],16,data,mode='train',rect=False,stride=32)
        assert len(ds)==1024
        # Capture actual installed transform matrix, not an invented geometry formula.
        captured={};original=RandomPerspective.get_params
        def capture(self,labels):
            params=original(self,labels);captured.update(params);return params
        RandomPerspective.get_params=capture
        try:
            for epoch in range(5):
                order=np.random.default_rng(20260922+mi*100+epoch).permutation(1024)
                for slot,index in enumerate(order):
                    seed=20260922+mi*100000+epoch*1024+slot
                    random.seed(seed);np.random.seed(seed);captured.clear()
                    path=Path(ds.im_files[int(index)]);real=path.name.startswith('real_')
                    sample=ds[int(index)]
                    assert sample['img'].shape==(3,640,640)
                    img=sample['img'].numpy();kp=sample['keypoints'].numpy();box=sample['bboxes'].numpy()
                    plan=dict(applied=False,reason='synthetic_unchanged',scheduled=False)
                    matrix=np.array(captured['M']);native_matrix=None;idx=None
                    if real:
                        idx=int(path.stem.split('_')[1]);target=targets[idx];audit['real']+=1
                        assert len(kp)==1,'Real instance removed by geometry; stop integrity audit'
                        assert captured['orig_shape']==tuple(ds.load_image(int(index))[0].shape[:2])
                        oh,ow=target['hw'];rh,rw=captured['orig_shape']
                        padded_to_resized=np.diag([rw/(ow+200),rh/(oh+200),1.])
                        pad=np.eye(3);pad[:2,2]=100
                        native_matrix=matrix@padded_to_resized@pad
                        raw=np.c_[target['target'],np.ones(9)]
                        projected=(raw@native_matrix.T)[:,:2]
                        e=float(np.max(np.abs(np.clip(projected,0,640)-kp[0,:,:2]*640)))
                        assert e<.001,('affine_coordinate_mismatch',e)
                        recovered=np.c_[projected,np.ones(9)]@np.linalg.inv(native_matrix).T
                        max_roundtrip=max(max_roundtrip,float(np.max(np.abs(recovered[:,:2]-raw[:,:2]))))
                        mask=np.array(target['mask'])
                        # Preserve excluded real sentinel even if stock geometry set out-of-frame to0.
                        kp[0,~mask,2]=1
                        canvas=np.array([[0,0,1],[ow,oh,1]])@native_matrix.T
                        canvas=np.r_[canvas[0,:2],canvas[1,:2]]
                        xywh=box[0]*640;b=np.r_[xywh[:2]-xywh[2:]/2,xywh[:2]+xywh[2:]/2]
                        plan=A.plan(kp[0,:,:2]*640,kp[0,:,2]==2,b,canvas,seed+1000000)
                        plan.update(native_canvas=canvas.tolist(),bbox=b.tolist())
                        for arm in ('S1','S2'):
                            applied=A.apply(img,plan,arm)
                            assert applied.shape==img.shape
                            if plan['applied']:
                                l,t,w,h=plan[arm];assert np.array_equal(applied[:,t:t+h,l:l+w],A.fill(plan))
                                assert w*h==plan['size'][0]*plan['size'][1]
                                audit[arm+'_masked']+=len(plan[arm+'_masked'])
                        audit[plan['reason']]+=1
                        audit['scheduled']+=int(plan['scheduled']);audit['applied']+=int(plan['applied'])
                    else:audit['synthetic']+=1
                    destination=C.RAW/'cache'/mat/f'{epoch}_{slot:04d}.npz';destination.parent.mkdir(parents=True,exist_ok=True)
                    arrays=dict(img=img,keypoints=kp,bboxes=box,cls=sample['cls'].numpy(),batch_idx=sample['batch_idx'].numpy())
                    if destination.exists():
                        with np.load(destination) as existing:
                            for key,value in arrays.items():np.testing.assert_array_equal(existing[key],value)
                    else:np.savez(destination,**arrays)
                    record=dict(material=mat,epoch=epoch,slot=slot,dataset_index=int(index),image=str(path),real=real,target_index=idx,
                        base_seed=seed,base_RGB_sha256=C.digest(img),target_sha256=C.digest(kp),box_sha256=C.digest(box),affine=matrix.tolist(),
                        native_to_model=None if native_matrix is None else native_matrix.tolist(),cache=C.bind(destination),plan=plan)
                    plans.append(record)
                print('CACHE',mat,epoch+1,'applied',audit['applied'],flush=True)
        finally:RandomPerspective.get_params=original
        transforms.append(dict(material=mat,transform_repr=str(ds.transforms)))
    with (C.DOC/'AUGMENTATION_PLAN.jsonl').open('x') as f:
        for r in plans:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    C.save(C.DOC/'AUGMENTATION_AUDIT.json',dict(counts=dict(audit),max_roundtrip_px=max_roundtrip,
        material={m:dict(real=sum(r['real'] for r in plans if r['material']==m),applied=sum(r['plan']['applied'] for r in plans if r['material']==m)) for m in C.MATERIALS},
        transforms=transforms,base_tensors_shared=True,original_historical_student_tensor_parity_not_claimed=True,
        paired_target_mask_unchanged=True,shape_fill_frequency_paired=True,graph=A.EDGES,
        status='READY' if audit['applied'] else 'AUGMENTATION_NOT_INSTANTIATED'))
    assert audit['applied']>0,'AUGMENTATION_NOT_INSTANTIATED'
    print('CACHE_COMPLETE',dict(audit),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['inputs','train_check','cache']);args=p.parse_args();globals()[args.stage]()
