"""Freeze replacement occurrences and shared hard RGB/box/augmentation/support."""
import hashlib
import json
from collections import Counter
import cv2
import numpy as np
from ultralytics.data.augment import RandomHSV
from . import common as C
from scripts.research.pallet_clean19_structured_easyhard_v1 import common as S


def key(epoch,slot):return hashlib.sha256(f'min-hard-slot-v1:{epoch}:{slot}'.encode()).hexdigest()


def main():
    if (C.DOC/'TRAIN_OCCURRENCE_LOCK.json').exists():
        C.verify(C.read(C.DOC/'TRAIN_OCCURRENCE_LOCK.json')['occurrences']);return
    lock=C.read(C.DOC/'HARD_LABEL_LOCK.json');C.verify(lock['labels'])
    tl=C.read(C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json');C.verify(tl['predictions'])
    assert tl['coverage']==1.,'Partial teacher support requires explicit incomplete-arm handling, not mask changes'
    frames=list(C.read(C.ROOT/lock['labels']['path'])['frames'].values());teacher=C.read(C.ROOT/tl['predictions']['path'])
    original=[r for r in map(json.loads,(S.DOC/'AUGMENTATION_PLAN.jsonl').read_text().splitlines()) if r['material']=='PLASTIC']
    oldtargets={r['index']:r for r in C.read(S.DOC/'TARGETS.json')}
    args=C.read(S.DOC/'PREFLIGHT.json')['args'];occ=[];n=0;epoch_counts=[];offscreen=0
    cv2.setNumThreads(1)
    for epoch in range(5):
        records=[r for r in original if r['epoch']==epoch]
        real=[r for r in records if r['real']];assert len(real)==512
        replaced={r['slot'] for r in sorted(real,key=lambda r:key(epoch,r['slot']))[:64]}
        counts=Counter()
        for r in records:
            C.verify(r['cache'])
            d=dict(r,hard=r['slot'] in replaced)
            if not d['hard']:
                d['kind']='CLEAN' if r['real'] else 'SYNTH';counts[d['kind']]+=1;occ.append(d);continue
            f=frames[n%len(frames)];n+=1;fid=f['frame_id'];C.verify(f['image'])
            im=cv2.imread(str(C.ROOT/f['image']['path']));h,w=im.shape[:2]
            # Recover old resized input shape from its audited native->model matrix.
            oldh,oldw=oldtargets[r['target_index']]['hw']
            before=np.linalg.inv(r['affine'])@np.asarray(r['native_to_model'])
            rw=round(before[0,0]*(oldw+200));rh=round(before[1,1]*(oldh+200))
            padded=cv2.copyMakeBorder(im,100,100,100,100,cv2.BORDER_REFLECT_101)
            padded=cv2.resize(padded,(rw,rh),interpolation=cv2.INTER_LINEAR)
            matrix=np.asarray(r['affine']);img=cv2.warpAffine(padded,matrix[:2],(640,640),borderValue=(114,114,114))
            np.random.seed(r['base_seed'])
            img=RandomHSV(args['hsv_h'],args['hsv_s'],args['hsv_v'])({'img':img})['img']
            img=np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1))
            pad=np.eye(3);pad[:2,2]=100
            affine=matrix@np.diag([rw/(w+200),rh/(h+200),1.])@pad
            xy=np.zeros((9,2),np.float64);support=np.zeros(9,bool)
            for k,p in enumerate(f['corners']):
                if p['status']=='DIRECT_VISIBLE':xy[k]=p['xy'];support[k]=True
            p=teacher[fid]['prediction'];pseudo=np.array(p['candidates'][p['selected_index']]['keypoints_xy'],float)
            def transform(q):return (np.c_[q,np.ones(len(q))]@affine.T)[:,:2]
            m=transform(xy);t=transform(pseudo)
            # Shared mask is driven by manual support and manual transformed frame bounds.
            inframe=(m[:,0]>=0)&(m[:,0]<640)&(m[:,1]>=0)&(m[:,1]<640)
            offscreen+=int((support&~inframe).sum());mask=support&inframe
            assert mask.sum()>=2,('too few surviving direct points',epoch,r['slot'])
            kp={}
            for arm,q in [('H_MANUAL',m),('H_PSEUDO',t)]:
                k=np.zeros((1,9,3),np.float32);k[:,:,2]=1;k[0,mask,:2]=q[mask]/640;k[0,mask,2]=2;kp[arm]=k
            x1,y1,x2,y2=f['bbox'];bp=transform(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))
            b=np.clip(np.r_[bp.min(0),bp.max(0)],0,640);assert (b[2:]>b[:2]).all()
            box=np.array([[*(b[:2]+b[2:])/2/640,*(b[2:]-b[:2])/640]],np.float32)
            path=C.RAW/'hard_cache'/f'{epoch}_{r["slot"]:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            arrays=dict(img=img,bboxes=box,cls=np.zeros((1,1),np.float32),batch_idx=np.zeros(1,np.float32),**kp)
            if path.exists():
                with np.load(path) as old:
                    for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
            else:np.savez(path,**arrays)
            d.update(kind='HARD',frame_id=fid,hard_cache=C.bind(path),hard_affine=affine.tolist(),
                     hard_RGB_sha256=S.digest(img),hard_box_sha256=S.digest(box),
                     hard_targets={a:S.digest(k) for a,k in kp.items()},supervised=int(mask.sum()),hard_extra_occlusion=False)
            counts['HARD']+=1;occ.append(d)
        assert counts==dict(CLEAN=448,SYNTH=512,HARD=64),counts
        epoch_counts.append(dict(counts));print('HARD_CACHE',epoch+1,dict(counts),flush=True)
    path=C.RAW/'TRAIN_OCCURRENCES_PRIVATE.json';C.save(path,occ,immutable=True)
    fit=C.read(S.DOC/'FIT_PLASTIC_S1.json');C.verify(fit['checkpoint'])
    C.save(C.DOC/'TRAIN_OCCURRENCE_LOCK.json',dict(created_at=C.now(),occurrences=C.bind(path),
           original_plan=C.bind(S.DOC/'AUGMENTATION_PLAN.jsonl'),label_lock=C.bind(C.DOC/'HARD_LABEL_LOCK.json'),teacher_lock=C.bind(C.DOC/'TEACHER_HARD_PREDICTION_LOCK.json'),
           epoch_counts=epoch_counts,hard_total=n,hard_per_frame=dict(Counter(r.get('frame_id') for r in occ if r['hard'])),
           original_init=C.bind(S.H.C.N.E.R0),base_checkpoint=fit['checkpoint'],fit_binding=C.bind(S.DOC/'FIT_PLASTIC_S1.json'),
           args=args,all_other_slots='original frozen S1 tensors and occlusion; exact order',
           hard_augmentation='same old affine on resized reflect100 hard canvas; canvas shape inherited from replaced clean slot; photometric HSV sampled at original base_seed; no extra rectangle',
           transformed_manual_points_outside=offscreen,hard_support='same manual-derived mask both arms; P8/automatic corners ignored',
           detection_normalization='original full-batch assigned-score denominator; hard detection numerators zero',
           checkpoint_rule='last',validation=False),immutable=True)


if __name__=='__main__':main()
