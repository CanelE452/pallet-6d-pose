"""R0 -> confidence/flip/LOO -> frozen Replay -> all8 LOO -> unchanged labels."""
import copy
import json
import time
import cv2
import numpy as np
import torch
from . import common as C
from scripts.research.pallet_posefix_utility_selector_v1.augmentation_stability import serial
from scripts.self_training_yolo.pseudo_label_filters import geometry_scores

PERM=[1,0,3,2,5,4,7,6,8]
FILTER=C.ROOT/'data/evaluation/pallet_eval_v1/adaptation/PSEUDOLABEL_FILTER_LOCK.json'


def top(pred):
    return None if pred['selected_index'] is None else pred['candidates'][pred['selected_index']]


def valid_points(candidate, confidence=True):
    q=np.asarray(candidate['keypoints_xy'],float)
    valid=np.isfinite(q).all(1)&~(q==-1).all(1)
    if confidence:valid &= np.asarray(candidate['keypoints_conf'])>=.5
    return valid


def passed(scores, keys):
    return all(scores.get(k) is not None and np.isfinite(scores[k]) and scores[k]<=.05 for k in keys)


def run():
    C.N.setup();torch.set_num_interop_threads(1);assert torch.cuda.is_available()
    C.N.E.gpu();pool=C.read(C.DOC/'POOL.json')
    for b in pool['sources']:C.verify(b)
    filt=C.read(FILTER)
    assert filt['TAU_BOX']==.85 and filt['keypoint_validity']['kp_conf_threshold']==.5 and filt['keypoint_validity']['min_valid_corners']==6
    assert filt['geometry_thresholds']['tau_remove']==filt['geometry_thresholds']['tau_flip']==.05
    registry_path=C.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'
    registry={r['object_type']:r['physical_dimensions_m'] for r in C.read(registry_path)['objects']}
    fit=C.read(C.N.DOC/'FIT.json')
    protocol=dict(order='R0 top confidence>=.85 and >=6 confident corners -> original raw flip+LOO<=.05 -> frozen Replay on survivors only -> all8 finite+median LOO<=.05',
        second_stage='No additional flip after Replay; all8 LOO includes low-confidence corners',
        no_filters=['augmentation stability','physical shape','X crossings','extra reprojection threshold'],
        inputs='Images, session cameraK and type registry only; no real GT coordinates/pose',
        output='Original R0 boxes/scores/confidence, Replay coordinates unchanged after stage2. Whole-image keep/drop, never N2 fallback.',
        student='separate R0 copy per kind; no pseudo-label refresh; no refiner training',
        precision='R0 cuDNN TF32=True; Replay=False; matmul=False',
        sources=[C.bound(C.DOC/'POOL.json'),C.bound(FILTER),C.bound(registry_path),C.bound(C.N.E.R0),fit['checkpoint'],
                 C.bound(C.N.__file__),C.bound(C.N.C.__file__),C.bound(C.ROOT/'scripts/self_training_yolo/pseudo_label_filters.py'),
                 C.bound(C.ROOT/'scripts/research/pallet_posefix_utility_selector_v1/augmentation_stability.py')],
        code=C.bound(__file__))
    C.freeze(C.DOC/'PSEUDO_PROTOCOL.json',protocol)
    extractor=C.N.E.old('features').FrozenYoloFeatures(C.N.E.R0);model=C.N.load_model()
    decisions=[];accepted=[];start=time.monotonic()
    def infer(image):
        torch.backends.cudnn.allow_tf32=True
        cap=extractor.predict(image)
        return dict(candidates=serial(cap['candidates']),selected_index=cap['selected_index'])
    def scores(q,v,K,d,fq=None,fv=None):
        try:s=geometry_scores(q,v,K,d,fq,fv)
        except cv2.error:return dict(s_remove=None,s_flip=None)
        return {k:float(s[k]) if s.get(k) is not None and np.isfinite(s[k]) else None for k in ['s_remove','s_flip']}
    try:
        for i,row in enumerate(pool['records']):
            dest=C.RAW/'pseudo_frames'/(row['id']+'.json')
            if dest.exists():
                result=C.read(dest);assert result['protocol_sha256']==C.sha(C.DOC/'PSEUDO_PROTOCOL.json')
            else:
                C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']));assert image is not None
                raw=infer(image);candidate=top(raw);reason='raw_confidence';s1=None;s2=None;refined=None
                if candidate is not None and candidate['score']>=.85 and valid_points(candidate)[:8].sum()>=6:
                    flip=top(infer(cv2.flip(image,1)));reason='raw_flip_missing'
                    if flip is not None:
                        fq=np.array(flip['keypoints_xy']);fq[:,0]=image.shape[1]-1-fq[:,0];fq=fq[PERM]
                        fv=valid_points(flip)[PERM];q=np.array(candidate['keypoints_xy']);v=valid_points(candidate)
                        s1=scores(q,v,np.array(row['K']),registry[row['object_type']],fq,fv);reason='raw_flip_LOO'
                        if passed(s1,['s_remove','s_flip']):
                            torch.backends.cudnn.allow_tf32=False
                            refined=C.N.C.predict(model,image,raw)
                            r=top(refined);v=valid_points(r,False);reason='refined_missing_corner'
                            if v[:8].all():
                                s2=scores(np.array(r['keypoints_xy']),v,np.array(row['K']),registry[row['object_type']]);reason='refined_all8_LOO'
                                if passed(s2,['s_remove']):reason='accepted'
                result=dict(**row,raw=raw,refined=refined,stage1=s1,stage2=s2,reason=reason,accepted=reason=='accepted',
                            raw_hw=list(image.shape[:2]),protocol_sha256=C.sha(C.DOC/'PSEUDO_PROTOCOL.json'))
                C.freeze(dest,result)
            decisions.append(dict(id=row['id'],kind=row['kind'],reason=result['reason'],accepted=result['accepted']))
            if result['accepted']:accepted.append(copy.deepcopy(result))
            if (i+1)%100==0 or i+1==len(pool['records']):
                from collections import Counter
                print(json.dumps(dict(done=i+1,total=len(pool['records']),accepted=dict(Counter(r['kind'] for r in accepted)),
                    seconds=time.monotonic()-start,gpu=C.N.E.gpu())),flush=True)
        for b in protocol['sources']+[protocol['code']]:C.verify(b)
        C.freeze(C.RAW/'PSEUDO_ACCEPTED.json',accepted);C.freeze(C.RAW/'PSEUDO_DECISIONS.json',decisions)
        from collections import Counter
        C.freeze(C.DOC/'PSEUDO_COMPLETE.json',dict(complete=True,counts=dict(Counter(r['kind'] for r in accepted)),
            reasons={kind:dict(Counter(r['reason'] for r in decisions if r['kind']==kind)) for kind in C.TYPES},
            artifacts=[C.bound(C.RAW/p) for p in ['PSEUDO_ACCEPTED.json','PSEUDO_DECISIONS.json']],protocol=C.bound(C.DOC/'PSEUDO_PROTOCOL.json')))
    finally:extractor.close()


if __name__=='__main__':run()
