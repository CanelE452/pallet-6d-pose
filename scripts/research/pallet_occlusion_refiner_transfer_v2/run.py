"""User-selected DAY264 pilot; frozen teachers, isolated outputs, no promotion."""
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from collections import Counter

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_occlusion_refiner_transfer_v1 import run as V
from scripts.research.pallet_posefix_replay_v1 import core as N
from scripts.research.pallet_type_selftrain_v1 import pseudo as P
from scripts.research.pallet_cad8_selftrain_v1.run import gpu
from scripts.research.pallet_posefix_replay_v1.evaluate import assert_preserved
from scripts.research.pallet_dim_conditioned_p_v1 import pose

NAME='pallet_occlusion_refiner_transfer_v2'
DOC=ROOT/'_docs/experiments'/NAME
RAW=ROOT/'data/pallet/results'/NAME
OUT=ROOT/'outputs'/NAME
SESSION=ROOT/'data/evaluation/pallet_eval_v1/incoming/sessions/real_unlabeled_day_20260830'
read,bind,verify=V.read,V.bind,V.verify
ARMS=['A00','A01','A10','A11']


def freeze(path,value):
    path=Path(path).resolve();assert any(path.is_relative_to(p) for p in (DOC,RAW,OUT))
    text=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists():assert path.read_text()==text,('No overwrite',str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:f.write(text)


def tensor_save(path,value):
    assert Path(path).is_relative_to(RAW) and not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:torch.save(value,f)


def prepare():
    selected=[r for r in csv.DictReader((SESSION/'manifests/frame_review.csv').open()) if r['review_label']=='plastic'][:264]
    assert selected[0]['frame']=='005480.png' and selected[-1]['frame']=='005743.png'
    records=[dict(id='DAY264__'+Path(r['frame']).stem,image=bind(SESSION/'rgb'/r['frame']),session=str(SESSION.relative_to(ROOT)),
        display_index=i+1,object_type='plastic_standard_110x130x11',condition='USER_APPROXIMATE_CLEAN_NOT_PER_FRAME_VERIFIED') for i,r in enumerate(selected)]
    assert len({r['image']['sha256'] for r in records})==264
    groups=read(ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    aliases={str(SESSION.relative_to(ROOT))}
    edges=[{s['session_key'] for s in g['sessions']} for g in groups['groups'] if not g['is_collection']]
    edges += [{r['session_a'],r['session_b']} for r in groups['partial_overlap_pairs']]
    while True:
        n=len(aliases)
        for edge in edges:
            if aliases&edge:aliases|=edge
        if n==len(aliases):break
    oldroles=read(V.DOC/'DATA_ROLE_MANIFEST.json'); hashes={r['image']['sha256'] for r in records}
    evaluation=[r for r in oldroles['records'] if str(Path(r['image']['path']).parent.parent) not in aliases and r['image']['sha256'] not in hashes]
    excluded=[r['id'] for r in oldroles['records'] if r not in evaluation]
    teacher=oldroles['teacher_manual_train'];teacher_overlap=[r['id'] for r in teacher if str(Path(r['image']['path']).parent.parent) in aliases]
    identical_train=[r['id'] for r in teacher if r['image']['sha256'] in hashes]
    assert not identical_train,('User selection overlaps manual teacher targets',identical_train)
    allowed={r['id'] for r in evaluation};populations={}
    for name,p in oldroles['populations'].items():
        ids=[fid for fid in p['ids'] if fid in allowed]
        if ids:populations[name]=dict(ids=ids,frames=len(ids),original_frames=p['frames'],removed=p['frames']-len(ids))
    K=np.loadtxt(SESSION/'cam_K.txt').reshape(3,3)
    bindings=read(V.DOC/'INPUT_BINDINGS.json')['checkpoints']
    for b in bindings.values():verify(b)
    freeze(DOC/'DATA_ROLE_MANIFEST.json',dict(status='[확인]',train_candidates=records,eval_records=evaluation,populations=populations,
        source_recording='REC_001',excluded_recording_aliases=sorted(aliases),excluded_eval_ids=excluded,
        eval_image_hash_overlap=0,eval_recording_overlap=0,teacher_training_same_recording=teacher_overlap,teacher_identical_image_overlap=identical_train,
        clean_label='User-approved approximate interval; not independent per-frame clean verification',
        global_split_unchanged=True,checkpoint_bindings=bindings,K=K.tolist(),
        sources=[bind(SESSION/'manifests/frame_review.csv'),bind(SESSION/'cam_K.txt'),bind(V.DOC/'DATA_ROLE_MANIFEST.json'),
                 bind(ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')]))
    filt=read(P.FILTER)
    assert filt['TAU_BOX']==.85 and filt['keypoint_validity']['kp_conf_threshold']==.5
    assert filt['geometry_thresholds']['tau_remove']==filt['geometry_thresholds']['tau_flip']==.05
    freeze(DOC/'PSEUDO_PROTOCOL.json',dict(status='[확인]',teacher='Frozen current Replay + existing self-occlusion PnP',
        pipeline='R0 confidence .85 />=6 kp .5 -> raw flip & median LOO .05 -> Replay -> hidden-only PnP -> all8 finite median LOO .05',
        no_GT=True,no_new_filter=True,confidence='Original R0 confidence unchanged',
        bindings=[bind(P.FILTER),bindings['R0'],bindings['REPLAY'],bind(Path(__file__)),bind(Path(P.__file__)),bind(Path(V.S.__file__))]))
    print('PREPARED',len(records),'candidates',len(evaluation),'eval','excluded',len(excluded),populations.keys(),flush=True)


def raw_infer(extractor,image):
    torch.backends.cudnn.allow_tf32=True
    out=extractor.predict(image)
    return dict(candidates=P.serial(out['candidates']),selected_index=out['selected_index'])


def scores(candidate,K,flip=None,hw=(480,640)):
    q=np.array(candidate['keypoints_xy']);v=P.valid_points(candidate,confidence=flip is not None)
    kwargs={}
    if flip is not None:
        fq=np.array(flip['keypoints_xy']);fq[:,0]=hw[1]-1-fq[:,0];fq=fq[P.PERM]
        kwargs=dict(flipped_keypoints_xy=fq,flipped_valid=P.valid_points(flip)[P.PERM])
    # Positional arguments exactly match the previously verified filter adapter.
    try:
        s=P.geometry_scores(q,v,K,dict(x=1.1,y=.11,z=1.3),
            kwargs.get('flipped_keypoints_xy'),kwargs.get('flipped_valid'))
    except cv2.error:return dict(s_remove=None,s_flip=None)
    return {k:float(s[k]) if s.get(k) is not None and np.isfinite(s[k]) else None for k in ('s_remove','s_flip')}


def pseudo():
    N.setup();gpu();assert torch.cuda.is_available()
    role=read(DOC/'DATA_ROLE_MANIFEST.json');K=np.array(role['K']);protocol=bind(DOC/'PSEUDO_PROTOCOL.json')
    for b in read(DOC/'PSEUDO_PROTOCOL.json')['bindings']:verify(b)
    extractor=N.E.old('features').FrozenYoloFeatures(N.E.R0);model=N.load_model();rows=[]
    try:
        for i,r in enumerate(role['train_candidates']):
            path=RAW/'pseudo_frames'/f'{r["id"]}.json'
            if path.exists():
                result=read(path);assert result['protocol']==protocol
            else:
                verify(r['image']);image=cv2.imread(str(ROOT/r['image']['path']));assert image.shape[:2]==(480,640)
                raw=raw_infer(extractor,image);c=P.top(raw);reason='raw_confidence';flip=None;stage1=None;stage2=None;refined=None;pnp_info=None
                if c and c['score']>=.85 and P.valid_points(c)[:8].sum()>=6:
                    flip=raw_infer(extractor,cv2.flip(image,1));f=P.top(flip);reason='raw_flip_missing'
                    if f:
                        stage1=scores(c,K,f);reason='raw_flip_LOO'
                        if P.passed(stage1,['s_remove','s_flip']):
                            torch.backends.cudnn.allow_tf32=False
                            refined=N.C.predict(model,image,raw);assert_preserved(raw,refined)
                            cc=P.top(refined);initial=pose.infer(cc['keypoints_xy'],K,np.array([1.1,.11,1.3]))
                            q,pnp_info=V.S.correct(cc,initial,K);cc['keypoints_xy']=q.tolist();assert_preserved(raw,refined)
                            reason='refined_missing_corner'
                            if P.valid_points(cc,False)[:8].all():
                                stage2=scores(cc,K);reason='refined_all8_LOO'
                                if P.passed(stage2,['s_remove']):reason='accepted'
                result=dict(**r,raw=raw,flip=flip,refined=refined,stage1=stage1,stage2=stage2,pnp=pnp_info,
                    reason=reason,accepted=reason=='accepted',protocol=protocol,GT_input=False)
                freeze(path,result)
            rows.append(result)
            if (i+1)%40==0 or i==263:print('PSEUDO',i+1,dict(Counter(x['reason'] for x in rows)),gpu(),flush=True)
    finally:extractor.close();del model;torch.cuda.empty_cache()
    accepted=[r for r in rows if r['accepted']]
    freeze(RAW/'PSEUDOLABEL_MANIFEST.json',accepted)
    freeze(DOC/'E1_PSEUDOLABEL_POOL.json',dict(status='[확인]',candidates=len(rows),accepted=len(accepted),unique_accepted=len({r['image']['sha256'] for r in accepted}),
        reasons=dict(Counter(r['reason'] for r in rows)),GT_correctness_unknown=True,recording_count=1,
        first_display=min((r['display_index'] for r in accepted),default=None),last_display=max((r['display_index'] for r in accepted),default=None),
        pool_gate=len(accepted)>=8))
    print('PSEUDO_DONE',len(accepted),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pseudo']);args=p.parse_args()
    globals()[args.stage]()
