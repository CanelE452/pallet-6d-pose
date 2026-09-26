"""Frozen Replay inference and common D9 pose generation. No reference coordinates."""
from concurrent.futures import ProcessPoolExecutor
import copy
import subprocess
import numpy as np
from . import common as C

def pose_job(row):
    fid,pred,meta=row
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    return fid,D.Pose.infer(D.points(pred),np.array(meta['K']),np.array(meta['xyz']),False)

def main():
    if (C.DOC/'PREDICTIONS_LOCK.json').exists():
        for b in C.read(C.DOC/'PREDICTIONS_LOCK.json')['files']: C.verify(b)
        print('PREDICTIONS_ALREADY_FROZEN');return
    rr=C.records();ids={r['id'] for r in rr}
    registry=C.read(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')
    sessions=set(C.read(C.DOC/'CORE_COMPARABILITY_AUDIT.json')['teacher_sessions'])
    groups={g['recording_id']:[s['session_key'] for s in g['sessions'] if s['session_key'].split('/')[-1] in sessions] for g in registry['groups']}
    groups={g:ss for g,ss in groups.items() if ss}
    assert groups and not set(groups)&{r['recording_group'] for r in rr}
    C.save(C.DOC/'TEACHER_RECORDING_AUDIT.json',dict(teacher_groups=groups,heldout_groups=sorted({r['recording_group'] for r in rr}),overlap=[],registry=C.bind(C.ROOT/'data/pallet/results/site_environment_audit_v1/SOURCE_RECORDING_GROUPS.json')),True)
    preds={}
    for arm in C.ARMS:
        payload=C.read(C.prediction_path(arm));preds[arm]={r['id']:r['prediction'] for r in payload['records'] if r['id'] in ids}
        assert set(preds[arm])==ids
    # All pose-only students retain exactly the detector's candidate order, boxes and scores.
    for arm in C.ARMS[1:]:
        for fid in ids:
            a,b=preds['R0'][fid],preds[arm][fid]
            assert a['selected_index']==b['selected_index'] and len(a['candidates'])==len(b['candidates'])
            for x,y in zip(a['candidates'],b['candidates']):
                assert x['box_xyxy']==y['box_xyxy'] and x['score']==y['score']
    if (C.RAW/'TEACHER_PREDICTIONS.json').exists():
        preds['TEACHER']=C.read(C.RAW/'TEACHER_PREDICTIONS.json')
    else:
        import torch,cv2
        from scripts.research.pallet_posefix_replay_v1 import core as N
        N.setup();assert torch.cuda.is_available(), 'Host CUDA required; never silently fall back'
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        C.save(C.DOC/'GPU_PREFLIGHT.json',dict(utc=C.now(),cuda=True,device=torch.cuda.get_device_name(0),status=gpu))
        fit=C.read(N.DOC/'FIT.json');C.verify(fit['checkpoint'])
        ck=torch.load(C.ROOT/fit['checkpoint']['path'],map_location='cpu',weights_only=False)
        assert ck['step']==300 and ck['protocol_sha256']==C.sha(N.DOC/'PROTOCOL.json')
        model=N.C.PoseFixPallet9();model.load_state_dict(ck['model_state_dict']);model.cuda().eval().requires_grad_(False)
        preds['TEACHER']={}
        for i,r in enumerate(rr):
            if i%16==0:
                temp=int(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                assert temp<80, 'Thermal safety stop'
                print('FROZEN_REPLAY',i,len(rr),'GPU_C',temp,flush=True)
            im=cv2.imread(str(C.ROOT/r['image']['path']));assert im is not None
            preds['TEACHER'][r['id']]=N.C.predict(model,im,preds['R0'][r['id']])
        C.save(C.RAW/'TEACHER_PREDICTIONS.json',preds['TEACHER'],True)
        del model;torch.cuda.empty_cache()
    assert set(preds['TEACHER'])==ids
    C.save(C.RAW/'PREDICTIONS.json',preds,True)
    metadata={r['id']:r for r in C.read(C.META)};poses={}
    for arm in C.ARMS:
        with ProcessPoolExecutor(max_workers=4) as pool:
            poses[arm]=dict(pool.map(pose_job,[(i,preds[arm][i],metadata[i]) for i in sorted(ids)],chunksize=8))
        print('COMMON_D9',arm,flush=True)
    C.save(C.RAW/'POSE_PREDICTIONS.json',poses,True)
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    files=[C.RAW/'PREDICTIONS.json',C.RAW/'POSE_PREDICTIONS.json',C.RAW/'TEACHER_PREDICTIONS.json',C.META,Path(D.Pose.__file__),Path(D.Selector.__file__),Path(N.C.__file__) if 'N' in locals() else C.ROOT/'scripts/research/pallet_posefix_large_error_v1/core.py']
    C.save(C.DOC/'PREDICTIONS_LOCK.json',dict(created_at=C.now(),files=[C.bind(p) for p in files],reference_coordinates_read=False,new_training_steps=0,
        inference='Frozen original Replay, no cap; common unchanged R0 detection. No filter at evaluation. Pose uses original D9 with corner-only solve; no oracle or trained router.'),True)

from pathlib import Path
if __name__=='__main__':main()
