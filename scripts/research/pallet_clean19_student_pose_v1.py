"""Frozen300 pose evaluation using existing canonical proper-symmetry contract."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from scripts.research import pallet_clean19_student_v1 as S


def main():
    cv2.setNumThreads(1);S.scope()
    doc=S.DOC/'pose';raw=S.RAW/'pose'
    assert not doc.exists() and not raw.exists()
    records=S.P.read(S.P.DOC/'SPLIT.json')['evaluation']
    ids={r['id'] for r in records}
    source=S.P.read(S.V.RAW/'PREDICTIONS.json')['predictions']
    predictions={'R0':source['R0_DIRECT'],'CORRECTED_TEACHER':source['TYPE_REPLAY_PIPELINE'],'STUDENT':{}}
    files=[S.C.bound(S.V.RAW/'PREDICTIONS.json'),S.C.bound(S.V.RAW/'INFERENCE_METADATA.json'),S.C.bound(Path(__file__)),S.C.bound(Path(S.V.Pose.__file__))]
    for a in ('PLASTIC','WOOD'):
        path=S.RAW/f'EVAL_PREDICTIONS_{a}.json';files.append(S.C.bound(path))
        predictions['STUDENT'].update({r['id']:r['prediction'] for r in S.P.read(path)['records']})
    assert all(set(p)==ids for p in predictions.values())
    inputs={r['id']:r for r in S.P.read(S.V.RAW/'INFERENCE_METADATA.json')}
    S.save(doc/'PROTOCOL.json',dict(frames=300,inputs=files,
        inference='Each final2D output -> same prediction-only W/D selector + SQPnP/LM; no GT detection gate; no new training',
        metrics='Existing dimension-conditioned proper-group pose.metric; per-object normalized ADDsym AUC to0.1 diameter, failures count0; conditional medians',
        caveat='Geometry-reconstructed annotation reference, not independent physical6D GT. Canonical axis mapping and predicted extents for IoU; historical CF-only table is not directly identical.',
        auto_promote=False))
    poses={}
    for arm,rows in predictions.items():
        poses[arm]={}
        for i,r in enumerate(records):
            fid=r['id'];meta=inputs[fid];c=S.P.C.selected(rows[fid])
            q=None if c is None else np.array(c['keypoints_xy'],float)
            if q is not None:q[(q==-1).all(1)]=np.nan
            poses[arm][fid]=S.V.Pose.infer(q,np.array(meta['K']),np.array(meta['xyz']),False)
            if (i+1)%100==0:print('POSE',arm,i+1,flush=True)
    S.save(raw/'PREDICTIONS.json',poses)
    S.save(doc/'PREDICTION_LOCK.json',dict(predictions=S.C.bound(raw/'PREDICTIONS.json'),before_scoring=True))
    _,truth=S.V.Pose.metadata('REAL_DEV')
    metrics={}
    for arm,rows in poses.items():
        with ProcessPoolExecutor(max_workers=4) as pool:
            mm=list(pool.map(S.V.Pose.metric,[(fid,p,truth[fid]) for fid,p in rows.items()],chunksize=16))
        metrics[arm]={m['id']:m for m in mm}
        for m in mm:
            if m['available']:
                m['axis_correct']=bool(abs(rows[m['id']]['cf_extents'][0]-truth[m['id']]['body_xyz'][0])<1e-6)
    groups={'ALL300':records,**{k:[r for r in records if r['object_type']==k] for k in ('plastic','wood')},**{k:[r for r in records if r['severity']==k] for k in S.P.SEVERITIES}}
    results={};lines=['# Clean19 학생: 동일300장6D pose 평가','',
        'R0·종류별Replay+자기 가림PnP 보정·종류별학생 단독의 최종2D예측에서 동일 prediction-only W/D selector + SQPnP/LM으로 pose를 계산했다. 평가GT로 축을 선택하거나 검출매칭으로 프레임을 제거하지 않았다.', '',
        '기존 DIM 조건부 평가기의 canonical 물리축/C2 대칭 규약 사용. IoU3D는 예측·정답 각각의 extents를 사용한다. ADDsym AUC는 각 팔레트 대각선으로 정규화한 오차를 0~0.1에서 적분(1001점); 성공률이 아니라0~1 AUC다. 실패는 AUC에0, 오차중앙값은 pose 성공건 기준. 이전 CF-only 방식 표와 수치를 그대로 혼합하지 않는다.', '',
        '**참조6D는 어노테이션에서 기하학적으로 복원한 값이며 독립 장비로 측정한 실측6D 정답이 아니다.** 같은세션 재사용 평가·단일seed 결과다.', '']
    for group,rr in groups.items():
        results[group]={};lines += ['## '+group,'','| 모델 | PoseCov↑ | AxisAcc↑ | R med°↓ | Yaw med°↓ | t med cm↓ | IoU3D med↑ | ADDsym AUC↑ |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in poses:
            rows=[metrics[arm][r['id']] for r in rr];ok=[r for r in rows if r['available']]
            result=dict(frames=len(rows),available=len(ok),coverage=len(ok)/len(rows),axis_accuracy=float(np.mean([r['axis_correct'] for r in ok])),
                ADDsym_AUC_full=S.V.Pose.pose_auc([r['ADDsym_normalized'] if r['available'] else float('inf') for r in rows],1.))
            for key in ('rotation_deg','yaw_deg','translation_cm','IoU3D'):
                result[key]=dict(median=float(np.median([r[key] for r in ok])),P90=float(np.quantile([r[key] for r in ok],.9)))
            results[group][arm]=result
            lines.append(f'| {arm} | {100*result["coverage"]:.2f}% | {100*result["axis_accuracy"]:.2f}% | {result["rotation_deg"]["median"]:.3f} | {result["yaw_deg"]["median"]:.3f} | {result["translation_cm"]["median"]:.3f} | {result["IoU3D"]["median"]:.4f} | {result["ADDsym_AUC_full"]:.4f} |')
        lines.append('')
    for b in files:S.C.verify(b)
    S.save(raw/'FRAME_METRICS.json',metrics);S.save(doc/'RESULTS.json',results)
    S.save(doc/'RESULTS_KO.md','\n'.join(lines)+'\n')
    S.save(doc/'AUDIT.json',dict(complete=True,frames=300,arms=3,predictions_locked_before_scoring=True,input_hashes_verified=True,
        reference=S.C.bound(S.V.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json')))
    print('ALL300',results['ALL300'],flush=True)


if __name__=='__main__':main()
