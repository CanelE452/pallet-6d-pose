"""Clean2×2 평가: 모든 RGB/D9 예측 lock 뒤 참조를 여는 별도 scoring 경로.

infer만 GPU를 사용한다. oracle-freeze/oracle-score는 기존 D9 후보의 CPU
진단이며 배포 결과·선택기를 바꾸지 않는다. 모든 쓰기는 새 namespace에 한정한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

from . import common as C
from scripts.research.pallet_pose_objective_followup_v2 import metric_baseline as M

ARMS = ('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR','CLEAN_RAW_OCC','CLEAN_REF_OCC')
BASELINES = {'R0':'R0','OLD_REF':'REF_LR5'}
PAIRS = (
    ('CLEAN_REF_CLEAR','CLEAN_REF_OCC'),
    ('CLEAN_RAW_CLEAR','CLEAN_RAW_OCC'),
    ('CLEAN_RAW_CLEAR','CLEAN_REF_CLEAR'),
    ('CLEAN_RAW_OCC','CLEAN_REF_OCC'),
    ('OLD_REF','CLEAN_REF_OCC'),
    ('R0','CLEAN_REF_OCC'),
)
PRIMARY = 'NATURAL99'


def save(path, value, freeze=True):
    C.save(path, M.clean(value), freeze)


def paths(seed):
    assert isinstance(seed,int) and seed >= 0
    raw = C.RAW/'evaluation'/f'S{seed}'
    return dict(raw=raw, metadata=raw/'METADATA.json', predictions=raw/'PREDICTIONS.json',
        poses=raw/'POSES.json', lock=raw/'PREDICTIONS_LOCK.json',
        results=C.DOC/f'EVAL_RESULTS_S{seed}.json',
        candidate_lock=raw/'CANDIDATES_LOCK.json', candidates=raw/'CANDIDATES.json',
        oracle_result=C.DOC/f'CANDIDATE_ORACLE_S{seed}.json')


def validate_membership(rows):
    """추론 metadata 화이트리스트로 reference 좌표 유입을 차단한다."""
    allowed = {'id','recording','recording_group','severity','image','K','xyz','hw','session','object_type'}
    assert len(rows) == len({r['id'] for r in rows}) == 128
    for row in rows:
        assert set(row) <= allowed, '추론 metadata에 허용하지 않은 참조 필드'
        assert {'id','recording','severity','image','K','xyz','hw'} <= set(row)
        assert np.asarray(row['K']).shape == (3,3) and np.isfinite(row['K']).all()
        assert np.asarray(row['xyz']).shape == (3,) and np.all(np.asarray(row['xyz']) > 0)
        assert len(row['hw']) == 2 and min(row['hw']) > 0
        assert row.get('recording_group',row['recording']) == row['recording']
    assert Counter(r['severity'] for r in rows) == dict(CLEAN=29,MODERATE_OCCLUSION=21,SEVERE_OCCLUSION=78)
    return [r['id'] for r in rows]


def group_ids(rows):
    ids = validate_membership(rows)
    groups = {'FULL128':ids,
        'CLEAN29':[r['id'] for r in rows if r['severity']=='CLEAN'],
        'MOD21':[r['id'] for r in rows if r['severity']=='MODERATE_OCCLUSION'],
        'SEV78':[r['id'] for r in rows if r['severity']=='SEVERE_OCCLUSION'],
        PRIMARY:[r['id'] for r in rows if r['severity']!='CLEAN']}
    groups.update({'recording:'+rec:[r['id'] for r in rows if r['recording']==rec] for rec in sorted({r['recording'] for r in rows})})
    return groups


def fits_for(seed):
    """학습 완료/320update/동결 상태의 공통 계약을 먼저 확인한다."""
    protocol_path = C.DOC/'PRIMARY_PROTOCOL.json'
    protocol = C.read(protocol_path)
    assert protocol.get('locked_before_fit') is True
    assert seed in protocol['seeds']
    C.verify(protocol['initialization'])
    parity_path = C.DOC/f'PAIR_INTEGRITY_S{seed}.json'
    assert C.read(parity_path)['passed'] is True
    if 'arms' in protocol:
        assert set(protocol['arms']) == set(ARMS)
    fits = {}
    for arm in ARMS:
        path = C.DOC/f'FIT_{arm}_S{seed}.json'
        fit = C.read(path)
        assert fit['complete'] and fit.get('optimizer_steps',fit.get('steps')) == 320
        assert fit.get('protected_state_exact',fit.get('frozen_state_exact')) is True
        assert fit.get('exact_R0_initialization') is True
        assert fit['initialization'] == protocol['initialization']
        for key in ('checkpoint','protocol','initialization','trace','results_csv'):
            if isinstance(fit.get(key),dict) and 'path' in fit[key]:
                C.verify(fit[key])
        assert fit['checkpoint']['path'] == str((C.RAW/f'runs/{arm}_S{seed}/weights/last.pt').relative_to(C.ROOT))
        fits[arm] = fit
    if all('initialization' in fit for fit in fits.values()):
        assert all(fit['initialization'] == fits[ARMS[0]]['initialization'] for fit in fits.values())
    return protocol,fits


def assert_native_parity(rows, reference, values):
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    assert list(values) == [r['id'] for r in rows]
    for row in rows:
        C.verify(row['image'])
        assert_detector_parity(reference[row['id']], values[row['id']])


def prediction_lock(seed):
    p = paths(seed)
    lock = C.read(p['lock'])
    assert lock['all_four_arms_locked'] and lock['no_evaluation_reference_coordinates_read'] and lock['original_D9']
    for binding in lock['files'] + lock['sources'] + list(lock['checkpoints'].values()):
        C.verify(binding)
    rows = C.read(p['metadata'])
    ids = validate_membership(rows)
    predictions, poses = C.read(p['predictions']), C.read(p['poses'])
    assert set(predictions) == set(poses) == set(ARMS) | set(BASELINES)
    for arm in predictions:
        assert list(predictions[arm]) == ids and set(poses[arm]) == set(ids)
        if arm in ARMS:
            assert_native_parity(rows,predictions['R0'],predictions[arm])
    return lock


def infer(seed=42):
    """참조GT 없이 RGB→고정 학생→기존D9 출력을 네팔 모두 먼저 잠근다."""
    p = paths(seed)
    if p['lock'].exists():
        prediction_lock(seed)
        print('CLEAN_FOUR_ARM_INFERENCE_ALREADY_LOCKED',seed,flush=True)
        return
    _,fits = fits_for(seed)
    from ultralytics import YOLO
    import cv2
    import torch
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import thermal_guard
    from scripts.research.pallet_oracle_mechanism_followup_v1 import cycle_affine_eval as E, pose_oracle as O
    assert torch.cuda.is_available(), 'Host CUDA 필요: 자동CPU신경망 추론은 하지 않음'
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    start=time.monotonic()
    rows,old_predictions,old_poses=E.metadata('PLASTIC')
    ids=validate_membership(rows)
    predictions={alias:old_predictions[original] for alias,original in BASELINES.items()}
    poses={alias:old_poses[original] for alias,original in BASELINES.items()}
    save(p['metadata'],rows)
    files=[p['metadata']]
    runtime={}
    try:
        for arm in ARMS:
            fit=fits[arm]
            cache=p['raw']/f'EVAL_{arm}.json'
            seal=p['raw']/f'EVAL_{arm}_LOCK.json'
            stamp=dict(checkpoint=fit['checkpoint'],metadata=C.bind(p['metadata']),
                implementation=C.bind(Path(__file__)), predictor=C.bind(Path(__import__(predict.__module__,fromlist=['']).__file__)))
            if cache.exists():
                assert seal.exists(), '미봉인 캐시는 자동 재사용하지 않음'
                C.verify(C.read(seal)['file'])
                stored=C.read(cache)
                assert stored['input_stamp']==stamp and stored['GT_input'] is False
            else:
                assert not seal.exists()
                begin=time.monotonic()
                thermal_guard()
                model=YOLO(str(C.ROOT/fit['checkpoint']['path']),task='pose')
                values={}
                for index,row in enumerate(rows):
                    if index%32==0:
                        thermal_guard()
                        print('CLEAN_INFER',arm,seed,index,len(rows),flush=True)
                    C.verify(row['image'])
                    image=cv2.imread(str(C.ROOT/row['image']['path']))
                    assert image is not None and list(image.shape[:2])==row['hw']
                    values[row['id']]=predict(model,image)
                assert_native_parity(rows,old_predictions['R0'],values)
                stored=dict(input_stamp=stamp,predictions=values,GT_input=False,seconds=time.monotonic()-begin,at=C.now())
                save(cache,stored)
                save(seal,dict(file=C.bind(cache)))
                del model
                torch.cuda.empty_cache()
            values=stored['predictions']
            assert_native_parity(rows,old_predictions['R0'],values)
            predictions[arm]=values
            runtime[arm]=stored['seconds']
            poses[arm]={row['id']:O.D.Pose.infer(O.D.points(values[row['id']]),np.asarray(row['K']),np.asarray(row['xyz']),False) for row in rows}
            assert set(poses[arm]) == set(ids)
            files.extend([cache,seal])
        for path,value in [(p['predictions'],predictions),(p['poses'],poses)]:
            save(path,value)
            files.append(path)
        sources=[C.DOC/'PRIMARY_PROTOCOL.json',C.DOC/f'PAIR_INTEGRITY_S{seed}.json',Path(__file__),Path(E.__file__),Path(O.__file__),Path(O.D.__file__),Path(O.D.Pose.__file__),
            C.ROOT/'challenge/evaluation_v2/pnp_selector.py',O.P.RAW/'PREDICTIONS.json',O.P.RAW/'POSE_PREDICTIONS.json',
            O.P.DOC/'PREDICTIONS_LOCK.json',C.ROOT/'_docs/experiments/pallet_pose_objective_followup_v2/METRIC_AND_SELECTION_LOCK.json']
        sources.extend(C.DOC/f'FIT_{arm}_S{seed}.json' for arm in ARMS)
        # 사전/사후parity 산출물이 있으면 모두 출처를 고정하되 평가코드가 학습검사를 대체하지 않는다.
        sources.extend(path for path in [C.DOC/'PRIMARY_PREFLIGHT.json',C.DOC/f'PRIMARY_PARITY_S{seed}.json'] if path.exists())
        save(p['lock'],dict(created_at=C.now(),seed=seed,all_four_arms_locked=True,
            no_evaluation_reference_coordinates_read=True,original_D9=True,detector_parity=True,
            files=[C.bind(path) for path in files],sources=[C.bind(path) for path in sources],
            checkpoints={arm:fit['checkpoint'] for arm,fit in fits.items()},
            seconds=time.monotonic()-start,per_arm_inference_seconds=runtime,
            new_fits=0,optimizer_updates=0,reference_use='RGB/등록치수/K만 사용. 평가xy/6D참조 사용없음.'))
        prediction_lock(seed)
        print('CLEAN_ALL_FOUR_NATIVE_D9_LOCKED',seed,flush=True)
    except BaseException as error:
        event=p['raw']/'INFERENCE_ATTEMPTS.json'
        events=C.read(event) if event.exists() else []
        events.append(dict(at=C.now(),error=repr(error),status='FAILED_PRESERVED',completed_arm_caches=list(runtime),seconds=time.monotonic()-start))
        save(event,events,False)
        raise


def pose_job(task):
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    fid,pose,truth=task
    return fid,M.extend_metric(O.D.metric(fid,pose,truth),pose,truth)


def aggregate(rows,frames,fixed,metrics):
    from scripts.research.pallet_oracle_mechanism_followup_v1 import cycle_affine_eval as E
    groups=group_ids(rows)
    result={}
    for group,ids in groups.items():
        result[group]={}
        for arm in metrics:
            value=M.summarize(metrics[arm][fid] for fid in ids)
            value.update(twoD=E.summary2([frames[arm][fid] for fid in ids]),
                fixed_ID=E.summary2([fixed[arm][fid] for fid in ids]))
            result[group][arm]=value
    return groups,result


def classify(groups):
    primary=groups[PRIMARY]
    return dict(population=PRIMARY,
        corrected_occlusion_value=M.classify_candidate(primary['CLEAN_REF_OCC'],primary['CLEAN_REF_CLEAR']),
        raw_occlusion_value=M.classify_candidate(primary['CLEAN_RAW_OCC'],primary['CLEAN_RAW_CLEAR']),
        clear_coordinate_correction_value=M.classify_candidate(primary['CLEAN_REF_CLEAR'],primary['CLEAN_RAW_CLEAR']),
        occluded_coordinate_correction_value=M.classify_candidate(primary['CLEAN_REF_OCC'],primary['CLEAN_RAW_OCC']),
        versus_OLD_REF=M.classify_candidate(primary['CLEAN_REF_OCC'],primary['OLD_REF']),
        versus_R0=M.classify_candidate(primary['CLEAN_REF_OCC'],primary['R0']),
        significance='부호 기반 기술적 판정이며 통계적/실무적 유의성·독립 검증을 자동 주장하지 않음')


def score(seed=42):
    p=paths(seed)
    lock=prediction_lock(seed)
    if p['results'].exists():
        result=C.read(p['results'])
        for binding in result['scoring_sources']+result['private_artifacts']:
            C.verify(binding)
        print('CLEAN_FOUR_ARM_SCORE_ALREADY_COMPLETE',seed,flush=True)
        return
    _,fits=fits_for(seed)
    start=time.monotonic()
    start_path=p['raw']/'SCORING_START.json'
    if start_path.exists():
        C.verify(C.read(start_path)['prediction_lock'])
    else:
        save(start_path,dict(at=C.now(),prediction_lock=C.bind(p['lock'])))
    # 평가 참조좌표를 여는 첫 경로: 위 네 학생 raw/D9 hash lock 검증 후에만 진입한다.
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O, cycle_affine_eval as E
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    truth=C.read(O.P.TRUTH)
    _,pose_truth=O.D.Pose.metadata('REAL_DEV')
    rows,predictions,poses=[C.read(p[key]) for key in ('metadata','predictions','poses')]
    ids=validate_membership(rows)
    oldframes=C.read(O.P.RAW/'FRAME_METRICS.json')
    oldfixed=C.read(O.P.RAW/'FIXED_ID_METRICS.json')
    oldmetrics=C.read(O.P.RAW/'POSE_METRICS.json')
    frames={a:oldframes[b] for a,b in BASELINES.items()}
    fixed={a:oldfixed[b] for a,b in BASELINES.items()}
    metrics={a:{fid:M.extend_metric(oldmetrics[b][fid],poses[a][fid],pose_truth[fid]) for fid in ids} for a,b in BASELINES.items()}
    for arm in ARMS:
        frames[arm]={fid:score_frame(fid,predictions[arm][fid],truth[fid]) for fid in ids}
        fixed[arm]={fid:score_frame(fid,predictions[arm][fid],truth[fid],fixed=True) for fid in ids}
        with ProcessPoolExecutor(max_workers=4) as pool:
            metrics[arm]=dict(pool.map(pose_job,[(fid,poses[arm][fid],pose_truth[fid]) for fid in ids],chunksize=8))
        print('CLEAN_SCORED',arm,len(metrics[arm]),flush=True)
    assert all(set(value)==set(ids) for collection in (frames,fixed,metrics) for value in collection.values())
    groups,summaries=aggregate(rows,frames,fixed,metrics)
    for arm,value in summaries['FULL128'].items():
        assert value['frames']==128
        assert value['twoD']['corners']==value['fixed_ID']['corners']==985
        assert value['twoD']['detected']==128 and value['twoD']['matched']==120,arm
    contrasts={group:{after+'-minus-'+before:dict(**M.paired(metrics[before],metrics[after],members),
        auxiliary_2D_and_ADD=E.contrast(frames[before],frames[after],metrics[before],metrics[after],members))
        for before,after in PAIRS} for group,members in groups.items()}
    private_pairs={group:{after+'-minus-'+before:[dict(id=fid,
        before_valid=metrics[before][fid]['available'],after_valid=metrics[after][fid]['available'],
        delta={k:metrics[after][fid][k]-metrics[before][fid][k] for k in M.KEYS}
            if metrics[before][fid]['available'] and metrics[after][fid]['available'] else None)
        for fid in members] for before,after in PAIRS} for group,members in groups.items()}
    loro={rec:{after+'-minus-'+before:M.paired(metrics[before],metrics[after],
        [r['id'] for r in rows if r['recording']!=rec and r['id'] in groups[PRIMARY]])
        for before,after in PAIRS} for rec in sorted({r['recording'] for r in rows})}
    artifacts=[]
    for name,value in [('FRAME_METRICS',frames),('FIXED_ID_METRICS',fixed),('POSE_METRICS',metrics),('PAIRED_FRAME_DELTAS',private_pairs)]:
        path=p['raw']/(name+'.json')
        save(path,value)
        artifacts.append(path)
    sources=[p['lock'],C.DOC/'PRIMARY_PROTOCOL.json',O.P.TRUTH,
        O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',O.D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json',
        O.P.RAW/'FRAME_METRICS.json',O.P.RAW/'FIXED_ID_METRICS.json',O.P.RAW/'POSE_METRICS.json',
        Path(__file__),Path(M.__file__),Path(E.__file__),Path(O.D.__file__),Path(O.D.Pose.__file__),
        C.ROOT/'scripts/research/pallet_material_selftrain_closure_v1/score_eval.py']
    result=dict(created_at=C.now(),seed=seed,groups=summaries,contrasts=contrasts,
        classification=classify(summaries),leave_one_recording_out=loro,LORO_population=PRIMARY,
        population='Plastic 동일128: CLEAN29/MODERATE21/SEVERE78; primary 자연99',
        primary='centroid T median/P90 cm + C2 full R median/P90 deg + valid99; 난도별median평균금지',
        reference='기존기하재구성6D/legacyxy; 반복DEV, 독립physical6D가아님',
        camera_axes='OpenCV camera x/z; 차량좌표아님',
        interpretation='과거Clean19와현재새4팔은teacher/학습계약차이를보존; 새2×2내에서만통제효과판단',
        prediction_lock=C.bind(p['lock']),scoring_sources=[C.bind(path) for path in dict.fromkeys(sources)],
        private_artifacts=[C.bind(path) for path in artifacts],
        fits={arm:{key:fit[key] for key in ('checkpoint','optimizer_steps','steps','seconds','seed') if key in fit} for arm,fit in fits.items()},
        added_deployment_components='없음: RGB학생과기존D9; teacher사용안함',
        seconds=time.monotonic()-start,CPU_score_wall_seconds=time.monotonic()-start,GPU_score_seconds=0,
        inference_seconds=lock['seconds'],new_scoring_fits=0,new_scoring_optimizer_updates=0,
        candidate_oracle_status='별도oracle-freeze/oracle-score전까지미실행; oldcandidateoracle값대체금지')
    save(p['results'],result)
    print('CLEAN_FOUR_ARM_EVAL_COMPLETE',seed,result['classification'],flush=True)


def oracle_freeze(seed=42):
    """동일 frozen 좌표의 기존 두W/D후보를 GT 없이 CPU로 저장한다."""
    p=paths(seed)
    prediction_lock(seed)
    if p['candidate_lock'].exists():
        for binding in C.read(p['candidate_lock'])['files']+C.read(p['candidate_lock'])['sources']:
            C.verify(binding)
        return
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    rows,predictions,poses=[C.read(p[key]) for key in ('metadata','predictions','poses')]
    start=time.monotonic()
    candidates={arm:{} for arm in predictions}
    for arm in candidates:
        for row in rows:
            fid=row['id']
            record=O.candidate_record(predictions[arm][fid],row)
            O.D.close(record['current'],poses[arm][fid])
            candidates[arm][fid]=record
        print('CLEAN_CANDIDATES_FROZEN',arm,len(candidates[arm]),flush=True)
    save(p['candidates'],candidates)
    sources=[p['lock'],Path(__file__),Path(O.__file__),Path(O.D.__file__),Path(O.D.Pose.__file__)]
    save(p['candidate_lock'],dict(created_at=C.now(),files=[C.bind(p['candidates'])],
        sources=[C.bind(path) for path in sources],no_reference_coordinates_read=True,
        unchanged_current_D9_verified=len(rows)*len(candidates),new_candidate_policy=False,
        new_fits=0,optimizer_updates=0,GPU_seconds=0,seconds=time.monotonic()-start))


def candidate_job(task):
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    fid,record,truth=task
    return fid,[dict(name=h['name'],metric=M.extend_metric(O.D.metric(fid,h['pose'],truth),h['pose'],truth)) for h in record['hypotheses']]


def same_candidate_joint_count(ids, choices, baseline):
    """단일한 한 후보가 T/R 모두 낮아야 한다. 서로 다른oracle최솟값을 조합하지 않음."""
    return sum(baseline[fid]['available'] and any(h['metric']['available']
        and h['metric']['translation_cm'] < baseline[fid]['translation_cm']
        and h['metric']['rotation_deg'] < baseline[fid]['rotation_deg']
        for h in choices[fid]) for fid in ids)


def oracle_score(seed=42):
    p=paths(seed)
    prediction_lock(seed)
    candidate_lock=C.read(p['candidate_lock'])
    for binding in candidate_lock['files']+candidate_lock['sources']:
        C.verify(binding)
    assert candidate_lock['no_reference_coordinates_read']
    if p['oracle_result'].exists():
        existing=C.read(p['oracle_result'])
        for binding in existing['sources']+existing['private_artifacts']:
            C.verify(binding)
        return
    assert p['results'].exists(), '실제D9 scoring을먼저완료할것'
    for binding in C.read(p['results'])['private_artifacts']:
        C.verify(binding)
    from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
    start=time.monotonic()
    _,truth=O.D.Pose.metadata('REAL_DEV')
    rows=C.read(p['metadata'])
    groups=group_ids(rows)
    candidates=C.read(p['candidates'])
    current=C.read(p['raw']/'POSE_METRICS.json')
    choices={}
    for arm,values in candidates.items():
        with ProcessPoolExecutor(max_workers=4) as pool:
            choices[arm]=dict(pool.map(candidate_job,[(fid,record,truth[fid]) for fid,record in values.items()],chunksize=8))
        for fid,record in values.items():
            selected=next((h['metric'] for h in choices[arm][fid] if h['name']==record['selected_name']),None)
            if selected is not None:
                O.D.close(selected,current[arm][fid])
        print('CLEAN_ORACLE_SCORED',arm,flush=True)
    output,private={},{}
    for group,ids in groups.items():
        output[group],private[group]={},{}
        for arm in candidates:
            output[group][arm],private[group][arm]=M.oracle_for(ids,choices[arm],current[arm])
        output[group]['CLEAN_REF_OCC_joint_candidate_vs_baseline']={baseline:same_candidate_joint_count(ids,choices['CLEAN_REF_OCC'],current[baseline])
            for baseline in ('R0','OLD_REF','CLEAN_REF_CLEAR','CLEAN_RAW_OCC')}
        output[group]['selected_axis_switch_counts']={baseline:sum(candidates['CLEAN_REF_OCC'][fid]['selected_name']!=candidates[baseline][fid]['selected_name'] for fid in ids)
            for baseline in ('R0','OLD_REF','CLEAN_REF_CLEAR','CLEAN_RAW_OCC')}
    private_path=p['raw']/'ORACLE_METRICS_SELECTIONS_PRIVATE.json'
    save(private_path,dict(candidate_metrics=choices,posthoc_selections=private))
    sources=[p['candidate_lock'],p['results'],p['raw']/'POSE_METRICS.json',Path(__file__),Path(M.__file__),
        O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',O.D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json']
    save(p['oracle_result'],dict(created_at=C.now(),seed=seed,groups=output,
        label='POSTHOC GT CANDIDATE ORACLE — NONDEPLOYABLE',
        actual_selector_unchanged=True,oracle_not_used_for_training_or_deployment=True,
        separate_T_R_oracles_are_not_one_attainable_pose=True,
        limitation='기존두W/D후보만의상한. 참조6D기하재구성/반복DEV. 후보성공만으로새selector일반화성공은아님.',
        sources=[C.bind(path) for path in sources],private_artifacts=[C.bind(private_path)],
        new_fits=0,optimizer_updates=0,GPU_seconds=0,seconds=time.monotonic()-start))
    print('CLEAN_CANDIDATE_ORACLE_COMPLETE',seed,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['infer','score','oracle-freeze','oracle-score'])
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    {'infer':infer,'score':score,'oracle-freeze':oracle_freeze,'oracle-score':oracle_score}[args.phase](args.seed)


if __name__=='__main__':
    main()
