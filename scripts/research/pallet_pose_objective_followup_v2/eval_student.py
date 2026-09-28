"""Lock a complete paired native/D9 inference before independent CPU scoring.

Only ``infer`` uses a GPU; no training or shared resource/state write occurs.
All writes are confined to the V2 cycle namespace, including partial attempts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import re
import time

import numpy as np
from . import common as C
from . import metric_baseline as M
from scripts.research.pallet_oracle_mechanism_followup_v1 import pose_oracle as O
from scripts.research.pallet_oracle_mechanism_followup_v1 import cycle_affine_eval as E

TARGETS=('RAW','REF')
EXPECTED={'PLASTIC':(128,985,120),'WOOD':(45,346,45)}


def paths(cycle,material,seed):
    assert re.fullmatch(r'[A-Za-z0-9_]+',cycle), 'Cycle must be a namespace identifier'
    assert material in EXPECTED and isinstance(seed,int) and seed>=0
    raw=C.RAW/'cycles'/cycle;doc=C.DOC/'cycles'/cycle;tag=f'{material}_S{seed}'
    return dict(raw=raw,doc=doc,tag=tag,lock=raw/f'PREDICTIONS_LOCK_{tag}.json',
                result=doc/f'RESULTS_{tag}.json',metadata=raw/f'METADATA_{tag}.json',
                predictions=raw/f'PREDICTIONS_{tag}.json',poses=raw/f'POSES_{tag}.json')


def fits_for(cycle,material,seed):
    p=paths(cycle,material,seed);result={}
    protocol=C.read(p['doc']/'PROTOCOL.json')
    assert protocol['locked_before_fit'] and material in protocol['materials'] and seed in protocol['seeds']
    assert C.read(p['doc']/f'PARITY_{p["tag"]}.json')['passed']
    for target in TARGETS:
        path=p['raw']/f'FIT_{material}_{target}_S{seed}.json';fit=C.read(path)
        assert fit['complete'] and fit['optimizer_steps']==320
        assert fit['material']==material and fit['target']==target and fit['seed']==seed and fit['cycle']==cycle
        assert fit['exact_R0_initialization'] and fit['protected_state_exact']
        for binding in (fit['checkpoint'],fit['protocol'],fit['initialization'],fit['trace'],fit['results_csv']):C.verify(binding)
        result[target]=fit
    assert result['RAW']['initialization']==result['REF']['initialization']
    return result


def assert_native_parity(rows,reference,values):
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    assert list(values)==[r['id'] for r in rows]
    for row in rows:
        C.verify(row['image'])
        assert_detector_parity(reference[row['id']],values[row['id']])


def inference_binding_checks(p):
    lock=C.read(p['lock'])
    assert lock['no_evaluation_reference_coordinates_read'] and lock['both_targets_locked'] and lock['original_D9']
    for binding in lock['files']+lock['sources']+list(lock['checkpoints'].values()):C.verify(binding)
    rows=C.read(p['metadata']);predictions=C.read(p['predictions']);poses=C.read(p['poses'])
    assert set(predictions)==set(poses)=={'R0','OLD_RAW','OLD_REF','SYN','NEW_RAW','NEW_REF'}
    for arm in ('NEW_RAW','NEW_REF'):
        assert_native_parity(rows,predictions['R0'],predictions[arm])
        assert set(poses[arm])=={r['id'] for r in rows}
    return lock


def infer(cycle,material,seed):
    """No truth/pose-reference coordinates are loaded by this inference path."""
    p=paths(cycle,material,seed)
    if p['lock'].exists():
        inference_binding_checks(p);print('PAIRED_INFERENCE_ALREADY_LOCKED',p['tag'],flush=True);return
    fits=fits_for(cycle,material,seed)
    metric_lock=C.DOC/'METRIC_AND_SELECTION_LOCK.json';assert metric_lock.exists()
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_oracle_mechanism_followup_v1.cycle_affine import gpu
    import cv2
    import torch
    assert torch.cuda.is_available();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    start=time.monotonic();rows,old_predictions,old_poses=E.metadata(material)
    baseline=M.MAIN_ALIASES[material]
    predictions={alias:old_predictions[original] for alias,original in baseline.items()}
    poses={alias:old_poses[original] for alias,original in baseline.items()}
    files=[];cache_runtime={}
    try:
        for target in TARGETS:
            arm='NEW_'+target;fit=fits[target];cache=p['raw']/f'EVAL_{material}_{target}_S{seed}.json'
            if cache.exists():
                stored=C.read(cache);assert stored['checkpoint']==fit['checkpoint'] and stored['GT_input'] is False
                values=stored['predictions']
            else:
                begin=time.monotonic();gpu();model=YOLO(str(C.ROOT/fit['checkpoint']['path']),task='pose');values={}
                for index,row in enumerate(rows):
                    if index%32==0:gpu();print('INFER',cycle,material,arm,seed,index,len(rows),flush=True)
                    C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']))
                    assert image is not None and list(image.shape[:2])==row['hw']
                    values[row['id']]=predict(model,image)
                assert_native_parity(rows,old_predictions['R0'],values)
                stored=dict(checkpoint=fit['checkpoint'],predictions=values,GT_input=False,
                            seconds=time.monotonic()-begin,created_at=C.now())
                C.save(cache,stored,True);del model;torch.cuda.empty_cache()
            assert_native_parity(rows,old_predictions['R0'],values)
            predictions[arm]=values;cache_runtime[target]=stored['seconds']
            poses[arm]={row['id']:O.D.Pose.infer(O.D.points(values[row['id']]),np.asarray(row['K']),np.asarray(row['xyz']),False) for row in rows}
            files.append(cache)
        for path,value in [(p['metadata'],rows),(p['predictions'],predictions),(p['poses'],poses)]:
            C.save(path,value,True);files.append(path)
        oldroot=C.P.RAW if material=='PLASTIC' else C.M.RAW
        source_paths=[metric_lock,p['doc']/'SPEC.md',p['doc']/'PROTOCOL.json',p['doc']/f'PARITY_{p["tag"]}.json',
                      Path(__file__),Path(O.__file__),Path(O.D.__file__),Path(O.D.Pose.__file__),
                      C.ROOT/'challenge/evaluation_v2/pnp_selector.py',
                      oldroot/'PREDICTIONS.json',oldroot/'POSE_PREDICTIONS.json']
        source_paths.extend(p['raw']/f'FIT_{material}_{target}_S{seed}.json' for target in TARGETS)
        C.save(p['lock'],dict(created_at=C.now(),cycle=cycle,material=material,seed=seed,
           files=[C.bind(path) for path in files],sources=[C.bind(path) for path in source_paths],
           checkpoints={target:fit['checkpoint'] for target,fit in fits.items()},
           both_targets_locked=True,no_evaluation_reference_coordinates_read=True,original_D9=True,
           detector_parity=True,RGB_only_student=True,no_teacher_inference=True,
           seconds=time.monotonic()-start,GPU_reservation_seconds_this_invocation=time.monotonic()-start,
           per_target_cached_inference_seconds=cache_runtime,
           new_fits=0,optimizer_updates=0),True)
        inference_binding_checks(p)
        print('BOTH_TARGETS_NATIVE_AND_D9_LOCKED',cycle,p['tag'],flush=True)
    except BaseException as error:
        event=p['raw']/f'INFERENCE_ATTEMPTS_{p["tag"]}.json'
        events=C.read(event) if event.exists() else []
        events.append(dict(utc=C.now(),status='FAILED_PRESERVED',error=repr(error),seconds=time.monotonic()-start,
                           completed_target_caches=list(cache_runtime),optimizer_updates=0,fit_attempts=0))
        C.save(event,events)
        raise


def pose_job(task):
    fid,pose,truth=task
    metric=O.D.metric(fid,pose,truth)
    return fid,M.extend_metric(metric,pose,truth)


def group_summaries(rows,material,frames,fixed,pose_metrics):
    groups=M.group_ids(rows,material);out={}
    for group,ids in groups.items():
        out[group]={}
        for arm in frames:
            summary=M.summarize([pose_metrics[arm][fid] for fid in ids])
            summary.update(twoD=E.summary2([frames[arm][fid] for fid in ids]) if ids else None,
                           fixed_ID=E.summary2([fixed[arm][fid] for fid in ids]) if ids else None,
                           legacy_sixD=E.summary6([pose_metrics[arm][fid] for fid in ids]) if ids else None)
            out[group][arm]=summary
    return groups,out


def verified66(predictions,rows,truth):
    from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics
    final=C.read(C.P.FINAL);assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    lookup={row['id']:row for row in rows};points=[]
    for frame_index,corner_index in final['review_queue']:
        frame=final['frames'][frame_index];fid=frame['frame_id'];corner=frame['corners'][corner_index]
        if fid not in lookup or corner['status']!='DIRECT_VISIBLE':continue
        assert corner_index<8 and corner['coordinate_source']=='manual_click'
        assert frame['image_sha256']==lookup[fid]['image']['sha256']
        errors={};missing={}
        for arm,values in predictions.items():
            q=point(values[fid],corner_index);missing[arm]=q is None
            errors[arm]=float(np.linalg.norm(q-corner['xy'])) if q is not None else float(np.hypot(*truth[fid]['hw']))
        points.append(dict(id=fid,corner=corner_index,recording=lookup[fid]['recording'],severity=lookup[fid]['severity'],errors=errors,missing=missing))
    assert len(points)==66 and len({r['id'] for r in points})==16
    groups={'ALL':points}
    for key in ('recording','severity'):
        groups.update({key+':'+value:[r for r in points if r[key]==value] for value in sorted({r[key] for r in points})})
    return dict(points=66,frames=16,fixed_identity=True,reference=C.bind(C.P.FINAL),
        groups={group:{arm:metrics([r['errors'][arm] for r in values]) for arm in predictions} for group,values in groups.items()},
        coverage={arm:sum(not r['missing'][arm] for r in points) for arm in predictions},
        DEV_reference_never_trained=True),points


def score(cycle,material,seed):
    p=paths(cycle,material,seed)
    lock=inference_binding_checks(p)
    if p['result'].exists():
        result=C.read(p['result'])
        for binding in result['scoring_sources']+result['private_artifacts']:C.verify(binding)
        print('PAIRED_SCORE_ALREADY_COMPLETE',p['tag'],flush=True);return
    fits=fits_for(cycle,material,seed);start=time.monotonic()
    scoring_start=p['raw']/f'SCORING_START_{p["tag"]}.json'
    if scoring_start.exists():C.verify(C.read(scoring_start)['prediction_lock'])
    else:C.save(scoring_start,dict(utc=C.now(),prediction_lock=C.bind(p['lock'])),True)
    # First interpretation of evaluation-only reference coordinates happens here,
    # after RAW and REF native predictions and common D9 outputs are immutable.
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    truth=C.read(C.P.TRUTH);_,pose_truth=O.D.Pose.metadata('REAL_DEV')
    rows=C.read(p['metadata']);predictions=C.read(p['predictions']);poses=C.read(p['poses'])
    oldroot=C.P.RAW if material=='PLASTIC' else C.M.RAW
    oldframes=C.read(oldroot/'FRAME_METRICS.json');oldfixed=C.read(oldroot/'FIXED_ID_METRICS.json');oldpose=C.read(oldroot/'POSE_METRICS.json')
    aliases=M.MAIN_ALIASES[material]
    frames={alias:oldframes[original] for alias,original in aliases.items()}
    fixed={alias:oldfixed[original] for alias,original in aliases.items()}
    metrics={alias:{row['id']:M.extend_metric(oldpose[original][row['id']],poses[alias][row['id']],pose_truth[row['id']]) for row in rows}
             for alias,original in aliases.items()}
    for arm in ('NEW_RAW','NEW_REF'):
        frames[arm]={row['id']:score_frame(row['id'],predictions[arm][row['id']],truth[row['id']]) for row in rows}
        fixed[arm]={row['id']:score_frame(row['id'],predictions[arm][row['id']],truth[row['id']],fixed=True) for row in rows}
        with ProcessPoolExecutor(max_workers=4) as pool:
            metrics[arm]=dict(pool.map(pose_job,[(row['id'],poses[arm][row['id']],pose_truth[row['id']]) for row in rows],chunksize=8))
    ids={row['id'] for row in rows};n,corners,matched=EXPECTED[material]
    assert len(rows)==n and all(set(values)==ids for collection in (frames,fixed,metrics,predictions,poses) for values in collection.values())
    groups,summaries=group_summaries(rows,material,frames,fixed,metrics)
    for value in summaries['ALL'].values():
        assert value['frames']==n and value['twoD']['corners']==value['fixed_ID']['corners']==corners
        assert value['twoD']['matched']==matched and value['twoD']['detected']==n
    if material=='PLASTIC':assert len(groups[M.PRIMARY])==99
    pairs=[('OLD_REF','NEW_REF'),('R0','NEW_REF'),('NEW_RAW','NEW_REF'),('OLD_RAW','NEW_RAW'),('SYN','NEW_REF')]
    contrasts={group:{after+'-minus-'+before:dict(**M.paired(metrics[before],metrics[after],members),
                 auxiliary_2D_and_ADD=E.contrast(frames[before],frames[after],metrics[before],metrics[after],members) if members else None)
               for before,after in pairs} for group,members in groups.items()}
    selection_group=M.PRIMARY if material=='PLASTIC' else 'ALL'
    classification={base:M.classify_candidate(summaries[selection_group]['NEW_REF'],summaries[selection_group][base])
                    for base in ('OLD_REF','R0','NEW_RAW')}
    loro={}
    for recording in sorted({row['recording'] for row in rows}):
        members=[row['id'] for row in rows if row['recording']!=recording and row['id'] in groups[selection_group]]
        loro[recording]={after+'-minus-'+before:M.paired(metrics[before],metrics[after],members) for before,after in pairs}
    common=[fid for fid in groups['ALL'] if all(frames[arm][fid]['matched'] for arm in frames)]
    common_matched={arm:E.summary2([frames[arm][fid] for fid in common]) for arm in frames}
    paths_saved=[]
    for name,value in [('FRAME_METRICS',frames),('FIXED_ID_METRICS',fixed),('POSE_METRICS',metrics)]:
        path=p['raw']/f'{name}_{p["tag"]}.json';C.save(path,value,True);paths_saved.append(path)
    visible=None
    if material=='PLASTIC':
        visible,points=verified66(predictions,rows,truth)
        path=p['raw']/f'VERIFIED66_POINT_METRICS_{p["tag"]}.json';C.save(path,points,True);paths_saved.append(path)
    sources=[C.P.TRUTH,C.P.FINAL,O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',O.D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json',
             oldroot/'FRAME_METRICS.json',oldroot/'FIXED_ID_METRICS.json',oldroot/'POSE_METRICS.json',
             Path(__file__),Path(M.__file__),Path(O.D.__file__),Path(O.D.Pose.__file__),
             C.ROOT/'scripts/research/pallet_material_selftrain_closure_v1/score_eval.py',
             C.DOC/'METRIC_AND_SELECTION_LOCK.json',p['lock']]
    source_validation={}
    for target in TARGETS:
        oldfit=C.read((C.P.REC/'pose_only'/f'FIT_{target}_LR5.json') if material=='PLASTIC' else C.M.DOC/f'FIT_WOOD_{target}_LR5.json')
        for label,fit in [('OLD_'+target,oldfit),('NEW_'+target,fits[target])]:
            C.verify(fit['results_csv']);path=C.ROOT/fit['results_csv']['path'];sources.append(path)
            with path.open() as stream:row=list(csv.DictReader(stream))[-1]
            source_validation[label]={key.strip():float(value) for key,value in row.items() if key.strip().startswith(('metrics/','val/'))}
    result=dict(created_at=C.now(),cycle=cycle,material=material,seed=seed,groups=summaries,contrasts=contrasts,
       classification=dict(population=selection_group,versus=classification,
           descriptive_only_for_Wood=material=='WOOD',winner_selected_here=False,
           note='Apply locked Pareto/cost/card rules centrally across eligible candidates; no AUC/PCK tie-break'),
       leave_one_recording_out=loro,LORO_population=selection_group,
       common_matched_count=len(common),common_matched_2D_supplement=common_matched,
       verified66=visible,strict_Wood_verified='NA_REFERENCE_NOT_VERIFIED',
       source_validation32=dict(results=source_validation,scope='Same original32 synthetic framework validation; not physical6D/source test or checkpoint selection'),
       fits={target:{key:fit[key] for key in ('checkpoint','optimizer_steps','seconds','seed','manual_added','checkpoint_selection')} for target,fit in fits.items()},
       prediction_lock=C.bind(p['lock']),scoring_sources=[C.bind(path) for path in dict.fromkeys(sources)],
       private_artifacts=[C.bind(path) for path in paths_saved],
       reference='Geometry-derived legacy pose/xy; verified66 direct-visible fixed-ID supplement; repeated DEV, no independent physical6D claim',
       oracle_gap_recovery='NA_DIFFERENT_STUDENT_CANDIDATE_SET; no old ADD oracle fraction claimed',
       teacher_inference_added=False,added_inference_components='None: same RGB student and D9 solver',
       seconds=time.monotonic()-start,CPU_score_wall_seconds=time.monotonic()-start,GPU_score_seconds=0,
       inference_seconds=lock['seconds'],fit_seconds=sum(fit['seconds'] for fit in fits.values()),
       new_scoring_fits=0,new_scoring_optimizer_updates=0)
    C.save(p['result'],result,True)
    print('POSE_SCORE_COMPLETE',cycle,p['tag'],classification,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['infer','score'])
    parser.add_argument('--cycle',required=True);parser.add_argument('--material',choices=list(EXPECTED),required=True)
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    (infer if args.phase=='infer' else score)(args.cycle,args.material,args.seed)


if __name__=='__main__':main()
