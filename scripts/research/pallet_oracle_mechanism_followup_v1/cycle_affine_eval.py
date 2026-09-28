"""Freeze all C2 RGB/D9 inference before evaluation-only reference scoring."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from . import cycle_affine as A
from . import pose_oracle as O
from . import common as C

BASE_ARMS = {'PLASTIC':dict(R0='R0',OLD_RAW='RAW_LR5',OLD_REF='REF_LR5',SYN='SYN_LR5'),
             'WOOD':dict(R0='R0',OLD_RAW='WOOD_RAW_LR5',OLD_REF='WOOD_REF_LR5',SYN='SYN_LR5')}
CONTRASTS = [('OLD_REF','NEW_REF'),('NEW_RAW','NEW_REF'),('R0','NEW_REF'),('OLD_RAW','NEW_RAW'),('OLD_RAW','NEW_REF'),('SYN','NEW_REF')]


def metadata(material):
    rows, predictions, poses, paths = O.population(material)
    if material == 'PLASTIC':
        records={r['id']:r for r in C.P.records()}
        rows=[dict(r,image=records[r['id']]['image']) for r in rows]
    return rows, predictions, poses


def infer():
    lock_path=A.RAW/'PREDICTIONS_LOCK.json'
    if lock_path.exists():
        for b in C.read(lock_path)['files']:C.verify(b)
        print('C2_PREDICTIONS_ALREADY_FROZEN');return
    from ultralytics import YOLO
    from scripts.research.pallet_visible_transfer_closure_v1.infer_train import predict
    from scripts.research.pallet_material_selftrain_closure_v1.infer_eval import assert_detector_parity
    start=time.monotonic();assert torch.cuda.is_available()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
    files=[];fits={}
    for material in BASE_ARMS:
        rows,old_predictions,old_poses=metadata(material)
        predictions={alias:old_predictions[original] for alias,original in BASE_ARMS[material].items()}
        poses={alias:old_poses[original] for alias,original in BASE_ARMS[material].items()}
        for target in ('RAW','REF'):
            arm=f'{material}_{target}';fit=C.read(A.RAW/f'FIT_{arm}.json')
            assert fit['complete'] and fit['optimizer_steps']==320;C.verify(fit['checkpoint']);fits[arm]=fit['checkpoint']
            destination=A.RAW/f'EVAL_{arm}.json'
            if destination.exists():
                stored=C.read(destination);assert stored['checkpoint']==fit['checkpoint'];values=stored['predictions']
            else:
                begin=time.monotonic();model=YOLO(str(C.ROOT/fit['checkpoint']['path']),task='pose');values={}
                for j,row in enumerate(rows):
                    if j%32==0:A.gpu();print('C2_INFER',arm,j,len(rows),flush=True)
                    C.verify(row['image']);image=cv2.imread(str(C.ROOT/row['image']['path']))
                    assert image is not None and list(image.shape[:2])==row['hw']
                    values[row['id']]=predict(model,image)
                    assert_detector_parity(old_predictions['R0'][row['id']],values[row['id']])
                A.save(destination,dict(checkpoint=fit['checkpoint'],predictions=values,GT_input=False,
                                       seconds=time.monotonic()-begin),True)
                del model;torch.cuda.empty_cache()
            assert list(values)==[r['id'] for r in rows]
            for row in rows:
                C.verify(row['image'])
                assert_detector_parity(old_predictions['R0'][row['id']],values[row['id']])
            predictions['NEW_'+target]=values
            poses['NEW_'+target]={r['id']:O.D.Pose.infer(O.D.points(values[r['id']]),np.asarray(r['K']),np.asarray(r['xyz']),False) for r in rows}
            files.append(destination)
        for filename,value in [(f'{material}_PREDICTIONS.json',predictions),(f'{material}_POSES.json',poses),(f'{material}_METADATA.json',rows)]:
            path=A.RAW/filename;A.save(path,value,True);files.append(path)
    A.save(lock_path,dict(created_at=C.now(),files=[C.bind(p) for p in files],checkpoints=fits,
                         spec=C.bind(A.DOC/'SPEC.md'),code=C.bind(Path(__file__)),
                         no_reference_coordinates_read=True,original_D9=True,detector_parity=True,
                         seconds=time.monotonic()-start,GPU_seconds=time.monotonic()-start,
                         fits=0,optimizer_updates=0),True)
    print('C2_ALL_PREDICTIONS_AND_POSES_LOCKED',flush=True)


def pose_job(task):
    fid,pose,truth=task
    return fid,O.D.metric(fid,pose,truth)


def groups(rows):
    out={'ALL':[r['id'] for r in rows]}
    for key in ('recording','severity'):
        for value in sorted({r[key] for r in rows}):out[f'{key}:{value}']=[r['id'] for r in rows if r[key]==value]
    return out


def summary2(rows):
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import summary
    result=summary(rows)
    errors=[e for row in rows for e in row['errors']]
    result['tail_gt20_count']=sum(e>20 for e in errors)
    result['tail_gt50_count']=sum(e>50 for e in errors)
    result['tail_gt100_count']=sum(e>100 for e in errors)
    return result


def summary6(rows):
    result=O.D.aggregate(rows);result['pose_coverage']=result['available']/result['frames'];return result


def contrast(left,right,poseleft,poseright,ids):
    from scripts.research.pallet_recording_disjoint_transfer_v1.evaluate import transitions
    a,b=[left[i] for i in ids],[right[i] for i in ids]
    la,lb=summary2(a),summary2(b)
    return dict(PCK10_delta_pp=100*(lb['PCK']['10']-la['PCK']['10']),correct10_delta=lb['correct']['10']-la['correct']['10'],
                ADDsym_AUC_delta=summary6([poseright[i] for i in ids])['ADDsym_AUC']-summary6([poseleft[i] for i in ids])['ADDsym_AUC'],
                transitions=transitions(a,b))


def verified(predictions,rows,truth):
    from scripts.research.pallet_verified_anchor_v1.evaluate import point,metrics
    final=C.read(C.P.FINAL);assert final['reference_version']=='VERIFIED_VISIBLE_ANCHOR_FINAL_V2'
    lookup={r['id']:r for r in rows};values=[]
    for fi,ci in final['review_queue']:
        frame=final['frames'][fi];fid=frame['frame_id'];corner=frame['corners'][ci]
        if fid not in lookup or corner['status']!='DIRECT_VISIBLE':continue
        assert ci<8 and corner['coordinate_source']=='manual_click'
        assert frame['image_sha256']==lookup[fid]['image']['sha256']
        errors={};missing={}
        for arm,pp in predictions.items():
            q=point(pp[fid],ci);missing[arm]=q is None
            errors[arm]=float(np.linalg.norm(q-corner['xy'])) if q is not None else math.hypot(*truth[fid]['hw'])
        values.append(dict(id=fid,corner=ci,recording=lookup[fid]['recording'],severity=lookup[fid]['severity'],errors=errors,missing=missing))
    assert len(values)==66 and len({r['id'] for r in values})==16
    grouped={'ALL':values}
    for key in ('recording','severity'):
        grouped.update({f'{key}:{v}':[r for r in values if r[key]==v] for v in sorted({r[key] for r in values})})
    A.save(A.RAW/'VERIFIED66_POINT_METRICS.json',values,True)
    return dict(points=66,frames=16,fixed_identity=True,GT_training=False,reference=C.bind(C.P.FINAL),
                groups={g:{a:metrics([r['errors'][a] for r in rr]) for a in predictions} for g,rr in grouped.items()},
                coverage={a:sum(not r['missing'][a] for r in values) for a in predictions})


def score():
    if (A.DOC/'RESULTS.json').exists():print('C2_RESULTS_ALREADY_FROZEN');return
    start=time.monotonic();lockpath=A.RAW/'PREDICTIONS_LOCK.json';lock=C.read(lockpath)
    for b in lock['files']:C.verify(b)
    for b in lock['checkpoints'].values():C.verify(b)
    A.save(A.RAW/'SCORING_START.json',dict(utc=C.now(),prediction_lock=C.bind(lockpath)),True)
    # First reference read by this workflow, after all four checkpoints/raw/poses are locked.
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    truth=C.read(C.P.TRUTH);_,pose_truth=O.D.Pose.metadata('REAL_DEV')
    results={};visible=None
    sources=[C.P.TRUTH,C.P.FINAL,Path(__file__),Path(O.D.__file__),Path(O.D.Pose.__file__),Path(O.D.Selector.__file__),
             O.D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json',O.D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json']
    for material,bindings in BASE_ARMS.items():
        rows=C.read(A.RAW/f'{material}_METADATA.json');gg=groups(rows)
        predictions=C.read(A.RAW/f'{material}_PREDICTIONS.json');poses=C.read(A.RAW/f'{material}_POSES.json')
        oldroot=C.P.RAW if material=='PLASTIC' else C.M.RAW
        oldframe=C.read(oldroot/'FRAME_METRICS.json');oldpose=C.read(oldroot/'POSE_METRICS.json');oldfixed=C.read(oldroot/'FIXED_ID_METRICS.json')
        sources.extend(oldroot/f for f in ('FRAME_METRICS.json','POSE_METRICS.json','FIXED_ID_METRICS.json'))
        frames={a:oldframe[b] for a,b in bindings.items()};metrics={a:oldpose[b] for a,b in bindings.items()};fixed={a:oldfixed[b] for a,b in bindings.items()}
        for target in ('RAW','REF'):
            arm='NEW_'+target
            frames[arm]={r['id']:score_frame(r['id'],predictions[arm][r['id']],truth[r['id']]) for r in rows}
            fixed[arm]={r['id']:score_frame(r['id'],predictions[arm][r['id']],truth[r['id']],fixed=True) for r in rows}
            with ProcessPoolExecutor(max_workers=4) as pool:
                metrics[arm]=dict(pool.map(pose_job,[(r['id'],poses[arm][r['id']],pose_truth[r['id']]) for r in rows],chunksize=8))
        summaries={g:{a:dict(twoD=summary2([frames[a][i] for i in ii]),fixed_ID=summary2([fixed[a][i] for i in ii]),
                                  sixD=summary6([metrics[a][i] for i in ii])) for a in frames} for g,ii in gg.items()}
        expected_corners=985 if material=='PLASTIC' else 346
        assert all(v['twoD']['corners']==expected_corners for v in summaries['ALL'].values())
        assert all(v['fixed_ID']['corners']==expected_corners for v in summaries['ALL'].values())
        assert len(rows)==(128 if material=='PLASTIC' else 45)
        assert all(v['twoD']['matched']==(120 if material=='PLASTIC' else 45) for v in summaries['ALL'].values())
        contrasts={g:{f'{b}-minus-{a}':contrast(frames[a],frames[b],metrics[a],metrics[b],ii) for a,b in CONTRASTS} for g,ii in gg.items()}
        recordings=sorted({r['recording'] for r in rows})
        loro={rec:{f'{b}-minus-{a}':contrast(frames[a],frames[b],metrics[a],metrics[b],[r['id'] for r in rows if r['recording']!=rec]) for a,b in CONTRASTS} for rec in recordings}
        common=[i for i in gg['ALL'] if all(frames[a][i]['matched'] for a in frames)]
        common_results={a:dict(twoD=summary2([frames[a][i] for i in common]),sixD=summary6([metrics[a][i] for i in common])) for a in frames} if common else None
        results[material]=dict(groups=summaries,contrasts=contrasts,leave_one_recording_out=loro,
                              frames=len(rows),corners=expected_corners,recordings=len(recordings),
                              common_matched_count=len(common),common_matched_supplement=common_results)
        for filename,value in [(f'{material}_FRAME_METRICS.json',frames),(f'{material}_POSE_METRICS.json',metrics),(f'{material}_FIXED_ID_METRICS.json',fixed)]:A.save(A.RAW/filename,value,True)
        if material=='PLASTIC':visible=verified(predictions,rows,truth)
        print('C2_SCORED',material,[(a,v['twoD']['PCK']['10'],v['sixD']['ADDsym_AUC']) for a,v in summaries['ALL'].items()],flush=True)
    fits={a:C.read(A.RAW/f'FIT_{a}.json') for a in A.ARMS}
    source_validation={}
    for material in BASE_ARMS:
        source_validation[material]={}
        for target in ('RAW','REF'):
            newfit=fits[f'{material}_{target}'];oldfit=C.read((C.P.REC/'pose_only'/f'FIT_{target}_LR5.json') if material=='PLASTIC' else C.M.DOC/f'FIT_WOOD_{target}_LR5.json')
            for name,fit in [('OLD_'+target,oldfit),('NEW_'+target,newfit)]:
                C.verify(fit['results_csv']);sources.append(C.ROOT/fit['results_csv']['path'])
                row=list(csv.DictReader((C.ROOT/fit['results_csv']['path']).open()))[-1]
                source_validation[material][name]={k.strip():float(v) for k,v in row.items() if k.strip().startswith(('metrics/','val/'))}
    A.save(A.DOC/'RESULTS.json',dict(cycle=A.IDENT,materials=results,verified66=visible,
            primary='Full denominator PCK10, per material; NEW_REF versus OLD_REF/NEW_RAW/R0',
            reference='Legacy xy and geometry-derived6D, not independent physical6D; verified66 fixed identity supplement only',
            evaluation_role='Repeated DEV; single optimizer seed42, no independent confirmation',
            raw_and_pose_frozen_before_scoring=True,prediction_lock=C.bind(lockpath),
            fits={a:{k:f[k] for k in ('checkpoint','optimizer_steps','seconds','GPU_seconds','frozen_state_exact','exact_R0_initialization')} for a,f in fits.items()},
            resources=dict(fits=4,optimizer_updates=1280,fit_GPU_seconds=sum(f['GPU_seconds'] for f in fits.values()),
                           inference_GPU_seconds=lock['GPU_seconds'],scoring_CPU_wall_seconds=time.monotonic()-start),
            training_parity=C.bind(A.DOC/'TRAINING_PARITY.json'),spec=C.bind(A.DOC/'SPEC.md'),
            source_validation32=dict(results=source_validation,scope='Original same32 synthetic framework validation; descriptive bookkeeping, not source heldout6D or checkpoint selection'),
            artifact_sources=[C.bind(p) for p in dict.fromkeys(sources)],
            oracle_gap_recovery='NA: keypoint model changed; previous frozen candidate set no longer identical',
            strict_Wood_visible='NA_REFERENCE_NOT_VERIFIED',manual_supervision_added=0),True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['infer','score']);args=parser.parse_args()
    infer() if args.stage=='infer' else score()
