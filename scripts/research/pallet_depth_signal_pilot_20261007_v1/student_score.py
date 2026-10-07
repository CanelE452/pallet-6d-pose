"""Score all three final RGB students on identical locked IDs and references.

Inference is performed by student_infer, before this reference-reading stage.
No checkpoint, teacher rule or automatically selected W/D branch is tuned here.
"""
from collections import Counter
import copy
import json
from pathlib import Path
import time
import numpy as np
from . import common as C
from . import pilot as P

STUDENTS=('RAW_TARGET','GLOBAL_TARGET','DEPTH_TARGET')
RGB_ARMS=('R0','GLOBAL_POST_RGB_ONLY',*STUDENTS)

def point_error(q, target, support):
    if q is None:
        return None
    mask=np.asarray(support,bool)[:8] & np.isfinite(np.asarray(target)[:8]).all(-1)
    return P.distribution(np.linalg.norm(np.asarray(q)[:8][mask]-np.asarray(target)[:8][mask],axis=-1))

def metric(pose,truth,fid,D):
    row=D.metric(fid,pose,truth)
    if row['available']:
        row['z_abs_cm']=float(abs(pose['centroid'][2]-truth['t'][2])*100)
    return row

def pair(rows,before,after,repetitions=2000):
    a=P.summarize(r['metrics'][before] for r in rows)
    b=P.summarize(r['metrics'][after] for r in rows)
    recordings=sorted({r['original_recording'] for r in rows})
    clusters={rec:[r for r in rows if r['original_recording']==rec] for rec in recordings}
    rng=np.random.default_rng(20261007)
    values={k:[] for k in ('z_abs_cm','translation_cm')}
    for _ in range(repetitions):
        draw=[r for rec in rng.choice(recordings,len(recordings),replace=True) for r in clusters[rec]]
        for key in values:
            old=np.median([r['metrics'][before].get(key,float('inf')) for r in draw])
            new=np.median([r['metrics'][after].get(key,float('inf')) for r in draw])
            values[key].append(new-old)
    deltas={k:{stat:b[k][stat]-a[k][stat] if b[k][stat] is not None and a[k][stat] is not None else None
               for stat in ('median','mean','P90')} for k in values}
    return C.clean(dict(frames=len(rows),recordings=len(recordings),differences= deltas,
        paired_original_recording_bootstrap=dict(repetitions=repetitions,seed=20261007,
            median_difference_95_spread={k:np.quantile(v,[.025,.975]) for k,v in values.items()}),
        new_pose_failures=sum(r['metrics'][before]['available'] and not r['metrics'][after]['available'] for r in rows)))

def main():
    if (C.DOC/'STUDENT_SUMMARY.json').exists():
        print('EXISTING_FINAL_STUDENT_SCORE',flush=True)
        return
    start=time.monotonic()
    gate=C.read(C.DOC/'SUMMARY.json')
    assert gate['student_training_eligible']
    lock=C.read(C.DOC/'STUDENT_PREDICTIONS_LOCK.json')
    # Accept bind-only files or filename keyed bindings, but always verify.
    files=lock['files']
    for b in files.values() if isinstance(files,dict) else files:
        C.verify(b)
    C.verify(lock['manifest'])
    for b in lock['receipts'].values():
        C.verify(b)
    for b in lock['checkpoints'].values():
        C.verify(b)
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    from scripts.research.pallet_selftraining_paper_closure_v1 import common as OLD
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    from . import source_metadata as S
    C.save(C.DOC/'STUDENT_SCORING_LOCK.json',dict(at=C.now(),predictions_lock=C.bind(C.DOC/'STUDENT_PREDICTIONS_LOCK.json'),
        scoring_code=C.bind(Path(__file__)),source_metadata_code=C.bind(Path(S.__file__)),
        pose_code=C.bind(Path(D.Pose.__file__)),no_model_selection=True,seed=42,
        metrics='Same automatic production F; original recording paired bootstrap; fixed R0 branch diagnostic only.'))
    metadata=C.read(C.PRIVATE/'INFERENCE_METADATA_PRIVATE.json')
    packets={arm:C.read(C.PRIVATE/f'INFERENCE_{arm}.json')['predictions'] for arm in ('R0',*STUDENTS)}
    inputs=C.read(C.PRIVATE/'INPUTS_PRIVATE.json')
    input_eval={r['id']:r for r in inputs['EVAL']};input_train={r['id']:r for r in inputs['TRAIN']}
    teacher=C.read(C.PRIVATE/'TEACHER_PRIVATE.json')
    teacher_eval={r['id']:r for r in teacher['EVAL']};teacher_train={r['id']:r for r in teacher['TRAIN']}
    _,truth=D.Pose.metadata('REAL_DEV');two_truth=C.read(OLD.TRUTH)
    source_meta,source_truth,source_two_truth=S.load()
    C.save(C.DOC/'SOURCE_COORDINATE_VALIDATION.json',S.validation_receipt())
    source_meta={r['id']:r for r in source_meta} if isinstance(source_meta,list) else source_meta
    counts=Counter();poses={arm:{split:{} for split in ('EVAL','SOURCE','TRAIN_PROBE')} for arm in ('R0',*STUDENTS)}
    for split in ('EVAL','SOURCE','TRAIN_PROBE'):
        rows=metadata[split]
        for row in rows:
            fid=row['id']
            base_meta=input_eval[fid] if split=='EVAL' else source_meta[fid] if split=='SOURCE' else input_train[fid]
            for arm in ('R0',*STUDENTS):
                if arm=='R0' and split=='EVAL':
                    pose=teacher_eval[fid]['R0']
                else:
                    pose=D.Pose.infer(D.points(packets[arm][split][fid]),np.asarray(base_meta['K']),np.asarray(base_meta['xyz']),split=='SOURCE')
                    counts[f'actual_F_{arm}_{split}']+=1
                poses[arm][split][fid]=pose
    visible=C.read(OLD.FINAL);vpoints={}
    for fi,ci in visible['review_queue']:
        frame=visible['frames'][fi];corner=frame['corners'][ci]
        if corner['status']=='DIRECT_VISIBLE':
            assert ci<8 and corner['coordinate_source']=='manual_click'
            vpoints.setdefault(frame['frame_id'],[]).append((ci,np.asarray(corner['xy'])))
    visible_count=sum(len(vpoints.get(r['id'],[])) for r in inputs['EVAL'])
    assert visible_count==66
    eval_rows=[]
    for row in inputs['EVAL']:
        fid=row['id'];result=dict(id=fid,original_recording=row['original_recording'],recording=row['recording'],severity=row['severity'],
                                metrics={},twoD={},damage={},fixed_branch={})
        for arm in RGB_ARMS:
            pose=poses[arm]['EVAL'][fid] if arm!='GLOBAL_POST_RGB_ONLY' else copy.deepcopy(poses['R0']['EVAL'][fid])
            if arm=='GLOBAL_POST_RGB_ONLY':
                pose['centroid']=(np.asarray(pose['centroid'])*gate['GLOBAL_POST']['scale']).tolist()
            result['metrics'][arm]=metric(pose,truth[fid],fid,D)
            prediction=packets['R0' if arm=='GLOBAL_POST_RGB_ONLY' else arm]['EVAL'][fid]
            result['twoD'][arm]=score_frame(fid,prediction,two_truth[fid])
            q=D.points(prediction);q0=D.points(packets['R0']['EVAL'][fid]);damage=[]
            for corner,xy in vpoints.get(fid,[]):
                a=float(np.linalg.norm(q0[corner]-xy));b=float(np.linalg.norm(q[corner]-xy)) if q is not None else float(np.hypot(*row['hw']))
                damage.append(dict(corner=corner,before_px=a,after_px=b,good5_to_bad10=a<5 and b>10))
            result['damage'][arm]=damage
            if arm in STUDENTS:
                raw=poses['R0']['EVAL'][fid];dims=raw['cf_extents']
                h=dict(success=True,name=raw['selected_hypothesis'],camera_facing_dimensions_m=dict(width=dims[0],height=dims[1],depth=dims[2]))
                fixed=D.hyp_pose(h,q,np.asarray(row['K']),np.asarray(row['xyz'])) if q is not None else dict(available=False)
                counts['fixed_branch_'+arm+'_EVAL']+=1
                result['fixed_branch'][arm]=metric(fixed,truth[fid],fid,D)
        eval_rows.append(result)
    source_rows=[]
    for row in metadata['SOURCE']:
        fid=row['id'];r=dict(id=fid,scenario=source_meta[fid].get('scenario_id'),metrics={},twoD={})
        for arm in ('R0',*STUDENTS):
            r['metrics'][arm]=metric(poses[arm]['SOURCE'][fid],source_truth[fid],fid,D)
            r['twoD'][arm]=S.measure_two_d(packets[arm]['SOURCE'][fid],source_two_truth[fid])
        source_rows.append(r)
    probe_rows=[]
    for row in metadata['TRAIN_PROBE']:
        fid=row['id'];target=teacher_train[fid];train=input_train[fid]
        r=dict(id=fid,original_recording=train['original_recording'],supervised_corners=int(np.asarray(train['support'])[:8].sum()),
            target_z_constructed_m=target['result']['zD'],target_z_actual_F_m=target['TEACHER_F_QD_RGBD']['centroid'][2],arms={})
        initial=poses['R0']['TRAIN_PROBE'][fid]
        for arm in ('R0',*STUDENTS):
            pose=poses[arm]['TRAIN_PROBE'][fid];q=D.points(packets[arm]['TRAIN_PROBE'][fid])
            r['arms'][arm]=dict(qD_residual_px=point_error(q,target['qD'],train['support']),available=pose['available'])
            if pose['available']:
                r['arms'][arm].update(z_m=pose['centroid'][2],z_target_actual_F_abs_cm=abs(pose['centroid'][2]-r['target_z_actual_F_m'])*100,
                    z_target_constructed_abs_cm=abs(pose['centroid'][2]-r['target_z_constructed_m'])*100,
                    signed_z_change_m=pose['centroid'][2]-initial['centroid'][2])
        probe_rows.append(r)
    groups={'FULL128':eval_rows,'CLEAN29':[r for r in eval_rows if r['severity']=='CLEAN'],
            'NATURAL99':[r for r in eval_rows if r['severity']!='CLEAN']}
    summary={g:{arm:P.summarize(r['metrics'][arm] for r in rows) for arm in RGB_ARMS} for g,rows in groups.items()}
    twod={g:{arm:P.summarize_two_d(r['twoD'][arm] for r in rows) for arm in RGB_ARMS} for g,rows in groups.items()}
    contrasts={g:{f'DEPTH_TARGET-minus-{arm}':pair(rows,arm,'DEPTH_TARGET') for arm in RGB_ARMS if arm!='DEPTH_TARGET'} for g,rows in groups.items()}
    follow={arm:dict(qD_residual_px_median_of_frame_means=P.distribution(r['arms'][arm]['qD_residual_px']['mean'] for r in probe_rows if r['arms'][arm]['qD_residual_px'] is not None),
        z_target_actual_F_abs_cm=P.distribution(r['arms'][arm]['z_target_actual_F_abs_cm'] for r in probe_rows if r['arms'][arm]['available'])) for arm in ('R0',*STUDENTS)}
    primary=summary['FULL128'];depth=primary['DEPTH_TARGET'];raw=primary['RAW_TARGET'];base=primary['R0']
    improves=lambda a,b:all(a[k]['median']<b[k]['median'] for k in ('z_abs_cm','translation_cm'))
    follows=follow['DEPTH_TARGET']['qD_residual_px_median_of_frame_means']['median']<follow['R0']['qD_residual_px_median_of_frame_means']['median']
    fixed_summary={g:{arm:P.summarize(r['fixed_branch'][arm] for r in rows) for arm in STUDENTS} for g,rows in groups.items()}
    fixed_gain=improves(fixed_summary['FULL128']['DEPTH_TARGET'],fixed_summary['FULL128']['RAW_TARGET']) and improves(fixed_summary['FULL128']['DEPTH_TARGET'],base)
    label=('TEACHER_GOOD_STUDENT_NOT_FOLLOWING' if not follows else 'STUDENT_FOLLOWS_NO_DEV_GAIN' if not(improves(depth,raw) and improves(depth,base)) else
           'GLOBAL_SUFFICIENT' if not(improves(depth,primary['GLOBAL_TARGET']) and improves(depth,primary['GLOBAL_POST_RGB_ONLY'])) else 'DEPTH_TO_RGB_SIGNAL')
    if follows and label=='STUDENT_FOLLOWS_NO_DEV_GAIN' and fixed_gain:
        label='FIXED_BRANCH_GAIN_ONLY'
    if label=='GLOBAL_SUFFICIENT':
        dominates=lambda a,b:all(a[k]['median']<=b[k]['median'] for k in ('z_abs_cm','translation_cm'))
        label=('GLOBAL_SUFFICIENT_ON_POINT_ESTIMATES' if any(dominates(primary[a],depth) for a in ('GLOBAL_TARGET','GLOBAL_POST_RGB_ONLY'))
               else 'GLOBAL_COMPARISON_UNRESOLVED')
    damage={arm:sum(e['good5_to_bad10'] for r in eval_rows for e in r['damage'][arm]) for arm in RGB_ARMS}
    C.save(C.DOC/'STUDENT_EVAL_ROWS.jsonl',''.join(json.dumps(C.clean(r),ensure_ascii=False,allow_nan=False)+'\n' for r in eval_rows))
    C.save(C.DOC/'SOURCE_EVAL_ROWS.jsonl',''.join(json.dumps(C.clean(r),ensure_ascii=False,allow_nan=False)+'\n' for r in source_rows))
    C.save(C.DOC/'TRAIN_PROBE_ROWS.jsonl',''.join(json.dumps(C.clean(r),ensure_ascii=False,allow_nan=False)+'\n' for r in probe_rows))
    C.save(C.DOC/'STUDENT_SUMMARY.json',dict(status=label,teacher_status='PILOT_GO',groups=summary,twoD=twod,
        paired=contrasts,probe=follow,direct_visible=dict(points=visible_count,good5_to_bad10=damage),
        fixed_branch=fixed_summary,
        source256={arm:P.summarize(r['metrics'][arm] for r in source_rows) for arm in ('R0',*STUDENTS)},
        source_twoD={arm:P.summarize_two_d(r['twoD'][arm] for r in source_rows) for arm in ('R0',*STUDENTS)},
        resources=dict(counts),score_seconds=time.monotonic()-start,
        inference_lock=C.bind(C.DOC/'STUDENT_PREDICTIONS_LOCK.json'),
        question_answer='Single-seed reused-DEV result; target following and actual RGB-only accuracy are distinct.',
        reference='Existing reconstructed real geometry and legacy 2D plus direct-visible66 damage; registered synthetic heldout. No new labels.',
        classification_scope='Operational sign-based pilot label; damage, means/tails and cluster intervals must accompany it. Not confirmation or all-6D solution.'))
    C.save(C.PRIVATE/'STUDENT_POSES_PRIVATE.json',poses)
    print('FINAL_RGB_ONLY_STUDENT_RESULT',label,dict(counts),flush=True)

if __name__=='__main__':
    main()
