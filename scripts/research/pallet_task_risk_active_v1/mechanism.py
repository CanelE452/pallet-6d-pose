"""Pool-only retrospective audit and the predeclared no-training mechanism gate."""
from collections import Counter
from types import SimpleNamespace
import sys
import numpy as np
from contracts import *
from pose_adapter import pose,pool_truth
from stats import errors,hard20,diagnostics,paired_bootstrap,interval,corr

METRICS=('kp_mean_px','kp_max_px','gross20','translation_cm','lateral_cm','depth_cm','yaw_deg','rotation_deg')


def summary(rows):
    out=dict(n=len(rows),sessions=dict(Counter(r['capture_session'] for r in rows)),
        material=dict(Counter(r['object_type'] for r in rows)),day_night=dict(Counter(r['paper_condition'] for r in rows)),
        detection_failures=sum(not r['detection'] for r in rows),
        pose_failures=sum(not r['pose_valid'] for r in rows),
        axis_errors=sum(r['axis_correct'] is False for r in rows))
    out['errors']={}
    for k in METRICS:
        a=errors([r[k] for r in rows]);finite=a[np.isfinite(a)]
        hits=sum(r['hard'][k] for r in rows)
        out['errors'][k]=dict(finite_n=len(finite),missing_n=len(a)-len(finite),
            mean=float(finite.mean()) if len(finite) else None,median=float(np.median(finite)) if len(finite) else None,
            p90=float(np.quantile(finite,.9)) if len(finite) else None,
            hard20_count=hits,hard_precision=hits/len(rows) if rows else None)
    return out


def unique_bootstrap(rows,a,b):
    groups=sorted({r['capture_session'] for r in rows});rng=np.random.default_rng(SEED)
    result={k:{s:[] for s in ('mean','median','p90')} for k in ('kp_mean_px','translation_cm','yaw_deg','axis_failure')}
    for _ in range(BOOTSTRAPS):
        sampled=[g for i in rng.integers(len(groups),size=len(groups)) for g in [groups[i]]]
        sides=[[r for g in sampled for r in rows if r['capture_session']==g and r['frame_id'] in ids] for ids in (a,b)]
        for k in result:
            arrays=[np.array([float(r['axis_correct'] is not True) if k=='axis_failure' else (np.inf if r[k] is None else r[k]) for r in side]) for side in sides]
            for name,fn in [('mean',np.mean),('median',np.median),('p90',lambda v:np.quantile(v,.9))]:
                result[k][name].append(float(fn(arrays[1])-fn(arrays[0])) if all(len(v) and np.isfinite(v).all() for v in arrays) else None)
    return {k:{s:interval(v) for s,v in vv.items()} for k,vv in result.items()}


def phase1():
    verify_lock()
    assert not (RAW/'POOL_GT_ERRORS.json').exists(), 'Do not recompute after seeing risks'
    # Explicitly block the reserved labels and mixed containers in this process.
    split=read(DOC/'SPLIT_BINDING.json');blocked={str((ROOT/r['label_path']).resolve()) for r in split['evaluation']}
    blocked.update(str((POSE/n).resolve()) for n in ('AXIS_REVIEW_MANIFEST.json','GEOMETRY_RESOLVED_POSE_GT.json'))
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and str(Path(args[0]).resolve()) in blocked:
            raise PermissionError('RESERVED_GT_DENIED_IN_PHASE1')
    sys.addaudithook(guard)
    from challenge.evaluation_v2 import paper_real_eval as evaluator
    _,_,metrics=canonical_modules()
    inputs=read(RAW/'TASK_INPUTS.json');base=read(BASELINE)['frames'];output=[];truths={}
    for row,inp in zip(split['pool'],inputs):
        assert row['frame_id']==inp['frame_id']
        label=read(ROOT/row['label_path']);k=label['camera_data']['intrinsics']
        camera=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1]],float)
        assert np.allclose(camera,inp['camera']['K'],rtol=0,atol=1e-9), ('K input mismatch',row['frame_id'])
        target=evaluator._legacy_forbidden_target(SimpleNamespace(frame_id=row['frame_id'],label=row['label_path'],object_type=row['object_type']))
        points=[p['xy'] if p['xy'] is not None else [np.nan,np.nan] for p in label['objects'][0]['keypoint_annotations']]
        truth=pool_truth(points,camera,inp['dimensions']);truths[row['frame_id']]=truth
        candidates=base[str((ROOT/row['image_path']).resolve().relative_to(ROOT))]
        selected=max(candidates,key=lambda p:p['score']) if candidates else None
        pred=pose(selected['keypoints_xy'] if selected else None,camera,inp['dimensions'])
        result={k:None for k in METRICS}
        if selected:
            distance=np.linalg.norm(np.asarray(selected['keypoints_xy'])-target.keypoints_xy,axis=1)[target.keypoint_supervision_mask]
            result.update(kp_mean_px=float(distance.mean()),kp_max_px=float(distance.max()),gross20=float(np.mean(distance>20)))
        if pred['pose_valid']:
            r,t=np.array(pred['rotation']),np.array(pred['translation']);gtR,gtT=np.array(truth['rotation']),np.array(truth['translation'])
            parts=metrics.translation_components_m(t,gtT)
            result.update(translation_cm=parts['total_m']*100,lateral_cm=parts['lateral_m']*100,depth_cm=parts['depth_m']*100,
                yaw_deg=metrics.yaw_error_degrees(r,gtR),rotation_deg=metrics.rotation_error_degrees(r,gtR))
        output.append(dict(frame_id=row['frame_id'],capture_session=row['capture_session'],object_type=row['object_type'],
            paper_condition=row['paper_condition'],detection=selected is not None,pose_valid=pred['pose_valid'],
            axis_correct=pred['axis_id']==truth['axis_id'] if pred['pose_valid'] else None,
            box_iou=evaluator._box_iou(np.array(selected['box_xyxy']),target.box_xyxy) if selected else None,**result))
    thresholds={}
    for k in METRICS:
        hard,threshold=hard20([r[k] for r in output]);thresholds[k]=dict(threshold=threshold if np.isfinite(threshold) else None,hard_count=int(hard.sum()))
        for r,h in zip(output,hard):r.setdefault('hard',{})[k]=bool(h)
    write(RAW/'POOL_GT_ERRORS.json',output);write(RAW/'POOL_GT_REFERENCE.json',truths)
    choices=read(OLD_DOC/'SELECTION_LOCK.json')['selections']
    write(DOC/'OLD_SELECTION_HARDNESS_AUDIT.json',dict(status='RETROSPECTIVE_DIAGNOSTIC_ONLY',
        old_verdict_unchanged='RETROSPECTIVE_AL_NO_SIGNAL',pool=summary(output),hard_thresholds=thresholds,
        selections={m:summary([r for r in output if r['frame_id'] in ids]) for m,ids in choices.items()},
        missing_error='Worst for hard-label ranks; severity summaries explicitly show finite and missing counts',
        independent_K_vs_annotation_parity_frames=174,reserved_GT_parsed=0,
        canonical_GT='Pool-only reconstruction with unchanged canonical solver, points, fixed category geometry and minimum residual; mixed319 file not parsed',
        error_sha256=sha(RAW/'POOL_GT_ERRORS.json')))
    a,b=map(set,(choices['diversity'],choices['geometry_weighted_diversity']))
    shared=a&b;au=a-b;bu=b-a
    write(DOC/'OLD_SELECTION_UNIQUE_SET_ANALYSIS.json',dict(overlap=len(shared),
        shared_ids=sorted(shared),diversity_unique_ids=sorted(au),geometry_unique_ids=sorted(bu),
        summaries={name:summary([r for r in output if r['frame_id'] in ids]) for name,ids in [('shared',shared),('diversity_unique',au),('geometry_unique',bu)]},
        session_cluster_bootstrap_geometry_minus_diversity=unique_bootstrap(output,au,bu),
        bootstrap_draws=BOOTSTRAPS,seed=SEED,causal_claim=False))
    print('PHASE1_COMPLETE',len(output),'pool GT only; old overlap',len(shared),flush=True)


def gate():
    verify_lock();freeze=read(DOC/'RISK_FREEZE.json')
    assert freeze['risk_sha256']==sha(RAW/'TASK_RISK.json')
    errors_rows=read(RAW/'POOL_GT_ERRORS.json');risks=read(RAW/'TASK_RISK.json')
    inputs=read(RAW/'TASK_INPUTS.json');old=read(OLD_RAW/'pool/ACQUISITION_SIGNALS.json')
    assert [r['frame_id'] for r in risks]==[r['frame_id'] for r in errors_rows]
    assert [r['image_sha256'] for r in inputs]==[r['image_sha256'] for r in old]
    task=np.array([r['R_task'] for r in risks]);base=np.array([r['selected_pose_instability'] for r in old])
    inverse=np.array([1-r['original_top_score'] for r in old]);groups=np.array([r['capture_session'] for r in errors_rows])
    report={};qualifying=[]
    for k in ('kp_mean_px','translation_cm','yaw_deg'):
        error=errors([r[k] for r in errors_rows]);hard,threshold=hard20(error)
        within={}
        for group in np.unique(groups):
            ix=(groups==group)&np.isfinite(error);n=int(ix.sum())
            within[group]=dict(n=n,eligible=n>=8,task_spearman=corr(task[ix],error[ix]) if n>=8 else None,
                brightness_spearman=corr(base[ix],error[ix]) if n>=8 else None)
        positive=sum(v['task_spearman'] is not None and v['task_spearman']>0 for v in within.values())
        d={name:diagnostics(score,error,hard) for name,score in [('task',task),('brightness',base),('inverse_confidence',inverse)]}
        report[k]=dict(scores=d,within_session=within,positive_task_sessions=positive,
            hard_threshold=threshold if np.isfinite(threshold) else None,hard_count=int(hard.sum()),
            paired_frame_bootstrap=paired_bootstrap(task,base,error,hard,groups,False),
            paired_session_bootstrap=paired_bootstrap(task,base,error,hard,groups,True))
        print('DIAGNOSTIC',k,d['task'], 'brightness',d['brightness'],'positive sessions',positive,flush=True)
    coverage=dict(original_pose=float(np.mean([r['original_pose_valid'] for r in risks])),
        at_least6of8=float(np.mean([r['valid_poses']>=6 for r in risks])))
    tests={}
    for k,other in [('translation_cm','yaw_deg'),('yaw_deg','translation_cm')]:
        d=report[k]['scores'];o=report[other]['scores']
        tests[k]=dict(G1=d['task']['AUROC'] is not None and d['task']['AUROC']>=.65,
            G2=d['task']['AUROC'] is not None and d['brightness']['AUROC'] is not None and d['task']['AUROC']-d['brightness']['AUROC']>=.05,
            G3=report[k]['positive_task_sessions']>=6,
            G4=o['task']['AUROC'] is not None and o['brightness']['AUROC'] is not None and o['task']['AUROC']>=o['brightness']['AUROC']-.03)
        if all(tests[k].values()):qualifying.append(k)
    g0=coverage['original_pose']>=.9 and coverage['at_least6of8']>=.9
    passed=bool(g0 and qualifying)
    write(DOC/'TASK_RISK_ERROR_DIAGNOSTIC.json',dict(targets=report,availability=coverage,
        GT_join_after_risk_freeze=True,risk_sha256=freeze['risk_sha256'],bootstrap_draws=BOOTSTRAPS,seed=SEED,
        score='Frozen maximum, not calibrated probability',population=174,reserved145_scored=False))
    write(DOC/'TASK_RISK_VERDICT.json',dict(verdict='TASK_RISK_MECHANISM_PASS' if passed else 'TASK_RISK_MECHANISM_FAIL',
        G0=g0,target_gates=tests,qualifying_targets=qualifying,coverage=coverage,
        new_student_fits=0,new_optimizer_updates=0,next_action='Train exactly four bounded fits' if passed else 'STOP; zero new student fits',
        old_verdict='RETROSPECTIVE_AL_NO_SIGNAL',old_verdict_unchanged=True))
    print('TASK_RISK_MECHANISM_PASS' if passed else 'TASK_RISK_MECHANISM_FAIL',flush=True)


if __name__=='__main__':
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    {'phase1':phase1,'gate':gate}[sys.argv[1]]()
