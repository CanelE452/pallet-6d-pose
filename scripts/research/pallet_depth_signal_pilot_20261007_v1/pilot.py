"""Lock inputs, create one GT-free teacher, then apply the preregistered gate.

Run ``python -B -m scripts.research.pallet_depth_signal_pilot_20261007_v1.pilot
lock|teacher|score``. References are only opened by score, after teacher freeze.
No student is fitted by this program; a GO receipt is necessary for any fit.
"""
import argparse
from collections import Counter
import copy
import csv
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from . import common as C
from . import teacher as T

OLD = C.ROOT / 'data/pallet/results/pallet_clean_to_pose_transfer_v1/evaluation/S42'
TEACHER_ARMS = ('TEACHER_PD_RGBD', 'TEACHER_F_QD_RGBD')
ARMS = ('R0', *TEACHER_ARMS)

def distribution(values):
    values = list(values)
    if not values:
        return dict(n=0, median=None, mean=None, P90=None)
    a = np.asarray(values, float)
    return dict(n=len(a), median=float(np.median(a)), mean=float(np.mean(a)),
                P90=float(np.quantile(a, .9)))

def summarize(rows):
    rows = list(rows)
    good = [r for r in rows if r['available']]
    result = dict(frames=len(rows), available=len(good), failed=len(rows)-len(good))
    for key in ('z_abs_cm', 'translation_cm', 'rotation_deg', 'ADDsym_m'):
        result[key] = distribution(r[key] if r['available'] else float('inf') for r in rows)
    return C.clean(result)

def summarize_two_d(rows):
    rows=list(rows)
    errors=[e for r in rows for e in r['errors']]
    return dict(frames=len(rows),corners=len(errors),matched=sum(r['matched'] for r in rows),
        pixel_error=distribution(errors),PCK10=float(np.mean(np.asarray(errors)<=10)) if errors else None,
        errors_gt20=sum(e>20 for e in errors),errors_gt50=sum(e>50 for e in errors))

def gate(raw, corrected, depth_reference_count, accepted_recordings, accepted_count):
    """Operational gate, not statistical confirmation or a learned decision."""
    if depth_reference_count < 16:
        return dict(status='TEACHER_UNRESOLVED', reason='fewer_than_16_depth_reference_frames')
    if accepted_count < 2 or len(accepted_recordings) < 2:
        return dict(status='TEACHER_UNRESOLVED', reason='extremely_few_corrections_or_one_recording')
    if corrected['failed'] > raw['failed']:
        return dict(status='TEACHER_UNRESOLVED', reason='new_numeric_pose_failures')
    z = corrected['z_abs_cm']['median'] - raw['z_abs_cm']['median']
    t = corrected['translation_cm']['median'] - raw['translation_cm']['median']
    mean_z = corrected['z_abs_cm']['mean'] - raw['z_abs_cm']['mean']
    status = ('PILOT_GO' if z < 0 and t < 0 and mean_z <= 0 else
              'TEACHER_NOT_HELPFUL' if z > 0 and t > 0 else 'TEACHER_UNRESOLVED')
    return dict(status=status, reason='frozen_full128_pose_gate', z_median_delta_cm=z,
                translation_median_delta_cm=t, z_mean_delta_cm=mean_z)

def lock():
    if (C.DOC/'PROTOCOL.json').exists():
        protocol=C.read(C.DOC/'PROTOCOL.json')
        C.verify(protocol['private_inputs']);C.verify(protocol['input_audit'])
        for b in protocol['sources']:
            C.verify(b)
        print('EXISTING_PROTOCOL_VERIFIED_NO_RESELECTION',flush=True)
        return
    from .prepare_inputs import build
    inputs, audit = build()
    assert len(inputs['EVAL']) == 128 and len(inputs['TRAIN']) <= 256
    assert {r['id'] for r in inputs['EVAL']} == {r['id'] for r in C.read(OLD/'METADATA.json')}
    assert not ({r['image']['sha256'] for r in inputs['TRAIN']} &
                {r['image']['sha256'] for r in inputs['EVAL']})
    assert not ({r['original_recording'] for r in inputs['TRAIN']} &
                {r['original_recording'] for r in inputs['EVAL'] if r.get('original_recording')})
    C.save(C.PRIVATE/'INPUTS_PRIVATE.json', inputs)
    C.save(C.DOC/'INPUTS.json', audit)
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    initial = C.read(C.ROOT/'_docs/experiments/pallet_clean_to_pose_transfer_v1/PRIMARY_PROTOCOL.json')['initialization']
    C.verify(initial)
    sources = [Path(__file__), Path(T.__file__), Path(__import__(build.__module__,fromlist=['']).__file__),
               Path(C.__file__), Path(D.Pose.__file__),
               C.ROOT/'challenge/evaluation_v2/pnp_selector.py',
               C.ROOT/'scripts/paper/pose_metric_closure_v1/run_pose_evaluation.py',
               C.ROOT/'scripts/self_training_yolo/depth_corrected/audit_sensor_geometry_validation.py',
               OLD/'METADATA.json', OLD/'PREDICTIONS.json', OLD/'POSES.json', OLD/'PREDICTIONS_LOCK.json']
    for filename in ('SOURCE_EVAL_MANIFEST.json','UNIT_CHECKS.json'):
        if (C.DOC/filename).exists():
            sources.append(C.DOC/filename)
    sources.extend(p for p in Path(__file__).parent.glob('test_*.py'))
    C.save(C.DOC/'PROTOCOL.json', dict(locked_at=C.now(), base_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'], cwd=C.ROOT).decode().strip(),
        branch='main', branch_instruction_resolution='Direct user main-only/no-new-branch instruction overrides attached plan branch suggestion.',
        plan=C.bind('/home/minjae/Downloads/pallet_depth_signal_pilot_plan_20261007.txt'),
        teacher=T.settings(), private_inputs=C.bind(C.PRIVATE/'INPUTS_PRIVATE.json'),
        input_audit=C.bind(C.DOC/'INPUTS.json'), sources=[C.bind(p) for p in sources],
        initialization=initial, reference='Existing geometric reconstruction; repeated DEV, not physical ground truth or independent test.',
        gate=dict(depth_reference_minimum=16, accepted_train_minimum=32, train_recording_minimum=2,
                  full128_z_median_and_translation_median='both strictly lower', mean_z_change='nonworse', new_numeric_failures=0,
                  extreme_correction_count='0 or 1', accepted_eval_recordings_minimum=2,
                  recording_dominance='GO must retain z/T median improvement when omitting each accepted recording'),
        budget=dict(teacher_train_max=256, teacher_eval_max=128, source_eval=256,
                    student_fits_max=3, optimizer_updates_per_fit=320, student_forward_max=1248,
                    missing_R0_forward_max=384, GPU_fit_seconds_max=1200, preparation_seconds_report_limit=1800),
        conditional_training=dict(arms=['RAW_TARGET','GLOBAL_TARGET','DEPTH_TARGET'], seed=42, epochs=5,
            batch=16, updates=320, real_slots_per_epoch=512, source_slots_per_epoch=512,
            optimizer='AdamW', lr0=1e-5, lrf=.1, weight_decay=1e-4, warmup_epochs=0,
            cosine=True, augmentation='Existing CLEAR affine/HSV, shared samples and common support; no new occlusion augmentation',
            initialization='Exact same R0, last checkpoint only', trainable='Protected pose/flow only; backbone/detection/buffers frozen'),
        operational_concretization='Grid/tile layout and 95% block spread intervals instantiate the attached fixed rules; no DEV-driven setting search.',
        training_reference_coordinates_used=False, tuning=0, new_annotations=0))
    print('INPUTS_AND_PROTOCOL_LOCKED', len(inputs['TRAIN']), len(inputs['EVAL']), flush=True)

def teacher():
    if (C.DOC/'TEACHER_LOCK.json').exists():
        stored=C.read(C.DOC/'TEACHER_LOCK.json');C.verify(stored['outputs']);C.verify(stored['protocol'])
        print('EXISTING_TEACHER_VERIFIED_NO_NEW_CALLS',flush=True)
        return
    start = time.monotonic()
    protocol = C.read(C.DOC/'PROTOCOL.json')
    assert protocol['teacher'] == T.settings()
    C.verify(protocol['private_inputs'])
    for b in protocol['sources']:
        C.verify(b)
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    inputs = C.read(C.PRIVATE/'INPUTS_PRIVATE.json')
    outputs = {'TRAIN':[], 'EVAL':[]}
    counts = Counter()
    cv2.setNumThreads(1)
    for split in ('TRAIN', 'EVAL'):
        for index, row in enumerate(inputs[split]):
            q0 = None if row.get('q0') is None else np.asarray(row['q0'], float)
            pose = row.get('pose0')
            if pose is None:
                pose = D.Pose.infer(q0, np.asarray(row['K']), np.asarray(row['xyz']), False)
                counts['actual_F_RAW_'+split] += 1
            elif split == 'EVAL' and index < 8:
                current = D.Pose.infer(q0, np.asarray(row['K']), np.asarray(row['xyz']), False)
                D.close(current, pose)
                counts['actual_F_RAW_parity_EVAL'] += 1
            result = dict(status='abstain', reason='missing_depth', accepted=False)
            qd = q0
            pd = copy.deepcopy(pose)
            pf = copy.deepcopy(pose)
            if row.get('depth') and pose['available']:
                C.verify(row['depth'])
                depth = cv2.imread(str(C.ROOT/row['depth']['path']), cv2.IMREAD_UNCHANGED)
                assert depth is not None and list(depth.shape) == row['hw'], row['id']
                counts['teacher_geometry_'+split] += 1
                try:
                    result = T.estimate(depth.astype(float)*row.get('depth_unit_scale',.001),
                        np.asarray(row['K']), np.asarray(pose['R_cf']), np.asarray(pose['centroid']),
                        np.asarray(pose['cf_extents']), invalid_codes_m=row.get('invalid_codes_m',()))
                except ValueError as error:
                    result = dict(status='ABSTAIN',reason='INVALID_INITIAL_GEOMETRY',accepted=False,
                                  detail=str(error))
                if result.get('accepted'):
                    tnew = np.asarray(result['tD'])
                    qd = T.transfer(q0, np.asarray(row['support'], bool), np.asarray(row['K']),
                        np.asarray(pose['R_cf']), np.asarray(pose['centroid']), tnew,
                        np.asarray(pose['cf_extents']))
                    pd['centroid'] = tnew.tolist()
                    pf = D.Pose.infer(qd, np.asarray(row['K']), np.asarray(row['xyz']), False)
                    counts['actual_F_QD_'+split] += 1
            elif not pose['available']:
                result['reason'] = 'initial_pose_unavailable'
            outputs[split].append(dict(id=row['id'], recording=row.get('recording'),
                original_recording=row.get('original_recording'), severity=row.get('severity'),
                depth_available=bool(row.get('depth')), result=result, q0=q0, qD=qd,
                R0=pose, TEACHER_PD_RGBD=pd, TEACHER_F_QD_RGBD=pf))
            if (index+1)%32 == 0:
                print('TEACHER',split,index+1,len(inputs[split]),flush=True)
    C.save(C.PRIVATE/'TEACHER_PRIVATE.json', outputs)
    C.save(C.DOC/'TEACHER_LOCK.json',dict(at=C.now(), protocol=C.bind(C.DOC/'PROTOCOL.json'),
        outputs=C.bind(C.PRIVATE/'TEACHER_PRIVATE.json'), counts=dict(counts),
        seconds=time.monotonic()-start, GT_teacher_inputs=False, neural_forward_examples=0,
        model_fits=0, optimizer_updates=0, rule_searches=0))
    print('GT_FREE_TEACHER_LOCKED',dict(counts),flush=True)

def bootstrap(rows, arm, repetitions=2000):
    recs = sorted({r['original_recording'] or r['recording'] for r in rows})
    by_rec = {rec:[r for r in rows if (r['original_recording'] or r['recording'])==rec] for rec in recs}
    rng = np.random.default_rng(20261007)
    changes = {k:[] for k in ('z_abs_cm','translation_cm')}
    for _ in range(repetitions):
        sample = [r for rec in rng.choice(recs,len(recs),replace=True) for r in by_rec[rec]]
        for key in changes:
            a = np.median([r['metrics']['R0'].get(key,float('inf')) for r in sample])
            b = np.median([r['metrics'][arm].get(key,float('inf')) for r in sample])
            changes[key].append(float(b-a))
    return dict(unit='original recording, paired whole clusters', repetitions=repetitions, seed=20261007,
        recordings=len(recs), median_error_difference_95_spread={k:list(np.quantile(v,[.025,.975])) for k,v in changes.items()},
        interpretation='Descriptive repeated-DEV cluster bootstrap; not calibration CI or independent confirmation.')

def score():
    if (C.DOC/'SUMMARY.json').exists():
        print('EXISTING_FROZEN_SCORE',C.read(C.DOC/'SUMMARY.json')['status'],flush=True)
        return
    start=time.monotonic()
    lock=C.read(C.DOC/'TEACHER_LOCK.json');C.verify(lock['outputs']);C.verify(lock['protocol'])
    outputs=C.read(C.PRIVATE/'TEACHER_PRIVATE.json')
    # All teacher proposals are frozen above; references do not enter estimate/transfer.
    from scripts.research.pallet_clean19_pose_mismatch_v1 import diagnose as D
    from scripts.research.pallet_selftraining_paper_closure_v1 import common as P
    from scripts.research.pallet_material_selftrain_closure_v1.score_eval import score_frame
    _,truth=D.Pose.metadata('REAL_DEV')
    two_truth=C.read(P.TRUTH)
    raw_preds=C.read(OLD/'PREDICTIONS.json')['R0']
    old_metrics=C.read(OLD/'POSE_METRICS.json')['R0']
    accepted_train=[r for r in outputs['TRAIN'] if r['result'].get('accepted')]
    global_m=float(np.median([np.log(r['result']['zD']/r['R0']['centroid'][2]) for r in accepted_train])) if accepted_train else None
    eval_arms=ARMS+('GLOBAL_POST_RGB_ONLY',) if global_m is not None else ARMS
    visible=C.read(P.FINAL)
    vpoints={}
    for fi,ci in visible['review_queue']:
        frame=visible['frames'][fi];corner=frame['corners'][ci]
        if corner['status']=='DIRECT_VISIBLE':
            assert corner['coordinate_source']=='manual_click' and ci<8
            vpoints.setdefault(frame['frame_id'],[]).append((ci,np.asarray(corner['xy'])))
    results=[]
    csv_rows=[]
    for row in outputs['EVAL']:
        fid=row['id'];r=copy.deepcopy(row);r.pop('q0');r.pop('qD');r['metrics']={};r['twoD']={};r['direct_visible_damage']={}
        for key in ('sample_pixels','sample_tile_ids','sample_face_names','sample_absolute_cosines'):
            r['result'].pop(key,None)  # Private sampling cache is not a public Git artifact.
        if global_m is not None:
            r['GLOBAL_POST_RGB_ONLY']=copy.deepcopy(row['R0'])
            if r['GLOBAL_POST_RGB_ONLY']['available']:
                r['GLOBAL_POST_RGB_ONLY']['centroid']=(np.asarray(row['R0']['centroid'])*np.exp(global_m)).tolist()
        for arm in eval_arms:
            current_pose=row[arm] if arm in ARMS else r[arm]
            metric=D.metric(fid,current_pose,truth[fid])
            if metric['available']:
                metric['z_abs_cm']=abs(current_pose['centroid'][2]-truth[fid]['t'][2])*100
            if arm=='R0':
                for key in ('translation_cm','rotation_deg','ADDsym_m'):
                    assert np.isclose(metric[key],old_metrics[fid][key],atol=1e-7,rtol=1e-7)
            r['metrics'][arm]=metric
        for arm,q in [('R0',row['q0']),('TEACHER_F_QD_RGBD',row['qD'])]:
            pred=copy.deepcopy(raw_preds[fid])
            if pred['selected_index'] is not None:
                pred['candidates'][pred['selected_index']]['keypoints_xy']=q
            r['twoD'][arm]=score_frame(fid,pred,two_truth[fid])
        for arm in ('TEACHER_PD_RGBD','TEACHER_F_QD_RGBD'):
            # PD is a pose only; its associated 2D target is qD, not a new measurement.
            errors=[]
            for corner,xy in vpoints.get(fid,[]):
                if row['q0'] is None:
                    continue
                a=float(np.linalg.norm(np.asarray(row['q0'][corner])-xy))
                b=float(np.linalg.norm(np.asarray(row['qD'][corner])-xy))
                errors.append(dict(corner=corner,before_px=a,after_px=b,good5_to_bad10=a<5 and b>10))
            r['direct_visible_damage'][arm]=errors
        results.append(r)
        s=row['result'];a=r['metrics']['R0'];b=r['metrics']['TEACHER_F_QD_RGBD']
        csv_rows.append(dict(id=fid,split='EVAL',recording=row['original_recording'] or row['recording'],
            depth_available=row['depth_available'],status=s['status'],reason=s.get('reason'),accepted=s.get('accepted',False),
            z0_m=row['R0']['centroid'][2] if row['R0']['available'] else None,
            zD_m=s.get('zD'),zF_m=row['TEACHER_F_QD_RGBD']['centroid'][2] if row['TEACHER_F_QD_RGBD']['available'] else None,
            samples=s.get('valid_depth_samples'),blocks=s.get('fit_nonempty_blocks',0)+s.get('check_nonempty_blocks',0),
            raw_z_error_cm=a.get('z_abs_cm'),corrected_z_error_cm=b.get('z_abs_cm'),
            raw_translation_cm=a.get('translation_cm'),corrected_translation_cm=b.get('translation_cm'),
            raw_rotation_deg=a.get('rotation_deg'),corrected_rotation_deg=b.get('rotation_deg'),
            raw_ADDsym_m=a.get('ADDsym_m'),corrected_ADDsym_m=b.get('ADDsym_m'),
            new_pose_failure=a['available'] and not b['available']))
    accepted=[r for r in results if r['result'].get('accepted')]
    groups={'FULL128':results,'ACCEPTED':accepted,'CLEAN29':[r for r in results if r['severity']=='CLEAN'],
        'NATURAL99':[r for r in results if r['severity']!='CLEAN']}
    summary={group:{arm:summarize(r['metrics'][arm] for r in rows) for arm in eval_arms} for group,rows in groups.items()}
    two_d={group:{arm:summarize_two_d(r['twoD'][arm] for r in rows) for arm in ('R0','TEACHER_F_QD_RGBD')}
           for group,rows in groups.items()}
    g=gate(summary['FULL128']['R0'],summary['FULL128']['TEACHER_F_QD_RGBD'],
        sum(r['depth_available'] for r in results),{r['original_recording'] for r in accepted},len(accepted))
    loro={}
    if g['status']=='PILOT_GO':
        for rec in sorted({r['original_recording'] for r in accepted}):
            subset=[r for r in results if r['original_recording']!=rec]
            a,b=[summarize(r['metrics'][arm] for r in subset) for arm in ('R0','TEACHER_F_QD_RGBD')]
            loro[rec]=dict(z_median_delta_cm=b['z_abs_cm']['median']-a['z_abs_cm']['median'],
                          translation_median_delta_cm=b['translation_cm']['median']-a['translation_cm']['median'])
        if any(v['z_median_delta_cm']>=0 or v['translation_median_delta_cm']>=0 for v in loro.values()):
            g.update(status='TEACHER_UNRESOLVED',reason='improvement_not_robust_to_omitting_one_accepted_recording')
    train_gate=len(accepted_train)>=32 and len({r['original_recording'] for r in accepted_train})>=2
    status=g['status'] if g['status']!='PILOT_GO' or train_gate else 'INSUFFICIENT_ACCEPTED_TRAIN'
    for row in outputs['TRAIN']:
        s=row['result'];pose=row['R0']
        csv_rows.append(dict(id=row['id'],split='TRAIN',recording=row['original_recording'],depth_available=row['depth_available'],
            status=s['status'],reason=s.get('reason'),accepted=s.get('accepted',False),
            z0_m=pose['centroid'][2] if pose['available'] else None,zD_m=s.get('zD'),
            zF_m=row['TEACHER_F_QD_RGBD']['centroid'][2] if row['TEACHER_F_QD_RGBD']['available'] else None))
    csv_path=C.DOC/'TEACHER_ROWS.csv';fields=list(dict.fromkeys(k for row in csv_rows for k in row))
    assert not csv_path.exists()
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(C.clean(csv_rows))
    path=C.DOC/'EVAL_ROWS.jsonl'
    C.save(path,''.join(__import__('json').dumps(C.clean(r),ensure_ascii=False,allow_nan=False)+'\n' for r in results))
    resource=dict(lock['counts'], actual_F_fixed_branch=0, student_fits=0, optimizer_updates=0,
        GPU_training_seconds=0, new_neural_forward_examples=0, teacher_seconds=lock['seconds'],score_seconds=time.monotonic()-start)
    C.save(C.DOC/'SUMMARY.json',dict(status=status,teacher_gate=g,student_training_eligible=g['status']=='PILOT_GO' and train_gate,
        eval_frames=128,depth_reference_frames=sum(r['depth_available'] for r in results),
        eval_accepted=len(accepted),eval_accepted_recordings=dict(Counter(r['original_recording'] for r in accepted)),
        train_candidates=len(outputs['TRAIN']),train_accepted=len(accepted_train),
        train_accepted_recordings=dict(Counter(r['original_recording'] for r in accepted_train)),
        abstentions={split:dict(Counter(r['result'].get('reason','accepted') for r in rows)) for split,rows in outputs.items()},
        groups=summary,twoD=two_d,paired_bootstrap={arm:bootstrap(results,arm) for arm in eval_arms[1:]},leave_one_recording_out=loro,
        GLOBAL_POST=dict(status='EVALUATED_NONLEARNING_CONTROL' if global_m is not None else 'NOT_RUN_NO_ACCEPTED_TRAIN',
                        train_only_log_z_median=global_m,scale=float(np.exp(global_m)) if global_m is not None else None,
                        train_frames=len(accepted_train),eval_depth_input=False,F_calls=0,optimizer_updates=0),
        direct_visible=dict(points=sum(len(vpoints.get(r['id'],[])) for r in results),
            good5_to_bad10=sum(e['good5_to_bad10'] for r in results for e in r['direct_visible_damage']['TEACHER_F_QD_RGBD'])),
        resources=resource,rule_searches=0,new_annotations=0,
        question_answer='RGB-only benefit cannot be claimed before all conditional student/control fits; teacher decision is separate.',
        reference_sources=[C.bind(P.TRUTH),C.bind(P.FINAL),C.bind(D.Pose.E.C.POSE/'GEOMETRY_RESOLVED_POSE_GT.json'),
                           C.bind(D.Pose.E.C.POSE/'AXIS_REVIEW_MANIFEST.json')]))
    if not (g['status']=='PILOT_GO' and train_gate):
        C.save(C.DOC/'TRAIN_RECEIPTS.json',dict(status='NOT_RUN_GATE_STOP',reason=status,
            arms={arm:dict(status='NOT_RUN',optimizer_updates=0,GPU_training_seconds=0,
                checkpoint=None,protected_state_exact=None,target_following=None) for arm in ('RAW_TARGET','GLOBAL_TARGET','DEPTH_TARGET')},
            fits=0,optimizer_updates=0,GPU_training_seconds=0,GLOBAL_POST='SEPARATE_NONLEARNING_CONTROL_IN_SUMMARY',
            student_failure_claim=False))
    print('PILOT_TEACHER_GATE',status,'train',len(accepted_train),'eval',len(accepted),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['lock','teacher','score']);args=p.parse_args()
    globals()[args.stage]()
