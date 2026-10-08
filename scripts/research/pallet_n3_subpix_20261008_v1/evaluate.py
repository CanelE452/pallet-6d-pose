"""Cached accuracy inputs, bounded parity, and 319 actual combined F calls."""
from collections import Counter
import copy
import gzip
import inspect
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from . import common as C
from .test_contracts import run as functional_checks
from scripts.research.pallet_training_free_compare_20261007_v1.evaluate import same_numeric
from scripts.research.pallet_training_free_compare_20261007_v1.reporting import canonical_observed

OLD = C.ROOT / '_docs/experiments/pallet_training_free_compare_20261007_v1'
N3_PATH = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
N3_POSES = C.ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/pose_predictions/REAL_DEV/N3_DIM_SYM_seed1.json'
PANEL = C.ROOT / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
COUNT_KEYS = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM')


def binding(path, label=None):
    path = Path(path)
    if label is None:
        label = str(path.relative_to(C.ROOT)) if path.is_relative_to(C.ROOT) else path.name
    return dict(path=label, sha256=C.sha(path), bytes=path.stat().st_size)


def score(E, frame, points, target, arm):
    """Capture the single existing F call; score only after fixed coordinates."""
    actual = []
    infer = E.POSE.infer
    def captured(*args, **kwargs):
        pose = infer(*args, **kwargs); actual.append(copy.deepcopy(pose)); return pose
    E.POSE.infer = captured
    try:
        row = E.scored(frame, points, target, frame['captured']['captured']['selected_index'], arm, 1)
    finally:
        E.POSE.infer = infer
    assert len(actual) == 1
    row.update(method=arm, actual_pose=actual[0], pose_parameters_saved=True)
    # Only the independent scoring stage maps prediction errors to canonical GT.
    pv = np.isfinite(points).all(-1) & ~(points == -1).all(-1) & bool(target['matched'])
    observed = np.zeros(9, bool)
    permutation = np.asarray(target['permutations'][row['corner']['branch']], int)
    observed[permutation] = pv
    observed &= np.asarray(target['valid'], bool)
    row['canonical_observed'] = observed[:8]
    return row


def prepare(E, frames, targets, baselines, baseline_path):
    assert not (C.DOC / 'PROTOCOL.json').exists(), 'Preserve prior executions'
    previous = C.read(OLD / 'PROTOCOL.json')
    old_inputs = {v['id']: v for v in previous['input_manifest']}
    n3_packet = C.read(N3_PATH)
    assert n3_packet['complete'] and n3_packet['GT_input'] is False
    n3 = {r['id']: r for r in n3_packet['records']}
    ids = [f['id'] for f in frames]
    assert len(n3) == 319 and set(n3) == set(ids)
    inputs = []; cap_excess = []
    for f in frames:
        old = old_inputs[f['id']]; r = n3[f['id']]
        cap = f['captured']['captured']; selected = cap['selected_index']
        assert selected is not None and selected == r['selected_index']
        assert r['key'] == f['axis']['image']
        assert len(r['candidates']) == len(cap['candidates'])
        for a, b in zip(cap['candidates'], r['candidates']):
            assert C.digest({k:v for k,v in a.items() if k != 'keypoints_xy'}) == C.digest({k:v for k,v in b.items() if k != 'keypoints_xy'})
        qN = np.asarray(r['candidates'][selected]['keypoints_xy'], np.float64)
        assert qN.shape == (9,2)
        assert np.array_equal(qN[8], f['q'][8], equal_nan=True)
        assert np.array_equal(qN[~f['point_valid']], f['q'][~f['point_valid']], equal_nan=True)
        assert np.isfinite(qN[f['point_valid']]).all()
        for key, value in [('initial_points', f['q']), ('prediction_support', f['point_valid']), ('raw_hw', f['raw_hw']), ('K', f['K']), ('dimensions_whd_m', f['xyz'])]:
            assert C.digest(old[key]) == C.digest(value), (f['id'], key)
        image = C.ROOT / old['image']; cache = C.ROOT / old['cache']
        assert C.sha(image) == old['image_sha256']
        assert C.sha(cache) == old['cache_sha256']
        h,w = f['raw_hw']; d = np.linalg.norm(qN[:8]-f['q'][:8], axis=-1)
        cap_excess.extend((d[d > .01*np.hypot(w,h)+1e-10]-.01*np.hypot(w,h)).tolist())
        inputs.append(dict(id=f['id'], session=f['session'], image_sha256=old['image_sha256'],
            cache_sha256=old['cache_sha256'], raw_hw=f['raw_hw'], K=f['K'], dimensions_whd_m=f['xyz'],
            q0=f['q'], qN=qN, prediction_support=f['point_valid'], selected_index=selected,
            candidate_metadata=old['candidate_metadata']))
    # Changes elsewhere since the prior protocol do not invalidate this comparison.
    for entry in previous['code'] + previous['dependencies']:
        assert C.sha(C.ROOT / entry['path']) == entry['sha256'], entry['path']
    assert C.sha(C.ROOT / n3_packet['checkpoint']['path']) == n3_packet['checkpoint']['sha256']
    assert C.sha(OLD / 'PREDICTIONS.jsonl.gz') == C.read(OLD / 'PREDICTIONS.json')['raw_rows_sha256']
    panel = C.read(PANEL)['selected']
    assert len(panel) == 26 and len({p['session_id'] for p in panel}) == 13
    assert all(p['frame_id'] in ids for p in panel)
    initial = C.read(C.OUTPUT / 'INITIAL_STATE.json')
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/main'], cwd=C.WORKTREE, text=True).split()[0]
    from scripts.research.pallet_training_free_compare_20261007_v1.methods import method_configuration
    cfg = method_configuration()['SUBPIX']
    deps = [binding(N3_PATH), binding(N3_POSES), binding(PANEL), binding(baseline_path, 'immutable_a22/A_REAL_DEV_BASELINES.json'),
        binding(OLD/'PROTOCOL.json'), binding(OLD/'PREDICTIONS.jsonl.gz'), binding(C.ROOT / n3_packet['checkpoint']['path'])]
    for p in [C.ROOT / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json',
              C.ROOT / 'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json',
              C.ROOT / 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',
              Path(E.POSE.__file__), Path(E.M.__file__)]:
        deps.append(binding(p))
    n3temp = C.read(C.ROOT / '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json')['temperatures']['N3_DIM_SYM_seed1']['temperature']
    protocol = dict(schema='pallet_n3_subpix_20261008_v1', date='2026-10-08',
        base_commit=initial['head'], investigated_main='4e4e3a5d201298a00447419d69d949d86878aafc', remote_main_at_start=remote,
        branch='research/n3-subpix-20261008', main_publication='explicit final user request overrides earlier no-main rule',
        methods=list(C.ARMS), N3_seed=1, N3_temperature=n3temp, N3_checkpoint=n3_packet['checkpoint'],
        SUBPIX=cfg, cap=dict(fraction=.01, reference='original Base q0 and raw image diagonal', arithmetic='float64'),
        population=dict(images=319, sessions=13, matched=311, reference_corners=2499, observed_corners=2445, frame_ids=ids),
        input_manifest=inputs, input_manifest_sha256=C.digest(inputs), dependencies=deps,
        code=[binding(p, str(p.relative_to(C.WORKTREE))) for p in [Path(__file__), Path(C.__file__), Path(__file__).with_name('test_contracts.py')]],
        parity=dict(ids=[p['frame_id'] for p in panel], absolute_tolerance=1e-7, F_calls_ceiling=78),
        evaluation=dict(final_F='existing prediction-only W/D hypothesis selection + SQPnP + LM, once per combined frame',
            reference='existing geometry-reconstructed reference; not independent measured physical GT',
            accuracy_cache_reuse=True, GT_selection=False, bootstrap=dict(units=13,resamples=10000,seed=20260917,shared_draws=True),
            std='sample error dispersion over corners/frames, ddof=1; not confidence interval or seed dispersion'),
        runtime=dict(panel_ids=[p['frame_id'] for p in panel], warmup_each=20,repeats=5, measured_each=130, pipelines_ceiling=600,
            boundary='RAM raw BGR -> actual YOLO and same selection -> correction -> single final F; loading/decoding excluded'),
        scope=dict(new_training=0,extra_seeds=0,reverse_route=0,tuning=0,new_gates=0,new_annotations=0,paper_changes=0),
        budget=dict(combined_accuracy_F=319,parity_F=78,runtime_F=600,active_evaluation_runtime_seconds=1800),
        input_checks=dict(all_319_image_cache_hashes_match=True,previous_evaluation_code_dependencies_match=True,
            native_numbering=True, all_candidate_metadata_preserved=True, center_mask_preserved=True),
        historical_N3_cap_rounding=dict(exceeding_corners=len(cap_excess), maximum_excess_px=max(cap_excess,default=0),
            explanation='original float32 N3 contract retained; combined strict cap may remove tiny rounding excess'),
        user_changes_at_start=dict(tracked_sha256=initial['tracked_sha256'], tracked_diff_sha256=initial['tracked_diff_sha256'],
            original_status_sha256=initial['status_sha256']), private_inputs_copied=False)
    C.write(C.DOC/'PROTOCOL.json', protocol)
    return protocol, n3


def main():
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (C.DOC/'PREDICTIONS.jsonl.gz').exists(), 'No outcome-based reruns'
    cv2.setNumThreads(1); torch.set_num_threads(4)
    started = time.monotonic()
    E, frames, targets, baselines, baseline_path = C.load_real()
    protocol, n3 = prepare(E, frames, targets, baselines, baseline_path)
    checks = dict(functional=functional_checks(), input=protocol['input_checks'], status='IN_PROGRESS')
    old_sub = {r['id']:r for r in C.iter_rows(OLD/'PREDICTIONS.jsonl.gz') if r['method']=='SUBPIX_CAP1'}
    old_native = {r['id']:r for r in C.iter_rows(OLD/'PREDICTIONS.jsonl.gz') if r['method']=='SUBPIX_NATIVE'}
    reused = {'BASE':{r['id']:r for r in baselines['rows']['RAW']},
              'N3':{r['id']:r for r in baselines['rows']['N3_seed1']}, 'SUBPIX':old_sub}
    n3poses = C.read(N3_POSES)['records']
    bounds = {r['id']:r for r in protocol['input_manifest']}
    frame_map = {f['id']:f for f in frames}
    parity = []; counts = Counter(); parity_poses = {}
    C.write(C.OUTPUT/'EXECUTION.json',dict(stage='PARITY',F_attempts=0,completed=0))
    for fid in protocol['parity']['ids']:
        f = frame_map[fid]; qN = np.asarray(bounds[fid]['qN'], float)
        bgr = cv2.imread(str(C.ROOT / f['axis']['image']), cv2.IMREAD_COLOR)
        assert bgr is not None and list(bgr.shape[:2]) == list(f['raw_hw'])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        native, diagnostics = C.correct(gray, f['q'], f['point_valid'], 'SUBPIX')
        sub = C.cap_points(f['q'], native, gray.shape[1], gray.shape[0], f['point_valid'])
        assert np.array_equal(sub, np.asarray(old_sub[fid]['native_points']))
        for arm, points in [('BASE',f['q']),('N3',qN),('SUBPIX',sub)]:
            C.write(C.OUTPUT/'EXECUTION.json',dict(stage='PARITY',F_attempts=len(parity)+1,completed=len(parity),last_id=fid,method=arm))
            row = score(E,f,points,targets[fid],arm); counts.update(row['PnP_counts'])
            same_numeric(reused[arm][fid]['corner'], C.finite(row['corner']), fid+'/'+arm+'/corner')
            same_numeric(reused[arm][fid]['pose'], C.finite(row['pose']), fid+'/'+arm+'/pose')
            assert row['final_hypothesis'] == reused[arm][fid]['final_hypothesis']
            if arm == 'N3': same_numeric(n3poses[fid], C.finite(row['actual_pose']), fid+'/N3/actual_pose')
            parity_poses[(fid,arm)] = row['actual_pose']
            parity.append(dict(id=fid,method=arm,PASS=True,PnP_counts=row['PnP_counts']))
    checks['parity'] = dict(PASS=True,F_calls=len(parity),rows=parity,absolute_tolerance=1e-7,PnP_counts=dict(counts))
    C.write(C.DOC/'CHECKS.json', checks)
    print('MINIMUM_CHECKS_PASS', len(parity), flush=True)
    path=C.DOC/'PREDICTIONS.jsonl.gz'; combined_calls=0; algo_calls=0; combo_counts=Counter()
    fallback_cap_rounding = []; fallback_corners = 0
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as stream:
        for i,f in enumerate(frames):
            assert time.monotonic()-started < 1800, 'Initial active execution budget reached'
            bound=bounds[f['id']]; q0=f['q'].copy(); qN=np.asarray(bound['qN'],np.float64)
            support=f['point_valid'].copy(); h,w=f['raw_hw']; cap=.01*np.hypot(w,h)
            bgr=cv2.imread(str(C.ROOT/f['axis']['image']),cv2.IMREAD_COLOR)
            assert bgr is not None and list(bgr.shape[:2])==[h,w], f['id']
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            frozen_metadata = {k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')}
            before=C.digest(frozen_metadata)
            qS,qFinal,diagnostics=C.combine(gray,q0,qN,support)
            algo_calls+=diagnostics['algorithm_corner_calls']
            usable=support[:8]&np.isfinite(q0[:8]).all(-1)&~(q0[:8]==-1).all(-1)
            total_before=np.linalg.norm(qS[:8]-q0[:8],axis=-1)
            total_final=np.linalg.norm(qFinal[:8]-q0[:8],axis=-1)
            assert np.max(total_final[usable],initial=0)<=cap+1e-10
            assert np.array_equal(qFinal[8],q0[8],equal_nan=True)
            assert np.array_equal(qFinal[~support],q0[~support],equal_nan=True)
            for d in diagnostics['corner_records']:
                if d['status']!='refined':
                    k=d['corner']; fallback_corners+=1
                    assert np.array_equal(qS[k],qN[k],equal_nan=True)
                    if not np.array_equal(qFinal[k],qN[k],equal_nan=True):
                        fallback_cap_rounding.append(dict(id=f['id'],corner=k,
                            final_minus_N3_px=float(np.linalg.norm(qFinal[k]-qN[k]))))
            C.write(C.OUTPUT/'EXECUTION.json',dict(stage='ACCURACY',combined_F_attempts=combined_calls+1,combined_F_completed=combined_calls,last_id=f['id'],parity_F=len(parity)))
            combined=score(E,f,qFinal,targets[f['id']],'N3_SUBPIX');combined_calls+=1
            combo_counts.update(combined['PnP_counts'])
            assert before==C.digest({k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')})
            combined.update(q0=q0,qN=qN,qS=qS,qFinal=qFinal,
                correction=dict(diagnostics=diagnostics,cap_px=cap,
                    total_before_cap_px8=total_before,total_final_px8=total_final,
                    additional_subpix_px8=np.linalg.norm(qS[:8]-qN[:8],axis=-1),cap_active8=(total_before>cap)&usable))
            for arm in C.ARMS:
                if arm=='N3_SUBPIX': row=combined
                else:
                    row=copy.deepcopy(reused[arm][f['id']]);row['method']=arm
                    row['canonical_observed']=canonical_observed(row)
                    points=q0 if arm=='BASE' else qN if arm=='N3' else np.asarray(old_sub[f['id']]['native_points'],float)
                    row.update(native_points=points,qFinal=points,F_attempt=False,F_complete=True,
                        accuracy_provenance='existing actual F result reused; 26-frame same-F parity passed')
                    row['actual_pose']=n3poses[f['id']] if arm=='N3' else parity_poses.get((f['id'],arm))
                    row['pose_parameters_saved']=row['actual_pose'] is not None
                    if arm=='SUBPIX':row['qS']=old_native[f['id']]['native_points']
                row.update(prediction_support=support,raw_hw=[h,w],
                    fixed_metadata=dict(selected_index=bound['selected_index'],candidate_metadata=bound['candidate_metadata'],preserved=True))
                stream.write(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
            if i%40==0 or i==318:
                stream.flush();print('ACCURACY',i+1,319,'combined_F',combined_calls,'seconds',round(time.monotonic()-started,2),flush=True)
    checks.update(status='PASS',complete=True,combined=dict(frames=319,center_mask_metadata_preserved=True,
        strict_Base_total_cap=True,SUBPIX_failure_native_qN_preserved=True,GT_inference_inputs=False,
        actual_pose_captured_from_single_F=True),
        code_signature=list(inspect.signature(C.combine).parameters),
        execution=dict(combined_accuracy_F=combined_calls,parity_F=len(parity),accuracy_PnP_counts=dict(combo_counts),
            parity_PnP_counts=dict(counts),combined_OpenCV_corner_calls=algo_calls,combined_image_calls=319,
            parity_SUBPIX_image_calls=26,parity_OpenCV_corner_calls=sum(old_sub[fid]['correction']['diagnostics']['algorithm_corner_calls'] for fid in protocol['parity']['ids']),
            detector_accuracy_calls=0,N3_accuracy_forwards=0,training_updates=0,
            cached_F_rows_reused=957,rows=1276,elapsed_seconds=time.monotonic()-started),
        raw_rows_sha256=C.sha(path),
        fallback=dict(corners_retaining_qN_native=fallback_corners,
            strict_cap_rounding_adjustments=fallback_cap_rounding,
            explanation='Failure retains qN as qS; final q0-based strict cap also removes historical float32 N3 cap excess. No fallback to Base.'),
        pose_parameter_cache_note='N3 all319 and new combined319 saved. BASE/SUBPIX parameter vectors saved only on parity26; other rows reuse existing actual F metric cache, not reconstructed/copied N3 poses.')
    C.write(C.DOC/'CHECKS.json',checks)
    print('ACCURACY_COMPLETE', C.finite(checks['execution']),flush=True)


if __name__=='__main__':
    try: main()
    except Exception as error:
        C.write(C.DOC/'INCOMPLETE.json',dict(complete=False,error_type=type(error).__name__,error=str(error),
            execution=C.read(C.OUTPUT/'EXECUTION.json') if (C.OUTPUT/'EXECUTION.json').exists() else {},
            instruction='Preserve partial rows; no silent rerun or tolerance relaxation'))
        raise
