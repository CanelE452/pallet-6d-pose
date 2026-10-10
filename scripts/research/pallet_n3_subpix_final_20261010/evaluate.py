"""Same319, seed1 parity first, fixed three-seed actual final-F evaluation.

No detector/head inference, training, tuning, or original-input writes. The
existing frozen checkpoint outputs are authenticated by preflight.py first.
"""
from collections import Counter
import copy
import gzip
import json
import platform
import subprocess
import time

import cv2
import numpy as np
import torch

from . import common as C

OLD = '_docs/experiments/pallet_n3_subpix_20261008_v1'
GRADE = '_docs/experiments/pallet_combined_closeout_20261003_v1/closeout_20261006_v1/static/LABEL_PROVENANCE_AUDIT.json'
ALIASES = {'BASE': 'BASE', 'N3_DIM_SYM': 'N3', 'SUBPIX': 'SUBPIX', 'N3_THEN_SUBPIX': 'N3_SUBPIX'}


def same(a, b, name, tolerance=1e-7):
    """Compare all old metric fields; never adapt settings to observed results."""
    if isinstance(a, dict):
        assert isinstance(b, dict), name
        for k, v in a.items():
            assert k in b, (name, k)
            same(v, b[k], name + '/' + k, tolerance)
    elif isinstance(a, list):
        assert len(a) == len(b), name
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, name + '/' + str(i), tolerance)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert b is not None and np.isclose(a, b, rtol=0, atol=tolerance), (name, a, b)
    else:
        assert a == b, (name, a, b)


def score(E, frame, points, target, method, seed):
    actual = []
    infer = E.POSE.infer
    def capture(*args, **kwargs):
        value = infer(*args, **kwargs)
        actual.append(copy.deepcopy(value))
        return value
    E.POSE.infer = capture
    try:
        row = E.scored(frame, points, target, frame['captured']['captured']['selected_index'], method, seed)
    finally:
        E.POSE.infer = infer
    assert len(actual) == 1
    row.update(seed=seed, method=method, actual_pose=actual[0], pose_parameters_saved=True,
               accuracy_provenance='actual existing F called on this final coordinate array')
    pv = np.isfinite(points).all(-1) & ~(points == -1).all(-1) & bool(target['matched'])
    perm = np.asarray(target['permutations'][row['corner']['branch']], int)
    observed = np.zeros(9, bool)
    observed[perm] = pv
    observed &= np.asarray(target['valid'], bool)
    row['canonical_observed'] = observed[:8]
    # Evaluation-only fields are appended after F and never enter its inputs.
    row.update(evaluation_reference_points=target['gt'], evaluation_reference_valid=target['valid'],
               evaluation_permutation=perm, evaluation_reference_used_in_inference=False)
    return row


def main():
    started = time.monotonic()
    audit = C.read(C.DOC / 'INPUT_AUDIT.json')
    assert audit['status'] == 'PASS'
    assert not (C.DOC / 'PREDICTIONS.jsonl.gz').exists(), 'Preserve previous executions'
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    load_real, correct, cap_points = C.existing()
    E, frames, targets, baselines, baseline_path = load_real()
    ids = [f['id'] for f in frames]
    previous = C.read(C.ROOT / OLD / 'PROTOCOL.json')
    assert ids == previous['population']['frame_ids']
    bounds = {r['id']: r for r in previous['input_manifest']}
    old_rows = {(r['id'], r['method']): r for r in C.iter_rows(C.ROOT / OLD / 'PREDICTIONS.jsonl.gz')}
    grade_packet = C.read(C.ROOT / GRADE)
    grades = {r['id']: r['severity'] for r in grade_packet['rows'] if r['population'] == 'DEV319'}
    assert set(grades) == set(ids)
    assert Counter(grades.values()) == {'clean': 153, 'moderate': 92, 'severe': 74}
    n3 = {}
    for seed in (1, 2, 3):
        packet = C.read(C.ROOT / f'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/N3_DIM_SYM_seed{seed}.json')
        assert packet['complete'] and packet['GT_input'] is False
        n3[seed] = {r['id']: r for r in packet['records']}
        assert set(n3[seed]) == set(ids)
    # Seal every inference input and fixed method before any final-F call.
    locked_inputs = []
    for f in frames:
        b = bounds[f['id']]
        for key, value in [('q0', f['q']), ('prediction_support', f['point_valid']), ('raw_hw', f['raw_hw']),
                           ('K', f['K']), ('dimensions_whd_m', f['xyz'])]:
            assert C.digest(b[key]) == C.digest(value), (f['id'], key)
        captured = f['captured']['captured']
        for seed in (1, 2, 3):
            r = n3[seed][f['id']]
            assert r['selected_index'] == captured['selected_index']
            assert r['key'] == f['axis']['image']
            assert len(r['candidates']) == len(captured['candidates'])
            for k, (a, z) in enumerate(zip(captured['candidates'], r['candidates'])):
                assert C.digest({x: v for x, v in a.items() if x != 'keypoints_xy'}) == C.digest({x: v for x, v in z.items() if x != 'keypoints_xy'})
                if k != captured['selected_index']:
                    assert np.array_equal(a['keypoints_xy'], z['keypoints_xy'], equal_nan=True)
            q = np.asarray(r['candidates'][r['selected_index']]['keypoints_xy'], float)
            assert q.shape == (9, 2)
            assert np.array_equal(q[8], f['q'][8], equal_nan=True)
            assert np.array_equal(q[~f['point_valid']], f['q'][~f['point_valid']], equal_nan=True)
        locked_inputs.append(dict(id=f['id'], session=f['session'], grade=grades[f['id']], raw_hw=f['raw_hw'],
            K=f['K'], dimensions_pnp_WH_D_m=f['xyz'], dimensions_context_WD_H_m=f['captured']['dimensions'],
            canonical_symmetry_order=f['captured']['order'],
            q0=f['q'], prediction_support=f['point_valid'], selected_index=captured['selected_index'],
            candidates_metadata=[{k: v for k, v in a.items() if k != 'keypoints_xy'} for a in captured['candidates']],
            image_sha256=b['image_sha256'], cache_sha256=b['cache_sha256']))
    lock = dict(schema='pallet_n3_subpix_final_20261010', status='LOCKED', date='2026-10-10',
        remote_main_at_start=subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=C.WORKTREE, text=True).strip(),
        methods=list(C.METHODS), seeds=[1,2,3], population=dict(images=319, sessions=13, matched=311,
        reference_corners=2499, observed_corners=2445, grades=dict(Counter(grades.values())), frame_ids=ids),
        input_audit_sha256=C.sha(C.DOC / 'INPUT_AUDIT.json'), input_manifest=locked_inputs,
        input_manifest_sha256=C.digest(locked_inputs),
        methods_fixed=dict(N3='existing authenticated seed-specific cached checkpoint predictions; calibration temperature=1.0, lambda=1',
            SUBPIX=dict(winSize=[5,5], zeroZone=[-1,-1], maxCount=40, epsilon=.001, grayscale='uint8 raw resolution'),
            cap=dict(fraction=.01, reference='original BASE q0', applications_after_combination=1,
                     note='existing cached N3 stage has its historical float32 cap; do not allow an additional1% around qN'),
            F='unchanged prediction-only W/D hypothesis selection + SQPnP + RefineLM'),
        reference='geometry reconstructed; scoring only; not independent physical metrology',
        grade_semantics='existing annotation difficulty categories; physical external occlusion fraction NOT_CONFIRMED',
        parity=dict(all_images=319, coordinates='exact', metrics_absolute_tolerance=1e-7, pose_available_vectors_absolute_tolerance=1e-7),
        accuracy=dict(detector_calls=0, head_forwards=0, new_training_updates=0, baseline_actual_F_once_then_shared_across_seeds=True),
        environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, torch=torch.__version__,
                         threads=dict(opencv=1,torch=4), requires_gpu_for_cached_replay=False),
        runtime=dict(reused_existing_same_environment_seed1=True, source=OLD+'/RUNTIME.json', sha256=C.sha(C.ROOT/OLD/'RUNTIME.json'),
                     new_benchmark_calls=0),
        code=[dict(path=str(p.relative_to(C.WORKTREE)),sha256=C.sha(p)) for p in [__file__ and C.WORKTREE/'scripts/research/pallet_n3_subpix_final_20261010/evaluate.py', C.WORKTREE/'scripts/research/pallet_n3_subpix_final_20261010/common.py']])
    C.write(C.DOC / 'INPUT_AND_METHOD_LOCK.json', lock)
    calls = 0
    counts = Counter()
    rows_written = 0
    gray_decodes = 0
    parity = []
    shared = {}
    subpix_calls = Counter()
    pose_failures = []
    journal_path = C.DOC / 'EXECUTION_AND_VERIFICATION.json'
    path = C.DOC / 'PREDICTIONS.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as stream:
        for seed in (1, 2, 3):
            assert seed == 1 or len(parity) == 319, 'seed1 parity must pass before seed2/3'
            for i, f in enumerate(frames):
                assert time.monotonic()-started < 1800, 'Accuracy initial30-minute limit reached'
                q0 = f['q'].copy()
                r = n3[seed][f['id']]
                qN = np.asarray(r['candidates'][r['selected_index']]['keypoints_xy'], float)
                support = f['point_valid'].copy()
                h,w = f['raw_hw']
                bgr = cv2.imread(str(C.ROOT / f['axis']['image']), cv2.IMREAD_COLOR)
                assert bgr is not None and list(bgr.shape[:2]) == [h,w]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                gray_decodes += 1
                qS, diag = correct(gray, qN, support, 'SUBPIX')
                qFinal = cap_points(q0, qS, w, h, support)
                subpix_calls[f'combined_seed{seed}'] += diag['algorithm_corner_calls']
                if seed == 1:
                    sub_native, sub_diag = correct(gray, q0, support, 'SUBPIX')
                    qSub = cap_points(q0, sub_native, w, h, support)
                    subpix_calls['SUBPIX'] += sub_diag['algorithm_corner_calls']
                else:
                    qSub = np.asarray(shared[(f['id'],'SUBPIX')]['qFinal'], float)
                    sub_native = np.asarray(shared[(f['id'],'SUBPIX')]['qS'], float)
                    sub_diag = shared[(f['id'],'SUBPIX')]['correction']['diagnostics']
                usable = support[:8] & np.isfinite(q0[:8]).all(-1) & ~(q0[:8] == -1).all(-1)
                metadata = {k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')}
                digest_before = C.digest(metadata)
                frame_rows = []
                for method, points in [('BASE',q0),('N3_DIM_SYM',qN),('SUBPIX',qSub),('N3_THEN_SUBPIX',qFinal)]:
                    if seed > 1 and method in ('BASE','SUBPIX'):
                        row = copy.deepcopy(shared[(f['id'], method)])
                        row.update(seed=seed, F_attempt=False, reused_from_seed1=True,
                                   accuracy_provenance='same BASE/SUBPIX actual F result computed once in this execution; shared across N3 seeds')
                    else:
                        row = score(E,f,points,targets[f['id']],method,seed)
                        calls += 1
                        counts.update(row['PnP_counts'])
                    native_stage = qS if method == 'N3_THEN_SUBPIX' else sub_native if method == 'SUBPIX' else points
                    row.update(grade=grades[f['id']], q0=q0, qN=qN if method not in ('BASE','SUBPIX') else q0,
                               qS=native_stage, qFinal=points, prediction_support=support, raw_hw=[h,w],
                        fixed_metadata=dict(selected_index=r['selected_index'], candidate_metadata=bounds[f['id']]['candidate_metadata'],
                                            K=f['K'],dimensions_pnp_WH_D_m=f['xyz'],canonical_symmetry_order=f['captured']['order'],preserved=True))
                    if method in ('SUBPIX','N3_THEN_SUBPIX'):
                        starting = qN if method == 'N3_THEN_SUBPIX' else q0
                        diagnostics = diag if method == 'N3_THEN_SUBPIX' else sub_diag
                        total = np.linalg.norm(native_stage[:8]-q0[:8],axis=-1)
                        row['correction'] = dict(diagnostics=diagnostics,cap_px=.01*np.hypot(w,h),
                            total_before_cap_px8=total,total_final_px8=np.linalg.norm(points[:8]-q0[:8],axis=-1),
                            additional_subpix_px8=np.linalg.norm(native_stage[:8]-starting[:8],axis=-1),cap_active8=(total>.01*np.hypot(w,h))&usable)
                    assert np.array_equal(points[8],q0[8],equal_nan=True)
                    assert np.array_equal(points[~support],q0[~support],equal_nan=True)
                    if method in ('SUBPIX','N3_THEN_SUBPIX'):
                        assert np.max(np.linalg.norm(points[:8]-q0[:8],axis=-1)[usable],initial=0)<=.01*np.hypot(w,h)+1e-10
                    assert digest_before == C.digest({k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')})
                    if seed == 1:
                        old = old_rows[(f['id'],ALIASES[method])]
                        assert np.array_equal(points,np.asarray(old['qFinal'],float),equal_nan=True), (f['id'],method,'coordinates')
                        same(old['corner'],C.finite(row['corner']),f['id']+'/'+method+'/corner')
                        same(old['pose'],C.finite(row['pose']),f['id']+'/'+method+'/metric')
                        assert old['final_hypothesis'] == row['final_hypothesis']
                        if old.get('actual_pose') is not None:
                            same(old['actual_pose'],C.finite(row['actual_pose']),f['id']+'/'+method+'/actual_pose')
                        if method == 'N3_THEN_SUBPIX':
                            for key in ('q0','qN','qS','qFinal'):
                                assert np.array_equal(row[key],np.asarray(old[key],float),equal_nan=True), (f['id'],key)
                        if method in ('BASE','SUBPIX'):
                            shared[(f['id'],method)] = copy.deepcopy(row)
                    if not row['pose']['available']:
                        pose_failures.append(dict(seed=seed,method=method,id=f['id'],status='no_pose'))
                    frame_rows.append(row)
                if seed == 1:
                    parity.append(dict(id=f['id'],coordinates_exact=True,all4_metrics_PASS=True,
                                       actual_pose_comparison='all available old R/t vectors; BASE/SUBPIX old vectors available on26-frame panel'))
                for row in frame_rows:
                    stream.write(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                    rows_written += 1
                if i%40 == 0 or i == 318:
                    stream.flush()
                    C.write(journal_path,dict(status='RUNNING',seed=seed,seed_frames=i+1,actual_F_calls=calls,rows_written=rows_written,
                        seed1_parity_frames=len(parity),PnP_counts=dict(counts),elapsed_seconds=time.monotonic()-started))
                    print('EVALUATION',seed,i+1,319,'F',calls,'seconds',round(time.monotonic()-started,2),flush=True)
            if seed == 1:
                C.write(C.DOC/'SEED1_PARITY.json',dict(status='PASS',images=319,methods=4,rows=parity,coordinates_exact=True,
                    metrics_absolute_tolerance=1e-7,old_rows_sha256=C.sha(C.ROOT/OLD/'PREDICTIONS.jsonl.gz'),
                    input_manifest_all319_exact=True,actual_R_t_coverage=dict(BASE=26,SUBPIX=26,N3_DIM_SYM=319,N3_THEN_SUBPIX=319),
                    note='Previous BASE/SUBPIX vectors did not exist beyond26; all319 coordinates, hypotheses and errors compared; new actual vectors saved for all319'))
                print('SEED1_ALL319_PARITY_PASS',flush=True)
    C.write(journal_path,dict(status='ACCURACY_COMPLETE',complete=True,images=319,sessions=13,seeds=[1,2,3],rows=rows_written,
        actual_F_calls=calls,expected_actual_F_calls=2552,shared_BASE_SUBPIX_rows=1276,
        PnP_counts=dict(counts),OpenCV_corner_calls=dict(subpix_calls),gray_decodes=gray_decodes,
        seed1_parity='PASS',seed1_parity_sha256=C.sha(C.DOC/'SEED1_PARITY.json'),
        coordinates_center_mask_metadata_preserved=True,total_cap_checked=True,GT_inference_inputs=False,
        no_pose=pose_failures,detector_calls=0,N3_forwards=0,new_training_updates=0,new_synthetic_images=0,
        runtime_reused=True,new_runtime_calls=0,raw_rows_sha256=C.sha(path),elapsed_seconds=time.monotonic()-started,
        published_raw_rows='numeric coordinates, R/t and evaluation errors only; no RGB, annotation paths or weights'))
    print('ACCURACY_COMPLETE',rows_written,calls,round(time.monotonic()-started,2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Failure text is private: it may contain a local dependency pathname.
        import os
        from pathlib import Path
        private = Path(os.environ.get('PALLET_PRIVATE_OUTPUT', '/dev/shm/pallet-n3-subpix-final-private'))
        C.write(private/'ERROR.json',dict(status='BLOCKED',type=type(e).__name__,error=str(e)))
        raise
