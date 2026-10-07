"""Fixed 319-frame accuracy replay: zero detector calls, four actual F calls/frame."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
import torch
from .common import ROOT, DOC, OUTPUT, ARMS, BASE_SHA, read, write, sha, digest, finite, load_real
from .methods import correct, cap_points, method_configuration

PANEL = ROOT / 'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json'
NEW_CODE = Path(__file__).parent


def array_json(value):
    return np.array(value, dtype=float)


def same_numeric(a, b, path='', tolerance=1e-7):
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        for k in a:
            if k in b:
                same_numeric(a[k], b[k], path + '/' + k, tolerance)
            else:
                raise AssertionError('Missing old key: ' + path + '/' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same_numeric(x, y, path + '/' + str(i), tolerance)
    elif isinstance(a, (float, int)) and not isinstance(a, bool):
        assert b is not None and np.isclose(a, b, rtol=0, atol=tolerance), (path, a, b)
    else:
        assert a == b, (path, a, b)


def prepare():
    """Seal prediction inputs/reference/code before any new final-pose call."""
    assert not (DOC / 'PROTOCOL.json').exists(), 'Do not replace the pre-evaluation lock'
    E, frames, targets, baselines, baseline_path = load_real()
    config = method_configuration()
    inputs = []
    for frame in frames:
        cap = frame['captured']['captured']
        selected = cap['selected_index']
        candidate = None if selected is None else cap['candidates'][selected]
        assert candidate is None or np.array_equal(frame['q'], candidate['keypoints_xy'], equal_nan=True)
        image = ROOT / frame['axis']['image']
        cache = ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1/DEV_cache' / (frame['axis']['frame_id'] + '.pt')
        inputs.append(dict(id=frame['id'], session=frame['session'], image=str(image.relative_to(ROOT)),
            image_sha256=sha(image), raw_hw=frame['raw_hw'], cache=str(cache.relative_to(ROOT)),
            cache_sha256=sha(cache), initial_points=frame['q'], prediction_support=frame['point_valid'],
            selected_index=selected, candidate_metadata=None if candidate is None else
                {k: v for k, v in candidate.items() if k != 'keypoints_xy'},
            K=frame['K'], dimensions_whd_m=frame['xyz']))
    panel = read(PANEL)['selected']
    ids = [f['id'] for f in frames]
    assert len(panel) == 26 and len({r['session_id'] for r in panel}) == 13
    assert all(r['frame_id'] in ids for r in panel)
    from scripts.research.pallet_pose_target_6d_20261006_v1.baseline import BASELINE_ROOT
    dependencies = []
    relatives = ['scripts/research/pallet_dim_conditioned_p_v1/pose.py',
                 'scripts/research/pallet_dim_conditioned_p_v1/eval_math.py',
                 'data/pallet/results/paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json',
                 'data/pallet/results/paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json']
    for relative in relatives:
        assert sha(ROOT / relative) == sha(BASELINE_ROOT / relative), relative
        dependencies.append(dict(path=relative, sha256=sha(ROOT / relative), main_a22_exact_bytes=True))
    relative='data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json'
    dependencies.append(dict(path=relative,sha256=sha(ROOT/relative),
        old_baseline_target_sha256_exact=baselines['target_sha256']==sha(ROOT/relative)))
    for name in ('N3_DIM_SYM',):
        for seed in (1, 2, 3):
            relative = f'data/pallet/results/pallet_dim_conditioned_p_v1/predictions/REAL_DEV/{name}_seed{seed}.json'
            dependencies.append(dict(path=relative, sha256=sha(ROOT / relative)))
    for seed in (1, 2, 3):
        relative = f'data/pallet/results/pallet_posefix_replay_diagnosis_v1/predictions/seed{seed}_REAL_DEV.npz'
        dependencies.append(dict(path=relative, sha256=sha(ROOT / relative)))
    initial = read(OUTPUT / 'INITIAL_STATE.json')
    current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT).decode().split()[0]
    assert current == remote == BASE_SHA
    protocol = dict(schema='pallet_training_free_compare_20261007_v1', base_commit=BASE_SHA,
        branch='main', remote_at_start=remote, created_unix=time.time(),
        purpose='Test whether fixed corrections without additional training replace the N3 accuracy gain',
        scope=dict(primary='REAL_DEV319 rectangular, 13 sessions', new_training=0, new_data=0,
            new_annotations=0, lifter_analysis=0, paper_latex_pdf_bibliography_changes=0,
            tuning_after_outcomes=False, initial_detector_accuracy_calls=0),
        population=dict(images=319, sessions=13, reference_corners=2499, matched_images=311,
            observed_corners=2445, frame_ids=ids), arms=list(ARMS), methods=config,
        input_manifest=finite(inputs), input_manifest_sha256=digest(inputs),
        baseline=dict(path=str(baseline_path), sha256=sha(baseline_path),
            original_commit='a22fb14beb5e8df08076385000e0d53503c1ae29',
            rows_reused=list(baselines['rows']), BASE_is_stored_RAW=True,
            baseline_root=str(BASELINE_ROOT), linked_N0='NOT_LINKED: not present in the same saved baseline packet'),
        dependencies=dependencies,
        evaluation=dict(corner='existing full-object symmetry branch; canonical-identity aligned',
            pose='unchanged prediction-only WD hypotheses + SQPnP + refinement, actual final F(q)',
            pose_reference='existing reconstructed geometry reference, not independent measured range',
            ADD='ADDsym over 8 cuboid corners/proper rotation group; not surface ADD-S',
            normalized_ADD_AUC='existing threshold0..0.1 /1001 trapezoidal curve, full319 including failures',
            failures='full319 denominator retained; successful-pose med/P90 labeled conditional',
            good_to_bad='canonical error before<5 and after>10px', bad_to_good='before>20 and after<10px',
            bootstrap=dict(resamples=10000, seed=20260917, units=13, shared_session_draws=True,
                statistic='paired frame-pooled mean; paired median separately; difference of medians separately'),
            visibility='existing frozen human states used only after predictions; unknown/unannotated distinct'),
        parity=dict(fixed_panel_ids=[r['frame_id'] for r in panel], calls=26,
            compare='all saved BASE corner/pose fields at absolute1e-7; no new baseline319 rerun'),
        runtime=dict(panel_path=str(PANEL.relative_to(ROOT)), panel_sha256=sha(PANEL),
            panel_ids=[r['frame_id'] for r in panel], routes=['BASE','N3_seed1','PoseFix_seed1',*ARMS],
            warmup_each=20, measured_panel_repeats=5, full_pipeline_ceiling=1050,
            boundary='RAM raw image through initial YOLO, same selection, correction, final F; decode/load excluded',
            correction_only='synchronized correction stages from the same complete pipeline; initial/F excluded',
            accuracy_replay_is_end_to_end=False),
        budget=dict(new_F_accuracy=1276, new_F_BASE_parity=26, new_F_runtime_ceiling=1050,
            new_F_total_ceiling=2352, optional_square_2D_ceiling=476, square_F=0,
            initial_operational_minutes=60, time_limit_applies='evaluation and measurement processes',
            extra_methods=0, CV_grid_search=0, learned_accuracy_forwards=0),
        user_changes_at_start=initial['tracked_sha256'],
        original_files_preserved=True, final_artifacts_compressed=True,
        code=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in
              (NEW_CODE/'common.py', NEW_CODE/'methods.py', Path(__file__))])
    DOC.mkdir(parents=True, exist_ok=True)
    write(DOC / 'PROTOCOL.json', protocol)
    print('PROTOCOL_SEALED', sha(DOC/'PROTOCOL.json'), len(inputs), flush=True)


def score(E, frame, points, target, arm):
    row = E.scored(frame, points, target, frame['captured']['captured']['selected_index'], arm, 0)
    row['method'] = arm
    pv = np.isfinite(points).all(-1) & ~(points == -1).all(-1) & bool(target['matched'])
    canonical_observed = np.zeros(9, dtype=bool)
    permutation = np.array(target['permutations'][row['corner']['branch']], dtype=int)
    canonical_observed[permutation] = pv
    canonical_observed &= np.array(target['valid'], dtype=bool)
    row['canonical_observed'] = canonical_observed[:8]
    return row


def parity(E, frames, targets, baselines, protocol):
    assert not (DOC / 'BASE_PARITY.json').exists(), 'BASE parity is bounded to one panel run'
    by_id = {f['id']: f for f in frames}
    old = {r['id']: r for r in baselines['rows']['RAW']}
    saved = []
    counts = dict(solvePnP=0, solvePnPGeneric=0, solvePnPRefineLM=0)
    for fid in protocol['parity']['fixed_panel_ids']:
        f = by_id[fid]
        write(OUTPUT / 'F_JOURNAL.json', dict(stage='BASE_PARITY', F_attempts=len(saved)+1, last_id=fid))
        row = score(E, f, f['q'], targets[fid], 'BASE')
        same_numeric(old[fid]['corner'], row['corner'], fid+'/corner')
        same_numeric(old[fid]['pose'], row['pose'], fid+'/pose')
        assert old[fid]['final_hypothesis'] == row['final_hypothesis']
        for k in counts:
            counts[k] += row['PnP_counts'][k]
        saved.append(finite(row))
    write(DOC / 'BASE_PARITY.json', dict(PASS=True, calls=26, new_detector_calls=0,
        full_metric_tolerance_absolute=1e-7, rows=saved, PnP_counts=counts,
        protocol_sha256=sha(DOC/'PROTOCOL.json')))
    print('BASE_PARITY_PASS', len(saved), flush=True)


def evaluate():
    protocol = read(DOC / 'PROTOCOL.json')
    assert read(DOC / 'CHECKS_METHODS.json')['status']=='PASS'
    assert not (DOC / 'PREDICTIONS.json').exists(), 'Do not silently replace completed results'
    assert not (OUTPUT / 'ACCURACY_STARTED.json').exists(), 'Interrupted attempts require explicit accounting'
    for binding in protocol['code']:
        assert sha(ROOT / binding['path']) == binding['sha256'], binding['path']
    started = time.monotonic()
    cv2.setNumThreads(1); torch.set_num_threads(4)
    E, frames, targets, baselines, _ = load_real()
    assert [f['id'] for f in frames] == protocol['population']['frame_ids']
    by_id = {r['id']: r for r in protocol['input_manifest']}
    for f in frames:
        bound = by_id[f['id']]
        assert digest(f['q']) == digest(bound['initial_points'])
        assert digest(f['point_valid']) == digest(bound['prediction_support'])
    write(OUTPUT / 'ACCURACY_STARTED.json', dict(time_unix=time.time(), protocol_sha256=sha(DOC/'PROTOCOL.json')))
    parity(E, frames, targets, baselines, protocol)
    path = DOC / 'PREDICTIONS.jsonl.gz'
    counts = dict(solvePnP=0, solvePnPGeneric=0, solvePnPRefineLM=0)
    F_calls=0; rows_written=0; gray_decodes=0
    native_algorithm_image_calls={'SUBPIX':0,'CVRANK':0}
    saved = []
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as stream:
        for i, frame in enumerate(frames):
            assert time.monotonic()-started < 3600, 'Initial operational limit reached; preserve partial rows'
            bound = by_id[frame['id']]
            image_path = ROOT / bound['image']
            assert sha(image_path) == bound['image_sha256']
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            assert bgr is not None and list(bgr.shape[:2]) == list(frame['raw_hw'])
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); gray_decodes+=1
            initial = np.array(frame['q'], dtype=float, copy=True)
            support = np.array(frame['point_valid'], dtype=bool, copy=True)
            h,w = gray.shape
            candidate = frame['captured']['captured']['candidates'][bound['selected_index']]
            before = digest(candidate)
            for algorithm in ('SUBPIX', 'CVRANK'):
                native, diagnostics = correct(gray, initial, support, algorithm)
                native_algorithm_image_calls[algorithm]+=1
                assert np.array_equal(initial, frame['q'], equal_nan=True)
                assert np.array_equal(native[8], initial[8], equal_nan=True)
                for suffix in ('NATIVE','CAP1'):
                    arm = algorithm+'_'+suffix
                    points = native.copy() if suffix=='NATIVE' else cap_points(initial,native,w,h,support)
                    assert np.array_equal(points[8], initial[8], equal_nan=True)
                    assert np.array_equal(points[~support], initial[~support], equal_nan=True)
                    usable=support[:8]&np.isfinite(initial[:8]).all(-1)&~(initial[:8]==-1).all(-1)
                    delta=np.linalg.norm(points[:8]-initial[:8],axis=-1)
                    if suffix=='CAP1':
                        assert np.max(delta[usable],initial=0)<=.01*np.hypot(w,h)+1e-10
                    write(OUTPUT/'F_JOURNAL.json',dict(stage='ACCURACY', completed=F_calls,
                        F_attempts=26+F_calls+1, last_id=frame['id'],method=arm))
                    row=score(E,frame,points,targets[frame['id']],arm); F_calls+=1
                    for k in counts:counts[k]+=row['PnP_counts'][k]
                    row.update(RAW_native_points=initial,prediction_support=support,
                        correction=dict(native_algorithm=algorithm,cap_applied=suffix=='CAP1',
                            cap_px=.01*np.hypot(w,h),diagnostics=diagnostics,displacement_px8=delta,
                            corner_unchanged8=[np.array_equal(points[k],initial[k],equal_nan=True) for k in range(8)]),
                        fixed_metadata=dict(selected_index=bound['selected_index'],
                            metadata=bound['candidate_metadata'],preserved=before==digest(candidate)),
                        raw_hw=[h,w],reference_sha256=protocol['baseline']['sha256'])
                    assert row['fixed_metadata']['preserved']
                    clean=finite(row)
                    stream.write(json.dumps(clean,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                    rows_written+=1; saved.append(clean)
            if i%40==0 or i==318:
                stream.flush()
                print('ACCURACY',i+1,319,'F',F_calls,'seconds',round(time.monotonic()-started,1),flush=True)
    assert rows_written==F_calls==1276
    for arm in ARMS:
        rows=[r for r in saved if r['method']==arm]
        assert len(rows)==319 and len({r['id'] for r in rows})==319
        assert sum(len(r['corner']['errors']) for r in rows)==2499
        assert sum(len(r['corner']['observed_errors']) for r in rows)==2445
        assert sum(r['corner']['matched'] for r in rows)==311
    receipt=dict(schema='training_free_native_coordinates_and_actual_F_v1',complete=True,
        protocol_sha256=sha(DOC/'PROTOCOL.json'),raw_rows_file=path.name,
        raw_rows_sha256=sha(path),raw_rows_bytes=path.stat().st_size,rows=rows_written,
        full_denominator_per_arm=319,methods=list(ARMS),
        execution=dict(new_F_accuracy=F_calls,new_F_BASE_parity=26,new_F_runtime=0,
            native_algorithm_image_calls=native_algorithm_image_calls,RGB_decodes=gray_decodes,
            initial_detector_calls=0,N3_PoseFix_accuracy_forwards=0,new_training=0,
            accuracy_PnP_counts=counts,BASE_parity_PnP_counts=read(DOC/'BASE_PARITY.json')['PnP_counts']),
        correction_inputs=['raw-resolution gray','initial selected9 points','prediction support'],
        GT_selection=False,original_points_copy_each_route=True,cap_recomputations=0,
        elapsed_seconds=time.monotonic()-started)
    write(DOC/'PREDICTIONS.json',receipt)
    print('ACCURACY_COMPLETE',finite(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','evaluate'],required=True)
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else evaluate()
