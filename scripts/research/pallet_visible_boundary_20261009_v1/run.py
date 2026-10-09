"""Seal prediction-only inputs, lock 319 coordinates, then evaluate actual final poses."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import inspect
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import common as C

OLD = C.ROOT / '_docs/experiments/pallet_n3_subpix_20261008_v1'
TF = C.ROOT / '_docs/experiments/pallet_training_free_compare_20261007_v1'
ATTACHMENT = Path('/home/minjae/.codex/attachments/4315d665-d2ba-45b6-b743-6e6d87ab22c9/Pasted text.txt')


def prepare():
    from . import methods as M
    assert not (C.DOC / 'PROTOCOL.json').exists(), 'Preserve the pre-outcome configuration'
    old, tf = C.read(OLD / 'PROTOCOL.json'), C.read(TF / 'PROTOCOL.json')
    images = {r['id']: r for r in tf['input_manifest']}
    inputs = []
    for row in old['input_manifest']:
        f = images[row['id']]
        for a, b in (('q0', 'initial_points'), ('prediction_support', 'prediction_support'),
                     ('raw_hw', 'raw_hw'), ('K', 'K'), ('dimensions_whd_m', 'dimensions_whd_m'),
                     ('selected_index', 'selected_index'), ('candidate_metadata', 'candidate_metadata')):
            assert C.digest(row[a]) == C.digest(f[b]), (row['id'], a)
        assert C.sha(C.ROOT / f['image']) == row['image_sha256'] == f['image_sha256']
        q0, qn = np.asarray(row['q0'], float), np.asarray(row['qN'], float)
        support = np.asarray(row['prediction_support'], bool)
        assert q0.shape == qn.shape == (9, 2) and support.shape == (9,)
        assert np.array_equal(q0[8], qn[8], equal_nan=True)
        assert np.array_equal(q0[~support], qn[~support], equal_nan=True)
        inputs.append(dict(**row, image=f['image']))
    assert len(inputs) == len({r['id'] for r in inputs}) == 319
    assert len({r['session'] for r in inputs}) == 13
    C.write(C.SCRATCH / 'INITIAL_STATE.json', C.source_state())
    C.write(C.DOC / 'INPUTS.json', dict(schema='prediction_only_input_manifest', frames=inputs,
            inference_fields=['image', 'q0', 'qN', 'prediction_support'], GT_inputs=False))
    config = M.method_configuration()
    code = [C.binding(p) for p in (Path(__file__), Path(C.__file__), Path(M.__file__),
                                   Path(M.__file__).with_name('test_methods.py'),
                                   Path(__file__).with_name('benchmark.py'))]
    protocol = dict(schema='pallet_visible_boundary_20261009_v1', date='2026-10-09',
        branch='research/visible-boundary-refine-20261009', base_commit=C.source_state()['head'],
        authorization='User approved applying the attached coarse-search/shared-boundary proposal as a separate experiment.',
        user_reference=dict(filename=ATTACHMENT.name, sha256=C.sha(ATTACHMENT)),
        population=dict(frames=319, sessions=13, reference_corners=2499, matched_frames=311,
                        observed_reference_corners=2445, ids=[r['id'] for r in inputs]),
        methods=config, controls=list(C.CONTROLS), new_arms=list(C.NEW_ARMS),
        N3=dict(seed=1, checkpoint=old['N3_checkpoint'], temperature=old['N3_temperature'],
                weights_input_output_unchanged=True, cached_coordinates=True),
        cap=dict(CAP1='final total displacement from ORIGINAL Base q0 <= raw image diagonal1%',
                 NATIVE='explicitly uncapped diagnostic route; different output-limit contract',
                 partial_cap_may_break_shared_line_intersections=True, no_post_cap_refit=True),
        configuration_selection='One fixed wider search and one RGB shared-line implementation; inspired by previously viewed DEV examples. No outcome-based parameter changes.',
        novelty='Shared-line reasoning already exists in prior learned line models. This experiment changes observation/estimation to training-free raw-RGB normal search and robust 2D fitting; no novelty claim.',
        inference=dict(inputs=['original_gray', 'predicted_seed9x2', 'prediction_support9'],
            human_visibility_used=False, reference_points_used=False, reference_pose_used=False,
            original_object_selection_box_score_confidence_center_missing_preserved=True,
            coordinates_locked_before_loading_scoring_targets=True),
        evaluation=dict(actual_final_F='existing prediction-only WD hypothesis selection + SQPnP + refinement',
            every_new_arm_F_recomputed=True, existing_four_paths_reused=True,
            visibility='existing human labels for post-hoc strata only',
            reference='existing coordinates and geometry-reconstructed pose, not independent measured physical truth',
            means_variance_SD='raw-point or raw-frame dispersion, ddof1; SD is not a CI or seed variation',
            bootstrap=dict(sessions=13, resamples=10000, seed=20260917, paired=True),
            matched_failures='same original denominator/penalties preserved'),
        timing=dict(routes=list(C.ARMS), panel_frames=26, sessions=13, warmup_each=20,
            repeats=5, measured_each=130, full_actual_pipeline_calls=1800,
            boundary='RAM original RGB -> actual Base detector -> optional fixed N3 -> correction -> actual final F',
            load_decode_excluded=True, cached_replay_is_not_runtime=True, no_parallel_benchmarks=True),
        scope=dict(new_training=0, extra_seeds=0, new_annotations=0, paper_latex_pdf_reference_edits=0,
                   outcome_tuning=False, automatic_main_merge=False),
        inputs=C.binding(C.DOC / 'INPUTS.json'), code=code,
        dependencies=[C.binding(p) for p in (OLD / 'PROTOCOL.json', TF / 'PROTOCOL.json',
            OLD / 'PREDICTIONS.jsonl.gz', C.ROOT / old['N3_checkpoint']['path'])])
    C.write(C.DOC / 'PROTOCOL.json', protocol)
    print('PROTOCOL_SEALED', C.sha(C.DOC / 'PROTOCOL.json'), 'frames319', flush=True)


def assert_preserved(q0, seed, native, final, support, raw_hw, capped):
    supported = np.asarray(support, bool)
    usable = supported & np.isfinite(q0).all(1) & ~(q0 == -1).all(1)
    for array in (native, final):
        assert array.shape == (9, 2)
        assert np.array_equal(array[8], q0[8], equal_nan=True)
        assert np.array_equal(array[~supported], q0[~supported], equal_nan=True)
        assert np.array_equal(array[~usable], q0[~usable], equal_nan=True)
        assert np.isfinite(array[usable]).all()
    if capped:
        active = supported[:8] & np.isfinite(q0[:8]).all(1) & ~(q0[:8] == -1).all(1)
        assert np.max(np.linalg.norm(final[:8][active] - q0[:8][active], axis=1), initial=0) <= .01*np.hypot(*raw_hw)+1e-10


def infer():
    from . import methods as M
    protocol = C.read(C.DOC / 'PROTOCOL.json')
    assert C.sha(C.DOC / 'INPUTS.json') == protocol['inputs']['sha256']
    for item in protocol['code']:
        assert C.sha(C.WORKTREE / item['path']) == item['sha256'], item['path']
    assert not (C.DOC / 'COORDINATES.jsonl.gz').exists()
    assert not (C.SCRATCH / 'INFERENCE_STARTED.json').exists(), 'Interrupted runs need explicit accounting'
    started = time.monotonic()
    cv2.setNumThreads(1)
    bounds = C.read(C.DOC / 'INPUTS.json')['frames']
    C.write(C.SCRATCH / 'INFERENCE_STARTED.json', dict(time=time.time(), protocol_sha256=C.sha(C.DOC/'PROTOCOL.json')))
    counter = Counter()
    path = C.DOC / 'COORDINATES.jsonl.gz'
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=6) as stream:
        for i, row in enumerate(bounds):
            assert C.sha(C.ROOT / row['image']) == row['image_sha256'], row['id']
            image = cv2.imread(str(C.ROOT / row['image']), cv2.IMREAD_COLOR)
            assert image is not None and list(image.shape[:2]) == row['raw_hw']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            q0, qn = np.asarray(row['q0'], float), np.asarray(row['qN'], float)
            support = np.asarray(row['prediction_support'], bool)
            for start, seed in (('BASE', q0), ('N3', qn)):
                for label, method in (('WIDE', 'WIDE_SUBPIX'), ('BOUNDARY', 'BOUNDARY')):
                    native, diag = M.correct(gray, seed.copy(), support.copy(), method)
                    counter[f'{start}_{label}_image_calls'] += 1
                    counter['cornerSubPix_calls'] += diag.get('algorithm_corner_calls', 0)
                    for limit in ('NATIVE', 'CAP1'):
                        arm = f'{start}_{label}_{limit}'
                        final = native.copy() if limit == 'NATIVE' else C.cap_points(q0, native, support, row['raw_hw'])
                        assert_preserved(q0, seed, native, final, support, row['raw_hw'], limit == 'CAP1')
                        packet = dict(id=row['id'], session=row['session'], method=arm,
                            native_points=final, q0=q0, qSeed=seed, qSearch=native,
                            prediction_support=support, raw_hw=row['raw_hw'],
                            fixed_metadata=dict(selected_index=row['selected_index'],
                                                candidate_metadata=row['candidate_metadata'], preserved=True),
                            correction=dict(diagnostics=diag, cap_px=.01*np.hypot(*row['raw_hw']),
                                cap_anchor='original_BASE', capped=limit=='CAP1',
                                native_uncapped_route=limit=='NATIVE',
                                total_before_cap_px8=np.linalg.norm(native[:8]-q0[:8],axis=1),
                                total_final_px8=np.linalg.norm(final[:8]-q0[:8],axis=1)))
                        line_residuals = []
                        for corner in diag.get('corner_records', []):
                            k = corner['corner']
                            for edge in corner.get('selected_edges') or []:
                                fitted = diag['edge_records'][edge]
                                normal, offset = np.asarray(fitted['normal']), fitted['offset']
                                line_residuals.append(dict(corner=k, edge=edge,
                                    native_distance_px=abs(float(native[k] @ normal + offset)),
                                    final_distance_px=abs(float(final[k] @ normal + offset))))
                        packet['correction']['selected_shared_line_residuals'] = line_residuals
                        packet['correction']['post_cap_refit'] = False
                        stream.write(json.dumps(C.finite(packet),ensure_ascii=False,separators=(',', ':'),allow_nan=False)+'\n')
                        counter['rows'] += 1
            if i % 40 == 0 or i == 318:
                stream.flush()
                C.write(C.SCRATCH / 'EXECUTION.json', dict(stage='INFERENCE', frames_complete=i+1, **counter))
                print('INFER',i+1,319,'seconds',round(time.monotonic()-started,2),flush=True)
    assert counter['rows'] == 2552
    C.write(C.DOC / 'COORDINATES_LOCK.json', dict(complete=True, predictions=C.binding(path),
        configuration_sha256=C.digest(protocol['methods']), protocol_sha256=C.sha(C.DOC/'PROTOCOL.json'),
        prediction_only=True, scoring_targets_loaded=False, human_visibility_loaded=False,
        execution=dict(counter, new_detector_calls=0, new_N3_forwards=0, new_training=0,
                       elapsed_seconds=time.monotonic()-started)))


def score():
    lock = C.read(C.DOC / 'COORDINATES_LOCK.json')
    assert lock['complete'] and C.sha(C.DOC/'COORDINATES.jsonl.gz') == lock['predictions']['sha256']
    assert not (C.DOC / 'PREDICTIONS.jsonl.gz').exists()
    assert not (C.SCRATCH / 'SCORING_STARTED.json').exists(), 'Interrupted F attempts require explicit accounting'
    C.source_modules()
    from scripts.research.pallet_n3_subpix_20261008_v1.evaluate import score as existing_score
    from scripts.research.pallet_training_free_compare_20261007_v1.common import load_real
    from scripts.research.pallet_training_free_compare_20261007_v1.evaluate import same_numeric
    from scripts.research.pallet_training_free_compare_20261007_v1.reporting import canonical_observed
    import torch
    torch.set_num_threads(4); cv2.setNumThreads(1)
    E, frames, targets, _, _ = load_real()
    frame_by_id = {r['id']: r for r in frames}
    bounds = {r['id']: r for r in C.read(C.DOC/'INPUTS.json')['frames']}
    original = {arm:{} for arm in C.CONTROLS}
    for row in C.iter_rows(OLD/'PREDICTIONS.jsonl.gz'):
        original[row['method']][row['id']] = row
    assert all(len(v)==319 for v in original.values())
    coords = {arm:{} for arm in C.NEW_ARMS}
    for row in C.iter_rows(C.DOC/'COORDINATES.jsonl.gz'):
        coords[row['method']][row['id']] = row
    assert all(len(v)==319 for v in coords.values())
    panel = C.read(C.ROOT/'data/pallet/results/pallet_n3_completion_v3/runtime/dope_seed1.json')['selected']
    ids = [r['frame_id'] for r in panel]
    # The existing F is checked on one fixed Base panel before any new-arm scoring.
    parity=[]
    started = time.monotonic()
    C.write(C.SCRATCH/'SCORING_STARTED.json',dict(time=time.time(),accuracy_F_started=0,parity_F_started=0))
    count=Counter()
    runtime_refs={arm:{} for arm in C.ARMS}
    for fid in ids:
        f=frame_by_id[fid]
        new=existing_score(E,f,np.asarray(bounds[fid]['q0'],float),targets[fid],'BASE')
        same_numeric(original['BASE'][fid]['corner'],C.finite(new['corner']),fid+'/BASE/corner')
        same_numeric(original['BASE'][fid]['pose'],C.finite(new['pose']),fid+'/BASE/pose')
        assert new['final_hypothesis']==original['BASE'][fid]['final_hypothesis']
        parity.append(dict(id=fid,PASS=True,PnP_counts=new['PnP_counts']))
        count['parity_F']+=1
    C.write(C.DOC/'BASE_PARITY.json',dict(PASS=True,rows=parity,actual_F_calls=len(parity),atol=1e-7))
    path=C.DOC/'PREDICTIONS.jsonl.gz'
    with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as stream:
        for i,f in enumerate(frames):
            fid=f['id'];bound=bounds[fid]
            assert C.digest(f['q'])==C.digest(bound['q0'])
            assert C.digest(f['point_valid'])==C.digest(bound['prediction_support'])
            before=C.digest({k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')})
            for arm in C.ARMS:
                if arm in C.CONTROLS:
                    row=copy.deepcopy(original[arm][fid])
                    row['canonical_observed']=canonical_observed(row)
                    row['accuracy_provenance']='Existing frozen four-route actual F metrics reused, Base fixed26 parity checked.'
                    count['cached_rows_reused']+=1
                else:
                    packet=coords[arm][fid]
                    q=np.asarray(packet['native_points'],float)
                    count['accuracy_F_started']+=1
                    C.write(C.SCRATCH/'EXECUTION.json',dict(stage='SCORING',last_id=fid,last_method=arm,**count))
                    row=existing_score(E,f,q,targets[fid],arm)
                    count['accuracy_F_completed']+=1
                    count.update({f'accuracy_{k}':v for k,v in row['PnP_counts'].items()})
                    row.update(packet)
                    row['accuracy_provenance']='Actual existing F applied to this exact final coordinate, after prediction-only coordinate lock.'
                    for diag in packet['correction']['diagnostics'].get('corner_records',[]):
                        if diag.get('changed') is False:
                            k=diag['corner']
                            assert np.array_equal(q[k],np.asarray(packet['qSeed'])[k],equal_nan=True) or packet['correction']['capped']
                stream.write(json.dumps(C.finite(row),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
                count['rows_written']+=1
                if fid in ids:
                    assert row['actual_pose'] is not None, (fid, arm, 'missing predicted pose parameter reference')
                    runtime_refs[arm][fid]=dict(points=row['native_points'],pose=row['actual_pose'])
            after=C.digest({k:v for k,v in f['captured']['captured'].items() if k not in ('p3','p4')})
            assert before==after
            if i%40==0 or i==318:
                stream.flush();print('SCORE',i+1,319,'F',count['accuracy_F_completed'],'seconds',round(time.monotonic()-started,2),flush=True)
    assert count['rows_written']==3828 and count['accuracy_F_completed']==2552 and count['parity_F']==26
    assert all(len(rows)==26 for rows in runtime_refs.values())
    C.write(C.DOC/'RUNTIME_REFERENCES.json',dict(schema='prediction_only_runtime_parity_references',
        GT_coordinates=False,GT_pose=False,panel_ids=ids,references=runtime_refs,
        head_references={fid:bounds[fid]['qN'] for fid in ids}))
    C.write(C.DOC/'CHECKS.json',dict(status='PASS',complete=True,
        original_four_controls_exact=True, Base_panel_pose_parity=True,
        GT_free_correction_signature=list(inspect.signature(__import__(__package__+'.methods',fromlist=['correct']).correct).parameters),
        all_new_final_poses_recomputed=True, copied_N3_poses=False,
        raw_rows=C.binding(path), coordinates_lock=C.binding(C.DOC/'COORDINATES_LOCK.json'),
        original_user_state_unchanged=C.source_state()==C.read(C.SCRATCH/'INITIAL_STATE.json'),
        execution=dict(count, frames=319, new_detector_calls=0,new_N3_forwards=0,new_training=0,
                       elapsed_seconds=time.monotonic()-started)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','infer','score'))
    args=parser.parse_args()
    try:
        {'prepare':prepare,'infer':infer,'score':score}[args.stage]()
    except Exception as exc:
        C.write(C.SCRATCH/f'INCOMPLETE_{args.stage}.json',dict(complete=False,stage=args.stage,
                  type=type(exc).__name__,error=str(exc)))
        raise
