"""Fixed face-aware partial BI on OLD V47 q; all95 before posthoc scoring.

Compare one unchanged sealed numeric v4.10 baseline with a fixed analytic
partial-BI/whole-outline arm. No corrected q is renamed a backbone fallback:
explicit source/q/noise archives authorize complete-likelihood revocation.
Old full-search poses initialize both registered WD families, never become
new image measurements. Reused TEST49 is exploratory, not a new holdout.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import shutil
import time
import types

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_global_outline_pose_v4_10 as OLD10
from . import statewise_partial_BI_outline_solver_v4_13 as G
from . import face_aware_partial_bi_observations_v4_13b as OBS
from . import global_outline_analytic_factor_v4_11 as ANALYTIC
from . import truncation_interval_v4_2 as LOCALIZATION
from .scoring import pose_metric,distribution

BASELINE=OLD10.OUTPUT.parent/'global_outline_all95_eval_v4_10'
OUTPUT=BASELINE.parent/'face_partial_BI_outline_eval_v4_13'
LOCALIZATION_PATH=BASELINE.parent.parent/'truncation_interval_model_CAL_v4_2'/'MODEL.json'
ARMS=('SAME_OLD_G10_NUMERIC','FACE_AWARE_PARTIAL_BI_OLD_Q')


def _load():
    rows,model,config,bindings,parent=OLD10._load('all95')
    protocol=C.read(BASELINE/'PROTOCOL.json');seal=C.read(BASELINE/'PREDICTION_SEAL.json');path=BASELINE/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=95 or not seal['before_new_reference_metric_read']
            or C.sha(path)!=seal['predictions_sha256'] or C.sha(BASELINE/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Complete sealed old numeric95 baseline required')
    for field in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[field])
    baselines=list(C.rows(path))
    if [r['id'] for r in baselines]!=[r['id'] for r in rows]:raise ValueError('Ordered original95 required')
    by_id={r['id']:r for r in baselines}
    for row in rows:
        baseline=by_id[row['id']]
        for key in ('states','image_sha256','frozen_BI_fit','partial_boundary_observation','u_region_observation'):
            if baseline[key]!=row[key]:raise ValueError('Old immutable observation mismatch '+key)
    locality=LOCALIZATION.load_truncation_interval_model(LOCALIZATION_PATH)
    if locality['eligible_point_count']!=264 or locality['radius_ratio']!=.544750754221928:
        raise ValueError('Fixed existing CAL264 borrowed locality model required')
    for p in (path,BASELINE/'PREDICTION_SEAL.json',BASELINE/'PROTOCOL.json',LOCALIZATION_PATH):
        bindings[str(p.relative_to(C.ROOT))]=C.sha(p)
    return rows,model,config,bindings,protocol,by_id,locality


def _run(task):
    row,model,config,baseline,locality=task;started=time.monotonic();before=E._digest(row)
    image_path=C.ROOT/row['image']
    if C.sha(image_path)!=row['image_sha256']:raise ValueError('Actual current RGB binding changed')
    image=cv2.imread(str(image_path));geometry=row['runtime']['initial_geometry']
    if image is None or tuple(image.shape[:2])!=tuple(geometry['image_hw']):raise ValueError('Actual RGB canvas mismatch')
    m=row['_MASK'];mask_path=OLD10.MASKS/m['mask_npz']
    if C.sha(mask_path)!=m['mask_file_sha256']:raise ValueError('Frozen actual SAM mask file changed')
    with np.load(mask_path) as arrays:mask=arrays['pallet_mask_clean'].copy()
    if mask.shape!=image.shape[:2]:raise ValueError('Frozen SAM mask canvas mismatch')
    observation=OBS.observe_face_aware_partial_bi(image,mask,geometry,row['states'],row['frozen_BI_fit'],
        raw_native_points=row['correction']['raw_points'],localization_model=locality,
        nonself_scale_px=row['_original_source']['runtime']['ot_domains']['nonself_prediction_scale_px'],
        image_binding={'sha256':row['image_sha256'],'image_hw':list(image.shape[:2]),
            'mask_file_sha256':m['mask_file_sha256'],'mask_npz':m['mask_npz']})
    typed=observation['boundary_observations'];fresh=deepcopy(row)
    merged={r['native_corner']:deepcopy(r) for r in row['partial_boundary_observation']['boundary_observations']}
    for record in typed:merged[record['native_corner']]=deepcopy(record)
    fresh['partial_boundary_observation']={**deepcopy(row['partial_boundary_observation']),
        'schema':'MERGED_LEGACY_B_AND_TYPED_FACE_BI_V4_13','boundary_observations':[merged[k] for k in sorted(merged)],
        'typed_face_observation':observation}
    if fresh['frozen_BI_fit']!=row['frozen_BI_fit'] or fresh['u_region_observation']!=row['u_region_observation']:
        raise ValueError('Complete q/noise archives or U regions changed')
    if not typed:
        result=deepcopy(baseline);cached=True
    else:
        operation=types.FunctionType(OLD10._run.__code__,{**OLD10._run.__globals__,'G':G},
            name='_old_V47_typed_BI_analytic_prediction')
        old_backend=OLD10.G
        result=operation((fresh,model,config))
        if OLD10.G is not old_backend:raise ValueError('Immutable numeric module backend mutated')
        cached=False
        for arm in result['arms'].values():
            readout=arm['keypoint_readout']
            readout['partial_BI_observations']=deepcopy(fresh['partial_boundary_observation']['boundary_observations'])
            readout['partial_B_observations']=[deepcopy(r) for r in readout['partial_BI_observations'] if r['state']=='B']
            for record in fresh['partial_boundary_observation']['boundary_observations']:
                k=record['native_corner'];readout['reported_points'][k]=None
                readout['point_sources'][k]='PARTIAL_'+record['state']+'_NORMAL_AND_TANGENT_INTERVAL'
            readout['explicit_complete_BI_likelihood_revocation_ids']=[r['native_corner'] for r in typed]
    if E._digest(row)!=before:raise ValueError('Producer/pose changed immutable old observations')
    return C.finite(dict(id=row['id'],partition=row['partition'],session=row['session'],image=row['image'],
        image_sha256=row['image_sha256'],states=row['states'],initial_geometry=geometry,
        arms={ARMS[0]:deepcopy(baseline),ARMS[1]:result},face_observation=observation,
        merged_partial_observation=fresh['partial_boundary_observation'],new_arm_solve_calls=0 if cached else 1,
        exact_whole_old_numeric_cache_reuse=cached,elapsed_seconds=time.monotonic()-started,
        contract=dict(GT_inputs=False,S_local_coordinate_reads=0,S_local_RGB_queries=0,U_local_queries=0,
            states_K_Q_xyz_scale_complete_q_observed_and_fit_covariance_weights_OT_U_unchanged=True,
            new_mask_backbone_or_state_forward=0,old_full_BI_seed_and_family_poses_are_starts_only=True,
            partial_normal_variance_is_borrowed_fixed_CAL_proxy_not_new_calibration=True,
            tangent_range_is_broad_CAL_borrowed_not_precise_coordinate_or_coverage_probability=True)))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve existing typed partial experiment')
    rows,model,config,bindings,parent,baselines,locality=_load();sources=deepcopy(parent['source_sha256'])
    for source in (Path(__file__),Path(OBS.__file__),Path(OBS.G.__file__),Path(G.__file__),Path(G.P.__file__),
                   Path(G.OLD.__file__),Path(ANALYTIC.__file__),Path(ANALYTIC.BASE.__file__)):
        sources[str(source.relative_to(C.ROOT))]=C.sha(source)
    protected=parent['protected_user_sha256']
    for hashes in (sources,bindings,protected):E._verify_sources(hashes)
    output.mkdir(parents=True)
    protocol=dict(schema='OLD_V47_FACE_PARTIAL_BI_ANALYTIC_OUTLINE_PAIRED95_V4_13',arms=list(ARMS),ids=[r['id'] for r in rows],
        primary_population={'CAL':46,'TEST':49,'severe':0},source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        solver_configuration=config,outline_configuration=G.DEFAULT_OUTLINE_CONFIGURATION,producer_configuration=OBS.DEFAULT_CONFIGURATION,
        cached_baseline_derivative='numerical_v4_10',active_partial_derivative='analytic_v4_11',
        cached_unchanged_modes_are_not_fresh_analytic_runs=True,old_V47_q_not_current_v4_9c=True,
        initial_registered_face_tolerance_deg=2.,line_reach_px='existing36+7',
        explicit_complete_BI_likelihood_revocation=True,old_full_BI_seed_preserved=True,
        partial_normal_total_variance_not_newly_calibrated=True,tangent_radius_borrowed_CAL264_max=True,
        every_OT_U_and_visible_physical_guard_preserved=True,whole_outline_fixed_engineering_noise=True,
        no_GT_or_frameID_runtime_routing=True,TEST_reused_exploratory=True,all95_allarms_seal_before_new_score=True,
        workers=workers,CPU_threads_per_worker=1)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        dest=output/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,dest)
        if C.sha(dest)!=digest:raise ValueError('Source snapshot mismatch')
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(_run,(r,model,config,baselines[r['id']],locality)):r['id'] for r in rows}
        for future in as_completed(futures):
            row=future.result();done[row['id']]=row
            C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=95,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/95 {row["id"]} typed={[b["native_corner"] for b in row["face_observation"]["boundary_observations"]]}',flush=True)
    for hashes in (sources,bindings,protected):E._verify_sources(hashes)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[r['id']] for r in rows])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=95,CAL=46,TEST=49,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_reference_metric_read=True,all2arms_complete_before_score=True,
        source_sha256=sources,input_sha256=bindings,elapsed_seconds=time.monotonic()-started,
        new_arm_solve_calls=sum(r['new_arm_solve_calls'] for r in done.values())))


def score(output=OUTPUT):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'PREDICTION_SEAL.json');path=output/'PREDICTIONS.jsonl.gz'
    if (seal['rows']!=95 or not seal['all2arms_complete_before_score'] or not seal['before_new_reference_metric_read']
            or C.sha(path)!=seal['predictions_sha256'] or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Seal all95 first')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    rows=list(C.rows(path));baselines={r['id']:r for r in C.rows(BASELINE/'PREDICTIONS.jsonl.gz')}
    if [r['id'] for r in rows]!=protocol['ids'] or any(r['arms'][ARMS[0]]!=baselines[r['id']] for r in rows):raise ValueError('Exact95 numeric baseline parity required')
    reports={}
    for part,count in (('CAL',46),('TEST',49)):
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];population=[r for r in rows if r['partition']==part]
        if len(population)!=count:raise ValueError('Every fallback/native-incompatible row must remain')
        values=[]
        for row in population:
            metrics={};accepted={};family={}
            for arm in ARMS:
                result=row['arms'][arm];pose=result['arms']['SELECTED_JOINT_OUTLINE']
                metrics[arm]=pose_metric(pose['actual_pose'],labels[row['id']]['target']['truth']);accepted[arm]=pose['accepted'];family[arm]=result['selected_family']
            values.append(dict(id=row['id'],metrics=metrics,accepted=accepted,selected_family=family,
                partial_BI_ids=[b['native_corner'] for b in row['face_observation']['boundary_observations']]))
        summary={arm:dict(translation_cm=distribution([r['metrics'][arm]['translation_cm'] for r in values]),
            rotation_deg=distribution([r['metrics'][arm]['rotation_deg'] for r in values]),
            translation_le5=sum(r['metrics'][arm]['translation_cm']<=5 for r in values),
            successes_5cm_5deg=sum(r['metrics'][arm]['success_5cm_5deg'] for r in values),
            fallback_count=sum(not r['accepted'][arm] for r in values),
            translation_over5_rows=[r for r in values if r['metrics'][arm]['translation_cm']>5]) for arm in ARMS}
        report=dict(partition=part,count=count,rows=values,summary=summary,all_native_incompatible_and_fallbacks_kept=True,
            target_binding=E._binding(target),prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),TEST_is_reused_exploratory=True)
        C.write(output/f'{part}_SCORE.json',report);reports[part]=report
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),score_sha256={p:C.sha(output/f'{p}_SCORE.json') for p in reports}))
    print({p:r['summary'] for p,r in reports.items()},flush=True)


def check():
    rows,model,config,bindings,parent,baselines,locality=_load()
    print('FACE_PARTIAL_BI_PREFLIGHT_PASS',len(rows),'CAL46_TEST49_severe0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=('check','predict','score','all'),default='all')
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.stage=='check':check()
    elif args.stage=='predict':predict(args.output,args.workers)
    elif args.stage=='score':score(args.output)
    else:predict(args.output,args.workers);score(args.output)
