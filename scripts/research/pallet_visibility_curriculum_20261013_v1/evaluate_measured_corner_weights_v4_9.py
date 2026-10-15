"""Diagnostic paired95: direct corner repeatability Sigma with fixed prior.

Actual corrected full B/I intersections replace only their old effective
fit covariance with the recorded image-measurement covariance. This does NOT
claim that conditional line repeatability is total localization uncertainty.
Every fallback and partial B noise budget is preserved, as are coordinates,
states, geometry, reliability, OT/U constraints, coarse prior and starts.
No labels are read until every prediction is sealed. Reused TEST exploratory.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_visibility_pose_v4_8 as P
from . import evaluate_shared_edge_partial_b_v4_7 as V47
from .statewise_partial_boundary_u_solver_v4_6 import solve_boundary_u_region_joint_ot_pnp
from .scoring import pose_metric,distribution

OUTPUT=P.OUTPUT.parent/'measured_corner_weight_diagnosis_v4_9'
ARMS=('SAME_V47','MEASURED_SIGMA_PRIOR1_DIAGNOSTIC')


def use_measured_sigma(fit,correction,states,partial):
    records={r['native_corner']:r for r in correction['records']}
    partial_ids={r['native_corner'] for r in partial['boundary_observations']}
    result=[];changed=[]
    for measurement in fit:
        k=measurement['native_corner']
        if states[k] not in ('B','I'):
            raise ValueError('State gate excludes S/U/O/T before measurement covariance access')
        current=deepcopy(measurement)
        if records[k]['corrected'] and k not in partial_ids:
            covariance=np.asarray(current['covariance_px2'],float)
            if (covariance.shape!=(2,2) or not np.isfinite(covariance).all()
                    or not np.allclose(covariance,covariance.T) or np.linalg.eigvalsh(covariance).min()<=0):
                raise ValueError('Positive definite recorded image covariance required')
            current['fit_covariance_px2']=covariance.tolist()
            changed.append(k)
        restored=deepcopy(current)
        restored['fit_covariance_px2']=measurement['fit_covariance_px2']
        if restored!=measurement:
            raise ValueError('Only effective covariance may change')
        result.append(current)
    return result,changed


def _run(task):
    row,model,config=task;started=time.monotonic()
    fit,changed=use_measured_sigma(row['frozen_BI_fit'],row['correction'],row['states'],row['partial_boundary_observation'])
    altered=deepcopy(row);altered['frozen_BI_fit']=fit
    original=P._spec(row,config);spec=P._spec(altered,config)
    immutable=E._digest(spec)
    if E._digest(original)==immutable:
        solution=deepcopy(row['arms'][V47.ARMS[2]]['solution']);cached=True
    else:
        solution=solve_boundary_u_region_joint_ot_pnp(**spec);cached=False
    if E._digest(spec)!=immutable:raise ValueError('Solver changed fixed inputs')
    selected=V47._finish(row['_original_source'],fit,model,solution,
        row['u_region_observation']['u_regions'],ARMS[1],row['partial_boundary_observation'])
    return C.finite(dict(id=row['id'],partition=row['partition'],image=row['image'],image_sha256=row['image_sha256'],
        states=row['states'],runtime=row['runtime'],original_fit=row['frozen_BI_fit'],diagnostic_fit=fit,
        changed_sigma_native_ids=changed,partial_boundary_observation=row['partial_boundary_observation'],
        u_region_observation=row['u_region_observation'],
        arms={ARMS[0]:row['arms'][V47.ARMS[2]],ARMS[1]:selected},
        exact_mathematical_input_cache=cached,mathematical_input_sha256=immutable,
        GT_inputs=False,conditional_sigma_is_calibrated_total_error=False,
        new_RGB_mask_state_line_queries=0,elapsed_seconds=time.monotonic()-started))


def main():
    if OUTPUT.exists():raise FileExistsError('Preserve previous paired diagnosis')
    rows,model,config,bindings,parent=P._load()
    if config['prior_weight']!=1.:raise ValueError('Fixed V47 prior weight required')
    sources=deepcopy(parent['source_sha256'])
    for path in (Path(__file__),Path(P.__file__)):
        sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    OUTPUT.mkdir(parents=True)
    C.write(OUTPUT/'PROTOCOL.json',dict(schema='MEASURED_CORNER_SIGMA_PAIRED_DIAGNOSTIC_V4_9',
        ids=[r['id'] for r in rows],CAL=46,TEST=49,severe=0,arms=ARMS,
        source_sha256=sources,input_sha256=bindings,protected_user_sha256=protected,
        only_change='actual corrected full BI recorded covariance replaces effective fit covariance',
        conditional_repeatability_is_total_error=False,not_deployment_calibration=True,
        coordinates_reliability_partial_normals_states_OT_U_K_Q_starts_prior_fixed=True,
        annotations_accessed_only_after_all95_seal=True,GT_or_record_ID_selection=False,
        TEST_is_reused_iterative_exploratory=True,new_RGB_mask_state_line_queries=0))
    values={};started=time.monotonic()
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        tasks={pool.submit(_run,(row,model,config)):row['id'] for row in rows}
        for task in as_completed(tasks):
            value=task.result();values[value['id']]=value
            C.write(OUTPUT/f"{value['partition']}_frames"/(value['id'].replace(':','__')+'.json'),value)
            C.write(OUTPUT/'LIVE_PROGRESS.json',dict(completed=len(values),total=95,latest_id=value['id'],elapsed_seconds=time.monotonic()-started))
            print(f"{len(values)}/95 {value['id']} sigma={value['changed_sigma_native_ids']}",flush=True)
    ordered=[values[row['id']] for row in rows]
    C.save_rows(OUTPUT/'PREDICTIONS.jsonl.gz',ordered)
    for binding in (sources,bindings,protected):E._verify_sources(binding)
    C.write(OUTPUT/'PREDICTION_SEAL.json',dict(rows=95,CAL=46,TEST=49,
        predictions_sha256=C.sha(OUTPUT/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(OUTPUT/'PROTOCOL.json'),
        before_new_annotation_metrics=True,source_sha256=sources,input_sha256=bindings,
        elapsed_seconds=time.monotonic()-started))
    reports={}
    for part in ('CAL','TEST'):
        target=E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json' if part=='CAL' else 'test_reused_v1/ISOLATED_TEST_SCORING.json')
        labels=C.read(target)['rows'];scored=[]
        for row in ordered:
            if row['partition']!=part:continue
            scored.append(dict(id=row['id'],changed_sigma_native_ids=row['changed_sigma_native_ids'],
                metrics={a:pose_metric(row['arms'][a]['actual_pose'],labels[row['id']]['target']['truth']) for a in ARMS}))
        report=dict(partition=part,frames=len(scored),all_fallbacks_kept=True,summary_all={a:dict(
            translation_cm=distribution([r['metrics'][a]['translation_cm'] for r in scored]),
            rotation_deg=distribution([r['metrics'][a]['rotation_deg'] for r in scored]),
            translation_le5=sum(r['metrics'][a]['translation_cm']<=5 for r in scored),
            successes_5cm_5deg=sum(r['metrics'][a]['success_5cm_5deg'] for r in scored)) for a in ARMS},rows=scored)
        C.write(OUTPUT/f'{part}_SCORE.json',report);reports[part]=report
    print({part:r['summary_all'] for part,r in reports.items()},flush=True)


if __name__=='__main__':main()
