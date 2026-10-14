"""Fixed-recipe CAL LOGO validation of conditional truncation localization.

The CAL-selected lambda/beta and old sources remain frozen.  Pixel calibration
may read isolated CAL annotations; pose metrics are evaluated only after a
complete46 prediction seal. No TEST or new RGB forward is used by this driver.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import numpy as np
import cv2
from threadpoolctl import threadpool_limits

from . import common as C
from . import statewise_selection_v4 as D
from . import statewise_bias_selection_v4_1 as BS
from . import statewise_uncertainty_calibration_v4 as UC
from . import evaluation_bias_v4_1 as E
from . import statewise_uncertain_solver_v4_bias as B
from .statewise_joint_api_v4 import finish_observations
from .scoring import pose_metric

OUTPUT=D.FIX/'truncation_pose_validation_v4_2'
MODEL_OUTPUT=D.FIX/'truncation_interval_model_CAL_v4_2'
LABEL=UC.BASE/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'

def build_CAL_interval_frames(rows,annotations):
    """Native CAL-only fit records with S raw/GT erased BEFORE coordinate reads."""
    labels=annotations.get('rows',annotations)
    if len({r['id'] for r in rows})!=len(rows) or any(r.get('partition')!='CAL' for r in rows):
        raise ValueError('Distinct CAL rows only')
    out=[]
    for row in rows:
        label=labels[row['id']]
        if label['metadata']['partition']!='CAL':raise ValueError('CAL labels only')
        g=row['runtime']['initial_geometry'];a=UC._reference_alignment(row,label)
        states=label['metadata']['human_states_native'];hidden=list(g['initial_self_hidden'])
        raw=[];gt=[];valid=[]
        for k in range(8):
            refid=int(a['mapping'][k]);v=bool(label['target']['valid'][refid]);valid.append(v)
            # No coordinate lookup at all for S. No unavailable/ambiguous label
            # is silently converted into a physical localization target.
            if hidden[k]:raw.append([None,None]);gt.append([None,None]);continue
            raw.append(deepcopy(row['runtime']['original_raw_points'][k]))
            eligible=a['compatible'] and v and states[k] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')
            gt.append(deepcopy(label['target']['gt'][refid]) if eligible else [None,None])
        out.append({'id':row['id'],'partition':'CAL','session':row['session'],
            'physical_native_compatible':bool(a['compatible']),'initial_self_hidden':hidden,
            'human_states_native':deepcopy(states),'valid_native':valid,'raw_points':raw,'gt_native':gt,
            'nonself_prediction_scale_px':row['runtime']['ot_domains']['nonself_prediction_scale_px'],
            'fixed_native_reference_phase':a['phase'],'native_registration_gap_m':a['geometry_gap_m']})
    return out

def _interval_module():
    from . import truncation_interval_v4_2 as T
    return T

def _adapt(row,model,uncertainty_model):
    # Decode actual CAL RGB for retained-domain rebinding, never a new state,
    # mask, or BI forward. This is shared with the actual full-RGB API.
    from .statewise_interval_api_v4_2 import StatewiseIntervalV4_2
    path=C.ROOT/row['image']
    if C.sha(path)!=row['image_sha256']:raise ValueError('CAL actual RGB binding changed')
    image=cv2.imread(str(path));assert image is not None and list(image.shape[:2])==row['raw_hw']
    _,paths,_=E.E._runtime_models()
    api=StatewiseIntervalV4_2(ot_model=C.read(paths['OT_model']),uncertainty_model=uncertainty_model,
        truncation_model=model,config={'prediction_source':'N3_SUBPIX','record_id':row['id']})
    return api.adapt_observation(image,deepcopy(row['runtime']),crop_transform=row.get('crop_transform'))

def source_bindings():
    values=E.source_bindings()
    for name in ('truncation_interval_v4_2.py','statewise_interval_api_v4_2.py','statewise_truncation_validation_v4_2.py','statewise_selection_v4.py','statewise_bias_selection_v4_1.py'):
        path=Path(__file__).parent/name;values[str(path.relative_to(C.ROOT))]=C.sha(path)
    return values

def prepare(output=OUTPUT):
    selected,_,cfg,sb,ssb=E.load_bias_selection()
    if cfg['prior_weight']!=1. or cfg['bias_strength']!=1.:raise ValueError('Frozen lambda/beta1 required')
    plan=C.read(D.FIX/'TRUNCATION_INTERVAL_FIX_PLAN_V1.json')
    acceptance_path=D.FIX/'TRUNCATION_INTERVAL_CAL_ACCEPTANCE_V1.json'
    acceptance=C.read(acceptance_path)
    assert acceptance['before_new_interval_pose_runs'] and acceptance['lambda_beta']==[1.,1.] and acceptance['no_new_prior_choice']
    assert plan['lambda_and_beta_fixed_at_existing_CAL_choice']==[1.,1.] and not plan['new_hyperparameter_selection_from_TEST']
    rows,folds,_,_=D.load_inputs()
    # This CAL target read serves the explicit pixel fit, not pose scoring.
    labels=C.read(LABEL)
    assert labels['TEST_rows']==0 and labels['FINAL_rows']==0 and set(labels['rows'])=={r['id'] for r in rows}
    frames=build_CAL_interval_frames(rows,labels)
    T=_interval_module()
    full=T.fit_truncation_interval_model(frames,calibration_bindings={'CAL_annotations':D.binding(LABEL)})
    assert full['eligible_point_count']==264 and abs(full['radius_ratio']-.544750754221928)<1e-14
    interval_folds={s:T.fit_truncation_interval_model(frames,heldout_session=s,
        calibration_bindings={'CAL_annotations':D.binding(LABEL)}) for s in sorted({r['session'] for r in rows})}
    old_seal=C.read(BS.OUTPUT/'PREDICTION_SEAL.json')
    assert C.sha(BS.OUTPUT/'CANDIDATE_PREDICTIONS.jsonl.gz')==old_seal['predictions_sha256']
    baseline={r['id']:r for r in D.read_lines(BS.OUTPUT/'CANDIDATE_PREDICTIONS.jsonl.gz') if r['bias_strength']==1.}
    assert set(baseline)=={r['id'] for r in rows}
    tasks=[]
    for row in rows:
        fold=folds[row['id']];interval=interval_folds[row['session']]
        assert row['id'] not in interval['fit_ids'] and row['id'] not in fold['model']['fit_ids']
        old=BS.build(row,fold,1.)
        assert old['solver_input_sha256']==baseline[row['id']]['solver_input_sha256']
        adapted=_adapt(row,interval,fold['model'])
        if isinstance(adapted,tuple):adapted=adapted[0]
        newrow={**row,'runtime':adapted};task=BS.build(newrow,fold,1.)
        task.update(arm='FIXED_INTERVAL_CAL_LOGO',interval_model=interval,
            interval_model_sha256=D.digest(interval),baseline_solver_input_sha256=old['solver_input_sha256'],
            exact_baseline_cache_reusable=task['solver_input_sha256']==old['solver_input_sha256'])
        assert task['original_BI_sha256']==old['original_BI_sha256']
        assert task['calibrated_BI_sha256']==old['calibrated_BI_sha256']
        assert task['solver_input']['coarse_pose']==old['solver_input']['coarse_pose']
        assert task['solver_input']['config']==old['solver_input']['config']
        assert task['nonself_prediction_scale_px']==old['nonself_prediction_scale_px']
        assert all(a==b if a!='T' else b in ('T','U') for a,b in zip(old['solver_input']['states'],task['solver_input']['states']))
        assert [d for d in task['solver_input']['domains'] if d['state']=='O']==[d for d in old['solver_input']['domains'] if d['state']=='O']
        tasks.append(task)
    sources=source_bindings()
    protected=C.read(E.OUTPUT/'PROTOCOL.json')['protected_user_sha256']
    inputs={'CAL_cache':D.binding(D.FIX/'fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz'),
            'CAL_cache_seal':D.binding(D.FIX/'fresh_observations_v2/OBSERVATION_SEAL.json'),
            'CAL_annotations_for_pixel_fit':D.binding(LABEL),'CAL_uncertainty_seal':D.binding(D.FIX/'calibration/final_adapter_CAL46/SEAL.json'),
            'CAL_fold_models':D.binding(D.FIX/'calibration/final_adapter_CAL46/CAL_LOGO_REPORT.json'),
            'fixed_plan':D.binding(D.FIX/'TRUNCATION_INTERVAL_FIX_PLAN_V1.json'),'CAL_acceptance':D.binding(acceptance_path),'selected_config':sb,'selection_seal':ssb,
            'baseline_CAL_predictions':D.binding(BS.OUTPUT/'CANDIDATE_PREDICTIONS.jsonl.gz'),'baseline_CAL_prediction_seal':D.binding(BS.OUTPUT/'PREDICTION_SEAL.json')}
    protocol={'schema':'FIXED_RECIPE_CAL_TRUNCATION_POSE_VALIDATION_V4_2','created_UTC':D.now(),'ids':[r['id'] for r in rows],
        'rows':46,'fixed_prior_weight':1.,'fixed_bias_strength':1.,'no_hyperparameter_grid':True,
        'CAL_fit_coordinate_access':'isolated CAL native human DIRECT/EXTERNAL for pixel localization max only; compatible44 fitting; all46 pose scores',
        'before_CAL_pose_metrics':True,'conditional_subsystem_LOGO_not_full_pipeline_OOF':True,
        'exact_fold_uncertainty_mu_covariance_and_raw_interval_fit_excludes_held_session':True,
        'cached_observations':'actual freshv2 CAL46 upstream unchanged; pure adapter then unchanged bias solver',
        'source_sha256':sources,'input_bindings':inputs,'protected_user_sha256':protected,
        'preserved_model_bindings':deepcopy(C.read(BS.OUTPUT/'PROTOCOL.json')['preserved_model_bindings']),
        'TEST_files_opened':False,'actual_CAL_RGB_decodes':46,'new_state_mask_BI_forward_calls':0,
        'retained_OT_domains_rebuilt_and_rebound_to_actual_RGB':True,'exact_baseline_cache_reuses':sum(t['exact_baseline_cache_reusable'] for t in tasks),
        'new_pending_solver_tasks':sum(not t['exact_baseline_cache_reusable'] for t in tasks)}
    D.verify_bindings(protocol)
    assert all(C.sha(C.ROOT/p)==h for p,h in protected.items())
    if (output/'PROTOCOL.json').exists():
        old=C.read(output/'PROTOCOL.json');assert {k:v for k,v in old.items() if k!='created_UTC'}=={k:v for k,v in protocol.items() if k!='created_UTC'}
    else:
        output.mkdir(parents=True,exist_ok=True);D.write(output/'PROTOCOL.json',protocol)
        coverage={'full':T.evaluate_interval_coverage(full,frames),
            'folds':{s:T.evaluate_interval_coverage(m,[f for f in frames if f['session']==s]) for s,m in interval_folds.items()}}
        D.write(output/'FOLD_INTERVAL_MODELS.json',interval_folds)
        D.write(output/'CAL_LOCALIZATION_COVERAGE.json',coverage)
        if MODEL_OUTPUT.exists():raise FileExistsError('Preserve previous interval calibration model')
        MODEL_OUTPUT.mkdir(parents=True)
        D.write(MODEL_OUTPUT/'MODEL.json',full)
        D.write(MODEL_OUTPUT/'CAL_LOGO_REPORT.json',{'coverage':coverage,'fold_models':interval_folds,
            'pixel_fit_only_CAL':True,'no_trimming_or_quantile_grid':True,'no_future_coverage_guarantee':True})
        D.write(MODEL_OUTPUT/'MODEL_SEAL.json',{'schema':'CAL_TRUNCATION_INTERVAL_MODEL_SEAL_V4_2',
            'MODEL_sha256':C.sha(MODEL_OUTPUT/'MODEL.json'),'LOGO_sha256':C.sha(MODEL_OUTPUT/'CAL_LOGO_REPORT.json'),
            'recipeplan_sha256':C.sha(D.FIX/'TRUNCATION_INTERVAL_FIX_PLAN_V1.json'),
            'CAL_acceptance_sha256':C.sha(acceptance_path),
            'source_sha256':sources,'CAL_cache':inputs['CAL_cache'],'CAL_annotations':inputs['CAL_annotations_for_pixel_fit'],
            'CAL_validation_protocol':D.binding(output/'PROTOCOL.json'),'TEST_files_opened':False,
            'fit_partition':'CAL','eligible_point_count':264,'radius_ratio':full['radius_ratio']})
        D.write_lines(output/'SOLVER_INPUTS.jsonl.gz',[{k:t[k] for k in ['id','session','solver_input','solver_input_sha256','baseline_solver_input_sha256','exact_baseline_cache_reusable','original_BI_sha256','calibrated_BI_sha256','interval_model_sha256']} for t in tasks])
        for path,h in sources.items():
            target=output/'code_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/path,target);assert C.sha(target)==h
        D.write(output/'INPUT_SEAL.json',{'protocol_sha256':C.sha(output/'PROTOCOL.json'),'inputs_sha256':C.sha(output/'SOLVER_INPUTS.jsonl.gz'),
            'before_pose_solves':True,'before_CAL_pose_scoring':True,'TEST_files_opened':False})
    return tasks,baseline

def worker(task):
    start=time.perf_counter()
    with threadpool_limits(limits=1):solution=B.solve_uncertain_joint_ot_pnp(**task['solver_input'])
    return {'solver_input_sha256':task['solver_input_sha256'],'solution':solution,'solve_seconds':time.perf_counter()-start,'baseline_cache_reused':False}

def predict(output=OUTPUT):
    tasks,baseline=prepare(output);protocol=C.read(output/'PROTOCOL.json');D.verify_bindings(protocol)
    if (output/'PREDICTION_SEAL.json').exists():
        assert C.sha(output/'PREDICTIONS.jsonl.gz')==C.read(output/'PREDICTION_SEAL.json')['predictions_sha256'];return
    checkpoint=output/'SOLVER_CHECKPOINTS.jsonl.gz'
    cache={r['solver_input_sha256']:r for r in D.read_lines(checkpoint)} if checkpoint.exists() else {}
    for task in tasks:
        if task['exact_baseline_cache_reusable'] and task['solver_input_sha256'] not in cache:
            old=baseline[task['id']]
            assert old['solver_input_sha256']==task['solver_input_sha256']
            solved={'solver_input_sha256':task['solver_input_sha256'],'solution':deepcopy(old['runtime']['joint_pose']),
                    'solve_seconds':0.,'baseline_cache_reused':True}
            D.append_checkpoint(checkpoint,solved);cache[task['solver_input_sha256']]=solved
    pending=[t for t in tasks if t['solver_input_sha256'] not in cache];started=time.perf_counter()
    print({'frames':46,'pending':len(pending),'exact_reuses':sum(t['exact_baseline_cache_reusable'] for t in tasks)},flush=True)
    with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
        futures={pool.submit(worker,t):t for t in pending}
        for n,future in enumerate(as_completed(futures),1):
            task=futures[future];result=future.result();assert result['solver_input_sha256']==task['solver_input_sha256']
            D.append_checkpoint(checkpoint,result);cache[result['solver_input_sha256']]=result
            print({'completed':n,'pending_initial':len(pending),'id':task['id'],'solve_seconds':result['solve_seconds']},flush=True)
    predictions=[]
    for t in tasks:
        solved=cache[t['solver_input_sha256']]
        runtime=finish_observations(t['observation'],deepcopy(solved['solution']),t['solver_input']['visible'],t['model'])
        runtime['CAL_interval_validation']={'interval_model_sha256':t['interval_model_sha256'],
            'prediction_scope':'actual cachedCAL observations / actualRGB domain adapter / frozen pure bias solver',
            'full_refine_RGB_forward_executed':False,'fold_uncertainty_and_median_bias_exclude_held_session':True}
        assert runtime['frozen_visible_measurements']==t['observation']['frozen_visible_measurements']
        predictions.append({'id':t['id'],'partition':'CAL','session':t['session'],'runtime':runtime,'actual_pose':runtime['actual_pose'],
            'solver_input_sha256':t['solver_input_sha256'],'baseline_solver_input_sha256':t['baseline_solver_input_sha256'],
            'baseline_cache_reused':solved['baseline_cache_reused'],'interval_model_sha256':t['interval_model_sha256'],'TEST_files_opened':False})
    D.write_lines(output/'PREDICTIONS.jsonl.gz',predictions);D.verify_bindings(protocol)
    assert all(C.sha(C.ROOT/p)==h for p,h in protocol['protected_user_sha256'].items())
    D.write(output/'PREDICTION_SEAL.json',{'schema':'FIXED_CAL_TRUNCATION_POSE_SEAL_V4_2','created_UTC':D.now(),'rows':46,
        'predictions_sha256':C.sha(output/'PREDICTIONS.jsonl.gz'),'protocol_sha256':C.sha(output/'PROTOCOL.json'),
        'source_sha256':protocol['source_sha256'],'before_CAL_pose_scoring':True,'TEST_files_opened':False,
        'new_solver_tasks':len(pending),'exact_baseline_cache_reuses':sum(p['baseline_cache_reused'] for p in predictions),
        'elapsed_seconds':time.perf_counter()-started})

def score(output=OUTPUT):
    seal=C.read(output/'PREDICTION_SEAL.json');protocol=C.read(output/'PROTOCOL.json');D.verify_bindings(protocol)
    assert seal['before_CAL_pose_scoring'] and seal['rows']==46 and not seal['TEST_files_opened']
    assert C.sha(output/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256'] and C.sha(output/'PROTOCOL.json')==seal['protocol_sha256']
    labels=C.read(LABEL)['rows'];rows=D.read_lines(output/'PREDICTIONS.jsonl.gz')
    base={r['id']:r for r in D.read_lines(BS.OUTPUT/'CANDIDATE_PREDICTIONS.jsonl.gz') if r['bias_strength']==1.}
    assert set(base)==set(labels)=={r['id'] for r in rows}
    baseline=[];regressions=[];improvements=[];known_inside=[];success_losses=[];success_gains=[]
    for row in rows:
        row['metrics']=pose_metric(row['actual_pose'],labels[row['id']]['target']['truth'])
        old=deepcopy(base[row['id']]);old['metrics']=pose_metric(old['actual_pose'],labels[row['id']]['target']['truth']);baseline.append(old)
        delta=row['metrics']['translation_cm']-old['metrics']['translation_cm']
        record={'id':row['id'],'translation_delta_cm':delta,'old_metrics':old['metrics'],'new_metrics':row['metrics'],
                'old_states':old['runtime']['states'],'new_states':row['runtime']['states']}
        if delta>1e-8:regressions.append(record)
        elif delta< -1e-8:improvements.append(record)
        if old['metrics']['success_5cm_5deg'] and not row['metrics']['success_5cm_5deg']:success_losses.append(row['id'])
        if row['metrics']['success_5cm_5deg'] and not old['metrics']['success_5cm_5deg']:success_gains.append(row['id'])
        label=labels[row['id']];mapping=label['metadata']['native_to_reference_corner_ids'];h,w=row['runtime']['initial_geometry']['image_hw']
        for k,state in enumerate(old['runtime']['states']):
            if state!='T':continue
            rid=mapping[k];human=label['metadata']['human_states_native'][k]
            if human not in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED') or not label['target']['valid'][rid]:continue
            x,y=label['target']['gt'][rid]
            if 0<=x<=w-1 and 0<=y<=h-1:
                known_inside.append({'id':row['id'],'native_id':k,'human_state':human,'annotation_xy':[x,y],
                    'old_state':'T','new_state':row['runtime']['states'][k]})
    new_summary=D.summarize(rows);old_summary=D.summarize(baseline)
    result={'schema':'FIXED_CAL_TRUNCATION_POSE_VALIDATION_SCORE_V4_2','created_UTC':D.now(),
        'current_interval':new_summary,'frozen_beta1_baseline':old_summary,
        'known_in_canvas_T_annotation_contradictions':known_inside,
        'all_four_known_in_canvas_T_contradictions_removed':len(known_inside)==4 and all(x['new_state']=='U' for x in known_inside),
        'CAL_total_success_count_not_lower':new_summary['success_count']>=old_summary['success_count'],
        'success_loss_frame_ids':success_losses,'success_gain_frame_ids':success_gains,
        'regressions_by_translation':sorted(regressions,key=lambda x:x['translation_delta_cm'],reverse=True),
        'material_translation_increases_ge1cm':[r for r in sorted(regressions,key=lambda x:x['translation_delta_cm'],reverse=True) if r['translation_delta_cm']>=1.],
        'improvements_by_translation':sorted(improvements,key=lambda x:x['translation_delta_cm']),
        'prediction_seal':D.binding(output/'PREDICTION_SEAL.json'),'native_incompatible2_kept_in_pose_score':True,
        'CAL_acceptance_plan':D.binding(D.FIX/'TRUNCATION_INTERVAL_CAL_ACCEPTANCE_V1.json'),
        'fixed_lambda1_beta1_no_grid':True,'TEST_files_opened':False,'adoption_not_automatically_decided':True,
        'conditional_subsystem_LOGO_not_full_pipeline_OOF':True}
    D.write(output/'CAL_POSE_VALIDATION_SCORE.json',result)
    D.write_lines(output/'POSTSEAL_CAL_SCORED_ROWS.jsonl.gz',rows)
    print({k:result[k] for k in ['current_interval','frozen_beta1_baseline','adoption_not_automatically_decided']},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','score']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='predict':predict()
    else:score()

if __name__=='__main__':main()
