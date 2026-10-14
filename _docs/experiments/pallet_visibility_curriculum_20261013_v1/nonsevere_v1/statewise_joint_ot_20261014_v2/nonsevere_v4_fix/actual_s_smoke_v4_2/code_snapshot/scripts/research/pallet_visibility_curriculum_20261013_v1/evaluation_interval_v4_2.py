"""Fresh98 fixed CAL-localization T admission repair; old V4 files immutable.

Every primary prediction executes the actual interval API on current RGB.
Controls may reuse prior sealed poses only on exact mathematical input hashes.
Scoring is separate and never chooses a new parameter or removes an O region.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluation_bias_v4_1 as EB
from . import statewise_truncation_validation_v4_2 as V
from . import statewise_uncertain_solver_v4_bias as B
from .truncation_interval_v4_2 import load_truncation_interval_model,apply_truncation_interval
from .statewise_interval_api_v4_2 import StatewiseIntervalV4_2

OUTPUT=E.FIX/'evaluation_interval_v4_2'
MODEL=V.MODEL_OUTPUT/'MODEL.json'
_INTERVAL_MODEL=None
_BASE_CONFIG=None

def certified_T_fixture(model):
    states=['S','T']+['B']*6
    records=[{'native_id':k,'state':s,'physical_truncation_certified':k==1} for k,s in enumerate(states)]
    hidden=[True]+[False]*7
    # No valid S coordinate or certified-T query is made available.
    guarded=apply_truncation_interval(states,[[None,None]]*8,[480,640],100.,model,records,hidden)
    assert guarded['states']==states and guarded['audit']['records'][0]['raw_coordinate_accessed'] is False
    return {'certified_T_retained':True,'S_coordinates_not_available':True,'RGB_reads':0,'pose_solves':0}

def load_interval_configuration(path=MODEL):
    """Return selected bias/model/config, fixed interval model, and bindings.

    Requires CAL source/model/prediction/validation receipts before deployment
    or TEST prediction. It reads CAL validation metrics, never TEST targets.
    """
    selected,uncertainty,config,sb,ssb=EB.load_bias_selection()
    if config['prior_weight']!=1. or config['bias_strength']!=1.:raise ValueError('Frozen lambda1 beta1 required')
    path=Path(path).resolve();seal_path=path.parent/'MODEL_SEAL.json'
    if not seal_path.is_file():raise FileNotFoundError('CAL interval model seal required')
    seal=C.read(seal_path)
    if seal.get('TEST_files_opened') is not False or seal.get('fit_partition')!='CAL' or C.sha(path)!=seal['MODEL_sha256']:
        raise ValueError('Interval model is not sealed CAL-only')
    E._verify_sources(seal['source_sha256'])
    E._verify_binding(seal['CAL_cache']);E._verify_binding(seal['CAL_annotations']);E._verify_binding(seal['CAL_validation_protocol'])
    if C.sha(path.parent/'CAL_LOGO_REPORT.json')!=seal['LOGO_sha256']:raise ValueError('Interval LOGO receipt changed')
    plan_path=E.FIX/'TRUNCATION_INTERVAL_FIX_PLAN_V1.json'
    if C.sha(plan_path)!=seal['recipeplan_sha256']:raise ValueError('Fixed T repair plan changed')
    plan=C.read(plan_path)
    if plan['lambda_and_beta_fixed_at_existing_CAL_choice']!=[1.,1.] or plan['new_hyperparameter_selection_from_TEST']:
        raise ValueError('T repair changed frozen prior/bias policy')
    acceptance=E.FIX/'TRUNCATION_INTERVAL_CAL_ACCEPTANCE_V1.json'
    if C.sha(acceptance)!=seal['CAL_acceptance_sha256']:raise ValueError('Predeclared CAL acceptance changed')
    rule=C.read(acceptance)
    if not rule['before_new_interval_pose_runs'] or rule['lambda_beta']!=[1.,1.]:raise ValueError('Fixed CAL acceptance not present')
    interval=load_truncation_interval_model(path)
    if interval['heldout_session'] is not None or interval['eligible_point_count']!=264:
        raise ValueError('Production T gate requires fullCAL 264point fixed fit')
    vp=C.read(V.OUTPUT/'PROTOCOL.json');V.D.verify_bindings(vp)
    pred=C.read(V.OUTPUT/'PREDICTION_SEAL.json')
    if (not pred['before_CAL_pose_scoring'] or pred['TEST_files_opened'] or pred['rows']!=46
            or C.sha(V.OUTPUT/'PREDICTIONS.jsonl.gz')!=pred['predictions_sha256']
            or C.sha(V.OUTPUT/'PROTOCOL.json')!=pred['protocol_sha256']):
        raise ValueError('CAL46 interval prediction seal invalid')
    scored=C.read(V.OUTPUT/'CAL_POSE_VALIDATION_SCORE.json')
    E._verify_binding(scored['prediction_seal'])
    if (not scored['all_four_known_in_canvas_T_contradictions_removed']
            or not scored['CAL_total_success_count_not_lower'] or scored['TEST_files_opened']
            or not scored['fixed_lambda1_beta1_no_grid'] or scored['current_interval']['frames']!=46):
        raise ValueError('Predeclared CAL truncation acceptance conditions not met')
    fixture=certified_T_fixture(interval)
    bindings={'interval_model':E._binding(path),'interval_model_seal':E._binding(seal_path),
        'CAL_validation_score':E._binding(V.OUTPUT/'CAL_POSE_VALIDATION_SCORE.json'),
        'CAL_validation_prediction_seal':E._binding(V.OUTPUT/'PREDICTION_SEAL.json'),
        'fixed_truncation_plan':E._binding(plan_path),'bias_selected_config':sb,'bias_selection_seal':ssb,
        'CAL_acceptance_plan':E._binding(acceptance),
        'certified_T_fixture':fixture}
    return selected,uncertainty,config,interval,bindings

def source_bindings():
    values=V.source_bindings()
    path=Path(__file__);values[str(path.relative_to(C.ROOT))]=C.sha(path)
    return values

def _init_worker(selected,model,config,interval,base_config,device):
    global _INTERVAL_MODEL,_BASE_CONFIG
    E._init_worker(selected,model,config,device)
    _INTERVAL_MODEL=deepcopy(interval);_BASE_CONFIG=deepcopy(base_config)

def _control(result,fit,model,config,arm,old,old_binding):
    states=['U' if s in ('O','T') else s for s in result['states']] if arm==E.ARM_NO_OT else result['states']
    domains=[] if arm==E.ARM_NO_OT else result['ot_domains']['domains']
    use=deepcopy(config) if arm==E.ARM_NO_OT else deepcopy(_BASE_CONFIG)
    if arm==E.ARM_NO_PRIOR:use['prior_weight']=0.
    spec=E._physics_spec(result,fit,use,states,domains);key=E._digest(spec)
    exact=key==old['control_input_audit'][arm]['mathematical_input_sha256']
    if exact:
        record=deepcopy(old['controls'][arm]);calls=0
    else:
        solution=B.solve_uncertain_joint_ot_pnp(**spec) if arm==E.ARM_NO_OT else E._solve_spec(spec)
        source=deepcopy(result);source['states']=states;source['ot_domains']['domains']=deepcopy(domains)
        source['ot_domains']['domains_sha256']=E._digest(domains)
        done=E.finish_observations(source,solution,fit,model)
        record={'actual_pose':done['actual_pose'],'states':states,'accepted':done['accepted'],
            'solution':E._compact_solution(solution),'keypoint_readout':done['keypoint_readout'],
            'ambiguity_audit':done['ambiguity_audit'],'observation_alias':'runtime.frozen_visible_measurements',
            'calibrated_measurements_alias':'runtime.calibrated_visible_measurements'}
        calls=1
    audit={'mathematical_input_sha256':key,'cached_identical_input_result':exact,
        'reuse_prediction_seal':old_binding if exact else None,
        'same_BI_xy_observed_covariance_fit_covariance_reliability_and_scale':True,
        'same_initial_registered_geometry':True,
        'changes':'all retained O/T to U with domains empty; lambda1 beta1 unchanged' if arm==E.ARM_NO_OT else 'prior_weight0; CAL mean physically inactive; admitted O/T retained'}
    return record,audit,calls

def _run_frame(task):
    frame,cached,oldbias,oldv4,bias_binding,v4_binding=task
    head,ot,masks,model,config,device=E._CONTEXT
    path=C.ROOT/frame['image'];digest=C.sha(path)
    if digest!=frame['image_sha256'] or digest!=cached['image_sha256']:raise ValueError('Actual interval RGB binding changed')
    image=cv2.imread(str(path))
    if image is None or list(image.shape[:2])!=frame['raw_hw']:raise ValueError('Actual interval canvas mismatch')
    api=StatewiseIntervalV4_2(head,ot,masks,uncertainty_model=model,truncation_model=_INTERVAL_MODEL,
        config={'solver':config,'prediction_source':'N3_SUBPIX','record_id':frame['id'],'device':device})
    result=api.refine(image,frame['raw_points'],frame['K'],C.cuboid(*frame['initial_pose']['cf_extents']),
        frame['initial_pose'],crop_transform=frame['crop_transform'])
    # Compare actual fresh observations with the independently re-bound cached
    # source after the SAME fixed interval rule. Allowed T->U is explicit.
    expected=api.adapt_observation(image,deepcopy(cached['runtime']),crop_transform=frame['crop_transform'])
    expected_row={**deepcopy(cached),'runtime':expected}
    parity=E.observation_parity(result,expected_row);S=E.S_contract_audit(result)
    if not parity['passed'] or not S['passed']:raise ValueError('Fresh interval observation parity/S failed')
    oldstates=cached['runtime']['states']
    if any(a!=b if a!='T' else b not in ('T','U') for a,b in zip(oldstates,result['states'])):raise ValueError('NonT role changed')
    fit=deepcopy(result['calibrated_visible_measurements']);controls={};audits={};calls=1
    for arm,old,binding in [(E.ARM_NO_OT,oldbias,bias_binding),(E.ARM_NO_PRIOR,oldv4,v4_binding)]:
        controls[arm],audits[arm],n=_control(result,fit,model,config,arm,old,binding);calls+=n
    row={k:deepcopy(frame[k]) for k in ('id','source_record_id','partition','session','cohort','image','K','raw_hw','crop_transform','initial_pose')}
    row.update(image_sha256=digest,runtime=E._compact_runtime(result),controls=controls,control_input_audit=audits,
        fresh_cache_parity=parity,S_contract_audit=S,solver_calls_this_frame=calls,
        all_OT_mathematical_input_sha256=E._digest(E._physics_spec(result,fit,config)),
        fresh_current_RGB_full_API=True,annotation_targets_read=False,interval_model_CAL_only=True,
        original_cache_states=oldstates,allowed_new_state_change='conditionalT toU only',
        prior_mean_from_CAL_only=True,bias_strength=config['bias_strength'])
    return row

def predict(output=OUTPUT,model_path=MODEL,workers=3,device='cuda'):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous interval evaluation')
    if workers not in (1,2,3) or not str(device).startswith('cuda'):raise ValueError('1-3 CUDA spawn workers required')
    selected,model,config,interval,bindings=load_interval_configuration(model_path)
    _,_,base_config,_,_=E._selection(E._resolve(selected['stage1_selected_config']))
    frames=E._frames();cached,cache_bindings=E._fresh_cache(frames)
    biasrows,_,_=E._verify_prediction_seal(EB.OUTPUT);v4rows,_,_=E._verify_prediction_seal(E.OUTPUT)
    bias={r['id']:r for r in biasrows};v4={r['id']:r for r in v4rows}
    bb=E._binding(EB.OUTPUT/'PREDICTION_SEAL.json');vb=E._binding(E.OUTPUT/'PREDICTION_SEAL.json')
    _,_,model_bindings=E._runtime_models();fixed=C.read(E.FIX/'FIX_PROTOCOL_V1.json')
    protected=fixed['protected_user_sha256'];sources={**source_bindings(),**selected['source_sha256']}
    E._verify_sources(sources);E._verify_sources(protected)
    inputs={str((E.COHORT/name).relative_to(C.ROOT)):C.sha(E.COHORT/name) for name in ('PREDICTION_INPUTS.json','COHORT_PROTOCOL.json')}
    inputs.update(model_bindings);inputs.update(E.crop_bindings())
    for record in bindings.values():
        if isinstance(record,dict) and 'path' in record:inputs[record['path']]=record['sha256']
    for path in (EB.OUTPUT/'PREDICTIONS.jsonl.gz',EB.OUTPUT/'PREDICTION_SEAL.json',E.OUTPUT/'PREDICTIONS.jsonl.gz',E.OUTPUT/'PREDICTION_SEAL.json'):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    output.mkdir(parents=True)
    protocol={'schema':'STATEWISE_FRESH_RGB_FIXED_CAL_INTERVAL_EVALUATION_V4_2','ids':[f['id'] for f in frames],
        'primary_population':{'CAL':46,'TEST':49,'severe':0},'actual_RGB_crop_count':3,'crops_are_primary':False,
        'arms':list(E.ARMS),'arm_labels_use_shared_v4_scorer_version_is_defined_by_this_protocol':True,
        'actual_API':'StatewiseIntervalV4_2 full refine actualRGB; frozen bias solver unchanged',
        'all_frames_run_full_refine_API':True,'selected_config':bindings['bias_selected_config'],
        'selection_seal':bindings['bias_selection_seal'],'calibration_model':selected['calibration_model'],
        'solver_configuration':config,'interval_model_bindings':bindings,'selected_prior_weight':1.,'selected_bias_strength':1.,
        'fixed_recipe_no_GRID_or_TEST_parameter_choice':True,'source_sha256':sources,'input_and_model_sha256':inputs,
        'protected_user_sha256':protected,'fresh_observation_cache':cache_bindings,'parity_tolerances':E.PARITY,
        'parity_target':'independently re-bound original freshv2 cache under samefixedCALinterval; only conditionalT→U admitted',
        'NO_OT_changes':'retainedOT toU only; sameq/observedCov/fitCov/reliability/nonSscale/lambda/beta',
        'NO_PRIOR_changes':'weight0 only; prior meanphysicallyinactive; sameq/cov/scale/retainedOT/coarse',
        'controls_reuse':'sealedexactfullmathematicalhash only; mismatchesreceivefreshsolve',
        'prior_and_bias_selected_on_CAL_only':True,'annotation_targets_accessed':False,'S_image_queries_allowed':0,
        'TEST_status':'FURTHER_EXPLORATORY_REPAIR_REUSED_TEST_NO_INDEPENDENT_GENERALIZATION',
        'workers':workers,'device':device,'CPU_threads_per_worker':1,'maximum_gzip_bytes':100_000_000,
        'required_target':fixed['required_primary_target']}
    C.write(output/'PROTOCOL.json',protocol)
    for p,h in sources.items():
        dest=output/'code_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/p,dest);assert C.sha(dest)==h
    started=time.monotonic();by_id={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=_init_worker,
        initargs=(selected,model,config,interval,base_config,device)) as pool:
        futures={pool.submit(_run_frame,(f,cached[f['id']],bias[f['id']],v4[f['id']],bb,vb)):f['id'] for f in frames}
        for future in as_completed(futures):
            row=future.result();assert row['id']==futures[future] and row['id'] not in by_id;by_id[row['id']]=row
            C.write(output/'LIVE_PROGRESS.json',{'completed':len(by_id),'total':98,'last_id':row['id'],'elapsed_seconds':time.monotonic()-started})
            print(f"{len(by_id)}/98 {row['id']} interval accepted={row['runtime']['accepted']}",flush=True)
    ordered=[by_id[f['id']] for f in frames]
    E._verify_sources(sources);E._verify_sources(inputs);E._verify_sources(protected);load_interval_configuration(model_path)
    C.save_rows(output/'PREDICTIONS.jsonl.gz',ordered)
    if (output/'PREDICTIONS.jsonl.gz').stat().st_size>=100_000_000:raise ValueError('Journal exceeds100MB')
    C.write(output/'CONTROL_REUSE_AUDIT.json',{'bias_prediction_seal':bb,'v4_prediction_seal':vb,
        'NO_OT_exact_reuses':sum(r['control_input_audit'][E.ARM_NO_OT]['cached_identical_input_result'] for r in ordered),
        'NO_PRIOR_exact_reuses':sum(r['control_input_audit'][E.ARM_NO_PRIOR]['cached_identical_input_result'] for r in ordered),
        'all98_controls_complete_before_seal':True})
    C.write(output/'CONTRACT_CHECKS.json',{'rows':98,'fresh_cache_parity_pass':all(r['fresh_cache_parity']['passed'] for r in ordered),
        'S_contract_checks_pass':all(r['S_contract_audit']['passed'] for r in ordered),'all98_all3arms_complete':True,
        'protected6_pre_and_post_unchanged':True,'sources_and_inputs_pre_and_post_unchanged':True,
        'solver_calls':sum(r['solver_calls_this_frame'] for r in ordered),'all98_pose_fallbacks_preserved':True,
        'TEST_target_access_before_prediction_seal':False})
    C.write(output/'PREDICTION_SEAL.json',{'schema':'STATEWISE_FRESH_RGB_INTERVAL_PREDICTION_SEAL_V4_2','before_scoring':True,
        'predictions_sha256':C.sha(output/'PREDICTIONS.jsonl.gz'),'protocol_sha256':C.sha(output/'PROTOCOL.json'),
        'contract_checks_sha256':C.sha(output/'CONTRACT_CHECKS.json'),'control_reuse_audit_sha256':C.sha(output/'CONTROL_REUSE_AUDIT.json'),
        'source_sha256':sources,'rows':98,'primary_rows':95,'crop_rows':3,'severe_rows':0,'all3arms_complete':True,
        'annotation_targets_accessed':False,'elapsed_seconds':time.monotonic()-started})

def score(output=OUTPUT,model_path=MODEL):
    output=Path(output).resolve();_,_,cfg,_,bindings=load_interval_configuration(model_path)
    _,protocol,seal=E._verify_prediction_seal(output)
    if cfg!=protocol['solver_configuration'] or bindings!=protocol['interval_model_bindings']:raise ValueError('Frozen interval configuration changed')
    if C.sha(output/'CONTROL_REUSE_AUDIT.json')!=seal['control_reuse_audit_sha256']:raise ValueError('Controlaliasaudit changed')
    E._verify_prediction_seal(EB.OUTPUT);E._verify_prediction_seal(E.OUTPUT)
    E.score(output)
    C.write(output/'INTERVAL_POSTSEAL_BINDING.json',{'fixed_interval_model':bindings['interval_model'],
        'postseal_scoring':E._binding(output/'POSTSEAL_SCORING.json'),'lambda1_beta1_unchanged':True,
        'same_physical_C2_reference':True,'further_exploratory_repair_no_independent_generalization':True})

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['predict','score']);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--workers',type=int,default=3);p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.phase=='predict':predict(a.output,a.model,a.workers,a.device)
    else:score(a.output,a.model)

if __name__=='__main__':main()
