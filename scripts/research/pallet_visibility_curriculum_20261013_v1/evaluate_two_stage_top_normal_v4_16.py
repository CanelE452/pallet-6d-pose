"""Two-stage normal-only prior controls over the immutable whole238 archive.

Stage1: full original BI/legacy partial+tangent4+outline+top-normal, both WD.
Stage2: freeze its predicted family and refine D-merged partial+tangent4 with
the same top-normal prior. Sigma2 and empirical CALmax/3 are fixed controls.
No new NN, line, P1/P3P/global search, calibration, or GT-based routing occurs.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import argparse
import shutil
import time

import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_fixed_wd_archived_tangent_v4_15 as ARCHIVE
from . import evaluate_two_wd_families_v4_9 as W
from . import statewise_two_wd_family_v4_9 as FAMILY
from . import statewise_archived_BI_tangent_solver_v4_15 as T
from . import statewise_top_normal_outline_solver_v4_16 as G
from . import top_plane_normal_factor_v4_16 as N
from . import calibrate_top_plane_normal_v4_16 as CAL
from . import calibrate_top_plane_normal_conservative_v4_16b as BOUND
from . import score_remainder_closed_sources_v4_14 as CLOSED
from .statewise_uncertainty_calibration_v4 import load_uncertainty_model

OUTPUT=ARCHIVE.OUTPUT.parent/'two_stage_top_normal_eval_v4_16'
ARMS=('SAME_FROZEN_E','NORMAL_SIGMA2_TWO_STAGE_ALL_T4','NORMAL_CALMAX_TWO_STAGE_ALL_T4')
PROFILES=((ARMS[1],CAL),(ARMS[2],BOUND))


def _load():
    parent=C.read(ARCHIVE.OUTPUT/'PROTOCOL.json');seal=C.read(ARCHIVE.OUTPUT/'INPUT_SEAL.json')
    journal=ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'
    if (seal['rows']!=238 or C.sha(journal)!=seal['input_sha256']
        or C.sha(ARCHIVE.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Exact original238 prepared mathematical archive required')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(parent[key])
    closure=CLOSED.verify_additional_source_closure()
    models={};inputs=deepcopy(parent['input_sha256']);sources=deepcopy(parent['source_sha256'])
    for arm,module in PROFILES:
        folder=module.OUTPUT;model=C.read(folder/'MODEL.json');modelseal=C.read(folder/'MODEL_SEAL.json')
        if C.sha(folder/'MODEL.json')!=modelseal['model_sha256']:raise ValueError('CAL normal model changed')
        models[arm]=module.normal_factor_parameters(model)
        for path in (folder/'MODEL.json',folder/'MODEL_SEAL.json',folder/'PROTOCOL.json'):
            inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for path in (ARCHIVE.OUTPUT/'PROTOCOL.json',ARCHIVE.OUTPUT/'INPUT_SEAL.json',journal,
                 CLOSED.CLOSURE/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json'):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for module in (__file__,ARCHIVE.__file__,G.__file__,N.__file__,CAL.__file__,BOUND.__file__,CLOSED.__file__):
        path=Path(module);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    tasks=list(C.rows(journal))
    if [t['id']for t in tasks]!=parent['ids']or len(tasks)!=238:raise ValueError('Every original238 row/order required')
    for task in tasks:
        geometry=task['finish_input']['_original_source']['initial_geometry']
        bank=task['stage1_family_seed_bank'];original=bank['ORIGINAL_WD']['family']
        for key,field in (('native_xyz','native_xyz'),('physical_registration_Q','physical_registration_Q'),('cf_extents','cf_extents')):
            if not np.allclose(original[key],geometry[field],atol=1e-12,rtol=0):raise ValueError('Original basis must precede WD generation')
        models_generated=FAMILY.make_wd_families(original['native_xyz'],original['coarse_pose'])
        for family in models_generated:
            saved=bank[family['name']]['family']
            for key in ('native_xyz','physical_registration_Q','cf_extents'):
                if not np.allclose(family[key],saved[key],atol=1e-12,rtol=0):raise ValueError('Both-family saved model/Q differs')
            if not np.allclose(np.asarray(family['coarse_pose']['R'])@N.TOP_NORMAL_CF,
                               np.asarray(original['coarse_pose']['R'])@N.TOP_NORMAL_CF,atol=1e-12,rtol=0):
                raise ValueError('Family generation changed initial top-normal center')
        for key in ('boundary_original_legacy','boundary_all'):
            records=T.archive_legacy_B_tangent_predictions(task[key],task['spec']['visible'],task['spec']['states'],task['spec']['config'])
            normalized=T._boundary_inputs(records,task['spec']['states'],task['spec']['image_hw'],original['native_xyz'])
            T.validate_archived_predictions(normalized,task['spec']['visible'],task['spec']['config'])
    return tasks,models,sources,inputs,parent['protected_user_sha256'],closure


def _finish(task,model,family,solution,records,name):
    finish=deepcopy(task['finish_input'])
    finish['partial_boundary_observation']['boundary_observations']=deepcopy(records)
    arm=W._family_finish(finish,model,family,name,solution)
    readout=arm['keypoint_readout'];readout['partial_BI_observations']=deepcopy(records)
    readout['partial_B_observations']=[deepcopy(r)for r in records if r['state']=='B']
    for r in records:
        k=r['native_corner'];readout['reported_points'][k]=None
        readout['point_sources'][k]='PARTIAL_'+r['state']+'_RGB_NORMAL_AND_UNCERTAIN_ARCHIVED_TANGENT'
    readout['uncertain_archived_tangent_factor']=deepcopy(solution['tangent_prediction_factor_audit'])
    arm['solution']=E._compact_solution(arm['solution'])
    return arm


def _preserve_ambiguity(arm,baseline,stage1):
    historical=deepcopy(baseline['hierarchical_correspondence_audit']['initial_family_ambiguity'])
    prior_ambiguous=bool(historical.get('detected')or stage1['family_ambiguity'].get('detected'))
    within=deepcopy(arm['ambiguity_audit']);ambiguous=bool(prior_ambiguous or within.get('material_ambiguity'))
    arm['hierarchical_correspondence_audit']=dict(
        rule='full BI/legacy tangent/outline/top-normal predicted family fixed before D+ALL_T refinement',
        initial_family_ambiguity=deepcopy(stage1['family_ambiguity']),
        historical_initial_family_ambiguity=historical,refined_within_family_ambiguity=within,
        fixed_family=stage1['selected_family'],old_best_pose_fallback=False,
        prior_is_correlated_regularizer_not_new_image_information=True,
        initial_family_or_refined_pose_globally_unique_proved=False,GT_inputs=False)
    if ambiguous:
        arm['ambiguity_audit'].update(material_ambiguity=True,normal_regularization_does_not_prove_uniqueness=True)
        arm['solution'].setdefault('ambiguity',{})['detected']=True
        arm['solution'].update(OT_solved=False,OT_unique_solution_claimed=False)
        arm['keypoint_readout'].update(OT_solved=False,OT_unique_solution_claimed=False)
    return arm


def _run(task,model,profiles):
    started=time.monotonic();before=E._digest(task);spec=task['spec'];bank=task['stage1_family_seed_bank']
    original=bank['ORIGINAL_WD']['family'];arms={ARMS[0]:deepcopy(task['baseline'])};audits={}
    legacy=T.archive_legacy_B_tangent_predictions(task['boundary_original_legacy'],spec['visible'],spec['states'],spec['config'])
    merged=deepcopy(task['boundary_all'])
    for name,parameters in profiles.items():
        def solve(xyz,K,states,visible,domains,hw,coarse,**kwargs):
            match=next(value for value in bank.values()if np.allclose(value['family']['native_xyz'],xyz,atol=1e-12,rtol=0))
            if not np.allclose(match['family']['physical_registration_Q'],coarse['physical_registration_Q'],atol=1e-12,rtol=0):
                raise ValueError('Saved starts must use their actual WD registration')
            return G.refine_global_outline_joint_pose(xyz,K,states,visible,domains,hw,coarse,
                config=kwargs['config'],outline_observation=task['outline'],boundary_observations=legacy,
                u_regions=task['u_regions'],initial_solutions=match['initial_solutions'],
                initializations=match['initializations'],normal_parameters=parameters)
        stage1=FAMILY.solve_two_wd_families(original['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
            spec['image_hw'],original['coarse_pose'],config=spec['config'],boundary_observations=legacy,
            u_regions=task['u_regions'],solver_callable=solve)
        chosen=bank[stage1['selected_family']];family=chosen['family']
        initials=deepcopy(chosen['initial_solutions'])
        # Stage1's newly obtained same-basis pose/sample bank supplements,
        # rather than substitutes for, all bound existing search seeds.
        initials.append(deepcopy(stage1['solution']))
        solution=G.refine_global_outline_joint_pose(family['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],
            spec['image_hw'],family['coarse_pose'],config=spec['config'],outline_observation=task['outline'],
            boundary_observations=merged,u_regions=task['u_regions'],initial_solutions=initials,
            initializations=chosen['initializations'],normal_parameters=parameters)
        arm=_finish(task,model,family,solution,merged,name)
        arm=_preserve_ambiguity(arm,task['baseline'],stage1);arms[name]=arm
        audits[name]=dict(normal_parameters=parameters,stage1_selected_family=stage1['selected_family'],
            stage1_family_ambiguity=stage1['family_ambiguity'],stage1_all_families=stage1['families'],
            stage1_full_BI_original_legacy_tangent_only=True,stage2_D_merged_all_tangent=True,
            stage2_family_frozen=True,same_initial_top_normal_center_both_stages=True,
            prior_excluded_from_image_information_rank=True,new_full_P1_P3P_global_search=False,
            tangent_variance_inflation=4.,normal_prior_audit=solution['top_plane_normal_prior_audit'])
    if E._digest(task)!=before:raise ValueError('Immutable archived measurements/states/domains changed')
    return C.finite(dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],
        image_sha256=task['image_sha256'],states=spec['states'],arms=arms,
        fixed_family={ARMS[0]:task['fixed_family'],**{name:a['stage1_selected_family']for name,a in audits.items()}},
        tangent_control_audit=audits,normal_two_stage_audit=audits,input_sha256=before,
        elapsed_seconds=time.monotonic()-started,
        contract=dict(S_U_local_queries=0,U_only_regions=True,all_OT_U_hard_joint=True,
            old_BI_q_covariance_reliability_and_partial_normal_variance_preserved=True,
            no_GT_or_frameID_or_error_parameter_routing=True,all_fallbacks_retained=True,
            top_normal_prior_is_not_full_R_t_yaw_prior=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every previous two-stage experiment')
    tasks,profiles,sources,inputs,protected,closure=_load()
    for hashes in (sources,inputs,protected):E._verify_sources(hashes)
    output.mkdir(parents=True)
    protocol=dict(schema='TWO_STAGE_TOP_NORMAL_ALL_T4_ALL238_V4_16',ids=[t['id']for t in tasks],arms=list(ARMS),
        population=dict(CAL=46,TEST=49,REMAINDER=143,severe=0),normal_profiles=profiles,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,source_closure=closure,
        immutable_prepared_archive=E._binding(ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'),
        stage1='full original BI + original legacy B normal/tangent4 + global outline + top-normal, both WD',
        stage2='freeze stage1 predicted family, D-merged all tangents4 + same normal and outline',
        regularizer_not_independent_image_evidence=True,coarse_native_yaw_position_depth_not_penalized=True,
        CAL_full_model_resubstitution_not_independent_CAL_holdout=True,
        CAL_SIGMA2_and_empirical_CAL_MAX_div3_fixed_before_this_control_metrics=True,
        all238_sealed_before_new_control_metrics=True,new_NN_line_or_full_global_search=False,
        exploratory_reused_TEST49_and_remaining143=True,all_flags_and_failures_kept=True)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        path=output/'code_snapshot'/relative;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,path)
        if C.sha(path)!=digest:raise ValueError('V16 source snapshot changed')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],protocol_sha256=C.sha(output/'PROTOCOL.json'),
        parent_input_seal=E._binding(ARCHIVE.OUTPUT/'INPUT_SEAL.json'),before_new_v16_pose_fits_and_metrics=True))


def predict(output=OUTPUT,workers=3):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if seal['rows']!=238 or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']:raise ValueError('V16 input seal changed')
    E._verify_binding(seal['parent_input_seal'])
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Preserve completed V16')
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    tasks=list(C.rows(ARCHIVE.OUTPUT/'INPUTS.jsonl.gz'));model=load_uncertainty_model(ARCHIVE.MODEL)
    if [t['id']for t in tasks]!=protocol['ids']:raise ValueError('Every ordered238 task required')
    started=time.monotonic();done={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))as pool:
        futures={pool.submit(_run,t,model,protocol['normal_profiles']):t['id']for t in tasks}
        for future in as_completed(futures):
            row=future.result();done[row['id']]=row;C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(f'{len(done)}/238 {row["id"]}',flush=True)
    for key in ('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    CLOSED.verify_additional_source_closure()
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[t['id']]for t in tasks])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),all3_arms_complete=True,before_new_control_metric_read=True,
        ordered_ids=protocol['ids'],elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    # The frozen scorer's all238/reference/seal checks are reused verbatim;
    # only the control arm names differ, via a private function namespace.
    return T.T._clone(ARCHIVE.score,{'ARMS':ARMS})(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('check','prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=min(3,os.cpu_count()or 1))
    args=parser.parse_args()
    if args.stage=='check':print('V16_PREFLIGHT',len(_load()[0]),flush=True)
    elif args.stage=='prepare':prepare(args.output)
    elif args.stage=='predict':predict(args.output,args.workers)
    else:score(args.output)
