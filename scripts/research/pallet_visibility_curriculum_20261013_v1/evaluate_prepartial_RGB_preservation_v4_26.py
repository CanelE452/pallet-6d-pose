"""Preserve pre-partial RGB points with cost-only duplicate removal.

One immutable238 input population, a separate CAL edge-noise fit, original
hard guards and v19 starts. Zero and translation coarse-prior controls and
optional one-sided containment are uniform. No numeric e-pose fallback.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_key]='1'
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import argparse
import shutil
import sys
import time
import types
import numpy as np

from . import evaluate_source_preserving_tangent_mask_v4_24 as V
from . import evaluation_bias_v4_1 as POSE_MODEL
from . import coarse_pose_prior_v4_25 as PRIOR
from . import coarse_prior_containment_problem_v4_25 as P
from . import evaluate_coarse_prior_containment_v4_25 as PREVIOUS
from . import prepartial_RGB_point_preservation_v4_26 as RESTORE
from . import calibrate_prepartial_RGB_edge_mixture_v4_26 as RESTORE_CAL

OLD,C,E,BASE,REFINE=V.OLD,V.C,V.E,V.BASE,V.REFINE
OUTPUT=V.OUTPUT.parent/'prepartial_RGB_preservation_eval_v4_26'
PLAN=V.OUTPUT.parent/'prepartial_RGB_preservation_plan_v4_26/PREDECLARATION.json'
ARMS=('SAME_FROZEN_E','PREPARTIAL_RGB_NATIVE','PREPARTIAL_RGB_NATIVE_E_FAMILY',
    'PREPARTIAL_RGB_TRANSLATION_NATIVE','PREPARTIAL_RGB_TRANSLATION_NATIVE_E_FAMILY',
    'PREPARTIAL_RGB_TRANSLATION_CONTAINMENT','PREPARTIAL_RGB_TRANSLATION_CONTAINMENT_E_FAMILY')
# FULL is only the shared prior schema at exactly zero weight in the no-P
# control. It is not an active full-rotation regularizer in this experiment.
PROFILES=(('NATIVE_ZERO_COARSE','FULL',False,ARMS[1:3]),
    ('TRANSLATION_NATIVE','TRANSLATION',False,ARMS[3:5]),
    ('TRANSLATION_CONTAINMENT','TRANSLATION',True,ARMS[5:7]))
PROFILE_ARMS={name:arms for name,mode,containment,arms in PROFILES}


def _verified_models():
    models=V._verified_models()
    models['RESTORED']=V._checked_model(RESTORE_CAL)
    selected,model,config,choice,seal=POSE_MODEL.load_bias_selection()
    if config['prior_weight']!=1. or config['bias_strength']!=1.:
        raise ValueError('Reuse exact frozen CAL lambda1 beta1, no new tuning')
    models.update(COARSE=model,COARSE_BINDING=selected['calibration_model'],COARSE_CHOICE=choice,COARSE_CHOICE_SEAL=seal)
    return models


def _prepare_pose_prior(task,models,mode):
    original=task['families']['ORIGINAL_WD']
    initial={**original['coarse_pose'],'physical_registration_Q':original['physical_registration_Q']}
    return PRIOR.prepare_coarse_pose_prior(original['native_xyz'],initial,models['COARSE']['prior_covariance6'],
        models['COARSE']['prior_fit']['median_bias6'],
        calibration_model_digest_sha256=models['COARSE_BINDING']['sha256'],mode=mode,
        factor_weight=0. if mode=='FULL' else 1.,bias_strength=1.)


def _check_declaration(declaration):
    expected=[dict(name=name,schema_mode=mode,coarse_prior_weight=0. if mode=='FULL' else 1.,
        containment=containment)for name,mode,containment,_ in PROFILES]
    if(declaration['arms']!=list(ARMS)or declaration['profiles']!=expected
            or declaration['all238']!=dict(CAL=46,TEST=49,REMAINDER=143,severe=0)
            or declaration['new_RGB_NN_queries']!=0 or declaration['distillation']!=0):
        raise ValueError('Exact preregistered restoration, zero/TP and containment profiles required')


class _RestoredProblem(V._ExactReplacementCount,RESTORE.CompleteBIImageChainProblem):pass


def _problem(task,family,models):
    spec=task['spec']
    original=OLD.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
        spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
        config={**spec['config'],'prior_weight':0.},boundary_observations=task['boundary_merged'],
        u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    if not np.allclose(original.Q,family['physical_registration_Q'],atol=1e-12,rtol=0):
        raise ValueError('Exact physical Q required')
    base=RESTORE.prepare_prepartial_native_problem(original,visible_measurements=spec['visible'],
        initial_self_hidden=task['initial_self_hidden'])
    noise=RESTORE_CAL.mixture_noise_parameters(models['RESTORED'])
    bundle=RESTORE.authenticate_complete_bi_image_chain_groups(base,task['new_chain_observation'],
        junction_proposals=task['complete_BI_evidence']['junction_proposals'],visible_measurements=spec['visible'],
        initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],**noise)
    problem=_RestoredProblem(base,bundle,overlap_policy=RESTORE.PROTOTYPE_OVERLAP_POLICY)
    V._calibrate_plan_tangents(problem,models['TANGENT'])
    problem.reference_start_problem=OLD.V.PARENT._problem(task,family,models['NMS']['OLD'])[0]
    return problem,{**bundle,'runtime_complete_BI_replacement_plan':problem.plan}


def _restored_ids(task):
    partial={b['native_corner']for b in task['boundary_merged']}
    result=[]
    for measurement in task['spec']['visible']:
        k=measurement['native_corner']
        if task['initial_self_hidden'][k]or task['states'][k]not in('B','I'):
            raise ValueError('NonS BI gate must precede restoration source fields')
        if k in partial and measurement['source']!=V.SOURCE.FALLBACK_SOURCE:result.append(k)
    return sorted(result)


def _readout(task,selected,pose,bundle):
    restored=_restored_ids(task)
    records=[b for b in task['boundary_merged']if b['native_corner']not in restored]
    readout=BASE._readout(task,records,selected,pose)
    plan=bundle['runtime_complete_BI_replacement_plan']
    for record in plan['corner_plans']:
        if not record['point_cost_revoked']:continue
        k=record['native_corner']
        if task['states'][k]not in('B','I')or task['initial_self_hidden'][k]or k in restored:
            raise ValueError('Only non-restored nonS BI can revoke a point cost')
        readout['reported_points'][k]=None
        readout['point_sources'][k]=('IMAGE_LINES_AND_UNCERTAIN_ARCHIVED_TANGENT'
            if 'tangent_kind'in record else'IMAGE_LINES_WITHOUT_PRECISE_2D_KP')
    readout.update(restored_prepartial_RGB_ids=restored,restored_points_are_exact_old_image_observations=True,
        no_new_keypoint_observations_created=True)
    return readout


def _run(task,models):
    started=time.monotonic();before=E._digest(task)
    regions,oparams,original_S=OLD._prior_inputs(task,models)
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration'])
    supported=bool(task['outline'].get('supported'));shape_weight=cfg['outline_factor_weight']if supported else 0.
    arms={ARMS[0]:deepcopy(task['baseline'])};audits={};start_banks={};native_endpoints={}
    prior_inputs={mode:_prepare_pose_prior(task,models,mode)for mode in('TRANSLATION','FULL')}
    for profile,mode,containment,profile_arms in PROFILES:
        candidates=[];details={};searches={};bundles={};endpoints=[]
        for name,family in task['families'].items():
            base,bundle=_problem(task,family,models);bundles[name]=bundle
            rebound=OLD.P.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
            bounded=OLD.P.BoundedGeometricPriorsProblem(base,regions,oparams,rebound)
            bound_prior=PRIOR.rebind_coarse_pose_prior(prior_inputs[mode],family['native_xyz'],family['physical_registration_Q'])
            regularized=PRIOR.CoarsePosePriorProblem(bounded,bound_prior)
            problem=P.ContinuousContainmentProblem(regularized,task['outline'],cfg)if containment else regularized
            Q=np.asarray(family['physical_registration_Q'])
            proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
                for i,p in enumerate(task['proposals'])]

            def add(R,t,token,origin):
                checked=problem.evaluate(R,t)
                shape=float(checked['global_containment_cost'])if containment and checked.get('objective')is not None else 0.
                if checked['accepted']:
                    expected=checked['native_observation_objective']+checked['regularizer_objective']+(shape_weight*shape if containment else 0.)
                    if not np.isclose(checked['objective'],expected,atol=1e-9,rtol=1e-10):raise ValueError('Identical optimizer/selector components required')
                candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                    all_joint_constraints_satisfied=checked['accepted'],native_costs=BASE._bridge_costs(checked.get('native_costs',[])),
                    regularizer_costs=checked.get('regularizer_costs',[]),global_containment_cost=shape,
                    global_mask_cost=shape,legacy_mask_field_contains_only_containment_scalar=True,origin=origin))
                if checked['accepted']:details[token]=dict(evaluation=checked,family=family,origin=origin)

            for index,proposal in enumerate(proposals):add(proposal['R'],proposal['t'],name+'/CACHED_'+str(index),'CACHED')
            if containment:
                for index,endpoint in enumerate(native_endpoints[mode]):
                    if endpoint['family']==name:add(endpoint['R'],endpoint['t'],name+'/NATIVE_ENDPOINT_'+str(index),'SAME_MODE_NATIVE_ENDPOINT')
            refined=V.V._REFINER(problem,proposals,family['coarse_pose'],configuration=REFINE.DEFAULT_CONFIG)
            for index,result in enumerate(refined['solutions']):
                add(result['R'],result['t'],name+'/REFINED_'+str(index),'CONTINUOUS_CONTAINMENT_ENDPOINT'if containment else'CONTINUOUS_PRIOR_NATIVE_ENDPOINT')
                endpoints.append(dict(family=name,R=result['R'],t=result['t']))
            names=refined['start_selection']['selected_start_names']
            if name in start_banks and names!=start_banks[name]:raise ValueError('Exact unchanged v19-native start names required across all profiles')
            start_banks[name]=names
            searches[name]=dict(search={**refined['search'],'pose_prior':mode!='FULL','CAL_coarse_prior_mode':mode,
                'zero_weight_FULL_schema_not_active_rotation_prior':mode=='FULL',
                'continuous_containment_gradient_active':bool(containment and supported and shape_weight>0),
                'mask_objective':False,'old_full_outline_factor_used':False,
                'optimization_objective':'native + boundedO/S + CALcoarseprior'+(' + one-sidedcontainment'if containment else'')},
                start_selection=refined['start_selection'],candidates=refined['candidates'],
                feasible_endpoints=[{k:deepcopy(r.get(k))for k in('R','t','R_physical','objective','native_observation_objective',
                    'regularizer_objective','CAL_coarse_pose_prior_objective','global_containment_cost','weighted_containment_objective',
                    'initialization','branch')}for r in refined['solutions']])
        if not containment:native_endpoints[mode]=endpoints
        selections=(P.select_candidates(task['states'],candidates,containment_weight=shape_weight if containment else 0.),
            P.select_candidates(task['states'],candidates,containment_weight=shape_weight if containment else 0.,locked_family=task['baseline']['WD_family']))
        for arm,selection in zip(profile_arms,selections):
            token=selection['selected_name'];locked=arm.endswith('_E_FAMILY')
            if token is None:
                family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose'];origin='OWN_INITIAL_COARSE_FALLBACK'
                selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,native_costs=[],
                    regularizer_costs=[],reason='no_current_eligible_feasible_candidate')
            else:
                detail=details[token];origin=detail['origin']
                selected={**detail['evaluation'],'family':detail['family'],'selected_endpoint_origin':origin}
                chosen=next(r for r in selection['candidates']if r['name']==token)
                if not np.isclose(chosen['total_energy'],selected['objective'],atol=1e-9,rtol=1e-10):raise ValueError('Current selected total must equal current optimized objective')
            pose=BASE._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
            readout=_readout(task,selected,pose,bundles[selected['family']['name']])
            readout.pop('finite_candidate_hypothesis_not_continuous_or_unique_solution',None)
            readout.update(continuous_SE3_search_performed=True,selected_endpoint_origin=origin,S_image_or_2D_terms=0,
                S_model_prior_enabled=True,U_range_only=True,O_point_terms=0,coarse_pose_prior_is_correlated_model_regularizer=True,
                old_full_outline_factor_used=False,containment_is_conditional_shape_cue_not_hidden_keypoint=True)
            if selected['accepted']:
                for k,state in enumerate(task['states']):
                    if state=='O':readout['point_sources'][k]='CANVAS_FEASIBLE_MODEL_PROJECTION_BOUNDED_COARSE_O_PRIOR_NOT_UNIQUE'
            lockaudit=dict(enabled=locked,requested_predicted_family=task['baseline']['WD_family']if locked else None,
                locked_family_constraint_satisfied=bool(locked and selected['accepted']and selected['family']['name']==task['baseline']['WD_family']),
                e_numeric_pose_used=False,e_family_is_prediction_not_physical_identity_certificate=True)
            if locked:lockaudit.update(e_source_ambiguity_audit=deepcopy(task['baseline'].get('ambiguity_audit')),
                e_source_fallback_audit=deepcopy(task['baseline'].get('fallback_audit')))
            arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),solution=selected,
                keypoint_readout=readout,WD_family=selected['family']['name'],method=arm,candidate_selection=selection,
                ambiguity_audit=deepcopy(selection['ambiguity']),new_image_line_group_authentication=bundles,
                early_family_audit=lockaudit,fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],
                    old_best_pose_fallback=False,lock_requested_but_unsatisfied=bool(locked and not selected['accepted'])))
            audits[arm]=dict(profile=profile,CAL_coarse_prior_mode=mode,containment_enabled=containment,
                family_refinements=searches,all_cached_contenders_retained=True,
                exact_start_names_across_profiles=deepcopy(start_banks),same_v19_unregularized_native_six_start_policy=True,
                same_mode_native_endpoints_retained=containment,old_full_outline_factor_or_tie_break_used=False,
                native_costs_authenticated_separately_from_regularizers=True,early_family_constraint=locked,
                cost_nonworsening_is_not_GT_accuracy_guarantee=True)
    if E._digest(task)!=before:raise ValueError('Frozen v24 inputs mutated')
    return C.finite(dict(schema='PREPARTIAL_RGB_PRESERVATION_PREDICTION_V26',id=task['id'],partition=task['partition'],session=task['session'],
        image=task['image'],image_sha256=task['image_sha256'],states=task['states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family')for a in ARMS},bounded_prior_refinement_audit=audits,input_sha256=before,
        original_first_pose_prior_inputs=prior_inputs,elapsed_seconds=time.monotonic()-started,
        O_localization_abstention=deepcopy(task['O_localization_abstention']),
        tangent_control_audit={a:dict(remaining_fallback_partial_variance_inflation=4.,archived_complete_variance_model=V.TCAL.tangent_variance_parameters(models['TANGENT']),
            source_preserving=True,original_tangent_covariance_q_guide_reliability_unchanged=True)for a in ARMS[1:]},
        contract=dict(all238_retained=True,severe=0,original_e_default_unchanged=True,new_RGB_or_NN_queries=0,new_CAL_edge_noise_fits=1,new_CAL_coarse_prior_fits=0,
            S_raw_2D_image_regions=0,S_readout_pose_projection_only=True,U_range_only=True,original_hard_guards_unchanged=True,
            remaining_fallback_partial_T4_unchanged=True,restored_RGB_partial_normal_tangent_costs_removed=True,
            FULL_prior_schema_is_zero_weight_noP_control=True,no_GT_ID_error_or_partition_routing=True,no_selective_oldpose_fallback=True,
            original_first_coarse_pose_target_not_e_pose=True,old_full_outline_factor_used=False,no_default_adoption=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous preparations')
    models=_verified_models();parent=C.read(PREVIOUS.OUTPUT/'PROTOCOL.json');seal=C.read(PREVIOUS.OUTPUT/'INPUT_SEAL.json')
    declaration=C.read(PLAN)
    _check_declaration(declaration)
    if declaration['arms']!=list(ARMS)or seal['rows']!=238 or C.sha(PREVIOUS.OUTPUT/'INPUTS.jsonl.gz')!=seal['input_sha256']:
        raise ValueError('Exact seven-arm declaration and frozen parent238 required')
    if C.sha(PREVIOUS.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']:raise ValueError('Parent protocol changed')
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    restored_seal=C.read(RESTORE_CAL.OUTPUT/'MODEL_SEAL.json')
    for key,target in(('source_sha256',sources),('input_sha256',inputs)):
        E._verify_sources(restored_seal[key])
        for relative,digest in restored_seal[key].items():
            if relative in target and target[relative]!=digest:
                raise ValueError('Frozen restoration CAL dependency conflicts with parent')
            target[relative]=digest
    for module in(sys.modules[__name__],RESTORE,RESTORE_CAL,PREVIOUS,PRIOR,PRIOR.BIAS,P,P.FACTOR,POSE_MODEL):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    choice=POSE_MODEL.C.read(POSE_MODEL.SELECTION)
    for relative,digest in choice['source_sha256'].items():
        if relative in sources and sources[relative]!=digest:raise ValueError('Frozen legacy prior source versions differ')
        sources[relative]=digest
    prior_folder=C.ROOT/Path(models['COARSE_BINDING']['path']).parent
    priorseal=C.read(prior_folder/'SEAL.json')
    for name,digest in priorseal['files'].items():
        path=prior_folder/name
        if C.sha(path)!=digest:raise ValueError('Frozen original coarse-prior calibration changed')
        inputs[str(path.relative_to(C.ROOT))]=digest
    for path in(PLAN,PREVIOUS.OUTPUT/'PROTOCOL.json',PREVIOUS.OUTPUT/'INPUT_SEAL.json',
            PREVIOUS.OUTPUT/'INPUTS.jsonl.gz',prior_folder/'SEAL.json',POSE_MODEL.SELECTION,POSE_MODEL.SELECTION.with_name('SELECTION_SEAL.json')):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    counts=Counter();ids=[];modes=Counter()
    for task in C.rows(PREVIOUS.OUTPUT/'INPUTS.jsonl.gz'):
        ids.append(task['id']);counts[task['partition']]+=1;OLD._prior_inputs(task,models)
        for mode in('TRANSLATION','FULL'):
            prepared=_prepare_pose_prior(task,models,mode)
            for family in task['families'].values():
                bound=PRIOR.rebind_coarse_pose_prior(prepared,family['native_xyz'],family['physical_registration_Q'])
                base,_=_problem(task,family,models)
                modes.update(groups=len(base.groups),revoked=len(base.revoked),prior_bindings=1)
    if counts!=Counter(CAL=46,TEST=49,REMAINDER=143)or ids!=parent['ids']:raise ValueError('Keep all original238/order including every fallback')
    for mapping in(sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol={**parent,'schema':'PREPARTIAL_RGB_PRESERVATION_ALL238_PROTOCOL_V26','arms':list(ARMS),'arm_count':len(ARMS),
        'source_sha256':sources,'input_sha256':inputs,'declaration':E._binding(PLAN),
        'original_full_schema_prior_disabled_in_noP_control':True,
        'coarse_prior_model_binding':models['COARSE_BINDING'],'coarse_prior_choice':models['COARSE_CHOICE'],
        'coarse_prior_choice_seal':models['COARSE_CHOICE_SEAL'],'new_CAL_parameter_fits':1,
        'new_CAL_edge_noise_fits':1,'new_CAL_coarse_prior_fits':0,
        'restored_point_fields_exact_from_prepartial_visible':True,'restored_same_owner_cost_factors_removed':True,
        'input_bytes_identical_to_v25':True,'restored_CAL_model_binding':E._binding(RESTORE_CAL.OUTPUT/'MODEL.json'),
        'objective':'restored prepartial RGB native + boundedO/S + optional translation CALprior + optional conditional containment',
        'old_full_outline_factor_or_tie_break_used':False,'coarse_prior_weights':dict(noP=0.,translation=1.,beta_bias=1.),
        'coarse_prior_modes':['ZERO_WEIGHT_FULL_SCHEMA','TRANSLATION'],'three_profiles_each_unlocked_and_e_family_selectors':True,
        'e_family_label_indirectly_depends_on_historical_whole_mask':True,
        'containment_is_soft_outside_only_not_hard_full_foreground_subset_certificate':True,
        'all238_all_declared_arms_prediction_seal_before_reference_decode':True,
        'score_streams_predictions_but_uses_same_frozen_numeric_metric_and_reference_functions':True,
        'no_default_adoption':True}
    for key in('mask_not_continuously_optimized','continuous_mask_only_in_last_two_arms','C_bank_adds_source_native_endpoints_uniformly',
            'legacy_parent_all5_flag_does_not_describe_this_experiment','mask_selects_same_profile_bank_only',
            'four_optimization_profiles_each_unlocked_and_e_family_selectors','declaration_metadata_clarification'):
        protocol.pop(key,None)
    for name in('MODEL.json','MODEL_SEAL.json','PROTOCOL.json','PREPARATION_SEAL.json','CAL_LOGO_REPORT.json','CAL_MIXTURE_ERRORS.jsonl.gz'):
        path=RESTORE_CAL.OUTPUT/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    C.write(output/'PROTOCOL.json',protocol);shutil.copyfile(PREVIOUS.OUTPUT/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=ids,protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),both_WD_both_coarse_prior_modes_preflight=True,
        metadata_inventory=dict(modes),before_new_predictions_or_metrics=True))
    print(dict(prepared=238,arms=len(ARMS),metadata_inventory=dict(modes)),flush=True)


def predict(output=OUTPUT,workers=8):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if (seal['rows']!=238 or seal['input_sha256']!=C.sha(output/'INPUTS.jsonl.gz')or
            seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')or protocol['arms']!=list(ARMS)):
        raise ValueError('Exact prepared all238 seven-arm seal required')
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Preserve completed predictions')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    models=_verified_models();started=time.monotonic();done=set();tasks=iter(C.rows(output/'INPUTS.jsonl.gz'))
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'));pending={}
    def submit_one():
        task=next(tasks,None)
        if task is None:return False
        pending[pool.submit(_run,task,models)]=task['id'];return True
    try:
        for _ in range(workers*2):
            if not submit_one():break
        while pending:
            ready,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                expected=pending.pop(future);row=future.result()
                if row['id']!=expected or expected in done or set(row['arms'])!=set(ARMS):raise ValueError('Unique complete seven-arm row required')
                C.write(output/'frames'/(expected.replace(':','__')+'.json'),row);done.add(expected)
                C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=expected,elapsed_seconds=time.monotonic()-started))
                print(str(len(done))+'/238 '+expected,flush=True);submit_one()
    except BaseException as error:
        C.write(output/'FAILED_BEFORE_SEAL.json',dict(completed=len(done),total=238,error_type=type(error).__name__,reason=str(error),new_reference_reads=0))
        for future in pending:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if done!=set(protocol['ids']):raise ValueError('Keep all238 including failures')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',(C.read(output/'frames'/(ident.replace(':','__')+'.json'))for ident in protocol['ids']))
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),ordered_ids=protocol['ids'],arms=list(ARMS),arm_count=len(ARMS),
        all_declared_arms_complete=True,all7_arms_complete=True,all3_arms_complete=True,
        legacy_all3_flag_is_scorer_compatibility_only=True,before_new_control_metric_read=True,elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    """Stream large predictions; unchanged old metric and reference functions."""
    output=Path(output).resolve();seal=C.read(output/'PREDICTION_SEAL.json');protocol=C.read(output/'PROTOCOL.json')
    if not(seal.get('all7_arms_complete')is True and seal.get('all_declared_arms_complete')is True
            and seal.get('arms')==list(ARMS)and seal.get('arm_count')==len(ARMS)and seal['rows']==238
            and seal['predictions_sha256']==C.sha(output/'PREDICTIONS.jsonl.gz')
            and seal['protocol_sha256']==C.sha(output/'PROTOCOL.json')and seal['input_seal_sha256']==C.sha(output/'INPUT_SEAL.json')
            and seal['ordered_ids']==protocol['ids']):raise ValueError('Full seven-arm seal required before numeric references')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    auditpath=output/'independent_root_audit_v1/AUDIT.json';audit=C.read(auditpath);auditseal=C.read(auditpath.with_name('AUDIT_SEAL.json'))
    if (audit['rows']!=238 or audit['failure_count']or audit['failures']or C.sha(auditpath)!=auditseal['receipt_sha256']
            or audit['prediction_seal']['sha256']!=C.sha(output/'PREDICTION_SEAL.json')):
        raise ValueError('Current independent all238 zero-failure pose audit required')
    A=BASE.ARCHIVE;references={};bindings={};reports={};rows={p:[]for p in('CAL','TEST','REMAINDER')}
    # Only small metric inputs survive each streamed large row. Validate the
    # complete ordered population before opening any numeric reference.
    metric_inputs=[]
    for row in C.rows(output/'PREDICTIONS.jsonl.gz'):
        if set(row['arms'])!=set(ARMS):raise ValueError('Every streamed row must contain all seven arms')
        metric_inputs.append(dict(id=row['id'],partition=row['partition'],fixed_family=row['fixed_family'],
            actual_poses={a:row['arms'][a]['actual_pose']for a in ARMS},
            accepted={a:row['arms'][a]['accepted']for a in ARMS},tangent_control_audit=row['tangent_control_audit']))
    if [r['id']for r in metric_inputs]!=protocol['ids']:raise ValueError('Keep original ordered238 before reference access')
    for partition in rows:
        requested=[r['id']for r in metric_inputs if r['partition']==partition]
        if partition=='REMAINDER':
            plan=C.read(A.REF.OUTPUT/'PLAN.json');annotations={a['id']:a for a in plan['annotations']}
            references[partition]={ident:A.REF.reference_from_document(C.read(C.ROOT/annotations[ident]['annotation']))for ident in requested}
            bindings[partition]=E._binding(A.REF.OUTPUT/'PLAN.json')
        else:
            path=A.E.COHORT/('OOF_pose_v1/ISOLATED_CAL_SCORING.json'if partition=='CAL'else'test_reused_v1/ISOLATED_TEST_SCORING.json')
            labels=C.read(path)['rows'];references[partition]={ident:labels[ident]['target']['truth']for ident in requested};bindings[partition]=E._binding(path)
    for row in metric_inputs:
        partition=row['partition']
        metric=A.REF.metric_for_reference if partition=='REMAINDER'else A.pose_metric
        rows[partition].append(dict(id=row['id'],fixed_family=row['fixed_family'],
            metrics={a:metric(row['actual_poses'][a],references[partition][row['id']])for a in ARMS},
            accepted=row['accepted'],tangent_control_audit=row['tangent_control_audit']))
    for partition,count in(('CAL',46),('TEST',49),('REMAINDER',143)):
        values=rows[partition]
        if len(values)!=count:raise ValueError('No frame/native/fallback filtering')
        summary={}
        for arm in ARMS:
            ts=[r['metrics'][arm]['translation_cm']for r in values]
            rs=[r['metrics'][arm]['rotation_deg']for r in values if r['metrics'][arm]['rotation_deg']is not None]
            summary[arm]=dict(translation_cm=A.distribution(ts),rotation_deg=A.distribution(rs),
                translation_le5=sum(t<=5 for t in ts),successes_5cm_5deg=sum(bool(r['metrics'][arm]['success_5cm_5deg'])for r in values),
                rotation_unresolved=count-len(rs),fallback_count=sum(not r['accepted'][arm]for r in values),
                max_translation_id=values[int(np.argmax(ts))]['id'])
        reports[partition]=dict(rows=values,summary=summary,count=count,reference_binding=bindings[partition],all_rows_kept=True,
            new_parameter_or_source_selection_by_metrics=False,TEST49_and_REMAINDER143_reused_exploratory=True,
            remaining_reference_is_derived_not_independent_metrology=True,streaming_only_adapter_numeric_functions_unchanged=True)
        C.write(output/(partition+'_SCORE.json'),reports[partition]);print(partition,{a:(s['translation_le5'],s['translation_cm']['mean'],s['translation_cm']['max'])for a,s in summary.items()},flush=True)
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.write(output/'SCORE_SEAL.json',dict(prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),
        score_sha256={p:C.sha(output/(p+'_SCORE.json'))for p in reports},reference_bindings=bindings,
        all_numeric_references_decoded_only_after_new_control238_seal=True,
        same_unchanged_numeric_metric_reference_and_summary_functions=True,streaming_large_prediction_storage_only=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
