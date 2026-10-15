"""Uniform bounded O / O+S coarse-model priors, all frozen238.

Use v19's closed-canvas O hard domains and archived ellipses, unchanged fresh
NMS likelihood/BI/normal/T4/U/Q, every saved physical proposal, and exactly
the v19 unregularized six-diverse-plus-coarse start policy. Only optimization
and candidate energies add separately authenticated bounded regularizers.
No image/NN/GT query, fresh radius/noise fit, frame routing or default change.
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

from . import evaluate_o_localization_abstention_v4_19 as V
from . import bounded_geometric_priors_problem_v4_22 as P
from . import calibrate_bounded_o_localization_v4_21 as OCAL
from . import calibrate_coarse_visibility_prior_v4_20 as SCAL

C,E,BASE,REFINE=V.C,V.E,V.BASE,V.REFINE
OUTPUT=V.OUTPUT.parent/'bounded_geometric_priors_eval_v4_22'
ARMS=('SAME_FROZEN_E','O_SOFT_NATIVE','O_SOFT_S_NATIVE','O_SOFT_PLUS_MASK','O_SOFT_S_PLUS_MASK')
PROFILE_ARMS={'O':(ARMS[1],ARMS[3]),'OS':(ARMS[2],ARMS[4])}
PROFILE_INCLUDE_S={'O':False,'OS':True}
PROFILES=tuple((name,PROFILE_INCLUDE_S[name],*arms)for name,arms in PROFILE_ARMS.items())


def _checked_CAL_model(folder,kind):
    folder=Path(folder);seal=C.read(folder/'MODEL_SEAL.json');model=C.read(folder/'MODEL.json')
    if seal['fit_partition']!='CAL'or C.sha(folder/'MODEL.json')!=seal['model_sha256']:
        raise ValueError('Completed CAL-only prior model required')
    for name,key in (('CAL_LOGO_REPORT.json','LOGO_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(folder/name)!=seal[key]:raise ValueError('Sealed prior calibration binding mismatch')
    for key in ('source_sha256','input_sha256'):E._verify_sources(seal[key])
    if kind=='O':OCAL.prior_parameters(model)
    else:P.S.validate_model(model)
    return model


def _verified_models():
    return dict(NMS=V._verified_model(),O=_checked_CAL_model(OCAL.OUTPUT,'O'),S=_checked_CAL_model(SCAL.OUTPUT,'S'))


def _old_native_starts(problem,proposals,coarse_pose,*,configuration=None):
    return REFINE.choose_native_starts(problem.base,proposals,coarse_pose,configuration=configuration)


_REFINER=types.FunctionType(REFINE.refine_native_edge_pose.__code__,
    {**REFINE.refine_native_edge_pose.__globals__,'choose_native_starts':_old_native_starts},
    REFINE.refine_native_edge_pose.__name__)


def _prior_inputs(task,models):
    domains=[]
    for audit in task['O_localization_abstention']['records']:
        k=audit['native_corner']
        if task['initial_self_hidden'][k]or task['states'][k]!='O':
            raise ValueError('O archive rejected before localization fields')
        original=audit['original_domain']
        if original['binding']['calibration_model_sha256']!=models['O']['source_O_model_digest']:
            raise ValueError('O floor source radius model differs from this exact archived domain')
        replacement=next(d for d in task['spec']['domains']if d['native_corner']==k)
        if E._digest(replacement)!=E._digest(audit['replacement_domain']):
            raise ValueError('Exact existing closed-canvas O replacement required')
        domains.append(original)
    regions=P.O.prepare_o_localization_regions(domains,task['states'],task['initial_self_hidden'])
    original=task['families']['ORIGINAL_WD'];pose=original['coarse_pose']
    prepared=P.S.prepare_bounded_coarse_visibility_prior(original['native_xyz'],pose['R'],pose['t'],
        np.asarray(task['initial_self_hidden'],dtype=bool),models['S'],factor_weight=1.)
    return regions,OCAL.prior_parameters(models['O']),prepared


def _run(task,models):
    started=time.monotonic();before=E._digest(task)
    if 'O_localization_abstention'not in task:raise ValueError('Frozen v19 canvas/ellipse archive required')
    regions,o_parameters,original_S=_prior_inputs(task,models)
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration']);supported=bool(task['outline'].get('supported'))
    arms={ARMS[0]:deepcopy(task['baseline'])};all_audits={};start_banks={}
    for profile,use_S,native_arm,mask_arm in PROFILES:
        candidates=[];details={};searches={};bundles={};cached=[]
        for name,family in task['families'].items():
            base,bundle=V.PARENT._problem(task,family,models['NMS']);bundles[name]=bundle
            rebound=P.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])if use_S else None
            problem=P.BoundedGeometricPriorsProblem(base,regions,o_parameters,rebound)
            Q=np.asarray(family['physical_registration_Q'])
            proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
                       for i,p in enumerate(task['proposals'])]

            def add(R,t,token,origin):
                checked=problem.evaluate(R,t);mask=0. if not supported else None;mask_audit=None
                if checked['accepted']and supported:
                    pose=BASE.NATIVE.V._pose_vector(np.asarray(R),np.asarray(t),problem.scale)
                    value,_,mask_audit=BASE.OUTLINE.outline_value_and_gradient(pose,problem.xyz,problem.K,
                        problem.scale,task['outline'],cfg)
                    if mask_audit.get('valid_projection')and np.isfinite(value):mask=float(value)
                candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                    all_joint_constraints_satisfied=checked['accepted'],
                    native_costs=BASE._bridge_costs(checked.get('native_costs',[])),
                    regularizer_costs=checked.get('regularizer_costs',[]),global_mask_cost=mask,origin=origin))
                if checked['accepted']:
                    details[token]=dict(evaluation=checked,family=family,origin=origin,mask_audit=mask_audit)
                    if origin=='CACHED':cached.append(checked['objective'])

            for index,proposal in enumerate(proposals):add(proposal['R'],proposal['t'],name+'/CACHED_'+str(index),'CACHED')
            refined=_REFINER(problem,proposals,family['coarse_pose'],configuration=REFINE.DEFAULT_CONFIG)
            for index,result in enumerate(refined['solutions']):add(result['R'],result['t'],name+'/REFINED_'+str(index),'CONTINUOUS_ENDPOINT')
            active=bool(regions or(use_S and original_S['eligible_initial_S']))
            search={**refined['search'],'pose_prior':active,'coarse_regularizers_active':active,
                'original_search_pose_prior_false_overridden':True,
                'optimization_objective':'native image likelihood + separately declared bounded O/S model priors'}
            names=refined['start_selection']['selected_start_names']
            if name in start_banks and start_banks[name]!=names:raise ValueError('O/OS must share exact v19-native start names')
            start_banks[name]=names
            searches[name]=dict(search=search,start_selection=refined['start_selection'],
                candidates=refined['candidates'],feasible_endpoints=[{k:deepcopy(r.get(k))for k in
                    ('R','t','R_physical','objective','native_observation_objective','regularizer_objective','regularizer_costs','initialization','branch')}
                    for r in refined['solutions']])
        native,mask=P.select_regularized_candidate_pair(task['states'],candidates,
            mask_weight=cfg['outline_factor_weight']if supported else 0.)
        for arm,selection in ((native_arm,native),(mask_arm,mask)):
            token=selection['selected_name']
            if token is None:
                family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
                selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,
                    native_costs=[],regularizer_costs=[],reason='no_current_feasible_regularized_candidate')
                origin='OWN_INITIAL_COARSE_FALLBACK'
            else:
                detail=details[token];origin=detail['origin']
                selected={**detail['evaluation'],'family':detail['family'],'selected_endpoint_origin':origin,
                          'global_mask_audit':detail['mask_audit']}
                if arm==native_arm and cached and selected['objective']>min(cached)+1e-10*max(1.,abs(min(cached))):
                    raise ValueError('All cached contenders must remain available to regularized native selector')
            pose=BASE._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
            readout=BASE._readout(task,task['boundary_merged'],selected,pose)
            readout.pop('finite_candidate_hypothesis_not_continuous_or_unique_solution',None)
            readout.update(continuous_SE3_search_performed=True,joint_feasible_representative_not_unique_solution=True,
                selected_endpoint_origin=origin,original_O_hard_ellipse_claimed_solved=False,
                O_point_terms=0,S_image_or_2D_terms=0,S_model_prior_enabled=use_S,
                dependent_model_prior_does_not_establish_visibility_or_unique_pose=True)
            if selected['accepted']:
                for k,state in enumerate(task['states']):
                    if state=='O':readout['point_sources'][k]='CANVAS_FEASIBLE_MODEL_PROJECTION_BOUNDED_COARSE_O_PRIOR_NOT_UNIQUE'
            arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),
                solution=selected,keypoint_readout=readout,WD_family=selected['family']['name'],method=arm,
                candidate_selection=selection,ambiguity_audit=deepcopy(selection['ambiguity']),
                new_image_line_group_authentication=bundles,
                fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False))
            all_audits[arm]=dict(profile=profile,family_refinements=searches,all_cached_contenders_retained=True,
                same_v19_unregularized_native_six_start_policy=True,exact_start_names_across_profiles=start_banks,
                native_costs_authenticated_separately_from_regularizers=True,
                existing_v19_continuous_endpoints_not_added_to_fixed_input_bank=True,
                bounded_regularization_is_not_accuracy_or_uniqueness_guarantee=True)
    if E._digest(task)!=before:raise ValueError('Frozen v19 observation/physics/bank changed')
    return C.finite(dict(schema='BOUNDED_GEOMETRIC_MODEL_PRIORS_PREDICTION_V4_22',id=task['id'],partition=task['partition'],
        session=task['session'],image=task['image'],image_sha256=task['image_sha256'],states=task['states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family')for a in ARMS},bounded_prior_refinement_audit=all_audits,
        O_localization_abstention=deepcopy(task['O_localization_abstention']),input_sha256=before,
        tangent_control_audit={a:dict(unchanged_variance_inflation=4.,unchanged_native_likelihood=True)for a in ARMS[1:]},
        elapsed_seconds=time.monotonic()-started,
        contract=dict(all238_retained=True,severe=0,original_e_default_unchanged=True,new_RGB_or_NN_queries=0,
            S_raw_2D_image_regions=0,S_model_visibility_prior_explicit=True,S_readout_pose_projection_only=True,
            U_range_only=True,O_not_precise_point=True,all_v19_observations_and_hard_constraints_unchanged=True,
            prior_terms_not_independent_image_evidence=True,no_GT_ID_or_partition_routing=True,no_default_adoption=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every preparation')
    models=_verified_models();parent=C.read(V.OUTPUT/'PROTOCOL.json');seal=C.read(V.OUTPUT/'INPUT_SEAL.json')
    if (seal['rows']!=238 or C.sha(V.OUTPUT/'INPUTS.jsonl.gz')!=seal['input_sha256']
            or C.sha(V.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Complete frozen238 parent inputs required')
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    for module in(sys.modules[__name__],P,P.O,P.S,OCAL,SCAL):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    for folder in(V.OUTPUT,OCAL.OUTPUT,SCAL.OUTPUT):
        names=('PROTOCOL.json','INPUT_SEAL.json','PREDICTION_SEAL.json','SCORE_SEAL.json')if folder==V.OUTPUT else(
            'PROTOCOL.json','PREPARATION_SEAL.json','MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json')
        for name in names:
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    count=Counter();points=0
    for task in C.rows(V.OUTPUT/'INPUTS.jsonl.gz'):
        _prior_inputs(task,models);count[task['partition']]+=1;points+=len(task['O_localization_abstention']['records'])
    if count!=Counter(CAL=46,TEST=49,REMAINDER=143)or points!=133:raise ValueError('Exact all238/O133 inventory required')
    for mapping in(sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    C.write(output/'PROTOCOL.json',dict(schema='BOUNDED_GEOMETRIC_PRIORS_ALL238_PROTOCOL_V4_22',ids=parent['ids'],arms=list(ARMS),
        population=parent['population'],source_sha256=sources,input_sha256=inputs,protected_user_sha256=parent['protected_user_sha256'],
        new_NMS_model_binding=parent['new_NMS_model_binding'],observation_generation=parent['observation_generation'],
        O_model_binding=E._binding(OCAL.OUTPUT/'MODEL.json'),S_model_binding=E._binding(SCAL.OUTPUT/'MODEL.json'),
        objective='unchanged authenticated native image likelihood + bounded O frame prior + optional bounded S frame prior',
        native_cost_basis_does_not_contain_priors=True,regularizer_selector_uses_all_authenticated_candidates=True,
        original_S_eligibility_prepared_once_then_WD_geometry_rebound=True,
        fixed_factor_weights=dict(O=1.,S=1.),refinement_configuration=REFINE.DEFAULT_CONFIG,
        same_v19_native_six_start_names_not_reranked_by_priors=True,
        all51539physical_proposals_and_all_new_feasible_endpoints_retained=True,
        parent_v19_continuous_endpoints_not_added_to_fixed_proposal_bank=True,
        mask_selects_same_profile_bank_only=True,mask_not_continuously_optimized=True,
        O_source_radius_model_digest_checked_per_original_domain=True,
        all238_all5arms_prediction_seal_before_reference_decode=True,
        exploratory_reused_populations=True,remaining_reference_is_derived_not_independent_metrology=True,
        source_ready_requires_root_memory_coordination_before_pool=True,new_RGB_or_NN_queries=0,default_adoption=False))
    shutil.copyfile(V.OUTPUT/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=parent['ids'],input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_predictions_or_metrics=True))
    print(dict(prepared=238,O_points=points,arms=list(ARMS)),flush=True)


def predict(output=OUTPUT,workers=8):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if seal['rows']!=238 or seal['input_sha256']!=C.sha(output/'INPUTS.jsonl.gz')or seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json'):
        raise ValueError('Exact prepared238 input/source seal required')
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
                if row['id']!=expected or expected in done or set(row['arms'])!=set(ARMS):raise ValueError('Complete unique all5-arm row required')
                C.write(output/'frames'/(expected.replace(':','__')+'.json'),row);done.add(expected)
                C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=expected,elapsed_seconds=time.monotonic()-started))
                print(str(len(done))+'/238 '+expected,flush=True);submit_one()
    except BaseException as error:
        C.write(output/'FAILED_BEFORE_SEAL.json',dict(completed=len(done),total=238,error_type=type(error).__name__,reason=str(error),new_reference_reads=0))
        for future in pending:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    if done!=set(protocol['ids']):raise ValueError('Keep all238, including fallback/zero-BI frames')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',(C.read(output/'frames'/(ident.replace(':','__')+'.json'))for ident in protocol['ids']))
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),ordered_ids=protocol['ids'],all5_arms_complete=True,
        arms=list(ARMS),arm_count=5,all_declared_arms_complete=True,
        all3_arms_complete=True,legacy_all3_flag_is_scorer_compatibility_only=True,
        before_new_control_metric_read=True,elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    output=Path(output);seal=C.read(output/'PREDICTION_SEAL.json')
    if (seal.get('all5_arms_complete')is not True or seal.get('all_declared_arms_complete')is not True
            or seal.get('arms')!=list(ARMS)or seal.get('arm_count')!=5):
        raise ValueError('All five full-population arms required before reference access')
    scorer=types.FunctionType(BASE.ARCHIVE.score.__code__,{**BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},BASE.ARCHIVE.score.__name__)
    return scorer(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
