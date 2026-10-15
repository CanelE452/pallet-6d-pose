"""Uniform source-preserving complete-BI, CAL tangent and continuous-mask controls.

All seven declared arms use the same frozen238 observations and hard guards.
No new image/network/hidden-point query or reference-conditioned routing occurs.
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

from . import evaluate_complete_bi_bounded_priors_v4_23 as V
from . import source_preserving_complete_bi_evidence_v4_24 as SOURCE
from . import calibrate_source_preserving_bi_edge_mixture_v4_24 as SOURCE_CAL
from . import calibrate_archived_complete_tangent_v4_24 as TCAL
from . import continuous_outline_problem_v4_24 as MASK

OLD,C,E,BASE,REFINE=V.OLD,V.C,V.E,V.BASE,V.REFINE
P=OLD.P
OUTPUT=V.OUTPUT.parent/'source_preserving_tangent_mask_eval_v4_24'
PLAN=V.OUTPUT.parent/'source_preserving_tangent_mask_plan_v4_24/PREDECLARATION.json'
ARMS=('SAME_FROZEN_E','ALL_COMPLETE_TCAL_NATIVE','ALL_COMPLETE_TCAL_PLUS_MASK',
    'SOURCE_PRESERVING_TCAL_NATIVE','SOURCE_PRESERVING_TCAL_PLUS_MASK',
    'SOURCE_PRESERVING_TCAL_MASK_FIT','SOURCE_PRESERVING_TCAL_MASK_FIT_E_FAMILY')
PROFILES=('ALL','SOURCE','MASK_FIT')
PROFILE_ARMS={'ALL':ARMS[1:3],'SOURCE':ARMS[3:5],'MASK_FIT':ARMS[5:7]}


def _checked_model(module):
    folder=module.OUTPUT;seal=C.read(folder/'MODEL_SEAL.json');model=C.read(folder/'MODEL.json')
    if seal['fit_partition']!='CAL'or seal['model_sha256']!=C.sha(folder/'MODEL.json'):
        raise ValueError('Separate sealed CAL model required')
    for key in ('source_sha256','input_sha256'):E._verify_sources(seal[key])
    for filename,key in (('CAL_LOGO_REPORT.json','LOGO_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(folder/filename)!=seal[key]:raise ValueError('CAL model evidence changed')
    errors='CAL_TANGENT_ERRORS.jsonl.gz'if module is TCAL else'CAL_MIXTURE_ERRORS.jsonl.gz'
    if C.sha(folder/errors)!=seal['CAL_errors_sha256']:raise ValueError('CAL error archive changed')
    if module is TCAL:module.tangent_variance_parameters(model)
    else:module.mixture_noise_parameters(model)
    return model


def _verified_models():
    models=V._verified_models()
    models.update(SOURCE=_checked_model(SOURCE_CAL),TANGENT=_checked_model(TCAL))
    return models


def _calibrate_plan_tangents(problem,model):
    for record in problem.plan['corner_plans']:
        if 'tangent_kind'not in record:continue
        archive=record['original_measurement_archive']
        parameters=TCAL.calibrated_archived_tangent_variance(model,record['tangent_unit_xy'],
            archive['original_effective_fit_covariance_px2'])
        if not np.isclose(record['tangent_variance_px2'],parameters['exact_old_x4_variance_px2'],atol=1e-10,rtol=1e-10):
            raise ValueError('Exact original archived-complete tangent required')
        record['legacy_uncalibrated_variance_px2']=record['tangent_variance_px2']
        record['legacy_uncalibrated_variance_inflation']=record.pop('tangent_variance_inflation')
        record['tangent_variance_px2']=parameters['tangent_variance_px2']
        record['tangent_variance_CAL']=parameters
        record['tangent_total_error_validation_still_required']=False
        record['tangent_CAL_reused_descriptive_not_future_coverage_guarantee']=True
    problem.plan.update(archived_complete_tangent_variance_model=TCAL.tangent_variance_parameters(model),
        original_q_covariance_reliability_guide_unchanged=True,old_partial_T4_unchanged=True)


class _ExactReplacementCount:
    def objective(self,p):
        value,gradient,parts=super().objective(p)
        parts.update(replaced_old_normal_factor_count=len(self.plan['replaces_normal_factor_refs']),
            count_source='exact v24 source/ownership replacement plan; unchanged hard guards')
        return value,gradient,parts


class _AllProblem(_ExactReplacementCount,V.COMP.CompleteBIImageChainProblem):pass
class _SourceProblem(_ExactReplacementCount,SOURCE.CompleteBIImageChainProblem):pass


def _problem(task,family,models,profile):
    source=profile!='ALL';module=SOURCE if source else V.COMP
    spec=task['spec']
    base=OLD.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
        spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],
        config={**spec['config'],'prior_weight':0.},boundary_observations=task['boundary_merged'],
        u_regions=task['u_regions'],edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    if not np.allclose(base.Q,family['physical_registration_Q'],atol=1e-12,rtol=0):raise ValueError('Exact physical Q required')
    noise=(SOURCE_CAL.mixture_noise_parameters(models['SOURCE'])if source else
        V.CAL.mixture_noise_parameters(models['NMS']['COMPLETE']))
    bundle=module.authenticate_complete_bi_image_chain_groups(base,task['new_chain_observation'],
        junction_proposals=task['complete_BI_evidence']['junction_proposals'],visible_measurements=spec['visible'],
        initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],**noise)
    problem=(_SourceProblem if source else _AllProblem)(base,bundle,overlap_policy=V.COMP.PROTOTYPE_OVERLAP_POLICY)
    _calibrate_plan_tangents(problem,models['TANGENT'])
    # Fixed starts are ranked by the immutable original v19 likelihood only.
    problem.reference_start_problem=OLD.V.PARENT._problem(task,family,models['NMS']['OLD'])[0]
    return problem,{**bundle,'runtime_complete_BI_replacement_plan':problem.plan}


def _select_locked(states,candidates,*,family,mask_weight):
    if family not in ('ORIGINAL_WD','SWAPPED_WD'):raise ValueError('Stored predicted e WD family required')
    eligible=[record for record in candidates if record['family']==family]
    result=P.select_regularized_native_candidates(states,eligible,mask_weight=mask_weight,mask_ties_only=False)
    result.update(predicted_e_family_constraint=family,filter_applied_to_all_candidates_before_authentication=True,
        global_native_Pareto_not_used_as_family_prefilter=True,e_numeric_pose_used=False,
        eligible_family_candidate_count=len(eligible),all_other_family_candidates_retained_in_unlocked_audit=True)
    return result


def _run(task,models):
    started=time.monotonic();before=E._digest(task)
    regions,oparams,original_S=OLD._prior_inputs(task,models)
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration'])
    supported=bool(task['outline'].get('supported'));weight=cfg['outline_factor_weight']if supported else 0.
    arms={ARMS[0]:deepcopy(task['baseline'])};audits={};start_banks={};source_endpoints=[]
    for profile in PROFILES:
        candidates=[];details={};searches={};bundles={};new_endpoints=[]
        for name,family in task['families'].items():
            base,bundle=_problem(task,family,models,profile);bundles[name]=bundle
            rebound=P.S.rebind_bounded_visibility_trial_geometry(original_S,family['native_xyz'])
            native=P.BoundedGeometricPriorsProblem(base,regions,oparams,rebound)
            problem=MASK.ContinuousOutlineProblem(native,task['outline'],cfg)if profile=='MASK_FIT'else native
            Q=np.asarray(family['physical_registration_Q'])
            proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
                for i,p in enumerate(task['proposals'])]

            def add(R,t,token,origin):
                checked=problem.evaluate(R,t);mask=0. if not supported else None;ma=None
                if checked['accepted']and supported:
                    p=BASE.NATIVE.V._pose_vector(np.asarray(R),np.asarray(t),problem.scale)
                    value,_,ma=BASE.OUTLINE.outline_value_and_gradient(p,problem.xyz,problem.K,problem.scale,task['outline'],cfg)
                    if ma.get('valid_projection')and np.isfinite(value)and value>=0:mask=float(value)
                if profile=='MASK_FIT'and checked['accepted']:
                    if mask is None or not np.isclose(checked['objective'],checked['native_observation_objective']+
                            checked['regularizer_objective']+weight*mask,atol=1e-9,rtol=1e-10):
                        raise ValueError('Continuous and selector objectives must contain mask once')
                candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                    all_joint_constraints_satisfied=checked['accepted'],
                    native_costs=BASE._bridge_costs(checked.get('native_costs',[])),
                    regularizer_costs=checked.get('regularizer_costs',[]),global_mask_cost=mask,origin=origin))
                if checked['accepted']:details[token]=dict(evaluation=checked,family=family,origin=origin,mask_audit=ma)

            for index,proposal in enumerate(proposals):add(proposal['R'],proposal['t'],name+'/CACHED_'+str(index),'CACHED')
            if profile=='MASK_FIT':
                for index,endpoint in enumerate(source_endpoints):
                    if endpoint['family']==name:add(endpoint['R'],endpoint['t'],name+'/SOURCE_ENDPOINT_'+str(index),'SOURCE_NATIVE_ENDPOINT')
            refined=V._REFINER(problem,proposals,family['coarse_pose'],configuration=REFINE.DEFAULT_CONFIG)
            for index,result in enumerate(refined['solutions']):
                add(result['R'],result['t'],name+'/REFINED_'+str(index),'CONTINUOUS_MASK_ENDPOINT'if profile=='MASK_FIT'else'CONTINUOUS_NATIVE_ENDPOINT')
                new_endpoints.append(dict(family=name,R=result['R'],t=result['t']))
            names=refined['start_selection']['selected_start_names']
            if name in start_banks and names!=start_banks[name]:raise ValueError('Exact v19 start names must agree across all profiles')
            start_banks[name]=names
            searches[name]=dict(search={**refined['search'],'pose_prior':bool(regions or original_S['eligible_initial_S']),
                'coarse_regularizers_active':True,'continuous_mask_gradient_active':profile=='MASK_FIT'and supported and weight>0,
                'mask_objective':profile=='MASK_FIT'and supported and weight>0,
                'optimization_objective':'native + bounded O/S'+(' + lambda*wholemask'if profile=='MASK_FIT'else'')},
                start_selection=refined['start_selection'],candidates=refined['candidates'],
                feasible_endpoints=[{k:deepcopy(r.get(k))for k in('R','t','R_physical','objective','native_observation_objective',
                    'regularizer_objective','global_mask_cost','weighted_mask_objective','initialization','branch')}for r in refined['solutions']])
        if profile=='SOURCE':source_endpoints=new_endpoints
        if profile=='MASK_FIT':
            unlocked=P.select_regularized_native_candidates(task['states'],candidates,mask_weight=weight,mask_ties_only=False)
            locked=_select_locked(task['states'],candidates,family=task['baseline']['WD_family'],mask_weight=weight)
            selections=(unlocked,locked)
            # Re-evaluated cached + source endpoints are a subset of this bank.
            oldrows=[r for r in unlocked['candidates']if details[r['name']]['origin']in('CACHED','SOURCE_NATIVE_ENDPOINT')]
            if unlocked['selected_name']is not None and oldrows:
                chosen=next(r for r in unlocked['candidates']if r['name']==unlocked['selected_name'])
                if chosen['total_energy']>min(r['total_energy']for r in oldrows)+1e-9:raise ValueError('Retained B bank cannot lose in identical combined cost')
        else:selections=P.select_regularized_candidate_pair(task['states'],candidates,mask_weight=weight)
        for arm,selection in zip(PROFILE_ARMS[profile],selections):
            token=selection['selected_name'];is_locked=arm==ARMS[6]
            if token is None:
                family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose'];origin='OWN_INITIAL_COARSE_FALLBACK'
                selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,native_costs=[],
                    regularizer_costs=[],reason='no_current_eligible_feasible_candidate')
            else:
                detail=details[token];origin=detail['origin']
                selected={**detail['evaluation'],'family':detail['family'],'selected_endpoint_origin':origin,'global_mask_audit':detail['mask_audit']}
            pose=BASE._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
            readout=BASE._readout(task,task['boundary_merged'],selected,pose)
            readout.pop('finite_candidate_hypothesis_not_continuous_or_unique_solution',None)
            readout.update(continuous_SE3_search_performed=True,selected_endpoint_origin=origin,
                S_image_or_2D_terms=0,S_model_prior_enabled=True,U_range_only=True,O_point_terms=0,
                dependent_priors_and_mask_do_not_establish_unique_pose=True)
            if selected['accepted']:
                for k,state in enumerate(task['states']):
                    if state=='O':readout['point_sources'][k]='CANVAS_FEASIBLE_MODEL_PROJECTION_BOUNDED_COARSE_O_PRIOR_NOT_UNIQUE'
            lock_audit=dict(enabled=is_locked,requested_predicted_family=task['baseline']['WD_family']if is_locked else None,
                locked_family_constraint_satisfied=bool(is_locked and selected['accepted']and selected['family']['name']==task['baseline']['WD_family']),
                e_numeric_pose_used=False,e_family_is_prediction_not_physical_identity_certificate=True)
            if is_locked:lock_audit.update(e_source_ambiguity_audit=deepcopy(task['baseline'].get('ambiguity_audit')),
                e_source_fallback_audit=deepcopy(task['baseline'].get('fallback_audit')))
            arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),solution=selected,
                keypoint_readout=readout,WD_family=selected['family']['name'],method=arm,candidate_selection=selection,
                ambiguity_audit=deepcopy(selection['ambiguity']),new_image_line_group_authentication=bundles,
                early_family_audit=lock_audit,fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],
                    old_best_pose_fallback=False,lock_requested_but_unsatisfied=bool(is_locked and not selected['accepted'])))
            audits[arm]=dict(profile=profile,family_refinements=searches,all_cached_contenders_retained=True,
                exact_start_names_across_profiles=deepcopy(start_banks),same_v19_unregularized_native_six_start_policy=True,
                source_native_endpoints_retained=profile=='MASK_FIT',mask_in_optimizer_once=profile=='MASK_FIT',
                native_costs_authenticated_separately_from_regularizers=True,mask_cost_separate_from_native_and_priors=True,
                early_family_constraint=is_locked,cost_nonworsening_is_not_GT_accuracy_guarantee=True)
    if E._digest(task)!=before:raise ValueError('Frozen inputs mutated')
    return C.finite(dict(schema='SOURCE_PRESERVING_TANGENT_MASK_PREDICTION_V24',id=task['id'],partition=task['partition'],
        session=task['session'],image=task['image'],image_sha256=task['image_sha256'],states=task['states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family')for a in ARMS},
        tangent_control_audit={a:dict(old_partial_variance_inflation=4.,
            archived_complete_variance_model=TCAL.tangent_variance_parameters(models['TANGENT']),
            original_tangent_covariance_q_guide_reliability_unchanged=True,
            source_preserving=a in ARMS[3:])for a in ARMS[1:]},
        bounded_prior_refinement_audit=audits,input_sha256=before,elapsed_seconds=time.monotonic()-started,
        O_localization_abstention=deepcopy(task['O_localization_abstention']),
        contract=dict(all238_retained=True,severe=0,original_e_default_unchanged=True,new_RGB_or_NN_queries=0,
            S_raw_2D_image_regions=0,S_model_prior_explicit=True,S_readout_pose_projection_only=True,U_range_only=True,
            original_hard_guards_unchanged=True,old_partial_T4_unchanged=True,no_GT_ID_error_or_partition_routing=True,
            e_predicted_WD_label_only_in_uniform_seventh_arm=True,no_selective_oldpose_fallback=True,no_default_adoption=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous preparation')
    models=_verified_models();parent=C.read(V.OUTPUT/'PROTOCOL.json');seal=C.read(V.OUTPUT/'INPUT_SEAL.json')
    declaration=C.read(PLAN)
    if declaration['arms']!=list(ARMS)or seal['rows']!=238 or C.sha(V.OUTPUT/'INPUTS.jsonl.gz')!=seal['input_sha256']:
        raise ValueError('Exact seven-arm declaration and parent238 inputs required')
    if C.sha(V.OUTPUT/'PROTOCOL.json')!=seal['protocol_sha256']:raise ValueError('Parent protocol changed')
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    for module in(sys.modules[__name__],SOURCE,SOURCE_CAL,TCAL,MASK):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    for module in(SOURCE_CAL,TCAL):
        calseal=C.read(module.OUTPUT/'MODEL_SEAL.json')
        for relative,digest in calseal['source_sha256'].items():
            if relative in sources and sources[relative]!=digest:raise ValueError('CAL source versions differ')
            sources[relative]=digest
        for name in('PROTOCOL.json','PREPARATION_SEAL.json','MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json'):
            p=module.OUTPUT/name;inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
        p=module.OUTPUT/('CAL_TANGENT_ERRORS.jsonl.gz'if module is TCAL else'CAL_MIXTURE_ERRORS.jsonl.gz')
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    for path in(PLAN,V.OUTPUT/'PROTOCOL.json',V.OUTPUT/'INPUT_SEAL.json',V.OUTPUT/'INPUTS.jsonl.gz'):
        inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    modes=Counter();counts=Counter();ids=[]
    for task in C.rows(V.OUTPUT/'INPUTS.jsonl.gz'):
        ids.append(task['id']);counts[task['partition']]+=1;OLD._prior_inputs(task,models)
        for profile in('ALL','SOURCE'):
            for family in task['families'].values():
                problem,_=_problem(task,family,models,profile)
                modes.update({profile+'_groups':len(problem.groups),profile+'_revoked':len(problem.revoked),
                    profile+'_tangents':sum('tangent_kind'in r for r in problem.plan['corner_plans'])})
    if counts!=Counter(CAL=46,TEST=49,REMAINDER=143)or ids!=parent['ids']:raise ValueError('All238 and original order required')
    for mapping in(sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol={**parent,'schema':'SOURCE_PRESERVING_TANGENT_MASK_PROTOCOL_V24','arms':list(ARMS),'arm_count':len(ARMS),
        'source_sha256':sources,'input_sha256':inputs,'declaration':E._binding(PLAN),
        'source_edge_model_binding':E._binding(SOURCE_CAL.OUTPUT/'MODEL.json'),'tangent_model_binding':E._binding(TCAL.OUTPUT/'MODEL.json'),
        'input_bytes_identical_to_v23':True,'mask_not_continuously_optimized':False,
        'objective':'authenticated native + bounded O/S; last two arms continuously optimize same lambda*mask once',
        'mask_selects_same_profile_bank_only':False,'C_bank_adds_source_native_endpoints_uniformly':True,
        'continuous_mask_only_in_last_two_arms':True,'all238_all_declared_arms_prediction_seal_before_reference_decode':True,
        'legacy_parent_all5_flag_does_not_describe_this_experiment':True,'no_default_adoption':True}
    protocol.pop('all238_all5arms_prediction_seal_before_reference_decode',None)
    C.write(output/'PROTOCOL.json',protocol);shutil.copyfile(V.OUTPUT/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Snapshot mismatch')
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=ids,protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),both_WD_both_source_profiles_preflight=True,
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
    output=Path(output);seal=C.read(output/'PREDICTION_SEAL.json')
    if not(seal.get('all7_arms_complete')is True and seal.get('all_declared_arms_complete')is True
            and seal.get('arms')==list(ARMS)and seal.get('arm_count')==len(ARMS)):
        raise ValueError('All seven full-population arms required before reference access')
    scorer=types.FunctionType(BASE.ARCHIVE.score.__code__,{**BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},'score_seven_uniform_arms')
    return scorer(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
