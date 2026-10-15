"""Continuous SE3 control with the sealed fresh grazing/NMS likelihood.

Keep all cached contenders, refine the same six native-ranked diverse
starts plus own coarse in both WD families, and retain every fresh-feasible
endpoint. The unchanged native and native+old-mask selectors use that common
bank. No observation/noise/guard tuning, new NN/RGB query or GT routing.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time
import types

import numpy as np

from . import common as C
from . import evaluate_grazing_image_chain_mixtures_v4_18b as PARENT
from . import refine_native_edge_pose_v4_17 as REFINE

OLD=PARENT.OLD
E,BASE,MIX,CAL=OLD.E,OLD.BASE,OLD.MIX,PARENT.CAL
OUTPUT=PARENT.OUTPUT.parent/'refined_grazing_image_chain_mixtures_v4_18c'
ARMS=('SAME_FROZEN_E','REFINED_GRAZING_NMS_NATIVE','REFINED_GRAZING_NMS_PLUS_MASK')
_verified_model=PARENT._verified_model


def _problem(task,family,model):
    spec=deepcopy(task['spec']);spec['config']['prior_weight']=0.
    base=BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
        spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],config=spec['config'],
        boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],
        edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    if not np.allclose(base.Q,family['physical_registration_Q'],atol=1e-12,rtol=0):
        raise ValueError('Exact physical registration Q required')
    bundle=MIX.authenticate_image_chain_groups(base,task['new_chain_observation'],
        initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        **CAL.mixture_noise_parameters(model))
    return MIX.ImageChainMixtureProblem(base,bundle),bundle


def _run(task,model):
    started=time.monotonic();before=E._digest(task)
    if task['fresh_grazing_observation_generation']!=PARENT.OBS.OBSERVATION_GENERATION:
        raise ValueError('Exact sealed fresh grazing observation generation required')
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration'])
    mask_supported=bool(task['outline'].get('supported'))
    candidates=[];details={};bundles={};searches={};cached_energies=[]
    for name,family in task['families'].items():
        problem,bundle=_problem(task,family,model);bundles[name]=bundle
        Q=np.asarray(family['physical_registration_Q'])
        proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
            for i,p in enumerate(task['proposals'])]

        def add(R,t,token,origin):
            R,t=np.asarray(R),np.asarray(t);checked=problem.evaluate(R,t)
            mask=0. if not mask_supported else None;mask_audit=None
            if checked['accepted']and mask_supported:
                p=BASE.NATIVE.V._pose_vector(R,t,problem.scale)
                value,_,mask_audit=BASE.OUTLINE.outline_value_and_gradient(
                    p,problem.xyz,problem.K,problem.scale,task['outline'],cfg)
                if mask_audit.get('valid_projection')and np.isfinite(value):mask=float(value)
            costs=BASE._bridge_costs(checked.get('native_costs',[]))
            candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                all_joint_constraints_satisfied=checked['accepted'],native_costs=costs,
                global_mask_cost=mask,origin=origin))
            if checked['accepted']:
                details[token]=dict(evaluation=checked,family=family,origin=origin,mask_audit=mask_audit)
                if origin=='CACHED':cached_energies.append(float(checked['objective']))

        for index,p in enumerate(proposals):
            add(p['R'],p['t'],name+'/CACHED_'+str(index),'CACHED')
        refined=REFINE.refine_native_edge_pose(problem,proposals,family['coarse_pose'],
            configuration=REFINE.DEFAULT_CONFIG)
        for index,p in enumerate(refined['solutions']):
            add(p['R'],p['t'],name+'/REFINED_'+str(index),'CONTINUOUS_ENDPOINT')
        searches[name]=dict(search=refined['search'],start_selection=refined['start_selection'],
            candidates=refined['candidates'],feasible_endpoints=[{k:deepcopy(p.get(k))
                for k in ('R','t','R_physical','objective','native_costs','initialization','branch')}
                for p in refined['solutions']])
    native=BASE.AUTH.select_hierarchical_native_candidates(task['states'],candidates)
    coupled=OLD._coupled_selection(candidates,cfg['outline_factor_weight']if mask_supported else 0.,native)
    arms={ARMS[0]:deepcopy(task['baseline'])};audits={}
    for arm,selection in ((ARMS[1],native),(ARMS[2],coupled)):
        token=selection['selected_name']
        if token is None:
            family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
            selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,
                native_costs=[],reason='no_fresh_feasible_cached_or_continuous_endpoint')
            origin='OWN_INITIAL_COARSE_FALLBACK'
        else:
            detail=details[token];origin=detail['origin']
            selected={**detail['evaluation'],'family':detail['family'],
                'selected_endpoint_origin':origin,'global_mask_audit':detail['mask_audit']}
        if arm==ARMS[1]and token is not None and cached_energies:
            best=min(cached_energies);now=float(selected['objective'])
            if now>best+128*np.finfo(float).eps*max(1.,abs(best),abs(now)):
                raise RuntimeError('Retaining all cached contenders must not worsen native minimum')
        pose=BASE._actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
        readout=BASE._readout(task,task['boundary_merged'],selected,pose)
        readout.pop('finite_candidate_hypothesis_not_continuous_or_unique_solution',None)
        readout.update(continuous_SE3_search_performed=True,joint_feasible_representative_not_unique_solution=True,
            selected_endpoint_origin=origin,selected_pose_is_cached_contender=origin=='CACHED')
        if selected['accepted']:
            for k,state in enumerate(task['states']):
                if state in ('O','T'):
                    readout['point_sources'][k]='JOINT_FEASIBLE_MODEL_PROJECTION_NOT_UNIQUE'
        arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=deepcopy(task['states']),
            solution=selected,keypoint_readout=readout,WD_family=selected['family']['name'],method=arm,
            candidate_selection=BASE._compact_selection(selection)if arm==ARMS[1]else selection,
            ambiguity_audit=deepcopy(selection['ambiguity']),new_image_line_group_authentication=bundles,
            fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False))
        audits[arm]=dict(evaluated_candidates=len(candidates),accepted_candidates=sum(c['accepted']for c in candidates),
            family_refinements=searches,all_cached_contenders_retained=True,
            cached_minimum_native_energy=None if not cached_energies else min(cached_energies),
            selected_native_energy=selected['objective'],continuous_objective='unchanged fresh NMS native likelihood',
            optimization_has_mask_or_pose_prior=False,new_P3P_or_global_starts=0,
            native_energy_nonworsening_not_GT_accuracy_guarantee=True)
    if E._digest(task)!=before:raise ValueError('Frozen input mutated')
    return C.finite(dict(schema='REFINED_GRAZING_NMS_PREDICTION_V18C',id=task['id'],partition=task['partition'],
        session=task['session'],image=task['image'],image_sha256=task['image_sha256'],states=task['states'],
        original_states=task['grazing_adaptation']['original_states'],grazing_adaptation=task['grazing_adaptation'],
        arms=arms,fixed_family={a:arms[a].get('WD_family')for a in ARMS},native_refinement_audit=audits,
        input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(uniform238=True,severe=0,S_model_projection_only=True,U_regions_only=True,
            new_RGB_or_NN_calls=0,S_or_U_local_queries=0,no_GT_ID_or_partition_cost_routing=True,
            all_cached_candidates_and_fresh_feasible_endpoints_retained=True,
            observation_noise_guards_and_start_budget_unchanged=True,no_default_adoption=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior preparations')
    parent,seal=PARENT._sealed(PARENT.OUTPUT,'INPUT_SEAL.json','INPUTS.jsonl.gz','input_sha256',238)
    prediction=C.read(PARENT.OUTPUT/'PREDICTION_SEAL.json')
    if prediction['rows']!=238 or C.sha(PARENT.OUTPUT/'PREDICTIONS.jsonl.gz')!=prediction['predictions_sha256']:
        raise ValueError('Complete parent cached-control prediction seal required')
    _verified_model()
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    for module in (PARENT,REFINE,__import__(__name__,fromlist=['_'])):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    for name in ('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz','PREDICTION_SEAL.json'):
        path=PARENT.OUTPUT/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol=dict(schema='REFINED_GRAZING_NMS_ALL238_PROTOCOL_V18C',ids=parent['ids'],arms=list(ARMS),
        population=parent['population'],source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],new_NMS_model_binding=parent['new_NMS_model_binding'],
        observation_generation=parent['observation_generation'],refinement_configuration=REFINE.DEFAULT_CONFIG,
        parent_prediction_seal=E._binding(PARENT.OUTPUT/'PREDICTION_SEAL.json'),
        candidate_policy='all cached physical proposals plus all fresh feasible continuous native endpoints',
        selection='same native minsum/numeric mask ties and native+fixed old mask',
        exact_parent_input_bytes=True,unchanged_likelihood_noise_guards_and_iterations=True,
        six_diverse_native_ranked_starts_and_own_coarse_per_family_and_T_union=True,
        new_RGB_or_NN_queries=0,new_P3P_or_global_starts=0,no_default_adoption=True,
        all238_prediction_seal_before_new_metrics=True,exploratory_reused_data=True,
        additional143_reference_not_independent_metrology=True)
    C.write(output/'PROTOCOL.json',protocol);shutil.copyfile(PARENT.OUTPUT/'INPUTS.jsonl.gz',output/'INPUTS.jsonl.gz')
    if C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']:raise ValueError('Parent input bytes changed')
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=parent['ids'],input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_continuous_predictions_and_metrics=True))
    print(dict(prepared=238,input_bytes_identical=True),flush=True)


def predict(output=OUTPUT,workers=8):
    runner=types.FunctionType(OLD.predict.__code__,{**OLD.predict.__globals__,
        '_run':_run,'_verified_model':_verified_model},OLD.predict.__name__)
    return runner(Path(output),workers)


def score(output=OUTPUT):
    scorer=types.FunctionType(BASE.ARCHIVE.score.__code__,{**BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},BASE.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
