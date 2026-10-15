"""Uniform238 actual-NMS mixture controls, using all frozen saved proposals.

One conservative merged-partial profile, native-only and native+old-mask
selectors, plus exact archived e baseline. All frames/fallbacks stay. New
observations/model are sealed; predictions complete before reference scores.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
import types
import numpy as np

from . import common as C
from . import statewise_evaluation_v4 as E
from . import evaluate_cached_native_edges_v4_17 as BASE
from . import calibrate_native_edge_mixture_v4_18 as CAL
from . import image_chain_mixture_problem_v4_18 as MIX
from . import observe_native_edge_chains_v4_17b as OBS

OUTPUT=BASE.OUTPUT.parent/'image_chain_mixture_candidates_v4_18'
ARMS=('SAME_FROZEN_E','NMS_NATIVE_D_MERGED','NMS_PLUS_MASK_D_MERGED')


def _verified_model():
    seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json');path=CAL.OUTPUT/'MODEL.json'
    if C.sha(path)!=seal['model_sha256']:raise ValueError('Sealed new NMS mixture model required')
    for name,key in(('CAL_MIXTURE_ERRORS.jsonl.gz','CAL_errors_sha256'),
            ('CAL_LOGO_REPORT.json','LOGO_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(CAL.OUTPUT/name)!=seal[key]:raise ValueError('Sealed new NMS calibration evidence changed')
    for key in('source_sha256','input_sha256'):E._verify_sources(seal[key])
    model=C.read(path);CAL.mixture_noise_parameters(model)
    return model


def _actual_pose(family,R,t,accepted,method):
    return BASE._actual_pose(family,R,t,accepted,method)


def _coupled_selection(candidates,weight,native_selection):
    eligible=[]
    for index,c in enumerate(candidates):
        if c['accepted']and c['global_mask_cost']is not None:
            native=float(sum(r['weighted_robust_cost']for r in c['native_costs']))
            eligible.append(dict(index=index,name=c['name'],family=c['family'],native_energy=native,
                global_mask_cost=c['global_mask_cost'],total_energy=native+weight*c['global_mask_cost']))
    chosen=min(eligible,key=lambda c:(c['total_energy'],c['index']))if eligible else None
    return dict(schema='NMS_NATIVE_PLUS_FIXED_OLD_MASK_SELECTION_V18',
        selected_name=None if chosen is None else chosen['name'],
        selected_family=None if chosen is None else chosen['family'],candidates=eligible,
        mask_weight=weight,rule='minimum native mixture energy plus frozen old mask weight times mask cost',
        ambiguity=deepcopy(native_selection['ambiguity']),global_uniqueness_proved=False,
        no_GT_or_frame_or_partition_method_routing=True)


def _run(task,model):
    started=time.monotonic();before=E._digest(task);records=task['boundary_merged'];candidates=[];details={};bundles={}
    parameters=CAL.mixture_noise_parameters(model)
    cfg=BASE.OUTLINE._outline_config(task['outline']['configuration'])
    supported=bool(task['outline'].get('supported'))
    for name,family in task['families'].items():
        spec=deepcopy(task['spec']);spec['config']['prior_weight']=0.
        base=BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],spec['visible'],
            spec['domains'],spec['image_hw'],family['coarse_pose'],config=spec['config'],
            boundary_observations=records,u_regions=task['u_regions'],
            edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
        Q=np.asarray(family['physical_registration_Q'])
        if not np.allclose(Q,base.Q,atol=1e-12,rtol=0):raise ValueError('Exact registered physical Q required')
        bundle=MIX.authenticate_image_chain_groups(base,task['new_chain_observation'],
            initial_self_hidden=task['initial_self_hidden'],image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
            **parameters)
        problem=MIX.ImageChainMixtureProblem(base,bundle);bundles[name]=bundle
        for index,proposal in enumerate(task['proposals']):
            R=np.asarray(proposal['R_physical'])@Q.T;t=np.asarray(proposal['t']);checked=problem.evaluate(R,t)
            mask=0. if not supported else None;mask_audit=None
            if checked['accepted']and supported:
                p=BASE.NATIVE.V._pose_vector(R,t,problem.scale)
                value,_,mask_audit=BASE.OUTLINE.outline_value_and_gradient(p,problem.xyz,problem.K,problem.scale,task['outline'],cfg)
                if mask_audit.get('valid_projection')and np.isfinite(value):mask=float(value)
            token=name+'/'+str(index)
            candidates.append(dict(name=token,family=name,accepted=checked['accepted'],
                all_joint_constraints_satisfied=checked['accepted'],
                native_costs=BASE._bridge_costs(checked.get('native_costs',[])),global_mask_cost=mask))
            if checked['accepted']:details[token]=dict(evaluation=checked,family=family,proposal_index=index,mask_audit=mask_audit)
    native=BASE.AUTH.select_hierarchical_native_candidates(task['states'],candidates)
    coupled=_coupled_selection(candidates,cfg['outline_factor_weight']if supported else 0.,native)
    arms={ARMS[0]:deepcopy(task['baseline'])};audits={};tangent={}
    for arm,selection in((ARMS[1],native),(ARMS[2],coupled)):
        token=selection['selected_name']
        if token is None:
            family=task['families']['ORIGINAL_WD'];coarse=family['coarse_pose']
            selected=dict(accepted=False,family=family,R=coarse['R'],t=coarse['t'],objective=None,native_costs=[],
                reason='no_saved_proposal_satisfies_current_native_guards_or_required_mask_math')
        else:
            detail=details[token];selected={**detail['evaluation'],'family':detail['family'],
                'proposal_index':detail['proposal_index'],'global_mask_audit':detail['mask_audit']}
        pose=_actual_pose(selected['family'],selected['R'],selected['t'],selected['accepted'],arm)
        arms[arm]=dict(actual_pose=pose,accepted=selected['accepted'],states=task['states'],solution=selected,
            keypoint_readout=BASE._readout(task,records,selected,pose),WD_family=selected['family']['name'],
            candidate_selection=BASE._compact_selection(selection)if arm==ARMS[1]else selection,
            ambiguity_audit=deepcopy(selection['ambiguity']),method=arm,
            new_image_line_group_authentication=bundles,
            fallback_audit=dict(own_original_initial_coarse_only=not selected['accepted'],old_best_pose_fallback=False))
        audits[arm]=dict(evaluated_candidates=len(candidates),accepted_candidates=sum(c['accepted']for c in candidates),
            physical_proposal_inventory=task['proposal_inventory'],new_full_keypoints=0,
            old_partial_normal_refs_replaced_once=True,other_complete_BI_T4_normal_and_hard_guards_preserved=True,
            selected_native_energy=selected['objective'],old_cached_angle_model_used=False,new_optimizations=0)
        tangent[arm]=dict(tangent_variance_inflation=4.,archived_prediction_tangent_not_precise_RGB_point=True)
    if E._digest(task)!=before:raise ValueError('Frozen NMS input changed')
    return C.finite(dict(id=task['id'],partition=task['partition'],session=task['session'],image=task['image'],
        image_sha256=task['image_sha256'],states=task['states'],arms=arms,
        fixed_family={a:arms[a].get('WD_family')for a in ARMS},tangent_control_audit=tangent,
        image_chain_mixture_audit=audits,input_sha256=before,elapsed_seconds=time.monotonic()-started,
        contract=dict(uniform238=True,severe=0,new_RGB_or_NN_calls=0,S_or_U_local_queries=0,
            S_model_projection_only=True,U_regions_only=True,new_full_BI_revocations=0,
            all_new_NMS_alternatives_retained=True,old_line_intercept_not_double_counted=True,
            no_default_adoption=True,no_GT_ID_or_partition_cost_routing=True)))


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every previous prediction preparation')
    parent=C.read(BASE.OUTPUT/'PROTOCOL.json');oldseal=C.read(BASE.OUTPUT/'INPUT_SEAL.json')
    if (oldseal['rows']!=238 or C.sha(BASE.OUTPUT/'INPUTS.jsonl.gz')!=oldseal['input_sha256']
            or C.sha(BASE.OUTPUT/'PROTOCOL.json')!=oldseal['protocol_sha256']):
        raise ValueError('Exact frozen238 candidate inputs required')
    observe=C.read(OBS.OUTPUT/'OBSERVATION_SEAL.json');observe_protocol=C.read(OBS.OUTPUT/'PROTOCOL.json')
    if (observe['rows']!=238 or C.sha(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')!=observe['observations_sha256']
            or C.sha(OBS.OUTPUT/'PROTOCOL.json')!=observe['protocol_sha256']):
        raise ValueError('Completed238 new NMS observations required')
    model=_verified_model();model_seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    sources={**parent['source_sha256'],**observe_protocol['source_sha256'],**model_seal['source_sha256']}
    inputs={**parent['input_sha256'],**observe_protocol['input_sha256'],**model_seal['input_sha256']}
    for folder,names in((BASE.OUTPUT,('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz')),
            (OBS.OUTPUT,('PROTOCOL.json','OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz')),
            (CAL.OUTPUT,('MODEL.json','MODEL_SEAL.json','PROTOCOL.json','PREPARATION_SEAL.json'))):
        for name in names:
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    for module in(__file__,BASE.__file__,CAL.__file__,MIX.__file__,MIX.M.__file__,OBS.__file__):
        path=Path(module);sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    for key in(sources,inputs,protected):E._verify_sources(key)
    observation_rows=list(C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'))
    observation_ids=[r['id']for r in observation_rows]
    if (len(observation_rows)!=238 or len(set(observation_ids))!=238
            or observation_ids!=observe['ordered_ids']or observation_ids!=parent['ids']):
        raise ValueError('Exact unique ordered238 new observation archive required')
    observations={r['id']:r for r in observation_rows};tasks=[]
    for task in C.rows(BASE.OUTPUT/'INPUTS.jsonl.gz'):
        row=observations[task['id']]
        if (row['states']!=task['states']or row['visible_measurements']!=task['spec']['visible']
                or row['domains']!=task['spec']['domains']or row['u_regions']!=task['u_regions']
                or row['image_sha256']!=task['image_sha256']):
            raise ValueError('New observation archive changed old q/covariance/states/domains/image')
        geometry=row['initial_geometry'];original=task['families']['ORIGINAL_WD']
        if (geometry['native_xyz']!=original['native_xyz']or geometry['K']!=task['spec']['K']
                or geometry['physical_registration_Q']!=original['physical_registration_Q']
                or geometry['initial_self_hidden']!=task['initial_self_hidden']
                or geometry['image_hw']!=task['spec']['image_hw']):
            raise ValueError('New NMS association guide must match exact initial ORIGINAL geometry/K/Q/S gates')
        chain=row['chain_observation']
        if chain['schema']!='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C':raise ValueError('Exact fitted-observer generation required')
        tasks.append({**task,'new_chain_observation':chain})
    if ([t['id']for t in tasks]!=parent['ids']or len(tasks)!=238
            or Counter(t['partition']for t in tasks)!=Counter(CAL=46,TEST=49,REMAINDER=143)):
        raise ValueError('Complete238 uniform population, severe0')
    output.mkdir(parents=True)
    protocol=dict(schema='ACTUAL_NMS_NATIVE_EDGE_MIXTURE_CANDIDATE_ALL238_V18',ids=parent['ids'],arms=list(ARMS),
        population=dict(CAL=46,TEST=49,REMAINDER=143,severe=0),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,new_NMS_model_binding=E._binding(CAL.OUTPUT/'MODEL.json'),
        conservative_matching_oldpartial_only=True,new_full_BI_point_revocations=0,
        all_existing_cached_physical_candidates_retained=True,new_optimizations=0,
        exact_fitted_observer='17c, all actual-pixel alternatives; normalized mixture including covariance volume',
        controls='native evidence only with mask numeric ties; native+frozen old mask weight',
        old_normal_refs_replaced_exactly_once=True,other_BI_q_covariance_T4_and_guards_preserved=True,
        no_default_adoption=True,all238_predictions_seal_before_new_metrics=True,
        reused_TEST_and_remaining_exploratory=True,additional143_reference_not_independent_metrology=True)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for rel,digest in sources.items():
        target=output/'code_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/rel,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    for key in(sources,inputs,protected):E._verify_sources(key)
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=protocol['ids'],
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        before_new_image_mixture_predictions_and_metrics=True))
    print(dict(prepared=238,model_digest=model['model_digest_sha256'],physical_proposals=sum(len(t['proposals'])for t in tasks)),flush=True)


def predict(output=OUTPUT,workers=8):
    output=Path(output).resolve();protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if (seal['rows']!=238 or C.sha(output/'INPUTS.jsonl.gz')!=seal['input_sha256']
            or C.sha(output/'PROTOCOL.json')!=seal['protocol_sha256']):raise ValueError('Complete immutable238 input seal required')
    if (output/'PREDICTION_SEAL.json').exists():raise FileExistsError('Preserve completed predictions')
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    tasks=list(C.rows(output/'INPUTS.jsonl.gz'));model=_verified_model();done={};started=time.monotonic()
    if [t['id']for t in tasks]!=protocol['ids']:raise ValueError('Original ordered population required')
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
    futures={pool.submit(_run,t,model):t['id']for t in tasks}
    try:
        for future in as_completed(futures):
            row=future.result();done[row['id']]=row
            C.write(output/'frames'/(row['id'].replace(':','__')+'.json'),row)
            C.write(output/'LIVE_PROGRESS.json',dict(completed=len(done),total=238,latest_id=row['id'],elapsed_seconds=time.monotonic()-started))
            print(str(len(done))+'/238 '+row['id'],flush=True)
    except BaseException as error:
        C.write(output/'FAILED_BEFORE_SEAL.json',dict(completed=len(done),total=238,error_type=type(error).__name__,
            reason=str(error),prediction_seal_written=False,new_reference_reads=0))
        for future in futures:future.cancel()
        for process in list(getattr(pool,'_processes',{}).values()):process.terminate()
        pool.shutdown(wait=False,cancel_futures=True);raise
    else:pool.shutdown(wait=True)
    for key in('source_sha256','input_sha256','protected_user_sha256'):E._verify_sources(protocol[key])
    C.save_rows(output/'PREDICTIONS.jsonl.gz',[done[t['id']]for t in tasks])
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,CAL=46,TEST=49,REMAINDER=143,severe=0,
        predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),all3_arms_complete=True,
        before_new_control_metric_read=True,ordered_ids=protocol['ids'],elapsed_seconds=time.monotonic()-started))


def score(output=OUTPUT):
    scorer=types.FunctionType(BASE.ARCHIVE.score.__code__,{**BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},BASE.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='predict':predict(a.output,a.workers)
    else:score(a.output)
