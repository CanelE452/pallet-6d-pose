"""Exact mathematical-input memoization of one original fixed-e stage run.

The original D1 candidate builder and family/selector logic still execute.
Only previously executed P1/WD0/G10/G13 numeric stage bodies are memoized.
No other research bank or SINGLE/additional-start result is admitted.
"""
from __future__ import annotations
from copy import deepcopy
import importlib,inspect
from pathlib import Path
from types import SimpleNamespace,FunctionType
import hashlib,json
import numpy as np

from . import d1_producer as P,d1_cpu_logging_v3 as V3
from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import rows,read,sha,canonical

PKG='scripts.research.pallet_visibility_curriculum_20261013_v1.'


def source_bindings():
    result=V3.source_bindings();root=Path(__file__).parents[3]
    result[str(Path(__file__).relative_to(root))]=sha(__file__);return result


def solver_job(frame,observation_folder,output_folder,old_run,entry):
    from . import d1_batch as B
    def cached(row,frame,mask,config,model,interval):
        return solve_e_from_entry(row,frame,mask,config,model,interval,entry)
    operation=FunctionType(B.solver_job.__code__,{**B.solver_job.__globals__,
        'P':SimpleNamespace(**{**vars(P),'solve_e':cached}),'source_bindings':source_bindings},name='_exact_cached_CPU_job')
    return operation(frame,observation_folder,output_folder,old_run)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def load_entry(ident,original_run,typed_run):
    original=Path(original_run);typed=Path(typed_run)
    row=next(r for r in rows(original/'PREDICTIONS.jsonl.gz')if r['id']==ident)
    obsdir=original.parent/'fresh_observations_v4_14'
    observation=next(r for r in rows(obsdir/'OBSERVATIONS.jsonl.gz')if r['id']==ident)
    packet=next(r for r in rows(typed/'e_trace_v2/TYPED_PACKETS.jsonl.gz')if r['id']==ident)
    # The old protocol's ID order, not a reference or success list, locates slot.
    protocol=read(typed/'REPLAY_PROTOCOL.json');index=protocol['ordered_ids'].index(ident)
    slotpath=typed/'slots'/f'{index:03d}_TYPED_MULTI.json';slot=read(slotpath)
    if slot['id']!=ident or slot['arm']!='TYPED_MULTI':raise ValueError('Exact original MULTI slot required')
    replay=slot['result']
    if replay['audit']['labels_or_metrics_read']is not False:raise ValueError('Prediction-only replay required')
    return dict(id=ident,original=row,observation=observation,packet=packet,replay=replay,
        bindings=dict(original_prediction_seal_sha256=sha(original/'PREDICTION_SEAL.json'),
            original_protocol_sha256=sha(original/'PROTOCOL.json'),typed_slot_sha256=sha(slotpath),
            typed_packet_row_sha256=digest(packet),old_observation_row_sha256=digest(observation)))


def load_entries(original_run,typed_run):
    """Read each original journal once; retain only the one fixed-e stages."""
    original=Path(original_run);typed=Path(typed_run);obsdir=original.parent/'fresh_observations_v4_14'
    seal=read(original/'PREDICTION_SEAL.json');oseal=read(obsdir/'OBSERVATION_SEAL.json')
    if sha(original/'PREDICTIONS.jsonl.gz')!=seal['predictions_sha256']or sha(obsdir/'OBSERVATIONS.jsonl.gz')!=oseal['observations_sha256']:
        raise ValueError('Exact original fixed-e prediction/observation bytes required')
    old={r['id']:dict(full_prior_seed=r['full_prior_seed'],full_WD_search={k:r['full_WD_search'][k]for k in ('families','solver_configuration','selected_family','family_ambiguity')})for r in rows(original/'PREDICTIONS.jsonl.gz')}
    entries={};wanted=set(old)
    for r in rows(obsdir/'OBSERVATIONS.jsonl.gz'):
        if r['id']in wanted:
            source=r['_original_source']
            minimal=dict(_original_source={k:source[k]for k in ('initial_geometry','states','ot_domains')},
                **{k:r[k]for k in ('frozen_BI_fit','partial_boundary_observation','u_region_observation')})
            entries[r['id']]=dict(id=r['id'],original=old[r['id']],observation=minimal,
                bindings=dict(original_prediction_seal_sha256=sha(original/'PREDICTION_SEAL.json'),
                    original_protocol_sha256=sha(original/'PROTOCOL.json'),old_observation_row_sha256=digest(r)))
    for r in rows(typed/'e_trace_v2/TYPED_PACKETS.jsonl.gz'):
        if r['id']in entries:
            entry=entries[r['id']];entry['packet']={k:r[k]for k in ('solver_input','original_e_legacy_boundary_observations')}
            entry['bindings']['typed_packet_row_sha256']=digest(r)
    for index,ident in enumerate(read(typed/'REPLAY_PROTOCOL.json')['ordered_ids']):
        if ident not in entries:continue
        p=typed/'slots'/f'{index:03d}_TYPED_MULTI.json';slot=read(p)
        if slot['id']!=ident or slot['arm']!='TYPED_MULTI'or slot['result']['audit']['labels_or_metrics_read']is not False:
            raise ValueError('Exact prediction-only original MULTI cache required')
        r=slot['result'];entry=entries[ident]
        stages={stage:{key:r['stage_results'][stage][key]for key in ('families','selected_family','family_ambiguity')}for stage in ('G10','G13')}
        entry['replay']=dict(actual_pose=r['actual_pose'],stage_results=stages,generated_candidates=r['generated_candidates'])
        entry['bindings']['typed_slot_sha256']=sha(p)
    if len(entries)!=143 or any('packet'not in e or'replay'not in e for e in entries.values()):
        raise ValueError('Complete exact original143 cache required')
    return entries


def _mathematical(value):
    """Drop only reporting fields after original input validators have run.

    Source inspection: normalized objective/constraint helpers do not read
    these reporting dictionaries. Validation still runs before this filter,
    including typed archive revocation guards and image SHA/canvas checks.
    """
    if isinstance(value,dict):
        return {k:_mathematical(v)for k,v in value.items()if k not in ('provenance','association_provenance','image_binding','source')}
    if isinstance(value,(list,tuple)):return [_mathematical(v)for v in value]
    if hasattr(value,'tolist'):return _mathematical(value.tolist())
    return value


def _incoming_pose_inputs(solution):
    return dict(accepted=solution.get('accepted',False),
        R=solution.get('R')if solution.get('accepted')else None,
        t=solution.get('t')if solution.get('accepted')else None,
        R_physical=solution.get('R_physical')if solution.get('accepted')else None,
        feasible_pose_samples=[dict(R=s['R'],t=s['t'])for s in solution.get('feasible_pose_samples',[])])


def solve_e_from_entry(row,frame,mask,config,model,interval,entry):
    """Execute unchanged pure D1 glue with exact-input original stage caches."""
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_partial_boundary_u_solver_v4_6 as PS
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_two_wd_family_v4_9 as W
    from scripts.research.pallet_visibility_curriculum_20261013_v1 import evaluate_visibility_pose_v4_8 as SPEC
    from scripts.research.pallet_statewise_correspondence_diagnosis_v1 import typed_replay as T
    from scripts.research.pallet_visibility_curriculum_20261013_v1.common import finite
    if frame['id']!=entry['id']:raise ValueError('Exact cached ID required')
    old=entry['original'];obs=entry['observation'];packet=entry['packet'];replay=entry['replay']
    cfg=deepcopy(config);cfg['prior_weight']=0.
    if finite(SPEC._spec(row,config))!=finite(SPEC._spec(obs,config)):
        raise ValueError('Current original e full P1 mathematical spec differs')
    families={f['name']:f for f in old['full_WD_search']['families']}
    input_checks=[];p_calls=[];family_calls=[];original_partial=PS.solve_boundary_u_region_joint_ot_pnp;original_backend=T._backend
    signature=inspect.signature(original_partial)
    def memo_partial(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();actual=finite(dict(bound.arguments))
        weight=float(actual['config']['prior_weight'])
        if weight>0:
            expected=SPEC._spec(obs,config);answer=old['full_prior_seed'];stage='P1'
            expected['boundary_factor_weight']=1.
        else:
            matches=[f for f in families.values()if np.array_equal(f['native_xyz'],actual['xyz'])]
            if len(matches)!=1:raise ValueError('Original exact WD family required')
            family=matches[0];expected=SPEC._spec(obs,cfg);stage='WD0/'+family['name'];answer=family['solution']
            expected.update(xyz=family['native_xyz'],coarse_pose=family['coarse_pose'],
                initializations=family['initializations'],config=old['full_WD_search']['solver_configuration'],boundary_factor_weight=1.)
        expected=finite(expected)
        if actual!=expected:raise ValueError('Original exact cached stage inputs differ: '+stage)
        input_checks.append(dict(stage=stage,mathematical_inputs_sha256=digest(actual),exact=True))
        p_calls.append(dict(call_index=len(p_calls),input_prior_weight=weight,returned_original_solution=V3._summary(answer),cache_reused=True))
        return deepcopy(answer)
    def memo_backend(name,logger,**kwargs):
        if kwargs:raise ValueError('No SINGLE/additional-start backend is allowed')
        stage='G10'if name=='statewise_global_outline_solver_v4_10'else'G13'if name=='statewise_partial_BI_outline_solver_v4_13'else None
        if stage is None:raise ValueError('Only original G10/G13 stage caches are allowed')
        backend=importlib.import_module(PKG+name)
        def refine(xyz,K,states,visible,domains,image_hw,coarse_pose,*,outline_observation,initial_solutions,initializations=None,u_regions=None,boundary_observations=None,config=None,outline_configuration=None):
            matches=[f for f in families.values()if np.array_equal(f['native_xyz'],xyz)]
            if len(matches)!=1:raise ValueError('Original cached family geometry required')
            family=matches[0];fname=family['name'];spec=packet['solver_input']
            expected_boundary=packet['original_e_legacy_boundary_observations']if stage=='G10'else spec['boundary_observations']
            actual_prepared=backend._prepare(xyz,K,states,visible,domains,image_hw,coarse_pose,config,boundary_observations,u_regions)
            expected_prepared=backend._prepare(family['native_xyz'],spec['K'],spec['states'],spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],spec['config'],expected_boundary,spec['u_regions'])
            if finite(_mathematical(actual_prepared))!=finite(_mathematical(expected_prepared)):
                raise ValueError('Prepared original cache mathematical inputs differ: '+stage+'/'+fname)
            keys=('supported','configuration','image_hw','observed_boundary_samples_xy','observed_boundary_segments_xy')
            actual_outline={k:outline_observation.get(k)for k in keys};expected_outline={k:spec['outline_observation'].get(k)for k in keys}
            if actual_outline!=expected_outline:raise ValueError('Original whole-outline numeric inputs differ')
            expected_solutions=[family['solution']];expected_starts=[]
            if fname=='ORIGINAL_WD':expected_solutions.append(old['full_prior_seed'])
            else:
                R,t,_=T.choose_readout(old['full_prior_seed'],families['ORIGINAL_WD']['coarse_pose'])
                expected_starts=[dict(R=(np.asarray(R)@W.RY90.T).tolist(),t=t,name='PRIOR_PHYSICAL_POSE_TRANSFORMED_SEED_ONLY')]
            actual_starts=finite(initializations or[])
            if actual_starts!=finite(expected_starts)or [_incoming_pose_inputs(s)for s in initial_solutions]!=[_incoming_pose_inputs(s)for s in expected_solutions]:
                raise ValueError('Original exact P1/WD0 incoming starts differ')
            source=next(f for f in replay['stage_results'][stage]['families']if f['name']==fname)
            for endpoint in replay['generated_candidates']:
                if endpoint['stage']==stage and endpoint['family']==fname:
                    logger(endpoint['R'],endpoint['t'],endpoint['original_endpoint_record'],endpoint['kind'])
            input_checks.append(dict(stage=stage+'/'+fname,mathematical_inputs_sha256=digest(finite(_mathematical(actual_prepared))),
                outline_inputs_sha256=digest(actual_outline),incoming_starts_sha256=digest(finite(dict(initializations=actual_starts,solutions=[_incoming_pose_inputs(s)for s in initial_solutions]))),exact=True))
            family_calls.append(dict(stage=stage,family=fname,solution=V3._summary(source['solution']),cache_reused=True))
            return deepcopy(source['solution'])
        return SimpleNamespace(refine_global_outline_joint_pose=refine)
    PS.solve_boundary_u_region_joint_ot_pnp=memo_partial;T._backend=memo_backend
    try:result=P.solve_e(row,frame,mask,config,model,interval)
    finally:PS.solve_boundary_u_region_joint_ot_pnp=original_partial;T._backend=original_backend
    if result['e0_regenerated']['R_physical']!=replay['actual_pose']['R_physical']or result['e0_regenerated']['centroid']!=replay['actual_pose']['centroid']:
        raise ValueError('Unchanged hierarchical D1 E0 readout differs from original cache')
    grouped=[]
    for stage in ('WD0_SEEDS','G10','G13'):
        if stage=='WD0_SEEDS':original_stage=old['full_WD_search']
        elif stage=='G10':original_stage=replay['stage_results']['G10']
        else:
            if not result['stage_summaries']['G13_executed']:continue
            original_stage=replay['stage_results']['G13']
        grouped.append(dict(stage=stage,selected_family=original_stage['selected_family'],family_ambiguity=deepcopy(original_stage.get('family_ambiguity')),
            families=[dict(name=f['name'],accepted=f['accepted'],objective=f['objective'],solution=V3._summary(f['solution']))for f in original_stage['families']]))
    result['stage_execution_diagnostics']=finite(dict(scope='same original fixed-e stage caches; complete original returned outcomes retained in their original compact archive form',
        P1_and_WD_backend_calls=p_calls,family_stage_calls=grouped,feasible_pose_coordinates_not_duplicated_here=True,
        metadata_only=True,source_archive_compaction='only repeated visible/domain audits removed; all original optimizer outcomes and distinct feasible samples retained'))
    result['old_full_run_reused']=True
    result['proof']['exact_stage_memoization']=dict(source_sha256=sha(__file__),input_checks=input_checks,
        original_execution_bindings=entry['bindings'],new_optimizer_or_PnP_calls=0,history_only_union=False,
        only_original_fixed_P1_WD0_G10_G13=True,E0_previously_independently_executed=True)
    return result
