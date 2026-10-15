"""Complete B/I actual-edge replacement with frozen bounded O/S priors.

This is a new uniform experiment, with separate CAL edge noise and immutable
v19 start names. It reuses cached real image chains and typed topology only.
All original q hard guards remain. No precise hidden point, new RGB/network
forward, GT-conditioned proposal or default replacement is introduced.
"""
from collections import Counter
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
import argparse
import shutil
import sys
import types
import numpy as np

from . import evaluate_bounded_geometric_priors_v4_22 as OLD
from . import complete_bi_image_chain_evidence_v4_20d as COMP
from . import calibrate_complete_bi_edge_mixture_v4_20d as CAL
from . import observe_grazing_native_edge_chains_v4_18b as OBS

C,E,REFINE=OLD.C,OLD.E,OLD.REFINE
OUTPUT=OLD.OUTPUT.parent/'complete_bi_bounded_priors_eval_v4_23'
PLAN=OLD.OUTPUT.parent/'complete_bi_bounded_priors_plan_v4_23/PREDECLARATION.json'
ARMS=('SAME_FROZEN_E','COMPLETE_BI_O_SOFT_NATIVE','COMPLETE_BI_O_SOFT_S_NATIVE',
    'COMPLETE_BI_O_SOFT_PLUS_MASK','COMPLETE_BI_O_SOFT_S_PLUS_MASK')
PROFILE_ARMS={'O':(ARMS[1],ARMS[3]),'OS':(ARMS[2],ARMS[4])}
PROFILES=tuple((name,name=='OS',*arms)for name,arms in PROFILE_ARMS.items())


def _verified_models():
    models=OLD._verified_models()
    seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    if seal['fit_partition']!='CAL'or seal['model_sha256']!=C.sha(CAL.OUTPUT/'MODEL.json'):
        raise ValueError('Separate full CAL complete-BI edge model must be frozen')
    for name,key in (('CAL_MIXTURE_ERRORS.jsonl.gz','CAL_errors_sha256'),
            ('CAL_LOGO_REPORT.json','LOGO_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(CAL.OUTPUT/name)!=seal[key]:raise ValueError('Sealed complete-BI calibration archive changed')
    for key in ('source_sha256','input_sha256'):E._verify_sources(seal[key])
    model=C.read(CAL.OUTPUT/'MODEL.json');CAL.mixture_noise_parameters(model)
    models['NMS']=dict(COMPLETE=model,OLD=models['NMS'])
    return models


class _CompleteProblem(COMP.CompleteBIImageChainProblem):
    def objective(self,p):
        value,gradient,parts=super().objective(p)
        parts.update(replaced_old_normal_factor_count=len(self.plan['replaces_normal_factor_refs']),
            count_source='exact new typed replacement plan; base.replaced was not mutated',
            normal_and_complete_cost_replacement_separate_from_unchanged_hard_guards=True)
        return value,gradient,parts


def _problem(task,family,models):
    spec=task['spec'];cfg={**spec['config'],'prior_weight':0.}
    base=OLD.BASE.NATIVE.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],
        spec['visible'],spec['domains'],spec['image_hw'],family['coarse_pose'],config=cfg,
        boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],
        edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
    if not np.allclose(base.Q,family['physical_registration_Q'],atol=1e-12,rtol=0):
        raise ValueError('Same physical registration required')
    bundle=COMP.authenticate_complete_bi_image_chain_groups(base,task['new_chain_observation'],
        junction_proposals=task['complete_BI_evidence']['junction_proposals'],
        visible_measurements=spec['visible'],initial_self_hidden=task['initial_self_hidden'],
        image_binding=dict(sha256=task['image_sha256'],image_hw=spec['image_hw']),
        initial_projection_xy=task['complete_BI_evidence']['initial_projection_xy'],
        **CAL.mixture_noise_parameters(models['COMPLETE']))
    problem=_CompleteProblem(base,bundle,overlap_policy=COMP.PROTOTYPE_OVERLAP_POLICY)
    # This is the unchanged v19-native start ranking, not the new likelihood.
    problem.reference_start_problem=OLD.V.PARENT._problem(task,family,models['OLD'])[0]
    return problem,{**bundle,'runtime_complete_BI_replacement_plan':problem.plan}


def _choose_original_starts(problem,proposals,coarse_pose,*,configuration=None):
    return REFINE.choose_native_starts(problem.reference_start_problem,proposals,coarse_pose,
        configuration=configuration)


_REFINER=types.FunctionType(REFINE.refine_native_edge_pose.__code__,
    {**REFINE.refine_native_edge_pose.__globals__,'choose_native_starts':_choose_original_starts},
    'refine_complete_BI_with_exact_original_starts')


def _bridge_costs(records):
    result=[]
    for record in records:
        if record['kind']=='ARCHIVED_COMPLETE_BI_TANGENT':
            result.append({**record,'kind':'ARCHIVED_TANGENT','original_native_cost_kind':record['kind'],
                'source':'original complete-BI measurement; uncertain tangent only'})
        else:result.extend(OLD.BASE._bridge_costs([record]))
    return result


def _readout(task,records,selected,pose):
    result=OLD.BASE._readout(task,records,selected,pose)
    plan=selected.get('complete_BI_replacement_audit')
    if plan is not None:
        for corner in plan['corner_plans']:
            if not corner['point_cost_revoked']:continue
            k=corner['native_corner'];result['reported_points'][k]=None
            result['point_sources'][k]=('ACTUAL_NATIVE_EDGE_AND_UNCERTAIN_ARCHIVED_COMPLETE_TANGENT'
                if 'tangent_kind'in corner else 'JOINT_NATIVE_EDGE_CONSTRAINTS_NOT_A_PRECISE_2D_KP')
        result.update(complete_BI_revoked_point_ids=plan['revoked_complete_ids'],
            original_complete_q_not_reported_as_new_corrected_point=True,
            model_projection8_is_joint_pose_hypothesis=True,new_full_2D_corner_measurements=0)
    return result


BASE=types.SimpleNamespace(**{**OLD.BASE.__dict__,'_bridge_costs':_bridge_costs,'_readout':_readout})
_FACADE=types.SimpleNamespace(PARENT=types.SimpleNamespace(_problem=_problem))
_WORKER=types.FunctionType(OLD._run.__code__,{**OLD._run.__globals__,
    'V':_FACADE,'BASE':BASE,'ARMS':ARMS,'PROFILES':PROFILES,'_REFINER':_REFINER},'run_complete_BI_bounded_priors')


def _run(task,models):
    row=_WORKER(task,models);row['schema']='COMPLETE_BI_BOUNDED_O_S_PREDICTION_V23'
    row['contract'].pop('all_v19_observations_and_hard_constraints_unchanged',None)
    row['contract'].update(actual_observation_archive_unchanged=True,original_hard_guards_unchanged=True,
        typed_complete_BI_cost_replacement_and_support_balanced_variants=True,
        separate_CAL_edge_model=True,original_v19_start_names_unchanged=True,
        revoked_complete_q_not_used_as_precise_point=True)
    for arm in ARMS[1:]:
        row['tangent_control_audit'][arm]=dict(unchanged_old_partial_tangent_inflation=4.,
            new_archived_complete_tangent_inflation=4.,unchanged_native_likelihood=False,
            uncertainty_is_existing_engineering_choice_not_new_precise_RGB_measurement=True)
    return row


def _join(task,observation):
    if any(task[k]!=observation[k]for k in ('id','partition','session','image','image_sha256')):
        raise ValueError('Exact sealed frame/image identity required')
    if (task['states']!=observation['states']or task['spec']['visible']!=observation['visible_measurements']
            or task['u_regions']!=observation['u_regions']
            or E._digest(task['new_chain_observation'])!=E._digest(observation['chain_observation'])):
        raise ValueError('Same actual NMS states, visible observations, and U ranges required')
    original={r['native_corner']:r['original_domain']for r in task['O_localization_abstention']['records']}
    restored=[original.get(d['native_corner'],d)for d in task['spec']['domains']]
    if E._digest(restored)!=E._digest(observation['domains']):
        raise ValueError('Only declared O hard localization adaptation may differ')
    geometry=observation['initial_geometry'];family=task['families']['ORIGINAL_WD'];pose=family['coarse_pose']
    if (geometry['native_xyz']!=family['native_xyz']or geometry['physical_registration_Q']!=family['physical_registration_Q']
            or geometry['K']!=task['spec']['K']or geometry['image_hw']!=task['spec']['image_hw']
            or geometry['initial_self_hidden']!=task['initial_self_hidden']
            or not np.allclose(geometry['initial_projection_xy'],C.project(family['native_xyz'],pose['R'],pose['t'],geometry['K']),atol=1e-8,rtol=0)):
        raise ValueError('Exact original-only model guide geometry required')
    return {**task,'complete_BI_evidence':dict(junction_proposals=observation['junction_proposals'],
        initial_projection_xy=geometry['initial_projection_xy'],
        observation_generation=observation['observation_generation'])}


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous preparations')
    models=_verified_models();parent=C.read(OLD.OUTPUT/'PROTOCOL.json');parentseal=C.read(OLD.OUTPUT/'INPUT_SEAL.json')
    obsseal=C.read(OBS.OUTPUT/'OBSERVATION_SEAL.json')
    obsjournal=OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'
    if (parentseal['rows']!=238 or parentseal['protocol_sha256']!=C.sha(OLD.OUTPUT/'PROTOCOL.json')
            or parentseal['input_sha256']!=C.sha(OLD.OUTPUT/'INPUTS.jsonl.gz')
            or obsseal['rows']!=238 or obsseal['observations_sha256']!=C.sha(obsjournal)
            or obsseal['protocol_sha256']!=C.sha(OBS.OUTPUT/'PROTOCOL.json')
            or obsseal['ordered_ids']!=parent['ids']):
        raise ValueError('Complete sealed task and real-observation journals required')
    declaration=C.read(PLAN)
    if declaration['arms']!=list(ARMS):raise ValueError('Preserve prior method declaration')
    sources=dict(parent['source_sha256']);inputs=dict(parent['input_sha256'])
    for module in (sys.modules[__name__],OLD,COMP,CAL):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    calseal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    for relative,digest in calseal['source_sha256'].items():
        if relative in sources and sources[relative]!=digest:raise ValueError('Frozen dependency versions differ')
        sources[relative]=digest
    for p in (PLAN,OLD.OUTPUT/'PROTOCOL.json',OLD.OUTPUT/'INPUT_SEAL.json',
            OLD.OUTPUT/'INPUTS.jsonl.gz',OBS.OUTPUT/'PROTOCOL.json',OBS.OUTPUT/'OBSERVATION_SEAL.json',obsjournal,
            CAL.OUTPUT/'PROTOCOL.json',CAL.OUTPUT/'PREPARATION_SEAL.json',CAL.OUTPUT/'MODEL.json',
            CAL.OUTPUT/'MODEL_SEAL.json',CAL.OUTPUT/'CAL_LOGO_REPORT.json'):
        inputs[str(p.relative_to(C.ROOT))]=C.sha(p)
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol={**parent,'schema':'COMPLETE_BI_BOUNDED_O_S_ALL238_PROTOCOL_V23','arms':list(ARMS),
        'source_sha256':sources,'input_sha256':inputs,'declaration':E._binding(PLAN),
        'new_complete_BI_model_binding':E._binding(CAL.OUTPUT/'MODEL.json'),
        'old_NMS_model_binding_for_v19_start_ranking':parent['new_NMS_model_binding'],
        'new_NMS_model_binding':E._binding(CAL.OUTPUT/'MODEL.json'),
        'new_evidence_family':COMP.NOISE_OBSERVATION_FAMILY,
        'exact_original_v19_start_names_despite_new_observation_likelihood':True,
        'old_raw_centered_visible_hard_guards_unchanged':True,'no_default_adoption':True}
    C.write(output/'PROTOCOL.json',protocol)
    counts=Counter();modes=Counter();total=0
    def records():
        nonlocal total
        for task,observation in zip_longest(C.rows(OLD.OUTPUT/'INPUTS.jsonl.gz'),C.rows(obsjournal)):
            if task is None or observation is None:raise ValueError('Complete original aligned238 required')
            joined=_join(task,observation)
            OLD._prior_inputs(joined,models)
            if joined['id']!=parent['ids'][total]:raise ValueError('Original order changed')
            total+=1;counts[joined['partition']]+=1
            for family in joined['families'].values():
                problem,bundle=_problem(joined,family,models['NMS'])
                modes.update(groups=len(problem.groups),revoked_complete=len(problem.revoked))
            yield joined
    C.save_rows(output/'INPUTS.jsonl.gz',records())
    if total!=238 or counts!=Counter(CAL=46,TEST=49,REMAINDER=143):raise ValueError('Keep all238 without severe')
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    for mapping in (sources,inputs,parent['protected_user_sha256']):E._verify_sources(mapping)
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=parent['ids'],protocol_sha256=C.sha(output/'PROTOCOL.json'),
        input_sha256=C.sha(output/'INPUTS.jsonl.gz'),before_new_predictions_or_metrics=True,
        both_WD_metadata_preflight_complete=True,metadata_inventory=dict(modes)))
    print(dict(prepared=238,both_WD_metadata_inventory=dict(modes)),flush=True)


def predict(output=OUTPUT,workers=8):
    runner=types.FunctionType(OLD.predict.__code__,{**OLD.predict.__globals__,
        '_run':_run,'_verified_models':_verified_models,'ARMS':ARMS},'predict_complete_BI_bounded_priors')
    return runner(Path(output),workers)


def score(output=OUTPUT):
    runner=types.FunctionType(OLD.score.__code__,{**OLD.score.__globals__,'ARMS':ARMS,'BASE':BASE},
        'score_complete_BI_bounded_priors')
    return runner(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
