"""Uniform238 fresh grazing-NMS mixture controls on the frozen pose bank.

The v18 worker mathematics and selectors are unchanged.  This separate
generation uses the sealed grazing states/filtered boundary profile, fresh
grazing-gated actual NMS observations, and their separately fitted CAL model.
No new pose optimization, NN forward, exact U point or GT routing occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import types

from . import common as C
from . import evaluate_image_chain_mixtures_v4_18 as OLD
from . import calibrate_grazing_edge_mixture_v4_18b as CAL
from . import observe_grazing_native_edge_chains_v4_18b as OBS


OUTPUT=OLD.OUTPUT.parent/'grazing_image_chain_mixture_candidates_v4_18b'
PROFILE=CAL.PROFILE
ARMS=('SAME_FROZEN_E','GRAZING_NMS_NATIVE_D_MERGED','GRAZING_NMS_PLUS_MASK_D_MERGED')
_WORKER=types.FunctionType(OLD._run.__code__,{**OLD._run.__globals__,'ARMS':ARMS,'CAL':CAL},OLD._run.__name__)


def _verified_model():
    seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    for name,key in (('MODEL.json','model_sha256'),('CAL_MIXTURE_ERRORS.jsonl.gz','CAL_errors_sha256'),
            ('CAL_LOGO_REPORT.json','LOGO_sha256'),('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(CAL.OUTPUT/name)!=seal[key]:raise ValueError('Separate grazing-NMS CAL evidence changed')
    for key in ('source_sha256','input_sha256'):OLD.E._verify_sources(seal[key])
    model=C.read(CAL.OUTPUT/'MODEL.json');CAL.mixture_noise_parameters(model)
    if model['recipe']['observation_generation']!=OBS.OBSERVATION_GENERATION:
        raise ValueError('New grazing-gated observation generation requires its own model')
    return model


def _join_task(task,observation,producer_input,original_v18):
    """Pure production-data join, with no raw keypoints or reference fields."""
    for key in ('id','partition','session','image','image_sha256'):
        if task[key]!=observation[key]or task[key]!=producer_input[key]or task[key]!=original_v18[key]:
            raise ValueError('Same frame/RGB identity required across all sealed profiles')
    if (task['states']!=task['spec']['states']or observation['states']!=task['states']
            or producer_input['states']!=task['states']
            or observation['visible_measurements']!=task['spec']['visible']
            or observation['domains']!=task['spec']['domains']or observation['u_regions']!=task['u_regions']):
        raise ValueError('Exact grazing states/retained q/covariance/domains/U ranges required')
    if (observation['observation_generation']!=OBS.OBSERVATION_GENERATION
            or producer_input['observation_generation']!=OBS.OBSERVATION_GENERATION
            or observation['chain_observation']['schema']!='IMAGE_SUPPORTED_EDGE_CHAINS_V4_17C'):
        raise ValueError('Fresh actual grazing-gated17c archive required')
    adaptation=task['grazing_adaptation'];digest=OLD.E._digest(adaptation)
    if (observation['sealed_grazing_adaptation_sha256']!=digest
            or producer_input['sealed_grazing_adaptation_sha256']!=digest
            or producer_input['sealed_grazing_filtered_boundary_sha256']!=OLD.E._digest(task['boundary_merged'])
            or observation['new_grazing_U_native_ids']!=adaptation['geometry_gate']['new_U_native_ids']):
        raise ValueError('Producer and solver must bind the same filtered grazing boundary profile')
    geometry=observation['initial_geometry'];family=task['families']['ORIGINAL_WD']
    if (geometry['native_xyz']!=family['native_xyz']or geometry['physical_registration_Q']!=family['physical_registration_Q']
            or geometry['K']!=task['spec']['K']or geometry['image_hw']!=task['spec']['image_hw']
            or geometry['initial_self_hidden']!=task['initial_self_hidden']):
        raise ValueError('Exact initial ORIGINAL model/K/Q/S/canvas must guide the observations')
    for key in ('families','proposals','proposal_inventory','outline','baseline'):
        if OLD.E._digest(task[key])!=OLD.E._digest(original_v18[key]):
            raise ValueError('Unchanged physical pose bank, models, mask and baseline required: '+key)
    if task['spec']['config']!=original_v18['spec']['config']:
        raise ValueError('No solver/visibility/noise configuration change is allowed')
    return {**task,'new_chain_observation':observation['chain_observation'],
        'fresh_grazing_observation_generation':OBS.OBSERVATION_GENERATION}


def _run(task,model):
    if task['fresh_grazing_observation_generation']!=OBS.OBSERVATION_GENERATION:
        raise ValueError('Declared fresh grazing observation generation required')
    row=_WORKER(task,model)
    row.update(schema='FRESH_GRAZING_NMS_MIXTURE_CACHED_PREDICTION_V18B',
        observation_generation=OBS.OBSERVATION_GENERATION,
        original_states=task['grazing_adaptation']['original_states'],
        grazing_adaptation=deepcopy(task['grazing_adaptation']))
    row['contract'].update(initial_grazing_U_applied_before_fresh_NMS=True,
        fresh_grazing_NMS_and_separate_CAL_model_used=True,old_NMS_chains_used_as_new_observations=False,
        retained_complete_BI_and_partial_archives_from_sealed_grazing_control=True,
        original_initial_S_and_previous_U_O_T_preserved=True,
        original_physical_pose_bank_unchanged=True)
    return row


def _sealed(folder,seal_name,journal,journal_key,rows):
    protocol=C.read(folder/'PROTOCOL.json');seal=C.read(folder/seal_name)
    if (seal['rows']!=rows or C.sha(folder/'PROTOCOL.json')!=seal['protocol_sha256']
            or C.sha(folder/journal)!=seal[journal_key]):
        raise ValueError('Required sealed production input changed: '+str(folder))
    for key in ('source_sha256','input_sha256','protected_user_sha256'):OLD.E._verify_sources(protocol[key])
    return protocol,seal


def prepare(output=OUTPUT):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier model/control preparations')
    parent,parent_seal=_sealed(PROFILE,'INPUT_SEAL.json','INPUTS.jsonl.gz','input_sha256',238)
    observe,observe_seal=_sealed(OBS.OUTPUT,'OBSERVATION_SEAL.json','OBSERVATIONS.jsonl.gz','observations_sha256',238)
    prep=C.read(OBS.OUTPUT/'PREPARATION_SEAL.json')
    if (C.sha(OBS.OUTPUT/'PREPARATION_SEAL.json')!=observe_seal['preparation_seal_sha256']
            or C.sha(OBS.OUTPUT/'INPUTS.jsonl.gz')!=prep['input_sha256']
            or observe_seal['observation_generation']!=OBS.OBSERVATION_GENERATION):
        raise ValueError('Exact fresh grazing-NMS producer input/preparation seal required')
    original,original_seal=_sealed(OLD.OUTPUT,'INPUT_SEAL.json','INPUTS.jsonl.gz','input_sha256',238)
    model=_verified_model();model_seal=C.read(CAL.OUTPUT/'MODEL_SEAL.json')
    profiles=list(C.rows(PROFILE/'INPUTS.jsonl.gz'));observations=list(C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'))
    producer=list(C.rows(OBS.OUTPUT/'INPUTS.jsonl.gz'));originals=list(C.rows(OLD.OUTPUT/'INPUTS.jsonl.gz'))
    ids=parent['ids']
    for rows in (profiles,observations,producer,originals):
        values=[r['id']for r in rows]
        if len(values)!=238 or len(set(values))!=238 or values!=ids:
            raise ValueError('Every original unique238 frame/order must remain')
    if ids!=observe_seal['ordered_ids']or ids!=original['ids']or ids!=parent_seal['ordered_ids']:
        raise ValueError('Population seals differ')
    tasks=[_join_task(*values)for values in zip(profiles,observations,producer,originals)]
    sources=OBS._merge_maps(parent['source_sha256'],observe['source_sha256'],model_seal['source_sha256'],original['source_sha256'])
    inputs=OBS._merge_maps(parent['input_sha256'],observe['input_sha256'],model_seal['input_sha256'],original['input_sha256'])
    for module in (sys.modules[__name__],OLD,CAL,OBS):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    for folder,names in ((PROFILE,('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz')),
            (OBS.OUTPUT,('PROTOCOL.json','PREPARATION_SEAL.json','INPUTS.jsonl.gz','OBSERVATIONS.jsonl.gz','OBSERVATION_SEAL.json')),
            (CAL.OUTPUT,('MODEL.json','MODEL_SEAL.json','PROTOCOL.json','PREPARATION_SEAL.json')),
            (OLD.OUTPUT,('PROTOCOL.json','INPUT_SEAL.json','INPUTS.jsonl.gz'))):
        for name in names:
            path=folder/name;inputs[str(path.relative_to(C.ROOT))]=C.sha(path)
    protected=parent['protected_user_sha256']
    if protected!=observe['protected_user_sha256']or protected!=original['protected_user_sha256']:
        raise ValueError('Protected user file inventories differ')
    for mapping in (sources,inputs,protected):OLD.E._verify_sources(mapping)
    output.mkdir(parents=True)
    protocol=dict(schema='FRESH_GRAZING_NMS_MIXTURE_ALL238_PROTOCOL_V18B',ids=ids,arms=list(ARMS),
        population=deepcopy(parent['population']),source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=protected,observation_generation=OBS.OBSERVATION_GENERATION,
        new_NMS_model_binding=OLD.E._binding(CAL.OUTPUT/'MODEL.json'),
        states_and_boundary_profile='sealed grazing INPUTS; retainedvisible plus filteredboundary_merged',
        all_original_physical_proposals_models_and_oldmask_unchanged=True,
        controls='same v18 native minsum/machine-mask-ties and native+fixed old mask weight',
        private_worker_code_identical_to_v18=True,no_global_module_mutation=True,
        conservative_matching_oldpartial_only=True,new_full_BI_point_revocations=0,
        no_new_pose_RGB_or_NN_fits=True,all238_before_new_metrics=True,no_default_adoption=True,
        exploratory_reused_data=True,additional143_reference_not_independent_metrology=True)
    C.write(output/'PROTOCOL.json',protocol);C.save_rows(output/'INPUTS.jsonl.gz',tasks)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot mismatch')
    for mapping in (sources,inputs,protected):OLD.E._verify_sources(mapping)
    C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=ids,input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),before_new_predictions_and_metrics=True))
    print(dict(prepared=238,model_digest=model['model_digest_sha256'],
        physical_proposals=sum(len(t['proposals'])for t in tasks)),flush=True)


def predict(output=OUTPUT,workers=8):
    runner=types.FunctionType(OLD.predict.__code__,{**OLD.predict.__globals__,
        '_run':_run,'_verified_model':_verified_model},OLD.predict.__name__)
    return runner(Path(output),workers)


def score(output=OUTPUT):
    scorer=types.FunctionType(OLD.BASE.ARCHIVE.score.__code__,
        {**OLD.BASE.ARCHIVE.score.__globals__,'ARMS':ARMS},OLD.BASE.ARCHIVE.score.__name__)
    return scorer(Path(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','predict','score'))
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.output)
    elif args.command=='predict':predict(args.output,args.workers)
    else:score(args.output)
