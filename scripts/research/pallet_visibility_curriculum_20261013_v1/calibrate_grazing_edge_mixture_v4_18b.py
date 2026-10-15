"""Separate CAL mixture noise for grazing-gated, fresh17c observations.

The noise mathematics, priors, floors and fixed optimizer starts are the
immutable v18 recipe.  Only the sealed observation population is different:
new states, retained visible records and filtered boundary_merged are used
before authenticating every actual-chain alternative.  No old boundary_all,
TEST/remaining labels, image forward or pose fit is read/run here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime,timezone
from itertools import islice
from pathlib import Path
import shutil
import sys
import types

import numpy as np

from . import common as C
from . import calibrate_native_edge_mixture_v4_18 as OLD
from . import observe_grazing_native_edge_chains_v4_18b as OBS


OUTPUT=OLD.OUTPUT.parent/'CAL_grazing_native_edge_mixture_v4_18b'
PROFILE=OBS.GRAZING
LABELS=OLD.LABELS
SCHEMA='CAL_GRAZING_ACTUAL_NMS_NATIVE_EDGE_MIXTURE_NOISE_V4_18B'
RECIPE={**deepcopy(OLD.RECIPE),
    'observation_generation':OBS.OBSERVATION_GENERATION,
    'applicability':'sealed grazing-filtered boundary_merged physical edges only; same runtime authentication',
    'state_profile':'initial ORIGINAL geometry B/I grazing->U under existing geometry tolerance; permanent S unchanged',
    'retained_visible_noise':'exact sealed grazing retained B/I q/covariance/fit covariance/reliability',
    'old_boundary_all_used':False,
    'previous_observer_models_reused':False}

# These private function objects keep the complete old mathematical recipe.
# Only schema/recipe globals differ; no frozen module attribute is mutated.
_FIT_MODEL=types.FunctionType(OLD.fit_native_edge_mixture_model.__code__,
    {**OLD.fit_native_edge_mixture_model.__globals__,'RECIPE':RECIPE,'SCHEMA':SCHEMA},
    OLD.fit_native_edge_mixture_model.__name__,OLD.fit_native_edge_mixture_model.__defaults__)
_FIT_MODEL.__kwdefaults__=deepcopy(OLD.fit_native_edge_mixture_model.__kwdefaults__)
_EXPORT=types.FunctionType(OLD.mixture_noise_parameters.__code__,
    {**OLD.mixture_noise_parameters.__globals__,'RECIPE':RECIPE,'SCHEMA':SCHEMA},
    OLD.mixture_noise_parameters.__name__,OLD.mixture_noise_parameters.__defaults__)


def fit_native_edge_mixture_model(records,**kwargs):
    return _FIT_MODEL(records,**kwargs)


def mixture_noise_parameters(model):
    result=_EXPORT(model)
    result['noise_provenance'].update(observation_generation=OBS.OBSERVATION_GENERATION,
        state_profile=RECIPE['state_profile'],applicability_profile=RECIPE['applicability'],
        old_boundary_all_or_old_observer_model_used=False)
    return result


def _verify_preparation(output):
    protocol=OLD._verify_preparation(output)
    if protocol['recipe']!=RECIPE or protocol['observation_generation']!=OBS.OBSERVATION_GENERATION:
        raise ValueError('Separate declared grazing-NMS preparation required')
    return protocol


def prepare(output=OUTPUT):
    """Seal all CAL sources/inputs before reference coordinates are decoded."""
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every earlier calibration preparation')
    obs_protocol=C.read(OBS.OUTPUT/'PROTOCOL.json');obs_seal=C.read(OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json')
    obs_preparation=C.read(OBS.OUTPUT/'PREPARATION_SEAL.json')
    obs_journal=OBS.OUTPUT/'CAL_OBSERVATIONS.jsonl.gz'
    if (obs_seal['rows']!=46 or C.sha(obs_journal)!=obs_seal['observations_sha256']
            or C.sha(OBS.OUTPUT/'PROTOCOL.json')!=obs_seal['protocol_sha256']
            or C.sha(OBS.OUTPUT/'INPUTS.jsonl.gz')!=obs_preparation['input_sha256']
            or C.sha(OBS.OUTPUT/'PREPARATION_SEAL.json')!=obs_seal['preparation_seal_sha256']
            or obs_protocol['observation_generation']!=OBS.OBSERVATION_GENERATION):
        raise ValueError('Exact new grazing-gated fresh17c CAL46 observations required')
    parent=C.read(PROFILE/'PROTOCOL.json');seal=C.read(PROFILE/'INPUT_SEAL.json');journal=PROFILE/'INPUTS.jsonl.gz'
    if (seal['rows']!=238 or C.sha(journal)!=seal['input_sha256']
            or C.sha(PROFILE/'PROTOCOL.json')!=seal['protocol_sha256']):
        raise ValueError('Sealed grazing states/retainedvisible/filteredboundary inputs required')
    for p in (obs_protocol,parent):
        for key in ('source_sha256','input_sha256','protected_user_sha256'):OLD.E._verify_sources(p[key])
    rows=list(C.rows(obs_journal));profiles=list(islice(C.rows(journal),46));ids=[r['id']for r in rows]
    observer_tasks=list(islice(C.rows(OBS.OUTPUT/'INPUTS.jsonl.gz'),46))
    if (len(rows)!=46 or len(profiles)!=46 or len(set(ids))!=46
            or any(r['partition']!='CAL'for r in rows+profiles)
            or ids!=obs_seal['ordered_ids']or ids!=seal['ordered_ids'][:46]
            or ids!=[r['id']for r in profiles]or ids!=[r['id']for r in observer_tasks]
            or any(t['partition']!='CAL'for t in observer_tasks)):
        raise ValueError('Exact unique CAL46 prefix only; no TEST/R rows decoded')
    if parent['protected_user_sha256']!=obs_protocol['protected_user_sha256']:
        raise ValueError('Protected user inventories differ')
    sources={**parent['source_sha256'],**obs_protocol['source_sha256']}
    for module in (sys.modules[__name__],C,OLD,OBS,OLD.ALIGN,OLD.N,OLD.MIX,OLD.MIX.M,OLD.FACTOR):
        path=Path(module.__file__).resolve();sources[str(path.relative_to(C.ROOT))]=C.sha(path)
    paths=(journal,PROFILE/'PROTOCOL.json',PROFILE/'INPUT_SEAL.json',obs_journal,
        OBS.OUTPUT/'PROTOCOL.json',OBS.OUTPUT/'CAL_OBSERVATION_SEAL.json',
        OBS.OUTPUT/'PREPARATION_SEAL.json',OBS.OUTPUT/'INPUTS.jsonl.gz',LABELS)
    inputs={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths}
    output.mkdir(parents=True)
    protocol=dict(schema='GRAZING_CAL_ACTUAL_NMS_MIXTURE_PREFIT_PROTOCOL_V18B',recipe=RECIPE,
        observation_generation=OBS.OBSERVATION_GENERATION,
        prepared_before_CAL_reference_decode_UTC=datetime.now(timezone.utc).isoformat(),
        ids=ids,rows=46,source_sha256=sources,input_sha256=inputs,
        protected_user_sha256=parent['protected_user_sha256'],CAL_label_path=str(LABELS.relative_to(C.ROOT)),
        source_binding_scope='conservative inherited source union plus actual new profile/fitter/auth modules',
        other_partition_prediction_rows_decoded=0,TEST_or_remaining_label_paths=False,
        boundary_profile='grazing INPUTS.boundary_merged; old boundary_all not decoded',
        new_pose_fits=0,new_RGB_or_model_forwards=0)
    C.write(output/'PROTOCOL.json',protocol)
    for relative,digest in sources.items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(C.ROOT/relative,target)
        if C.sha(target)!=digest:raise ValueError('Source snapshot changed')
    prepared=[]
    for task,row,observer_task in zip(profiles,rows,observer_tasks):
        spec=task['spec'];family=task['families']['ORIGINAL_WD'];g=row['initial_geometry']
        if (row['states']!=task['states']or row['states']!=spec['states']
                or row['visible_measurements']!=spec['visible']or row['domains']!=spec['domains']
                or row['u_regions']!=task['u_regions']or row['image_sha256']!=task['image_sha256']
                or row['new_grazing_U_native_ids']!=task['grazing_adaptation']['geometry_gate']['new_U_native_ids']
                or g['native_xyz']!=family['native_xyz']or g['physical_registration_Q']!=family['physical_registration_Q']
                or g['K']!=spec['K']or g['image_hw']!=spec['image_hw']
                or g['initial_self_hidden']!=task['initial_self_hidden']
                or row['observation_generation']!=OBS.OBSERVATION_GENERATION):
            raise ValueError('Exact grazing-profile geometry/state/q/covariance/regions/image binding required')
        if (observer_task['sealed_grazing_filtered_boundary_sha256']!=OLD.E._digest(task['boundary_merged'])
                or row['sealed_grazing_adaptation_sha256']!=observer_task['sealed_grazing_adaptation_sha256']
                or row['sealed_grazing_adaptation_sha256']!=OLD.E._digest(task['grazing_adaptation'])):
            raise ValueError('Fresh observer must bind the same filtered boundary_merged profile')
        base=OLD.N.prepare_native_edge_problem(family['native_xyz'],spec['K'],spec['states'],spec['visible'],
            spec['domains'],spec['image_hw'],family['coarse_pose'],config={**spec['config'],'prior_weight':0.},
            boundary_observations=task['boundary_merged'],u_regions=task['u_regions'],
            edge_bundle=dict(edges=[],replaces_normal_factor_refs=[]))
        authenticated=OLD.MIX.authenticate_image_chain_groups(base,row['chain_observation'],
            initial_self_hidden=g['initial_self_hidden'],image_binding=row['chain_observation']['image_binding'],
            normal_variance_component_px2=0.,angular_variance_component_sine2=0.,normal_angle_correlation=0.,
            noise_provenance=dict(fit_partition='CAL',observation_family=OLD.MIX.NOISE_OBSERVATION_FAMILY,
                observation_generation=OBS.OBSERVATION_GENERATION,purpose='prefit authentication only'))
        count,alternatives=OLD._observation_inventory(row['chain_observation'])
        prepared.append(dict(id=row['id'],partition='CAL',session=row['session'],
            initial_geometry=deepcopy(g),new_grazing_U_native_ids=row['new_grazing_U_native_ids'],
            all_NMS_physical_edges=count,all_NMS_alternatives=alternatives,
            authenticated_group_bundle=authenticated))
    C.save_rows(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz',prepared)
    for values in (sources,inputs,parent['protected_user_sha256']):OLD.E._verify_sources(values)
    C.write(output/'PREPARATION_SEAL.json',dict(protocol_sha256=C.sha(output/'PROTOCOL.json'),
        CAL_inputs_sha256=C.sha(output/'CAL_AUTHENTICATED_INPUTS.jsonl.gz'),rows=46,
        source_sha256=sources,observation_generation=OBS.OBSERVATION_GENERATION,
        before_CAL_reference_decode=True,no_TEST_or_remaining_label_decode=True,
        all_NMS_edge_count=sum(r['all_NMS_physical_edges']for r in prepared),
        all_NMS_alternative_count=sum(r['all_NMS_alternatives']for r in prepared),
        runtime_applicable_group_count=sum(len(r['authenticated_group_bundle']['groups'])for r in prepared)))
    print(dict(prepared_CAL=46,grazing_U=sum(len(r['new_grazing_U_native_ids'])for r in prepared),
        applicable_groups=sum(len(r['authenticated_group_bundle']['groups'])for r in prepared)),flush=True)


def fit(output=OUTPUT):
    """Same immutable numerical fitter, isolated new preparation and schema."""
    output=Path(output).resolve()
    runner=types.FunctionType(OLD.fit.__code__,{**OLD.fit.__globals__,
        'RECIPE':RECIPE,'SCHEMA':SCHEMA,'fit_native_edge_mixture_model':fit_native_edge_mixture_model,
        '_verify_preparation':_verify_preparation},OLD.fit.__name__,OLD.fit.__defaults__)
    return runner(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('prepare','fit'))
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    (prepare if args.mode=='prepare'else fit)(args.output)
