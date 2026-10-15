"""Positive-input V32 controller; e is a separate parent-only comparison.

No historical evaluator run/reference-start/refiner wrapper is called.  The
worker accepts the seven-key positive packet only and constructs every seed
from its independent first pose and admitted B/I archives.  Runtime stages
remain separately ROOT-gated; importing this module does not read data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from . import common as C
from . import clean_fresh_BI_runtime_v4_32 as CLEAN
from . import clean_fresh_BI_input_projection_v4_32 as PROJECT
from . import solve_clean_fresh_BI_poses_v4_32 as SOLVER
from . import prepare_geometry_guided_native_evidence_v4_32 as FACTORY
from . import prepare_geometry_guided_native_metadata_v4_32 as META
from . import geometry_guided_edge_problem_v4_32 as G
from . import geometry_guided_edge_authentication_v4_32 as AUTH
from . import geometry_guided_edge_noise_calibration_v4_32 as NOISE
from . import calibrate_geometry_guided_edge_noise_v4_32 as CAL
from . import rerank_initial_only_observed_ray_population_v4_32 as OBS
from . import calibrate_bounded_o_localization_v4_21 as OCAL
from . import calibrate_coarse_visibility_prior_v4_20 as SCAL
from . import bounded_o_localization_prior_v4_21 as O
from . import bounded_coarse_visibility_prior_v4_20 as S


B = OBS.B
OUTPUT = B / 'clean_fresh_BI_pose_eval_v4_32_r2'
PLAN_OUTPUT = B / 'clean_fresh_BI_pose_eval_plan_v4_32_r2'
TASKS = META.TASKS
ARMS = ('SAME_FROZEN_E', *SOLVER.ARMS)
POLICY_ARMS = SOLVER.POLICY_ARMS
FAMILIES = SOLVER.FAMILIES
SOLVER_SHA256 = 'cf1256956ec0da7784c98d0b2a860af4105a1a1ba557b0f5fcef85407f0c3237'
PROJECTOR_PEER_PATH = B / 'clean_fresh_BI_input_projection_peer_review_v4_32/SOURCE_REVIEW_V1.json'
PROJECTOR_PEER_SHA256 = '53d8e1a50b52bbd3de2b0af828de5aa71c8784c87810784a7d1f8d54dac0e9cb'
PARTITIONS = dict(CAL=46, TEST=49, REMAINDER=143)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _line(value):
    return json.dumps(C.finite(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n'


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(C.ROOT)), sha256=C.sha(path))


def _verify(*maps):
    OBS._verify(*maps)


def _actual_pose(family, R, t):
    """Explicit native translation and geometric center, including shifted XYZ."""
    xyz = np.asarray(family['native_xyz'], float)
    Q = np.asarray(family['physical_registration_Q'], float)
    rotation = FACTORY.FRESH._rotation(R, 'Selected native rotation')
    translation = FACTORY.FRESH._array(t, (3,), 'Selected native translation')
    center = rotation@xyz.mean(axis=0)+translation
    return dict(R_cf=rotation.tolist(), R_physical=(rotation@Q).tolist(),
        native_t=translation.tolist(), centroid=center.tolist(),
        model_center_camera_xyz=center.tolist(), cf_extents=np.ptp(xyz, axis=0).tolist(),
        physical_registration_Q=Q.tolist(), WD_family=family['name'],
        centroid_semantics='GEOMETRIC_MODEL_CENTER_NOT_NATIVE_ORIGIN_TRANSLATION')


def _readout(packet, clean, core, result):
    """No raw task or old readout is available at this boundary."""
    source = packet['clean_input']
    states, hidden = source['states'], source['initial_self_hidden']
    points, sources = [None]*8, [None]*8
    regions = deepcopy(packet['physics']['u_regions'])
    family = clean['families'][result['WD_family']]
    R, t = np.asarray(result['R']), np.asarray(result['t'])
    xyz, K = np.asarray(family['native_xyz']), np.asarray(source['K'])
    camera = xyz@R.T+t
    hom = camera@K.T
    model = hom[:, :2]/hom[:, 2:3]
    archive = {record['native_corner']:record for record in source['visible_measurements']}
    complete = {record['native_corner'] for record in core.complete}
    owners = set(core.plan['new_owner_ids'])
    for k, state in enumerate(states):
        if hidden[k] or state == 'S':
            sources[k] = 'MODEL_PNP_PROJECTION_ONLY_NO_S_IMAGE_TERM'
            # This is an explicitly named model projection, never observed S.
            points[k] = model[k].tolist()
        elif state in ('U', 'O', 'T'):
            sources[k] = 'APPROXIMATE_REGION_ONLY_NO_POINT_CORRESPONDENCE'
        elif k in owners:
            sources[k] = 'ACTUAL_NATIVE_EDGE_CONSTRAINTS_NO_PRECISE_2D_KP'
        elif k in complete:
            points[k] = deepcopy(archive[k]['xy'])
            sources[k] = archive[k]['source']
        else:
            sources[k] = 'ARCHIVED_PARTIAL_NORMAL_AND_UNCERTAIN_TANGENT_NO_PRECISE_KP'
    return dict(states=list(states), reported_points=points, point_sources=sources,
        u_regions=regions, S_model_projection_ids=[k for k,h in enumerate(hidden) if h],
        S_image_raw_2D_or_region_terms=0, U_range_only=True,
        new_full_2D_corner_measurements=0, current_active_owner_ids=sorted(owners),
        current_complete_BI_ids=sorted(complete), used_in_accepted_pose=bool(result['accepted']),
        continuous_SE3_search_performed=True, OT_solved=False,
        OT_unique_solution_claimed=False, geometric_global_uniqueness_proved=False,
        accepted_fresh_or_refined_pose=bool(result['accepted']))


def build_clean_native_cores(packet):
    """Shared smoke/worker construction; fresh PnP, no continuous optimizer."""
    if not isinstance(packet, dict) or set(packet) != PROJECT.WORKER_KEYS:
        raise ValueError('Only the exact positive seven-key V32 worker packet is accepted')
    if packet['schema'] != PROJECT.WORKER_SCHEMA:
        raise ValueError('New clean V32 worker schema required')
    clean_input = packet['clean_input']
    if (clean_input['initial_source_binding'] != packet['source_contract']['initial_source_binding']
            or packet['identifiers'] != clean_input['identifiers']):
        raise ValueError('Same caller-verified positive source/identity required')
    clean = CLEAN.prepare_clean_fresh_BI_task(**clean_input)
    if not clean['ready']:
        return clean, None
    return clean, build_native_cores(packet, clean)


def build_native_cores(packet, clean):
    """Build typed cores from the original BI guard archives, never old poses."""
    CLEAN._validate(clean)
    if not clean['ready']:
        return None
    clean_input = packet['clean_input']
    if packet['source_contract'].get('ready') is not True:
        raise ValueError('Ready clean construction differs from parent positive projection')
    physics = packet['physics'];contract = packet['source_contract']
    cores = {}
    for policy in G.SOURCE_POLICIES:
        available = contract['model_availability'][policy]
        parameters = contract['tangent_parameters'][policy]
        if available is True:
            PROJECT._runtime_export(parameters, policy, tangent=True)
            gamma = parameters['gamma']
        elif available is False and parameters is None:
            gamma = contract['unavailable_policy_tangent_gamma']
            if gamma != 0. or contract['old_model_substituted'] is not False:
                raise ValueError('Explicit unavailable new model uses gamma0 without old substitution')
        else:
            raise ValueError('Boolean per-policy new model availability required')
        cores[policy] = {}
        for name in FAMILIES:
            # Use full original admitted BI archives for original hardguards.
            # clean.spec.visible is only the minimal SQPNP archive.
            base = FACTORY.native_baseline(family=clean['families'][name], K=clean_input['K'],
                states=clean_input['states'], visible_measurements=clean_input['visible_measurements'],
                domains=physics['domains'], image_hw=physics['image_hw'], config=physics['config'],
                boundary_merged=physics['boundary_merged'], u_regions=physics['u_regions'],
                initial_self_hidden=physics['initial_self_hidden'],
                initial_source_binding=clean_input['initial_source_binding'])
            bundle = packet['typed_bundles'][policy][name]
            if not available and bundle['groups']:
                raise ValueError('Unavailable new32 policy cannot retain an old line factor')
            cores[policy][name] = G.GeometryGuidedEdgeProblem(base, bundle,
                source_policy=policy, tangent_gamma=gamma)
    return cores


def solve_worker_packet(packet):
    """Pure numeric path; no files, images, references or e comparison argument."""
    clean, cores = build_clean_native_cores(packet)
    clean_input = packet['clean_input']
    if not clean['ready']:
        # Frozen solver returns before examining any core/regularizer payload.
        result = SOLVER.solve_clean_fresh_BI_poses(clean, None, o_regions=None,
            o_parameters=None, original_S_prepared=None, outline_observation=None,
            outline_configuration=None)
        for arm in SOLVER.ARMS:
            result['arms'][arm].update(actual_pose=None, readout=dict(
                states=list(clean_input['states']), reported_points=[None]*8,
                point_sources=['NO_POSE_INVALID_INDEPENDENT_INITIAL_COARSE']*8,
                S_image_raw_2D_or_region_terms=0, U_range_only=True,
                continuous_SE3_search_performed=False,
                OT_solved=False, OT_unique_solution_claimed=False))
        result.update(source_contract=deepcopy(packet['source_contract']),
            states=list(clean_input['states']),initial_self_hidden=list(clean_input['initial_self_hidden']),
            fresh_seed_audit={},parent_only_e_comparison_not_supplied_to_worker=True)
        return result
    contract = packet['source_contract']
    result = SOLVER.solve_clean_fresh_BI_poses(clean, cores, **packet['regularizers'])
    for policy, arms in POLICY_ARMS.items():
        for arm in arms:
            entry = result['arms'][arm]
            family = clean['families'][entry['WD_family']]
            entry['actual_pose'] = _actual_pose(family, entry['R'], entry['t'])
            entry['readout'] = _readout(packet, clean, cores[policy][entry['WD_family']], entry)
            entry['new_noise_availability'] = contract['model_availability'][policy]
    result.update(source_contract=deepcopy(contract), states=list(clean_input['states']),
        initial_self_hidden=list(clean_input['initial_self_hidden']),
        fresh_seed_audit={name:dict(input_binding_sha256=bank['input_binding_sha256'],
            proposal_count=len(bank['proposals']), old_pose_bank_inputs=0)
            for name,bank in clean['fresh_seed_banks'].items()},
        parent_only_e_comparison_not_supplied_to_worker=True)
    return result


def _checked_prior(folder, kind):
    folder = Path(folder)
    seal = C.read(folder/'MODEL_SEAL.json');model = C.read(folder/'MODEL.json')
    if seal['fit_partition'] != 'CAL' or C.sha(folder/'MODEL.json') != seal['model_sha256']:
        raise ValueError('Frozen CAL-only bounded prior required')
    for filename, key in (('CAL_LOGO_REPORT.json','LOGO_sha256'),
                          ('PREPARATION_SEAL.json','preparation_seal_sha256')):
        if C.sha(folder/filename) != seal[key]:
            raise ValueError('Frozen bounded prior seal changed')
    _verify(seal['source_sha256'], seal['input_sha256'])
    if kind == 'O': OCAL.prior_parameters(model)
    else: S.validate_model(model)
    return model


def _model_record(value, policy, *, tangent):
    if value.get('model') is None and value.get('status','').startswith('MODEL_UNAVAILABLE'):
        if value.get('old_model_substituted') is not False:
            raise ValueError('Unavailable new population may not substitute old model')
        return False
    model = value.get('model', value)
    NOISE._require_model(model,policy,tangent=tangent)
    if model['model_digest_sha256'] != NOISE.PARENT._digest(
            {key:item for key,item in model.items() if key != 'model_digest_sha256'}):
        raise ValueError('Exact new32 model digest required')
    if tangent: NOISE.tangent_variance_parameters(model,source_policy=policy)
    else: NOISE.mixture_noise_parameters(model,source_policy=policy)
    return True


def _validate_model_inventory(seal, full, report):
    """Four full files plus sixteen LOGO entries inside the report, not20files."""
    expected = {policy+'/'+name for policy in G.SOURCE_POLICIES
        for name in ('MODEL.json','TANGENT_MODEL.json')}
    if {path for path in seal['assets'] if Path(path).name in ('MODEL.json','TANGENT_MODEL.json')} != expected:
        raise ValueError('Exactly four full model/status files required')
    if set(report) != set(G.SOURCE_POLICIES) or set(full) != set(G.SOURCE_POLICIES):
        raise ValueError('Both typed source-policy inventories required')
    count = 0
    for policy in G.SOURCE_POLICIES:
        for kind in ('line','tangent'):
            _model_record(full[policy][kind],policy,tangent=kind=='tangent');count += 1
        entry = report[policy];folds = entry['LOGO_folds']
        if (entry['inventory46'] != 46 or entry['compatible44'] != 44
                or len(folds) != 4 or len({fold['heldout_session'] for fold in folds}) != 4
                or entry['old_model_substituted'] is not False):
            raise ValueError('Four separately held-out CAL sessions per source policy required')
        for fold in folds:
            for kind in ('line','tangent'):
                _model_record(fold[kind],policy,tangent=kind=='tangent');count += 1
    if count != 20: raise ValueError('Exactly twenty model/status entries required')
    return dict(full_model_files=4,nested_LOGO_model_entries=16,total_models=20)


def _models(expected_central_plan_sha256):
    # This verifier reads only frozen metadata/source/model bytes, never labels.
    protocol = CAL._verify_preparation(CAL.OUTPUT,
        expected_central_plan_sha256=expected_central_plan_sha256)
    folder = CAL.OUTPUT;seal = C.read(folder/'MODEL_SEAL.json')
    if (seal['fit_partition'] != 'CAL' or seal['CAL_inventory'] != 46
            or seal['compatible_inventory'] != 44
            or seal['selection_profile'] != G.SELECTION_PROFILE
            or seal['observation_generation'] != G.OBSERVATION_FAMILY
            or seal['noise_population'] != G.POPULATION
            or seal['raw_observation_generation'] != G.RAW_OBSERVATION_GENERATION
            or seal['preparation_seal_sha256'] != C.sha(folder/'PREPARATION_SEAL.json')):
        raise ValueError('New separately sealed V32 CAL46/fixed44 model population required')
    _verify(seal['source_sha256'],seal['input_sha256'])
    for relative, digest in seal['assets'].items():
        if C.sha(folder/relative) != digest: raise ValueError('New CAL32 fitted asset changed')
    full = {policy:dict(line=C.read(folder/policy/'MODEL.json'),
        tangent=C.read(folder/policy/'TANGENT_MODEL.json')) for policy in G.SOURCE_POLICIES}
    _validate_model_inventory(seal,full,C.read(folder/'CAL_LOGO_REPORT.json'))
    result = dict(O=_checked_prior(OCAL.OUTPUT,'O'), S=_checked_prior(SCAL.OUTPUT,'S'), policies={})
    for policy in G.SOURCE_POLICIES:
        line, tangent = full[policy]['line'], full[policy]['tangent']
        unavailable = any(value.get('model') is None and value.get('status','').startswith('MODEL_UNAVAILABLE')
            for value in (line,tangent))
        if unavailable:
            result['policies'][policy] = dict(available=False, old_model_substituted=False,
                noise=None, tangent=None)
        else:
            result['policies'][policy] = dict(available=True,
                noise=NOISE.mixture_noise_parameters(line,source_policy=policy),
                tangent=NOISE.tangent_variance_parameters(tangent,source_policy=policy),
                old_model_substituted=False)
    return result, protocol, seal


def _regularizers(task, projected, models):
    source = projected['clean_input'];physics = projected['physics'];domains = []
    for archived in task['O_localization_abstention']['records']:
        k = G._id(archived['native_corner'])
        if source['initial_self_hidden'][k] or source['states'][k] != 'O':
            raise ValueError('O state gate before original localized domain fields')
        original = archived['original_domain']
        if original['binding']['calibration_model_sha256'] != models['O']['source_O_model_digest']:
            raise ValueError('O frame floor must bind the exact original localization model')
        replacement = next(d for d in physics['domains'] if d['native_corner'] == k)
        if G._digest(replacement) != G._digest(archived['replacement_domain']):
            raise ValueError('Exact archived closed-canvas O replacement required')
        domains.append(original)
    coarse = source['initial_coarse_pose']
    return dict(o_regions=O.prepare_o_localization_regions(domains,source['states'],source['initial_self_hidden']),
        o_parameters=OCAL.prior_parameters(models['O']),
        original_S_prepared=S.prepare_bounded_coarse_visibility_prior(source['native_xyz'],
            coarse['R'],coarse['t'],source['initial_self_hidden'],models['S'],factor_weight=1.),
        outline_observation=deepcopy(task['outline']),
        outline_configuration=deepcopy(task['outline']['configuration']))


def _prepare_packet(task, row, binding, models):
    projected = PROJECT.project_clean_input(task,row,initial_source_binding=binding)
    if not projected['ready']:
        return PROJECT.assemble_worker_input(projected,typed_bundles=None,regularizers=None,
            model_availability=None,tangent_parameters=None,line_noise_provenance=None)
    source, physics = projected['clean_input'], projected['physics']
    families = CLEAN._families(np.asarray(source['native_xyz']), source['initial_coarse_pose'],
        np.asarray(source['physical_registration_Q']))  # Geometry only, no PnP.
    bundles, availability, tangents, provenance = {}, {}, {}, {}
    for policy in G.SOURCE_POLICIES:
        parameters = models['policies'][policy];availability[policy] = parameters['available']
        tangents[policy] = parameters['tangent']
        provenance[policy] = None if not availability[policy] else parameters['noise']['noise_provenance']
        bundles[policy] = {}
        for name in FAMILIES:
            base = FACTORY.native_baseline(family=families[name],K=source['K'],states=source['states'],
                visible_measurements=source['visible_measurements'],domains=physics['domains'],
                image_hw=physics['image_hw'],config=physics['config'],boundary_merged=physics['boundary_merged'],
                u_regions=physics['u_regions'],initial_self_hidden=physics['initial_self_hidden'],
                initial_source_binding=source['initial_source_binding'])
            auth = projected['observation_auth_arguments']
            if availability[policy]:
                bundles[policy][name] = AUTH.authenticate_geometry_guided_edges(base,row['observation'],
                    initial_geometry=auth['initial_geometry'],image_binding=auth['image_binding'],
                    expected_source_policy=policy,**parameters['noise'])
            else:
                bundles[policy][name] = G.empty_geometry_guided_evidence(base,
                    initial_geometry=auth['initial_geometry'],image_binding=auth['image_binding'],
                    association_configuration=row['observation']['association_configuration'],
                    reason='NEW_V32_POLICY_NOISE_UNAVAILABLE_NO_OLD_MODEL_SUBSTITUTION')
    return PROJECT.assemble_worker_input(projected,typed_bundles=bundles,
        regularizers=_regularizers(task,projected,models),model_availability=availability,
        tangent_parameters=tangents,line_noise_provenance=provenance)


def _observation_metadata(expected_sha):
    protocol = OBS._load_rest(OBS.OUTPUT,expected_sha)
    final = C.read(OBS.OUTPUT/'FINAL_PROTOCOL.json')
    seal = C.read(OBS.OUTPUT/'OBSERVATION_SEAL.json')
    completion = C.read(OBS.OUTPUT/'FINAL_COMPLETION_RECEIPT_V32.json')
    ids = protocol['ids'];entries = []
    if (len(ids) != 238 or len(set(ids)) != 238 or final['ids'] != ids
            or final['central_plan_sha256'] != expected_sha or seal['ordered_ids'] != ids
            or seal['rows'] != 238 or seal['protocol_sha256'] != C.sha(OBS.OUTPUT/'FINAL_PROTOCOL.json')
            or seal['observations_sha256'] != C.sha(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz')
            or completion['observation_seal'] != _binding(OBS.OUTPUT/'OBSERVATION_SEAL.json')
            or completion['final_protocol'] != _binding(OBS.OUTPUT/'FINAL_PROTOCOL.json')
            or seal['selection_profile'] != G.SELECTION_PROFILE
            or seal['canonical_validation'] != dict(rows=238,canonical_row_digests_verified=238,
                missing_duplicate_or_extra_rows=0)):
        raise ValueError('Completed new32 ordered/canonical all238 observation required')
    for part,count in (('CAL',46),('REST',192)):
        partition = C.read(OBS.OUTPUT/(part+'_OBSERVATION_SEAL.json'))
        index = C.read(OBS.OUTPUT/(part+'_ROW_INDEX.json'))['entries']
        if (len(index) != count or partition['rows'] != count
                or partition['ordered_ids'] != [entry['id'] for entry in index]
                or partition['row_index_sha256'] != C.sha(OBS.OUTPUT/(part+'_ROW_INDEX.json'))
                or seal['partition_seals'][part] != C.sha(OBS.OUTPUT/(part+'_OBSERVATION_SEAL.json'))):
            raise ValueError('Both exact current partition inventories/seals required')
        entries.extend(index)
    if ([entry['id'] for entry in entries] != ids
            or Counter(entry['partition'] for entry in entries) != Counter(PARTITIONS)):
        raise ValueError('No missing/duplicate/severe rows or partition substitution')
    if final['original_CAL_file_sha256'] != OBS._cal_immutable(OBS.OUTPUT):
        raise ValueError('Original CAL53 bytes changed')
    _verify(*(final[key] for key in ('source_sha256','input_sha256','protected_user_sha256',
        'original_CAL_file_sha256','original_parent_immutable_file_sha256')))
    return final,entries


def _source_map():
    modules = (C,CLEAN,PROJECT,SOLVER,FACTORY,META,G,AUTH,NOISE,CAL,OBS,OCAL,SCAL,O,S,
        CLEAN.FRESH,CLEAN.REFINE,G.N,G.J,G.EDGE,SOLVER.BOUNDED,SOLVER.CONTAIN,SOLVER.SCALAR)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths += [Path(__file__).resolve(),Path(__file__).with_name('test_evaluate_clean_fresh_BI_poses_v4_32_r2.py').resolve(),
        Path(PROJECT.__file__).with_name('test_clean_fresh_BI_input_projection_v4_32.py').resolve()]
    return {str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}


def _peer(binding):
    from . import declare_initial_only_observed_ray_v4_32 as DECLARE
    return DECLARE._peer(binding, (PROJECT,__import__(__name__,fromlist=['*'])))


def _stage_inputs(expected_sha, expected_model_seal_sha):
    if not _sha(expected_sha) or not _sha(expected_model_seal_sha):
        raise ValueError('Explicit central and newly fitted model seal SHA required')
    central = OBS._central(expected_sha)
    from . import declare_initial_only_observed_ray_v4_32 as DECLARE
    DECLARE._peer(dict(path=str(PROJECTOR_PEER_PATH.relative_to(C.ROOT)),
        sha256=PROJECTOR_PEER_SHA256),(PROJECT,))
    if C.sha(SOLVER.__file__) != SOLVER_SHA256:
        raise ValueError('Frozen independently reviewed fresh solver changed')
    if C.sha(CAL.OUTPUT/'MODEL_SEAL.json') != expected_model_seal_sha:
        raise ValueError('Caller-pinned new32 model seal changed before decode')
    models,cal_protocol,modelseal = _models(expected_sha)
    observation,entries = _observation_metadata(expected_sha)
    task_protocol = C.read(TASKS/'PROTOCOL.json');taskseal = C.read(TASKS/'INPUT_SEAL.json')
    if (taskseal['rows'] != 238 or taskseal['ordered_ids'] != observation['ids']
            or task_protocol['ids'] != observation['ids']
            or taskseal['protocol_sha256'] != C.sha(TASKS/'PROTOCOL.json')
            or taskseal['input_sha256'] != C.sha(TASKS/'INPUTS.jsonl.gz')):
        raise ValueError('Exact original sealed parent238 task archive required')
    initial_binding,proof_sources = META._initial_source_closure()
    sources = OBS._union(*[value['source_sha256'] for value in
        (central,observation,cal_protocol,task_protocol)],proof_sources,_source_map())
    inputs = OBS._union(*[value['input_sha256'] for value in
        (central,observation,cal_protocol,task_protocol)])
    paths = [OBS.DECLARATION,PROJECTOR_PEER_PATH,OBS.OUTPUT/'FINAL_PROTOCOL.json',OBS.OUTPUT/'OBSERVATION_SEAL.json',
        OBS.OUTPUT/'FINAL_COMPLETION_RECEIPT_V32.json',OBS.OUTPUT/'OBSERVATIONS.jsonl.gz',
        CAL.OUTPUT/'MODEL_SEAL.json',CAL.OUTPUT/'PREPARATION_SEAL.json',
        TASKS/'PROTOCOL.json',TASKS/'INPUT_SEAL.json',TASKS/'INPUTS.jsonl.gz']
    paths += [CAL.OUTPUT/relative for relative in modelseal['assets']]
    paths += [folder/name for folder in (OCAL.OUTPUT,SCAL.OUTPUT) for name in
        ('PROTOCOL.json','PREPARATION_SEAL.json','MODEL.json','MODEL_SEAL.json','CAL_LOGO_REPORT.json')]
    inputs = OBS._union(inputs,{str(path.relative_to(C.ROOT)):C.sha(path) for path in paths})
    protected = OBS._union(*[value['protected_user_sha256'] for value in
        (central,observation,cal_protocol,task_protocol)])
    if len(protected) != 6: raise ValueError('Exactly original protected6 files required')
    _verify(sources,inputs,protected)
    return dict(models=models,entries=entries,ids=observation['ids'],initial_binding=initial_binding,
        source_sha256=sources,input_sha256=inputs,protected_user_sha256=protected,
        original_CAL_file_sha256=observation['original_CAL_file_sha256'])


def plan(*,root_metadata_GO=False,expected_central_plan_sha256=None,
        expected_model_seal_sha256=None,controller_peer_binding=None,output=PLAN_OUTPUT):
    if root_metadata_GO is not True:
        raise PermissionError('Root separate execution plan GO before IO')
    if not _sha(expected_central_plan_sha256) or not _sha(expected_model_seal_sha256) or controller_peer_binding is None:
        raise ValueError('Explicit immutable central/model/peer bindings before IO')
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError('Preserve every separate evaluation plan')
    peerpath = _peer(controller_peer_binding)
    inputs = _stage_inputs(expected_central_plan_sha256,expected_model_seal_sha256)
    inputs['input_sha256'] = OBS._union(inputs['input_sha256'],{str(peerpath.relative_to(C.ROOT)):C.sha(peerpath)})
    value = {key:inputs[key] for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')}
    value.update(schema='CLEAN_FRESH_BI_SEPARATE_EXECUTION_PLAN_V32',UTC=datetime.now(timezone.utc).isoformat(),
        central_plan_sha256=expected_central_plan_sha256,new_model_seal_sha256=expected_model_seal_sha256,
        controller_peer_binding=controller_peer_binding,ordered_ids=inputs['ids'],arms=list(ARMS),partitions=PARTITIONS,
        positive_worker_schema=PROJECT.WORKER_SCHEMA,worker_keys=sorted(PROJECT.WORKER_KEYS),
        parent_e_comparison_separate_from_numeric_inputs=True,old_pose_bank_start_or_fallback_inputs=0,
        full_pose_TP_topnormal_prior_weights=0.,old_O_S_and_optional_one_sided_C_math_unchanged=True,
        actual_PnP_posefit_or_prediction_in_plan=0,predictions_before_reference_or_score=True,
        invalid_coarse_row_retained_as_no_pose=True,default='SAME_FROZEN_E unchanged')
    output.mkdir(parents=True);C.write(output/'PREDECLARATION.json',value)
    for relative,digest in value['source_sha256'].items():
        target=output/'code_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((C.ROOT/relative).read_bytes())
        if C.sha(target)!=digest:raise ValueError('Current execution source snapshot changed')
    C.write(output/'PLAN_SEAL.json',dict(plan_sha256=C.sha(output/'PREDECLARATION.json'),source_sha256=value['source_sha256']))
    return _binding(output/'PREDECLARATION.json')


def _load_plan(plan_folder):
    folder=Path(plan_folder);value=C.read(folder/'PREDECLARATION.json');seal=C.read(folder/'PLAN_SEAL.json')
    if C.sha(folder/'PREDECLARATION.json')!=seal['plan_sha256'] or value['arms']!=list(ARMS):
        raise ValueError('Exact separate five-arm execution plan required')
    _verify(*(value[key] for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')))
    if C.sha(OBS.DECLARATION)!=value['central_plan_sha256']:
        raise ValueError('Original central declaration must remain immutable')
    return value


def _comparison_row(task):
    """Legacy e uses native t in centroid; current centered XYZ makes it center.

    This metadata check belongs only to the parent comparison archive.  The
    prediction bytes are preserved and never influence a new numerical input.
    """
    xyz=np.asarray(task['families']['ORIGINAL_WD']['native_xyz'],float)
    if xyz.shape!=(8,3) or not np.isfinite(xyz).all() or not np.allclose(xyz.mean(axis=0),0.,atol=1e-12,rtol=0):
        raise ValueError('Legacy e centroid convention requires the verified centered current body')
    return dict(id=task['id'],partition=task['partition'],prediction=deepcopy(task['baseline']),
        legacy_e_pose_semantics=dict(centroid_field='NATIVE_T_EQUAL_MODEL_CENTER_FOR_CENTERED_XYZ',
            original_native_xyz_zero_mean_verified=True,comparison_only=True,numeric_worker_exposure=False))


def _prepared_rows(tasks,observations,entries,ids,binding,models,comparison_stream,records):
    seen=[]
    for task,row,entry in zip_longest(tasks,observations,entries):
        if task is None or row is None or entry is None:
            raise ValueError('Exact all238 task/observation/index counts required')
        index=len(seen)
        if (index>=238 or task['id']!=ids[index] or row['id']!=ids[index] or entry['id']!=ids[index]
                or task['partition']!=row['partition'] or entry['partition']!=row['partition']
                or OBS.D.R._canonical_sha(row)!=entry['canonical_row_sha256']):
            raise ValueError('Exact same ordered canonical task/observation join required')
        OBS._validate_selection_counts(row)
        packet=_prepare_packet(task,row,binding,models)
        # This prediction archive is parent-only; no score/reference is read.
        comparison=_comparison_row(task)
        comparison_stream.write(_line(comparison))
        records.append(dict(id=task['id'],partition=task['partition'],ready=packet['source_contract']['ready'],
            positive_packet_sha256=G._digest(C.finite(packet))))
        seen.append(task['id'])
        del task,row,entry,comparison
        yield packet
    if seen!=ids or Counter(record['partition'] for record in records)!=Counter(PARTITIONS):
        raise ValueError('No row dropping, severe frames or population routing')


def prepare(*,root_metadata_GO=False,plan_folder=PLAN_OUTPUT,output=OUTPUT):
    if root_metadata_GO is not True:raise PermissionError('Root positive-input preparation GO before IO')
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve every positive preparation attempt')
    declared=_load_plan(plan_folder)
    inputs=_stage_inputs(declared['central_plan_sha256'],declared['new_model_seal_sha256'])
    for key in ('source_sha256','protected_user_sha256','original_CAL_file_sha256'):
        if inputs[key]!=declared[key]:raise ValueError('Separate plan source/protected/CAL53 binding changed')
    output.mkdir(parents=True);records=[]
    try:
        with gzip.open(output/'E_COMPARISON.jsonl.gz','xt',encoding='utf-8',compresslevel=1) as comparison:
            OBS.save_rows_large(output/'INPUTS.jsonl.gz',_prepared_rows(C.rows(TASKS/'INPUTS.jsonl.gz'),
                C.rows(OBS.OUTPUT/'OBSERVATIONS.jsonl.gz'),inputs['entries'],inputs['ids'],
                inputs['initial_binding'],inputs['models'],comparison,records))
        if len(records)!=238:raise ValueError('Every positive/e-comparison row required')
        for mapping in (declared['source_sha256'],declared['input_sha256'],declared['protected_user_sha256']):_verify(mapping)
        protocol={**declared,'schema':'SEALED_POSITIVE_FRESH_BI_PREDICTION_INPUTS_V32',
            'execution_plan':_binding(Path(plan_folder)/'PREDECLARATION.json'),
            'ordered_ids':inputs['ids'],'records':records,'e_comparison_parent_only':True,
            'e_comparison_sha256':C.sha(output/'E_COMPARISON.jsonl.gz')}
        C.write(output/'PROTOCOL.json',protocol)
        C.write(output/'INPUT_SEAL.json',dict(rows=238,ordered_ids=inputs['ids'],
            arms=list(ARMS),arm_count=5,protocol_sha256=C.sha(output/'PROTOCOL.json'),
            input_sha256=C.sha(output/'INPUTS.jsonl.gz'),
            e_comparison_sha256=protocol['e_comparison_sha256'],before_fresh_PnP_or_prediction=True))
    except BaseException as error:
        C.write(output/'PREPARATION_FAILURE_V1.json',dict(exception=type(error).__name__,reason=str(error),
            prepared_rows=len(records),attempt_preserved=True,source_or_inputs_silently_modified=False))
        raise
    return _binding(output/'INPUT_SEAL.json')


def _load_prepared(output):
    output=Path(output);protocol=C.read(output/'PROTOCOL.json');seal=C.read(output/'INPUT_SEAL.json')
    if (seal['rows']!=238 or seal['ordered_ids']!=protocol['ordered_ids'] or seal['arms']!=list(ARMS)
            or seal['protocol_sha256']!=C.sha(output/'PROTOCOL.json')
            or seal['input_sha256']!=C.sha(output/'INPUTS.jsonl.gz')
            or seal['e_comparison_sha256']!=C.sha(output/'E_COMPARISON.jsonl.gz')):
        raise ValueError('Exact238 positive inputs and separate e prediction stream required')
    _verify(*(protocol[key] for key in ('source_sha256','input_sha256','protected_user_sha256','original_CAL_file_sha256')))
    return protocol


def _spool_path(output,identifier):
    return Path(output)/'prediction_spool'/(hashlib.sha256(identifier.encode()).hexdigest()+'.jsonl.gz')


def _worker(packet,output):
    result=solve_worker_packet(packet)
    row={**result,'id':packet['identifiers']['id'],'partition':packet['identifiers']['partition'],
        'image_sha256':packet['identifiers']['image_sha256'],'states':packet['clean_input']['states'],
        'initial_self_hidden':packet['clean_input']['initial_self_hidden']}
    if set(row['arms'])!=set(SOLVER.ARMS):raise ValueError('All four fresh results required')
    path=_spool_path(output,row['id']);OBS.save_rows_large(path,[row])
    return dict(id=row['id'],partition=row['partition'],path=str(path.relative_to(C.ROOT)),
        sha256=C.sha(path),canonical_row_sha256=OBS.D.R._canonical_sha(row),arm_count=4)


def _bounded_predict(rows,ids,output,workers):
    iterator=iter(rows);pending={};entries={};started=time.monotonic()
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'))
    def submit():
        try:packet=next(iterator)
        except StopIteration:return False
        identifier=packet['identifiers']['id']
        if identifier not in ids:raise ValueError('Unexpected positive row ID')
        pending[pool.submit(_worker,packet,output)]=identifier
        return True
    try:
        for _ in range(workers):
            if not submit():break
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                identifier=pending.pop(future);entry=future.result()
                if entry['id']!=identifier or identifier in entries:raise ValueError('Duplicate/wrong worker result')
                entries[identifier]=entry
                C.write(Path(output)/'LIVE_PROGRESS.json',dict(completed=len(entries),total=238,
                    last_id=identifier,elapsed_seconds=time.monotonic()-started,workers=workers,maximum_pending=workers))
                print(dict(completed=len(entries),total=238,id=identifier),flush=True)
                del future,entry
                submit()
    except BaseException as error:
        C.write(Path(output)/'PREDICTION_FAILURE_V1.json',dict(exception=type(error).__name__,reason=str(error),
            completed_ids=list(entries),pending_ids=list(pending.values()),attempt_preserved=True))
        for future in pending:future.cancel()
        for process in (getattr(pool,'_processes',None) or {}).values():
            if process.is_alive():process.terminate()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:pool.shutdown(wait=True)
    if set(entries)!=set(ids) or len(entries)!=238:raise ValueError('All238 results including abstentions required')
    return entries


def _merged_rows(output,ids,entries,comparisons):
    seen=[]
    for identifier,comparison in zip_longest(ids,comparisons):
        if identifier is None or comparison is None or comparison['id']!=identifier:
            raise ValueError('Exact ordered parent-only e stream required before merge')
        entry=entries[identifier];path=C.ROOT/entry['path']
        if C.sha(path)!=entry['sha256']:raise ValueError('Fresh four-arm spool changed')
        row=OBS.OLD._read_single(path,entry)
        if OBS.D.R._canonical_sha(row)!=entry['canonical_row_sha256'] or row['partition']!=comparison['partition']:
            raise ValueError('Current four-arm/e identity/canonical join changed')
        row['arms']={ARMS[0]:comparison['prediction'],**row['arms']}
        row['e_comparison_parent_only_no_numeric_exposure']=True
        row['legacy_e_pose_semantics']=comparison['legacy_e_pose_semantics']
        if set(row['arms'])!=set(ARMS):raise ValueError('All five declared arms required')
        frame=Path(output)/'frames'/(identifier.replace(':','__').replace('/','_').replace('\\','_')+'.json')
        C.write(frame,row)
        seen.append(identifier)
        yield row
    if seen!=ids:raise ValueError('All238 rows retained after e merge')


def predict(*,root_prediction_GO=False,workers=3,output=OUTPUT):
    if root_prediction_GO is not True:raise PermissionError('Root full prediction GO before any IO/PnP')
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1<=workers<=8:
        raise ValueError('Explicit bounded1..8 worker budget required')
    output=Path(output).resolve();protocol=_load_prepared(output)
    execution=output/'PREDICTION_EXECUTION_PROTOCOL.json'
    if execution.exists() or (output/'PREDICTIONS.jsonl.gz').exists():
        raise FileExistsError('Preserve every prediction attempt')
    (output/'prediction_spool').mkdir();(output/'frames').mkdir()
    C.write(execution,dict(workers=workers,maximum_pending=workers,source_sha256=C.sha(__file__),
        UTC=datetime.now(timezone.utc).isoformat(),positive_worker_keys=sorted(PROJECT.WORKER_KEYS),
        e_numeric_worker_inputs=0,reference_or_score_inputs=0))
    ids=protocol['ordered_ids']
    entries=_bounded_predict(C.rows(output/'INPUTS.jsonl.gz'),ids,output,workers)
    C.write(output/'PREDICTION_ROW_INDEX.json',dict(entries=[entries[identifier] for identifier in ids]))
    OBS.save_rows_large(output/'PREDICTIONS.jsonl.gz',_merged_rows(output,ids,entries,C.rows(output/'E_COMPARISON.jsonl.gz')))
    if [row['id'] for row in C.rows(output/'PREDICTIONS.jsonl.gz')]!=ids:
        raise ValueError('Final all238 prediction journal order changed')
    _load_prepared(output)
    C.write(output/'PREDICTION_SEAL.json',dict(rows=238,ordered_ids=ids,arms=list(ARMS),arm_count=5,
        all_declared_arms_complete=True,predictions_sha256=C.sha(output/'PREDICTIONS.jsonl.gz'),
        protocol_sha256=C.sha(output/'PROTOCOL.json'),input_seal_sha256=C.sha(output/'INPUT_SEAL.json'),
        execution_protocol_sha256=C.sha(execution),row_index_sha256=C.sha(output/'PREDICTION_ROW_INDEX.json'),
        separate_e_comparison_sha256=C.sha(output/'E_COMPARISON.jsonl.gz'),
        before_fixed_reference_scoring=True,no_population_drop=True,default='SAME_FROZEN_E unchanged'))
    return _binding(output/'PREDICTION_SEAL.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','prepare','predict'))
    parser.add_argument('--root-metadata-GO',action='store_true')
    parser.add_argument('--root-prediction-GO',action='store_true')
    parser.add_argument('--central-plan-sha256')
    parser.add_argument('--model-seal-sha256')
    parser.add_argument('--controller-peer')
    parser.add_argument('--controller-peer-sha256')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.stage=='plan':
        print(plan(root_metadata_GO=args.root_metadata_GO,expected_central_plan_sha256=args.central_plan_sha256,
            expected_model_seal_sha256=args.model_seal_sha256,
            controller_peer_binding=dict(path=args.controller_peer,sha256=args.controller_peer_sha256)))
    elif args.stage=='prepare':print(prepare(root_metadata_GO=args.root_metadata_GO))
    else:print(predict(root_prediction_GO=args.root_prediction_GO,workers=args.workers))
