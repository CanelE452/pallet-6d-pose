"""Pure, partition-independent projection to the clean V32 worker interface.

The parent owns archive/source/model byte verification and raw-row AUTH32.
Only ORIGINAL independent coarse physics and gated prepartial B/I archives
are read here. Historical poses/banks, SWAPPED legacy fields, raw observation
and external guides are never forwarded. Fresh PnP belongs to the worker,
after this projection; no IO, PnP, fit or image/reference query occurs here.
"""
from copy import deepcopy

import numpy as np

from . import prepare_geometry_guided_native_evidence_v4_32 as FACTORY
from . import clean_fresh_BI_runtime_v4_32 as CLEAN

G, FRESH = FACTORY.G, FACTORY.FRESH
SCHEMA = 'POSITIVE_FULL_COHORT_CLEAN_INPUT_PROJECTION_V32'
WORKER_SCHEMA = 'POSITIVE_CLEAN_FRESH_BI_WORKER_PACKET_V32'
PARTITIONS = ('CAL', 'TEST', 'REMAINDER')
FAMILIES = ('ORIGINAL_WD', 'SWAPPED_WD')
GEOMETRY_FIELDS = ('native_xyz', 'R_cf', 'centroid', 'physical_registration_Q',
                   'K', 'image_hw', 'initial_self_hidden')
CLEAN_INPUT_KEYS = frozenset(('native_xyz', 'K', 'states', 'initial_self_hidden',
    'visible_measurements', 'initial_coarse_pose', 'physical_registration_Q',
    'initial_source_binding', 'identifiers'))
PHYSICS_KEYS = frozenset(('domains', 'image_hw', 'config', 'boundary_merged',
                          'u_regions', 'initial_self_hidden'))
REGULARIZER_KEYS = frozenset(('o_regions', 'o_parameters', 'original_S_prepared',
                            'outline_observation', 'outline_configuration'))
WORKER_KEYS = frozenset(('schema', 'identifiers', 'clean_input', 'physics',
                        'typed_bundles', 'regularizers', 'source_contract'))
DEPENDENCY_SHA256 = {
    'clean_fresh_BI_runtime_v4_32.py': '8cf1b99c3430946bb187434b1f7e671e54dad49f73a53288ea154fb375940a84',
    'prepare_geometry_guided_native_evidence_v4_32.py': 'db6467591e237e5d6689935fd744f3d28903465a1c4990feaf09f5b564822b20',
    'geometry_guided_edge_problem_v4_32.py': '7cedbdd09c00604ecfa81fe15d9795f8121a43f0ecc893f9f397d7c3aee00465',
    'geometry_guided_edge_authentication_v4_32.py': '199cb21c85e4908c08ef466b7636463bf4b1a09481d7f82b90ef8d2495808f2b',
    'initial_only_observed_ray_association_v4_32.py': 'd4ab01e8ffb0d3ee00e18730fdd7976ca9debf84e7c432c216df6cc648931674',
}
_BUNDLE_REQUIRED = ('schema', 'observation_family', 'population', 'producer_generation',
    'selection_profile', 'association_configuration', 'groups', 'ordered_corner_pairs',
    'initial_geometry_digest', 'initial_model_ray_binding', 'image_binding')
_BUNDLE_OPTIONAL = ('evidence_availability', 'abstention_reason', 'old_noise_substitution',
    'new_image_queries', 'new_precise_KPs', 'permanent_S_U_local_queries',
    'old_V31_noise_substitution')


def _tags():
    return dict(observation_family=G.OBSERVATION_FAMILY, noise_population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, raw_observation_generation=G.RAW_OBSERVATION_GENERATION)


def _identifiers(task, row):
    partition = task['partition']
    if (partition not in PARTITIONS or row['partition'] != partition
            or not isinstance(task['id'], str) or not task['id'] or row['id'] != task['id']):
        raise ValueError('Same supported partition and string row identity required before physics')
    if (not isinstance(task['session'], str) or not task['session']
            or row['session'] != task['session'] or row['image_sha256'] != task['image_sha256']
            or not _sha(task['image_sha256'])):
        raise ValueError('Same session and exact image SHA metadata required')
    return dict(id=task['id'], partition=partition, session=task['session'],
                image_sha256=task['image_sha256'])


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _positive_digest(clean, physics, geometry):
    return G._digest(dict(clean_input={key:value for key,value in clean.items()
        if key != 'identifiers'}, physics=physics, initial_geometry=geometry))


def project_clean_input(task, row, *, initial_source_binding):
    """Return positive physics plus parent-only AUTH arguments; raw row stays out.

    Source binding is an explicit caller attestation, not an ancestry proof
    created by this function. Valid source/geometry/archive mismatches fail
    closed. An invalid ORIGINAL coarse is instead a retained no-pose row,
    before accessing B/I q, domains, U regions, observation or guard payload.
    Neither partition nor identifier participates in numerical projection.
    """
    ids = _identifiers(task, row)
    binding = FACTORY._source_binding(initial_source_binding)
    if tuple(G.GEOMETRY_DIGEST_FIELDS) != GEOMETRY_FIELDS:
        raise ValueError('Frozen seven-key geometry excludes all supplied projection slots')
    spec = task['spec']
    states, hidden = list(task['states']), list(task['initial_self_hidden'])
    if list(spec['states']) != states or list(row['states']) != states:
        raise ValueError('Same authoritative state binding before coordinate payload')
    FACTORY._BI_records(states, hidden, [])  # State/S consistency, no q access.
    original = task['families']['ORIGINAL_WD']
    xyz = FRESH._array(original['native_xyz'], (8, 3), 'Original known native XYZ')
    Q = FRESH._rotation(original['physical_registration_Q'], 'Original known Q')
    K = FRESH._array(spec['K'], (3, 3), 'Known camera K')
    coarse = dict(source=FRESH.COARSE_SOURCE, independent_of_cached_pose_bank=True)
    reason = None
    try:
        # Deliberately never inspect other coarse/family fields or old poses.
        R = FRESH._rotation(original['coarse_pose']['R'], 'Independent original initial R')
        t = FRESH._array(original['coarse_pose']['t'], (3,), 'Independent original native t')
        FRESH._proposal(R, t, xyz, Q, 'OWN', FRESH.COARSE_SOURCE, {})
        coarse.update(R=R.tolist(), t=t.tolist(), physical_registration_Q=Q.tolist(),
                      R_physical=(R @ Q).tolist())
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        reason = 'invalid_independent_initial_coarse: ' + str(error)
    clean = dict(native_xyz=xyz.tolist(), K=K.tolist(), states=states,
        initial_self_hidden=hidden, visible_measurements=[], initial_coarse_pose=coarse,
        physical_registration_Q=Q.tolist(), initial_source_binding=binding, identifiers=ids)
    contract = dict(**_tags(), ready=reason is None, initial_source_binding=deepcopy(binding),
        row_retained=True, caller_verifies_actual_source_input_and_model_bytes=True,
        independent_initial_input_attestation_not_statistical_independence=True,
        historical_task_or_pose_bank_forwarded=False, raw_observation_forwarded=False,
        fresh_PnP_performed_by_projector=False, old_model_substituted=False,
        native_factory_uses_exact_clean_input_visible_archive=True,
        clean_spec_visible_is_PnP_only_not_native_guard_archive=True,
        frozen_dependency_sha256=deepcopy(DEPENDENCY_SHA256))
    if reason is not None:
        contract['abstention_reason'] = reason
        return dict(schema=SCHEMA, ready=False, identifiers=ids, clean_input=clean,
            physics=None, observation_auth_arguments=None, source_contract=contract)
    geometry = row['initial_geometry']
    if (not np.array_equal(xyz, np.asarray(geometry['native_xyz'], float))
            or not np.array_equal(Q, np.asarray(geometry['physical_registration_Q'], float))
            or not np.array_equal(K, np.asarray(geometry['K'], float))
            or not np.allclose(R, geometry['R_cf'], atol=1e-12, rtol=0)
            or not np.allclose(R @ xyz.mean(axis=0)+t, geometry['centroid'], atol=1e-12, rtol=0)
            or list(geometry['image_hw']) != list(spec['image_hw'])
            or list(geometry['initial_self_hidden']) != hidden):
        raise ValueError('Exact independent ORIGINAL geometry/camera/canvas/S join required')
    visible = FACTORY._BI_records(states, hidden, spec['visible'])
    observed = FACTORY._BI_records(states, hidden, row['visible_measurements'])
    if (visible != observed or row['domains'] != spec['domains']
            or row['u_regions'] != task['u_regions']):
        raise ValueError('Exact archived BI q/source/covariance/reliability/domain/U join required')
    if row['selection_profile'] != G.SELECTION_PROFILE:
        raise ValueError('Exact initial-only V32 selected profile required')
    projection = [None] * 8
    for k, is_hidden in enumerate(hidden):
        if is_hidden:
            continue  # Never index/copy supplied S slots; unknown is MODEL only.
        supplied = geometry['initial_projection_xy'][k]
        if task['complete_BI_evidence']['initial_projection_xy'][k] != supplied:
            raise ValueError('Original nonS MODEL projection source changed')
        value = G._array(supplied, (2,), 'Original nonS MODEL projection')
        camera = R @ xyz[k] + t
        hom = K @ camera
        if abs(hom[2]) <= 1e-9 or not np.allclose(value, hom[:2]/hom[2], atol=1e-6, rtol=0):
            raise ValueError('Original nonS MODEL ray must match independent coarse physics')
        projection[k] = value.tolist()
    sanitized = {key: deepcopy(geometry[key]) for key in G.GEOMETRY_DIGEST_FIELDS}
    sanitized['initial_projection_xy'] = projection
    clean['visible_measurements'] = visible
    physics = dict(domains=deepcopy(spec['domains']), image_hw=list(spec['image_hw']),
        config=deepcopy(spec['config']), boundary_merged=deepcopy(task['boundary_merged']),
        u_regions=deepcopy(task['u_regions']), initial_self_hidden=hidden)
    # IDs/session/partition do not enter this same-graph numerical binding.
    contract['positive_physics_sha256'] = _positive_digest(clean, physics, sanitized)
    return dict(schema=SCHEMA, ready=True, identifiers=ids, clean_input=clean, physics=physics,
        observation_auth_arguments=dict(initial_geometry=sanitized,
            image_binding=dict(sha256=ids['image_sha256'], image_hw=list(spec['image_hw']))),
        source_contract=contract)


def _runtime_export(value, policy, *, tangent):
    if not isinstance(value, dict):
        raise ValueError('Typed newly fitted V32 runtime export required')
    G.validate_noise_population(value)
    if (value.get('source_policy') != policy or not _sha(value.get('model_digest_sha256'))
            or value.get('newly_fitted_selection_population') is not True
            or any(value.get('prior_V'+str(v)+'_fitted_model_relabelled') is not False for v in (29, 30, 31))):
        raise ValueError('New V32 policy/model digest; no prior model substitution')
    if tangent:
        gamma = float(value['gamma'])
        if (not np.isfinite(gamma) or gamma < 0 or value['pixel_floor_variance_px2'] != 1.
                or value['tangent_axis'] != 'SELECTED_ACTUAL_MEASURED_LINE_TANGENT'):
            raise ValueError('Typed selected actual-axis gamma and fixed floor1 required')
    return deepcopy(value)


def _bundle(bundle, projected, policy, line_provenance, *, available):
    if not isinstance(bundle, dict):
        raise ValueError('AUTH32 selected bundle or explicit newempty32 required')
    empty = bundle.get('schema') == 'EXPLICIT_EMPTY_INITIAL_ONLY_GEOMETRY_EDGE_EVIDENCE_V32'
    if (bundle.get('schema') not in ('AUTHENTICATED_INITIAL_ONLY_OBSERVED_NATIVE_RAY_GROUPS_V32',
                                   'EXPLICIT_EMPTY_INITIAL_ONLY_GEOMETRY_EDGE_EVIDENCE_V32')
            or bundle.get('observation_family') != G.OBSERVATION_FAMILY
            or bundle.get('population') != G.POPULATION
            or bundle.get('selection_profile') != G.SELECTION_PROFILE
            or bundle.get('producer_generation') != G.RAW_OBSERVATION_GENERATION):
        raise ValueError('New typed V32 selected bundle required; previous generations rejected')
    groups = bundle['groups']
    states = projected['clean_input']['states']
    hidden = projected['clean_input']['initial_self_hidden']
    for group in groups:
        k, mate = G._id(group['native_corner']), G._id(group['native_neighbor'])
        if states[k] not in ('B', 'I') or hidden[k] or states[mate] == 'S' or hidden[mate]:
            raise ValueError('Selected bundle state/S gate before query/noise/TLS payload')
    if not available and (not empty or groups or bundle['ordered_corner_pairs']
            or bundle.get('evidence_availability') != 'UNAVAILABLE_UNIFORM_ABSTENTION'
            or bundle.get('old_noise_substitution') is not False):
        raise ValueError('Unavailable new model requires explicit uniform empty32; no old noise')
    auth = projected['observation_auth_arguments']
    geometry = auth['initial_geometry']
    expected_geometry = {key:geometry[key] for key in G.GEOMETRY_DIGEST_FIELDS}
    binding = bundle['initial_model_ray_binding']
    archives = [dict(native_corner=record['native_corner'], xy=deepcopy(record['xy']), source=record['source'])
                for record in sorted(projected['clean_input']['visible_measurements'], key=lambda m:m['native_corner'])]
    if (bundle['image_binding'] != auth['image_binding']
            or bundle['initial_geometry_digest'] != G._digest(expected_geometry)
            or binding['schema'] != G.GEOMETRY_BINDING_SCHEMA
            or binding['geometry'] != expected_geometry
            or binding['initial_projection_xy'] != geometry['initial_projection_xy']
            or binding['initial_geometry_digest'] != bundle['initial_geometry_digest']
            or binding['original_BI_archives'] != archives
            or binding['original_BI_archive_digest'] != G._digest(archives)):
        raise ValueError('AUTH32 exact ORIGINAL geometry/BI archives/image binding required')
    seen = set()
    points = {record['native_corner']:dict(record=record) for record in projected['clean_input']['visible_measurements']}
    for group in groups:
        # Source/state gate occurs before reading TLS values or copying support.
        aliases = group['authenticated_raw_query_aliases']
        stub = dict(native_edge=[group['native_corner'], group['native_neighbor']],
            native_corner=group['native_corner'], native_neighbor=group['native_neighbor'],
            line_record=dict(guide_reference=group['selected_chain_ref']['guide_reference'], **aliases))
        verdict = G.ASSOCIATION._source_gate(stub, states, hidden)
        if not verdict['passes']:
            raise ValueError('Authenticated initial-only query source rejected: '+verdict['reason'])
        node = G._query_node(group, bundle['initial_geometry_digest'])
        check = G.ASSOCIATION.assess_initial_only_observed_ray_association(node, points,
            states, hidden, bundle['association_configuration'], geometry=geometry)
        if not check['passes']:
            raise ValueError('Authenticated q-anchor/native ray admission changed: '+check['reason'])
        if group['noise_provenance'] != line_provenance:
            raise ValueError('Selected component must use exact policy line model exporter')
        pair = tuple(sorted((G._id(group['native_corner']), G._id(group['native_neighbor']))))
        if pair in seen or list(pair) != list(group['physical_native_edge']):
            raise ValueError('One authenticated selected component per named physical edge required')
        seen.add(pair)
    # Only the selected evidence/binding interface crosses the process boundary.
    # All raw alternatives remain in the parent's immutable archive.
    return {key:deepcopy(bundle[key]) for key in (*_BUNDLE_REQUIRED, *_BUNDLE_OPTIONAL) if key in bundle}


def assemble_worker_input(projected, *, typed_bundles, regularizers,
        model_availability, tangent_parameters, line_noise_provenance):
    """Close a worker packet after parent AUTH32; no raw observation is accepted.

    Available policies require their own newly fitted line/tangent exports.
    Unavailable policies require explicit empty32 bundles, no substitute model
    and no tangent export; the worker uses gamma0 for that uniform abstention.
    Invalid-coarse rows return before inspecting any other supplied payload.
    """
    if (projected.get('schema') != SCHEMA or projected.get('ready') is not projected['source_contract']['ready']
            or not isinstance(projected['ready'], bool) or set(projected['clean_input']) != CLEAN_INPUT_KEYS
            or projected['clean_input']['initial_source_binding'] != projected['source_contract']['initial_source_binding']
            or projected['identifiers'] != projected['clean_input']['identifiers']):
        raise ValueError('Same projected source binding/identity/readiness required')
    contract = deepcopy(projected['source_contract'])
    if not projected['ready']:
        return dict(schema=WORKER_SCHEMA, identifiers=deepcopy(projected['identifiers']),
            clean_input=deepcopy(projected['clean_input']), physics=None,
            typed_bundles=None, regularizers=None, source_contract=contract)
    if (set(projected['physics']) != PHYSICS_KEYS
            or projected['source_contract']['positive_physics_sha256'] != _positive_digest(
                projected['clean_input'], projected['physics'],
                projected['observation_auth_arguments']['initial_geometry'])):
        raise ValueError('Exact positive original archive/physics projection changed after parent join')
    for mapping in (typed_bundles, model_availability, tangent_parameters, line_noise_provenance):
        if not isinstance(mapping, dict) or set(mapping) != set(G.SOURCE_POLICIES):
            raise ValueError('Exact two V32 source policies required')
    if not isinstance(regularizers, dict) or set(regularizers) != REGULARIZER_KEYS:
        raise ValueError('Only five named O/S/containment regularizer inputs allowed')
    bundles, tangent, line = {}, {}, {}
    for policy in G.SOURCE_POLICIES:
        available = model_availability[policy]
        if not isinstance(available, bool):
            raise ValueError('Explicit boolean new model availability required')
        if available:
            tangent[policy] = _runtime_export(tangent_parameters[policy], policy, tangent=True)
            line[policy] = _runtime_export(line_noise_provenance[policy], policy, tangent=False)
        else:
            if tangent_parameters[policy] is not None or line_noise_provenance[policy] is not None:
                raise ValueError('Unavailable new model cannot substitute old gamma or noise')
            tangent[policy] = line[policy] = None
        if not isinstance(typed_bundles[policy], dict) or set(typed_bundles[policy]) != set(FAMILIES):
            raise ValueError('Both clean WD families must retain typed evidence/abstention')
        bundles[policy] = {name:_bundle(typed_bundles[policy][name], projected, policy,
                                      line[policy], available=available) for name in FAMILIES}
    contract.update(model_availability=deepcopy(model_availability),
        tangent_parameters=tangent, line_noise_provenance=line,
        unavailable_policy_tangent_gamma=0., old_model_substituted=False)
    return dict(schema=WORKER_SCHEMA, identifiers=deepcopy(projected['identifiers']),
        clean_input=deepcopy(projected['clean_input']), physics=deepcopy(projected['physics']),
        typed_bundles=bundles, regularizers=deepcopy(regularizers), source_contract=contract)
