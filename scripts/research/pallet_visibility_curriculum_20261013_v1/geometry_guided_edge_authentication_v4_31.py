"""V31 original-q actual-query and signed-native-ray provenance authentication.

All actual RGB/lattice/TLS/component/floor/owner mathematics stays in frozen
V29 authentication. A private namespace changes metadata constants only. The
new admission is recomputed on exact raw selected refs and again by the core;
neither a stored pass ledger nor a trial WD/pose establishes native identity.
"""
from copy import deepcopy
from types import FunctionType
import numpy as np

from . import geometry_guided_edge_authentication_v4_29 as PARENT
from . import geometry_guided_edge_problem_v4_31 as G

SCHEMA = 'AUTHENTICATED_OBSERVED_Q_ANCHORED_SIGNED_NATIVE_RAY_GROUPS_V31'
PRODUCER_GENERATION = G.RAW_OBSERVATION_GENERATION
GEOMETRY_DIGEST_FIELDS = G.GEOMETRY_DIGEST_FIELDS
RELATION, OBS = PARENT.RELATION, PARENT.OBS

_authenticate = FunctionType(PARENT.authenticate_geometry_guided_edges.__code__,
    {**PARENT.authenticate_geometry_guided_edges.__globals__, 'G': G, 'SCHEMA': SCHEMA},
    '_authenticate_V31_population', PARENT.authenticate_geometry_guided_edges.__defaults__)
_authenticate.__kwdefaults__ = deepcopy(PARENT.authenticate_geometry_guided_edges.__kwdefaults__)


def authenticate_geometry_guided_edges(base, observation, *, noise_provenance, **kwargs):
    if not isinstance(base, G.BASELINE.PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Exact prepartial native baseline required')
    geometry = kwargs['initial_geometry']
    if list(geometry['initial_self_hidden']) != [s == 'S' for s in base.states]:
        raise ValueError('Permanent initial S binding precedes selected payload')
    if (observation.get('selection_profile') != G.SELECTION_PROFILE
            or observation.get('observation_generation') != PRODUCER_GENERATION):
        raise ValueError('New V31 selection with unchanged raw V29 observation required')
    G.validate_noise_population(noise_provenance)
    cfg = observation['association_configuration']
    if cfg != observation['chain_observation']['configuration']:
        raise ValueError('Association configuration must bind the unchanged actual raw chain query')
    nodes = {node['name']: node for node in observation['joint_graph']['line_candidates']}
    records = observation['chain_observation']['edge_records']
    checks = []
    for reference in observation['selected_chain_refs']:
        ri = reference['ray_index']
        if isinstance(ri, (bool, np.bool_)) or not isinstance(ri, (int, np.integer)) or not 0 <= ri < len(records):
            raise ValueError('Exact nonnegative selected raw ray index required')
        ray = records[ri]
        k, mate = G._id(ray['native_corner']), G._id(ray['native_neighbor'])
        if base.states[k] not in ('B', 'I') or base.states[mate] == 'S':
            raise ValueError('State/S gates precede actual chain/query/observed archive payload')
        ci = reference['chain_index']
        if isinstance(ci, (bool, np.bool_)) or not isinstance(ci, (int, np.integer)) or not 0 <= ci < len(ray['chains']):
            raise ValueError('Exact nonnegative selected raw chain index required')
        chain = ray['chains'][ci]
        if (G._id(reference['native_corner']) != k or G._id(reference['native_neighbor']) != mate
                or chain['guide_reference'] != reference['guide_reference']
                or chain['guide_reference_id'] != reference['guide_reference_id']):
            raise ValueError('Actual raw chain must retain its exact original query dictionary/ref')
        node = nodes[reference['line_name']]
        query = dict(native_corner=k, native_neighbor=mate, native_edge=[k, mate],
            normal=node['normal'], offset=node['offset'], line_record=chain)
        verdict = G.assess_bound_line(base, query, cfg, geometry=geometry)
        if not verdict['passes']:
            raise ValueError('Selected V31 line fails original q/query/native-ray admission: ' + verdict['reason'])
        checks.append(dict(line_name=reference['line_name'], **verdict))
    binding = G.bind_initial_geometry(base, geometry, expected_digest=observation['initial_geometry_digest'])
    result = _authenticate(base, observation, noise_provenance=noise_provenance, **kwargs)
    result.update(selection_profile=G.SELECTION_PROFILE, association_configuration=deepcopy(cfg),
        initial_model_ray_binding=binding, original_BI_query_and_native_ray_association_recomputed=checks,
        raw_measurement_generation_unchanged=True, old_V29_or_V30_noise_relabelled=False)
    # Core also verifies exact query/ref IDs, original geometry and immutable
    # q sources; do this even when planning is deferred until after CAL.
    G._parent_bundle(base, result)
    return result
