"""Bind the new selection population to old, unchanged actual RGB chains.

The frozen V29 provenance/TLS/finite-support authenticator is reused through
private function globals. All selected lines additionally recheck association
against immutable original B/I q, before source-policy planning. No labels,
pixels, pose fitting or global module mutation occurs here.
"""
from copy import deepcopy
from types import FunctionType

from . import geometry_guided_edge_authentication_v4_29 as PARENT
from . import geometry_guided_edge_problem_v4_30 as G

SCHEMA = 'AUTHENTICATED_OBSERVED_BI_ASSOCIATED_ACTUAL_EDGE_GROUPS_V30'
PRODUCER_GENERATION = G.RAW_OBSERVATION_GENERATION
GEOMETRY_DIGEST_FIELDS = PARENT.GEOMETRY_DIGEST_FIELDS
RELATION, OBS = PARENT.RELATION, PARENT.OBS

_authenticate = FunctionType(PARENT.authenticate_geometry_guided_edges.__code__,
    {**PARENT.authenticate_geometry_guided_edges.__globals__, 'G': G, 'SCHEMA': SCHEMA},
    '_authenticate_V30_population', PARENT.authenticate_geometry_guided_edges.__defaults__)
_authenticate.__kwdefaults__ = deepcopy(PARENT.authenticate_geometry_guided_edges.__kwdefaults__)


def authenticate_geometry_guided_edges(base, observation, *, noise_provenance, **kwargs):
    if not isinstance(base, G.BASELINE.PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Exact prepartial native baseline required')
    if list(kwargs['initial_geometry']['initial_self_hidden']) != [s == 'S' for s in base.states]:
        raise ValueError('Permanent initial S binding precedes selected payload')
    if observation.get('selection_profile') != G.SELECTION_PROFILE:
        raise ValueError('New V30 selected observation required; V29 winner is not interchangeable')
    if (noise_provenance.get('selection_profile') != G.SELECTION_PROFILE
            or noise_provenance.get('observation_family') != G.OBSERVATION_FAMILY):
        raise ValueError('Separately fitted V30 selection-population noise required')
    configuration = observation['association_configuration']
    if configuration != observation['chain_observation']['configuration']:
        raise ValueError('Association reach must bind original raw-chain configuration')
    nodes = {node['name']: node for node in observation['joint_graph']['line_candidates']}
    records = observation['chain_observation']['edge_records']
    checks = []
    for reference in observation['selected_chain_refs']:
        ri = reference['ray_index']
        if isinstance(ri, bool) or not isinstance(ri, int) or not 0 <= ri < len(records):
            raise ValueError('Exact nonnegative ray array index required')
        ray = records[ri]
        k, mate = G._id(ray['native_corner']), G._id(ray['native_neighbor'])
        if base.states[k] not in ('B', 'I') or base.states[mate] == 'S':
            raise ValueError('State/S gates precede line or observed archive payload')
        node = nodes[reference['line_name']]
        # The raw ray IDs, not an optional node ledger, determine the edge.
        check = G.assess_bound_line(base, dict(native_edge=[k, mate],
            normal=node['normal'], offset=node['offset']), configuration)
        if not check['passes']:
            raise ValueError('Selected V30 line fails immutable original B/I association')
        checks.append(dict(line_name=reference['line_name'], **check))
    result = _authenticate(base, observation, noise_provenance=noise_provenance, **kwargs)
    result.update(selection_profile=G.SELECTION_PROFILE,
        association_configuration=deepcopy(configuration), original_BI_association_recomputed=checks,
        raw_measurement_generation_unchanged=True, old_V29_noise_relabelled=False)
    return result
