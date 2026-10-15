"""Authenticate unchanged raw V29 measurements under initial-only V32 policy.

Frozen V31 raw-chain checks and frozen V29 numerical authentication are
private function copies. Original raw query aliases are archived separately
before the new core independently repeats the strict initial-only admission.
"""
from copy import deepcopy
import numpy as np

from . import geometry_guided_edge_authentication_v4_31 as PREVIOUS
from . import geometry_guided_edge_problem_v4_32 as G

PARENT = PREVIOUS.PARENT  # Frozen numerical V29 authenticator.
RELATION, OBS = PREVIOUS.RELATION, PREVIOUS.OBS
SCHEMA = 'AUTHENTICATED_INITIAL_ONLY_OBSERVED_NATIVE_RAY_GROUPS_V32'
PRODUCER_GENERATION = G.RAW_OBSERVATION_GENERATION
GEOMETRY_DIGEST_FIELDS = G.GEOMETRY_DIGEST_FIELDS

_raw_authenticate = G._clone(PARENT.authenticate_geometry_guided_edges,
                            dict(G=G, SCHEMA=SCHEMA))


def _authenticate_with_raw_aliases(base, observation, **kwargs):
    result = _raw_authenticate(base, observation, **kwargs)
    records = observation['chain_observation']['edge_records']
    for group in result['groups']:
        reference = group['selected_chain_ref']
        chain = records[reference['ray_index']]['chains'][reference['chain_index']]
        # Copy the values actually archived by the original producer. Do not
        # manufacture missing aliases from the outer selected-reference ID.
        aliases = {key: deepcopy(chain[key]) for key in G.RAW_QUERY_ALIAS_KEYS}
        if (aliases['guide_reference_id'] != reference['guide_reference_id']
                or aliases['generating_guide_reference_id'] != reference['guide_reference_id']):
            raise ValueError('Original raw query aliases differ from authenticated selected reference')
        group['authenticated_raw_query_aliases'] = aliases
    return result


_authenticate = G._clone(PREVIOUS.authenticate_geometry_guided_edges,
    dict(G=G, SCHEMA=SCHEMA, PRODUCER_GENERATION=PRODUCER_GENERATION,
         _authenticate=_authenticate_with_raw_aliases))


def authenticate_geometry_guided_edges(base, observation, *, noise_provenance, **kwargs):
    # Reject an ineligible source before looking up any selected node's TLS
    # numbers. The frozen V31 wrapper repeats all state/ref/geometry checks;
    # this additional source-only preflight does not alter its mathematics.
    G.validate_noise_population(noise_provenance)
    if not isinstance(base, G.BASELINE.PrepartialRGBPointPreservingNativeProblem):
        raise ValueError('Exact prepartial native baseline required')
    hidden = [s == 'S' for s in base.states]
    if list(kwargs['initial_geometry']['initial_self_hidden']) != hidden:
        raise ValueError('Permanent initial S binding precedes selected payload')
    if (observation.get('selection_profile') != G.SELECTION_PROFILE
            or observation.get('observation_generation') != PRODUCER_GENERATION):
        raise ValueError('New V32 initial-only selection with unchanged raw V29 required')
    records = observation['chain_observation']['edge_records']
    for reference in observation['selected_chain_refs']:
        ri = reference['ray_index']
        if isinstance(ri, (bool, np.bool_)) or not isinstance(ri, (int, np.integer)) or not 0 <= ri < len(records):
            raise ValueError('Exact nonnegative selected raw ray index required')
        ray = records[ri]
        k, mate = G._id(ray['native_corner']), G._id(ray['native_neighbor'])
        if base.states[k] not in ('B', 'I') or base.states[mate] == 'S':
            raise ValueError('State/S gates precede original query and TLS payload')
        ci = reference['chain_index']
        if isinstance(ci, (bool, np.bool_)) or not isinstance(ci, (int, np.integer)) or not 0 <= ci < len(ray['chains']):
            raise ValueError('Exact nonnegative selected raw chain index required')
        stub = dict(native_corner=k, native_neighbor=mate, native_edge=[k, mate])
        verdict = G.ASSOCIATION._source_gate(stub, base.states, hidden, chain=ray['chains'][ci])
        if not verdict['passes']:
            raise ValueError('Selected V32 initial-only query source failed: ' + verdict['reason'])
    result = _authenticate(base, observation, noise_provenance=noise_provenance, **kwargs)
    result.update(schema=SCHEMA, old_V31_noise_relabelled=False,
        initial_only_query_source_recomputed=True, original_query_aliases_preserved_not_fabricated=True)
    return result
