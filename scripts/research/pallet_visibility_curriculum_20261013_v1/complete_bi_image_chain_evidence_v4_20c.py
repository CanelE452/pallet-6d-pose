"""Serialization-only closure of immutable typed finite/topology v20b.

Actual archive JSON turns DEFAULT tuples into lists. Require identical
canonical JSON values before substituting the immutable source recipe. No
measurement, threshold, covariance, authentication or objective math changes.
"""
from copy import deepcopy
import json

from . import complete_bi_image_chain_evidence_v4_20b as TYPED

BROAD, N, M, OBS = TYPED.BROAD, TYPED.N, TYPED.M, TYPED.OBS
NOISE_OBSERVATION_FAMILY = TYPED.NOISE_OBSERVATION_FAMILY
PROTOTYPE_OVERLAP_POLICY = TYPED.PROTOTYPE_OVERLAP_POLICY
plan_complete_bi_replacement = TYPED.plan_complete_bi_replacement
CompleteBIImageChainProblem = TYPED.CompleteBIImageChainProblem


def _same_recipe(configuration):
    encode = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if encode(configuration) != encode(OBS.DEFAULT_CONFIGURATION):
        raise ValueError('Actual NMS numerical recipe changed')
    return deepcopy(OBS.DEFAULT_CONFIGURATION)


def authenticate_complete_bi_image_chain_groups(base, observation, *, junction_proposals, **kwargs):
    normalized = {**observation, 'configuration': _same_recipe(observation.get('configuration'))}
    topology = {**junction_proposals, 'configuration': _same_recipe(junction_proposals.get('configuration'))}
    result = TYPED.authenticate_complete_bi_image_chain_groups(base, normalized,
        junction_proposals=topology, **kwargs)
    result['configuration_serialization_normalized_only'] = True
    result['actual_numeric_recipe_and_all_observations_unchanged'] = True
    return result
