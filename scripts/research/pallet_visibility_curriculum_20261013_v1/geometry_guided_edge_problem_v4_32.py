"""Initial-only actual-query population; frozen V29 factor/guard physics.

Independent initial geometry and image B/I archives remain prerequisites.
No historical pose bank, selected e, prior, start or solver is fetched here.
Actual query aliases are copied from authenticated raw chains by AUTH32 and
rechecked here; missing aliases are never invented from a desired source ID.
"""
from copy import deepcopy
from types import FunctionType

import numpy as np

from . import geometry_guided_edge_problem_v4_31 as PREVIOUS
from . import initial_only_observed_ray_association_v4_32 as ASSOCIATION

PARENT = PREVIOUS.PARENT  # The unchanged V29 numerical problem.
BASELINE, EDGE, N = PREVIOUS.BASELINE, PREVIOUS.EDGE, PREVIOUS.N
V, J, P = PREVIOUS.V, PREVIOUS.J, PREVIOUS.P
SOURCE_POLICIES, FALLBACK = PREVIOUS.SOURCE_POLICIES, PREVIOUS.FALLBACK
RAW_OBSERVATION_GENERATION = PREVIOUS.RAW_OBSERVATION_GENERATION
OBSERVATION_FAMILY = 'INITIAL_ONLY_OBSERVED_BI_NATIVE_RAY_ACTUAL_EDGES_V32'
POPULATION = 'V32_INITIAL_ONLY_QUERY_RUNTIME_ACTIVE_ONE_LINE_PER_PHYSICAL_EDGE'
SELECTION_PROFILE = ASSOCIATION.SELECTION_PROFILE
GEOMETRY_DIGEST_FIELDS = PREVIOUS.GEOMETRY_DIGEST_FIELDS
GEOMETRY_BINDING_SCHEMA = 'ORIGINAL_NON_S_INITIAL_ONLY_RAYS_AND_PREPARTIAL_BI_ARCHIVES_V32'
INITIAL_SOURCE_CLOSURE_SHA256 = ASSOCIATION.INITIAL_SOURCE_CLOSURE_SHA256
_digest, _id, _unit = PREVIOUS._digest, PREVIOUS._id, PREVIOUS._unit
_archive, _finite_actual_pair = PREVIOUS._archive, PREVIOUS._finite_actual_pair
_array, bound_BI_points = PREVIOUS._array, PREVIOUS.bound_BI_points
RAW_QUERY_ALIAS_KEYS = ('guide_reference_id', 'generating_guide_reference_id',
                        'generating_guide_direction_xy')


def _clone(function, updates):
    result = FunctionType(function.__code__, {**function.__globals__, **updates},
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return result


bind_initial_geometry = _clone(PREVIOUS.bind_initial_geometry,
    dict(GEOMETRY_BINDING_SCHEMA=GEOMETRY_BINDING_SCHEMA))
_bound_geometry = _clone(PREVIOUS._bound_geometry,
    dict(GEOMETRY_BINDING_SCHEMA=GEOMETRY_BINDING_SCHEMA, bind_initial_geometry=bind_initial_geometry))


def validate_noise_population(provenance):
    if (provenance.get('fit_partition') != 'CAL'
            or provenance.get('observation_family') != OBSERVATION_FAMILY
            or provenance.get('noise_population') != POPULATION
            or ('population' in provenance and provenance['population'] != POPULATION)
            or provenance.get('selection_profile') != SELECTION_PROFILE
            or provenance.get('raw_observation_generation') != RAW_OBSERVATION_GENERATION):
        raise ValueError('Separate V32 initial-only CAL family/population/profile/raw-generation required')


def _query_node(group, geometry_digest):
    node = PREVIOUS._query_node(group, geometry_digest)
    aliases = group['authenticated_raw_query_aliases']
    reference = group['selected_chain_ref']
    identity = reference['guide_reference_id']
    if (set(aliases) != set(RAW_QUERY_ALIAS_KEYS)
            or aliases['guide_reference_id'] != identity
            or aliases['generating_guide_reference_id'] != identity
            or not np.array_equal(np.asarray(aliases['generating_guide_direction_xy'], float),
                                  np.asarray(group['generating_guide_direction_xy'], float))):
        raise ValueError('Exact authenticated original raw query aliases must remain bound')
    node['line_record'] = dict(guide_reference=reference['guide_reference'], **deepcopy(aliases))
    return node


def assess_bound_line(base, node, configuration, *, geometry):
    return ASSOCIATION.assess_initial_only_observed_ray_association(node,
        bound_BI_points(base), base.states, [s == 'S' for s in base.states],
        configuration, geometry=geometry)


_parent_bundle = _clone(PREVIOUS._parent_bundle, dict(
    OBSERVATION_FAMILY=OBSERVATION_FAMILY, POPULATION=POPULATION,
    SELECTION_PROFILE=SELECTION_PROFILE, RAW_OBSERVATION_GENERATION=RAW_OBSERVATION_GENERATION,
    validate_noise_population=validate_noise_population, _bound_geometry=_bound_geometry,
    _query_node=_query_node, assess_bound_line=assess_bound_line, PARENT=PARENT))
_old_tag_plan = _clone(PREVIOUS._tag_plan, dict(
    OBSERVATION_FAMILY=OBSERVATION_FAMILY, POPULATION=POPULATION,
    RAW_OBSERVATION_GENERATION=RAW_OBSERVATION_GENERATION, SELECTION_PROFILE=SELECTION_PROFILE))


def _tag_plan(plan, bundle, checks):
    _old_tag_plan(plan, bundle, checks)
    plan.update(old_V31_noise_relabelled=False, initial_only_query_population=True,
        external_poseguide_consumed=False,
        initial_source_closure_sha256=INITIAL_SOURCE_CLOSURE_SHA256,
        original_query_aliases_preserved_not_fabricated=True)
    return plan


_empty = _clone(PREVIOUS.empty_geometry_guided_evidence, dict(
    bind_initial_geometry=bind_initial_geometry, OBSERVATION_FAMILY=OBSERVATION_FAMILY,
    POPULATION=POPULATION, RAW_OBSERVATION_GENERATION=RAW_OBSERVATION_GENERATION,
    SELECTION_PROFILE=SELECTION_PROFILE))


def empty_geometry_guided_evidence(*args, **kwargs):
    result = _empty(*args, **kwargs)
    result.update(schema='EXPLICIT_EMPTY_INITIAL_ONLY_GEOMETRY_EDGE_EVIDENCE_V32',
                  old_V31_noise_substitution=False)
    return result


def plan_geometry_guided_replacement(base, bundle, *, source_policy):
    converted, checks = _parent_bundle(base, bundle)
    plan = PARENT.plan_geometry_guided_replacement(base, converted, source_policy=source_policy)
    return _tag_plan(plan, bundle, checks)


class GeometryGuidedEdgeProblem(PARENT.GeometryGuidedEdgeProblem):
    def __init__(self, base, bundle, *, source_policy, tangent_gamma):
        converted, checks = _parent_bundle(base, bundle)
        super().__init__(base, converted, source_policy=source_policy, tangent_gamma=tangent_gamma)
        _tag_plan(self.plan, bundle, checks)

    def objective(self, p):
        value, gradient, parts = super().objective(p)
        parts.update(v29_observation_family=OBSERVATION_FAMILY,
            v32_observation_family=OBSERVATION_FAMILY, v32_selection_profile=SELECTION_PROFILE,
            v32_cost_plan=self.plan)
        return value, gradient, parts
