"""V30 association population with the unchanged V29 pose mathematics.

The raw RGB chains still belong to V29. Only their deterministic selection
population changes. A private metadata adapter lets the frozen numerical
problem handle the newly authenticated population without changing its
module, old models, archived coordinates, factor ordering or derivatives.
Legacy ``v29_cost_plan`` keys remain wire-format aliases, not model versions.
"""
from copy import deepcopy

from . import geometry_guided_edge_problem_v4_29 as PARENT
from . import observed_bi_line_association_v4_30 as ASSOCIATION

BASELINE, EDGE, N = PARENT.BASELINE, PARENT.EDGE, PARENT.N
V, J, P = PARENT.V, PARENT.J, PARENT.P
SOURCE_POLICIES = PARENT.SOURCE_POLICIES
FALLBACK = PARENT.FALLBACK
RAW_OBSERVATION_GENERATION = 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
OBSERVATION_FAMILY = 'OBSERVED_BI_ASSOCIATED_MULTIGUIDE_ACTUAL_NATIVE_EDGES_V30'
POPULATION = 'V30_ASSOCIATION_GATED_RUNTIME_ACTIVE_DETERMINISTIC_ONE_LINE_PER_PHYSICAL_EDGE'
SELECTION_PROFILE = ASSOCIATION.SELECTION_PROFILE
_digest, _id, _unit = PARENT._digest, PARENT._id, PARENT._unit
_archive, _finite_actual_pair = PARENT._archive, PARENT._finite_actual_pair


def bound_BI_points(base):
    """Use the immutable prepartial q archive, gated before its payload."""
    points = {}
    for measurement in base.measurements:
        k = _id(measurement['native_corner'])
        if base.states[k] not in ('B', 'I'):
            raise ValueError('Only nonS B/I archives may enter association')
        archive = _archive(base, k)
        points[k] = dict(record=dict(xy=archive['original_observed_xy'],
            source=archive['original_measurement_source']))
    return points


def assess_bound_line(base, node, configuration):
    return ASSOCIATION.assess_observed_BI_line_association(node,
        bound_BI_points(base), base.states, [s == 'S' for s in base.states], configuration)


def _parent_bundle(base, bundle):
    if (bundle.get('observation_family') != OBSERVATION_FAMILY
            or bundle.get('population') != POPULATION
            or bundle.get('selection_profile') != SELECTION_PROFILE):
        raise ValueError('Separate V30 family, population and selection profile required')
    configuration = bundle['association_configuration']
    checks = []
    for group in bundle['groups']:
        # Native IDs and the initial-S gate precede normal/support/archive reads.
        edge = [_id(group['native_corner']), _id(group['native_neighbor'])]
        if any(base.states[k] == 'S' for k in edge):
            raise ValueError('S physical endpoints excluded before line payload')
        node = dict(native_edge=edge, normal=group['normal_unit_xy'], offset=group['offset_px'])
        check = assess_bound_line(base, node, configuration)
        if not check['passes']:
            raise ValueError('V30 original B/I normal association failed')
        checks.append(check)
    return ({**bundle, 'observation_family': PARENT.OBSERVATION_FAMILY,
        'population': PARENT.POPULATION}, checks)


def _tag_plan(plan, bundle, checks):
    plan.update(observation_family=OBSERVATION_FAMILY, population=POPULATION,
        raw_observation_generation=RAW_OBSERVATION_GENERATION,
        selection_profile=SELECTION_PROFILE,
        association_configuration=deepcopy(bundle['association_configuration']),
        original_BI_association_recomputed=checks,
        old_V29_noise_relabelled=False, numerical_factor_and_guard_math='UNCHANGED_V29')
    return plan


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
            v30_observation_family=OBSERVATION_FAMILY, v30_selection_profile=SELECTION_PROFILE,
            v30_cost_plan=self.plan)
        return value, gradient, parts
