"""Pure V31 CAL/runtime join and zero-component metadata preparation.

The v30 native baseline and input join are private function copies with G31
bound. No production source, module global, guard or numerical function is
changed. Full CAL46 IO/central declaration/operational driver are deliberately
outside this pure module; callers must seal and verify that ancestry first.
"""
from collections import Counter
from copy import deepcopy
from types import FunctionType

from . import prepare_geometry_guided_native_evidence_v4_30 as PARENT
from . import geometry_guided_edge_problem_v4_31 as G
from . import geometry_guided_edge_authentication_v4_31 as AUTH

C, PREFIX, B = PARENT.C, PARENT.PREFIX, PARENT.B
TASKS = PARENT.TASKS
OUTPUT = B / 'geometry_guided_native_evidence_v4_31_checks'


def _parent_function(function):
    copied = FunctionType(function.__code__, {**function.__globals__, 'G': G},
                          function.__name__, function.__defaults__, function.__closure__)
    copied.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return copied


native_baseline = _parent_function(PARENT.native_baseline)
validate_runtime_join = _parent_function(PARENT.validate_runtime_join)


def validate_join(task, row):
    if task['partition'] != 'CAL' or row['partition'] != 'CAL' or task['id'] != row['id']:
        raise ValueError('Exact CAL-only V31 metadata join required before coordinate payload')
    return validate_runtime_join(task, row)


def prepare_one(task, row):
    geometry = validate_join(task, row)
    if (row.get('selection_profile') != G.SELECTION_PROFILE
            or row['observation'].get('selection_profile') != G.SELECTION_PROFILE):
        raise ValueError('Exact new V31 selected observation/profile required')
    family = task['families']['ORIGINAL_WD']
    base = native_baseline(task, family)
    bundle = AUTH.authenticate_geometry_guided_edges(base, row['observation'],
        image_binding=dict(sha256=task['image_sha256'], image_hw=task['spec']['image_hw']),
        initial_geometry=geometry, normal_variance_component_px2=0., angular_variance_component_sine2=0.,
        noise_provenance=dict(fit_partition='CAL', observation_family=G.OBSERVATION_FAMILY,
            noise_population=G.POPULATION, selection_profile=G.SELECTION_PROFILE,
            raw_observation_generation=G.RAW_OBSERVATION_GENERATION,
            mode='ZERO_COMPONENT_PREFIT_METADATA_NOT_FITTED'), calibration_preparation=True)
    profiles, metadata = {}, {}
    for policy in G.SOURCE_POLICIES:
        plan = G.plan_geometry_guided_replacement(base, bundle, source_policy=policy)
        groups, tangents = [], []
        for group in plan['active_groups']:
            component = deepcopy(group)
            component['mixture_component_prior'] = 1.
            groups.append(dict(physical_native_edge=group['physical_native_edge'],
                reliability_weight=group['fit_reliability_weight'], components=[component],
                selected_chain_ref=deepcopy(group['selected_chain_ref']),
                activation_owners=deepcopy(group['activation_owners'])))
        for record in plan['corner_plans']:
            if record['mode'] != 'ONE_SELECTED_LINE_ARCHIVED_Q_TANGENT':
                continue
            archive = record['source_archive']
            tangents.append(dict(kind='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT',
                native_corner=record['native_corner'], state=record['state'],
                source=archive['original_measurement_source'], archived_q_xy=archive['original_observed_xy'],
                selected_line_tangent_unit_xy=record['selected_line_tangent_unit_xy'],
                effective_fit_covariance_px2=archive['original_effective_fit_covariance_px2'],
                fit_reliability_weight=archive['original_fit_reliability_weight'],
                selected_line_ref=record['selected_line_ref']))
        profiles[policy] = dict(groups=groups, tangent_records=tangents,
            runtime_plan_digest=G._digest(plan), selection_profile=G.SELECTION_PROFILE)
        metadata[policy] = dict(groups=len(groups), tangents=len(tangents), owners=len(plan['new_owner_ids']),
            modes=dict(Counter(record['mode'] for record in plan['corner_plans'])), ledger=plan['ledger'])
    prepared = dict(id=task['id'], partition='CAL', session=task['session'], states=task['states'],
        initial_geometry=geometry, observation_generation=G.OBSERVATION_FAMILY,
        raw_observation_generation=AUTH.PRODUCER_GENERATION, noise_population=G.POPULATION,
        selection_profile=G.SELECTION_PROFILE, profiles=profiles)
    return prepared, dict(id=task['id'], session=task['session'], profiles=metadata,
        authenticated_groups=len(bundle['groups']), skipped=bundle['skipped'],
        cross_native_overlap_withheld=bundle['cross_native_support_overlap_withheld_component_ids'])
