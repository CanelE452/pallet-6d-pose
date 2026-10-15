"""Separately fitted initial-only V32 population; unchanged V29 likelihood.

Only provenance/selection recipe changes. No IO, historical model substitution,
actual CAL records, labels or runtime pose search occurs in this pure facade.
"""
from copy import deepcopy

from . import geometry_guided_edge_noise_calibration_v4_31 as PREVIOUS
from . import geometry_guided_edge_problem_v4_32 as CORE

PARENT = PREVIOUS.PARENT  # Frozen V29 fitting/export mathematics.
LINE, TANGENT = PREVIOUS.LINE, PREVIOUS.TANGENT
RAW_OBSERVATION_GENERATION = CORE.RAW_OBSERVATION_GENERATION
OBSERVATION_FAMILY, POPULATION = CORE.OBSERVATION_FAMILY, CORE.POPULATION
SELECTION_PROFILE, SOURCE_POLICIES = CORE.SELECTION_PROFILE, CORE.SOURCE_POLICIES
TANGENT_FAMILY = 'V32_INITIAL_ONLY_QUERY_ACTIVE_ARCHIVED_Q_SELECTED_ACTUAL_LINE_TANGENT'
SCHEMA = 'CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_EDGE_NOISE_V32'
TANGENT_SCHEMA = 'CAL_INITIAL_ONLY_OBSERVED_NATIVE_RAY_TANGENT_SCALE_V32'
RECIPE = {**deepcopy(PREVIOUS.RECIPE),
    'observation_generation': OBSERVATION_FAMILY,
    'raw_observation_generation': RAW_OBSERVATION_GENERATION,
    'applicability': POPULATION, 'selection_profile': SELECTION_PROFILE,
    'graph_population': 'initial-only two original actual-query kinds, deterministic graph winner; raw V29 indices and samples unchanged; sealed before CAL labels',
    'association': 'strict original17c/measuredBI source IDs/query dictionaries plus unchanged F31 qanchor/signedray/normal43; external pose_guides forbidden before graph construction',
    'initial_source_closure_sha256': CORE.INITIAL_SOURCE_CLOSURE_SHA256,
    'permitted_query_kinds': list(CORE.ASSOCIATION.ALLOWED_GUIDE_KINDS),
    'external_cached_poseguides_consumed': False,
    'new_selection_population_old_model_relabelled': False}
TANGENT_RECIPE = {**deepcopy(PREVIOUS.TANGENT_RECIPE),
    'population': POPULATION, 'observation_family': TANGENT_FAMILY,
    'raw_observation_generation': RAW_OBSERVATION_GENERATION,
    'selection_profile': SELECTION_PROFILE,
    'initial_source_closure_sha256': CORE.INITIAL_SOURCE_CLOSURE_SHA256,
    'permitted_query_kinds': list(CORE.ASSOCIATION.ALLOWED_GUIDE_KINDS),
    'external_cached_poseguides_consumed': False,
    'new_selection_population_old_model_relabelled': False}

conditional_covariance_floors = PREVIOUS.conditional_covariance_floors
mixture_noise_nll_and_gradient = PREVIOUS.mixture_noise_nll_and_gradient
scalar_tangent_nll_and_gradient = PREVIOUS.scalar_tangent_nll_and_gradient
_component_priors = PREVIOUS._component_priors
_validate_support_priors = PREVIOUS._validate_support_priors
_digest, _clone, prepared_tangent = PREVIOUS._digest, PREVIOUS._clone, PREVIOUS.prepared_tangent


def recipe_for(source_policy, *, tangent=False):
    if source_policy not in SOURCE_POLICIES:
        raise ValueError('Explicit V32 source policy required')
    return {**deepcopy(TANGENT_RECIPE if tangent else RECIPE), 'source_policy': source_policy}


def _family(records, source_policy):
    recipe_for(source_policy)
    records = list(records)
    for row in records:
        if (row.get('partition') != 'CAL'
                or row.get('observation_generation') != OBSERVATION_FAMILY
                or row.get('raw_observation_generation') != RAW_OBSERVATION_GENERATION
                or row.get('noise_population') != POPULATION
                or row.get('selection_profile') != SELECTION_PROFILE
                or row.get('source_policy') != source_policy):
            raise ValueError('New V32 CAL family/population/profile/source policy and unchanged raw V29 required')
    return records


def _parent_function(function):
    return CORE._clone(function, dict(LINE=LINE, TANGENT=TANGENT, SCHEMA=SCHEMA,
        TANGENT_SCHEMA=TANGENT_SCHEMA, TANGENT_FAMILY=TANGENT_FAMILY,
        RAW_OBSERVATION_GENERATION=RAW_OBSERVATION_GENERATION,
        OBSERVATION_FAMILY=OBSERVATION_FAMILY, POPULATION=POPULATION,
        SOURCE_POLICIES=SOURCE_POLICIES, recipe_for=recipe_for,
        _family=_family, _clone=_clone))


def _finish_model(model):
    model.update(selection_profile=SELECTION_PROFILE,
        newly_fitted_selection_population=True, raw_observation_generation_unchanged=True,
        prior_V29_fitted_model_relabelled=False, prior_V30_fitted_model_relabelled=False,
        prior_V31_fitted_model_relabelled=False)
    model['model_digest_sha256'] = _digest({k: v for k, v in model.items() if k != 'model_digest_sha256'})
    return model


def fit_geometry_guided_line_model(records, *, source_policy, **kwargs):
    return _finish_model(_parent_function(PARENT.fit_geometry_guided_line_model)(
        records, source_policy=source_policy, **kwargs))


def fit_geometry_guided_tangent_model(records, *, source_policy, **kwargs):
    return _finish_model(_parent_function(PARENT.fit_geometry_guided_tangent_model)(
        records, source_policy=source_policy, **kwargs))


def _require_model(model, source_policy, *, tangent=False):
    recipe_for(source_policy, tangent=tangent)
    if (model.get('schema') != (TANGENT_SCHEMA if tangent else SCHEMA)
            or model.get('observation_generation') != OBSERVATION_FAMILY
            or model.get('raw_observation_generation') != RAW_OBSERVATION_GENERATION
            or model.get('noise_population') != POPULATION
            or model.get('selection_profile') != SELECTION_PROFILE
            or model.get('source_policy') != source_policy
            or model.get('newly_fitted_selection_population') is not True
            or model.get('raw_observation_generation_unchanged') is not True
            or any(model.get('prior_V'+str(v)+'_fitted_model_relabelled') is not False for v in (29, 30, 31))):
        raise ValueError('Newly fitted V32 model/source policy required; old29/30/31 models rejected')


def mixture_noise_parameters(model, *, source_policy):
    _require_model(model, source_policy)
    result = _parent_function(PARENT.mixture_noise_parameters)(model, source_policy=source_policy)
    result['noise_provenance'].update(selection_profile=SELECTION_PROFILE,
        raw_observation_generation=RAW_OBSERVATION_GENERATION,
        observation_family=OBSERVATION_FAMILY, noise_population=POPULATION,
        newly_fitted_selection_population=True, prior_V29_fitted_model_relabelled=False,
        prior_V30_fitted_model_relabelled=False, prior_V31_fitted_model_relabelled=False)
    return result


def tangent_variance_parameters(model, *, source_policy):
    _require_model(model, source_policy, tangent=True)
    result = _parent_function(PARENT.tangent_variance_parameters)(model, source_policy=source_policy)
    return {**result, 'selection_profile': SELECTION_PROFILE,
        'raw_observation_generation': RAW_OBSERVATION_GENERATION,
        'observation_family': OBSERVATION_FAMILY, 'noise_population': POPULATION,
        'newly_fitted_selection_population': True, 'prior_V29_fitted_model_relabelled': False,
        'prior_V30_fitted_model_relabelled': False, 'prior_V31_fitted_model_relabelled': False}


def calibrated_archived_tangent_variance(model, tangent_unit_xy,
        original_effective_fit_covariance_px2, *, source_policy):
    parameters = tangent_variance_parameters(model, source_policy=source_policy)
    a = TANGENT._variance_axis(tangent_unit_xy, original_effective_fit_covariance_px2)
    return dict(tangent_variance_px2=parameters['pixel_floor_variance_px2'] + parameters['gamma'] * a,
        original_variance_axis_px2=a, old_x4_engineering_control_variance_px2=4. * a, **parameters)
