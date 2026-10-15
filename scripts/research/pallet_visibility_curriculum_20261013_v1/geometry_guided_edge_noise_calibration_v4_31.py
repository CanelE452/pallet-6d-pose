"""Pure NEW V31 q-anchor/signed-ray population noise models.

Only family, profile and recipe provenance change. Numerical likelihood,
analytic gradients, floors, frame/session weights, optimizer starts and LOGO
behavior are the frozen V29 functions in private namespaces. No V29/V30 model
is renamed, accepted or substituted. There is no IO or actual-data execution.
"""
from __future__ import annotations
from copy import deepcopy
from types import FunctionType

from . import geometry_guided_edge_noise_calibration_v4_29 as PARENT
from . import geometry_guided_edge_problem_v4_31 as CORE

LINE,TANGENT=PARENT.LINE,PARENT.TANGENT
RAW_OBSERVATION_GENERATION=CORE.RAW_OBSERVATION_GENERATION
OBSERVATION_FAMILY,POPULATION=CORE.OBSERVATION_FAMILY,CORE.POPULATION
SELECTION_PROFILE,SOURCE_POLICIES=CORE.SELECTION_PROFILE,CORE.SOURCE_POLICIES
TANGENT_FAMILY='V31_OBSERVED_Q_ANCHORED_SIGNED_RAY_ACTIVE_ARCHIVED_Q_SELECTED_ACTUAL_LINE_TANGENT'
SCHEMA='CAL_OBSERVED_Q_ANCHORED_SIGNED_NATIVE_RAY_EDGE_NOISE_V31'
TANGENT_SCHEMA='CAL_OBSERVED_Q_ANCHORED_SIGNED_NATIVE_RAY_TANGENT_SCALE_V31'
RECIPE={**deepcopy(PARENT.RECIPE),
    'observation_generation':OBSERVATION_FAMILY,
    'raw_observation_generation':RAW_OBSERVATION_GENERATION,
    'applicability':POPULATION,'selection_profile':SELECTION_PROFILE,
    'graph_population':'new deterministic q-query-anchor/fixed-signed-native-ray graph winner; raw V29 arrays and original indices retained; population sealed before CAL labels',
    'association':'unchanged V30 normal43 gate plus actual query anchor equals immutable source BI q within metadata1e-6, model-anchor tag rejected; positive immutable BI or ORIGINAL initial model signed ray, existing minimum10px; short/missing/orthogonal reference abstains',
    'anchor_equality_is_metadata_not_spatial_correction_cap':True,
    'finite_alongline_native_identity_certified':False,
    'new_selection_population_old_model_relabelled':False}
TANGENT_RECIPE={**deepcopy(PARENT.TANGENT_RECIPE),
    'population':POPULATION,'observation_family':TANGENT_FAMILY,
    'raw_observation_generation':RAW_OBSERVATION_GENERATION,
    'selection_profile':SELECTION_PROFILE,
    'new_selection_population_old_model_relabelled':False}

conditional_covariance_floors=PARENT.conditional_covariance_floors
mixture_noise_nll_and_gradient=PARENT.mixture_noise_nll_and_gradient
scalar_tangent_nll_and_gradient=PARENT.scalar_tangent_nll_and_gradient
_component_priors=PARENT._component_priors
_validate_support_priors=PARENT._validate_support_priors
_digest=PARENT._digest
_clone=PARENT._clone
prepared_tangent=PARENT.prepared_tangent


def recipe_for(source_policy,*,tangent=False):
    if source_policy not in SOURCE_POLICIES:
        raise ValueError('Explicit V31 source policy required')
    return {**deepcopy(TANGENT_RECIPE if tangent else RECIPE),'source_policy':source_policy}


def _family(records,source_policy):
    recipe_for(source_policy)
    records=list(records)
    for row in records:
        if (row.get('partition')!='CAL'
                or row.get('observation_generation')!=OBSERVATION_FAMILY
                or row.get('raw_observation_generation')!=RAW_OBSERVATION_GENERATION
                or row.get('noise_population')!=POPULATION
                or row.get('selection_profile')!=SELECTION_PROFILE
                or row.get('source_policy')!=source_policy):
            raise ValueError('New V31 CAL family/population/selection profile/source policy and raw V29 generation required')
    return records


def _parent_function(function):
    updates=dict(LINE=LINE,TANGENT=TANGENT,SCHEMA=SCHEMA,
        TANGENT_SCHEMA=TANGENT_SCHEMA,TANGENT_FAMILY=TANGENT_FAMILY,
        RAW_OBSERVATION_GENERATION=RAW_OBSERVATION_GENERATION,
        OBSERVATION_FAMILY=OBSERVATION_FAMILY,POPULATION=POPULATION,
        SOURCE_POLICIES=SOURCE_POLICIES,recipe_for=recipe_for,
        _family=_family,_clone=_clone)
    result=FunctionType(function.__code__,{**function.__globals__,**updates},
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=deepcopy(function.__kwdefaults__)
    return result


def _finish_model(model):
    # One final digest covers all numerical outputs and NEW provenance flags;
    # no detached parent digest is retained as the runtime model identity.
    model.update(selection_profile=SELECTION_PROFILE,
        newly_fitted_selection_population=True,raw_observation_generation_unchanged=True,
        prior_V29_fitted_model_relabelled=False,prior_V30_fitted_model_relabelled=False)
    model['model_digest_sha256']=_digest({k:v for k,v in model.items()if k!='model_digest_sha256'})
    return model


def fit_geometry_guided_line_model(records,*,source_policy,**kwargs):
    """Frozen three-start normal/sine likelihood on V31 population only."""
    return _finish_model(_parent_function(PARENT.fit_geometry_guided_line_model)(
        records,source_policy=source_policy,**kwargs))


def fit_geometry_guided_tangent_model(records,*,source_policy,**kwargs):
    """Frozen scalar gamma recipe on V31 selected measured-line axes only."""
    return _finish_model(_parent_function(PARENT.fit_geometry_guided_tangent_model)(
        records,source_policy=source_policy,**kwargs))


def _require_model(model,source_policy,*,tangent=False):
    recipe_for(source_policy,tangent=tangent)
    if (model.get('schema')!=(TANGENT_SCHEMA if tangent else SCHEMA)
            or model.get('observation_generation')!=OBSERVATION_FAMILY
            or model.get('raw_observation_generation')!=RAW_OBSERVATION_GENERATION
            or model.get('noise_population')!=POPULATION
            or model.get('selection_profile')!=SELECTION_PROFILE
            or model.get('source_policy')!=source_policy
            or model.get('newly_fitted_selection_population')is not True
            or model.get('raw_observation_generation_unchanged')is not True
            or model.get('prior_V29_fitted_model_relabelled')is not False
            or model.get('prior_V30_fitted_model_relabelled')is not False):
        raise ValueError('Bound newly fitted V31 model/source policy/selection profile required; V29/V30 models rejected')


def mixture_noise_parameters(model,*,source_policy):
    _require_model(model,source_policy)
    result=_parent_function(PARENT.mixture_noise_parameters)(model,source_policy=source_policy)
    result['noise_provenance'].update(selection_profile=SELECTION_PROFILE,
        raw_observation_generation=RAW_OBSERVATION_GENERATION,
        observation_family=OBSERVATION_FAMILY,noise_population=POPULATION,
        newly_fitted_selection_population=True,
        prior_V29_fitted_model_relabelled=False,prior_V30_fitted_model_relabelled=False)
    return result


def tangent_variance_parameters(model,*,source_policy):
    _require_model(model,source_policy,tangent=True)
    result=_parent_function(PARENT.tangent_variance_parameters)(model,source_policy=source_policy)
    return {**result,'selection_profile':SELECTION_PROFILE,
        'raw_observation_generation':RAW_OBSERVATION_GENERATION,
        'observation_family':OBSERVATION_FAMILY,'noise_population':POPULATION,
        'newly_fitted_selection_population':True,
        'prior_V29_fitted_model_relabelled':False,'prior_V30_fitted_model_relabelled':False}


def calibrated_archived_tangent_variance(model,tangent_unit_xy,
        original_effective_fit_covariance_px2,*,source_policy):
    parameters=tangent_variance_parameters(model,source_policy=source_policy)
    a=TANGENT._variance_axis(tangent_unit_xy,original_effective_fit_covariance_px2)
    return dict(tangent_variance_px2=parameters['pixel_floor_variance_px2']+parameters['gamma']*a,
        original_variance_axis_px2=a,old_x4_engineering_control_variance_px2=4.*a,**parameters)
