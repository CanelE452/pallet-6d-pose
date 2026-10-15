"""Pure CAL models for the separately versioned v29 multiguide observer.

No old fitted model is relabelled. The immutable v20d normal/sine mixture
likelihood and v24 scalar tangent likelihood are reused with new generation,
active-population and selection provenance. Runtime adds conditional floors
once. Search/graph conditioning and named-edge association are not independent
image information or a guarantee of future coverage.
"""
from __future__ import annotations
from copy import deepcopy
from types import FunctionType, SimpleNamespace
import numpy as np
from . import calibrate_complete_bi_edge_mixture_v4_20d as LINE
from . import calibrate_archived_complete_tangent_v4_24 as TANGENT

RAW_OBSERVATION_GENERATION='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
OBSERVATION_FAMILY='FULL_PROVENANCE_MULTIGUIDE_ACTUAL_NMS_ORDERED_NATIVE_EDGES_V29'
TANGENT_FAMILY='V29_ACTIVE_ONE_DIRECTION_ARCHIVED_Q_SELECTED_ACTUAL_LINE_TANGENT'
SCHEMA='CAL_MULTIGUIDE_ORDERED_ACTUAL_EDGE_MIXTURE_NOISE_V29'
TANGENT_SCHEMA='CAL_MULTIGUIDE_ACTIVE_ARCHIVED_TANGENT_SCALE_V29'
POPULATION='V29_RUNTIME_ACTIVE_DETERMINISTIC_GRAPH_WINNER_ONE_ACTUAL_LINE_PER_PHYSICAL_EDGE'
SOURCE_POLICIES=('ALL_BI_REPLACE','RGB_NONFALLBACK_PRESERVE')
RECIPE={**deepcopy(LINE.RECIPE),
    'observation_generation':OBSERVATION_FAMILY,
    'raw_observation_generation':RAW_OBSERVATION_GENERATION,
    'applicability':POPULATION,
    'replacement':'v29 image-supported native physical-edge normal replaces q normal once; one direction keeps archived q in selected actual-line tangent; finite ordered nonparallel pair removes full q cost; no new owner preserves v26 measurement state: nonfallback q2D, fallback legacy normal+T4',
    'source_17c_guard_policy':'new full-provenance engineering partial source; any superseded old endpoint/shift guard is explicit, never renamed a passing old17c measurement',
    'search_angle_reference':'per-chain actual generating guide, with immutable ORIGINAL angle retained separately; never interpreted as independent measured direction',
    'group':'one physical native edge; one deterministic graphselected actual TLS line; all other alternatives are archive only',
    'graph_population':'exact deterministic graphselected runtime population sealed before CAL labels; no label-dependent line selection',
    'old_noise_relabelled':False,
    'one_direction_complete_tangent':'separate new active-population gamma model; original q/effective covariance/reliability retained; selected actual measured-line tangent fixed before labels',
    'tangent_total_error_calibrated_by_this_model':False,
    'original_q_covariance_state_or_hardguard_changed_by_noise_fitter':False}
TANGENT_RECIPE={**deepcopy(TANGENT.RECIPE),
    'population':'v29 active one-direction archived B/I q only, sealed before fixed CAL44 reference decode',
    'observation_family':TANGENT_FAMILY,
    'raw_observation_generation':RAW_OBSERVATION_GENERATION,
    'reference':'original fixed-native CAL44 model projection, no GT/error-based selector',
    'tangent':'selected actual measured-line unit tangent; sign has no information; same across both WD trial families',
    'variance':'existing_pixel_floor1px^2 + gamma * (t_selected_actual @ effective_fit_cov @ t_selected_actual)',
    'matched_old_control':'4*a engineering sensitivity computed from the same immutable covariance axis; not claimed to be a saved v29 likelihood',
    'new_population_old_model_relabelled':False}
conditional_covariance_floors=LINE.conditional_covariance_floors
mixture_noise_nll_and_gradient=LINE.mixture_noise_nll_and_gradient
scalar_tangent_nll_and_gradient=TANGENT.scalar_tangent_nll_and_gradient
_component_priors=LINE._component_priors
_validate_support_priors=LINE._validate_support_priors
_digest=LINE.OLD._digest

def recipe_for(source_policy,*,tangent=False):
    if source_policy not in SOURCE_POLICIES:raise ValueError('Explicit declared v29 source policy required')
    return {**deepcopy(TANGENT_RECIPE if tangent else RECIPE),'source_policy':source_policy}


def _clone(function,updates,name):
    result=FunctionType(function.__code__,{**function.__globals__,**updates},name)
    result.__kwdefaults__=deepcopy(function.__kwdefaults__)
    return result


def _family(records,source_policy):
    recipe_for(source_policy)
    records=list(records)
    for row in records:
        if row.get('partition')!='CAL':raise ValueError('CAL only; no TEST/remaining calibration')
        if (row.get('observation_generation')!=OBSERVATION_FAMILY or row.get('noise_population')!=POPULATION
                or row.get('source_policy')!=source_policy or row.get('raw_observation_generation')!=RAW_OBSERVATION_GENERATION):
            raise ValueError('New v29 generation and runtime-active population required')
    return records


def fit_geometry_guided_line_model(records,*,source_policy,**kwargs):
    """Same uncentered likelihood; one component per graphselected edge."""
    fit=_clone(LINE.fit_native_edge_mixture_model,{'SCHEMA':SCHEMA,'RECIPE':recipe_for(source_policy)},'fit_v29_lines')
    records=_family(records,source_policy)
    allowed=set(kwargs.get('fit_ids') or [r['id'] for r in records]);held=kwargs.get('heldout_session')
    for row in records:
        if row['id'] not in allowed or row.get('compatible') is not True or row['session']==held:continue
        seen=set()
        for group in row['groups']:
            pair=tuple(group['physical_native_edge'])
            if pair in seen or len(group['components'])!=1:raise ValueError('One graphselected component per unique physical edge required')
            seen.add(pair);_validate_support_priors(group['components'])
    result=fit(records,**kwargs)
    result.update(observation_generation=OBSERVATION_FAMILY,raw_observation_generation=RAW_OBSERVATION_GENERATION,noise_population=POPULATION,
        source_policy=source_policy,old_fitted_model_relabelled=False,graph_selector_conditioned_population_fitted=True,
        independent_graph_selection_coverage_guaranteed=False,
        scale_training_session_count=len(result['fit_sessions']),single_session_scale_caveat=len(result['fit_sessions'])<2)
    result['model_digest_sha256']=_digest({k:v for k,v in result.items() if k!='model_digest_sha256'})
    return result


def fit_geometry_guided_tangent_model(records,*,source_policy,**kwargs):
    """New population and selected-image-line axis; original q/Σ are unchanged."""
    recipe=recipe_for(source_policy,tangent=True)
    describe=_clone(TANGENT._descriptive,{'RECIPE':recipe},'describe_v29_tangents')
    fit=_clone(TANGENT.fit_archived_complete_tangent_model,{'SCHEMA':TANGENT_SCHEMA,
        'OBSERVATION_FAMILY':TANGENT_FAMILY,'RECIPE':recipe,'_descriptive':describe},'fit_v29_tangents')
    result=fit(_family(records,source_policy),**kwargs)
    result.update(observation_generation=OBSERVATION_FAMILY,raw_observation_generation=RAW_OBSERVATION_GENERATION,noise_population=POPULATION,source_policy=source_policy,
        old_fitted_model_relabelled=False,new_image_independence_claimed=False,
        tangent_axis='SELECTED_ACTUAL_MEASURED_LINE_TANGENT',
        scale_training_session_count=len(result['fit_sessions']),single_session_scale_caveat=len(result['fit_sessions'])<2)
    result['model_digest_sha256']=_digest({k:v for k,v in result.items() if k!='model_digest_sha256'})
    return result


def mixture_noise_parameters(model,*,source_policy):
    export=_clone(LINE.OLD.mixture_noise_parameters,{'SCHEMA':SCHEMA,'RECIPE':recipe_for(source_policy),
        'MIX':SimpleNamespace(NOISE_OBSERVATION_FAMILY=OBSERVATION_FAMILY)},'export_v29_line_noise')
    result=export(model)
    if (model.get('observation_generation')!=OBSERVATION_FAMILY or model.get('noise_population')!=POPULATION
            or model.get('source_policy')!=source_policy or model.get('raw_observation_generation')!=RAW_OBSERVATION_GENERATION):raise ValueError('Bound new v29 model policy/population required')
    result['noise_provenance'].update(noise_population=POPULATION,source_policy=source_policy,
        graph_selector_conditioned_population_fitted=True,independent_graph_selection_coverage_guaranteed=False,
        archived_q_tangent_model_separate=True,calibration_does_not_authenticate_named_edge_identity=True)
    return result


def tangent_variance_parameters(model,*,source_policy):
    export=_clone(TANGENT.tangent_variance_parameters,{'SCHEMA':TANGENT_SCHEMA,
        'OBSERVATION_FAMILY':TANGENT_FAMILY,'RECIPE':recipe_for(source_policy,tangent=True)},'export_v29_tangent_noise')
    result=export(model)
    if (model.get('observation_generation')!=OBSERVATION_FAMILY or model.get('noise_population')!=POPULATION
            or model.get('source_policy')!=source_policy or model.get('raw_observation_generation')!=RAW_OBSERVATION_GENERATION):raise ValueError('Bound new v29 tangent policy/population required')
    return {**result,'source_policy':source_policy,'tangent_axis':'SELECTED_ACTUAL_MEASURED_LINE_TANGENT'}


def calibrated_archived_tangent_variance(model,tangent_unit_xy,original_effective_fit_covariance_px2,*,source_policy):
    parameters=tangent_variance_parameters(model,source_policy=source_policy)
    a=TANGENT._variance_axis(tangent_unit_xy,original_effective_fit_covariance_px2)
    return dict(tangent_variance_px2=parameters['pixel_floor_variance_px2']+parameters['gamma']*a,
        original_variance_axis_px2=a,old_x4_engineering_control_variance_px2=4.*a,**parameters)


def prepared_tangent(record,states,hidden):
    """Gate authoritative state before archive coordinates/covariance fields."""
    k=record.get('native_corner')
    if isinstance(k,(bool,np.bool_))or not isinstance(k,(int,np.integer))or not 0<=int(k)<8:
        raise ValueError('Explicit native index required')
    k=int(k)
    if hidden[k]or states[k]not in ('B','I'):return None
    if record.get('state')!=states[k] or record.get('kind')!='ARCHIVED_Q_SELECTED_MEASURED_LINE_TANGENT':
        raise ValueError('Authoritative B/I and selected measured-line tangent source required')
    q=np.asarray(record['archived_q_xy'],float);t=np.asarray(record['selected_line_tangent_unit_xy'],float)
    cov=np.asarray(record['effective_fit_covariance_px2'],float)
    if q.shape!=(2,)or not np.isfinite(q).all():raise ValueError('Finite archived B/I q required')
    a=TANGENT._variance_axis(t,cov)
    weight=float(record['fit_reliability_weight'])
    if not np.isfinite(weight)or weight<0:raise ValueError('Finite nonnegative fixed reliability required')
    return dict(native_corner=k,state=states[k],source=str(record['source']),
        source_group=TANGENT.DIAG._source_group(record['source']),archived_q_xy=q.tolist(),
        selected_line_tangent_unit_xy=t.tolist(),selected_line_ref=deepcopy(record['selected_line_ref']),effective_fit_covariance_px2=cov.tolist(),
        fit_reliability_weight=weight,original_variance_axis_px2=a,
        exact_saved_old_tangent_variance_px2=4.*a,
        old_x4_field_is_derived_sensitivity_not_claimed_saved_v29_variance=True)
