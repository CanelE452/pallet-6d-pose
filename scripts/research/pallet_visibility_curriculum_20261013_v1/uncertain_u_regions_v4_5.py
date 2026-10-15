"""Pure U disks from immutable raw predictions and a borrowed frozen CAL max.

The CAL model was fitted to initial non-S DIRECT/EXTERNAL localization errors,
not to U coverage. A disk is a spatial hypothesis, never a precise image point.
Initial S is excluded before coordinate indexing. No image or file is read.
"""
from collections.abc import Mapping

import numpy as np

from .truncation_interval_v4_2 import load_truncation_interval_model


SCHEMA='UNCERTAIN_U_REGIONS_BORROWED_CAL_MAX_V4_5'
STATES=frozenset(('S','B','I','O','T','U'))


def _validated_model(value):
    # Mapping only: the shared validator's file-path branch is unreachable.
    if not isinstance(value,Mapping):raise ValueError('Explicit frozen CAL model mapping required; no file loading')
    model=load_truncation_interval_model(value)
    if model.get('eligible_point_count')!=264 or model.get('heldout_session') is not None:
        raise ValueError('The frozen full-CAL264 maximum model is required')
    fit=model.get('fit_ids',[]);compatible=model.get('compatible_ids',[]);excluded=model.get('excluded_native_incompatible_ids',[])
    if (len(fit)!=46 or len(set(fit))!=46 or len(compatible)!=44 or len(set(compatible))!=44
            or len(excluded)!=2 or len(set(excluded))!=2
            or set(compatible)&set(excluded) or set(compatible)|set(excluded)!=set(fit)):
        raise ValueError('Frozen CAL46/compatible44 model population binding required')
    required_false=('initial_S_coordinates_accessed','scale_recomputed',
        'trimming_or_quantile_or_hyperparameter_selection','physical_truncation_certificate',
        'future_coverage_guarantee','runtime_annotation_or_GT_compatibility_inputs')
    if any(model.get(key) is not False for key in required_false) or model.get('empirical_training_coverage_only') is not True:
        raise ValueError('Untrimmed initial-nonS empirical CAL-only model provenance required')
    ratio=float(model['radius_ratio'])
    if ratio<=0:raise ValueError('Positive localization radius required; U cannot become an exact point')
    maximum=model.get('max_error_point',{})
    if maximum.get('normalized_error')!=model['radius_ratio']:
        raise ValueError('Radius must equal the bound untrimmed CAL maximum')
    return model


def build_uncertain_u_regions(runtime,model):
    """Build U ellipses without altering states, raw points, or existing B/I.

Only ``original_raw_points[k]`` for an initial non-S U is indexed. Outside-frame
centers remain outside; no canvas halfplanes or clipping are manufactured.
Missing predictions remain unresolved and receive no pose-based substitute.
"""
    if not isinstance(runtime,Mapping):raise ValueError('Frozen runtime observation mapping required')
    states=runtime.get('states');geometry=runtime.get('initial_geometry',{})
    hidden=geometry.get('initial_self_hidden') if isinstance(geometry,Mapping) else None
    if (not isinstance(states,(list,tuple,np.ndarray)) or len(states)!=8 or any(s not in STATES for s in states)
            or not isinstance(hidden,(list,tuple,np.ndarray)) or len(hidden)!=8
            or any(not isinstance(s,(bool,np.bool_)) for s in hidden)):
        raise ValueError('Eight native states and explicit initial-S gates required')
    frozen=_validated_model(model)
    domains=runtime.get('ot_domains',{})
    if not isinstance(domains,Mapping):raise ValueError('Frozen non-S scale metadata required')
    scale=domains.get('nonself_prediction_scale_px')
    if isinstance(scale,(bool,np.bool_)) or not np.isscalar(scale) or not np.isfinite(scale) or scale<=0:
        raise ValueError('Positive finite frozen non-S prediction scale required')
    radius=float(frozen['radius_ratio'])*float(scale)
    if not np.isfinite(radius) or radius<=0:raise ValueError('Finite nonzero U spatial region required')
    provenance=dict(borrowed_model_schema=frozen['schema'],borrowed_model_sha256=frozen['model_sha256'],
        borrowed_model_fit_partition='CAL',borrowed_model_fit_recipe=frozen['recipe'],
        borrowed_model_eligible_points=264,borrowed_model_native_compatible_frames=44,
        U_specific_coverage_calibrated=False,simultaneous_U_coverage_calibrated=False,
        future_coverage_guaranteed=False,physical_visibility_or_truncation_certificate=False,
        radius_rule='provided_frozen_CAL_radius_ratio * immutable_nonS_prediction_scale_px',
        scale_recomputed=False,center_source='immutable_original_raw_nonS_image_prediction',
        candidate_pose_used_as_center=False,precise_image_point_objective=False,
        canvas_clipping_or_halfplanes_added=False,runtime_GT_inputs=False,image_reads=0,
        original_U_states_preserved=True)
    regions=[];records=[];raw=None;raw_source_checked=False;U_coordinate_reads=0
    for k,state in enumerate(states):
        record=dict(native_corner=k,state=state,region_created=False,coordinate_read=False)
        records.append(record)
        # Both gates precede fetching the raw source or indexing any coordinate.
        if bool(hidden[k]) or state=='S':
            record['reason']='initial_or_declared_S_excluded_before_coordinate_read'
            continue
        if state!='U':
            record['reason']='non_U_state_preserved_without_coordinate_read'
            continue
        if not raw_source_checked:
            raw=runtime.get('original_raw_points');raw_source_checked=True
            if raw is not None:
                try:valid_length=len(raw) in (8,9)
                except TypeError:valid_length=False
                if not valid_length:raise ValueError('Native original_raw_points8/9 source required')
        if raw is None:
            record['reason']='missing_original_raw_prediction_source'
            continue
        record['coordinate_read']=True;U_coordinate_reads+=1
        try:point=np.asarray(raw[k],dtype=float)
        except (TypeError,ValueError,KeyError,IndexError):
            record['reason']='missing_or_malformed_U_raw_prediction'
            continue
        if point.shape!=(2,) or not np.isfinite(point).all() or np.all(point==[-1.,-1.]):
            record['reason']='missing_or_nonfinite_U_raw_prediction'
            continue
        center=point.tolist()
        region=dict(schema=SCHEMA,native_corner=k,native_id=k,state='U',shape='ellipse',
            center_xy=center,half_axes_px=[radius,radius],angle_deg=0.,halfplanes=[],
            independent_precise_support=False,source='raw_U_prediction_plus_borrowed_CAL_nonS_untrimmed_max',
            candidate_pose_inputs_used=False,GT_coordinates_used=False,
            canvas_clipped=False,U_state_preserved=True,provenance=dict(provenance))
        regions.append(region)
        record.update(region_created=True,reason='unclipped_raw_center_U_circle',
            center_xy=center,radius_px=radius,exact_point_measurement=False,
            borrowed_U_coverage_guarantee=False)
    return dict(schema=SCHEMA,u_regions=regions,records=records,S_coordinate_reads=0,image_reads=0,
        U_coordinate_reads=U_coordinate_reads,radius_px=radius,
        radius_ratio=float(frozen['radius_ratio']),nonself_prediction_scale_px=float(scale),
        states_preserved=list(states),provenance=provenance)


build_u_regions=build_uncertain_u_regions
