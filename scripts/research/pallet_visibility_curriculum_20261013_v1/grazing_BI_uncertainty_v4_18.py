"""Proposed pure cached B/I grazing->U control, without changing old modules.

Only current B/I in the ORIGINAL initial geometry's existing tolerance band
are withheld. Fixed S and all previous U/O/T survive. No image, coordinate,
label or file is read by the geometry decision or precise-factor filter.
The optional U factory reads only newly unresolved initial-nonS raw centers.
Historical cached local queries are disclosed, never retrospectively denied.
This module has not been adopted or pose-evaluated.
"""
from collections.abc import Mapping
import numpy as np

from . import joint_ot_pnp as V
from . import geometric_visibility_uncertainty_v4_9 as OLD
from . import uncertain_u_regions_v4_5 as U


def apply_grazing_BI_uncertainty(geometry, states, *, geometry_tolerance_deg):
    states=[V._state(s)for s in states]
    if len(states)!=8 or any(s not in('S','B','I','O','T','U')for s in states):
        raise ValueError('Eight authoritative native states required')
    # Reuse the frozen original-geometry calculation. This temporary list is
    # solely a math-query gate: O/T/U are restored exactly in the output.
    query_states=[s if s in('S','B','I')else'U'for s in states]
    measured=OLD.apply_geometric_visibility_uncertainty(geometry,query_states,
        geometry_tolerance_deg=geometry_tolerance_deg)
    new_ids=list(measured['new_U_native_ids']);result=list(states)
    records=[]
    for k,state in enumerate(states):
        audit={**measured['records'][k],'original_state':state,'state':state}
        if state not in('S','B','I'):
            audit['reason']='existing_U_O_T_preserved_before_geometry_or_observation_fields'
        if k in new_ids:result[k]='U';audit['state']='U'
        records.append(audit)
    return dict(schema='INITIAL_GRAZING_BI_TO_U_PROPOSAL_V4_18',states=result,records=records,
        new_U_native_ids=new_ids,initial_S_native_ids=measured['initial_S_native_ids'],
        geometry_tolerance_deg=measured['geometry_tolerance_deg'],
        predicate='abs(maximum_incident_outward_normal_cosine)<=sin(existing_geometry_tolerance_deg)',
        initial_S_unchanged=True,previous_U_O_T_unchanged=True,
        original_initial_geometry_only=True,not_current_candidate_geometry=True,
        conditional_visibility_uncertainty_not_physical_SELF_certificate=True,
        raw_coordinate_reads=0,RGB_queries=0,GT_inputs=False,
        historical_cached_local_queries_may_have_occurred=True,
        pose_evaluation_or_default_adoption=False)


def _records(values):
    if isinstance(values,Mapping):
        if 'native_corner'in values or'native_id'in values:return[values]
        raise ValueError('Use explicit native-ID record sequences, not keyed records')
    return values


def filter_grazing_BI_reuse(visible,boundary_observations,states,new_U_native_ids,*,initial_self_hidden):
    states=[V._state(s)for s in states];hidden=list(initial_self_hidden)
    if len(states)!=8 or len(hidden)!=8 or any(bool(hidden[k])!=(states[k]=='S')for k in range(8)):
        raise ValueError('Initial S is fixed before factor fields')
    uncertain=set(new_U_native_ids)
    if any(isinstance(k,bool)or not isinstance(k,(int,np.integer))or not 0<=k<8 or states[k]!='U'for k in uncertain):
        raise ValueError('New uncertainty IDs must be authoritative U')
    retained=[];removed=[]
    for record in _records(visible):
        k=V._id(record)
        if hidden[k]or states[k]not in('B','I'):
            removed.append(k);continue
        retained.append(record)
    partial=[];discarded=[];restored=[]
    for record in _records(boundary_observations or []):
        k=V._id(record)
        if hidden[k]or states[k]not in('B','I'):
            discarded.append(dict(native_corner=k,reason='non_BI_receiver_before_line_or_archive_fields'))
            continue
        lines=[]
        for index,line in enumerate(record['lines']):
            provenance=line.get('provenance',{})
            source=provenance.get('source_query_native_corner')
            origin='explicit_source_query_native_corner'
            if source is None:
                source=provenance.get('line_record',{}).get('native_corner',k)
                origin='actual_line_record_anchor_or_receiver_fallback'
            if isinstance(source,bool)or not isinstance(source,(int,np.integer))or not 0<=source<8:
                raise ValueError('Explicit valid native source-query identity required')
            if bool(hidden[source])or source in uncertain:
                discarded.append(dict(native_corner=k,line_index=index,source_query_native_corner=int(source),
                    source_identity_origin=origin,reason='line_queried_at_initial_S_or_new_U_not_reused'))
                continue
            actual=provenance.get('line_record',{})
            mate=actual.get('native_neighbor',provenance.get('native_neighbor'))
            # Frozen whole-edge searches explicitly used both old B/I image
            # endpoints. A newly uncertain mate must not survive indirectly
            # as a precise endpoint guide. Local unknown-mate searches used
            # only model direction and remain admissible at a retained BI.
            endpoint_used=(actual.get('whole_edge_search')is True
                or bool(actual.get('neighbor_coordinate_reads',0))
                or bool(actual.get('neighbor_corner_queries',0)))
            if mate in uncertain and endpoint_used:
                discarded.append(dict(native_corner=k,line_index=index,native_neighbor=mate,
                    reason='historical_full_edge_used_new_U_precise_endpoint_not_reused'))
                continue
            lines.append(line)
        if lines:
            partial.append(record if len(lines)==len(record['lines'])else{**record,'lines':lines})
        else:
            discarded.append(dict(native_corner=k,reason='all_partial_lines_withheld'))
            restored.append(k)
    return dict(visible=retained,boundary_observations=partial,
        removed_visible_native_ids=removed,discarded_partial_records_or_lines=discarded,
        retained_BI_original_complete_cost_restored_if_no_partial_lines=restored,
        retained_measurements_noise_reliability_and_archives_recomputed=False,
        retained_U_mate_model_geometry_only_allowed=True,new_U_exact_point_or_line_costs=0,
        raw_coordinate_reads=0,RGB_queries=0)


def append_grazing_U_regions(geometry,states,new_U_native_ids,*,raw_native_points,
        nonself_prediction_scale_px,localization_model,existing_u_regions):
    """Reuse the frozen CAL264 factory, query only newly uncertain centers.

    A private factory query gate suppresses preexisting U coordinates: those
    exact domain objects are kept rather than rebuilt. It is not a classifier
    output. Actual returned states remain the caller's authoritative states.
    """
    states=[V._state(s)for s in states];hidden=geometry['initial_self_hidden'];new=set(new_U_native_ids)
    if len(states)!=8 or len(hidden)!=8 or any(bool(hidden[k])!=(states[k]=='S')for k in range(8)):
        raise ValueError('Eight fixed initial-S gates required')
    if any(isinstance(k,bool)or not isinstance(k,(int,np.integer))or not 0<=k<8 or states[k]!='U'or hidden[k]for k in new):
        raise ValueError('Only newly unresolved nonS native vertices may get a circle')
    old=[];seen=set()
    for region in existing_u_regions:
        k=V._id(region)
        if hidden[k]or states[k]!='U'or k in new or k in seen:
            raise ValueError('Distinct unchanged preexisting U regions required')
        old.append(region);seen.add(k)
    query_gate=['S'if hidden[k]else'U'if k in new else'B'for k in range(8)]
    result=U.build_u_regions(dict(states=query_gate,initial_geometry=dict(initial_self_hidden=hidden),
        original_raw_points=raw_native_points,ot_domains=dict(nonself_prediction_scale_px=nonself_prediction_scale_px)),localization_model)
    created={V._id(r)for r in result['u_regions']}
    return dict(schema='APPEND_NEW_GRAZING_U_REGIONS_V4_18',states_preserved=states,
        u_regions=old+result['u_regions'],new_u_regions=result['u_regions'],
        new_region_records=[r for r in result['records']if r['native_corner']in new],
        missing_new_U_raw_native_ids=sorted(new-created),existing_U_regions_exact_objects_preserved=True,
        query_gate_is_private_not_a_state_classification=True,
        new_U_raw_coordinate_reads=result['U_coordinate_reads'],S_coordinate_reads=0,RGB_queries=0,
        radius_px=result['radius_px'],radius_ratio=result['radius_ratio'],provenance=result['provenance'],
        U_point_or_line_costs=0,U_canvas_or_visibility_guards_added=False,
        U_specific_coverage_calibrated=False)
