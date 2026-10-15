"""Opt-in cached candidate admission: observed q anchor and fixed signed ray.

This module creates no measurement and is not connected to the frozen V30
run. It retains V30's43px infinite-line normal association and adds metadata
anchor provenance and immutable direction tests before deduplication. A model
anchor is never moved to q or presented as a fresh q-anchored NMS measurement.
Raw chains, indices, heap/memo/ordered-score arithmetic stay unchanged. This
NEW selection population requires separate CAL46 calibration and evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import FunctionType
import numpy as np

from . import observed_bi_line_association_v4_30 as F

P,G,B,M=F.P,F.G,F.B,F.M
PARENT_ASSOCIATION_SHA256='02cca91ff2076a17840a33892ef85e32d80ff821be824d205f1d09cea2387a2d'
SELECTION_PROFILE='OBSERVED_BI_Q_QUERY_ANCHOR_AND_FIXED_SIGNED_NATIVE_RAY_PLUS_NORMAL_REACH_V31'
OBSERVATION_GENERATION='CACHED_FULL_PROVENANCE_V29_WITH_OBSERVED_ANCHOR_NATIVE_RAY_SELECTION_V31'
ANCHOR_METADATA_TOLERANCE_PX=1e-6
MINIMUM_REFERENCE_LENGTH_PX=10.
ORTHOGONAL_NUMERIC_TOLERANCE=1e-12
RECIPE=dict(selection_profile=SELECTION_PROFILE,observation_generation=OBSERVATION_GENERATION,
    parent_raw_observation_generation=P.OBSERVATION_GENERATION,parent_selection_profile=F.SELECTION_PROFILE,
    normal_gate='unchanged V30 all eligible original BI q to infinite TLS within existing cfg.normal_search_radius_px43',
    actual_query_anchor='must equal immutable source BI q componentwise within1e-6 metadata tolerance; not a spatial correction cap',
    disallowed_query_anchor_kind='MODEL_PROJECTION_SEARCH_ONLY_BAND even when numerically coincident with q',
    both_BI_direction='positive hemisphere of immutable qmate-qsource, length>=existing10px',
    unknown_mate_direction='fixed ORIGINAL initial model projection mate-source only; O/T/U observation coordinates never read',
    short_or_orthogonal_reference='abstain; no alternate pose-selected direction rescue',
    normal_line_sign='unoriented TLS sign reversal does not change admission',
    graph_hook='one private continue after frozen fresh node and before dedup; same R30 heap/memo/ordered scorer',
    cached_query_anchor_or_pixels_not_relocated=True,raw_chain_indices_and_provenance_unchanged=True,
    new_precise_KP_or_pose_adoption=False,new_RGB_NMS_TLS_NN_posefit_or_labels=0,
    old_V29_or_V30_noise_validated_for_new_population=False,
    required_before_adoption='separate CAL46 inventory/fixed44 applicable noise population and uniform primary evaluation',
    finite_alongline_identity_certified=False)


def _ids(node):
    edge=tuple(node['native_edge']);k=node['native_corner'];mate=node['native_neighbor']
    values=(*edge,k,mate)
    if (len(edge)!=2 or any(isinstance(v,(bool,np.bool_))or not isinstance(v,(int,np.integer))or not 0<=v<8 for v in values)
            or k==mate or set(edge)!={k,mate}):
        raise ValueError('Strict directed source/mate IDs must match two native endpoints')
    return int(k),int(mate),list(map(int,edge))


def assess_observed_anchor_native_ray_association(node,points,states,initial_self_hidden,cfg,*,geometry):
    """Pure admission; state gates precede q/line/reference/model coordinates."""
    k,mate,edge=_ids(node)
    base=dict(native_edge=edge,source_native_corner=k,source_native_neighbor=mate,
        anchor_metadata_tolerance_px=ANCHOR_METADATA_TOLERANCE_PX,
        minimum_reference_length_px=MINIMUM_REFERENCE_LENGTH_PX,
        unknown_S_U_observed_coordinates_read=0,pose_selected_orientation=False,
        actual_measurement_or_anchor_relocation=False)
    def reject(reason,**values):return dict(base,passes=False,reason=reason,**values)
    if any(initial_self_hidden[i]or states[i]=='S'for i in edge):
        return reject('initial_S_gate_before_line_q_or_reference_payload')
    if states[k]not in('B','I')or k not in points:
        return reject('source_anchor_must_be_original_eligible_BI_before_payload')
    if states[mate]not in('B','I','O','T','U'):
        raise ValueError('Authoritative original mate state required')
    if states[mate]in('B','I')and mate not in points:
        return reject('both_BI_mate_missing_immutable_measurement')
    normal=F.assess_observed_BI_line_association(node,points,states,initial_self_hidden,cfg)
    if not normal['passes']:
        return reject('parent_V30_original_BI_normal_gate_failed',normal_association=normal)
    q=np.asarray(points[k]['record']['xy'],float)
    line=node.get('line_record');reference=line.get('guide_reference')if isinstance(line,dict)else None
    if not isinstance(reference,dict):
        return reject('missing_original_actual_query_reference',normal_association=normal)
    anchor_kind=reference.get('search_anchor_kind');guide_kind=reference.get('guide_kind')
    if anchor_kind=='MODEL_PROJECTION_SEARCH_ONLY_BAND':
        return reject('model_projection_query_is_not_an_observed_q_query',normal_association=normal,
            query_anchor_kind=anchor_kind,guide_kind=guide_kind)
    allowed=(anchor_kind=='OBSERVED_BI_REFERENCE_BAND'and guide_kind in(
        'PHYSICAL_VP_THROUGH_OBSERVED_BI_ANCHOR','MEASURED_BI_ENDPOINT_DIRECTION'))
    # Frozen base17c did not add search_anchor_kind. Its actual query is still
    # explicitly archived as q in actual_search_anchor_xy.
    allowed=allowed or(anchor_kind is None and guide_kind=='FROZEN17C_ORIGINAL_MODEL_SEED')
    if not allowed:
        return reject('unrecognized_observed_query_anchor_provenance',normal_association=normal,
            query_anchor_kind=anchor_kind,guide_kind=guide_kind)
    try:
        anchor=np.asarray(reference['actual_search_anchor_xy'],float)
        direction=np.asarray(reference['actual_search_direction_xy'],float)
    except(KeyError,TypeError,ValueError):
        return reject('missing_or_invalid_actual_query_anchor_direction',normal_association=normal)
    if (anchor.shape!=(2,)or direction.shape!=(2,)or not np.isfinite([*anchor,*direction]).all()
            or np.linalg.norm(direction)<=ORTHOGONAL_NUMERIC_TOLERANCE):
        return reject('nonfinite_or_degenerate_actual_query_anchor_direction',normal_association=normal)
    difference=float(np.max(np.abs(anchor-q)))
    anchor_audit=dict(query_anchor_kind=anchor_kind,guide_kind=guide_kind,
        original_source_q_xy=q.tolist(),actual_query_anchor_xy=anchor.tolist(),
        metadata_anchor_max_absolute_difference_px=difference,normal_association=normal)
    if difference>ANCHOR_METADATA_TOLERANCE_PX:
        return reject('actual_query_anchor_does_not_equal_original_source_BI_q',**anchor_audit)
    # Some synthetic/legacy fresh nodes omit the flat alias. When present it
    # must match the same archived actual query, never be silently replaced.
    if 'generating_guide_direction_xy'in line:
        alias=np.asarray(line['generating_guide_direction_xy'],float)
        if alias.shape!=(2,)or not np.array_equal(alias,direction):
            return reject('actual_query_direction_alias_disagrees',**anchor_audit)
    if states[mate]in('B','I'):
        if mate not in points:
            return reject('both_BI_mate_missing_immutable_measurement',**anchor_audit)
        # This field is reached only for an authoritative BI mate.
        other=np.asarray(points[mate]['record']['xy'],float)
        reference_delta=other-q
        reference_source='IMMUTABLE_OBSERVED_BI_MATE_MINUS_SOURCE'
    else:
        # Initial nonS MODEL projection only; never points[mate], an O/T/U q,
        # a trial pose, a selected pose or a freshly chosen guide direction.
        try:
            original=geometry['initial_projection_xy']
            start=np.asarray(original[k],float);other=np.asarray(original[mate],float)
        except(KeyError,IndexError,TypeError,ValueError):
            return reject('missing_original_native_model_ray',**anchor_audit)
        if start.shape!=(2,)or other.shape!=(2,):
            return reject('invalid_original_native_model_ray',**anchor_audit)
        reference_delta=other-start
        reference_source='ORIGINAL_INITIAL_MODEL_NATIVE_RAY_FOR_UNKNOWN_MATE'
    if reference_delta.shape!=(2,)or not np.isfinite(reference_delta).all():
        return reject('nonfinite_fixed_native_reference_ray',**anchor_audit)
    length=float(np.linalg.norm(reference_delta))
    ray_audit=dict(**anchor_audit,reference_source=reference_source,
        fixed_reference_delta_xy=reference_delta.tolist(),fixed_reference_length_px=length)
    if length<MINIMUM_REFERENCE_LENGTH_PX:
        return reject('fixed_native_reference_ray_shorter_than_existing10px',**ray_audit)
    direction=direction/np.linalg.norm(direction);fixed=reference_delta/length
    cosine=float(direction@fixed)
    ray_audit.update(actual_query_direction_unit_xy=direction.tolist(),fixed_reference_direction_unit_xy=fixed.tolist(),
        signed_direction_cosine=cosine)
    if abs(cosine)<=ORTHOGONAL_NUMERIC_TOLERANCE:
        return reject('fixed_native_direction_orthogonal_ambiguity',**ray_audit)
    if cosine<0.:
        return reject('actual_query_direction_opposes_fixed_native_ray',**ray_audit)
    return dict(base,passes=True,reason='original_q_query_anchor_and_fixed_native_ray_pass',**ray_audit,
        TLS_normal_sign_is_not_native_ray_sign=True,finite_alongline_identity_certified=False)


def rank_observed_anchor_native_ray_graph(chain_observation,geometry,states,visible_measurements,*,
        image_binding,configuration=None,pose_guides=None,pallet_mask=None,visibility_context=None,
        maximum_cache_entries=4096,intersection_cache_limit=4096):
    """Same frozen R30 rank with one combined admission before deduplication."""
    if hashlib.sha256(Path(F.__file__).read_bytes()).hexdigest()!=PARENT_ASSOCIATION_SHA256:
        raise ValueError('Exact frozen V30 private admission-hook source required')
    ledger=[]
    def check(node,points,current_states,hidden,cfg):
        verdict=assess_observed_anchor_native_ray_association(node,points,current_states,hidden,cfg,geometry=geometry)
        ledger.append(dict(line_name=node['name'],node_provenance=list(node['provenance']),**verdict))
        return verdict['passes']
    namespace=dict(G.rank_joint_geometry_line_graph.__globals__)
    namespace.update(_association_pass=check,_node=P._fresh_node,
        _score_graph=M.make_ordered_scorer(maximum_cache_entries=maximum_cache_entries),
        _bounded_beam_step=B.bounded_beam_step,
        _new_intersection_cache=lambda:B.BoundedIntersectionCache(intersection_cache_limit))
    rank=FunctionType(F._filtered_rank_code(),namespace)
    rank.__kwdefaults__=deepcopy(G.rank_joint_geometry_line_graph.__kwdefaults__)
    graph=rank(chain_observation,geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides,pallet_mask=pallet_mask,visibility_context=visibility_context)
    graph['joint_alternatives']=[a for a in graph['joint_alternatives']if not a['ordered_support_unavailable']]
    return dict(selection_profile=SELECTION_PROFILE,observation_generation=OBSERVATION_GENERATION,
        parent_raw_observation_generation=P.OBSERVATION_GENERATION,parent_selection_profile=F.SELECTION_PROFILE,
        graph=graph,node_association_ledger=ledger,
        rejected_node_count=sum(not r['passes']for r in ledger),candidate_nodes_checked=len(ledger),
        raw_archive_unchanged=True,original_chain_indices_and_node_provenance_preserved=True,
        old_V29_or_V30_noise_validated_for_this_population=False,
        new_CAL46_noise_and_uniform_validation_required=True,
        new_RGB_NMS_TLS_posefit_labels_or_precise_KPs=0,operational_opt_in_only=True)
