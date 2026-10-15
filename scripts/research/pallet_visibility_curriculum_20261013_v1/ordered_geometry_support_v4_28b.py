"""Cached-only ordered image-line support proposals; never adopted corner/PnP.

The frozen v28 graph is reused through private function globals. This adds a
conditional endpoint-order check, retaining measured line geometry, all old
angle guards, and every original candidate. An inlier lies on the k->mate side
of a proposed B/I junction if its signed measured-tangent coordinate is >=-1px.
At least the existing0.6 fraction must pass. BI endpoint orientation grounds
the sign when available; opposite model guides cannot rescue a reversed end.

Only an exact saved insufficient_along_edge_span TLS rejection can become an
explicit engineering PARTIAL_LINE_PROPOSAL. No gradient/NMS/occupancy values
are fabricated. No candidate TLS is refit and no pixel, reference, NN or
optimizer runs; the inherited rounded diagnostic may refit cached support.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from types import FunctionType
import numpy as np

from . import joint_geometry_guided_lines_v4_28 as OLD


SCHEMA='ORDERED_CACHED_GEOMETRY_LINE_SUPPORT_V4_28B'
SIDE_MARGIN_PX=1.
PARTIAL_GRADE='PARTIAL_LINE_PROPOSAL'
RECIPE=dict(schema=SCHEMA,side_margin_px=SIDE_MARGIN_PX,
    minimum_fraction='unchanged original minimum_inlier_fraction0.6',
    angle_guard='unchanged line direction compatibility12deg; intersection sine0.25 separately audited',
    sign='measured BI mate direction when available; compatible fixed geometry guide alternatives otherwise',
    wrong_endpoint='fewer than0.6 actual inliers on k-to-mate side => junction combination unavailable',
    partial='sole insufficient_along_edge_span rejection, all other saved count/fraction/RMS/angle guards pass',
    missing_RGB_arrays='not fabricated; explicit unknown-quality engineering proposal penalty',
    production_17c_guards_changed=False,full_corner_or_PnP_adoption=False,
    local_S_U_coordinate_or_pixel_queries=0,U_junctions=0,new_RGB_NMS_candidate_TLS_NN_CAL_or_posefit_calls=0,
    cached_support_rounded_diagnostic_TLS_refit_allowed=True,
    calibrated_confidence_or_likelihood=False,user_selected_two_case_posthoc_only=True)


def _compatible_directions(k,mate,p,normal,points,guide_bank,cfg):
    tangent=np.array([-normal[1],normal[0]])
    measured=None
    if k in points and mate in points:
        delta=np.asarray(points[mate]['record']['xy'])-np.asarray(points[k]['record']['xy'])
        if np.linalg.norm(delta)>=10.:
            measured=delta/np.linalg.norm(delta)
    records=[]
    if measured is not None:
        records.append(dict(source='MEASURED_BI_ENDPOINT_ORDER',direction=measured))
    edge=sorted([k,mate])
    for guide in guide_bank['guide_proposals']:
        if guide['native_edge']!=edge:continue
        sign=1. if guide['native_corner']==k else -1.
        reference=np.asarray(guide['direction_xy'])*sign
        if guide.get('vanishing_point_h') is None:
            direction=reference/np.linalg.norm(reference)
        else:
            try:direction=OLD.vanishing_direction_at(guide['vanishing_point_h'],p,sign_reference=reference)
            except ValueError:continue
        # Endpoint positions may be noisy normal to the edge; the orientation
        # hemisphere is still measured. Do not allow a coarse pose to invert it.
        if measured is not None and direction@measured<=0:continue
        records.append(dict(source=guide['guide_id'],direction=direction))
    unique={}
    diagnostics=[]
    for r in records:
        d=np.asarray(r['direction'])
        cosine=float(tangent@d)
        angle=float(np.degrees(np.arccos(np.clip(abs(cosine),0,1))))
        compatible=angle<=cfg['maximum_direction_change_deg']+1e-9
        diagnostics.append(dict(source=r['source'],angle_to_measured_line_deg=angle,
            line_angle_compatible=compatible,measured_BI_hemisphere_used=measured is not None))
        if compatible:
            orientation=1 if cosine>=0 else -1
            unique.setdefault(orientation,[]).append(r['source'])
    return tangent,unique,diagnostics,measured is not None


def _support(k,mate,p,node,points,guide_bank,cfg):
    normal=np.asarray(node['normal'],float)
    actual=np.asarray(node['actual_inliers'],float)
    tangent,orientations,angles,measured=_compatible_directions(k,mate,p,normal,points,guide_bank,cfg)
    alternatives=[]
    for sign,sources in sorted(orientations.items()):
        ordered_tangent=tangent*sign
        distances=(actual-p)@ordered_tangent
        fraction=float(np.mean(distances>=-SIDE_MARGIN_PX))
        alternatives.append(dict(direction_sources=sources,ordered_tangent_xy=ordered_tangent.tolist(),
            supported_fraction=fraction,passes=fraction+1e-12>=cfg['minimum_inlier_fraction'],
            minimum_signed_support_px=float(distances.min()),maximum_signed_support_px=float(distances.max()),
            actual_inlier_count=len(actual)))
    passing=[a for a in alternatives if a['passes']]
    return dict(native_corner=k,native_neighbor=mate,line_name=node['name'],
        passes=bool(passing),side_margin_px=SIDE_MARGIN_PX,minimum_fraction=cfg['minimum_inlier_fraction'],
        measured_BI_mate_order_used=measured,orientation_ambiguous_across_model_guides=len(orientations)>1,
        angle_compatibility=angles,directed_support_alternatives=alternatives,
        best_supported_fraction=max([a['supported_fraction'] for a in alternatives],default=0.),
        reason='ordered_actual_support_pass' if passing else
            'no_angle_compatible_order_direction' if not alternatives else 'actual_support_is_on_wrong_native_endpoint_side',
        native_identity_or_precise_corner_certified=False)


def assess_ordered_incident_support(native_corner,native_neighbor,junction_xy,line_node,*,
        geometry,states,visible_measurements,image_binding,pose_guides=None,configuration=None):
    """Pure public audit with authoritative B/I and initial-S gates first."""
    for value in (native_corner,native_neighbor):
        if isinstance(value,(bool,np.bool_))or not isinstance(value,(int,np.integer))or not 0<=value<8:
            raise ValueError('Strict native integer0..7 IDs required')
    k,mate=int(native_corner),int(native_neighbor)
    hidden=geometry['initial_self_hidden']
    if hidden[k] or hidden[mate] or states[k]not in('B','I') or states[mate]=='S':
        return dict(passes=False,reason='authoritative_non_BI_or_initial_S_gate_before_fields',excluded=True)
    if mate not in OLD.NEIGHBORS[k]:raise ValueError('Physical incident native edge required')
    cfg,_=OLD._config(configuration)
    points=OLD.NMS.G._eligible_points(visible_measurements,states,hidden,geometry['image_hw'])
    if k not in points:return dict(passes=False,reason='no_eligible_BI_anchor',excluded=True)
    guides=OLD.prepare_joint_geometry_guides(geometry,states,visible_measurements,
        image_binding=image_binding,pose_guides=pose_guides,configuration=configuration)
    p=OLD._finite(junction_xy,(2,),'candidate B/I junction')
    return _support(k,mate,p,line_node,points,guides,cfg)


def _promote_span_only(record,chain,cfg):
    if chain.get('reason')!='insufficient_along_edge_span':return None
    required=('normal','offset','observed_points','inlier_mask','inlier_count','inlier_fraction',
              'inlier_span_fraction','inlier_rms_px','angle_change_deg','observation_confidence')
    if any(key not in chain for key in required):return None
    normal,offset,actual=OLD._actual_line(chain)
    confidence=np.asarray(chain['observation_confidence'],float)
    observed=np.asarray(chain['observed_points'],float)
    if confidence.shape!=(len(observed),) or not np.isfinite(confidence).all() or np.any(confidence<0):return None
    values=np.array([chain['inlier_fraction'],chain['inlier_span_fraction'],chain['inlier_rms_px'],chain['angle_change_deg']],float)
    if (not np.isfinite(values).all() or not np.isfinite(offset)
            or len(actual)!=chain['inlier_count'] or len(actual)<cfg['minimum_inlier_count']
            or not cfg['minimum_inlier_fraction']<=values[0]<=1.
            or not 0<=values[1]<cfg['minimum_inlier_span_fraction']
            or not 0<=values[2]<=cfg['maximum_line_rms_px']
            or not 0<=values[3]<=cfg['maximum_direction_change_deg']):return None
    residual_rms=float(np.sqrt(np.mean((actual@normal+offset)**2)))
    if abs(residual_rms-values[2])>1e-7:return None
    if abs(float(np.mean(np.asarray(chain['inlier_mask'],bool)))-values[0])>1e-9:return None
    tangent=np.array([-normal[1],normal[0]])
    span=float(np.ptp(actual@tangent))
    minimum_span=float(cfg['along_start_px']+cfg['boundary_context_px'])
    if span<minimum_span:return None
    # Parent edge guarantees image NMS/RGB occupancy gates preceded fitting.
    if record.get('measurement_source')!='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT':return None
    out=deepcopy(chain)
    out.update(proposal_grade=PARTIAL_GRADE,
        measurement_source='CACHED_ACTUAL_NMS_TLS_PARTIAL_SUPPORT_PROPOSAL_V28B',
        cached_TLS_rejection_reason=chain['reason'],production_17c_accepted=False,
        RGB_gradient_lattice_provenance_complete=False,full_2D_corner_measurement=False,
        actual_points_and_TLS_coefficients_unchanged=True,not_newly_measured_or_refit=True,
        actual_finite_inlier_span_px=span,minimum_engineering_partial_span_px=minimum_span,
        old_corrected_endpoint_guard_status='not_evaluated_after_span_reject',
        conditional_parent_mask_context_kind=record.get('mask_context_kind'),
        parent_RGB_NMS_context_gate_provenance='frozen parent edge record and source binding; missing per-sample arrays remain missing')
    return out


def _proposal_rows(observation,states,hidden,cfg):
    rows=[];ledger=[]
    for index,record in enumerate(observation['edge_records']):
        k,mate=int(record['native_corner']),int(record['native_neighbor'])
        # Do not copy/read rejected line fields on excluded S/U anchors.
        if hidden[k] or hidden[mate] or states[k]not in('B','I') or states[mate]=='S':
            rows.append(dict(native_corner=k,native_neighbor=mate,supported=False,chains=[]));continue
        row=dict(record);row['chains']=list(record.get('chains',[]))
        for ci,chain in enumerate(record.get('rejected_chains',[])):
            partial=_promote_span_only(record,chain,cfg)
            if partial is None:continue
            row['chains'].append(partial)
            ledger.append(dict(parent_edge_record=index,parent_rejected_chain=ci,native_edge=sorted([k,mate]),
                proposal_grade=PARTIAL_GRADE,source_span_fraction=chain['inlier_span_fraction'],
                old_minimum_span_fraction=cfg['minimum_inlier_span_fraction'],actual_inliers=chain['inlier_count'],
                cached_TLS_rejection_reason='insufficient_along_edge_span',production_guard_changed=False))
        rows.append(row)
    return dict(observation,edge_records=rows),ledger


def _ordered_score(selected,points,states,guide_bank,cfg,graph_cfg):
    score=OLD._score_graph(selected,points,states,guide_bank,cfg,graph_cfg)
    nodes=[n for n in selected if n is not None]
    checks=[];angle_checks=[]
    for k in sorted(points):
        if states[k]not in('B','I'):continue
        incident=[n for n in nodes if k in n['native_edge']]
        for a,b in combinations(incident,2):
            key=tuple(sorted([a['name'],b['name']]))
            cache=guide_bank['_cached_intersections']
            if key not in cache:
                cache[key]=OLD._intersection((a['normal'],a['offset']),(b['normal'],b['offset']),cfg['minimum_intersection_sine'])
            p,sine=cache[key]
            angle_checks.append(dict(native_corner=k,line_names=[a['name'],b['name']],
                sine=sine,minimum_sine=cfg['minimum_intersection_sine'],passes=p is not None))
            if p is None:continue  # Near-parallel remains a line/range hypothesis, never a junction.
            for node in (a,b):
                mate=next(i for i in node['native_edge'] if i!=k)
                checks.append(_support(k,mate,p,node,points,guide_bank,cfg))
    failed=[c for c in checks if not c['passes']]
    score.update(parent_v28_engineering_energy=score['total_engineering_energy'],
        ordered_support_unavailable=bool(failed),ordered_support_checks=checks,
        unchanged_intersection_angle_checks=angle_checks,
        directed_support_is_separate_from_angle_conditioning=True)
    if failed:score['total_engineering_energy']=float('inf')
    return score


def _graded_node(record,chain,reference_index,cfg):
    node=OLD._node(record,chain,reference_index,cfg)
    grade=chain.get('proposal_grade','FULL_LENGTH_IMAGE_LINE_PROPOSAL')
    node.update(proposal_grade=grade,production_17c_accepted=grade!=PARTIAL_GRADE,
        RGB_gradient_lattice_provenance_complete=grade!=PARTIAL_GRADE,
        unknown_RGB_quality_proposal_penalty=.5 if grade==PARTIAL_GRADE else None)
    if grade==PARTIAL_GRADE:
        node.update(old_corrected_endpoint_normal_guard_pass=None,
            old_corrected_endpoint_guard_status='not_evaluated_after_span_reject',
            actual_finite_inlier_span_px=chain['actual_finite_inlier_span_px'],
            unknown_local_mask_transition_engineering_fraction=.5,
            actual_points_and_TLS_coefficients_unchanged=True)
    else:
        node['production_17c_accepted']=bool(node['old_corrected_endpoint_normal_guard_pass'])
    return node


def rank_ordered_geometry_line_graph(chain_observation,geometry,states,visible_measurements,*,
        image_binding,configuration=None,pose_guides=None,pallet_mask=None,visibility_context=None):
    cfg,_=OLD._config(configuration)
    states=[OLD.NMS.V3._state(s)for s in states]
    hidden=geometry['initial_self_hidden']
    augmented,ledger=_proposal_rows(chain_observation,states,hidden,cfg)
    namespace=dict(OLD.rank_joint_geometry_line_graph.__globals__)
    namespace.update(_score_graph=_ordered_score,_node=_graded_node)
    rank=FunctionType(OLD.rank_joint_geometry_line_graph.__code__,namespace)
    graph=rank(augmented,geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides,pallet_mask=pallet_mask,visibility_context=visibility_context)
    graph['joint_alternatives']=[r for r in graph['joint_alternatives'] if not r['ordered_support_unavailable']]
    graph.update(schema=SCHEMA,ordered_support_recipe=deepcopy(RECIPE),partial_cached_TLS_promotions=ledger,
        source_v28_observation_changed=False,new_pixel_NMS_candidate_TLS_NN_pose_or_GT_queries=0,
        cached_support_rounded_diagnostic_TLS_refit_allowed=True,
        full_KP_or_PnP_adoption=False)
    return graph
