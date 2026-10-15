"""Pure v29 selected-line provenance, normal replacement and CAL population.

This authenticates actual cached RGB/NMS/TLS samples against their generating
search guide, rather than pretending that a new multiguide line passed the old
ORIGINAL-guide/old-q shift guards. No pixels, poses, labels or files are read.
"""
from __future__ import annotations
from copy import deepcopy
from itertools import combinations
import numpy as np

from . import geometry_guided_edge_problem_v4_29 as G
from . import complete_bi_image_chain_evidence_v4_20d as RELATION
from .interior_observation import NEIGHBORS

OBS = RELATION.OBS
SCHEMA = 'AUTHENTICATED_DETERMINISTIC_MULTIGUIDE_EDGE_GROUPS_V29'
PRODUCER_GENERATION = 'FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
GEOMETRY_DIGEST_FIELDS=('native_xyz','R_cf','centroid','physical_registration_Q','K','image_hw','initial_self_hidden')


def _component(chain, ray, reference, image_hw, vn, va, noise):
    cfg=OBS.DEFAULT_CONFIGURATION;k=G._id(ray['native_corner']);mate=G._id(ray['native_neighbor'])
    if (chain.get('native_corner')!=k or chain.get('native_neighbor')!=mate
            or sorted(chain.get('native_edge',[]))!=sorted([k,mate])
            or chain.get('invented_bridge_points')!=0):
        raise ValueError('Exact fresh actual-chain native IDs/no invented support required')
    xy=np.asarray(chain['observed_points'],float);mask=np.asarray(chain['inlier_mask'],bool)
    n=G._unit(chain['normal'],'measured TLS normal');offset=float(chain['offset'])
    if xy.ndim!=2 or xy.shape[1]!=2 or mask.shape!=(len(xy),) or not np.isfinite(xy).all() or not np.isfinite(offset):
        raise ValueError('Bound finite observed points/inlier mask required')
    actual=xy[mask];quality=np.asarray([chain['inlier_fraction'],chain['inlier_rms_px']],float)
    if (len(actual)<cfg['minimum_inlier_count'] or chain['inlier_count']!=len(actual)
            or not np.isfinite(quality).all() or not cfg['minimum_inlier_fraction']<=quality[0]<=1.
            or not np.isclose(quality[0],mask.mean(),atol=1e-10,rtol=0)
            or not 0<=quality[1]<=cfg['maximum_line_rms_px']):
        raise ValueError('Actual count/fraction/RMS guards must pass for full or partial lines')
    guide=G._unit(chain['generating_guide_direction_xy'],'actual generating search guide')
    guide_reference=reference['guide_reference']
    if (guide_reference.get('guide_reference_id')!=reference['guide_reference_id']
            or G._digest({key:value for key,value in guide_reference.items()if key!='guide_reference_id'})!=reference['guide_reference_id']
            or not np.array_equal(np.asarray(guide_reference['actual_search_direction_xy'],float),guide)):
        raise ValueError('Selected exact generating-guide dictionary/digest/direction must bind')
    seed_normal=G._unit(guide_reference['seed_normal_xy'],'actual TLS fitting seed normal')
    angle=float(np.degrees(np.arccos(np.clip(abs(n@seed_normal),0.,1.))))
    if (not np.isclose(angle,float(chain['angle_change_deg']),atol=1e-5,rtol=0)
            or angle>cfg['maximum_direction_change_deg']+1e-9
            or chain.get('generating_guide_reference_id')!=reference['guide_reference_id']):
        raise ValueError('Measured angle must bind the exact selected generating guide')
    h,w=image_hw
    if np.any(xy<0) or np.any(xy[:,0]>=w) or np.any(xy[:,1]>=h):
        raise ValueError('Actual observed pixels must lie on the same canvas')
    samples={}
    for key in ('RGB_gradient_energy','RGB_alignment','RGB_contrast','mask_plus','mask_minus'):
        values=np.asarray(chain[key],float)
        if values.shape!=(len(xy),) or not np.isfinite(values).all():
            raise ValueError('Full RGB/occupancy provenance must cover every observed sample')
        samples[key]=values
    mp,mm=samples['mask_plus'],samples['mask_minus'];kind=chain['mask_context_kind']
    if np.any(mp<0)or np.any(mp>1)or np.any(mm<0)or np.any(mm>1):raise ValueError('Finite occupancy probabilities required')
    if kind=='TARGET_TARGET_INTERNAL':
        occupancy=(mp>=cfg['minimum_inside'])&(mm>=cfg['minimum_inside'])
    elif kind=='TARGET_BACKGROUND_OUTLINE':
        occupancy=(np.maximum(mp,mm)>=cfg['minimum_inside'])&(np.minimum(mp,mm)<=cfg['maximum_outside'])&(np.abs(mp-mm)>=cfg['minimum_transition'])
    else:raise ValueError('Explicit supported target/background or target/target context required')
    if not np.all(occupancy&(samples['RGB_gradient_energy']>=cfg['minimum_gradient'])
            &(samples['RGB_alignment']>=cfg['minimum_gradient_alignment'])
            &(samples['RGB_contrast']>=cfg['minimum_color_transition'])):
        raise ValueError('Actual RGB gradient/alignment/contrast/occupancy guard failed')
    relation_error=RELATION._relation(chain,image_hw)
    lattice=np.asarray(chain['actual_inlier_image_lattice_xy'],float)
    if not np.array_equal(lattice,np.round(lattice)) or len({tuple(p)for p in lattice})!=len(actual):
        raise ValueError('Distinct actual image lattice sites required')
    rms=float(np.sqrt(np.mean((actual@n+offset)**2)))
    if not np.isclose(rms,quality[1],atol=1e-7,rtol=1e-7):raise ValueError('Saved weighted TLS coefficients must match actual inlier RMS')
    if n[int(np.argmax(np.abs(n)))]<0:n=-n;offset=-offset
    tangent=np.array([-n[1],n[0]]);coordinate=actual@tangent;span=float(np.ptp(coordinate))
    minimum_span=float(cfg['along_start_px']+cfg['boundary_context_px'])
    if span<minimum_span:raise ValueError('Actual finite support shorter than existing11px engineering bound')
    confidence=np.asarray(chain['observation_confidence'],float)
    if confidence.shape!=(len(xy),)or not np.isfinite(confidence).all()or np.any(confidence<0):
        raise ValueError('Actual observation-confidence array must remain bound')
    partial=chain.get('proposal_grade')=='PARTIAL_LINE_PROPOSAL'
    if not partial and chain['inlier_span_fraction']<cfg['minimum_inlier_span_fraction']:
        raise ValueError('Span-ratio rejection must be explicitly graded as fresh partial evidence')
    center=actual.mean(axis=0);anchor=center-n*(n@center+offset);fn=max(1.,rms)
    fs=float(np.sqrt(2.*fn*fn/(span*span+2.*fn*fn)))
    support_id=RELATION.BROAD._pixel_digest(lattice)
    fit_variant_id=G._digest(dict(normal=n.tolist(),offset=offset,anchor=anchor.tolist(),
        support_interval=[float(coordinate.min()),float(coordinate.max())],normal_floor_px=fn,angular_sine_floor=fs))
    component_id=support_id+':'+fit_variant_id
    return dict(native_corner=k,native_neighbor=mate,normal_unit_xy=n.tolist(),offset_px=offset,
        support_anchor_xy=anchor.tolist(),normal_variance_px2=float(vn)+fn*fn,
        actual_unweighted_inlier_centroid_xy=center.tolist(),
        angular_variance_rad2=float(va)+fs*fs,normal_angle_correlation=0.,
        actual_inlier_xy=actual.tolist(),actual_inlier_lattice_xy=lattice.astype(int).tolist(),
        actual_support_tangent_interval_px=[float(coordinate.min()),float(coordinate.max())],
        actual_inlier_span_px=span,normal_floor_px=fn,angular_sine_floor=fs,
        support_id=support_id,fit_variant_id=fit_variant_id,component_id=component_id,mixture_component_prior=1.,
        selected_chain_ref=deepcopy(reference),noise_provenance=deepcopy(noise),
        generating_guide_direction_xy=guide.tolist(),proposal_grade='PARTIAL_LINE_PROPOSAL'if partial else'FULL_LENGTH_IMAGE_LINE_PROPOSAL',
        generating_TLS_seed_normal_xy=seed_normal.tolist(),
        per_guide_angle_change_deg=angle,mask_context_kind=kind,lattice_relation_error_px=relation_error,
        original_q_endpoint_guard_and_12px_corner_cap_superseded=True,
        full_2D_corner_measurement=False,actual_pixels_and_TLS_coefficients_not_refit=True)


def _order_owner(base,k,mate,component,source_anchor):
    n=np.asarray(component['normal_unit_xy']);axis=np.array([-n[1],n[0]])
    parsed={m['native_corner']:m for m in base.measurements}
    if mate in parsed:
        # Both lookups happen only after the authoritative BI/S gate.
        direction=np.asarray(parsed[mate]['xy'])-np.asarray(parsed[k]['xy'])
        value=float(axis@direction);provenance='IMMUTABLE_BI_ENDPOINT_HEMISPHERE'
    else:
        direction=np.asarray(component['generating_guide_direction_xy'])
        if k!=source_anchor:direction=-direction
        value=float(axis@direction);provenance='SELECTED_GENERATING_GUIDE_UNIQUE_ORDER'
    sign=None if abs(value)<=1e-12 else(1 if value>0 else-1)
    return dict(native_corner=k,
        owner_provenance='SELF_ANCHOR_ACTUAL_SUPPORT'if k==source_anchor else'NAMED_PHYSICAL_EDGE_MODEL_CONSTRAINT_ONLY',
        fixed_tangent_sign=sign,order_provenance=provenance if sign is not None else None,
        order_ambiguous=sign is None,opposite_corner_new_image_measurement_created=False)


def authenticate_geometry_guided_edges(base,observation,*,image_binding,initial_geometry,
        normal_variance_component_px2=0.,angular_variance_component_sine2=0.,noise_provenance,
        calibration_preparation=False,expected_source_policy=None,normal_angle_correlation=0.):
    """Bind selected full-provenance lines before any source-policy/noise fit."""
    if not isinstance(base,G.BASELINE.PrepartialRGBPointPreservingNativeProblem):raise ValueError('Prepare exact v26 no-edge cost baseline first')
    if observation.get('observation_generation')!=PRODUCER_GENERATION:
        raise ValueError('New fresh multiguide observation generation required')
    binding=observation['image_binding']
    if (binding.get('sha256')!=image_binding.get('sha256')or tuple(binding['image_hw'])!=tuple(base.image_hw)
            or tuple(image_binding['image_hw'])!=tuple(base.image_hw)
            or not np.array_equal(np.asarray(initial_geometry['K']),base.K)
            or list(initial_geometry['initial_self_hidden'])!=[s=='S'for s in base.states]):
        raise ValueError('Exact image/camera/permanent initial-S binding required')
    if (list(observation.get('initial_geometry_digest_fields',[]))!=list(GEOMETRY_DIGEST_FIELDS)
            or observation.get('initial_geometry_digest')!=G._digest({k:initial_geometry[k]for k in GEOMETRY_DIGEST_FIELDS})):
        raise ValueError('Exact original observer geometry guide whitelist required')
    if noise_provenance.get('fit_partition')!='CAL'or noise_provenance.get('observation_family')!=G.OBSERVATION_FAMILY:
        raise ValueError('Separately fitted new observation-family CAL noise required')
    if not calibration_preparation and (not isinstance(noise_provenance.get('model_digest_sha256'),str)
            or len(noise_provenance['model_digest_sha256'])!=64):
        raise ValueError('Runtime must bind its separately fitted new CAL model')
    if not calibration_preparation and (expected_source_policy not in G.SOURCE_POLICIES
            or noise_provenance.get('source_policy')!=expected_source_policy):
        raise ValueError('Runtime CAL model must match the exact declared source policy')
    vn,va=float(normal_variance_component_px2),float(angular_variance_component_sine2)
    if not np.isfinite([vn,va,float(normal_angle_correlation)]).all()or min(vn,va)<0:
        raise ValueError('Finite nonnegative CAL variance components required')
    if float(normal_angle_correlation)!=0.:raise ValueError('Declared zero normal/angle correlation required')
    records=observation['chain_observation']['edge_records'];graph=observation['joint_graph']
    references=observation['selected_chain_refs']
    if {r['line_name']for r in references}!=set(graph['selected_line_names'])or len(references)!=len(graph['selected_line_names']):
        raise ValueError('One exact chain reference per deterministic graphselected physical line required')
    nodes={n['name']:n for n in graph['line_candidates']};groups=[];skipped=[];pairs=set()
    hidden=initial_geometry['initial_self_hidden']
    for reference in references:
        ri,ci=reference['ray_index'],reference['chain_index']
        if isinstance(ri,(bool,np.bool_))or not isinstance(ri,(int,np.integer))or not 0<=ri<len(records):
            raise ValueError('Exact nonnegative ray/chain array indices required')
        ray=records[ri]
        k,mate=G._id(ray['native_corner']),G._id(ray['native_neighbor'])
        if hidden[k]or hidden[mate]or base.states[k]not in('B','I')or base.states[mate]=='S':
            raise ValueError('State/S gates must precede selected actual chain fields')
        if isinstance(ci,(bool,np.bool_))or not isinstance(ci,(int,np.integer))or not 0<=ci<len(ray.get('chains',[])):
            raise ValueError('Exact nonnegative ray/chain array indices required')
        if mate not in NEIGHBORS[k]or reference['native_corner']!=k or reference['native_neighbor']!=mate:
            raise ValueError('Exact physical incident edge reference required')
        pair=tuple(sorted([k,mate]))
        if pair in pairs:raise ValueError('Graph must select one line per unique physical edge')
        pairs.add(pair);chain=ray['chains'][ci]
        component=_component(chain,ray,reference,base.image_hw,vn,va,noise_provenance)
        node=nodes[reference['line_name']]
        sign=float(np.sign(np.asarray(node['normal'])@component['normal_unit_xy']))
        if (not np.array_equal(np.asarray(node['actual_inliers']),np.asarray(component['actual_inlier_xy']))
                or not np.allclose(np.asarray(node['normal'])*sign,component['normal_unit_xy'],atol=1e-12,rtol=0)
                or not np.isclose(float(node['offset'])*sign,component['offset_px'],atol=1e-12,rtol=0)):
            raise ValueError('Selected graph node must bind its exact actual line support/coefficients')
        role='TARGET_BACKGROUND_OUTLINE'if base.states[k]=='B'else'TARGET_TARGET_INTERNAL'
        if component['mask_context_kind']!=role:
            skipped.append(dict(physical_native_edge=list(pair),reason='source_anchor_role_context_mismatch'));continue
        if base.states[k]=='B':
            outside=node.get('global_external_context',{})
            if (outside.get('available')is not True or outside.get('actual_support_near_global_external_fraction',0.)<.6
                    or outside.get('same_foreground_component_fraction',0.)<.6):
                skipped.append(dict(physical_native_edge=list(pair),reason='B_actual_global_external_support_ambiguous'));continue
        owners=[]
        for owner in pair:
            if hidden[owner]or base.states[owner]not in('B','I'):continue
            other=mate if owner==k else k
            owners.append(_order_owner(base,owner,other,component,k))
        groups.append({**component,'activation_owners':owners,'physical_native_edge':list(pair),
                       'image_quality_energy':float(node['image_quality_energy'])})
    overlapping=set()
    for first,second in combinations(groups,2):
        if {tuple(p)for p in first['actual_inlier_lattice_xy']}&{tuple(p)for p in second['actual_inlier_lattice_xy']}:
            overlapping.update([first['component_id'],second['component_id']])
    active=[g for g in groups if g['component_id']not in overlapping]
    corner_pairs=[]
    for k,state in enumerate(base.states):
        if hidden[k]or state not in('B','I'):continue
        incident=[g for g in active if any(o['native_corner']==k for o in g['activation_owners'])]
        for first,second in combinations(incident,2):
            check=G._finite_actual_pair(first,second,base.image_hw);direction=[]
            if check['passed']:
                point=np.asarray(check['diagnostic_intersection_xy'])
                for g in (first,second):
                    owner=next(o for o in g['activation_owners']if o['native_corner']==k);sign=owner['fixed_tangent_sign']
                    n=np.asarray(g['normal_unit_xy']);axis=np.array([-n[1],n[0]])
                    fraction=None if sign is None else float(np.mean(sign*((np.asarray(g['actual_inlier_xy'])-point)@axis)>=-1.))
                    direction.append(dict(component_id=g['component_id'],fixed_tangent_sign=sign,
                        actual_support_fraction=fraction,passed=sign is not None and fraction>=.6))
            corner_pairs.append(dict(native_corner=k,component_ids=[first['component_id'],second['component_id']],
                passed=bool(check['passed']and direction and all(d['passed']for d in direction)),
                finite_actual_support=check,directed_support=direction,precise_corner_created=False))
    return dict(schema=SCHEMA,observation_family=G.OBSERVATION_FAMILY,population=G.POPULATION,
        producer_generation=PRODUCER_GENERATION,
        groups=active,ordered_corner_pairs=corner_pairs,skipped=skipped,
        cross_native_support_overlap_withheld_component_ids=sorted(overlapping),
        selected_graph_population_before_source_policy=True,new_image_queries=0,new_precise_KPs=0,
        permanent_S_U_local_queries=0,initial_geometry_digest=observation['initial_geometry_digest'],
        image_binding=deepcopy(image_binding),all_alternatives_retained_in_source_archive=True)
