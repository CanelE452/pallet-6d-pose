"""Uniform fresh image-line observation generation for CAL46 and primary192.

Frozen17c image-lattice NMS/IRLS TLS and v28 guide/visibility kernels are reused
through private facades. A span-only rejection can survive as an explicit
partial LINE, with all actual gradient/lattice/mask provenance constructed in
the same fresh pass. It is never promoted to a precise 2D keypoint. The ordered
graph chooses at most one actual line per physical edge; all alternatives stay
archived. This generation requires its own selected-population CAL noise.

No file/label/model loading, learned inference, pose optimization, S/U local
point query or state/corner modification is performed by this module.
"""
from __future__ import annotations

from copy import deepcopy
from types import FunctionType, SimpleNamespace
import numpy as np

from . import joint_geometry_guided_lines_v4_28 as G
from . import ordered_geometry_support_v4_28b as ORDER
from . import edge_visibility_v4_28 as VIS
from . import refine_native_edge_pose_v4_17 as STARTS


SCHEMA='FRESH_JOINT_GEOMETRY_IMAGE_LINE_OBSERVATION_V4_29'
OBSERVATION_GENERATION='FRESH_GUIDED_NMS_TLS_PARTIAL_DIRECTED_GRAPH_WINNER_V29'
MEASUREMENT_SOURCE='ACTUAL_IMAGE_NMS_TLS_WITH_GUIDE_AND_VISIBILITY_V29'
CENTRAL_PREDECLARATION_SHA256='4ba0b55e7c60bccf36e8479ce393e42a1fc1b7500dc3918f689aa0d5cd99c41f'
FULL_GRADE='FULL_LENGTH_LINE_PROPOSAL'
PARTIAL_GRADE='PARTIAL_LINE_PROPOSAL'
DEFAULT_CONFIGURATION=dict(line_configuration=deepcopy(G.NMS.DEFAULT_CONFIGURATION),
    graph_configuration=deepcopy(G.DEFAULT_GRAPH_CONFIGURATION))
RECIPE=dict(schema=SCHEMA,observation_generation=OBSERVATION_GENERATION,
    central_predeclaration_sha256=CENTRAL_PREDECLARATION_SHA256,
    full_and_partial_measurement_source=MEASUREMENT_SOURCE,
    guide_bank='both original task families coarse plus same unregularized six-diverse input proposals',
    no_e_numeric_pose_GT_or_error_selected_guide=True,
    frozen_pixel_kernel='17c image-lattice gradient NMS, local RGB/occupancy gates, IRLS TLS',
    span_partial='only span ratio fails; same count6/fraction0.6/RMS1.25/guide-angle12 and actual span>=11',
    minimum_actual_span_px='existing along_start4 + boundary_context7',
    graph='frozen v28 geometry/exterior graph plus v28b ordered1px/0.6 support',
    active_policy='exact graph winner, at most one line per physical edge; all ALT retained',
    CAL_noise_population='same selected physical-edge observations, distinct generation required',
    precise_KP_created=False,hard12_shift_is_line_admission_criterion=False,
    original_support_and_rounding_excluded_support_gaps_are_distinct=True,
    S_U_local_coordinate_or_corner_queries=0,U_intersections=0,
    target_mask_absence_is_occlusion=False,independent_occluder_mask_optional=True,
    corner_states_q_original_covariance_effective_weights_unchanged=True,
    new_NN_posefit_CAL_or_label_IO=0,
    fresh_candidate_TLS=True,cached_support_rounded_diagnostic_TLS_not_factor_input=True,
    calibrated_confidence_or_independent_image_likelihood_claim=False)


def prepare_task_pose_guides(task,unregularized_problems):
    """No fits: the exact existing six-diverse selector on frozen input poses.

    Caller supplies the old unregularized NativeEdgeProblem per family, never
    an O/S/pose/mask-regularized wrapper. Saved accepted/error fields and the
    task's numeric baseline are not accessed.
    """
    if set(task['families'])!={'ORIGINAL_WD','SWAPPED_WD'}:
        raise ValueError('Both original registered task families required')
    guides=[];audit={}
    original=task['families']['ORIGINAL_WD']
    original_xyz=G._finite(original['native_xyz'],(8,3),'original native XYZ')
    original_Q=G._rotation(original['physical_registration_Q'],'original fixed Q')
    for name,family in task['families'].items():
        problem=unregularized_problems[name]
        if not np.allclose(problem.xyz,family['native_xyz'],atol=0,rtol=0):
            raise ValueError('Unregularized start problem has wrong family model')
        Q=G._rotation(family['physical_registration_Q'],'task physical Q')
        proposals=[dict(name='CACHED_'+str(i),R=(np.asarray(p['R_physical'])@Q.T).tolist(),t=p['t'])
            for i,p in enumerate(task['proposals'])]
        selected,selection=STARTS.choose_native_starts(problem,proposals,family['coarse_pose'],
            configuration=STARTS.DEFAULT_CONFIG)
        # Coarse is explicitly a guide even when numerically equal to a saved
        # start. Equality is audited as an alias, not independent direction data.
        bank=[dict(name='OWN_INITIAL_COARSE',**family['coarse_pose'])]
        bank.extend(s for s in selected if s['name']!='OWN_INITIAL_COARSE')
        for seed in bank:
            source_R=G._rotation(seed['R'],'task guide R');source_t=G._finite(seed['t'],(3,),'task guide t')
            source_xyz=np.asarray(family['native_xyz'],float)
            physical_R=source_R@Q
            center=source_R@source_xyz.mean(axis=0)+source_t
            # W/D families retain image-channel labels but can permute physical
            # corners. Observation rays use ONE original native table always;
            # only the physical pose and center are borrowed from a start.
            xyz=original_xyz;Qguide=original_Q;R=physical_R@Qguide.T
            t=center-R@xyz.mean(axis=0);K=np.asarray(task['spec']['K'],float)
            camera=xyz@R.T+t;h=camera@K.T
            projection=[]
            for k in range(8):
                # S is kept as a model-only placeholder, never an image query.
                if task['initial_self_hidden'][k]or task['states'][k]=='S':projection.append(None)
                elif abs(h[k,2])>1e-12:projection.append((h[k,:2]/h[k,2]).tolist())
                else:projection.append([float('nan'),float('nan')])
            geometry=dict(native_xyz=xyz.tolist(),physical_registration_Q=Qguide.tolist(),R_cf=R.tolist(),
                centroid=(R@xyz.mean(axis=0)+t).tolist(),K=K.tolist(),image_hw=task['spec']['image_hw'],
                initial_self_hidden=deepcopy(task['initial_self_hidden']),initial_projection_xy=projection)
            guides.append(dict(guide_id=name+'/'+seed['name'],
                source_kind='ORIGINAL_TASK_COARSE'if seed['name']=='OWN_INITIAL_COARSE'else'UNIFORM_INITIAL_PROPOSAL_BANK',
                geometry=geometry,family=name,original_input_proposal_name=seed['name'],
                source_family_pose=dict(R_cf=source_R.tolist(),t=source_t.tolist(),
                    physical_registration_Q=Q.tolist(),R_physical=physical_R.tolist(),model_center_camera=center.tolist()),
                native_channel_geometry='ORIGINAL_WD fixed XYZ/Q; source-family IDs are not mixed into image rays',
                start_selection_policy='frozen choose_native_starts; native image cost only, six diverse plus coarse'))
        audit[name]=selection
    return guides,dict(families=audit,configuration=deepcopy(STARTS.DEFAULT_CONFIG),
        old_numeric_e_pose_or_GT_error_read=False,new_pose_optimization_calls=0,
        source='original task physical proposals reexpressed with ORIGINAL fixed XYZ/Q and same camera model center',
        source_family_corner_permutation_not_treated_as_same_native_image_edge=True)


def _fit_proxy(line_cfg):
    """Keep the exact measured TLS even when global span alone is insufficient."""
    def fit(observations,confidence,t,seed_normal,total_samples,fit_cfg):
        fitted,diag=G.NMS.F._fit_line(observations,confidence,t,seed_normal,total_samples,fit_cfg)
        if fitted is not None:
            return fitted,dict(diag,proposal_grade=FULL_GRADE,original_fitter_accepted=True,
                original_span_guard_pass=True,original_fitter_reason=diag['reason'])
        # Reuse the already tested *admissibility only*. Its cached provenance
        # labels are not copied into a fresh measurement.
        parent=dict(measurement_source='ACTUAL_IMAGE_NMS_CHAIN_WITH_MASK_CONTEXT')
        admissible=ORDER._promote_span_only(parent,diag,line_cfg)
        if admissible is None:return fitted,diag
        return dict(normal=np.asarray(diag['normal']),offset=float(diag['offset'])),dict(diag,
            proposal_grade=PARTIAL_GRADE,original_fitter_accepted=False,
            original_span_guard_pass=False,original_fitter_reason=diag['reason'],
            actual_finite_inlier_span_px=admissible['actual_finite_inlier_span_px'],
            minimum_engineering_partial_span_px=admissible['minimum_engineering_partial_span_px'])
    return fit


def _kernel_facade(cfg):
    fit=SimpleNamespace(_fit_line=_fit_proxy(cfg))
    edge_ns=dict(G.NMS._edge_chains.__globals__);edge_ns['F']=fit
    edge=FunctionType(G.NMS._edge_chains.__code__,edge_ns)
    observe_ns=dict(G.NMS.observe_image_supported_edge_chains.__globals__)
    observe_ns['_edge_chains']=edge
    observe=FunctionType(G.NMS.observe_image_supported_edge_chains.__code__,observe_ns)
    return SimpleNamespace(**{**G.NMS.__dict__,'_edge_chains':edge,
        'observe_image_supported_edge_chains':observe})


def _fresh_node(record,chain,reference_index,cfg):
    node=G._node(record,chain,reference_index,cfg)
    grade=chain.get('proposal_grade',FULL_GRADE)
    node.update(proposal_grade=grade,RGB_gradient_lattice_provenance_complete=True,
        full_actual_image_provenance=True,
        production_17c_accepted=bool(chain.get('original_fitter_accepted',True)
            and chain.get('corrected_endpoint_normal_guard_pass',True)),
        actual_finite_inlier_span_px=float(np.ptp(np.asarray(node['actual_inliers'])@
            np.array([-node['normal'][1],node['normal'][0]]))))
    return node


def _rank_fresh(observation,geometry,states,visible_measurements,**kwargs):
    ns=dict(G.rank_joint_geometry_line_graph.__globals__)
    ns.update(_node=_fresh_node,_score_graph=ORDER._ordered_score)
    graph=FunctionType(G.rank_joint_geometry_line_graph.__code__,ns)(
        observation,geometry,states,visible_measurements,**kwargs)
    graph['joint_alternatives']=[a for a in graph['joint_alternatives']if not a['ordered_support_unavailable']]
    return graph


def _geometry_record(geometry,states):
    g=G._geometry(geometry)
    # Model geometry is allowed; no S 2D input slot is touched.
    projection={str(k):G._finite(geometry['initial_projection_xy'][k],(2,),'nonS model projection').tolist()
        for k in range(8)if not geometry['initial_self_hidden'][k]and states[k]!='S'}
    return dict(native_xyz=g['xyz'].tolist(),R_cf=g['R'].tolist(),physical_registration_Q=g['Q'].tolist(),
        K=g['K'].tolist(),centroid=g['center'].tolist(),image_hw=list(geometry['image_hw']),
        initial_self_hidden=list(geometry['initial_self_hidden']),nonS_model_projection_xy=projection)


def _chain_reference(ray,chain,ri,ci,geometry,states,points):
    k,mate=int(ray['native_corner']),int(ray['native_neighbor'])
    proposal=ray.get('guide_proposal')
    if proposal is None:
        a=G._finite(geometry['initial_projection_xy'][k],(2,),'base nonS projection')
        b=G._finite(geometry['initial_projection_xy'][mate],(2,),'base nonS mate projection')
        direction=(b-a)/np.linalg.norm(b-a)
        ref=dict(guide_id='ORIGINAL_INITIAL',guide_kind='FROZEN17C_ORIGINAL_MODEL_SEED',
            seed_direction_xy=direction.tolist(),seed_normal_xy=[float(-direction[1]),float(direction[0])],
            actual_strip=ray.get('along_interval_px'),
            strip_direction_source='BI endpoint vector if both BI, otherwise original model direction',
            original_query_strip_metadata_is_frozen17c=True)
        anchor=np.asarray(points[k]['record']['xy'],float)
        actual_delta=(np.asarray(points[mate]['record']['xy'],float)-anchor)if mate in points else direction
        actual_direction=actual_delta/np.linalg.norm(actual_delta)
        ref.update(actual_search_anchor_xy=anchor.tolist(),actual_search_direction_xy=actual_direction.tolist(),
            actual_search_direction_source='gated immutable BI endpoint hemisphere'if mate in points else'original model direction')
    else:
        direction=np.asarray(proposal['direction_xy'],float)
        ref=dict(guide_id=proposal['guide_id'],guide_kind=proposal['guide_kind'],
            seed_direction_xy=direction.tolist(),seed_normal_xy=[float(-direction[1]),float(direction[0])],
            search_anchor_kind=proposal['search_anchor_kind'],search_anchor_xy=proposal['anchor_xy'],
            original_observed_BI_reference_xy=proposal['observed_BI_reference_xy'],
            actual_query_strip_audit=ray['actual_query_strip_audit'],
            vanishing_point_h=proposal.get('vanishing_point_h'),physical_axis=proposal['physical_axis'])
        ref.update(actual_search_anchor_xy=proposal['anchor_xy'],actual_search_direction_xy=direction.tolist(),
            actual_search_direction_source=proposal['guide_kind'])
    normal=np.asarray(chain['normal'],float);normal/=np.linalg.norm(normal)
    seed=np.asarray(ref['seed_normal_xy'])
    angle=float(np.degrees(np.arccos(np.clip(abs(normal@seed),0,1))))
    if abs(angle-float(chain['angle_change_deg']))>1e-5:
        raise ValueError('Actual TLS angle differs from its explicit search-guide seed')
    ref.update(native_corner=k,native_neighbor=mate,unoriented_angle_to_proposal_deg=angle,
        physical_native_ids=[k,mate],
        original_reference_projection_xy=[list(geometry['initial_projection_xy'][k]),list(geometry['initial_projection_xy'][mate])],
        stored_fitter_angle_change_deg=chain['angle_change_deg'],
        original_geometry_digest=G._digest({key:geometry[key]for key in
            ('native_xyz','R_cf','centroid','physical_registration_Q','K','image_hw','initial_self_hidden')}),
        native_neighbor_is_model_association_not_unknown_coordinate=True)
    ref['geometry_ref_id']='ORIGINAL_INITIAL'if ref['guide_id']=='MEASURED_BI_ENDPOINTS'else ref['guide_id']
    ref['guide_reference_id']=G._digest(ref)
    return dict(ray_index=ri,chain_index=ci,native_corner=k,native_neighbor=mate,
        guide_reference=ref,guide_reference_id=ref['guide_reference_id'])


def _decorate_chain(ray,chain,reference,cfg):
    out=deepcopy(chain)
    observed=np.asarray(out['observed_points'],float)
    inliers=np.asarray(out['inlier_mask'],bool)
    normals=np.asarray(out['image_gradient_normal_xy'],float)
    shifts=np.asarray(out['measured_subpixel_shift_px'],float)
    if normals.shape!=observed.shape or shifts.shape!=(len(observed),):
        raise ValueError('Full per-sample actual NMS normal and shift provenance required')
    lattice_float=observed-shifts[:,None]*normals
    lattice=np.rint(lattice_float)
    error=float(np.max(np.abs(lattice_float-lattice)))
    if error>1e-6:raise ValueError('Actual selected NMS lattice cannot be reconstructed exactly')
    if not np.array_equal(lattice[inliers],np.asarray(out['actual_inlier_image_lattice_xy'])):
        raise ValueError('Reconstructed full selected lattice disagrees with actual inlier lattice')
    for field in ('RGB_gradient_energy','RGB_alignment','RGB_contrast','mask_plus','mask_minus','observation_confidence'):
        array=np.asarray(out[field],float)
        if array.shape!=(len(observed),)or not np.isfinite(array).all():
            raise ValueError('Complete finite actual sample provenance required: '+field)
    normal,offset,actual=G._actual_line(out);tangent=np.array([-normal[1],normal[0]])
    span=float(np.ptp(actual@tangent))
    if span<cfg['along_start_px']+cfg['boundary_context_px']:
        out['factor_eligible_actual_span']=False
    else:out['factor_eligible_actual_span']=True
    out.update(measurement_source=MEASUREMENT_SOURCE,observation_generation=OBSERVATION_GENERATION,
        full_actual_image_provenance=True,RGB_gradient_lattice_provenance_complete=True,
        actual_image_lattice_xy=lattice.astype(int).tolist(),
        full_lattice_reconstruction='exact original observed_xy minus measured_shift*image_gradient_normal; integer-grid verification',
        full_lattice_reconstruction_error_px=error,
        actual_finite_inlier_span_px=span,minimum_actual_factor_span_px=cfg['along_start_px']+cfg['boundary_context_px'],
        guide_reference=reference['guide_reference'],guide_reference_id=reference['guide_reference_id'],
        generating_guide_direction_xy=reference['guide_reference']['actual_search_direction_xy'],
        generating_guide_reference_id=reference['guide_reference_id'],
        original_proposal_measurement_source=chain.get('measurement_source'),
        original_corrected_endpoint_guard_pass=chain.get('corrected_endpoint_normal_guard_pass'),
        old12_corner_cap_not_a_line_factor_admission_rule=True,
        full_2D_corner_measurement_created=False,
        proposal_grade=chain.get('proposal_grade',FULL_GRADE))
    out['chain_id']=G._digest(dict(native_edge=sorted([ray['native_corner'],ray['native_neighbor']]),
        observed_points=out['observed_points'],inlier_mask=out['inlier_mask'],normal=out['normal'],offset=out['offset'],
        observation_confidence=out['observation_confidence'],guide_reference_id=out['guide_reference_id']))
    return out


def _support_gap(node,point):
    normal=np.asarray(node['normal']);tangent=np.array([-normal[1],normal[0]])
    support=np.asarray(node['actual_inliers'])@tangent;value=float(np.asarray(point)@tangent)
    return float(max(support.min()-value,0.,value-support.max()))


def _joint_support_audit(graph,cfg):
    byname={n['name']:n for n in graph['line_candidates']}
    for junction in graph['rounded_BI_junctions']:
        nodes=[byname[name]for name in junction['line_names']]
        point,sine=G._intersection((nodes[0]['normal'],nodes[0]['offset']),
            (nodes[1]['normal'],nodes[1]['offset']),cfg['minimum_intersection_sine'])
        if point is not None:
            gaps=[_support_gap(n,point)for n in nodes]
            junction.update(original_measured_line_intersection_xy=point.tolist(),
                original_actual_support_extrapolation_gap_px=gaps,
                original_actual_support_finite11_pass=max(gaps)<=cfg['along_start_px']+cfg['boundary_context_px'],
                rounding_excluded_straight_support_gap_px=junction.get('extrapolation_beyond_straight_support_px'),
                rounding_excluded_straight_support_finite11_pass=junction.get('finite_support_11px_guard_would_pass'),
                rounding_guard_failure_does_not_mean_original_observation_missing=True,
                corner_or_pose_adopted=False)


def observe_joint_geometry_edge_evidence(image_bgr,pallet_mask,geometry,states,visible_measurements,*,
        image_binding,pose_guides=None,configuration=None,occluder_mask=None,occluder_binding=None):
    """Fresh global/edge RGB only, with permanent initial S and unknown gates."""
    cfg,_=G._config(configuration)
    states=[G.NMS.V3._state(s)for s in states]
    for guide in pose_guides or []:
        if guide['geometry']['initial_self_hidden']!=geometry['initial_self_hidden']:
            raise ValueError('Every guide must retain the original permanent initial-S gate')
        if not np.array_equal(guide['geometry']['native_xyz'],geometry['native_xyz'])or not np.array_equal(
                guide['geometry']['physical_registration_Q'],geometry['physical_registration_Q']):
            raise ValueError('Every image guide must use ORIGINAL fixed native XYZ and Q')
    context=VIS.prepare_edge_visibility(pallet_mask,geometry,states,visible_measurements,
        image_binding=image_binding,occluder_mask=occluder_mask,occluder_binding=occluder_binding)
    facade=_kernel_facade(cfg)
    guided_ns=dict(G._guided_edge.__globals__);guided_ns['NMS']=facade
    guided=FunctionType(G._guided_edge.__code__,guided_ns)
    # The rank runs after decoration below; a tiny temporary hook keeps the
    # observer from reranking twice while preserving its fresh query kernel.
    observe_ns=dict(G.observe_joint_geometry_guided_lines.__globals__)
    observe_ns.update(NMS=facade,_guided_edge=guided,
        rank_joint_geometry_line_graph=lambda *args,**kwargs:{})
    observed=FunctionType(G.observe_joint_geometry_guided_lines.__code__,observe_ns)(
        image_bgr,pallet_mask,geometry,states,visible_measurements,image_binding=image_binding,
        pose_guides=pose_guides,configuration=configuration,visibility_context=context,base_observation=None)
    records=[*observed['base_observation']['edge_records'],*observed['additional_guided_edge_records']]
    points=G.NMS.G._eligible_points(visible_measurements,states,geometry['initial_self_hidden'],geometry['image_hw'])
    decorated=[];partial=[];refs={}
    for ri,ray in enumerate(records):
        k,mate=ray['native_corner'],ray['native_neighbor']
        if geometry['initial_self_hidden'][k]or geometry['initial_self_hidden'][mate]or states[k]not in('B','I')or states[mate]=='S':
            decorated.append(dict(ray));continue
        row=dict(ray,measurement_source=MEASUREMENT_SOURCE,observation_generation=OBSERVATION_GENERATION)
        candidates=[*ray.get('chains',[]),*[c for c in ray.get('rejected_chains',[])
            if c.get('reason')=='existing_corrected_endpoint_normal_guard_failed']]
        row['chains']=[]
        row['rejected_chains']=[c for c in ray.get('rejected_chains',[])
            if c.get('reason')!='existing_corrected_endpoint_normal_guard_failed']
        for ci,chain in enumerate(candidates):
            ref=_chain_reference(ray,chain,ri,ci,geometry,states,points)
            line=_decorate_chain(ray,chain,ref,cfg)
            if not line['factor_eligible_actual_span']:
                row['rejected_chains'].append(dict(line,reason='actual_finite_span_below_existing11px_factor_budget'))
                continue
            ref['chain_index']=len(row['chains']);row['chains'].append(line)
            refs[f'{ri}:{ref["chain_index"]}']=dict(ref,chain_id=line['chain_id'])
            if line['proposal_grade']==PARTIAL_GRADE:
                partial.append(dict(ray_index=ri,chain_index=ref['chain_index'],chain_id=line['chain_id'],
                    original_span_fraction=line['inlier_span_fraction'],actual_span_px=line['actual_finite_inlier_span_px'],
                    full_actual_image_provenance=True,original_fitter_accepted=False))
        decorated.append(row)
    chain_observation=dict(schema=SCHEMA,observation_generation=OBSERVATION_GENERATION,
        edge_records=decorated,configuration=cfg,image_binding=observed['image_binding'])
    graph=_rank_fresh(chain_observation,geometry,states,visible_measurements,image_binding=image_binding,
        configuration=configuration,pose_guides=pose_guides,pallet_mask=pallet_mask,visibility_context=context)
    _joint_support_audit(graph,cfg)
    selected=[];seen=set()
    for node in graph['line_candidates']:
        # Exact fit variants are alternatives; the winning physical edge is
        # represented once, never independent precision from repeated guides.
        if node['name']not in graph['selected_line_names']:continue
        edge=tuple(node['native_edge'])
        if edge in seen:raise ValueError('At most one graph-winner line per physical edge')
        seen.add(edge)
        ref=refs[node['provenance'][0]]
        selected.append(dict(ref,line_name=node['name'],native_edge=node['native_edge'],
            proposal_grade=node['proposal_grade'],
            eligible_BI_owner_ids=[k for k in node['native_edge']
                if not geometry['initial_self_hidden'][k]and states[k]in('B','I')],
            endpoint_provenance=[dict(native_corner=k,state=states[k],
                actual_self_anchor=k==ref['native_corner'],
                model_only_other_BI_endpoint=k!=ref['native_corner']and states[k]in('B','I'),
                image_coordinate_used_as_search_guide=(k==ref['native_corner']or
                    decorated[ref['ray_index']].get('neighbor_image_coordinate_used',False)),
                full_2D_point_observation_created=False)for k in node['native_edge']]))
    visibility=[]
    for ref in selected:
        line=decorated[ref['ray_index']]['chains'][ref['chain_index']]
        actual=np.asarray(line['observed_points'])[np.asarray(line['inlier_mask'],bool)]
        visibility.append(context.classify_expected_edge(ref['native_corner'],ref['native_neighbor'],
            observed_chain_points=actual,mask_context_kind=decorated[ref['ray_index']].get('mask_context_kind')))
    guide_geometries={'ORIGINAL_INITIAL':_geometry_record(geometry,states)}
    for item in pose_guides or []:guide_geometries[item['guide_id']]=_geometry_record(item['geometry'],states)
    # Large actual sample arrays occur once in chain_observation, not again in
    # graph nodes. Exact refs retain every fit/support variant and its aliases.
    for node in graph['line_candidates']:
        node.pop('line_record',None)
        node['chain_references']=[refs[key]for key in node['provenance']]
    points=G.NMS.G._eligible_points(visible_measurements,states,geometry['initial_self_hidden'],geometry['image_hw'])
    return dict(schema=SCHEMA,observation_generation=OBSERVATION_GENERATION,recipe=deepcopy(RECIPE),
        image_binding=observed['image_binding'],target_mask_pixel_sha256=VIS.mask_pixel_sha256(pallet_mask),
        initial_geometry_digest=graph['original_guide_bank']['original_geometry_digest'],
        initial_geometry_digest_fields=['native_xyz','R_cf','centroid','physical_registration_Q','K','image_hw','initial_self_hidden'],
        guide_geometries=guide_geometries,guide_bank=graph['original_guide_bank'],
        chain_observation=chain_observation,joint_graph=graph,selected_chain_refs=selected,
        partial_line_ledger=partial,edge_visibility=visibility,
        independent_occluder_binding=deepcopy(occluder_binding),
        gated_BI_archive_sha256=G._digest([points[k]['record']for k in sorted(points)]),
        selected_physical_edges=len(selected),all_actual_chains_preserved=True,
        old_q_covariance_states_U_regions_and_initialS_changed=False,
        S_U_local_coordinate_or_corner_queries=0,U_intersections_created=0,
        new_NN_or_pose_solver_calls=0,labels_or_reference_access=False,
        full_2D_corner_measurements_created=0,requires_new_selected_population_CAL_noise=True)
