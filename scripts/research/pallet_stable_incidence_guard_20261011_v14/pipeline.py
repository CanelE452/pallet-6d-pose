"""Keep finite support in the fit, use point/line incidence for observability.

Free positions along an edge are inequalities, not additional 3D point IDs.
The selected V13 candidate is unchanged; this guard never selects another one.
"""
from __future__ import annotations
import copy
import cv2
import numpy as np

from ..pallet_boundary_corner_refiner_20261010_v2.pipeline import assemble
from ..pallet_same_observation_controls_20261010_v8 import controls as C
from ..pallet_partial_line_independent_20261010_v5 import solver as V5
from ..pallet_observation_refiner_20261009_v1.solver import cuboid
from ..pallet_finite_edge_consensus_repair_20261011_v13 import pipeline as V13

METHOD='FINITE_EDGE_CONSENSUS_STABLE_INCIDENCE'
METHODS=(METHOD,)
PRIMARY=METHOD
PARENT_METHOD=V13.PRIMARY
POLICY=dict(schema='selected_finite_pose_stable_incidence_policy_v14',
    selected_V13_actual_fit_only=True,finite_optimizer_and_inlier_score_unchanged=True,
    observability='actual fitted point Jacobian and two endpoint-normal rows per actual fitted unused edge',
    longitudinal_hinges_used_for_optimizer_consensus_and_score=True,
    longitudinal_hinges_used_as_observability_equalities=False,
    hinge_rank_rows_retained=0,numerical_rank_relative_threshold=1e-10,
    weak_relative_threshold=1e-4,weak_normalization='unchanged V5 unit L2 Jacobian columns',
    candidate_reselection=False,new_pose_optimizer_calls=0,new_PnP_calls=0,
    new_pixel_epsilon=False,new_seeds_training_RGB=False,
    insufficient_state='STABLE_INCIDENCE_INSUFFICIENT',weak_state='STABLE_INCIDENCE_WEAK',
    numerical_failure_state='STABLE_INCIDENCE_NUMERICAL_FAILURE',
    decline_output='full original N3 output and original initial pose',
    initial_H_image_coordinates_used=False,accepted_H_projection_reused_without_refit=True,
    center_used_for_fit=False,physical_ownership_certified=False,global_uniqueness_proven=False,
    cause_declared_before_V13_GT_scoring=True,known_DEV_structural_operating_ablation=True,
    performance_threshold_search=False)
TRUTH={'pose','corner','GT','ground_truth','reference_pose','mask_audit','evaluation_reference','human_states_native'}

def require(value,message):C.require(value,message)

def normal_row_indices(geometry,point_ids,edges):
    """Select saved point rows and the two incidence rows of each edge block."""
    require(geometry['point_ids']==list(point_ids) and geometry['line_edges']==list(edges),'selected actual fit geometry identity')
    indices=list(range(2*len(point_ids)));blocks=geometry['edge_blocks']
    require([b['edge'] for b in blocks]==list(edges),'selected edge block order')
    previous=2*len(point_ids)
    for b in blocks:
        require(b['scalar_start']==previous and b['scalar_end']>=previous+5,'finite block scalar partition')
        indices.extend((previous,previous+1));previous=b['scalar_end']
    return indices,previous

def guard_solver(solved,points,K,xyz,hidden):
    """Return one guarded solver view and complete selected-candidate evidence."""
    before=C.semantic_sha256(dict(solved=solved,points=points,K=K,xyz=xyz,hidden=hidden))
    result=copy.deepcopy(solved)
    counts=dict(guard_calls=1,geometry_info_calls=0,geometry_projection_entries=0,
        complete_geometry_info_calls=0,new_optimizer_calls=0,new_PnP_calls=0,
        actual_rank_info_dispatches_independently_counted=False)
    witness=dict(schema='selected_stable_incidence_witness_v14',policy=copy.deepcopy(POLICY),
        original_solver_semantic_sha256=C.semantic_sha256(solved),
        original_numerical_pose_available=bool(solved['available']),gate_applied=False,
        guard_state='PARENT_HAS_NO_NEW_POSE',accepted=False,selected_candidate_reselected=False,
        original_all_candidates_semantic_sha256=C.semantic_sha256(solved.get('all_candidate_solutions',[])),
        counts=counts,physical_ownership_truth='UNKNOWN',global_uniqueness_proven=False)
    if not solved['available']:
        result['stable_incidence_guard']=copy.deepcopy(witness)
        return result,witness
    q,K,xyz=(np.asarray(x,float) for x in (points,K,xyz));H=list(hidden)
    require(q.shape==(9,2) and K.shape==(3,3) and xyz.shape==(3,) and np.isfinite(K).all()
        and np.isfinite(xyz).all() and (xyz>0).all(),'original numeric observation contract')
    best=solved['best_candidate'];ids=list(solved['fit_input_ids']);edges=list(solved['fit_line_edges'])
    require(best['accepted'] is True and ids==best['fit_input_ids'] and edges==best['fit_line_edges'],
        'only the original accepted selected actual fit may be guarded')
    require(len(set(ids))==len(ids) and all(type(k) is int and 0<=k<8 for k in ids)
        and set(ids).isdisjoint(H) and set(ids)<=set(solved['used']),'H/center/point identity leakage')
    require(len(set(edges))==len(edges) and set(edges)<=set(solved['line_edges'])
        and set(edges).isdisjoint(solved['consumed_edges']),'line identity or duplicate point/line fit')
    sources={line['edge']:line for line in solved['actual_source_lines']}
    require(len(sources)==len(solved['actual_source_lines']) and all(e in sources for e in edges),'actual selected line source missing')
    lines=[dict(edge=e,normal=np.asarray(sources[e]['normal'],float),offset=float(sources[e]['offset'])) for e in edges]
    # Keep the unchanged HypothesisBank registry enumeration, including its
    # original near-square tolerance; this guard creates no new model branch.
    dims=[xyz] if abs(xyz[0]-xyz[2])<1e-9 else [xyz,xyz[[2,1,0]]]
    dim=best['dimension_index'];require(type(dim) is int and 0<=dim<len(dims),'original dimension identity')
    require(np.array_equal(np.asarray(best['cf_extents']),dims[dim]) and best['final_z'][3:]==solved['centroid'],
        'selected model/physical pose differs')
    z=np.asarray(best['final_z'],float);require(z.shape==(6,) and np.isfinite(z).all(),'selected pose must be finite')
    original=best['point_line_geometry'];indices,total=normal_row_indices(original,ids,edges)
    require(total==len(original['observed_normal_J'])==len(original['modeled_normal_J']),'selected finite J partition')
    observed=np.asarray(original['observed_normal_J'],float)[indices].reshape(-1,6)
    modeled=np.asarray(original['modeled_normal_J'],float)[indices].reshape(-1,6)
    witness.update(gate_applied=True,selected_candidate=copy.deepcopy(best),
        selected_dimension_index=dim,selected_extents=dims[dim].tolist(),selected_final_z=z.tolist(),
        actual_fit_input_ids=ids,actual_fit_line_edges=edges,hidden=H,
        actual_normal_lines=[dict(edge=l['edge'],normal=l['normal'].tolist(),offset=l['offset']) for l in lines],
        incidence_source_row_indices=indices,finite_scalar_rows=total,
        incidence_scalar_rows=2*len(ids)+2*len(edges),removed_longitudinal_rows=total-len(indices),
        observed_normal_J=observed.tolist(),modeled_normal_J=modeled.tolist(),
        exposed_J_source='exact point/endpoint-normal rows of immutable selected V13 candidate',
        new_geometry_source='unchanged V5.geometry_info on exactly the same selected actual fit')
    def projected():counts['geometry_projection_entries']+=1
    try:
        counts['geometry_info_calls']+=1
        geometry=V5.geometry_info(cuboid(*dims[dim]),z,K,q,ids,lines,projection_counter=projected)
        counts['complete_geometry_info_calls']+=1
        for key,J in (('observed_normal_joint',observed),('modeled_normal_joint',modeled)):
            scale=np.maximum(np.linalg.norm(J,axis=0),1e-300)
            require(np.array_equal(np.asarray(geometry[key]['normalized_column_scale']),scale),'recomputed incidence J scale differs from saved normal rows')
        rank=all(geometry[k]['numerical_rank']==6 for k in ('observed_normal_joint','modeled_normal_joint'))
        weak={name:len(geometry[name+'_normal_joint']['singular_values'])!=6 or
            geometry[name+'_normal_joint']['singular_values'][-1]<geometry[name+'_normal_joint']['singular_values'][0]*1e-4
            for name in ('observed','modeled')}
        state='PASSED_STABLE_INCIDENCE' if rank and not any(weak.values()) else 'STABLE_INCIDENCE_INSUFFICIENT' if not rank else 'STABLE_INCIDENCE_WEAK'
        witness.update(stable_incidence_geometry=geometry,observed_weak=weak['observed'],modeled_weak=weak['modeled'],
            guard_state=state,accepted=state=='PASSED_STABLE_INCIDENCE',complete_rank_info_calls_from_source=5)
    except (ValueError,TypeError,FloatingPointError,np.linalg.LinAlgError,cv2.error) as error:
        witness.update(guard_state='STABLE_INCIDENCE_NUMERICAL_FAILURE',accepted=False,
            error=dict(type=type(error).__name__,message=str(error)))
    if not witness['accepted']:
        result.update(available=False,pose_available=False,new_pose_estimated=False,no_pose=True,
            state=witness['guard_state'],reason='selected actual fit lacks sufficient stable point/line incidence',
            output_status='NO_NEW_POSE',local_refinement_estimated=False,
            rejected_finite_pose={k:copy.deepcopy(solved.get(k)) for k in ('state','R_cf','R_physical','centroid','cf_extents','projected','fit_input_ids','fit_line_edges','point_line_geometry')},
            R_cf=None,R_physical=None,centroid=None,cf_extents=None,projected=None,
            fit_input_ids=[],fit_line_edges=[],input_inliers=[],final_inliers=[],inliers=[],final_line_inliers=[],
            point_inlier_count=0,unique_line_inlier_count=0,total_inlier_factor_count=0,
            points_final=q.tolist(),hidden_reprojected=False)
    result['stable_incidence_guard']=copy.deepcopy(witness)
    require(before==C.semantic_sha256(dict(solved=solved,points=points,K=K,xyz=xyz,hidden=hidden)),
        'guard mutated original selected observations or candidate')
    return result,witness

def guard_row(parent):
    """Guard one complete GT-free V13 geometry row without another fit."""
    require(parent['method']==PARENT_METHOD and not TRUTH.intersection(parent),'complete unscored V13 geometry required')
    before=C.semantic_sha256(parent);solved,witness=guard_solver(parent['solver'],parent['input_points'],parent['K'],parent['xyz'],parent['hidden_initial'])
    row=copy.deepcopy(parent)
    if parent['solver']['available'] and not solved['available']:
        row.update(assemble(parent['original_native_N3_points'],parent['input_points'],parent['initial_pose'],
            parent['hidden_initial'],solved,'VALIDATED_ROLE_ONLY',parent['observation_contract']))
    else:
        row['solver']=solved
        if solved['available']:row['actual_pose']=solved
    row.update(method=METHOD,parent_method=PARENT_METHOD,stable_incidence_policy=copy.deepcopy(POLICY),
        stable_incidence_guard=copy.deepcopy(witness),parent_V13_row_semantic_sha256=before,
        numerical_NEW_is_not_accuracy_success=True,global_uniqueness_proven=False,
        actual_existing_finite_optimizer_refit_repeated=False,accepted_H_projection_assigned_again=False)
    require(before==C.semantic_sha256(parent),'original V13 geometry row changed')
    return row,witness
