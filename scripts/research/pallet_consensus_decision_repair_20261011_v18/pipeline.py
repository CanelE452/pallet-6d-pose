"""Saved decision replay and one real fresh point-only solve share one policy.

Imports of numerical production code occur only in the fresh bank factory.
The saved driver has no production imports or newly computed pose.
"""
import math
from .solver import ORIGINAL,METHOD,METHODS,PRIMARY,PARENT_METHOD,POLICY,clone,require,semantic,restore_solver,finite

def hidden_after_diagnostic(pose):
    """The unchanged V1 convex-cuboid, two-degree display diagnostic.

    This is pose-to-mask scalar arithmetic only, never a fit/ranking gate.
    The driver binds inference.hidden_mask and cuboid as source authority.
    """
    extents,R,t=pose['cf_extents'],pose['R_cf'],pose['centroid']
    camera=[-sum(R[j][i]*t[j] for j in range(3)) for i in range(3)]
    signs=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
    hidden=[];cosines=[]
    for k,sign in enumerate(signs):
        ray=[camera[i]-sign[i]*extents[i]/2 for i in range(3)]
        norm=math.sqrt(sum(v*v for v in ray));require(norm>0,'finite diagnostic camera ray required')
        cosine=max(sign[i]*ray[i]/norm for i in range(3));cosines.append(cosine)
        if cosine < -math.sin(math.radians(2.)):hidden.append(k)
    return hidden,dict(margin_deg=2.,cosine=cosines,geometry='convex cuboid proxy; not actual mesh visibility',
        scalar_pose_to_mask_diagnostic_only=True,used_for_acceptance_or_pose_selection=False)

def assemble_row(parent,solved,decision,*,execution_kind):
    """Preserve q and metadata; project H only for an approved final pose."""
    before=semantic(parent);row=clone(parent);H=parent['hidden_initial']
    native=clone(parent['original_native_N3_points']);q=clone(parent['input_points'])
    initial=parent['initial_pose'];new=bool(solved['available']);fallback=not new and bool(initial['available'])
    points=clone(native)
    if new:
        for k in range(8):
            if k in H:points[k]=clone(solved['projected'][k])
            elif all(finite(v) for v in q[k]) and q[k]!=[-1,-1]:points[k]=clone(q[k])
    if decision['restored']:
        after,visibility=hidden_after_diagnostic(solved)
    else:after,visibility=clone(parent['hidden_after']),None
    adopted=set(parent['selected_corner_ids'])
    sources=['FINAL_POSE_REPROJECTION' if new and k in H else
        'VALIDATED_BOUNDARY_INTERSECTION' if new and k in adopted else 'NATIVE_N3_RGB' for k in range(8)]+['UNCHANGED_DETECTOR_CENTER']
    row.update(method=METHOD,parent_method=PARENT_METHOD,solver=solved,
        native_points=points,actual_pose=solved if new else clone(initial),
        new_pose_estimated=new,pose_available=new or bool(initial['available']),fallback_used=fallback,
        output_status='NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE',
        no_pose=not(new or bool(initial['available'])),hidden_reprojected=bool(new and H),
        hidden_after=after,hidden_set_changed=bool(new and set(after)!=set(H)),
        output_coordinate_sources=sources,restored_pose_mask_diagnostic=visibility,
        reprojected_ids=clone(H) if new else [],initial_H_applied=True,
        original_self_hidden_initial=clone(H),reprojections_reused_as_observations=False,
        partial_lines_not_pose_inputs=True,local_point_line_refinement=False,
        global_uniqueness_proven=False,decision_restoration=clone(decision),
        decision_restoration_policy=clone(POLICY),decision_execution_kind=execution_kind,
        original_point_only_row_semantic_sha256=before,
        selected_output_pose_was_actually_fit_in_parent=bool(new and solved.get('best_candidate') is not None),
        saved_decision_replay=execution_kind=='SAVED_DECISION_REPLAY',
        cached_accuracy_row_used=execution_kind=='SAVED_DECISION_REPLAY',
        solver_replay_latency_claim=False,numerical_NEW_is_not_pose_accuracy_success=True)
    require(row['input_points']==q and row['native_points'][8]==native[8]
        and not(set(H)&set(solved['fit_input_ids'])) and semantic(parent)==before,
        'q/center/parent/Hfit changed')
    return row

def restore_saved_row(parent):
    require(parent['method']==PARENT_METHOD and parent['oracle'] is False
        and parent['partial_lines_not_pose_inputs'] is True,'original sealed nonoracle point-only arm required')
    for field in ('evaluation_reference','pose','human','GT','visibility_truth','metric_targets'):
        require(field not in parent,'scored or truth input forbidden: '+field)
    solved,decision=restore_solver(parent['solver'],parent['input_points'],parent['hidden_initial'],
        execution_kind='SAVED_DECISION_REPLAY')
    # These operations belong to replay display, not the parent's fit counters.
    row=assemble_row(parent,solved,decision,execution_kind='SAVED_DECISION_REPLAY')
    ledger=dict(schema='saved_point_only_decision_ledger_v18',id=parent['id'],session=parent['session'],method=METHOD,
        parent_row_semantic_sha256=semantic(parent),parent_solver_semantic_sha256=semantic(parent['solver']),
        decision=clone(decision),actual_new_solver_calls=0,actual_new_optimizer_projection_J_SVD_calls=0,
        inherited_solver_operation_counts=clone(parent['solver']['operation_counts']),
        replay_display_H_assignments=decision['new_display_H_assignments'],
        scalar_pose_to_mask_diagnostic_calls=int(decision['restored']),
        original_H=clone(parent['hidden_initial']),input_points=clone(parent['input_points']),
        original_native_N3_points=clone(parent['original_native_N3_points']),
        independent_point_only_scalar_authority_is_driver_requirement=True,
        latency_claim=False,cached_replay_latency_claim=False)
    return row,ledger

def restore_packet(parent):
    """Return two displays of the same already fitted point-only candidate pool."""
    before=semantic(parent);restored,detail=restore_saved_row(parent)
    original=clone(parent)
    original.update(method=ORIGINAL,parent_method=PARENT_METHOD,
        original_point_only_row_semantic_sha256=before,
        decision_execution_kind='SAVED_ORIGINAL_DECISION_COPY',actual_new_solver_calls=0,
        saved_decision_replay=True,cached_accuracy_row_used=True,solver_replay_latency_claim=False)
    ledger=dict(schema='saved_point_only_decision_pair_ledger_v18',id=parent['id'],session=parent['session'],
        methods=list(METHODS),parent_method=PARENT_METHOD,parent_row_semantic_sha256=before,
        parent_solver_semantic_sha256=semantic(parent['solver']),
        original_row_semantic_sha256=semantic(original),restored_row_semantic_sha256=semantic(restored),
        decision=clone(detail['decision']),inherited_solver_operation_counts=clone(detail['inherited_solver_operation_counts']),
        actual_new_solver_calls=0,actual_new_optimizer_projection_J_SVD_calls=0,
        original_output_view_copies=1,restored_output_view_decisions=1,
        replay_display_H_assignments=detail['replay_display_H_assignments'],
        scalar_pose_to_mask_diagnostic_calls=detail['scalar_pose_to_mask_diagnostic_calls'],
        original_H=clone(detail['original_H']),input_points=clone(detail['input_points']),
        original_native_N3_points=clone(detail['original_native_N3_points']),
        independent_point_only_scalar_authority_is_driver_requirement=True,
        latency_claim=False,cached_replay_latency_claim=False)
    require(semantic(parent)==before,'parent packet changed')
    return [original,restored],ledger

def fresh_bank_class():
    """Return a subclass; never replace globals or edit the frozen solver."""
    from ..pallet_factor_subset_consensus_20261011_v15.solver import FactorSubsetConsensusBank
    from ..pallet_sparse_local_line_20261010_v9 import common as C
    class FreshPointOnlyBank(FactorSubsetConsensusBank):
        def solve(self,observation,adopted_boundary_corner_ids=(),hidden=(),*,initial_pose=None,point_result=None):
            require(observation.get('lines')==[] and not adopted_boundary_corner_ids,
                'fresh restoration is strictly point-only')
            original=super().solve(observation,adopted_boundary_corner_ids=(),hidden=hidden,
                initial_pose=initial_pose,point_result=point_result)
            H=list(original['hidden'])
            # Use the unchanged wire normalizer, including absent XY as None.
            # This changes no numeric input and matches the saved-row contract.
            solved,decision=restore_solver(C.finite(original),C.finite(self.points),H,
                execution_kind='FRESH_ORIGINAL_POINT_ONLY_SOLVE')
            assignments=decision['new_display_H_assignments']
            if assignments:
                self.counts['hidden_reprojection_assignments']+=assignments
                solved['operation_counts']['hidden_reprojection_assignments']+=assignments
                solved['local_bank_counts']=dict(self.counts)
            solved['decision_restoration']=decision
            solved['fresh_original_point_only_solver_calls']=1
            solved['fresh_second_solver_calls']=0
            solved['saved_accuracy_row_used_as_input']=False
            # Parent counters describe real fits. Only new H copies are added.
            return solved
    return FreshPointOnlyBank
