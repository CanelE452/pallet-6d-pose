"""Pure decision restoration; no numerical geometry or fitting imports.

The original candidate generator/refit and its independent point-only scalar
audit own geometry validity. This module checks their saved decision fields.
Same inlier IDs do not erase a strict difference in same-whole-U score.
"""
import copy
import hashlib
import json
import math

ORIGINAL='HYBRID_POINT_CONSENSUS_ORIGINAL'
METHOD='HYBRID_POINT_CONSENSUS_RESTORED'
METHODS=(ORIGINAL,METHOD)
PRIMARY=METHOD
PARENT_METHOD='HYBRID_FACTOR_SUBSET_POINT_ONLY'
TIE_PX2=1e-8
POLICY=dict(schema='point_only_count_SSE_decision_restoration_v18',
    parent_method=PARENT_METHOD,method=METHOD,methods=list(METHODS),
    candidate_generator_refit_geometry_or_threshold_changes=0,
    ranking='original -whole_U_inlier_count, whole_U_truncated_SSE, dimension, logical_start_index',
    numerical_score_tie_px2=TIE_PX2,
    ambiguity='non-equivalent accepted alternative with same vote count and abs whole-U truncated-SSE difference <= inherited1e-8',
    same_support_different_cost_veto_removed=True,
    different_fit_pool_soft_l1_costs_compared=False,
    chosen_pose_or_observations_reselected=False,
    original_actual_inlier_rank_weak_depth_and_min4_gates_preserved=True,
    H_and_center_not_fit_or_scored=True,H_projected_once_without_refit=True,
    cornerwise_H_union_k_LOO_unchanged=True,
    physical_ownership_certified=False,global_uniqueness_proven=False,
    known_DEV=True,unseen_evaluation=False,new_training_RGB_seeds_or_threshold_search=False)

def require(value,message):
    if not value:raise ValueError(message)
def clone(value):return copy.deepcopy(value)
def semantic(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def ids(value):
    require(isinstance(value,list) and value==sorted(set(value)) and all(type(k) is int and 0<=k<8 for k in value),'sorted corner IDs0..7 required')
    return value
def key(candidate):return (-candidate['score_count'],candidate['truncated_sse_px2'],candidate['dimension_index'],candidate['logical_start_index'])

def verify_point_only_decision(s,H):
    """Validate stored flags/whole-pool ordering, not independent geometry."""
    ids(H);U=ids(s['used'])
    require(s['hidden']==H and not(set(U)&set(H)) and s['line_edges']==s['fit_line_edges']==s['final_line_inliers']==s['consumed_edges']==[],
        'point-only same-H observations required')
    for field in ('prior_used','known_dimension_constraint_used','reprojected_points_reused_as_observations','same_edge_point_line_double_count','center_used_for_fit_or_score'):
        require(s[field] is False,'original no-prior/no-projection-refit contract '+field)
    require(s['hidden_initial_excluded'] is True and s.get('actual_source_lines',[])==[],'H exclusion/no source line factors')
    accepted=[c for c in s['all_candidate_solutions'] if c['accepted'] is True]
    for c in accepted:
        fit,final=ids(c['fit_input_ids']),ids(c['final_inliers'])
        require(c['rejected'] is None and len(fit)>=4 and set(fit)<=set(U) and set(final)<=set(U)
            and c['fit_line_edges']==c['final_line_inliers']==[] and c['whole_pool_point_ids']==U and c['whole_pool_line_edges']==[],
            'original sufficient actual point-only refit/whole-U score required')
        require(c['score_count']==len(final) and finite(c['truncated_sse_px2']) and c['truncated_sse_px2']>=0
            and c['fit_pool_all_final8px_inliers'] is True,'original actual-inlier consensus acceptance')
        g,guard=c['point_line_geometry'],c['stable_incidence_guard']
        require(g['point_ids']==fit and g['line_edges']==[] and all(g[n]['numerical_rank']==6 for n in ('observed_normal_joint','modeled_normal_joint'))
            and g['observed_weak'] is False and g['modeled_weak'] is False and guard['accepted'] is True,
            'stored finite and stable-incidence gates retained; independent proof is external')
        require(len(c['model_camera_depths_m'])==8 and all(finite(z) and z>0 for z in c['model_camera_depths_m'])
            and len(c['projected'])==8 and all(len(p)==2 and all(finite(v) for v in p) for p in c['projected']),
            'original accepted finite/positive-depth projection witness')
    best=s.get('best_candidate');alternatives=s['alternatives']
    if best is None:
        require(not accepted and not alternatives and not s['available'],'no accepted best must stay unavailable')
        return dict(accepted_candidates=0,extra_veto_alternatives=0,true_tie_alternatives=0,non_equivalent_alternatives=0)
    require(accepted and best==sorted(accepted,key=key)[0] and s['viable_candidate_count']==len(accepted)==len(alternatives)+1,
        'original count/full-U SSE/dimension/start sorted best and complete viable pool')
    ranked=sorted(accepted,key=key);extra=ties=non_equivalent=0
    for original,relation in zip(ranked[1:],alternatives):
        require(all(relation[k]==v for k,v in original.items()),'complete original alternative candidate retained')
        delta=relation['truncated_sse_px2']-best['truncated_sse_px2']
        tied=relation['score_count']==best['score_count'] and abs(delta)<=TIE_PX2
        support=relation['final_inliers']==best['final_inliers'] and relation['final_line_inliers']==best['final_line_inliers']
        equivalent=relation['numerically_equivalent_to_selected']
        require(type(equivalent) is bool and relation['numerical_objective_tie'] is tied
            and relation['same_actual8px_consensus_as_selected'] is support
            and relation['objective_difference_from_selected_px2']==delta,'original comparable whole-U alternative predicates')
        if not equivalent:
            non_equivalent+=1;ties+=int(tied);extra+=int(support and not tied)
    require(s['multiple_solutions'] is bool(non_equivalent),'preserved non-equivalent multiplicity flag')
    return dict(accepted_candidates=len(accepted),extra_veto_alternatives=extra,true_tie_alternatives=ties,
        non_equivalent_alternatives=non_equivalent)

def restore_solver(original,input_points,H,*,execution_kind):
    """Keep the original best. Remove only its extra any-cost support veto."""
    require(execution_kind in ('SAVED_DECISION_REPLAY','FRESH_ORIGINAL_POINT_ONLY_SOLVE'),'explicit execution provenance required')
    before=semantic(original);checks=verify_point_only_decision(original,H);s=clone(original)
    require(isinstance(input_points,list) and len(input_points)==9 and all(isinstance(p,list) and len(p)==2 for p in input_points),'original9x2 q required')
    repaired=False;assignments=0;old_state=s['state']
    if old_state=='AMBIGUOUS_FINITE_CONSENSUS':
        old_reasons=set(s['reason'].split(';'))
        require(old_reasons and old_reasons<={'OBSERVATION_CONSENSUS_AMBIGUITY','NUMERICAL_CONSENSUS_TIE'}
            and s['available'] is False and s['unresolved_ambiguity'] is True,'only declared original ambiguity may change')
        if checks['true_tie_alternatives']:
            s['reason']='NUMERICAL_CONSENSUS_TIE'
        else:
            require(checks['extra_veto_alternatives']>0 and old_reasons=={'OBSERVATION_CONSENSUS_AMBIGUITY'},'only different-cost same-support veto may restore')
            best=s['best_candidate'];points=clone(input_points)
            for k in H:points[k]=clone(best['projected'][k])
            assignments=len(H);repaired=True
            s.update(available=True,pose_available=True,new_pose_estimated=True,no_pose=False,
                unresolved_ambiguity=False,state='NEW_POSE',reason='restored_original_count_SSE_point_consensus_decision',output_status='NEW_POSE',
                local_refinement_estimated=True,dimension_index=best['dimension_index'],selected_hypothesis=best['selected_hypothesis'],
                fit_input_ids=clone(best['fit_input_ids']),input_inliers=clone(best['fit_input_ids']),
                final_inliers=clone(best['final_inliers']),inliers=clone(best['final_inliers']),
                fit_line_edges=[],final_line_inliers=[],point_inlier_count=len(best['final_inliers']),unique_line_inlier_count=0,
                total_inlier_factor_count=best['score_count'],points_final=points,hidden_reprojected=bool(H),
                factor_sse_px2=best['whole_pool_score']['factor_sse_px2'],
                residuals_used_px=clone(best['whole_pool_score']['point_residual_norm_px']),
                line_residual_RMS_px=clone(best['whole_pool_score']['edge_block_norm_px']))
            for field in ('R_cf','R_physical','centroid','cf_extents','projected','point_line_geometry','stable_incidence_guard',
                'scalar_residuals_px','soft_l1_cost','sse_px2','truncated_sse_px2'):s[field]=clone(best[field])
    else:
        require(not checks['true_tie_alternatives'] or not s['available'],'true non-equivalent numeric tie cannot already be accepted')
    s.update(global_uniqueness_proven=False,decision_restoration_policy=clone(POLICY),
        inherited_solver_policy_is_candidate_generation_history=True,
        operative_output_decision_policy=clone(POLICY),
        original_same_support_veto_state=old_state,original_same_support_veto_reason=original['reason'])
    require(s.get('best_candidate')==original.get('best_candidate'),'best pose must never be reselected')
    require(s['all_candidate_solutions']==original['all_candidate_solutions'] and s['alternatives']==original['alternatives']
        and semantic(original)==before and not(set(s['fit_input_ids'])&set(H)),'immutable full candidate pool and H exclusion')
    decision=dict(schema='point_only_decision_restoration_receipt_v18',original_solver_semantic_sha256=before,
        execution_kind=execution_kind,checks=checks,original_state=old_state,new_state=s['state'],restored=repaired,
        true_non_equivalent_numeric_tie_retained=checks['true_tie_alternatives']>0,
        new_display_H_assignments=assignments,best_candidate_reselected=False,all_candidate_witnesses_changed=False,
        new_solver_optimizer_projection_J_SVD_calls=0,saved_replay_new_solver_calls=0,
        inherited_geometry_validity_independent_audit_required=True,latency_claim=False,cached_replay_latency_claim=False)
    return s,decision
