"""Preserve the inherited posthoc schema failure and adapt finite-edge metadata.

The frozen score/statistics scalar, moment, bootstrap and verdict functions
remain unchanged. V7 posthoc is reused with a label adapter; V13 fit geometry
is described truthfully instead of claiming the older V9 diagnostic contract.
"""
import traceback
from pathlib import Path
from . import evaluation as E
from . import evaluation_review as A
from . import runner as R
from ..pallet_sparse_local_line_20261010_v9 import common as C
from ..pallet_sparse_local_line_20261010_v9 import statistics as S9
from ..pallet_three_head_observation_20261010_v7 import statistics as S


def finite_fields(row,n3,point):
    solver=row['solver'];new=bool(row['new_pose_estimated']);pool=solver.get('factor_pool')
    supplied_points=list(solver['used']);supplied_edges=list(solver['line_edges'])
    fitted_points=list(solver['fit_input_ids']) if new else []
    fitted_edges=list(solver['fit_line_edges']) if new else []
    geometry=solver.get('point_line_geometry') if new else None
    if new:
        C.require(solver['local_refinement_estimated'] is True and solver['initial_pose_start_used'] is True
            and solver['initial_pose_residual_prior'] is False and solver['global_uniqueness_proven'] is False
            and solver['inlier_geometry_is_acceptance_gate'] is True,
            'accepted finite-edge actual-inlier contract required')
        C.require(geometry['point_ids']==fitted_points and geometry['line_edges']==fitted_edges
            and geometry['observed_normal_joint']['numerical_rank']==geometry['modeled_normal_joint']['numerical_rank']==6
            and not geometry['observed_weak'] and not geometry['modeled_weak'],
            'stored actual finite-inlier fit geometry differs')
        C.require(pool['point_ids']==supplied_points and pool['line_edges']==supplied_edges,
            'stored full finite scoring pool differs')
    return dict(actual_supplied_point_ids=supplied_points,actual_supplied_unused_line_edges=supplied_edges,
        supplied_factor_pool=pool,accepted_fit_point_ids=fitted_points,accepted_fit_line_edges=fitted_edges,
        accepted_final_inlier_ids=list(solver['final_inliers']) if new else [],
        accepted_final_line_inliers=list(solver['final_line_inliers']) if new else [],
        numeric_finite_NEW=new,point_comparator_NEW=point['new_pose_estimated'],
        numeric_rescue_over_point=bool(new and not point['new_pose_estimated']),
        actual_fit_has_fewer_than4_points=bool(new and len(fitted_points)<4),
        actual_fit_has_unused_lines=bool(new and fitted_edges),
        point_only_local_refinement=bool(new and not fitted_edges),
        actual_fit_observed_rank=geometry['observed_normal_joint']['numerical_rank'] if geometry else None,
        actual_fit_modeled_rank=geometry['modeled_normal_joint']['numerical_rank'] if geometry else None,
        actual_fit_scalar_residuals=len(geometry['actual_scalar_residuals_px']) if geometry else None,
        actual_fit_observed_weak=geometry['observed_weak'] if geometry else None,
        actual_fit_modeled_weak=geometry['modeled_weak'] if geometry else None,
        actual_inlier_geometry_is_acceptance_gate=solver['inlier_geometry_is_acceptance_gate'],
        finite_hinge_residuals_included=True,scalar_rows_are_not_2_times_points_plus_edges=True,
        consensus_votes_one_per_distinct_edge=True,whole_pool_scoring_is_not_all_actual_fit_observations=True,
        finite_inlier_geometry_is_not_old_V9_diagnostic_support_schema=True,
        one_sided_hinge_rank_is_not_stable_two_sided_observability_proof=True,
        initial_pose_start_used=solver['initial_pose_start_used'],initial_pose_residual_prior=False,
        global_uniqueness_proven=False,physical_ownership_certified=False,numeric_NEW_is_not_goal_success=True,
        actual_pose_vs_N3=S9.outcome(row,n3),actual_pose_vs_point=S9.outcome(row,point),
        line_effect_causally_isolated=False)


def posthoc(row,n3,point,observation):
    adapted=dict(row,method=C.PARENT_PRIMARY)
    result=S.posthoc(adapted,n3,observation)
    result.update(method=row['method'],posthoc_kernel_label_adapter_only=True,
        finite_edge_diagnostics=finite_fields(row,n3,point),
        old_V9_local_point_line_diagnostic_contract_not_claimed=True)
    return result


def authority_inputs():
    paths=dict(wrapper=Path(__file__),original_statistics_code=Path(E.__file__),
        inherited_diagnostic_code=Path(S9.__file__),unchanged_V7_statistics=Path(S.__file__),
        geometry_authority_wrapper=Path(A.__file__),
        original_started=R.DOC/'STATISTICS_STARTED.json',original_failure=R.DOC/'STATISTICS_FIRST_FAILURE.json',
        scoring_authority=R.DOC/'SCORING_AUTHORITY_SUPPLEMENT.json',scoring_receipt=R.DOC/'SCORING_RECEIPT.json')
    return {key:C.binding(path) for key,path in paths.items()}


def main():
    parser=R.parser();next(a for a in parser._actions if a.dest=='stage').choices=('stats',);args=parser.parse_args()
    A.geometry_gate(args);bindings=authority_inputs()
    failed=C.read(R.DOC/'STATISTICS_FIRST_FAILURE.json');first=C.read(R.DOC/'STATISTICS_STARTED.json')
    C.require(failed['schema']=='preserved_v13_statistics_attempt_failure' and failed['complete'] is False
        and failed['passed'] is False and failed['actual_exit_code']==1
        and failed['stage']=='posthoc_contract_before_summaries_and_CIs'
        and failed['error']['type']=='RuntimeError'
        and failed['error']['message']=='SAME_OBSERVATION_GUARD: accepted local diagnostic support schema required'
        and failed['models_GT_scoring_fits_new_draws']==0 and failed['produced_Metrics_or_posthoc_rows'] is False,
        'preserve the actual inherited V9 posthoc schema failure')
    for key in ('original_started','original_statistics_code','inherited_diagnostic_code','scoring_receipt'):
        C.require(C.same_binding(failed[key],bindings[key]),'preserved first statistics failure bytes differ '+key)
    for key,value in first['inputs'].items():
        path=C.REPO/value['path'];C.bound(path,value,'unchanged original statistics input '+key)
    for name in ('STATISTICS_AUTHORITY_SUPPLEMENT.json','REVIEW_STATISTICS_STARTED.json','REVIEW_STATISTICS_FAILURE.json'):
        R.output(args,name)
    C.write_new(R.output(args,'STATISTICS_AUTHORITY_SUPPLEMENT.json'),dict(
        schema='preserved_posthoc_schema_failure_finite_metadata_review_v13',inputs=bindings,
        original_failed_attempt_preserved=True,original_scores_and_statistics_code_changed=False,
        moments_bootstrap_pairing_and_verdict_functions_changed=False,V7_posthoc_function_changed=False,
        V9_local_fields_not_used=True,new_finite_diagnostic_metadata_describes_actual_fit=True,
        new_model_RGB_GT_scoring_fit_SVD_draw_calls=0,automatic_retry=False))
    old_gate,old_posthoc,old_output=E.geometry_gate,S9.posthoc,R.output
    def output(a,name):return old_output(a,'REVIEW_STATISTICS_STARTED.json' if name=='STATISTICS_STARTED.json' else name)
    E.geometry_gate=A.geometry_gate;S9.posthoc=posthoc;R.output=output
    try:
        E.stats(args)
        C.require(authority_inputs()==bindings,'statistics review authority bytes changed')
        current=C.read(R.DOC/'REVIEW_STATISTICS_STARTED.json')
        C.require(current==first,'review statistics input scope changed')
    except BaseException as exc:
        C.write_new(old_output(args,'REVIEW_STATISTICS_FAILURE.json'),dict(complete=False,
            error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),
            original_failed_attempt_preserved=True,automatic_retry=False))
        raise
    finally:E.geometry_gate=old_gate;S9.posthoc=old_posthoc;R.output=old_output


if __name__=='__main__':main()
