"""Use unchanged frozen statistics arithmetic with actual reviewed authority.

The original failed inspector remains a distinct input. Successful aliases
point to the separately frozen full review receipts; no receipt is synthesized
or changed. This wrapper changes no scalar, bootstrap, verdict or scoring AST.
"""
import ast
import hashlib
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
CODE=Path(__file__).resolve().parent
SOURCE=CODE/'verify.py'
SOURCE_SHA='bad35cd12dd449d2f02e772e715b4dc3b04c74f74b1e34f508a0288bb53e018e'
FILES=('REVIEW_STATISTICS_VERIFICATION_PROTOCOL.json','REVIEW_STATISTICS_VERIFICATION_STARTED.json','REVIEW_VERIFICATION.json')
MARKER=dict(original_statistics_verifier_changed=False,statistics_computation_AST_changed=False,
    original_failed_geometry_audit_preserved=True,original_complete_audit_passed=False,
    original_checks=713802,original_failure_count=74,successful_aliases_are_actual_separate_review_files=True,
    saved_rows_score_model_GT_fit_SVD_new_draw_changes=0,original_statistics_verifier_CLI_invocation=False,unchanged_statistics_AST_used=True)


def require(value,message):
    if not value:raise ValueError(message)


def load_kernel():
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SOURCE_SHA,'frozen original V13 verifier bytes differ')
    tree=ast.parse(SOURCE.read_text());witness=dict(source_sha256=SOURCE_SHA,
        full_unchanged_AST_sha256=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest())
    env=dict(__file__=str(SOURCE),__name__='unchanged_frozen_v13_statistics_review_kernel')
    exec(compile(tree,str(SOURCE)+' [unchanged full statistics AST]','exec'),env)
    return env,witness


def main():
    env,witness=load_kernel();old_inputs,old_metadata,old_write=env['inputs'],env['metadata'],env['write']
    env['FILES']=FILES
    def reviewed_inputs(args):
        paths=old_inputs(args);p=args.input
        for alias,name in (('geometry_validation','CHECKS'),('geometry_validation_protocol','PROTOCOL'),
            ('geometry_validation_started','STARTED')):
            paths['original_'+alias]=paths[alias];paths[alias]=p/('REVIEW_VALIDATION_'+name+'.json')
        paths['original_statistics_started']=paths['statistics_started'];paths['statistics_started']=p/'REVIEW_STATISTICS_STARTED.json'
        paths.update(statistics_review=CODE/'statistics_review.py',statistics_authority=p/'STATISTICS_AUTHORITY_SUPPLEMENT.json',
            statistics_first_failure=p/'STATISTICS_FIRST_FAILURE.json',inherited_diagnostic_code=REPO/'scripts/research/pallet_sparse_local_line_20261010_v9/statistics.py',
            review_verifier=Path(__file__).resolve(),review_checker=CODE/'validation_review.py',
            review_contract=p/'CUSP_REVIEW_CONTRACT_KO.md',authority_wrapper=CODE/'evaluation_review.py',
            scoring_authority=p/'SCORING_AUTHORITY_SUPPLEMENT.json')
        return paths
    def reviewed_metadata(paths,b):
        read,same,binding=env['read'],env['same'],env['binding']
        failed,first,first_started,check,review,started,authority=[read(paths[k]) for k in (
            'original_geometry_validation','original_geometry_validation_protocol','original_geometry_validation_started',
            'geometry_validation','geometry_validation_protocol','geometry_validation_started','scoring_authority')]
        require(failed['schema']=='finite_edge_consensus_scalar_checks_v13' and failed['complete'] is True
            and failed['passed'] is False and failed['checks']==713802 and failed['failure_count']==74
            and failed['exception'] is None and failed['inputs']==first['inputs'],
            'preserve complete original 713802-check/74-failure inspector')
        require(first['schema']=='finite_edge_consensus_scalar_protocol_v13'
            and same(failed['own_protocol'],b['original_geometry_validation_protocol'])
            and same(failed['checker'],b['geometry_checker']) and first_started['inputs']==first['inputs']
            and same(first_started['own_protocol'],b['original_geometry_validation_protocol'])
            and first_started['new_production_calls']==0,'original failed inspector authority differs')
        for key,value in first['inputs'].items():
            require(key in review['inputs'] and same(value,review['inputs'][key]),'review changed original inspector '+key)
        for key,alias in (('review_checker','review_checker'),('review_contract','review_contract'),
            ('original_failure','original_geometry_validation'),('original_validation_protocol','original_geometry_validation_protocol')):
            require(same(review['inputs'][key],b[alias]),'review supplemental source binding differs '+key)
        marker=check['review']
        require(marker['original_checker_changed'] is False and marker['numerical_core_or_saved_rows_changed'] is False
            and marker['performance_thresholds_changed'] is False and marker['original_complete_failed_audit_preserved'] is True
            and marker['original_checks']==713802 and marker['original_failure_count']==74
            and marker['AST_patch_sites']==7 and marker['actual_roundoff_branch_reconciliations']==4
            and marker['one_sided_cusp_rank_is_not_two_sided_observability_proof'] is True,
            'exact separate scalar-cusp review marker required')
        require(same(marker['AST_witness']['original'],b['geometry_checker'])
            and marker['AST_witness']['AST_patch_sites']==7,'original checker AST witness differs')
        events=check['cusp_reconciliations']
        require(len(events)==4 and all(e['id']=='plastic_night_01:035421' and e['edge']==2 and e['query_id']==20
            and e['phase'] in ('proposal','refit') and e['analytic_derivative_recomputed'] is True
            and e['performance_threshold_changed'] is False for e in events),'only original rounding cusp may be reconciled')
        require(authority['schema']=='preserved_cusp_inspector_joined_review_authority_v13'
            and authority['original_complete_audit_passed'] is False and authority['original_checks']==713802
            and authority['original_failure_count']==74 and authority['separate_full_review_PASS'] is True
            and authority['numerical_core_or_saved_rows_changed'] is False
            and authority['original_score_statistics_kernels_changed'] is False
            and authority['GT_access_only_after_complete_geometry_cleanup_and_full_review_PASS'] is True,
            'preserved-failure scoring supplement differs')
        keys={'wrapper':'authority_wrapper','original_score_kernel':'own_code','original_checker':'geometry_checker',
            'review_checker':'review_checker','review_contract':'review_contract',
            'VALIDATION_PROTOCOL.json':'original_geometry_validation_protocol',
            'VALIDATION_STARTED.json':'original_geometry_validation_started','VALIDATION_CHECKS.json':'original_geometry_validation',
            'REVIEW_VALIDATION_PROTOCOL.json':'geometry_validation_protocol',
            'REVIEW_VALIDATION_STARTED.json':'geometry_validation_started','REVIEW_VALIDATION_CHECKS.json':'geometry_validation'}
        require(set(authority['inputs'])==set(keys),'full scoring authorization closure required')
        for key,alias in keys.items():require(same(authority['inputs'][key],b[alias]),'scoring authorization current bytes differ '+key)
        first_stats=read(paths['original_statistics_started']);review_stats=read(paths['statistics_started'])
        failed_stats=read(paths['statistics_first_failure']);stats_authority=read(paths['statistics_authority'])
        require(first_stats==review_stats,'actual reviewed statistics STARTED scope differs')
        require(failed_stats['schema']=='preserved_v13_statistics_attempt_failure' and failed_stats['complete'] is False
            and failed_stats['passed'] is False and failed_stats['actual_exit_code']==1
            and failed_stats['stage']=='posthoc_contract_before_summaries_and_CIs'
            and failed_stats['error']['type']=='RuntimeError'
            and failed_stats['error']['message']=='SAME_OBSERVATION_GUARD: accepted local diagnostic support schema required'
            and failed_stats['models_GT_scoring_fits_new_draws']==0 and failed_stats['produced_Metrics_or_posthoc_rows'] is False,
            'preserve actual original posthoc schema failure')
        stats_keys={'wrapper':'statistics_review','original_statistics_code':'own_code',
            'inherited_diagnostic_code':'inherited_diagnostic_code','unchanged_V7_statistics':'statistics_kernel',
            'geometry_authority_wrapper':'authority_wrapper','original_started':'original_statistics_started',
            'original_failure':'statistics_first_failure','scoring_authority':'scoring_authority','scoring_receipt':'scoring_receipt'}
        require(stats_authority['schema']=='preserved_posthoc_schema_failure_finite_metadata_review_v13'
            and set(stats_authority['inputs'])==set(stats_keys) and stats_authority['original_failed_attempt_preserved'] is True
            and stats_authority['original_scores_and_statistics_code_changed'] is False
            and stats_authority['moments_bootstrap_pairing_and_verdict_functions_changed'] is False
            and stats_authority['V7_posthoc_function_changed'] is False and stats_authority['V9_local_fields_not_used'] is True
            and stats_authority['new_finite_diagnostic_metadata_describes_actual_fit'] is True
            and stats_authority['new_model_RGB_GT_scoring_fit_SVD_draw_calls']==0,'statistics metadata review authority differs')
        for key,alias in stats_keys.items():require(same(stats_authority['inputs'][key],b[alias]),'statistics authority byte join '+key)
        for key,alias in (('original_started','original_statistics_started'),('original_statistics_code','own_code'),
            ('inherited_diagnostic_code','inherited_diagnostic_code'),('scoring_receipt','scoring_receipt')):
            require(same(failed_stats[key],b[alias]),'original statistics failure authority differs '+key)
        require(binding(SOURCE)['sha256']==SOURCE_SHA,'original statistics verifier changed')
        # The immutable original metadata receives actual separately successful
        # files. Its original fixed-input, scoring, cleanup and population tests
        # all execute unchanged before its arithmetic is authorized.
        return old_metadata(paths,b)
    def reviewed_write(path,value):
        value['authority_review']=dict(MARKER,unchanged_full_statistics_AST=witness,
            review_verifier=env['binding'](__file__))
        old_write(path,value)
    env['inputs']=reviewed_inputs;env['metadata']=reviewed_metadata;env['write']=reviewed_write;env['main']()


if __name__=='__main__':main()
