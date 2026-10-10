"""Preserve the complete failed cusp inspector and join its separate full review.

The geometry and score/statistic kernels remain byte-identical. Only the
independent inspector's one-sided derivative at an endpoint rounding cusp is
reviewed; no failed receipt is replaced or represented as a PASS.
"""
from pathlib import Path
from . import evaluation as E
from . import runner as R
from ..pallet_sparse_local_line_20261010_v9 import common as C


REVIEW_FILES=('REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json')
ORIGINAL_FILES=('VALIDATION_PROTOCOL.json','VALIDATION_STARTED.json','VALIDATION_CHECKS.json')


def geometry_gate(args):
    own=R.frozen(args);p=R.DOC
    seal,receipt,failed,first,first_started,check,review,started=[C.read(p/n) for n in (
        'GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json',
        'VALIDATION_STARTED.json','REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json','REVIEW_VALIDATION_STARTED.json')]
    C.require(seal['schema']=='endpoint_observability_geometry_seal_v13' and seal['complete'] is True
        and seal['frames']==seal['ledger_rows']==seal['rows']==245 and seal['GT_read_allowed'] is False
        and seal['cleanup_completed_before_seal'] is True,'complete full245 GT-free seal required')
    C.require(receipt['schema']=='endpoint_observability_repair_execution_v13' and receipt['complete'] is True
        and receipt['cleanup_completed'] is True and receipt['error'] is None and receipt['cleanup_error'] is None,
        'complete restored fitting scope required')
    C.require(first['schema']=='finite_edge_consensus_scalar_protocol_v13'
        and failed['schema']=='finite_edge_consensus_scalar_checks_v13'
        and failed['complete'] is True and failed['passed'] is False and failed['checks']==713802
        and failed['failure_count']==74 and failed['exception'] is None and failed['inputs']==first['inputs'],
        'preserve the complete original 713802-check/74-failure inspector')
    C.bound(p/'VALIDATION_PROTOCOL.json',failed['own_protocol'],'original failed checker protocol')
    C.bound(R.CODE/'validation_checks.py',failed['checker'],'unchanged original failed checker')
    C.require(first_started['inputs']==first['inputs'] and first_started['new_production_calls']==0,
        'original inspector STARTED authority differs')
    C.bound(p/'VALIDATION_PROTOCOL.json',first_started['own_protocol'],'original inspector STARTED protocol')
    C.require(review['schema']=='finite_edge_consensus_scalar_protocol_v13'
        and check['schema']=='finite_edge_consensus_scalar_checks_v13' and check['complete'] is True
        and check['passed'] is True and check['exception'] is None and check['failure_count']==0
        and check['inputs']==review['inputs'],'separate complete reviewed inspector PASS required')
    C.bound(p/'REVIEW_VALIDATION_PROTOCOL.json',check['own_protocol'],'actual reviewed inspector protocol')
    C.bound(R.CODE/'validation_checks.py',check['checker'],'unchanged underlying inspector')
    C.require(started['inputs']==review['inputs'] and started['new_production_calls']==0,
        'review STARTED authority differs')
    C.bound(p/'REVIEW_VALIDATION_PROTOCOL.json',started['own_protocol'],'actual reviewed STARTED protocol')
    for alias,value in first['inputs'].items():
        C.require(alias in review['inputs'] and C.same_binding(value,review['inputs'][alias]),
            'review changed an original inspector input '+alias)
    for alias,path in (('geometry',p/'GEOMETRY_SEALED.jsonl.gz'),('ledgers',p/'CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol',Path(args.protocol)),('geometry_seal',p/'GEOMETRY_SEAL.json'),
        ('repair_receipt',p/'CONTROL_RECEIPT.json'),('parent_population_checks',p/'PARENT_POPULATION_CHECKS.json'),
        ('checker',R.CODE/'validation_checks.py'),('review_checker',R.CODE/'validation_review.py'),
        ('original_failure',p/'VALIDATION_CHECKS.json'),('original_validation_protocol',p/'VALIDATION_PROTOCOL.json'),
        ('review_contract',p/'CUSP_REVIEW_CONTRACT_KO.md')):
        C.bound(path,review['inputs'][alias],'actual reviewed input '+alias)
    marker=check['review']
    C.require(marker['original_checker_changed'] is False and marker['numerical_core_or_saved_rows_changed'] is False
        and marker['performance_thresholds_changed'] is False and marker['original_complete_failed_audit_preserved'] is True
        and marker['original_checks']==713802 and marker['original_failure_count']==74
        and marker['AST_patch_sites']==7 and marker['actual_roundoff_branch_reconciliations']==4
        and marker['one_sided_cusp_rank_is_not_two_sided_observability_proof'] is True,
        'exact floating-cusp inspector supplement required')
    C.bound(R.CODE/'validation_checks.py',marker['AST_witness']['original'],'review unchanged inspector AST source')
    C.require(marker['AST_witness']['AST_patch_sites']==7,'declared inspector AST sites differ')
    events=check['cusp_reconciliations']
    C.require(len(events)==4 and all(e['id']=='plastic_night_01:035421' and e['edge']==2 and e['query_id']==20
        and e['phase'] in ('proposal','refit') and e['analytic_derivative_recomputed'] is True
        and e['performance_threshold_changed'] is False for e in events),'only the original rounding cusp may be reconciled')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('CONTROL_RECEIPT.json','control_receipt'),
        ('PARENT_POPULATION_CHECKS.json','parent_population_checks')):
        C.bound(p/name,seal[key],'unchanged sealed '+name)
    return own,seal,receipt,check


def authority_inputs():
    paths=dict(wrapper=Path(__file__),original_score_kernel=Path(E.__file__),
        original_checker=R.CODE/'validation_checks.py',review_checker=R.CODE/'validation_review.py',
        review_contract=R.DOC/'CUSP_REVIEW_CONTRACT_KO.md')
    paths.update({n:R.DOC/n for n in (*ORIGINAL_FILES,*REVIEW_FILES)})
    return {key:C.binding(path) for key,path in paths.items()}


def scoring_authority(args):
    geometry_gate(args);path=R.DOC/'SCORING_AUTHORITY_SUPPLEMENT.json';expected=authority_inputs()
    if args.stage=='score':
        C.write_new(R.output(args,path.name),dict(schema='preserved_cusp_inspector_joined_review_authority_v13',
            inputs=expected,original_complete_audit_passed=False,original_checks=713802,original_failure_count=74,
            separate_full_review_PASS=True,numerical_core_or_saved_rows_changed=False,
            original_score_statistics_kernels_changed=False,
            GT_access_only_after_complete_geometry_cleanup_and_full_review_PASS=True))
    else:
        value=C.read(path)
        C.require(value['schema']=='preserved_cusp_inspector_joined_review_authority_v13' and value['inputs']==expected
            and value['original_complete_audit_passed'] is False and value['separate_full_review_PASS'] is True
            and value['numerical_core_or_saved_rows_changed'] is False,
            'same preserved-failure scoring authorization required')


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('score','stats');args=p.parse_args()
    scoring_authority(args)
    original=E.geometry_gate;E.geometry_gate=geometry_gate
    try:{'score':E.score,'stats':E.stats}[args.stage](args)
    finally:E.geometry_gate=original


if __name__=='__main__':main()
