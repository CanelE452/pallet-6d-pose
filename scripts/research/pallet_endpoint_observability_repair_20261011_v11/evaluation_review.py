"""Join the preserved first checker failure and its separately frozen repair.

Only an inspector's optional absent best_candidate field is corrected. The
original numerical source, geometry, policy and score/statistic kernels stay
byte-identical. No geometric rerun or replacement of a failed receipt occurs.
"""
from pathlib import Path
from . import evaluation as E
from . import runner as R
from ..pallet_sparse_local_line_20261010_v9 import common as C


def geometry_gate(args):
    own=R.frozen(args);p=R.DOC
    seal,receipt,failed,first,check,review=[C.read(p/n) for n in (
        'GEOMETRY_SEAL.json','CONTROL_RECEIPT.json','VALIDATION_CHECKS.json','VALIDATION_PROTOCOL.json',
        'REVIEW_VALIDATION_CHECKS.json','REVIEW_VALIDATION_PROTOCOL.json')]
    C.require(seal['complete'] is True and seal['frames']==seal['ledger_rows']==245 and seal['rows']==490
              and seal['GT_read_allowed'] is False and seal['cleanup_completed_before_seal'] is True,
              'complete full245/490 GT-free seal required')
    C.require(receipt['complete'] is True and receipt['cleanup_completed'] is True and receipt['error'] is None
              and receipt['cleanup_error'] is None,'complete restored fit scope required')
    C.require(failed['complete'] is False and failed['passed'] is False and failed['checks']==95
              and failed['exception']['type']=='KeyError' and failed['exception']['message']=="'best_candidate'"
              and failed['inputs']==first['inputs'],'preserved specific inspector schema failure required')
    C.bound(p/'VALIDATION_PROTOCOL.json',failed['own_protocol'],'original checker protocol')
    C.bound(R.CODE/'validation_checks.py',failed['checker'],'unchanged original failed checker')
    C.require(check['schema']=='supplemental_endpoint_observability_scalar_checks_v11'
              and check['complete'] is True and check['passed'] is True and check['inputs']==review['inputs'],
              'separate full repair scalar PASS required')
    C.bound(p/'REVIEW_VALIDATION_PROTOCOL.json',check['own_protocol'],'actual review checker protocol')
    for alias,path in (('geometry',p/'GEOMETRY_SEALED.jsonl.gz'),('ledgers',p/'CONTROL_LEDGERS.jsonl.gz'),
        ('repair_protocol',Path(args.protocol)),('geometry_seal',p/'GEOMETRY_SEAL.json'),
        ('repair_receipt',p/'CONTROL_RECEIPT.json'),('checker',R.CODE/'validation_checks.py')):
        C.bound(path,review['inputs'][alias],'separate checked '+alias)
        C.require(C.same_binding(first['inputs'][alias],review['inputs'][alias]),'review changed original '+alias)
    # Review metadata and code are themselves declared and frozen independently.
    for alias,path in (('review_checker',R.CODE/'validation_review.py'),
        ('original_failure',p/'VALIDATION_CHECKS.json'),('original_validation_protocol',p/'VALIDATION_PROTOCOL.json')):
        C.bound(path,review['inputs'][alias],'supplemental inspector '+alias)
    C.require(check['review']['original_checker_changed'] is False
              and check['review']['numerical_core_or_saved_rows_changed'] is False
              and check['review']['AST_optional_best_candidate_access_changes']==1,
              'exact one-field inspector supplement required')
    for name,key in (('PROTOCOL.json','protocol'),('GEOMETRY_SEALED.jsonl.gz','geometry'),
        ('CONTROL_LEDGERS.jsonl.gz','ledgers'),('CONTROL_RECEIPT.json','control_receipt')):
        C.bound(p/name,seal[key],'unchanged sealed '+name)
    return own,seal,receipt,check


def execute(args):
    # Restore the original guard after this call; all original model/fit bans
    # inside scoring and the full freeze/cleanup guards remain in force.
    original=E.geometry_gate;E.geometry_gate=geometry_gate
    try:{'score':E.score,'stats':E.stats}[args.stage](args)
    finally:E.geometry_gate=original


def main():
    p=R.parser();next(a for a in p._actions if a.dest=='stage').choices=('score','stats');args=p.parse_args()
    # Scorer authorization supplement is fixed after the inspector review and
    # before the first score; its files are added, never overwrite old files.
    geometry_gate(args)
    if args.stage=='score':
        C.write_new(R.output(args,'SCORING_AUTHORITY_SUPPLEMENT.json'),dict(
            schema='preserved_inspector_failure_joined_review_authority_v11',
            wrapper=C.binding(__file__),review_protocol=C.binding(R.DOC/'REVIEW_VALIDATION_PROTOCOL.json'),
            review_PASS=C.binding(R.DOC/'REVIEW_VALIDATION_CHECKS.json'),
            original_failure=C.binding(R.DOC/'VALIDATION_CHECKS.json'),
            original_score_kernel=C.binding(E.__file__),numerical_core_or_saved_rows_changed=False,
            GT_access_only_after_complete_geometry_cleanup_and_full_review_PASS=True))
    else:
        authority=C.read(R.DOC/'SCORING_AUTHORITY_SUPPLEMENT.json')
        C.bound(__file__,authority['wrapper'],'same scoring authorization wrapper')
        C.bound(E.__file__,authority['original_score_kernel'],'unchanged score/statistics kernel')
    execute(args)


if __name__=='__main__':main()
