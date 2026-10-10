"""Seal verified public artifacts and check the original checkout is preserved."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from . import common as C


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--visual-inspected', action='store_true',
                        help='Record that a reviewer actually opened and inspected all seven PNGs')
    args = parser.parse_args()
    assert args.visual_inspected, 'Actual PNG visual review must precede finalization'
    verification = C.read(C.DOC / 'INDEPENDENT_VERIFICATION.json')
    assert verification['status'] == 'PASS' and verification['figures']['status'] == 'PASS'
    assert C.read(C.DOC / 'EVALUATOR_CONTRACT_REVIEW.json')['status'] == 'PASS'
    lock = C.read(C.DOC / 'INPUT_AND_METHOD_LOCK.json')
    assert C.sha(C.DOC / 'INPUT_AUDIT.json') == lock['input_audit_sha256']
    for bound in lock['code']:
        assert C.sha(C.WORKTREE / bound['path']) == bound['sha256'], bound['path']
    original = C.read(args.private_dir / 'input_audit_before_snapshot.json')
    source = args.source_root.resolve()
    assert git(source, 'rev-parse', 'HEAD').decode().strip() == original['head']
    assert git(source, 'branch', '--show-current').decode().strip() == original['branch']
    assert hashlib.sha256(git(source, 'status', '--porcelain=v1', '-z')).hexdigest() == original['status_sha256']
    assert hashlib.sha256(git(source, 'diff', '--binary', 'HEAD')).hexdigest() == original['tracked_diff_sha256']
    for path, expected in original['changed_tracked_sha256'].items():
        assert C.sha(source / path) == expected, path
    figure_index = C.read(C.DOC / 'FIGURE_INDEX.json')
    assert len(figure_index['figures']) == 7
    for fig in figure_index['figures']:
        assert C.sha(C.DOC / fig['path']) == fig['sha256']
    figures = dict(status='PASS', count=7, PNG_decode='PASS', actual_visual_inspection='PASS',
                   review='All seven PNGs opened; labels, axes, full error tails, CI zero lines and coordinate-only examples checked',
                   fixes='Moved accuracy labels away from point markers; replaced overlapping movement labels with legend',
                   RGB_published=False, rights='Original RGB redistribution rights unconfirmed; coordinate-only examples used')
    figure_index['visual_inspection'] = figures
    C.write(C.DOC / 'FIGURE_INDEX.json', figure_index)
    result = C.read(C.DOC / 'EXECUTION_AND_VERIFICATION.json')
    result.update(status='COMPLETE', complete=True,
        independent_verification=dict(status='PASS',file='INDEPENDENT_VERIFICATION.json',sha256=C.sha(C.DOC/'INDEPENDENT_VERIFICATION.json')),
        independent_evaluator_review=dict(status='PASS',file='EVALUATOR_CONTRACT_REVIEW.json',sha256=C.sha(C.DOC/'EVALUATOR_CONTRACT_REVIEW.json')),
        figures=figures,
        input_audit=dict(status='PASS',sealed_hash_matches=True,supplement_disclosed=True,
                         note='Original audit bytes exactly restored and retained; later evidence-only runtime/baseline supplement is separately published'),
        original_checkout=dict(status='PRESERVED',head_unchanged=True,branch_unchanged=True,porcelain_unchanged=True,
            tracked_binary_diff_unchanged=True,changed_tracked_files_unchanged=len(original['changed_tracked_sha256']),
            original_status_sha256=original['status_sha256'],original_tracked_diff_sha256=original['tracked_diff_sha256']),
        publication=dict(target='CanelE452/pallet-6d-pose:main',authorized_by='explicit user request',force_push=False,
                         status='READY_FOR_FAST_FORWARD_PUSH',remote_SHA_and_links_checked_after_push='final response and private publication receipt'),
        task_timing=dict(finalized_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                         accuracy_seconds=result['elapsed_seconds'],input_audit_seconds=C.read(C.DOC/'INPUT_AUDIT.json')['execution']['elapsed_seconds'],
                         note='Accuracy replay wall time is not deployment end-to-end latency; historical runtime reused'))
    result['artifact_sha256'] = {str(p.relative_to(C.DOC)):C.sha(p) for p in sorted(C.DOC.rglob('*'))
                                if p.is_file() and p.name != 'EXECUTION_AND_VERIFICATION.json'}
    result['source_sha256'] = {str(p.relative_to(C.WORKTREE)):C.sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    C.write(C.DOC / 'EXECUTION_AND_VERIFICATION.json', result)
    print(json.dumps(dict(status='COMPLETE',figures=7,original_checkout='PRESERVED',
                         actual_F_calls=result['actual_F_calls'],rows=result['rows'],new_training=0)))


if __name__ == '__main__':
    main()
