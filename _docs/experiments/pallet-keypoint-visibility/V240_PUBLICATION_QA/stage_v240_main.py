"""Stage only the reviewed V240 publication allowlist and verify every index blob."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
QA = BASE + '/V240_PUBLICATION_QA'


def safe(rel):
    assert isinstance(rel, str) and not Path(rel).is_absolute() and '..' not in Path(rel).parts
    assert not any(x in rel.lower() for x in ('v112', '__pycache__', '.pyc'))
    return rel


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--independent-receipt', required=True)
    ap.add_argument('--independent-sha256', required=True)
    ap.add_argument('--independent-code', required=True)
    a = ap.parse_args()
    prep = json.loads((R / 'V240_MAIN_PREPARATION_V2.json').read_bytes())
    root = json.loads((R / 'V240_ROOT_PUBLICATION_CHECKS.json').read_bytes())
    assert prep['PASS'] and root['PASS']
    assert git('rev-parse', 'HEAD').decode().strip() == prep['parent_commit']
    assert not git('diff', '--cached', '--name-only', '-z')
    ip = R / safe(a.independent_receipt)
    ib = ip.read_bytes()
    assert sha(ib) == a.independent_sha256
    ind = json.loads(ib)
    assert ind['completed'] is True and ind['PASS'] is True
    assert ind['manifest_sha256'] == prep['manifest_sha256']
    expected = {safe(f['path']): {'sha256': f['sha256'], 'bytes': f['bytes']}
                for f in prep['copied'] + prep['entry_points']}
    for rel, item in expected.items():
        b = (REPO / rel).read_bytes()
        assert (len(b), sha(b)) == (item['bytes'], item['sha256'])
    copies = [
        ('V240_MAIN_PREPARATION.json', 'first_attempt/V240_MAIN_PREPARATION.json'),
        ('V240_MAIN_PREPARATION_V2.json', 'V240_MAIN_PREPARATION_V2.json'),
        ('V240_ROOT_PUBLICATION_CHECKS.json', 'V240_ROOT_PUBLICATION_CHECKS.json'),
        ('prepare_v240_main.py', 'prepare_v240_main.py'),
        ('update_v240_main_v2.py', 'update_v240_main_v2.py'),
        ('check_v240_publication.py', 'check_v240_publication.py'),
        ('stage_v240_main.py', 'stage_v240_main.py'),
        ('V240_PUBLICATION_CPU_CONTROL.json', 'V240_PUBLICATION_CPU_CONTROL.json'),
        ('V240_PUBLICATION_CPU_TREATMENT.json', 'V240_PUBLICATION_CPU_TREATMENT.json'),
        ('V240_PUBLICATION_ARITHMETIC_REPLAY.json', 'V240_PUBLICATION_ARITHMETIC_REPLAY.json'),
        (safe(a.independent_receipt), 'independent/ACTUAL_PACKAGE_QA.json'),
        (safe(a.independent_code), 'independent/verify_declared_package_v2.py'),
        ('V240_PUBLICATION_VERIFY_C/actual_package_v2/ROOT_ACTUAL_ENTRY_PACKAGE_V2.json', 'independent/ROOT_ACTUAL_ENTRY_PACKAGE_V2.json'),
        ('V240_PUBLICATION_VERIFY_C/FAILED_EXECUTION.json', 'first_attempt/INDEPENDENT_FAILED_EXECUTION.json'),
    ]
    for src, dst in copies:
        p = R / safe(src)
        assert not p.is_symlink()
        b = p.read_bytes()
        rel = safe(QA + '/' + dst)
        target = REPO / rel
        assert not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b)
        expected[rel] = {'sha256': sha(b), 'bytes': len(b)}
    status = json.loads((REPO / BASE / 'V240_CANONICAL_CONTEXT_ACTUAL_RESULTS/STATUS.json').read_bytes())
    review = {
        'schema': 'V240_PUBLICATION_REVIEW_V2', 'completed': True, 'PASS': True,
        'parent_commit': prep['parent_commit'], 'versions': ['V239', 'V240'],
        'manifest_sha256': prep['manifest_sha256'],
        'published_files_including_this_review': len(expected) + 1,
        'canonical_features': 'COMPLETE_7617_ACTIVE_BYTES_EXACT',
        'paired_heads': 'BOTH_NEW_1000_AND_FROZEN',
        'SOURCE_gate_PASS': False, 'model_adopted': False,
        'source_status': status, 'source_gate_gain_pp': 0.3456571415751661,
        'new_current_or_held_payload': 0, 'new_current_evaluation': False, 'new_PnP': False,
        'private_evaluation_used_for_fit_or_selection': False,
        'original_q_L_native_U_FIT_DEV_preserved': True,
        'relocated_CPU_predictions_exact_each_arm': 7617,
        'relocated_stdlib_metric_replay_PASS': True,
        'large_duplicate_NODE_PIXEL_FEATURES_excluded_with_binding': True,
        'six_classifier_input_arrays_and_two_small_models_included': True,
        'entrypoints_old_bytes_preserved_as_suffix': True,
        'root_checks': root, 'independent_checks': ind,
        'first_publication_failure': {
            'terminal': '125682 exit1',
            'reason': 'Report linked raw paired JSONL but first package supplied gzip only; standalone scalar replay also needs original raw leaf.',
            'repair': 'New immutable V2 package includes exact raw paired rows plus gzip. Original source report/model/features unchanged.',
            'failed_original_package_manifest': 'fb27e4cb2e8c79cd5a15545b600c259538a6643f6464d5be02989cab64a08560',
            'failure_receipt_sha256': 'a951d3211a4d5d4aba34c6968e20923817337f948b693cb9fa2e70496afad98c',
            'original_package_preserved': True,
        },
        'limits': ['Publication and replay PASS is not model success; SOURCE gate failed.',
                   'SOURCE DEV is exposed/reused; YCB AUX native correspondence UNKNOWN.',
                   'Six frozen descriptors replay heads, not raw image/GT/encoder generation.',
                   'No next CF generation or next model training result exists.'],
    }
    rel = BASE + '/V240_PUBLICATION_REVIEW.json'
    b = (json.dumps(review, ensure_ascii=False, indent=2) + '\n').encode()
    assert not (REPO / rel).exists()
    (REPO / rel).write_bytes(b)
    expected[rel] = {'sha256': sha(b), 'bytes': len(b)}
    deferred = root['new_prefix_review_link_checks_deferred']
    assert len(deferred) == 1
    assert deferred[0]['target_relative_to_repo'] == rel
    assert (REPO / safe(deferred[0]['target_relative_to_repo'])).is_file()
    allowlist = R / 'V240_GIT_PATHS.nul'
    with allowlist.open('xb') as f:
        f.write(b''.join(safe(p).encode() + b'\0' for p in sorted(expected)))
    subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', '-f',
                    '--pathspec-from-file=' + str(allowlist), '--pathspec-file-nul'], cwd=REPO, check=True)
    staged = [safe(p.decode()) for p in git('diff', '--cached', '--name-only', '-z').split(b'\0') if p]
    assert set(staged) == set(expected)
    for rel in staged:
        b = git('show', ':' + rel)
        assert (len(b), sha(b)) == (expected[rel]['bytes'], expected[rel]['sha256']), rel
    check = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,-blank-at-eof,cr-at-eol',
                            'diff', '--cached', '--check'], cwd=REPO, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    out = {'completed': True, 'PASS': True, 'parent_commit': prep['parent_commit'],
           'staged_files': len(staged), 'index_blobs_byte_exact': True,
           'whitespace_check_exit_code': check.returncode, 'files': expected}
    p = R / 'V240_STAGED_QA.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'staged_files': len(staged),
                      'receipt_sha256': sha(p.read_bytes())}))


if __name__ == '__main__':
    main()
