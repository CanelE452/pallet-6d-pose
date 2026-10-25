"""Stage only the reviewed V239 publication allowlist and verify every index blob."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
QA = BASE + '/V239_PUBLICATION_QA'


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
    prep = json.loads((R / 'V239_MAIN_PREPARATION.json').read_bytes())
    root = json.loads((R / 'V239_ROOT_PUBLICATION_CHECKS.json').read_bytes())
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
        ('V239_MAIN_PREPARATION.json', 'V239_MAIN_PREPARATION.json'),
        ('V239_ROOT_PUBLICATION_CHECKS.json', 'V239_ROOT_PUBLICATION_CHECKS.json'),
        ('check_v239_publication.py', 'check_v239_publication.py'),
        ('check_v239_publication_first_draft.py', 'check_v239_publication_first_draft.py'),
        ('V239_FIRST_PUBLICATION_CHECK_FAILURE.json', 'V239_FIRST_PUBLICATION_CHECK_FAILURE.json'),
        ('prepare_v239_main.py', 'prepare_v239_main.py'),
        ('stage_v239_main.py', 'stage_v239_main.py'),
        (safe(a.independent_receipt), 'independent/ACTUAL_PACKAGE_QA.json'),
        (safe(a.independent_code), 'independent/verify_declared_package.py'),
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
    review = {
        'schema': 'V239_PUBLICATION_REVIEW_V1', 'completed': True, 'PASS': True,
        'parent_commit': prep['parent_commit'], 'versions': ['V237', 'V238', 'V239'],
        'manifest_sha256': prep['manifest_sha256'],
        'published_files_including_this_review': len(expected) + 1,
        'scientific_status': {'V237_input_rebuild': 'COMPLETE',
                              'V238_actual': 'FAILED_BEFORE_OPTIMIZER',
                              'V239_actual': 'NOT_EXECUTED_PREPARATION_ONLY',
                              'model_adopted': False, 'new_current_evaluation': False},
        'V238_fit_descriptor_diagnosis': {'FIT_rows': 5876, 'exceed_1e_6': 66,
                                         'max_abs': 1.5515834093093872e-6,
                                         'first_failing_batch': 4, 'first_failing_sample': 397,
                                         'new_optimizer_updates': 0, 'new_DEV_outputs': 0},
        'current_or_held_new_payload': 0, 'new_PnP': False,
        'tolerance_relaxed': False, 'source_gate_pass': False,
        'private_evaluation_used_for_fit_or_selection': False,
        'large_RGB_valid_and_failed_feature_caches_excluded_with_bindings': True,
        'entrypoints_old_bytes_preserved_as_suffix': True,
        'absolute_metadata_pointers_followed': False,
        'root_checks': root, 'independent_checks': ind,
        'limits': ['Package QA is not model success.',
                   'V239 paired CODE integration QA remains pending at this checkpoint.',
                   'Prepared toy checks are not actual SOURCE feature generation or training.',
                   'Copied V238 CONTROL weights and outputs are historical V233 bytes.'],
    }
    rel = BASE + '/V239_PUBLICATION_REVIEW.json'
    b = (json.dumps(review, ensure_ascii=False, indent=2) + '\n').encode()
    assert not (REPO / rel).exists()
    (REPO / rel).write_bytes(b)
    expected[rel] = {'sha256': sha(b), 'bytes': len(b)}
    deferred = root['new_prefix_review_link_checks_deferred']
    assert len(deferred) == 1
    assert deferred[0]['target_relative_to_repo'] == rel
    assert (REPO / safe(deferred[0]['target_relative_to_repo'])).is_file()
    allowlist = R / 'V239_GIT_PATHS.nul'
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
    p = R / 'V239_STAGED_QA.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'staged_files': len(staged),
                      'receipt_sha256': sha(p.read_bytes())}))


if __name__ == '__main__':
    main()
