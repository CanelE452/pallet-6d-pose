"""Publish QA evidence and stage only the V236 explicit path allowlist."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = '_docs/experiments/pallet-keypoint-visibility'
QA = f'{BASE}/V236_PUBLICATION_QA'


def safe(value):
    value = str(value)
    assert 'v112' not in value.lower()
    assert '__pycache__' not in value and not value.endswith('.pyc')
    assert not Path(value).is_absolute() and '..' not in Path(value).parts
    return value


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


prep = json.loads((ROOT / 'V236_MAIN_PREPARATION.json').read_text())
assert git('rev-parse', 'HEAD').decode().strip() == prep['parent_commit']
assert not git('diff', '--cached', '--name-only', '-z')
expected = {safe(f['path']): {'sha256': f['sha256'], 'bytes': f['bytes']}
            for f in prep['copied'] + prep['entry_points']}
for rel, binding in expected.items():
    b = (REPO / rel).read_bytes()
    assert sha(b) == binding['sha256'] and len(b) == binding['bytes']

ready_path = ROOT / 'V236_PUBLICATION_VERIFY_A/FINAL_READY.json'
ready_bytes = ready_path.read_bytes()
assert sha(ready_bytes) == 'e348cf0dd7db83e3581cc57c2caacef4b646fec1d0886a794c15236aecd90121'
ready = json.loads(ready_bytes)
assert ready['completed'] and ready['PASS'] and ready['package_mutations'] == 0
independent = (ROOT / 'V236_PUBLICATION_VERIFY_A/ACTUAL_PACKAGE_QA.json').read_bytes()
assert sha(independent) == '1e7464b2c86d6b57b94b9435835e77218f7463d363f8278d88a66c233a010833'
root_checks_bytes = (ROOT / 'V236_ROOT_PUBLICATION_CHECKS.json').read_bytes()
assert sha(root_checks_bytes) == '395c06e69e5be16215de7ee71ea95df0b29dec7d1171546dd4d88c36ad617f89'
root_checks = json.loads(root_checks_bytes)
assert root_checks['PASS']

qa_copies = []
root_names = ['V236_MAIN_PREPARATION.json', 'V236_COPIED_BYTES_AND_ENTRYPOINT_QA.json',
              'V236_ROOT_PUBLICATION_CHECKS.json', 'prepare_v236_main.py', 'stage_v236_main.py']
agent_names = ['verify_declared_package.py', 'verify_declared_package_v2.py',
               'ACTUAL_PACKAGE_QA.json', 'FIRST_EXECUTION_FAILURE.json',
               'VERIFICATION_POLICY_CORRECTION.json', 'FINAL_READY.json']
for src_rel, dst_name in [(n, n) for n in root_names] + [
        (f'V236_PUBLICATION_VERIFY_A/{n}', f'independent/{n}') for n in agent_names]:
    src = ROOT / safe(src_rel)
    assert not src.is_symlink()
    b = src.read_bytes()
    rel = safe(f'{QA}/{dst_name}')
    target = REPO / rel
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b)
    binding = {'sha256': sha(b), 'bytes': len(b)}
    expected[rel] = binding
    qa_copies.append({'path': rel, **binding})

review = {
    'schema': 'V236_PUBLICATION_REVIEW_V1', 'completed': True, 'PASS': True,
    'versions': ['V234', 'V235', 'V236'], 'parent_commit': prep['parent_commit'],
    'package_manifest_sha256': prep['manifest_sha256'],
    'package_files_including_manifest': 223, 'package_bytes_excluding_manifest': 47182119,
    'published_file_count_including_this_review': 238,
    'model_adopted': False, 'source_gate_pass': False, 'source_gain_pp': 0.113491,
    'YCB_internal_DEV_external_recall': {'correct': 0, 'total': 57},
    'new_current_or_held_payload_reads': 0, 'new_current_model_evaluation': False,
    'private_evaluation_used_for_fit_or_selection': False, 'new_PnP': False,
    'original_q_L_native_and_unknown_preserved': True,
    'large_pair_RGB_and_existing_raw_or_prefix_caches_not_copied': True,
    'source_paths_and_SHA_bindings': 'V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/PUBLICATION_SOURCE_MAP.json',
    'root_actual_verification': root_checks,
    'independent_ready': ready,
    'qa_evidence_files': qa_copies,
    'preserved_first_strict_link_failure': {
        'terminal': '649eb5 exit1', 'reason': 'Standalone package checker rejected intentional historical ../V229 link.',
        'resolution': 'Classified that exact link as external-relative and not followed by independent verifier. Root confirmed the exact historical target exists in the main checkout.',
        'package_or_experiment_mutation': False,
    },
    'copied_byte_QA_and_three_entrypoint_suffix_preservation': 'V236_PUBLICATION_QA/V236_COPIED_BYTES_AND_ENTRYPOINT_QA.json',
    'git_whitespace_policy': 'blank-at-eol,space-before-tab,-blank-at-eof,cr-at-eol; preserve byte-pinned executed source endings and CSV CRLF.',
    'limits': ['Publication verification is not scientific model success.',
               'Only manifest/package files and new entrypoint prefixes were checked; historical documents are retained byte-for-byte as suffixes.',
               'Independent package hash/gzip/link verification did not run models or decode images/NPY numerically.',
               'Package includes saved-result verification, not a one-command full training environment.'],
}
review_rel = f'{BASE}/V236_PUBLICATION_REVIEW.json'
review_bytes = (json.dumps(review, ensure_ascii=False, indent=2) + '\n').encode()
assert not (REPO / review_rel).exists()
(REPO / review_rel).write_bytes(review_bytes)
expected[review_rel] = {'sha256': sha(review_bytes), 'bytes': len(review_bytes)}
assert len(expected) == 238

allowlist = ROOT / 'V236_GIT_PATHS.nul'
allowlist.write_bytes(b''.join(safe(p).encode() + b'\0' for p in sorted(expected)))
subprocess.run(['git', '-c', 'core.autocrlf=false', 'add', '-f',
                f'--pathspec-from-file={allowlist}', '--pathspec-file-nul'], cwd=REPO, check=True)
staged = [safe(p.decode()) for p in git('diff', '--cached', '--name-only', '-z').split(b'\0') if p]
assert set(staged) == set(expected)
for rel in staged:
    b = git('show', ':' + rel)
    assert sha(b) == expected[rel]['sha256'] and len(b) == expected[rel]['bytes'], rel
whitespace = subprocess.run(['git', '-c', 'core.whitespace=blank-at-eol,space-before-tab,-blank-at-eof,cr-at-eol',
                             'diff', '--cached', '--check'], cwd=REPO, capture_output=True, text=True)
assert whitespace.returncode == 0, whitespace.stdout + whitespace.stderr
out = {'completed': True, 'PASS': True, 'staged_files': len(staged), 'index_blobs_byte_exact': True,
       'whitespace_check_exit_code': whitespace.returncode, 'parent_commit': prep['parent_commit'],
       'review_sha256': sha(review_bytes), 'files': expected}
receipt = ROOT / 'V236_STAGED_QA.json'
receipt.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'completed': True, 'PASS': True, 'staged_files': len(staged),
                  'index_blobs_byte_exact': True, 'review_sha256': sha(review_bytes),
                  'receipt_sha256': sha(receipt.read_bytes())}))
