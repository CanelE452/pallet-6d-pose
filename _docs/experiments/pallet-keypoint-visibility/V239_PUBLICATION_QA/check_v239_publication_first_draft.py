"""Verify copied package, entry point suffixes and bounded checkpoint claims."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = REPO / '_docs/experiments/pallet-keypoint-visibility'
PACKAGE = BASE / 'V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION'


def safe(p):
    p = Path(p)
    assert not any(x in str(p).lower() for x in ('v112', '__pycache__', '.pyc'))
    assert not p.is_symlink()
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(rel):
    return safe(PACKAGE / rel).read_bytes()


def main():
    prep = json.loads((R / 'V239_MAIN_PREPARATION.json').read_bytes())
    vb = read('verify_package.py')
    assert sha(vb) == 'bd34cada69b56591171a3ee0686bb4a6eb8c16c3bdf8ebfe994db13b7dac30bc'
    run = subprocess.run([sys.executable, '-B', str(PACKAGE / 'verify_package.py'),
                          '--package', str(PACKAGE), '--manifest', 'MANIFEST.json'],
                         capture_output=True, text=True, check=True)
    result = json.loads(run.stdout)
    assert result['PASS'] and result['manifest_sha256'] == prep['manifest_sha256']
    for f in prep['copied'] + prep['entry_points']:
        b = safe(REPO / f['path']).read_bytes()
        assert (len(b), sha(b)) == (f['bytes'], f['sha256'])
    prefix_links = []
    for entry in prep['entry_points']:
        p = safe(REPO / entry['path'])
        b = p.read_bytes()
        old = subprocess.check_output(['git', 'show', prep['parent_commit'] + ':' + entry['path']], cwd=REPO)
        assert len(old) == entry['old_bytes'] and sha(old) == entry['old_sha256']
        assert b[entry['prefix_bytes']:] == old
        prefix = b[:entry['prefix_bytes']].decode()
        for link in re.findall(r'!?\[[^\]\n]*\]\(([^\s\)]+)\)', prefix):
            assert ':' not in link and not link.startswith('/')
            target = safe(p.parent / link)
            assert target.is_file(), link
            prefix_links.append({'entrypoint': entry['path'], 'link': link})
    complete = json.loads(read('artifacts/V237_ROOT_SOURCE_CONTEXT_INPUTS/original_source_inputs_v1/COMPLETE.json'))
    assert complete['completed'] and complete['PASS']
    assert complete['known7617_original_RGB96_and_valid_SHA_exact'] == 7617
    failed = json.loads(read('artifacts/V238_ROOT_CONTEXT_RESIDUAL/context_residual_source_fit_v1/FAILED_EXECUTION.json'))
    assert not failed['completed'] and not failed['PASS']
    diag = json.loads(read('artifacts/V238_ALL_FIT_DESCRIPTOR_DIAG/DIAGNOSIS.json'))
    assert diag['FIT_rows'] == 5876 and diag['exceed_1e_6_rows'] == 66
    assert diag['max_abs'] == 1.5515834093093872e-6 and diag['first_failing_batch'] == 4
    assert diag['new_head_or_optimizer_updates'] == 0
    assert diag['tolerance_or_threshold_changed'] is False
    scope = json.loads(read('artifacts/V239_ROOT_CANONICAL_CONTEXT/PUBLICATION_CHECKPOINT_STATUS.json'))
    assert scope['V239_actual_feature_generation'] == scope['V239_actual_head_fit'] == scope['V239_new_DEV_outputs'] == 0
    assert scope['new_current_or_held_payload'] == 0 and not scope['new_PnP']
    assert scope['V239_independent_paired_CODE_QA'] == 'PENDING'
    for folder in ('V239_CANONICAL_FEATURE_PRODUCER_B', 'V239_CANONICAL_PAIRED_TRAINER_A'):
        ready = json.loads(read('artifacts/' + folder + '/FINAL_READY.json'))
        assert ready['completed'] and ready['PASS'] and ready['prepared_only']
    historical = 'artifacts/V238_ROOT_CONTEXT_RESIDUAL/context_residual_source_fit_v1/'
    assert sha(read(historical + 'CONTROL_LOCAL_CROP_FINAL1000.pt')) == '6f6f2e0d82a3bd9296991e50d8a7e3f8151df78a01d140a4d658504091340e0a'
    assert sha(read(historical + 'CONTROL_LOCAL_CROP_ALL_SOURCE_ROWS.csv')) == 'e638a38c196f58fe287f998f8636e102f5ea3abe43a7aede9f953c9d527dab8b'
    old_report = safe(BASE / 'V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING/artifacts/V235_SOURCE_REPORT_C/results/PORTABLE_REPORT_KO.md')
    assert old_report.is_file()
    out = {'completed': True, 'PASS': True, 'manifest_sha256': prep['manifest_sha256'],
           'copied_package_verifier': result, 'copied_files_verified': len(prep['copied']),
           'entrypoints_suffix_preserved': 3, 'new_prefix_links_verified': prefix_links,
           'bounded_scientific_status_checks': True, 'historical_CONTROL_bytes_exact': True,
           'external_historical_report_exact_target_exists': True,
           'metadata_pointers_followed': False, 'arrays_numeric_decoded': False,
           'model_run': False, 'current_or_held_new_payload': 0,
           'verifier': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).read_bytes())},
           'limits': 'Publication byte/link/status verification only; no new classifier performance.'}
    p = R / 'V239_ROOT_PUBLICATION_CHECKS.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'receipt_sha256': sha(p.read_bytes())}))


if __name__ == '__main__':
    main()
