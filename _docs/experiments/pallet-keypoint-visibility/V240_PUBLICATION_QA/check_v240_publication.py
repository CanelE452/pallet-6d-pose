"""Verify copied V240 bytes, old entry point suffixes and relocated public replays."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys

R = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
REPO = Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
BASE = REPO / '_docs/experiments/pallet-keypoint-visibility'
PACKAGE = BASE / 'V240_CANONICAL_CONTEXT_ACTUAL_RESULTS'


def safe(p):
    p = Path(p)
    assert not any(x in str(p).lower() for x in ('v112', '__pycache__', '.pyc'))
    assert not p.is_symlink()
    return p


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    prep = json.loads((R / 'V240_MAIN_PREPARATION_V2.json').read_bytes())
    assert sha((PACKAGE / 'verify_package.py').read_bytes()) == 'bd34cada69b56591171a3ee0686bb4a6eb8c16c3bdf8ebfe994db13b7dac30bc'
    run = subprocess.run([sys.executable, '-B', str(PACKAGE / 'verify_package.py'),
                          '--package', str(PACKAGE), '--manifest', 'MANIFEST.json'],
                         capture_output=True, text=True, check=True)
    integrity = json.loads(run.stdout)
    assert integrity['completed'] and integrity['PASS']
    assert integrity['manifest_sha256'] == prep['manifest_sha256'] == '3204602a3a2abe306a02e65e95fd13c7759bf6cac40679dd37dc6a1e414df5fa'
    for f in prep['copied'] + prep['entry_points']:
        b = safe(REPO / f['path']).read_bytes()
        assert (len(b), sha(b)) == (f['bytes'], f['sha256'])
    links = []; deferred = []
    for entry in prep['entry_points']:
        p = safe(REPO / entry['path']); b = p.read_bytes()
        old = subprocess.check_output(['git', 'show', prep['parent_commit'] + ':' + entry['path']], cwd=REPO)
        assert b[entry['prefix_bytes']:] == old and sha(old) == entry['old_sha256']
        for link in re.findall(r'!?\[[^\]\n]*\]\(([^\s\)]+)\)', b[:entry['prefix_bytes']].decode()):
            assert ':' not in link and not link.startswith('/')
            target = safe(p.parent / link)
            if entry['path'] == '_docs/experiments/pallet-keypoint-visibility/README.md' and link == 'V240_PUBLICATION_REVIEW.json':
                deferred.append({'entrypoint': entry['path'], 'link': link,
                                 'target_relative_to_repo': target.relative_to(REPO).as_posix()})
                continue
            assert target.is_file(), link
            links.append({'entrypoint': entry['path'], 'link': link})
    assert len(deferred) == 1
    status = json.loads((PACKAGE / 'STATUS.json').read_bytes())
    expected = {'canonical_features_complete': True, 'both_models_completed1000': True,
                'independent_feature_and_training_QA_PASS': True, 'SOURCE_gate_PASS': False,
                'model_adopted': False, 'current_or_held_new_payload': 0, 'new_PnP': False}
    assert all(status[k] is v if isinstance(v, bool) else status[k] == v for k, v in expected.items())
    replays = {}
    for short, arm in [('CONTROL', 'CONTROL_CANONICAL_LOCAL'), ('TREATMENT', 'TREAT_CANONICAL_CONTEXT')]:
        p = R / ('V240_PUBLICATION_CPU_' + short + '.json')
        d = json.loads(p.read_bytes())
        assert d['completed'] and d['PASS'] and d['arm'] == arm
        assert d['predictions_exact'] == 7617 and d['both_logits_checked'] == 15234
        assert d['canonical_active_bytes_exact'] and d['active_node_max_abs'] == 0
        assert d['atol'] == d['rtol'] == 1e-5
        replays[arm] = {'path': str(p), 'sha256': sha(p.read_bytes()),
                        'predictions_exact': 7617, 'max_abs': d['CPU_GPU_FP32_max_abs']}
    arithmetic = json.loads((R / 'V240_PUBLICATION_ARITHMETIC_REPLAY.json').read_bytes())
    assert arithmetic['completed'] and arithmetic['PASS'] and arithmetic['saved_arithmetic']['rows'] == 7617
    for name in ('V239_SOURCE_CONTEXT_INPUTS_AND_PREPARATION', 'V236_SOURCE_DIAGNOSIS_AND_PAIRED_TRAINING'):
        assert safe(BASE / name / 'README_KO.md').is_file()
    out = {'completed': True, 'PASS': True, 'manifest_sha256': prep['manifest_sha256'],
           'package_integrity': integrity, 'copied_files_verified': len(prep['copied']),
           'entrypoints_suffix_preserved': 3, 'new_prefix_links_verified': links,
           'new_prefix_review_link_checks_deferred': deferred,
           'seven_status_axes': expected, 'relocated_CPU_replays': replays,
           'relocated_stdlib_arithmetic_PASS': True, 'historical_readme_targets_exist': 2,
           'scientific_gate_pass': False, 'model_adopted': False,
           'metadata_pointers_followed': False, 'new_current_or_held_payload': 0,
           'Root_verifier': {'path': str(Path(__file__).absolute()), 'sha256': sha(Path(__file__).read_bytes())},
           'limits': 'Publication integrity and saved small-head replays only; no new encoder/raw data/annotation/physical correspondence audit.'}
    p = R / 'V240_ROOT_PUBLICATION_CHECKS.json'
    with p.open('x') as f:
        json.dump(out, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'completed': True, 'PASS': True, 'receipt_sha256': sha(p.read_bytes())}))


if __name__ == '__main__':
    main()
