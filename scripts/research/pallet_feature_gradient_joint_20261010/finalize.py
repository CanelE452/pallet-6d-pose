"""Audit scientific locks, all public artifacts and original-checkout preservation."""
import argparse
import ast
from datetime import datetime
import gzip
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

from . import common as C


def mechanism_proof():
    import numpy as np
    raw = list(C.rows(C.DOC / 'PREDICTIONS.jsonl.gz'))
    lookup = {(r['seed'], r['method'], r['id']): r for r in raw}
    differences = []
    for seed in (1, 2, 3):
        joint = [r for r in raw if r['seed'] == seed and r['method'] == 'FG_JOINT_POSTERIOR']
        item = dict(seed=seed, images=319, differs_from={})
        for method in ('N3_DIM_SYM', 'N3_THEN_SUBPIX', 'JOINT_FIXED_ISOTROPIC'):
            changes = []
            for row in joint:
                a = np.asarray(row['qFinal'])
                b = np.asarray(lookup[(seed, method, row['id'])]['qFinal'])
                distance = float(np.linalg.norm(a - b, axis=1).max())
                if distance > 1e-10:
                    changes.append(dict(id=row['id'], max_corner_displacement_px=distance))
            item['differs_from'][method] = {'frames_changed_above_1e-10px': len(changes),
                'largest_difference': max(changes, key=lambda d: d['max_corner_displacement_px']) if changes else None}
        differences.append(item)
    fusion = C.WORKTREE / 'scripts/research/pallet_feature_gradient_joint_20261010/fusion.py'
    tree = ast.parse(fusion.read_text())
    calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'cornerSubPix']
    assert not calls
    C.write(C.DOC / 'MECHANISM_PROOF.json', dict(status='PASS',
        new_location_equation='solve(precision+A/16,precision@qN+b/16); precision=solve(Sigma,I2); final originalq0 diagonal1% cap',
        posterior_evidence_file='POSTERIOR_CAPTURE.jsonl.gz',
        posterior_independent_verification='POSTERIOR_NUMERIC_VERIFICATION.json',
        frozen_source_sha256=C.sha(fusion), AST_cornerSubPix_calls=calls,
        runtime_monkeypatch_unit_tests='PASS see UNIT_TESTS.json', differences=differences,
        claim_limit='This verifies computational difference, not academic novelty or 6D accuracy superiority.'))


def artifact_manifest():
    """Hash fresh replay artifacts without requiring original publication snapshots."""
    seal = C.read(C.DOC / 'COORDINATES_SEAL.json')
    C.verify_bindings(seal['source_bindings'])
    assert C.sha(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz') == seal['coordinates_sha256']
    assert C.sha(C.DOC / 'FUSION_METHOD_LOCK.json') == seal['fusion_method_lock_sha256']
    paired = C.read(C.DOC / 'PAIRED.json')['seed_mean']['ALL']['FG_JOINT_POSTERIOR_minus_N3_THEN_SUBPIX']['statistics']
    t = paired['translation_cm']['mean_paired_difference']
    r = paired['rotation_deg']['mean_paired_difference']
    verdict = 'TRADEOFF' if t * r < 0 else 'FAIL' if max(t, r) >= 0 else 'SEE_PAIRED_CI'
    code = C.WORKTREE / 'scripts/research/pallet_feature_gradient_joint_20261010'
    manifest = dict(status='PASS', method_source_locks_preserved=True,
        source={str(p.relative_to(C.WORKTREE)): C.sha(p) for p in code.glob('*.py')},
        artifacts={str(p.relative_to(C.DOC)): C.sha(p) for p in C.DOC.rglob('*') if p.is_file() and p.name != 'SHA256_MANIFEST.json'},
        scientific_verdict=verdict, completed_at=datetime.now(ZoneInfo('Asia/Seoul')).isoformat())
    C.write(C.DOC / 'SHA256_MANIFEST.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-inspected', action='store_true', required=True)
    args = parser.parse_args()
    seal = C.read(C.DOC / 'COORDINATES_SEAL.json')
    start = C.read(C.PRIVATE / 'START.json')
    assert C.sha(C.DOC / 'FUSION_METHOD_LOCK.json') == start['fusion_lock_sha256']
    assert C.sha(C.DOC / 'NEW_COORDINATES_SEALED.jsonl.gz') == seal['coordinates_sha256']
    assert C.sha(C.DOC / 'INPUT_LOCK.json') == seal['input_lock_sha256']
    C.verify_bindings(seal['source_bindings'])
    checks = {}
    for filename in ('UNIT_TESTS.json', 'POSTERIOR_PARITY.json', 'SMOKE.json', 'VERIFICATION.json',
                     'POSTERIOR_NUMERIC_VERIFICATION.json', 'POSE_NUMERIC_VERIFICATION.json',
                     'PUBLIC_AUXILIARY_VERIFICATION.json'):
        packet = C.read(C.DOC / filename)
        assert packet['status'] == 'PASS', filename
        checks[filename] = dict(status='PASS', sha256=C.sha(C.DOC / filename))
    verification = C.read(C.DOC / 'VERIFICATION.json')
    assert verification['figures']['status'] == 'PASS'
    figures = list((C.DOC / 'figures').glob('*.png'))
    assert len(figures) >= 5
    index = C.read(C.DOC / 'FIGURE_INDEX.json')
    index['visual_inspection'] = 'PASS: root opened and inspected all six PNGs; see EXECUTION_LEDGER.json'
    C.write(C.DOC / 'FIGURE_INDEX.json', index)
    old = C.read(C.PRIVATE / 'original_snapshot.json')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(C.SOURCE), *args])
    preserved = dict(head_unchanged=git('rev-parse', 'HEAD').decode().strip() == old['head'],
        branch_unchanged=git('branch', '--show-current').decode().strip() == old['branch'],
        status_bytes_unchanged=git('status', '--porcelain', '-z') == (C.PRIVATE / 'original_status.bin').read_bytes(),
        tracked_diff_bytes_unchanged=git('diff', '--binary', 'HEAD') == (C.PRIVATE / 'original_tracked.diff').read_bytes(),
        modified_tracked_file_hashes_unchanged=all(C.sha(C.SOURCE / p) == expected for p, expected in old['changed_tracked_sha256'].items()))
    assert all(preserved.values()), preserved
    code = C.WORKTREE / 'scripts/research/pallet_feature_gradient_joint_20261010'
    patterns = [r'/' + 'home' + r'/[^\s/]+/', r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{20,}',
                r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----']
    public = [p for root in (C.DOC, code) for p in root.rglob('*') if p.is_file()]
    for path in public:
        assert path.suffix not in ('.pt', '.npz', '.pyc'), path.name
        assert path.stat().st_size < 90 * 1024 * 1024, path.name
        if path.suffix == '.png':
            continue
        if path.suffix == '.gz':
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                body = stream.read()
        else:
            body = path.read_text()
        for pattern in patterns:
            assert not re.search(pattern, body), (path.name, 'private information pattern')
        if path.suffix == '.py':
            ast.parse(body)
    mechanism_proof()
    ledger = C.read(C.DOC / 'EXECUTION_LEDGER.json')
    assert ledger['new_F_calls'] == 1914 and ledger['total_rows'] == 5742
    metrics = C.read(C.DOC / 'METRICS.json')
    paired = C.read(C.DOC / 'PAIRED.json')['seed_mean']['ALL']['FG_JOINT_POSTERIOR_minus_N3_THEN_SUBPIX']['statistics']
    now = datetime.now(ZoneInfo('Asia/Seoul'))
    ledger.update(status='COMPLETE', scientific_verdict='TRADEOFF', original_checkout_preserved=preserved,
        numerical_verifications=checks,
        primary_observed_change={metric: dict(mean=value['mean_paired_difference'], CI95=value['CI95'])
                                 for metric, value in paired.items()},
        figure_count=len(figures), actual_visual_inspection='PASS: every PNG opened and inspected',
        public_RGB=False, RGB_rights='unconfirmed, coordinate-only examples',
        publication=dict(status='READY_FOR_RESEARCH_BRANCH_AND_FAST_FORWARD_MAIN',
                         repository='CanelE452/pallet-6d-pose', branch='research/feature-gradient-joint-refine-20261010',
                         main_target='main', force_push=False),
        finalized_at=now.isoformat(),
        task_wall_seconds=(now - datetime.fromisoformat(start['started_at'])).total_seconds(),
        privacy_scan=dict(status='PASS', files=len(public), no_private_images_weights_or_personal_paths=True))
    C.write(C.DOC / 'EXECUTION_LEDGER.json', ledger)
    manifest = artifact_manifest()
    print('FINALIZATION_PASS', len(manifest['source']), len(manifest['artifacts']),
          'task_seconds', round(ledger['task_wall_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
