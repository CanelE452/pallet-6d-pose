"""Validate the partial preflight handoff, never label it a completed experiment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import unittest

from preflight import ROOT, HERE, DOC, R0, R0_SHA, read, sha, write


def main():
    binding = read(DOC / 'SOURCE_BINDING.json')
    assert sha(R0) == R0_SHA
    for path, expected in {**binding['originals'], **binding['new_code']}.items():
        assert sha(ROOT / path) == expected, path
    data = read(DOC / 'DATA_AUDIT.json')
    retention = read(DOC / 'SOURCE_RETENTION_SPLIT.json')
    paths = {}
    for row in [*data['target_bindings'], *data['synthetic_replay'], *retention['records']]:
        for field in ('image', 'label'):
            p, h = row[field], row[field + '_sha256']
            if p in paths:
                assert paths[p] == h
            paths[p] = h
    def verify(pair):
        path, expected = pair
        assert sha(ROOT / path) == expected, path
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, paths.items()))
    tests = unittest.defaultTestLoader.discover(str(HERE), pattern='test_contracts.py')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful()
    train = read(DOC / 'TRAINING_AUDIT.json')
    assert train['completed_fits'] == train['actual_main_optimizer_updates'] == train['actual_smoke_optimizer_updates'] == 0
    assert len(train['per_fit_actual_updates']) == 12 and set(train['per_fit_actual_updates'].values()) == {0}
    assert read(DOC / 'VERDICT.json')['scientific_verdict'] is None
    assert not read(DOC / 'CURRENT_RESULT.json')['scientific_complete']
    raw = ROOT / 'data/pallet/results/pallet_transfer_replay_control_v1'
    assert not list(raw.rglob('*.pt')), 'Unexpected training checkpoint'
    # Check tracked modifications, both staged and unstaged, for scope leaks.
    changed = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).splitlines()
    prefixes = ('scripts/research/pallet_transfer_replay_control_v1/', '_docs/experiments/pallet_transfer_replay_control_v1/')
    assert all(path.startswith(prefixes) for path in changed), changed
    write(DOC / 'FINAL_AUDIT.json', dict(
        execution='TECHNICAL_STOP_BEFORE_TRAINING', integrity='PARTIAL_PREFLIGHT_HANDOFF_PASS',
        scientific_complete=False, scientific_verdict=None, training_gate_passed=False,
        completed_fits=0, actual_main_updates=0, actual_smoke_updates=0,
        actual_evaluation_model_forwards=0, CPU_helper_tests=result.testsRun,
        preserved_original_bindings=len(binding['originals']),
        rehashed_data_image_label_paths=len(paths), source_label_and_image_preservation=True,
        paper_final_bound_files_unchanged=True, no_tracked_modifications_outside_new_scope=True,
        external_GPU_checks=3, driver_or_system_mutation=False,
        pending='Loader, actual-model gates, resumable trainer, full panel evaluator/statistics, all12 fits and evaluation',
        git_publication='Verified separately after commit/push; not a performance gate',
        unrelated_user_untracked=['_docs/experiments/pallet_capacity_screen_v1/',
            'data/pallet/results/pallet_capacity_screen_v1/', 'scripts/research/pallet_capacity_screen_v1/']))
    paths = sorted(p for directory in (DOC, HERE) for p in directory.rglob('*')
                   if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json' and '__pycache__' not in p.parts)
    write(DOC / 'ARTIFACT_MANIFEST.json', dict(scope='Partial preflight, not trained experiment',
        files={str(p.relative_to(ROOT)): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}))
    print('PARTIAL_HANDOFF_AUDIT_PASS; EXPERIMENT_NOT_COMPLETE; ALL12_FITS_NOT_STARTED', flush=True)


if __name__ == '__main__':
    main()
