"""Reproduce immutable settings into a fresh output directory, never overwrite results."""
import argparse
import io
from pathlib import Path
import shutil
import time
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert not output.exists() or not any(output.iterdir()), 'Use an empty new output directory'
    output.mkdir(parents=True, exist_ok=True)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    from . import common as C
    published_lock = C.WORKTREE / '_docs/experiments/pallet_feature_gradient_joint_20261010/FUSION_METHOD_LOCK.json'
    shutil.copyfile(published_lock, output / 'FUSION_METHOD_LOCK.json')
    C.DOC = output
    C.SOURCE = args.source_root.resolve()
    C.PRIVATE = args.private_dir.resolve()
    from . import preflight, capture
    preflight.DOC = capture.DOC = output
    preflight.run(C.SOURCE, C.PRIVATE, output / 'INPUT_LOCK.json')
    capture.run(C.SOURCE, C.PRIVATE)
    from . import test_fusion
    log = io.StringIO()
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_fusion)
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert result.wasSuccessful(), log.getvalue()
    (C.PRIVATE / 'unit_tests_replay.log').write_text(log.getvalue())
    C.write(output / 'UNIT_TESTS.json', dict(status='PASS', tests=result.testsRun,
        elapsed_seconds=time.monotonic() - started, source_files=C.source_bindings(('fusion.py', 'test_fusion.py')),
        fixture_source='synthetic numerical arrays only; no REAL_DEV or GT',
        cornerSubPix_monkeypatch='PASS', explicit_inverse_monkeypatch='PASS', new_training_updates=0))
    from . import coordinates, evaluate
    coordinates.main()
    evaluate.main()
    print('REPLAY_COMPLETE; aggregate/verify with --doc pointing to the fresh output.', flush=True)


if __name__ == '__main__':
    main()
