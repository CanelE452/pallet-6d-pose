"""Resolve the immutable a22 experiment without copying it into main.

The publication contains only the new experiment. Its unchanged scorer, pose
function and old evidence remain an explicit read-only baseline dependency.
"""
from pathlib import Path
import hashlib
import os
import subprocess

ROOT = Path(__file__).resolve().parents[3]
BASE = 'a22fb14beb5e8df08076385000e0d53503c1ae29'
OLD_CODE = 'scripts/research/pallet_joint_action_handoff_20261006_v1'
OLD_DOC = '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
_REQUIRED_CODE = ('a_common.py', 'a_data.py', 'a_experiment.py', 'a_evaluate.py',
                  'geometry.py', 'scorer.py')
_REQUIRED_EVIDENCE = ('A_protocol.json', 'A_manifest.json', 'results/A_ID_MANIFEST.json')


def resolve():
    specified = os.environ.get('PALLET_BASELINE_ROOT')
    candidates = [Path(specified)] if specified else [ROOT.parent / 'pallet-pose-handoff-20261006', ROOT]
    for candidate in candidates:
        if not all((candidate / OLD_CODE / n).is_file() for n in _REQUIRED_CODE):
            continue
        for relative in [OLD_CODE + '/' + n for n in _REQUIRED_CODE] + [OLD_DOC + '/' + n for n in _REQUIRED_EVIDENCE]:
            expected = subprocess.check_output(['git', '-C', str(candidate), 'show', BASE + ':' + relative])
            actual = (candidate / relative).read_bytes()
            if hashlib.sha256(actual).digest() != hashlib.sha256(expected).digest():
                raise ValueError('BLOCKED_INTEGRITY: baseline differs from a22: ' + relative)
        import scripts.research
        location = str(candidate / 'scripts/research')
        scripts.research.__path__ = [location] + [p for p in scripts.research.__path__ if p != location]
        return candidate.resolve()
    raise FileNotFoundError('BLOCKED_DATA: unchanged a22 worktree is required; set PALLET_BASELINE_ROOT. Keep all bank/checkpoint/data dependencies read-only.')


BASELINE_ROOT = resolve()
