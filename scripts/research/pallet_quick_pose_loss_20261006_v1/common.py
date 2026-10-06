"""Read-only inherited contracts and separate quick-experiment destinations."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.dont_write_bytecode = True
from scripts.research.pallet_pose_target_6d_20261006_v1.baseline import BASELINE_ROOT
from scripts.research.pallet_pose_target_6d_20261006_v1.training import (
    initialize, numeric_contract, learning_rate, forward_bank)
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_common import (
    read, write, sha, hash_value, dcp_env)
from scripts.research.pallet_joint_action_handoff_20261006_v1.a_data import Data, network_bank
from scripts.research.pallet_joint_action_handoff_20261006_v1.scorer import action_scores

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_quick_pose_loss_20261006_v1'
HARD_DOC = ROOT / '_docs/experiments/pallet_pose_target_6d_20261006_v1'
OLD_DOC = BASELINE_ROOT / '_docs/experiments/pallet_joint_action_handoff_20261006_v1'
OUTPUT = Path('/tmp/pallet-quick-pose-loss-20261006-cache')
COST = Path('/tmp/pallet-pose-target-6d-cache')
BANKS = Path('/tmp/pallet-joint-action-cache')


class Banks:
    """Immutable loader after setup's one content verification.

    Reusing its recorded digest here is not a new full-content hash check.
    Native NoOp and center equality are asserted for every accessed row.
    """
    def __init__(self, data, directory=BANKS, setup_report=None):
        self.data, self.directory = data, Path(directory)
        self.points = np.load(self.directory / 'source_banks.npy', mmap_mode='r')
        self.counts = np.load(self.directory / 'source_counts.npy', mmap_mode='r')
        self.hypotheses = np.load(self.directory / 'source_hypothesis.npy', mmap_mode='r')
        self.names = read(self.directory / 'BANK_NAMES.json')
        self.binding = read(self.directory / 'BANK_BINDING.json')['binding']
        manifest = read(OLD_DOC / 'A_manifest.json')
        assert self.binding == manifest['bank_binding']
        saved = {v['name']: v for v in manifest['cache_files']}
        for name in ('source_banks.npy', 'source_counts.npy', 'source_hypothesis.npy'):
            assert (self.directory / name).stat().st_size == saved[name]['bytes']
        self.bank_sha256 = saved['source_banks.npy']['sha256']

    def get(self, row, frame=None):
        count = int(self.counts[row])
        assert 1 <= count <= 201
        bank = dict(points=np.array(self.points[row, :count], copy=True),
                    hypotheses=[self.names[int(v)] for v in self.hypotheses[row, :count]])
        frame = self.data.source_frame(row) if frame is None else frame
        assert np.array_equal(bank['points'][0], frame['q'], equal_nan=True)
        assert np.array_equal(bank['points'][:, 8], np.broadcast_to(frame['q'][8], (count, 2)), equal_nan=True)
        return bank

    def tensor_batch(self, rows, batch, device='cuda'):
        maximum = max(2, max(int(self.counts[r]) for r in rows))
        points, valid = [], []
        for i, row in enumerate(rows):
            frame = self.data.source_frame(row)
            bank = self.get(row, frame)
            q = network_bank(bank, frame, batch['points'][i].detach().cpu().numpy())
            padded = np.repeat(q[:1], maximum, axis=0)
            padded[:len(q)] = q
            points.append(padded)
            valid.append(np.arange(maximum) < len(q))
        return torch.from_numpy(np.stack(points)).to(device), torch.from_numpy(np.stack(valid)).to(device)


def code_bindings():
    # Freeze every formal-training dependency before the first update. Separate
    # reporting/evaluation code is recorded before those operations execute.
    names=('__init__.py','common.py','losses.py','setup.py','training.py')
    return {name: sha(Path(__file__).parent/name) for name in names}


def verify_derived(setup):
    """Derived small arrays are freshly hashed; frozen big features are not."""
    for entry in setup.get('artifacts', setup.get('array_artifacts', [])):
        if 'path' in entry and 'sha256' in entry:
            path = Path(entry['path'])
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
