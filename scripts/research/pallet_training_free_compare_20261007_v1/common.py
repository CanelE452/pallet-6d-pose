"""Read-only adapters and bounded, compressed outputs for the fixed comparison."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_training_free_compare_20261007_v1'
OUTPUT = Path('/dev/shm/pallet-training-free-20261007')
ARMS = ('SUBPIX_NATIVE', 'SUBPIX_CAP1', 'CVRANK_NATIVE', 'CVRANK_CAP1')
BASE_SHA = '7b98efa62f3a2cb6488fd7905c25a2a28471b6c5'


def read(path):
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(finite(value), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def legacy():
    # Resolver verifies unchanged a22 source. No training/evaluation entrypoint is run.
    from scripts.research.pallet_pose_target_6d_20261006_v1 import evaluation as E
    from scripts.research.pallet_pose_target_6d_20261006_v1.baseline import BASELINE_ROOT
    return E, BASELINE_ROOT


def load_real():
    E, baseline_root = legacy()
    # real_frames needs only root/dim: do not initialize the synthetic training bank.
    data = object.__new__(E.Data)
    data.root = ROOT
    data.dim = ROOT / 'data/pallet/results/pallet_dim_conditioned_p_v1'
    frames = data.real_frames()
    target_path = ROOT / 'data/pallet/results/pallet_posefix_replay_diagnosis_v1/TARGETS.json'
    targets = read(target_path)
    baseline_path = baseline_root / '_docs/experiments/pallet_joint_action_handoff_20261006_v1/results/A_REAL_DEV_BASELINES.json'
    baselines = read(baseline_path)
    assert baselines['target_sha256'] == sha(target_path)
    ids = [f['id'] for f in frames]
    assert len(ids) == len(set(ids)) == 319
    assert len({f['session'] for f in frames}) == 13
    for name, rows in baselines['rows'].items():
        assert len(rows) == 319 and {r['id'] for r in rows} == set(ids), name
    assert sum(sum(targets[i]['valid'][:8]) for i in ids) == 2499
    assert sum(targets[i]['matched'] for i in ids) == 311
    assert sum(len(r['corner']['observed_errors']) for r in baselines['rows']['RAW']) == 2445
    return E, frames, targets, baselines, baseline_path


def iter_rows(path=None):
    with gzip.open(path or DOC / 'PREDICTIONS.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)
