"""Explicit read-only inputs and compact, versioned experiment outputs."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / '_docs/experiments/pallet_visibility_curriculum_20261013_v1'
OLD = Path('/dev/shm/pallet-observation-worktree-20261009/_docs/experiments/pallet_observation_refiner_20261009_v1')
SCRATCH = Path('/dev/shm/pallet-visibility-curriculum-20261013-v1')


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read(path):
    path = Path(path)
    with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def save_rows(path, values):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve completed experiment rows: ' + str(path))
    with gzip.open(path, 'wt', compresslevel=5) as stream:
        for value in values:
            stream.write(json.dumps(finite(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def cuboid(width, height, depth):
    a, b, c = np.asarray([width, height, depth], float) / 2
    return np.array([[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
                     [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]])


def project(xyz, R, t, K):
    camera = np.asarray(xyz) @ np.asarray(R).T + np.asarray(t).reshape(3)
    homogeneous = camera @ np.asarray(K).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def self_hidden(xyz, R, t):
    rays = (-np.asarray(R).T @ np.asarray(t).reshape(3))[None] - xyz
    cosines = np.sign(xyz) * rays / np.linalg.norm(rays, axis=1)[:, None]
    return cosines.max(1) < -np.sin(np.deg2rad(2))


def prediction_inputs():
    """Extract only prediction fields; annotations are opened by scoring only."""
    isolated = DOC / 'PREDICTION_INPUTS.json'
    if isolated.exists():
        return read(isolated)['frames']
    frames = read(OLD / 'INPUTS.json')['frames']
    controls = {}
    for row in rows(OLD / 'FIXED_CONTROLS.jsonl.gz'):
        if row['method'] in ('BASE', 'N3_SUBPIX'):
            controls[(row['id'], row['method'])] = row.get('actual_pose')
    if any(value is None for value in controls.values()):
        for row in rows(OLD / 'OBSERVATIONS.jsonl.gz'):
            arm = 'BASE' if row['method'].startswith('BASE_') else 'N3_SUBPIX'
            if row['method'] == arm + '_NO_MASK_STANDARD':
                controls[(row['id'], arm)] = row['initial_pose']
    for frame in frames:
        frame['initial_poses'] = {arm: controls[(frame['id'], arm)] for arm in ('BASE', 'N3_SUBPIX')}
    return frames
