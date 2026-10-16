"""Public metadata, immutable inputs and exclusive new-output helpers.

Importing this module does not import a detector, Torch, OpenCV or pose solver.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
OLD = REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1'
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
PRIVATE = Path('/dev/shm/pallet-boundary-corner-private-20261010-v2')
SOURCE = os.environ.get('PALLET_SOURCE_ROOT')
BASELINE = os.environ.get('PALLET_BASELINE_ROOT')
FITS = Path('/dev/shm/pallet-kp-supervision-repair-private-20261010/learned_fits')
METHODS = ('VALIDATED_ROLE_ONLY', 'N3_VALIDATED_ROLE', 'N3_VALIDATED_ROLE_NO_MASK',
           'N3_BASIN_ROBUST', 'N3_BASIN_STANDARD')
PRIMARY = 'N3_VALIDATED_ROLE'
NATIVE_REPLACEMENT_CAP_PX = 8.0
CORNER_OBSERVATION_UNCERTAINTY_CAP_PX = 8.0
BASE_WEIGHT_REL = 'challenge/yolo_pose_one_model/spatial_concat_scratch/runs/YOLO26N_G38_P0_TEX20K_CLEANSTART_60EP_SEED42/weights/best.pt'
N3_WEIGHT_REL = 'data/pallet/results/pallet_dim_conditioned_p_v1/runs/N3_DIM_SYM_seed1/last.pt'
BASE_WEIGHT_SHA = '970a0913b38ed4c9e3662837abccbf9d91b8b0858deafae854c1055e477644f7'
N3_WEIGHT_SHA = 'ceea743b7b8467cf43eef66582b923dd980e5b2fb27f5575af5df185109f22dd'
ROLE_WEIGHT_SHA = '882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d'


def require(condition, message):
    if not condition:
        raise RuntimeError('BOUNDARY_REFINER_GUARD: ' + message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    public = path.is_relative_to(REPO)
    return dict(path=str(path.relative_to(REPO)) if public else path.name,
                origin='public_repository' if public else 'external_readonly_dependency',
                sha256=sha(path), bytes=path.stat().st_size)


def bound(path, expected, label):
    require(Path(path).is_file(), label + ' missing')
    require(sha(path) == expected['sha256'] and Path(path).stat().st_size == expected['bytes'],
            label + ' SHA/bytes changed')


def finite(value):
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def primitive_counter():
    """Count actual OpenCV pose entries; does not change numerical arguments."""
    from collections import Counter
    import cv2
    counts = Counter()
    names = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'cornerSubPix')
    original = {k: getattr(cv2, k) for k in names}
    for key, fn in original.items():
        def wrapped(*a, _key=key, _fn=fn, **kw):
            counts[_key] += 1
            return _fn(*a, **kw)
        setattr(cv2, key, wrapped)
    try:
        yield counts
    finally:
        for key, fn in original.items():
            setattr(cv2, key, fn)


def write_new(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'preserve existing ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def save_rows(path, values):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'preserve existing ' + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as zipped:
            for value in values:
                zipped.write((json.dumps(finite(value), ensure_ascii=False, separators=(',', ':'),
                                         allow_nan=False) + '\n').encode())
        raw.flush()
        os.fsync(raw.fileno())


def output_path(args, name, allow_existing=False):
    require(str(name)==Path(name).name and str(name) not in ('','.', '..'), 'output name must be one basename')
    require(args.source_root and args.baseline_root, 'supply --source-root/--baseline-root or their PALLET environment variables')
    unexpanded = Path(args.output)
    require(not unexpanded.is_symlink(), 'output directory may not be a symlink')
    output = unexpanded.resolve()
    roots = [Path(args.source_root).resolve(), Path(args.baseline_root).resolve(), Path(args.fits).resolve(),
             Path('/dev/shm/pallet-observation-private-20261009'),
             Path('/dev/shm/pallet-kp-supervision-gate-private-20261010'),
             Path('/dev/shm/pallet-kp-difficulty-private-20261010'),
             Path('/dev/shm/pallet-kp-supervision-repair-private-20261010'),
             Path('/dev/shm/pallet-kp-corrected-supervision-private-20261010')]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots),
            'output overlaps original data/baseline/weights/previous experiment')
    require(not output.is_relative_to(REPO) or output == DOC.resolve(), 'repository output must be new DOC')
    path = output / name
    require(not path.is_symlink(), 'output file may not be a symlink')
    require(allow_existing or not path.exists(), 'preserve completed/interrupted ' + name)
    return path


def prior_snapshot(path):
    protected = read(path)
    entries = protected['files']
    require(len(entries) == len({b['path'] for b in entries}), 'duplicate protected file')
    for b in entries:
        path = (REPO / b['path']).resolve()
        require(path.is_relative_to(REPO) and not path.is_relative_to(DOC.resolve()) and
                not path.is_relative_to(CODE.resolve()), 'prior protection includes current outputs or escapes repo')
        bound(path, b, 'prior ' + b['path'])
    return entries


def cohort_frames(args):
    cohort = read(args.cohort)
    authority = {f['id']: f for f in read(OLD / 'INPUTS.json')['frames']}
    require(len(authority) == 319 and len(cohort['ids']) == len(set(cohort['ids'])) == 245,
            'frozen 319/selected245 population differs')
    require(cohort['counts'] == dict(clean=153, moderate=92, severe_excluded=74, original=319),
            'frozen difficulty scope differs')
    require(set(cohort['ids']).isdisjoint(cohort['excluded_ids']) and
            set(cohort['ids']) | set(cohort['excluded_ids']) == set(authority), 'cohort partition differs')
    session_metadata = {}
    for r in read(CORRECTED / 'RUNTIME_PANEL.json')['frames']:
        previous = session_metadata.setdefault(r['session_id'], r)
        require(previous['object_type'] == r['object_type'] and
                previous['dimensions_wdh_m'] == r['dimensions_wdh_m'], 'registry session metadata differs')
    selected = []
    for label in cohort['frames']:
        f = authority[label['id']]
        require(label['label'] in ('clean', 'moderate') and label['session'] == f['session'] and
                label['image']['sha256'] == f['image_sha256'], 'cohort image/label identity differs')
        dims = [f['xyz'][0], f['xyz'][2], f['xyz'][1]]
        require(dims == session_metadata[f['session']]['dimensions_wdh_m'], 'WDH/WH D registry mapping differs')
        selected.append(dict(f, label=label['label'], object_type=session_metadata[f['session']]['object_type'],
            frame_id=f['id'], session_id=f['session'], image_key=f['image'], original_hw=f['raw_hw'],
            camera_intrinsics=f['K'], dimensions_wdh_m=dims))
    require([f['id'] for f in selected] == cohort['ids'], 'cohort order differs')
    return selected


def protocol_inputs(args):
    require(args.source_root and args.baseline_root, 'explicit source/baseline dependencies required')
    paths = {'cohort': args.cohort, 'calibration': args.calibration,
             'prior_bindings': args.prior_bindings, 'goal': DOC / 'GOAL.md',
             'statistics_protocol': DOC / 'STATISTICS_PROTOCOL.json',
             'visual_case_protocol': DOC / 'VISUAL_CASE_PROTOCOL.json',
             'original_inputs': OLD / 'INPUTS.json', 'runtime_panel': CORRECTED / 'RUNTIME_PANEL.json',
             'reference_id_mapping': CORRECTED / 'REFERENCE_ID_MAPPING.json',
             'original_pose_diagnostics': OLD / 'POSE_DIAGNOSTICS.jsonl.gz',
             'original_fixed_controls': OLD / 'FIXED_CONTROLS.jsonl.gz',
             'training_completion': Path(args.fits) / 'TRAINING_COMPLETION.json',
             'corrected_role_last': Path(args.fits) / 'IMAGE_ROLE.pt',
             'base_weights': args.base_weights or Path(args.source_root) / BASE_WEIGHT_REL,
             'N3_weights': args.N3_weights or Path(args.source_root) / N3_WEIGHT_REL,
             'N3_configuration': Path(args.source_root) / '_docs/experiments/pallet_dim_conditioned_p_v1/CALIBRATION_AND_SELECTION.json',
             'N3_normalization': Path(args.source_root) / '_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json',
             'original_N3_runtime': Path(args.source_root) / 'scripts/research/pallet_n3_subpix_20261008_v1/runtime.py',
             'original_model': REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
             'original_context': REPO / 'scripts/research/pallet_kp_repair_runtime_20261010_v1/adapter.py',
             'original_metric': REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/evaluate.py'}
    paths.update({name: CODE / name for name in ('common.py', 'pipeline.py', 'evaluate.py', 'runtime.py',
                                               'observations.py', 'pose.py')})
    return {name: binding(path) for name, path in paths.items()}


def verify_protocol(args):
    protocol = read(args.protocol)
    require(protocol['fixed_inputs'] == protocol_inputs(args), 'frozen protocol code/input drift')
    require(protocol['methods'] == list(METHODS) and protocol['frames'] == 245 and
            protocol['primary'] == PRIMARY and protocol['native_replacement_cap_px'] == 8.0 and
            protocol['corner_observation_uncertainty_cap_px'] == 8.0,
            'method/population/policy differs')
    require(protocol['GT_tuning'] is False and protocol['new_training_updates'] == 0,
            'unauthorized tuning/training')
    completion = read(Path(args.fits) / 'TRAINING_COMPLETION.json')
    require(completion['complete'] and completion['formal_updates'] == completion['total_updates'] == 9000 and
            completion['seed'] == 1 and completion['batch'] == 16 and completion['throwaway_updates'] == 0 and
            completion['same_initial_tensor_sha'] and completion['same_batch_order'] and
            completion['same_update_budget'] and completion['input_hashes_unchanged'], 'corrected training completion invalid')
    role = next(c for c in completion['checkpoints'] if c['arm'] == 'IMAGE_ROLE')
    bound(Path(args.fits) / 'IMAGE_ROLE.pt', role['checkpoint'], 'completed ROLE checkpoint')
    require(sha(Path(args.fits)/'IMAGE_ROLE.pt')==ROLE_WEIGHT_SHA,'fixed corrected last ROLE identity differs')
    return protocol


@contextmanager
def legacy_context(args):
    require(args.source_root and args.baseline_root, 'source-root/baseline-root are required deployment dependencies')
    from scripts.research.pallet_kp_repair_runtime_20261010_v1 import adapter
    with adapter.original_context(args, Path(args.output).resolve(), Path(args.output).resolve()) as (C, L):
        from scripts.research.pallet_n3_subpix_20261008_v1 import runtime as R
        require(C.ROOT.resolve() == Path(args.source_root).resolve(), 'original source resolver differs')
        yield C, L, R


def parser(description, stages=None):
    p = argparse.ArgumentParser(description=description)
    if stages:
        p.add_argument('stage', choices=stages)
    p.add_argument('--output', default=str(PRIVATE / 'accuracy'))
    p.add_argument('--source-root', default=SOURCE)
    p.add_argument('--baseline-root', default=BASELINE)
    p.add_argument('--fits', default=str(FITS))
    p.add_argument('--base-weights', help='optional byte-identical fixed Base checkpoint copy')
    p.add_argument('--N3-weights', help='optional byte-identical fixed N3 checkpoint copy')
    p.add_argument('--cohort', default=str(CORRECTED / 'COHORT.json'))
    p.add_argument('--calibration', default=str(DOC / 'CALIBRATION.json'))
    p.add_argument('--protocol', default=str(DOC / 'PROTOCOL.json'))
    p.add_argument('--prior-bindings', default=str(DOC / 'PRIOR_MATERIALIZED_BINDINGS.json'))
    return p
