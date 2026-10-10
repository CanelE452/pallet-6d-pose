"""Two fixed source-CAL128 head passes and byte-exact reuse of ROLE calibration.

freeze binds the inputs and the unchanged v2 calibration algorithm before any
model is loaded. run claims one execution before Torch import. Each added arm
uses its own eight batches and its own threshold/uncertainty calibration. ROLE
artifacts retain their original bytes, including the historical eight-forward
execution receipt; REUSE_RECEIPT and COMPLETION record zero new ROLE forwards.

The only adaptation of v2.calibrate's result is explicit arm/checkpoint/new
protocol provenance. Its numerical body and global output directory are never
modified. Its original protocol hash is retained as source_algorithm_protocol.
Source CAL precision and propagated uncertainty remain calibration diagnostics,
not independent transfer, physical corner ownership, or pose accuracy proofs.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
DOC = REPO / '_docs/experiments/pallet_three_head_observation_20261010_v7'
PRIVATE = Path('/tmp/pallet-three-head-observation-private-20261010-v7')
V2_CODE = REPO / 'scripts/research/pallet_boundary_corner_refiner_20261010_v2'
V2_DOC = REPO / '_docs/experiments/pallet_boundary_corner_refiner_20261010_v2'
REPAIR = REPO / '_docs/experiments/pallet_kp_supervision_repair_20261010_v1'
CORRECTED = REPO / '_docs/experiments/pallet_kp_corrected_supervision_20261010_v1'
ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
NEW_ARMS = ARMS[:2]
INDICES = tuple(range(768, 896))
ROLE_FILES = ('CALIBRATION_PROTOCOL.json', 'CALIBRATION_START.json',
              'CALIBRATION_ROWS.jsonl.gz', 'CALIBRATION_THRESHOLD_SCAN.jsonl.gz',
              'CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz',
              'CALIBRATION_GEOMETRY_ROWS.jsonl.gz', 'CALIBRATION.json',
              'CALIBRATION_EXECUTION.json', 'CALIBRATION_CHECKS.json')
ARM_OUTPUTS = ROLE_FILES[:-1]
MODES = dict(GEOMETRY_ONLY='zero channels0:19; retain geometry19:25 and role25:28',
             IMAGE_NO_ROLE='zero role25:28; retain image/neck0:19 and geometry19:25',
             IMAGE_ROLE='unchanged all28 channels')
CHECKPOINT_SHAS = dict(
    GEOMETRY_ONLY='d188dcc68bd8795c88232d5bf1b85259d684695723b96809017abd47d6ac009b',
    IMAGE_NO_ROLE='9c52a2e2036ee8f65ba2e191bdcbebe93ac3ecde60a6aa31e71a4ffbadc0f835',
    IMAGE_ROLE='882a6eddc964258abfbc1ad3baeee380ab9fac42b3b90c7f64166bd98d777d5d')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else path.name,
                origin='public_repository' if path.is_relative_to(REPO) else 'external_readonly_dependency',
                sha256=sha(path), bytes=path.stat().st_size)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_new(path, rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n').encode())


def reject_symlinks(path):
    path = Path(path).absolute()
    require(all(not p.is_symlink() for p in (path, *path.parents)), 'symlink output ancestor')


def output_root(args):
    candidate = Path(args.output).absolute()
    reject_symlinks(candidate)
    root = candidate.resolve()
    require(root == (DOC / 'source_calibration').resolve() or root.is_relative_to(PRIVATE.resolve()),
            'calibration output must be new source_calibration or new private subtree')
    protected = (Path(args.source_root).resolve(), Path(args.fits).resolve(),
                 Path(args.features).resolve().parent, V2_DOC.resolve(), REPAIR.resolve(), CORRECTED.resolve())
    require(all(not root.is_relative_to(p) and not p.is_relative_to(root) for p in protected),
            'calibration output overlaps immutable inputs')
    return root


def validate_metadata(meta, completion):
    require(completion['complete'] and completion['formal_updates'] == completion['total_updates'] == 9000
            and completion['throwaway_updates'] == 0 and completion['seed'] == 1 and completion['batch'] == 16,
            'completed fixed9000 training required')
    rows = meta['rows']
    require(len(rows) == 3 and {r['arm'] for r in rows} == set(ARMS), 'exact three original arms required')
    by_arm = {r['arm']: r for r in rows}
    done = {r['arm']: r for r in completion['checkpoints']}
    require(set(done) == set(ARMS), 'completion checkpoint arms differ')
    for arm, row in by_arm.items():
        require(row['steps'] == done[arm]['updates'] == 3000 and row['parameters'] == 5890,
                'last step3000 tiny head required: ' + arm)
        require(all(row['checkpoint'][k] == done[arm]['checkpoint'][k] for k in ('sha256', 'bytes')),
                'metadata/completion checkpoint mismatch: ' + arm)
        require(row['checkpoint']['sha256'] == CHECKPOINT_SHAS[arm], 'fixed corrected last weights differ')
    for key in ('config', 'initial_state_sha256', 'batch_order_sha256',
                'protocol_sha256', 'repaired_target_sha256'):
        require(all(r[key] == rows[0][key] for r in rows), 'unequal original training ' + key)
    return by_arm


def validate_source(source):
    require([r['index'] for r in source] == list(INDICES), 'exact source CAL indices768:896 required')
    require(all(r['partition'] == 'calibration' and len(r['queries']) == 84 for r in source),
            'source CAL partition/query count differs')
    require(len({r['family'] for r in source}) == 128 and len({r['id'] for r in source}) == 128,
            'CAL families/IDs must be distinct')


def source_rows(path):
    # Non-CAL records are skipped without decoding their labels or geometry.
    selected = []
    count = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            count += 1
            if 768 <= index < 896:
                selected.append(json.loads(line))
    require(count == 1024, 'source row population must remain1024')
    validate_source(selected)
    return selected


def validate_cache_sources(source, manifest, split):
    require(manifest['complete'] and len(manifest['records']) == 1024
            and manifest['specs']['features'] == dict(shape=[1024, 84, 28, 65], dtype='float16'),
            'source feature manifest differs')
    require(split['counts'] == dict(train=768, calibration=128, source_test=128), 'source family split differs')
    for row in source:
        cached, family = manifest['records'][row['index']], split['records'][row['index']]
        require(cached['index'] == row['index'] and cached['id'] == row['id'] == family['id']
                and cached['family'] == row['family'] == family['family']
                and cached['partition'] == row['partition'] == family['partition'] == 'calibration',
                'CAL feature/source/family identity differs')
        require(cached['selected_points'] == row['frozen_selected_points']
                and cached['variant'] == 0 and cached['composition'] == dict(kind='existing_P0_original', generated=False),
                'CAL query coordinates/original RGB provenance differs')


def inputs(args):
    paths = dict(features=Path(args.features), ready_targets=REPAIR / 'READY_PREPARED_TARGETS.npz',
                 cache_manifest=Path(args.cache_manifest) if args.cache_manifest else Path(args.features).with_name('CACHE_MANIFEST.json'),
                 source_family_split=REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1/SOURCE_FAMILY_SPLIT.json',
                 ready_rows=REPAIR / 'READY_SOURCE_TARGET_ROWS.jsonl.gz',
                 checkpoint_metadata=CORRECTED / 'CHECKPOINT_METADATA.json',
                 training_completion=Path(args.fits) / 'TRAINING_COMPLETION.json',
                 original_model=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/model.py',
                 original_solver_import=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/solver.py',
                 original_source_import=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/source_audit.py',
                 original_common_import=REPO / 'scripts/research/pallet_observation_refiner_20261009_v1/common.py',
                 algorithm=V2_CODE / 'calibration.py', observations=V2_CODE / 'observations.py',
                 source_algorithm_protocol=V2_DOC / 'CALIBRATION_PROTOCOL.json',
                 quiet_guard=REPO / 'scripts/research/pallet_corner_mechanism_audit_20261010_v1/deployment_smoke.py',
                 calibration=Path(__file__), tests=CODE / 'test_calibration.py', common=CODE / 'common.py',
                 protection=Path(args.prior_bindings))
    paths.update({'checkpoint:' + a: Path(args.fits) / (a + '.pt') for a in ARMS})
    paths.update({'ROLE:' + name: V2_DOC / name for name in ROLE_FILES})
    return {key: binding(path) for key, path in paths.items()}


def protect(args):
    from . import common as C
    return C.protect(args)


def checkpoint_identity(args, locked_inputs):
    meta = validate_metadata(read(CORRECTED / 'CHECKPOINT_METADATA.json'),
                             read(Path(args.fits) / 'TRAINING_COMPLETION.json'))
    for arm, row in meta.items():
        actual = locked_inputs['checkpoint:' + arm]
        require(all(actual[k] == row['checkpoint'][k] for k in ('sha256', 'bytes')),
                'corrected checkpoint bytes differ: ' + arm)
    old = read(V2_DOC / 'CALIBRATION_PROTOCOL.json')['inputs']
    for name in ('features', 'ready_targets', 'ready_rows', 'checkpoint_metadata', 'original_model', 'observations'):
        require(all(locked_inputs[name][k] == old[name][k] for k in ('sha256', 'bytes')),
                'source algorithm dependency differs: ' + name)
    require(all(locked_inputs['checkpoint:IMAGE_ROLE'][k] == old['checkpoint'][k] for k in ('sha256', 'bytes')),
            'reused ROLE calibration weights differ')
    require(meta['IMAGE_ROLE']['repaired_target_sha256'] == locked_inputs['ready_targets']['sha256'],
            'repaired supervision provenance differs')
    return meta


def destinations(args, frozen=False):
    root = output_root(args)
    for name in ('STARTED.json', 'COMPLETION.json'):
        require(not (root / name).exists(), 'preserve execution ' + name)
    require(frozen or not (root / 'PROTOCOL.json').exists(), 'preserve source calibration protocol')
    for arm in ARMS:
        reject_symlinks(root / arm)
        require(not (root / arm).exists(), 'preserve arm directory: ' + arm)
    return root


def freeze(args):
    root = destinations(args)
    locked = inputs(args)
    meta = checkpoint_identity(args, locked)
    preservation = protect(args)
    old_protocol = read(V2_DOC / 'CALIBRATION_PROTOCOL.json')
    policy = {k: v for k, v in old_protocol.items() if k not in ('schema', 'inputs', 'arm')}
    require(policy['indices'] == list(INDICES) and policy['head_forward_calls'] == 8
            and policy['batch_size'] == 16, 'frozen CAL128 algorithm policy differs')
    root.mkdir(parents=True, exist_ok=True)
    value = dict(schema='three_fixed_head_source_calibration_protocol_v7', created_before_new_head_calls=True,
                 inputs=locked, source_algorithm_protocol=locked['source_algorithm_protocol'],
                 source_algorithm_policy=policy, indices=list(INDICES), new_arms=list(NEW_ARMS),
                 reused_arm='IMAGE_ROLE', head_modes=MODES, checkpoint_metadata=meta,
                 new_head_forward_calls_by_arm=dict(GEOMETRY_ONLY=8, IMAGE_NO_ROLE=8, IMAGE_ROLE=0),
                 new_head_image_exposures_by_arm=dict(GEOMETRY_ONLY=128, IMAGE_NO_ROLE=128, IMAGE_ROLE=0),
                 torch_threads=1, device='cuda', head_dtype='float32', source_cache_dtype='float16',
                 backend_flags='keep process defaults, record actual flags; historical v2 flags not recorded',
                 numerical_algorithm='call unmodified v2.calibrate with original v2 protocol; no global monkeypatch',
                 output_adaptation='retain old algorithm protocolSHA; add own arm/protocol/checkpoint provenance only',
                 ROLE_reuse='copy all9 files byte-exact; historical receipt8 calls remain; REUSE_RECEIPT newcalls0',
                 source_test_exposures=0, real_frames=0, new_training_updates=0, new_RGB=0,
                 detector_calls=0, N3_calls=0, PnP_calls=0, rays=0, feature_recomputations=0,
                 protection_before=preservation)
    write_new(root / 'PROTOCOL.json', value)
    print('SOURCE_CALIBRATION_FROZEN', sha(root / 'PROTOCOL.json'), flush=True)


def annotate_calibration(value, arm, own_protocol, source_protocol, checkpoint):
    require(arm in NEW_ARMS, 'ROLE calibration must retain original bytes')
    require(value['protocol_sha256'] == source_protocol['sha256'], 'original algorithm protocol hash differs')
    result = dict(value)
    result.update(protocol_sha256=own_protocol['sha256'], arm=arm, training_arm=arm,
                  checkpoint=checkpoint, source_algorithm_protocol=source_protocol,
                  source_algorithm_protocol_sha256=value['protocol_sha256'],
                  provenance_only_adaptation=True)
    return result


def check_binding(path, expected, name):
    actual = binding(path)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'binding differs: ' + name)


def validate_deployment_inputs(args, head_arms):
    """Stdlib-only head/calibration provenance check before any model import."""
    require(len(head_arms) == len(set(head_arms)) and set(head_arms) <= set(ARMS), 'distinct original head arms required')
    root = Path(args.calibration_root).resolve()
    protocol = read(root / 'PROTOCOL.json')
    completion = read(root / 'COMPLETION.json')
    require(protocol['schema'] == 'three_fixed_head_source_calibration_protocol_v7'
            and protocol['indices'] == list(INDICES) and protocol['head_modes'] == MODES
            and protocol['new_arms'] == list(NEW_ARMS) and protocol['reused_arm'] == 'IMAGE_ROLE',
            'source-only fixed calibration plan differs')
    require(completion['complete'] and completion['inputs_before_after_equal']
            and set(completion['arms']) == set(ARMS)
            and completion['actual_new_head_forward_calls'] == 16
            and completion['actual_new_head_image_exposures'] == 256
            and completion['ROLE_new_head_forward_calls'] == 0, 'completed source-only calibration required')
    check_binding(root / 'PROTOCOL.json', completion['protocol'], 'source calibration protocol')
    require(completion['input_bindings'] == protocol['inputs'], 'source calibration frozen input join differs')
    for key in ('source_test_exposures', 'real_frames', 'new_training_updates', 'detector_calls',
                'N3_calls', 'PnP_calls', 'rays', 'new_RGB', 'feature_recomputations'):
        require(protocol[key] == completion[key] == 0, 'source calibration scope violation: ' + key)
    meta = validate_metadata(read(CORRECTED / 'CHECKPOINT_METADATA.json'),
                             read(Path(args.fits) / 'TRAINING_COMPLETION.json'))
    for arm in ARMS:
        info = completion['arms'][arm]
        check_binding(Path(args.fits) / (arm + '.pt'), info['checkpoint'], 'fixed weights ' + arm)
        require(info['checkpoint']['sha256'] == meta[arm]['checkpoint']['sha256'], 'head checkpoint identity differs')
        expected_calls = 0 if arm == 'IMAGE_ROLE' else 8
        require(info['new_head_forward_calls'] == completion['actual_counts'][arm]['completed'] == expected_calls
                and completion['actual_counts'][arm]['attempted'] == expected_calls
                and info['new_head_image_exposures'] == expected_calls * 16, 'actual calibration call ledger differs')
    for arm in head_arms:
        directory = root / arm
        info = completion['arms'][arm]
        check_binding(directory / 'CALIBRATION.json', info['calibration'], 'calibration ' + arm)
        check_binding(directory / 'CALIBRATION_EXECUTION.json', info['execution'], 'execution ' + arm)
        value, execution = read(directory / 'CALIBRATION.json'), read(directory / 'CALIBRATION_EXECUTION.json')
        require(execution['complete'] and execution['arm'] == arm and execution['partition'] == 'calibration',
                'head-specific source calibration not completed')
        require(execution['head_forward_calls'] == 8 and execution['head_image_exposures'] == 128,
                'per-arm calibration historical/current eight-call ledger differs')
        if arm == 'IMAGE_ROLE':
            reuse = read(directory / 'REUSE_RECEIPT.json')
            check_binding(directory / 'REUSE_RECEIPT.json', info['reuse'], 'ROLE reuse receipt')
            require(reuse['complete'] and reuse['current_new_head_forward_calls'] == 0
                    and reuse['current_new_image_exposures'] == 0 and reuse['numerical_recalibration'] is False,
                    'ROLE must use byte-exact old calibration with zero new model calls')
            require(reuse['checkpoint']['sha256'] == CHECKPOINT_SHAS[arm], 'ROLE calibration/checkpoint differs')
            for name in ROLE_FILES:
                check_binding(directory / name, protocol['inputs']['ROLE:' + name], 'reused ROLE ' + name)
                check_binding(directory / name, reuse['files'][name], 'ROLE copied file receipt ' + name)
            require(value['protocol_sha256'] == protocol['inputs']['source_algorithm_protocol']['sha256'],
                    'ROLE coefficient protocol must stay original')
        else:
            own = read(directory / 'CALIBRATION_PROTOCOL.json')
            check_binding(directory / 'CALIBRATION_PROTOCOL.json', execution['protocol'], 'arm protocol ' + arm)
            require(own['arm'] == value['arm'] == value['training_arm'] == execution['training_arm'] == arm
                    and own['head_mode'] == MODES[arm] and own['indices'] == list(INDICES),
                    'head-specific calibration modes/provenance differ')
            require(value['checkpoint']['sha256'] == execution['checkpoint']['sha256'] == CHECKPOINT_SHAS[arm],
                    'calibration selected another arm checkpoint')
            require(value['protocol_sha256'] == execution['protocol']['sha256']
                    and value['source_algorithm_protocol_sha256'] == protocol['inputs']['source_algorithm_protocol']['sha256']
                    and own['source_algorithm_policy'] == protocol['source_algorithm_policy'],
                    'head-specific coefficients/source algorithm protocol differ')
        for key in ('source_test_exposures', 'real_frames', 'training_updates', 'detector_calls',
                    'PnP_calls', 'rays', 'new_RGB', 'feature_recomputations'):
            require(execution[key] == 0, 'per-arm calibration exposure/scope violation: ' + key)
    return dict(passed=True, arms=list(head_arms), source_only=True, new_head_forward_calls=16,
                ROLE_new_head_forward_calls=0, protocol=binding(root / 'PROTOCOL.json'))


def copy_role(root, locked):
    directory = root / 'IMAGE_ROLE'
    directory.mkdir()
    copied = {}
    for name in ROLE_FILES:
        target = directory / name
        actual = copy_exact(V2_DOC / name, target, locked['ROLE:' + name])
        copied[name] = actual
    receipt = dict(schema='exact_ROLE_source_calibration_reuse_v7', complete=True,
                   arm='IMAGE_ROLE', current_new_head_forward_calls=0, current_new_image_exposures=0,
                   historical_head_forward_calls=8, historical_image_exposures=128,
                   checkpoint=locked['checkpoint:IMAGE_ROLE'], files=copied,
                   protocol=binding(root / 'PROTOCOL.json'), numerical_recalibration=False)
    write_new(directory / 'REUSE_RECEIPT.json', receipt)
    return receipt


def copy_exact(source, target, expected):
    require(not Path(target).is_symlink(), 'symlink byte reuse destination')
    with Path(source).open('rb') as original, Path(target).open('xb') as destination:
        shutil.copyfileobj(original, destination)
    actual = binding(target)
    require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'byte-exact reuse failed')
    return actual


@contextmanager
def source_only_canary():
    import builtins
    import io
    forbidden = ('TARGETS.json', 'GEOMETRY_RESOLVED_POSE_GT', 'STATIC_VISIBILITY',
                 'PREDICTIONS.json', 'METRICS.json', 'AXIS_REVIEW', 'severity_snapshot')
    original = (builtins.open, io.open)
    def guarded(fn):
        def call(path, *args, **kwargs):
            if isinstance(path, (str, Path)):
                require(not any(t.lower() in str(path).lower() for t in forbidden)
                        and Path(path).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.usd'),
                        'source calibration attempted real evidence/RGB/mesh read')
            return fn(path, *args, **kwargs)
        return call
    builtins.open, io.open = map(guarded, original)
    try:
        yield
    finally:
        builtins.open, io.open = original


@contextmanager
def no_pose_calls():
    # These guards fail on an accidental numeric-pose call; they do not execute
    # any pose primitive. The copied calibration algorithm has no such calls.
    import cv2
    names = ('solvePnP', 'solvePnPGeneric', 'solvePnPRansac', 'solvePnPRefineLM', 'solvePnPRefineVVS')
    originals = {name: getattr(cv2, name) for name in names if hasattr(cv2, name)}
    def forbidden(*_args, **_kwargs):
        raise AssertionError('source calibration attempted a numeric pose primitive')
    for name in originals: setattr(cv2, name, forbidden)
    try:
        yield
    finally:
        for name, function in originals.items(): setattr(cv2, name, function)


def run(args):
    begin = time.monotonic()
    root = destinations(args, frozen=True)
    protocol = read(root / 'PROTOCOL.json')
    before = inputs(args)
    require(before == protocol['inputs'] and protocol['new_arms'] == list(NEW_ARMS)
            and protocol['head_modes'] == MODES, 'source calibration frozen inputs/policy differ')
    meta = checkpoint_identity(args, before)
    protect(args)
    write_new(root / 'STARTED.json', dict(protocol=binding(root / 'PROTOCOL.json'), utc=utc(),
              attempted_head_calls=0, automatic_retry=False))
    counts = {arm: dict(attempted=0, completed=0, image_exposures=0, attempted_image_exposures=0) for arm in ARMS}
    resources, completed, error, preservation = [], {}, None, None
    old_env = os.environ.get('PALLET_SOURCE_ROOT')
    try:
        from scripts.research.pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet
        probe = quiet(); resources.append(probe)
        write_new(root / 'RESOURCE_000.json', probe)
        require(probe['quiet'] and probe['temperature_under_80'], 'source calibration resource guard')
        role = copy_role(root, before)
        completed['IMAGE_ROLE'] = dict(calibration=binding(root / 'IMAGE_ROLE' / 'CALIBRATION.json'),
             execution=binding(root / 'IMAGE_ROLE' / 'CALIBRATION_EXECUTION.json'),
             reuse=binding(root / 'IMAGE_ROLE' / 'REUSE_RECEIPT.json'), checkpoint=before['checkpoint:IMAGE_ROLE'],
             new_head_forward_calls=0, new_head_image_exposures=0, reused_byte_exact=True)
        import numpy as np
        source = source_rows(REPAIR / 'READY_SOURCE_TARGET_ROWS.jsonl.gz')
        cache_manifest = Path(args.cache_manifest) if args.cache_manifest else Path(args.features).with_name('CACHE_MANIFEST.json')
        validate_cache_sources(source, read(cache_manifest),
             read(REPO / '_docs/experiments/pallet_observation_refiner_20261009_v1/SOURCE_FAMILY_SPLIT.json'))
        features = np.load(args.features, mmap_mode='r')
        require(features.shape == (1024, 84, 28, 65) and features.dtype == np.float16, 'immutable feature cache schema')
        with np.load(REPAIR / 'READY_PREPARED_TARGETS.npz') as archive:
            targets = {key: archive[key] for key in ('lo', 'hi', 'weight', 'valid')}
        require(all(x.shape == (1024, 84) for x in targets.values()), 'target array schema')
        os.environ['PALLET_SOURCE_ROOT'] = str(Path(args.source_root).resolve())
        import torch
        from scripts.research.pallet_observation_refiner_20261009_v1 import model as M
        from scripts.research.pallet_boundary_corner_refiner_20261010_v2 import calibration as A
        torch.set_num_threads(1)
        require(M.CONFIG == meta[ARMS[0]]['config'] and tuple(M.ARMS) == ARMS, 'original tiny model contract changed')
        versions = dict(Python=sys.version, numpy=np.__version__, torch=torch.__version__, CUDA=torch.version.cuda,
                        cudnn=torch.backends.cudnn.version(), threads=torch.get_num_threads(),
                        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                        cudnn_benchmark=torch.backends.cudnn.benchmark,
                        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                        historical_v2_backend_flags='not recorded')
        for arm in NEW_ARMS:
            directory = root / arm; directory.mkdir()
            arm_protocol = dict(schema='per_head_source_calibration_protocol_v7', arm=arm, head_mode=MODES[arm],
                 inputs=before, parent_protocol=binding(root / 'PROTOCOL.json'), checkpoint=before['checkpoint:' + arm],
                 source_algorithm_protocol=before['source_algorithm_protocol'],
                 source_algorithm_policy=protocol['source_algorithm_policy'], indices=list(INDICES),
                 batch_size=16, head_forward_calls=8, head_image_exposures=128)
            write_new(directory / 'CALIBRATION_PROTOCOL.json', arm_protocol)
            write_new(directory / 'CALIBRATION_START.json', dict(protocol=binding(directory / 'CALIBRATION_PROTOCOL.json'),
                       arm=arm, attempted_head_forward_calls=0, training_updates=0))
            saved, head, hooks, arm_error = [], None, [], None
            arm_begin = time.monotonic()
            try:
                checkpoint = torch.load(Path(args.fits) / (arm + '.pt'), map_location='cpu', weights_only=False)
                require(checkpoint['arm'] == arm and checkpoint['steps'] == 3000, 'checkpoint arm/step differs')
                for key in ('config', 'initial_state_sha256', 'batch_order_sha256', 'protocol_sha256', 'repaired_target_sha256'):
                    require(checkpoint[key] == meta[arm][key], 'checkpoint training provenance differs: ' + key)
                head = M.CorrespondenceHead().cuda().eval()
                require(sum(p.numel() for p in head.parameters()) == 5890, 'head parameter count differs')
                head.load_state_dict(checkpoint['model'], strict=True); head.requires_grad_(False)
                def attempted(_module, _args):
                    counts[arm]['attempted'] += 1
                    counts[arm]['attempted_image_exposures'] += int(_args[0].shape[0])
                def done(_module, _args, _result):
                    counts[arm]['completed'] += 1
                    counts[arm]['image_exposures'] += int(_args[0].shape[0])
                hooks = [head.register_forward_pre_hook(attempted), head.register_forward_hook(done)]
                with source_only_canary(), no_pose_calls(), torch.no_grad():
                    for start in range(768, 896, 16):
                        x = torch.tensor(np.asarray(features[start:start + 16]), dtype=torch.float32, device='cuda')
                        logits = head(x, arm).float().cpu().numpy()
                        require(logits.shape == (16, 84, 66) and np.isfinite(logits).all(), 'finite original head logits required')
                        for j, index in enumerate(range(start, start + 16)):
                            raw = logits[j].astype('<f4').tobytes(); row = source[index - 768]
                            saved.append(dict(index=index, id=row['id'], family=row['family'], partition='calibration',
                                arm=arm, logits_shape=[84, 66], raw_logits_sha256=hashlib.sha256(raw).hexdigest(),
                                logits_fp32_zlib_base64=base64.b64encode(zlib.compress(raw)).decode(),
                                source_ready_row_semantic_sha256=hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()))
                require(counts[arm] == dict(attempted=8, completed=8, image_exposures=128, attempted_image_exposures=128) and len(saved) == 128,
                        'fixed eight CAL batches incomplete')
                save_new(directory / 'CALIBRATION_ROWS.jsonl.gz', saved)
                with source_only_canary(), no_pose_calls():
                    value, scan, query, geometry = A.calibrate(saved, source, targets, read(V2_DOC / 'CALIBRATION_PROTOCOL.json'))
                value = annotate_calibration(value, arm, binding(directory / 'CALIBRATION_PROTOCOL.json'),
                                             before['source_algorithm_protocol'], before['checkpoint:' + arm])
                for name, rows in (('CALIBRATION_THRESHOLD_SCAN.jsonl.gz', scan),
                                   ('CALIBRATION_QUERY_DIAGNOSTICS.jsonl.gz', query),
                                   ('CALIBRATION_GEOMETRY_ROWS.jsonl.gz', geometry)):
                    save_new(directory / name, rows)
                value.update(raw_rows=binding(directory / 'CALIBRATION_ROWS.jsonl.gz'),
                             threshold_scan=binding(directory / 'CALIBRATION_THRESHOLD_SCAN.jsonl.gz'),
                             geometry_rows=binding(directory / 'CALIBRATION_GEOMETRY_ROWS.jsonl.gz'))
                write_new(directory / 'CALIBRATION.json', value)
            except BaseException as exc:
                arm_error = dict(type=type(exc).__name__, message=str(exc))
                if not (directory / 'CALIBRATION_ROWS.jsonl.gz').exists():
                    save_new(directory / 'CALIBRATION_ROWS.jsonl.gz', saved)
                raise
            finally:
                for hook in hooks: hook.remove()
                if head is not None: del head
                torch.cuda.empty_cache()
                execution = dict(complete=arm_error is None, arm=arm, training_arm=arm,
                    protocol=binding(directory / 'CALIBRATION_PROTOCOL.json'),
                    checkpoint=before['checkpoint:' + arm], input_bindings=before,
                    calibration=binding(directory / 'CALIBRATION.json') if (directory / 'CALIBRATION.json').exists() else None,
                    head_forward_calls=counts[arm]['completed'], attempted_head_forward_calls=counts[arm]['attempted'],
                    head_image_exposures=counts[arm]['image_exposures'], partition='calibration', raw_rows=len(saved),
                    attempted_head_image_exposures=counts[arm]['attempted_image_exposures'],
                    source_test_exposures=0, real_frames=0, training_updates=0, detector_calls=0, N3_calls=0,
                    PnP_calls=0, rays=0, new_RGB=0, feature_recomputations=0,
                    versions=versions, error=arm_error, wall_seconds=time.monotonic() - arm_begin)
                execution.update(source_only_file_canary=True, numeric_pose_call_guard=True,
                                 target_array_scope='NPZ members loaded whole; only indices768:896 accessed by calibration')
                write_new(directory / 'CALIBRATION_EXECUTION.json', execution)
            completed[arm] = dict(calibration=binding(directory / 'CALIBRATION.json'),
                 execution=binding(directory / 'CALIBRATION_EXECUTION.json'), checkpoint=before['checkpoint:' + arm],
                 new_head_forward_calls=8, new_head_image_exposures=128, reused_byte_exact=False)
        require(inputs(args) == before, 'source/head/algorithm/protection input bytes changed')
        preservation = protect(args)
        require(all(counts[a]['completed'] == 8 for a in NEW_ARMS) and counts['IMAGE_ROLE']['completed'] == 0,
                'new calibration call count differs')
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if old_env is None: os.environ.pop('PALLET_SOURCE_ROOT', None)
        else: os.environ['PALLET_SOURCE_ROOT'] = old_env
        after = inputs(args)
        write_new(root / 'COMPLETION.json', dict(schema='three_fixed_head_source_calibration_completion_v7',
              complete=error is None, protocol=binding(root / 'PROTOCOL.json'), inputs_before_after_equal=before == after,
              input_bindings=before, arms=completed, actual_counts=counts,
              actual_new_head_forward_calls=sum(c['completed'] for c in counts.values()),
              actual_new_head_image_exposures=sum(c['image_exposures'] for c in counts.values()),
              ROLE_new_head_forward_calls=0, historical_ROLE_head_forward_calls=8,
              source_test_exposures=0, real_frames=0, new_training_updates=0, detector_calls=0, N3_calls=0,
              PnP_calls=0, rays=0, new_RGB=0, feature_recomputations=0,
              protection_after=preservation, resource_probes=resources, error=error, wall_seconds=time.monotonic() - begin))
    print('SOURCE_CALIBRATION_COMPLETE', sha(root / 'COMPLETION.json'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--features', required=True)
    parser.add_argument('--cache-manifest', help='defaults to CACHE_MANIFEST.json beside immutable features.npy')
    parser.add_argument('--fits', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--prior-bindings', default=str(DOC / 'PROTECTION_BEFORE.json'))
    parser.add_argument('--output', default=str(DOC / 'source_calibration'))
    args = parser.parse_args()
    freeze(args) if args.stage == 'freeze' else run(args)


if __name__ == '__main__':
    main()
