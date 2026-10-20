"""Resumable execution of the unchanged original D1 on new frozen inputs.

Only this controller is new. Each available input goes through the original
RGB observer, original P1/WD0/G10/G13 hierarchy, and original62 features.
Existing YOLO observations, stages, poses, or feature caches are never loaded.
Unavailable upstream rows remain explicit NO_POSE rows. This module neither
trains nor selects a head, scores a pose, reads a reference, or draws a figure.

Observation is one persistent CUDA job, followed by at most4 CPU jobs. Stages
do not overlap. Persistent neural hooks repair the historical per-frame SAM
counter gap without another forward. Invocation and failed-attempt costs are
retained separately from cumulative completed-row costs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import time
import traceback
from types import FunctionType, SimpleNamespace
import uuid


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'))


def write(path, value):
    """Write once; exact metadata reuse is allowed."""
    path = Path(path)
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Preserve existing artifact: ' + str(path))
        return
    temporary = path.with_name(path.name + '.pending.' + uuid.uuid4().hex)
    temporary.write_bytes(data)
    temporary.rename(path)


def save_rows(path, values):
    path = Path(path)
    payload = ''.join(canonical(value) + '\n' for value in values).encode()
    # Stable bytes across resume and independent machines.
    data = gzip.compress(payload, compresslevel=5, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Completed row archive changed: ' + str(path))
    else:
        path.write_bytes(data)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def utc():
    return datetime.now(timezone.utc).isoformat()


def producer_modules():
    from scripts.research.pallet_boundary_action_repair_v1 import d1_batch as B
    from scripts.research.pallet_boundary_action_repair_v1 import d1_producer as P
    from scripts.research.pallet_boundary_action_repair_v1 import d1_cpu_logging_v3 as V
    return B, P, V


def sources(repo):
    B, P, V = producer_modules()
    repo = Path(repo).resolve()
    if any(not Path(module.__file__).resolve().is_relative_to(repo) for module in (B, P, V)):
        raise ValueError('Producer imports must come from the prepared isolated repository')
    result = {str((Path(repo) / key).resolve()): value
              for key, value in V.source_bindings().items()}
    result[str(Path(__file__).resolve())] = sha(__file__)
    return result


def model_inventory(repo):
    """Metadata and hashes only: no network/model constructor is called."""
    repo = Path(repo)
    study = repo / ('_docs/experiments/pallet_visibility_curriculum_20261013_v1/'
                    'nonsevere_v1/statewise_joint_ot_20261014_v2')
    runtime_path = study / 'RUNTIME_CONFIG_V3.json'
    runtime = read(runtime_path)
    bindings = {str(runtime_path): sha(runtime_path)}
    asset_paths = {}
    descriptors = {}
    for key in ('state_model', 'OT_model', 'mask_descriptor', 'boundary_mask_config'):
        path = (study / runtime[key]).absolute()
        if sha(path) != runtime['bindings_sha256'][key]:
            raise ValueError('Frozen descriptor mismatch: ' + key)
        bindings[str(path)] = runtime['bindings_sha256'][key]
        descriptors[key] = path, read(path)
    for key in ('state_model', 'mask_descriptor'):
        path, desc = descriptors[key]
        bindings[str(path.parent / desc['arrays_file'])] = desc['arrays_sha256']
        for role in ('backbone', 'pallet_segmenter'):
            item = desc[role]
            bindings[str(repo / item['weights_repo_relative'])] = item['weights_sha256']
    path, boundary = descriptors['boundary_mask_config']
    bindings[str((path.parent / boundary['mobile_sam_checkpoint']).absolute())] = \
        boundary['model_sha256']['mobile_sam_checkpoint']
    vendor = (path.parent / boundary['mobile_sam_vendor']).absolute()
    # Bind vendor implementation, not merely a successful import elsewhere.
    if vendor.is_dir():
        for path in sorted(vendor.rglob('*.py')):
            bindings[str(path.absolute())] = sha(path)
    asset_paths['mobile_sam_vendor'] = str(vendor)
    fix = study / 'nonsevere_v4_fix'
    for rel in ('calibration/final_adapter_CAL46/MODEL.json',
                'truncation_interval_model_CAL_v4_2/MODEL.json',
                'boundary_recovery_v1/shared_edge_partial_b_eval_v4_7/PROTOCOL.json'):
        path = fix / rel
        bindings[str(path)] = sha(path)
    readiness = {path: Path(path).is_file() and sha(path) == digest
                 for path, digest in bindings.items()}
    return dict(bindings=bindings, paths=asset_paths,
                missing_or_mismatched=[path for path, ok in readiness.items() if not ok],
                vendor_available=(vendor / 'mobile_sam/__init__.py').is_file(),
                weights_decoded=False, neural_forwards=0)


def verify_upstream(path, seal_path, frames, backbone=None):
    seal = read(seal_path)
    if seal.get('complete') is not True or seal.get('rows') != 238:
        raise ValueError('Complete238 upstream seal required')
    if seal.get('schema') != 'D1_THREE_BACKBONE_UPSTREAM_SEAL_V1' or \
            seal.get('all238_including_failures_retained') is not True:
        raise ValueError('Explicit all238 transfer upstream contract required')
    if backbone is not None and seal.get('backbone') != backbone:
        raise ValueError('Upstream backbone identity changed')
    candidates = [seal.get(key) for key in ('upstream_sha256', 'predictions_sha256',
                  'input_sha256', 'output_sha256', 'inputs_sha256')]
    artifact = seal.get('artifacts', {}).get(Path(path).name)
    candidates.append(artifact.get('sha256') if isinstance(artifact, dict) else artifact)
    if sha(path) not in candidates:
        raise ValueError('Upstream input bytes not authenticated')
    if seal.get('ordered_ids') is not None and seal['ordered_ids'] != [f['id'] for f in frames]:
        raise ValueError('Upstream ordered IDs mismatch')


def validate_frame_set(frames):
    from scripts.research.pallet_boundary_action_repair_v1.pipeline_contract import ALLOWED, SCHEMA
    if len(frames) != 238 or len({f['id'] for f in frames}) != 238:
        raise ValueError('All238 unique identities are required')
    if Counter(f['partition'] for f in frames) != Counter(CAL=46, TEST=49, REMAINDER=143):
        raise ValueError('Original CAL46/TEST49/REMAINDER143 metadata required')
    for frame in frames:
        if set(frame) - ALLOWED or frame.get('schema') != SCHEMA:
            raise ValueError('Original closed upstream schema required')
        available = frame.get('initial_pose', {}).get('available')
        if not isinstance(available, bool):
            raise ValueError('Explicit boolean upstream availability required')
        if sha(frame['image']) != frame['image_sha256']:
            raise ValueError('Original RGB input hash changed')
        if not available and set(frame['initial_pose']) != {'available'}:
            raise ValueError('Unavailable input must not carry a fabricated fallback pose')
    return frames


def prepare(repo, run, inputs_by_backbone, old_run, *, workers=4, device='cuda',
            upstream_seals=None):
    """Create a read-only-input plan. This does not run observation or solving."""
    if not 1 <= workers <= 4 or device != 'cuda':
        raise ValueError('One CUDA observation job, then at most4 CPU workers')
    if set(inputs_by_backbone) - {'yolo', 'dope', 'resnet18'} or not inputs_by_backbone:
        raise ValueError('Explicit supported backbone input map required')
    if not upstream_seals or set(upstream_seals) != set(inputs_by_backbone):
        raise ValueError('Each input must have its own upstream seal')
    root, repo, old = Path(run).resolve(), Path(repo).resolve(), Path(old_run).resolve()
    arms = {}
    fixed = {str(old / 'CONFIG_FROZEN.json'): sha(old / 'CONFIG_FROZEN.json')}
    common_ids = None
    for name, path in inputs_by_backbone.items():
        path = Path(path).resolve()
        frames = validate_frame_set(rows(path))
        seal_path = Path(upstream_seals[name]).resolve()
        verify_upstream(path, seal_path, frames, name)
        ids = [f['id'] for f in frames]
        if common_ids is not None and ids != common_ids:
            raise ValueError('All backbones must retain the same238 identities and order')
        common_ids = ids
        fixed.update({str(path): sha(path), str(seal_path): sha(seal_path)})
        arms[name] = dict(inputs=str(path), upstream_seal=str(seal_path), ordered_ids=ids,
                         available=sum(f['initial_pose']['available'] for f in frames),
                         unavailable=sum(not f['initial_pose']['available'] for f in frames))
    inventory = model_inventory(repo)
    plan = dict(schema='D1_THREE_BACKBONE_EXECUTION_PLAN_V1', complete=True,
                prepared_utc=utc(), repo=str(repo), run=str(root), old_run=str(old),
                workers=workers, device=device, arms=arms, sources=sources(repo),
                fixed_inputs=fixed, models=inventory['bindings'],
                asset_paths=inventory['paths'], readiness_at_prepare=inventory,
                algorithm='UNCHANGED_ORIGINAL_D1_OBSERVE_P1_WD0_G10_G13_FEATURES',
                stage_memoization=False, previous_YOLO_observation_or_pose_reuse=False,
                numeric_original_CAL_and_state_models_frozen=True,
                legacy_runtime_prediction_source_tag='N3_SUBPIX',
                actual_input_backbone_is_external_upstream_provenance=True,
                new_training_or_H1_prediction_or_scoring=False,
                timeouts_or_candidate_truncation=False,
                unavailable_rows='NO_POSE; candidates[]; no observation or solver calls',
                whole_campaign_complete=False)
    root.mkdir(parents=True, exist_ok=True)
    write(root / 'EXECUTION_PLAN.json', plan)
    return plan


def guard(plan, *, require_assets=True):
    if sha(__file__) != plan['sources'][str(Path(__file__).resolve())]:
        raise ValueError('Execution controller changed after preparation')
    for group in ('sources', 'fixed_inputs'):
        for path, digest in plan[group].items():
            if sha(path) != digest:
                raise ValueError('Frozen ' + group + ' binding changed: ' + path)
    if sources(plan['repo']) != plan['sources']:
        raise ValueError('Loaded producer source closure differs from prepared repository')
    if require_assets:
        for path, digest in plan['models'].items():
            if not Path(path).is_file() or sha(path) != digest:
                raise ValueError('Frozen model or descriptor unavailable/changed: ' + path)
        vendor = Path(plan['asset_paths']['mobile_sam_vendor'])
        if not (vendor / 'mobile_sam/__init__.py').is_file():
            raise ValueError('Frozen local MobileSAM vendor unavailable')
        expected = {str(p.absolute()): sha(p) for p in sorted(vendor.rglob('*.py'))}
        if not expected or any(plan['models'].get(p) != v for p, v in expected.items()):
            raise ValueError('Vendor Python source not bound at prepare; prepare after asset linking')


def archive_incomplete(folder, seal_name, attempts):
    folder = Path(folder)
    if folder.exists() and not (folder / seal_name).exists() and any(folder.iterdir()):
        target = Path(attempts) / (folder.name + '_' + uuid.uuid4().hex)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(folder), str(target))
        return str(target)
    return None


class PersistentFrameMeter:
    """Small adapter used by unchanged observe_batch's per-row context."""
    def __init__(self, persistent, current, journal):
        self.persistent, self.current, self.journal = persistent, current, Path(journal)
        self.record = None

    def __enter__(self):
        self.persistent.meter.reset()
        self.persistent.events = []
        self.started, self.cpu = time.monotonic(), time.process_time()
        self.current.clear()
        return self

    def receipt(self):
        self.record = self.persistent.receipt()
        return deepcopy(self.record)

    def __exit__(self, kind, exc, tb):
        timing_error = None
        if self.record is None:
            try:
                self.receipt()
            except BaseException as failure:
                timing_error = repr(failure)
                self.record = dict(actual_calls=self.persistent.meter.snapshot(),
                                   CUDA_forward_seconds=None,
                                   CUDA_duration_status='UNKNOWN_NOT_ZERO')
        append(self.journal, dict(stage='OBSERVATION_ATTEMPT', id=self.current.get('id'),
               status='FAILED_TECHNICAL' if kind else 'RETURNED', resource=self.record,
               wall_seconds=time.monotonic() - self.started,
               CPU_seconds=time.process_time() - self.cpu,
               error=repr(exc) if exc else None, timing_error=timing_error,
               failed_forward_GPU_duration_status='UNKNOWN_NOT_ZERO' if kind else None,
               neural_counts_scope='actual forward entry hooks; attempted calls include failures'))
        return False


def observe_valid(inputs, output, old_run, ledger, model_bindings):
    """The original observer batch with persistent measurement hooks only."""
    B, P, _ = producer_modules()
    current = {}
    original_meter = B.NeuralMeter
    with original_meter() as persistent:
        def meter_factory():
            return PersistentFrameMeter(persistent, current, ledger)

        def observe(frame, bundle):
            current['id'] = frame['id']
            return P.observe(frame, bundle)

        proxy = SimpleNamespace(**{**vars(P), 'observe': observe})
        original_read = B.observe_batch.__globals__['read']

        def read_with_completed_frame_recovery(path):
            # A process can exit after the last row seal, before the aggregate
            # seal. Original observe_batch then has no pending model job and
            # reads the not-yet-written aggregate for its model bindings.
            if Path(path) == Path(output) / 'OBSERVATION_SEAL.json' and not Path(path).exists():
                return {'model_bindings': model_bindings}
            return original_read(path)

        operation = FunctionType(B.observe_batch.__code__,
            {**B.observe_batch.__globals__, 'NeuralMeter': meter_factory, 'P': proxy,
             'read': read_with_completed_frame_recovery})
        operation.__kwdefaults__ = deepcopy(B.observe_batch.__kwdefaults__)
        return operation(inputs, output, old_run, 'real', device='cuda', GO=True)


@contextmanager
def existing_counts(counter):
    yield counter


def solve_worker(frame, obs, output, old_run):
    """Record all original numerical calls even if the original job raises."""
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    B, _, V = producer_modules()
    B._preload_solver()
    tick, cpu = time.monotonic(), time.process_time()
    original = B.numerical_meter
    failure, result = None, None
    with original() as counts:
        # Reuse this one counter in the original nested solver_job context.
        # A second numerical hook would hide minimize aliases from its own
        # counter. Arguments, arithmetic order and return values stay original.
        B.numerical_meter = lambda: existing_counts(counts)
        try:
            result = V.solver_job(frame, obs, output, old_run)
        except BaseException as exc:
            failure = dict(error=repr(exc), traceback=traceback.format_exc())
        finally:
            B.numerical_meter = original
    receipt = dict(id=frame['id'], complete=failure is None,
                   status='SEALED' if failure is None else 'FAILED_TECHNICAL',
                   resource=dict(wall_seconds=time.monotonic() - tick,
                                 CPU_seconds=time.process_time() - cpu,
                                 max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                   operations=dict(counts), failure=failure,
                   frame_seal_sha256=sha(Path(output) / 'FRAME_SEAL.json') if result else None,
                   new_neural_forwards=0, stage_memoization=False)
    write(Path(output) / 'CONTROLLER_ATTEMPT.json', receipt)
    return receipt


def validate_observation(frame, folder, B):
    folder = Path(folder)
    seal = read(folder / 'OBSERVATION_SEAL.json')
    if not seal.get('complete') or seal['id'] != frame['id']:
        raise ValueError('Completed observation identity/status changed')
    if seal['source_sha256'] != B.source_bindings():
        raise ValueError('Completed observation source changed')
    if seal['input_row_sha256'] != hashlib.sha256(canonical(frame).encode()).hexdigest():
        # Original common.canonical uses the same sorted compact encoding.
        from scripts.research.pallet_statewise_end_to_end_recovery_v1.common import canonical as original
        if seal['input_row_sha256'] != hashlib.sha256(original(frame).encode()).hexdigest():
            raise ValueError('Completed observation input row changed')
    for name, key in (('OBSERVATION.jsonl.gz', 'observation_sha256'), ('MASK.npz', 'mask_sha256')):
        if sha(folder / name) != seal[key]:
            raise ValueError('Completed observation output changed')
    return seal


def validate_solve(frame, obs, folder, V):
    folder = Path(folder)
    seal = read(folder / 'FRAME_SEAL.json')
    if seal['id'] != frame['id'] or seal.get('complete') is not True:
        raise ValueError('Completed solver identity/status changed')
    if seal['source_sha256'] != V.source_bindings():
        raise ValueError('Completed solver source changed')
    if seal['observation_seal_sha256'] != sha(Path(obs) / 'OBSERVATION_SEAL.json'):
        raise ValueError('Completed solver observation parent changed')
    for name, key in (('E_RUN.jsonl.gz', 'e_run_sha256'), ('FEATURES.jsonl.gz', 'features_sha256'),
                      ('BOUNDARY_BUNDLE.jsonl.gz', 'boundary_bundle_sha256')):
        if sha(folder / name) != seal[key]:
            raise ValueError('Completed solver output changed')
    if not seal['e0_independently_executed'] or seal['history_only_union'] or seal['new_B_or_A_candidates']:
        raise ValueError('Completed original D1 contract changed')
    return seal


def unavailable_feature(frame):
    if frame['initial_pose'] != {'available': False}:
        raise ValueError('Only explicit unavailable upstream enters zero-call branch')
    return dict(id=frame['id'], partition=frame['partition'], split=frame['split'],
                session=frame['session'], cluster_id=frame['cluster_id'],
                candidates=[], fallback={'available': False}, e0_regenerated={'available': False},
                feature_names=[],
                transformation='signed_log1p31_plus31_explicit_validity_bits',
                prediction_status='NO_POSE_UPSTREAM_UNAVAILABLE',
                no_pose_reason='See bound upstream journal: no valid same-backbone F initial pose',
                reference_inputs=False, D1_observation_calls=0, D1_solver_calls=0,
                proof=dict(upstream_unavailable=True, object_scale_not_applicable=True,
                           e0_independently_executed=False,
                           no_fabricated_pose=True, reference_reads=0))


def import_exact_pilot(plan, name, valid, ledger):
    """Reuse only a new same-backbone technical row with exact bound inputs."""
    B, _, V = producer_modules()
    root = Path(plan['run'])
    pilot_root = root / 'technical/cal_pilot_v1' / name
    receipt_path = pilot_root / 'PILOT_RECEIPT.json'
    if not receipt_path.exists():
        return None
    receipt, lock = read(receipt_path), read(pilot_root / 'PILOT_LOCK.json')
    if receipt.get('complete') is not True or receipt['backbone'] != name or \
            receipt['plan_sha256'] != sha(root / 'EXECUTION_PLAN.json'):
        raise ValueError('Technical pilot receipt is not a completed exact parent')
    validate_complete(pilot_root, receipt)
    matching = [(i, f) for i, f in enumerate(valid) if f['id'] == receipt['id']]
    if len(matching) != 1:
        raise ValueError('Technical pilot identity absent from available production input')
    index, frame = matching[0]
    if lock['input_row_sha256'] != hashlib.sha256(canonical(frame).encode()).hexdigest():
        raise ValueError('Technical pilot and production input rows differ')
    op, candidate = pilot_root / 'observations/frames/0000', pilot_root / 'candidates/frames/0000'
    validate_observation(frame, op, B)
    validate_solve(frame, op, candidate, V)
    import_receipt = root / name / 'EXACT_TECHNICAL_PILOT_IMPORT.json'
    if import_receipt.exists():
        previous = read(import_receipt)
        if previous['id'] != frame['id'] or previous['pilot_receipt_sha256'] != sha(receipt_path) or \
                previous['exact_input_row_sha256'] != lock['input_row_sha256']:
            raise ValueError('Completed technical import parent changed')
        for path, digest in previous['copied_artifacts'].items():
            if sha(root / path) != digest:
                raise ValueError('Completed exact technical import changed')
        return previous
    copied = {}
    for source, destination in ((op, root / name / 'observations/frames' / f'{index:04d}'),
                                (candidate, root / name / 'candidates/frames' / f'{index:04d}')):
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            # A prior completed row is independently validated by run_arm;
            # an incomplete attempt is archived before this helper is called.
            continue
        for path in source.iterdir():
            if path.is_file():
                target = destination / path.name
                shutil.copy2(path, target)
                if sha(target) != sha(path):
                    raise ValueError('Exact technical row copy failed')
                copied[str(target.relative_to(root))] = sha(target)
    result = dict(id=frame['id'], backbone=name, pilot_receipt_sha256=sha(receipt_path),
                  exact_input_row_sha256=lock['input_row_sha256'], copied_artifacts=copied,
                  new_observation_or_solver_calls=0,
                  cost_scope='technical work was measured once in pilot; production row totals are cumulative and must not be added again')
    write(import_receipt, result)
    append(ledger, dict(stage='EXACT_SAME_BACKBONE_TECHNICAL_PILOT_REUSE', **result))
    return result


def run_arm(plan, name, invocation):
    B, _, V = producer_modules()
    spec = plan['arms'][name]
    frames = validate_frame_set(rows(spec['inputs']))
    valid = [f for f in frames if f['initial_pose']['available']]
    root = Path(plan['run']) / name
    root.mkdir(parents=True, exist_ok=True)
    valid_input = root / 'AVAILABLE_UPSTREAM.jsonl.gz'
    save_rows(valid_input, valid)
    obs, out = root / 'observations', root / 'candidates'
    attempted = Path(plan['run']) / 'attempts' / invocation / name
    ledger = Path(plan['run']) / 'ACTUAL_OPERATION_LEDGER.jsonl'
    for index, frame in enumerate(valid):
        for base, seal, category in ((obs, 'OBSERVATION_SEAL.json', 'observations'),
                                     (out, 'FRAME_SEAL.json', 'solves')):
            folder = base / 'frames' / f'{index:04d}'
            archived = archive_incomplete(folder, seal, attempted / category)
            if archived:
                append(ledger, dict(stage='PRESERVE_INCOMPLETE_ATTEMPT', id=frame['id'],
                       category=category, archive=archived,
                       interrupted_partial_cost='UNKNOWN_NOT_ZERO unless a durable attempt receipt tracks it'))
    pilot_import = import_exact_pilot(plan, name, valid, ledger)
    if valid:
        for index, frame in enumerate(valid):
            folder = obs / 'frames' / f'{index:04d}'
            if (folder / 'OBSERVATION_SEAL.json').exists():
                validate_observation(frame, folder, B)
            else:
                archived = archive_incomplete(folder, 'OBSERVATION_SEAL.json', attempted / 'observations')
                if archived:
                    append(ledger, dict(stage='PRESERVE_INCOMPLETE_OBSERVATION', id=frame['id'], archive=archived))
        if (obs / 'OBSERVATION_SEAL.json').exists():
            sealed = read(obs / 'OBSERVATION_SEAL.json')
            if not sealed['complete'] or sealed['input_sha256'] != sha(valid_input):
                raise ValueError('Completed observation aggregate changed')
        else:
            observe_valid(valid_input, obs, plan['old_run'], ledger, plan['models'])
        done, jobs = {}, []
        for index, frame in enumerate(valid):
            folder, op = out / 'frames' / f'{index:04d}', obs / 'frames' / f'{index:04d}'
            if (folder / 'FRAME_SEAL.json').exists():
                done[frame['id']] = validate_solve(frame, op, folder, V)
            else:
                archived = archive_incomplete(folder, 'FRAME_SEAL.json', attempted / 'solves')
                if archived:
                    append(ledger, dict(stage='PRESERVE_INCOMPLETE_SOLVE', id=frame['id'], archive=archived,
                           interrupted_partial_cost='UNKNOWN_NOT_ZERO unless existing attempt receipt tracks it'))
                jobs.append((frame, str(op), str(folder), plan['old_run']))
        failures = []
        # spawn avoids inheriting the parent's completed CUDA context.
        with ProcessPoolExecutor(max_workers=plan['workers'],
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(solve_worker, *job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    receipt = future.result()
                except BaseException as exc:
                    receipt = dict(id=job[0]['id'], complete=False, status='WORKER_PROCESS_FAILURE',
                                   error=repr(exc), partial_CPU_and_operations='UNKNOWN_NOT_ZERO')
                append(ledger, dict(stage='SOLVER_ATTEMPT', backbone=name, invocation=invocation, **receipt))
                if receipt['complete']:
                    done[receipt['id']] = validate_solve(job[0], job[1], job[2], V)
                else:
                    failures.append(receipt)
                print(canonical(dict(backbone=name, stage='D1_ORIGINAL_E', completed=len(done),
                                     total=len(valid), id=receipt['id'], status=receipt['status'])), flush=True)
        if failures:
            raise RuntimeError('Technical failures retained; no completion seal: ' + name)
    else:
        done = {}
    feature_map = {}
    row_bindings, operation_totals = {}, Counter()
    for index, frame in enumerate(valid):
        folder = out / 'frames' / f'{index:04d}'
        feature_map[frame['id']] = rows(folder / 'FEATURES.jsonl.gz')[0]
        row_bindings[frame['id']] = dict(kind='EXECUTED_ORIGINAL_D1',
            frame_seal_sha256=sha(folder / 'FRAME_SEAL.json'),
            observation_seal_sha256=sha(obs / 'frames' / f'{index:04d}' / 'OBSERVATION_SEAL.json'))
        operation_totals.update(done[frame['id']]['operations'])
    for frame in frames:
        if not frame['initial_pose']['available']:
            feature_map[frame['id']] = unavailable_feature(frame)
            row_bindings[frame['id']] = dict(kind='UPSTREAM_UNAVAILABLE_NO_CALLS',
                input_row_sha256=hashlib.sha256(canonical(frame).encode()).hexdigest())
    save_rows(out / 'HEAD_FEATURES.jsonl.gz', [feature_map[f['id']] for f in frames])
    seal = dict(schema='D1_THREE_BACKBONE_FEATURE_SEAL_V1', complete=True, rows=238,
                backbone=name, ordered_ids=[f['id'] for f in frames],
                head_features_sha256=sha(out / 'HEAD_FEATURES.jsonl.gz'),
                input_sha256=sha(spec['inputs']), upstream_seal_sha256=sha(spec['upstream_seal']),
                execution_plan_sha256=sha(Path(plan['run']) / 'EXECUTION_PLAN.json'),
                row_bindings=row_bindings, source_sha256=plan['sources'],
                operations_completed_rows=dict(operation_totals),
                e0_independently_executed=True, e0_execution_scope='available upstream rows only',
                rows_with_executed_e0=len(valid), upstream_unavailable_rows=238 - len(valid),
                all_rows_kept=True, previous_YOLO_observation_or_stage_reuse=False,
                new_B_or_A_candidates=False, history_only_union=False,
                H1_prediction_or_fit_or_scoring_calls=0, reference_reads=0,
                numeric_original_CAL_preserved=True, transfer_generalization_unproven=True)
    if pilot_import is not None:
        seal['same_backbone_technical_pilot_import_sha256'] = sha(root / 'EXACT_TECHNICAL_PILOT_IMPORT.json')
    write(out / 'FEATURE_SEAL.json', seal)
    return seal


def validate_complete(root, completion):
    for path, digest in completion['artifacts'].items():
        if sha(Path(root) / path) != digest:
            raise ValueError('Completed controller output changed: ' + path)


def validate_arm_closure(plan, name):
    B, _, V = producer_modules()
    spec = plan['arms'][name]
    frames = rows(spec['inputs'])
    root = Path(plan['run']) / name
    seal = read(root / 'candidates/FEATURE_SEAL.json')
    if not seal.get('complete') or seal['input_sha256'] != sha(spec['inputs']) or \
            seal['upstream_seal_sha256'] != sha(spec['upstream_seal']) or \
            seal['execution_plan_sha256'] != sha(Path(plan['run']) / 'EXECUTION_PLAN.json'):
        raise ValueError('Completed transfer feature parents changed')
    if sha(root / 'candidates/HEAD_FEATURES.jsonl.gz') != seal['head_features_sha256']:
        raise ValueError('Completed transfer feature bytes changed')
    valid = [f for f in frames if f['initial_pose']['available']]
    for index, frame in enumerate(valid):
        op = root / 'observations/frames' / f'{index:04d}'
        folder = root / 'candidates/frames' / f'{index:04d}'
        validate_observation(frame, op, B)
        validate_solve(frame, op, folder, V)
        bound = seal['row_bindings'][frame['id']]
        if bound['frame_seal_sha256'] != sha(folder / 'FRAME_SEAL.json') or \
                bound['observation_seal_sha256'] != sha(op / 'OBSERVATION_SEAL.json'):
            raise ValueError('Completed transfer row ancestry changed')


def pilot(plan_path, backbone, *, root_GO=False):
    """One fixed common-available CAL row, separate from production caches."""
    if not root_GO:
        raise ValueError('Explicit root technical CAL execution GO required')
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    guard(plan)
    if backbone not in plan['arms']:
        raise ValueError('Prepared backbone required')
    populations = {name: rows(spec['inputs']) for name, spec in plan['arms'].items()}
    common = set.intersection(*({f['id'] for f in frames
                                 if f['partition'] == 'CAL' and f['initial_pose']['available']}
                                for frames in populations.values()))
    frame = next((f for f in populations[backbone] if f['id'] in common), None)
    if frame is None:
        raise ValueError('No common available CAL input; technical pilot unavailable')
    root = Path(plan['run']) / 'technical' / 'cal_pilot_v1' / backbone
    root.mkdir(parents=True, exist_ok=True)
    with (Path(plan['run']) / 'OWNER.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / 'PILOT_LOCK.json', dict(schema='D1_TRANSFER_TECHNICAL_CAL_LOCK_V1',
              id=frame['id'], backbone=backbone, plan_sha256=sha(plan_path),
              input_row_sha256=hashlib.sha256(canonical(frame).encode()).hexdigest(),
              selection='first canonical input-order CAL identity available in every prepared backbone; no metric read',
              production_cache_reuse='only same-backbone exact input/source/mask/seal-verified row',
              new_fit_or_H1_forward_or_scoring=False))
        if (root / 'PILOT_RECEIPT.json').exists():
            receipt = read(root / 'PILOT_RECEIPT.json')
            if receipt['plan_sha256'] != sha(plan_path) or receipt['id'] != frame['id']:
                raise ValueError('Completed technical pilot binding changed')
            validate_complete(root, receipt)
            B, _, V = producer_modules()
            validate_observation(frame, root / 'observations/frames/0000', B)
            validate_solve(frame, root / 'observations/frames/0000', root / 'candidates/frames/0000', V)
            return receipt
        B, _, V = producer_modules()
        inputs, obs, out = root / 'ONE_INPUT.jsonl.gz', root / 'observations', root / 'candidates'
        save_rows(inputs, [frame])
        ledger = root / 'ACTUAL_OPERATION_LEDGER.jsonl'
        op, folder = obs / 'frames/0000', out / 'frames/0000'
        if (op / 'OBSERVATION_SEAL.json').exists():
            validate_observation(frame, op, B)
        else:
            archive_incomplete(op, 'OBSERVATION_SEAL.json', root / 'prior_incomplete_observation_attempts')
        if not (obs / 'OBSERVATION_SEAL.json').exists():
            observe_valid(inputs, obs, plan['old_run'], ledger, plan['models'])
        if (folder / 'FRAME_SEAL.json').exists():
            validate_solve(frame, op, folder, V)
        else:
            archive_incomplete(folder, 'FRAME_SEAL.json', root / 'prior_incomplete_solver_attempts')
            attempt = solve_worker(frame, str(op), str(folder), plan['old_run'])
            append(ledger, dict(stage='SOLVER_ATTEMPT', backbone=backbone, **attempt))
            if not attempt['complete']:
                raise RuntimeError('Technical CAL failure retained')
        guard(plan)
        artifacts = {str(path.relative_to(root)): sha(path) for path in (
            root / 'PILOT_LOCK.json', inputs, obs / 'OBSERVATION_SEAL.json',
            op / 'OBSERVATION_SEAL.json', folder / 'FRAME_SEAL.json',
            folder / 'E_RUN.jsonl.gz', folder / 'FEATURES.jsonl.gz',
            folder / 'BOUNDARY_BUNDLE.jsonl.gz', ledger)}
        receipt = dict(schema='D1_TRANSFER_TECHNICAL_CAL_RECEIPT_V1', complete=True,
              id=frame['id'], backbone=backbone, plan_sha256=sha(plan_path), artifacts=artifacts,
              observation_resource=read(op / 'OBSERVATION_SEAL.json')['resource'],
              solver_resource=read(folder / 'FRAME_SEAL.json')['resource'],
              operations=read(folder / 'FRAME_SEAL.json')['operations'],
              before_any_new_reference_decode=True, new_training_or_H1_or_scoring_calls=0,
              extra_technical_observation_and_solver_row=1, production_outputs_replaced=False)
        write(root / 'PILOT_RECEIPT.json', receipt)
        return receipt


def run(plan_path, *, root_GO=False):
    if not root_GO:
        raise ValueError('Explicit root execution GO required; prepare does not run science')
    plan_path = Path(plan_path).resolve()
    plan = read(plan_path)
    if plan_path != Path(plan['run']) / 'EXECUTION_PLAN.json':
        raise ValueError('Canonical prepared plan path required')
    guard(plan)
    root = Path(plan['run'])
    with (root / 'OWNER.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete_path = root / 'EXECUTION_COMPLETION.json'
        if complete_path.exists():
            completion = read(complete_path)
            if completion['execution_plan_sha256'] != sha(plan_path):
                raise ValueError('Completed execution plan changed')
            validate_complete(root, completion)
            for name in plan['arms']:
                validate_arm_closure(plan, name)
            return completion
        invocation = 'invocation_' + uuid.uuid4().hex
        tick, cpu = time.monotonic(), time.process_time()
        before_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        write(root / 'attempts' / invocation / 'START.json', dict(started_utc=utc(),
            plan_sha256=sha(plan_path), actual_argv=list(os.sys.argv),
            worker_limit=plan['workers'], stage_overlap=False, no_fits_or_scores=True))
        failure, results = None, {}
        try:
            for name in plan['arms']:
                results[name] = run_arm(plan, name, invocation)
            guard(plan)
        except BaseException as exc:
            failure = dict(error=repr(exc), traceback=traceback.format_exc())
        after_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        receipt = dict(schema='D1_TRANSFER_INVOCATION_RECEIPT_V1', complete=failure is None,
            invocation=invocation, ended_utc=utc(), wall_seconds=time.monotonic() - tick,
            parent_CPU_seconds=time.process_time() - cpu,
            exited_child_CPU_seconds=after_child - before_child,
            CPU_scope='parent process plus exited children, disjoint from row timers; do not sum nested scopes',
            pre_invocation_guard_and_import_cost='outside this timer; use external service/command capture',
            receipt_and_completion_finalization_cost='UNKNOWN_NOT_ZERO; outside internal timer',
            failure=failure, completed_backbones=list(results),
            interrupted_unsealed_cost='UNKNOWN_NOT_ZERO where no durable attempt record',
            plan_sha256=sha(plan_path), source_sha256=sha(__file__),
            no_H1_forward_or_training_or_scoring=True)
        write(root / 'attempts' / invocation / 'RECEIPT.json', receipt)
        if failure:
            raise RuntimeError(failure['error'])
        artifacts = {}
        for name in results:
            for filename in ('HEAD_FEATURES.jsonl.gz', 'FEATURE_SEAL.json'):
                rel = name + '/candidates/' + filename
                artifacts[rel] = sha(root / rel)
        completion = dict(schema='D1_TRANSFER_EXECUTION_COMPLETION_V1', complete=True,
            execution_plan_sha256=sha(plan_path), artifacts=artifacts,
            backbone_rows={name: 238 for name in results}, PRIMARY192_and_CAL46_retained=True,
            invocation_receipt_sha256=sha(root / 'attempts' / invocation / 'RECEIPT.json'),
            H1_prediction_or_training_or_scoring_calls=0, source_sha256=sha(__file__),
            whole_campaign_complete=False)
        write(complete_path, completion)
        return completion


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--repo', required=True)
    prep.add_argument('--run', required=True)
    prep.add_argument('--old-run', required=True)
    prep.add_argument('--inputs', required=True, help='JSON map backbone -> sanitized238 path')
    prep.add_argument('--upstream-seals', required=True, help='JSON map backbone -> upstream seal')
    prep.add_argument('--workers', type=int, default=4)
    execute = sub.add_parser('run')
    execute.add_argument('--plan', required=True)
    execute.add_argument('--root-go', action='store_true')
    smoke = sub.add_parser('pilot')
    smoke.add_argument('--plan', required=True)
    smoke.add_argument('--backbone', choices=('yolo', 'dope', 'resnet18'), required=True)
    smoke.add_argument('--root-go', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare(args.repo, args.run, read(args.inputs), args.old_run,
                         workers=args.workers, upstream_seals=read(args.upstream_seals))
    elif args.mode == 'pilot':
        result = pilot(args.plan, args.backbone, root_GO=args.root_go)
    else:
        result = run(args.plan, root_GO=args.root_go)
    print(canonical(result), flush=True)


if __name__ == '__main__':
    main()
