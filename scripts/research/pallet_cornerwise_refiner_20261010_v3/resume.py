"""One explicit data-only continuation of the interrupted frozen245 inference.

The first160 compressed rows are copied verbatim, not recomputed. Only the
remaining85 RGB frames enter the unchanged Pipeline. This module never scores
poses, modifies a policy, or automatically retries an interrupted continuation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from . import common as C

PREFIX = 160
REMAINDER = 85
PREFIX_NAMES = ('INFERENCE_STARTED.json', 'INFERENCE_RECEIPT.json',
                'INTERRUPTED_GEOMETRY.jsonl.gz',
                'INTERRUPTED_FIXED_GEOMETRY.jsonl.gz',
                'INTERRUPTED_OBSERVATIONS.jsonl.gz')
FINAL_NAMES = ('OBSERVATIONS.jsonl.gz', 'GEOMETRY_SEALED.jsonl.gz',
               'FIXED_GEOMETRY_SEALED.jsonl.gz', 'BASE_N3_PARITY.json',
               'GEOMETRY_SEAL.json')
ATTEMPT_NAMES = ('RESUME_STARTED.json', 'RESUME_RECEIPT.json',
                 'RESUME_INTERRUPTED_GEOMETRY.jsonl.gz',
                 'RESUME_INTERRUPTED_FIXED_GEOMETRY.jsonl.gz',
                 'RESUME_INTERRUPTED_OBSERVATIONS.jsonl.gz')
SCORE_KEYS = ('pose', 'corner', 'mask_audit', 'baseline_pose',
              'baseline_corner', 'evaluation_reference')


def utc():
    return datetime.now(timezone.utc).isoformat()


def prefix_inputs(args):
    folder = Path(args.prefix)
    paths = {name: folder / name for name in PREFIX_NAMES}
    paths.update(original_protocol=args.protocol, continuation_code=Path(__file__),
                 resource_probe=C.REPO / 'scripts/research/'
                     'pallet_corner_mechanism_audit_20261010_v1/deployment_smoke.py')
    return {name: C.binding(path) for name, path in paths.items()}


def load_prefix(args, frames):
    folder = Path(args.prefix)
    started = C.read(folder / 'INFERENCE_STARTED.json')
    receipt = C.read(folder / 'INFERENCE_RECEIPT.json')
    C.require(started['protocol'] == C.binding(args.protocol) and
              started['configured_frames'] == 245 and
              started['configured_rows'] == 980 and
              started['configured_fixed_rows'] == 490 and
              started['no_automatic_retry'] is True, 'original start contract differs')
    C.require(receipt['complete'] is False and receipt['actual_complete_frames'] == PREFIX and
              receipt['method_rows'] == PREFIX * 4 and receipt['fixed_rows'] == PREFIX * 2 and
              receipt['cleanup_error'] is None and receipt['GT_access_during_inference'] is False and
              receipt['new_training_updates'] == receipt['new_RGB'] == 0 and
              receipt['no_automatic_retry'] is True, 'original interrupted receipt differs')
    expected = dict(detector_calls=PREFIX, N3_route_calls=PREFIX,
                    initial_pose_calls=PREFIX, base_control_pose_calls=PREFIX,
                    ROLE_head_calls=PREFIX, feature_initial_pose_calls=PREFIX,
                    final_pose_paths=PREFIX * 4)
    C.require(all(receipt['actual_calls'].get(k) == v for k, v in expected.items()),
              'prefix actual stage counts do not establish160 completed captures')
    C.require(receipt['actual_model_forwards'] ==
              dict(detector=PREFIX + 1, N3=PREFIX, ROLE=PREFIX), 'prefix actual model counts differ')
    C.require(set(receipt['actual_OpenCV_entry_calls']) <=
              {'cornerSubPix', 'solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM'} and
              all(isinstance(n, int) and n >= 0 for n in receipt['actual_OpenCV_entry_calls'].values()),
              'prefix primitive counters invalid')
    geometry = list(C.rows(folder / 'INTERRUPTED_GEOMETRY.jsonl.gz'))
    fixed = list(C.rows(folder / 'INTERRUPTED_FIXED_GEOMETRY.jsonl.gz'))
    observations = list(C.rows(folder / 'INTERRUPTED_OBSERVATIONS.jsonl.gz'))
    ids = [f['id'] for f in frames[:PREFIX]]
    for values, methods in ((geometry, C.METHODS), (fixed, ('BASE', 'N3_SUBPIX'))):
        C.require([(r['id'], r['method']) for r in values] ==
                  [(fid, m) for fid in ids for m in methods], 'prefix row population/order differs')
        C.require(not any(k in r for r in values for k in SCORE_KEYS),
                  'prefix geometry contains posthoc scoring data')
    C.require([r['id'] for r in observations] == ids and
              all(r['GT_input'] is False and r['session'] == frames[i]['session']
                  for i, r in enumerate(observations)), 'prefix observation population differs')
    C.require(len(frames) == 245 and len(frames[PREFIX:]) == REMAINDER,
              'continuation must contain exactly85 remaining frozen frames')
    C.require(not (folder / 'GEOMETRY_SEAL.json').exists(), 'prefix already has a complete seal')
    return receipt, geometry, fixed, observations


def freeze(args):
    original = C.verify_protocol(args)
    C.protect(args)
    frames = C.cohort_frames(args)
    load_prefix(args, frames)
    C.require(Path(args.resume_protocol).resolve() ==
              C.output_path(args, 'RESUME_PROTOCOL.json').resolve(), 'resume protocol output differs')
    protocol = dict(schema='cornerwise_data_only_continuation_protocol_v3',
        frozen_before_remaining_inference=True, original_protocol=C.binding(args.protocol),
        original_fixed_inputs=original['inputs'], continuation_inputs=prefix_inputs(args),
        prefix_frames=PREFIX, remaining_frames=REMAINDER, complete_frames=245,
        prefix_ids=[f['id'] for f in frames[:PREFIX]],
        remaining_ids=[f['id'] for f in frames[PREFIX:]],
        methods=list(C.METHODS), method_policy_unchanged=True,
        new_training_updates=0, new_RGB=0, GT_tuning=False, scoring_calls=0,
        no_automatic_retry=True, only_remaining_images_enter_capture=True,
        prefix_storage='copy entire original gzip bytes verbatim as first member; append85-frame member',
        original_failed_resource_snapshot=None,
        original_failure_trigger='resource guard reported by supervising execution; exact failed snapshot unavailable',
        prefix_original_bank_records=None, prefix_original_resource_snapshots=None,
        prefix_original_parity_checklist=None,
        reconstructed_prefix_parity='stored GT-free observation Base/N3 coordinates only; diagnostic after load',
        expected_total_model_forwards=dict(detector=247, N3=245, ROLE=245),
        constructor_detector_forwards=dict(original=1, continuation=1),
        resource_policy='unchanged quiet and temperature_under_80; persist probe before asserting')
    C.write_new(C.output_path(args, 'RESUME_PROTOCOL.json'), protocol)
    print('CORNERWISE_RESUME_FROZEN', C.sha(args.resume_protocol), flush=True)


def prerequisites(args, check_destinations=True):
    original = C.verify_protocol(args)
    preserved = C.protect(args)
    frames = C.cohort_frames(args)
    p = C.read(args.resume_protocol)
    C.require(p['original_protocol'] == C.binding(args.protocol) and
              p['original_fixed_inputs'] == original['inputs'] and
              p['continuation_inputs'] == prefix_inputs(args), 'continuation input/code drift')
    C.require(p['prefix_frames'] == PREFIX and p['remaining_frames'] == REMAINDER and
              p['complete_frames'] == 245 and p['methods'] == list(C.METHODS) and
              p['method_policy_unchanged'] is True and p['no_automatic_retry'] is True and
              p['new_training_updates'] == p['new_RGB'] == p['scoring_calls'] == 0 and
              p['GT_tuning'] is False, 'continuation scope/policy differs')
    C.require(p['prefix_ids'] == [f['id'] for f in frames[:PREFIX]] and
              p['remaining_ids'] == [f['id'] for f in frames[PREFIX:]], 'continuation ID order differs')
    receipt, geometry, fixed, observations = load_prefix(args, frames)
    if check_destinations:
        for name in FINAL_NAMES + ATTEMPT_NAMES:
            C.output_path(args, name)
        C.output_path(args, 'RESUME_RESOURCE_000.json')
    return p, frames, receipt, geometry, fixed, observations, preserved


def record_quiet(args, probes, phase, completed):
    from ..pallet_corner_mechanism_audit_20261010_v1.deployment_smoke import quiet
    # The exact unmodified probe is written before its unchanged gate is applied.
    value = dict(phase=phase, complete_remaining_frames=completed,
                 complete_total_frames=PREFIX + completed, utc=utc(), **quiet())
    name = 'RESUME_RESOURCE_%03d.json' % len(probes)
    C.write_new(C.output_path(args, name), value)
    probes.append(dict(snapshot=C.binding(Path(args.output) / name), **value))
    C.require(value['quiet'] and value['temperature_under_80'],
              'pending competing workload/thermal guard; exact snapshot preserved')
    return value


def append_preserved(prefix, destination, suffix):
    """Preserve the complete compressed prefix byte sequence, including headers."""
    prefix = Path(prefix)
    destination = Path(destination)
    before = C.binding(prefix)
    C.require(not destination.exists() and not destination.is_symlink(), 'merged output exists')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with prefix.open('rb') as source, destination.open('xb') as raw:
        shutil.copyfileobj(source, raw, length=8 * 1024 * 1024)
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
            for value in suffix:
                stream.write((json.dumps(C.finite(value), ensure_ascii=False,
                    separators=(',', ':'), allow_nan=False) + '\n').encode())
        raw.flush()
        os.fsync(raw.fileno())
    C.bound(prefix, before, 'untouched interrupted prefix')
    h = hashlib.sha256()
    remaining = before['bytes']
    with destination.open('rb') as stream:
        while remaining:
            part = stream.read(min(8 * 1024 * 1024, remaining))
            C.require(bool(part), 'merged prefix truncated')
            h.update(part)
            remaining -= len(part)
    C.require(h.hexdigest() == before['sha256'], 'compressed prefix bytes were not preserved')
    return dict(prefix=before, merged=C.binding(destination),
                preserved_compressed_prefix_bytes=before['bytes'],
                preserved_compressed_prefix_sha256=h.hexdigest(), exact_prefix_copy=True,
                suffix_rows=len(suffix), gzip_members=2)


def prefix_parity(pipeline, frames, observations):
    import numpy as np
    values = []
    for frame, row in zip(frames[:PREFIX], observations):
        base = pipeline.runtime._point_parity(row['original_base_points'], frame['points']['BASE'], 1e-7, frame['id'])
        n3 = pipeline.runtime._point_parity(row['native_N3_points'], frame['points']['N3_SUBPIX'], 1e-7, frame['id'])
        C.require(np.array_equal(np.asarray(row['original_base_points'][8], float),
                                 np.asarray(row['native_N3_points'][8], float), equal_nan=True),
                  'stored prefix center differs')
        values.append(dict(id=frame['id'], BASE=base, N3_SUBPIX=n3,
            center_preserved=True, stored_parity_not_prediction_input=True,
            provenance='reconstructed_saved_prefix_coordinate_parity_only',
            metadata_preserved=None, original_metadata_parity_checklist=None,
            original_parity_checklist_not_persisted=True))
    return values


def resume(args):
    p, frames, prefix_receipt, prefix_geometry, prefix_fixed, prefix_observations, before = prerequisites(args)
    # Claim once before the first resource probe/model import. Even a zero-forward
    # failure retains this claim and cannot be automatically restarted.
    C.write_new(C.output_path(args, 'RESUME_STARTED.json'), dict(
        schema='cornerwise_data_only_continuation_started_v3', utc=utc(),
        original_protocol=C.binding(args.protocol), resume_protocol=C.binding(args.resume_protocol),
        configured_remaining_frames=REMAINDER, configured_remaining_rows=REMAINDER * 4,
        configured_remaining_fixed_rows=REMAINDER * 2, no_automatic_retry=True,
        prefix_inference_receipt=p['continuation_inputs']['INFERENCE_RECEIPT.json']))
    start = time.monotonic()
    geometry = []; controls = []; observations = []; parity = []; banks = []; probes = []
    pipeline = None; primitive = None; head_hook = None; head_actual = Counter()
    calls = {}; model_calls = {}; cv_calls = {}; model_bindings = None
    complete = False; failure = None; cleanup_error = None; merge_witnesses = {}
    reconstructed = []; preserved = before
    try:
        record_quiet(args, probes, 'before_models', 0)
        import cv2
        import numpy as np
        from .pipeline import Pipeline, preserve_prediction
        with C.inference_canary():
            pipeline = Pipeline(args)
        head_hook = pipeline.head.register_forward_pre_hook(lambda *_: head_actual.update(ROLE=1))
        model_bindings = pipeline.bindings
        reconstructed = prefix_parity(pipeline, frames, prefix_observations)
        with C.primitive_counter() as primitive:
            for frame in frames[PREFIX:]:
                path = Path(args.source_root) / frame['image']
                C.require(C.sha(path) == frame['image_sha256'], 'original RGB SHA changed')
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                C.require(image is not None and list(image.shape[:2]) == frame['raw_hw'], 'RGB decode/shape differs')
                meta = {key: frame[key] for key in ('id', 'session', 'object_type')}
                with C.inference_canary(), pipeline.torch.no_grad():
                    captured = pipeline.capture(image, frame['K'], frame['xyz'], meta, need_base_pose=True)
                    preserve_prediction(captured['raw'], captured['prediction'])
                    result, ledger = pipeline.outcomes(captured)
                    fixed = [pipeline.fixed_result(captured, arm) for arm in ('BASE', 'N3_SUBPIX')]
                # Authority comparisons follow fresh capture/selection; they are
                # never an observation, pose prior, head input or candidate input.
                idx = captured['raw']['selected_index']
                metadata = None if idx is None else {k: v for k, v in captured['raw']['candidates'][idx].items() if k != 'keypoints_xy'}
                C.require(idx == frame['selected_index'] and
                          C.finite(metadata) == C.finite(frame['candidate_metadata']), 'detector metadata differs')
                base = pipeline.runtime._point_parity(captured['original_base_points'], frame['points']['BASE'], 1e-7, frame['id'])
                n3 = pipeline.runtime._point_parity(captured['native_N3_points'], frame['points']['N3_SUBPIX'], 1e-7, frame['id'])
                C.require(np.array_equal(captured['native_N3_points'][8], captured['original_base_points'][8], equal_nan=True), 'center changed')
                parity.append(dict(id=frame['id'], BASE=base, N3_SUBPIX=n3,
                    metadata_preserved=True, center_preserved=True, stored_parity_not_prediction_input=True,
                    provenance='fresh_remaining_capture_post_inference_parity'))
                geometry.extend(result); controls.extend(fixed)
                banks.append(dict(id=frame['id'], **ledger))
                observations.append(dict(id=frame['id'], session=frame['session'], GT_input=False,
                    original_base_points=captured['original_base_points'], native_N3_points=captured['native_N3_points'],
                    initial_N3_pose=captured['initial_pose'], predicted_N3_hidden=captured['hidden'], **captured['observation']))
                if len(observations) % 32 == 0:
                    record_quiet(args, probes, 'after_remaining_frame', len(observations))
                    print('CORNERWISE_RESUME_GEOMETRY', PREFIX + len(observations), 245, flush=True)
        calls = dict(pipeline.counts)
        model_calls = dict(detector=pipeline.models.detector_forwards, N3=pipeline.models.n3_forwards, ROLE=head_actual['ROLE'])
        cv_calls = dict(primitive)
        C.require(len(observations) == REMAINDER and len(geometry) == REMAINDER * 4 and
                  len(controls) == REMAINDER * 2, 'remaining population incomplete')
        C.require(all(calls.get(k) == REMAINDER for k in
                  ('detector_calls', 'N3_route_calls', 'initial_pose_calls', 'base_control_pose_calls',
                   'ROLE_head_calls', 'feature_initial_pose_calls')) and calls['final_pose_paths'] == REMAINDER * 4,
                  'actual remaining stage counts differ')
        C.require(model_calls == dict(detector=REMAINDER + 1, N3=REMAINDER, ROLE=REMAINDER),
                  'actual remaining model forwards differ')
        record_quiet(args, probes, 'after_remaining_geometry', len(observations))
        prerequisites(args, check_destinations=False)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), phase='remaining_inference_or_preseal')
    finally:
        if primitive is not None:
            cv_calls = dict(primitive)
        if pipeline is not None:
            calls = dict(pipeline.counts)
            model_calls = dict(detector=pipeline.models.detector_forwards, N3=pipeline.models.n3_forwards, ROLE=head_actual['ROLE'])
            if head_hook is not None:
                try:
                    head_hook.remove()
                except BaseException as error:
                    cleanup_error = dict(type=type(error).__name__, message=str(error), phase='head_hook_remove')
            try:
                pipeline.close()
            except BaseException as error:
                close_error = dict(type=type(error).__name__, message=str(error), phase='pipeline_close')
                cleanup_error = close_error if cleanup_error is None else dict(first=cleanup_error, second=close_error)
    if failure is None and cleanup_error is None:
        try:
            C.verify_protocol(args); preserved = C.protect(args)
            # Recheck every prefix binding immediately before the byte-copy merge.
            C.require(prefix_inputs(args) == p['continuation_inputs'], 'prefix changed before merge')
            for source_name, name, suffix in (
                    ('INTERRUPTED_OBSERVATIONS.jsonl.gz', 'OBSERVATIONS.jsonl.gz', observations),
                    ('INTERRUPTED_GEOMETRY.jsonl.gz', 'GEOMETRY_SEALED.jsonl.gz', geometry),
                    ('INTERRUPTED_FIXED_GEOMETRY.jsonl.gz', 'FIXED_GEOMETRY_SEALED.jsonl.gz', controls)):
                merge_witnesses[name] = append_preserved(Path(args.prefix) / source_name,
                                                        C.output_path(args, name), suffix)
            for name, expected in (('OBSERVATIONS.jsonl.gz', prefix_observations + C.finite(observations)),
                                   ('GEOMETRY_SEALED.jsonl.gz', prefix_geometry + C.finite(geometry)),
                                   ('FIXED_GEOMETRY_SEALED.jsonl.gz', prefix_fixed + C.finite(controls))):
                C.require(list(C.rows(Path(args.output) / name)) == expected, 'merged decompressed rows differ')
            C.write_new(C.output_path(args, 'BASE_N3_PARITY.json'), dict(
                passed=True, rows=reconstructed + parity, rtol=0, atol=1e-7,
                prefix_rows=PREFIX, freshly_rechecked_remaining_rows=REMAINDER,
                prefix_metadata_parity_checklist=None, prefix_parity_scope='saved coordinate replay only'))
            total_calls = dict(Counter(prefix_receipt['actual_calls']) + Counter(calls))
            total_model = dict(Counter(prefix_receipt['actual_model_forwards']) + Counter(model_calls))
            total_cv = dict(Counter(prefix_receipt['actual_OpenCV_entry_calls']) + Counter(cv_calls))
            C.require(total_model == dict(detector=247, N3=245, ROLE=245), 'composed actual model counts differ')
            seal = dict(schema='cornerwise_geometry_seal_v3', complete=True, GT_read_allowed=False,
                frames=245, rows=980, fixed_rows=490, methods=list(C.METHODS), protocol=C.binding(args.protocol),
                resume_protocol=C.binding(args.resume_protocol), resume_code=C.binding(Path(__file__)),
                original_inference_receipt=p['continuation_inputs']['INFERENCE_RECEIPT.json'],
                observations=C.binding(Path(args.output) / 'OBSERVATIONS.jsonl.gz'),
                geometry=C.binding(Path(args.output) / 'GEOMETRY_SEALED.jsonl.gz'),
                fixed_geometry=C.binding(Path(args.output) / 'FIXED_GEOMETRY_SEALED.jsonl.gz'),
                parity=C.binding(Path(args.output) / 'BASE_N3_PARITY.json'),
                actual_calls=total_calls, actual_model_forwards=total_model, actual_OpenCV_entry_calls=total_cv,
                prefix_actual_calls=prefix_receipt['actual_calls'], remaining_actual_calls=calls,
                prefix_actual_model_forwards=prefix_receipt['actual_model_forwards'], remaining_actual_model_forwards=model_calls,
                prefix_actual_OpenCV_entry_calls=prefix_receipt['actual_OpenCV_entry_calls'], remaining_actual_OpenCV_entry_calls=cv_calls,
                banks=banks, banks_population_scope='fresh remaining85 only; original160 per-frame records not persisted',
                prefix_bank_records=None, prefix_original_parity_checklist=None,
                prefix_resource_snapshots=None, original_failed_resource_snapshot=None,
                model_bindings=model_bindings, resource_snapshots=probes,
                prefix_copy_witnesses=merge_witnesses, protection=preserved,
                remaining_wall_seconds=time.monotonic() - start,
                original_prefix_wall_seconds=prefix_receipt['wall_seconds'],
                wall_seconds=prefix_receipt['wall_seconds'] + time.monotonic() - start,
                wall_scope='sum of two inference attempts; downtime/freeze/scoring excluded; not runtime benchmark',
                runtime_benchmark=False, new_training_updates=0, new_RGB=0,
                GT_access_during_inference=False, no_automatic_retry=True)
            C.write_new(C.output_path(args, 'GEOMETRY_SEAL.json'), seal)
            complete = True
        except BaseException as error:
            failure = dict(type=type(error).__name__, message=str(error), phase='merge_or_seal')
    if not complete:
        for name, values in (('RESUME_INTERRUPTED_GEOMETRY.jsonl.gz', geometry),
                             ('RESUME_INTERRUPTED_FIXED_GEOMETRY.jsonl.gz', controls),
                             ('RESUME_INTERRUPTED_OBSERVATIONS.jsonl.gz', observations)):
            C.save_rows(C.output_path(args, name), values)
    receipt = dict(schema='cornerwise_data_only_continuation_receipt_v3', complete=complete,
        original_protocol=C.binding(args.protocol), resume_protocol=C.binding(args.resume_protocol),
        continuation_inputs=p['continuation_inputs'], prefix_complete_frames=PREFIX,
        actual_complete_remaining_frames=len(observations), actual_complete_frames=PREFIX + len(observations),
        method_rows=PREFIX * 4 + len(geometry), fixed_rows=PREFIX * 2 + len(controls),
        actual_calls=dict(Counter(prefix_receipt['actual_calls']) + Counter(calls)),
        actual_model_forwards=dict(Counter(prefix_receipt['actual_model_forwards']) + Counter(model_calls)),
        actual_OpenCV_entry_calls=dict(Counter(prefix_receipt['actual_OpenCV_entry_calls']) + Counter(cv_calls)),
        remaining_actual_calls=calls, remaining_actual_model_forwards=model_calls,
        remaining_actual_OpenCV_entry_calls=cv_calls,
        constructor_failure_counters_unavailable=pipeline is None,
        resource_snapshots=probes, failure=failure, cleanup_error=cleanup_error,
        prefix_copy_witnesses=merge_witnesses, prefix_original_bank_records=None,
        prefix_original_resource_snapshots=None, prefix_original_parity_checklist=None,
        original_failed_resource_snapshot=None, new_training_updates=0, new_RGB=0,
        GT_access_during_inference=False, scoring_calls=0, no_automatic_retry=True,
        remaining_wall_seconds=time.monotonic() - start,
        original_prefix_wall_seconds=prefix_receipt['wall_seconds'], protection=preserved)
    if complete:
        receipt['geometry_seal'] = C.binding(Path(args.output) / 'GEOMETRY_SEAL.json')
    C.write_new(C.output_path(args, 'RESUME_RECEIPT.json'), receipt)
    C.require(complete, 'continuation incomplete; preserve attempt and do not automatically retry')
    print('CORNERWISE_RESUME_SEALED', 980, 490, 'before_GT', flush=True)


def main():
    parser = C.parser(__doc__, ('freeze', 'preflight', 'resume'))
    parser.set_defaults(output=str(C.DOC))
    parser.add_argument('--prefix', default=str(C.DOC), help='read-only original interrupted files')
    parser.add_argument('--resume-protocol', default=str(C.DOC / 'RESUME_PROTOCOL.json'))
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze(args)
    elif args.stage == 'preflight':
        prerequisites(args)
        print('CORNERWISE_RESUME_PREFLIGHT_PASS_NO_MODELS', flush=True)
    else:
        resume(args)


if __name__ == '__main__':
    main()
