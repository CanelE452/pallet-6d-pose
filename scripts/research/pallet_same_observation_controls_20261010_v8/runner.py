"""Fixed original-C3 PnP diagnostics on complete, GT-free V7 observations.

This is an actual 735-solve diagnostic replay with immutable selected sparse
coordinates, not fresh image inference and not a latency benchmark. Freeze and
execution are explicit separate stages; no stage is run merely by importing.
"""
from __future__ import annotations

import builtins
from collections import Counter
from contextlib import contextmanager
import copy
import io
import os
from pathlib import Path
import sys
import time
import traceback

from . import common as C

STREAMS = dict(geometry='GEOMETRY_SEALED.jsonl.gz', ledgers='CONTROL_LEDGERS.jsonl.gz')
PRIMITIVES = ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'projectPoints', 'Rodrigues', 'cornerSubPix')
RUN_NAMES = ('CONTROL_STARTED.json', 'CONTROL_RECEIPT.json', 'GEOMETRY_SEAL.json',
    'GEOMETRY_SEALED.jsonl.gz', 'CONTROL_LEDGERS.jsonl.gz', 'INTERRUPTED_CONTROL_PACKET.json',
    'INTERRUPTED_GEOMETRY.jsonl.gz', 'INTERRUPTED_CONTROL_LEDGERS.jsonl.gz')


def CPU_authority(bindings):
    protocol = C.read(C.DOC / 'CPU_TEST_PROTOCOL.json')
    B = C.read(C.DOC / 'CONTROL_CHECKS_ATTEMPT_B.json')
    A = C.read(C.DOC / 'CONTROL_CHECKS.json')
    C.require(protocol['schema'] == 'pre_execution_synthetic_same_observation_CPU_contract_v8' and
        protocol['synthetic_only'] is True and protocol['test_groups_expected'] == 10,
        'fixed synthetic CPU protocol absent')
    C.require(B['complete'] is True and B['passed'] is True and B['tests_run'] == 10 and
        not B['failures'] and not B['errors'] and not B['forbidden_asset_reads'], 'successful CPU attempt B required')
    C.require(A['complete'] is True and A['passed'] is False and A['tests_run'] == 10,
              'preserve original failed CPU attempt A')
    for entry in protocol['code']:
        C.bound(C.REPO / entry['path'], entry, 'CPU-bound immutable code')
    for attempt, filename in ((A, 'CONTROL_CHECKS_STARTED.json'), (B, 'CONTROL_CHECKS_ATTEMPT_B_STARTED.json')):
        started = C.read(C.DOC / filename)
        C.require(started['code'] == attempt['code'], 'CPU attempt/start code differs')
        for key, filename in (('controls', 'controls.py'), ('tests', 'test_controls.py')):
            C.require(C.same_binding(attempt['code'][key], bindings[filename]), 'CPU-tested code differs: ' + filename)
        for key in ('model_calls', 'real_GT_reads', 'new_RGB'):
            C.require(attempt[key] == 0, 'CPU fixture unexpectedly accessed assets')
    return dict(successful_attempt=bindings['CONTROL_CHECKS_ATTEMPT_B.json'],
        failed_attempt_preserved=bindings['CONTROL_CHECKS.json'],
        failure_explanation=bindings['CONTROL_CHECKS_ATTEMPT_A_REASON.json'],
        protocol=bindings['CPU_TEST_PROTOCOL.json'],
        both_attempts_count_as_executed_work=True, numeric_policy_unchanged_between_attempts=True)


def freeze(args):
    for name in ('PARENT_POPULATION_CHECKS.json',):
        C.output_path(args, name)
    protocol_path = Path(args.protocol)
    C.require(protocol_path.resolve() == C.DOC / 'PROTOCOL.json', 'formal protocol must be new V8 DOC')
    protocol_args = copy.copy(args)
    protocol_args.output = str(C.DOC)
    C.output_path(protocol_args, 'PROTOCOL.json')
    bindings = C.input_bindings(args)  # Every input, including full raw bytes, bound before arithmetic.
    CPU = CPU_authority(bindings)
    before = C.protect(args)
    proof = C.population_proof(args, bindings)
    C.require(C.input_bindings(args) == bindings, 'parent/code changed during population verification')
    C.write_new(C.output_path(args, 'PARENT_POPULATION_CHECKS.json'), proof)
    C.write_new(protocol_path, dict(schema='fixed_same_observation_C3_protocol_v8',
        policy=C.POLICY, methods=list(C.METHODS), primary=C.PRIMARY, frames=245, rows=735,
        contrasts=[list(pair) for pair in C.CONTRASTS], contrast_direction='first minus second; negative is improvement',
        inputs=bindings, CPU_authority=CPU,
        postseal_only_inputs=list(C.POSTSEAL_ONLY_INPUTS),
        postseal_bindings_are_hashes_not_pre_fit_reference_decoding=True,
        parent_population_checks=C.binding(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'),
        protected_before=before, parent_completion=proof['completion'],
        source_root=str(Path(args.source_root).resolve()), baseline_root=str(Path(args.baseline_root).resolve()),
        new_accuracy_primary=False, evaluation_is_unseen=False, GT_tuning=False,
        accuracy_policy_changes=0, requested_original_C3_diagnostic=True,
        actual_new_geometry_executed_at_freeze=0, expected_execution=dict(frames=245, geometry_rows=735,
            ledger_rows=245, coordinate_banks=245, logical_pose_paths=735,
            existing_control_replay_pose_paths=245, new_diagnostic_pose_paths=490),
        output_schema=dict(geometry='full solve_controls normalized rows + original native OBS reference + Jacobian fit IDs',
            ledgers='id/session plus full solve_controls per-frame bank ledger',
            unavailable='distinct solver state retained, separate N3 baseline return or complete failure'),
        latency='No fresh detector/head or full-path timing. Wall seconds are replay execution accounting only.',
        limits=['Known DEV245 geometry-reconstructed reference; no independent real physical 6D truth.',
            'Sparse H/no-H fit can be identical if H never entered q; record same-U parity and display differences.',
            'ROLE features and original H remain conditional on Base/N3; this does not test removal of those feature dependencies.',
            'Four controlled synthetic variants and independent role semantics are outside this solver/mask diagnostic.',
            'A frozen successful CPU check is not an accuracy result.'], no_automatic_retry=True))
    C.verify_protocol(args)
    print('SAME_OBSERVATION_FROZEN', C.binding(protocol_path)['sha256'], flush=True)


def preflight(args):
    protocol = C.verify_protocol(args)
    C.require(protocol['source_root'] == str(Path(args.source_root).resolve()) and
        protocol['baseline_root'] == str(Path(args.baseline_root).resolve()), 'split deployment roots differ')
    before = C.protect(args)
    proof = C.population_proof(args, protocol['inputs'])
    C.bound(Path(args.output) / 'PARENT_POPULATION_CHECKS.json', protocol['parent_population_checks'], 'population proof')
    C.require(proof == C.read(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'), 'parent population proof differs')
    probe = C.quiet_snapshot()
    passed = probe['quiet'] and probe['temperature_under_80']
    C.write_new(C.output_path(args, 'PREFLIGHT.json'), dict(schema='independent_same_observation_preflight_v8',
        passed=bool(passed), protocol=C.binding(args.protocol), parent_population_checked=True,
        protection=before, resource_snapshot=probe, new_pose_paths=0, no_automatic_retry=True))
    C.require(passed, 'PENDING competing workload/thermal guard; preflight snapshot preserved')
    print('SAME_OBSERVATION_PREFLIGHT_PASS', flush=True)


@contextmanager
def diagnostic_canary(args, attempted):
    """Deny source assets, images, weights and scored/human data during solves.

    Sealed parent streams are authorized replay inputs. Local Python imports
    and new V8 output writes remain possible; no source_modules is invoked.
    """
    roots = (Path(args.source_root).resolve(), Path(args.baseline_root).resolve(), Path(args.fits).resolve())
    forbidden = ('TARGETS.json', 'PREDICTIONS.jsonl', 'POSTHOC', 'METRICS.json',
        'REAL_CORRESPONDENCE', 'STATIC_VISIBILITY', 'COCO', 'annotation', 'target_cache',
        'target_arrays', 'READY_PREPARED_TARGETS', 'GROUND_TRUTH', 'GEOMETRY_RESOLVED_POSE_GT',
        'AXIS_REVIEW_MANIFEST')
    asset_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pt', '.pth', '.onnx', '.npz', '.npy')
    saved = [(builtins, 'open', builtins.open), (io, 'open', io.open), (Path, 'open', Path.open)]
    def guarded(fn):
        def call(path, *a, **kw):
            if isinstance(path, (str, os.PathLike)):
                actual = Path(path).absolute()
                mode = str(kw.get('mode', a[0] if a else 'r'))
                if 'r' in mode or '+' in mode:
                    text = str(actual).lower()
                    blocked = any(actual.is_relative_to(root) for root in roots) or actual.suffix.lower() in asset_extensions
                    blocked = blocked or any(token.lower() in text for token in forbidden)
                    if blocked:
                        attempted.append(dict(path=str(actual), mode=mode))
                        raise AssertionError('SAME_OBSERVATION_ASSET_CANARY:' + actual.name)
            return fn(path, *a, **kw)
        return call
    for module, name, function in saved:
        setattr(module, name, guarded(function))
    try:
        yield
    finally:
        for module, name, function in saved:
            setattr(module, name, function)


@contextmanager
def primitive_entries(counts):
    import cv2
    saved = {name: getattr(cv2, name) for name in PRIMITIVES}
    for name, function in saved.items():
        def counted(*a, _name=name, _function=function, **kw):
            counts[_name] += 1
            return _function(*a, **kw)
        setattr(cv2, name, counted)
    try:
        yield
    finally:
        for name, function in saved.items():
            setattr(cv2, name, function)


def run(args):
    protocol = C.verify_protocol(args)
    C.require(protocol['source_root'] == str(Path(args.source_root).resolve()) and
        protocol['baseline_root'] == str(Path(args.baseline_root).resolve()), 'split deployment roots differ')
    for name in RUN_NAMES:
        C.output_path(args, name)
    for name in STREAMS.values():
        C.output_path(args, 'PENDING_' + name)
    C.bound(Path(args.output) / 'PARENT_POPULATION_CHECKS.json', protocol['parent_population_checks'], 'population proof')
    preflight_receipt = C.read(Path(args.output) / 'PREFLIGHT.json')
    C.require(preflight_receipt['passed'] is True and
        C.same_binding(preflight_receipt['protocol'], C.binding(args.protocol)), 'independent quiet preflight required')
    protection_before = C.protect(args)
    # Recheck all735/1960/490 input rows before the first numeric bank.
    proof = C.population_proof(args, protocol['inputs'])
    C.require(proof == C.read(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'), 'complete parent population drift')
    C.write_new(C.output_path(args, 'CONTROL_STARTED.json'), dict(schema='fixed_C3_control_started_v8',
        protocol=C.binding(args.protocol), configured_frames=245, configured_rows=735,
        configured_pose_paths=735, parent_checked_before_first_pose=True, no_automatic_retry=True))
    start = time.monotonic()
    counts = Counter()
    primitive = Counter({name: 0 for name in PRIMITIVES})
    statuses, methods = Counter(), Counter()
    streams, probes, attempted = {}, [], []
    active = None
    finished = False
    error = None
    cleanup_errors = []
    current_id = None
    protection_after = None
    shared_equal_U = 0
    removed_H = Counter()
    environment = None
    library_environment = None
    controls_module = None
    original_bank = None
    previous_cv_threads = None

    def guard(phase):
        snapshot = dict(phase=phase, complete_frames=counts['frames_complete'], **C.quiet_snapshot())
        C.write_new(C.output_path(args, 'RESOURCE_%03d.json' % len(probes)), snapshot)
        probes.append(snapshot)
        C.require(snapshot['quiet'] and snapshot['temperature_under_80'], 'PENDING competing workload/thermal guard')

    def stream_diagnostics():
        return {kind: dict(rows=writer.count, published=writer.published,
            pending_basename=writer.pending_path.name, final_basename=writer.final_path.name,
            close_errors=writer.close_errors, interruption_preservation_errors=writer.preservation_errors)
            for kind, writer in streams.items()}

    try:
        guard('before_control_imports')
        from ..pallet_three_head_observation_20261010_v7.run import RowWriter
        for kind, name in STREAMS.items():
            interrupted = 'INTERRUPTED_GEOMETRY.jsonl.gz' if kind == 'geometry' else 'INTERRUPTED_CONTROL_LEDGERS.jsonl.gz'
            streams[kind] = RowWriter(C.output_path(args, name), interrupted_path=C.output_path(args, interrupted))
        with C.original_environment(args) as environment, diagnostic_canary(args, attempted):
            import cv2
            import numpy as np
            from . import controls as controls_module
            C.require(controls_module.METHODS == C.METHODS and controls_module.PRIMARY == C.PRIMARY,
                      'immutable controls have different fixed routes')
            previous_cv_threads = cv2.getNumThreads()
            library_environment = dict(python=sys.version.split()[0], numpy=np.__version__,
                opencv=cv2.__version__, original_OpenCV_threads=previous_cv_threads, solve_OpenCV_threads=1,
                thread_environment={name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                Torch_or_model_library_initialized=False)
            cv2.setNumThreads(1)
            original_bank = controls_module.PoseBank

            class CountedBank(original_bank):
                """Actual entries only; leave every numerical argument unchanged."""
                def __init__(self, *a, **kw):
                    counts['coordinate_banks_started'] += 1
                    super().__init__(*a, **kw)
                    counts['coordinate_banks_complete'] += 1
                def solve(self, *a, **kw):
                    index = len(active['solver_returns'])
                    C.require(index < 3, 'unexpected fourth solve within one fixed control packet')
                    method = C.METHODS[index]
                    counts['logical_pose_paths_started'] += 1
                    counts[method + ':started'] += 1
                    result = super().solve(*a, **kw)
                    counts['logical_pose_paths_complete'] += 1
                    counts[method + ':complete'] += 1
                    active['solver_returns'].append(dict(method=method, solver=result))
                    return result

            controls_module.PoseBank = CountedBank
            try:
                with primitive_entries(primitive):
                    for parent, observation in C.parent_frames(args):
                        current_id = parent['id']
                        C.require(time.monotonic() - start <= args.remaining_seconds, 'operating time limit; preserved prefix')
                        active = dict(id=current_id, solver_returns=[], parent_row=parent,
                            native_N3_points=observation['native_N3_points'])
                        counts['frames_started'] += 1
                        packet = dict(parent_row=parent, native_N3_points=observation['native_N3_points'],
                            parent_completion=proof['completion'])
                        outputs, ledger = controls_module.solve_controls(packet)
                        C.require(len(outputs) == 3 and ledger['parent_replay_parity']['passed'], 'parent replay gate failed')
                        for output in outputs:
                            solver = output['solver']
                            robust = output['method'] != 'ROLE_BOUNDARY_H_STANDARD'
                            output.update(original_native_N3_points=copy.deepcopy(observation['native_N3_points']),
                                native_reference_id=dict(id=current_id, head_arm='IMAGE_ROLE'),
                                parent_observation_row_semantic_sha256=C.semantic_sha256(observation),
                                stored_jacobian_fit_ids=list(solver['final_inliers'] if robust else solver['used'])
                                    if solver.get('geometry', {}).get('jacobian') is not None else [],
                                stored_jacobian_fit_ids_source='unchanged V4 solve rank_ids: final_inliers if robust else used',
                                full_candidate_and_operation_witnesses_preserved=True,
                                solver_replay_latency_claim=False)
                            streams['geometry'].write(output)
                            methods[output['method']] += 1
                            statuses[output['method'] + ':' + output['output_status']] += 1
                            counts['geometry_rows'] += 1
                        streams['ledgers'].write(dict(id=current_id, session=parent['session'], **ledger))
                        counts['ledger_rows'] += 1
                        counts['frames_complete'] += 1
                        shared_equal_U += int(ledger['H_and_no_H_used_U_equal'])
                        removed_H.update(ledger['actual_H_removed_sparse_ids'])
                        active = None
                        if counts['frames_complete'] % 32 == 0:
                            guard('after_frame')
                            print('SAME_OBSERVATION_GEOMETRY', counts['frames_complete'], 245, flush=True)
            finally:
                controls_module.PoseBank = original_bank
        # All monkeypatches/environment/IO guards have been restored here.
        C.require(counts['frames_complete'] == 245 and counts['geometry_rows'] == 735 and
            counts['ledger_rows'] == 245 and dict(methods) == {method: 245 for method in C.METHODS}, 'output population incomplete')
        C.require(counts['logical_pose_paths_started'] == counts['logical_pose_paths_complete'] == 735 and
            counts['coordinate_banks_started'] == counts['coordinate_banks_complete'] == 245, 'actual entry counts differ')
        C.require(not attempted and primitive['solvePnP'] == primitive['cornerSubPix'] == 0,
                  'unexpected source/GT/model/initial solver work')
        guard('after_control_cleanup')
        C.verify_protocol(args)
        protection_after = C.protect(args)
        C.require(protection_before == protection_after, 'prior protection changed during controls')
        for writer in streams.values():
            writer.close()
        C.require(not any(w.close_errors for w in streams.values()), 'row close/fsync failed')
        for writer in streams.values():
            writer.promote()
        finished = True
    except BaseException as exception:
        error = dict(type=type(exception).__name__, message=str(exception), frame_id=current_id,
                     traceback=traceback.format_exc())
    finally:
        if controls_module is not None and original_bank is not None:
            controls_module.PoseBank = original_bank
        if previous_cv_threads is not None:
            try:
                cv2.setNumThreads(previous_cv_threads)
            except BaseException as exception:
                cleanup_errors.append(dict(type=type(exception).__name__, message=str(exception)))
        if not finished:
            for writer in streams.values():
                writer.preserve_interrupted()
            if active is not None:
                try:
                    C.write_new(C.output_path(args, 'INTERRUPTED_CONTROL_PACKET.json'), active)
                except BaseException as exception:
                    cleanup_errors.append(dict(type=type(exception).__name__, message=str(exception)))
        for writer in streams.values():
            cleanup_errors.extend(writer.close_errors)
            cleanup_errors.extend(writer.preservation_errors)
        complete = finished and not cleanup_errors and error is None
        receipt = dict(schema='fixed_same_observation_control_receipt_v8', complete=complete,
            geometry_ready_for_seal=complete, formal_GT_permission_requires_complete_seal=True,
            protocol=C.binding(args.protocol), error=error, cleanup_error=cleanup_errors or None,
            actual_complete_frames=counts['frames_complete'], actual_counts=dict(counts),
            method_rows=dict(methods), output_status_counts=dict(statuses),
            actual_OpenCV_entry_calls=dict(primitive),
            actual_OpenCV_entry_counter_code=protocol['inputs']['runner.py'],
            actual_OpenCV_entry_counter_scope='actual entries during the three solve_controls routes, including their projection/Jacobian/rotation work',
            actual_OpenCV_primitive_names=list(PRIMITIVES), library_environment=library_environment,
            existing_control_replay_pose_paths_started=counts[C.PRIMARY + ':started'],
            existing_control_replay_pose_paths_complete=counts[C.PRIMARY + ':complete'],
            new_diagnostic_pose_paths_started=sum(counts[m + ':started'] for m in C.METHODS[1:]),
            new_diagnostic_pose_paths_complete=sum(counts[m + ':complete'] for m in C.METHODS[1:]),
            actual_detector_calls=0, actual_N3_calls=0, actual_head_calls=0, actual_initial_pose_calls=0,
            actual_CAL_calls=0, actual_query_decode_calls=0, actual_model_constructors=0,
            new_training_updates=0, new_RGB=0, GT_access_during_controls=False,
            source_asset_or_GT_attempted_reads=attempted, explicit_split_environment=environment,
            environment_and_monkeypatch_cleanup_completed=not cleanup_errors, serialization=dict(mode='per_frame_full_witness_stream',
                diagnostic_fields_removed=0, full_parent_population_retained=False, streams=stream_diagnostics()),
            H_and_no_H_equal_used_U_frames=shared_equal_U, actual_H_removed_sparse_ID_counts=dict(removed_H),
            resource_snapshots=probes, protection_before=protection_before, protection_after=protection_after,
            wall_seconds=time.monotonic() - start, wall_seconds_purpose='actual replay execution accounting only',
            runtime_benchmark=False, deployment_latency_claim=False, no_automatic_retry=True)
        C.write_new(C.output_path(args, 'CONTROL_RECEIPT.json'), receipt)
    C.require(receipt['complete'], 'controls did not complete; failed receipt and prefix preserved')
    # Receipt proves cleanup before this seal exists. Scoring must verify both.
    C.write_new(C.output_path(args, 'GEOMETRY_SEAL.json'), dict(schema='fixed_same_observation_geometry_seal_v8',
        complete=True, GT_read_allowed=False, frames=245, rows=735, ledger_rows=245,
        methods=list(C.METHODS), primary=C.PRIMARY, protocol=C.binding(args.protocol),
        geometry=C.binding(Path(args.output) / STREAMS['geometry']),
        ledgers=C.binding(Path(args.output) / STREAMS['ledgers']),
        control_receipt=C.binding(Path(args.output) / 'CONTROL_RECEIPT.json'),
        parent_population_checks=C.binding(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'),
        parent_completion=proof['completion'], cleanup_completed_before_seal=True,
        no_scored_or_human_inputs=True, latency_benchmark=False))
    print('SAME_OBSERVATION_SEALED', 735, 245, 'before_GT', flush=True)


def main():
    args = C.parser(__doc__).parse_args()
    C.require(args.remaining_seconds > 0, 'positive operating budget required')
    if args.stage == 'freeze':
        freeze(args)
    elif args.stage == 'preflight':
        preflight(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
