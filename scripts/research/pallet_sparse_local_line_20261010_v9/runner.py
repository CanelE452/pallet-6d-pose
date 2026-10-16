"""Explicit freeze/preflight/run of the single fixed local C2 diagnostic.

Actual local SciPy solves consume sealed sparse observations, not live images.
Replay wall time is execution accounting and never deployment latency.
"""
from __future__ import annotations

from collections import Counter
import copy
import os
from pathlib import Path
import sys
import time
import traceback

from . import common as C
from ..pallet_same_observation_controls_20261010_v8.runner import diagnostic_canary, primitive_entries, PRIMITIVES

STREAMS = dict(geometry='GEOMETRY_SEALED.jsonl.gz', ledgers='CONTROL_LEDGERS.jsonl.gz')
RUN_NAMES = ('CONTROL_STARTED.json', 'CONTROL_RECEIPT.json', 'GEOMETRY_SEAL.json',
    'GEOMETRY_SEALED.jsonl.gz', 'CONTROL_LEDGERS.jsonl.gz', 'INTERRUPTED_CONTROL_PACKET.json',
    'INTERRUPTED_GEOMETRY.jsonl.gz', 'INTERRUPTED_CONTROL_LEDGERS.jsonl.gz')


def CPU_authority(args, bindings):
    protocol, original = C.read(args.cpu_protocol), C.read(args.cpu_original)
    original_started = C.read(Path(args.cpu_original).with_name(Path(args.cpu_original).stem + '_STARTED.json'))
    review_protocol, review, joined = [C.read(path) for path in
        (args.cpu_review_protocol, args.cpu_review_success, args.cpu_success)]
    review_started = C.read(Path(args.cpu_review_success).with_name(Path(args.cpu_review_success).stem + '_STARTED.json'))
    targets = ['test_no_unused_lines_three_actual_points_remains_LOCAL',
               'test_packet_wrapper_NEW_H_projection_and_full_fallback']
    C.require(protocol['schema'] == 'pre_execution_synthetic_sparse_local_line_CPU_contract_v9' and
        protocol['synthetic_only'] is True and protocol['test_groups_expected'] == 14,
        'fixed local synthetic CPU protocol required')
    failed_names = [item['test'].split(' ')[0] for item in original['failures'] + original['errors']]
    C.require(original['schema'] == 'sparse_local_line_CPU_checks_v9' and original['complete'] is True and
        original['passed'] is False and original['tests_run'] == 14 and len(original['failures']) == 1 and
        len(original['errors']) == 1 and sorted(failed_names) == sorted(targets) and
        not original['forbidden_asset_reads'] and not original['cleanup_errors'] and original['exception'] is None and
        original['code_unchanged_after_tests'] is True and original['protocol_unchanged_after_tests'] is True and
        original['production_local_optimizer_dispatch_count_join'] is True,
        'the unchanged actual fourteen-group attempt and its two named expectation issues must survive')
    C.require(original['code'] == original_started['code'] == protocol['code'] and
        C.same_binding(original['protocol'], bindings['cpu_protocol']) and
        C.same_binding(original_started['protocol'], bindings['cpu_protocol']), 'original CPU byte bindings differ')
    C.require(review_protocol['schema'] == 'pre_execution_synthetic_sparse_local_line_CPU_review_protocol_v9' and
        review_protocol['synthetic_only'] is True and review_protocol['test_groups_expected'] == 2 and
        review_protocol['targeted_test_names'] == targets and
        C.same_binding(review_protocol['original_receipt'], bindings['cpu_original']),
        'the two targeted CPU expectations require their own pre-execution review protocol')
    C.require(review['schema'] == 'sparse_local_line_CPU_review_checks_v9' and review['complete'] is True and
        review['passed'] is True and review['tests_run'] == 2 and review['targeted_test_names'] == targets and
        not review['failures'] and not review['errors'] and not review['forbidden_asset_reads'] and
        not review['cleanup_errors'] and review['exception'] is None and review['other_twelve_rerun'] is False and
        review['original_attempt_preserved_unchanged'] is True and review['code_unchanged_after_tests'] is True and
        review['protocol_unchanged_after_tests'] is True and review['production_local_optimizer_dispatch_count_join'] is True,
        'both corrected CPU expectations must actually pass without rerunning the twelve original passes')
    C.require(review['code'] == review_started['code'] == review_protocol['code'] == joined['code'] and
        {key: value for key, value in review['code'].items() if key != 'review'} == protocol['code'],
        'review/original solver dependencies differ')
    C.require(joined['schema'] == 'sparse_local_line_joined_CPU_contract_checks_v9' and joined['complete'] is True and
        joined['passed'] is True and joined['original_attempt_tests'] == 14 and
        joined['original_attempt_passed_groups'] == joined['accepted_original_unchanged_groups'] == 12 and
        joined['original_attempt_failed_groups'] == joined['review_targeted_tests'] ==
        joined['accepted_review_corrected_groups'] == 2 and joined['contract_groups'] == 14 and
        joined['original_attempt_reported_passed'] is False and joined['other_twelve_rerun'] is False and
        joined['solver_or_threshold_changed'] is False and joined['review_is_new_performance_tuning'] is False and
        joined['original_attempt_preserved_unchanged'] is True and not joined['cleanup_errors'],
        'joined CPU authority must account for original12 and corrected2 while preserving the original failure')
    for key, source in (('original_protocol', 'cpu_protocol'), ('original_receipt', 'cpu_original'),
        ('original_started', 'cpu_original_started'), ('review_protocol', 'cpu_review_protocol'),
        ('review_receipt', 'cpu_review_success'), ('review_started', 'cpu_review_success_started')):
        C.require(C.same_binding(joined[key], bindings[source]), 'joined CPU input differs:' + key)
    for evidence in (review, review_started):
        for key, source in (('protocol', 'cpu_review_protocol'), ('original_receipt', 'cpu_original'),
            ('original_started', 'cpu_original_started'), ('original_protocol', 'cpu_protocol')):
            C.require(C.same_binding(evidence[key], bindings[source]), 'review CPU input differs:' + key)
    for code in joined['code'].values():
        C.bound(C.REPO / code['path'], code, 'CPU-bound source')
    for key, filename in (('solver', 'solver.py'), ('pipeline', 'pipeline.py'), ('tests', 'test_solver.py')):
        C.require(C.same_binding(joined['code'][key], bindings[filename]), 'CPU-tested own source differs:' + filename)
    C.require(C.same_binding(joined['code']['review'], bindings['test_solver_review.py']), 'CPU review code differs')
    for key in ('model_calls', 'real_images', 'real_GT_reads', 'source_cache_reads', 'new_training_updates', 'new_RGB'):
        C.require(original[key] == 0, 'original CPU suite accessed prohibited real assets:' + key)
    C.require(review['new_model_real_GT_image_training_RGB_calls'] ==
        joined['new_model_real_GT_image_training_RGB_calls'] == 0, 'CPU review accessed prohibited assets')
    history = {key: value for key, value in bindings.items() if key.startswith('cpu_history:')}
    return dict(original_protocol=bindings['cpu_protocol'], original_attempt=bindings['cpu_original'],
        original_started=bindings['cpu_original_started'], review_protocol=bindings['cpu_review_protocol'],
        review_attempt=bindings['cpu_review_success'], review_started=bindings['cpu_review_success_started'],
        joined_success_authority=bindings['cpu_success'], all_existing_CPU_attempts_and_protocols_bound=history,
        original_attempt_reported_passed=False, original_groups_passed=12, review_groups_passed=2,
        contract_groups=14, other_twelve_rerun=False, solver_or_threshold_changed=False,
        actual_attempts_preserved=True, automatic_retry=False, no_accuracy_result_claim=True)


def freeze(args):
    C.output_path(args, 'PARENT_POPULATION_CHECKS.json')
    path = Path(args.protocol)
    C.require(path.resolve() == C.DOC / 'PROTOCOL.json', 'formal protocol must be the new V9 DOC')
    target_args = copy.copy(args); target_args.output = str(C.DOC)
    C.output_path(target_args, 'PROTOCOL.json')
    bindings = C.input_bindings(args)
    cpu = CPU_authority(args, bindings)
    protection = C.protect(args)
    proof = C.population_proof(args, bindings)
    C.require(C.input_bindings(args) == bindings, 'code/input bytes changed during full-population checks')
    C.write_new(C.output_path(args, 'PARENT_POPULATION_CHECKS.json'), proof)
    C.write_new(path, dict(schema='fixed_same_observation_local_C2_protocol_v9', policy=C.POLICY,
        methods=list(C.METHODS), primary=C.PRIMARY, comparator=C.COMPARATOR, frames=245, rows=245,
        contrasts=[list(pair) for pair in C.CONTRASTS], contrast_direction='local minus comparator; negative is improvement',
        inputs=bindings, CPU_authority=cpu, postseal_only_inputs=list(C.POSTSEAL_ONLY_INPUTS),
        postseal_bindings_are_hashes_not_pre_fit_reference_decoding=True,
        parent_population_checks=C.binding(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'),
        parent_completion=proof['completion'], point_completion=proof['point_completion'], protected_before=protection,
        source_root=str(Path(args.source_root).resolve()), baseline_root=str(Path(args.baseline_root).resolve()),
        requested_original_C2_missing_latest_sparse_local_block=True,
        historical_v1_C2_and_corrected_66way_and_V5_V6_completed=True,
        initial_pose_is_local_start_not_residual_prior=True,
        NEW_is_numeric_local_output_not_accuracy_success=True, global_uniqueness_proven=False,
        expected_execution=dict(frames=245, geometry_rows=245, ledger_rows=245,
            coordinate_banks=245, logical_local_pose_paths=245,
            maximum_logical_local_starts_per_frame=4, new_PnP_generators=0,
            detector_N3_head_RGB_initial_PnP_CAL_decode_calls=0),
        standalone_geometry_validation_PASS_required_before_GT=True,
        no_fresh_runtime_or_deployment_latency_claim=True, actual_geometry_executed_at_freeze=0,
        evaluation_is_unseen=False, GT_tuning=False, performance_based_policy_changes=0,
        limits=['Local seeded optimization only; whole-factor rank6 does not prove correct correspondences or global uniqueness.',
                'Eight-pixel inlier support and its rank are diagnostics, not the acceptance gate.',
                'Zero unused-line point-local frames must not be called evidence of a line effect.',
                'Geometry-reconstructed known DEV references are not independent measured physical 6D truth.',
                'Source CAL support/straightness does not certify real physical edge ownership.'],
        no_automatic_retry=True))
    C.verify_protocol(args)
    print('SPARSE_LOCAL_LINE_FROZEN', C.binding(path)['sha256'], flush=True)


def preflight(args):
    protocol = C.verify_protocol(args)
    C.require(protocol['source_root'] == str(Path(args.source_root).resolve()) and
        protocol['baseline_root'] == str(Path(args.baseline_root).resolve()), 'split CLI roots differ')
    protection = C.protect(args)
    proof = C.population_proof(args, protocol['inputs'])
    C.bound(Path(args.output) / 'PARENT_POPULATION_CHECKS.json', protocol['parent_population_checks'], 'population proof')
    C.require(proof == C.read(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'), 'parent population proof differs')
    probe = C.quiet_snapshot()
    passed = bool(probe['quiet'] and probe['temperature_under_80'])
    C.write_new(C.output_path(args, 'PREFLIGHT.json'), dict(schema='independent_sparse_local_line_preflight_v9',
        passed=passed, protocol=C.binding(args.protocol), parent_population_checked=True,
        resource_snapshot=probe, protection=protection, local_optimizer_calls=0, no_automatic_retry=True))
    C.require(passed, 'PENDING competing workload/thermal guard; snapshot preserved')
    print('SPARSE_LOCAL_LINE_PREFLIGHT_PASS', flush=True)


def run(args):
    protocol = C.verify_protocol(args)
    C.require(protocol['source_root'] == str(Path(args.source_root).resolve()) and
        protocol['baseline_root'] == str(Path(args.baseline_root).resolve()), 'split CLI roots differ')
    for name in RUN_NAMES:
        C.output_path(args, name)
    for name in STREAMS.values():
        C.output_path(args, 'PENDING_' + name)
    C.bound(Path(args.output) / 'PARENT_POPULATION_CHECKS.json', protocol['parent_population_checks'], 'population proof')
    preflight_receipt = C.read(Path(args.output) / 'PREFLIGHT.json')
    C.require(preflight_receipt['passed'] is True and
        C.same_binding(preflight_receipt['protocol'], C.binding(args.protocol)), 'independent quiet preflight required')
    before = C.protect(args)
    proof = C.population_proof(args, protocol['inputs'])
    C.require(proof == C.read(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'), 'full parent population drift')
    C.write_new(C.output_path(args, 'CONTROL_STARTED.json'), dict(schema='fixed_local_C2_started_v9',
        protocol=C.binding(args.protocol), configured_frames=245, configured_rows=245,
        parent_checked_before_first_local_optimizer=True, no_automatic_retry=True))
    start = time.monotonic()
    counts, operations = Counter(), Counter()
    primitive = Counter({name: 0 for name in PRIMITIVES})
    statuses, methods, diagnostics = Counter(), Counter(), Counter()
    streams, probes, attempted = {}, [], []
    active, active_bank = None, None
    current_id = None
    error, cleanup_errors, finished = None, [], False
    after, environment, library_environment = None, None, None
    pipeline_module, solver_module = None, None
    original_bank, original_optimizer, cv_threads = None, None, None

    def guard(phase):
        snapshot = dict(phase=phase, complete_frames=counts['frames_complete'], **C.quiet_snapshot())
        C.write_new(C.output_path(args, 'RESOURCE_%03d.json' % len(probes)), snapshot)
        probes.append(snapshot)
        C.require(snapshot['quiet'] and snapshot['temperature_under_80'], 'PENDING competing workload/thermal guard')

    def serialization():
        return {kind: dict(rows=writer.count, published=writer.published,
            pending_basename=writer.pending_path.name, final_basename=writer.final_path.name,
            close_errors=writer.close_errors, interruption_preservation_errors=writer.preservation_errors)
            for kind, writer in streams.items()}

    try:
        guard('before_local_imports')
        from ..pallet_three_head_observation_20261010_v7.run import RowWriter
        for kind, name in STREAMS.items():
            interrupted = 'INTERRUPTED_GEOMETRY.jsonl.gz' if kind == 'geometry' else 'INTERRUPTED_CONTROL_LEDGERS.jsonl.gz'
            streams[kind] = RowWriter(C.output_path(args, name), interrupted_path=C.output_path(args, interrupted))
        with C.original_environment(args) as environment, diagnostic_canary(args, attempted):
            import cv2
            import numpy as np
            import scipy
            from . import pipeline as pipeline_module, solver as solver_module
            C.require(pipeline_module.METHOD == C.PRIMARY and pipeline_module.COMPARATOR == C.COMPARATOR,
                      'local pipeline method/comparator differs')
            cv_threads = cv2.getNumThreads()
            cv2.setNumThreads(1)
            library_environment = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                opencv=cv2.__version__, original_OpenCV_threads=cv_threads, solve_OpenCV_threads=1,
                thread_environment={name: os.environ.get(name) for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                Torch_or_model_library_initialized=False)
            original_bank = pipeline_module.LocalPointLineBank
            original_optimizer = solver_module.least_squares

            class CountedBank(original_bank):
                def __init__(self, *a, **kw):
                    nonlocal active_bank
                    counts['coordinate_banks_started'] += 1
                    super().__init__(*a, **kw)
                    counts['coordinate_banks_complete'] += 1
                    active_bank = self
                def solve(self, *a, **kw):
                    counts['logical_local_pose_paths_started'] += 1
                    result = super().solve(*a, **kw)
                    counts['logical_local_pose_paths_complete'] += 1
                    active['solver_returns'].append(result)
                    return result

            def counted_optimizer(fun, x0, *a, **kw):
                counts['actual_SciPy_least_squares_entries'] += 1
                def actual_residual(*values, **options):
                    counts['actual_optimizer_residual_entries'] += 1
                    return fun(*values, **options)
                numeric_jac = kw.get('jac')
                if callable(numeric_jac):
                    def actual_jacobian(*values, **options):
                        counts['actual_optimizer_jacobian_entries'] += 1
                        return numeric_jac(*values, **options)
                    kw['jac'] = actual_jacobian
                try:
                    fit = original_optimizer(actual_residual, x0, *a, **kw)
                except BaseException:
                    counts['actual_SciPy_least_squares_exceptions'] += 1
                    raise
                counts['actual_SciPy_least_squares_returns'] += 1
                counts['actual_SciPy_success_reports'] += int(bool(fit.success))
                return fit

            pipeline_module.LocalPointLineBank = CountedBank
            solver_module.least_squares = counted_optimizer
            try:
                with primitive_entries(primitive):
                    for parent, observation, point in C.parent_frames(args):
                        current_id = parent['id']
                        C.require(time.monotonic() - start <= args.remaining_seconds, 'operating limit; preserve actual prefix')
                        packet = dict(parent_row=parent, role_observation=observation,
                            native_N3_points=observation['native_N3_points'], parent_completion=proof['completion'])
                        active = dict(id=current_id, packet=packet, solver_returns=[])
                        active_bank = None
                        counts['frames_started'] += 1
                        output, ledger = pipeline_module.solve_local_packet(packet)
                        solved = output['solver']
                        ledger.update(point_control_row_semantic_sha256=C.semantic_sha256(point),
                            point_comparator_is_saved_sealed_control=True, point_control_recomputed_here=False)
                        streams['geometry'].write(output)
                        streams['ledgers'].write(dict(id=current_id, session=parent['session'], **ledger))
                        counts.update(geometry_rows=1, ledger_rows=1, frames_complete=1)
                        methods[output['method']] += 1
                        statuses[output['output_status']] += 1
                        operations.update(solved['operation_counts'])
                        pool = solved.get('factor_pool', {})
                        line_count, point_count = pool.get('unique_line_factors', 0), len(solved['used'])
                        diagnostics['unused_lines_positive'] += int(line_count > 0)
                        diagnostics['unused_lines_zero'] += int(line_count == 0)
                        diagnostics['under4_points_with_lines_NEW'] += int(point_count < 4 and line_count > 0 and output['new_pose_estimated'])
                        diagnostics['no_line_point_LOCAL_NEW'] += int(line_count == 0 and output['new_pose_estimated'])
                        diagnostics['NEW_with_weak_inlier_support'] += int(output['new_pose_estimated'] and not solved.get('diagnostic_inlier_support_rank6', False))
                        active, active_bank = None, None
                        if counts['frames_complete'] % 32 == 0:
                            guard('after_frame')
                            print('SPARSE_LOCAL_LINE_GEOMETRY', counts['frames_complete'], 245, flush=True)
            finally:
                pipeline_module.LocalPointLineBank = original_bank
                solver_module.least_squares = original_optimizer
        C.require(counts['frames_complete'] == counts['geometry_rows'] == counts['ledger_rows'] == 245 and
            dict(methods) == {C.PRIMARY: 245}, 'local output population incomplete')
        C.require(counts['coordinate_banks_started'] == counts['coordinate_banks_complete'] ==
            counts['logical_local_pose_paths_started'] == counts['logical_local_pose_paths_complete'] == 245,
            'actual bank/local-path counts differ')
        for key, operation in (('actual_SciPy_least_squares_entries', 'local_optimizer_calls'),
            ('actual_SciPy_least_squares_returns', 'local_optimizer_completions'),
            ('actual_optimizer_residual_entries', 'optimizer_residual_calls'),
            ('actual_optimizer_jacobian_entries', 'optimizer_jacobian_calls')):
            C.require(counts[key] == operations[operation], 'actual local optimizer/callback dispatch join differs:' + key)
        C.require(not attempted and all(primitive[name] == 0 for name in
            ('solvePnP', 'solvePnPGeneric', 'solvePnPRefineLM', 'cornerSubPix')), 'unexpected source/GT/model/initial PnP work')
        guard('after_local_cleanup')
        C.verify_protocol(args)
        after = C.protect(args)
        C.require(before == after, 'prior protection changed')
        for writer in streams.values():
            writer.close()
        C.require(not any(w.close_errors for w in streams.values()), 'stream close/fsync failed')
        for writer in streams.values():
            writer.promote()
        finished = True
    except BaseException as exception:
        error = dict(type=type(exception).__name__, message=str(exception), frame_id=current_id,
                     traceback=traceback.format_exc())
    finally:
        if pipeline_module is not None and original_bank is not None:
            pipeline_module.LocalPointLineBank = original_bank
        if solver_module is not None and original_optimizer is not None:
            solver_module.least_squares = original_optimizer
        if cv_threads is not None:
            try:
                cv2.setNumThreads(cv_threads)
            except BaseException as exception:
                cleanup_errors.append(dict(type=type(exception).__name__, message=str(exception)))
        if not finished:
            for writer in streams.values():
                writer.preserve_interrupted()
            if active is not None:
                if active_bank is not None:
                    active['partial_local_bank_counts'] = dict(active_bank.counts)
                    active['partial_hypothesis_bank_counts'] = dict(active_bank.bank.ledger)
                try:
                    C.write_new(C.output_path(args, 'INTERRUPTED_CONTROL_PACKET.json'), active)
                except BaseException as exception:
                    cleanup_errors.append(dict(type=type(exception).__name__, message=str(exception)))
        for writer in streams.values():
            cleanup_errors.extend(writer.close_errors)
            cleanup_errors.extend(writer.preservation_errors)
        complete = finished and error is None and not cleanup_errors
        receipt = dict(schema='fixed_sparse_local_line_control_receipt_v9', complete=complete,
            protocol=C.binding(args.protocol), error=error, cleanup_error=cleanup_errors or None,
            geometry_ready_for_seal=complete, formal_GT_permission_requires_complete_seal_and_validation_PASS=True,
            actual_complete_frames=counts['frames_complete'], actual_counts=dict(counts),
            local_operation_counts=dict(operations), method_rows=dict(methods), output_status_counts=dict(statuses),
            local_observation_diagnostics=dict(diagnostics), actual_OpenCV_entry_calls=dict(primitive),
            actual_OpenCV_primitive_names=list(PRIMITIVES), actual_entry_counter_code=protocol['inputs']['runner.py'],
            actual_counter_scope='full local packet including optimizer/callbacks, diagnostics, equivalence and final display',
            actual_optimizer_dispatch_count_join=all(counts[a] == operations[b] for a, b in (
                ('actual_SciPy_least_squares_entries', 'local_optimizer_calls'),
                ('actual_SciPy_least_squares_returns', 'local_optimizer_completions'),
                ('actual_optimizer_residual_entries', 'optimizer_residual_calls'),
                ('actual_optimizer_jacobian_entries', 'optimizer_jacobian_calls'))) if complete else False,
            actual_detector_calls=0, actual_N3_calls=0, actual_head_calls=0, actual_initial_pose_calls=0,
            actual_CAL_calls=0, actual_query_decode_calls=0, actual_model_constructors=0,
            new_training_updates=0, new_RGB=0, GT_access_during_controls=False,
            source_asset_or_GT_attempted_reads=attempted, explicit_split_environment=environment,
            environment_and_monkeypatch_cleanup_completed=not cleanup_errors,
            library_environment=library_environment, serialization=dict(mode='full_per_frame_local_witness_stream',
                diagnostic_fields_removed=0, full_parent_population_retained=False, streams=serialization()),
            resource_snapshots=probes, protection_before=before, protection_after=after,
            wall_seconds=time.monotonic() - start, wall_seconds_purpose='replay execution accounting only',
            runtime_benchmark=False, deployment_latency_claim=False, no_automatic_retry=True)
        C.write_new(C.output_path(args, 'CONTROL_RECEIPT.json'), receipt)
    C.require(receipt['complete'], 'local geometry failed; actual receipt/prefix preserved')
    C.write_new(C.output_path(args, 'GEOMETRY_SEAL.json'), dict(schema='fixed_sparse_local_line_geometry_seal_v9',
        complete=True, GT_read_allowed=False, frames=245, rows=245, ledger_rows=245,
        methods=list(C.METHODS), primary=C.PRIMARY, comparator=C.COMPARATOR, protocol=C.binding(args.protocol),
        geometry=C.binding(Path(args.output) / STREAMS['geometry']), ledgers=C.binding(Path(args.output) / STREAMS['ledgers']),
        control_receipt=C.binding(Path(args.output) / 'CONTROL_RECEIPT.json'),
        parent_population_checks=C.binding(Path(args.output) / 'PARENT_POPULATION_CHECKS.json'),
        parent_completion=proof['completion'], point_completion=proof['point_completion'],
        cleanup_completed_before_seal=True, no_scored_or_human_inputs=True,
        standalone_geometry_validation_PASS_required_before_GT=True, latency_benchmark=False))
    print('SPARSE_LOCAL_LINE_SEALED', 245, 'before_GT', flush=True)


def main():
    args = C.parser(__doc__).parse_args()
    C.require(args.remaining_seconds > 0, 'positive operating budget required')
    {'freeze': freeze, 'preflight': preflight, 'run': run}[args.stage](args)


if __name__ == '__main__':
    main()
