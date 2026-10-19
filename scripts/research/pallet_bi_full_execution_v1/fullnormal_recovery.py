"""Recover interrupted D4 rows without changing its numerical implementation.

The original producer lock, row receipts and parent job envelopes stay intact.
Only a fresh invocation receives new dispatch envelopes. Production is launched
by ROOT, after an approval binding this adapter and the original producer lock.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import resource
import shutil
import time

_IMPORTED_AT = time.monotonic()
_IMPORT_CPU = time.process_time()
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '1'

from . import fullnormal_dispatch_four as D4

Q = D4.Q
SCHEMA = 'FULL_NORMAL_D4_RECOVERY_INVOCATION_V1'
D4_SHA256 = '42da73724ea0a7a51eaa167786f3723bdbba38404317ab2932ffa0d76a3f6df8'


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing receipt: ' + str(path))
    Q.write(path, value)


def source_bindings():
    if Q.sha(D4.__file__) != D4_SHA256:
        raise ValueError('Reviewed immutable D4 source changed')
    return dict(recovery_controller_sha256=Q.sha(__file__),
                delegated_D4_source_binding=D4.source_bindings())


def hashes(root):
    """Bytes only: never decode a pose/result/reference."""
    root = Path(root)
    return {str(p.relative_to(root)): Q.sha(p)
            for p in sorted(root.rglob('*')) if p.is_file()} if root.exists() else {}


def require_preserved(root, previous):
    for relative, digest in previous.items():
        if Q.sha(Path(root)/relative) != digest:
            raise ValueError('Historical completed bytes changed: ' + relative)


def require_first_aggregate(base, arms):
    names = ['SUMMARY.json', 'DISPATCH_COMPLETION.json']
    names += [f'{a}/{d}/{n}' for a in arms for d in ('real', 'synthetic')
              for n in ('FEATURES.jsonl.gz', 'FEATURE_SEAL.json')]
    for name in names:
        if (Path(base)/name).exists():
            raise FileExistsError('Recovery permits only first aggregation: ' + name)


def inspect_rows(base, frames, arms, binding, expected_packets, packet_root, reuse_root=None):
    """Authenticate complete metadata; list incomplete files for preservation.

    A ROW without FRAME, or FRAME without ROW, requires ROOT disposition rather
    than silently inventing an execution account. Exact packet payloads are
    recomputed again by the unchanged row helper before any completed-row reuse.
    """
    base, packet_root = Path(base), Path(packet_root)
    completed, preserved, pending, calls = [], {}, [], 0
    registered = {(r['domain'], r['ordinal'], r['arm']): r
                  for r in Q.read(Path(reuse_root)/'REUSE_MANIFEST.json')['rows']} if reuse_root else {}
    for domain, ff in frames.items():
        for ordinal, frame in enumerate(ff):
            packet = None
            for arm in arms:
                folder = base/arm/domain/f'{ordinal:04d}'
                receipt_path, seal_path = folder/'ROW_EXECUTION.json', folder/'FRAME_SEAL.json'
                if receipt_path.exists() != seal_path.exists():
                    raise ValueError('Ambiguous FRAME/ROW history requires ROOT disposition: ' + str(folder))
                if not seal_path.exists():
                    if folder.exists():
                        pending.append(dict(arm=arm, domain=domain, ordinal=ordinal, id=frame['id'],
                            path=str(folder), artifacts=hashes(folder),
                            prior_optimizer_progress='UNKNOWN_NOT_ZERO'))
                    continue
                receipt, seal = Q.read(receipt_path), Q.read(seal_path)
                identity = dict(id=frame['id'], arm=arm, domain=domain, ordinal=ordinal)
                if not receipt.get('complete') or any(receipt.get(k) != v for k, v in identity.items()):
                    raise ValueError('Completed ROW identity differs')
                if not seal.get('complete') or any(seal.get(k) != identity[k] for k in ('id', 'domain', 'ordinal')):
                    raise ValueError('Completed FRAME identity differs')
                if receipt['driver_sha256'] != binding['row_helper_sha256']:
                    raise ValueError('Historical ROW helper changed')
                if (receipt['numerical_kernel_source_binding'] != binding['numerical_kernel_source_binding'] or
                        seal['source_binding'] != binding['numerical_kernel_source_binding']):
                    raise ValueError('Completed ROW numerical closure differs')
                if (receipt['original_frame_seal_sha256'] != Q.sha(seal_path) or
                        receipt['cache_payload_sha256'] != seal['cache_payload_sha256']):
                    raise ValueError('Completed ROW/FRAME binding differs')
                if seal.get('arm') != arm:
                    proof = receipt.get('pilot_reuse_proof') or {}
                    entry = registered.get((domain, ordinal, arm))
                    staged = Path(reuse_root)/arm/domain/f'{ordinal:04d}' if reuse_root else None
                    if (entry is None or Path(proof.get('source_folder', '')).resolve() != staged.resolve() or
                            proof.get('source_frame_seal_sha256') != Q.sha(seal_path) or
                            proof.get('copied_artifact_sha256') != seal['artifacts'] or
                            proof.get('dependency_payload_equality_verified') is not True or
                            proof.get('pilot_source_preserved') is not True or
                            entry['source_frame_seal_sha256'] != Q.sha(seal_path) or
                            entry['desired_cache_payload_sha256'] != seal['cache_payload_sha256'] or
                            Q.sha(staged/'FRAME_SEAL.json') != Q.sha(seal_path) or
                            Q.sha(Path(entry['source'])/'FRAME_SEAL.json') != Q.sha(seal_path)):
                        raise ValueError('Cross-arm FRAME header lacks exact source proof')
                    for name, digest in seal['artifacts'].items():
                        if Q.sha(staged/name) != digest or Q.sha(Path(entry['source'])/name) != digest:
                            raise ValueError('Cross-arm registered origin/staging artifact changed')
                if packet is None:
                    packet = Q.read(packet_root/'packets'/domain/f'{ordinal:04d}.json.gz')
                if (receipt['requested_packet_file_sha256'] != expected_packets[(domain, ordinal)] or
                        receipt['requested_selection_sha256'] != Q.P.digest(packet['selections'][arm])):
                    raise ValueError('Historical ROW requested packet/selection changed')
                if set(seal['artifacts']) != {'RESULT.json.gz', 'FEATURE.json.gz'}:
                    raise ValueError('Unexpected completed FRAME artifacts')
                for name, digest in seal['artifacts'].items():
                    if Q.sha(folder/name) != digest:
                        raise ValueError('Completed FRAME artifact changed')
                value = receipt['resource']['actual_new_pose_optimizer_calls']
                if type(value) is not int or value < 0:
                    raise ValueError('Invalid historical optimizer account')
                calls += value
                for name in ('ROW_EXECUTION.json', 'FRAME_SEAL.json', *seal['artifacts']):
                    p = folder/name
                    preserved[str(p.relative_to(base))] = Q.sha(p)
                completed.append(identity)
    return dict(completed_rows=completed, completed_row_count=len(completed),
                known_stored_new_optimizer_calls=calls, completed_artifacts_sha256=preserved,
                incomplete_arm_directories=pending,
                interrupted_prior_optimizer_calls='UNKNOWN_NOT_ZERO',
                historical_cost_scope='original-base ROW accounts only; never add again as recovery calls')


def preserve_unsealed(base, invocation, pending):
    """Move every unsealed leaf byte to fresh history before a retry may write."""
    base, invocation = Path(base), Path(invocation)
    records = []
    for row in pending:
        folder = Path(row['path'])
        if (folder/'FRAME_SEAL.json').exists() or (folder/'ROW_EXECUTION.json').exists():
            raise ValueError('Unsealed state changed before preservation')
        if hashes(folder) != row['artifacts']:
            raise ValueError('Unsealed bytes changed since census')
        for relative, digest in row['artifacts'].items():
            source = folder/relative
            destination = invocation/'unsealed_history'/source.relative_to(base)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError('Never overwrite unsealed history')
            shutil.move(str(source), str(destination))
            if Q.sha(destination) != digest:
                raise ValueError('Unsealed history copy differs')
            records.append(dict(original_path=str(source), preserved_path=str(destination), sha256=digest))
    return records


def validate_results(results, frames, arms, base):
    expected = {(d, i): f['id'] for d, ff in frames.items() for i, f in enumerate(ff)}
    seen, calls = set(), 0
    for result in results:
        key = result['domain'], result['ordinal']
        if key in seen or expected.get(key) != result['id']:
            raise ValueError('Invocation job identity/denominator differs')
        seen.add(key)
        if [r['arm'] for r in result['records']] != list(arms):
            raise ValueError('Invocation job arm order differs')
        for row in result['records']:
            path = Path(base)/row['arm']/key[0]/f'{key[1]:04d}'/'ROW_EXECUTION.json'
            if Path(row['path']) != path or Q.sha(path) != row['sha256'] or Q.read(path) != row['receipt']:
                raise ValueError('Invocation embedded ROW account changed')
        value = result['actual_new_optimizer_calls_this_job']
        if type(value) is not int or value < 0:
            raise ValueError('Invalid invocation optimizer account')
        calls += value
    if seen != set(expected):
        raise ValueError('Every original row must finish before aggregation')
    return calls


def feature_bindings(base, arms):
    result = {}
    for arm in arms:
        for domain, count in (('real', 238), ('synthetic', 1003)):
            folder = Path(base)/arm/domain
            seal = Q.read(folder/'FEATURE_SEAL.json')
            if (not seal['complete'] or seal['rows'] != count or
                    Q.sha(folder/'FEATURES.jsonl.gz') != seal['head_features_sha256'] or
                    seal['producer_lock_sha256'] != Q.sha(Path(base)/'PRODUCER_LOCK.json')):
                raise ValueError('Full recovered feature binding differs')
            result[arm+'/'+domain] = dict(feature_seal_sha256=Q.sha(folder/'FEATURE_SEAL.json'),
                                         features_sha256=Q.sha(folder/'FEATURES.jsonl.gz'))
    return result


def run(prior_run, base, invocation, approved_go, approved_go_sha256, *,
        F_completion=None, original_F_controller=None):
    started, cpu_started = time.monotonic(), time.process_time()
    child_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    prior_run, base, invocation, approved_go = map(Path, (prior_run, base, invocation, approved_go))
    if invocation.exists() or invocation == base or base in invocation.parents:
        raise FileExistsError('Require a fresh invocation outside the original producer root')
    if Q.sha(approved_go) != approved_go_sha256:
        raise ValueError('Recovery ROOT GO digest differs')
    go = Q.read(approved_go)
    if (go.get('approved') is not True or go.get('recovery_controller_sha256') != Q.sha(__file__) or
            go.get('producer_lock_sha256', {}).get(str(base)) != Q.sha(base/'PRODUCER_LOCK.json')):
        raise PermissionError('ROOT GO must bind this adapter and immutable original producer lock')
    with (base/'PRODUCER_LOCK.json').open('rb') as ownership:
        # Advisory ownership adds no file and changes no original lock bytes.
        fcntl.flock(ownership.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = source_bindings()
        original = Q.read(base/'PRODUCER_LOCK.json')
        delegated = binding['delegated_D4_source_binding']
        for key in ('orchestration_sha256', 'preserved_D3_validation_sha256',
                    'row_helper_sha256', 'numerical_kernel_source_binding'):
            if original[key] != delegated[key]:
                raise ValueError('Immutable original D4 source/configuration differs: ' + key)
        if original['schema'] != D4.SCHEMA or original['workers'] != 4:
            raise ValueError('Only the reviewed original four-worker producer is recoverable')
        arms = original['arms']
        is_F = arms == ['F_COMMON', 'F_BI']
        if arms not in (['C0', 'CP', 'CB'], ['F_COMMON', 'F_BI']):
            raise ValueError('Unexpected original arm protocol')
        if is_F != bool(F_completion) or is_F != bool(original_F_controller):
            raise ValueError('F core requires an explicit new completion path and original controller source')
        if F_completion and Path(F_completion).exists():
            raise FileExistsError('Preserve existing F completion')
        require_first_aggregate(base, arms)
        packet_root, packet_seal = Path(original['packet_root']), Path(original['packet_seal'])
        for path, expected in ((packet_seal, original['packet_seal_sha256']),
                               (Path(original['environment_path']), original['environment_sha256']),
                               (prior_run/'P0_INPUT_CONTRACT.json', original['prior_P0_sha256']),
                               (base/'PACKET_VALIDATION.json', original['packet_validation_receipt_sha256'])):
            if Q.sha(path) != expected:
                raise ValueError('Original producer input changed: ' + str(path))
        for path, digest in original['driver_validation_source_sha256'].items():
            if Q.sha(path) != digest:
                raise ValueError('Original row validation source changed')
        frames = {d: D4.CAL.load_bound_frames(prior_run, d)[0] for d in ('real', 'synthetic')}
        if {d: len(ff) for d, ff in frames.items()} != {'real': 238, 'synthetic': 1003}:
            raise ValueError('Original real238/synthetic1003 required')
        if {d: [f['id'] for f in ff] for d, ff in frames.items()} != original['static_ordered_row_ids']:
            raise ValueError('Original static row identity/order differs')
        expected, validation = D4.verify_packet_bindings(packet_root, packet_seal, frames)
        reuse_root = (original.get('exact_reuse_binding') or {}).get('root')
        if D4.verify_reuse_root(reuse_root, packet_seal, delegated['numerical_kernel_source_binding']) != original['exact_reuse_binding']:
            raise ValueError('Original future-pilot/exact staging changed')
        before = inspect_rows(base, frames, arms, delegated, expected, packet_root, reuse_root)
        old_jobs = hashes(base/'row_jobs')
        p0 = Q.read(prior_run/'P0_INPUT_CONTRACT.json')
        jobs = [(d, i, f, p0[d]['inventory'][i]['observation_folder'], expected[(d, i)])
                for d, ff in frames.items() for i, f in enumerate(ff)]
        invocation.mkdir(parents=True, exist_ok=False)
        initial = dict(schema=SCHEMA, source_binding=binding, approved_GO=str(approved_go),
            approved_GO_sha256=approved_go_sha256, original_output=str(base), workers=4,
            original_producer_lock_sha256=Q.sha(base/'PRODUCER_LOCK.json'),
            original_packet_seal_sha256=Q.sha(packet_seal), original_environment_sha256=original['environment_sha256'],
            original_row_jobs_sha256=old_jobs, total_jobs=len(jobs), actual_pid=os.getpid(),
            actual_argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v],
            process_startticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            delegated_execute_output=str(invocation), delegated_initializer_output=str(base),
            original_F_controller_source_sha256=Q.sha(original_F_controller) if is_F else None,
            new_H1_fits=0, F_choice_calls=0, numeric_reference_inputs=False, epoch=time.time())
        write_new(invocation/'INVOCATION_LOCK.json', initial)
        write_new(invocation/'PACKET_VALIDATION.json', validation)
        write_new(invocation/'HISTORICAL_STATE.json', before)
        archive = preserve_unsealed(base, invocation, before['incomplete_arm_directories'])
        write_new(invocation/'UNSEALED_HISTORY.json', dict(files=archive,
            interrupted_prior_optimizer_calls='UNKNOWN_NOT_ZERO', numeric_reference_inputs=False))
        setup_done = time.monotonic()
        results, failures, dispatch = D4.execute_static(jobs, invocation, 4, D4.initialize_worker,
            (str(prior_run), str(packet_root), str(base), arms, str(reuse_root) if reuse_root else None),
            D4.row_job, original.get('stop_file'))
        dispatch_done = time.monotonic()
        write_new(invocation/'DISPATCH_RESULT.json', dispatch)
        require_preserved(base, before['completed_artifacts_sha256'])
        if hashes(base/'row_jobs') != old_jobs:
            raise ValueError('Original parent row_jobs were changed')
        if Q.sha(base/'PRODUCER_LOCK.json') != initial['original_producer_lock_sha256']:
            raise ValueError('Original producer lock was changed')
        full = (not failures and len(results) == len(jobs) and dispatch['submitted_jobs'] == len(jobs)
                and dispatch['unsubmitted_jobs'] == 0 and not dispatch['checkpoint_stop_observed']
                and dispatch['active_rows_drained_before_exit'])
        calls = sum(r['actual_new_optimizer_calls_this_job'] for r in results)
        cumulative = None
        summary = None
        if full:
            calls = validate_results(results, frames, arms, base)
            after = inspect_rows(base, frames, arms, delegated, expected, packet_root, reuse_root)
            if after['completed_row_count'] != len(jobs)*len(arms) or after['incomplete_arm_directories']:
                raise ValueError('Incomplete arm metadata before aggregate')
            cumulative = after['known_stored_new_optimizer_calls']
            if cumulative - before['known_stored_new_optimizer_calls'] != calls:
                raise ValueError('Current calls must equal new stored accounts, excluding historical copies')
            require_first_aggregate(base, arms)
            summary = D4.aggregate(base, frames, arms, original, results, failures, started)
            if summary['status'] != 'COMPLETE_NEW' or summary['current_invocation_new_pose_optimizer_calls'] != calls:
                raise ValueError('Unexpected aggregate completion/accounting')
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt = dict(schema=SCHEMA, complete=full, status='COMPLETE' if full else 'INCOMPLETE',
            recovery_controller_sha256=Q.sha(__file__), invocation=str(invocation), original_output=str(base),
            invocation_lock_sha256=Q.sha(invocation/'INVOCATION_LOCK.json'),
            historical_state_sha256=Q.sha(invocation/'HISTORICAL_STATE.json'),
            unsealed_history_sha256=Q.sha(invocation/'UNSEALED_HISTORY.json'),
            dispatch_result_sha256=Q.sha(invocation/'DISPATCH_RESULT.json'), dispatch=dispatch,
            job_envelopes_sha256=hashes(invocation/'row_jobs'), technical_failures=failures,
            original_producer_lock_sha256=initial['original_producer_lock_sha256'],
            original_completed_artifacts_preserved=True, original_parent_row_jobs_preserved=True,
            prior_stored_known_calls=before['known_stored_new_optimizer_calls'],
            actual_new_optimizer_calls_this_invocation=calls if full else None,
            successful_returned_jobs_new_optimizer_calls_lower_bound=calls,
            current_invocation_call_count_status='EXACT_FULL_ROW_IDENTITY' if full else 'LOWER_BOUND_SUCCESSFUL_JOBS_ONLY',
            current_invocation_unattributed_optimizer_calls='UNKNOWN_NOT_ZERO' if failures else 0,
            current_invocation_unknown_scope='failed jobs may have sealed additional arms or unsealed work' if failures else None,
            base_stored_calls_cumulative=cumulative,
            cost_identity_verified=full, copied_historical_intrinsic_calls_added=0,
            interrupted_historical_optimizer_calls='UNKNOWN_NOT_ZERO',
            total_campaign_actual_calls='UNKNOWN_NOT_ZERO',
            historical_unknown_scope='lost original D4 in-memory work; not erased by successful recovery',
            wall_seconds=time.monotonic()-started, setup_wall_seconds=setup_done-started,
            dispatch_wall_seconds=dispatch_done-setup_done,
            aggregate_and_verification_wall_seconds=time.monotonic()-dispatch_done,
            module_import_wall_seconds=started-_IMPORTED_AT, module_import_CPU_seconds=cpu_started-_IMPORT_CPU,
            controller_CPU_seconds=time.process_time()-cpu_started,
            worker_CPU_seconds=(children.ru_utime+children.ru_stime-child_started.ru_utime-child_started.ru_stime),
            worker_peak_RSS_bytes=children.ru_maxrss*1024,
            per_job_CPU_seconds=sum(r['current_job_CPU_seconds'] for r in results),
            summary_sha256=Q.sha(base/'SUMMARY.json') if full else None,
            new_H1_fits=0, F_choice_calls=0, numeric_reference_inputs=False, epoch=time.time())
        write_new(invocation/'RECOVERY_COMPLETION.json', receipt)
        if full:
            write_new(base/'DISPATCH_COMPLETION.json', dict(dispatch,
                orchestration_sha256=delegated['orchestration_sha256'],
                preserved_D3_validation_sha256=delegated['preserved_D3_validation_sha256'],
                row_helper_sha256=delegated['row_helper_sha256'], summary_sha256=Q.sha(base/'SUMMARY.json'),
                recovery_controller_sha256=Q.sha(__file__), recovery_invocation=str(invocation),
                recovery_completion_sha256=Q.sha(invocation/'RECOVERY_COMPLETION.json'),
                actual_new_optimizer_calls_this_invocation=calls,
                interrupted_historical_optimizer_calls='UNKNOWN_NOT_ZERO'))
            if is_F:
                write_new(F_completion, dict(complete=True, component='unconditional F core banks only',
                    final_F_complete=False, unconditional_core_only=True, source_sha256=Q.sha(__file__),
                    original_controller_source_sha256=Q.sha(original_F_controller),
                    recovery_invocation=str(invocation), recovery_completion_sha256=Q.sha(invocation/'RECOVERY_COMPLETION.json'),
                    root_GO_sha256=approved_go_sha256, workers=4,
                    producer_summary_sha256=Q.sha(base/'SUMMARY.json'), producer_lock_sha256=Q.sha(base/'PRODUCER_LOCK.json'),
                    producer_dispatch_sha256=Q.sha(base/'DISPATCH_COMPLETION.json'),
                    feature_bindings=feature_bindings(base, arms), H1_fits=0, real_reference_reads=0,
                    F_choice_calls=0, interrupted_historical_optimizer_calls='UNKNOWN_NOT_ZERO', epoch=time.time()))
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior-run', 'base', 'invocation', 'approved-go', 'approved-go-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--F-completion')
    parser.add_argument('--original-F-controller')
    args = parser.parse_args()
    value = run(args.prior_run, args.base, args.invocation, args.approved_go, args.approved_go_sha256,
                F_completion=args.F_completion, original_F_controller=args.original_F_controller)
    print({k: value[k] for k in ('status', 'actual_new_optimizer_calls_this_invocation',
                               'interrupted_historical_optimizer_calls')}, flush=True)
    if not value['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
