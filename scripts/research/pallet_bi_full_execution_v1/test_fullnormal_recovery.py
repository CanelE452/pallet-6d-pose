"""Constructed metadata only: no solver, model, reference or dataset execution."""
from pathlib import Path
import shutil

import pytest

from scripts.research.pallet_bi_full_execution_v1 import fullnormal_recovery as R


def row_fixture(tmp_path, *, calls=17):
    base, packets = tmp_path/'base', tmp_path/'packets'
    folder = base/'C0/real/0000'
    requested = dict(constructed='selection')
    R.Q.write(packets/'packets/real/0000.json.gz', dict(selections=dict(C0=requested)))
    packet_sha = R.Q.sha(packets/'packets/real/0000.json.gz')
    for name in ('RESULT.json.gz', 'FEATURE.json.gz'):
        R.Q.write(folder/name, dict(constructed=True))
    kernel = {'constructed_source': 'immutable'}
    seal = dict(complete=True, id='a', arm='C0', domain='real', ordinal=0,
                source_binding=kernel, cache_payload_sha256='payload',
                artifacts={name: R.Q.sha(folder/name) for name in ('RESULT.json.gz', 'FEATURE.json.gz')})
    R.Q.write(folder/'FRAME_SEAL.json', seal)
    receipt = dict(complete=True, id='a', arm='C0', domain='real', ordinal=0,
        driver_sha256='row-helper', numerical_kernel_source_binding=kernel,
        original_frame_seal_sha256=R.Q.sha(folder/'FRAME_SEAL.json'),
        cache_payload_sha256='payload', requested_packet_file_sha256=packet_sha,
        requested_selection_sha256=R.Q.P.digest(requested),
        resource=dict(actual_new_pose_optimizer_calls=calls), pilot_reuse_proof=None)
    R.Q.write(folder/'ROW_EXECUTION.json', receipt)
    return dict(base=base, packet_root=packets, frames={'real': [dict(id='a')]}, arms=['C0'],
                binding=dict(row_helper_sha256='row-helper', numerical_kernel_source_binding=kernel),
                expected_packets={('real', 0): packet_sha})


def test_completed_metadata_and_historical_calls_are_preserved(tmp_path):
    args = row_fixture(tmp_path)
    state = R.inspect_rows(**args)
    assert state['completed_row_count'] == 1 and state['known_stored_new_optimizer_calls'] == 17
    assert state['incomplete_arm_directories'] == []
    R.require_preserved(args['base'], state['completed_artifacts_sha256'])
    folder = args['base']/'C0/real/0000'
    receipt = R.Q.read(folder/'ROW_EXECUTION.json')
    result = dict(domain='real', ordinal=0, id='a', current_job_CPU_seconds=0,
        records=[dict(arm='C0', path=str(folder/'ROW_EXECUTION.json'),
                      sha256=R.Q.sha(folder/'ROW_EXECUTION.json'), receipt=receipt)],
        actual_new_optimizer_calls_this_job=0)
    assert R.validate_results([result], args['frames'], args['arms'], args['base']) == 0
    assert result['records'][0]['receipt']['resource']['actual_new_pose_optimizer_calls'] == 17


@pytest.mark.parametrize('changed', ['id', 'arm', 'domain', 'ordinal', 'cache_payload_sha256',
                                     'original_frame_seal_sha256', 'requested_packet_file_sha256',
                                     'requested_selection_sha256', 'driver_sha256'])
def test_mismatched_complete_receipt_rejected(tmp_path, changed):
    args = row_fixture(tmp_path)
    path = args['base']/'C0/real/0000/ROW_EXECUTION.json'
    value = R.Q.read(path);value[changed] = 'different'
    R.Q.write(path, value)
    with pytest.raises(ValueError):
        R.inspect_rows(**args)


def test_changed_result_bytes_rejected_without_pose_decode(tmp_path):
    args = row_fixture(tmp_path)
    (args['base']/'C0/real/0000/RESULT.json.gz').write_bytes(b'changed opaque bytes')
    with pytest.raises(ValueError, match='artifact changed'):
        R.inspect_rows(**args)


@pytest.mark.parametrize('missing', ['ROW_EXECUTION.json', 'FRAME_SEAL.json'])
def test_ambiguous_cost_history_rejected_instead_of_zero_filled(tmp_path, missing):
    args = row_fixture(tmp_path)
    (args['base']/'C0/real/0000'/missing).unlink()
    with pytest.raises(ValueError, match='ROOT disposition'):
        R.inspect_rows(**args)


def test_cross_arm_frame_requires_exact_original_seal_proof(tmp_path):
    args = row_fixture(tmp_path)
    folder = args['base']/'C0/real/0000'
    seal = R.Q.read(folder/'FRAME_SEAL.json');seal['arm'] = 'prior_arm'
    R.Q.write(folder/'FRAME_SEAL.json', seal)
    receipt = R.Q.read(folder/'ROW_EXECUTION.json')
    receipt['original_frame_seal_sha256'] = R.Q.sha(folder/'FRAME_SEAL.json')
    R.Q.write(folder/'ROW_EXECUTION.json', receipt)
    with pytest.raises(ValueError, match='Cross-arm'):
        R.inspect_rows(**args)
    origin = tmp_path/'original_source';origin.mkdir()
    staging = tmp_path/'registered_staging';staged = staging/'C0/real/0000';staged.mkdir(parents=True)
    for name in ('FRAME_SEAL.json', 'RESULT.json.gz', 'FEATURE.json.gz'):
        shutil.copy2(folder/name, origin/name);shutil.copy2(folder/name, staged/name)
    receipt['pilot_reuse_proof'] = dict(source_folder=str(staged),
        source_frame_seal_sha256=R.Q.sha(staged/'FRAME_SEAL.json'), copied_artifact_sha256=seal['artifacts'],
        dependency_payload_equality_verified=True, pilot_source_preserved=True)
    R.Q.write(folder/'ROW_EXECUTION.json', receipt)
    with pytest.raises(ValueError, match='Cross-arm'):
        R.inspect_rows(**args)
    R.Q.write(staging/'REUSE_MANIFEST.json', dict(rows=[dict(domain='real', ordinal=0, arm='C0',
        source=str(origin), source_frame_seal_sha256=R.Q.sha(staged/'FRAME_SEAL.json'),
        desired_cache_payload_sha256='payload')]))
    args['reuse_root'] = staging
    assert R.inspect_rows(**args)['completed_row_count'] == 1
    (origin/'RESULT.json.gz').write_bytes(b'changed origin')
    with pytest.raises(ValueError, match='origin/staging artifact'):
        R.inspect_rows(**args)


def test_unsealed_and_pending_bytes_preserved_before_retry(tmp_path):
    args = row_fixture(tmp_path)
    frames = {'real': [dict(id='a'), dict(id='interrupted')]}
    folder = args['base']/'C0/real/0001';folder.mkdir()
    (folder/'RESULT.json.gz').write_bytes(b'opaque interrupted result')
    (folder/'.pending').write_bytes(b'opaque partial write')
    original = R.hashes(folder)
    args['frames'] = frames
    state = R.inspect_rows(**args)
    assert state['known_stored_new_optimizer_calls'] == 17
    assert state['incomplete_arm_directories'][0]['prior_optimizer_progress'] == 'UNKNOWN_NOT_ZERO'
    invocation = tmp_path/'fresh_invocation'
    archive = R.preserve_unsealed(args['base'], invocation, state['incomplete_arm_directories'])
    assert len(archive) == 2 and R.hashes(folder) == {}
    assert R.hashes(invocation/'unsealed_history/C0/real/0001') == original
    R.require_preserved(args['base'], state['completed_artifacts_sha256'])


def test_unsealed_change_is_not_overwritten(tmp_path):
    base = tmp_path/'base';folder = base/'C0/real/0000';folder.mkdir(parents=True)
    path = folder/'.pending';path.write_text('first')
    pending = [dict(path=str(folder), artifacts=R.hashes(folder))]
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed since census'):
        R.preserve_unsealed(base, tmp_path/'invocation', pending)
    assert path.read_text() == 'changed'


@pytest.mark.parametrize('name', ['SUMMARY.json', 'DISPATCH_COMPLETION.json',
                                'C0/real/FEATURE_SEAL.json', 'C0/synthetic/FEATURES.jsonl.gz'])
def test_aggregate_can_only_be_created_first_time(tmp_path, name):
    path = tmp_path/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text('original')
    with pytest.raises(FileExistsError, match='first aggregation'):
        R.require_first_aggregate(tmp_path, ['C0'])
    assert path.read_text() == 'original'


def test_new_writes_never_overwrite(tmp_path):
    path = tmp_path/'receipt.json';path.write_text('preserved')
    with pytest.raises(FileExistsError):
        R.write_new(path, dict(changed=True))
    assert path.read_text() == 'preserved'


def constructed_run(tmp_path, monkeypatch, *, F=False, stopped=False, failed=False):
    """A full1241 metadata plan, with all scientific/packet readers replaced."""
    prior, base, invocation = tmp_path/'prior', tmp_path/'base', tmp_path/'new_invocation'
    packets = tmp_path/'packet_root';env = tmp_path/'environment.json'
    arms = ['F_COMMON', 'F_BI'] if F else ['C0', 'CP', 'CB']
    frames = {d: [dict(id=f'{d}{i}') for i in range(n)] for d, n in (('real', 238), ('synthetic', 1003))}
    binding = dict(orchestration_sha256='D4', preserved_D3_validation_sha256='D3',
                   row_helper_sha256='ROW', numerical_kernel_source_binding={'kernel': 'unchanged'})
    R.Q.write(env, {'constructed': True})
    R.Q.write(packets/'seal.json', {'constructed': True})
    R.Q.write(prior/'P0_INPUT_CONTRACT.json', {d: dict(inventory=[dict(observation_folder='unused') for f in ff]) for d, ff in frames.items()})
    R.Q.write(base/'PACKET_VALIDATION.json', {'constructed': True})
    R.Q.write(base/'row_jobs/real/0000.json', dict(historical_current_calls=17))
    original = dict(binding, schema=R.D4.SCHEMA, workers=4, arms=arms,
        packet_root=str(packets), packet_seal=str(packets/'seal.json'), packet_seal_sha256=R.Q.sha(packets/'seal.json'),
        environment_path=str(env), environment_sha256=R.Q.sha(env), prior_P0_sha256=R.Q.sha(prior/'P0_INPUT_CONTRACT.json'),
        packet_validation_receipt_sha256=R.Q.sha(base/'PACKET_VALIDATION.json'), driver_validation_source_sha256={},
        exact_reuse_binding=None, static_ordered_row_ids={d: [f['id'] for f in ff] for d, ff in frames.items()}, stop_file=None)
    R.Q.write(base/'PRODUCER_LOCK.json', original)
    go = tmp_path/'go.json'
    R.Q.write(go, dict(approved=True, recovery_controller_sha256=R.Q.sha(R.__file__),
                      producer_lock_sha256={str(base): R.Q.sha(base/'PRODUCER_LOCK.json')}))
    monkeypatch.setattr(R, 'source_bindings', lambda: dict(recovery_controller_sha256=R.Q.sha(R.__file__), delegated_D4_source_binding=binding))
    monkeypatch.setattr(R.D4.CAL, 'load_bound_frames', lambda prior, domain: (frames[domain], {}))
    monkeypatch.setattr(R.D4, 'verify_packet_bindings', lambda *args: ({(d, i): 'packet' for d, ff in frames.items() for i, f in enumerate(ff)}, {'constructed': True}))
    monkeypatch.setattr(R.D4, 'verify_reuse_root', lambda *args: None)
    inspections = []
    def inspect(*args):
        inspections.append(True)
        return dict(completed_rows=[], completed_row_count=1 if len(inspections) == 1 else 1241*len(arms),
            known_stored_new_optimizer_calls=17 if len(inspections) == 1 else 22,
            completed_artifacts_sha256={}, incomplete_arm_directories=[])
    monkeypatch.setattr(R, 'inspect_rows', inspect)
    def execute(jobs, output, workers, initializer, initargs, callable_row, stop_file):
        assert Path(output) == invocation and Path(initargs[2]) == base and workers == 4
        assert initializer is R.D4.initialize_worker and callable_row is R.D4.row_job
        assert len(jobs) == 1241
        result = [dict(domain=j[0], ordinal=j[1], id=j[2]['id'], records=[],
                       actual_new_optimizer_calls_this_job=5 if i == 0 else 0,
                       current_job_CPU_seconds=0) for i, j in enumerate(jobs[:3] if failed else jobs[:4] if stopped else jobs)]
        R.Q.write(Path(output)/'row_jobs/real/0000.json', result[0])
        return result, [dict(status='INCOMPLETE_TECHNICAL')] if failed else [], dict(submitted_jobs=4 if stopped or failed else 1241,
            unsubmitted_jobs=1237 if stopped or failed else 0, checkpoint_stop_observed=stopped,
            active_rows_drained_before_exit=True)
    monkeypatch.setattr(R.D4, 'execute_static', execute)
    monkeypatch.setattr(R, 'validate_results', lambda *args: 5)
    def aggregate(output, actual_frames, actual_arms, lock, results, failures, started):
        assert Path(output) == base and lock == original and actual_frames == frames and actual_arms == arms
        assert len(results) == 1241 and failures == []
        value = dict(status='COMPLETE_NEW', current_invocation_new_pose_optimizer_calls=5)
        R.Q.write(base/'SUMMARY.json', value)
        return value
    monkeypatch.setattr(R.D4, 'aggregate', aggregate)
    monkeypatch.setattr(R, 'feature_bindings', lambda base, arms: {'constructed_features': True})
    before = R.hashes(base)
    original_controller = tmp_path/'old_F_controller.py';original_controller.write_text('old controller source')
    completion = tmp_path/'F_COMPLETION.json'
    value = R.run(prior, base, invocation, go, R.Q.sha(go),
        F_completion=completion if F else None, original_F_controller=original_controller if F else None)
    R.require_preserved(base, before)
    return value, base, invocation, completion, original_controller


def test_full_delegation_old_job_preservation_and_once_cost_identity(tmp_path, monkeypatch):
    value, base, invocation, _, _ = constructed_run(tmp_path, monkeypatch)
    assert value['complete'] and value['dispatch']['submitted_jobs'] == 1241
    assert value['prior_stored_known_calls'] == 17
    assert value['actual_new_optimizer_calls_this_invocation'] == 5
    assert value['current_invocation_call_count_status'] == 'EXACT_FULL_ROW_IDENTITY'
    assert value['base_stored_calls_cumulative'] == 22
    assert value['copied_historical_intrinsic_calls_added'] == 0
    assert value['interrupted_historical_optimizer_calls'] == 'UNKNOWN_NOT_ZERO'
    assert R.Q.read(base/'row_jobs/real/0000.json') == dict(historical_current_calls=17)
    dispatch = R.Q.read(base/'DISPATCH_COMPLETION.json')
    assert dispatch['recovery_controller_sha256'] == R.Q.sha(R.__file__)
    assert dispatch['recovery_invocation'] == str(invocation)


def test_checkpoint_stop_does_not_create_partial_aggregate(tmp_path, monkeypatch):
    value, base, invocation, _, _ = constructed_run(tmp_path, monkeypatch, stopped=True)
    assert not value['complete'] and value['status'] == 'INCOMPLETE'
    assert value['base_stored_calls_cumulative'] is None
    assert value['actual_new_optimizer_calls_this_invocation'] is None
    assert value['successful_returned_jobs_new_optimizer_calls_lower_bound'] == 5
    assert value['current_invocation_call_count_status'] == 'LOWER_BOUND_SUCCESSFUL_JOBS_ONLY'
    assert not (base/'SUMMARY.json').exists() and not (base/'DISPATCH_COMPLETION.json').exists()
    assert (invocation/'RECOVERY_COMPLETION.json').exists()


def test_failed_jobs_disclose_current_unknown_work_in_addition_to_old_unknown(tmp_path, monkeypatch):
    value, base, invocation, _, _ = constructed_run(tmp_path, monkeypatch, failed=True)
    assert not value['complete'] and value['actual_new_optimizer_calls_this_invocation'] is None
    assert value['successful_returned_jobs_new_optimizer_calls_lower_bound'] == 5
    assert value['current_invocation_unattributed_optimizer_calls'] == 'UNKNOWN_NOT_ZERO'
    assert value['interrupted_historical_optimizer_calls'] == 'UNKNOWN_NOT_ZERO'
    assert not (base/'SUMMARY.json').exists() and not (base/'DISPATCH_COMPLETION.json').exists()


def test_F_completion_truthfully_binds_recovery_source_and_zero_heads(tmp_path, monkeypatch):
    value, base, invocation, completion, old_controller = constructed_run(tmp_path, monkeypatch, F=True)
    result = R.Q.read(completion)
    assert result['complete'] and result['unconditional_core_only'] and not result['final_F_complete']
    assert result['source_sha256'] == R.Q.sha(R.__file__) != R.Q.sha(old_controller)
    assert result['original_controller_source_sha256'] == R.Q.sha(old_controller)
    assert result['producer_dispatch_sha256'] == R.Q.sha(base/'DISPATCH_COMPLETION.json')
    assert result['H1_fits'] == result['F_choice_calls'] == result['real_reference_reads'] == 0
