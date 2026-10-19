"""Constructed scheduler contracts only; no model/pose/reference API calls."""
from multiprocessing import get_context
from pathlib import Path
import ast
import inspect
import time
import pytest
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_dispatch_four as D4
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_dispatch as D3
from scripts.research.pallet_bi_full_execution_v1 import fullnormal_parallel as ROW


def stop_probe(job):
    domain, ordinal, frame, *_ = job
    if ordinal == 0:
        Path(frame['stop']).write_text('constructed checkpoint request')
    # All four tasks were submitted before the initializer barrier opens;
    # they finish even when the first task requests a stop.
    time.sleep(.08)
    return dict(domain=domain, ordinal=ordinal, id=frame['id'], completed=True,
        actual_new_optimizer_calls_this_job=0, new_H1_fits=0)


def test_exact_existing_primitives_validation_and_numerical_source_binding():
    for name in ('row_job', 'initialize_worker', 'aggregate', 'verify_packet_bindings'):
        assert getattr(D4, name) is getattr(ROW, name) is getattr(D3, name)
    assert D4.verify_reuse_root is D3.verify_reuse_root
    b3, b4 = D3.source_bindings(), D4.source_bindings()
    assert b4['numerical_kernel_source_binding'] == b3['numerical_kernel_source_binding']
    assert b4['row_helper_sha256'] == b3['row_helper_sha256'] == D3.ROW_HELPER_EXPECTED_SHA256
    assert b4['preserved_D3_validation_sha256'] == b3['orchestration_sha256'] == D4.D3_EXPECTED_SHA256
    assert b4['orchestration_sha256'] == D4.Q.sha(D4.__file__)
    assert b4['orchestration_sha256'] != b3['orchestration_sha256']
    assert D4.SCHEMA != 'FULL_NORMAL_ORDERED_DISPATCH_V1' and D4.WORKERS == 4


def test_four_spawned_workers_same_constructed_contract_as_D3_probe(tmp_path):
    jobs = [('constructed', i, dict(id=str(i)), None, None) for i in range(12)]
    ctx = get_context('spawn');barrier = ctx.Barrier(4)
    results, failures, dispatch = D4.execute_static(jobs, tmp_path, 4,
        D3.initialize_constructed_probe, (barrier,), D3.constructed_probe)
    assert len(results) == len(jobs) and not failures
    assert len({r['worker_pid'] for r in results}) == 4
    assert all(set(r['thread_limits'].values()) == {'1'} for r in results)
    assert sum(r['actual_new_optimizer_calls_this_job'] for r in results) == 0
    for job, result in zip(jobs, sorted(results, key=lambda r:r['ordinal'])):
        expected = D3.constructed_probe(job)
        assert {k:v for k,v in result.items() if k != 'worker_pid'} == {
            k:v for k,v in expected.items() if k != 'worker_pid'}
    assert dispatch == dict(submitted_jobs=12, unsubmitted_jobs=0,
        checkpoint_stop_observed=False, active_rows_drained_before_exit=True)


def test_four_active_rows_drain_then_stop_without_dropping_unsubmitted_rows(tmp_path):
    stop = tmp_path/'STOP'
    jobs = [('constructed', i, dict(id=str(i), stop=str(stop)), None, None) for i in range(12)]
    ctx = get_context('spawn');barrier = ctx.Barrier(4)
    results, failures, dispatch = D4.execute_static(jobs, tmp_path, 4,
        D3.initialize_constructed_probe, (barrier,), stop_probe, stop)
    assert not failures and len(results) == 4
    assert sorted(r['ordinal'] for r in results) == [0,1,2,3]
    assert len(list((tmp_path/'row_jobs/constructed').glob('*.json'))) == 4
    assert dispatch == dict(submitted_jobs=4, unsubmitted_jobs=8,
        checkpoint_stop_observed=True, active_rows_drained_before_exit=True)


def test_preexisting_stop_submits_zero_and_explicit_workers_truth(tmp_path):
    stop = tmp_path/'STOP';stop.write_text('constructed stop before submit')
    jobs = [('constructed',0,dict(id='unstarted'),None,None)]
    results, failures, dispatch = D4.execute_static(jobs,tmp_path,4,None,(),D3.constructed_probe,stop)
    assert not results and not failures and dispatch['submitted_jobs']==0 and dispatch['unsubmitted_jobs']==1
    assert dispatch['checkpoint_stop_observed'] and dispatch['active_rows_drained_before_exit']
    with pytest.raises(ValueError,match='workers=4'):
        D4.execute_static(jobs,tmp_path,3,None,(),D3.constructed_probe)
    with pytest.raises(PermissionError,match='Fresh root'):
        D4.run('unused','unused','unused','unused',tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_existing_output_rejected_before_reads_and_preserves_every_byte(tmp_path, monkeypatch):
    root = tmp_path/'existing';root.mkdir()
    (root/'PRODUCER_LOCK.json').write_text('immutable old schedule')
    (root/'PACKET_VALIDATION.json').write_text('immutable old validation')
    before = {p.name:p.read_bytes() for p in root.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Old-output rejection must precede source checks/frame loading')
    monkeypatch.setattr(D4, 'source_bindings', forbidden)
    monkeypatch.setattr(D4.CAL, 'load_bound_frames', forbidden)
    with pytest.raises(FileExistsError,match='new output namespace'):
        D4.run('unused','unused','unused','unused',root,root_GO=True)
    assert {p.name:p.read_bytes() for p in root.iterdir()} == before


def test_initialization_source_has_finite_ordered_enumeration_and_no_process_rng():
    # Static source-contract check only: do not execute any pose constructor.
    original = inspect.getsource(D4.K.B.V._initial_poses)
    backend = inspect.getsource(D4.K.B._initial_poses)
    source = original + '\n' + backend
    names = {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node,ast.Attribute)}
    assert {'solveP3P','solvePnPGeneric','SOLVEPNP_P3P','SOLVEPNP_SQPNP'} <= names
    assert not names & {'random','default_rng','RandomState','rand','randn','choice','shuffle',
                        'solvePnPRansac','setRNGSeed'}
    assert 'combinations(range(len(ids)), 3)' in original
    assert 'triples[:max(0, limit)]' in original
    assert 'permutations(range(3))' in original and 'product([-1, 1], repeat=3)' in original
    assert 'permutations(range(3))' in backend and 'product([-1,1],repeat=3)' in backend
    assert 'ransac_control' not in inspect.getsource(D4.K.solve_frame)
    assert 'ransac_control' not in inspect.getsource(D4.K.feature_frame)
