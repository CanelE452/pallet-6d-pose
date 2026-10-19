"""Constructed metadata only; no actual scalar child, data or reference/RGB."""
import json
from pathlib import Path

import pytest

from scripts.research.pallet_bi_full_execution_v1 import error40_waiter as W

F = W.F


def put(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def forbidden(*args, **kwargs):
    raise AssertionError('Constructed test forbids real derived/analysis/child calls')


def test_dependencies_are_the_exact_frozen_descriptive_sources():
    W.dependency_guard()
    assert F.sha(W.D.__file__) == W.DERIVED_SHA and F.sha(W.B.__file__) == W.BUDGET_SHA


def test_prepare_only_metadata_then_refuses_existing_namespaces(tmp_path, monkeypatch):
    dp, dg = tmp_path/'derived_plan.json', tmp_path/'derived_go.json'
    put(dp, dict(controller_sha256=W.DERIVED_SHA, sources={}, output=str(tmp_path/'derived'), worktree='constructed'))
    put(dg, dict(approved=True, plan_sha256=F.sha(dp), controller_sha256=W.DERIVED_SHA))
    monkeypatch.setattr(W.D, 'guard', lambda plan: None)
    monkeypatch.setattr(W.D, 'run', forbidden);monkeypatch.setattr(F.subprocess, 'call', forbidden)
    plan = W.prepare(dp, dg, tmp_path/'result', tmp_path/'waiter')
    assert plan['expected_arm_count'] == 40 and plan['new_fits'] == plan['new_solves'] == 0
    assert set(p.name for p in (tmp_path/'waiter').iterdir()) == {'PLAN.json'}
    with pytest.raises(FileExistsError):W.prepare(dp, dg, tmp_path/'result', tmp_path/'waiter')


def completed_fixture(tmp_path, monkeypatch):
    root = tmp_path/'derived';terminal = root/'DERIVED_COMPLETION_SEAL.json';put(terminal, dict(complete=True))
    (root/'ACTUAL_COMMAND_LEDGER.jsonl').write_text('preserved completed derived ledger\n')
    export = root/'PUBLIC_METRICS40/EXPORT_SEAL.json';put(export, dict(complete=True, constructed=True))
    put(root/'checkpoints/export.json', dict(complete=True, completed_files={str(export.resolve()): F.sha(export)}))
    plan = dict(derived_output=str(root), derived_plan='unused', derived_GO='unused')
    calls = []
    monkeypatch.setattr(W.D, 'run', lambda *args: calls.append('completed_read_only_verification'))
    return plan, root, export, calls


def test_missing_derived_terminal_cannot_call_active_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(W.D, 'run', forbidden)
    with pytest.raises(FileNotFoundError):W.completed_inputs(dict(derived_output=str(tmp_path/'pending')))


def test_trusted_export_digest_comes_from_completed_checkpoint_not_current_reseal(tmp_path, monkeypatch):
    plan, root, export, calls = completed_fixture(tmp_path, monkeypatch)
    before = (root/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes()
    value = W.completed_inputs(plan)
    assert value['export_seal_sha256'] == F.sha(export) and calls == ['completed_read_only_verification']
    assert (root/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes() == before
    put(export, dict(complete=True, resealed=True))
    with pytest.raises(ValueError, match='checkpoint'):W.completed_inputs(plan)


def test_completed_helper_is_required_to_make_zero_ledger_changes(tmp_path, monkeypatch):
    plan, root, export, calls = completed_fixture(tmp_path, monkeypatch)
    def changed(*args):
        with (root/'ACTUAL_COMMAND_LEDGER.jsonl').open('a') as f:f.write('unexpected append')
    monkeypatch.setattr(W.D, 'run', changed)
    with pytest.raises(ValueError, match='Frozen'):W.completed_inputs(plan)


def native_fixture(tmp_path):
    out = tmp_path/'result'
    names = ['JOINT_CATEGORIES.csv', 'ERROR_BANDS.csv', 'CONSERVATION_CHECKS.csv', 'ERROR_BUDGET_SCHEMA.json', 'INPUT_BINDINGS.json']
    for name in names:put(out/name, {'export_seal_sha256': 'trusted'} if name == 'INPUT_BINDINGS.json' else {'constructed': True})
    seal = dict(schema='SAVED_SCALAR_ERROR_BUDGET_SEAL_V1', complete=True, source_sha256=W.BUDGET_SHA,
        input_export_seal_sha256='trusted', arms=W.B.named_arms(40), populations=W.B.E.EXPECTED,
        campaign_scope='ALL40_COMPLETED_ARMS', joint_rows=800, band_rows=1120, conservation_rows=320,
        posthoc_descriptive=True, preregistered_test=False, selector_performance_claim=False, no_outlier_exclusion=True,
        **{k: 0 for k in ('raw_reference_reads', 'pose_or_RGB_reads', 'new_pose_metric_calls', 'new_solves', 'new_fits', 'independent_real_test_n')},
        artifacts={n: F.sha(out/n) for n in names})
    put(out/'ERROR_BUDGET_SEAL.json', seal)
    return dict(result_output=str(out)), dict(export_seal_sha256='trusted'), out, seal


@pytest.mark.parametrize('change', ['scope', 'counter', 'rows', 'hash'])
def test_native_requires_final40_scope_zero_calls_counts_and_actual_hashes(tmp_path, change):
    plan, inputs, out, seal = native_fixture(tmp_path)
    assert len(W.native(plan, inputs)) == 6
    if change == 'scope':seal['campaign_scope'] = 'INITIAL25_COMPLETED_ARMS_ONLY'
    elif change == 'counter':seal['new_fits'] = 1
    elif change == 'rows':seal['band_rows'] = 1119
    else:(out/'JOINT_CATEGORIES.csv').write_text('changed')
    put(out/'ERROR_BUDGET_SEAL.json', seal)
    with pytest.raises(ValueError):W.native(plan, inputs)


def execution_fixture(tmp_path, monkeypatch):
    out = tmp_path/'execution';out.mkdir();result = tmp_path/'result'
    plan = dict(output=str(out), result_output=str(result), derived_output='constructed_derived', python='python', worktree='constructed')
    pp, gp = out/'PLAN.json', tmp_path/'GO.json';put(pp, plan)
    put(gp, dict(approved=True, plan_sha256=F.sha(pp), waiter_source_sha256=F.sha(W.__file__),
        new_fits_authorized=0, new_pose_solves_authorized=0, expected_arm_count=40, sequential_child_workers=1))
    monkeypatch.setattr(W, 'guard', lambda plan: None)
    inputs = dict(derived_completion_sha256='derived', export_checkpoint_sha256='checkpoint',
                  export_seal_sha256='trusted', export_dir='scalar_only_input')
    monkeypatch.setattr(W, 'completed_inputs', lambda plan: inputs)
    monkeypatch.setattr(F, 'wait_for', lambda path, validator: validator())
    def native(plan, inputs):
        path = result/'ERROR_BUDGET_SEAL.json'
        if not F.read(path)['complete']:raise ValueError('Fixture not complete')
        return {str(path): F.sha(path)}
    monkeypatch.setattr(W, 'native', native)
    calls = []
    def execute(key, argv, output, worktree):
        assert key == 'error_budget40' and argv[-2:] == ['--expected-arm-count', '40']
        assert argv[argv.index('--export-seal-sha256')+1] == 'trusted'
        assert argv[3] == F.NS+'error_budget'
        calls.append(list(argv));put(result/'ERROR_BUDGET_SEAL.json', dict(complete=True))
    monkeypatch.setattr(F, 'execute', execute);monkeypatch.setattr(F.subprocess, 'call', forbidden)
    return pp, gp, out, result, calls


def test_ROOT_GO_rejected_before_any_wait_or_helper_call(tmp_path, monkeypatch):
    pp, gp = tmp_path/'p', tmp_path/'g';put(pp, {});put(gp, {'approved': False})
    monkeypatch.setattr(W, 'guard', forbidden);monkeypatch.setattr(F, 'wait_for', forbidden)
    with pytest.raises(PermissionError):W.run(pp, gp)


def test_one_child_then_completed_return_never_appends_or_runs_again(tmp_path, monkeypatch):
    pp, gp, out, result, calls = execution_fixture(tmp_path, monkeypatch)
    value = W.run(pp, gp)
    assert len(calls) == 1 and value['current_invocation_child_calls'] == 1
    before = (out/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes()
    assert W.run(pp, gp) == value and len(calls) == 1
    assert (out/'ACTUAL_COMMAND_LEDGER.jsonl').read_bytes() == before
    put(result/'ERROR_BUDGET_SEAL.json', dict(complete=True, changed=True))
    with pytest.raises(ValueError):W.run(pp, gp)


def test_pending_completion_and_partial_output_never_start_analysis(tmp_path, monkeypatch):
    pp, gp, out, result, calls = execution_fixture(tmp_path, monkeypatch)
    def pending(*args):raise RuntimeError('constructed pending completion')
    monkeypatch.setattr(F, 'wait_for', pending)
    with pytest.raises(RuntimeError, match='pending'):W.run(pp, gp)
    assert calls == [] and not (out/'ACTUAL_COMMAND_LEDGER.jsonl').exists()
    monkeypatch.setattr(F, 'wait_for', lambda path, validator: validator())
    sentinel = result/'partial.csv';sentinel.parent.mkdir();sentinel.write_text('keep')
    with pytest.raises(RuntimeError, match='Partial output preserved'):W.run(pp, gp)
    assert calls == [] and sentinel.read_text() == 'keep'
