"""ROOT-GO-only waiter for the frozen final40 saved-scalar error budget.

It authenticates an already completed derived controller, then invokes exactly
one existing scalar-only child. No producer, fit, pose metric or reference/RGB
reader is called. Existing completed/partial outputs are always preserved.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time

from . import derived_completion as D
from . import error_budget as B

F = D.F
DERIVED_SHA = '42fd83e31b47e89aaa94c4b7bf6b5573c7eaecec1c90a6251fcaf72a5f69a03d'
BUDGET_SHA = 'b534ecce4426121d3d0c7a773e310f7453bdd1f8ac6f1145c1af7863074c62a2'


def dependency_guard():
    if F.sha(D.__file__) != DERIVED_SHA or F.sha(B.__file__) != BUDGET_SHA:
        raise ValueError('Frozen derived/error-budget source changed')


def prepare(derived_plan, derived_GO, result_output, output):
    derived_plan, derived_GO, result_output, output = map(Path, (derived_plan, derived_GO, result_output, output))
    if output.exists() or result_output.exists():raise FileExistsError('Prepare fresh waiter and final40 result namespaces')
    if output.resolve() == result_output.resolve():raise ValueError('Waiter metadata and public results require separate folders')
    dependency_guard();dp = F.read(derived_plan);go = F.read(derived_GO)
    if (go.get('approved') is not True or go['plan_sha256'] != F.sha(derived_plan) or
            go['controller_sha256'] != DERIVED_SHA or dp['controller_sha256'] != DERIVED_SHA):
        raise ValueError('Reviewed source-bound derived PLAN/GO required')
    D.guard(dp)
    sources = dict(dp['sources'])
    for path in (Path(__file__).resolve(), Path(B.__file__).resolve()):sources[str(path)] = F.sha(path)
    plan = dict(schema='FINAL40_ERROR_BUDGET_WAITER_PLAN_V1', waiter_source_sha256=F.sha(__file__),
        derived_plan=str(derived_plan.resolve()), derived_GO=str(derived_GO.resolve()),
        derived_output=dp['output'], result_output=str(result_output.resolve()), output=str(output.resolve()),
        worktree=dp['worktree'], python=sys.executable, sources=sources,
        fixed_inputs={str(p.resolve()): F.sha(p) for p in (derived_plan, derived_GO)},
        expected_arm_count=40, child_source_sha256=BUDGET_SHA, derived_source_sha256=DERIVED_SHA,
        sequential_child_workers=1, new_solves=0, new_fits=0, raw_reference_pose_or_RGB_readers=False)
    F.write(output/'PLAN.json', plan)
    return plan


def guard(plan):
    dependency_guard();F.verify_map(plan['sources']);F.verify_map(plan['fixed_inputs'])
    D.guard(F.read(plan['derived_plan']))


def completed_inputs(plan):
    """Call only the frozen controller's completed, read-only return branch."""
    root = Path(plan['derived_output']);terminal = root/'DERIVED_COMPLETION_SEAL.json'
    if not terminal.exists():raise FileNotFoundError('Full derived completion must exist before authentication')
    before = {str(p): F.sha(p) for p in (terminal, root/'ACTUAL_COMMAND_LEDGER.jsonl')}
    # The required terminal is immutable. D.run's completed branch verifies all
    # eight checkpoints/native artifacts/input ancestry and never appends/runs.
    D.run(plan['derived_plan'], plan['derived_GO'])
    F.verify_map(before)
    cp_path = root/'checkpoints/export.json';cp = F.read(cp_path)
    export = root/'PUBLIC_METRICS40/EXPORT_SEAL.json';trusted = cp['completed_files'].get(str(export.resolve()))
    if not cp.get('complete') or trusted != F.sha(export):raise ValueError('Trusted completed export checkpoint differs')
    return dict(derived_completion_sha256=F.sha(terminal), export_checkpoint_sha256=F.sha(cp_path),
        export_seal_sha256=trusted, export_dir=str(export.parent))


def native(plan, inputs):
    output = Path(plan['result_output']);sp = output/'ERROR_BUDGET_SEAL.json';seal = F.read(sp)
    if (seal.get('schema') != 'SAVED_SCALAR_ERROR_BUDGET_SEAL_V1' or seal.get('complete') is not True or
            seal['source_sha256'] != BUDGET_SHA or seal['input_export_seal_sha256'] != inputs['export_seal_sha256'] or
            seal['arms'] != B.named_arms(40) or seal['populations'] != B.E.EXPECTED or
            seal['campaign_scope'] != 'ALL40_COMPLETED_ARMS' or
            (seal['joint_rows'], seal['band_rows'], seal['conservation_rows']) != (800, 1120, 320) or
            seal['posthoc_descriptive'] is not True or seal['preregistered_test'] or seal['selector_performance_claim'] or
            not seal['no_outlier_exclusion'] or any(seal[k] != 0 for k in
                ('raw_reference_reads', 'pose_or_RGB_reads', 'new_pose_metric_calls', 'new_solves', 'new_fits', 'independent_real_test_n'))):
        raise ValueError('Completed final40 descriptive scalar contract differs')
    names = {'JOINT_CATEGORIES.csv', 'ERROR_BANDS.csv', 'CONSERVATION_CHECKS.csv', 'ERROR_BUDGET_SCHEMA.json', 'INPUT_BINDINGS.json'}
    if set(seal['artifacts']) != names:raise ValueError('Exactly five public artifacts plus the final40 seal required')
    bindings = {str(sp): F.sha(sp)}
    for name, digest in seal['artifacts'].items():
        if F.sha(output/name) != digest:raise ValueError('Completed final40 artifact changed')
        bindings[str(output/name)] = digest
    if F.read(output/'INPUT_BINDINGS.json')['export_seal_sha256'] != inputs['export_seal_sha256']:
        raise ValueError('Public final40 input binding differs')
    return bindings


def run(plan_path, GO_path):
    started, cpu = time.monotonic(), time.process_time()
    plan, go = F.read(plan_path), F.read(GO_path)
    if (go.get('approved') is not True or go.get('plan_sha256') != F.sha(plan_path) or
            go.get('waiter_source_sha256') != F.sha(__file__) or go.get('new_fits_authorized') != 0 or
            go.get('new_pose_solves_authorized') != 0 or go.get('expected_arm_count') != 40 or
            go.get('sequential_child_workers') != 1):
        raise PermissionError('Reviewed source/plan-bound final40 waiter ROOT GO required')
    guard(plan);out = Path(plan['output']);terminal = out/'ERROR40_COMPLETION.json'
    if terminal.exists():
        value = F.read(terminal)
        if (not value.get('complete') or value['waiter_source_sha256'] != F.sha(__file__) or
                value['plan_sha256'] != F.sha(plan_path) or value['ROOT_GO_sha256'] != F.sha(GO_path) or
                value['actual_command_ledger_sha256'] != F.sha(out/'ACTUAL_COMMAND_LEDGER.jsonl')):
            raise ValueError('Completed final40 waiter ancestry changed')
        inputs = completed_inputs(plan)
        if inputs != value['authenticated_inputs'] or native(plan, inputs) != value['result_artifacts']:
            raise ValueError('Completed final40 output/input ancestry changed')
        cp = out/'checkpoints/error_budget40.json'
        if F.sha(cp) != value['checkpoint_sha256']:raise ValueError('Completed final40 checkpoint changed')
        return value
    with (out/'OWNER.lock').open('a+') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inputs = F.wait_for(Path(plan['derived_output'])/'DERIVED_COMPLETION_SEAL.json', lambda: completed_inputs(plan))
        guard(plan)
        argv = [plan['python'], '-B', '-m', F.NS+'error_budget', '--export-dir', inputs['export_dir'],
                '--export-seal-sha256', inputs['export_seal_sha256'], '--output', plan['result_output'], '--expected-arm-count', '40']
        environment = dict(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONPATH=plan['worktree'])
        F.append_ledger(out, dict(event='CONTROLLER_START', actual_argv=[v.decode() for v in Path('/proc/self/cmdline').read_bytes().split(b'\0') if v],
            waiter_source_sha256=F.sha(__file__), child_source_sha256=BUDGET_SHA, child_environment=environment,
            cwd=plan['worktree'], plan_sha256=F.sha(plan_path), ROOT_GO_sha256=F.sha(GO_path),
            authenticated_inputs=inputs, epoch=time.time()))
        identity = {'PLAN.json': F.sha(plan_path), 'ROOT_GO': F.sha(GO_path), **inputs}
        action = F.stage(out, 'error_budget40', argv, identity, lambda: native(plan, inputs),
            Path(plan['result_output'])/'ERROR_BUDGET_SEAL.json', Path(plan['result_output']),
            lambda key, command: F.execute(key, command, out, plan['worktree']))
        guard(plan);artifacts = native(plan, inputs)
        F.append_ledger(out, dict(event='CHECKPOINT_ACTION', phase='error_budget40', action=action, epoch=time.time()))
        value = dict(schema='FINAL40_ERROR_BUDGET_WAITER_COMPLETION_V1', complete=True,
            waiter_source_sha256=F.sha(__file__), child_source_sha256=BUDGET_SHA, plan_sha256=F.sha(plan_path),
            ROOT_GO_sha256=F.sha(GO_path), authenticated_inputs=inputs, result_artifacts=artifacts,
            actual_command_ledger_sha256=F.sha(out/'ACTUAL_COMMAND_LEDGER.jsonl'),
            checkpoint_sha256=F.sha(out/'checkpoints/error_budget40.json'), action=action,
            current_invocation_child_calls=int(action == 'COMPLETE'),
            historical_analysis_cost='see preserved ledger;unknown if an earlier native completion was adopted',
            controller_wall_seconds_including_wait=time.monotonic()-started, controller_CPU_seconds=time.process_time()-cpu,
            child_time_scope='actual one-child CPU/wall separately recorded;controller elapsed includes upstream waiting',
            expected_arm_count=40, new_fits=0, new_solves=0, new_pose_metric_calls=0, raw_reference_pose_or_RGB_reads=0,
            posthoc_descriptive=True, epoch=time.time())
        F.write(terminal, value)
        return value


if __name__ == '__main__':
    p = argparse.ArgumentParser();sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare')
    for name in ('derived-plan', 'derived-GO', 'result-output', 'output'):a.add_argument('--'+name, required=True, type=Path)
    a = sub.add_parser('run');a.add_argument('--plan', required=True, type=Path);a.add_argument('--root-GO', required=True, type=Path)
    a = p.parse_args()
    if a.mode == 'prepare':prepare(a.derived_plan, a.derived_GO, a.result_output, a.output)
    else:print(run(a.plan, a.root_GO))
