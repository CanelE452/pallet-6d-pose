"""Pure packaging/clone checks: no optimizer or reference invocation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from . import typed_replay as T


def _outline_stub(starts):
    records = []
    unique = starts
    choices = [None]
    def consider(p, name, optimizer=None):
        R, t = p, p + 1
        record = {'value': p * p, 'name': name}
        records.append(record)
    for start in unique:
        consider(start['R'], start['name'])
    return records


def _partial_stub(initial):
    starts, search = V3._initial_poses(initial)
    def fit(start, use_boundary=True):
        return start, {'R': start['R'], 't': start['R']+1,
                       'objective': start['R']**2, 'feasible': False}
    seeds = []
    for start in starts:
        answer, seed = fit(start, use_boundary=False)
        seeds.append(seed)
    output = []
    for start in starts:
        output.append(fit(start))
    return output, seeds, search


def test_outline_logging_preserves_output_and_order():
    ledger = []
    clone = T._clone(_outline_stub, {}, lambda R, t, record, kind: ledger.append((R, t, deepcopy(record), kind)))
    starts = [{'R': 2, 'name': 'first'}, {'R': 3, 'name': 'second'}]
    assert clone(starts) == _outline_stub(starts)
    assert [r[0] for r in ledger] == [2, 3]
    assert clone.__replay_clone_audit__['edits'] == ['log_consider_Rt']


def test_single_outline_replaces_only_start_policy():
    ledger = []
    clone = T._clone(_outline_stub, {}, lambda *args: ledger.append(args),
                     single_start={'R': 7, 'name': T.START_NAME})
    assert clone([{'R': 2, 'name': 'discarded'}]) == [{'value': 49, 'name': T.START_NAME}]
    assert len(ledger) == 1


def test_partial_single_omits_new_initial_generation_and_BI_prefit():
    calls = []
    # Would fail if P3P/initial-generation function were evaluated.
    initial_api = SimpleNamespace(_initial_poses=lambda *args: (_ for _ in ()).throw(AssertionError('new PnP')))
    clone = T._clone(_partial_stub, {'V3': initial_api}, lambda *args: calls.append(args),
                     single_start={'R': 7, 'name': T.START_NAME}, partial=True)
    out, seeds, search = clone(None)
    assert len(out) == 1 and out[0][1]['objective'] == 49
    assert seeds == [] and len(calls) == 1
    assert calls[0][-1] == 'partial_joint_fit'
    assert search['new_P3P_starts'] == 0


def test_native_fallback_selection_matches_original_readout():
    coarse = dict(R=np.eye(3).tolist(), t=[0, 0, 3])
    seed = dict(R=np.eye(3).tolist(), t=[0, 0, 2], visible_support_guards_pass=True,
                contains_underconstrained_pose_family=False)
    solution = dict(accepted=False, BI_prior_seed=seed)
    assert T.choose_readout(solution, coarse)[1] == [0, 0, 2]
    seed['contains_underconstrained_pose_family'] = True
    assert T.choose_readout(solution, coarse)[1] == [0, 0, 3]


def test_original_functions_clone_without_running_fit():
    for name in ('statewise_global_outline_solver_v4_10', 'statewise_global_outline_solver_v4_11',
                 'statewise_partial_BI_outline_solver_v4_13'):
        module = T._backend(name, lambda *args: None)
        assert module.refine_global_outline_joint_pose.__replay_clone_audit__['edits'] == ['log_consider_Rt']
        single = T._backend(name, lambda *args: None, single_start={'R': np.eye(3).tolist(), 't': [0, 0, 3]})
        assert set(single.refine_global_outline_joint_pose.__replay_clone_audit__['edits']) == {
            'single_outline_start', 'log_consider_Rt'}
    partial = T._module('statewise_partial_BI_u_solver_v4_13')
    cloned = T._clone(partial.solve_partial_BI_u_region_joint_ot_pnp, {}, lambda *args: None,
                      single_start={'R': np.eye(3).tolist(), 't': [0, 0, 3]}, partial=True)
    assert set(cloned.__replay_clone_audit__['edits']) == {
        'single_partial_start_no_new_PnP', 'log_partial_fit_Rt', 'single_skip_BI_seed_fit'}
