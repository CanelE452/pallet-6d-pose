"""Synthetic optimizer integration, no dataset or image access."""
from copy import deepcopy

import numpy as np
import pytest

from . import native_edge_problem_v4_17 as N
from . import refine_native_edge_pose_v4_17 as NEW
from .test_statewise_partial_boundary_u_solver_v4_6 import arguments, region


def setup():
    a = arguments()
    p = N.prepare_native_edge_problem(*a[:7], config=a[7], boundary_observations=[],
        u_regions=[region(a)], edge_bundle=dict(edges=[], replaces_normal_factor_refs=[]))
    R, t = np.asarray(a[6]['R']), np.asarray(a[6]['t'])
    return a, p, R, t


def test_saved_start_continuous_fit_reduces_native_cost_and_keeps_all_current_guards():
    a, problem, R, t = setup()
    shifted = t + [.015, -.008, .025]
    original = problem.evaluate(R, shifted)
    assert original['accepted']
    proposals = [dict(name='saved', R=R.tolist(), t=shifted.tolist(), accepted=False,
        objective=-1e9, metrics=object())]
    saved = deepcopy(proposals[0]['R']), deepcopy(proposals[0]['t'])
    result = NEW.refine_native_edge_pose(problem, proposals, a[6],
        configuration=dict(maximum_native_ranked_starts=1))
    assert result['chosen']['accepted']
    assert result['chosen']['objective'] <= original['objective'] + 1e-8
    assert result['chosen']['objective'] < 1e-8
    assert result['chosen']['all_original_hard_guards_recomputed']
    assert result['search']['continuous_SE3'] and not result['search']['mask_objective']
    assert result['search']['new_P3P_or_global_starts'] == 0
    assert (proposals[0]['R'], proposals[0]['t']) == saved


def test_no_old_accepted_flag_can_admit_an_infeasible_start_and_coarse_is_rechecked():
    a, problem, R, t = setup()
    proposals = [dict(name='invalid', R=R.tolist(), t=[0., 0., -4.], accepted=True,
        objective=-1e100)]
    starts, audit = NEW.choose_native_starts(problem, proposals, a[6])
    assert [v['name'] for v in starts] == ['OWN_INITIAL_COARSE']
    assert audit['rejected_saved_candidates'][0]['name'] == 'invalid'
    assert audit['fresh_feasible_saved_candidates'] == 0


def test_start_budget_is_explicit_and_preserves_pose_basins_before_coarse():
    a, problem, R, t = setup()
    proposals = [dict(name='zero', R=R.tolist(), t=t.tolist()),
        dict(name='duplicate', R=R.tolist(), t=t.tolist()),
        dict(name='near', R=R.tolist(), t=(t + [.001, 0., 0.]).tolist()),
        dict(name='other', R=R.tolist(), t=(t + [.03, 0., 0.]).tolist())]
    starts, audit = NEW.choose_native_starts(problem, proposals, a[6],
        configuration=dict(maximum_native_ranked_starts=2))
    assert len(starts) == 2 and set(v['name'] for v in starts) == {'zero', 'other'}
    assert audit['configuration']['maximum_native_ranked_starts'] == 2
    with pytest.raises(ValueError):
        NEW.choose_native_starts(problem, proposals, a[6],
            configuration=dict(maximum_native_ranked_starts=0))
