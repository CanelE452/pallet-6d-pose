"""Meaningful accounting contracts; real-data parity is a separately saved pilot."""
import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import fixed_cost as C


def test_outline_weight_applied_once_and_metadata_not_loss():
    p=dict(soft_BI_robust_loss=2.,partial_boundary_normal_robust_loss=3.,
           global_outline_robust_mean_loss=5.,global_outline_factor_weight=.2,
           coarse_pose_prior_loss=0.,complete_BI_likelihood_count=100,
           prior_whitened_residual=[999.]*6,boundary_factor_weight=4.)
    assert C.weighted_components(p)==dict(full_BI=2.,partial_normal=3.,outline=1.,prior=0.)
    assert sum(C.weighted_components(p).values())==6.


def test_retained_and_endpoint_record_join_without_mutation():
    v=dict(original_record=dict(initialization='A',objective=2.,R=np.eye(3).tolist()),
           original_endpoint_record=dict(branch=0,objective_components=dict(soft_BI_robust_loss=2.)))
    before=deepcopy(v);out=C._candidate_record(v)
    assert out['objective']==2. and out['branch']==0
    assert v==before


def test_reject_nonoriginal_stage_or_family_before_input_read():
    with pytest.raises(ValueError,match='original'):
        C.prepare({},'G17','ORIGINAL_WD')
    with pytest.raises(ValueError,match='original'):
        C.prepare({},'G13','NEW_FAMILY')


def test_execution_source_contains_no_fit_or_candidate_generator_call():
    tree=ast.parse(Path(C.__file__).read_text())
    forbidden={'minimize','least_squares','solvePnP','solvePnPGeneric','solveP3P',
               'solve_uncertain_joint_ot_pnp','solve_two_wd_families','refine_global_outline_joint_pose',
               '_initial_poses','observe_global_outline','fit','forward'}
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            target=node.func
            calls.append(target.attr if isinstance(target,ast.Attribute) else target.id if isinstance(target,ast.Name) else None)
    assert not forbidden.intersection(calls)
